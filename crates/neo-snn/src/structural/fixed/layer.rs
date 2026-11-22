//! Masks and statistics of a single fixed-shape layer.
//!
//! Weight layout is the same as in `burn::nn::Linear`: an `[in, out]` matrix, element
//! `(i, o)` located at index `i * out + o`.

use crate::structural::error::{StructuralError, StructuralResult};
use crate::structural::rng::Rng64;

/// Layer role: it determines whether neurons can be deactivated.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum LayerRole {
    /// Hidden layer: neurons can be activated from the reserve and deactivated.
    Hidden,
    /// Output layer: all neurons are always active, only connections change.
    Output,
}

/// Layer description.
#[derive(Debug, Clone, PartialEq)]
pub struct LayerSpec {
    /// Name (for the log).
    pub name: String,
    /// Number of inputs (rows of the weight matrix).
    pub in_features: usize,
    /// Capacity in output neurons (columns of the weight matrix), including the reserve.
    pub out_features: usize,
    /// Which layer feeds the inputs of this layer. For a recurrent matrix of layer `k`
    /// this is `k` itself. `None` — the input is external (all inputs are alive).
    pub input_from: Option<usize>,
    /// Role.
    pub role: LayerRole,
    /// How many output neurons are active initially; the rest are reserve.
    pub initial_active: usize,
    /// Initial fraction of enabled connections among the possible ones (0..=1].
    pub initial_density: f32,
}

impl LayerSpec {
    /// A fully connected layer with all neurons active.
    pub fn dense(
        name: impl Into<String>,
        in_features: usize,
        out_features: usize,
        role: LayerRole,
    ) -> Self {
        Self {
            name: name.into(),
            in_features,
            out_features,
            input_from: None,
            role,
            initial_active: out_features,
            initial_density: 1.0,
        }
    }

    /// Set the input source.
    pub fn fed_by(mut self, layer: usize) -> Self {
        self.input_from = Some(layer);
        self
    }

    /// Keep `n` neurons active, the rest as reserve.
    pub fn with_initial_active(mut self, n: usize) -> Self {
        self.initial_active = n;
        self
    }

    /// Initial sparsity.
    pub fn with_density(mut self, d: f32) -> Self {
        self.initial_density = d;
        self
    }

    /// Number of elements in the weight matrix.
    pub fn weight_len(&self) -> usize {
        self.in_features * self.out_features
    }

    pub(crate) fn validate(&self, layer: usize, layers: usize) -> StructuralResult<()> {
        let bad = |m: String| {
            Err(StructuralError::InvalidConfig(format!(
                "layer {layer} \"{}\": {m}",
                self.name
            )))
        };
        if self.in_features == 0 || self.out_features == 0 {
            return bad("dimensions must be > 0".into());
        }
        if self.initial_active == 0 || self.initial_active > self.out_features {
            return bad(format!(
                "initial_active {} outside 1..={}",
                self.initial_active, self.out_features
            ));
        }
        if !(self.initial_density > 0.0 && self.initial_density <= 1.0) {
            return bad("initial_density must be in (0, 1]".into());
        }
        if self.role == LayerRole::Output && self.initial_active != self.out_features {
            return bad("an output layer cannot have a reserve".into());
        }
        if let Some(src) = self.input_from
            && src >= layers
        {
            return bad(format!("input_from {src} outside 0..{layers}"));
        }
        Ok(())
    }
}

/// Sliding statistics of a layer, collected during training.
#[derive(Debug, Clone, PartialEq)]
pub struct LayerStats {
    /// Input rate (spikes per step), EMA.
    pub pre_rate: Vec<f32>,
    /// Output rate, EMA.
    pub post_rate: Vec<f32>,
    /// Co-activity `pre·post` for the pair `(i, o)`, EMA.
    pub coactivity: Vec<f32>,
    /// Importance `|w·∂L/∂w|`, EMA.
    pub saliency: Vec<f32>,
    /// How many times `observe_activity` was called.
    pub activity_samples: u64,
    /// How many times `observe_grads` was called.
    pub grad_samples: u64,
}

impl LayerStats {
    pub(crate) fn new(spec: &LayerSpec) -> Self {
        Self {
            pre_rate: vec![0.0; spec.in_features],
            post_rate: vec![0.0; spec.out_features],
            coactivity: vec![0.0; spec.weight_len()],
            saliency: vec![0.0; spec.weight_len()],
            activity_samples: 0,
            grad_samples: 0,
        }
    }

    pub(crate) fn reset_pair(&mut self, idx: usize) {
        self.coactivity[idx] = 0.0;
        self.saliency[idx] = 0.0;
    }

    pub(crate) fn reset_neuron(&mut self, o: usize, spec: &LayerSpec) {
        self.post_rate[o] = 0.0;
        for i in 0..spec.in_features {
            self.reset_pair(i * spec.out_features + o);
        }
    }
}

/// Layer state: which connections and neurons exist.
#[derive(Debug, Clone, PartialEq)]
pub struct LayerState {
    pub(crate) spec: LayerSpec,
    pub(crate) syn_active: Vec<bool>,
    pub(crate) syn_born: Vec<u64>,
    pub(crate) neuron_active: Vec<bool>,
    pub(crate) neuron_born: Vec<u64>,
    pub(crate) stats: LayerStats,
}

impl LayerState {
    pub(crate) fn new(spec: LayerSpec, rng: &mut Rng64) -> Self {
        let mut syn_active = vec![false; spec.weight_len()];
        let mut neuron_active = vec![false; spec.out_features];
        for a in neuron_active.iter_mut().take(spec.initial_active) {
            *a = true;
        }
        for i in 0..spec.in_features {
            for o in 0..spec.initial_active {
                if spec.initial_density >= 1.0 || rng.chance(spec.initial_density) {
                    syn_active[i * spec.out_features + o] = true;
                }
            }
        }
        // Every active neuron needs at least one input, otherwise it is dead from birth.
        for o in 0..spec.initial_active {
            if (0..spec.in_features).all(|i| !syn_active[i * spec.out_features + o]) {
                let i = rng.below(spec.in_features);
                syn_active[i * spec.out_features + o] = true;
            }
        }
        let stats = LayerStats::new(&spec);
        Self {
            syn_born: vec![0; spec.weight_len()],
            neuron_born: vec![0; spec.out_features],
            spec,
            syn_active,
            neuron_active,
            stats,
        }
    }

    /// Layer description.
    pub fn spec(&self) -> &LayerSpec {
        &self.spec
    }

    /// Activity flags of output neurons.
    pub fn neuron_active(&self) -> &[bool] {
        &self.neuron_active
    }

    /// Structural connection flags (excluding neurons).
    pub fn synapse_flags(&self) -> &[bool] {
        &self.syn_active
    }

    /// Statistics.
    pub fn stats(&self) -> &LayerStats {
        &self.stats
    }

    /// Number of active output neurons.
    pub fn active_neurons(&self) -> usize {
        self.neuron_active.iter().filter(|&&a| a).count()
    }
}

/// Change of a single connection: "set the flag and the weight".
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SynChange {
    /// Index in the `[in, out]` matrix (`i * out + o`).
    pub idx: usize,
    /// New structural flag.
    pub active: bool,
    /// Weight to write into the tensor.
    pub weight: f32,
}

/// Neuron change.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct NeuronChange {
    /// Output neuron number.
    pub o: usize,
    /// New activity flag.
    pub active: bool,
}

/// Changes of a single layer. Applied as a whole; the reverse change is built from
/// the previous values and applied by the same code.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct LayerUpdate {
    /// Layer number.
    pub layer: usize,
    /// Connections.
    pub syn: Vec<SynChange>,
    /// Neurons.
    pub neurons: Vec<NeuronChange>,
}

impl LayerUpdate {
    /// Whether there are any changes.
    pub fn is_empty(&self) -> bool {
        self.syn.is_empty() && self.neurons.is_empty()
    }

    /// Write the weights from the changes into a flat `[in, out]` buffer.
    pub fn write_weights(&self, weights: &mut [f32]) -> StructuralResult<()> {
        for c in &self.syn {
            let n = weights.len();
            *weights.get_mut(c.idx).ok_or_else(|| {
                StructuralError::InvariantViolation(format!(
                    "connection index {} outside a tensor of length {n}",
                    c.idx
                ))
            })? = c.weight;
        }
        Ok(())
    }
}
