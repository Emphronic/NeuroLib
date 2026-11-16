//! Spike network obtained by conversion: [`SpikingNet`].
//!
//! Neuron - integrate-and-fire without leakage (IF): `V += I`, spike at
//! `V ≥ 1`, then reset by subtracting the threshold ([`ResetMode::Subtract`]) or
//! zeroing. The threshold is always `1` - the scale is already taken into account in the scales by normalization.
//! The initial membrane `v_init` (default `0.5`) turns truncation
//! frequencies in rounding and noticeably reduces the error at small `T`.

use burn::tensor::{Distribution, Tensor};

use super::ann::{Act, pool};
use crate::error::{Result, SnnError, tensor_to_vec};
use crate::state::ResetMode;

use burn::module::Param;
use burn::nn::Linear;
use burn::nn::conv::Conv2d;

/// How to feed the input of the first layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum InputEncoding {
    /// The input is supplied with current at each step as is (direct coding).
    /// More precisely: the first layer performs MACs, not ACs.
    #[default]
    Analog,
    /// Bernoulli spikes with probability `x ∈ [0, 1]` at each step. All layers
    /// networks work on spikes, but larger ones are needed `T` - sampling error
    /// decreases as `1/√T`. The input must be in `[0, 1]`.
    Rate,
}

/// Parameters of neurons and input of the spike network.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SpikeConfig {
    /// Membrane reset method (`None` not allowed).
    pub reset: ResetMode,
    /// The initial membrane in fractions of the threshold, `[0, 1)`.
    pub v_init: f32,
    /// Input coding.
    pub input: InputEncoding,
}

impl Default for SpikeConfig {
    fn default() -> Self {
        Self {
            reset: ResetMode::Subtract,
            v_init: 0.5,
            input: InputEncoding::Analog,
        }
    }
}

impl SpikeConfig {
    pub(crate) fn validate(&self) -> Result<()> {
        if matches!(self.reset, ResetMode::None) {
            return Err(SnnError::InvalidConfig {
                what: "SpikeConfig::reset",
                reason: "ResetMode::None is not suitable for conversion: without a reset, the frequency does not correspond to ReLU".into(),
            });
        }
        if !(self.v_init >= 0.0 && self.v_init < 1.0) {
            return Err(SnnError::InvalidConfig {
                what: "SpikeConfig::v_init",
                reason: format!(
                    "the initial membrane must lie in [0, 1), obtained {}",
                    self.v_init
                ),
            });
        }
        Ok(())
    }
}

#[derive(Debug, Clone)]
pub(crate) enum SnnLayer {
    Linear {
        lin: Linear,
        spiking: bool,
    },
    Conv {
        conv: Conv2d,
        spiking: bool,
    },
    AvgPool {
        kernel: [usize; 2],
        stride: [usize; 2],
    },
    Flatten,
}

/// A network of IF neurons with weights recalculated from ANN.
#[derive(Debug, Clone)]
pub struct SpikingNet {
    pub(crate) layers: Vec<SnnLayer>,
    pub(crate) cfg: SpikeConfig,
    /// `λ` of each spike layer (in layer order).
    pub(crate) scales: Vec<f32>,
    /// Output multiplier: `λ` of the last layer if the network ends
    /// with a spike layer, otherwise `1`.
    pub(crate) out_scale: f32,
}

/// Network state between steps: membranes, spike counters, output sum.
#[derive(Debug, Clone)]
pub struct SnnState {
    mem: Vec<Option<Act>>,
    spike_sum: Vec<Option<Tensor<1>>>,
    out_sum: Option<Act>,
    steps: usize,
}

impl SnnState {
    /// How many steps have already been completed?
    pub fn steps(&self) -> usize {
        self.steps
    }
}

/// Statistics of spikes of one spike layer per run.
#[derive(Debug, Clone, PartialEq)]
pub struct LayerStats {
    /// Number among the spike layers (from scratch).
    pub layer: usize,
    /// Neurons per batch example.
    pub neurons: usize,
    /// Total spikes per run (by batch and time).
    pub spikes: f32,
    /// Average frequency: spikes per neuron per step, `[0, 1]`.
    pub rate: f32,
}

/// Result [`SpikingNet::run`].
#[derive(Debug, Clone)]
pub struct SnnOutput {
    /// Output averaged over time and reduced to ANN units.
    pub output: Act,
    /// The same output, averaged over the first `t` steps, for `t = 1..=T`.
    pub per_step: Vec<Act>,
    /// Spike statistics by layer.
    pub stats: Vec<LayerStats>,
}

impl SnnOutput {
    /// Total spikes across all layers (a measure of “energy” in the number of AC operations).
    pub fn total_spikes(&self) -> f32 {
        self.stats.iter().map(|s| s.spikes).sum()
    }

    /// Classification accuracy after each step `t = 1..=T`
    /// (`argmax` by output `[batch, classes]` versus `labels`).
    pub fn accuracy_over_time(&self, labels: &[usize]) -> Result<Vec<f32>> {
        self.per_step
            .iter()
            .map(|act| {
                let (b, c) = match act.dims().as_slice() {
                    [b, c] => (*b, *c),
                    d => {
                        return Err(SnnError::shape_msg(
                            "accuracy_over_time",
                            "[batch, classes]",
                            d,
                        ));
                    }
                };
                if labels.len() != b {
                    return Err(SnnError::shape("accuracy_over_time", [b], [labels.len()]));
                }
                let v = act.to_vec("accuracy_over_time")?;
                let correct = (0..b)
                    .filter(|&i| argmax(&v[i * c..(i + 1) * c]) == labels[i])
                    .count();
                Ok(correct as f32 / b as f32)
            })
            .collect()
    }
}

/// Maximum index (first in case of equality).
pub fn argmax(row: &[f32]) -> usize {
    let mut best = 0;
    for (i, v) in row.iter().enumerate() {
        if *v > row[best] {
            best = i;
        }
    }
    best
}

impl SpikingNet {
    /// Neuron parameters.
    pub fn config(&self) -> &SpikeConfig {
        &self.cfg
    }

    /// Scales `λ` of spike layers.
    pub fn scales(&self) -> &[f32] {
        &self.scales
    }

    /// Number of spike layers.
    pub fn spiking_layers(&self) -> usize {
        self.scales.len()
    }

    /// Zero state (shapes are created lazily in the first step).
    pub fn init_state(&self) -> SnnState {
        let n = self.scales.len();
        SnnState {
            mem: vec![None; n],
            spike_sum: vec![None; n],
            out_sum: None,
            steps: 0,
        }
    }

    /// One step in time. Returns the output of the last layer in this step
    /// (spikes or read current - not to scale) and the new state.
    pub fn step(&self, x: &Act, mut state: SnnState) -> Result<(Act, SnnState)> {
        let mut h = match self.cfg.input {
            InputEncoding::Analog => x.clone(),
            InputEncoding::Rate => match x {
                Act::D2(t) => Act::D2(bernoulli(t.clone())),
                Act::D4(t) => Act::D4(bernoulli(t.clone())),
            },
        };
        let mut si = 0;
        for layer in &self.layers {
            let (cur, spiking) = match layer {
                SnnLayer::Linear { lin, spiking } => (
                    Act::D2(lin.forward(h.into_d2("SpikingNet::Linear")?)),
                    *spiking,
                ),
                SnnLayer::Conv { conv, spiking } => (
                    Act::D4(conv.forward(h.into_d4("SpikingNet::Conv2d")?)),
                    *spiking,
                ),
                SnnLayer::AvgPool { kernel, stride } => {
                    h = Act::D4(pool(h.into_d4("SpikingNet::AvgPool2d")?, *kernel, *stride));
                    continue;
                }
                SnnLayer::Flatten => {
                    h = h.flatten();
                    continue;
                }
            };
            h = if spiking {
                let (spikes, mem, count) = self.neuron(state.mem[si].take(), cur)?;
                state.mem[si] = Some(mem);
                state.spike_sum[si] = Some(match state.spike_sum[si].take() {
                    Some(prev) => prev + count,
                    None => count,
                });
                si += 1;
                spikes
            } else {
                cur
            };
        }
        state.out_sum = Some(match state.out_sum.take() {
            Some(sum) => sum.add(h.clone(), "SpikingNet::step")?,
            None => h.clone(),
        });
        state.steps += 1;
        Ok((h, state))
    }

    /// Running `steps` steps on a constant input `x`.
    pub fn run(&self, x: &Act, steps: usize) -> Result<SnnOutput> {
        if steps == 0 {
            return Err(SnnError::EmptySequence {
                op: "SpikingNet::run",
            });
        }
        let mut state = self.init_state();
        let mut per_step = Vec::with_capacity(steps);
        for t in 1..=steps {
            let (_, next) = self.step(x, state)?;
            state = next;
            if let Some(sum) = &state.out_sum {
                per_step.push(sum.clone().scale(self.out_scale / t as f32));
            }
        }
        let batch = x.batch();
        let mut stats = Vec::with_capacity(self.scales.len());
        for (i, (mem, sum)) in state.mem.iter().zip(&state.spike_sum).enumerate() {
            let (Some(mem), Some(sum)) = (mem, sum) else {
                continue;
            };
            let neurons: usize = mem.dims()[1..].iter().product();
            let spikes = tensor_to_vec(sum.clone(), "SpikingNet::run")?
                .first()
                .copied()
                .unwrap_or(0.0);
            let rate = spikes / (batch * neurons * steps) as f32;
            stats.push(LayerStats {
                layer: i,
                neurons,
                spikes,
                rate,
            });
        }
        let output = per_step.last().cloned().ok_or(SnnError::EmptySequence {
            op: "SpikingNet::run",
        })?;
        Ok(SnnOutput {
            output,
            per_step,
            stats,
        })
    }

    /// Convenient wrapper: output `[batch, classes]` after `steps` steps.
    pub fn predict(&self, x: &Act, steps: usize) -> Result<Tensor<2>> {
        self.run(x, steps)?.output.into_d2("SpikingNet::predict")
    }

    fn neuron(&self, mem: Option<Act>, cur: Act) -> Result<(Act, Act, Tensor<1>)> {
        match cur {
            Act::D2(c) => {
                let m = match mem {
                    Some(m) => Some(m.into_d2("SpikingNet::state")?),
                    None => None,
                };
                let (s, m, n) = if_step(m, c, &self.cfg);
                Ok((Act::D2(s), Act::D2(m), n))
            }
            Act::D4(c) => {
                let m = match mem {
                    Some(m) => Some(m.into_d4("SpikingNet::state")?),
                    None => None,
                };
                let (s, m, n) = if_step(m, c, &self.cfg);
                Ok((Act::D4(s), Act::D4(m), n))
            }
        }
    }
}

/// IF neuron step: `(spikes, new membrane, number of spikes)`.
fn if_step<const D: usize>(
    mem: Option<Tensor<D>>,
    cur: Tensor<D>,
    cfg: &SpikeConfig,
) -> (Tensor<D>, Tensor<D>, Tensor<1>) {
    let mem = match mem {
        Some(m) => m + cur,
        None => cur.add_scalar(cfg.v_init),
    };
    let spike = mem.clone().greater_equal_elem(1.0).float();
    let count = spike.clone().sum();
    let mem = match cfg.reset {
        ResetMode::Subtract => mem - spike.clone(),
        // `None` is eliminated to `SpikeConfig::validate`.
        _ => mem.mul(spike.clone().neg().add_scalar(1.0)),
    };
    (spike, mem, count)
}

fn bernoulli<const D: usize>(x: Tensor<D>) -> Tensor<D> {
    let u = x.random_like(Distribution::Uniform(0.0, 1.0));
    x.greater(u).float()
}

/// Recalculate the layer with weights: `w·k`, `b·kb`.
pub(crate) fn scaled_linear(l: &Linear, k: f32, kb: f32) -> Linear {
    let mut out = l.clone();
    out.weight = Param::from_tensor(l.weight.val().mul_scalar(k));
    out.bias = l
        .bias
        .as_ref()
        .map(|b| Param::from_tensor(b.val().mul_scalar(kb)));
    out
}

pub(crate) fn scaled_conv(c: &Conv2d, k: f32, kb: f32) -> Conv2d {
    let mut out = c.clone();
    out.weight = Param::from_tensor(c.weight.val().mul_scalar(k));
    out.bias = c
        .bias
        .as_ref()
        .map(|b| Param::from_tensor(b.val().mul_scalar(kb)));
    out
}
