//! Spike KAN layer: trace + KAN + shortcut + mixing + norm + LIF.

use burn::module::{Module, Param};
use burn::nn::{LayerNorm, LayerNormConfig, Linear, LinearConfig};
use burn::tensor::Device;
use burn::tensor::Tensor;
use burn::tensor::activation::sigmoid;

use crate::event_driven::EventDrivenKANSynapse;
use crate::neuron::{Leaky, LeakyConfig};
use crate::state::{LayerState, NeuronState, ResetMode};
use crate::surrogate::Surrogate;

/// KAN based spike layer with optional residual shortcut.
#[derive(Module, Debug)]
pub struct EventDrivenSpikingKANLayer {
    /// KAN synapse (with optional delta gating).
    pub kan_synapse: EventDrivenKANSynapse,
    /// Residual projection if `in_features != out_features`.
    pub shortcut: Option<Linear>,
    /// Raw mixing ratio of KAN path and shortcut path.
    pub kan_weight: Param<Tensor<1>>,
    /// Normalization before LIF.
    pub layer_norm: LayerNorm,
    /// Weekend LIF.
    pub lif: Leaky,
    #[module(skip)]
    trace_alpha: f32,
    #[module(skip)]
    in_features: usize,
    #[module(skip)]
    out_features: usize,
}

impl EventDrivenSpikingKANLayer {
    /// Create a layer.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        in_features: usize,
        out_features: usize,
        trace_alpha: f32,
        delta: f32,
        kan_grid_size: usize,
        use_residual: bool,
        device: &Device,
    ) -> Self {
        let kan_synapse = EventDrivenKANSynapse::new(
            in_features,
            out_features,
            kan_grid_size,
            delta,
            true,
            device,
        );

        let shortcut = if use_residual && in_features != out_features {
            Some(LinearConfig::new(in_features, out_features).init(device))
        } else {
            None
        };

        let kan_weight = Param::from_tensor(Tensor::from_floats([0.1], device));
        let layer_norm = LayerNormConfig::new(out_features).init(device);

        let lif = Leaky::new(
            &LeakyConfig {
                beta_init: 0.95,
                threshold: 1.0,
                learn_threshold: false,
                reset_mode: ResetMode::Subtract,
                reset_order: crate::state::ResetOrder::BeforeDecay,
                inhibition: false,
                graded_spikes_factor: 1.0,
                learn_graded_spikes_factor: false,
            },
            device,
        );

        Self {
            kan_synapse,
            shortcut,
            kan_weight,
            layer_norm,
            lif,
            trace_alpha,
            in_features,
            out_features,
        }
    }

    /// Input size.
    pub fn in_features(&self) -> usize {
        self.in_features
    }

    /// Output size.
    pub fn out_features(&self) -> usize {
        self.out_features
    }

    /// Direct passage of one time step.
    pub fn forward<S: Surrogate + ?Sized>(
        &self,
        x: Tensor<2>,
        state: &LayerState,
        surrogate: &S,
    ) -> (Tensor<2>, LayerState) {
        let trace_new =
            state.trace.clone().mul_scalar(self.trace_alpha) + x.mul_scalar(1.0 - self.trace_alpha);

        // The reference `ref_trace`/`ref_output` are returned by the synapse taking into account
        // gate: they change only where KAN was actually recalculated.
        let (kan_out, ref_trace, ref_output) = self.kan_synapse.forward(
            trace_new.clone(),
            state.last_trace.clone(),
            state.last_output.clone(),
        );

        let synaptic_current = match &self.shortcut {
            Some(sc) => {
                let short = sc.forward(trace_new.clone());
                // Mixing is kept by TENSOR, and is not read on the CPU via
                // into_data() at each time step (as was the case in
                // original) - eliminates GPU↔CPU synchronization internally
                // time cycle.
                let mix = sigmoid(self.kan_weight.val()).reshape([1, 1]);
                kan_out.clone().mul(mix.clone()) + short.mul(mix.mul_scalar(-1.0).add_scalar(1.0))
            }
            None => kan_out.clone(),
        };

        let normed = self.layer_norm.forward(synaptic_current);

        let neuron_state = NeuronState {
            mem: state.mem.clone(),
            syn: None,
        };
        let (spike, new_neuron_state) = self.lif.step(normed, &neuron_state, surrogate);

        let new_state = LayerState {
            trace: trace_new.clone(),
            mem: new_neuron_state.mem,
            last_trace: Some(ref_trace),
            last_output: Some(ref_output),
        };

        (spike, new_state)
    }
}
