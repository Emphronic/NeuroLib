//! Recurrent Synaptic: LIF with synaptic trace and recurrent weights.
//!
//! `I = W_in·x + W_rec·spike_prev`; `syn' = α·syn + I`; `mem' = β·mem + syn`.

use burn::module::{Module, Param};
use burn::nn::{Linear, LinearConfig};
use burn::tensor::Device;
use burn::tensor::Tensor;
use burn::tensor::activation::sigmoid;

use super::{logit, spike_with_scalar_threshold};
use crate::state::{NeuronState, ResetMode, ResetOrder, carry_membrane, decay_membrane};
use crate::surrogate::Surrogate;

/// Recurrent LIF with synaptic trace.
#[derive(Module, Debug)]
pub struct RSynaptic {
    /// Direct projection of the input.
    pub linear_in: Linear,
    /// Recurrent projection of previous spikes.
    pub linear_rec: Linear,
    /// Raw β (membrane attenuation).
    pub beta: Param<Tensor<1>>,
    /// Raw α (synaptic current decay).
    pub alpha_syn: Param<Tensor<1>>,
    /// Spike threshold.
    pub threshold: f32,
    /// Reset mode.
    #[module(skip)]
    pub reset_mode: ResetMode,
    /// The order is “reset ↔ decay” (see [`ResetOrder`]); not included in record.
    #[module(skip)]
    pub reset_order: ResetOrder,
}

impl RSynaptic {
    /// Create a layer.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        device: &Device,
        in_features: usize,
        out_features: usize,
        beta_init: f32,
        alpha_syn_init: f32,
        threshold: f32,
        reset_mode: ResetMode,
    ) -> Self {
        Self {
            linear_in: LinearConfig::new(in_features, out_features).init(device),
            linear_rec: LinearConfig::new(out_features, out_features)
                .with_bias(false)
                .init(device),
            beta: Param::from_tensor(Tensor::from_floats([logit(beta_init)], device)),
            alpha_syn: Param::from_tensor(Tensor::from_floats([logit(alpha_syn_init)], device)),
            threshold,
            reset_mode,
            reset_order: ResetOrder::BeforeDecay,
        }
    }

    /// Set the order “reset ↔ attenuation” (default `BeforeDecay`).
    pub fn with_reset_order(mut self, order: ResetOrder) -> Self {
        self.reset_order = order;
        self
    }

    /// One step in time.
    pub fn step<S: Surrogate + ?Sized>(
        &self,
        input: Tensor<2>,
        spike_prev: Tensor<2>,
        state: &NeuronState,
        surrogate: &S,
    ) -> (Tensor<2>, NeuronState) {
        let dims = state.mem.dims();
        let device = state.mem.device();

        let current = self.linear_in.forward(input) + self.linear_rec.forward(spike_prev);

        let syn = state
            .syn
            .clone()
            .unwrap_or_else(|| Tensor::zeros(dims, &device));

        let b = sigmoid(self.beta.val()).reshape([1, 1]).expand(dims);
        let a = sigmoid(self.alpha_syn.val()).reshape([1, 1]).expand(dims);

        let syn_new = syn.mul(a) + current;
        let mem_new = decay_membrane(
            state.mem.clone(),
            b,
            self.threshold,
            self.reset_mode,
            self.reset_order,
        ) + syn_new.clone();

        let (spike, spike_hard) =
            spike_with_scalar_threshold(mem_new.clone(), self.threshold, surrogate);
        let mem_reset = carry_membrane(
            mem_new,
            &spike_hard,
            self.threshold,
            self.reset_mode,
            self.reset_order,
        );

        (
            spike,
            NeuronState {
                mem: mem_reset,
                syn: Some(syn_new),
            },
        )
    }
}
