//! Event-driven full SKAN: `input_fc → norm → LIF → layer1 → layer2 → readout`.

use burn::module::{Module, Param};
use burn::nn::{LayerNorm, LayerNormConfig, Linear, LinearConfig};
use burn::tensor::Device;
use burn::tensor::Tensor;

use super::EventDrivenSpikingKANLayer;
use crate::neuron::{Leaky, LeakyConfig};
use crate::state::{LayerState, NeuronState, ResetMode};
use crate::surrogate::Surrogate;

/// Complete event-driven SKAN network: input → hidden SKAN layers → readout.
#[derive(Module, Debug)]
pub struct EventDrivenFullSKAN {
    /// Input projection.
    pub input_fc: Linear,
    /// Input normalization.
    pub input_norm: LayerNorm,
    /// Input LIF.
    pub input_lif: Leaky,
    /// The first hidden SKAN layer.
    pub layer1: EventDrivenSpikingKANLayer,
    /// Second hidden SKAN layer.
    pub layer2: EventDrivenSpikingKANLayer,
    /// Trainable membrane readout-accumulation weight.
    pub output_weight: Param<Tensor<1>>,
    #[module(skip)]
    num_steps: usize,
    #[module(skip)]
    hidden1: usize,
    #[module(skip)]
    hidden2: usize,
    #[module(skip)]
    output_size: usize,
}

impl EventDrivenFullSKAN {
    /// Create a network.
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        input_size: usize,
        hidden1: usize,
        hidden2: usize,
        output_size: usize,
        num_steps: usize,
        beta: f32,
        trace_alpha: f32,
        delta: f32,
        kan_grid_size: usize,
        device: &Device,
    ) -> Self {
        assert!(num_steps > 0, "num_steps must be > 0");

        let input_fc = LinearConfig::new(input_size, hidden1).init(device);
        let input_norm = LayerNormConfig::new(hidden1).init(device);
        let input_lif = Leaky::new(
            &LeakyConfig {
                beta_init: beta,
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

        let layer1 = EventDrivenSpikingKANLayer::new(
            hidden1,
            hidden2,
            trace_alpha,
            delta,
            kan_grid_size,
            true,
            device,
        );
        let layer2 = EventDrivenSpikingKANLayer::new(
            hidden2,
            output_size,
            trace_alpha,
            delta,
            kan_grid_size,
            false,
            device,
        );

        let output_weight = Param::from_tensor(Tensor::ones([output_size], device));

        Self {
            input_fc,
            input_norm,
            input_lif,
            layer1,
            layer2,
            output_weight,
            num_steps,
            hidden1,
            hidden2,
            output_size,
        }
    }

    /// Direct passage through all `num_steps` time steps.
    ///
    /// Input `x`: `[batch, input_size]`, supplied unchanged at each step
    /// (static image/vector repeated over time - typical
    /// pattern for SNN on static data, as in the original). Returns
    /// `(readout [batch, output_size], spike_counter [batch, output_size])`.
    pub fn forward<S: Surrogate + ?Sized>(
        &self,
        x: Tensor<2>,
        surrogate: &S,
    ) -> (Tensor<2>, Tensor<2>) {
        let device = x.device();
        let batch = x.dims()[0];

        let mut input_state = NeuronState::zeros_mem_only(&device, batch, self.hidden1);
        let mut layer1_state = LayerState::zeros(&device, batch, self.hidden1, self.hidden2, true);
        let mut layer2_state =
            LayerState::zeros(&device, batch, self.hidden2, self.output_size, true);

        let mut mem_accum: Tensor<2> = Tensor::zeros([batch, self.output_size], &device);
        let mut spike_count: Tensor<2> = Tensor::zeros([batch, self.output_size], &device);

        // The input projection and its normalization do not change from step to step
        // (x is the same at each step) - taken out of the loop, unlike
        // original, where `input_fc`/`input_norm` were recalculated
        // num_steps times is wasted.
        let cur_input = self.input_norm.forward(self.input_fc.forward(x));

        for _ in 0..self.num_steps {
            let (spk_input, new_input_state) =
                self.input_lif
                    .step(cur_input.clone(), &input_state, surrogate);
            input_state = new_input_state;

            let (spk1, new_l1) = self.layer1.forward(spk_input, &layer1_state, surrogate);
            layer1_state = new_l1;

            let (spk2, new_l2) = self.layer2.forward(spk1, &layer2_state, surrogate);
            layer2_state = new_l2;

            mem_accum = mem_accum + layer2_state.mem.clone();
            spike_count = spike_count + spk2;
        }

        let readout = mem_accum.div_scalar(self.num_steps as f32).mul(
            self.output_weight
                .val()
                .reshape([1, self.output_size])
                .expand([batch, self.output_size]),
        ) + spike_count.clone();

        (readout, spike_count)
    }
}
