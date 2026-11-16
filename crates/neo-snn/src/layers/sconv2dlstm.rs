//! `SConv2dLSTM`: spiking convolutional LSTM.
//!
//! A ConvLSTM whose hidden output is passed through a spiking
//! nonlinearity. Input `[B, C_in, H, W]` → spikes `[B, C_hidden, H, W]`.
//!
//! ## How it differs from [`SLSTM`](crate::slstm::SLSTM)
//!
//! In a regular LSTM the state is a vector, and the spatial structure of the
//! input is lost at the very first `Linear`. Here all four gates are computed
//! by convolution, so the state stays a map `[C, H, W]` and connections are
//! local. For video streams and event-camera data this is exactly what is
//! needed: neighboring pixels are correlated, and flattening them into a
//! vector means discarding precisely the structure the convolution was chosen
//! for.
//!
//! ## Why two convolutions instead of one over concatenated input
//!
//! Reference implementations compute the gates with a single convolution
//! over `concat([x, h])`. Convolution is linear, so this is identical to the
//! sum of two convolutions — over the input and over the hidden state — with
//! the same kernels. The split changes nothing numerically, but avoids
//! concatenating tensors at every step and makes the parameter structure
//! explicit: [`RLeaky`](crate::neuron::RLeaky) with its `linear_in`/`linear_rec`
//! is organized in exactly the same way.

use burn::module::Module;
use burn::nn::PaddingConfig2d;
use burn::nn::conv::{Conv2d, Conv2dConfig};
use burn::tensor::Device;
use burn::tensor::Tensor;
use burn::tensor::activation::sigmoid;

use crate::neuron::spike_with_scalar_threshold;
use crate::state::ResetMode;
use crate::surrogate::Surrogate;

/// ConvLSTM state: hidden map and cell map.
///
/// Both are `[batch, hidden_channels, height, width]`.
#[derive(Debug, Clone)]
pub struct ConvLstmState {
    /// Hidden state (also the previous spikes).
    pub h: Tensor<4>,
    /// Cell state.
    pub c: Tensor<4>,
}

impl ConvLstmState {
    /// Zero state for the given map shape.
    pub fn zeros(
        device: &Device,
        batch: usize,
        hidden_channels: usize,
        height: usize,
        width: usize,
    ) -> Self {
        let shape = [batch, hidden_channels, height, width];
        Self {
            h: Tensor::zeros(shape, device),
            c: Tensor::zeros(shape, device),
        }
    }

    /// Detach the state from the graph (truncated BPTT).
    ///
    /// Without this the graph grows over the whole sequence — the same
    /// reason as in [`NeuronState::detach`](crate::state::NeuronState::detach).
    pub fn detach(self) -> Self {
        Self {
            h: self.h.detach(),
            c: self.c.detach(),
        }
    }
}

/// Spiking convolutional LSTM.
#[derive(Module, Debug)]
pub struct SConv2dLSTM {
    /// Convolution over the input: produces all four gates at once.
    pub conv_x: Conv2d,
    /// Convolution over the hidden state, without bias — it is already in `conv_x`.
    pub conv_h: Conv2d,
    /// Number of hidden channels (a structural constant, not a parameter).
    #[module(skip)]
    pub hidden_channels: usize,
    /// Spike threshold.
    pub threshold: f32,
    /// Mode for resetting the cell state after a spike.
    #[module(skip)]
    pub reset_mode: ResetMode,
}

impl SConv2dLSTM {
    /// Create the layer.
    ///
    /// `kernel_size` is taken odd so that `Same` padding preserves the
    /// spatial dimensions.
    pub fn new(
        device: &Device,
        in_channels: usize,
        hidden_channels: usize,
        kernel_size: usize,
        threshold: f32,
        reset_mode: ResetMode,
    ) -> Self {
        let gates = 4 * hidden_channels;

        let conv_x = Conv2dConfig::new([in_channels, gates], [kernel_size, kernel_size])
            .with_padding(PaddingConfig2d::Same)
            .init(device);

        let conv_h = Conv2dConfig::new([hidden_channels, gates], [kernel_size, kernel_size])
            .with_padding(PaddingConfig2d::Same)
            .with_bias(false)
            .init(device);

        Self {
            conv_x,
            conv_h,
            hidden_channels,
            threshold,
            reset_mode,
        }
    }

    /// One time step.
    ///
    /// `input` — `[batch, in_channels, h, w]`.
    /// Returns `(spikes [batch, hidden_channels, h, w], new state)`.
    pub fn step<S: Surrogate + ?Sized>(
        &self,
        input: Tensor<4>,
        state: &ConvLstmState,
        surrogate: &S,
    ) -> (Tensor<4>, ConvLstmState) {
        let gates = self.conv_x.forward(input) + self.conv_h.forward(state.h.clone());
        let [batch, _, height, width] = gates.dims();
        let hc = self.hidden_channels;

        // Gate order is the same as in reference ConvLSTM implementations:
        // input, forget, cell, output.
        let slice_gate = |idx: usize| {
            gates
                .clone()
                .slice([0..batch, idx * hc..(idx + 1) * hc, 0..height, 0..width])
        };

        let i = sigmoid(slice_gate(0));
        let f = sigmoid(slice_gate(1));
        let g = slice_gate(2).tanh();
        let o = sigmoid(slice_gate(3));

        let c_new = f.mul(state.c.clone()) + i.mul(g);
        let h_pre = o.mul(c_new.clone().tanh());

        // The surrogate works with rank 2, so the map is flattened into
        // [batch·h·w, channels]. permute BEFORE reshape is mandatory: reshape
        // does not transpose data, and without it channels would be mixed with
        // spatial positions — the same case as in
        // [`SpikingConv2d`](super::SpikingConv2d).
        let flat = h_pre
            .permute([0, 2, 3, 1])
            .reshape([batch * height * width, hc]);

        let (spike_flat, spike_hard_flat) =
            spike_with_scalar_threshold(flat, self.threshold, surrogate);

        let to_map = |t: Tensor<2>| t.reshape([batch, height, width, hc]).permute([0, 3, 1, 2]);

        let spikes = to_map(spike_flat);
        let spike_hard = to_map(spike_hard_flat);

        // Reset is applied to the CELL state: it is the one that accumulates
        // charge between steps, and without a reset a spike has no consequences.
        let c_reset = match self.reset_mode {
            ResetMode::Subtract => c_new - spike_hard.mul_scalar(self.threshold),
            ResetMode::Zero => c_new.mul(spike_hard.mul_scalar(-1.0).add_scalar(1.0)),
            ResetMode::None => c_new,
        };

        (
            spikes.clone(),
            ConvLstmState {
                h: spikes,
                c: c_reset,
            },
        )
    }
}
