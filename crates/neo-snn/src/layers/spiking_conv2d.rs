//! `SpikingConv2d`: Conv2d followed by LIF.
//! Input `[B, C_in, H, W]` → spikes `[B, C_out, H', W']`.

use burn::module::Module;
use burn::nn::conv::{Conv2d, Conv2dConfig};
use burn::tensor::Device;
use burn::tensor::Tensor;

use crate::neuron::{Leaky, LeakyConfig};
use crate::state::NeuronState;
use crate::surrogate::Surrogate;

/// Conv2d + LIF. LIF is applied per channel at each spatial location.
#[derive(Module, Debug)]
pub struct SpikingConv2d {
    /// Convolutional layer.
    pub conv: Conv2d,
    /// LIF layer on top of the convolution output.
    pub lif: Leaky,
}

impl SpikingConv2d {
    /// Create the layer.
    pub fn new(conv_config: Conv2dConfig, lif_config: &LeakyConfig, device: &Device) -> Self {
        Self {
            conv: conv_config.init(device),
            lif: Leaky::new(lif_config, device),
        }
    }

    /// One time step.
    ///
    /// `input`: `[batch, c_in, h, w]`. `state.mem` must have the shape
    /// `[batch * h' * w', c_out]` — see [`SpikingConv2d::state_zeros`].
    /// Returns `(spikes [batch, c_out, h', w'], new state)`.
    pub fn step<S: Surrogate + ?Sized>(
        &self,
        input: Tensor<4>,
        state: &NeuronState,
        surrogate: &S,
    ) -> (Tensor<4>, NeuronState) {
        let out = self.conv.forward(input);
        self.step_from_conv_out(out, state, surrogate)
    }

    /// Second half of [`step`](Self::step): LIF on top of the ALREADY computed
    /// convolution output `[batch, c_out, h', w']`.
    ///
    /// Split out so that `SpikingLayer::step` can first discover the output
    /// shape (for lazy state) without running the convolution twice.
    pub(crate) fn step_from_conv_out<S: Surrogate + ?Sized>(
        &self,
        out: Tensor<4>,
        state: &NeuronState,
        surrogate: &S,
    ) -> (Tensor<4>, NeuronState) {
        let [batch, c_out, h, w] = out.dims();

        // The convolution produces layout [batch, channel, h, w]; LIF works with
        // [batch, units]. permute moves channel to the last axis BEFORE
        // reshape: reshaping directly from [b,c,h,w] would silently mix spatial
        // positions and channels together, because reshape does not transpose
        // data, it simply reinterprets their linear order in memory (the same
        // pitfall as "reshape instead of transpose" in PyTorch).
        let flat = out
            .clone()
            .permute([0, 2, 3, 1])
            .reshape([batch * h * w, c_out]);

        let (spikes_flat, new_state) = self.lif.step(flat, state, surrogate);

        let spikes = spikes_flat
            .reshape([batch, h, w, c_out])
            .permute([0, 3, 1, 2]);

        (spikes, new_state)
    }

    /// Zero state for the output shape `(batch, c_out, h, w)`.
    ///
    /// `(h, w)` are the spatial dimensions AFTER the convolution; they are
    /// easiest to obtain with a single trial `self.conv.forward(...)` on an
    /// input of the required shape before the main time loop.
    pub fn state_zeros(
        device: &Device,
        batch: usize,
        c_out: usize,
        h: usize,
        w: usize,
    ) -> NeuronState {
        NeuronState::zeros_mem_only(device, batch * h * w, c_out)
    }
}
