//! ReGLU — Gated ReLU (Shazeer, «GLU Variants Improve Transformer», 2020,
//! arXiv:2002.05202).
//!
//! Same as [`super::geglu`], but ReLU is used instead of GELU:
//!
//! ```text
//!                     ┌── Linear A ── ReLU ──┐
//!  x ─────────────────┤                      ⊙ ── Linear out ── y
//!                     └── Linear B ──────────┘
//! ```
//!
//! ```text
//! A = W_a·x (+ b_a)
//! B = W_b·x (+ b_b)
//! H = ReLU(A) ⊙ B
//! y = W_o·H (+ b_o)
//! ```
//!
//! The layer works along the last axis and accepts a tensor of any rank `D >= 1`.
//! The general case with activation selection is [`super::gated::GatedFfn`].
//!
//! # Example
//!
//! ```ignore
//! use burn::tensor::Device;
//! use burn::tensor::Tensor;
//! use neo_nn::ffn::RegluConfig;
//!
//! let device = Device::flex(); // CPU backend (feature "flex")
//!
//! let ffn = RegluConfig::new(64, 256).init(&device);
//! let y = ffn.forward(Tensor::<3>::zeros([2, 10, 64], &device));
//! ```

use burn::nn::{Linear, LinearConfig};
use burn::prelude::*;
use burn::tensor::activation::relu;

/// Configuration [`Reglu`].
#[derive(Config, Debug)]
pub struct RegluConfig {
    /// Input and output size (`d_model`).
    pub d_model: usize,
    /// Hidden layer size (width of both branches A and B).
    pub d_hidden: usize,
    /// Whether to use biases in all three linear layers.
    #[config(default = true)]
    pub bias: bool,
}

impl RegluConfig {
    /// Select `d_hidden = round(2/3 · d_ff)` so that the number of parameters
    /// approximately coincided with a conventional two-layer FFN with size `d_ff`.
    pub fn with_param_parity(d_model: usize, d_ff: usize) -> Self {
        Self::new(d_model, (2 * d_ff + 1) / 3)
    }

    /// Initialize the layer on the specified device.
    pub fn init(&self, device: &Device) -> Reglu {
        Reglu {
            linear_a: LinearConfig::new(self.d_model, self.d_hidden)
                .with_bias(self.bias)
                .init(device),
            linear_b: LinearConfig::new(self.d_model, self.d_hidden)
                .with_bias(self.bias)
                .init(device),
            linear_out: LinearConfig::new(self.d_hidden, self.d_model)
                .with_bias(self.bias)
                .init(device),
        }
    }
}

/// ReGLU layer: `y = W_o · (ReLU(W_a·x) ⊙ (W_b·x))`.
#[derive(Module, Debug)]
pub struct Reglu {
    /// Branch with ReLU activation (`W_a`).
    pub linear_a: Linear,
    /// Gate branch without activation (`W_b`).
    pub linear_b: Linear,
    /// The output projection is back to `d_model` (`W_o`).
    pub linear_out: Linear,
}

impl Reglu {
    /// Direct passage.
    ///
    /// *input: `[..., d_model]`
    /// * output: `[..., d_model]`
    pub fn forward<const D: usize>(&self, x: Tensor<D>) -> Tensor<D> {
        let a = self.linear_a.forward(x.clone());
        let b = self.linear_b.forward(x);

        let h = relu(a).mul(b);

        self.linear_out.forward(h)
    }
}
