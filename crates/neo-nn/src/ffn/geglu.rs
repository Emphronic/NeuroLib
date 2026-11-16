//! GEGLU — Gated GELU (Shazeer, «GLU Variants Improve Transformer», 2020,
//! arXiv:2002.05202).
//!
//! Regular FFN: `Linear → GELU → Linear`. GEGLU replaces the first projector
//! two parallel ones, one of which serves as a “gate”:
//!
//! ```text
//!                     ┌── Linear A ── GELU ──┐
//!  x ─────────────────┤                      ⊙ ── Linear out ── y
//!                     └── Linear B ──────────┘
//! ```
//!
//! ```text
//! A = W_a·x (+ b_a)
//! B = W_b·x (+ b_b)
//! H = GELU(A) ⊙ B
//! y = W_o·H (+ b_o)
//! ```
//!
//! where `⊙` is element-wise multiplication. The layer works along the last axis and
//! accepts a tensor of any rank `D >= 1` (`[batch, seq, d_model]`,
//! `[batch, d_model]`, etc.).
//!
//! About the size of the hidden layer: GEGLU has three matrices instead of two, so in order
//! the number of parameters coincides with the usual FFN with `d_ff`, take
//! `d_hidden ≈ 2/3 · d_ff` (see [`GegluConfig::with_param_parity`]).
//!
//! The general case with activation selection is [`super::gated::GatedFfn`].
//!
//! # Example
//!
//! ```ignore
//! use burn::tensor::Device;
//! use burn::tensor::Tensor;
//! use neo_nn::ffn::GegluConfig;
//!
//! let device = Device::flex(); // CPU backend (feature "flex")
//!
//! let ffn = GegluConfig::new(64, 256).init(&device);
//! let x = Tensor::<3>::zeros([2, 10, 64], &device);
//! let y = ffn.forward(x); // [2, 10, 64]
//! ```

use burn::nn::{Linear, LinearConfig};
use burn::prelude::*;
use burn::tensor::activation::gelu;

/// Configuration [`Geglu`].
#[derive(Config, Debug)]
pub struct GegluConfig {
    /// Input and output size (`d_model`).
    pub d_model: usize,
    /// Hidden layer size (width of both branches A and B).
    pub d_hidden: usize,
    /// Whether to use biases in all three linear layers.
    ///
    /// In the original article and in most LLM (LLaMA, PaLM) bias is not
    /// is used, but the default is `true` as in `nn::Linear`.
    #[config(default = true)]
    pub bias: bool,
}

impl GegluConfig {
    /// Select `d_hidden` so that the number of parameters approximately coincides with
    /// conventional two-layer FFN with intermediate size `d_ff`:
    /// `d_hidden = round(2/3 · d_ff)`.
    pub fn with_param_parity(d_model: usize, d_ff: usize) -> Self {
        Self::new(d_model, (2 * d_ff + 1) / 3)
    }

    /// Initialize the layer on the specified device.
    pub fn init(&self, device: &Device) -> Geglu {
        Geglu {
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

/// GEGLU layer: `y = W_o · (GELU(W_a·x) ⊙ (W_b·x))`.
#[derive(Module, Debug)]
pub struct Geglu {
    /// Branch with GELU activation (`W_a`).
    pub linear_a: Linear,
    /// Gate branch without activation (`W_b`).
    pub linear_b: Linear,
    /// The output projection is back to `d_model` (`W_o`).
    pub linear_out: Linear,
}

impl Geglu {
    /// Direct passage.
    ///
    /// *input: `[..., d_model]`
    /// * output: `[..., d_model]`
    pub fn forward<const D: usize>(&self, x: Tensor<D>) -> Tensor<D> {
        let a = self.linear_a.forward(x.clone());
        let b = self.linear_b.forward(x);

        let h = gelu(a).mul(b);

        self.linear_out.forward(h)
    }
}
