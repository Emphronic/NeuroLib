//! Universal gated FFN (GLU family).
//!
//! GEGLU, ReGLU, SwiGLU and others are special cases of one scheme
//! (Shazeer, «GLU Variants Improve Transformer», 2020, arXiv:2002.05202):
//!
//! ```text
//!               ┌── W_g ── activation ──┐
//!  x ───────────┤                       ⊙ ── W_o ── y
//!               └── W_v ────────────────┘
//! ```
//!
//! ```text
//! gate   = activation(W_g·x)
//! value  = W_v·x
//! hidden = gate ⊙ value
//! y      = W_o·hidden
//! ```
//!
//! Selecting activation gives a specific option:
//!
//! | [`GateActivation`] | Title |
//! |--------------------|----------------------------------|
//! | `Gelu`             | GEGLU                            |
//! | `Relu`             | ReGLU                            |
//! | `Silu`             | SwiGLU (LLaMA, Mistral, PaLM)    |
//! | `Sigmoid` | classic GLU (Dauphin, 2017) |
//! | `Tanh`             | tanh-gated MLP                   |
//!
//! The layer works along the last axis and accepts a tensor of any rank.
//!
//! # Example
//!
//! ```ignore
//! use burn::tensor::Device;
//! use burn::tensor::Tensor;
//! use neo_nn::ffn::{GateActivation, GatedFfnConfig};
//!
//! let device = Device::flex(); // CPU backend (feature "flex")
//!
//! let ffn = GatedFfnConfig::swiglu(64, 172).init(&device);
//! let y = ffn.forward(Tensor::<3>::zeros([2, 10, 64], &device));
//! ```

use burn::nn::{Linear, LinearConfig};
use burn::prelude::*;
use burn::tensor::activation::{gelu, relu, sigmoid, silu};

/// Shutter activation.
#[derive(Config, Debug, PartialEq)]
pub enum GateActivation {
    /// GELU → GEGLU.
    Gelu,
    /// ReLU → ReGLU.
    Relu,
    /// SiLU (Swish) → SwiGLU.
    Silu,
    /// Sigmoid → classic GLU.
    Sigmoid,
    /// Hyperbolic tangent.
    Tanh,
}

impl GateActivation {
    /// Apply activation element by element.
    pub fn apply<const D: usize>(&self, x: Tensor<D>) -> Tensor<D> {
        match self {
            GateActivation::Gelu => gelu(x),
            GateActivation::Relu => relu(x),
            GateActivation::Silu => silu(x),
            GateActivation::Sigmoid => sigmoid(x),
            GateActivation::Tanh => x.tanh(),
        }
    }
}

/// Configuration [`GatedFfn`].
#[derive(Config, Debug)]
pub struct GatedFfnConfig {
    /// Input and output size (`d_model`).
    pub d_model: usize,
    /// Hidden layer size (width of `gate` and `value` branches).
    pub d_hidden: usize,
    /// Shutter activation.
    pub activation: GateActivation,
    /// Whether to use offsets in all three linear layers.
    #[config(default = true)]
    pub bias: bool,
}

impl GatedFfnConfig {
    /// GEGLU: GELU activation.
    pub fn geglu(d_model: usize, d_hidden: usize) -> Self {
        Self::new(d_model, d_hidden, GateActivation::Gelu)
    }

    /// ReGLU: ReLU activation.
    pub fn reglu(d_model: usize, d_hidden: usize) -> Self {
        Self::new(d_model, d_hidden, GateActivation::Relu)
    }

    /// SwiGLU: SiLU activation.
    pub fn swiglu(d_model: usize, d_hidden: usize) -> Self {
        Self::new(d_model, d_hidden, GateActivation::Silu)
    }

    /// Select `d_hidden = round(2/3 · d_ff)` so that the number of parameters
    /// approximately coincided with the usual two-layer FFN with size `d_ff`
    /// (the gated version has three matrices instead of two).
    pub fn with_param_parity(d_model: usize, d_ff: usize, activation: GateActivation) -> Self {
        Self::new(d_model, (2 * d_ff + 1) / 3, activation)
    }

    /// Initialize the layer on the specified device.
    pub fn init(&self, device: &Device) -> GatedFfn {
        GatedFfn {
            gate: LinearConfig::new(self.d_model, self.d_hidden)
                .with_bias(self.bias)
                .init(device),
            value: LinearConfig::new(self.d_model, self.d_hidden)
                .with_bias(self.bias)
                .init(device),
            out: LinearConfig::new(self.d_hidden, self.d_model)
                .with_bias(self.bias)
                .init(device),
            activation: self.activation.clone(),
        }
    }
}

/// Universal gated FFN: `y = W_o · (act(W_g·x) ⊙ (W_v·x))`.
#[derive(Module, Debug)]
pub struct GatedFfn {
    /// Shutter projector `W_g` (activation applied to it).
    pub gate: Linear,
    /// Projector value `W_v`.
    pub value: Linear,
    /// Output projection `W_o`.
    pub out: Linear,
    /// Gate activation (not a trainable parameter, not stored in the model record).
    #[module(skip)]
    pub activation: GateActivation,
}

impl GatedFfn {
    /// Direct pass: `[..., d_model] → [..., d_model]`.
    pub fn forward<const D: usize>(&self, x: Tensor<D>) -> Tensor<D> {
        let gate = self.activation.apply(self.gate.forward(x.clone()));
        let value = self.value.forward(x);

        self.out.forward(gate.mul(value))
    }
}
