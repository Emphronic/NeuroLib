//! Parameter-efficient fine-tuning (PEFT): LoRA, DoRA, IA³.
//!
//! ```text
//!   x ─┬─ frozen base ──── y_base ─┐
//!      │                           ├─ combine ── y
//!      └─ trainable adapter ─ y_ad ┘
//! ```
//!
//! Base weights live in [`FrozenLinear`] and participate in the forward pass only in
//! detached form: `dL/dW_base` does not exist by construction, while `dL/dx` and
//! adapter gradients proceed as usual. The optimizer itself should not freeze
//! anything.
//!
//! **The layout of the weights is as in Burn:** `Linear.weight` has shape
//! `[d_input, d_output]`, computed as `y = x · W + b`. Therefore
//! `lora_a: [in, r]`, `lora_b: [r, out]`, `ΔW = scaling · A · B: [in, out]`
//! (in PyTorch spec notation this is `BA` with transposed shapes).
//!
//! Layers accept a tensor of any rank and work along the last axis.
//!
//! `merge()` does not overwrite `W₀`: the merged weight is stored separately, `unmerge()`
//! it simply discards it, so the restoration is accurate.

mod dora;
mod ia3;
mod lora;

#[cfg(test)]
mod tests;

pub use dora::{DoraConfig, DoraLinear};
pub use ia3::{Ia3Config, Ia3Linear, Ia3Mode};
pub use lora::{LoraConfig, LoraLinear};

use burn::module::Param;
use burn::nn::Linear;
use burn::prelude::*;
use burn::tensor::module::linear;

/// General interface of the PEFT layer.
pub trait PeftAdapter {
    /// Direct pass: `[..., in] → [..., out]`.
    fn forward<const D: usize>(&self, x: Tensor<D>) -> Tensor<D>;

    /// Number of learnable adapter parameters (without base layer).
    fn adapter_parameter_count(&self) -> usize;

    /// Whether the adapter participates in the calculation (`false`, if merged with the base).
    fn is_active(&self) -> bool;
}

/// Merging the adapter with the base weight for inference.
///
/// `merge()` and `unmerge()` are idempotent.
pub trait MergeablePeft {
    fn merge(&mut self);
    fn unmerge(&mut self);
    fn is_merged(&self) -> bool;
}

/// Linear layer with frozen weights: `y = x · W + b`.
///
/// The parameters remain in the module (saved, transferred between devices),
/// but in the calculation they are used through `detach()`.
#[derive(Module, Debug)]
pub struct FrozenLinear {
    /// Weight `[d_input, d_output]`.
    pub weight: Param<Tensor<2>>,
    /// Offset `[d_output]`.
    pub bias: Option<Param<Tensor<1>>>,
}

impl FrozenLinear {
    /// Freeze ready [`Linear`].
    pub fn from_linear(l: Linear) -> Self {
        Self {
            weight: l.weight,
            bias: l.bias,
        }
    }

    pub fn d_input(&self) -> usize {
        self.weight.dims()[0]
    }

    pub fn d_output(&self) -> usize {
        self.weight.dims()[1]
    }

    /// Weight without gradient connection.
    pub fn weight(&self) -> Tensor<2> {
        self.weight.val().detach()
    }

    /// Displacement without gradient coupling.
    pub fn bias(&self) -> Option<Tensor<1>> {
        self.bias.as_ref().map(|b| b.val().detach())
    }

    /// `x · W + b`.
    pub fn forward<const D: usize>(&self, x: Tensor<D>) -> Tensor<D> {
        linear(x, self.weight(), self.bias())
    }

    /// `x · W` (no offset).
    pub fn forward_no_bias<const D: usize>(&self, x: Tensor<D>) -> Tensor<D> {
        linear(x, self.weight(), None)
    }
}

/// `[n] → [1, …, 1, n]` of rank `D` for broadcast along the last axis.
pub(crate) fn along_last<const D: usize>(v: Tensor<1>) -> Tensor<D> {
    let n = v.dims()[0];
    let mut shape = [1usize; D];
    shape[D - 1] = n;
    v.reshape(shape)
}

pub(crate) fn numel<const D: usize>(t: &Param<Tensor<D>>) -> usize {
    t.dims().iter().product()
}

pub(crate) fn check_dims(in_f: usize, out_f: usize) {
    assert!(in_f > 0, "in_features must be > 0");
    assert!(out_f > 0, "out_features must be > 0");
}

pub(crate) fn check_lora(rank: usize, alpha: f32, dropout: f64) {
    assert!(rank > 0, "rank must be > 0");
    assert!(alpha > 0.0 && alpha.is_finite(), "alpha must be > 0");
    assert!((0.0..1.0).contains(&dropout), "dropout should be in [0, 1)");
}
