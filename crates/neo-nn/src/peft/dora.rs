//! DoRA (Liu et al., 2024, arXiv:2402.09353): weight-decomposed LoRA.
//!
//! ```text
//! V   = W₀ + (α/r)·A·B
//! W = m ⊙ V / ‖V‖ (norm along the input axis, separately for each output)
//! y   = x·W + b
//! ```
//!
//! Calculated without materializing `W`: `y = (x·W₀ + (α/r)·drop(x)·A·B) · (m/‖V‖) + b`.
//! Initially `B = 0`, `m = ‖W₀‖` ⇒ `DoraLinear(x) == Linear(x)`.
//! `‖V‖ = sqrt(Σ V² + eps)` - protection against division by zero.

use burn::module::Param;
use burn::nn::{Dropout, DropoutConfig, Initializer, Linear, LinearConfig};
use burn::prelude::*;
use burn::tensor::module::linear;

use super::{FrozenLinear, MergeablePeft, PeftAdapter, along_last, check_dims, check_lora, numel};

/// Configuration [`DoraLinear`].
#[derive(Config, Debug)]
pub struct DoraConfig {
    pub in_features: usize,
    pub out_features: usize,
    pub rank: usize,
    #[config(default = "1.0")]
    pub alpha: f32,
    #[config(default = "0.0")]
    pub dropout: f64,
    #[config(default = true)]
    pub bias: bool,
    /// Addition under the root when calculating the norm.
    #[config(default = "1e-6")]
    pub eps: f32,
}

impl DoraConfig {
    pub fn scaling(&self) -> f32 {
        self.alpha / self.rank as f32
    }

    pub fn validate(&self) {
        check_dims(self.in_features, self.out_features);
        check_lora(self.rank, self.alpha, self.dropout);
        assert!(self.eps > 0.0 && self.eps.is_finite(), "eps must be > 0");
    }

    /// `in·r + r·out + out` (adapter + magnitude).
    pub fn parameter_count(&self) -> usize {
        self.rank * (self.in_features + self.out_features) + self.out_features
    }

    pub fn init(&self, device: &Device) -> DoraLinear {
        self.validate();
        let base = LinearConfig::new(self.in_features, self.out_features)
            .with_bias(self.bias)
            .init(device);
        self.init_from_linear(base)
    }

    pub fn init_from_linear(&self, base: Linear) -> DoraLinear {
        self.validate();
        let [d_in, d_out] = base.weight.dims();
        assert_eq!(
            (d_in, d_out),
            (self.in_features, self.out_features),
            "Linear dimensions do not match DoraConfig"
        );
        let device = base.weight.val().device();
        let magnitude = column_norm(base.weight.val(), self.eps);
        DoraLinear {
            base: FrozenLinear::from_linear(base),
            lora_a: LinearConfig::new(d_in, self.rank)
                .with_bias(false)
                .init(&device),
            lora_b: LinearConfig::new(self.rank, d_out)
                .with_bias(false)
                .with_initializer(Initializer::Zeros)
                .init(&device),
            magnitude: Param::from_tensor(magnitude.detach()),
            dropout: DropoutConfig::new(self.dropout).init(),
            scaling: self.scaling(),
            eps: self.eps,
            merged_weight: None,
        }
    }
}

/// `sqrt(Σ_in W² + eps)` for each output: `[in, out] → [out]`.
fn column_norm(w: Tensor<2>, eps: f32) -> Tensor<1> {
    let out = w.dims()[1];
    w.powf_scalar(2.0)
        .sum_dim(0)
        .add_scalar(eps)
        .sqrt()
        .reshape([out])
}

/// Linear layer with DoRA adapter.
#[derive(Module, Debug)]
pub struct DoraLinear {
    pub base: FrozenLinear,
    /// `A: [in, r]`.
    pub lora_a: Linear,
    /// `B: [r, out]`.
    pub lora_b: Linear,
    /// Trainable length `m: [out]`.
    pub magnitude: Param<Tensor<1>>,
    pub dropout: Dropout,
    #[module(skip)]
    pub scaling: f32,
    #[module(skip)]
    pub eps: f32,
    /// Merged weight `m ⊙ V/‖V‖: [in, out]` if the layer is merged.
    pub merged_weight: Option<Tensor<2>>,
}

impl DoraLinear {
    /// `V = W₀ + scaling·A·B`: `[in, out]`.
    fn direction_weight(&self) -> Tensor<2> {
        let ab = self
            .lora_a
            .weight
            .val()
            .matmul(self.lora_b.weight.val())
            .mul_scalar(self.scaling);
        self.base.weight() + ab
    }

    pub fn forward<const D: usize>(&self, x: Tensor<D>) -> Tensor<D> {
        if let Some(w) = &self.merged_weight {
            return linear(x, w.clone(), self.base.bias());
        }
        let norm = column_norm(self.direction_weight(), self.eps);
        let ratio = self.magnitude.val() / norm;

        let raw = self.base.forward_no_bias(x.clone())
            + self
                .lora_b
                .forward(self.lora_a.forward(self.dropout.forward(x)))
                .mul_scalar(self.scaling);
        let y = raw * along_last::<D>(ratio);
        match self.base.bias() {
            Some(b) => y + along_last::<D>(b),
            None => y,
        }
    }

    /// `m ⊙ V/‖V‖`, shape `[in, out]`, without gradient connection.
    pub fn merged_weight(&self) -> Tensor<2> {
        let v = self.direction_weight();
        let norm = column_norm(v.clone(), self.eps);
        let ratio = (self.magnitude.val() / norm).reshape([1, self.base.d_output()]);
        (v * ratio).detach()
    }
}

impl PeftAdapter for DoraLinear {
    fn forward<const D: usize>(&self, x: Tensor<D>) -> Tensor<D> {
        DoraLinear::forward(self, x)
    }

    fn adapter_parameter_count(&self) -> usize {
        numel(&self.lora_a.weight) + numel(&self.lora_b.weight) + numel(&self.magnitude)
    }

    fn is_active(&self) -> bool {
        self.merged_weight.is_none()
    }
}

impl MergeablePeft for DoraLinear {
    fn merge(&mut self) {
        if self.merged_weight.is_none() {
            self.merged_weight = Some(self.merged_weight());
        }
    }

    fn unmerge(&mut self) {
        self.merged_weight = None;
    }

    fn is_merged(&self) -> bool {
        self.merged_weight.is_some()
    }
}
