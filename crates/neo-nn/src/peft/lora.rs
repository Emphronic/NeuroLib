//! LoRA (Hu et al., 2021, arXiv:2106.09685).
//!
//! ```text
//! y = x·W₀ + b + (α/r) · ((drop(x)·A)·B)
//! ```
//!
//! `A` is initialized randomly, `B` with zeros ⇒ at the time of creation
//! `LoraLinear(x) == Linear(x)`.

use burn::nn::{Dropout, DropoutConfig, Initializer, Linear, LinearConfig};
use burn::prelude::*;
use burn::tensor::module::linear;

use super::{FrozenLinear, MergeablePeft, PeftAdapter, check_dims, check_lora, numel};

/// Configuration [`LoraLinear`].
#[derive(Config, Debug)]
pub struct LoraConfig {
    pub in_features: usize,
    pub out_features: usize,
    /// Adapter rank `r`.
    pub rank: usize,
    /// `α`; adapter multiplier `α / r`.
    #[config(default = "1.0")]
    pub alpha: f32,
    /// Dropout is only on the adapter branch.
    #[config(default = "0.0")]
    pub dropout: f64,
    /// Does the base layer have bias (for [`LoraConfig::init`]).
    #[config(default = true)]
    pub bias: bool,
}

impl LoraConfig {
    /// `α / r`.
    pub fn scaling(&self) -> f32 {
        self.alpha / self.rank as f32
    }

    /// Panics when the configuration is incorrect.
    pub fn validate(&self) {
        check_dims(self.in_features, self.out_features);
        check_lora(self.rank, self.alpha, self.dropout);
    }

    /// Number of trainable parameters: `in·r + r·out`.
    pub fn parameter_count(&self) -> usize {
        self.rank * (self.in_features + self.out_features)
    }

    /// New random base layer + adapter (for tests and training from scratch).
    pub fn init(&self, device: &Device) -> LoraLinear {
        self.validate();
        let base = LinearConfig::new(self.in_features, self.out_features)
            .with_bias(self.bias)
            .init(device);
        self.init_from_linear(base)
    }

    /// Wrap existing [`Linear`]: its weights are frozen.
    pub fn init_from_linear(&self, base: Linear) -> LoraLinear {
        self.validate();
        let [d_in, d_out] = base.weight.dims();
        assert_eq!(
            (d_in, d_out),
            (self.in_features, self.out_features),
            "Linear dimensions do not match LoraConfig"
        );
        let device = base.weight.val().device();
        LoraLinear {
            base: FrozenLinear::from_linear(base),
            lora_a: LinearConfig::new(d_in, self.rank)
                .with_bias(false)
                .init(&device),
            lora_b: LinearConfig::new(self.rank, d_out)
                .with_bias(false)
                .with_initializer(Initializer::Zeros)
                .init(&device),
            dropout: DropoutConfig::new(self.dropout).init(),
            scaling: self.scaling(),
            merged_weight: None,
        }
    }
}

/// Linear layer with LoRA adapter.
#[derive(Module, Debug)]
pub struct LoraLinear {
    pub base: FrozenLinear,
    /// `A: [in, r]`.
    pub lora_a: Linear,
    /// `B: [r, out]`.
    pub lora_b: Linear,
    pub dropout: Dropout,
    #[module(skip)]
    pub scaling: f32,
    /// `W₀ + scaling·A·B`, if the layer is merged; otherwise `None`.
    pub merged_weight: Option<Tensor<2>>,
}

impl LoraLinear {
    pub fn forward<const D: usize>(&self, x: Tensor<D>) -> Tensor<D> {
        if let Some(w) = &self.merged_weight {
            return linear(x, w.clone(), self.base.bias());
        }
        let base = self.base.forward(x.clone());
        let delta = self
            .lora_b
            .forward(self.lora_a.forward(self.dropout.forward(x)));
        base + delta * self.scaling
    }

    /// `W₀ + scaling · A · B`, shape `[in, out]`, without gradient connection.
    pub fn merged_weight(&self) -> Tensor<2> {
        let ab = self
            .lora_a
            .weight
            .val()
            .matmul(self.lora_b.weight.val())
            .mul_scalar(self.scaling);
        (self.base.weight() + ab).detach()
    }
}

impl PeftAdapter for LoraLinear {
    fn forward<const D: usize>(&self, x: Tensor<D>) -> Tensor<D> {
        LoraLinear::forward(self, x)
    }

    fn adapter_parameter_count(&self) -> usize {
        numel(&self.lora_a.weight) + numel(&self.lora_b.weight)
    }

    fn is_active(&self) -> bool {
        self.merged_weight.is_none()
    }
}

impl MergeablePeft for LoraLinear {
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
