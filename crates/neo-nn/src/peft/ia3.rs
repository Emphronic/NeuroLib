//! IA³ (Liu et al., 2022, arXiv:2205.05638): trainable activation scales.
//!
//! ```text
//! Input : y = (s ⊙ x)·W + b,   s: [in]
//! Output: y = s ⊙ (x·W + b),   s: [out]
//! ```
//!
//! `s` is initialized with ones ⇒ `Ia3Linear(x) == Linear(x)`.

use burn::module::Param;
use burn::nn::Linear;
use burn::nn::LinearConfig;
use burn::prelude::*;
use burn::tensor::module::linear;

use super::{FrozenLinear, MergeablePeft, PeftAdapter, along_last, check_dims, numel};

/// Which side of the layer the scale is applied to.
#[derive(Config, Debug, PartialEq)]
pub enum Ia3Mode {
    /// `s ∈ [in]`, input is scaled.
    Input,
    /// `s ∈ [out]`, output is scaled (including bias).
    Output,
}

/// Configuration [`Ia3Linear`].
#[derive(Config, Debug)]
pub struct Ia3Config {
    pub in_features: usize,
    pub out_features: usize,
    #[config(default = "Ia3Mode::Input")]
    pub mode: Ia3Mode,
    #[config(default = true)]
    pub bias: bool,
}

impl Ia3Config {
    pub fn validate(&self) {
        check_dims(self.in_features, self.out_features);
    }

    /// Vector size `s`.
    pub fn parameter_count(&self) -> usize {
        match self.mode {
            Ia3Mode::Input => self.in_features,
            Ia3Mode::Output => self.out_features,
        }
    }

    pub fn init(&self, device: &Device) -> Ia3Linear {
        self.validate();
        let base = LinearConfig::new(self.in_features, self.out_features)
            .with_bias(self.bias)
            .init(device);
        self.init_from_linear(base)
    }

    pub fn init_from_linear(&self, base: Linear) -> Ia3Linear {
        self.validate();
        let [d_in, d_out] = base.weight.dims();
        assert_eq!(
            (d_in, d_out),
            (self.in_features, self.out_features),
            "Linear dimensions do not match with Ia3Config"
        );
        let device = base.weight.val().device();
        Ia3Linear {
            base: FrozenLinear::from_linear(base),
            scale: Param::from_tensor(Tensor::ones([self.parameter_count()], &device)),
            mode: self.mode.clone(),
            merged_weight: None,
            merged_bias: None,
        }
    }
}

/// Linear layer with IA³ scale.
#[derive(Module, Debug)]
pub struct Ia3Linear {
    pub base: FrozenLinear,
    /// Trainable scale `s`.
    pub scale: Param<Tensor<1>>,
    #[module(skip)]
    pub mode: Ia3Mode,
    pub merged_weight: Option<Tensor<2>>,
    /// Merge bias (only Output mode with bias).
    pub merged_bias: Option<Tensor<1>>,
}

impl Ia3Linear {
    pub fn forward<const D: usize>(&self, x: Tensor<D>) -> Tensor<D> {
        if let Some(w) = &self.merged_weight {
            let bias = self.merged_bias.clone().or_else(|| self.base.bias());
            return linear(x, w.clone(), bias);
        }
        let s = along_last::<D>(self.scale.val());
        match self.mode {
            Ia3Mode::Input => self.base.forward(x * s),
            Ia3Mode::Output => self.base.forward(x) * s,
        }
    }

    /// Merged `(W', b')`, without gradient connection.
    ///
    /// Input: `W'[i,o] = s[i]·W[i,o]`, `b' = b`.
    /// Output: `W'[i,o] = W[i,o]·s[o]`, `b' = s ⊙ b`.
    pub fn merged_parameters(&self) -> (Tensor<2>, Option<Tensor<1>>) {
        let s = self.scale.val();
        let w = self.base.weight();
        match self.mode {
            Ia3Mode::Input => {
                let n = s.dims()[0];
                ((w * s.reshape([n, 1])).detach(), None)
            }
            Ia3Mode::Output => {
                let n = s.dims()[0];
                let bias = self.base.bias().map(|b| (b * s.clone()).detach());
                ((w * s.reshape([1, n])).detach(), bias)
            }
        }
    }
}

impl PeftAdapter for Ia3Linear {
    fn forward<const D: usize>(&self, x: Tensor<D>) -> Tensor<D> {
        Ia3Linear::forward(self, x)
    }

    fn adapter_parameter_count(&self) -> usize {
        numel(&self.scale)
    }

    fn is_active(&self) -> bool {
        self.merged_weight.is_none()
    }
}

impl MergeablePeft for Ia3Linear {
    fn merge(&mut self) {
        if self.merged_weight.is_none() {
            let (w, b) = self.merged_parameters();
            self.merged_weight = Some(w);
            self.merged_bias = b;
        }
    }

    fn unmerge(&mut self) {
        self.merged_weight = None;
        self.merged_bias = None;
    }

    fn is_merged(&self) -> bool {
        self.merged_weight.is_some()
    }
}
