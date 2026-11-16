//! ScaleNorm (Nguyen & Salazar, 2019, arXiv:1910.05895).
//!
//! ```text
//! r = sqrt(Σ xᵢ² + eps),   yᵢ = g · xᵢ / r
//! ```
//!
//! `g` is one trainable scalar (shape parameter `[1]`).

use burn::module::Param;
use burn::prelude::*;

use crate::error::{NnError, fail};

use super::{assert_last_dim, check_d_model, check_eps, numel};

/// Configuration [`ScaleNorm`].
#[derive(Config, Debug)]
pub struct ScaleNormConfig {
    pub d_model: usize,
    #[config(default = "1e-5")]
    pub eps: f32,
    #[config(default = "1.0")]
    pub init_scale: f32,
}

impl ScaleNormConfig {
    pub fn validate(&self) -> Result<(), String> {
        check_d_model(self.d_model)?;
        check_eps(self.eps)?;
        if !self.init_scale.is_finite() {
            return Err("init_scale must be finite".into());
        }
        Ok(())
    }

    /// Always 1.
    pub fn parameter_count(&self) -> usize {
        1
    }

    /// Create a layer; `Err` if [`validate`](Self::validate) failed.
    pub fn try_init(&self, device: &Device) -> Result<ScaleNorm, NnError> {
        self.validate()
            .map_err(|e| NnError::config("ScaleNormConfig", e))?;
        Ok(ScaleNorm {
            scale: Param::from_tensor(Tensor::full([1], self.init_scale, device)),
            d_model: self.d_model,
            eps: self.eps,
        })
    }

    /// Create a layer.
    ///
    /// # Panics
    /// If [`validate`](Self::validate) returned an error (the way without panic is [`try_init`](Self::try_init)).
    pub fn init(&self, device: &Device) -> ScaleNorm {
        self.try_init(device).unwrap_or_else(|e| fail(e))
    }
}

/// ScaleNorm: `y = g · x / sqrt(Σx² + eps)` along the last axis.
#[derive(Module, Debug)]
pub struct ScaleNorm {
    /// Scalar `g`, shape `[1]`.
    pub scale: Param<Tensor<1>>,
    #[module(skip)]
    pub d_model: usize,
    #[module(skip)]
    pub eps: f32,
}

impl ScaleNorm {
    pub fn forward<const D: usize>(&self, x: Tensor<D>) -> Tensor<D> {
        assert_last_dim(&x, self.d_model, "ScaleNorm");
        let r = x
            .clone()
            .powf_scalar(2.0)
            .sum_dim(D - 1)
            .add_scalar(self.eps)
            .sqrt();
        (x / r) * self.scale.val().reshape([1usize; D])
    }

    pub fn parameter_count(&self) -> usize {
        numel(&self.scale)
    }
}
