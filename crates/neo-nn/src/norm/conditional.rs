//! Normalization with ready-made external modulation.
//!
//! ```text
//! y = (1 + scale) ⊙ RMSNorm(x) + shift
//! ```
//!
//! The layer knows nothing about the origin of `scale`/`shift`. At zero
//! modulation `y == RMSNorm(x)` (convenient for AdaLN-Zero).
//!
//! Shapes: `scale`/`shift` of the same rank as `x`, last axis `d_model`,
//! the remaining axes are the same as for `x` or `1` (broadcast): `[B,1,D]` or `[B,T,D]`.

use burn::nn::{RmsNorm, RmsNormConfig};
use burn::prelude::*;

use crate::error::{NnError, fail};

use super::{assert_last_dim, check_d_model, check_eps};

/// Configuration [`ConditionalRmsNorm`].
#[derive(Config, Debug)]
pub struct ConditionalRmsNormConfig {
    pub d_model: usize,
    #[config(default = "1e-5")]
    pub eps: f32,
}

impl ConditionalRmsNormConfig {
    pub fn validate(&self) -> Result<(), String> {
        check_d_model(self.d_model)?;
        check_eps(self.eps)
    }

    /// Trained weight of RMSNorm itself (`d_model`); modulation does not have its own parameters.
    pub fn parameter_count(&self) -> usize {
        self.d_model
    }

    /// Create a layer; `Err` if [`validate`](Self::validate) failed.
    pub fn try_init(&self, device: &Device) -> Result<ConditionalRmsNorm, NnError> {
        self.validate()
            .map_err(|e| NnError::config("ConditionalRmsNormConfig", e))?;
        Ok(ConditionalRmsNorm {
            norm: RmsNormConfig::new(self.d_model)
                .with_epsilon(self.eps as f64)
                .init(device),
            d_model: self.d_model,
        })
    }

    /// Create a layer.
    ///
    /// # Panics
    /// If [`validate`](Self::validate) returned an error (the way without panic is [`try_init`](Self::try_init)).
    pub fn init(&self, device: &Device) -> ConditionalRmsNorm {
        self.try_init(device).unwrap_or_else(|e| fail(e))
    }
}

/// RMSNorm + external modulation.
#[derive(Module, Debug)]
pub struct ConditionalRmsNorm {
    pub norm: RmsNorm,
    #[module(skip)]
    pub d_model: usize,
}

impl ConditionalRmsNorm {
    /// `(1 + scale) ⊙ RMSNorm(x) + shift`.
    pub fn forward_with_modulation<const D: usize>(
        &self,
        x: Tensor<D>,
        scale: Tensor<D>,
        shift: Tensor<D>,
    ) -> Tensor<D> {
        assert_last_dim(&shift, self.d_model, "ConditionalRmsNorm.shift");
        self.forward_scale_only(x, scale) + shift
    }

    /// `(1 + scale) ⊙ RMSNorm(x)` - no shift.
    pub fn forward_scale_only<const D: usize>(&self, x: Tensor<D>, scale: Tensor<D>) -> Tensor<D> {
        assert_last_dim(&x, self.d_model, "ConditionalRmsNorm.x");
        assert_last_dim(&scale, self.d_model, "ConditionalRmsNorm.scale");
        self.norm.forward(x) * scale.add_scalar(1.0)
    }
}
