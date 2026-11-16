//! LayerScale (Touvron et al., 2021, arXiv:2103.17239): `y = γ ⊙ x`.
//!
//! Not normalization. Applies to the output of the residual branch: `h = x + γ ⊙ F(x)`;
//! addition with residual is left to the calling code.

use burn::module::Param;
use burn::prelude::*;

use crate::error::{NnError, fail};

use super::{assert_last_dim, check_d_model, numel};

/// Configuration [`LayerScale`].
#[derive(Config, Debug)]
pub struct LayerScaleConfig {
    pub d_model: usize,
    /// The initial value of all `γᵢ` (for example `1e-5` for deep networks).
    #[config(default = "1.0")]
    pub init_value: f32,
}

impl LayerScaleConfig {
    pub fn validate(&self) -> Result<(), String> {
        check_d_model(self.d_model)?;
        if !self.init_value.is_finite() {
            return Err("init_value must be finite".into());
        }
        Ok(())
    }

    /// `d_model`.
    pub fn parameter_count(&self) -> usize {
        self.d_model
    }

    /// Create a layer; `Err` if [`validate`](Self::validate) failed.
    pub fn try_init(&self, device: &Device) -> Result<LayerScale, NnError> {
        self.validate()
            .map_err(|e| NnError::config("LayerScaleConfig", e))?;
        Ok(LayerScale {
            gamma: Param::from_tensor(Tensor::full([self.d_model], self.init_value, device)),
        })
    }

    /// Create a layer.
    ///
    /// # Panics
    /// If [`validate`](Self::validate) returned an error (the way without panic is [`try_init`](Self::try_init)).
    pub fn init(&self, device: &Device) -> LayerScale {
        self.try_init(device).unwrap_or_else(|e| fail(e))
    }
}

/// Component-wise learnable scale on the last axis.
#[derive(Module, Debug)]
pub struct LayerScale {
    /// `γ: [D]`.
    pub gamma: Param<Tensor<1>>,
}

impl LayerScale {
    pub fn forward<const D: usize>(&self, x: Tensor<D>) -> Tensor<D> {
        let d = self.gamma.dims()[0];
        assert_last_dim(&x, d, "LayerScale");
        let mut shape = [1usize; D];
        shape[D - 1] = d;
        x * self.gamma.val().reshape(shape)
    }

    pub fn parameter_count(&self) -> usize {
        numel(&self.gamma)
    }
}
