//! Adaptive RMSNorm: modulation is based on the condition.
//!
//! ```text
//! c → Linear(c) → [γ, β]
//! y = (1 + γ) ⊙ RMSNorm(x) + β
//! ```
//!
//! At `zero_init` the projection is zero, and the fresh layer is equal to the usual RMSNorm
//! under any condition - it can be inserted into an already trained model.
//!
//! Two condition modes:
//! * [`AdaptiveRmsNorm::forward_global`]: `c: [B, C]`, one modulation per
//!   batch element, stretches along all intermediate axes `x`;
//! * [`AdaptiveRmsNorm::forward`]: `c` of the same rank as `x` - `[B,1,C]`
//!   or per-token `[B,T,C]`.

use burn::nn::{Initializer, Linear, LinearConfig};
use burn::prelude::*;

use crate::error::{NnError, fail};

use super::{
    ConditionalRmsNorm, ConditionalRmsNormConfig, assert_last_dim, check_d_model, check_eps,
};

/// Configuration [`AdaptiveRmsNorm`].
#[derive(Config, Debug)]
pub struct AdaptiveRmsNormConfig {
    pub d_model: usize,
    pub condition_dim: usize,
    #[config(default = "1e-5")]
    pub eps: f32,
    /// If `false`, only `scale` is constructed (the projection gives `d_model`).
    #[config(default = true)]
    pub use_shift: bool,
    /// Zero initialization of the projection (weight and bias).
    #[config(default = true)]
    pub zero_init: bool,
}

impl AdaptiveRmsNormConfig {
    pub fn validate(&self) -> Result<(), String> {
        check_d_model(self.d_model)?;
        if self.condition_dim == 0 {
            return Err("condition_dim must be > 0".into());
        }
        check_eps(self.eps)
    }

    fn modulation_dim(&self) -> usize {
        if self.use_shift {
            2 * self.d_model
        } else {
            self.d_model
        }
    }

    /// Weight RMSNorm + projection with bias.
    pub fn parameter_count(&self) -> usize {
        let m = self.modulation_dim();
        self.d_model + self.condition_dim * m + m
    }

    /// Create a layer; `Err` if [`validate`](Self::validate) failed.
    pub fn try_init(&self, device: &Device) -> Result<AdaptiveRmsNorm, NnError> {
        self.validate()
            .map_err(|e| NnError::config("AdaptiveRmsNormConfig", e))?;
        let mut proj = LinearConfig::new(self.condition_dim, self.modulation_dim());
        if self.zero_init {
            proj = proj.with_initializer(Initializer::Zeros);
        }
        Ok(AdaptiveRmsNorm {
            conditional: ConditionalRmsNormConfig::new(self.d_model)
                .with_eps(self.eps)
                .try_init(device)?,
            modulation: proj.init(device),
            d_model: self.d_model,
            condition_dim: self.condition_dim,
            use_shift: self.use_shift,
        })
    }

    /// Create a layer.
    ///
    /// # Panics
    /// If [`validate`](Self::validate) returned an error (the way without panic is [`try_init`](Self::try_init)).
    pub fn init(&self, device: &Device) -> AdaptiveRmsNorm {
        self.try_init(device).unwrap_or_else(|e| fail(e))
    }
}

/// RMSNorm modulated by condition.
#[derive(Module, Debug)]
pub struct AdaptiveRmsNorm {
    pub conditional: ConditionalRmsNorm,
    /// `condition_dim → d_model` (scale) or `2·d_model` (scale, shift).
    pub modulation: Linear,
    #[module(skip)]
    pub d_model: usize,
    #[module(skip)]
    pub condition_dim: usize,
    #[module(skip)]
    pub use_shift: bool,
}

impl AdaptiveRmsNorm {
    /// `condition` is of the same rank as `x`: `[B,1,C]` or `[B,T,C]`.
    pub fn forward<const D: usize>(&self, x: Tensor<D>, condition: Tensor<D>) -> Tensor<D> {
        assert_last_dim(&condition, self.condition_dim, "AdaptiveRmsNorm.condition");
        let m = self.modulation.forward(condition);
        self.apply(x, m)
    }

    /// `condition: [B, C]` — common for all positions; `x: [B, …, D]`.
    pub fn forward_global<const D: usize>(&self, x: Tensor<D>, condition: Tensor<2>) -> Tensor<D> {
        assert_last_dim(&condition, self.condition_dim, "AdaptiveRmsNorm.condition");
        assert_eq!(
            condition.dims()[0],
            x.dims()[0],
            "AdaptiveRmsNorm: condition and input batch sizes are different"
        );
        let m = self.modulation.forward(condition);
        let [b, n] = m.dims();
        let mut shape = [1usize; D];
        shape[0] = b;
        shape[D - 1] = n;
        self.apply(x, m.reshape(shape))
    }

    fn apply<const D: usize>(&self, x: Tensor<D>, m: Tensor<D>) -> Tensor<D> {
        let d = self.d_model;
        let scale = m.clone().narrow(D - 1, 0, d);
        if self.use_shift {
            let shift = m.narrow(D - 1, d, d);
            self.conditional.forward_with_modulation(x, scale, shift)
        } else {
            self.conditional.forward_scale_only(x, scale)
        }
    }
}
