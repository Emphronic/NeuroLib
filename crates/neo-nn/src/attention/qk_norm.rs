//! QK-Norm: RMSNorm by `head_dim` separately for Q and K (up to RoPE).
//!
//! `Q: [B, Hq, T, Dh]`, `K: [B, Hkv, T, Dh]`; Only the last axis is normalized.
//! The weights Q and K are independent.

use burn::nn::{RmsNorm, RmsNormConfig};
use burn::prelude::*;

use crate::error::{NnError, fail};

/// Configuration [`QkNorm`].
#[derive(Config, Debug)]
pub struct QkNormConfig {
    pub head_dim: usize,
    #[config(default = "1e-5")]
    pub eps: f32,
}

impl QkNormConfig {
    pub fn validate(&self) -> Result<(), String> {
        if self.head_dim == 0 {
            return Err("head_dim must be > 0".into());
        }
        if !(self.eps.is_finite() && self.eps > 0.0) {
            return Err("eps must be finite and > 0".into());
        }
        Ok(())
    }

    /// `2 · head_dim`.
    pub fn parameter_count(&self) -> usize {
        2 * self.head_dim
    }

    /// Create a layer; `Err` if [`validate`](Self::validate) failed.
    pub fn try_init(&self, device: &Device) -> Result<QkNorm, NnError> {
        self.validate()
            .map_err(|e| NnError::config("QkNormConfig", e))?;
        let n = RmsNormConfig::new(self.head_dim).with_epsilon(self.eps as f64);
        Ok(QkNorm {
            q_norm: n.init(device),
            k_norm: n.init(device),
        })
    }

    /// Create a layer.
    ///
    /// # Panics
    /// If [`validate`](Self::validate) returned an error (the way without panic is [`try_init`](Self::try_init)).
    pub fn init(&self, device: &Device) -> QkNorm {
        self.try_init(device).unwrap_or_else(|e| fail(e))
    }
}

/// Independent RMSNorms for Q and K.
#[derive(Module, Debug)]
pub struct QkNorm {
    pub q_norm: RmsNorm,
    pub k_norm: RmsNorm,
}

impl QkNorm {
    pub fn forward_q(&self, q: Tensor<4>) -> Tensor<4> {
        self.q_norm.forward(q)
    }

    pub fn forward_k(&self, k: Tensor<4>) -> Tensor<4> {
        self.k_norm.forward(k)
    }

    pub fn forward(&self, q: Tensor<4>, k: Tensor<4>) -> (Tensor<4>, Tensor<4>) {
        (self.forward_q(q), self.forward_k(k))
    }
}
