//! MQA is a thin wrapper over GQA with `num_kv_heads = 1`.

use burn::prelude::*;

use super::{GqaAttention, GqaAttentionConfig};
use crate::transformer::{KvCache, RopeLayout};

/// Configuration [`MqaAttention`] (`num_kv_heads` is not needed outside).
#[derive(Config, Debug)]
pub struct MqaAttentionConfig {
    pub d_model: usize,
    pub num_heads: usize,
    #[config(default = false)]
    pub bias: bool,
    #[config(default = false)]
    pub o_bias: bool,
    #[config(default = "0.0")]
    pub dropout: f64,
    #[config(default = true)]
    pub causal: bool,
    #[config(default = false)]
    pub use_qk_norm: bool,
    #[config(default = "1e-5")]
    pub qk_norm_eps: f32,
    #[config(default = true)]
    pub rope: bool,
    #[config(default = "10000.0")]
    pub rope_theta: f32,
    #[config(default = 2048)]
    pub max_seq_len: usize,
    #[config(default = "RopeLayout::Interleaved")]
    pub rope_layout: RopeLayout,
}

impl MqaAttentionConfig {
    /// Equivalent GQA config with `num_kv_heads = 1`.
    pub fn to_gqa(&self) -> GqaAttentionConfig {
        GqaAttentionConfig::new(self.d_model, self.num_heads, 1)
            .with_bias(self.bias)
            .with_o_bias(self.o_bias)
            .with_dropout(self.dropout)
            .with_causal(self.causal)
            .with_use_qk_norm(self.use_qk_norm)
            .with_qk_norm_eps(self.qk_norm_eps)
            .with_rope(self.rope)
            .with_rope_theta(self.rope_theta)
            .with_max_seq_len(self.max_seq_len)
            .with_rope_layout(self.rope_layout.clone())
    }

    pub fn validate(&self) -> Result<(), String> {
        self.to_gqa().validate()
    }

    pub fn init(&self, device: &Device) -> MqaAttention {
        MqaAttention {
            inner: self.to_gqa().init(device),
        }
    }
}

/// Multi-query attention = [`GqaAttention`] with one KV head.
#[derive(Module, Debug)]
pub struct MqaAttention {
    pub inner: GqaAttention,
}

impl MqaAttention {
    pub fn forward(&self, x: Tensor<3>, mask: Option<Tensor<4>>) -> Tensor<3> {
        self.inner.forward(x, mask)
    }

    pub fn forward_cached(&self, x: Tensor<3>, cache: &mut KvCache, position: usize) -> Tensor<3> {
        self.inner.forward_cached(x, cache, position)
    }

    pub fn new_cache(&self) -> KvCache {
        self.inner.new_cache()
    }
}
