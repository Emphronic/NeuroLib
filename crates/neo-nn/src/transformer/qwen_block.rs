//! Qwen block: the same `LlmBlock`, but with optional QK-Norm (Qwen3),
//! bias in Q/K/V (Qwen2) and the choice of gated activation.

use burn::prelude::*;

use super::llm_block::{AttentionKind, FfnKind, KvCache, LlmBlock, LlmBlockConfig};

#[derive(Config, Debug)]
pub struct QwenBlockConfig {
    pub d_model: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub d_ff: usize,
    #[config(default = 1_000_000.0)]
    pub rope_theta: f32,
    #[config(default = 1e-6)]
    pub rms_norm_eps: f32,
    #[config(default = 2048)]
    pub max_seq_len: usize,
    /// RMSNorm by `head_dim` for Q and K to RoPE; V is not affected.
    #[config(default = false)]
    pub use_qk_norm: bool,
    /// Bias in Q/K/V (Qwen2); `o_proj` and FFN without bias.
    #[config(default = false)]
    pub qkv_bias: bool,
    /// `SwiGlu` / `GeGlu` / `ReGlu`.
    #[config(default = "FfnKind::SwiGlu")]
    pub ffn_kind: FfnKind,
    #[config(default = 0)]
    pub layer_index: usize,
}

impl QwenBlockConfig {
    pub fn to_block_config(&self) -> LlmBlockConfig {
        LlmBlockConfig::new(self.d_model, self.num_heads, self.num_kv_heads, self.d_ff)
            .with_rope_theta(self.rope_theta)
            .with_rms_norm_eps(self.rms_norm_eps)
            .with_max_seq_len(self.max_seq_len)
            .with_attention_kind(AttentionKind::infer(self.num_heads, self.num_kv_heads))
            .with_ffn_kind(self.ffn_kind.clone())
            .with_use_qk_norm(self.use_qk_norm)
            .with_use_bias(self.qkv_bias)
            .with_layer_index(self.layer_index)
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.ffn_kind == FfnKind::Moe {
            return Err("QwenBlock: For MoE use MoeBlock".into());
        }
        self.to_block_config().validate()
    }

    /// # Panics
    /// If the config fails [`validate`](Self::validate).
    pub fn init(&self, device: &Device) -> QwenBlock {
        if let Err(e) = self.validate() {
            panic!("QwenBlockConfig: {e}");
        }
        QwenBlock {
            block: self.to_block_config().init(device),
        }
    }
}

#[derive(Module, Debug)]
pub struct QwenBlock {
    pub block: LlmBlock,
}

impl QwenBlock {
    pub fn forward(&self, x: Tensor<3>) -> Tensor<3> {
        self.block.forward(x)
    }

    pub fn forward_with_mask(&self, x: Tensor<3>, mask: Option<Tensor<4>>) -> Tensor<3> {
        self.block.forward_with_mask(x, mask)
    }

    pub fn forward_cached(&self, x: Tensor<3>, cache: &mut KvCache, position: usize) -> Tensor<3> {
        self.block.forward_cached(x, cache, position)
    }

    pub fn new_cache(&self) -> KvCache {
        self.block.new_cache()
    }
}
