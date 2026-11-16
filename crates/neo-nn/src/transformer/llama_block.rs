//! Llama block: pre-norm, GQA, RoPE, SwiGLU, without bias. Thin wrapper over
//! [`LlmBlock`].

use burn::prelude::*;

use super::llm_block::{AttentionKind, FfnKind, KvCache, LlmBlock, LlmBlockConfig};

#[derive(Config, Debug)]
pub struct LlamaBlockConfig {
    pub d_model: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    pub d_ff: usize,
    #[config(default = 10000.0)]
    pub rope_theta: f32,
    #[config(default = 1e-5)]
    pub rms_norm_eps: f32,
    #[config(default = 2048)]
    pub max_seq_len: usize,
    #[config(default = 0)]
    pub layer_index: usize,
}

impl LlamaBlockConfig {
    pub fn to_block_config(&self) -> LlmBlockConfig {
        LlmBlockConfig::new(self.d_model, self.num_heads, self.num_kv_heads, self.d_ff)
            .with_rope_theta(self.rope_theta)
            .with_rms_norm_eps(self.rms_norm_eps)
            .with_max_seq_len(self.max_seq_len)
            .with_attention_kind(AttentionKind::infer(self.num_heads, self.num_kv_heads))
            .with_ffn_kind(FfnKind::SwiGlu)
            .with_layer_index(self.layer_index)
    }

    pub fn validate(&self) -> Result<(), String> {
        self.to_block_config().validate()
    }

    /// # Panics
    /// If the config fails [`validate`](Self::validate).
    pub fn init(&self, device: &Device) -> LlamaBlock {
        LlamaBlock {
            block: self.to_block_config().init(device),
        }
    }
}

#[derive(Module, Debug)]
pub struct LlamaBlock {
    pub block: LlmBlock,
}

impl LlamaBlock {
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
