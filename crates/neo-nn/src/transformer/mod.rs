//! Decoder-only Transformer (LLM) on top of Burn Tensor.
//!
//! ```text
//!                    LlmBlock
//!        ┌────────────┼────────────┐
//!   LlmAttention   LlmFfn      residual
//!  (GQA/MQA, RoPE,  (SwiGLU/GeGLU/ReGLU
//!   KV-cache) or MoE)
//! ```
//!
//! * [`LlmBlock`] - general pre-norm block; all mathematics lives here.
//! * [`LlamaBlock`], [`QwenBlock`], [`MoeBlock`] - subtle specializations:
//!   collect [`LlmBlockConfig`] and delegate to `LlmBlock`.
//!
//! Activation format everywhere `[B, T, D]`; inside attention - `[B, H, T, Dh]`.
//! Trainable parameters (module) and inference state ([`KvCache`]) are separated:
//! The cache creates and holds calling code, one per layer.

mod llama_block;
mod llm_block;
mod moe_block;
mod qwen_block;

#[cfg(test)]
mod tests;

pub use llama_block::{LlamaBlock, LlamaBlockConfig};
pub use llm_block::{
    AttentionKind, FfnKind, KvCache, LlmAttention, LlmBlock, LlmBlockConfig, LlmFfn, RopeLayout,
    RotaryEmbedding, SwiGluFfn, causal_mask,
};
pub use moe_block::{MoeBlock, MoeBlockConfig};
pub use qwen_block::{QwenBlock, QwenBlockConfig};
