//! Specialized embedding layers.
//!
//! These are not three variants of the same technology, but three different ideas:
//!
//! ```text
//! Embedding family
//! │
//! ├── FactorizedEmbedding - fewer parameters: [V,R] · [R,D]
//! ├── AdaptiveEmbedding - different token capacity: tiers with different dims
//! └── FourierEmbedding - deterministic sin/cos position attributes
//!                             or continuous value
//! ```
//!
//! The usual `burn::nn::Embedding` (`[vocab, d_model]`) is not reinvented here:
//! modules are built on top of it and on top of `burn::nn::Linear`, so they work on
//! any backend and differentiate without their own backward code. Their
//! There are no CUDA/WGPU/CubeCL kernels; candidates for fused kernels (`neo-synk`) are listed
//! in the documentation of each module.
//!
//! # API contract
//!
//! | module | input | output |
//! |-----------------------|----------------------------|-------------------|
//! | [`FactorizedEmbedding`] | `Tensor<2, Int>` `[B,T]` | `Tensor<3>` `[B,T,D]` |
//! | [`AdaptiveEmbedding`]   | `Tensor<2, Int>` `[B,T]` | `Tensor<3>` `[B,T,D]` |
//! | [`FourierEmbedding`] | `Tensor<D>`, last axis = `input_dim` | `Tensor<D>`, last axis = feature size |
//!
//! # Selection of positioning mechanism
//!
//! `FourierEmbedding` is an additional component, not part of `LlamaBlock` /
//! `QwenBlock`: for them the main mechanism is RoPE inside attention, and mix
//! it is not automatically needed with Fourier positions.

pub mod adaptive;
pub mod factorized;
pub mod fourier;

pub use adaptive::{AdaptiveEmbedding, AdaptiveEmbeddingConfig, AdaptiveTier, AdaptiveTierConfig};
pub use factorized::{FactorizedEmbedding, FactorizedEmbeddingConfig};
pub use fourier::{FourierEmbedding, FourierEmbeddingConfig, FrequencyScale};

#[cfg(test)]
mod tests;
