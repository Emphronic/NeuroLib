//! Attention: GQA is the canonical dense implementation, the rest is built around it.
//!
//! ```text
//! GQA owns Q/K/V/O, RoPE, QK-Norm, KV-cache
//! MQA     = GQA(num_kv_heads = 1)
//! Sliding = GQA + strip mask
//! Sparse = GQA + block-sparse mask (block indices are the basis of the future kernel)
//! Linear = separate algorithm (feature map φ = elu+1), O(T)
//! QkNorm = RMSNorm by head_dim for Q and K, up to RoPE
//! ```
//!
//! The order in GQA is: `Linear → heads → QK-Norm → RoPE → grouped QKᵀ/√Dh → mask →
//! softmax → ·V → merge → Wo`. K/V are not copied to `G = Hq/Hkv`: group
//! query heads shape the token axis.
//!
//! Additive masks `[1, 1, Tq, Tk]`. Causal mask - single [`causal_mask`]
//! from `transformer`; Sliding/Sparse patterns are built on the host once per
//! call (reference path, `O(Tq·Tk)` from memory). A query row fully
//! covered by a mask - panic (otherwise softmax gives NaN).
//!
//! RoPE, KV-cache and causal mask are currently being reused from `transformer`
//! (`RotaryEmbedding`, `KvCache`, `causal_mask`); `LlmAttention` is not touched.

mod gqa;
mod linear;
mod mqa;
mod qk_norm;
mod sliding;
mod sparse;

#[cfg(test)]
mod tests;

pub use gqa::{GqaAttention, GqaAttentionConfig};
pub use linear::{LinearAttention, LinearAttentionConfig, LinearAttentionState};
pub use mqa::{MqaAttention, MqaAttentionConfig};
pub use qk_norm::{QkNorm, QkNormConfig};
pub use sliding::{SlidingWindowAttention, SlidingWindowConfig};
pub use sparse::{SparseAttention, SparseAttentionConfig, SparsePattern};

use burn::prelude::*;

use crate::transformer::causal_mask;

/// Which pairs (query, key) are allowed.
#[derive(Clone, Debug, PartialEq)]
pub enum AttentionPattern {
    /// All (with a causal mask if the layer is causal).
    Dense,
    /// Window of `window_size` latest positions (including current).
    SlidingWindow { window_size: usize },
    /// Block-sparse pattern.
    Sparse(SparsePattern),
}

/// Dimensional checks common to GQA/MQA/Linear.
pub(crate) fn check_heads(
    d_model: usize,
    num_heads: usize,
    num_kv_heads: usize,
) -> Result<(), String> {
    if d_model == 0 {
        return Err("d_model must be > 0".into());
    }
    if num_heads == 0 {
        return Err("num_heads must be > 0".into());
    }
    if num_kv_heads == 0 {
        return Err("num_kv_heads must be > 0".into());
    }
    if !d_model.is_multiple_of(num_heads) {
        return Err(format!(
            "d_model ({d_model}) must be divisible by num_heads ({num_heads})"
        ));
    }
    if !num_heads.is_multiple_of(num_kv_heads) {
        return Err(format!(
            "num_heads ({num_heads}) must be divisible by num_kv_heads ({num_kv_heads})"
        ));
    }
    Ok(())
}

/// Additive mask `[1,1,tq,tk]` from predicate `allowed(i, j)`:
/// `i` is the query index inside the chunk, `j` is the key index.
pub(crate) fn host_mask(
    tq: usize,
    tk: usize,
    device: &Device,
    allowed: impl Fn(usize, usize) -> bool,
) -> Tensor<4> {
    let mut data = Vec::with_capacity(tq * tk);
    for i in 0..tq {
        let before = data.len();
        for j in 0..tk {
            data.push(if allowed(i, j) {
                0.0
            } else {
                f32::NEG_INFINITY
            });
        }
        assert!(
            data[before..].contains(&0.0),
            "attention: query row {i} is completely masked"
        );
    }
    Tensor::<4>::from_data(TensorData::new(data, [1, 1, tq, tk]), device)
}

/// Mask for the pattern. `past` — the number of tokens in the cache before the first request
/// (the absolute position of the request `i` is equal to `i + past`). `None` - no mask needed.
pub fn pattern_mask(
    pattern: &AttentionPattern,
    tq: usize,
    tk: usize,
    past: usize,
    causal: bool,
    device: &Device,
) -> Option<Tensor<4>> {
    match pattern {
        AttentionPattern::Dense => (causal && tq > 1).then(|| causal_mask(tq, tk, past, device)),
        AttentionPattern::SlidingWindow { window_size } => {
            let w = *window_size;
            assert!(w > 0, "window_size must be > 0");
            Some(host_mask(tq, tk, device, |i, j| {
                let qi = i + past;
                if causal {
                    j <= qi && qi - j < w
                } else {
                    qi.abs_diff(j) < w
                }
            }))
        }
        AttentionPattern::Sparse(p) => {
            let bs = p.block_size();
            let nqb = (past + tq).div_ceil(bs);
            let nkb = tk.div_ceil(bs);
            let table = p.block_table(nqb, nkb);
            Some(host_mask(tq, tk, device, |i, j| {
                let qi = i + past;
                table[qi / bs][j / bs] && (!causal || j <= qi)
            }))
        }
    }
}
