//! GQA is the canonical dense implementation (MHA for `Hkv = Hq`, MQA for `Hkv = 1`).
//!
//! Along the way, it also owns RoPE (preceded by QK-Norm) and works with `KvCache` from
//! `transformer`. Sliding/Sparse patterns are mixed through the mask
//! ([`forward_with_pattern`](GqaAttention::forward_with_pattern)).

use burn::nn::{Dropout, DropoutConfig, Linear, LinearConfig};
use burn::prelude::*;

use crate::error::{NnError, fail};
use burn::tensor::activation::softmax;

use super::{AttentionPattern, QkNorm, QkNormConfig, check_heads, pattern_mask};
use crate::transformer::{KvCache, RopeLayout, RotaryEmbedding, causal_mask};

/// Configuration [`GqaAttention`].
///
/// `causal` defaults to `true` (decoder-only; same as `LlmAttention`).
#[derive(Config, Debug)]
pub struct GqaAttentionConfig {
    pub d_model: usize,
    pub num_heads: usize,
    pub num_kv_heads: usize,
    /// Bias in Q/K/V.
    #[config(default = false)]
    pub bias: bool,
    /// Bias in `o_proj`.
    #[config(default = false)]
    pub o_bias: bool,
    /// Dropout on probabilities attention.
    #[config(default = "0.0")]
    pub dropout: f64,
    #[config(default = true)]
    pub causal: bool,
    #[config(default = false)]
    pub use_qk_norm: bool,
    #[config(default = "1e-5")]
    pub qk_norm_eps: f32,
    /// Use RoPE (requires even `head_dim`).
    #[config(default = true)]
    pub rope: bool,
    #[config(default = "10000.0")]
    pub rope_theta: f32,
    #[config(default = 2048)]
    pub max_seq_len: usize,
    #[config(default = "RopeLayout::Interleaved")]
    pub rope_layout: RopeLayout,
}

impl GqaAttentionConfig {
    pub fn head_dim(&self) -> usize {
        self.d_model / self.num_heads.max(1)
    }

    pub fn validate(&self) -> Result<(), String> {
        check_heads(self.d_model, self.num_heads, self.num_kv_heads)?;
        if !(0.0..1.0).contains(&self.dropout) {
            return Err("dropout should be in [0, 1)".into());
        }
        if self.max_seq_len == 0 {
            return Err("max_seq_len must be > 0".into());
        }
        if self.rope {
            if !self.head_dim().is_multiple_of(2) {
                return Err(format!(
                    "head_dim ({}) must be even (RoPE)",
                    self.head_dim()
                ));
            }
            if !(self.rope_theta.is_finite() && self.rope_theta > 0.0) {
                return Err("rope_theta must be > 0".into());
            }
        }
        if self.use_qk_norm {
            QkNormConfig::new(self.head_dim())
                .with_eps(self.qk_norm_eps)
                .validate()?;
        }
        Ok(())
    }

    /// Create a layer; `Err` if [`validate`](Self::validate) failed.
    pub fn try_init(&self, device: &Device) -> Result<GqaAttention, NnError> {
        self.validate()
            .map_err(|e| NnError::config("GqaAttentionConfig", e))?;
        let (h, kv, dh) = (self.num_heads, self.num_kv_heads, self.head_dim());
        let lin =
            |i: usize, o: usize, bias: bool| LinearConfig::new(i, o).with_bias(bias).init(device);
        Ok(GqaAttention {
            q_proj: lin(self.d_model, h * dh, self.bias),
            k_proj: lin(self.d_model, kv * dh, self.bias),
            v_proj: lin(self.d_model, kv * dh, self.bias),
            o_proj: lin(h * dh, self.d_model, self.o_bias),
            qk_norm: self.use_qk_norm.then(|| {
                QkNormConfig::new(dh)
                    .with_eps(self.qk_norm_eps)
                    .init(device)
            }),
            dropout: DropoutConfig::new(self.dropout).init(),
            rotary: self.rope.then(|| {
                RotaryEmbedding::new(self.rope_theta, self.max_seq_len, dh)
                    .with_layout(self.rope_layout.clone())
            }),
            num_heads: h,
            num_kv_heads: kv,
            head_dim: dh,
            max_seq_len: self.max_seq_len,
            causal: self.causal,
        })
    }

    /// Create a layer.
    ///
    /// # Panics
    /// If [`validate`](Self::validate) returned an error (the way without panic is [`try_init`](Self::try_init)).
    pub fn init(&self, device: &Device) -> GqaAttention {
        self.try_init(device).unwrap_or_else(|e| fail(e))
    }
}

/// Grouped-query self-attention.
#[derive(Module, Debug)]
pub struct GqaAttention {
    pub q_proj: Linear,
    pub k_proj: Linear,
    pub v_proj: Linear,
    pub o_proj: Linear,
    pub qk_norm: Option<QkNorm>,
    pub dropout: Dropout,
    #[module(skip)]
    pub rotary: Option<RotaryEmbedding>,
    #[module(skip)]
    pub num_heads: usize,
    #[module(skip)]
    pub num_kv_heads: usize,
    #[module(skip)]
    pub head_dim: usize,
    #[module(skip)]
    pub max_seq_len: usize,
    #[module(skip)]
    pub causal: bool,
}

impl GqaAttention {
    /// `x: [B,T,D]` → `Q [B,Hq,T,Dh]`, `K,V [B,Hkv,T,Dh]`; QK-Norm then RoPE
    /// (from the position of `start_pos`) for Q and K.
    pub fn project_qkv(&self, x: Tensor<3>, start_pos: usize) -> (Tensor<4>, Tensor<4>, Tensor<4>) {
        let [b, t, _] = x.dims();
        let heads = |y: Tensor<3>, n: usize| y.reshape([b, t, n, self.head_dim]).swap_dims(1, 2);
        let q = heads(self.q_proj.forward(x.clone()), self.num_heads);
        let k = heads(self.k_proj.forward(x.clone()), self.num_kv_heads);
        let v = heads(self.v_proj.forward(x), self.num_kv_heads);
        let (q, k) = match &self.qk_norm {
            Some(n) => n.forward(q, k),
            None => (q, k),
        };
        match &self.rotary {
            Some(r) => (r.apply(q, start_pos), r.apply(k, start_pos), v),
            None => (q, k, v),
        }
    }

    /// `QKᵀ/√Dh (+ mask)`: `[B,Hq,Tq,Tk]`. K/V are not copied.
    pub fn attention_scores(
        &self,
        q: Tensor<4>,
        k: Tensor<4>,
        mask: Option<Tensor<4>>,
    ) -> Tensor<4> {
        let [b, hq, tq, dh] = q.dims();
        let tk = k.dims()[2];
        let hkv = self.num_kv_heads;
        let g = hq / hkv;
        let scores = q
            .reshape([b, hkv, g * tq, dh])
            .matmul(k.swap_dims(2, 3))
            .div_scalar((dh as f32).sqrt())
            .reshape([b, hq, tq, tk]);
        match mask {
            Some(m) => scores + m,
            None => scores,
        }
    }

    /// `probs · V`: `[B,Hq,Tq,Tk] × [B,Hkv,Tk,Dh] → [B,Hq,Tq,Dh]`.
    pub fn attention_values(&self, probs: Tensor<4>, v: Tensor<4>) -> Tensor<4> {
        let [b, hq, tq, tk] = probs.dims();
        let dh = v.dims()[3];
        let hkv = self.num_kv_heads;
        probs
            .reshape([b, hkv, (hq / hkv) * tq, tk])
            .matmul(v)
            .reshape([b, hq, tq, dh])
    }

    /// `[B,Hq,T,Dh] → [B,T,Hq·Dh]`.
    pub fn merge_heads(&self, y: Tensor<4>) -> Tensor<3> {
        let [b, h, t, dh] = y.dims();
        y.swap_dims(1, 2).reshape([b, t, h * dh])
    }

    /// Full attention to ready-made Q/K/V (with RoPE) and additive mask.
    pub fn attend(
        &self,
        q: Tensor<4>,
        k: Tensor<4>,
        v: Tensor<4>,
        mask: Option<Tensor<4>>,
    ) -> Tensor<3> {
        let probs = self
            .dropout
            .forward(softmax(self.attention_scores(q, k, mask), 3));
        self.o_proj
            .forward(self.merge_heads(self.attention_values(probs, v)))
    }

    /// Full sequence. `mask = None` — causal (if `self.causal`);
    /// `Some` is the entire additive mask; you need to include causal in it yourself.
    pub fn forward(&self, x: Tensor<3>, mask: Option<Tensor<4>>) -> Tensor<3> {
        let t = x.dims()[1];
        let device = x.device();
        let (q, k, v) = self.project_qkv(x, 0);
        let mask = mask.or_else(|| (self.causal && t > 1).then(|| causal_mask(t, t, 0, &device)));
        self.attend(q, k, v, mask)
    }

    /// Like [`forward`](Self::forward), but the mask is built from a pattern.
    pub fn forward_with_pattern(&self, x: Tensor<3>, pattern: &AttentionPattern) -> Tensor<3> {
        let t = x.dims()[1];
        let mask = pattern_mask(pattern, t, t, 0, self.causal, &x.device());
        let (q, k, v) = self.project_qkv(x, 0);
        self.attend(q, k, v, mask)
    }

    /// Empty cache under `max_seq_len` layer.
    pub fn new_cache(&self) -> KvCache {
        KvCache::new(self.max_seq_len)
    }

    /// Inference with cache; `position` must coincide with `cache.len()`.
    pub fn forward_cached(&self, x: Tensor<3>, cache: &mut KvCache, position: usize) -> Tensor<3> {
        self.forward_cached_with_pattern(x, cache, position, &AttentionPattern::Dense)
    }

    /// Inference with cache and pattern. The cache stores the entire history; pattern only
    /// limits visibility.
    pub fn forward_cached_with_pattern(
        &self,
        x: Tensor<3>,
        cache: &mut KvCache,
        position: usize,
        pattern: &AttentionPattern,
    ) -> Tensor<3> {
        assert_eq!(
            position,
            cache.len(),
            "forward_cached: position ({position}) != cache length ({})",
            cache.len()
        );
        let t = x.dims()[1];
        let device = x.device();
        let (q, k, v) = self.project_qkv(x, position);
        let (k_all, v_all) = cache.append_and_get(k, v);
        let tk = k_all.dims()[2];
        let mask = pattern_mask(pattern, t, tk, position, self.causal, &device);
        self.attend(q, k_all, v_all, mask)
    }
}
