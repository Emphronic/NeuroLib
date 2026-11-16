//! General infrastructure decoder-only LLM: configuration, RoPE, KV-cache,
//! GQA-attention, FFN and the pre-norm block itself.
//!
//! ```text
//! h = x + Attention(RMSNorm(x))
//! y = h + FFN(RMSNorm(h))
//! ```
//!
//! Attention is computed without copying K/V: query-heads of the group
//! add up to the token axis (`[B, Hkv, G·T, Dh]`), so the same
//! `K`/`V` `[B, Hkv, Tk, Dh]` is reused by all `G = Hq/Hkv` heads.

use std::sync::Arc;

use burn::nn::{Linear, LinearConfig, RmsNorm, RmsNormConfig};
use burn::prelude::*;
use burn::tensor::activation::softmax;
use burn::tensor::{Bool, TensorData};

use crate::error::{NnError, fail};
use crate::ffn::{GatedFfn, GatedFfnConfig, Moe, MoeAuxStats, MoeConfig};

// Configuration ─────────────────────────

/// Type of attention based on the number of KV heads.
#[derive(Config, Debug, PartialEq)]
pub enum AttentionKind {
    /// `num_kv_heads == num_heads`.
    Mha,
    /// `num_heads % num_kv_heads == 0` (extreme cases MHA/MQA are also acceptable).
    Gqa,
    /// `num_kv_heads == 1`.
    Mqa,
}

impl AttentionKind {
    /// The type corresponding to the number of heads.
    pub fn infer(num_heads: usize, num_kv_heads: usize) -> Self {
        if num_kv_heads == num_heads {
            Self::Mha
        } else if num_kv_heads == 1 {
            Self::Mqa
        } else {
            Self::Gqa
        }
    }
}

/// View of FFN inside the block.
#[derive(Config, Debug, PartialEq)]
pub enum FfnKind {
    /// `W_o(SiLU(W_g x) ⊙ W_v x)`.
    SwiGlu,
    /// GELU shutter.
    GeGlu,
    /// ReLU gate.
    ReGlu,
    /// Mixture of Experts (needs `moe` in the config).
    Moe,
}

/// Pair layout in RoPE.
#[derive(Config, Debug, PartialEq)]
pub enum RopeLayout {
    /// Pairs `(x[2i], x[2i+1])` - as in the original RoFormer article.
    Interleaved,
    /// `(x[i], x[i + Dh/2])` pairs - Hugging Face Llama/Qwen weights layout.
    HalfSplit,
}

/// Common block configuration.
///
/// Bias: `use_bias` – Q/K/V projections, `use_o_bias` – `o_proj`,
/// `use_ffn_bias` - FFN (Qwen2: `use_bias = true`, the rest `false`).
#[derive(Config, Debug)]
pub struct LlmBlockConfig {
    pub d_model: usize,
    pub num_attention_heads: usize,
    pub num_kv_heads: usize,
    pub d_ff: usize,
    #[config(default = 10000.0)]
    pub rope_theta: f32,
    #[config(default = false)]
    pub use_bias: bool,
    #[config(default = 1e-5)]
    pub rms_norm_eps: f32,
    #[config(default = 2048)]
    pub max_seq_len: usize,
    #[config(default = "AttentionKind::Gqa")]
    pub attention_kind: AttentionKind,
    #[config(default = "FfnKind::SwiGlu")]
    pub ffn_kind: FfnKind,
    /// RMSNorm by `head_dim` for Q and K (up to RoPE), as in Qwen3.
    #[config(default = false)]
    pub use_qk_norm: bool,
    #[config(default = 0)]
    pub layer_index: usize,
    #[config(default = false)]
    pub use_o_bias: bool,
    #[config(default = false)]
    pub use_ffn_bias: bool,
    #[config(default = "RopeLayout::Interleaved")]
    pub rope_layout: RopeLayout,
    /// Mandatory for `ffn_kind == Moe`.
    pub moe: Option<MoeConfig>,
}

impl LlmBlockConfig {
    /// `d_model / num_attention_heads`.
    pub fn head_dim(&self) -> usize {
        self.d_model / self.num_attention_heads.max(1)
    }

    /// Checking parameters with a clear error message.
    pub fn validate(&self) -> Result<(), String> {
        let (d, h, kv) = (self.d_model, self.num_attention_heads, self.num_kv_heads);
        if d == 0 {
            return Err("d_model must be > 0".into());
        }
        if h == 0 {
            return Err("num_attention_heads must be > 0".into());
        }
        if kv == 0 {
            return Err("num_kv_heads must be > 0".into());
        }
        if d % h != 0 {
            return Err(format!(
                "d_model ({d}) must be divisible by num_attention_heads ({h})"
            ));
        }
        if h % kv != 0 {
            return Err(format!(
                "num_attention_heads ({h}) must be divisible by num_kv_heads ({kv})"
            ));
        }
        if (d / h) % 2 != 0 {
            return Err(format!("head_dim ({}) must be even (RoPE)", d / h));
        }
        match self.attention_kind {
            AttentionKind::Mha if kv != h => {
                return Err(format!(
                    "AttentionKind::Mha requires num_kv_heads == num_heads ({kv} != {h})"
                ));
            }
            AttentionKind::Mqa if kv != 1 => {
                return Err(format!(
                    "AttentionKind::Mqa requires num_kv_heads == 1, received {kv}"
                ));
            }
            _ => {}
        }
        if self.d_ff == 0 {
            return Err("d_ff must be > 0".into());
        }
        if self.max_seq_len == 0 {
            return Err("max_seq_len must be > 0".into());
        }
        if !(self.rms_norm_eps > 0.0) || !self.rms_norm_eps.is_finite() {
            return Err("rms_norm_eps must be > 0".into());
        }
        if !(self.rope_theta > 0.0) || !self.rope_theta.is_finite() {
            return Err("rope_theta must be > 0".into());
        }
        match (&self.ffn_kind, &self.moe) {
            (FfnKind::Moe, None) => return Err("ffn_kind = Moe requires moe field".into()),
            (FfnKind::Moe, Some(m)) => {
                m.validate()?;
                if m.d_model != d {
                    return Err(format!("moe.d_model ({}) != d_model ({d})", m.d_model));
                }
            }
            _ => {}
        }
        Ok(())
    }

    /// RoPE tables for this config (can be divided between layers via
    /// [`init_with_rotary`](Self::init_with_rotary)).
    pub fn build_rotary(&self) -> RotaryEmbedding {
        RotaryEmbedding::new(self.rope_theta, self.max_seq_len, self.head_dim())
            .with_layout(self.rope_layout.clone())
    }

    /// Create a block.
    ///
    /// # Panics
    /// If [`validate`](Self::validate) returned an error (the way without panic is
    /// [`try_init`](Self::try_init)).
    pub fn init(&self, device: &Device) -> LlmBlock {
        self.try_init(device).unwrap_or_else(|e| fail(e))
    }

    /// Like [`init`](Self::init), but with `Result` instead of panic.
    pub fn try_init(&self, device: &Device) -> Result<LlmBlock, NnError> {
        self.try_init_with_rotary(device, self.build_rotary())
    }

    /// Same as [`init`](Self::init), but with ready-made ones (common between layers)
    /// RoPE-tables.
    ///
    /// # Panics
    /// See [`init`](Self::init).
    pub fn init_with_rotary(&self, device: &Device, rotary: RotaryEmbedding) -> LlmBlock {
        self.try_init_with_rotary(device, rotary)
            .unwrap_or_else(|e| fail(e))
    }

    /// Like [`init_with_rotary`](Self::init_with_rotary), but with `Result`.
    pub fn try_init_with_rotary(
        &self,
        device: &Device,
        rotary: RotaryEmbedding,
    ) -> Result<LlmBlock, NnError> {
        self.validate()
            .map_err(|e| NnError::config("LlmBlockConfig", e))?;
        if rotary.head_dim != self.head_dim() {
            return Err(NnError::config(
                "LlmBlockConfig",
                format!(
                    "head_dim RoPE ({}) does not match the block's head_dim ({})",
                    rotary.head_dim,
                    self.head_dim()
                ),
            ));
        }
        let norm = RmsNormConfig::new(self.d_model).with_epsilon(self.rms_norm_eps as f64);
        Ok(LlmBlock {
            attn_norm: norm.init(device),
            ffn_norm: norm.init(device),
            attention: LlmAttention::new(self, rotary, device),
            ffn: LlmFfn::try_new(self, device)?,
            layer_index: self.layer_index,
        })
    }
}

// ───────────────────────── RoPE ─────────────────────────

/// Rotary position embedding. cos/sin tables are regular `f32` buffers (not
/// module parameters) shared between clones via `Arc`.
#[derive(Clone, Debug)]
pub struct RotaryEmbedding {
    pub theta: f32,
    pub max_seq_len: usize,
    pub head_dim: usize,
    pub layout: RopeLayout,
    cos: Arc<Vec<f32>>,
    sin: Arc<Vec<f32>>,
}

impl RotaryEmbedding {
    /// # Panics
    /// For odd/zero `head_dim`, zero `max_seq_len`, `theta <= 0`.
    pub fn new(theta: f32, max_seq_len: usize, head_dim: usize) -> Self {
        assert!(
            head_dim > 0 && head_dim % 2 == 0,
            "RoPE: head_dim ({head_dim}) must be even and > 0"
        );
        assert!(max_seq_len > 0, "RoPE: max_seq_len must be > 0");
        assert!(theta > 0.0 && theta.is_finite(), "RoPE: theta must be > 0");
        let half = head_dim / 2;
        let mut cos = Vec::with_capacity(max_seq_len * half);
        let mut sin = Vec::with_capacity(max_seq_len * half);
        for pos in 0..max_seq_len {
            for i in 0..half {
                let inv_freq = (theta as f64).powf(-2.0 * i as f64 / head_dim as f64);
                let angle = pos as f64 * inv_freq;
                cos.push(angle.cos() as f32);
                sin.push(angle.sin() as f32);
            }
        }
        Self {
            theta,
            max_seq_len,
            head_dim,
            layout: RopeLayout::Interleaved,
            cos: Arc::new(cos),
            sin: Arc::new(sin),
        }
    }

    pub fn with_layout(mut self, layout: RopeLayout) -> Self {
        self.layout = layout;
        self
    }

    /// Rotate `x: [B, H, T, Dh]`, where the token `i` has position `start_pos + i`.
    pub fn apply(&self, x: Tensor<4>, start_pos: usize) -> Tensor<4> {
        let [b, h, t, d] = x.dims();
        assert_eq!(
            d, self.head_dim,
            "RoPE: expected head_dim = {}, received {d}",
            self.head_dim
        );
        assert!(
            start_pos + t <= self.max_seq_len,
            "RoPE: positions {}..{} exceed max_seq_len = {}",
            start_pos,
            start_pos + t,
            self.max_seq_len
        );
        let half = d / 2;
        let device = x.device();
        let rows = start_pos * half..(start_pos + t) * half;
        let cos = Tensor::<2>::from_data(
            TensorData::new(self.cos[rows.clone()].to_vec(), [t, half]),
            &device,
        );
        let sin =
            Tensor::<2>::from_data(TensorData::new(self.sin[rows].to_vec(), [t, half]), &device);

        match self.layout {
            RopeLayout::Interleaved => {
                let x5 = x.reshape([b, h, t, half, 2]);
                let xe = x5.clone().narrow(4, 0, 1);
                let xo = x5.narrow(4, 1, 1);
                let c = cos.reshape([1, 1, t, half, 1]);
                let s = sin.reshape([1, 1, t, half, 1]);
                let re = xe.clone() * c.clone() - xo.clone() * s.clone();
                let ro = xe * s + xo * c;
                Tensor::cat(vec![re, ro], 4).reshape([b, h, t, d])
            }
            RopeLayout::HalfSplit => {
                let x1 = x.clone().narrow(3, 0, half);
                let x2 = x.narrow(3, half, half);
                let c = cos.reshape([1, 1, t, half]);
                let s = sin.reshape([1, 1, t, half]);
                let r1 = x1.clone() * c.clone() - x2.clone() * s.clone();
                let r2 = x1 * s + x2 * c;
                Tensor::cat(vec![r1, r2], 3)
            }
        }
    }
}

// ───────────────────────── mask and KV-cache ─────────────────────────

/// Additive causal mask `[1, 1, tq, tk]`: `0` where you can look,
/// `-inf` in the future. `past` — the number of tokens in the cache before the first request
/// (the request `i` sees the keys `j <= i + past`).
pub fn causal_mask(tq: usize, tk: usize, past: usize, device: &Device) -> Tensor<4> {
    let m = Tensor::<2, Bool>::tril_mask([tq, tk], past as i64, device);
    Tensor::<2>::zeros([tq, tk], device)
        .mask_fill(m, f32::NEG_INFINITY)
        .reshape([1, 1, tq, tk])
}

/// The mask is not needed for one request: it is the last one and sees all the keys.
fn causal_mask_opt(tq: usize, tk: usize, past: usize, device: &Device) -> Option<Tensor<4>> {
    (tq > 1).then(|| causal_mask(tq, tk, past, device))
}

/// KV-cache of one layer: `[B, Hkv, T, Dh]`. Inference state, not part of the module.
#[derive(Clone, Debug)]
pub struct KvCache {
    pub key: Option<Tensor<4>>,
    pub value: Option<Tensor<4>>,
    /// How many tokens are already in the cache (= next token position).
    pub position: usize,
    pub max_seq_len: usize,
}

impl KvCache {
    pub fn new(max_seq_len: usize) -> Self {
        Self {
            key: None,
            value: None,
            position: 0,
            max_seq_len,
        }
    }

    pub fn len(&self) -> usize {
        self.position
    }

    pub fn is_empty(&self) -> bool {
        self.position == 0
    }

    /// Add `[B, Hkv, T_new, Dh]` to the end along the time axis.
    pub fn append(&mut self, key: Tensor<4>, value: Tensor<4>) {
        let _ = self.append_and_get(key, value);
    }

    /// Like [`append`](Self::append), but immediately returns all accumulated
    /// `(K, V)` - without re-accessing `Option` via [`get`](Self::get).
    pub fn append_and_get(&mut self, key: Tensor<4>, value: Tensor<4>) -> (Tensor<4>, Tensor<4>) {
        let t_new = key.dims()[2];
        assert_eq!(
            value.dims()[2],
            t_new,
            "KvCache: K and V of different lengths"
        );
        assert!(
            self.position + t_new <= self.max_seq_len,
            "KvCache: overflow ({} + {} > {})",
            self.position,
            t_new,
            self.max_seq_len
        );
        let k_all = match self.key.take() {
            Some(old) => Tensor::cat(vec![old, key], 2),
            None => key,
        };
        let v_all = match self.value.take() {
            Some(old) => Tensor::cat(vec![old, value], 2),
            None => value,
        };
        self.key = Some(k_all.clone());
        self.value = Some(v_all.clone());
        self.position += t_new;
        (k_all, v_all)
    }

    pub fn get(&self) -> Option<(&Tensor<4>, &Tensor<4>)> {
        self.key.as_ref().zip(self.value.as_ref())
    }

    pub fn clear(&mut self) {
        self.key = None;
        self.value = None;
        self.position = 0;
    }
}

// ───────────────────────── attention ─────────────────────────

/// Self-attention for MHA/GQA/MQA with RoPE, optional QK-norm and KV-cache.
#[derive(Module, Debug)]
pub struct LlmAttention {
    pub q_proj: Linear,
    pub k_proj: Linear,
    pub v_proj: Linear,
    pub o_proj: Linear,
    /// RMSNorm for `head_dim` (weights `[Dh]`), only for `use_qk_norm`.
    pub q_norm: Option<RmsNorm>,
    pub k_norm: Option<RmsNorm>,
    #[module(skip)]
    pub rotary: RotaryEmbedding,
    #[module(skip)]
    pub num_heads: usize,
    #[module(skip)]
    pub num_kv_heads: usize,
    #[module(skip)]
    pub head_dim: usize,
}

impl LlmAttention {
    pub fn new(cfg: &LlmBlockConfig, rotary: RotaryEmbedding, device: &Device) -> Self {
        let (h, kv, dh) = (cfg.num_attention_heads, cfg.num_kv_heads, cfg.head_dim());
        let lin =
            |i: usize, o: usize, bias: bool| LinearConfig::new(i, o).with_bias(bias).init(device);
        let qk_norm = || {
            RmsNormConfig::new(dh)
                .with_epsilon(cfg.rms_norm_eps as f64)
                .init(device)
        };
        Self {
            q_proj: lin(cfg.d_model, h * dh, cfg.use_bias),
            k_proj: lin(cfg.d_model, kv * dh, cfg.use_bias),
            v_proj: lin(cfg.d_model, kv * dh, cfg.use_bias),
            o_proj: lin(h * dh, cfg.d_model, cfg.use_o_bias),
            q_norm: cfg.use_qk_norm.then(&qk_norm),
            k_norm: cfg.use_qk_norm.then(&qk_norm),
            rotary,
            num_heads: h,
            num_kv_heads: kv,
            head_dim: dh,
        }
    }

    /// `x: [B, T, D]` → `Q [B, Hq, T, Dh]`, `K, V [B, Hkv, T, Dh]`.
    /// RoPE (from the position of `start_pos`) applies to Q and K, not to V.
    pub fn project_qkv(&self, x: Tensor<3>, start_pos: usize) -> (Tensor<4>, Tensor<4>, Tensor<4>) {
        let [b, t, _] = x.dims();
        let heads = |y: Tensor<3>, n: usize| y.reshape([b, t, n, self.head_dim]).swap_dims(1, 2);
        let q = heads(self.q_proj.forward(x.clone()), self.num_heads);
        let k = heads(self.k_proj.forward(x.clone()), self.num_kv_heads);
        let v = heads(self.v_proj.forward(x), self.num_kv_heads);
        let (q, k) = match (&self.q_norm, &self.k_norm) {
            (Some(qn), Some(kn)) => (qn.forward(q), kn.forward(k)),
            _ => (q, k),
        };
        (
            self.rotary.apply(q, start_pos),
            self.rotary.apply(k, start_pos),
            v,
        )
    }

    /// `softmax(QKᵀ/√Dh + mask) V` → `o_proj`. `mask` - additive,
    /// broadcast on `[B, Hq, Tq, Tk]`.
    fn attend(
        &self,
        q: Tensor<4>,
        k: Tensor<4>,
        v: Tensor<4>,
        mask: Option<Tensor<4>>,
    ) -> Tensor<3> {
        let [b, hq, tq, dh] = q.dims();
        let tk = k.dims()[2];
        let hkv = self.num_kv_heads;
        let g = hq / hkv;

        // A group of query heads goes into the token axis: K/V are not copied.
        let scores = q
            .reshape([b, hkv, g * tq, dh])
            .matmul(k.swap_dims(2, 3))
            .div_scalar((dh as f32).sqrt())
            .reshape([b, hq, tq, tk]);
        let scores = match mask {
            Some(m) => scores + m,
            None => scores,
        };
        let probs = softmax(scores, 3).reshape([b, hkv, g * tq, tk]);
        let y = probs
            .matmul(v)
            .reshape([b, hq, tq, dh])
            .swap_dims(1, 2)
            .reshape([b, tq, hq * dh]);
        self.o_proj.forward(y)
    }

    /// Full sequence (train/prefill without cache).
    /// `mask = None` — causal; `Some` — additive mask as a whole (causal to it
    /// you need to enable it yourself).
    pub fn forward(&self, x: Tensor<3>, mask: Option<Tensor<4>>) -> Tensor<3> {
        let t = x.dims()[1];
        let device = x.device();
        let (q, k, v) = self.project_qkv(x, 0);
        let mask = mask.or_else(|| causal_mask_opt(t, t, 0, &device));
        self.attend(q, k, v, mask)
    }

    /// Inference with cache: `x` - new tokens (1 or chunk) from position `position`,
    /// which must coincide with `cache.len()`.
    pub fn forward_cached(&self, x: Tensor<3>, cache: &mut KvCache, position: usize) -> Tensor<3> {
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
        let mask = causal_mask_opt(t, tk, position, &device);
        self.attend(q, k_all, v_all, mask)
    }
}

// ───────────────────────── FFN ─────────────────────────

/// SwiGLU FFN: `W_o(SiLU(W_g x) ⊙ W_v x)` is [`GatedFfn`] with SiLU activation
/// (`GatedFfnConfig::swiglu`).
pub type SwiGluFfn = GatedFfn;

/// Block FFN: regular gated or MoE.
#[derive(Module, Debug)]
pub enum LlmFfn {
    Gated(GatedFfn),
    /// Single implementation of MoE from [`crate::ffn`].
    Moe(Moe),
}

impl LlmFfn {
    /// # Panics
    /// If `cfg.moe` is not specified or invalid for `FfnKind::Moe` (path without
    /// panic — [`try_new`](Self::try_new)).
    pub fn new(cfg: &LlmBlockConfig, device: &Device) -> Self {
        Self::try_new(cfg, device).unwrap_or_else(|e| fail(e))
    }

    /// Like [`new`](Self::new), but with `Result`.
    pub fn try_new(cfg: &LlmBlockConfig, device: &Device) -> Result<Self, NnError> {
        let gated = |c: GatedFfnConfig| Self::Gated(c.with_bias(cfg.use_ffn_bias).init(device));
        Ok(match cfg.ffn_kind {
            FfnKind::SwiGlu => gated(GatedFfnConfig::swiglu(cfg.d_model, cfg.d_ff)),
            FfnKind::GeGlu => gated(GatedFfnConfig::geglu(cfg.d_model, cfg.d_ff)),
            FfnKind::ReGlu => gated(GatedFfnConfig::reglu(cfg.d_model, cfg.d_ff)),
            FfnKind::Moe => {
                let moe = cfg.moe.as_ref().ok_or_else(|| {
                    NnError::config("LlmBlockConfig", "ffn_kind = Moe requires moe field")
                })?;
                Self::Moe(moe.try_init(device)?)
            }
        })
    }

    /// `[B, T, D] → [B, T, D]` and router statistics (only for MoE).
    ///
    /// # Panics
    /// If MoE failed to read routing indexes (see.
    /// [`try_forward`](Self::try_forward)).
    pub fn forward(&self, x: Tensor<3>) -> (Tensor<3>, Option<MoeAuxStats>) {
        self.try_forward(x).unwrap_or_else(|e| fail(e))
    }

    /// Like [`forward`](Self::forward), but with `Result`.
    pub fn try_forward(&self, x: Tensor<3>) -> Result<(Tensor<3>, Option<MoeAuxStats>), NnError> {
        Ok(match self {
            Self::Gated(f) => (f.forward(x), None),
            Self::Moe(m) => {
                let (y, aux) = m.try_forward_with_aux(x)?;
                (y, Some(aux))
            }
        })
    }
}

// ──────────────────────── block ─────────────────────────

/// Pre-norm decoder block.
#[derive(Module, Debug)]
pub struct LlmBlock {
    pub attn_norm: RmsNorm,
    pub ffn_norm: RmsNorm,
    pub attention: LlmAttention,
    pub ffn: LlmFfn,
    #[module(skip)]
    pub layer_index: usize,
}

impl LlmBlock {
    /// Causal passage through the entire sequence.
    pub fn forward(&self, x: Tensor<3>) -> Tensor<3> {
        self.forward_with_mask(x, None)
    }

    /// `mask = None` — causal; otherwise additive mask `[*, *, T, T]`.
    pub fn forward_with_mask(&self, x: Tensor<3>, mask: Option<Tensor<4>>) -> Tensor<3> {
        self.forward_with_aux(x, mask).0
    }

    /// Like `forward_with_mask`, plus MoE router statistics (`None` for dense FFN).
    pub fn forward_with_aux(
        &self,
        x: Tensor<3>,
        mask: Option<Tensor<4>>,
    ) -> (Tensor<3>, Option<MoeAuxStats>) {
        let h = x.clone() + self.attention.forward(self.attn_norm.forward(x), mask);
        let (f, aux) = self.ffn.forward(self.ffn_norm.forward(h.clone()));
        (h + f, aux)
    }

    /// Inference with KV-cache (`position == cache.len()`).
    pub fn forward_cached(&self, x: Tensor<3>, cache: &mut KvCache, position: usize) -> Tensor<3> {
        self.forward_cached_with_aux(x, cache, position).0
    }

    pub fn forward_cached_with_aux(
        &self,
        x: Tensor<3>,
        cache: &mut KvCache,
        position: usize,
    ) -> (Tensor<3>, Option<MoeAuxStats>) {
        let h = x.clone()
            + self
                .attention
                .forward_cached(self.attn_norm.forward(x), cache, position);
        let (f, aux) = self.ffn.forward(self.ffn_norm.forward(h.clone()));
        (h + f, aux)
    }

    /// Empty cache under `max_seq_len` of this block.
    pub fn new_cache(&self) -> KvCache {
        KvCache::new(self.attention.rotary.max_seq_len)
    }
}
