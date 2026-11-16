//! Linear attention (Katharopoulos et al., 2020, arXiv:2006.16236).
//!
//! ```text
//! φ(x) = elu(x) + 1                       (> 0)
//! causal:  S_t = Σ_{j≤t} φ(K_j)ᵀ V_j,   Z_t = Σ_{j≤t} φ(K_j)
//! y_t = φ(Q_t) S_t / (φ(Q_t)·Z_t + eps)
//! ```
//!
//! Not a softmax-attention approximation, but a separate algorithm. In causal mode
//! the fixed size `(S, Z)` state allows you to generate a token in
//! token without history recalculation ([`LinearAttentionState`]).
//! Reference: sequential scan by `T` (time `O(T)`), without RoPE and KV-cache.

use burn::nn::{Linear, LinearConfig};
use burn::prelude::*;

use crate::error::{NnError, fail};

/// Configuration [`LinearAttention`].
#[derive(Config, Debug)]
pub struct LinearAttentionConfig {
    pub d_model: usize,
    pub num_heads: usize,
    #[config(default = "1e-5")]
    pub eps: f32,
    #[config(default = true)]
    pub causal: bool,
    #[config(default = false)]
    pub bias: bool,
}

impl LinearAttentionConfig {
    pub fn head_dim(&self) -> usize {
        self.d_model / self.num_heads.max(1)
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.d_model == 0 || self.num_heads == 0 {
            return Err("d_model and num_heads must be > 0".into());
        }
        if self.d_model % self.num_heads != 0 {
            return Err(format!(
                "d_model ({}) must be divisible by num_heads ({})",
                self.d_model, self.num_heads
            ));
        }
        if !(self.eps > 0.0) || !self.eps.is_finite() {
            return Err("eps must be finite and > 0".into());
        }
        Ok(())
    }

    /// Create a layer; `Err` if [`validate`](Self::validate) failed.
    pub fn try_init(&self, device: &Device) -> Result<LinearAttention, NnError> {
        self.validate()
            .map_err(|e| NnError::config("LinearAttentionConfig", e))?;
        let d = self.d_model;
        let lin = || LinearConfig::new(d, d).with_bias(self.bias).init(device);
        Ok(LinearAttention {
            q_proj: lin(),
            k_proj: lin(),
            v_proj: lin(),
            o_proj: lin(),
            num_heads: self.num_heads,
            head_dim: self.head_dim(),
            eps: self.eps,
            causal: self.causal,
        })
    }

    /// Create a layer.
    ///
    /// # Panics
    /// If [`validate`](Self::validate) returned an error (the way without panic is [`try_init`](Self::try_init)).
    pub fn init(&self, device: &Device) -> LinearAttention {
        self.try_init(device).unwrap_or_else(|e| fail(e))
    }
}

/// Autoregressive state of one layer.
#[derive(Clone, Debug)]
pub struct LinearAttentionState {
    /// `Σ φ(K)ᵀ V`: `[B, H, Dh, Dh]`.
    pub kv_state: Tensor<4>,
    /// `Σ φ(K)`: `[B, H, Dh]`.
    pub k_state: Tensor<3>,
}

impl LinearAttentionState {
    pub fn new(batch: usize, num_heads: usize, head_dim: usize, device: &Device) -> Self {
        Self {
            kv_state: Tensor::zeros([batch, num_heads, head_dim, head_dim], device),
            k_state: Tensor::zeros([batch, num_heads, head_dim], device),
        }
    }
}

#[derive(Module, Debug)]
pub struct LinearAttention {
    pub q_proj: Linear,
    pub k_proj: Linear,
    pub v_proj: Linear,
    pub o_proj: Linear,
    #[module(skip)]
    pub num_heads: usize,
    #[module(skip)]
    pub head_dim: usize,
    #[module(skip)]
    pub eps: f32,
    #[module(skip)]
    pub causal: bool,
}

/// `elu(x) + 1 = exp(min(x,0)) + max(x,0)`.
fn feature_map<const D: usize>(x: Tensor<D>) -> Tensor<D> {
    x.clone().clamp_max(0.0).exp() + x.clamp_min(0.0)
}

impl LinearAttention {
    pub fn new_state(&self, batch: usize, device: &Device) -> LinearAttentionState {
        LinearAttentionState::new(batch, self.num_heads, self.head_dim, device)
    }

    /// `φ(Q), φ(K), V`: `[B, H, T, Dh]`.
    fn project(&self, x: Tensor<3>) -> (Tensor<4>, Tensor<4>, Tensor<4>) {
        let [b, t, _] = x.dims();
        let heads = |y: Tensor<3>| {
            y.reshape([b, t, self.num_heads, self.head_dim])
                .swap_dims(1, 2)
        };
        (
            feature_map(heads(self.q_proj.forward(x.clone()))),
            feature_map(heads(self.k_proj.forward(x.clone()))),
            heads(self.v_proj.forward(x)),
        )
    }

    fn finish(&self, y: Tensor<4>) -> Tensor<3> {
        let [b, h, t, dh] = y.dims();
        self.o_proj
            .forward(y.swap_dims(1, 2).reshape([b, t, h * dh]))
    }

    /// The complete sequence is `[B,T,D] → [B,T,D]`.
    /// Causal — scan by token, otherwise the global amount.
    pub fn forward(&self, x: Tensor<3>) -> Tensor<3> {
        let [b, t, _] = x.dims();
        let device = x.device();
        let (q, k, v) = self.project(x);
        let y = if self.causal {
            let mut state = self.new_state(b, &device);
            self.scan(q, k, v, &mut state)
        } else {
            let s = k.clone().swap_dims(2, 3).matmul(v); // [B,H,Dh,Dh]
            let z = k.sum_dim(2); // [B,H,1,Dh]
            let num = q.clone().matmul(s);
            let den = (q * z).sum_dim(3).add_scalar(self.eps);
            num / den
        };
        debug_assert_eq!(y.dims()[2], t);
        self.finish(y)
    }

    /// The next chunk (1 or more tokens) with state accumulation. Only causal.
    pub fn forward_stateful(&self, x: Tensor<3>, state: &mut LinearAttentionState) -> Tensor<3> {
        assert!(self.causal, "forward_stateful requires causal = true");
        let (q, k, v) = self.project(x);
        let y = self.scan(q, k, v, state);
        self.finish(y)
    }

    /// Serial scan; updates `state`.
    fn scan(
        &self,
        q: Tensor<4>,
        k: Tensor<4>,
        v: Tensor<4>,
        state: &mut LinearAttentionState,
    ) -> Tensor<4> {
        let [b, h, t, dh] = q.dims();
        let mut outs = Vec::with_capacity(t);
        for i in 0..t {
            let qi = q.clone().narrow(2, i, 1); // [B,H,1,Dh]
            let ki = k.clone().narrow(2, i, 1);
            let vi = v.clone().narrow(2, i, 1);
            state.kv_state = state.kv_state.clone() + ki.clone().swap_dims(2, 3).matmul(vi);
            state.k_state = state.k_state.clone() + ki.reshape([b, h, dh]);
            let num = qi.clone().matmul(state.kv_state.clone()); // [B,H,1,Dh]
            let z = state.k_state.clone().reshape([b, h, 1, dh]);
            let den = (qi * z).sum_dim(3).add_scalar(self.eps); // [B,H,1,1]
            outs.push(num / den);
        }
        Tensor::cat(outs, 2)
    }
}
