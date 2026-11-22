//! MoE — Mixture of Experts with top-k routing: **the only one**
//! implementation of MoE in `neo-nn`.
//!
//! Instead of one FFN, we keep several “experts”, and a small router for
//! each token selects `top_k` from them:
//!
//! ```text
//!                     ┌→ Expert 0 ─┐
//!                     ├→ Expert 1 ─┤
//!  x → Router → top-k ├→ Expert 2 ─┤ → weighted sum (+ shared) → y
//!                     ├→ Expert 3 ─┤
//!                     └→ Expert 4 ─┘
//! ```
//!
//! For each token:
//!
//! ```text
//! probs   = softmax(W_r·x [+ b_r])        [num_experts]
//! (p, i) = top_k(probs) values and indices
//! w = p / Σp (normalize_top_k) weights of selected experts, sum = 1
//! y       = Σ_j w_j · Expert_{i_j}(x)  (+ Shared(x))
//! ```
//!
//! For example, with `num_experts = 8`, `top_k = 2` the router can choose
//! `expert 3 : 0.72`, `expert 6 : 0.28`, and then
//! `y = 0.72·Expert3(x) + 0.28·Expert6(x)`. The remaining six experts for
//! of this token are not calculated.
//!
//! Renormalization of the selected probabilities gives the same thing as softmax by
//! `top_k` logits (as in Mixtral). They remain with `normalize_top_k = false`
//! "raw" probabilities (as in Qwen-MoE).
//!
//! Each expert is [`GatedFfn`] (default SwiGLU) with internal width
//! `d_ff`.
//!
//! ## Capacity and overflow
//!
//! `capacity = ceil(capacity_factor · N · top_k / num_experts)`, no more than `N`.
//! By default `capacity_factor = f32::MAX` - tokens are not discarded (for
//! training usually put `1.25`). Tokens in excess of capacity are processed according to
//! [`OverflowPolicy`] (currently `Drop`: the contribution of this expert is zero,
//! remains residual). Overflow priority: all first selections first,
//! then the second ones, etc.
//!
//! ## Shared expert
//!
//! With `shared_expert = true` another [`GatedFfn`] is added, which sees
//! all tokens; its output is added to the output of the routed experts.
//!
//! ## Implementation
//!
//! * [`Moe::forward`] - sparse: tokens are grouped by experts,
//!   each expert is evaluated only on its tokens. For this purpose the indexes
//!   routings are read to the host (one sync per call).
//! * [`Moe::forward_dense`] - standard: each expert is evaluated on all
//!   tokens and is weighted by a mask. Slow, needed for testing and debugging.
//! * [`Moe::forward_with_aux`] - the same as `forward`, plus statistics
//!   router [`MoeAuxStats`], from which the balancing loss Switch is taken
//!   Transformer (Fedus et al., 2021): `E · Σ f_e · P_e`. Equal to 1 when ideal
//!   uniform loading.
//!
//! ## Errors
//!
//! `try_*` methods return [`NnError`]; methods of the same name without `try_`
//! panic with the same message.
//!
//! # Example
//!
//! ```ignore
//! use burn::tensor::Device;
//! use burn::tensor::Tensor;
//! use neo_nn::ffn::MoeConfig;
//!
//! let device = Device::flex(); // CPU backend (feature "flex")
//!
//! let moe = MoeConfig::new(64, 172, 8).with_top_k(2).init(&device);
//! let y = moe.forward(Tensor::<3>::zeros([2, 10, 64], &device));
//! ```

use burn::nn::{Linear, LinearConfig};
use burn::prelude::*;
use burn::tensor::activation::softmax;
use burn::tensor::{IndexingUpdateOp, Int, TensorData};

use super::gated::{GateActivation, GatedFfn, GatedFfnConfig};
use crate::error::{NnError, data_to_vec, fail};

/// What to do with tokens in excess of the expert's capacity.
#[derive(Config, Debug, PartialEq)]
#[non_exhaustive]
pub enum OverflowPolicy {
    /// Reset: the contribution of this expert for the token is zero (remains residual).
    Drop,
}

/// Configuration [`Moe`].
///
/// `capacity = ceil(capacity_factor · N · top_k / num_experts)`, no more than `N`.
#[derive(Config, Debug)]
pub struct MoeConfig {
    /// Input and output size (`d_model`).
    pub d_model: usize,
    /// Internal width of each expert (and shared expert).
    pub d_ff: usize,
    /// Number of experts.
    pub num_experts: usize,
    /// How many experts are selected for the token.
    #[config(default = 2)]
    pub top_k: usize,
    /// Activating the shutter inside experts.
    #[config(default = "GateActivation::Silu")]
    pub activation: GateActivation,
    /// Whether to use biases in Expert Advisors.
    #[config(default = true)]
    pub bias: bool,
    /// Router offset (none by default).
    #[config(default = false)]
    pub router_bias: bool,
    /// Expert capacity multiplier; `f32::MAX` - do not discard tokens.
    #[config(default = "f32::MAX")]
    pub capacity_factor: f32,
    /// Policy for tokens over capacity.
    #[config(default = "OverflowPolicy::Drop")]
    pub overflow_policy: OverflowPolicy,
    /// Renormalize the selected weights so that the sum per token is 1.
    #[config(default = true)]
    pub normalize_top_k: bool,
    /// Add shared expert (visible to all tokens).
    #[config(default = false)]
    pub shared_expert: bool,
}

/// `ceil(capacity_factor · tokens · top_k / num_experts)`, in `1..=tokens`.
fn capacity_for(capacity_factor: f32, tokens: usize, top_k: usize, num_experts: usize) -> usize {
    let c =
        (capacity_factor as f64 * tokens as f64 * top_k as f64 / num_experts.max(1) as f64).ceil();
    (c as usize).clamp(1, tokens.max(1))
}

impl MoeConfig {
    /// Check parameters; error text - for people.
    pub fn validate(&self) -> Result<(), String> {
        if self.d_model == 0 {
            return Err("moe.d_model must be > 0".into());
        }
        if self.d_ff == 0 {
            return Err("moe.d_ff must be > 0".into());
        }
        if self.num_experts == 0 {
            return Err("moe.num_experts must be > 0".into());
        }
        if self.top_k == 0 {
            return Err("moe.top_k must be > 0".into());
        }
        if self.top_k > self.num_experts {
            return Err(format!(
                "moe.top_k ({}) > num_experts ({})",
                self.top_k, self.num_experts
            ));
        }
        if self.capacity_factor.is_nan() || self.capacity_factor <= 0.0 {
            return Err("moe.capacity_factor must be > 0".into());
        }
        Ok(())
    }

    /// Capacity of one expert for `tokens` tokens.
    pub fn capacity(&self, tokens: usize) -> usize {
        capacity_for(self.capacity_factor, tokens, self.top_k, self.num_experts)
    }

    /// Create a layer; `Err` if [`validate`](Self::validate) failed.
    pub fn try_init(&self, device: &Device) -> Result<Moe, NnError> {
        self.validate().map_err(|e| NnError::config("Moe", e))?;
        let expert = GatedFfnConfig::new(self.d_model, self.d_ff, self.activation.clone())
            .with_bias(self.bias);
        Ok(Moe {
            router: LinearConfig::new(self.d_model, self.num_experts)
                .with_bias(self.router_bias)
                .init(device),
            experts: (0..self.num_experts).map(|_| expert.init(device)).collect(),
            shared: self.shared_expert.then(|| expert.init(device)),
            top_k: self.top_k,
            capacity_factor: self.capacity_factor,
            normalize_top_k: self.normalize_top_k,
            overflow_policy: self.overflow_policy.clone(),
        })
    }

    /// Create a layer.
    ///
    /// # Panics
    /// If [`validate`](Self::validate) returned an error (the way without panic is
    /// [`try_init`](Self::try_init)).
    pub fn init(&self, device: &Device) -> Moe {
        self.try_init(device).unwrap_or_else(|e| fail(e))
    }
}

/// Router statistics for balancing loss.
#[derive(Debug, Clone)]
pub struct MoeAuxStats {
    /// Probabilities for all experts `[N, E]` (differentiable).
    pub router_probs: Tensor<2>,
    /// Selected experts `[N, k]`.
    pub expert_indices: Tensor<2, Int>,
    /// The share of appointments for each expert before discarding `[E]` (sum 1).
    pub expert_load: Tensor<1>,
    /// How many assignments (token, expert) are discarded by capacity.
    pub dropped_assignments: usize,
}

impl MoeAuxStats {
    /// Switch-loss `E · Σ f_e · P_e` (shape `[1]`); equals 1 for uniform
    /// loading. Add to the main loss with a small coefficient
    /// (about `1e-2`).
    pub fn load_balance_loss(&self) -> Tensor<1> {
        let e = self.expert_load.dims()[0];
        let p = self.router_probs.clone().mean_dim(0).reshape([e]);
        self.expert_load.clone().mul(p).sum().mul_scalar(e as f32)
    }
}

/// Mixture of Experts: router + `num_experts` layers [`GatedFfn`] +
/// optional shared expert.
#[derive(Module, Debug)]
pub struct Moe {
    /// Router `W_r`: `d_model → num_experts`.
    pub router: Linear,
    /// Experts.
    pub experts: Vec<GatedFfn>,
    /// Shared expert (visible to all tokens).
    pub shared: Option<GatedFfn>,
    /// How many experts are selected for the token.
    #[module(skip)]
    pub top_k: usize,
    /// Expert capacity multiplier.
    #[module(skip)]
    pub capacity_factor: f32,
    /// Whether to renormalize the selected weights.
    #[module(skip)]
    pub normalize_top_k: bool,
    /// Policy for tokens over capacity.
    #[module(skip)]
    pub overflow_policy: OverflowPolicy,
}

impl Moe {
    /// Number of experts.
    pub fn num_experts(&self) -> usize {
        self.experts.len()
    }

    /// Capacity of one expert for `tokens` tokens.
    pub fn capacity(&self, tokens: usize) -> usize {
        capacity_for(self.capacity_factor, tokens, self.top_k, self.num_experts())
    }

    /// Routing for a flat batch of `[tokens, d_model]` tokens.
    ///
    /// Returns `(probs, weights, indices)`: probabilities for all experts
    /// `[tokens, E]`, weights of selected experts and their indices (descending
    /// probability), both `[tokens, top_k]`. For `normalize_top_k` the sum of the weights
    /// along the line is equal to 1.
    pub fn route(&self, x: Tensor<2>) -> (Tensor<2>, Tensor<2>, Tensor<2, Int>) {
        let probs = softmax(self.router.forward(x), 1);
        let (top, indices) = probs.clone().topk_with_indices(self.top_k, 1);
        let weights = if self.normalize_top_k {
            let s = top.clone().sum_dim(1);
            top / s
        } else {
            top
        };
        (probs, weights, indices)
    }

    /// Direct pass (sparse): `[..., d_model] → [..., d_model]`.
    ///
    /// # Panics
    /// If routing indexes could not be read to the host (see
    /// [`try_forward`](Self::try_forward)).
    pub fn forward<const D: usize>(&self, x: Tensor<D>) -> Tensor<D> {
        self.forward_with_aux(x).0
    }

    /// Like [`forward`](Self::forward), but with `Result`.
    pub fn try_forward<const D: usize>(&self, x: Tensor<D>) -> Result<Tensor<D>, NnError> {
        Ok(self.try_forward_with_aux(x)?.0)
    }

    /// Direct pass plus router statistics (taken from
    /// [`load_balance_loss`](MoeAuxStats::load_balance_loss)).
    ///
    /// # Panics
    /// See [`forward`](Self::forward).
    pub fn forward_with_aux<const D: usize>(&self, x: Tensor<D>) -> (Tensor<D>, MoeAuxStats) {
        self.try_forward_with_aux(x).unwrap_or_else(|e| fail(e))
    }

    /// Like [`forward_with_aux`](Self::forward_with_aux), but with `Result`.
    pub fn try_forward_with_aux<const D: usize>(
        &self,
        x: Tensor<D>,
    ) -> Result<(Tensor<D>, MoeAuxStats), NnError> {
        let dims = x.dims();
        let (tokens, d_model) = Self::flat_dims(&dims);
        let (y, aux) = self.try_forward_tokens(x.reshape([tokens, d_model]))?;
        Ok((y.reshape(dims), aux))
    }

    /// Flat version: `[N, D] → [N, D]` and router statistics.
    ///
    /// # Panics
    /// See [`forward`](Self::forward).
    pub fn forward_tokens(&self, x: Tensor<2>) -> (Tensor<2>, MoeAuxStats) {
        self.try_forward_tokens(x).unwrap_or_else(|e| fail(e))
    }

    /// Like [`forward_tokens`](Self::forward_tokens), but with `Result`.
    pub fn try_forward_tokens(&self, x: Tensor<2>) -> Result<(Tensor<2>, MoeAuxStats), NnError> {
        let [n, d] = x.dims();
        let device = x.device();
        let (k, n_exp) = (self.top_k, self.num_experts());

        if n == 0 {
            let aux = MoeAuxStats {
                router_probs: Tensor::zeros([0, n_exp], &device),
                expert_indices: Tensor::zeros([0, k], &device),
                expert_load: Tensor::zeros([n_exp], &device),
                dropped_assignments: 0,
            };
            return Ok((x, aux));
        }

        let (probs, weights, indices) = self.route(x.clone());
        // Routing indexes - per host: you need to group tokens by
        // experts (dynamic dimensions).
        let idx: Vec<i64> =
            data_to_vec(indices.clone().into_data(), "Moe::forward: routing indexes")?;

        // Overflow priority: first all first selections, then second...
        let cap = self.capacity(n);
        let mut counts = vec![0usize; n_exp];
        let mut load = vec![0usize; n_exp];
        let mut toks: Vec<Vec<i64>> = vec![Vec::new(); n_exp];
        let mut flats: Vec<Vec<i64>> = vec![Vec::new(); n_exp];
        let mut dropped = 0usize;
        for j in 0..k {
            for tok in 0..n {
                let flat = tok * k + j;
                let e = usize::try_from(idx[flat])
                    .ok()
                    .filter(|&e| e < n_exp)
                    .ok_or_else(|| {
                        NnError::state(
                            "Moe::forward",
                            format!(
                                "The router returned the expert index {} outside 0..{n_exp}",
                                idx[flat]
                            ),
                        )
                    })?;
                load[e] += 1;
                match self.overflow_policy {
                    OverflowPolicy::Drop if counts[e] >= cap => dropped += 1,
                    OverflowPolicy::Drop => {
                        counts[e] += 1;
                        toks[e].push(tok as i64);
                        flats[e].push(flat as i64);
                    }
                }
            }
        }

        let w_flat = weights.reshape([n * k]);
        let mut out = Tensor::<2>::zeros([n, d], &device);
        for (e, expert) in self.experts.iter().enumerate() {
            if toks[e].is_empty() {
                continue;
            }
            let m = toks[e].len();
            let tok_idx = int_tensor(std::mem::take(&mut toks[e]), &device);
            let w = w_flat
                .clone()
                .select(0, int_tensor(std::mem::take(&mut flats[e]), &device))
                .reshape([m, 1]);
            let ye = expert.forward(x.clone().select(0, tok_idx.clone())).mul(w);
            out = out.select_assign(0, tok_idx, ye, IndexingUpdateOp::Add);
        }
        if let Some(shared) = &self.shared {
            out = out + shared.forward(x);
        }

        let total = (n * k).max(1) as f32;
        let load = Tensor::<1>::from_data(
            TensorData::new(
                load.iter().map(|&c| c as f32 / total).collect::<Vec<_>>(),
                [n_exp],
            ),
            &device,
        );
        let aux = MoeAuxStats {
            router_probs: probs,
            expert_indices: indices,
            expert_load: load,
            dropped_assignments: dropped,
        };
        Ok((out, aux))
    }

    /// Reference direct pass: each expert is evaluated on all tokens and
    /// weighted by a selection mask, without a container. Same as
    /// [`Moe::forward`] at `capacity_factor = f32::MAX`, but costs
    /// `num_experts` full passes - for testing and debugging only.
    pub fn forward_dense<const D: usize>(&self, x: Tensor<D>) -> Tensor<D> {
        let dims = x.dims();
        let (tokens, d_model) = Self::flat_dims(&dims);
        if tokens == 0 {
            return x;
        }

        let x2 = x.reshape([tokens, d_model]);
        let (_, weights, indices) = self.route(x2.clone());

        let mut y = Tensor::<2>::zeros([tokens, d_model], &x2.device());
        for (e, expert) in self.experts.iter().enumerate() {
            // Expert weight e for each token: Σ_j w_j · [i_j == e].
            let mask = indices.clone().equal_elem(e as i64).float();
            let w_e = mask.mul(weights.clone()).sum_dim(1); // [tokens, 1]
            y = y.add(expert.forward(x2.clone()).mul(w_e));
        }
        if let Some(s) = &self.shared {
            y = y + s.forward(x2);
        }
        y.reshape(dims)
    }

    /// Balancing loss according to router statistics (same as
    /// [`MoeAuxStats::load_balance_loss`]).
    pub fn load_balance_loss(&self, stats: &MoeAuxStats) -> Tensor<1> {
        stats.load_balance_loss()
    }

    /// `[..., d_model]` → (number of tokens, `d_model`).
    fn flat_dims<const D: usize>(dims: &[usize; D]) -> (usize, usize) {
        let d_model = dims[D - 1];
        let tokens = dims[..D - 1].iter().product::<usize>();
        (tokens, d_model)
    }
}

/// One-dimensional integer tensor from a vector of indices.
fn int_tensor(values: Vec<i64>, device: &Device) -> Tensor<1, Int> {
    let n = values.len();
    Tensor::from_data(TensorData::new(values, [n]), device)
}
