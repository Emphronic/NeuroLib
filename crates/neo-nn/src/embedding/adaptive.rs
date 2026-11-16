//! Adaptive embedding: tokens of different dictionary ranges receive embedding
//! different sizes.
//!
//! ```text
//!  token id ─┬─ [0, 50k)       ──► Embedding dim 4096 ─────────────┐
//!            ├─ [50k, 150k)    ──► Embedding dim 2048 ─► Linear ───┼─► [.., 4096]
//!            └─ [150k, 200k)   ──► Embedding dim 1024 ─► Linear ───┘
//! ```
//!
//! Idea (Baevski & Auli, “Adaptive Input Representations”, 2019): Dictionary
//! sorted by frequency, frequent tokens receive wide vectors, rare tokens -
//! narrow. Tier, which has `dim == output_dim`, has no projection; the rest
//! are reduced to `output_dim` by a linear layer.
//!
//! The mapping “token → tier” is static (by id ranges). Frequency
//! the policy needs to be applied in advance, when building the dictionary/configuration
//! (see [`AdaptiveEmbeddingConfig::from_cutoffs`]); in `forward` frequencies are not
//! taken into account.
//!
//! # Two forward modes
//!
//! The size of the set of tokens of each tier depends on the data, and the sizes of the tensors
//! must be known to the calling code. Therefore, “select tier tokens on
//! device" without reading data to the host is impossible - a compromise has been chosen:
//!
//! | method | host sync | calculations |
//! |-------|-----------|------------|
//! | [`AdaptiveEmbedding::forward`] (main) | **no** | all `N = B·T` tokens pass through each tier, the contribution of other people's tokens is nulled by a mask |
//! | [`AdaptiveEmbedding::forward_routed`] | yes: id is read once | each tier processes only its tokens, the result is collected `select_assign` |
//!
//! `forward` works with fixed shapes and is suitable for GPU without
//! synchronizations. Its price - projections are calculated for all `N` tokens for each
//! tier (for 3 tiers from the example above this is about `N·(2048+1024)·4096`
//! multiplications and additions for the entire batch are negligible next to the transformer, but
//! not free), plus one intermediate `[N, output_dim]` per tier.
//! `forward_routed` saves calculations at the cost of synchronization - according to the same scheme
//! the same as routing in `Moe`.
//!
//! Both methods give the same result for valid ids.
//!
//! # Boundary ids
//!
//! Id must be in `[0, vocab_size)`. `forward` for out of range returns
//! zero vector (you can’t check this on a device without synchronization);
//! `forward_routed` panics.
//!
//! Candidate for fused core in `neo-synk`: routing + lookup + projection + scatter.

use burn::nn::{Embedding, EmbeddingConfig, Linear, LinearConfig};
use burn::prelude::*;
use burn::tensor::{IndexingUpdateOp, Int, TensorData};

use crate::error::{NnError, data_to_vec, fail};

/// The range of the dictionary is `[start, end)` and the embedding dimension of this tier.
#[derive(Config, Debug)]
pub struct AdaptiveTierConfig {
    /// The first global id of the tier (inclusive).
    pub start: usize,
    /// Last global id of the tier (exclusively).
    pub end: usize,
    /// Embedding dimension in this tier.
    pub dim: usize,
}

/// Configuration [`AdaptiveEmbedding`].
///
/// Tiers must go in a row and cover the entire dictionary: the first begins with `0`,
/// each next one starts from the end of the previous one, the last one ends with
/// `vocab_size`.
#[derive(Config, Debug)]
pub struct AdaptiveEmbeddingConfig {
    /// Dictionary size.
    pub vocab_size: usize,
    /// The total output dimension is `D`.
    pub output_dim: usize,
    /// Tiers in ascending order of id.
    pub tiers: Vec<AdaptiveTierConfig>,
    /// Offset in tier projections (by default `false`, as in the original).
    #[config(default = false)]
    pub bias: bool,
}

impl AdaptiveEmbeddingConfig {
    /// Construct tiers from boundaries, as in adaptive input / adaptive softmax:
    /// `cutoffs = [c0, c1, ..]` divide the dictionary into `[0,c0)`, `[c0,c1)`, …,
    /// `[c_last, vocab_size)`, and the dimension of tier `i` is equal to
    /// `output_dim / factor^i` (not less than 1).
    ///
    /// ```ignore
    /// // 4096 / 2048 / 1024 for a dictionary of 200,000 tokens:
    /// AdaptiveEmbeddingConfig::from_cutoffs(200_000, 4096, &[50_000, 150_000], 2);
    /// ```
    ///
    /// Validity of boundaries (increasing, `< vocab_size`) checks
    /// [`validate`](Self::validate) on initialization.
    pub fn from_cutoffs(
        vocab_size: usize,
        output_dim: usize,
        cutoffs: &[usize],
        factor: usize,
    ) -> Self {
        let factor = factor.max(1);
        let mut bounds = vec![0];
        bounds.extend_from_slice(cutoffs);
        bounds.push(vocab_size);

        let tiers = bounds
            .windows(2)
            .enumerate()
            .map(|(i, w)| {
                let dim = u32::try_from(i)
                    .ok()
                    .and_then(|i| factor.checked_pow(i))
                    .map_or(1, |p| (output_dim / p).max(1));
                AdaptiveTierConfig::new(w[0], w[1], dim)
            })
            .collect();
        Self::new(vocab_size, output_dim, tiers)
    }

    /// Check configuration: non-empty tiers, `start < end`, `dim > 0`,
    /// continuous coverage `[0, vocab_size)` without gaps or overlaps.
    pub fn validate(&self) -> Result<(), String> {
        if self.vocab_size == 0 {
            return Err("vocab_size must be > 0".into());
        }
        if self.output_dim == 0 {
            return Err("output_dim must be > 0".into());
        }
        let first = self.tiers.first().ok_or("tiers must not be empty")?;
        if first.start != 0 {
            return Err(format!(
                "the first tier should start from 0, but starts from {}",
                first.start
            ));
        }
        let mut expected_start = 0;
        for (i, t) in self.tiers.iter().enumerate() {
            if t.dim == 0 {
                return Err(format!("tier {i}: dim must be > 0"));
            }
            if t.start >= t.end {
                return Err(format!(
                    "tier {i}: start ({}) must be < end ({})",
                    t.start, t.end
                ));
            }
            if t.start != expected_start {
                return Err(format!(
                    "tier {i}: start = {}, expected {expected_start} (gap or overlap)",
                    t.start
                ));
            }
            expected_start = t.end;
        }
        if expected_start != self.vocab_size {
            return Err(format!(
                "the last tier ends with {expected_start} and vocab_size = {}",
                self.vocab_size
            ));
        }
        Ok(())
    }

    /// Tier index for global id (`None`, if id is outside the dictionary).
    pub fn tier_of(&self, id: usize) -> Option<usize> {
        self.tiers.iter().position(|t| id >= t.start && id < t.end)
    }

    /// `(tier index, local id)` for global id:
    /// `local = id - tier.start`.
    pub fn locate(&self, id: usize) -> Option<(usize, usize)> {
        self.tier_of(id).map(|k| (k, id - self.tiers[k].start))
    }

    /// Number of trainable parameters: tier tables and their projections.
    pub fn parameter_count(&self) -> usize {
        self.tiers
            .iter()
            .map(|t| {
                let table = (t.end - t.start) * t.dim;
                let proj = if t.dim != self.output_dim {
                    t.dim * self.output_dim + if self.bias { self.output_dim } else { 0 }
                } else {
                    0
                };
                table + proj
            })
            .sum()
    }

    /// The number of parameters of a regular embedding `[vocab, output_dim]` is for comparison.
    pub fn dense_parameter_count(&self) -> usize {
        self.vocab_size * self.output_dim
    }

    /// Initialize the layer.
    ///
    /// # Panics
    /// If the configuration is invalid (see [`validate`](Self::validate)).
    pub fn init(&self, device: &Device) -> AdaptiveEmbedding {
        self.try_init(device)
            .unwrap_or_else(|e| panic!("AdaptiveEmbeddingConfig: {e}"))
    }

    /// Like [`init`](Self::init), but with `Result` instead of panic.
    pub fn try_init(&self, device: &Device) -> Result<AdaptiveEmbedding, String> {
        self.validate()?;
        let tiers = self
            .tiers
            .iter()
            .map(|t| AdaptiveTier {
                start: t.start,
                end: t.end,
                dim: t.dim,
                embedding: EmbeddingConfig::new(t.end - t.start, t.dim).init(device),
                projection: (t.dim != self.output_dim).then(|| {
                    LinearConfig::new(t.dim, self.output_dim)
                        .with_bias(self.bias)
                        .init(device)
                }),
            })
            .collect();
        Ok(AdaptiveEmbedding {
            tiers,
            vocab_size: self.vocab_size,
            output_dim: self.output_dim,
        })
    }
}

/// One tier: table for the range `[start, end)` and (if necessary) projection
/// `dim → output_dim`.
#[derive(Module, Debug)]
pub struct AdaptiveTier {
    /// First global id (inclusive).
    #[module(skip)]
    pub start: usize,
    /// Last global id (exclusively).
    #[module(skip)]
    pub end: usize,
    /// Dimension of the table.
    #[module(skip)]
    pub dim: usize,
    /// Table `[end - start, dim]`, indexed by **local** id.
    pub embedding: Embedding,
    /// Projection `dim → output_dim`; `None` if `dim == output_dim`.
    pub projection: Option<Linear>,
}

impl AdaptiveTier {
    /// Lookup by local id and projection: `[M]` → `[M, output_dim]`.
    fn embed_local(&self, local: Tensor<1, Int>) -> Tensor<2> {
        let m = local.dims()[0];
        let e = self
            .embedding
            .forward(local.reshape([1, m]))
            .reshape([m, self.dim]);
        match &self.projection {
            Some(p) => p.forward(e),
            None => e,
        }
    }
}

/// Embedding with different dimensions for different vocabulary ranges.
#[derive(Module, Debug)]
pub struct AdaptiveEmbedding {
    /// Tiers in ascending order of id.
    pub tiers: Vec<AdaptiveTier>,
    /// Dictionary size.
    #[module(skip)]
    pub vocab_size: usize,
    /// Overall output dimension.
    #[module(skip)]
    pub output_dim: usize,
}

impl AdaptiveEmbedding {
    /// Tier index for global id (`None`, if id is outside the dictionary).
    pub fn tier_index(&self, id: usize) -> Option<usize> {
        let k = self.tiers.partition_point(|t| t.end <= id);
        (k < self.tiers.len() && id >= self.tiers[k].start).then_some(k)
    }

    /// Basic pass without synchronizing with the host: `[B,T]` (Int) → `[B,T,D]`.
    ///
    /// For each tier:
    /// 1. mask `start <= id < end` (float `[N,1]`);
    /// 2. local id `clamp(id - start, 0, size - 1)` - other people’s tokens
    ///    safely read any line, their contribution is reset to zero by the mask;
    /// 3. lookup → projection → multiplication by mask → sum by tier.
    ///
    /// The token belongs to exactly one tier, so the amount is equal to the contribution
    /// "your" tier. The gradient of other people's tokens is reset to zero by the same mask.
    /// Id outside `[0, vocab_size)` give a zero vector.
    pub fn forward(&self, indices: Tensor<2, Int>) -> Tensor<3> {
        let [batch, seq] = indices.dims();
        let n = batch * seq;
        let device = indices.device();
        if n == 0 {
            return Tensor::zeros([batch, seq, self.output_dim], &device);
        }

        let flat = indices.reshape([n]);
        let mut out = Tensor::<2>::zeros([n, self.output_dim], &device);

        for tier in &self.tiers {
            let (start, end) = (tier.start as i64, tier.end as i64);

            let ge = flat.clone().greater_equal_elem(start).float();
            let lt = flat.clone().lower_elem(end).float();
            let mask = ge.mul(lt).reshape([n, 1]);

            let local = flat.clone().sub_scalar(start).clamp(0, end - start - 1);
            let rows = tier.embed_local(local); // [N, D]

            out = out + rows.mul(mask);
        }

        out.reshape([batch, seq, self.output_dim])
    }

    /// Option with real routing: each tier processes only its own
    /// tokens. **Reads host ids once** (like `Moe`), but does not compute
    /// projections for other tokens. The result is collected via `select_assign`
    /// in the original order.
    ///
    /// # Panics
    /// If some id is not in `[0, vocab_size)` or indexes cannot be
    /// read (path without panic - [`try_forward_routed`](Self::try_forward_routed)).
    pub fn forward_routed(&self, indices: Tensor<2, Int>) -> Tensor<3> {
        self.try_forward_routed(indices).unwrap_or_else(|e| fail(e))
    }

    /// Like [`forward_routed`](Self::forward_routed), but errors (id is out of dictionary,
    /// failed to read indexes from device) are returned as [`NnError`].
    pub fn try_forward_routed(&self, indices: Tensor<2, Int>) -> Result<Tensor<3>, NnError> {
        let [batch, seq] = indices.dims();
        let n = batch * seq;
        let device = indices.device();
        if n == 0 {
            return Ok(Tensor::zeros([batch, seq, self.output_dim], &device));
        }

        let ids: Vec<i64> = data_to_vec(
            indices.reshape([n]).into_data(),
            "AdaptiveEmbedding::forward_routed",
        )?;

        let mut positions: Vec<Vec<i64>> = vec![Vec::new(); self.tiers.len()];
        let mut locals: Vec<Vec<i64>> = vec![Vec::new(); self.tiers.len()];
        for (pos, &id) in ids.iter().enumerate() {
            let k = usize::try_from(id)
                .ok()
                .and_then(|id| self.tier_index(id))
                .ok_or_else(|| {
                    NnError::input(
                        "AdaptiveEmbedding::forward_routed",
                        format!(
                            "AdaptiveEmbedding: id {id} outside dictionary [0, {})",
                            self.vocab_size
                        ),
                    )
                })?;
            positions[k].push(pos as i64);
            locals[k].push(id - self.tiers[k].start as i64);
        }

        let mut out = Tensor::<2>::zeros([n, self.output_dim], &device);
        for (k, tier) in self.tiers.iter().enumerate() {
            if positions[k].is_empty() {
                continue;
            }
            let pos = int_tensor(std::mem::take(&mut positions[k]), &device);
            let local = int_tensor(std::mem::take(&mut locals[k]), &device);
            let rows = tier.embed_local(local);
            out = out.select_assign(0, pos, rows, IndexingUpdateOp::Add);
        }

        Ok(out.reshape([batch, seq, self.output_dim]))
    }
}

fn int_tensor(values: Vec<i64>, device: &Device) -> Tensor<1, Int> {
    let n = values.len();
    Tensor::from_data(TensorData::new(values, [n]), device)
}
