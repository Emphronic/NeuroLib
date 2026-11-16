//! Factorized embedding: one large matrix `[vocab, d_model]` is replaced
//! two small ones.
//!
//! ```text
//!  token id ──► E1 [vocab, rank] ──► E2 [rank, d_model] ──► vector[d_model]
//! ```
//!
//! ```text
//! Z[b,t,:] = E1[token[b,t], :]                 // [B,T,R]
//! Y[b,t,:] = Z[b,t,:] · W_proj (+ bias)        // [B,T,D]
//! ```
//!
//! Parameters: `vocab·rank + rank·d_model` instead of `vocab·d_model`.
//! For `vocab = 200 000`, `d_model = 4096` regular embedding - 819,200,000
//! parameters; with `rank = 256` factorized - 52,248,576 (~15.7 times less).
//!
//! This is a low-rank approximation: the resulting matrix `E1·E2` has rank no higher than
//! `rank`. The choice of `rank` is a “memory / expressiveness” compromise, with
//! `rank >= d_model` no savings.
//!
//! # Projection counts as one GEMM
//!
//! `[B,T,R]` before `Linear` is flattened into `[B·T, R]`, and after it returns to
//! `[B,T,D]`: backend performs single matrix multiplication rather than batched matmul with
//! broadcast of the scales by `B`.
//!
//! # Link with LM head
//!
//! [`FactorizedEmbedding::project_to_vocab`] gives logits using the same weights
//! (`hidden → rank → vocab`). Weight tying is **not** automatically enabled:
//! the method is called explicitly, and a separate `FactorizedLmHead` can be done later.
//!
//! # Example
//!
//! ```ignore
//! use burn::tensor::Device;
//! use neo_nn::embedding::FactorizedEmbeddingConfig;
//!
//! let device = Device::flex(); // CPU backend (feature "flex")
//! let emb = FactorizedEmbeddingConfig::new(200_000, 256, 4096).init(&device);
//! // ids: Tensor<2, Int> [batch, seq]  →  Tensor<3> [batch, seq, 4096]
//! ```
//!
//! Candidate for fused core in `neo-synk`: lookup + projection.

use burn::nn::{Embedding, EmbeddingConfig, Linear, LinearConfig};
use burn::prelude::*;
use burn::tensor::Int;

/// Configuration [`FactorizedEmbedding`].
#[derive(Config, Debug)]
pub struct FactorizedEmbeddingConfig {
    /// Dictionary size `V`.
    pub vocab_size: usize,
    /// Decomposition rank `R` (bottleneck width).
    pub rank: usize,
    /// Output dimension `D`.
    pub d_model: usize,
    /// Projection offset `R → D`.
    ///
    /// By default `false`: for embedding the offset is the same vector,
    /// added to each token, and ALBERT-like schemes do not use it.
    #[config(default = false)]
    pub bias: bool,
}

impl FactorizedEmbeddingConfig {
    /// Checking parameters: `vocab_size > 0`, `rank > 0`, `d_model > 0`.
    pub fn validate(&self) -> Result<(), String> {
        if self.vocab_size == 0 {
            return Err("vocab_size must be > 0".into());
        }
        if self.rank == 0 {
            return Err("rank must be > 0".into());
        }
        if self.d_model == 0 {
            return Err("d_model must be > 0".into());
        }
        Ok(())
    }

    /// Number of parameters: `vocab·rank + rank·d_model` (+ `d_model`, if `bias`).
    pub fn parameter_count(&self) -> usize {
        self.vocab_size * self.rank
            + self.rank * self.d_model
            + if self.bias { self.d_model } else { 0 }
    }

    /// The number of parameters of a regular embedding `[vocab, d_model]` is for comparison.
    pub fn dense_parameter_count(&self) -> usize {
        self.vocab_size * self.d_model
    }

    /// Initialize the layer.
    ///
    /// # Panics
    /// If the configuration is invalid (see [`validate`](Self::validate)).
    pub fn init(&self, device: &Device) -> FactorizedEmbedding {
        self.try_init(device)
            .unwrap_or_else(|e| panic!("FactorizedEmbeddingConfig: {e}"))
    }

    /// Like [`init`](Self::init), but with `Result` instead of panic.
    pub fn try_init(&self, device: &Device) -> Result<FactorizedEmbedding, String> {
        self.validate()?;
        Ok(FactorizedEmbedding {
            token_embedding: EmbeddingConfig::new(self.vocab_size, self.rank).init(device),
            projection: LinearConfig::new(self.rank, self.d_model)
                .with_bias(self.bias)
                .init(device),
            vocab_size: self.vocab_size,
            rank: self.rank,
            d_model: self.d_model,
        })
    }
}

/// Embedding with low-rank factorization `E1[vocab, rank] · E2[rank, d_model]`.
#[derive(Module, Debug)]
pub struct FactorizedEmbedding {
    /// `E1`: table `[vocab_size, rank]`.
    pub token_embedding: Embedding,
    /// `E2`: projection of `rank → d_model` (weight `[rank, d_model]`).
    pub projection: Linear,
    /// Dictionary size.
    #[module(skip)]
    pub vocab_size: usize,
    /// Decomposition rank.
    #[module(skip)]
    pub rank: usize,
    /// Output dimension.
    #[module(skip)]
    pub d_model: usize,
}

impl FactorizedEmbedding {
    /// Direct pass: `[B,T]` (Int) → `[B,T,D]`.
    pub fn forward(&self, indices: Tensor<2, Int>) -> Tensor<3> {
        let [batch, seq] = indices.dims();
        let z = self.lookup(indices);

        // One GEMM by [B·T, R] instead of batched matmul.
        let y = self.projection.forward(z.reshape([batch * seq, self.rank]));
        y.reshape([batch, seq, self.d_model])
    }

    /// Only lookup in `E1`, without projection: `[B,T]` → `[B,T,R]`.
    pub fn lookup(&self, indices: Tensor<2, Int>) -> Tensor<3> {
        self.token_embedding.forward(indices)
    }

    /// Collapse the hidden state to rank: `[B,T,D]` → `[B,T,R]`, that is
    /// `hidden · W_projᵀ`. Projection offset is not used.
    ///
    /// # Panics
    /// If the last axis `hidden` is not equal to `d_model`.
    pub fn project_to_rank(&self, hidden: Tensor<3>) -> Tensor<3> {
        let [batch, seq, d] = hidden.dims();
        assert_eq!(
            d, self.d_model,
            "FactorizedEmbedding: expected d_model = {}, received {d}",
            self.d_model
        );

        let w = self.projection.weight.val(); // [R, D]
        hidden
            .reshape([batch * seq, d])
            .matmul(w.transpose())
            .reshape([batch, seq, self.rank])
    }

    /// Logits according to the dictionary using the **same** weights (tied LM head):
    /// `[B,T,D]` → `[B,T,rank]` → `[B,T,vocab]`, that is
    /// `(hidden · W_projᵀ) · E1ᵀ`.
    ///
    /// Called explicitly; weight tying does not turn on by itself. Offset
    /// projection is not used.
    pub fn project_to_vocab(&self, hidden: Tensor<3>) -> Tensor<3> {
        let [batch, seq, _] = hidden.dims();
        let r = self
            .project_to_rank(hidden)
            .reshape([batch * seq, self.rank]);

        let e = self.token_embedding.weight.val(); // [V, R]
        r.matmul(e.transpose())
            .reshape([batch, seq, self.vocab_size])
    }
}
