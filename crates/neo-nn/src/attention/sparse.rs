//! Block-sparse attention: GQA + pattern over blocks of positions.
//!
//! The pattern is a list of allowed key blocks for each query block
//! ([`SparsePattern::allowed_blocks`]); they are the input of the future kernel,
//! which will compute only the selected blocks. Here - reference through a mask.
//! In causal mode, `j ≤ i` acts on top of blocks.

use burn::prelude::*;

use super::{AttentionPattern, GqaAttention, GqaAttentionConfig};
use crate::error::{NnError, fail};
use crate::transformer::KvCache;

/// Structure of allowed blocks.
#[derive(Config, Debug, PartialEq)]
pub enum SparsePattern {
    /// Query block `q` sees the latest `blocks_per_query` key blocks,
    /// including yours: `q-n+1 ..= q`.
    BlockLocal {
        block_size: usize,
        blocks_per_query: usize,
    },
    /// Explicit pairs `(query_block, key_block)`.
    Explicit {
        block_size: usize,
        pairs: Vec<(usize, usize)>,
    },
}

impl SparsePattern {
    pub fn block_size(&self) -> usize {
        match self {
            Self::BlockLocal { block_size, .. } | Self::Explicit { block_size, .. } => *block_size,
        }
    }

    pub fn validate(&self) -> Result<(), String> {
        if self.block_size() == 0 {
            return Err("block_size must be > 0".into());
        }
        match self {
            Self::BlockLocal {
                blocks_per_query, ..
            } if *blocks_per_query == 0 => Err("blocks_per_query must be > 0".into()),
            Self::Explicit { pairs, .. } if pairs.is_empty() => {
                Err("Explicit: block list is empty".into())
            }
            _ => Ok(()),
        }
    }

    /// For each of the `nq` query blocks - sorted indexes of the `< nk` key blocks.
    pub fn allowed_blocks(&self, nq: usize, nk: usize) -> Vec<Vec<usize>> {
        let mut out = vec![Vec::new(); nq];
        match self {
            Self::BlockLocal {
                blocks_per_query, ..
            } => {
                for (q, row) in out.iter_mut().enumerate() {
                    let lo = (q + 1).saturating_sub(*blocks_per_query);
                    row.extend((lo..=q).filter(|k| *k < nk));
                }
            }
            Self::Explicit { pairs, .. } => {
                for &(q, k) in pairs {
                    if q < nq && k < nk {
                        out[q].push(k);
                    }
                }
                for row in &mut out {
                    row.sort_unstable();
                    row.dedup();
                }
            }
        }
        out
    }

    /// Boolean table `[nq][nk]`.
    pub fn block_table(&self, nq: usize, nk: usize) -> Vec<Vec<bool>> {
        let mut t = vec![vec![false; nk]; nq];
        for (q, ks) in self.allowed_blocks(nq, nk).into_iter().enumerate() {
            for k in ks {
                t[q][k] = true;
            }
        }
        t
    }
}

/// Configuration [`SparseAttention`].
#[derive(Config, Debug)]
pub struct SparseAttentionConfig {
    pub pattern: SparsePattern,
    pub inner: GqaAttentionConfig,
}

impl SparseAttentionConfig {
    pub fn validate(&self) -> Result<(), String> {
        self.pattern.validate()?;
        self.inner.validate()
    }

    /// Create a layer; `Err` if [`validate`](Self::validate) failed.
    pub fn try_init(&self, device: &Device) -> Result<SparseAttention, NnError> {
        self.validate()
            .map_err(|e| NnError::config("SparseAttentionConfig", e))?;
        Ok(SparseAttention {
            inner: self.inner.try_init(device)?,
            pattern: self.pattern.clone(),
        })
    }

    /// Create a layer.
    ///
    /// # Panics
    /// If [`validate`](Self::validate) returned an error (the way without panic is [`try_init`](Self::try_init)).
    pub fn init(&self, device: &Device) -> SparseAttention {
        self.try_init(device).unwrap_or_else(|e| fail(e))
    }
}

#[derive(Module, Debug)]
pub struct SparseAttention {
    pub inner: GqaAttention,
    #[module(skip)]
    pub pattern: SparsePattern,
}

impl SparseAttention {
    pub fn pattern(&self) -> AttentionPattern {
        AttentionPattern::Sparse(self.pattern.clone())
    }

    pub fn forward(&self, x: Tensor<3>) -> Tensor<3> {
        self.inner.forward_with_pattern(x, &self.pattern())
    }

    pub fn forward_cached(&self, x: Tensor<3>, cache: &mut KvCache, position: usize) -> Tensor<3> {
        self.inner
            .forward_cached_with_pattern(x, cache, position, &self.pattern())
    }

    pub fn new_cache(&self) -> KvCache {
        self.inner.new_cache()
    }
}
