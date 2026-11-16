//! Sliding-window attention: GQA + strip mask, no projections.
//!
//! Causal: the request `i` sees `j` with `i-W < j ≤ i`. Non-causal: `|i-j| < W`.
//! The mask is a reference implementation (`O(T²)` from memory).

use burn::prelude::*;

use super::{AttentionPattern, GqaAttention, GqaAttentionConfig};
use crate::error::{NnError, fail};
use crate::transformer::KvCache;

/// Configuration [`SlidingWindowAttention`].
#[derive(Config, Debug)]
pub struct SlidingWindowConfig {
    pub window_size: usize,
    pub inner: GqaAttentionConfig,
}

impl SlidingWindowConfig {
    pub fn validate(&self) -> Result<(), String> {
        if self.window_size == 0 {
            return Err("window_size must be > 0".into());
        }
        self.inner.validate()
    }

    /// Create a layer; `Err` if [`validate`](Self::validate) failed.
    pub fn try_init(&self, device: &Device) -> Result<SlidingWindowAttention, NnError> {
        self.validate()
            .map_err(|e| NnError::config("SlidingWindowConfig", e))?;
        Ok(SlidingWindowAttention {
            inner: self.inner.try_init(device)?,
            window_size: self.window_size,
        })
    }

    /// Create a layer.
    ///
    /// # Panics
    /// If [`validate`](Self::validate) returned an error (the way without panic is [`try_init`](Self::try_init)).
    pub fn init(&self, device: &Device) -> SlidingWindowAttention {
        self.try_init(device).unwrap_or_else(|e| fail(e))
    }
}

#[derive(Module, Debug)]
pub struct SlidingWindowAttention {
    pub inner: GqaAttention,
    #[module(skip)]
    pub window_size: usize,
}

impl SlidingWindowAttention {
    pub fn pattern(&self) -> AttentionPattern {
        AttentionPattern::SlidingWindow {
            window_size: self.window_size,
        }
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
