//! MoE Transformer block: a thin wrapper over [`LlmBlock`], where FFN is replaced with
//! [`crate::ffn::Moe`].
//!
//! All MoE logic (router, top-k, capacity, shared expert, balancing
//! loss) lives in [`crate::ffn`] - it is not here. The MoE configuration is
//! [`MoeConfig`](crate::ffn::MoeConfig), router statistics -
//! [`MoeAuxStats`](crate::ffn::MoeAuxStats).

use burn::prelude::*;

use super::llm_block::{FfnKind, KvCache, LlmBlock, LlmBlockConfig, LlmFfn};
use crate::error::{NnError, fail};
use crate::ffn::{Moe, MoeAuxStats, MoeConfig};

/// Configuration [`MoeBlock`]: common block + MoE (substituted as FFN).
#[derive(Config, Debug)]
pub struct MoeBlockConfig {
    pub base: LlmBlockConfig,
    pub moe: MoeConfig,
}

impl MoeBlockConfig {
    pub fn to_block_config(&self) -> LlmBlockConfig {
        let mut c = self.base.clone();
        c.ffn_kind = FfnKind::Moe;
        c.moe = Some(self.moe.clone());
        c
    }

    pub fn validate(&self) -> Result<(), String> {
        self.to_block_config().validate()
    }

    /// # Panics
    /// If [`validate`](Self::validate) returned an error (the way without panic is
    /// [`try_init`](Self::try_init)).
    pub fn init(&self, device: &Device) -> MoeBlock {
        self.try_init(device).unwrap_or_else(|e| fail(e))
    }

    /// Like [`init`](Self::init), but with `Result` instead of panic.
    pub fn try_init(&self, device: &Device) -> Result<MoeBlock, NnError> {
        Ok(MoeBlock {
            block: self.to_block_config().try_init(device)?,
        })
    }
}

/// Transformer block, where FFN is replaced by MoE. All logic is in [`LlmBlock`].
#[derive(Module, Debug)]
pub struct MoeBlock {
    pub block: LlmBlock,
}

impl MoeBlock {
    /// MoE-block layer; `Err`, if the internal FFN is not MoE (the block is assembled in
    /// bypass [`MoeBlockConfig`]).
    pub fn try_moe(&self) -> Result<&Moe, NnError> {
        match &self.block.ffn {
            LlmFfn::Moe(m) => Ok(m),
            LlmFfn::Gated(_) => Err(NnError::state(
                "MoeBlock",
                "internal FFN not MoE (the block is not assembled via MoeBlockConfig)",
            )),
        }
    }

    /// # Panics
    /// See [`try_moe`](Self::try_moe).
    pub fn moe(&self) -> &Moe {
        self.try_moe().unwrap_or_else(|e| fail(e))
    }

    pub fn forward(&self, x: Tensor<3>) -> Tensor<3> {
        self.block.forward(x)
    }

    pub fn forward_with_mask(&self, x: Tensor<3>, mask: Option<Tensor<4>>) -> Tensor<3> {
        self.block.forward_with_mask(x, mask)
    }

    /// Block output and router statistics (aux-loss does not mix with the output).
    ///
    /// # Panics
    /// See [`try_forward_with_aux`](Self::try_forward_with_aux).
    pub fn forward_with_aux(
        &self,
        x: Tensor<3>,
        mask: Option<Tensor<4>>,
    ) -> (Tensor<3>, MoeAuxStats) {
        self.try_forward_with_aux(x, mask)
            .unwrap_or_else(|e| fail(e))
    }

    /// Like [`forward_with_aux`](Self::forward_with_aux), but with `Result`.
    pub fn try_forward_with_aux(
        &self,
        x: Tensor<3>,
        mask: Option<Tensor<4>>,
    ) -> Result<(Tensor<3>, MoeAuxStats), NnError> {
        let (y, aux) = self.block.forward_with_aux(x, mask);
        let aux = aux.ok_or_else(|| {
            NnError::state("MoeBlock", "the block did not return router statistics")
        })?;
        Ok((y, aux))
    }

    pub fn forward_cached(&self, x: Tensor<3>, cache: &mut KvCache, position: usize) -> Tensor<3> {
        self.block.forward_cached(x, cache, position)
    }

    pub fn load_balance_loss(&self, stats: &MoeAuxStats) -> Tensor<1> {
        stats.load_balance_loss()
    }
}
