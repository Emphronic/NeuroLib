//! `SpikingTransformerBlock`: one block of spike transformer -
//! attention (SSA or SDSA, interchangeably) + spiking MLP, each with
//! with its residual connection.
//!
//! # Residual - along the membrane, not along the spikes
//!
//! Both Spikformer and Spike-Driven Transformer stack the subblock input with
//! its output BEFORE the final spike nonlinearity - that is, residual
//! goes according to the membrane potential, and not according to already discretized ones (0/1)
//! spikes. If residual already took the spike output, the addition of two
//! binary signals would give the values `{0, 1, 2}` - not a spike - and that’s all
//! would equally require either repeated spike nonlinearity (which is
//! membrane residual, only with an extra step), or interpretation
//! “graded spike” at the input of the next subblock, which is what the original
//! they don't make architecture. Therefore, at [`SpikingSelfAttention`],
//! [`SpikeDrivenSelfAttention`] and [`SpikingMlp`] are a pair method
//! `forward_pre_spike`, returning the domain BEFORE the final SN(·), and precisely
//! the input of the block is added to it, and SN(·) is applied to the sum:
//!
//! ```text
//! attn_mem = Attention.forward_pre_spike(x)
//! x1       = SN(x + attn_mem)
//! mlp_mem  = Mlp.forward_pre_spike(x1)
//! x2       = SN(x1 + mlp_mem)
//! ```
//!
//! Please note: `x` (already the spike output of the previous block or
//! patch embedding) is added to the `attn_mem` MEMBRANE, and not vice versa -
//! it is asymmetrical with respect to types (spike + membrane = membrane),
//! but this is exactly how residual is designed in both reference architectures: it
//! continues to run in the real domain, only occasionally being discretized.

use burn::module::Module;
use burn::tensor::Device;

use super::{
    SpikeDrivenSelfAttention, SpikeDrivenSelfAttentionConfig, SpikeSequence, SpikingMlp,
    SpikingMlpConfig, SpikingSelfAttention, SpikingSelfAttentionConfig, apply_spiking_nonlinearity,
    spiking_nonlinearity,
};
use crate::neuron::Leaky;
use crate::surrogate::Surrogate;

/// Which variant of spike-based attention does the block use?
///
/// SSA (Spikformer) is more expressive (full-rank attention matrix `[N,N]`
/// per head), but requires two matrix multiplications per head per challenge.
/// SDSA (Spike-Driven Transformer) is cheaper on neuromorphic hardware (only
/// addition and Hadamard-products, without mathmuls), but an attention mask
/// single-channel for a pair of tokens - structurally less expressive.
#[derive(Module, Debug)]
pub enum SpikingAttentionVariant {
    /// Spiking Self-Attention (Spikformer).
    Ssa(SpikingSelfAttention),
    /// Spike-Driven Self-Attention (Spike-Driven Transformer).
    Sdsa(SpikeDrivenSelfAttention),
}

impl SpikingAttentionVariant {
    fn forward_pre_spike<S: Surrogate + ?Sized>(
        &self,
        x: SpikeSequence,
        surrogate: &S,
    ) -> SpikeSequence {
        match self {
            Self::Ssa(layer) => layer.forward_pre_spike(x, surrogate),
            Self::Sdsa(layer) => layer.forward_pre_spike(x, surrogate),
        }
    }
}

/// Selecting an attention option when configuring a block.
#[derive(Debug, Clone, Copy)]
pub enum AttentionKind {
    /// Spiking Self-Attention (Spikformer): `SN(Q·Kᵀ·V·scale)`.
    Ssa {
        /// Scale `s`, see [`SpikingSelfAttentionConfig::scale`].
        scale: f32,
    },
    /// Spike-Driven Self-Attention (Spike-Driven Transformer):
    /// `SN(Σ_c(Q⊗K)) ⊗ V`, without matrix multiplications.
    Sdsa,
}

/// Configuration [`SpikingTransformerBlock`].
#[derive(Debug, Clone, Copy)]
pub struct SpikingTransformerBlockConfig {
    /// The dimension of the feature is `D`.
    pub dim: usize,
    /// Number of attention heads.
    pub num_heads: usize,
    /// Dimension of the MLP hidden layer (default `4 * dim`).
    pub mlp_hidden_dim: usize,
    /// Attention option.
    pub attention: AttentionKind,
    /// Spike threshold common to all SN(·) in the block (including both residual-SNs).
    pub threshold: f32,
}

impl SpikingTransformerBlockConfig {
    /// New configuration with SSA attention (Spikformer) - a sensible choice
    /// by default if there are no specific requirements for neuromorphic
    /// hardware, where [`AttentionKind::Sdsa`] is preferable.
    pub fn new(dim: usize, num_heads: usize) -> Self {
        assert!(dim > 0, "dim must be > 0");
        assert!(num_heads > 0, "num_heads must be > 0");
        assert!(
            dim.is_multiple_of(num_heads),
            "dim({dim}) must be divisible by num_heads({num_heads})"
        );
        Self {
            dim,
            num_heads,
            mlp_hidden_dim: dim * 4,
            attention: AttentionKind::Ssa { scale: 0.125 },
            threshold: 1.0,
        }
    }

    /// Use SDSA (Spike-Driven Transformer) instead of SSA.
    pub fn with_sdsa(mut self) -> Self {
        self.attention = AttentionKind::Sdsa;
        self
    }

    /// Use SSA (Spikformer) with explicit scale `s`.
    pub fn with_ssa_scale(mut self, scale: f32) -> Self {
        self.attention = AttentionKind::Ssa { scale };
        self
    }

    /// Set the dimension of the MLP hidden layer.
    pub fn with_mlp_hidden_dim(mut self, hidden_dim: usize) -> Self {
        assert!(hidden_dim > 0, "hidden_dim must be > 0");
        self.mlp_hidden_dim = hidden_dim;
        self
    }

    /// Set the spike threshold.
    pub fn with_threshold(mut self, threshold: f32) -> Self {
        self.threshold = threshold;
        self
    }

    /// Create a block.
    pub fn init(&self, device: &Device) -> SpikingTransformerBlock {
        SpikingTransformerBlock::new(self, device)
    }
}

/// One spike transformer block: attention + MLP, each with
/// residual connection across the membrane potential. See module documentation.
#[derive(Module, Debug)]
pub struct SpikingTransformerBlock {
    /// Attention layer - SSA or SDSA, see [`SpikingAttentionVariant`].
    pub attention: SpikingAttentionVariant,
    /// Spiking MLP subblock.
    pub mlp: SpikingMlp,
    /// SN(·), discretizing the sum `x + attention_pre_spike`.
    #[module(skip)]
    pub sn_attn_residual: Leaky,
    /// SN(·), discretizing the sum `x1 + mlp_pre_spike`.
    #[module(skip)]
    pub sn_mlp_residual: Leaky,
}

impl SpikingTransformerBlock {
    /// Create a block from the configuration.
    pub fn new(config: &SpikingTransformerBlockConfig, device: &Device) -> Self {
        let attention = match config.attention {
            AttentionKind::Ssa { scale } => SpikingAttentionVariant::Ssa(
                SpikingSelfAttentionConfig::new(config.dim, config.num_heads)
                    .with_scale(scale)
                    .with_threshold(config.threshold)
                    .init(device),
            ),
            AttentionKind::Sdsa => SpikingAttentionVariant::Sdsa(
                SpikeDrivenSelfAttentionConfig::new(config.dim, config.num_heads)
                    .with_threshold(config.threshold)
                    .init(device),
            ),
        };

        let mlp = SpikingMlpConfig::new(config.dim)
            .with_hidden_dim(config.mlp_hidden_dim)
            .with_threshold(config.threshold)
            .init(device);

        Self {
            attention,
            mlp,
            sn_attn_residual: spiking_nonlinearity(config.threshold, device),
            sn_mlp_residual: spiking_nonlinearity(config.threshold, device),
        }
    }

    /// Direct passage. `x` — spike input `[T, B, N, D]` (patch output
    /// embedding for the first block of the stack, or the output of the previous block).
    /// Returns a spike output of the same shape.
    pub fn forward<S: Surrogate + ?Sized>(&self, x: SpikeSequence, surrogate: &S) -> SpikeSequence {
        let attn_pre_spike = self.attention.forward_pre_spike(x.clone(), surrogate);
        let x1 = apply_spiking_nonlinearity(&self.sn_attn_residual, x + attn_pre_spike, surrogate);

        let mlp_pre_spike = self.mlp.forward_pre_spike(x1.clone(), surrogate);
        apply_spiking_nonlinearity(&self.sn_mlp_residual, x1 + mlp_pre_spike, surrogate)
    }
}

#[cfg(test)]
mod residual_tests {
    use super::*;
    use crate::surrogate::ATan;
    use burn::tensor::Device;
    use burn::tensor::Tensor;

    fn dev() -> Device {
        Device::flex()
    }

    fn spike_input(t: usize, b: usize, n: usize, d: usize) -> Tensor<4> {
        let total = t * b * n * d;
        let data: Vec<f32> = (0..total).map(|i| (i as f32 * 0.37).sin().abs()).collect();
        Tensor::<1>::from_floats(data.as_slice(), &dev()).reshape([t, b, n, d])
    }

    fn vec4(x: Tensor<4>) -> Vec<f32> {
        x.into_data().try_to_vec::<f32>().unwrap()
    }

    /// `forward` must actually add `x` (and `x1`) with pre-spike output
    /// subblocks before the final SN - rather than just passing the input through
    /// attention+MLP by residual connection. Compare with MANUAL composition
    /// THE SAME block fields (`self.attention`, `self.mlp`,
    /// `self.sn_*_residual`), but without adding `x`/`x1` - if
    /// residual was accidentally discarded (for example, due to a typo of the shape
    /// `apply_spiking_nonlinearity(&sn, attn_pre_spike, surrogate)` without
    /// `x + ...`), this version would match `forward`, which is what the test catches.
    #[test]
    fn transformer_block_residual_changes_output() {
        let device = dev();
        let surrogate = ATan::default();
        let x = spike_input(2, 2, 4, 8);

        let block = SpikingTransformerBlockConfig::new(8, 2).init(&device);

        let with_residual = vec4(block.forward(x.clone(), &surrogate));

        // The same sequence of calls as inside `forward`, but WITHOUT
        // addition with `x`/`x1` - uses the same trained/
        // initialized `self.attention`/`self.mlp`/`self.sn_*`, so
        // that the discrepancy with `with_residual` can only be explained
        // the absence of residual addition, and not different initialization of weights.
        let attn_pre_spike = block.attention.forward_pre_spike(x.clone(), &surrogate);
        let x1_no_residual =
            apply_spiking_nonlinearity(&block.sn_attn_residual, attn_pre_spike, &surrogate);

        let mlp_pre_spike = block
            .mlp
            .forward_pre_spike(x1_no_residual.clone(), &surrogate);
        let x2_no_residual =
            apply_spiking_nonlinearity(&block.sn_mlp_residual, mlp_pre_spike, &surrogate);

        let without_residual = vec4(x2_no_residual);

        assert_ne!(
            with_residual, without_residual,
            "the result of forward coincided with the composition without residual addition: \
             it looks like x (or x1) is not added to the pre-spike output before SN"
        );
    }
}
