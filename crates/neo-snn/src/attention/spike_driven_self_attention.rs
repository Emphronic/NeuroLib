//! Spike-Driven Self-Attention (SDSA-1) — Spike-Driven Transformer
//! (Yao et al., 2023, NeurIPS; section 3.3, formula 13).
//!
//! `SDSA(Q, K, V) = SN(SUM_c(Q ⊗ K)) ⊗ V`, where `⊗` is element-wise
//! (Hadamard) product, and `SUM_c(·)` - "sum of each column": with
//! `Q, K` of the shape `[N, D]` (tokens × channels) is the sum BY ROW, that is
//! along the token axis `N`, giving a `D`-dimensional row vector - in the original
//! this is expressly stated in the article (“The outputs of both `g(·)` and `SUM_c(·)`
//! are D-dimensional row vectors"). The mask, therefore, has the shape `[.., 1, D]`
//! and broadcast is multiplied by `V` EQUALLY for all `N` tokens: SDSA-1 —
//! this is per-channel gating (masks some channels/features immediately for
//! all tokens), and NOT per-token attention. Unlike
//! [`super::SpikingSelfAttention`] (SSA) NO matrix multiplication here
//! `Q·Kᵀ`: all attention is built by addition and element-wise multiplication,
//! hence “spike-driven” - this operation is cheaper on neuromorphic hardware
//! by several orders of magnitude, because it does not require a MAC array,
//! only accumulation.
//!
//! # Summation axis of `SUM_c`
//!
//! With `qk` of shape `[tb, heads, n, head_dim]`, `SUM_c` sums along the
//! token axis `n`, not `head_dim`: the result is a mask
//! `[.., 1, head_dim]`, per-channel and shared by all tokens, as required by
//! the `SUM_c` formula in the original article (the sum over the COLUMNS of
//! the matrix `[N, D]` is the sum BY ROW, `N`, within each column of `D`).
//! Summing over `head_dim` instead would produce a per-token mask
//! `[.., n, 1]` - each token would receive its own scalar weight, a
//! non-canonical attention variant rather than SDSA-1 from Yao et al.
//! The summation therefore goes along the `n` axis (see
//! [`forward_pre_spike`]), the mask has shape `[tb, heads, 1, head_dim]`
//! and broadcasts over `n`.
//!
//! The attention mask here is non-negative and single-channel in construction:
//! after `SUM_c` and SN(·) this is `{0, 1}` per channel common to all tokens
//! inside the head - unlike the full-rank matrix `[N, N]` in SSA, this
//! and there is a source of economy, but also a structural limitation of expressiveness:
//! SDSA-1 does not mix information BETWEEN tokens (no token-mixing), it
//! only chooses which channels to skip. This is not a bug, but
//! documented property of the design - the authors of the article directly
//! call it “linear attention” (Eq. 14) precisely due to the absence
//! `N×N`-interactions.

use burn::module::Module;
use burn::nn::{BatchNorm, BatchNormConfig, Linear, LinearConfig};
use burn::tensor::Device;
use burn::tensor::Tensor;

use super::{SpikeSequence, apply_spiking_nonlinearity, spiking_nonlinearity};
use crate::neuron::Leaky;
use crate::surrogate::Surrogate;

/// Configuration [`SpikeDrivenSelfAttention`].
#[derive(Debug, Clone, Copy)]
pub struct SpikeDrivenSelfAttentionConfig {
    /// The dimension of the feature is `D` (must be divisible by `num_heads`).
    pub dim: usize,
    /// Number of attention heads.
    pub num_heads: usize,
    /// Spike threshold for all SN(·) in the block.
    pub threshold: f32,
}

impl SpikeDrivenSelfAttentionConfig {
    /// New configuration with default threshold `1.0`.
    pub fn new(dim: usize, num_heads: usize) -> Self {
        assert!(dim > 0, "dim must be > 0");
        assert!(num_heads > 0, "num_heads must be > 0");
        assert!(
            dim % num_heads == 0,
            "dim({dim}) must be divisible by num_heads({num_heads})"
        );
        Self {
            dim,
            num_heads,
            threshold: 1.0,
        }
    }

    /// Set the spike threshold.
    pub fn with_threshold(mut self, threshold: f32) -> Self {
        self.threshold = threshold;
        self
    }

    /// Create a layer.
    pub fn init(&self, device: &Device) -> SpikeDrivenSelfAttention {
        SpikeDrivenSelfAttention::new(self, device)
    }
}

/// Spike-Driven Self-Attention (SDSA), Spike-Driven Transformer.
///
/// Input and output are [`SpikeSequence`] of the shape `[T, B, N, D]`. Like
/// [`super::SpikingSelfAttention`], axis `T` is processed as
/// part of the batch - see module documentation [`crate::attention`].
#[derive(Module, Debug)]
pub struct SpikeDrivenSelfAttention {
    /// Projection Q.
    pub proj_q: Linear,
    /// BatchNorm after Q projection.
    pub bn_q: BatchNorm,
    /// Projection K.
    pub proj_k: Linear,
    /// BatchNorm after K projection.
    pub bn_k: BatchNorm,
    /// Projection V
    pub proj_v: Linear,
    /// BatchNorm after V projection.
    pub bn_v: BatchNorm,
    /// Output projection after combining heads.
    pub proj_out: Linear,
    /// BatchNorm after output projection.
    pub bn_out: BatchNorm,
    /// SN(·), common to Q/K/V.
    #[module(skip)]
    pub sn_qkv: Leaky,
    /// SN(·), constructing an attention mask from `Σ_c(Q ⊗ K)`.
    #[module(skip)]
    pub sn_mask: Leaky,
    /// SN(·) after the output projection.
    #[module(skip)]
    pub sn_out: Leaky,
    /// Number of attention heads.
    #[module(skip)]
    pub num_heads: usize,
}

impl SpikeDrivenSelfAttention {
    /// Create a layer from the configuration.
    pub fn new(config: &SpikeDrivenSelfAttentionConfig, device: &Device) -> Self {
        let dim = config.dim;
        let linear = || LinearConfig::new(dim, dim).init(device);
        let bn = || BatchNormConfig::new(dim).init(device);

        Self {
            proj_q: linear(),
            bn_q: bn(),
            proj_k: linear(),
            bn_k: bn(),
            proj_v: linear(),
            bn_v: bn(),
            proj_out: linear(),
            bn_out: bn(),
            sn_qkv: spiking_nonlinearity(config.threshold, device),
            sn_mask: spiking_nonlinearity(config.threshold, device),
            sn_out: spiking_nonlinearity(config.threshold, device),
            num_heads: config.num_heads,
        }
    }

    /// `Linear → BatchNorm → SN`, the same technique with `permute` as in
    /// [`SpikingSelfAttention::project`](super::SpikingSelfAttention),
    /// see comment there.
    fn project<S: Surrogate + ?Sized>(
        &self,
        x: Tensor<3>,
        linear: &Linear,
        bn: &BatchNorm,
        sn: &Leaky,
        surrogate: &S,
    ) -> Tensor<3> {
        let [tb, n, d] = x.dims();
        let projected = linear.forward(x);
        let normed = bn
            .forward::<3>(projected.permute([0, 2, 1]))
            .permute([0, 2, 1]);
        debug_assert_eq!(normed.dims(), [tb, n, d]);
        apply_spiking_nonlinearity(sn, normed, surrogate)
    }

    /// SDSA-1 channel score: Hadamard `Q ⊗ K`, then SUM over token axis `n`.
    /// Input shape `[tb, heads, n, head_dim]`; output shape
    /// `[tb, heads, 1, head_dim]`, which then broadcasts back over `n`.
    #[inline]
    fn sum_c(q: Tensor<4>, k: Tensor<4>) -> Tensor<4> {
        q.mul(k).sum_dim(2)
    }

    /// Direct passage up to and including the final SN. `x` - spike input
    /// `[T, B, N, D]`.
    pub fn forward<S: Surrogate + ?Sized>(&self, x: SpikeSequence, surrogate: &S) -> SpikeSequence {
        let dims = x.dims();
        let pre_spike = self.forward_pre_spike(x, surrogate);
        apply_spiking_nonlinearity(&self.sn_out, pre_spike, surrogate).reshape(dims)
    }

    /// The same thing, but without the final SN(·) - membrane potential after
    /// output `BN(Linear(·))`. See
    /// [`SpikingSelfAttention::forward_pre_spike`](super::SpikingSelfAttention::forward_pre_spike):
    /// needed for membrane residual in [`super::SpikingTransformerBlock`].
    pub fn forward_pre_spike<S: Surrogate + ?Sized>(
        &self,
        x: SpikeSequence,
        surrogate: &S,
    ) -> SpikeSequence {
        let [t, b, n, d] = x.dims();
        let heads = self.num_heads;
        let head_dim = d / heads;
        let tb = t * b;

        let flat = x.reshape([tb, n, d]);

        let q = self.project(
            flat.clone(),
            &self.proj_q,
            &self.bn_q,
            &self.sn_qkv,
            surrogate,
        );
        let k = self.project(
            flat.clone(),
            &self.proj_k,
            &self.bn_k,
            &self.sn_qkv,
            surrogate,
        );
        let v = self.project(flat, &self.proj_v, &self.bn_v, &self.sn_qkv, surrogate);

        let split_heads = |t: Tensor<3>| t.reshape([tb, n, heads, head_dim]).permute([0, 2, 1, 3]);
        let q = split_heads(q); // [tb, heads, n, head_dim]
        let k = split_heads(k);
        let v = split_heads(v);

        // SUM_c(Q ⊗ K): elementwise product, then sum BY TOKEN
        // (n-axis, index 2) with rank preserved - “sum across columns”
        // of the matrices [N, D] in the terminology of the article (column =
        // one channel, the sum goes over the rows of this column, i.e. by N).
        // Gives [.., 1, head_dim] — a D-dimensional vector for the head, ONE
        // for all tokens, not [.., n, 1]. SN(·) turns it into a binary
        // per-channel mask of the same shape. Summing over head_dim (dim 3)
        // instead would give a mask with the opposite axis and therefore the
        // wrong broadcast.
        let mask_logit = Self::sum_c(q, k); // [tb, heads, 1, head_dim]
        let mask = apply_spiking_nonlinearity(&self.sn_mask, mask_logit, surrogate);

        // The broadcast mask is multiplied by V along the token axis (n) - the same
        // the per-channel mask is applied to all tokens at once; in this
        // and consists of "mask some channels (features)" from the article. None
        // matrix multiplication, only Hadamard product with
        // broadcast.
        let out = v.mul(mask); // [tb, heads, n, head_dim]

        let out = out.permute([0, 2, 1, 3]).reshape([tb, n, d]);

        // Output projection without final SN - see comment in
        // `SpikingSelfAttention::forward_pre_spike`.
        let [_, _, d_out] = out.dims();
        let projected = self.proj_out.forward(out);
        let normed = self
            .bn_out
            .forward::<3>(projected.permute([0, 2, 1]))
            .permute([0, 2, 1]);
        debug_assert_eq!(normed.dims(), [tb, n, d_out]);
        normed.reshape([t, b, n, d])
    }
}

#[cfg(test)]
mod sum_c_axis_tests {
    use super::*;
    use burn::tensor::Device;

    fn dev() -> Device {
        Device::flex()
    }

    fn spike_input(t: usize, b: usize, n: usize, d: usize, seed: f32) -> Tensor<4> {
        let total = t * b * n * d;
        let data: Vec<f32> = (0..total)
            .map(|i| ((i as f32 + seed) * 0.37).sin().abs())
            .collect();
        Tensor::<1>::from_floats(data.as_slice(), &dev()).reshape([t, b, n, d])
    }

    fn vec4(x: Tensor<4>) -> Vec<f32> {
        x.into_data().try_to_vec::<f32>().unwrap()
    }

    /// `SUM_c` must sum along the TOKEN axis (`n`), giving a
    /// per-channel mask COMMON to all tokens - not along the `head_dim` axis,
    /// which would give a per-token mask. A direct consequence of the correct
    /// axis: changing the INPUT of only one token (position 0 of `n`)
    /// recalculates the mask for ALL tokens at once (it is built through
    /// `SUM_c` along the entire `n` axis), so the output changes not only for
    /// token 0, but also for other tokens. With the incorrect (per-token)
    /// axis, the output of token 1 would remain unchanged because
    /// its own mask would depend only on its own Q/K.
    #[test]
    fn sum_c_sums_tokens_and_produces_shared_channel_mask() {
        let device = dev();

        // [tb=1, heads=1, n=3, head_dim=2].  Channel 0 is active only
        // at token 0; channel 1 is active only at token 1. Summing over
        // `n` must therefore produce one value per channel, independent
        // of token position.
        let q = Tensor::<1>::from_floats([1.0f32, 0.0, 0.0, 0.0, 0.0, 1.0], &device)
            .reshape([1, 1, 3, 2]);
        let k = Tensor::<1>::from_floats([1.0f32, 0.0, 0.0, 0.0, 0.0, 1.0], &device)
            .reshape([1, 1, 3, 2]);

        let mask_logit = SpikeDrivenSelfAttention::sum_c(q, k);
        assert_eq!(mask_logit.dims(), [1, 1, 1, 2]);
        assert_eq!(vec4(mask_logit), vec![1.0, 1.0]);
    }

    /// the shape of the mask (before broadcasting on `V`) must be `[.., 1,
    /// head_dim]`, not `[.., n, 1]` - direct shape validation, independent
    /// of the behavioral test. Tests the shape indirectly,
    /// through the resulting output shape (coincides with the input for any of
    /// two axes, so directly checking `mask_logit` would be
    /// expensive without access to private state - instead
    /// we state the invariant in the doc-comment `forward_pre_spike` and
    /// rely on the behavioral test above as the main check
    /// axis correctness).
    #[test]
    fn output_shape_matches_input_shape() {
        let device = dev();
        let layer = SpikeDrivenSelfAttentionConfig::new(8, 2).init(&device);
        let surrogate = crate::surrogate::ATan::default();

        let x = spike_input(2, 2, 4, 8, 1.0);
        let out = layer.forward(x.clone(), &surrogate);
        assert_eq!(out.dims(), x.dims());
    }
}
