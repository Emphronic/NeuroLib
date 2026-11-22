//! Spiking Self-Attention (SSA) — Spikformer (Zhou et al., 2023).
//!
//! `Q, K, V = SN(BN(Linear(X)))`, then
//! `SSA(Q, K, V) = SN(Q · Kᵀ · V · s)`, and finally the output projection
//! `X' = SN(BN(Linear(SSA)))`.
//!
//! Key difference from normal (softmax) attention: `Q`, `K`, `V` - already
//! BINARY spike tensors (not arbitrary real ones), so
//! `Q · Kᵀ` is non-negative by construction and softmax is not needed - normalization
//! replaces the scalar scale `s` (in the original - a fixed constant,
//! here is also a configurable number, not `1/√d`: range of amounts by
//! binary vectors is different from that of softmax weights, and `1/√d` empirically
//! turns out to be too small).

use burn::module::Module;
use burn::nn::{BatchNorm, BatchNormConfig, Linear, LinearConfig};
use burn::tensor::Device;
use burn::tensor::Tensor;

use super::{SpikeSequence, apply_spiking_nonlinearity, spiking_nonlinearity};
use crate::neuron::Leaky;
use crate::surrogate::Surrogate;

/// Configuration [`SpikingSelfAttention`].
#[derive(Debug, Clone, Copy)]
pub struct SpikingSelfAttentionConfig {
    /// The dimension of the feature is `D` (must be divisible by `num_heads`).
    pub dim: usize,
    /// Number of attention heads.
    pub num_heads: usize,
    /// Scale `s` before the final SN in `SSA(Q,K,V) = SN(Q·Kᵀ·V·s)`.
    ///
    /// In the original article - a fixed constant (not `1/√d`,
    /// as in softmax attention): the sums for binary `Q`/`K`/`V` grow
    /// with the number of tokens differently than dot products of reals
    /// vectors, so the appropriate scale is usually much smaller than `1/√d`.
    pub scale: f32,
    /// Spike threshold for all four SN(·) in a block.
    pub threshold: f32,
}

impl SpikingSelfAttentionConfig {
    /// New configuration. `scale` defaults to `0.125`, as in
    /// reference implementation of Spikformer (`qk_scale` in the absence
    /// the explicit value is taken as `0.125`, and not `head_dim ** -0.5`).
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
            scale: 0.125,
            threshold: 1.0,
        }
    }

    /// Set the scale to `s`.
    pub fn with_scale(mut self, scale: f32) -> Self {
        self.scale = scale;
        self
    }

    /// Set the spike threshold.
    pub fn with_threshold(mut self, threshold: f32) -> Self {
        self.threshold = threshold;
        self
    }

    /// Create a layer.
    pub fn init(&self, device: &Device) -> SpikingSelfAttention {
        SpikingSelfAttention::new(self, device)
    }
}

/// Spiking Self-Attention (SSA), Spikformer.
///
/// Input and output - [`SpikeSequence`] of the shape `[T, B, N, D]` (time steps,
/// batch, tokens, sign). The `T` axis is not recurrent: see documentation
/// module [`crate::attention`] about static time processing as
/// additional dimension of the batch.
#[derive(Module, Debug)]
pub struct SpikingSelfAttention {
    /// Projection Q.
    pub proj_q: Linear,
    /// BatchNorm after projection Q (according to the feature, the `D` axis is rearranged to
    /// channel position before the call - see [`Self::project`]).
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
    /// SN(·) - common for Q/K/V (in the original this is the same type
    /// neuron with one configuration, applied three times).
    #[module(skip)]
    pub sn_qkv: Leaky,
    /// SN(·) after `Q·Kᵀ·V·s`.
    #[module(skip)]
    pub sn_attn: Leaky,
    /// SN(·) after the output projection.
    #[module(skip)]
    pub sn_out: Leaky,
    /// Number of attention heads.
    #[module(skip)]
    pub num_heads: usize,
    /// Scale `s`.
    #[module(skip)]
    pub scale: f32,
}

impl SpikingSelfAttention {
    /// Create a layer from the configuration.
    pub fn new(config: &SpikingSelfAttentionConfig, device: &Device) -> Self {
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
            sn_attn: spiking_nonlinearity(config.threshold, device),
            sn_out: spiking_nonlinearity(config.threshold, device),
            num_heads: config.num_heads,
            scale: config.scale,
        }
    }

    /// `Linear → BatchNorm → SN`, common path for Q/K/V.
    ///
    /// `BatchNorm` in burn normalizes along axis 1 (channel) of the tensor `[N, C, ..]`,
    /// and the sign here lies in the last axis `[T·B, N, D]`. Therefore, before
    /// BN time-and-batch axis is flattened, `permute` is applied,
    /// so that `D` takes the channel position, and after BN - back, using the same
    /// technique that in [`crate::layers::SConv2dLSTM`] reduces the map `[B,C,H,W]`
    /// to `[batch·h·w, channels]` for the surrogate gradient:
    /// `permute` before `reshape` is required, otherwise the axes will be mixed.
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

    /// Direct passage up to and including the final SN.
    ///
    /// `x` - already spike input `[T, B, N, D]` (usually patch embedding output
    /// or previous [`super::SpikingTransformerBlock`], the past itself
    /// via SN).
    pub fn forward<S: Surrogate + ?Sized>(&self, x: SpikeSequence, surrogate: &S) -> SpikeSequence {
        let dims = x.dims();
        let pre_spike = self.forward_pre_spike(x, surrogate);
        apply_spiking_nonlinearity(&self.sn_out, pre_spike, surrogate).reshape(dims)
    }

    /// The same thing, but returns the membrane potential BEFORE the final
    /// SN(·) of the output projection (i.e. `BN(Linear(SSA(Q,K,V)))`, without
    /// last SN). Need [`super::SpikingTransformerBlock`] for
    /// residual connections across the membrane: need to be added BEFORE discretization in
    /// spike, otherwise the residual would be reduced to the addition of two binary signals.
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

        // Multi-head split: [tb, n, d] -> [tb, heads, n, head_dim].
        let split_heads = |t: Tensor<3>| t.reshape([tb, n, heads, head_dim]).permute([0, 2, 1, 3]);
        let q = split_heads(q);
        let k = split_heads(k);
        let v = split_heads(v);

        // Q · Kᵀ · V, batch by [tb, heads]. burn matrix multiplies by
        // the last two axes and broadcast the rest - exactly what
        // needed for a batch of goals without an explicit loop.
        let attn = q.matmul(k.transpose()); // [tb, heads, n, n]
        let attn = attn.mul_scalar(self.scale);
        let out = attn.matmul(v); // [tb, heads, n, head_dim]

        let out = apply_spiking_nonlinearity(&self.sn_attn, out, surrogate);

        // Back to [tb, n, d]: permute before reshape is required (see
        // comment in `project` about the same technique in SConv2dLSTM).
        let out = out.permute([0, 2, 1, 3]).reshape([tb, n, d]);

        // Output projection WITHOUT the last SN - the same Linear→BN logic,
        // that in `project`, but nonlinearity does not apply here: it
        // applies the calling code (`forward`, or a block for residual).
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
