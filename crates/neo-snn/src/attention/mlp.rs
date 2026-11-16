//! Spiking MLP: `fc1 → BatchNorm → SN → fc2 → BatchNorm → SN`.
//!
//! Standard feed-forward subblock of the transformer, converted to spikes -
//! used in both Spikformer and Spike-Driven Transformer as a second
//! half [`super::SpikingTransformerBlock`] after attention.

use burn::module::Module;
use burn::nn::{BatchNorm, BatchNormConfig, Linear, LinearConfig};
use burn::tensor::Device;

use super::{SpikeSequence, apply_spiking_nonlinearity, spiking_nonlinearity};
use crate::neuron::Leaky;
use crate::surrogate::Surrogate;

/// Configuration [`SpikingMlp`].
#[derive(Debug, Clone, Copy)]
pub struct SpikingMlpConfig {
    /// Dimension of input/output `D`.
    pub dim: usize,
    /// The dimension of the hidden layer (usually `4 * dim`, as in ANN transformers).
    pub hidden_dim: usize,
    /// Spike threshold for both SN(·).
    pub threshold: f32,
}

impl SpikingMlpConfig {
    /// New configuration. `hidden_dim` defaults to `4 * dim`.
    pub fn new(dim: usize) -> Self {
        assert!(dim > 0, "dim must be > 0");
        Self {
            dim,
            hidden_dim: dim * 4,
            threshold: 1.0,
        }
    }

    /// Set the hidden layer dimension explicitly.
    pub fn with_hidden_dim(mut self, hidden_dim: usize) -> Self {
        assert!(hidden_dim > 0, "hidden_dim must be > 0");
        self.hidden_dim = hidden_dim;
        self
    }

    /// Set the spike threshold.
    pub fn with_threshold(mut self, threshold: f32) -> Self {
        self.threshold = threshold;
        self
    }

    /// Create a layer.
    pub fn init(&self, device: &Device) -> SpikingMlp {
        SpikingMlp::new(self, device)
    }
}

/// Spiking MLP subblock.
#[derive(Module, Debug)]
pub struct SpikingMlp {
    /// First linear layer (`dim -> hidden_dim`).
    pub fc1: Linear,
    /// BatchNorm after `fc1`.
    pub bn1: BatchNorm,
    /// Second linear layer (`hidden_dim -> dim`).
    pub fc2: Linear,
    /// BatchNorm after `fc2`.
    pub bn2: BatchNorm,
    /// SN(·) after `fc1`/`bn1`.
    #[module(skip)]
    pub sn1: Leaky,
    /// SN(·) after `fc2`/`bn2`.
    #[module(skip)]
    pub sn2: Leaky,
}

impl SpikingMlp {
    /// Create a layer from the configuration.
    pub fn new(config: &SpikingMlpConfig, device: &Device) -> Self {
        Self {
            fc1: LinearConfig::new(config.dim, config.hidden_dim).init(device),
            bn1: BatchNormConfig::new(config.hidden_dim).init(device),
            fc2: LinearConfig::new(config.hidden_dim, config.dim).init(device),
            bn2: BatchNormConfig::new(config.dim).init(device),
            sn1: spiking_nonlinearity(config.threshold, device),
            sn2: spiking_nonlinearity(config.threshold, device),
        }
    }

    /// Direct passage up to and including the final SN. `x` - spike input
    /// `[T, B, N, D]`.
    pub fn forward<S: Surrogate + ?Sized>(&self, x: SpikeSequence, surrogate: &S) -> SpikeSequence {
        let dims = x.dims();
        let pre_spike = self.forward_pre_spike(x, surrogate);
        apply_spiking_nonlinearity(&self.sn2, pre_spike, surrogate).reshape(dims)
    }

    /// The same thing, but without the final SN(·) - membrane potential after
    /// `bn2(fc2(·))`. Need [`super::SpikingTransformerBlock`] for
    /// residual connections along the membrane, see
    /// [`super::SpikingSelfAttention::forward_pre_spike`].
    pub fn forward_pre_spike<S: Surrogate + ?Sized>(
        &self,
        x: SpikeSequence,
        surrogate: &S,
    ) -> SpikeSequence {
        let [t, b, n, d] = x.dims();
        let tb = t * b;
        let flat = x.reshape([tb, n, d]);

        // permute before BN: see comment in
        // `SpikingSelfAttention::project` about why the trait is transferred
        // to the channel axis in front of `BatchNorm::forward`.
        let h = self.fc1.forward(flat);
        let hidden_dim = h.dims()[2];
        let h = self
            .bn1
            .forward::<3>(h.permute([0, 2, 1]))
            .permute([0, 2, 1]);
        let h = apply_spiking_nonlinearity(&self.sn1, h, surrogate);
        debug_assert_eq!(h.dims(), [tb, n, hidden_dim]);

        let projected = self.fc2.forward(h);
        let normed = self
            .bn2
            .forward::<3>(projected.permute([0, 2, 1]))
            .permute([0, 2, 1]);
        debug_assert_eq!(normed.dims(), [tb, n, d]);
        normed.reshape([t, b, n, d])
    }
}
