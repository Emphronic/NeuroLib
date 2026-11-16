//! BatchNorm through time (BNTT): separate BatchNorm parameters for each
//! time step. Used in SNNs to normalize activations independently at every
//! time step (Kim & Panda, 2021).
//!
//! # Layer variants
//!
//! Burn's `BatchNorm` takes the tensor rank as a parameter of `forward::<D>()`,
//! so `BatchNormTT1d` and `BatchNormTT2d` differ only in the input rank accepted
//! by `forward`. The separate names are kept for API compatibility.

use burn::module::Module;
use burn::nn::{BatchNorm, BatchNormConfig};
use burn::tensor::Device;
use burn::tensor::Tensor;

/// BatchNorm through time for 1D (dense) features.
///
/// Input: `[batch, channels, length]`. One BatchNorm for each time step.
#[derive(Module, Debug)]
pub struct BatchNormTT1d {
    /// One BatchNorm layer for each time step.
    pub layers: Vec<BatchNorm>,
}

impl BatchNormTT1d {
    /// Create a layer.
    pub fn new(num_steps: usize, num_features: usize, device: &Device) -> Self {
        assert!(num_steps > 0, "num_steps must be > 0");
        let config = BatchNormConfig::new(num_features);
        let layers = (0..num_steps).map(|_| config.init(device)).collect();
        Self { layers }
    }

    /// Direct pass for a given time step. Input: `[batch, channels, length]`.
    pub fn forward(&self, step_index: usize, x: Tensor<3>) -> Tensor<3> {
        self.layers[step_index].forward(x)
    }

    /// Number of time steps (length `layers`).
    pub fn num_steps(&self) -> usize {
        self.layers.len()
    }
}

/// BatchNorm through time for 2D (spatial) features.
///
/// Input: `[batch, channels, height, width]`. One BatchNorm for each time step.
#[derive(Module, Debug)]
pub struct BatchNormTT2d {
    /// One BatchNorm layer for each time step.
    pub layers: Vec<BatchNorm>,
}

impl BatchNormTT2d {
    /// Create a layer.
    pub fn new(num_steps: usize, num_features: usize, device: &Device) -> Self {
        assert!(num_steps > 0, "num_steps must be > 0");
        let config = BatchNormConfig::new(num_features);
        let layers = (0..num_steps).map(|_| config.init(device)).collect();
        Self { layers }
    }

    /// Direct pass for a given time step.
    /// Input: `[batch, channels, height, width]`.
    pub fn forward(&self, step_index: usize, x: Tensor<4>) -> Tensor<4> {
        self.layers[step_index].forward(x)
    }

    /// Number of time steps (length `layers`).
    pub fn num_steps(&self) -> usize {
        self.layers.len()
    }
}
