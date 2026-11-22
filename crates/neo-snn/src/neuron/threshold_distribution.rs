//! Threshold distribution — the stochastic (Bayesian) threshold of
//! [`StochasticThresholdLeaky`](crate::neuron::StochasticThresholdLeaky).

use burn::tensor::{Device, Tensor};

use crate::probabilistic::RngSource;

/// Threshold distribution — the neuron draws its threshold `θ ~ N(mean, std_dev)`
/// (the paper: "for every neuron the parameters of the threshold distribution — mean and
/// variance — are trained").
#[derive(Debug, Clone, Copy)]
pub struct ThresholdDistribution {
    /// Mean of the threshold distribution.
    pub mean: f32,
    /// Std of the threshold distribution.
    pub std_dev: f32,
}

impl ThresholdDistribution {
    /// New distribution.
    pub fn new(mean: f32, std_dev: f32) -> Self {
        Self {
            mean,
            std_dev: std_dev.max(1e-3),
        }
    }

    /// Draw a single threshold sample.
    pub fn sample(&self, rng: &mut RngSource) -> f32 {
        rng.sample_normal(self.mean, self.std_dev)
    }

    /// Draw a tensor of threshold samples of shape `dims` — per-neuron draws.
    pub fn sample_tensor(
        &self,
        dims: [usize; 2],
        device: &Device,
        rng: &mut RngSource,
    ) -> Tensor<2> {
        rng.normal_tensor(dims, device)
            .mul_scalar(self.std_dev)
            .add_scalar(self.mean)
    }
}
