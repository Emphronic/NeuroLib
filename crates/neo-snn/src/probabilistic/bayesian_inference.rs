//! Bayesian inference — the stochastic threshold IS the posterior over "how excitable is this
//! neuron": `ThresholdPosterior` — mean/std of the threshold distribution — can be combined
//! (product of Gaussians) and sampled — Bayesian sampling, without running the stochastic
//! neuron step by step.

use burn::tensor::{Device, Tensor};

use crate::neuron::StochasticThresholdLeaky;
use crate::probabilistic::rng::RngSource;

/// Posterior over the threshold — the distribution of the "device randomness" the neuron draws.
#[derive(Debug, Clone)]
pub struct ThresholdPosterior {
    /// Posterior mean of the threshold, `[batch, units]`.
    pub mean: Tensor<2>,
    /// Posterior std of the threshold, `[batch, units]`.
    pub std_dev: Tensor<2>,
}

impl ThresholdPosterior {
    /// Posterior from ready-made mean/std.
    pub fn new(mean: Tensor<2>, std_dev: Tensor<2>) -> Self {
        Self { mean, std_dev }
    }

    /// Posterior of the layer's trained threshold distribution, broadcast to `dims`.
    pub fn from_layer(layer: &StochasticThresholdLeaky, dims: [usize; 2], device: &Device) -> Self {
        Self {
            mean: layer.threshold_distribution_tensor(dims, device),
            std_dev: layer.threshold_std_tensor(dims, device),
        }
    }

    /// Draw a threshold sample — Bayesian sampling.
    pub fn sample(&self, rng: &mut RngSource) -> Tensor<2> {
        let device = self.mean.device();
        let noise = rng.normal_tensor(self.mean.dims(), &device);
        self.mean.clone() + self.std_dev.clone().mul(noise)
    }

    /// Combine two independent posteriors — product of Gaussians:
    /// `σ² = 1/(1/σ_a² + 1/σ_b²)`, `μ = (μ_a·σ_b² + μ_b·σ_a²)/(σ_a² + σ_b²)`.
    pub fn combine(a: &Self, b: &Self) -> Self {
        let sa2 = a.std_dev.clone().mul(a.std_dev.clone());
        let sb2 = b.std_dev.clone().mul(b.std_dev.clone());
        let den = sa2.clone().add(sb2.clone());
        let mean = a
            .mean
            .clone()
            .mul(sb2.clone())
            .add(b.mean.clone().mul(sa2.clone()))
            .div(den.clone());
        let variance = sa2.mul(sb2).div(den);
        Self {
            mean,
            std_dev: variance.sqrt(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::Device;

    #[test]
    fn posterior_samples_within_distribution_spread() {
        let device = Device::flex();
        let mean = Tensor::<1>::from_floats(vec![1.0].as_slice(), &device).reshape([1, 1]);
        let std = Tensor::<1>::from_floats(vec![0.5].as_slice(), &device).reshape([1, 1]);
        let posterior = ThresholdPosterior::new(mean, std);
        let mut rng = RngSource::new(Some(5));
        let mut samples: Vec<f32> = Vec::new();
        for _ in 0..2000 {
            samples.push(
                posterior
                    .sample(&mut rng)
                    .into_data()
                    .try_to_vec::<f32>()
                    .unwrap()[0],
            );
        }
        let mean: f32 = samples.iter().sum::<f32>() / samples.len() as f32;
        let var: f32 =
            samples.iter().map(|x| (x - mean) * (x - mean)).sum::<f32>() / samples.len() as f32;
        assert!((mean - 1.0).abs() < 0.1, "mean {mean}");
        assert!((var - 0.25).abs() < 0.1, "variance {var}");
    }

    #[test]
    fn combine_product_of_gaussians() {
        let device = Device::flex();
        let ones = |v: f32| Tensor::<1>::from_floats(vec![v].as_slice(), &device).reshape([1, 1]);
        let a = ThresholdPosterior::new(ones(0.0), ones(1.0));
        let b = ThresholdPosterior::new(ones(0.0), ones(1.0));
        // N(0,1) ⊛ N(0,1) = N(0, 1/√2)
        let c = ThresholdPosterior::combine(&a, &b);
        let std_c = c.std_dev.into_data().try_to_vec::<f32>().unwrap()[0];
        assert!(
            (std_c - std::f32::consts::FRAC_1_SQRT_2).abs() < 1e-5,
            "std_c = {std_c}"
        );
    }
}
