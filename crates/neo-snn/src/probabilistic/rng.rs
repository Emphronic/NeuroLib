//! Controllable RNG source — the software model of the paper's hardware randomness.
//!
//! The paper draws the stochastic threshold from the physical stochasticity of the MTJ
//! (magnetic tunnel junction) switching. Software: a seeded pseudo-random generator
//! ([`StochasticThresholdLeakyConfig::seed`](crate::neuron::StochasticThresholdLeakyConfig::seed))
//! makes runs reproducible.

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use burn::tensor::{Device, Tensor};

/// Seeded RNG source with per-call samples.
#[derive(Debug, Clone)]
pub struct RngSource {
    rng: StdRng,
}

impl RngSource {
    /// New source. `seed = Some(s)` — reproducible; `None` — nondeterministic.
    pub fn new(seed: Option<u64>) -> Self {
        let seed = seed.unwrap_or_else(|| rand::thread_rng().gen_range(0..u64::MAX));
        Self {
            rng: StdRng::seed_from_u64(seed),
        }
    }

    /// Uniform sample `U[0, 1)`.
    pub fn next_uniform(&mut self) -> f32 {
        self.rng.gen_range(0.0..1.0)
    }

    /// Standard normal sample — Box–Muller.
    pub fn next_normal(&mut self) -> f32 {
        let u1 = self.rng.gen_range(f32::EPSILON..1.0);
        let u2 = self.rng.gen_range(0.0..1.0);
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos()
    }

    /// Normal sample `N(mean, std_dev)`.
    pub fn sample_normal(&mut self, mean: f32, std_dev: f32) -> f32 {
        mean + std_dev * self.next_normal()
    }

    /// Independent normal samples of shape `dims` — per-neuron draws of the stochastic threshold.
    pub fn normal_tensor(&mut self, dims: [usize; 2], device: &Device) -> Tensor<2> {
        let [batch, units] = dims;
        let mut values = Vec::with_capacity(batch * units);
        for _ in 0..batch * units {
            values.push(self.next_normal());
        }
        Tensor::<1>::from_floats(values.as_slice(), device).reshape([batch, units])
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn same_seed_reproduces_the_sequence() {
        let mut a = RngSource::new(Some(42));
        let mut b = RngSource::new(Some(42));
        let v_a: Vec<f32> = (0..16).map(|_| a.next_normal()).collect();
        let v_b: Vec<f32> = (0..16).map(|_| b.next_normal()).collect();
        assert_eq!(v_a, v_b);
        let mut c = RngSource::new(Some(43));
        assert!(v_a.iter().any(|&v| v != c.next_normal()));
    }

    #[test]
    fn normal_samples_are_centered() {
        let mut rng = RngSource::new(Some(7));
        let samples: Vec<f32> = (0..4096).map(|_| rng.next_normal()).collect();
        let mean: f32 = samples.iter().sum::<f32>() / samples.len() as f32;
        let var: f32 =
            samples.iter().map(|x| (x - mean) * (x - mean)).sum::<f32>() / samples.len() as f32;
        assert!(mean.abs() < 0.05, "mean {mean}");
        assert!((var - 1.0).abs() < 0.1, "variance {var}");
    }
}
