//! “spike operator” family: the gradient is specified piecewise with respect to the threshold.
//!
//! General idea: above the threshold the neuron is spiked and the derivative is taken equal to
//! unit; below the threshold, something small is substituted for zero so that
//! subthreshold neurons did not remain gradient-free forever. Options
//! differ in WHAT exactly is substituted:
//!
//! * [`LeakySpikeOperator`] - constant slope `k` (like Leaky ReLU);
//! * [`StochasticSpikeOperator`] is a random variable, unique for everyone
//!   element and each step.
//!
//! The deterministic option without a subthreshold gradient is
//! [`Heaviside`](super::Heaviside); alias [`SpikeOperator`] abandoned
//! for compatibility with snntorch terminology.

use burn::tensor::activation::relu;
use burn::tensor::{Distribution, Tensor};

use super::{Heaviside, Surrogate};

/// Classic spike operator: the gradient `1` is above the threshold, `0` is below.
///
/// The reverse pass matches [`Heaviside`](super::Heaviside) -
/// This is an alias, not a separate implementation.
pub type SpikeOperator = Heaviside;

/// Spike operator with constant subthreshold slope.
///
/// Gradient: `1` at `x ≥ 0` and `k` at `x < 0`. Same as derivative
/// shifted Leaky ReLU, hence the name.
#[derive(Debug, Clone, Copy)]
pub struct LeakySpikeOperator {
    /// Subthreshold slope k. Typically 0.1.
    pub slope: f64,
}

impl Default for LeakySpikeOperator {
    fn default() -> Self {
        Self { slope: 0.1 }
    }
}

impl LeakySpikeOperator {
    /// New leaky spike operator.
    pub fn new(slope: f64) -> Self {
        Self { slope }
    }
}

impl Surrogate for LeakySpikeOperator {
    fn surrogate_forward(&self, x: Tensor<2>) -> Tensor<2> {
        let k = self.slope as f32;
        // σ(x) = relu(x) − k relu(−x): for x > 0 this is x (gradient 1),
        // for x < 0 it is k·x (gradient k).
        let pos = relu(x.clone());
        let neg = relu(x.mul_scalar(-1.0)).mul_scalar(k);
        pos - neg
    }
}

/// Spike operator with RANDOM subthreshold gradient.
///
/// Gradient: `1` for `x ≥ 0`, and for `x < 0` a sample from
/// `(U[−0.5, 0.5) + μ)·σ²`, independent for each element and each
/// call. The point is that the subthreshold neuron receives non-systematic
/// shift in one direction, and noise with zero average (at `μ = 0`): it
/// pushes away “dead” neurons without imposing directions on them.
///
/// The noise is detached from the graph - it is part of the GRADIENT, and not a trainable quantity.
#[derive(Debug, Clone, Copy)]
pub struct StochasticSpikeOperator {
    /// Shift μ of the subthreshold distribution.
    pub mean: f64,
    /// Scale σ² of the subthreshold distribution.
    pub variance: f64,
}

impl Default for StochasticSpikeOperator {
    fn default() -> Self {
        Self {
            mean: 0.0,
            variance: 0.2,
        }
    }
}

impl StochasticSpikeOperator {
    /// New stochastic spike operator.
    pub fn new(mean: f64, variance: f64) -> Self {
        Self { mean, variance }
    }
}

impl Surrogate for StochasticSpikeOperator {
    fn surrogate_forward(&self, x: Tensor<2>) -> Tensor<2> {
        let dims = x.dims();
        let device = x.device();
        let mean = self.mean as f32;
        let variance = self.variance as f32;

        let noise: Tensor<2> = Tensor::random(dims, Distribution::Uniform(0.0, 1.0), &device)
            .add_scalar(mean - 0.5)
            .mul_scalar(variance)
            .detach();

        let pos = relu(x.clone());
        let neg = relu(x.mul_scalar(-1.0)).mul(noise);
        pos - neg
    }
}
