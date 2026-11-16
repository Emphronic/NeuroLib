//! Surrogate "escape rate": gradient `k·exp(−β·|x|)`.
//!
//! The motivation is biophysical, not computational. In noisy models
//! neuron spike is not determined: the probability of spike increases with
//! the membrane approaches the threshold exponentially (Boltzmann shape
//! escape noise). The gradient here is the probability density
//! spike, that is, “how close the neuron was to changing its mind.”
//!
//! W. Gerstner, W. M. Kistler (2002), «Spiking Neuron Models»,
//! Cambridge University Press, ch. 5 (escape noise / escape rate).

use burn::tensor::Tensor;
use burn::tensor::activation::relu;

use super::Surrogate;

/// Boltzmann-type surrogate with adjustable width β and height k.
#[derive(Debug, Clone, Copy)]
pub struct SpikeRateEscape {
    /// Width β: the larger, the faster the gradient decays from the threshold.
    pub beta: f64,
    /// Height k: The gradient value is exactly at the threshold.
    pub slope: f64,
}

impl Default for SpikeRateEscape {
    fn default() -> Self {
        Self {
            beta: 1.0,
            slope: 25.0,
        }
    }
}

/// The lower limit for `beta`/`slope` is the same clamp-bottom as for
/// `graded_spikes_factor`/`gamma` in the rest of the crate, and not a ban through
/// `assert!`/`Result`: configuration constructors are everywhere here
/// "silently lead to the nearest valid value" rather than panic.
const MIN_POSITIVE: f64 = 1e-6;

impl SpikeRateEscape {
    /// New escape-rate surrogate.
    ///
    /// `beta` and `slope` are clamped from below to a positive value:
    /// surrogate formula - `k·exp(−β|x|)` with antiderivative divided by
    /// `β` (see [`Surrogate::surrogate_forward`] below), so `beta <= 0`
    /// gives division by zero (`beta == 0` → `inf`/`NaN` via `k/beta`)
    /// or violates the biophysical motivation of escape-rate (the gradient
    /// must DECREASE with distance from the threshold, not increase, which is
    /// what happens at `beta < 0`).
    pub fn new(beta: f64, slope: f64) -> Self {
        Self {
            beta: beta.max(MIN_POSITIVE),
            slope: slope.max(MIN_POSITIVE),
        }
    }

    /// Set the width β. Clamped from below - see [`Self::new`].
    pub fn with_beta(mut self, beta: f64) -> Self {
        self.beta = beta.max(MIN_POSITIVE);
        self
    }

    /// Set height k. Clamped from below - see [`Self::new`].
    pub fn with_slope(mut self, slope: f64) -> Self {
        self.slope = slope.max(MIN_POSITIVE);
        self
    }
}

impl Surrogate for SpikeRateEscape {
    fn surrogate_forward(&self, x: Tensor<2>) -> Tensor<2> {
        // Clamped here too, and not only in constructors: fields `beta`/
        // `slope` public, literal `SpikeRateEscape { beta: -1.0, .. }`
        // bypasses `new`/`with_beta`/`with_slope`. Invariant "beta, slope
        // > 0" should be maintained regardless of construction method.
        let beta = self.beta.max(MIN_POSITIVE) as f32;
        let k = self.slope.max(MIN_POSITIVE) as f32;

        // The antiderivative of k·exp(−β|x|), written through relu, so that neither
        // one branch did not have exp(+large): both exponents have
        // non-positive argument by construction.
        //
        //   F(x) = (k/β)·[ exp(−β·relu(−x)) + 1 − exp(−β·relu(x)) ]
        //
        // For x < 0 it is (k/β)·exp(βx), for x > 0 it is (k/β)·(2 − exp(−βx)).
        // The derivative in both cases is k·exp(−β|x|).
        //
        // Exactly at x = 0 both branches of relu give zero gradient - the measure point
        // zero, does not affect learning.
        let left = relu(x.clone().mul_scalar(-1.0)).mul_scalar(-beta).exp();
        let right = relu(x).mul_scalar(-beta).exp();

        (left - right).add_scalar(1.0).mul_scalar(k / beta)
    }
}

#[cfg(test)]
mod validation_tests {
    use super::*;
    use burn::tensor::Device;

    fn dev() -> Device {
        Device::flex()
    }

    fn t2(data: Vec<f32>, shape: [usize; 2]) -> Tensor<2> {
        Tensor::<1>::from_floats(data.as_slice(), &dev()).reshape(shape)
    }

    fn vec2(t: Tensor<2>) -> Vec<f32> {
        t.into_data().try_to_vec::<f32>().unwrap()
    }

    /// `beta = 0.0` must not give `inf`/`NaN` via `k/beta` - the
    /// constructor clamps it from below to a positive value.
    #[test]
    fn zero_beta_does_not_produce_nan_or_inf() {
        let s = SpikeRateEscape::new(0.0, 25.0);
        assert!(s.beta > 0.0, "beta must be clamped above zero: {}", s.beta);

        let out = vec2(s.surrogate_forward(t2(vec![0.0, 1.0, -1.0], [1, 3])));
        for v in out {
            assert!(
                v.is_finite(),
                "surrogate_forward gave a non-finite value: {v}"
            );
        }
    }

    /// Negative `beta` must also be clamped - otherwise the gradient
    /// increases with distance from the threshold instead of decreasing, which violates
    /// biophysical motivation escape-rate.
    #[test]
    fn negative_beta_is_clamped_positive() {
        let s = SpikeRateEscape::new(-3.0, 10.0);
        assert!(s.beta > 0.0, "beta must be positive: {}", s.beta);
    }

    /// `with_beta`/`with_slope` must clamp in the same way as `new`.
    #[test]
    fn setters_clamp_like_constructor() {
        let s = SpikeRateEscape::default().with_beta(-1.0).with_slope(0.0);
        assert!(s.beta > 0.0);
        assert!(s.slope > 0.0);
    }

    /// Literal structure construction (public fields) bypasses
    /// constructors - `surrogate_forward` must remain finite even
    /// then, clamping the value on the use.
    #[test]
    fn struct_literal_with_zero_beta_still_finite_on_forward() {
        let s = SpikeRateEscape {
            beta: 0.0,
            slope: 25.0,
        };
        let out = vec2(s.surrogate_forward(t2(vec![0.5], [1, 1])));
        assert!(out[0].is_finite(), "received {}", out[0]);
    }

    /// Valid positive values should not be changed by clamping.
    #[test]
    fn valid_values_are_unaffected() {
        let s = SpikeRateEscape::new(2.0, 30.0);
        assert!((s.beta - 2.0).abs() < 1e-9);
        assert!((s.slope - 30.0).abs() < 1e-9);
    }
}
