//! Arctan style surrogate (snnTorch: `spikegen.ATan` / `surrogate.atan`).
//!
//! # Fixed
//!
//! The previous implementation was called `ATan`, but actually computed
//! `0.5 + 0.5·tanh(α·x)`, a tanh-based sigmoid shape, not
//! arctangent. In snnTorch, an ATan surrogate is a shifted and scaled
//! `arctan`:
//!
//! ```text
//! S(x) ≈ (1/π)·arctan(π·α·x/2)
//! ```
//!
//! with derivative (used as a surrogate gradient via
//! straight-through, see [`super::spike_operator`]):
//!
//! ```text
//! dS/dx = (α/2) / (1 + (π·α·x/2)²)
//! ```
//!
//! `tanh` and `arctan` are both smooth sigmoid-like functions, but with different
//! tails (`tanh` decreases exponentially, `arctan` decreases exponentially), therefore
//! these are not interchangeable approximations: substituting one for the other
//! changes the shape of the surrogate gradient, not just its scale. Here
//! uses real `arctan` via `Tensor::atan` (available in burn
//! since at least 0.21) so that the behavior matches snnTorch.
//!
//! The direct value is shifted to `[0, 1]` (the constant `0.5` is added), as in
//! remaining sigmoid surrogates in this crate ([`super::FastSigmoid`],
//! [`super::SigmoidSurrogate`]) - the shift does not affect the derivative.

use burn::tensor::Tensor;

use super::Surrogate;

/// ATan surrogate: `0.5 + (1/π)·arctan(π·α·x/2)`, values in `(0, 1)`.
#[derive(Debug, Clone, Copy)]
pub struct ATan {
    /// Slope α.
    pub alpha: f64,
}

impl Default for ATan {
    fn default() -> Self {
        Self { alpha: 2.0 }
    }
}

impl ATan {
    /// New ATan surrogate with a given slope.
    pub fn new(alpha: f64) -> Self {
        Self { alpha }
    }
}

impl Surrogate for ATan {
    fn fused_kind(&self) -> Option<neo_synk::snn::SurrogateKind> {
        Some(neo_synk::snn::SurrogateKind::ATan {
            alpha: self.alpha as f32,
        })
    }

    fn surrogate_forward(&self, x: Tensor<2>) -> Tensor<2> {
        let k = (std::f64::consts::PI * self.alpha / 2.0) as f32;
        x.mul_scalar(k)
            .atan()
            .mul_scalar(1.0 / std::f32::consts::PI)
            .add_scalar(0.5)
    }
}

#[cfg(test)]
mod tests {
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

    /// At zero arctan(0) = 0, so the surrogate must give EXACTLY 0.5 -
    /// the same symmetry point as the previous tanh implementation, but this
    /// property of any odd S-shaped function shifted by 0.5, not
    /// a sign that the implementation is exactly arctan.
    #[test]
    fn zero_input_gives_half() {
        let x = t2(vec![0.0], [1, 1]);
        let out = vec2(ATan::default().surrogate_forward(x));
        assert!((out[0] - 0.5).abs() < 1e-6);
    }

    /// The values must lie strictly in (0, 1) - arctan is limited, just like
    /// tanh, but asymptotically approaches the boundaries more slowly.
    #[test]
    fn stays_within_unit_interval() {
        let x = t2(vec![-100.0, -1.0, 0.5, 1.0, 100.0], [1, 5]);
        let out = vec2(ATan::default().surrogate_forward(x));
        for v in out {
            assert!(v > 0.0 && v < 1.0, "went beyond (0,1): {v}");
        }
    }

    /// Oddness around 0.5: S(-x) - 0.5 == -(S(x) - 0.5), since
    /// arctan is an odd function.
    #[test]
    fn antisymmetric_around_half() {
        let surrogate = ATan::new(3.0);
        let pos = vec2(surrogate.surrogate_forward(t2(vec![0.7], [1, 1])))[0];
        let neg = vec2(surrogate.surrogate_forward(t2(vec![-0.7], [1, 1])))[0];
        assert!(((pos - 0.5) + (neg - 0.5)).abs() < 1e-5);
    }

    /// Difference from the tanh-variant: in average values, arctan and tanh give
    /// noticeably different numbers for the same α - if the implementation were secret
    /// remained tanh, this test would have failed.
    #[test]
    fn differs_from_tanh_shape_at_moderate_x() {
        let alpha = 2.0f64;
        let x = 1.0f32;

        let atan_out = vec2(ATan::new(alpha).surrogate_forward(t2(vec![x], [1, 1])))[0];

        let tanh_out = 0.5 + 0.5 * (alpha as f32 * x).tanh();

        assert!(
            (atan_out - tanh_out).abs() > 1e-3,
            "arctan and tanh coincided at x={x}, alpha={alpha}: {atan_out} vs {tanh_out} - the surrogate is similar to the old tanh implementation"
        );
    }
}
