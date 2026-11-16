//! Triangular surrogate: gradient `max(0, 1 − k·|x|)`.
//!
//! The cheapest of all in terms of computation: no exponents, no divisions, just
//! clamp and multiply. The gradient support is FINITE - beyond
//! `|x| > 1/k` the gradient is strictly zero, whereas for sigmoid surrogates it is
//! only asymptotically small. In practice, this noticeably thins the backward
//! pass in deep SNNs.
//!
//! Suggested by: G. Bellec et al. (2018), «Long short-term memory and
//! learning-to-learn in networks of spiking neurons», NeurIPS.

use burn::tensor::Tensor;

use super::Surrogate;

/// Triangular surrogate with adjustable slope.
///
/// The direct meaning is the antiderivative of the triangle:
/// `σ(x) = 0.5 + c − (k/2)·c·|c|`, where `c = clamp(x, −1/k, 1/k)`.
/// Its derivative is exactly `max(0, 1 − k·|x|)`: peak 1 at zero, zero outside
/// `[−1/k, 1/k]`. The values lie in `0.5 ± 1/(2k)`.
#[derive(Debug, Clone, Copy)]
pub struct Triangular {
    /// Slope k. The larger, the narrower the non-zero gradient window.
    pub slope: f64,
}

impl Default for Triangular {
    fn default() -> Self {
        Self { slope: 1.0 }
    }
}

impl Triangular {
    /// New triangular surrogate.
    pub fn new(slope: f64) -> Self {
        Self { slope }
    }

    /// Non-zero gradient window width, `1/k`.
    pub fn support(&self) -> f64 {
        1.0 / self.slope
    }
}

impl Surrogate for Triangular {
    fn fused_kind(&self) -> Option<neo_synk::snn::SurrogateKind> {
        Some(neo_synk::snn::SurrogateKind::Triangular {
            slope: self.slope as f32,
        })
    }

    fn surrogate_forward(&self, x: Tensor<2>) -> Tensor<2> {
        let k = self.slope as f32;
        let bound = 1.0 / k;

        // The clamp gives a zero gradient outside the support — exactly what
        // is required from a triangle, and without a single mask.
        let c = x.clamp(-bound, bound);
        let quad = c.clone().mul(c.clone().abs()).mul_scalar(k * 0.5);
        (c - quad).add_scalar(0.5)
    }
}
