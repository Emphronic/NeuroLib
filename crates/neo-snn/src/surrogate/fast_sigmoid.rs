//! Fast sigmoid surrogate (snntorch `fast_sigmoid`).
//!
//! The smooth forward `0.5 + 0.5·x/(1 + k·|x|)` gives the gradient
//! `0.5/(1 + k·|x|)²`, exactly the polynomial decay implied by the name.
//!
//! A plain sigmoid surrogate with a high slope (25) would instead decay
//! exponentially in |x|, so neurons far from the threshold would stop
//! receiving gradient at all (the classic “dead neuron” problem in SNNs).
//! Fast sigmoid keeps the damping polynomial, which is the point of this
//! surrogate.

use burn::tensor::Tensor;

use super::Surrogate;

/// Fast sigmoid surrogate with adjustable slope.
#[derive(Debug, Clone, Copy)]
pub struct FastSigmoid {
    /// Slope k; higher - sharper. Typically 25.
    pub slope: f64,
}

impl Default for FastSigmoid {
    fn default() -> Self {
        Self { slope: 25.0 }
    }
}

impl FastSigmoid {
    /// New fast sigmoid surrogate.
    pub fn new(slope: f64) -> Self {
        Self { slope }
    }
}

impl Surrogate for FastSigmoid {
    fn fused_kind(&self) -> Option<neo_synk::snn::SurrogateKind> {
        Some(neo_synk::snn::SurrogateKind::FastSigmoid {
            slope: self.slope as f32,
        })
    }

    fn surrogate_forward(&self, x: Tensor<2>) -> Tensor<2> {
        let k = self.slope as f32;
        let denom = x.clone().abs().mul_scalar(k).add_scalar(1.0);
        x.div(denom).mul_scalar(0.5).add_scalar(0.5)
    }
}
