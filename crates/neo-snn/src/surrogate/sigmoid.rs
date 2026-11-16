//! Regular sigmoid surrogate: `sigmoid(α·x)`.

use burn::tensor::Tensor;
use burn::tensor::activation::sigmoid;

use super::Surrogate;

/// Sigmoid surrogate with adjustable slope.
#[derive(Debug, Clone, Copy)]
pub struct SigmoidSurrogate {
    /// Slope α.
    pub alpha: f64,
}

impl Default for SigmoidSurrogate {
    fn default() -> Self {
        Self { alpha: 10.0 }
    }
}

impl SigmoidSurrogate {
    /// New sigmoid surrogate.
    pub fn new(alpha: f64) -> Self {
        Self { alpha }
    }
}

impl Surrogate for SigmoidSurrogate {
    fn fused_kind(&self) -> Option<neo_synk::snn::SurrogateKind> {
        Some(neo_synk::snn::SurrogateKind::Sigmoid {
            alpha: self.alpha as f32,
        })
    }

    fn surrogate_forward(&self, x: Tensor<2>) -> Tensor<2> {
        sigmoid(x.mul_scalar(self.alpha as f32))
    }
}
