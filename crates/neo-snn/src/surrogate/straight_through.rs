//! Straight-through estimator: gradient is 1, forward remains Heaviside.

use burn::tensor::Tensor;

use super::Surrogate;

/// Straight-through: The surrogate value is equal to the input (gradient 1).
#[derive(Debug, Clone, Copy, Default)]
pub struct StraightThrough;

impl Surrogate for StraightThrough {
    fn fused_kind(&self) -> Option<neo_synk::snn::SurrogateKind> {
        Some(neo_synk::snn::SurrogateKind::Straight)
    }

    fn surrogate_forward(&self, x: Tensor<2>) -> Tensor<2> {
        x
    }
}
