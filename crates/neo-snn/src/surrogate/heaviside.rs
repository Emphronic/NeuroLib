//! Heaviside “as is”: gradient 1 above the threshold and 0 below.
//!
//! This is not an approximation, but an exact derivative of the shifted step where
//! it is defined. Training with it only works because after the spike
//! the membrane is still reset; a neuron that has gone below the threshold
//! does not receive a gradient at all - the classic problem of “dead neurons”.
//!
//! It is worth keeping it as a STANDARD FOR COMPARISON: if training with the real
//! the surrogate is no better than with this, which means the surrogate is connected incorrectly.

use burn::tensor::Tensor;
use burn::tensor::activation::relu;

use super::Surrogate;

/// Heaviside surrogate: `σ(x) = relu(x)`, that is, the gradient of `1` at `x ≥ 0`
/// and `0` with `x < 0`.
#[derive(Debug, Clone, Copy, Default)]
pub struct Heaviside;

impl Surrogate for Heaviside {
    fn surrogate_forward(&self, x: Tensor<2>) -> Tensor<2> {
        relu(x)
    }
}
