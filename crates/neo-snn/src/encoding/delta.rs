//! Delta encoding: spike when change exceeds threshold.

use burn::tensor::Device;
use burn::tensor::Tensor;

/// Delta encoding: compares `data` with the previous value.
///
/// In the first step, `prev` can be passed to `None` (equivalent to zeros).
/// Returns `[batch, features]`: `+1` if rising, `-1` if falling, `0` otherwise.
pub fn delta_encode(
    data: Tensor<2>,
    prev: Option<Tensor<2>>,
    threshold: f32,
    device: &Device,
) -> Tensor<2> {
    let prev = prev.unwrap_or_else(|| Tensor::zeros(data.dims(), device));
    let diff = data - prev;

    let pos = diff.clone().greater_equal_elem(threshold).float();
    let neg = diff.lower_equal_elem(-threshold).float();

    pos - neg
}
