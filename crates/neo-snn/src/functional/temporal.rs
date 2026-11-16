//! Reductions of a spike train along the time axis.
//!
//! All functions take `[batch, num_steps, features]` — the shape returned by
//! the encoders in [`crate::encoding`] and the one in which the network output
//! is usually accumulated — and return `[batch, features]`.
//!
//! They are implemented as a loop over steps with `slice` rather than a single
//! reduction along the axis: [`LeakyParallel`](crate::neuron::LeakyParallel) works
//! the same way, and compared with it there is no extra cost — the autodiff
//! graph still contains each step separately.

use burn::tensor::Tensor;

/// A slice of one time step: `[batch, steps, features]` → `[batch, features]`.
///
/// Panics if `step` is out of bounds.
pub fn time_slice(x: Tensor<3>, step: usize) -> Tensor<2> {
    let [batch, steps, features] = x.dims();
    assert!(
        step < steps,
        "time_slice: step {step} out of range 0..{steps}"
    );
    x.slice([0..batch, step..step + 1, 0..features])
        .reshape([batch, features])
}

/// Sum over time — the total NUMBER of spikes of each neuron.
pub fn sum_over_time(x: Tensor<3>) -> Tensor<2> {
    let [batch, steps, features] = x.dims();
    assert!(steps > 0, "sum_over_time: empty time axis");

    // One reduction instead of `steps` slices: the backward pass of `slice` builds a zero tensor
    // the size of the whole input for each slice (O(T²) on the backward pass).
    x.sum_dim(1).reshape([batch, features])
}

/// Mean over time — the FIRING RATE of each neuron in [0, 1].
pub fn mean_over_time(x: Tensor<3>) -> Tensor<2> {
    let steps = x.dims()[1];
    assert!(steps > 0, "mean_over_time: empty time axis");
    sum_over_time(x).div_scalar(steps as f32)
}

/// Maximum over time.
///
/// Needed for membrane-potential readout: the winning class is the one whose
/// membrane rose highest at least once, even if no spike ever occurred.
pub fn max_over_time(x: Tensor<3>) -> Tensor<2> {
    let steps = x.dims()[1];
    assert!(steps > 0, "max_over_time: empty time axis");

    // Slices via `split_time` (two-level slicing, backward O(T·√T) instead of O(T²)).
    let mut xs = crate::neuron::split_time(x).into_iter();
    let mut best = xs.next().expect("steps > 0 checked above");
    for cur in xs {
        // mask = 1 where the current step is greater than or equal to the accumulated maximum.
        // The comparison is non-differentiable, so the gradient flows only into the
        // step that actually turned out to be the maximum — as it should.
        let mask = best.clone().lower_equal(cur.clone()).float();
        let keep = mask.clone().mul_scalar(-1.0).add_scalar(1.0);
        best = cur.mul(mask) + best.mul(keep);
    }
    best
}
