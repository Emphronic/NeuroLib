//! Activity regularization: penalties for too high or too low a firing
//! rate.
//!
//! ## Why SNNs specifically need this
//!
//! In an ordinary network, regularization fights overfitting. Here it has a
//! second, more pressing role: firing rate is a DIRECT metric of energy
//! consumption on neuromorphic hardware, where every spike counts. A network
//! that solves a task with two spikes instead of two hundred is not just more
//! elegant — it is exactly one hundred times cheaper.
//!
//! There is a third role as well. An unregularized SNN easily falls into one
//! of two degenerate regimes: the layer goes completely silent (no gradient,
//! training stalls) or fires at every step (spikes stop encoding anything and
//! the network degenerates into an ordinary one). A penalty for deviating from
//! a sensible rate keeps it between these extremes — for that there is
//! [`rate_bounds_penalty`].
//!
//! All functions take `[batch, num_steps, units]` and return a scalar that is
//! added to the main loss with its own coefficient.

use burn::tensor::Tensor;
use burn::tensor::activation::relu;

use super::temporal::mean_over_time;

/// L1 activity penalty: the mean firing rate over all neurons.
///
/// Pushes the rate toward zero uniformly. The most direct way to buy sparsity
/// at the cost of accuracy; the coefficient is chosen according to the
/// acceptable loss in quality.
pub fn l1_spike_rate(spikes: Tensor<3>) -> Tensor<1> {
    // Spikes are non-negative, so |s| = s and no absolute value is needed.
    mean_over_time(spikes).mean()
}

/// L2 penalty for a neuron's rate deviating from the target.
///
/// Unlike [`l1_spike_rate`], the goal is not silence but a GIVEN activity
/// level (for example, 0.1 — one spike every ten steps). The rate is computed
/// per neuron, so the penalty balances the load between them, not just the
/// total activity of the layer.
pub fn l2_spike_rate(spikes: Tensor<3>, target_rate: f32) -> Tensor<1> {
    let diff = mean_over_time(spikes).sub_scalar(target_rate);
    diff.clone().mul(diff).mean()
}

/// Quadratic penalty for the rate LEAVING the corridor `[lower, upper]`.
///
/// Inside the corridor the penalty is exactly zero — the network is free to
/// use activity however it likes as long as it stays within reasonable bounds.
/// Only degeneration is penalized: complete silence of the layer (rate below
/// `lower`) and continuous firing (above `upper`).
///
/// This is gentler than [`l2_spike_rate`] with a single target point: there any
/// deviation from it is penalized, even if harmless.
pub fn rate_bounds_penalty(spikes: Tensor<3>, lower: f32, upper: f32) -> Tensor<1> {
    assert!(
        lower <= upper,
        "rate_bounds_penalty: lower bound {lower} is above upper bound {upper}"
    );

    let rate = mean_over_time(spikes);

    // relu zeroes the gradient inside the corridor — exactly the behavior
    // required here, without any masks.
    let over = relu(rate.clone().sub_scalar(upper));
    let under = relu(rate.mul_scalar(-1.0).add_scalar(lower));

    let over_sq = over.clone().mul(over).mean();
    let under_sq = under.clone().mul(under).mean();
    over_sq + under_sq
}
