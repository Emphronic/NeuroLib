//! Lateral inhibition: suppressing all responses except the strongest one.

use burn::tensor::Tensor;

/// Hard lateral inhibition for `[batch, units]`.
///
/// Subtracts the maximum along the feature axis, so the strongest response is
/// zeroed out and the rest become negative. Apply BEFORE the threshold/spike.
///
/// Note: the result is not in [0, 1] — it is a difference signal, not spikes.
/// If you need the soft variant, see [`lateral_inhibition_soft`].
pub fn lateral_inhibition(spikes: Tensor<2>) -> Tensor<2> {
    let max_val = spikes.clone().max_dim(1);
    spikes - max_val
}

/// Soft lateral inhibition: subtracts a `strength` fraction of the maximum.
///
/// `strength = 0.0` — no inhibition, `1.0` — equivalent to
/// [`lateral_inhibition`]. Intermediate values preserve some activity of weak
/// neurons, which in practice trains more stably than hard
/// winner-take-all.
pub fn lateral_inhibition_soft(spikes: Tensor<2>, strength: f32) -> Tensor<2> {
    let max_val = spikes.clone().max_dim(1).mul_scalar(strength);
    spikes - max_val
}

/// Winner-take-all: only the strongest neuron in the layer fires.
///
/// This is the `inhibition=True` from snntorch exposed as a separate
/// function. Spikes of all neurons except the one with the highest membrane
/// potential are zeroed out.
///
/// The difference from [`lateral_inhibition`] is substantial: there the signal
/// is transformed BEFORE the threshold and remains real-valued, here the
/// already computed spikes are filtered. The two do not conflict, but usually
/// only one of them is needed.
///
/// The mask is built from the membrane and detached from the graph: inhibition
/// is a structural rule of the layer, not a learnable operation, and the
/// gradient must flow through the spike itself, not through the choice of
/// winner.
///
/// Both tensors are `[batch, units]`. When maxima are equal, all neurons that
/// reach the maximum fire.
pub fn winner_take_all(spikes: Tensor<2>, mem: Tensor<2>) -> Tensor<2> {
    let dims = mem.dims();
    crate::error::assert_dims("winner_take_all", "spikes", spikes.dims(), dims);
    let max_val = mem.clone().max_dim(1).expand(dims);
    let mask = max_val.lower_equal(mem).float().detach();
    spikes.mul(mask)
}
