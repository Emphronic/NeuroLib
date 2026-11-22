//! Spike neuron models (LIF, Synaptic, Alpha, RLeaky, RSynaptic,
//! Lapicque, AdaptiveLeaky, Izhikevich, AdEx, Hodgkin-Huxley, and
//! multilevel and ternary spikes).

mod adaptive_leaky;
mod adex;
mod alpha;
mod graded;
mod hodgkin_huxley;
mod izhikevich;
mod lapicque;
mod leaky;
/// A wrapper around fused/reference leaky-scan (see `neo_synk::snn`).
/// `LeakyParallel` always goes through [`leaky_scan_op::leaky_scan_fused`]:
/// implementation - `neo_snn::leaky_scan_forward`: on CubeCL devices
/// (wgpu / CUDA / ROCm) this is a fused GPU core with manual backward, on the rest -
/// standard for tensor operations (autodiff works in both cases).
/// Not `pub use`
/// entirely outside the crate - this is an internal implementation detail.
pub mod leaky_scan_op;
mod multistep;
mod parallel;
mod rleaky;
mod rsynaptic;
mod stochastic_threshold;
mod synaptic;
mod threshold_distribution;

pub use adaptive_leaky::{AdaptiveLeaky, AdaptiveLeakyConfig};
pub use adex::{AdEx, AdExConfig, AdExPreset, AdExState};
pub use alpha::Alpha;
pub use graded::{GradedLeaky, GradedLeakyConfig, expand_levels, multi_level_spike, split_ternary};
pub use hodgkin_huxley::{
    HodgkinHuxley, HodgkinHuxleyConfig, HodgkinHuxleyState, hh_rates, hh_rates_f32,
    hh_steady_state, hh_steady_state_as,
};
pub use izhikevich::{Izhikevich, IzhikevichConfig, IzhikevichPreset, IzhikevichState};
pub use lapicque::Lapicque;
pub use leaky::{Leaky, LeakyConfig};
pub use multistep::{MultiStepLeaky, MultiStepRLeaky, split_time};
pub use parallel::LeakyParallel;
pub use rleaky::RLeaky;
pub use rsynaptic::RSynaptic;
pub use stochastic_threshold::{
    StochasticThresholdLeaky, StochasticThresholdLeakyConfig, StochasticThresholdState,
};
pub use synaptic::Synaptic;
pub use threshold_distribution::ThresholdDistribution;

use burn::tensor::Tensor;

use crate::surrogate::Surrogate;

/// Edge-protected logit: inverse sigmoid for initialization
/// attenuation parameters.
///
/// The raw form `(beta / (1.0 - beta.max(0.01))).ln()` gives NaN for
/// `beta >= 1.0` (ln of a negative number) even before the first learning
/// step, and `-inf` for `beta == 0`. The input is clamped into the open
/// interval (0, 1).
pub(crate) fn logit(p: f32) -> f32 {
    const EPS: f32 = 1e-4;
    let p = p.clamp(EPS, 1.0 - EPS);
    (p / (1.0 - p)).ln()
}

/// Spike with TENSOR threshold.
///
/// The spike condition is strict: `mem > threshold` (like `(input > 0)` in snnTorch).
/// With `mem == threshold` there is no spike - this is important for bit matching
/// rasters when transferring ready-made snnTorch models.
///
/// Returns `(spike, spike_hard)`:
/// * `spike` — straight-through output (forward = Heaviside, backward — through a surrogate);
/// * `spike_hard` - the same step, but already detached from the graph; needed to reset the membrane.
///
/// In the original, the hard spike was restored again as
/// `spike.greater_equal_elem(0.5)`, that is, an unnecessary comparison at each step
/// in time and an implicit dependence on the fact that the surrogate does not go beyond 0.5.
pub fn spike_with_threshold<S: Surrogate + ?Sized>(
    mem: Tensor<2>,
    threshold: Tensor<2>,
    surrogate: &S,
) -> (Tensor<2>, Tensor<2>) {
    spike_from_x(mem - threshold, surrogate)
}

/// Spike by `x = mem − θ`: `(spike, spike_hard)`, see [`spike_with_threshold`].
///
/// If the surrogate has a closed derivative ([`Surrogate::fused_kind`]) and its structure
/// is supported (Flex), the straight-through estimator is ONE fused operation
/// (`neo_synk::snn::spike_surrogate`: forward - Heaviside, backward - `grad·σ'(x)`); otherwise -
/// `hard + (soft − soft.detach())`. The values and gradients are the same.
pub fn spike_from_x<S: Surrogate + ?Sized>(x: Tensor<2>, surrogate: &S) -> (Tensor<2>, Tensor<2>) {
    if let Some(kind) = surrogate.fused_kind()
        && neo_synk::snn::spike_fused_supported(&x.device())
    {
        let spike = neo_synk::snn::spike_surrogate(x, kind);
        // the forward value of the spike is the hard spike; `detach` does not copy data.
        let hard = spike.clone().detach();
        return (spike, hard);
    }
    let hard = x.clone().greater_elem(0.0).float().detach();
    let soft = surrogate.surrogate_forward(x);
    let spike = hard.clone() + (soft.clone() - soft.detach());
    (spike, hard)
}

/// Spike with SCALAR threshold. See [`spike_with_threshold`].
pub fn spike_with_scalar_threshold<S: Surrogate + ?Sized>(
    mem: Tensor<2>,
    threshold: f32,
    surrogate: &S,
) -> (Tensor<2>, Tensor<2>) {
    spike_from_x(mem.sub_scalar(threshold), surrogate)
}

/// Amplitude modulation of a finished spike (graded spikes, snntorch:
/// `graded_spikes_factor`).
///
/// `spike` — already calculated straight-through spike of the shape `[batch, units]`
/// (see [`spike_with_threshold`]); `factor` — shape amplitude multiplier
/// `[1]` (scalar, the same for all units), usually from
/// `Leaky::graded_factor_tensor`.
///
/// Multiplies `spike` element-wise by `factor`, expanded to the shape
/// `spike`. This does NOT change the fact of the spike (it has already been recorded by Heaviside
/// inside `spike`) and does not affect the surrogate gradient across the membrane -
/// the multiplication is linear, so the backward pass just multiplies
/// gradient to the same `factor`, including the case when `factor` itself
/// trainable (`learn_graded_spikes_factor == true`): gradient across it
/// also flows naturally through the multiplication of tensors.
///
/// At `factor == 1.0` (default value, see
/// [`LeakyConfig::graded_spikes_factor`](crate::neuron::LeakyConfig))
/// this is an identical transformation - the behavior is no different from
/// a regular binary spike.
pub fn apply_graded_spikes(spike: Tensor<2>, factor: Tensor<1>) -> Tensor<2> {
    let dims = spike.dims();
    spike.mul(factor.reshape([1, 1]).expand(dims))
}

#[cfg(test)]
mod graded_spikes_tests {
    use super::*;
    use burn::tensor::Device;
    fn dev() -> Device {
        Device::flex()
    }

    fn t2(data: Vec<f32>, shape: [usize; 2]) -> Tensor<2> {
        Tensor::<1>::from_floats(data.as_slice(), &dev()).reshape(shape)
    }

    fn t1(data: Vec<f32>) -> Tensor<1> {
        Tensor::<1>::from_floats(data.as_slice(), &dev())
    }

    fn vec2(t: Tensor<2>) -> Vec<f32> {
        t.into_data().try_to_vec::<f32>().unwrap()
    }

    /// `factor == 1.0` must be an identical transformation - the same
    /// contract, which is documented by `apply_graded_spikes` and
    /// `LeakyConfig::graded_spikes_factor` (default `1.0`).
    #[test]
    fn factor_one_is_identity() {
        let spike = t2(vec![0.0, 1.0, 1.0, 0.0], [2, 2]);
        let factor = t1(vec![1.0]);
        let out = vec2(apply_graded_spikes(spike, factor));
        assert_eq!(out, vec![0.0, 1.0, 1.0, 0.0]);
    }

    /// The scalar factor equally scales the amplitude of all spiked
    /// units and does not create spikes where there were none (0 * factor == 0).
    #[test]
    fn factor_scales_active_spikes_uniformly() {
        let spike = t2(vec![0.0, 1.0, 1.0, 1.0], [2, 2]);
        let factor = t1(vec![2.5]);
        let out = vec2(apply_graded_spikes(spike, factor));
        assert_eq!(out, vec![0.0, 2.5, 2.5, 2.5]);
    }

    /// The shape factor `[1]` must expand to an arbitrary shape
    /// `[batch, units]`, and not just square.
    #[test]
    fn factor_broadcasts_over_batch_and_units() {
        let spike = t2(vec![1.0, 0.0, 1.0, 1.0, 0.0, 1.0], [3, 2]);
        let factor = t1(vec![0.3]);
        let out = vec2(apply_graded_spikes(spike, factor));
        for (i, &v) in out.iter().enumerate() {
            let expected = if [0, 2, 3, 5].contains(&i) { 0.3 } else { 0.0 };
            assert!((v - expected).abs() < 1e-6, "index {i}: {v} vs {expected}");
        }
    }
}
