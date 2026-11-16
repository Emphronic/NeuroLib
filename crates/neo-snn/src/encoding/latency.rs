//! Latency encoding: intensity → first spike time
//! (the stronger the signal, the earlier the spike).

use burn::tensor::Device;
use burn::tensor::Tensor;

/// Latency encoding: `data` in [0, 1] → EXACTLY ONE spike per neuron at the moment,
/// proportional `(1 - value)`.
///
/// Returns `[batch, num_steps, features]`: 1 in the spike step, 0 everywhere else.
/// Step index = `round((1 - data)·(num_steps - 1))`, sandwiched in `[0, num_steps-1]`.
///
/// # Fixed
///
/// There were two independent bugs in the same function, noticed at different times
/// time:
///
/// 1. The mask was built like `(t <= s) & (s <= t+1)` with both non-strict
///    boundaries - a value that falls exactly on the integer `t` satisfies
///    condition for both bin `t` and bin `t-1`, that is, the neuron produced TWO
///    a spike instead of one (the latency code is, by definition, one-time). This
///    was fixed earlier - the upper bound became strict, `[t, t+1)`.
/// 2. That fix, however, made the mask interval: `is_spike =
///    (step_float >= t) & (step_float < t+1)` is a FLOOR function
///    (`floor`), and not `round`, which the documentation and signature promises
///    contract. At `step_float = 1.5`, the interval `[1, 2)` gives a spike at
///    `t=1`, whereas `round(1.5) = 2`. The documentation and code are different.
///
/// Now the spike index is calculated explicitly as `round(step_float)` (method
/// `Tensor::round`, bank rounding is not required - border `.5`
/// occurs only when `num_steps - 1` is even and is not the place
/// where correctness depends on the direction of rounding of the halves), clamped
/// to `[0, num_steps-1]`, and the mask is constructed by exact comparison `t ==
/// round(step_float)`, and not a half-open interval - so the code and
/// The documentation now describes the same formula.
pub fn latency_encode(data: Tensor<2>, num_steps: usize, device: &Device) -> Tensor<3> {
    assert!(num_steps > 0, "latency_encode: num_steps must be > 0");

    let shape = data.dims();
    let max_step = (num_steps - 1) as f32;

    let one: Tensor<2> = Tensor::ones(shape, device);
    // Clamp values outside [0, 1] so they stay within the step range, then
    // round to the nearest integer step as documented.
    let step_idx = (one - data)
        .clamp(0.0, 1.0)
        .mul_scalar(max_step)
        .round()
        .clamp(0.0, max_step);

    let mut spikes = Vec::with_capacity(num_steps);
    for t in 0..num_steps {
        let is_spike = step_idx.clone().equal_elem(t as f32).float();
        spikes.push(is_spike);
    }

    Tensor::stack::<3>(spikes, 1)
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::Device;

    fn dev() -> Device {
        Device::flex()
    }

    fn t2(data: Vec<f32>, shape: [usize; 2]) -> Tensor<2> {
        Tensor::<1>::from_floats(data.as_slice(), &dev()).reshape(shape)
    }

    fn vec3(t: Tensor<3>) -> Vec<f32> {
        t.into_data().try_to_vec::<f32>().unwrap()
    }

    /// `data == 1.0` (maximum intensity) must give a spike at
    /// The EARLIEST step is step 0.
    #[test]
    fn max_intensity_spikes_immediately() {
        let device = dev();
        let data = t2(vec![1.0], [1, 1]);
        let out = vec3(latency_encode(data, 4, &device));
        assert_eq!(out, vec![1.0, 0.0, 0.0, 0.0]);
    }

    /// `data == 0.0` (minimum intensity) must give a spike at the
    /// LATEST step, `num_steps - 1`.
    #[test]
    fn min_intensity_spikes_last() {
        let device = dev();
        let data = t2(vec![0.0], [1, 1]);
        let out = vec3(latency_encode(data, 4, &device));
        assert_eq!(out, vec![0.0, 0.0, 0.0, 1.0]);
    }

    /// Exactly ONE spike per neuron, even when `step_float` lands exactly
    /// on an integer boundary; the spike mask must not fire twice there.
    #[test]
    fn exactly_one_spike_at_integer_boundary() {
        let device = dev();
        // num_steps=5, data=0.5 -> step_float = 0.5 * 4 = 2.0 (exactly integer).
        let data = t2(vec![0.5], [1, 1]);
        let out = vec3(latency_encode(data, 5, &device));
        let spike_count: f32 = out.iter().sum();
        assert_eq!(spike_count, 1.0, "there must be exactly one spike: {out:?}");
        assert_eq!(out, vec![0.0, 0.0, 1.0, 0.0, 0.0]);
    }

    /// The spike position must match exactly `round`, not `floor`.
    /// `num_steps=4, data=0.5` gives `step_float = 0.5 * 3 = 1.5`;
    /// floor-style rounding would place the spike at step 1, whereas
    /// `round(1.5) = 2`, as documented.
    #[test]
    fn spike_position_matches_round_not_floor() {
        let device = dev();
        let data = t2(vec![0.5], [1, 1]);
        let out = vec3(latency_encode(data, 4, &device));
        assert_eq!(
            out,
            vec![0.0, 0.0, 1.0, 0.0],
            "a spike was expected at the round(1.5)=2 step, received {out:?}"
        );
    }

    /// Values outside `[0, 1]` must be clamped:
    /// `data > 1.0` behaves like `data == 1.0` (spike at step 0),
    /// `data < 0.0` - like `data == 0.0` (spike at the last step).
    #[test]
    fn out_of_range_values_are_clamped() {
        let device = dev();
        let data = t2(vec![5.0, -3.0], [1, 2]);
        let out = vec3(latency_encode(data, 3, &device));
        // Memory order: t0f0,t0f1, t1f0,t1f1, t2f0,t2f1
        let feature0: Vec<f32> = (0..3).map(|t| out[t * 2]).collect();
        let feature1: Vec<f32> = (0..3).map(|t| out[t * 2 + 1]).collect();
        assert_eq!(feature0, vec![1.0, 0.0, 0.0], "data=5.0 -> spike at step 0");
        assert_eq!(
            feature1,
            vec![0.0, 0.0, 1.0],
            "data=-3.0 -> spike at the last step"
        );
    }

    /// Batch/feature independence, as in other coders.
    #[test]
    fn independent_per_batch_and_feature() {
        let device = dev();
        let data = t2(vec![1.0, 0.0], [1, 2]);
        let out = vec3(latency_encode(data, 3, &device));
        let feature0: Vec<f32> = (0..3).map(|t| out[t * 2]).collect();
        let feature1: Vec<f32> = (0..3).map(|t| out[t * 2 + 1]).collect();
        assert_eq!(feature0, vec![1.0, 0.0, 0.0]);
        assert_eq!(feature1, vec![0.0, 0.0, 1.0]);
    }
}
