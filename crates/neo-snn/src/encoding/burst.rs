//! Burst coding: a burst of several spikes in a row instead of
//! one, the number of spikes in a burst is proportional to the intensity.
//!
//! # Relation to snnTorch
//!
//! In `snntorch.spikegen` (current documentation, versions 0.9.4/1.0.0)
//! there is no function `burst` - there are three official schemes: `rate`, `latency`,
//! `delta` (all three are already in this crate: [`super::rate_encode`],
//! [`super::latency_encode`], [`super::delta_encode`]). The closest thing
//! snnTorch has the parameter `firing_pattern` (`'regular'` / `'uniform'`
//! / `'poisson'`) at `targets_rate`/`targets_convert`, which sets
//! DISTRIBUTION of individual spikes over time, not bursts; `'regular'`
//! matches [`super::periodic_encode`] in this crate.
//!
//! Burst coding here is independent, but widely used in
//! neurobiological and SNN-literature scheme (burst/burst encoding
//! frequencies through the number of spikes in a short group, and not through the interspike
//! interval throughout the sequence). Added as an extra
//! parity with snnTorch, and not as a implementation of its specific function.
//!
//! # Scheme
//!
//! The intensity of `data ∈ [0, 1]` determines the LENGTH of the burst: `burst_len =
//! round(data · max_burst)` spikes in a row, starting from step `0`, with
//! interspike interval (ISI) `1` inside the burst (spike at each step
//! packs). After the pack there is silence until the end `num_steps`. `max_burst`
//! is limited to `num_steps` so that the pack does not go beyond
//! sequences.

use burn::tensor::Tensor;

/// Burst encoding: `data` to `[0, 1]` → burst of `round(data ·
/// max_burst)` spikes in a row (ISI = 1), starting from step `0`, after which
/// silence follows. Returns `[batch, num_steps, features]`.
///
/// `max_burst`—packet length at `data == 1.0`; clamped to `num_steps`,
/// if more is specified (the packet cannot be longer than the
/// sequences). `data` outside `[0, 1]` is clamped in the same way as in
/// [`super::periodic_encode`]: negative and zero values give
/// an empty pack (no spikes), values above `1.0` - maximum pack.
pub fn burst_encode(data: Tensor<2>, num_steps: usize, max_burst: usize) -> Tensor<3> {
    assert!(num_steps > 0, "burst_encode: num_steps must be > 0");
    assert!(max_burst > 0, "burst_encode: max_burst must be > 0");

    let max_burst = max_burst.min(num_steps);

    let clamped = data.clamp(0.0, 1.0);
    // burst_len[b,f] = round(data * max_burst), in range [0, max_burst].
    let burst_len = clamped.mul_scalar(max_burst as f32).round();

    let mut spikes = Vec::with_capacity(num_steps);
    for t in 0..num_steps {
        // Spike at step t, if t < burst_len (the burst takes steps
        // [0, burst_len), ISI=1 inside it).
        let is_spike = burst_len.clone().greater_elem(t as f32).float();
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

    /// `data == 1.0` is obliged to give a packet of maximum length `max_burst`.
    #[test]
    fn max_intensity_gives_full_burst() {
        let data = t2(vec![1.0], [1, 1]);
        let out = burst_encode(data, 6, 3);
        assert_eq!(vec3(out), vec![1.0, 1.0, 1.0, 0.0, 0.0, 0.0]);
    }

    /// `data == 0.0` is obliged not to give a SINGLE spike.
    #[test]
    fn zero_intensity_gives_no_spikes() {
        let data = t2(vec![0.0], [1, 1]);
        let out = burst_encode(data, 5, 3);
        assert_eq!(vec3(out), vec![0.0, 0.0, 0.0, 0.0, 0.0]);
    }

    /// Intermediate intensity produces a proportionally shorter burst.
    #[test]
    fn intermediate_intensity_gives_partial_burst() {
        // data=0.5, max_burst=4 -> round(0.5*4)=2 spikes in a row.
        let data = t2(vec![0.5], [1, 1]);
        let out = burst_encode(data, 6, 4);
        assert_eq!(vec3(out), vec![1.0, 1.0, 0.0, 0.0, 0.0, 0.0]);
    }

    /// `max_burst` more than `num_steps` must be clamped to `num_steps` -
    /// the packet cannot go out of sequence.
    #[test]
    fn max_burst_clamped_to_num_steps() {
        let data = t2(vec![1.0], [1, 1]);
        let out = burst_encode(data, 3, 10);
        assert_eq!(vec3(out), vec![1.0, 1.0, 1.0]);
    }

    /// Inside the ISI pack there is strictly 1: spikes go in a row, without gaps.
    #[test]
    fn burst_has_no_gaps_inside() {
        let data = t2(vec![0.8], [1, 1]);
        let out = vec3(burst_encode(data, 5, 5));
        // round(0.8*5) = 4 spikes in a row, then silence.
        assert_eq!(out, vec![1.0, 1.0, 1.0, 1.0, 0.0]);
    }

    /// Different batch/feature elements receive independent batch lengths.
    #[test]
    fn independent_per_batch_and_feature() {
        let data = t2(vec![1.0, 0.25], [1, 2]);
        let out = burst_encode(data, 4, 4);
        let v = vec3(out); // [1, 4, 2]
        let feature0: Vec<f32> = (0..4).map(|t| v[t * 2]).collect();
        let feature1: Vec<f32> = (0..4).map(|t| v[t * 2 + 1]).collect();
        assert_eq!(feature0, vec![1.0, 1.0, 1.0, 1.0]);
        assert_eq!(feature1, vec![1.0, 0.0, 0.0, 0.0]);
    }
}
