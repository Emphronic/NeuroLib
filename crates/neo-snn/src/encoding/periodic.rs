//! Periodic (regular) coding: deterministic spike train with
//! fixed period, inversely proportional to intensity.
//!
//! Corresponds to `firing_pattern='regular'` in `snntorch.spikegen.targets_rate`/
//! `targets_convert` - there this is one of three spike generation modes for
//! target tensors (`'regular'` / `'uniform'` / `'poisson'`), see
//! `snntorch.spikegen` documentation. Unlike [`super::rate_encode`]
//! (Poisson sampling - stochastic, different result for each
//! call), periodic - DETERMINISTIC regular spike train: the same
//! intensity always gives the same pattern, which is convenient for targets
//! (target) tensors for training and for reproducible tests.

use burn::tensor::Device;
use burn::tensor::Tensor;

/// Periodic encoding: `data` to `[0, 1]` → regular spike train
/// at `num_steps` steps, `[batch, num_steps, features]`.
///
/// `data` is interpreted as frequency (`rate`), as in [`super::rate_encode`]:
/// `1.0` - spike at each step, `0.5` - spike every step, `0.0` - spikes
/// no at all. The period between spikes is `round(1 / rate)` steps (the higher
/// `rate`, the more often the spikes), the first spike is at step `0`.
///
/// The implementation does not sample `T` random numbers like [`super::rate_encode`],
/// and builds a pattern directly according to the formula - cheaper and without RNG.
///
/// `data` outside `[0, 1]` is clamped: `rate <= 0` gives zero (no spikes)
/// train, `rate`, no matter how big it is, behaves like `rate = 1.0` (spike
/// at each step) - the same clamping convention as [`super::latency_encode`].
pub fn periodic_encode(data: Tensor<2>, num_steps: usize, device: &Device) -> Tensor<3> {
    assert!(num_steps > 0, "periodic_encode: num_steps must be > 0");

    let [batch, features] = data.dims();
    let rate = data.clamp(0.0, 1.0);

    // period[b,f] = round(1 / rate[b,f]), rate == 0 -> "infinite" period
    // (more num_steps so that there are no spikes at all). We compute tensorically,
    // to avoid reading data on the CPU until the end of the function.
    //
    // `1 / (rate + eps)` with `rate == 0` gives a FINITE period
    // (`round(1/eps) = 100_000_000` at `eps = 1e-8`), not true infinity.
    // For `num_steps <= period` this is indistinguishable from "no spikes" -
    // the small-step tests below pass - but past 1e8 steps a zero-rate input
    // would produce one spike at step `period`, violating the contract
    // "`rate=0.0` -> no spikes". A separate `zero_rate` mask forcibly resets
    // the whole train for those (batch, feature) pairs, so the guarantee does
    // not depend on `num_steps`.
    let eps = 1e-8f32;
    let zero_rate = rate.clone().lower_equal_elem(0.0).float(); // 1.0 where rate == 0
    let not_zero_rate = zero_rate.mul_scalar(-1.0).add_scalar(1.0); // 1.0 where rate != 0
    let period = Tensor::<2>::ones([batch, features], device)
        .div(rate.add_scalar(eps))
        .round();

    // period == 0 (when rate was so close to 1 that round gave 0)
    // unacceptable - minimum period 1 step (spike at each step).
    let period = period.clamp_min(1.0);

    let mut spikes = Vec::with_capacity(num_steps);
    for t in 0..num_steps {
        // Spike at step t, if t is divisible by period without remainder, that is
        // t - period * floor(t / period) == 0. Tensor version of the remainder:
        // burn does not provide a ready-made modulo for float, we build using floor.
        let t_val = Tensor::<2>::ones([batch, features], device).mul_scalar(t as f32);
        let quotient = t_val.clone().div(period.clone()).floor();
        let remainder = t_val - quotient.mul(period.clone());
        let is_spike = remainder.abs().lower_equal_elem(0.5).float();
        // We force it to zero where rate == 0 - regardless of
        // whether an “infinite” period fits into num_steps.
        spikes.push(is_spike.mul(not_zero_rate.clone()));
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

    /// `rate == 1.0` must give a spike at EVERY step - period 1.
    #[test]
    fn rate_one_spikes_every_step() {
        let device = dev();
        let data = t2(vec![1.0], [1, 1]);
        let out = periodic_encode(data, 5, &device);
        assert_eq!(vec3(out), vec![1.0, 1.0, 1.0, 1.0, 1.0]);
    }

    /// `rate == 0.5` is required to give period 2: spike at steps 0, 2, 4, ...
    #[test]
    fn rate_half_spikes_every_other_step() {
        let device = dev();
        let data = t2(vec![0.5], [1, 1]);
        let out = periodic_encode(data, 6, &device);
        assert_eq!(vec3(out), vec![1.0, 0.0, 1.0, 0.0, 1.0, 0.0]);
    }

    /// `rate == 0.0` is obliged not to give a SINGLE spike.
    #[test]
    fn rate_zero_spikes_never() {
        let device = dev();
        let data = t2(vec![0.0], [1, 1]);
        let out = periodic_encode(data, 5, &device);
        assert_eq!(vec3(out), vec![0.0, 0.0, 0.0, 0.0, 0.0]);
    }

    /// `rate == 0.0` must not spike even beyond the finite period implied by
    /// `1/(rate+eps)`: with `eps = 1e-8`, `round(1/eps) = 100_000_000`, so
    /// this test runs past that step to verify the zero-rate mask.
    ///
    /// `#[ignore]`: 10⁸ steps is expensive (allocations and a cycle on the
    /// CPU backend), not for a normal run. Run separately:
    /// `cargo test -- --ignored rate_zero_spikes_never_past_eps_period`.
    #[test]
    #[ignore]
    fn rate_zero_spikes_never_past_eps_period() {
        let device = dev();
        let data = t2(vec![0.0], [1, 1]);
        let num_steps = 100_000_005; // the old "period" would have been 100_000_000
        let out = vec3(periodic_encode(data, num_steps, &device));
        assert!(
            out.iter().all(|&x| x == 0.0),
            "rate=0.0 should not spike at any of the {num_steps} steps"
        );
    }

    /// Determinism is the main difference from `rate_encode`: two
    /// calls with the same data must give an IDENTICAL pattern.
    #[test]
    fn deterministic_across_calls() {
        let device = dev();
        let data = t2(vec![0.3, 0.7], [1, 2]);
        let out1 = vec3(periodic_encode(data.clone(), 10, &device));
        let out2 = vec3(periodic_encode(data, 10, &device));
        assert_eq!(out1, out2);
    }

    /// Different elements of a batch/feature are coded independently with their own period.
    #[test]
    fn independent_per_batch_and_feature() {
        let device = dev();
        // batch=1, 2 features: rate=1.0 (period 1) and rate=0.25 (period 4).
        let data = t2(vec![1.0, 0.25], [1, 2]);
        let out = periodic_encode(data, 8, &device);
        let v = vec3(out); // [1, 8, 2], order: t0f0,t0f1,t1f0,t1f1,...
        let feature0: Vec<f32> = (0..8).map(|t| v[t * 2]).collect();
        let feature1: Vec<f32> = (0..8).map(|t| v[t * 2 + 1]).collect();
        assert_eq!(feature0, vec![1.0; 8], "rate=1.0 -> spike every step");
        assert_eq!(
            feature1,
            vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0],
            "rate=0.25 -> period 4"
        );
    }
}
