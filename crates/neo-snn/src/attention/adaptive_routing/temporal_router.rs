//! Temporal router — adaptive precision along the time axis.
//!
//! In the paper the gated spike firing of AdaE-IF/S-GSA also applies along the time axis —
//! the per-step spike activity routes each step to a fine/coarse firing pattern. Here the
//! gate is a monotone interpolation of the per-step spike energy (summed over the batch,
//! the tokens and the channels — `[T]`, [`route_temporal`]) between
//! [`AdaptiveRoutingConfig::coarse_threshold`] (quiet steps — gate 0, activity suppressed)
//! and [`AdaptiveRoutingConfig::fine_threshold`] (active steps — gate 1).

use burn::tensor::Tensor;

use super::AdaptiveRoutingConfig;
use crate::attention::SpikeSequence;

/// Per-step temporal gate — `[T]` — 1.0 — fine (active step), 0.0 — coarse (quiet step) —
/// interpolated by the per-step spike energy between
/// [`AdaptiveRoutingConfig::coarse_threshold`] and [`AdaptiveRoutingConfig::fine_threshold`].
pub fn route_temporal(seq: &SpikeSequence, cfg: &AdaptiveRoutingConfig) -> Tensor<1> {
    seq.clone()
        .sum_dims_squeeze::<1, _>(&[1, 2, 3])
        .sub_scalar(cfg.coarse_threshold)
        .div_scalar((cfg.fine_threshold - cfg.coarse_threshold).max(1e-6))
        .clamp(0.0f32, 1.0f32)
}

/// Apply the temporal gate: `seq * gate` — the `[T]` gate broadcast over `[T, B, N, D]` —
/// quiet steps' activity is suppressed.
pub fn apply_temporal(seq: &SpikeSequence, gate: &Tensor<1>) -> SpikeSequence {
    let [t, b, n, d] = seq.dims();
    let gate = gate.clone().reshape([t, 1, 1, 1]).expand([t, b, n, d]);
    seq.clone().mul(gate)
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::Device;

    #[test]
    fn route_temporal_gates_steps_by_activity() {
        let device = Device::flex();
        // seq [T=2, B=1, N=2, D=1]: t0: n0 = [1], n1 = [1] — active; t1: all zero
        let seq: SpikeSequence =
            Tensor::<1>::from_floats(vec![1.0f32, 1.0, 0.0, 0.0].as_slice(), &device)
                .reshape([2, 1, 2, 1]);
        let cfg = AdaptiveRoutingConfig {
            fine_threshold: 1.0,
            coarse_threshold: 0.05,
            ..AdaptiveRoutingConfig::default()
        };
        let gate = route_temporal(&seq, &cfg)
            .into_data()
            .try_to_vec::<f32>()
            .unwrap();
        // t0: energy 2 → (2 - 0.05)/(1 - 0.05) ≈ 2.16 → clamped to 1; t1: energy 0 → 0
        assert!((gate[0] - 1.0).abs() < 1e-6, "gate = {gate:?}");
        assert!((gate[1] - 0.0).abs() < 1e-6, "gate = {gate:?}");
    }

    #[test]
    fn route_temporal_gives_partial_gates_in_between() {
        let device = Device::flex();
        // seq [T=2, B=1, N=1, D=1]: [0.5, 0] — step energy 0.5
        let seq: SpikeSequence =
            Tensor::<1>::from_floats(vec![0.5f32, 0.0].as_slice(), &device).reshape([2, 1, 1, 1]);
        let cfg = AdaptiveRoutingConfig {
            fine_threshold: 1.0,
            coarse_threshold: 0.0,
            ..AdaptiveRoutingConfig::default()
        };
        let gate = route_temporal(&seq, &cfg)
            .into_data()
            .try_to_vec::<f32>()
            .unwrap();
        assert!((gate[0] - 0.5).abs() < 1e-6, "gate = {gate:?}");
        assert!((gate[1] - 0.0).abs() < 1e-6, "gate = {gate:?}");
    }

    #[test]
    fn apply_temporal_attenuates_quiet_steps() {
        let device = Device::flex();
        // seq [T=2, B=1, N=1, D=1]: [1, 0]
        let seq: SpikeSequence =
            Tensor::<1>::from_floats(vec![1.0f32, 0.0].as_slice(), &device).reshape([2, 1, 1, 1]);
        let gate = Tensor::<1>::from_floats(vec![0.5f32, 0.25].as_slice(), &device);
        let out = apply_temporal(&seq, &gate)
            .into_data()
            .try_to_vec::<f32>()
            .unwrap();
        assert_eq!(out, vec![0.5, 0.0]);
    }
}
