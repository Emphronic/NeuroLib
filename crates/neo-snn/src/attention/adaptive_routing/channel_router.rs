//! Channel router — S-LGU (Spiking Local-Gated Unit): allows controlling activity at the
//! channel level.
//!
//! In the paper the LGU gates the channels through a local gating unit — the channel
//! activity itself selects whether the channel passes through. Here the gate is a monotone
//! interpolation of the channel energy (spike counts per channel, summed over the time
//! axis and the tokens — [`channel_energy`]) between
//! [`AdaptiveRoutingConfig::coarse_threshold`] (silent channels — gate 0, activity
//! suppressed) and [`AdaptiveRoutingConfig::fine_threshold`] (active channels — gate 1).

use burn::tensor::Tensor;

use super::AdaptiveRoutingConfig;
use crate::attention::SpikeSequence;

/// Channel energy — spike counts per channel, summed over the time axis and the tokens —
/// `[B, D]` (the sequence is `[T, B, N, D]`).
pub fn channel_energy(seq: &SpikeSequence) -> Tensor<2> {
    seq.clone().sum_dims_squeeze::<2, _>(&[0, 2])
}

/// Channel gate — `[B, D]` in `[0, 1]`: 1 — active channel, 0 — silent channel —
/// interpolated by the channel energy between [`AdaptiveRoutingConfig::coarse_threshold`]
/// and [`AdaptiveRoutingConfig::fine_threshold`].
pub fn channel_gate(energy: &Tensor<2>, cfg: &AdaptiveRoutingConfig) -> Tensor<2> {
    energy
        .clone()
        .sub_scalar(cfg.coarse_threshold)
        .div_scalar((cfg.fine_threshold - cfg.coarse_threshold).max(1e-6))
        .clamp(0.0f32, 1.0f32)
}

/// Apply the channel gate (S-LGU) with the default config: `seq * gate` — the `[B, D]`
/// gate broadcast over `[T, B, N, D]` — silent channels' activity is suppressed.
pub fn apply(seq: &SpikeSequence) -> SpikeSequence {
    let cfg = AdaptiveRoutingConfig::default();
    let [t, b, n, d] = seq.dims();
    let gate = channel_gate(&channel_energy(seq), &cfg)
        .reshape([b, 1, 1, d])
        .expand([t, b, n, d]);
    seq.clone().mul(gate)
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::Device;

    #[test]
    fn channel_energy_sums_over_time_and_tokens() {
        let device = Device::flex();
        // seq [T=2, B=1, N=2, D=2]: t0: n0 = [1, 0], n1 = [0, 1]; t1: n0 = [1, 0], n1 = [0, 0]
        let data = vec![1.0f32, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0];
        let seq: SpikeSequence =
            Tensor::<1>::from_floats(data.as_slice(), &device).reshape([2, 1, 2, 2]);
        // [B=1, D=2]: d0 = 1 (t0n0) + 1 (t1n0); d1 = 1 (t0n1)
        assert_eq!(
            channel_energy(&seq)
                .into_data()
                .try_to_vec::<f32>()
                .unwrap(),
            vec![2.0, 1.0]
        );
    }

    #[test]
    fn channel_gate_is_zero_below_coarse_and_one_above_fine() {
        let device = Device::flex();
        let cfg = AdaptiveRoutingConfig {
            fine_threshold: 1.0,
            coarse_threshold: 0.0,
            ..AdaptiveRoutingConfig::default()
        };
        let energy =
            Tensor::<1>::from_floats(vec![0.0f32, 0.5, 1.0].as_slice(), &device).reshape([1, 3]);
        assert_eq!(
            channel_gate(&energy, &cfg)
                .into_data()
                .try_to_vec::<f32>()
                .unwrap(),
            vec![0.0, 0.5, 1.0]
        );
    }

    #[test]
    fn apply_passes_active_channels_and_suppresses_silent_ones() {
        let device = Device::flex();
        // seq [T=1, B=1, N=2, D=2]: t0: n0 = [1, 0] — active channel 0; n1 = [0, 0] — silent
        let seq: SpikeSequence =
            Tensor::<1>::from_floats(vec![1.0f32, 0.0, 0.0, 0.0].as_slice(), &device)
                .reshape([1, 1, 2, 2]);
        let out = apply(&seq).into_data().try_to_vec::<f32>().unwrap();
        assert!(
            (out[0] - 1.0).abs() < 1e-6,
            "active channel passes: {out:?}"
        );
        assert!(
            out[1..].iter().all(|v| v.abs() < 1e-6),
            "silent channels suppressed: {out:?}"
        );
    }

    #[test]
    fn apply_attenuates_low_energy_channels() {
        let device = Device::flex();
        // seq [T=1, B=1, N=2, D=2]: t0: n0 = [0.1, 0] — low-energy channel 0; n1 = [0, 0]
        let seq: SpikeSequence =
            Tensor::<1>::from_floats(vec![0.1f32, 0.0, 0.0, 0.0].as_slice(), &device)
                .reshape([1, 1, 2, 2]);
        let out = apply(&seq).into_data().try_to_vec::<f32>().unwrap();
        // default config: gate = (0.1 - 0.05)/0.45 ≈ 0.1111
        let expected = 0.1f32 * (0.1 - 0.05) / 0.45;
        assert!((out[0] - expected).abs() < 1e-6, "out = {out:?}");
        assert!(out[1..].iter().all(|v| v.abs() < 1e-6), "out = {out:?}");
    }
}
