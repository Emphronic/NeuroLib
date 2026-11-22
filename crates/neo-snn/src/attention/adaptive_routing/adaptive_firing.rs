//! AdaE-IF — Adaptive Efficient Integrate-and-Fire: the neuron picks a more precise (fine) or
//! coarser mode of spiking activity depending on the token importance.
//!
//! Precision is in levels between [`AdaptiveRoutingConfig::min_firing_precision`] (coarse)
//! and [`AdaptiveRoutingConfig::max_firing_precision`] (fine), interpolated by the importance
//! between [`AdaptiveRoutingConfig::coarse_threshold`] and
//! [`AdaptiveRoutingConfig::fine_threshold`]; [`firing_gate`] scales the token's firing
//! activity by the interpolated precision — fine tokens keep the full activity, coarse tokens'
//! activity is attenuated.

use burn::tensor::Tensor;

use super::AdaptiveRoutingConfig;
use crate::attention::SpikeSequence;

/// Firing precision per token (levels) — between `min_firing_precision` (coarse) and
/// `max_firing_precision` (fine), interpolated by the importance between
/// [`AdaptiveRoutingConfig::coarse_threshold`] and [`AdaptiveRoutingConfig::fine_threshold`].
pub fn fire_precision(importance: &Tensor<2>, cfg: &AdaptiveRoutingConfig) -> Tensor<2> {
    let t = importance
        .clone()
        .sub_scalar(cfg.coarse_threshold)
        .div_scalar((cfg.fine_threshold - cfg.coarse_threshold).max(1e-6))
        .clamp(0.0f32, 1.0f32);
    let lo = cfg.min_firing_precision as f32;
    let hi = cfg.max_firing_precision as f32;
    t.mul_scalar(hi - lo).add_scalar(lo)
}

/// Gate in `[0, 1]` per token — 1.0 — fine, 0.0 — coarse — proportional between the
/// [`AdaptiveRoutingConfig::coarse_threshold`] and [`AdaptiveRoutingConfig::fine_threshold`].
pub fn firing_gate(importance: &Tensor<2>, cfg: &AdaptiveRoutingConfig) -> Tensor<2> {
    importance
        .clone()
        .sub_scalar(cfg.coarse_threshold)
        .div_scalar((cfg.fine_threshold - cfg.coarse_threshold).max(1e-6))
        .clamp(0.0f32, 1.0f32)
}

/// Apply the firing gate: `seq * gate` expanded over `[T, B, N, D]` — coarse tokens' activity
/// is attenuated (fine — unchanged).
pub fn apply(
    seq: &SpikeSequence,
    importance: &Tensor<2>,
    cfg: &AdaptiveRoutingConfig,
) -> SpikeSequence {
    let [t, b, n, d] = seq.dims();
    let gate = firing_gate(importance, cfg)
        .reshape([1, b, n, 1])
        .expand([t, b, n, d]);
    seq.clone().mul(gate)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::attention::SpikeSequence;
    use burn::tensor::Device;

    fn cfg() -> AdaptiveRoutingConfig {
        AdaptiveRoutingConfig {
            fine_threshold: 1.0,
            coarse_threshold: 0.0,
            min_firing_precision: 1,
            max_firing_precision: 4,
        }
    }

    #[test]
    fn fire_precision_interpolates_levels() {
        let device = Device::flex();
        let imp =
            Tensor::<1>::from_floats(vec![0.0f32, 0.5, 1.0].as_slice(), &device).reshape([1, 3]);
        let p = fire_precision(&imp, &cfg());
        let v = p.into_data().try_to_vec::<f32>().unwrap();
        assert!((v[0] - 1.0).abs() < 1e-5, "v = {v:?}");
        assert!((v[1] - 2.5).abs() < 1e-5, "v = {v:?}");
        assert!((v[2] - 4.0).abs() < 1e-5, "v = {v:?}");
    }

    #[test]
    fn firing_gate_is_one_fine_zero_coarse() {
        let device = Device::flex();
        let imp =
            Tensor::<1>::from_floats(vec![0.0f32, 0.5, 1.0].as_slice(), &device).reshape([1, 3]);
        assert_eq!(
            firing_gate(&imp, &cfg())
                .into_data()
                .try_to_vec::<f32>()
                .unwrap(),
            vec![0.0, 0.5, 1.0]
        );
    }

    #[test]
    fn apply_gates_coarse_tokens_activity() {
        let device = Device::flex();
        // seq [T=2, B=1, N=2, D=1] — all spikes: [t0n0=1, t0n1=1, t1n0=1, t1n1=1]
        let seq: SpikeSequence =
            Tensor::<1>::from_floats(vec![1.0f32; 4].as_slice(), &device).reshape([2, 1, 2, 1]);
        // importance [B=1, N=2]: token0 — coarse (0), token1 — fine (1.0)
        let imp = Tensor::<1>::from_floats(vec![0.0f32, 1.0].as_slice(), &device).reshape([1, 2]);
        let out = apply(&seq, &imp, &cfg());
        // gate = [t0n0=0, t0n1=1, t1n0=0, t1n1=1]
        assert_eq!(
            out.into_data().try_to_vec::<f32>().unwrap(),
            vec![0.0, 1.0, 0.0, 1.0]
        );
    }
}
