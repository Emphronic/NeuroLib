//! Token router — per-token importance and routing decisions.

use burn::tensor::Tensor;

use super::AdaptiveRoutingConfig;
use crate::attention::SpikeSequence;

/// Token importance of a spike sequence — per-token spike energy: the sum of the token's
/// spikes over the time and channel axes — `[B, N]` (the sequence is `[T, B, N, D]`).
pub fn token_importance(seq: &SpikeSequence) -> Tensor<2> {
    seq.clone().sum_dims_squeeze::<2, _>(&[0, 3])
}

/// Routing decision per token — 1.0 — "fine" (importance ≥ `fine_threshold`),
/// 0.0 — "coarse" — via [`AdaptiveRoutingConfig::fine_threshold`].
pub fn route_tokens(importance: &Tensor<2>, cfg: &AdaptiveRoutingConfig) -> Tensor<2> {
    importance
        .clone()
        .greater_equal_scalar(cfg.fine_threshold)
        .float()
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::Device;

    #[test]
    fn token_importance_sums_over_time_and_channels() {
        // seq [T=2, B=1, N=2, D=2]: t0: token0=[1,0], token1=[0,1]; t1: token0=[1,0], token1=[0,0]
        let device = Device::flex();
        let data = vec![1.0f32, 0.0, 0.0, 1.0, 1.0, 0.0, 0.0, 0.0];
        let seq: SpikeSequence =
            Tensor::<1>::from_floats(data.as_slice(), &device).reshape([2, 1, 2, 2]);
        // token0: t0(1) + t1(1) = 2; token1: t0(1) + 0 = 1
        assert_eq!(
            token_importance(&seq)
                .into_data()
                .try_to_vec::<f32>()
                .unwrap(),
            vec![2.0, 1.0]
        );
    }

    #[test]
    fn route_tokens_is_fine_above_threshold() {
        let device = Device::flex();
        let imp = Tensor::<1>::from_floats(vec![0.9f32, 0.01].as_slice(), &device).reshape([1, 2]);
        let cfg = AdaptiveRoutingConfig {
            fine_threshold: 0.5,
            coarse_threshold: 0.05,
            ..AdaptiveRoutingConfig::default()
        };
        assert_eq!(
            route_tokens(&imp, &cfg)
                .into_data()
                .try_to_vec::<f32>()
                .unwrap(),
            vec![1.0, 0.0]
        );
    }
}
