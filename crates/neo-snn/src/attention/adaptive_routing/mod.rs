//! Test-Time Adaptive Routing for Spiking Vision Transformers (arXiv:2611.23728, AdaS-ViT).
//!
//! The paper: usually all tokens are processed with the same temporal and firing precision.
//! In an image some areas are important, some contain background, some tokens barely
//! influence the result — why spend the same number of spikes on all?
//!
//! AdaS-ViT proposes an adaptive distribution of computation during inference:
//!
//! 1. **AdaE-IF — Adaptive Efficient Integrate-and-Fire** — the neuron picks a more precise
//!    (fine) or coarser mode of spiking activity depending on the token importance
//!    ([`adaptive_firing`]);
//! 2. **S-GSA — Spiking Gated Self-Attention** — controls the spike activity of tokens by
//!    attention-aware importance ([`token_router`]);
//! 3. **S-LGU — Spiking Local-Gated Unit** — allows controlling activity at the channel level
//!    ([`channel_router`]).
//!
//! The computational precision adapts in three directions: the time axis
//! ([`temporal_router`]), the spatial tokens, and the channels.
//!
//! Pipeline: Activity monitoring → Routing decision → Gated execution
//! ([`AdaptiveRouting::route`] + [`AdaptiveRouting::apply`]).

pub mod adaptive_firing;
pub mod channel_router;
pub mod temporal_router;
pub mod token_router;

pub use adaptive_firing::{apply as firing_apply, fire_precision, firing_gate};
pub use channel_router::{apply as channel_apply, channel_energy, channel_gate};
pub use temporal_router::{apply_temporal, route_temporal};
pub use token_router::{route_tokens, token_importance};

use burn::tensor::Tensor;

use crate::attention::SpikeSequence;

/// Adaptive routing configuration — [`AdaptiveRouting::new`].
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AdaptiveRoutingConfig {
    /// Importance ≥ this — "fine" (full-precision) firing.
    pub fine_threshold: f32,
    /// Importance < this — "coarse" mode.
    pub coarse_threshold: f32,
    /// Minimal firing precision (levels) — coarse.
    pub min_firing_precision: usize,
    /// Maximal firing precision (levels) — fine.
    pub max_firing_precision: usize,
}

impl Default for AdaptiveRoutingConfig {
    fn default() -> Self {
        Self {
            fine_threshold: 0.5,
            coarse_threshold: 0.05,
            min_firing_precision: 1,
            max_firing_precision: 8,
        }
    }
}

/// Routing decision of [`AdaptiveRouting::route`].
#[derive(Debug, Clone)]
pub struct AdaptiveRoutingOut {
    /// Token importance — `[B, N]`.
    pub importance: Tensor<2>,
    /// Fine/coarse routing per token — `[B, N]` — 1.0 — fine, 0.0 — coarse.
    pub route: Tensor<2>,
    /// Firing precision per token (levels) — `[B, N]`.
    pub precision: Tensor<2>,
    /// Per-step temporal gate — `[T]` — 1.0 — fine (active step), 0.0 — coarse (quiet step).
    pub temporal: Tensor<1>,
}

/// Test-time adaptive routing — the mechanism of the paper: the distribution of computation
/// adapts during inference.
///
/// ```text
/// Activity monitoring → Routing decision (importance → fine/coarse + precision)
///      → Gated execution (AdaE-IF + S-GSA + S-LGU)
/// ```
pub struct AdaptiveRouting {
    /// Routing configuration.
    pub config: AdaptiveRoutingConfig,
}

impl AdaptiveRouting {
    /// New routing with the config.
    pub fn new(config: AdaptiveRoutingConfig) -> Self {
        Self { config }
    }

    /// Routing decision — the token importance ([`token_importance`]), the fine/coarse route
    /// ([`route_tokens`]), the firing precision ([`fire_precision`]) and the per-step temporal
    /// gate ([`route_temporal`]).
    pub fn route(&self, seq: &SpikeSequence) -> AdaptiveRoutingOut {
        let importance = token_router::token_importance(seq);
        let route = token_router::route_tokens(&importance, &self.config);
        let precision = adaptive_firing::fire_precision(&importance, &self.config);
        let temporal = temporal_router::route_temporal(seq, &self.config);
        AdaptiveRoutingOut {
            importance,
            route,
            precision,
            temporal,
        }
    }

    /// Apply the routing to the sequence: the firing-precision gate (AdaE-IF) + the temporal
    /// gate + the channel gate (S-LGU) — `seq * gates`.
    pub fn apply(&self, seq: &SpikeSequence) -> SpikeSequence {
        let out = self.route(seq);
        let gated = adaptive_firing::apply(seq, &out.importance, &self.config);
        let gated = temporal_router::apply_temporal(&gated, &out.temporal);
        channel_router::apply(&gated)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::Device;

    #[test]
    fn apply_keeps_active_steps_and_suppresses_quiet_ones() {
        let device = Device::flex();
        // seq [T=2, B=1, N=2, D=2]:
        // t0: n0 = [2, 0], n1 = [2, 0]; t1: n0 = [0.02, 0] (a quiet step), n1 = [0, 0]
        let data = vec![2.0f32, 0.0, 2.0, 0.0, 0.02, 0.0, 0.0, 0.0];
        let seq: SpikeSequence =
            Tensor::<1>::from_floats(data.as_slice(), &device).reshape([2, 1, 2, 2]);
        let out = AdaptiveRouting::new(AdaptiveRoutingConfig::default()).apply(&seq);
        // the active step passes through unchanged; the quiet step is suppressed
        assert_eq!(
            out.into_data().try_to_vec::<f32>().unwrap(),
            vec![2.0, 0.0, 2.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        );
    }
}
