//! The estimator contract and the estimate type.

use super::ActivityHistory;
use crate::structural::NeuronId;

/// An estimated effective connection between two neurons —
/// [Wakhloo & Mizera, "Connectivity from activity in spiking neural networks",
/// arXiv:2611.21592](https://arxiv.org/abs/2611.21592).
#[derive(Debug, Clone, PartialEq)]
pub struct ConnectivityEstimate {
    /// Source neuron of the connection.
    pub source: NeuronId,
    /// Target neuron.
    pub target: NeuronId,
    /// Estimated effective coupling — the paper's \(V_{ij}\) scale (spikes of the target per
    /// spike of the source); the sign reflects the polarity.
    pub estimated_weight: f32,
    /// Confidence — 0..1 — the squared cosine between the cross-correlation and the target's
    /// autocorrelation (how well the linear relation fits); 0 = no correlation.
    pub confidence: f32,
}

/// Estimates the connectivity from the observable activity.
pub trait ConnectivityEstimator {
    /// Estimate the connections — one [`ConnectivityEstimate`] per ordered pair (i, j), i ≠ j,
    /// with activity.
    fn estimate(&self, activity: &ActivityHistory) -> Vec<ConnectivityEstimate>;
}
