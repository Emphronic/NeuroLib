//! Reconstruction of the effective connectivity from the estimates.

use super::connectivity_estimator::ConnectivityEstimate;
use crate::structural::NeuronId;

/// The reconstructed effective connectivity — the estimates sorted by confidence (strongest
/// first).
#[derive(Debug, Clone)]
pub struct ReconstructedConnectivity {
    /// The estimates, sorted by `confidence` descending.
    pub estimates: Vec<ConnectivityEstimate>,
    /// The neurons in the estimation order.
    pub neurons: Vec<NeuronId>,
}

impl ReconstructedConnectivity {
    /// Build from estimates — sorted by confidence descending.
    pub fn new(mut estimates: Vec<ConnectivityEstimate>, neurons: Vec<NeuronId>) -> Self {
        estimates.sort_by(|a, b| {
            b.confidence
                .partial_cmp(&a.confidence)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        Self { estimates, neurons }
    }

    /// The estimate of the `source → target` connection, if any.
    pub fn estimate(&self, source: NeuronId, target: NeuronId) -> Option<&ConnectivityEstimate> {
        self.estimates
            .iter()
            .find(|e| e.source == source && e.target == target)
    }

    /// The estimates with `confidence ≥ min`.
    pub fn with_confidence(&self, min: f32) -> Vec<ConnectivityEstimate> {
        self.estimates
            .iter()
            .filter(|e| e.confidence >= min)
            .cloned()
            .collect()
    }

    /// The top `n` strongest estimates.
    pub fn top(&self, n: usize) -> Vec<ConnectivityEstimate> {
        self.estimates.iter().take(n).cloned().collect()
    }
}

/// Reconstruct the effective connectivity — sorts the estimates by confidence (strongest first).
pub fn reconstruct(
    estimates: Vec<ConnectivityEstimate>,
    neurons: Vec<NeuronId>,
) -> ReconstructedConnectivity {
    ReconstructedConnectivity::new(estimates, neurons)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reconstruct_sorts_and_looks_up() {
        let e1 = ConnectivityEstimate {
            source: NeuronId(0),
            target: NeuronId(1),
            estimated_weight: 0.5,
            confidence: 0.9,
        };
        let e2 = ConnectivityEstimate {
            source: NeuronId(1),
            target: NeuronId(0),
            estimated_weight: -0.3,
            confidence: 0.5,
        };
        let r = reconstruct(vec![e2.clone(), e1.clone()], vec![NeuronId(0), NeuronId(1)]);
        assert_eq!(r.estimates[0], e1); // stronger first
        assert_eq!(r.estimates[1], e2);
        assert_eq!(
            r.estimate(NeuronId(1), NeuronId(0))
                .unwrap()
                .estimated_weight,
            -0.3
        );
        assert_eq!(r.with_confidence(0.7).len(), 1);
        assert_eq!(r.top(1), vec![e1]);
        assert!(r.estimate(NeuronId(0), NeuronId(2)).is_none());
    }
}
