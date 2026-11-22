//! The spectral method — the paper's estimator.

use super::ActivityHistory;
use super::activity_correlation::autocorr;
use super::connectivity_estimator::{ConnectivityEstimate, ConnectivityEstimator};
use super::cross_correlation::cross_corr;

/// The paper's spectral estimator ([arXiv:2611.21592]).
///
/// The paper: "For two neurons i and j in a certain frequency regime, we get an approximation:
/// \(C_{ij}(\omega)\approx V_{ij}(\omega)C_{jj}(\omega) + C_{ii}(\omega)V_{ji}^{*}(\omega)\)" —
/// the effective connections \(V_{ij}\) are recovered from the observed cross-correlations.
///
/// This implementation computes the cross- and autocorrelation functions of the spike trains
/// and estimates each ordered pair's effective coupling as the least-squares projection of the
/// cross-correlation onto the autocorrelation of the target (the time-domain dual of the
/// paper's spectral relation); the confidence is the squared cosine between the two.
///
/// The paper's numerical results (LIF networks): 0.86 correlation between the true and
/// reconstructed excitatory connections, 0.99 for inhibitory (network 784–1000–10, T=64).
pub struct SpectralConnectivity {
    /// Max lag (steps) of the correlation functions.
    pub max_lag: usize,
    /// Epsilon added to the denominators (regularization).
    pub eps: f64,
}

impl Default for SpectralConnectivity {
    fn default() -> Self {
        Self {
            max_lag: 100,
            eps: 1e-6,
        }
    }
}

impl SpectralConnectivity {
    /// New estimator with the given max lag.
    pub fn new(max_lag: usize) -> Self {
        Self {
            max_lag,
            ..Default::default()
        }
    }
}

impl ConnectivityEstimator for SpectralConnectivity {
    fn estimate(&self, activity: &ActivityHistory) -> Vec<ConnectivityEstimate> {
        let n = activity.neurons();
        if n == 0 {
            return Vec::new();
        }
        let trains: Vec<Vec<u64>> = (0..n).map(|i| activity.spike_train(i)).collect();
        let autos: Vec<Vec<f64>> = trains.iter().map(|t| autocorr(t, self.max_lag)).collect();
        let mut out = Vec::with_capacity(n * (n - 1));
        for i in 0..n {
            for j in 0..n {
                if i == j {
                    continue;
                }
                let cross = cross_corr(&trains[i], &trains[j], self.max_lag);
                let num: f64 = cross.iter().zip(autos[j].iter()).map(|(c, a)| c * a).sum();
                let den: f64 = autos[j].iter().map(|a| a * a).sum::<f64>() + self.eps;
                let v = num / den;
                let cross_pow: f64 = cross.iter().map(|c| c * c).sum::<f64>().max(self.eps);
                let conf = num.abs() / (cross_pow.sqrt() * den.sqrt());
                out.push(ConnectivityEstimate {
                    source: activity.neuron_id(i),
                    target: activity.neuron_id(j),
                    estimated_weight: v as f32,
                    confidence: conf as f32,
                });
            }
        }
        // Strongest first.
        out.sort_by(|a, b| {
            b.confidence
                .partial_cmp(&a.confidence)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::structural::NeuronId;
    use crate::structural::analysis::ActivityHistory;

    fn history(n: usize, steps: Vec<Vec<bool>>) -> ActivityHistory {
        ActivityHistory {
            ids: (0..n).map(|i| NeuronId(i as u64)).collect(),
            steps,
            start_step: 0,
        }
    }

    #[test]
    fn estimates_all_pairs_sorted_by_confidence() {
        let h = history(
            3,
            vec![
                vec![true, false, true],
                vec![false, true, false],
                vec![true, true, false],
            ],
        );
        let ests = SpectralConnectivity::default().estimate(&h);
        assert_eq!(ests.len(), 6);
        assert!(ests.iter().all(|e| e.source != e.target));
        assert!(ests.windows(2).all(|w| w[0].confidence >= w[1].confidence));
        assert!(ests.iter().all(|e| (0.0..=1.0).contains(&e.confidence)));
    }

    #[test]
    fn silent_neurons_give_zero_estimates() {
        let h = history(
            2,
            vec![vec![false, false], vec![false, false], vec![false, false]],
        );
        let ests = SpectralConnectivity::default().estimate(&h);
        assert_eq!(ests.len(), 2);
        assert!(
            ests.iter()
                .all(|e| e.estimated_weight == 0.0 && e.confidence == 0.0)
        );
    }

    #[test]
    fn repeatable() {
        let h = history(
            3,
            vec![
                vec![true, false, true],
                vec![false, true, false],
                vec![true, true, false],
            ],
        );
        assert_eq!(
            SpectralConnectivity::new(50).estimate(&h),
            SpectralConnectivity::new(50).estimate(&h)
        );
    }

    #[test]
    fn nonzero_confidence_when_spikes_together() {
        // Neurons 0 and 1 spike together on all 10 steps.
        let steps: Vec<Vec<bool>> = (0..10).map(|_| vec![true, true]).collect();
        let h = history(2, steps);
        let ests = SpectralConnectivity::default().estimate(&h);
        assert!(ests.iter().any(|e| e.confidence > 0.0));
    }
}
