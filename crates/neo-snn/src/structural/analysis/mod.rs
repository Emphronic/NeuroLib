//! Connectivity from activity — the analysis of the network from its observable spike activity
//! (the first stage of the loop:
//! `Activity monitoring → Connectivity estimation → Candidate connections → Causal validation → Rewire/prune/strengthen`).
//!
//! Reference: [Wakhloo & Mizera, "Connectivity from activity in spiking neural networks",
//! arXiv:2611.21592](https://arxiv.org/abs/2611.21592) — a theoretical work with mathematical
//! derivation and a numerical check on LIF networks (preprint).
//!
//! # The method of the paper
//!
//! For two neurons i and j in a frequency regime, the paper gets the relation between the
//! cross-spectral density and the effective connections:
//!
//! \[
//! C_{ij}(\omega)\approx V_{ij}(\omega)C_{jj}(\omega)+C_{ii}(\omega)V_{ji}^{*}(\omega)
//! \]
//!
//! The effective connections \(V_{ij}\) are recovered from the observed cross-correlations —
//! the second term (back-coupling through the network) is included because the relation is the
//! linear response of the network — indirect paths through other neurons are accounted for.
//! This implementation computes the cross- and autocorrelation functions of the spike trains
//! ([`activity_correlation`], [`cross_correlation`]) and estimates each pair's effective coupling
//! as the least-squares projection of the cross-correlation onto the autocorrelation of the
//! target ([`spectral_connectivity`]) — the time-domain dual of the paper's spectral relation —
//! with a confidence — the squared cosine between the two (1 when the cross-correlation is a
//! multiple of the target's autocorrelation).
//!
//! The paper's numerical results: correlation 0.86 between the true and reconstructed
//! excitatory-connection strengths, 0.99 for inhibitory ones in an LIF network
//! (network 784–1000–10, T=64).
//!
//! # Activity monitoring
//!
//! [`ActivityHistory`] records the spikes of all neurons — [`DynamicNetwork::last_spikes`]
//! ([`ActivityHistory::record_step`] per step). [`SpectralConnectivity`] is the estimator —
//! implement [`ConnectivityEstimator`] for your own method.
//! [`causal_validation::validate_causal`] — the "Causal validation" stage: the influence of a
//! connection on the network behavior when it is temporarily disabled.

pub mod activity_correlation;
pub mod causal_validation;
pub mod connectivity_estimator;
pub mod cross_correlation;
pub mod reconstruction;
pub mod spectral_connectivity;

pub use activity_correlation::autocorr;
pub use causal_validation::{CausalValidation, validate_causal};
pub use connectivity_estimator::{ConnectivityEstimate, ConnectivityEstimator};
pub use reconstruction::{ReconstructedConnectivity, reconstruct};
pub use spectral_connectivity::SpectralConnectivity;

use super::{DynamicNetwork, NeuronId};
use crate::scalar::Scalar;

/// Recorded spike activity: one bit per neuron per step — the input of the connectivity estimation.
#[derive(Debug, Clone, Default)]
pub struct ActivityHistory {
    /// `ids[i]` — the id of the neuron whose bit is `steps[t][i]`
    /// (order — [`Topology::neurons`](super::Topology::neurons)).
    pub ids: Vec<NeuronId>,
    /// `steps[t][i]` — whether neuron `i` spiked at step `t` of the recording.
    pub steps: Vec<Vec<bool>>,
    /// Step index the recording starts from (`net.step_count()` at the start).
    pub start_step: u64,
}

impl ActivityHistory {
    /// New history — `ids` — the neuron ids in spike order; `start_step` — the step index of
    /// the first recorded step.
    pub fn new(ids: Vec<NeuronId>, start_step: u64) -> Self {
        Self {
            ids,
            steps: Vec::new(),
            start_step,
        }
    }

    /// New history initialized from the network — the neuron ids in
    /// [`Topology::neurons`](super::Topology::neurons) order, the current step count as the start.
    pub fn from_network<T: Scalar>(net: &DynamicNetwork<T>) -> Self {
        Self::new(
            net.topology().neurons().map(|n| n.id).collect(),
            net.step_count(),
        )
    }

    /// Record the spikes of the last network step ([`DynamicNetwork::last_spikes`]).
    pub fn record_step<T: Scalar>(&mut self, net: &DynamicNetwork<T>) {
        self.steps.push(net.last_spikes().to_vec());
    }

    /// Record raw bits — one bit per neuron, in `ids` order.
    pub fn record(&mut self, spikes: &[bool]) {
        self.steps.push(spikes.to_vec());
    }

    /// Number of recorded steps.
    pub fn len(&self) -> usize {
        self.steps.len()
    }

    /// Whether the history is empty.
    pub fn is_empty(&self) -> bool {
        self.steps.is_empty()
    }

    /// Number of neurons (see `ids` and the recorded bits).
    pub fn neurons(&self) -> usize {
        self.ids
            .len()
            .max(self.steps.first().map_or(0, |s| s.len()))
    }

    /// The id of neuron `i` (the index into `ids`; falls back to the index itself).
    pub fn neuron_id(&self, i: usize) -> NeuronId {
        self.ids.get(i).copied().unwrap_or(NeuronId(i as u64))
    }

    /// Spike train of neuron `i` — the sample indices of its spikes (the time base is
    /// `start_step + t`).
    pub fn spike_train(&self, i: usize) -> Vec<u64> {
        self.steps
            .iter()
            .enumerate()
            .filter(|(_, s)| s.get(i).copied().unwrap_or(false))
            .map(|(t, _)| self.start_step + t as u64)
            .collect()
    }
}
