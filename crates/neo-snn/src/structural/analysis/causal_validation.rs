//! Causal validation — the influence of a connection on the network behavior when it is
//! temporarily disabled (the last stage of the pipeline:
//! `… → Causal validation → …`).

use super::connectivity_estimator::ConnectivityEstimate;
use crate::scalar::Scalar;
use crate::structural::{DynamicNetwork, StructuralResult};

/// The measured influence of a candidate connection on the network behavior.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct CausalValidation {
    /// Mean over episodes and steps of `|spikes(with) − spikes(without)|` — the absolute change
    /// of output spikes when the connection is disabled.
    pub abs_output_change: f32,
    /// Mean output spike rate with the connection present (per step).
    pub rate_with: f32,
    /// Mean output spike rate with the connection disabled (per step).
    pub rate_without: f32,
}

impl CausalValidation {
    /// Whether disabling the connection has a measurable effect
    /// (`abs_output_change > 0.5`).
    pub fn has_effect(&self) -> bool {
        self.abs_output_change > 0.5
    }
}

/// Run the same steps from the same inputs (one episode per call).
fn run_steps<T: Scalar>(
    net: &mut DynamicNetwork<T>,
    steps: &[Vec<T>],
) -> StructuralResult<Vec<Vec<bool>>> {
    steps
        .iter()
        .map(|x| net.step(x).map(|s| s.output_spikes))
        .collect()
}

/// Temporarily disable the candidate connection (`source → target`), run the same steps again,
/// re-enable, and measure the behavior change: [`CausalValidation`].
///
/// If the connection does not exist, `Ok(CausalValidation::default())` is returned
/// (there is nothing to disable).
pub fn validate_causal<T: Scalar>(
    net: &mut DynamicNetwork<T>,
    candidate: &ConnectivityEstimate,
    steps: &[Vec<T>],
    episodes: usize,
) -> StructuralResult<CausalValidation> {
    let Some(sid) = net
        .topology()
        .find_synapse(candidate.source, candidate.target)
    else {
        return Ok(CausalValidation::default());
    };
    let step = net.step_count();
    let mut with = Vec::new();
    for _ in 0..episodes.max(1) {
        net.reset_state();
        with.extend(run_steps(net, steps)?);
    }
    net.topology_mut().disable_synapse(sid, step)?;
    let mut without = Vec::new();
    for _ in 0..episodes.max(1) {
        net.reset_state();
        without.extend(run_steps(net, steps)?);
    }
    net.topology_mut().enable_synapse(sid)?;
    Ok(compare(&with, &without))
}

fn compare(with: &[Vec<bool>], without: &[Vec<bool>]) -> CausalValidation {
    let mut change = 0.0f32;
    let mut rate_with = 0.0f32;
    let mut rate_without = 0.0f32;
    let mut n = 0.0f32;
    for (a, b) in with.iter().zip(without) {
        for (&u, &v) in a.iter().zip(b) {
            change += (u as i32 - v as i32).abs() as f32;
            rate_with += u as i32 as f32;
            rate_without += v as i32 as f32;
            n += 1.0;
        }
    }
    CausalValidation {
        abs_output_change: change / n.max(1.0),
        rate_with: rate_with / n.max(1.0),
        rate_without: rate_without / n.max(1.0),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::structural::analysis::ConnectivityEstimate;
    use crate::structural::{
        Birth, EngineConfig, LifParams, NeuronId, NeuronRole, Origin, StructuralConfig, Topology,
    };

    fn net() -> DynamicNetwork<f32> {
        let cfg = StructuralConfig::default();
        let mut topo = Topology::new(cfg.topology, cfg.params);
        let b = Birth::default();
        let a = topo
            .add_neuron(NeuronRole::Input, LifParams::default(), Origin::Initial, b)
            .unwrap();
        let o = topo
            .add_neuron(
                NeuronRole::Output,
                LifParams::new(0.9, 1.0),
                Origin::Initial,
                b,
            )
            .unwrap();
        topo.add_synapse(a, o, 1.2, 0, Origin::Initial, b).unwrap();
        DynamicNetwork::new(topo, EngineConfig::default()).unwrap()
    }

    #[test]
    fn missing_candidate_is_default() {
        let mut net = net();
        let cand = ConnectivityEstimate {
            source: NeuronId(0),
            target: NeuronId(1),
            estimated_weight: 1.0,
            confidence: 1.0,
        };
        let v = validate_causal(&mut net, &cand, &[vec![1.0]], 1).unwrap();
        assert_eq!(v, CausalValidation::default());
        assert!(!v.has_effect());
    }

    #[test]
    fn existing_candidate_measures_behavior() {
        let mut net = net();
        let a = net.topology().inputs()[0];
        let o = net.topology().outputs()[0];
        let cand = ConnectivityEstimate {
            source: a,
            target: o,
            estimated_weight: 1.0,
            confidence: 1.0,
        };
        let steps: Vec<Vec<f32>> = (0..6)
            .map(|t| if t % 2 == 0 { vec![1.0] } else { vec![0.0] })
            .collect();
        let v = validate_causal(&mut net, &cand, &steps, 2).unwrap();
        assert!(v.rate_with >= 0.0);
        assert!(v.rate_without >= 0.0);
        assert_eq!(v.has_effect(), v.abs_output_change > 0.5);
    }
}
