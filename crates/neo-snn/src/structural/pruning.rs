//! Utility estimation and safe network pruning.
//!
//! Principles (see ToR, section 9):
//!
//! * do not remove due to low weight or low activity per se: synapse
//!   estimated by **frequency of use** (how many spikes are actually
//!   delivered per step, relative to the network median - weight in the estimate
//!   not involved), in terms of sample coverage and contribution to the quality signal
//!   (`credit_ema`);
//! * item is only rated after minimum age **and** minimum
//!   observed samples, then no more often than `reeval_interval_steps`;
//! * two-stage removal: first `DisableSynapse`, and only after
//!   `grace_steps` - physical `RemoveSynapse`;
//! * protection: labeled `protected`, synapses with positive contribution and “rare”
//!   useful" synapses (low coverage, but positive `credit_ema`);
//! * minimums: active synapses in the network, incoming degree of output neurons,
//!   number of hidden neurons.
//!
//! The `propose_*` functions do not change anything; `prune_*` - apply suggestions.

use std::collections::BTreeMap;

use super::error::StructuralResult;
use super::mutation::{Reason, StructuralAction, apply_action};
use super::network::DynamicNetwork;
use super::policy::PruningConfig;
use super::rng::Rng64;
use super::topology::{NeuronId, NeuronRole, Synapse, SynapseId, SynapseState};
use crate::scalar::Scalar;

/// Evaluation of a single synapse.
#[derive(Debug, Clone, PartialEq)]
pub struct SynapseVerdict<T = f32> {
    /// Synapse.
    pub id: SynapseId,
    /// Has reached the age and scope for assessment.
    pub evaluable: bool,
    /// Relative frequency of use (to the median across mature synapses).
    pub utility: T,
    /// Proportion of samples in which the synapse transmitted spikes.
    pub coverage: T,
    /// Evaluation of the contribution to the quality signal.
    pub credit: T,
    /// Is it protected from deletion?
    pub protected: bool,
    /// Candidate for shutdown.
    pub candidate: bool,
}

fn median<T: Scalar>(mut v: Vec<T>) -> T {
    if v.is_empty() {
        return T::ZERO;
    }
    v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
    v[v.len() / 2]
}

/// Evaluate all active synapses (no network changes).
pub fn synapse_verdicts<T: Scalar>(
    net: &DynamicNetwork<T>,
    cfg: &PruningConfig<T>,
) -> Vec<SynapseVerdict<T>> {
    let step = net.step_count();
    let samples = net.sample_count();
    let topo = net.topology();
    let rate = |s: &Synapse<T>| {
        T::from_f64(s.stats.uses as f64)
            / T::from_f64(step.saturating_sub(s.meta.born_step).max(1) as f64)
    };
    let mature: Vec<T> = topo
        .synapses()
        .filter(|s| s.is_active() && step.saturating_sub(s.meta.born_step) >= cfg.min_age_steps)
        .map(rate)
        .collect();
    let med = median(mature);
    topo.synapses()
        .filter(|s| s.is_active())
        .map(|s| {
            let age = step.saturating_sub(s.meta.born_step);
            let observed = samples.saturating_sub(s.meta.born_sample);
            let due = step.saturating_sub(s.meta.last_evaluated_step) >= cfg.reeval_interval_steps;
            let evaluable = age >= cfg.min_age_steps && observed >= cfg.min_observed_samples && due;
            let coverage =
                T::from_f64(s.stats.active_samples as f64) / T::from_f64(observed.max(1) as f64);
            let credit = s.stats.credit_ema;
            let utility = rate(s) / (med + T::c(1e-9));
            let protected = s.meta.protected
                || credit > cfg.protect_credit
                || (s.stats.uses > 0 && coverage < cfg.rare_coverage && credit > T::ZERO);
            let candidate =
                evaluable && !protected && (s.stats.uses == 0 || utility < cfg.drive_rel_threshold);
            SynapseVerdict {
                id: s.id,
                evaluable,
                utility,
                coverage,
                credit,
                protected,
                candidate,
            }
        })
        .collect()
}

/// Mark synapses as scored (starts counting `reeval_interval_steps`).
pub fn mark_evaluated<T: Scalar>(net: &mut DynamicNetwork<T>, verdicts: &[SynapseVerdict<T>]) {
    let step = net.step_count();
    for v in verdicts.iter().filter(|v| v.evaluable) {
        if let Some(s) = net.topology_mut().synapse_mut(v.id) {
            s.meta.last_evaluated_step = step;
        }
    }
}

/// Suggest reduction of synapses: first, physical removal of disabled ones,
/// spent `grace_steps`, then disabling the worst candidates.
pub fn propose_prune_synapses<T: Scalar>(
    net: &DynamicNetwork<T>,
    cfg: &PruningConfig<T>,
    verdicts: &[SynapseVerdict<T>],
    max_ops: usize,
) -> Vec<(StructuralAction<T>, Reason)> {
    let mut out: Vec<(StructuralAction<T>, Reason)> = Vec::new();
    if !cfg.enabled || max_ops == 0 {
        return out;
    }
    let step = net.step_count();
    let topo = net.topology();
    for s in topo.synapses() {
        if let SynapseState::Disabled { since_step } = s.state
            && step.saturating_sub(since_step) >= cfg.grace_steps
            && out.len() < max_ops
        {
            out.push((
                StructuralAction::RemoveSynapse { id: s.id },
                Reason::GraceElapsed,
            ));
        }
    }
    let active = topo.active_synapse_count();
    let budget = ((cfg.max_fraction_per_interval * T::from_usize(active))
        .to_f64()
        .ceil() as usize)
        .min(max_ops.saturating_sub(out.len()))
        .min(active.saturating_sub(cfg.min_synapses));
    if budget == 0 {
        return out;
    }
    let mut cand: Vec<&SynapseVerdict<T>> = verdicts.iter().filter(|v| v.candidate).collect();
    cand.sort_by(|a, b| {
        a.utility
            .partial_cmp(&b.utility)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.id.cmp(&b.id))
    });
    let mut in_left: BTreeMap<NeuronId, usize> = BTreeMap::new();
    for n in topo.neurons().filter(|n| n.role == NeuronRole::Output) {
        let c = topo
            .in_synapses(n.id)
            .iter()
            .filter(|s| topo.synapse(**s).is_some_and(|x| x.is_active()))
            .count();
        in_left.insert(n.id, c);
    }
    let mut taken = 0;
    for v in cand {
        if taken >= budget {
            break;
        }
        let Some(s) = topo.synapse(v.id) else {
            continue;
        };
        if let Some(left) = in_left.get_mut(&s.dst) {
            if *left <= cfg.min_output_in_degree {
                continue;
            }
            *left -= 1;
        }
        out.push((
            StructuralAction::DisableSynapse { id: v.id },
            Reason::LowUtility {
                utility: v.utility.to_f32(),
            },
        ));
        taken += 1;
    }
    out
}

/// Suggest removal of neurons: “dead” (no active inputs or outputs)
/// and silent without contribution. A minimum of hidden neurons is maintained.
pub fn propose_prune_neurons<T: Scalar>(
    net: &DynamicNetwork<T>,
    cfg: &PruningConfig<T>,
) -> Vec<(StructuralAction<T>, Reason)> {
    let mut out = Vec::new();
    if !cfg.enabled || cfg.max_neuron_prunes == 0 {
        return out;
    }
    let step = net.step_count();
    let samples = net.sample_count();
    let topo = net.topology();
    let mut hidden_left = topo.hidden().len();
    for n in topo.neurons().filter(|n| n.role == NeuronRole::Hidden) {
        if out.len() >= cfg.max_neuron_prunes || hidden_left <= cfg.min_hidden {
            break;
        }
        if n.meta.protected {
            continue;
        }
        let age = step.saturating_sub(n.meta.born_step);
        let observed = samples.saturating_sub(n.meta.born_sample);
        if age < cfg.min_age_steps || observed < cfg.min_observed_samples {
            continue;
        }
        let active = |ids: &[SynapseId], incoming: bool| -> usize {
            ids.iter()
                .filter_map(|s| topo.synapse(*s))
                .filter(|s| s.is_active() && s.src != s.dst)
                .filter(|s| {
                    if incoming {
                        s.src != n.id
                    } else {
                        s.dst != n.id
                    }
                })
                .count()
        };
        let (ins, outs) = (
            active(topo.in_synapses(n.id), true),
            active(topo.out_synapses(n.id), false),
        );
        if cfg.prune_dead_neurons && (ins == 0 || outs == 0) {
            out.push((
                StructuralAction::RemoveNeuron { id: n.id },
                Reason::DeadNeuron,
            ));
            hidden_left -= 1;
            continue;
        }
        if cfg.prune_silent_neurons {
            let coverage =
                T::from_f64(n.stats.active_samples as f64) / T::from_f64(observed.max(1) as f64);
            let out_credit = topo
                .out_synapses(n.id)
                .iter()
                .filter_map(|s| topo.synapse(*s))
                .map(|s| s.stats.credit_ema.max(T::ZERO))
                .fold(T::ZERO, |a, b| a + b);
            if n.stats.rate_ema < cfg.neuron_min_rate
                && coverage < cfg.neuron_min_coverage
                && out_credit <= cfg.protect_credit
            {
                out.push((
                    StructuralAction::RemoveNeuron { id: n.id },
                    Reason::SilentNeuron {
                        rate: n.stats.rate_ema.to_f32(),
                    },
                ));
                hidden_left -= 1;
            }
        }
    }
    out
}

/// Evaluate and disable/remove synapses according to the rules `cfg` (applies
/// proposals). Returns the identifiers of those on which the action was triggered.
pub fn prune_synapses<T: Scalar>(
    net: &mut DynamicNetwork<T>,
    cfg: &PruningConfig<T>,
    max_ops: usize,
) -> StructuralResult<Vec<SynapseId>> {
    let verdicts = synapse_verdicts(net, cfg);
    let plan = propose_prune_synapses(net, cfg, &verdicts, max_ops);
    let mut rng = Rng64::new(0);
    let mut done = Vec::new();
    for (a, _) in &plan {
        apply_action(net, a, &mut rng)?;
        match a {
            StructuralAction::DisableSynapse { id } | StructuralAction::RemoveSynapse { id } => {
                done.push(*id)
            }
            _ => {}
        }
    }
    mark_evaluated(net, &verdicts);
    Ok(done)
}

/// Remove neurons deemed unnecessary (applies suggestions).
pub fn prune_neurons<T: Scalar>(
    net: &mut DynamicNetwork<T>,
    cfg: &PruningConfig<T>,
) -> StructuralResult<Vec<NeuronId>> {
    let plan = propose_prune_neurons(net, cfg);
    let mut rng = Rng64::new(0);
    let mut done = Vec::new();
    for (a, _) in &plan {
        apply_action(net, a, &mut rng)?;
        if let StructuralAction::RemoveNeuron { id } = a {
            done.push(*id);
        }
    }
    Ok(done)
}
