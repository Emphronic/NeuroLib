//! Network growth policies: new synapses from co-activity and new neurons
//! (synapse splitting, cloning, creation).
//!
//! `propose_*` change nothing and return actions with reasons;
//! `grow_*` apply the proposals. Growth is not uncontrolled: every action
//! passes the topology limit checks (`check_add_synapse`, `max_neurons`),
//! and the number of actions is bounded by the `max_new` argument and the
//! controller schedule.

use std::collections::BTreeSet;

use super::error::StructuralResult;
use super::mutation::{ActionKind, Reason, StructuralAction, apply_action};
use super::network::{DynamicNetwork, SynapseModel};
use super::policy::{GrowthConfig, GrowthStrategy, NeuronGrowthConfig};
use super::rng::Rng64;
use super::topology::{NeuronId, NeuronRole, SynapseId};
use crate::scalar::Scalar;

/// Propose new synapses based on accumulated co-activity. The weight of a new
/// synapse is `init_weight·(1 + jitter·u)` and is always positive (excitatory).
/// With no co-activity tracker enabled there are no proposals.
pub fn propose_grow_synapses<T: Scalar>(
    net: &DynamicNetwork<T>,
    cfg: &GrowthConfig<T>,
    rng: &mut Rng64,
    max_new: usize,
) -> Vec<(StructuralAction<T>, Reason)> {
    let mut out = Vec::new();
    if !cfg.enabled || max_new == 0 {
        return out;
    }
    let Some(tracker) = net.tracker() else {
        return out;
    };
    let c = &cfg.synapses;
    let topo = net.topology();
    for ((src, dst), score) in tracker.top(topo, max_new * 4, c.min_score) {
        if out.len() >= max_new {
            break;
        }
        if !cfg.allow_output_to_output {
            let kind = |n| topo.neuron(n).map(|x| x.role);
            if kind(src) == Some(NeuronRole::Output) && kind(dst) == Some(NeuronRole::Output) {
                continue;
            }
        }
        if topo
            .check_add_synapse(src, dst, c.delay, None, SynapseModel::CurrentBased)
            .is_err()
        {
            continue;
        }
        let w = c.init_weight * (T::ONE + c.weight_jitter * rng.symmetric_as::<T>());
        out.push((
            StructuralAction::AddSynapse {
                src,
                dst,
                weight: w.max(T::c(1e-3)),
                delay: c.delay,
            },
            Reason::CoActivity {
                score: score.to_f32(),
            },
        ));
    }
    out
}

/// Apply the proposed synapse growth; returns the created synapses.
pub fn grow_synapses<T: Scalar>(
    net: &mut DynamicNetwork<T>,
    cfg: &GrowthConfig<T>,
    rng: &mut Rng64,
    max_new: usize,
) -> StructuralResult<Vec<SynapseId>> {
    let plan = propose_grow_synapses(net, cfg, rng, max_new);
    let mut out = Vec::new();
    for (a, _) in &plan {
        out.extend(apply_action(net, a, rng)?.synapses);
    }
    Ok(out)
}

/// Propose one new neuron. Strategies are tried in the order given by
/// `cfg.strategies`; action kinds in `skip` (paused after a rollback) are
/// skipped.
///
/// * `SplitSynapse` — the synapse with the highest transmission load;
/// * `CloneNeuron` — the most active hidden neuron with both inputs and outputs;
/// * `NewNeuron` — a new neuron with `new_neuron_fan` inputs from the most
///   active sources and outputs to the output neurons with the smallest
///   in-degree.
pub fn propose_grow_neuron<T: Scalar>(
    net: &DynamicNetwork<T>,
    cfg: &NeuronGrowthConfig<T>,
    rng: &mut Rng64,
    skip: &BTreeSet<ActionKind>,
    plateau_evaluations: usize,
) -> Option<(StructuralAction<T>, Reason)> {
    if !cfg.enabled {
        return None;
    }
    let topo = net.topology();
    if topo.neuron_count() >= topo.limits().max_neurons {
        return None;
    }
    let reason = Reason::Plateau {
        evaluations: plateau_evaluations,
    };
    for strategy in &cfg.strategies {
        match strategy {
            GrowthStrategy::SplitSynapse if !skip.contains(&ActionKind::SplitSynapse) => {
                if topo.synapse_count() + 1 > topo.limits().max_synapses {
                    continue;
                }
                let best = topo
                    .synapses()
                    .filter(|s| s.stats.uses > 0 && topo.can_split_synapse(s.id))
                    .max_by(|a, b| {
                        a.stats
                            .drive_sum
                            .partial_cmp(&b.stats.drive_sum)
                            .unwrap_or(std::cmp::Ordering::Equal)
                            .then(b.id.cmp(&a.id))
                    });
                if let Some(s) = best {
                    return Some((
                        StructuralAction::SplitSynapse {
                            id: s.id,
                            init: cfg.split,
                        },
                        reason,
                    ));
                }
            }
            GrowthStrategy::CloneNeuron if !skip.contains(&ActionKind::CloneNeuron) => {
                let best = topo
                    .neurons()
                    .filter(|n| n.role == NeuronRole::Hidden && n.stats.rate_ema > T::ZERO)
                    .filter(|n| {
                        !topo.in_synapses(n.id).is_empty() && !topo.out_synapses(n.id).is_empty()
                    })
                    .max_by(|a, b| {
                        a.stats
                            .rate_ema
                            .partial_cmp(&b.stats.rate_ema)
                            .unwrap_or(std::cmp::Ordering::Equal)
                            .then(b.id.cmp(&a.id))
                    });
                if let Some(n) = best {
                    return Some((
                        StructuralAction::CloneNeuron {
                            id: n.id,
                            jitter: cfg.clone_jitter,
                        },
                        reason,
                    ));
                }
            }
            GrowthStrategy::NewNeuron if !skip.contains(&ActionKind::AddNeuron) => {
                let fan = cfg.new_neuron_fan.max(1);
                let mut srcs: Vec<_> = topo
                    .neurons()
                    .filter(|n| n.role != NeuronRole::Output && n.stats.rate_ema > T::ZERO)
                    .map(|n| (n.id, n.stats.rate_ema))
                    .collect();
                srcs.sort_by(|a, b| {
                    b.1.partial_cmp(&a.1)
                        .unwrap_or(std::cmp::Ordering::Equal)
                        .then(a.0.cmp(&b.0))
                });
                srcs.truncate(fan);
                let mut dsts: Vec<_> = topo
                    .neurons()
                    .filter(|n| n.role == NeuronRole::Output)
                    .map(|n| (n.id, topo.in_synapses(n.id).len()))
                    .collect();
                dsts.sort_by(|a, b| a.1.cmp(&b.1).then(a.0.cmp(&b.0)));
                dsts.truncate(fan);
                if srcs.is_empty()
                    || dsts.is_empty()
                    || topo.synapse_count() + srcs.len() + dsts.len() > topo.limits().max_synapses
                {
                    continue;
                }
                let w = |rng: &mut Rng64| {
                    (cfg.new_neuron_weight * (T::ONE + T::c(0.2) * rng.symmetric_as::<T>()))
                        .max(T::c(1e-3))
                };
                let srcs: Vec<(NeuronId, T)> =
                    srcs.into_iter().map(|(id, _)| (id, w(rng))).collect();
                let dsts: Vec<(NeuronId, T)> =
                    dsts.into_iter().map(|(id, _)| (id, w(rng))).collect();
                let model = cfg
                    .new_neuron_model
                    .clone()
                    .unwrap_or_else(|| cfg.new_neuron_params.into());
                return Some((
                    StructuralAction::AddNeuron {
                        model,
                        polarity: cfg.new_neuron_polarity,
                        srcs,
                        dsts,
                    },
                    reason,
                ));
            }
            _ => {}
        }
    }
    None
}

/// Apply neuron growth (up to `max_new` neurons); returns the created neurons.
pub fn grow_neurons<T: Scalar>(
    net: &mut DynamicNetwork<T>,
    cfg: &NeuronGrowthConfig<T>,
    rng: &mut Rng64,
    max_new: usize,
) -> StructuralResult<Vec<NeuronId>> {
    let mut out = Vec::new();
    for _ in 0..max_new {
        let Some((a, _)) = propose_grow_neuron(net, cfg, rng, &BTreeSet::new(), 0) else {
            break;
        };
        out.extend(apply_action(net, &a, rng)?.neurons);
    }
    Ok(out)
}
