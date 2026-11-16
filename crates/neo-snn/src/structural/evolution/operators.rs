//! Operators of evolution: mutation and genome crossover.

use std::collections::{BTreeMap, BTreeSet};

use crate::scalar::Scalar;
use crate::state::ResetMode;

use super::super::model::{LifParams, NeuronModel, NeuronParam, NeuronPolarity, NeuronRole};
use super::super::rng::Rng64;
use super::super::topology::{ParamLimits, TopologyLimits};
use super::genome::{ConnGene, Genome, Innov, InnovationTracker, NodeGene};

/// Probabilities and extent of mutations.
#[derive(Debug, Clone, PartialEq)]
pub struct MutationConfig<T = f32> {
    /// The probability that the weights of the genome generally mutate.
    pub weight_mutate_prob: T,
    /// The share of connections whose weight is replaced by a new random one rather than shifted.
    pub weight_replace_prob: T,
    /// Weight shift scale: `w += U(-1,1)·weight_power`.
    pub weight_power: T,
    /// Weight of new connection and replacement: `U(-init_weight, init_weight)`.
    pub init_weight: T,
    /// Possibility of adding a connection.
    pub add_conn_prob: T,
    /// Probability of adding a neuron by split connection (NEAT).
    pub add_node_prob: T,
    /// Probability to enable/disable each connection.
    pub toggle_prob: T,
    /// Probability to shift the neuron parameters (β, threshold), for each hidden and output.
    pub param_mutate_prob: T,
    /// The probability of changing the model of each hidden and output neuron.
    pub model_mutate_prob: T,
    /// Model pool for `model_mutate_prob`; empty - the model does not change.
    pub model_candidates: Vec<NeuronModel<T>>,
    /// The probability of changing the polarity of each hidden and output neuron.
    pub polarity_mutate_prob: T,
    /// Shift scale β.
    pub beta_power: T,
    /// Relative scale of threshold shift.
    pub threshold_power: T,
    /// Relative scale of shift of other physical parameters
    /// (Izhikevich, AdEx, Hodgkin–Huxley and ρ/ALIF adaptations).
    pub param_power: T,
    /// The absolute minimum of the parameter step (for values around zero).
    pub param_min_step: T,
    /// Trying to find a valid pair for a new connection.
    pub add_conn_attempts: usize,
    /// Parameters of a neuron created by connection separation. By default - “relay”: β=0, threshold 0.5,
    /// reset to zero; together with the weight `split_in_weight ≥ threshold` it transmits the spikes as is
    /// (delayed by a step), so splitting hardly changes the behavior.
    pub split_params: LifParams<T>,
    /// Model of a neuron created by separation of connections. `None` - LIF with `split_params`.
    /// For example, `Some(NeuronModel::AdaptiveLif(AdaptiveLifParams::slow_adaptation()))`
    /// gives a specialized, slowly adapting node.
    pub split_model: Option<NeuronModel<T>>,
    /// Weight of `src → new neuron` when splitting.
    pub split_in_weight: T,
}

impl<T: Scalar> Default for MutationConfig<T> {
    fn default() -> Self {
        Self {
            weight_mutate_prob: T::c(0.8),
            weight_replace_prob: T::c(0.1),
            weight_power: T::c(0.3),
            init_weight: T::c(1.0),
            add_conn_prob: T::c(0.2),
            add_node_prob: T::c(0.08),
            toggle_prob: T::c(0.02),
            param_mutate_prob: T::c(0.05),
            model_mutate_prob: T::c(0.0),
            model_candidates: Vec::new(),
            polarity_mutate_prob: T::c(0.0),
            beta_power: T::c(0.05),
            threshold_power: T::c(0.1),
            param_power: T::c(0.02),
            param_min_step: T::c(1e-3),
            add_conn_attempts: 24,
            split_params: LifParams {
                beta: T::c(0.0),
                threshold: T::c(0.5),
                reset: ResetMode::Zero,
            },
            split_model: None,
            split_in_weight: T::c(1.0),
        }
    }
}

/// What the mutation did (for reports and tests).
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct MutationSummary {
    /// Added neurons.
    pub nodes_added: usize,
    /// Added links (including two from the split).
    pub conns_added: usize,
    /// Switched connections.
    pub toggled: usize,
    /// Changed weights.
    pub weights_changed: usize,
    /// Changed neurons (parameters).
    pub params_changed: usize,
    /// Changed neuron models.
    pub models_changed: usize,
    /// The polarities of neurons have been changed.
    pub polarities_changed: usize,
}

/// Mutation environment: constraints that any descendant must satisfy.
pub(crate) struct Env<'a, T = f32> {
    pub cfg: &'a MutationConfig<T>,
    pub limits: &'a TopologyLimits,
    pub plimits: &'a ParamLimits<T>,
}

/// Is `to` reachable from `from` by expressed connections (`from == to` is reachable).
fn reaches<T: Scalar>(g: &Genome<T>, from: Innov, to: Innov) -> bool {
    let mut out: BTreeMap<Innov, Vec<Innov>> = BTreeMap::new();
    for c in g.conns.iter().filter(|c| c.enabled) {
        out.entry(c.src).or_default().push(c.dst);
    }
    let mut seen = BTreeSet::new();
    let mut stack = vec![from];
    while let Some(n) = stack.pop() {
        if n == to {
            return true;
        }
        if seen.insert(n) {
            stack.extend(out.get(&n).into_iter().flatten().copied());
        }
    }
    false
}

fn degree_ok<T: Scalar>(g: &Genome<T>, limits: &TopologyLimits, src: Innov, dst: Innov) -> bool {
    let outd = g.conns.iter().filter(|c| c.enabled && c.src == src).count();
    let ind = g.conns.iter().filter(|c| c.enabled && c.dst == dst).count();
    outd < limits.max_out_degree && ind < limits.max_in_degree
}

fn clamp_model<T: Scalar>(m: &mut NeuronModel<T>, pl: &ParamLimits<T>) {
    m.clamp(pl);
}

/// The genome mutates on the spot. The result always satisfies `limits` (the number of neurons and connections,
/// degrees, loops, recurrence) and `plimits` (weights, β, threshold).
pub(crate) fn mutate<T: Scalar>(
    g: &mut Genome<T>,
    env: &Env<T>,
    tracker: &mut InnovationTracker,
    rng: &mut Rng64,
) -> MutationSummary {
    let summary = mutate_inner(g, env, tracker, rng);
    if env.limits.enforce_dale {
        g.normalize_dale();
    }
    summary
}

fn mutate_inner<T: Scalar>(
    g: &mut Genome<T>,
    env: &Env<T>,
    tracker: &mut InnovationTracker,
    rng: &mut Rng64,
) -> MutationSummary {
    let (cfg, limits, pl) = (env.cfg, env.limits, env.plimits);
    let mut s = MutationSummary::default();

    if rng.chance(cfg.weight_mutate_prob.to_f32()) {
        for c in &mut g.conns {
            let w = if rng.chance(cfg.weight_replace_prob.to_f32()) {
                rng.symmetric_as::<T>() * cfg.init_weight
            } else {
                c.weight + rng.symmetric_as::<T>() * cfg.weight_power
            };
            let w = w.clamp(-pl.weight_abs_max, pl.weight_abs_max);
            if w != c.weight {
                s.weights_changed += 1;
            }
            c.weight = w;
        }
    }

    if rng.chance(cfg.add_node_prob.to_f32())
        && g.nodes.len() < limits.max_neurons
        && g.enabled_count() + 2 <= limits.max_synapses
    {
        let candidates: Vec<usize> = g
            .conns
            .iter()
            .enumerate()
            .filter(|(_, c)| c.enabled && c.src != c.dst)
            // Dale's principle: coupling of inhibitory or modulating source
            // cannot be divided by a relay neuron - it would change its meaning.
            .filter(|(_, c)| {
                !limits.enforce_dale
                    || g.node(c.src)
                        .is_none_or(|n| n.polarity == NeuronPolarity::Excitatory)
            })
            .map(|(i, _)| i)
            .collect();
        if !candidates.is_empty() {
            let old = g.conns[candidates[rng.below(candidates.len())]].clone();
            let (mut node, mut a, mut b) = tracker.split(old.innov, old.src, old.dst);
            if g.node(node).is_some() {
                // This connection has already been divided in this genome (and the gene was then turned on again):
                // the new neuron must be different
                node = tracker.fresh();
                a = tracker.conn(old.src, node);
                b = tracker.conn(node, old.dst);
            }
            let mut model = cfg
                .split_model
                .unwrap_or(NeuronModel::Lif(cfg.split_params));
            clamp_model(&mut model, pl);
            let in_w = cfg
                .split_in_weight
                .clamp(-pl.weight_abs_max, pl.weight_abs_max);
            g.insert_node(NodeGene {
                innov: node,
                kind: NeuronRole::Hidden,
                polarity: NeuronPolarity::Excitatory,
                model,
            });
            g.insert_conn(ConnGene {
                innov: a,
                src: old.src,
                dst: node,
                weight: in_w,
                delay: old.delay.min(pl.delay_max),
                enabled: true,
            });
            g.insert_conn(ConnGene {
                innov: b,
                src: node,
                dst: old.dst,
                weight: old.weight,
                delay: 0,
                enabled: true,
            });
            if let Some(c) = g.conns.iter_mut().find(|c| c.innov == old.innov) {
                c.enabled = false;
            }
            s.nodes_added += 1;
            s.conns_added += 2;
        }
    }

    if rng.chance(cfg.add_conn_prob.to_f32()) && g.enabled_count() < limits.max_synapses {
        for _ in 0..cfg.add_conn_attempts {
            let src = g.nodes[rng.below(g.nodes.len())].innov;
            let targets: Vec<Innov> = g
                .nodes
                .iter()
                .filter(|n| n.kind != NeuronRole::Input)
                .map(|n| n.innov)
                .collect();
            if targets.is_empty() {
                break;
            }
            let dst = targets[rng.below(targets.len())];
            if g.has_conn(src, dst)
                || (src == dst && (!limits.allow_self_loops || !limits.allow_recurrent))
            {
                continue;
            }
            if !limits.allow_recurrent && reaches(g, dst, src) {
                continue;
            }
            if !degree_ok(g, limits, src, dst) {
                continue;
            }
            let innov = tracker.conn(src, dst);
            let weight = (rng.symmetric_as::<T>() * cfg.init_weight)
                .clamp(-pl.weight_abs_max, pl.weight_abs_max);
            if g.insert_conn(ConnGene {
                innov,
                src,
                dst,
                weight,
                delay: 0,
                enabled: true,
            }) {
                s.conns_added += 1;
                break;
            }
        }
    }

    if cfg.toggle_prob > T::ZERO {
        for i in 0..g.conns.len() {
            if !rng.chance(cfg.toggle_prob.to_f32()) {
                continue;
            }
            let c = g.conns[i].clone();
            if c.enabled {
                g.conns[i].enabled = false;
                s.toggled += 1;
            } else if g.enabled_count() < limits.max_synapses
                && degree_ok(g, limits, c.src, c.dst)
                && (limits.allow_recurrent || (c.src != c.dst && !reaches(g, c.dst, c.src)))
            {
                g.conns[i].enabled = true;
                s.toggled += 1;
            }
        }
    }

    if cfg.param_mutate_prob > T::ZERO {
        for n in g.nodes.iter_mut().filter(|n| n.kind != NeuronRole::Input) {
            if !rng.chance(cfg.param_mutate_prob.to_f32()) {
                continue;
            }
            let mut changed = false;
            // We go around all the physical parameters of the model, not just β/threshold:
            // so evolution covers Izhikevich, AdEx and Hodgkin–Huxley.
            for (param, value) in n.model.params() {
                let delta = match param {
                    NeuronParam::Beta => rng.symmetric_as::<T>() * cfg.beta_power,
                    NeuronParam::Threshold => value * rng.symmetric_as::<T>() * cfg.threshold_power,
                    _ => {
                        rng.symmetric_as::<T>()
                            * (value.abs() * cfg.param_power + cfg.param_min_step)
                    }
                };
                if let Ok(new) = n.model.set_param(param, value + delta, pl) {
                    changed |= new != value;
                }
            }
            if changed {
                s.params_changed += 1;
            }
        }
    }
    if cfg.model_mutate_prob > T::ZERO && !cfg.model_candidates.is_empty() {
        for n in g.nodes.iter_mut().filter(|n| n.kind != NeuronRole::Input) {
            if !rng.chance(cfg.model_mutate_prob.to_f32()) {
                continue;
            }
            let mut model = cfg.model_candidates[rng.below(cfg.model_candidates.len())];
            clamp_model(&mut model, pl);
            if model.tag() != n.model.tag() {
                n.model = model;
                s.models_changed += 1;
            }
        }
    }

    if cfg.polarity_mutate_prob > T::ZERO {
        for n in g.nodes.iter_mut().filter(|n| n.kind != NeuronRole::Input) {
            if !rng.chance(cfg.polarity_mutate_prob.to_f32()) {
                continue;
            }
            let polarity = match rng.below(3) {
                0 => NeuronPolarity::Excitatory,
                1 => NeuronPolarity::Inhibitory,
                _ => NeuronPolarity::Modulatory,
            };
            if polarity != n.polarity {
                n.polarity = polarity;
                s.polarities_changed += 1;
            }
        }
    }
    s
}

/// Disables connections that close the loop (in token order): needed when recurrence
/// is forbidden, and two acyclic parents add up to a cycle.
pub(crate) fn enforce_acyclic<T: Scalar>(g: &mut Genome<T>) {
    let mut kept = Genome {
        nodes: g.nodes.clone(),
        conns: Vec::new(),
    };
    for c in &mut g.conns {
        if c.enabled && (c.src == c.dst || reaches(&kept, c.dst, c.src)) {
            c.enabled = false;
        }
        kept.conns.push(c.clone());
    }
}

/// Crossover. Paired genes (same marker) are taken from a random parent; unpaired and
/// the extra ones are from the more adapted one (if equal, from the smaller genome, so that the complexity does not
/// grew on its own). A gene disabled in any parent is disabled in a descendant with
/// probability 0.75.
pub(crate) fn crossover<T: Scalar>(
    a: &Genome<T>,
    fit_a: f32,
    b: &Genome<T>,
    fit_b: f32,
    limits: &TopologyLimits,
    rng: &mut Rng64,
) -> Genome<T> {
    let a_is_fitter = if fit_a == fit_b {
        a.conns.len() <= b.conns.len()
    } else {
        fit_a > fit_b
    };
    let (fit, other) = if a_is_fitter { (a, b) } else { (b, a) };
    let mut child = Genome {
        nodes: Vec::new(),
        conns: Vec::new(),
    };
    for c in &fit.conns {
        match other.conn(c.innov) {
            Some(o) => {
                let mut pick = if rng.chance(0.5) {
                    c.clone()
                } else {
                    o.clone()
                };
                if !c.enabled || !o.enabled {
                    pick.enabled = !rng.chance(0.75);
                }
                child.conns.push(pick);
            }
            None => child.conns.push(c.clone()),
        }
    }
    // Neurons: inputs and outputs always; the rest are those referenced by child connections
    let mut need: BTreeSet<Innov> = fit
        .nodes
        .iter()
        .filter(|n| n.kind != NeuronRole::Hidden)
        .map(|n| n.innov)
        .collect();
    for c in &child.conns {
        need.insert(c.src);
        need.insert(c.dst);
    }
    for innov in need {
        if let Some(n) = fit.node(innov).or_else(|| other.node(innov)) {
            child.nodes.push(n.clone());
        }
    }
    if !limits.allow_recurrent {
        enforce_acyclic(&mut child);
    }
    // The descendant must comply with the limits; if the combination of genes exceeded them,
    // take the more adapted parent as is
    if fits(&child, limits) {
        child
    } else {
        fit.clone()
    }
}

/// Does the genome fit within the topology limits (number of neurons and connections, degree).
pub(crate) fn fits<T: Scalar>(g: &Genome<T>, limits: &TopologyLimits) -> bool {
    if g.nodes.len() > limits.max_neurons || g.enabled_count() > limits.max_synapses {
        return false;
    }
    let mut ind: BTreeMap<Innov, usize> = BTreeMap::new();
    let mut outd: BTreeMap<Innov, usize> = BTreeMap::new();
    for c in g.conns.iter().filter(|c| c.enabled) {
        *ind.entry(c.dst).or_default() += 1;
        *outd.entry(c.src).or_default() += 1;
    }
    ind.values().all(|d| *d <= limits.max_in_degree)
        && outd.values().all(|d| *d <= limits.max_out_degree)
}
