//! `StructuralPlasticityController` - automatic growth decisions,
//! reduction and reconfiguration of the structure.
//!
//! # How to use
//!
//! Call [`StructuralPlasticityController::adapt`] **on secure
//! boundary** (end of sample, minibatch or epoch), passing quality metrics
//! (`loss` - less is better; for reinforcement learning - `reward`, for
//! self-learning - an explicitly specified goal, for example, the error of predicting the next
//! events). The topology does not change between boundaries.
//!
//! # What is taken into account
//!
//! 1. **Problem quality** — `Metrics::task_loss`; the plateau is determined by the number
//!    consecutive ratings without improvement by `plateau_min_delta`.
//! 2. **Activity**—neuron spike frequencies (silence/overload).
//! 3. **Utility of connections** - transmission, sample coverage, contribution to signal
//!    (`credit_ema`) - via `pruning`.
//! 4. **Homeostasis** - a shift in the threshold of neurons with a frequency out of range.
//! 5. **Cost** - [`StructureCost`] (default `LinearCost`) and budget
//!    memory `max_state_bytes`, upon reaching which growth is blocked.
//! 6. **History** - counters of outcomes by type of operation; after rollback type
//!    The operation pauses (doubling with repeated rollbacks).
//!
//! # Check and rollback
//!
//! If `eval` is passed, quality is measured before and after changes on copies
//! networks ([`DynamicNetwork::probe_copy`]; plasticity is off, original network
//! is not affected), and deterioration above the tolerance rolls back the entire change. Without
//! `eval` the change is checked later, using training metrics: through
//! `verify_after_calls` calls `adapt` loss is compared with the loss at the moment
//! changes, and if the topology deteriorates, it returns to the saved snapshot.
//! Additionally, a snapshot of the best cost-effective topology is stored
//! ([`StructuralPlasticityController::restore_best`]).

use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::fmt;

use super::error::{StructuralError, StructuralResult};
use super::growth::{propose_grow_neuron, propose_grow_synapses};
use super::model_selection::{propose_model_switches, propose_model_switches_evaluated};
use super::mutation::{
    ActionKind, EventLog, Outcome, Reason, StructuralAction, StructuralEvent, apply_action,
};
use super::network::DynamicNetwork;
use super::neuron_ops::NeuronMutation;
use super::param_learning::propose_param_updates;
use super::policy::StructuralConfig;
use super::pruning::{
    mark_evaluated, propose_prune_neurons, propose_prune_synapses, synapse_verdicts,
};
use super::rng::Rng64;
use super::stats::{LinearCost, StructuralStats, StructureCost};
use super::topology::{NeuronId, NeuronRole, SynapseId, Topology};
use crate::scalar::Scalar;

/// Quality metrics at the learning frontier.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Metrics<T = f32> {
    /// Loss function or self-learning goal error (less is better).
    pub task_loss: T,
    /// Remuneration (for RL), if any.
    pub reward: Option<T>,
    /// Accuracy, if any.
    pub accuracy: Option<T>,
}

impl<T: Scalar> Metrics<T> {
    /// Metrics with loss only.
    pub fn loss(task_loss: T) -> Self {
        Self {
            task_loss,
            reward: None,
            accuracy: None,
        }
    }
}

/// Why the analysis did not lead to changes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SkipReason {
    /// Metrics are non-numeric.
    InvalidMetrics,
    /// Not another call from `analyze_every`.
    NotDue,
    /// Too early after the previous change.
    TooSoon,
    /// No offers.
    NothingToDo,
    /// The previous change is still being checked (deferred check): new ones
    /// no structural changes are made until its completion.
    Verifying,
}

/// Call report `adapt`.
#[derive(Debug, Clone, PartialEq)]
pub struct AdaptReport<T = f32> {
    /// Reason for lack of change.
    pub skipped: Option<SkipReason>,
    /// Operations applied (and saved).
    pub applied: usize,
    /// Failed to apply.
    pub failed: usize,
    /// Whether the change was rolled back (now or by a deferred check).
    pub rolled_back: bool,
    /// Loss before change (if measured).
    pub loss_before: Option<T>,
    /// Loss after change (if measured).
    pub loss_after: Option<T>,
    /// Events of this call.
    pub events: Vec<StructuralEvent>,
    /// Structural indicators at the time of analysis.
    pub stats: StructuralStats<T>,
}

/// A snapshot of the best cost-effective topology.
#[derive(Debug, Clone)]
pub struct BestSnapshot<T = f32> {
    /// Topology.
    pub topology: Topology<T>,
    /// Full cost `J` at the time of the photo.
    pub cost: T,
    /// Loss at the time of the snapshot.
    pub loss: T,
    /// Simulation step.
    pub step: u64,
}

impl<T: Scalar> PartialEq for BestSnapshot<T> {
    fn eq(&self, other: &Self) -> bool {
        self.topology == other.topology
            && self.cost == other.cost
            && self.loss == other.loss
            && self.step == other.step
    }
}

#[derive(Debug, Clone)]
pub(super) struct Pending<T = f32> {
    pub(super) backup: Topology<T>,
    pub(super) loss_before: T,
    pub(super) cost_before: T,
    pub(super) calls_left: u64,
    pub(super) seqs: Vec<(u64, ActionKind)>,
}

/// Controller of structural plasticity.
pub struct StructuralPlasticityController<T = f32> {
    pub(super) cfg: StructuralConfig<T>,
    pub(super) rng: Rng64,
    pub(super) log: EventLog,
    pub(super) losses: VecDeque<T>,
    pub(super) best_loss: T,
    pub(super) since_improvement: usize,
    pub(super) calls: u64,
    pub(super) last_change_step: Option<u64>,
    pub(super) cooldown: BTreeMap<ActionKind, u64>,
    pub(super) streak: BTreeMap<ActionKind, u32>,
    pub(super) pending: Option<Pending<T>>,
    pub(super) best: Option<BestSnapshot<T>>,
    cost: Box<dyn StructureCost<T> + Send>,
}

impl<T: Scalar> fmt::Debug for StructuralPlasticityController<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("StructuralPlasticityController")
            .field("calls", &self.calls)
            .field("since_improvement", &self.since_improvement)
            .field("events", &self.log.len())
            .finish_non_exhaustive()
    }
}

/// Measure the quality of a network copy.
fn measure<T: Scalar>(
    eval: &mut Option<&mut dyn FnMut(&mut DynamicNetwork<T>) -> T>,
    net: &DynamicNetwork<T>,
) -> Option<T> {
    let f = eval.as_mut()?;
    let mut probe = net.probe_copy();
    let v = (**f)(&mut probe);
    v.is_finite().then_some(v)
}

impl<T: Scalar> StructuralPlasticityController<T> {
    /// Create a controller; the configuration is checked.
    pub fn new(cfg: StructuralConfig<T>) -> StructuralResult<Self> {
        cfg.validate()?;
        let cost = cfg.cost;
        Ok(Self {
            rng: Rng64::new(cfg.seed),
            log: EventLog::new(cfg.log_capacity),
            losses: VecDeque::new(),
            best_loss: T::from_f64(f64::INFINITY),
            since_improvement: 0,
            calls: 0,
            last_change_step: None,
            cooldown: BTreeMap::new(),
            streak: BTreeMap::new(),
            pending: None,
            best: None,
            cost: Box::new(cost),
            cfg,
        })
    }

    /// Replace the cost criterion (for example, with a combination of your own fines).
    pub fn with_cost(mut self, cost: Box<dyn StructureCost<T> + Send>) -> Self {
        self.cost = cost;
        self
    }

    /// Switch the controller to E/I mode (Dale principle): turns on the flag in
    /// configurations and gives the signs of the weights in the topologies that the controller
    /// stores for rollback (`pending`-test and best snapshot). Without this, a rollback
    /// would return the network to legacy signs. Returns the number of synapses modified.
    /// The network itself is converted by [`Checkpoint::enable_dale`](super::Checkpoint::enable_dale)
    /// or `Topology::set_enforce_dale(true)`.
    pub fn enable_dale(&mut self) -> usize {
        self.cfg.topology.enforce_dale = true;
        let mut changed = 0;
        if let Some(p) = self.pending.as_mut() {
            changed += p.backup.set_enforce_dale(true);
        }
        if let Some(b) = self.best.as_mut() {
            changed += b.topology.set_enforce_dale(true);
        }
        changed
    }

    /// Configuration.
    pub fn config(&self) -> &StructuralConfig<T> {
        &self.cfg
    }

    /// Event log.
    pub fn log(&self) -> &EventLog {
        &self.log
    }

    /// How many ratings in a row did the quality not improve?
    pub fn plateau_evaluations(&self) -> usize {
        self.since_improvement
    }

    /// The best cost-effective topology snapshot.
    pub fn best(&self) -> Option<&BestSnapshot<T>> {
        self.best.as_ref()
    }

    /// Return the network to the best-cost preserved topology.
    pub fn restore_best(&self, net: &mut DynamicNetwork<T>) -> bool {
        match &self.best {
            Some(b) => {
                net.replace_topology(b.topology.clone());
                true
            }
            None => false,
        }
    }

    fn recent_mean(&self, k: usize) -> T {
        let k = k.max(1).min(self.losses.len());
        if k == 0 {
            return T::from_f64(f64::NAN);
        }
        self.losses
            .iter()
            .rev()
            .take(k)
            .fold(T::ZERO, |a, &b| a + b)
            / T::from_usize(k)
    }

    fn is_worse(&self, before: T, after: T, cost_before: T, cost_after: T) -> bool {
        let r = &self.cfg.rollback;
        let (b, a) = if r.use_structure_cost {
            (cost_before, cost_after)
        } else {
            (before, after)
        };
        a > b + r.abs_tolerance + r.rel_tolerance * b.abs()
    }

    fn penalize(&mut self, kinds: &[ActionKind], step: u64) {
        let r = &self.cfg.rollback;
        for k in kinds {
            let s = self.streak.entry(*k).or_insert(0);
            *s += 1;
            let pause = r
                .cooldown_steps
                .saturating_mul(1u64 << (*s - 1).min(30))
                .min(r.max_cooldown_steps);
            self.cooldown.insert(*k, step.saturating_add(pause));
        }
    }

    fn event(
        &self,
        net: &DynamicNetwork<T>,
        action: &StructuralAction<T>,
        reason: &Reason,
    ) -> StructuralEvent {
        let t = net.topology();
        StructuralEvent {
            seq: 0,
            step: net.step_count(),
            sample: net.sample_count(),
            kind: action.kind(),
            action: action.to_string(),
            reason: reason.to_string(),
            outcome: Outcome::Applied,
            loss_before: None,
            loss_after: None,
            neurons_before: t.neuron_count(),
            neurons_after: t.neuron_count(),
            synapses_before: t.synapse_count(),
            synapses_after: t.synapse_count(),
        }
    }

    /// Analyze the network and change the structure if necessary.
    /// Call on the safe boundary of learning.
    pub fn adapt(
        &mut self,
        net: &mut DynamicNetwork<T>,
        metrics: Metrics<T>,
        eval: Option<&mut dyn FnMut(&mut DynamicNetwork<T>) -> T>,
    ) -> StructuralResult<AdaptReport<T>> {
        let mut eval = eval;
        let silent = self.cfg.pruning.neuron_min_rate;
        let over = self.cfg.homeostasis.rate_max;
        let mut report = AdaptReport {
            skipped: None,
            applied: 0,
            failed: 0,
            rolled_back: false,
            loss_before: None,
            loss_after: None,
            events: Vec::new(),
            stats: net.stats(silent, over),
        };
        if !metrics.task_loss.is_finite() {
            report.skipped = Some(SkipReason::InvalidMetrics);
            return Ok(report);
        }
        self.calls += 1;
        self.losses.push_back(metrics.task_loss);
        while self.losses.len() > self.cfg.loss_window {
            self.losses.pop_front();
        }
        if metrics.task_loss < self.best_loss - self.cfg.growth.neurons.plateau_min_delta {
            self.best_loss = metrics.task_loss;
            self.since_improvement = 0;
        } else {
            self.since_improvement += 1;
        }
        let step = net.step_count();
        if self.cfg.growth.enabled && self.cfg.growth.track_coactivity {
            let g = &self.cfg.growth;
            net.enable_coactivity(
                g.coactivity_window,
                g.coactivity_capacity,
                g.coactivity_max_recent,
            );
        }

        // Delayed check of previous change (no external evaluation).
        if let Some(p) = self.pending.as_mut() {
            p.calls_left = p.calls_left.saturating_sub(1);
        }
        if self.pending.as_ref().is_some_and(|p| p.calls_left == 0) {
            if let Some(p) = self.pending.take() {
                let now = self.recent_mean(self.cfg.rollback.verify_after_calls as usize);
                let cost_now = self.cost.cost(now, &report.stats);
                if self.is_worse(p.loss_before, now, p.cost_before, cost_now) {
                    net.replace_topology(p.backup);
                    let kinds: Vec<ActionKind> = p.seqs.iter().map(|s| s.1).collect();
                    for (seq, kind) in &p.seqs {
                        self.log.reclassify_rolled_back(*seq, *kind, now.to_f32());
                    }
                    self.penalize(&kinds, step);
                    self.last_change_step = Some(step);
                    report.rolled_back = true;
                    report.loss_before = Some(p.loss_before);
                    report.loss_after = Some(now);
                    report.skipped = Some(SkipReason::TooSoon);
                    report.stats = net.stats(silent, over);
                    return Ok(report);
                }
                for (_, k) in &p.seqs {
                    self.streak.insert(*k, 0);
                }
            }
        }

        if self.pending.is_some() {
            report.skipped = Some(SkipReason::Verifying);
            return Ok(report);
        }
        if self.calls % self.cfg.schedule.analyze_every != 0 {
            report.skipped = Some(SkipReason::NotDue);
            return Ok(report);
        }
        if let Some(l) = self.last_change_step {
            if step.saturating_sub(l) < self.cfg.schedule.min_interval_steps {
                report.skipped = Some(SkipReason::TooSoon);
                return Ok(report);
            }
        }

        // Best cost topology.
        let stats0 = report.stats.clone();
        let j0 = self.cost.cost(metrics.task_loss, &stats0);
        if self.best.as_ref().is_none_or(|b| j0 < b.cost) {
            self.best = Some(BestSnapshot {
                topology: net.topology().clone(),
                cost: j0,
                loss: metrics.task_loss,
                step,
            });
        }

        // Change plan.
        let skip: BTreeSet<ActionKind> = self
            .cooldown
            .iter()
            .filter(|(_, until)| **until > step)
            .map(|(k, _)| *k)
            .collect();
        let max_ops = self.cfg.schedule.max_ops_per_interval;
        let mut plan: Vec<(StructuralAction<T>, Reason)> = Vec::new();
        let verdicts = synapse_verdicts(net, &self.cfg.pruning);
        if self.cfg.pruning.enabled {
            plan.extend(propose_prune_neurons(net, &self.cfg.pruning));
            plan.extend(propose_prune_synapses(
                net,
                &self.cfg.pruning,
                &verdicts,
                max_ops,
            ));
        }
        let blocked = self
            .cfg
            .max_state_bytes
            .is_some_and(|b| stats0.state_bytes >= b);
        let needed = self
            .cfg
            .growth
            .target_loss
            .is_none_or(|t| metrics.task_loss > t);
        if self.cfg.growth.enabled && !blocked && needed {
            plan.extend(propose_grow_synapses(
                net,
                &self.cfg.growth,
                &mut self.rng,
                self.cfg.growth.synapses.max_new_per_interval,
            ));
            let ng = &self.cfg.growth.neurons;
            if ng.enabled
                && self.since_improvement >= ng.plateau_patience
                && metrics.task_loss >= ng.min_loss
                && ng.max_new_per_interval > 0
            {
                if let Some(a) =
                    propose_grow_neuron(net, ng, &mut self.rng, &skip, self.since_improvement)
                {
                    plan.push(a);
                }
            }
        }
        if self.cfg.model_selection.enabled {
            let switches = if self.cfg.model_selection.loss_evaluation {
                match eval.as_mut() {
                    Some(f) => {
                        propose_model_switches_evaluated(net, &self.cfg.model_selection, &mut **f)
                    }
                    None => propose_model_switches(net, &self.cfg.model_selection),
                }
            } else {
                propose_model_switches(net, &self.cfg.model_selection)
            };
            plan.extend(switches);
        }
        if self.cfg.param_learning.enabled {
            if let Some(f) = eval.as_mut() {
                // Changing the model and training the parameters of one neuron in one
                // terms conflict: the parameters refer to the old model.
                // We leave priority to changing the model, we will study the parameters on
                // next interval.
                let switched: BTreeSet<NeuronId> = plan
                    .iter()
                    .filter_map(|(a, _)| match a {
                        StructuralAction::ChangeNeuronModel { id, .. } => Some(*id),
                        _ => None,
                    })
                    .collect();
                let updates = propose_param_updates(net, &self.cfg.param_learning, &mut **f);
                plan.extend(updates.into_iter().filter(|(a, _)| match a {
                    StructuralAction::MutateNeuron { id, .. } => !switched.contains(id),
                    _ => true,
                }));
            }
        }
        // Local regulation of inhibitory strength: this is homeostasis of inhibitory
        // synapse magnitudes (signs and structure do not change), so it happens outside
        // the plan of structural actions, just like plasticity itself.
        if net.config().balance.enabled {
            let cfg = net.config().balance;
            net.balance_inhibition(&cfg);
        }
        if self.cfg.homeostasis.enabled {
            let h = &self.cfg.homeostasis;
            let mut dev: Vec<(T, StructuralAction<T>, Reason)> = Vec::new();
            for n in net
                .topology()
                .neurons()
                .filter(|n| n.role != NeuronRole::Input && n.stats.steps >= h.min_observed_steps)
            {
                let r = n.stats.rate_ema;
                let th = n.threshold();
                // For thresholds near zero (AdEx `v_peak`, HH `spike_threshold`)
                // Relative shift doesn't work - use absolute shift.
                let delta = if th.abs() < T::c(1e-3) {
                    T::ONE
                } else {
                    th * h.threshold_step
                };
                if r > h.rate_max {
                    dev.push((
                        r - h.rate_max,
                        StructuralAction::MutateNeuron {
                            id: n.id,
                            mutation: NeuronMutation {
                                threshold: Some(th + delta),
                                ..Default::default()
                            },
                        },
                        Reason::Overactive { rate: r.to_f32() },
                    ));
                } else if r < h.rate_min {
                    dev.push((
                        h.rate_min - r,
                        StructuralAction::MutateNeuron {
                            id: n.id,
                            mutation: NeuronMutation {
                                threshold: Some(th - delta),
                                ..Default::default()
                            },
                        },
                        Reason::Underactive { rate: r.to_f32() },
                    ));
                }
            }
            dev.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
            plan.extend(
                dev.into_iter()
                    .take(h.max_neurons_per_interval)
                    .map(|(_, a, r)| (a, r)),
            );
        }
        plan.retain(|(a, _)| !skip.contains(&a.kind()));
        // Actions on neurons that the same plan deletes are meaningless.
        let removed: BTreeSet<NeuronId> = plan
            .iter()
            .filter_map(|(a, _)| {
                if let StructuralAction::RemoveNeuron { id } = a {
                    Some(*id)
                } else {
                    None
                }
            })
            .collect();
        if !removed.is_empty() {
            let topo = net.topology();
            let syn_hits = |id: &SynapseId| {
                topo.synapse(*id)
                    .is_some_and(|s| removed.contains(&s.src) || removed.contains(&s.dst))
            };
            plan.retain(|(a, _)| match a {
                StructuralAction::DisableSynapse { id }
                | StructuralAction::RemoveSynapse { id }
                | StructuralAction::MutateSynapse { id, .. }
                | StructuralAction::RewireSynapse { id, .. }
                | StructuralAction::SplitSynapse { id, .. } => !syn_hits(id),
                StructuralAction::AddSynapse { src, dst, .. } => {
                    !removed.contains(src) && !removed.contains(dst)
                }
                StructuralAction::CloneNeuron { id, .. }
                | StructuralAction::MutateNeuron { id, .. }
                | StructuralAction::ChangeNeuronModel { id, .. } => !removed.contains(id),
                StructuralAction::AddNeuron { srcs, dsts, .. } => !srcs
                    .iter()
                    .chain(dsts.iter())
                    .any(|(n, _)| removed.contains(n)),
                StructuralAction::RemoveNeuron { .. } => true,
            });
        }
        plan.truncate(max_ops);

        if let Some(t) = net.tracker_mut() {
            t.decay(self.cfg.growth.coactivity_decay);
        }
        if plan.is_empty() {
            report.skipped = Some(SkipReason::NothingToDo);
            return Ok(report);
        }

        // Application with the possibility of rollback.
        let backup = net.topology().clone();
        let rng_backup = self.rng.clone();
        mark_evaluated(net, &verdicts);
        let loss_before = measure(&mut eval, net);
        let mut done: Vec<StructuralEvent> = Vec::new();
        let mut failed: Vec<StructuralEvent> = Vec::new();
        for (action, reason) in &plan {
            let mut ev = self.event(net, action, reason);
            match apply_action(net, action, &mut self.rng) {
                Ok(_) => {
                    ev.neurons_after = net.topology().neuron_count();
                    ev.synapses_after = net.topology().synapse_count();
                    done.push(ev);
                }
                Err(e) => {
                    ev.outcome = Outcome::Failed(e.to_string());
                    failed.push(ev);
                }
            }
        }
        report.failed = failed.len();
        for mut ev in failed {
            ev.seq = self.log.push(ev.clone());
            report.events.push(ev);
        }
        if done.is_empty() {
            report.skipped = Some(SkipReason::NothingToDo);
            return Ok(report);
        }
        net.topology().validate().map_err(|e| {
            StructuralError::InvariantViolation(format!("after applying the plan: {e}"))
        })?;

        let kinds: Vec<ActionKind> = done.iter().map(|e| e.kind).collect();
        self.last_change_step = Some(step);
        let mut keep = true;
        let mut loss_after = None;
        if self.cfg.rollback.enabled {
            if let Some(lb) = loss_before {
                loss_after = measure(&mut eval, net);
                if let Some(la) = loss_after {
                    let stats1 = net.stats(silent, over);
                    let (jb, ja) = (self.cost.cost(lb, &stats0), self.cost.cost(la, &stats1));
                    keep = !self.is_worse(lb, la, jb, ja);
                }
            }
        }
        report.loss_before = loss_before;
        report.loss_after = loss_after;
        if !keep {
            net.replace_topology(backup);
            self.rng = rng_backup;
            self.penalize(&kinds, step);
            report.rolled_back = true;
            for mut ev in done {
                ev.outcome = Outcome::RolledBack;
                ev.loss_before = loss_before.map(Scalar::to_f32);
                ev.loss_after = loss_after.map(Scalar::to_f32);
                ev.neurons_after = ev.neurons_before;
                ev.synapses_after = ev.synapses_before;
                ev.seq = self.log.push(ev.clone());
                report.events.push(ev);
            }
        } else {
            let mut seqs = Vec::new();
            for mut ev in done {
                ev.loss_before = loss_before.map(Scalar::to_f32);
                ev.loss_after = loss_after.map(Scalar::to_f32);
                let seq = self.log.push(ev.clone());
                ev.seq = seq;
                seqs.push((seq, ev.kind));
                report.events.push(ev);
                report.applied += 1;
            }
            for k in &kinds {
                self.streak.insert(*k, 0);
            }
            if kinds.iter().any(|k| k.is_growth()) {
                self.since_improvement = 0;
            }
            if self.cfg.rollback.enabled {
                self.pending = Some(Pending {
                    backup,
                    loss_before: metrics.task_loss,
                    cost_before: j0,
                    calls_left: self.cfg.rollback.verify_after_calls.max(1),
                    seqs,
                });
            }
        }
        report.stats = net.stats(silent, over);
        Ok(report)
    }

    /// Cost `J` for a given loss and structure indicators.
    pub fn cost(&self, task_loss: T, stats: &StructuralStats<T>) -> T {
        self.cost.cost(task_loss, stats)
    }

    /// Default cost from configuration.
    pub fn default_cost(&self) -> LinearCost<T> {
        self.cfg.cost
    }
}
