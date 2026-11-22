//! Structural plasticity telemetry: what the controller communicates to the outside and how to read it.
//!
//! * [`StructureReport`] - the result of one call `StructuralPlasticityController::adapt`:
//!   absolute [`StructuralStats`], NEW log events, loss before/after and (optional)
//!   [`TopologySnapshot`] for drawing a graph. The report is self-sufficient: the numbers in it are absolute,
//!   therefore, the loss of one report in a crowded queue (see `TrainTx`) does not distort
//!   graphs, and the consumer notices and shows the gap in `StructuralEvent::seq`.
//! * [`TopologySnapshot`] - a copy of the graph limited in size (nodes and edges without state
//!   simulation): based on it, the GUI draws what actually exists, and not animation.
//! * [`assess_structure`] - verbal verdict (“growth without improvement”, “quality has deteriorated”...),
//!   like [`super::assess`]: pure function without I/O.

use std::sync::Arc;

use crate::structural::{
    ActionKind, AdaptReport, NeuronKind, NeuronModel, NeuronPolarity, Outcome, StructuralEvent,
    StructuralStats, Topology,
};

use super::channel::Progress;
use super::health::{Finding, Severity};
use super::history::History;

/// How many neurons and synapses are included in the snapshot by default.
pub const SNAPSHOT_NEURONS: usize = 256;
/// See [`SNAPSHOT_NEURONS`].
pub const SNAPSHOT_SYNAPSES: usize = 2048;

/// Snapshot node.
#[derive(Debug, Clone, PartialEq)]
pub struct NodeView {
    /// Stable neuron identifier.
    pub id: u64,
    /// Role.
    pub kind: NeuronKind,
    /// Physiological polarity.
    pub polarity: NeuronPolarity,
    /// Dynamics model.
    pub model: NeuronModel,
    /// Sliding spike frequency (spikes per step).
    pub rate: f32,
    /// Creation step (0 - initial structure).
    pub born_step: u64,
}

/// Edge of the photo.
#[derive(Debug, Clone, PartialEq)]
pub struct EdgeView {
    /// Stable synapse ID.
    pub id: u64,
    /// Source (`NodeView::id`).
    pub src: u64,
    /// Receiver.
    pub dst: u64,
    /// Weight.
    pub weight: f32,
    /// Is it active (otherwise disabled and awaiting deletion).
    pub active: bool,
}

/// A copy of the graph for drawing. Size is limited: if exceeded, all inputs and outputs remain,
/// hidden neurons with the smallest id and the strongest `|w|` edges between the abandoned nodes
/// ([`truncated`](Self::truncated) = `true`).
#[derive(Debug, Clone, PartialEq, Default)]
pub struct TopologySnapshot {
    /// `Topology::version` at the time of the photo.
    pub version: u64,
    /// Nodes in ascending order by id.
    pub nodes: Vec<NodeView>,
    /// Edges in ascending order by id.
    pub edges: Vec<EdgeView>,
    /// How many neurons are there actually in the network?
    pub total_neurons: usize,
    /// How many synapses are there actually in the network?
    pub total_synapses: usize,
    /// Part of the graph was not included in the snapshot.
    pub truncated: bool,
}

impl TopologySnapshot {
    /// Make a copy with no more than `max_neurons` nodes and `max_synapses` edges.
    pub fn capture(t: &Topology, max_neurons: usize, max_synapses: usize) -> Self {
        let total_neurons = t.neuron_count();
        let total_synapses = t.synapse_count();
        let mut nodes: Vec<NodeView> = t
            .neurons()
            .map(|n| NodeView {
                id: n.id.0,
                kind: n.role,
                polarity: n.polarity,
                model: n.model.clone(),
                rate: n.stats.rate_ema,
                born_step: n.meta.born_step,
            })
            .collect();
        if nodes.len() > max_neurons {
            // Inputs and outputs are more important than hidden ones; within a role - by id (stable).
            nodes.sort_by_key(|n| (n.kind == NeuronKind::Hidden, n.id));
            nodes.truncate(max_neurons);
        }
        nodes.sort_by_key(|n| n.id);
        let kept = |id: u64| nodes.binary_search_by_key(&id, |n| n.id).is_ok();
        let mut edges: Vec<EdgeView> = t
            .synapses()
            .filter(|s| kept(s.src.0) && kept(s.dst.0))
            .map(|s| EdgeView {
                id: s.id.0,
                src: s.src.0,
                dst: s.dst.0,
                weight: s.weight,
                active: s.is_active(),
            })
            .collect();
        if edges.len() > max_synapses {
            edges.sort_by(|a, b| {
                b.weight
                    .abs()
                    .total_cmp(&a.weight.abs())
                    .then(a.id.cmp(&b.id))
            });
            edges.truncate(max_synapses);
        }
        edges.sort_by_key(|e| e.id);
        let truncated = nodes.len() < total_neurons || edges.len() < total_synapses;
        Self {
            version: t.version(),
            nodes,
            edges,
            total_neurons,
            total_synapses,
            truncated,
        }
    }
}

/// The limits against which growth is measured.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct StructureLimits {
    /// Maximum neurons (0 - unlimited).
    pub max_neurons: usize,
    /// Maximum synapses (0 - unlimited).
    pub max_synapses: usize,
}

/// The result of one controller call.
#[derive(Debug, Clone, PartialEq)]
pub struct StructureReport {
    /// Where is the training located?
    pub progress: Progress,
    /// Structural indicators at the time of analysis (absolute).
    pub stats: StructuralStats,
    /// New log events (in order `seq`).
    pub events: Vec<StructuralEvent>,
    /// Loss before the last change (if measured).
    pub loss_before: Option<f32>,
    /// Loss after it.
    pub loss_after: Option<f32>,
    /// Why there were no changes (`None` - the analysis did something).
    pub skipped: Option<String>,
    /// Limits to growth.
    pub limits: StructureLimits,
    /// Topology, if requested.
    pub topology: Option<Arc<TopologySnapshot>>,
}

impl StructureReport {
    /// Collect a report from the result `adapt`. `snapshot` — dimensions of the graph snapshot
    /// (`Some((SNAPSHOT_NEURONS, SNAPSHOT_SYNAPSES))`) or `None` if the graph is not needed.
    pub fn from_adapt(
        progress: Progress,
        r: &AdaptReport,
        t: &Topology,
        snapshot: Option<(usize, usize)>,
    ) -> Self {
        Self {
            progress,
            stats: r.stats.clone(),
            events: r.events.clone(),
            loss_before: r.loss_before,
            loss_after: r.loss_after,
            skipped: r.skipped.map(|s| format!("{s:?}")),
            limits: StructureLimits {
                max_neurons: t.limits().max_neurons,
                max_synapses: t.limits().max_synapses,
            },
            topology: snapshot.map(|(n, s)| Arc::new(TopologySnapshot::capture(t, n, s))),
        }
    }
}

impl StructureReport {
    /// Collect a report from the result `FixedShapeController::adapt` (fixed shape layers).
    /// There is no graph (`topology: None`); limits - the capacity of tensors.
    pub fn from_fixed(
        progress: Progress,
        r: &crate::structural::fixed::FixedAdaptReport,
        capacity: (usize, usize),
    ) -> Self {
        Self {
            progress,
            stats: r.stats.clone(),
            events: r.events.clone(),
            loss_before: r.loss_before,
            loss_after: r.loss_after,
            skipped: r.skipped.map(|s| format!("{s:?}")),
            limits: StructureLimits {
                max_neurons: capacity.0,
                max_synapses: capacity.1,
            },
            topology: None,
        }
    }
}

fn is_growth(k: ActionKind) -> bool {
    matches!(
        k,
        ActionKind::AddSynapse
            | ActionKind::AddNeuron
            | ActionKind::CloneNeuron
            | ActionKind::SplitSynapse
    )
}

/// Verdict on the structure: points from the most serious to the mildest, the list is never empty.
///
/// `events` - recent events (the window is selected by the caller, for example the last 64),
/// `loss` — loss history by training steps.
pub fn assess_structure(
    stats: &StructuralStats,
    limits: StructureLimits,
    events: &[StructuralEvent],
    loss: &History,
) -> Vec<Finding> {
    let f = |severity, what: String, advice: &str| Finding {
        severity,
        what,
        advice: advice.to_string(),
    };
    let mut out = Vec::new();

    // Growth: how much have we hit the limits and are we growing without return.
    let ratio = |n: usize, max: usize| if max == 0 { 0.0 } else { n as f32 / max as f32 };
    let n_ratio = ratio(stats.neurons, limits.max_neurons);
    let s_ratio = ratio(stats.synapses, limits.max_synapses);
    let near_limit = n_ratio.max(s_ratio) >= 0.9;
    let growth = events
        .iter()
        .filter(|e| e.outcome == Outcome::Applied && is_growth(e.kind))
        .count();
    let improving = loss.trend().is_some_and(|t| t < -0.01);
    if growth >= 6 && !improving {
        let sev = if near_limit {
            Severity::Problem
        } else {
            Severity::Notice
        };
        out.push(f(
            sev,
            format!("the structure grows without improvement: {growth} additions in the window, loss does not fall"),
            "raise growth.synapses.min_score or set growth.target_loss; check that the quality metric is informative",
        ));
    } else if near_limit {
        out.push(f(
            Severity::Notice,
            format!(
                "growth has reached a limit: neurons {}/{}, synapses {}/{}",
                stats.neurons, limits.max_neurons, stats.synapses, limits.max_synapses
            ),
            "new elements are no longer created; strengthen pruning or consciously raise topology.max_*",
        ));
    }

    // The quality has deteriorated relative to the best.
    if let (Some(best), Some(now), true) =
        (loss.best(), loss.recent_mean(4), loss.values().len() >= 8)
        && now > best * 1.25 + 0.02
    {
        let touched = events.iter().any(|e| e.outcome == Outcome::Applied);
        out.push(f(
                Severity::Problem,
                format!("quality has deteriorated: loss {now:.3} with better {best:.3}"),
                if touched {
                    "after structural changes: return the best snapshot (controller.restore_best()) or strengthen rollback"
                } else {
                    "the structure did not change: the reason is in training (lr, data), not in topology"
                },
            ));
    }

    // Kickbacks and refusals.
    let rolled = events
        .iter()
        .filter(|e| e.outcome == Outcome::RolledBack)
        .count();
    let decided = events
        .iter()
        .filter(|e| matches!(e.outcome, Outcome::Applied | Outcome::RolledBack))
        .count();
    if rolled >= 3 && rolled * 2 >= decided {
        out.push(f(
            Severity::Notice,
            format!("{rolled} out of {decided} changes rolled back"),
            "controller suggestions degrade the network: reduce max_ops_per_interval or raise growth thresholds",
        ));
    }
    if let Some(e) = events
        .iter()
        .rev()
        .find(|e| matches!(e.outcome, Outcome::Failed(_)))
    {
        let n = events
            .iter()
            .filter(|e| matches!(e.outcome, Outcome::Failed(_)))
            .count();
        let why = if let Outcome::Failed(w) = &e.outcome {
            w.as_str()
        } else {
            ""
        };
        out.push(f(
            Severity::Notice,
            format!(
                "{n} operation{s} failed to apply: {why}",
                s = if n == 1 { "" } else { "s" }
            ),
            "operation violated topology constraints",
        ));
    }

    // Activity.
    if stats.overactive_neurons > 0 {
        out.push(f(
            Severity::Notice,
            format!(
                "{} neur. overloaded (frequency above threshold)",
                stats.overactive_neurons
            ),
            "enable threshold homeostasis or increase threshold",
        ));
    }
    let working = stats.hidden + stats.outputs;
    if working > 0 && stats.silent_neurons * 2 > working {
        out.push(f(
            Severity::Notice,
            format!("silent {} out of {working} neurons", stats.silent_neurons),
            "they do not participate in the calculation; pruning will remove hidden outputs, you should check them manually",
        ));
    }

    if out.is_empty() {
        out.push(f(Severity::Ok, "structure is ok".into(), ""));
    }
    out.sort_by_key(|a| std::cmp::Reverse(a.severity));
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::structural::{Birth, LifParams, Origin, Reason, TopologyLimits};

    fn ev(seq: u64, kind: ActionKind, outcome: Outcome) -> StructuralEvent {
        StructuralEvent {
            seq,
            step: seq * 10,
            sample: seq,
            kind,
            action: kind.as_str().into(),
            reason: Reason::Manual.to_string(),
            outcome,
            loss_before: None,
            loss_after: None,
            neurons_before: 5,
            neurons_after: 5,
            synapses_before: 5,
            synapses_after: 6,
        }
    }

    fn stats(n: usize, s: usize) -> StructuralStats {
        StructuralStats {
            neurons: n,
            synapses: s,
            hidden: n.saturating_sub(4),
            outputs: 2,
            ..Default::default()
        }
    }

    fn hist(v: &[f32]) -> History {
        let mut h = History::new(64);
        v.iter().for_each(|x| h.push(*x));
        h
    }

    const LIM: StructureLimits = StructureLimits {
        max_neurons: 20,
        max_synapses: 100,
    };

    #[test]
    fn quiet_structure_is_ok() {
        let r = assess_structure(&stats(8, 20), LIM, &[], &hist(&[1.0, 0.8, 0.6, 0.5]));
        assert_eq!(r.len(), 1);
        assert_eq!(r[0].severity, Severity::Ok);
    }

    #[test]
    fn growth_without_gain_is_flagged_and_escalates_near_the_limit() {
        let evs: Vec<_> = (0..6)
            .map(|i| ev(i, ActionKind::AddSynapse, Outcome::Applied))
            .collect();
        let flat = hist(&[0.9, 0.9, 0.9, 0.9, 0.9, 0.9]);
        let r = assess_structure(&stats(8, 20), LIM, &evs, &flat);
        assert_eq!(r[0].severity, Severity::Notice);
        assert!(r[0].what.contains("grows without improvement"));
        let r = assess_structure(&stats(19, 20), LIM, &evs, &flat);
        assert_eq!(r[0].severity, Severity::Problem);
        // If the loss falls, growth is justified.
        let falling = hist(&[1.0, 0.8, 0.6, 0.4]);
        let r = assess_structure(&stats(8, 20), LIM, &evs, &falling);
        assert_eq!(r[0].severity, Severity::Ok);
    }

    #[test]
    fn limit_reached_is_reported() {
        let r = assess_structure(&stats(20, 30), LIM, &[], &hist(&[]));
        assert!(r.iter().any(|f| f.what.contains("reached a limit")));
        // Limit 0 = unlimited.
        let r = assess_structure(&stats(20, 30), StructureLimits::default(), &[], &hist(&[]));
        assert_eq!(r[0].severity, Severity::Ok);
    }

    #[test]
    fn degradation_from_best_is_a_problem() {
        let h = hist(&[0.2, 0.2, 0.2, 0.2, 0.9, 0.9, 0.9, 0.9]);
        let evs = [ev(0, ActionKind::SplitSynapse, Outcome::Applied)];
        let r = assess_structure(&stats(8, 20), LIM, &evs, &h);
        assert_eq!(r[0].severity, Severity::Problem);
        assert!(r[0].advice.contains("restore_best"));
        // Without structural changes, the advice is different.
        let r = assess_structure(&stats(8, 20), LIM, &[], &h);
        assert!(r[0].advice.contains("did not change"));
        // Short story - no conclusions.
        let r = assess_structure(&stats(8, 20), LIM, &[], &hist(&[0.2, 0.9]));
        assert_eq!(r[0].severity, Severity::Ok);
    }

    #[test]
    fn rollbacks_failures_and_activity_are_reported() {
        let mut evs: Vec<_> = (0..3)
            .map(|i| ev(i, ActionKind::AddNeuron, Outcome::RolledBack))
            .collect();
        evs.push(ev(
            3,
            ActionKind::RemoveNeuron,
            Outcome::Failed("no neuron".into()),
        ));
        let mut s = stats(8, 20);
        s.overactive_neurons = 2;
        s.silent_neurons = 5;
        let r = assess_structure(&s, LIM, &evs, &hist(&[]));
        let text: String = r
            .iter()
            .map(|f| f.what.clone())
            .collect::<Vec<_>>()
            .join("|");
        assert!(text.contains("3 out of 3 changes rolled back"), "{text}");
        assert!(text.contains("failed to apply: no neuron"));
        assert!(text.contains("2 neur. overloaded"));
        assert!(text.contains("silent 5 out of 6"));
        assert!(r.windows(2).all(|w| w[0].severity >= w[1].severity));
    }

    fn topo(hidden: usize) -> Topology {
        let lim = TopologyLimits {
            max_neurons: 100,
            max_synapses: 1000,
            ..TopologyLimits::default()
        };
        let mut t = Topology::new(lim, Default::default());
        let b = Birth::default();
        let i = t
            .add_neuron(NeuronKind::Input, LifParams::default(), Origin::Initial, b)
            .unwrap();
        let o = t
            .add_neuron(
                NeuronKind::Output,
                LifParams::new(0.9, 1.0),
                Origin::Initial,
                b,
            )
            .unwrap();
        for k in 0..hidden {
            let h = t
                .add_neuron(
                    NeuronKind::Hidden,
                    LifParams::new(0.9, 1.0),
                    Origin::Initial,
                    b,
                )
                .unwrap();
            t.add_synapse(i, h, 0.1 + k as f32 * 0.01, 0, Origin::Initial, b)
                .unwrap();
            t.add_synapse(h, o, 0.2, 0, Origin::Initial, b).unwrap();
        }
        t
    }

    #[test]
    fn snapshot_is_exact_when_it_fits() {
        let t = topo(3);
        let s = TopologySnapshot::capture(&t, 100, 100);
        assert!(!s.truncated);
        assert_eq!(
            (s.nodes.len(), s.edges.len()),
            (t.neuron_count(), t.synapse_count())
        );
        assert_eq!(s.version, t.version());
        assert!(s.nodes.windows(2).all(|w| w[0].id < w[1].id));
        assert!(s.edges.iter().all(|e| e.active));
    }

    #[test]
    fn snapshot_truncation_keeps_io_and_consistent_edges() {
        let t = topo(10);
        let s = TopologySnapshot::capture(&t, 5, 4);
        assert!(s.truncated);
        assert_eq!(s.nodes.len(), 5);
        assert_eq!(
            s.nodes
                .iter()
                .filter(|n| n.kind != NeuronKind::Hidden)
                .count(),
            2,
            "input and output saved"
        );
        assert!(s.edges.len() <= 4);
        for e in &s.edges {
            assert!(
                s.nodes.iter().any(|n| n.id == e.src) && s.nodes.iter().any(|n| n.id == e.dst),
                "dangling edge"
            );
        }
        assert_eq!(
            (s.total_neurons, s.total_synapses),
            (t.neuron_count(), t.synapse_count())
        );
    }
}
