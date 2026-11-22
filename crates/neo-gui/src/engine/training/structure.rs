//! What the dashboard knows about the network's STRUCTURE: folded from the
//! [`StructureReport`]s `neo_snn::telemetry` delivers, one at a time. Pure data, like
//! [`super::state`]: no UI type, tested headless.
//!
//! The reports carry absolute counters and the NEW events of the structural controller's log, so
//! this state keeps three things: histories of the counters (one point per report), a bounded
//! list of events with their reasons and the loss measured around each change, and the latest
//! graph snapshot. A hole in the events' `seq` means the training thread had to drop a report
//! (its queue was full); it is counted and shown, not hidden.

use std::collections::VecDeque;
use std::sync::Arc;

use neo_snn::structural::{ActionKind, Outcome, StructuralEvent, StructuralStats};
use neo_snn::telemetry::{
    Finding, History, Severity, StructureLimits, StructureReport, TopologySnapshot,
    assess_structure,
};

/// Reports kept per counter history.
pub const STRUCTURE_HISTORY: usize = 1024;
/// Events kept in the list (the newest).
pub const EVENT_LOG: usize = 256;
/// Newest events the health verdict looks at.
pub const FINDING_WINDOW: usize = 64;

/// What a change did, for the colour of its marker on the charts.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum MarkerKind {
    /// A synapse or neuron was added, cloned or a synapse split.
    Growth,
    /// A synapse was disabled or removed, or a neuron removed.
    Pruning,
    /// The change made the network worse and was undone.
    Rollback,
    /// The change was refused (an invariant of the topology).
    Failed,
}

impl MarkerKind {
    pub const ALL: [MarkerKind; 4] = [
        MarkerKind::Growth,
        MarkerKind::Pruning,
        MarkerKind::Rollback,
        MarkerKind::Failed,
    ];

    pub fn label(self) -> &'static str {
        match self {
            MarkerKind::Growth => "growth",
            MarkerKind::Pruning => "pruning",
            MarkerKind::Rollback => "rolled back",
            MarkerKind::Failed => "failed",
        }
    }

    /// The marker an event gets; parameter changes and rewiring have none.
    pub fn of(e: &StructuralEvent) -> Option<MarkerKind> {
        match (&e.outcome, e.kind) {
            (Outcome::RolledBack, _) => Some(MarkerKind::Rollback),
            (Outcome::Failed(_), _) => Some(MarkerKind::Failed),
            (
                Outcome::Applied,
                ActionKind::AddSynapse
                | ActionKind::AddNeuron
                | ActionKind::CloneNeuron
                | ActionKind::SplitSynapse,
            ) => Some(MarkerKind::Growth),
            (
                Outcome::Applied,
                ActionKind::DisableSynapse | ActionKind::RemoveSynapse | ActionKind::RemoveNeuron,
            ) => Some(MarkerKind::Pruning),
            (Outcome::Applied, _) => None,
        }
    }
}

/// A change on the time axis of the charts: the index of the report it arrived in.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct Marker {
    pub report: u64,
    pub kind: MarkerKind,
}

#[derive(Debug, Clone)]
pub struct StructureState {
    pub neurons: History,
    pub synapses: History,
    pub active_synapses: History,
    pub mean_rate: History,
    /// Counters of the latest report.
    pub stats: Option<StructuralStats>,
    pub limits: StructureLimits,
    /// The latest graph snapshot (reports may come without one).
    pub topology: Option<Arc<TopologySnapshot>>,
    /// Reports folded in.
    pub reports: u64,
    /// Why the latest analysis changed nothing.
    pub last_skip: Option<String>,
    /// `(before, after)` of the latest change whose loss was measured.
    pub last_loss: Option<(f32, f32)>,
    /// Events missing from the log: their reports were dropped by a full queue.
    pub missed_events: u64,
    /// Events seen, by outcome (all run long, not only the ones still in the list).
    pub applied: u64,
    pub rolled_back: u64,
    pub failed: u64,
    events: VecDeque<StructuralEvent>,
    markers: VecDeque<Marker>,
    last_seq: Option<u64>,
}

impl Default for StructureState {
    fn default() -> Self {
        Self::new()
    }
}

impl StructureState {
    pub fn new() -> Self {
        Self {
            neurons: History::new(STRUCTURE_HISTORY),
            synapses: History::new(STRUCTURE_HISTORY),
            active_synapses: History::new(STRUCTURE_HISTORY),
            mean_rate: History::new(STRUCTURE_HISTORY),
            stats: None,
            limits: StructureLimits::default(),
            topology: None,
            reports: 0,
            last_skip: None,
            last_loss: None,
            missed_events: 0,
            applied: 0,
            rolled_back: 0,
            failed: 0,
            events: VecDeque::new(),
            markers: VecDeque::new(),
            last_seq: None,
        }
    }

    /// No report yet: the Structure tab has nothing to show (the run does not use the mechanism).
    pub fn is_empty(&self) -> bool {
        self.reports == 0
    }

    /// Folds one report in.
    pub fn apply(&mut self, r: &StructureReport) {
        let index = self.reports;
        self.reports += 1;
        self.neurons.push(r.stats.neurons as f32);
        self.synapses.push(r.stats.synapses as f32);
        self.active_synapses.push(r.stats.active_synapses as f32);
        self.mean_rate.push(r.stats.mean_rate);
        self.stats = Some(r.stats.clone());
        self.limits = r.limits;
        if r.topology.is_some() {
            self.topology = r.topology.clone();
        }
        self.last_skip = r.skipped.clone();
        if let (Some(b), Some(a)) = (r.loss_before, r.loss_after) {
            self.last_loss = Some((b, a));
        }

        for e in &r.events {
            match self.last_seq {
                // Already seen (a report delivered twice): ignore
                Some(prev) if e.seq <= prev => continue,
                Some(prev) => self.missed_events += e.seq - prev - 1,
                None => {}
            }
            self.last_seq = Some(e.seq);
            match e.outcome {
                Outcome::Applied => self.applied += 1,
                Outcome::RolledBack => self.rolled_back += 1,
                Outcome::Failed(_) => self.failed += 1,
            }
            if let Some(kind) = MarkerKind::of(e) {
                self.markers.push_back(Marker {
                    report: index,
                    kind,
                });
            }
            if self.events.len() == EVENT_LOG {
                self.events.pop_front();
            }
            self.events.push_back(e.clone());
        }
        // Markers older than the history have no point to sit on
        let oldest = self.reports.saturating_sub(STRUCTURE_HISTORY as u64);
        while self.markers.front().is_some_and(|m| m.report < oldest) {
            self.markers.pop_front();
        }
    }

    /// The events kept, newest first.
    pub fn events_newest_first(&self) -> impl Iterator<Item = &StructuralEvent> {
        self.events.iter().rev()
    }

    #[cfg(test)]
    pub fn event_count(&self) -> usize {
        self.events.len()
    }

    /// Markers within the history window, with `report` made relative to the first point of the
    /// counter histories (what a chart drawn from `neurons.values()` needs).
    pub fn markers_in_window(&self) -> Vec<(usize, MarkerKind)> {
        let first = self
            .reports
            .saturating_sub(self.neurons.values().len() as u64);
        self.markers
            .iter()
            .filter(|m| m.report >= first)
            .map(|m| ((m.report - first) as usize, m.kind))
            .collect()
    }

    /// The verdict on the structure, worst first (empty before the first report): the library's
    /// [`assess_structure`] over the newest events and the loss history, plus the one thing only
    /// this side can see — reports lost on the way.
    pub fn findings(&self, loss: &History) -> Vec<Finding> {
        let Some(stats) = &self.stats else {
            return Vec::new();
        };
        let recent: Vec<StructuralEvent> = self
            .events
            .iter()
            .rev()
            .take(FINDING_WINDOW)
            .rev()
            .cloned()
            .collect();
        let mut out = assess_structure(stats, self.limits, &recent, loss);
        if self.missed_events > 0 {
            out.retain(|f| f.severity != Severity::Ok);
            out.push(Finding {
                severity: Severity::Notice,
                what: format!(
                    "{} structural events were lost (the queue was full)",
                    self.missed_events
                ),
                advice: "the list has gaps; the counters and the chart are still exact".into(),
            });
            out.sort_by_key(|a| std::cmp::Reverse(a.severity));
        }
        out
    }
}

/// One line of the event list, as text: what the view prints, separated from the view so it is
/// tested without a window.
#[derive(Debug, Clone, PartialEq)]
pub struct EventRow {
    /// For the colour of the dot (`None`: parameter changes and rewiring).
    pub marker: Option<MarkerKind>,
    /// `#3 · step 800 · add_synapse N1→N6 w=0.115 d=0`.
    pub head: String,
    /// Why the controller did it.
    pub reason: String,
    /// `loss 0.775 → 0.050`, when it was measured around the change.
    pub loss: Option<String>,
    /// `neurons 8 → 8 · synapses 5 → 6`.
    pub size: String,
    /// `applied`, `rolled back (made the network worse)` or `refused: why`.
    pub outcome: String,
}

impl EventRow {
    pub fn of(e: &StructuralEvent) -> Self {
        let loss = match (e.loss_before, e.loss_after) {
            (Some(b), Some(a)) => Some(format!("loss {b:.3} → {a:.3}")),
            (Some(b), None) => Some(format!("loss {b:.3} → ?")),
            _ => None,
        };
        let outcome = match &e.outcome {
            Outcome::Applied => "applied".to_string(),
            Outcome::RolledBack => "rolled back (it made the network worse)".to_string(),
            Outcome::Failed(why) => format!("refused: {why}"),
        };
        Self {
            marker: MarkerKind::of(e),
            head: format!("#{} · step {} · {}", e.seq, e.step, e.action),
            reason: e.reason.clone(),
            loss,
            size: format!(
                "neurons {} → {} · synapses {} → {}",
                e.neurons_before, e.neurons_after, e.synapses_before, e.synapses_after
            ),
            outcome,
        }
    }
}

fn skip_text(raw: &str) -> &str {
    match raw {
        "NotDue" => "not due yet",
        "TooSoon" => "too soon after the last change",
        "NothingToDo" => "nothing to change",
        "Verifying" => "verifying the previous change",
        "InvalidMetrics" => "the metrics were not numbers",
        other => other,
    }
}

impl StructureState {
    /// The facts of the tab's header: `(label, value)` pairs, in the order shown.
    pub fn summary(&self) -> Vec<(&'static str, String)> {
        let Some(s) = &self.stats else {
            return Vec::new();
        };
        let of = |n: usize, max: usize| {
            if max > 0 {
                format!("{n} of {}", max)
            } else {
                n.to_string()
            }
        };
        let mut out = vec![
            (
                "Neurons",
                format!(
                    "{} · {} working · {} hidden",
                    of(s.neurons, self.limits.max_neurons),
                    s.active_neurons,
                    s.hidden
                ),
            ),
            (
                "Synapses",
                format!(
                    "{} · {} active · {} disabled",
                    of(s.synapses, self.limits.max_synapses),
                    s.active_synapses,
                    s.disabled_synapses
                ),
            ),
            (
                "Firing",
                format!(
                    "mean rate {:.3} · {} overloaded · {} silent",
                    s.mean_rate, s.overactive_neurons, s.silent_neurons
                ),
            ),
            (
                "Changes",
                format!(
                    "{} applied · {} rolled back · {} refused",
                    self.applied, self.rolled_back, self.failed
                ),
            ),
            (
                "Loss around the last change",
                self.last_loss
                    .map_or("–".to_string(), |(b, a)| format!("{b:.3} → {a:.3}")),
            ),
            (
                "Last analysis",
                self.last_skip
                    .as_deref()
                    .map_or("changed the structure".to_string(), |r| {
                        skip_text(r).to_string()
                    }),
            ),
        ];
        if self.missed_events > 0 {
            out.push(("Events lost", self.missed_events.to_string()));
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use super::*;
    use neo_snn::structural::Outcome;
    use neo_snn::telemetry::structural_demo::{self, DemoConfig};
    use neo_snn::telemetry::{ChannelConfig, Progress, TrainEvent, channel};

    fn event(seq: u64, kind: ActionKind, outcome: Outcome) -> StructuralEvent {
        StructuralEvent {
            seq,
            step: seq,
            sample: seq,
            kind,
            action: kind.as_str().into(),
            reason: "test".into(),
            outcome,
            loss_before: Some(1.0),
            loss_after: Some(0.5),
            neurons_before: 4,
            neurons_after: 5,
            synapses_before: 6,
            synapses_after: 7,
        }
    }

    fn report(events: Vec<StructuralEvent>, neurons: usize) -> StructureReport {
        StructureReport {
            progress: Progress::default(),
            stats: StructuralStats {
                neurons,
                synapses: neurons * 2,
                active_synapses: neurons * 2,
                hidden: neurons.saturating_sub(4),
                outputs: 2,
                ..Default::default()
            },
            events,
            loss_before: Some(1.0),
            loss_after: Some(0.5),
            skipped: None,
            limits: StructureLimits {
                max_neurons: 20,
                max_synapses: 100,
            },
            topology: None,
        }
    }

    #[test]
    fn counters_markers_and_losses_are_folded() {
        let mut s = StructureState::new();
        assert!(s.is_empty() && s.findings(&History::new(4)).is_empty());
        s.apply(&report(
            vec![event(0, ActionKind::AddSynapse, Outcome::Applied)],
            5,
        ));
        s.apply(&report(
            vec![
                event(1, ActionKind::RemoveNeuron, Outcome::Applied),
                event(2, ActionKind::SplitSynapse, Outcome::RolledBack),
            ],
            6,
        ));
        assert_eq!(
            (
                s.reports,
                s.applied,
                s.rolled_back,
                s.failed,
                s.missed_events
            ),
            (2, 2, 1, 0, 0)
        );
        assert_eq!(s.neurons.values(), &[5.0, 6.0]);
        assert_eq!(s.last_loss, Some((1.0, 0.5)));
        let kinds: Vec<_> = s.markers_in_window();
        assert_eq!(
            kinds,
            vec![
                (0, MarkerKind::Growth),
                (1, MarkerKind::Pruning),
                (1, MarkerKind::Rollback)
            ]
        );
        // Newest first, with the reason kept
        let seqs: Vec<_> = s.events_newest_first().map(|e| e.seq).collect();
        assert_eq!(seqs, vec![2, 1, 0]);
        assert_eq!(s.events_newest_first().next().unwrap().reason, "test");
    }

    #[test]
    fn a_hole_in_the_sequence_is_counted_and_a_repeat_ignored() {
        let mut s = StructureState::new();
        s.apply(&report(
            vec![event(0, ActionKind::AddSynapse, Outcome::Applied)],
            5,
        ));
        s.apply(&report(
            vec![event(4, ActionKind::AddSynapse, Outcome::Applied)],
            5,
        ));
        assert_eq!(s.missed_events, 3);
        s.apply(&report(
            vec![event(4, ActionKind::AddSynapse, Outcome::Applied)],
            5,
        ));
        assert_eq!((s.applied, s.event_count(), s.missed_events), (2, 2, 3));
        let f = s.findings(&History::new(4));
        assert!(
            f.iter()
                .any(|f| f.what.contains("3 structural events were lost")),
            "{f:?}"
        );
    }

    #[test]
    fn the_event_list_and_markers_are_bounded() {
        let mut s = StructureState::new();
        for i in 0..(EVENT_LOG as u64 + 50) {
            s.apply(&report(
                vec![event(i, ActionKind::AddSynapse, Outcome::Applied)],
                5,
            ));
        }
        assert_eq!(s.event_count(), EVENT_LOG);
        assert_eq!(
            s.events_newest_first().next().unwrap().seq,
            EVENT_LOG as u64 + 49
        );
        assert_eq!(
            s.applied,
            EVENT_LOG as u64 + 50,
            "counters are not bounded by the list"
        );
        for i in 0..(STRUCTURE_HISTORY as u64 * 2) {
            s.apply(&report(vec![], 5));
            let _ = i;
        }
        assert!(
            s.markers_in_window().is_empty(),
            "markers older than the history are gone"
        );
        assert_eq!(s.neurons.values().len(), STRUCTURE_HISTORY);
    }

    #[test]
    fn findings_flag_growth_without_gain() {
        let mut s = StructureState::new();
        let evs: Vec<_> = (0..8)
            .map(|i| event(i, ActionKind::AddSynapse, Outcome::Applied))
            .collect();
        s.apply(&report(evs, 19));
        let mut loss = History::new(16);
        (0..8).for_each(|_| loss.push(0.9));
        let f = s.findings(&loss);
        assert_eq!(f[0].severity, Severity::Problem);
        assert!(f[0].what.contains("grows without improvement"));
    }

    /// The whole path of the mechanism: the real controller on a thread-less run, its reports
    /// through the real channel, folded by the same state the GUI owns.
    #[test]
    fn a_real_run_is_folded_end_to_end() {
        let (tx, rx) = channel(
            ChannelConfig::new("e2e")
                .with_capacity(4096)
                .with_frame_interval(Duration::ZERO),
        );
        let summary = structural_demo::run(
            &tx,
            &DemoConfig {
                pace: Duration::ZERO,
                ..Default::default()
            },
        );
        let mut state = crate::engine::training::TrainingState::new();
        let mut structure_events = 0;
        while let Ok(e) = rx.events.try_recv() {
            structure_events += matches!(e, TrainEvent::Structure(_)) as usize;
            state.apply(e);
        }
        let s = &state.structure;
        assert_eq!(s.reports as usize, structure_events);
        assert_eq!(s.missed_events, 0);
        assert!(s.applied as usize >= summary.applied.min(3));
        let last = s.stats.as_ref().unwrap();
        // The counters and the graph agree
        let snap = s.topology.as_ref().unwrap();
        assert_eq!(
            (snap.total_neurons, snap.total_synapses),
            (last.neurons, last.synapses)
        );
        // The dead neuron went, synapses grew, and the loss fell as the events say
        let removed = s
            .events_newest_first()
            .any(|e| e.kind == ActionKind::RemoveNeuron && e.outcome == Outcome::Applied);
        let grown = s
            .events_newest_first()
            .filter(|e| e.kind == ActionKind::AddSynapse && e.outcome == Outcome::Applied)
            .count();
        assert!(removed && grown >= 2, "removed {removed}, grown {grown}");
        assert!(state.loss.last().unwrap() < state.loss.values()[0] - 0.5);
        // Every event has its reason; the markers cover every applied growth/pruning
        assert!(s.events_newest_first().all(|e| !e.reason.is_empty()));
        assert!(!s.markers_in_window().is_empty());
        // A healthy finished run has no problems
        assert_ne!(
            neo_snn::telemetry::overall(&state.findings()),
            Severity::Problem,
            "{:?}",
            state.findings()
        );
    }

    #[test]
    fn event_rows_and_summary_say_what_happened() {
        let mut s = StructureState::new();
        assert!(s.summary().is_empty());
        let mut e = event(7, ActionKind::SplitSynapse, Outcome::RolledBack);
        e.action = "split_synapse s3".into();
        let row = EventRow::of(&e);
        assert_eq!(row.marker, Some(MarkerKind::Rollback));
        assert_eq!(row.head, "#7 · step 7 · split_synapse s3");
        assert_eq!(row.loss.as_deref(), Some("loss 1.000 → 0.500"));
        assert_eq!(row.size, "neurons 4 → 5 · synapses 6 → 7");
        assert!(row.outcome.starts_with("rolled back"));
        let failed = EventRow::of(&event(
            8,
            ActionKind::RemoveNeuron,
            Outcome::Failed("no neuron".into()),
        ));
        assert_eq!(failed.outcome, "refused: no neuron");
        let mut unmeasured = event(9, ActionKind::MutateNeuron, Outcome::Applied);
        unmeasured.loss_before = None;
        assert_eq!(
            (
                EventRow::of(&unmeasured).marker,
                EventRow::of(&unmeasured).loss
            ),
            (None, None)
        );

        let mut r = report(vec![e], 6);
        r.skipped = Some("Verifying".into());
        s.apply(&r);
        let sum: std::collections::BTreeMap<_, _> = s.summary().into_iter().collect();
        assert_eq!(sum["Neurons"], "6 of 20 · 0 working · 2 hidden");
        assert_eq!(sum["Changes"], "0 applied · 1 rolled back · 0 refused");
        assert_eq!(sum["Loss around the last change"], "1.000 → 0.500");
        assert_eq!(sum["Last analysis"], "verifying the previous change");
        assert!(!sum.contains_key("Events lost"));
    }
}
