//! What the dashboard knows about a run: folded from [`TrainEvent`]s, one at a time.
//!
//! Holds the metric histories, the latest spike frame of every layer and a bounded history of
//! frames to scrub back through (the old terminal dashboard's `←` / `→`), and answers the health
//! question ([`TrainingState::findings`]). Pure data: the view model owns one and calls
//! [`TrainingState::apply`] for every event that arrives.

use std::collections::{BTreeMap, VecDeque};
use std::sync::Arc;
use std::time::{Duration, Instant};

use neo_snn::demo::{
    LAYER_HIDDEN, LAYER_INPUT, LAYER_OUTPUT, WEIGHTS_HIDDEN_OUTPUT, WEIGHTS_INPUT_HIDDEN,
};
use neo_snn::functional::ActivityStats;
use neo_snn::telemetry::{Finding, History, Progress, Severity, SpikeFrame, TrainEvent, assess};

use super::brain::{BrainCamera, BrainLayer, BrainRequest, RASTER_FRAMES};
use super::dashboard::{DashboardRequest, WeightMatrix};
use super::evolution::EvolutionState;
use super::links::{Links, MAX_LINKS, Regions, SparseLinks};
use super::network::{NetCamera, NetLayer, NetworkRequest};
use super::structure::StructureState;
use crate::engine::palette::Palette;

/// Metric values kept per curve.
pub const METRIC_HISTORY: usize = 4096;
/// Spike frames kept per layer for scrubbing back.
pub const FRAME_HISTORY: usize = 48;
/// Hidden-layer frames the state-space (PCA) plot of the dashboard looks back over.
pub const PCA_FRAMES: usize = 12;
/// Memory ceiling of one layer's frame history (the newest frame is always kept).
pub const FRAME_BUDGET_BYTES: usize = 256 * 1024 * 1024;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum RunStatus {
    /// Nothing started yet.
    #[default]
    Idle,
    Running,
    Paused,
    /// Ran to the end.
    Finished,
    /// Ended on the user's request.
    Stopped,
}

impl RunStatus {
    /// A run is in progress (Running or Paused).
    pub fn active(self) -> bool {
        matches!(self, RunStatus::Running | RunStatus::Paused)
    }

    pub fn label(self) -> &'static str {
        match self {
            RunStatus::Idle => "Idle",
            RunStatus::Running => "Running",
            RunStatus::Paused => "Paused",
            RunStatus::Finished => "Finished",
            RunStatus::Stopped => "Stopped",
        }
    }
}

/// A spike frame with where in the run it was taken.
#[derive(Debug, Clone)]
pub struct StoredFrame {
    pub id: u64,
    pub progress: Progress,
    pub frame: Arc<SpikeFrame>,
    bytes: usize,
}

/// Which frame of a layer is shown.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FrameView {
    /// Always the newest.
    Live,
    /// Pinned to the frame with this id; newer frames do not move it.
    Pinned(u64),
}

/// One layer's spike frames.
#[derive(Debug, Clone)]
pub struct LayerState {
    pub name: String,
    live: Option<StoredFrame>,
    ring: VecDeque<StoredFrame>,
    ring_bytes: usize,
    next_id: u64,
    pub view: FrameView,
}

impl LayerState {
    fn new(name: &str) -> Self {
        Self {
            name: name.to_owned(),
            live: None,
            ring: VecDeque::new(),
            ring_bytes: 0,
            next_id: 0,
            view: FrameView::Live,
        }
    }

    fn push(&mut self, progress: Progress, frame: Arc<SpikeFrame>, stride: usize) {
        let id = self.next_id;
        self.next_id += 1;
        let stored = StoredFrame {
            id,
            progress,
            bytes: frame.size_bytes(),
            frame,
        };
        if FRAME_HISTORY > 0 && id.is_multiple_of(stride.max(1) as u64) {
            self.ring_bytes += stored.bytes;
            self.ring.push_back(stored.clone());
            while self.ring.len() > FRAME_HISTORY
                || (self.ring_bytes > FRAME_BUDGET_BYTES && self.ring.len() > 1)
            {
                match self.ring.pop_front() {
                    Some(old) => self.ring_bytes = self.ring_bytes.saturating_sub(old.bytes),
                    None => break,
                }
            }
        }
        self.live = Some(stored);
    }

    /// The frame on screen.
    pub fn shown(&self) -> Option<&StoredFrame> {
        match self.view {
            FrameView::Live => self.live.as_ref(),
            // A pinned frame that fell out of the history: the oldest newer one that is still
            // there, else the live one
            FrameView::Pinned(id) => self.ring.iter().find(|e| e.id >= id).or(self.live.as_ref()),
        }
    }

    /// `(position, length)` of the shown frame in the history, if it is in it.
    pub fn position(&self) -> Option<(usize, usize)> {
        let id = self.shown()?.id;
        let idx = self.ring.iter().position(|e| e.id == id)?;
        Some((idx + 1, self.ring.len()))
    }

    pub fn is_live(&self) -> bool {
        self.view == FrameView::Live
    }

    /// The newest `n` frames of the history, oldest first.
    pub fn recent_frames(&self, n: usize) -> Vec<Arc<SpikeFrame>> {
        let mut frames: Vec<Arc<SpikeFrame>> = self
            .ring
            .iter()
            .rev()
            .take(n)
            .map(|e| e.frame.clone())
            .collect();
        frames.reverse();
        frames
    }

    #[cfg(test)]
    pub fn history_len(&self) -> usize {
        self.ring.len()
    }

    fn current_id(&self) -> Option<u64> {
        match self.view {
            FrameView::Live => self.live.as_ref().map(|s| s.id),
            FrameView::Pinned(id) => Some(id),
        }
    }

    /// One frame back in the history.
    pub fn step_back(&mut self) {
        if let Some(cur) = self.current_id()
            && let Some(prev) = self.ring.iter().rev().find(|e| e.id < cur)
        {
            self.view = FrameView::Pinned(prev.id);
        }
    }

    /// One frame forward; past the newest one in the history it follows the live frame again.
    pub fn step_forward(&mut self) {
        if let FrameView::Pinned(cur) = self.view {
            match self.ring.iter().find(|e| e.id > cur) {
                Some(next) => self.view = FrameView::Pinned(next.id),
                None => self.view = FrameView::Live,
            }
        }
    }

    pub fn go_live(&mut self) {
        self.view = FrameView::Live;
    }
}

/// A whole run as the dashboard sees it.
#[derive(Debug, Clone)]
pub struct TrainingState {
    pub title: String,
    pub status: RunStatus,
    pub loss: History,
    pub accuracy: History,
    pub grad: History,
    pub progress: Progress,
    /// Steps received (one per `Step` event).
    pub steps_seen: u64,
    /// `Step` events whose loss was NaN or infinite (a [`History`] drops those silently).
    pub nonfinite_loss: u32,
    /// Latest `Activity` summary per name (`logits`, a layer, …).
    pub activity: BTreeMap<String, ActivityStats>,
    /// Latest weight matrix per name (`input→hidden`, …).
    pub weights: BTreeMap<String, WeightMatrix>,
    /// The same for layers sent as sparse link lists (big networks); preferred over a dense matrix of that name.
    pub sparse: BTreeMap<String, SparseLinks>,
    /// Regions the model named, per layer.
    pub regions: BTreeMap<String, Regions>,
    /// Spikes in all the frames that arrived (frames are samples of the run, not every step).
    pub spikes_seen: u64,
    /// Events the training thread dropped because this side did not keep up.
    pub dropped: u64,
    /// The structural plasticity of the network (empty when the run does not use it).
    pub structure: StructureState,
    pub evolution: EvolutionState,
    layers: Vec<LayerState>,
    selected: usize,
    /// Keep every n-th frame of a layer in its history.
    pub frame_stride: usize,
    last_step: Option<u64>,
    /// Times the step number went backwards: progress reported out of order.
    pub regressions: u32,
    started: Option<Instant>,
    ended: Option<Instant>,
    revision: u64,
}

impl Default for TrainingState {
    fn default() -> Self {
        Self::new()
    }
}

impl TrainingState {
    pub fn new() -> Self {
        Self {
            title: String::new(),
            status: RunStatus::Idle,
            loss: History::new(METRIC_HISTORY),
            accuracy: History::new(METRIC_HISTORY),
            grad: History::new(METRIC_HISTORY),
            progress: Progress::default(),
            steps_seen: 0,
            nonfinite_loss: 0,
            activity: BTreeMap::new(),
            weights: BTreeMap::new(),
            sparse: BTreeMap::new(),
            regions: BTreeMap::new(),
            spikes_seen: 0,
            dropped: 0,
            structure: StructureState::new(),
            evolution: EvolutionState::new(),
            layers: Vec::new(),
            selected: 0,
            frame_stride: 1,
            last_step: None,
            regressions: 0,
            started: None,
            ended: None,
            revision: 0,
        }
    }

    /// Changes with every update of the data: what a view compares to know it must redraw.
    pub fn revision(&self) -> u64 {
        self.revision
    }

    /// Starts over for a new run (the title and the frame stride survive).
    pub fn reset(&mut self, title: &str) {
        let (stride, revision) = (self.frame_stride, self.revision);
        *self = Self::new();
        self.title = title.to_owned();
        self.frame_stride = stride;
        self.revision = revision + 1;
    }

    /// Folds one event in.
    pub fn apply(&mut self, event: TrainEvent) {
        self.revision += 1;
        match event {
            TrainEvent::Started { title } => {
                self.reset(&title);
                self.status = RunStatus::Running;
                self.started = Some(Instant::now());
            }
            TrainEvent::Step(m) => {
                if self.status == RunStatus::Idle {
                    self.status = RunStatus::Running;
                    self.started.get_or_insert_with(Instant::now);
                }
                let step = m.progress.step();
                if self.last_step.is_some_and(|prev| step < prev) {
                    self.regressions += 1;
                }
                self.last_step = Some(step);
                self.progress = m.progress;
                self.steps_seen += 1;
                if m.loss.is_finite() {
                    self.loss.push(m.loss);
                } else {
                    self.nonfinite_loss += 1;
                }
                if let Some(a) = m.accuracy {
                    self.accuracy.push(a);
                }
                if let Some(g) = m.grad_norm {
                    self.grad.push(g);
                }
            }
            TrainEvent::Spikes {
                layer,
                progress,
                frame,
            } => {
                let stride = self.frame_stride;
                self.spikes_seen += frame.total_spikes();
                let i = match self.layers.iter().position(|l| l.name == layer) {
                    Some(i) => i,
                    None => {
                        self.layers.push(LayerState::new(&layer));
                        self.layers.len() - 1
                    }
                };
                self.layers[i].push(progress, frame, stride);
            }
            TrainEvent::Activity { layer, stats } => {
                self.activity.insert(layer, stats);
            }
            TrainEvent::Weights {
                layer,
                rows,
                cols,
                values,
                ..
            } => {
                self.weights
                    .insert(layer, WeightMatrix { rows, cols, values });
            }
            TrainEvent::Links {
                layer,
                rows,
                cols,
                from,
                to,
                weight,
                ..
            } => {
                // A list longer than the ceiling would make every frame slow: keep the previous one
                if from.len().min(to.len()).min(weight.len()) <= MAX_LINKS {
                    self.sparse.insert(
                        layer,
                        SparseLinks {
                            rows,
                            cols,
                            from,
                            to,
                            weight,
                        },
                    );
                }
            }
            TrainEvent::Regions {
                layer,
                names,
                labels,
            } => {
                self.regions.insert(layer, Regions { names, labels });
            }
            TrainEvent::Structure(report) => self.structure.apply(&report),
            TrainEvent::Evolution(report) => self.evolution.apply(&report),
            TrainEvent::Finished { stopped } => {
                self.status = if stopped {
                    RunStatus::Stopped
                } else {
                    RunStatus::Finished
                };
                self.ended = Some(Instant::now());
            }
        }
    }

    /// Pause / resume as the user asked. Only a run in progress changes.
    pub fn set_paused(&mut self, paused: bool) {
        match (self.status, paused) {
            (RunStatus::Running, true) => self.status = RunStatus::Paused,
            (RunStatus::Paused, false) => self.status = RunStatus::Running,
            _ => return,
        }
        self.revision += 1;
    }

    pub fn set_dropped(&mut self, dropped: u64) {
        if self.dropped != dropped {
            self.dropped = dropped;
            self.revision += 1;
        }
    }

    // ─── layers and frames ──────────────────────────────────────────────

    pub fn layers(&self) -> &[LayerState] {
        &self.layers
    }

    pub fn selected_index(&self) -> Option<usize> {
        (!self.layers.is_empty()).then(|| self.selected.min(self.layers.len() - 1))
    }

    /// The layer called `name`, if its frames have arrived.
    pub fn layer_named(&self, name: &str) -> Option<&LayerState> {
        self.layers.iter().find(|l| l.name == name)
    }

    pub fn selected_layer(&self) -> Option<&LayerState> {
        self.selected_index().map(|i| &self.layers[i])
    }

    pub fn select_layer(&mut self, index: usize) {
        if index < self.layers.len() && index != self.selected {
            self.selected = index;
            self.revision += 1;
        }
    }

    fn with_selected(&mut self, f: impl FnOnce(&mut LayerState)) {
        if let Some(i) = self.selected_index() {
            f(&mut self.layers[i]);
            self.revision += 1;
        }
    }

    pub fn step_back(&mut self) {
        self.with_selected(LayerState::step_back);
    }

    pub fn step_forward(&mut self) {
        self.with_selected(LayerState::step_forward);
    }

    pub fn go_live(&mut self) {
        self.with_selected(LayerState::go_live);
    }

    /// The frame on screen for the selected layer.
    pub fn shown_frame(&self) -> Option<&StoredFrame> {
        self.selected_layer()?.shown()
    }

    /// What the dashboard image is drawn from: the shown frame of each layer, the weights, the
    /// recent hidden frames for the PCA and the accuracy curve. All shared, so it can go to the
    /// work thread.
    pub fn dashboard_request(
        &self,
        width: u32,
        height: u32,
        scale: f32,
        palette: Palette,
    ) -> DashboardRequest {
        let shown = |name: &str| {
            self.layer_named(name)
                .and_then(|l| l.shown())
                .map(|s| s.frame.clone())
        };
        DashboardRequest {
            input: shown(LAYER_INPUT),
            hidden: shown(LAYER_HIDDEN),
            output: shown(LAYER_OUTPUT),
            w_input_hidden: self.weights.get(WEIGHTS_INPUT_HIDDEN).cloned(),
            w_hidden_output: self.weights.get(WEIGHTS_HIDDEN_OUTPUT).cloned(),
            pca_frames: self
                .layer_named(LAYER_HIDDEN)
                .map(|l| l.recent_frames(PCA_FRAMES))
                .unwrap_or_default(),
            accuracy: self.accuracy.values().to_vec(),
            width,
            height,
            scale,
            palette,
        }
    }

    /// What the network view is drawn from: the three layers with their shown frames, the two weight
    /// matrices between them, and where the camera looks. A layer whose frames have not arrived yet
    /// still gets its size from the weights.
    pub fn network_request(
        &self,
        width: u32,
        height: u32,
        scale: f32,
        camera: NetCamera,
        selected: Option<(usize, usize)>,
    ) -> NetworkRequest {
        let shown = |name: &str| {
            self.layer_named(name)
                .and_then(|l| l.shown())
                .map(|s| s.frame.clone())
        };
        let (w_in, w_out) = (
            self.connections(WEIGHTS_INPUT_HIDDEN),
            self.connections(WEIGHTS_HIDDEN_OUTPUT),
        );
        let (input, hidden, output) =
            (shown(LAYER_INPUT), shown(LAYER_HIDDEN), shown(LAYER_OUTPUT));
        let size = |frame: &Option<Arc<SpikeFrame>>, from_weights: usize| {
            frame.as_ref().map_or(from_weights, |f| f.units)
        };
        let n_in = size(&input, w_in.as_ref().map_or(0, |m| m.rows()));
        let n_hidden = size(
            &hidden,
            w_in.as_ref()
                .map_or(w_out.as_ref().map_or(0, |m| m.rows()), |m| m.cols()),
        );
        let n_out = size(&output, w_out.as_ref().map_or(0, |m| m.cols()));
        NetworkRequest {
            layers: vec![
                NetLayer {
                    units: n_in,
                    frame: input,
                },
                NetLayer {
                    units: n_hidden,
                    frame: hidden,
                },
                NetLayer {
                    units: n_out,
                    frame: output,
                },
            ],
            weights: vec![w_in, w_out],
            camera,
            selected,
            phase: self.revision() as f32 * 0.02,
            width,
            height,
            scale,
        }
    }

    /// The links of the pair of layers named `name`: the sparse list if one came, else the dense matrix.
    pub fn connections(&self, name: &str) -> Option<Links> {
        self.sparse
            .get(name)
            .cloned()
            .map(Links::Sparse)
            .or_else(|| self.weights.get(name).cloned().map(Links::Dense))
    }

    /// What the brain view is drawn from: for each layer the shown frame and the recent frames (usage
    /// and the raster), the weights between them, the camera and the picked neuron.
    pub fn brain_request(
        &self,
        width: u32,
        height: u32,
        scale: f32,
        camera: BrainCamera,
        selected: Option<(usize, usize)>,
    ) -> BrainRequest {
        let (w_in, w_out) = (
            self.connections(WEIGHTS_INPUT_HIDDEN),
            self.connections(WEIGHTS_HIDDEN_OUTPUT),
        );
        let layer = |name: &str, from_weights: usize| {
            let state = self.layer_named(name);
            let frame = state.and_then(|l| l.shown()).map(|s| s.frame.clone());
            let history = state
                .map(|l| l.recent_frames(RASTER_FRAMES))
                .unwrap_or_default();
            BrainLayer {
                units: frame.as_ref().map_or(from_weights, |f| f.units),
                frame,
                history,
                regions: self.regions.get(name).cloned(),
            }
        };
        let hidden_from_weights = w_in
            .as_ref()
            .map_or(w_out.as_ref().map_or(0, |m| m.rows()), |m| m.cols());
        BrainRequest {
            layers: vec![
                layer(LAYER_INPUT, w_in.as_ref().map_or(0, |m| m.rows())),
                layer(LAYER_HIDDEN, hidden_from_weights),
                layer(LAYER_OUTPUT, w_out.as_ref().map_or(0, |m| m.cols())),
            ],
            weights: vec![w_in, w_out],
            camera,
            selected,
            phase: self.revision() as f32 * 0.02,
            width,
            height,
            scale,
        }
    }

    // ─── time ───────────────────────────────────────────────────────────

    /// Time since the run started (frozen once it ended).
    pub fn elapsed(&self) -> Option<Duration> {
        let start = self.started?;
        Some(
            self.ended
                .unwrap_or_else(Instant::now)
                .duration_since(start),
        )
    }

    /// Fraction of the run done, `0..=1`.
    pub fn fraction_done(&self) -> f32 {
        let total = self.progress.total_steps();
        if total == 0 || self.steps_seen == 0 {
            return 0.0;
        }
        match self.status {
            RunStatus::Finished => 1.0,
            _ => ((self.progress.step() + 1) as f32 / total as f32).clamp(0.0, 1.0),
        }
    }

    // ─── health ─────────────────────────────────────────────────────────

    /// The verdict on the network, worst first: [`assess`] over the shown frame and the metric
    /// histories, plus the two things the histories hide (a loss that is not a number; progress
    /// reported out of order).
    pub fn findings(&self) -> Vec<Finding> {
        let mut out = self.training_findings();
        let mut structural = self.structure.findings(&self.loss);
        structural.extend(self.evolution.findings());
        // A structure with nothing to report adds nothing; otherwise its findings join the
        // list and the "all fine" line (which no longer holds) goes.
        if structural.iter().any(|f| f.severity != Severity::Ok) {
            out.retain(|f| f.severity != Severity::Ok);
            out.extend(
                structural
                    .into_iter()
                    .filter(|f| f.severity != Severity::Ok),
            );
            out.sort_by_key(|a| std::cmp::Reverse(a.severity));
        }
        out
    }

    /// [`assess`] over the shown frame and the metric histories, and the two things they hide.
    fn training_findings(&self) -> Vec<Finding> {
        let empty = SpikeFrame::default();
        let frame = self.shown_frame().map_or(&empty, |s| s.frame.as_ref());
        let mut out = assess(frame, &self.loss, &self.accuracy, &self.grad);

        let mut extra = Vec::new();
        if self.nonfinite_loss > 0 {
            extra.push(Finding {
                severity: Severity::Problem,
                what: format!(
                    "loss is not a number (NaN/inf) per {} step.",
                    self.nonfinite_loss
                ),
                advice: "training diverged: reduce lr, enable gradient norm constraint".into(),
            });
        }
        if self.regressions > 0 {
            extra.push(Finding {
                severity: Severity::Notice,
                what: "step number went backwards".into(),
                advice: "progress in TrainTx::step should grow in batch order".into(),
            });
        }
        if !extra.is_empty() {
            // The "all fine" line no longer holds
            // (A loss that is not a number outranks every other finding of its severity.)
            out.retain(|f| f.severity != Severity::Ok);
            extra.append(&mut out);
            extra.sort_by_key(|a| std::cmp::Reverse(a.severity));
            return extra;
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use neo_snn::telemetry::{StepMetrics, overall};

    fn step(epoch: usize, batch: usize, loss: f32) -> TrainEvent {
        TrainEvent::Step(StepMetrics {
            progress: Progress {
                epoch,
                epochs: 2,
                batch,
                batches: 3,
            },
            loss,
            accuracy: Some(0.5),
            grad_norm: Some(1.0),
        })
    }

    fn spikes(layer: &str, step: usize, on: bool) -> TrainEvent {
        TrainEvent::Spikes {
            layer: layer.into(),
            progress: Progress {
                epoch: 0,
                epochs: 1,
                batch: step,
                batches: 100,
            },
            frame: Arc::new(SpikeFrame::from_bits(4, 4, vec![on; 16])),
        }
    }

    #[test]
    fn steps_fill_histories_and_progress() {
        let mut s = TrainingState::new();
        s.apply(TrainEvent::Started {
            title: "run".into(),
        });
        assert_eq!(s.status, RunStatus::Running);
        for (i, l) in [1.0, 0.5, 0.25].into_iter().enumerate() {
            s.apply(step(0, i, l));
        }
        assert_eq!(s.loss.values(), [1.0, 0.5, 0.25]);
        assert_eq!(s.accuracy.total(), 3);
        assert_eq!(s.steps_seen, 3);
        assert_eq!(s.progress.batch, 2);
        assert!((s.fraction_done() - 0.5).abs() < 1e-6, "3 of 6 steps");
        s.apply(TrainEvent::Finished { stopped: false });
        assert_eq!(s.status, RunStatus::Finished);
        assert_eq!(s.fraction_done(), 1.0);
    }

    #[test]
    fn a_new_run_starts_over_but_keeps_settings() {
        let mut s = TrainingState::new();
        s.frame_stride = 5;
        s.apply(step(0, 0, 1.0));
        s.apply(spikes("hidden", 0, true));
        let rev = s.revision();
        s.apply(TrainEvent::Started {
            title: "second".into(),
        });
        assert!(s.loss.is_empty() && s.layers().is_empty());
        assert_eq!((s.title.as_str(), s.frame_stride), ("second", 5));
        assert!(s.revision() > rev);
    }

    #[test]
    fn every_event_changes_the_revision() {
        let mut s = TrainingState::new();
        let mut last = s.revision();
        for e in [
            step(0, 0, 1.0),
            spikes("h", 0, true),
            TrainEvent::Finished { stopped: true },
        ] {
            s.apply(e);
            assert!(s.revision() > last);
            last = s.revision();
        }
        assert_eq!(s.status, RunStatus::Stopped);
    }

    #[test]
    fn pause_only_changes_a_running_run() {
        let mut s = TrainingState::new();
        s.set_paused(true);
        assert_eq!(s.status, RunStatus::Idle);
        s.apply(TrainEvent::Started { title: "t".into() });
        s.set_paused(true);
        assert_eq!(s.status, RunStatus::Paused);
        s.set_paused(false);
        assert_eq!(s.status, RunStatus::Running);
        s.apply(TrainEvent::Finished { stopped: false });
        s.set_paused(true);
        assert_eq!(s.status, RunStatus::Finished);
    }

    #[test]
    fn layers_are_created_on_first_frame_and_selected() {
        let mut s = TrainingState::new();
        assert!(s.selected_layer().is_none() && s.shown_frame().is_none());
        s.apply(spikes("hidden", 0, true));
        s.apply(spikes("output", 0, false));
        s.apply(spikes("hidden", 1, true));
        let names: Vec<_> = s.layers().iter().map(|l| l.name.as_str()).collect();
        assert_eq!(names, ["hidden", "output"]);
        assert_eq!(s.selected_layer().unwrap().name, "hidden");
        s.select_layer(1);
        assert_eq!(s.selected_layer().unwrap().name, "output");
        s.select_layer(9); // out of range: ignored
        assert_eq!(s.selected_index(), Some(1));
    }

    #[test]
    fn scrubbing_back_pins_a_frame_and_live_follows_again() {
        let mut s = TrainingState::new();
        for i in 0..5 {
            s.apply(spikes("h", i, true));
        }
        let live_id = s.shown_frame().unwrap().id;
        assert_eq!(live_id, 4);
        s.step_back();
        s.step_back();
        assert_eq!(s.shown_frame().unwrap().id, 2);
        assert_eq!(s.selected_layer().unwrap().position(), Some((3, 5)));
        s.apply(spikes("h", 5, true)); // a newer frame does not move a pinned view
        assert_eq!(s.shown_frame().unwrap().id, 2);
        s.step_forward();
        s.step_forward();
        s.step_forward();
        s.step_forward();
        assert!(
            s.selected_layer().unwrap().is_live(),
            "past the newest → live again"
        );
        assert_eq!(s.shown_frame().unwrap().id, 5);
        s.step_back();
        s.go_live();
        assert!(s.selected_layer().unwrap().is_live());
    }

    #[test]
    fn stepping_back_from_the_oldest_stays_put() {
        let mut s = TrainingState::new();
        s.apply(spikes("h", 0, true));
        s.step_back();
        assert!(s.selected_layer().unwrap().is_live(), "no frame before");
    }

    #[test]
    fn frame_history_is_bounded_and_respects_the_stride() {
        let mut s = TrainingState::new();
        s.frame_stride = 2;
        for i in 0..(FRAME_HISTORY * 4) {
            s.apply(spikes("h", i, true));
        }
        let layer = s.selected_layer().unwrap();
        assert_eq!(layer.history_len(), FRAME_HISTORY);
        assert!(
            layer.ring.iter().all(|f| f.id % 2 == 0),
            "only every second frame"
        );
        assert_eq!(
            layer.live.as_ref().unwrap().id as usize,
            FRAME_HISTORY * 4 - 1,
            "there is always the last frame"
        );
    }

    #[test]
    fn a_pinned_frame_that_was_evicted_falls_forward() {
        let mut s = TrainingState::new();
        for i in 0..3 {
            s.apply(spikes("h", i, true));
        }
        s.step_back();
        s.step_back();
        assert_eq!(s.shown_frame().unwrap().id, 0);
        for i in 3..(FRAME_HISTORY + 10) {
            s.apply(spikes("h", i, true));
        }
        let shown = s.shown_frame().unwrap().id;
        assert!(
            shown > 0,
            "frame 0 is supplanted, the oldest remaining one is shown"
        );
        assert_eq!(
            shown as usize, 10,
            "FRAME_HISTORY + 10 frames, last remaining FRAME_HISTORY"
        );
    }

    #[test]
    fn healthy_and_dead_layers_get_different_verdicts() {
        let mut s = TrainingState::new();
        s.apply(TrainEvent::Spikes {
            layer: "h".into(),
            progress: Progress::default(),
            frame: Arc::new(SpikeFrame::from_fn(24, 8, |t, u| (t + u) % 3 == 0)),
        });
        assert_eq!(overall(&s.findings()), Severity::Ok);
        s.apply(spikes("h", 1, false)); // the layer went silent
        let f = s.findings();
        assert_eq!(overall(&f), Severity::Problem);
        assert!(f[0].what.contains("does not spike"));
    }

    #[test]
    fn scrubbing_back_judges_the_older_frame() {
        let mut s = TrainingState::new();
        s.apply(spikes("h", 0, true)); // saturated
        s.apply(spikes("h", 1, false)); // dead
        assert!(s.findings()[0].what.contains("does not spike"));
        s.step_back();
        assert!(
            s.findings()[0].what.contains("every step"),
            "{:?}",
            s.findings()
        );
    }

    #[test]
    fn a_nan_loss_is_reported_not_swallowed() {
        let mut s = TrainingState::new();
        s.apply(spikes("h", 0, true));
        s.apply(step(0, 0, f32::NAN));
        assert_eq!(s.loss.total(), 0, "History discards non-numbers");
        assert_eq!(s.nonfinite_loss, 1);
        let f = s.findings();
        assert_eq!(f[0].severity, Severity::Problem);
        assert!(f[0].what.contains("NaN"));
        assert!(f.iter().all(|x| x.severity != Severity::Ok));
    }

    #[test]
    fn out_of_order_progress_is_noticed() {
        let mut s = TrainingState::new();
        s.apply(step(0, 2, 1.0));
        s.apply(step(0, 1, 1.0));
        assert_eq!(s.regressions, 1);
        assert!(s.findings().iter().any(|f| f.what.contains("back")));
    }

    #[test]
    fn activity_keeps_the_latest_summary_per_name() {
        let mut s = TrainingState::new();
        let st = |mean| ActivityStats {
            mean,
            abs_max: 1.0,
            zero_fraction: 0.0,
            count: 4,
        };
        s.apply(TrainEvent::Activity {
            layer: "logits".into(),
            stats: st(0.1),
        });
        s.apply(TrainEvent::Activity {
            layer: "logits".into(),
            stats: st(0.2),
        });
        assert_eq!(s.activity.len(), 1);
        assert_eq!(s.activity["logits"].mean, 0.2);
    }

    #[test]
    fn weights_are_kept_per_name_and_the_latest_wins() {
        let mut s = TrainingState::new();
        let w = |name: &str, v: f32| TrainEvent::Weights {
            layer: name.into(),
            progress: Progress::default(),
            rows: 1,
            cols: 2,
            values: Arc::new(vec![v, v]),
        };
        s.apply(w("a", 1.0));
        s.apply(w("a", 2.0));
        s.apply(w("b", 3.0));
        assert_eq!(s.weights.len(), 2);
        assert_eq!(s.weights["a"].values[0], 2.0);
        assert_eq!((s.weights["b"].rows, s.weights["b"].cols), (1, 2));
    }

    #[test]
    fn dashboard_request_picks_each_layer_and_the_recent_hidden_frames() {
        let mut s = TrainingState::new();
        for step in 0..3 {
            s.apply(spikes(LAYER_HIDDEN, step, true));
        }
        s.apply(spikes(LAYER_INPUT, 0, false));
        let req = s.dashboard_request(100, 50, 2.0, Palette::DARK);
        assert!(req.hidden.is_some() && req.input.is_some() && req.output.is_none());
        assert_eq!(req.pca_frames.len(), 3);
        assert_eq!((req.width, req.height, req.scale), (100, 50, 2.0));
        assert!(req.w_input_hidden.is_none());
    }

    #[test]
    fn network_request_takes_sparse_links_for_a_big_network() {
        let mut s = TrainingState::new();
        let links = |name: &str, rows: usize, cols: usize| TrainEvent::Links {
            layer: name.into(),
            progress: Progress::default(),
            rows,
            cols,
            from: Arc::new(vec![0, 1]),
            to: Arc::new(vec![1, 0]),
            weight: Arc::new(vec![0.5, -0.5]),
        };
        s.apply(links(WEIGHTS_INPUT_HIDDEN, 2_000, 1_000_000));
        s.apply(links(WEIGHTS_HIDDEN_OUTPUT, 1_000_000, 40));
        let req = s.network_request(10, 10, 1.0, NetCamera::default(), None);
        assert_eq!(
            req.layers.iter().map(|l| l.units).collect::<Vec<_>>(),
            vec![2_000, 1_000_000, 40]
        );
        assert!(
            req.weights
                .iter()
                .all(|w| matches!(w, Some(Links::Sparse(_))))
        );
    }

    #[test]
    fn network_request_sizes_a_layer_from_its_weights_before_its_frames_arrive() {
        let mut s = TrainingState::new();
        let w = |name: &str, rows: usize, cols: usize| TrainEvent::Weights {
            layer: name.into(),
            progress: Progress::default(),
            rows,
            cols,
            values: Arc::new(vec![0.1; rows * cols]),
        };
        s.apply(w(WEIGHTS_INPUT_HIDDEN, 8, 24));
        s.apply(w(WEIGHTS_HIDDEN_OUTPUT, 24, 2));
        let req = s.network_request(10, 10, 1.0, NetCamera::default(), Some((1, 3)));
        assert_eq!(
            req.layers.iter().map(|l| l.units).collect::<Vec<_>>(),
            vec![8, 24, 2]
        );
        assert!(req.layers.iter().all(|l| l.frame.is_none()));
        assert!(req.weights.iter().all(Option::is_some));
        assert_eq!(req.selected, Some((1, 3)));
    }

    #[test]
    fn sparse_links_and_regions_are_kept_and_sparse_wins_over_dense() {
        let mut s = TrainingState::new();
        let dense = TrainEvent::Weights {
            layer: "a→b".into(),
            progress: Progress::default(),
            rows: 2,
            cols: 2,
            values: Arc::new(vec![1.0; 4]),
        };
        let links = |n: usize| TrainEvent::Links {
            layer: "a→b".into(),
            progress: Progress::default(),
            rows: 2,
            cols: 2,
            from: Arc::new(vec![0; n]),
            to: Arc::new(vec![1; n]),
            weight: Arc::new(vec![0.5; n]),
        };
        s.apply(dense);
        assert!(matches!(s.connections("a→b"), Some(Links::Dense(_))));
        s.apply(links(3));
        assert!(matches!(s.connections("a→b"), Some(Links::Sparse(ref l)) if l.weight.len() == 3));
        s.apply(links(MAX_LINKS + 1)); // too long: the previous list stays
        assert!(matches!(s.connections("a→b"), Some(Links::Sparse(ref l)) if l.weight.len() == 3));
        s.apply(TrainEvent::Regions {
            layer: "b".into(),
            names: Arc::new(vec!["optic".into()]),
            labels: Arc::new(vec![0, 0]),
        });
        assert_eq!(s.regions["b"].names.len(), 1);
        assert!(s.connections("missing").is_none());
    }

    #[test]
    fn spikes_are_counted_across_frames_and_layers() {
        let mut s = TrainingState::new();
        s.apply(spikes(LAYER_HIDDEN, 0, true)); // 4 steps x 4 units, all firing
        s.apply(spikes(LAYER_INPUT, 0, true));
        s.apply(spikes(LAYER_HIDDEN, 1, false));
        assert_eq!(s.spikes_seen, 32);
    }

    #[test]
    fn brain_request_carries_recent_frames_per_layer() {
        let mut s = TrainingState::new();
        for step in 0..3 {
            s.apply(spikes(LAYER_HIDDEN, step, true));
        }
        s.apply(spikes(LAYER_OUTPUT, 0, true));
        let req = s.brain_request(10, 10, 1.0, BrainCamera::default(), None);
        assert_eq!(req.layers.len(), 3);
        assert_eq!(req.layers[1].history.len(), 3);
        assert_eq!(req.layers[2].history.len(), 1);
        assert!(req.layers[0].frame.is_none() && req.layers[0].history.is_empty());
        assert_eq!(req.layers[1].units, 4);
        assert!(req.layers.iter().all(|l| l.regions.is_none()));
    }

    #[test]
    fn empty_state_is_harmless() {
        let s = TrainingState::new();
        assert_eq!(overall(&s.findings()), Severity::Ok);
        assert_eq!(s.fraction_done(), 0.0);
        assert!(s.elapsed().is_none());
        assert!(!RunStatus::Idle.active() && RunStatus::Paused.active());
    }
}
