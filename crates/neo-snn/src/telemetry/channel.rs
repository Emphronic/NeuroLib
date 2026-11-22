//! Training event channel and its management (pause, stop).

use std::sync::atomic::{AtomicBool, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use async_channel::{Receiver, Sender, TrySendError};
use burn::tensor::Tensor;

use super::evolution::EvolutionReport;
use super::frame::SpikeFrame;
use super::structure::StructureReport;
use crate::functional::ActivityStats;

/// Where is the training located? Numbers - from scratch.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Progress {
    pub epoch: usize,
    pub epochs: usize,
    pub batch: usize,
    pub batches: usize,
}

impl Progress {
    /// The end-to-end number of the step `epoch of batches + batch` (from scratch): only grows.
    pub fn step(&self) -> u64 {
        self.epoch as u64 * self.batches.max(1) as u64 + self.batch as u64
    }

    /// Total steps in training.
    pub fn total_steps(&self) -> u64 {
        self.epochs as u64 * self.batches.max(1) as u64
    }
}

/// Metrics of one learning step.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct StepMetrics {
    pub progress: Progress,
    pub loss: f32,
    /// Proportion of correct answers in `[0, 1]`, if computed.
    pub accuracy: Option<f32>,
    /// Gradient norm ([`crate::functional::grad_norm`]), if computed.
    pub grad_norm: Option<f32>,
}

/// What learning communicates outward.
#[derive(Debug, Clone)]
pub enum TrainEvent {
    /// Training has begun; `title` is the name of the experiment.
    Started { title: String },
    /// Step metrics.
    Step(StepMetrics),
    /// Spike activity of layer `layer` at step `progress`.
    ///
    /// Snapshot in `Arc`: the consumer keeps it in history and gives it to rendering threads without copying it.
    Spikes {
        layer: String,
        progress: Progress,
        frame: Arc<SpikeFrame>,
    },
    /// Tensor summary (mean, `abs_max`, sparsity) from [`crate::functional::probe`].
    Activity { layer: String, stats: ActivityStats },
    /// Result of structural analysis (`StructuralPlasticityController::adapt`): counters,
    /// new events with reasons, loss before/after, if desired, a snapshot of the graph.
    ///
    /// In `Arc` for the same reason as the spike snapshot. The numbers in the report are absolute, but for events
    /// there is `seq`: the loss of a report when the queue overflows is visible to the consumer as a break in `seq`.
    Structure(Arc<StructureReport>),
    /// The result of a generation of architecture evolution (`structural::evolution::Population::step`).
    Evolution(Arc<EvolutionReport>),
    /// Weight matrix `layer`, `rows × cols` (row - input, column - output), row by row.
    ///
    /// It is filmed with the same frames as spikes: the dashboard needs maps of weights and their distribution.
    Weights {
        layer: String,
        progress: Progress,
        rows: usize,
        cols: usize,
        values: Arc<Vec<f32>>,
    },
    /// Connections of layer `layer` in sparse shape: neuron `from[k]` (from `rows`) acts on `to[k]` (from `cols`)
    /// with weight `weight[k]`. For networks where a dense matrix would not fit into memory (tens and hundreds
    /// thousand neurons); are shown in the same way as dense weights.
    Links {
        layer: String,
        progress: Progress,
        rows: usize,
        cols: usize,
        from: Arc<Vec<u32>>,
        to: Arc<Vec<u32>>,
        weight: Arc<Vec<f32>>,
    },
    /// Regions of layer `layer`: neuron `u` lies in region `labels[u]`, region `r` has name `names[r]`
    /// (cell types, departments). It is sent once: the brain map colors and groups neurons by region.
    Regions {
        layer: String,
        names: Arc<Vec<String>>,
        labels: Arc<Vec<u16>>,
    },
    /// The training is over; `stopped` - interrupted by the user, but did not reach the end.
    Finished { stopped: bool },
}

/// Channel settings.
#[derive(Debug, Clone)]
pub struct ChannelConfig {
    /// Experiment name, goes to [`TrainEvent::Started`].
    pub title: String,
    /// Queue length. Overflow = consumer can't keep up: new metrics events
    /// discarded and counted, learning does not wait.
    pub capacity: usize,
    /// No more than one snapshot of spikes in such an interval ([`TrainTx::wants_frame`]).
    pub frame_interval: Duration,
}

impl ChannelConfig {
    pub fn new(title: impl Into<String>) -> Self {
        Self {
            title: title.into(),
            capacity: 512,
            frame_interval: Duration::from_millis(50),
        }
    }

    pub fn with_capacity(mut self, capacity: usize) -> Self {
        self.capacity = capacity.max(4);
        self
    }

    pub fn with_frame_interval(mut self, interval: Duration) -> Self {
        self.frame_interval = interval;
        self
    }
}

#[derive(Debug)]
struct Shared {
    stop: AtomicBool,
    paused: AtomicBool,
    dropped: AtomicU64,
    last_frame: Mutex<Option<Instant>>,
    frame_interval: Duration,
}

/// The learning side. Cheap to clone; can be kept in multiple threads.
#[derive(Debug, Clone)]
pub struct TrainTx {
    tx: Sender<TrainEvent>,
    shared: Arc<Shared>,
    title: Arc<str>,
}

/// Consumer side: events and control buttons.
#[derive(Debug, Clone)]
pub struct TrainRx {
    /// Flow of events. In GUI: `rx.events.recv().await`.
    pub events: Receiver<TrainEvent>,
    control: TrainControl,
}

/// Managing learning from the consumer: pause and stop. Learning obeys the step boundary
/// ([`TrainTx::checkpoint`]), and not in the middle of calculations, so that the checkpoint has time to register.
#[derive(Debug, Clone)]
pub struct TrainControl {
    shared: Arc<Shared>,
}

/// Create a channel. `TrainTx` - to the learning stream, `TrainRx` - to the consumer.
pub fn channel(config: ChannelConfig) -> (TrainTx, TrainRx) {
    let (tx, rx) = async_channel::bounded(config.capacity.max(4));
    let shared = Arc::new(Shared {
        stop: AtomicBool::new(false),
        paused: AtomicBool::new(false),
        dropped: AtomicU64::new(0),
        last_frame: Mutex::new(None),
        frame_interval: config.frame_interval,
    });
    let title: Arc<str> = config.title.into();
    (
        TrainTx {
            tx,
            shared: shared.clone(),
            title,
        },
        TrainRx {
            events: rx,
            control: TrainControl { shared },
        },
    )
}

impl TrainRx {
    pub fn control(&self) -> TrainControl {
        self.control.clone()
    }
}

impl TrainControl {
    /// Ask the training to stop at the nearest step boundary. Takes off the pause.
    pub fn stop(&self) {
        self.shared.stop.store(true, Ordering::Release);
        self.shared.paused.store(false, Ordering::Release);
    }

    pub fn set_paused(&self, paused: bool) {
        self.shared.paused.store(paused, Ordering::Release);
    }

    pub fn is_paused(&self) -> bool {
        self.shared.paused.load(Ordering::Acquire)
    }

    pub fn is_stopping(&self) -> bool {
        self.shared.stop.load(Ordering::Acquire)
    }

    /// How many events were discarded due to queue overflow.
    pub fn dropped(&self) -> u64 {
        self.shared.dropped.load(Ordering::Relaxed)
    }
}

impl TrainTx {
    fn emit(&self, event: TrainEvent) {
        match self.tx.try_send(event) {
            Ok(()) => {}
            Err(TrySendError::Full(_)) => {
                self.shared.dropped.fetch_add(1, Ordering::Relaxed);
            }
            Err(TrySendError::Closed(_)) => {}
        }
    }

    /// Notify about the start of training (name taken from [`ChannelConfig::title`]). Doesn't get lost.
    pub fn start(&self) {
        let _ = self.tx.send_blocking(TrainEvent::Started {
            title: self.title.to_string(),
        });
    }

    /// Step metrics. When overflowing, queues are discarded (training does not wait).
    pub fn step(&self, metrics: StepMetrics) {
        self.emit(TrainEvent::Step(metrics));
    }

    /// Will the spike image be accepted now.
    ///
    /// Snapshot of roads: the tensor is read to the host and folded into a bitmap. Examination
    /// allows you not to shoot something that won’t reach the screen anyway: not more often
    /// [`ChannelConfig::frame_interval`] and only until the queue is full.
    pub fn wants_frame(&self) -> bool {
        if self.tx.is_closed() || self.tx.len() * 2 >= self.tx.capacity().unwrap_or(usize::MAX) {
            return false;
        }
        let last = self
            .shared
            .last_frame
            .lock()
            .unwrap_or_else(|e| e.into_inner());
        last.is_none_or(|t| t.elapsed() >= self.shared.frame_interval)
    }

    /// Remove sample `sample` from `[batch, steps, units]` and send as activity of layer `layer`.
    pub fn spikes(&self, layer: &str, progress: Progress, spikes: Tensor<3>, sample: usize) {
        self.spike_frame(layer, progress, SpikeFrame::from_tensor(spikes, sample));
    }

    /// Same for the finished photo.
    pub fn spike_frame(&self, layer: &str, progress: Progress, frame: SpikeFrame) {
        *self
            .shared
            .last_frame
            .lock()
            .unwrap_or_else(|e| e.into_inner()) = Some(Instant::now());
        self.emit(TrainEvent::Spikes {
            layer: layer.to_owned(),
            progress,
            frame: Arc::new(frame),
        });
    }

    /// Summary of any rank tensor ([`ActivityStats`]); tensor not readable as `f32`,
    /// silently skipped (telemetry should not drop training).
    pub fn activity<const D: usize>(&self, layer: &str, tensor: Tensor<D>) {
        if let Ok(stats) = ActivityStats::try_of(tensor) {
            self.emit(TrainEvent::Activity {
                layer: layer.to_owned(),
                stats,
            });
        }
    }

    /// Report on structural plasticity. Like step metrics, it does not block learning: when
    /// when the queue overflows, it is discarded and counted ([`TrainControl::dropped`]).
    pub fn structural(&self, report: StructureReport) {
        self.emit(TrainEvent::Structure(Arc::new(report)));
    }

    /// Generation Evolution Report. Like the others, it does not block: when the queue is full
    /// discarded and counted; the loss is visible from the gap `generation`.
    pub fn evolution(&self, report: EvolutionReport) {
        self.emit(TrainEvent::Evolution(Arc::new(report)));
    }

    /// Weight matrix `[rows, cols]` of layer `layer` (read to the host; not `f32` - silently skipped).
    pub fn weights(&self, layer: &str, progress: Progress, weights: Tensor<2>) {
        let [rows, cols] = weights.dims();
        let Ok(values) = weights.into_data().try_to_vec::<f32>() else {
            return;
        };
        self.emit(TrainEvent::Weights {
            layer: layer.to_owned(),
            progress,
            rows,
            cols,
            values: Arc::new(values),
        });
    }

    #[allow(clippy::too_many_arguments)]
    /// Sparse connections (`from[k] → to[k]` with weight `weight[k]`) of layer `layer` from `rows` neurons in `cols`.
    /// The three lists must be the same length, otherwise the event is silently skipped. Send once per frame, just like spikes.
    pub fn sparse_links(
        &self,
        layer: &str,
        progress: Progress,
        rows: usize,
        cols: usize,
        from: Vec<u32>,
        to: Vec<u32>,
        weight: Vec<f32>,
    ) {
        if from.len() != to.len() || from.len() != weight.len() {
            return;
        }
        self.emit(TrainEvent::Links {
            layer: layer.to_owned(),
            progress,
            rows,
            cols,
            from: Arc::new(from),
            to: Arc::new(to),
            weight: Arc::new(weight),
        });
    }

    /// Dense weights `[rows, cols]`, but only `per_source` of the strongest modulo connections of each are sent
    /// source neuron: this way the dense layer is shown sparsely as long as it is not too large to be read by the host.
    pub fn strongest_links(
        &self,
        layer: &str,
        progress: Progress,
        weights: Tensor<2>,
        per_source: usize,
    ) {
        let [rows, cols] = weights.dims();
        let Ok(values) = weights.into_data().try_to_vec::<f32>() else {
            return;
        };
        let keep = per_source.min(cols);
        let (mut from, mut to, mut weight) = (
            Vec::with_capacity(rows * keep),
            Vec::with_capacity(rows * keep),
            Vec::with_capacity(rows * keep),
        );
        let mut order: Vec<usize> = Vec::with_capacity(cols);
        for i in 0..rows {
            let row = &values[i * cols..(i + 1) * cols];
            order.clear();
            order.extend((0..cols).filter(|&j| row[j].is_finite()));
            order.sort_by(|&a, &b| row[b].abs().total_cmp(&row[a].abs()));
            for &j in order.iter().take(keep) {
                from.push(i as u32);
                to.push(j as u32);
                weight.push(row[j]);
            }
        }
        self.sparse_links(layer, progress, rows, cols, from, to, weight);
    }

    /// Regions of the `layer` layer: `labels[u]` is the region number of the `u` neuron, `names[r]` is the name of the `r` region.
    /// Sent once (and again if the marking has changed); labels outside `names` are ignored.
    pub fn regions(&self, layer: &str, names: Vec<String>, labels: Vec<u16>) {
        // Markings are rarely sent and without them the map will not be colored: like `start`, it is not lost
        let _ = self.tx.send_blocking(TrainEvent::Regions {
            layer: layer.to_owned(),
            names: Arc::new(names),
            labels: Arc::new(labels),
        });
    }

    /// Step boundary: waits until the pause is removed and tells whether it is time to stop.
    ///
    /// `true` — the consumer pressed Stop or closed; we need to get out of the learning cycle. Replaces
    /// `SpikeDashboard::tick()`.
    pub fn checkpoint(&self) -> bool {
        while self.shared.paused.load(Ordering::Acquire) && !self.should_stop() {
            std::thread::sleep(Duration::from_millis(30));
        }
        self.should_stop()
    }

    fn should_stop(&self) -> bool {
        self.shared.stop.load(Ordering::Acquire) || self.tx.is_closed()
    }

    /// The training has reached its end. Doesn't get lost.
    pub fn finish(&self) {
        let stopped = self.should_stop();
        let _ = self.tx.send_blocking(TrainEvent::Finished { stopped });
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn metrics(step: usize) -> StepMetrics {
        StepMetrics {
            progress: Progress {
                epoch: 0,
                epochs: 1,
                batch: step,
                batches: 100,
            },
            loss: step as f32,
            ..Default::default()
        }
    }

    #[test]
    fn progress_step_is_monotonic_across_epochs() {
        let p = |e, b| Progress {
            epoch: e,
            epochs: 3,
            batch: b,
            batches: 4,
        };
        assert_eq!(p(0, 3).step() + 1, p(1, 0).step());
        assert_eq!(p(2, 3).step() + 1, p(2, 3).total_steps());
        assert_eq!(Progress::default().step(), 0);
    }

    #[test]
    fn events_arrive_in_order() {
        let (tx, rx) = channel(ChannelConfig::new("t"));
        tx.start();
        for i in 0..5 {
            tx.step(metrics(i));
        }
        tx.finish();
        let mut losses = Vec::new();
        let mut seen = (false, false);
        while let Ok(e) = rx.events.try_recv() {
            match e {
                TrainEvent::Started { title } => {
                    assert_eq!(title, "t");
                    seen.0 = true;
                }
                TrainEvent::Step(m) => losses.push(m.loss),
                TrainEvent::Finished { stopped } => {
                    assert!(!stopped);
                    seen.1 = true;
                }
                _ => {}
            }
        }
        assert_eq!(losses, [0.0, 1.0, 2.0, 3.0, 4.0]);
        assert_eq!(seen, (true, true));
    }

    #[test]
    fn sparse_links_and_regions_arrive_and_a_ragged_link_list_is_skipped() {
        let (tx, rx) = channel(ChannelConfig::new("t"));
        tx.sparse_links(
            "a→b",
            Progress::default(),
            3,
            4,
            vec![0, 2],
            vec![1, 3],
            vec![0.5, -0.25],
        );
        tx.sparse_links(
            "bad",
            Progress::default(),
            3,
            4,
            vec![0, 1],
            vec![1],
            vec![0.5],
        ); // lengths do not match
        tx.regions("b", vec!["optic".into(), "motor".into()], vec![0, 1, 1, 0]);
        let events: Vec<TrainEvent> = std::iter::from_fn(|| rx.events.try_recv().ok()).collect();
        assert_eq!(events.len(), 2, "the ragged one never leaves");
        match &events[0] {
            TrainEvent::Links {
                layer,
                rows,
                cols,
                from,
                to,
                weight,
                ..
            } => {
                assert_eq!((layer.as_str(), *rows, *cols), ("a→b", 3, 4));
                assert_eq!(
                    (from.as_slice(), to.as_slice(), weight.as_slice()),
                    (&[0u32, 2][..], &[1u32, 3][..], &[0.5f32, -0.25][..])
                );
            }
            other => panic!("{other:?}"),
        }
        assert!(
            matches!(&events[1], TrainEvent::Regions { names, labels, .. } if names.len() == 2 && labels.as_slice() == [0, 1, 1, 0])
        );
    }

    #[test]
    fn a_slow_consumer_never_blocks_training_and_drops_are_counted() {
        let (tx, rx) = channel(ChannelConfig::new("t").with_capacity(4));
        for i in 0..100 {
            tx.step(metrics(i)); // shouldn't freeze
        }
        assert_eq!(rx.events.len(), 4);
        assert_eq!(rx.control().dropped(), 96);
    }

    #[test]
    fn stop_reaches_the_checkpoint_and_lifts_the_pause() {
        let (tx, rx) = channel(ChannelConfig::new("t"));
        let ctl = rx.control();
        assert!(!tx.checkpoint());
        ctl.set_paused(true);
        let t = std::thread::spawn(move || tx.checkpoint()); // is on pause...
        std::thread::sleep(Duration::from_millis(80));
        assert!(!t.is_finished(), "on pause the checkpoint must wait");
        ctl.stop(); // …Stop unpauses and stops
        assert!(t.join().unwrap());
        assert!(!ctl.is_paused());
        assert!(ctl.is_stopping());
    }

    #[test]
    fn resume_lets_training_continue() {
        let (tx, rx) = channel(ChannelConfig::new("t"));
        let ctl = rx.control();
        ctl.set_paused(true);
        let t = std::thread::spawn(move || tx.checkpoint());
        std::thread::sleep(Duration::from_millis(60));
        ctl.set_paused(false);
        assert!(!t.join().unwrap(), "no need to stop after Resume");
    }

    #[test]
    fn a_closed_consumer_stops_training() {
        let (tx, rx) = channel(ChannelConfig::new("t"));
        drop(rx);
        assert!(
            tx.checkpoint(),
            "The GUI has been closed - training should not go in vain"
        );
        assert!(!tx.wants_frame());
        tx.step(metrics(0)); // doesn't panic
        tx.finish(); // and doesn't hang
    }

    #[test]
    fn wants_frame_is_rate_limited() {
        let (tx, _rx) =
            channel(ChannelConfig::new("t").with_frame_interval(Duration::from_millis(60)));
        assert!(tx.wants_frame(), "the first frame is sent immediately");
        tx.spike_frame(
            "h",
            Progress::default(),
            SpikeFrame::from_bits(2, 2, vec![true; 4]),
        );
        assert!(!tx.wants_frame(), "the second right after the first - no");
        std::thread::sleep(Duration::from_millis(70));
        assert!(tx.wants_frame());
    }

    #[test]
    fn wants_frame_backs_off_when_the_queue_is_filling_up() {
        let (tx, _rx) = channel(
            ChannelConfig::new("t")
                .with_capacity(8)
                .with_frame_interval(Duration::ZERO),
        );
        for i in 0..4 {
            tx.step(metrics(i));
        }
        assert!(
            !tx.wants_frame(),
            "The queue is half full - we are not taking pictures"
        );
    }

    #[test]
    fn finish_reports_whether_it_was_stopped() {
        let (tx, rx) = channel(ChannelConfig::new("t"));
        rx.control().stop();
        tx.finish();
        assert!(matches!(
            rx.events.try_recv(),
            Ok(TrainEvent::Finished { stopped: true })
        ));
    }

    fn report(seq: u64) -> StructureReport {
        use crate::structural::{ActionKind, Outcome, StructuralEvent};
        StructureReport {
            progress: Progress::default(),
            stats: Default::default(),
            events: vec![StructuralEvent {
                seq,
                step: 1,
                sample: 1,
                kind: ActionKind::AddSynapse,
                action: "add_synapse".into(),
                reason: "test".into(),
                outcome: Outcome::Applied,
                loss_before: Some(1.0),
                loss_after: Some(0.5),
                neurons_before: 3,
                neurons_after: 3,
                synapses_before: 2,
                synapses_after: 3,
            }],
            loss_before: Some(1.0),
            loss_after: Some(0.5),
            skipped: None,
            limits: Default::default(),
            topology: None,
        }
    }

    #[test]
    fn structure_reports_arrive_and_never_block_when_the_queue_is_full() {
        let (tx, rx) = channel(ChannelConfig::new("t").with_capacity(4));
        tx.structural(report(0));
        match rx.events.try_recv() {
            Ok(TrainEvent::Structure(r)) => assert_eq!(r.events[0].seq, 0),
            other => panic!("a report was expected, {other:?} arrived"),
        }
        // The queue is not read: unnecessary reports are discarded and counted, the call does not hang.
        for i in 0..10 {
            tx.structural(report(i));
        }
        assert_eq!(rx.control().dropped(), 6);
    }
}
