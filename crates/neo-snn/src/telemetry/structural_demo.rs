//! Demonstration of structural plasticity through a telemetry channel: the learning flow sends
//! metrics, spike frames and [`StructureReport`], the GUI (or any consumer) reads them.
//!
//! This is the same task as in `examples/dynamic_snn.rs` and `tests/structural_plasticity.rs`:
//! 6 inputs, 2 outputs, class A - spikes on inputs 0-1, class B - on 2-3, inputs 4-5 - noise;
//! the weight of the synapse is limited, so one input per output class is not enough, and one hidden
//! a neuron has no inputs. STDP alone cannot solve the problem: the controller must grow the missing
//! connections, remove dead neurons and roll back what degrades the quality.
//!
//! Runs on a real CPU engine [`crate::structural`], without Burn and GPU: GUI shows
//! namely topology changes, not animation.
//!
//! ```ignore
//! let (tx, rx) = channel(ChannelConfig::new("structural plasticity"));
//! std::thread::spawn(move || structural_demo::run(&tx, &DemoConfig::default()));
//! ```

use std::time::Duration;

use crate::functional::plasticity::WeightBound;
use crate::structural::{
    Birth, DynamicNetwork, EngineConfig, LifParams, Metrics, NeuronId, NeuronKind, Origin,
    PlasticityMode, Rng64, StructuralConfig, StructuralPlasticityController, Topology,
};

use super::channel::{Progress, StepMetrics, TrainTx};
use super::frame::SpikeFrame;
use super::structure::{SNAPSHOT_NEURONS, SNAPSHOT_SYNAPSES, StructureReport};

const INPUTS: usize = 6;
const STEPS: usize = 20;
const TRAIN_PER_EPOCH: usize = 20;
const EVAL_SAMPLES: usize = 20;

/// Demonstration options.
#[derive(Debug, Clone)]
pub struct DemoConfig {
    /// Number of epochs.
    pub epochs: usize,
    /// Pause after each epoch (so that the GUI has time to show; in tests - zero).
    pub pace: Duration,
    /// Whether to remove the graph to draw the topology.
    pub snapshot: bool,
    /// Controller grain.
    pub seed: u64,
}

impl Default for DemoConfig {
    fn default() -> Self {
        Self {
            epochs: 16,
            pace: Duration::from_millis(350),
            snapshot: true,
            seed: 7,
        }
    }
}

/// What the demonstration returned.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DemoSummary {
    /// Loss on the check sample at the beginning.
    pub start_loss: f32,
    /// Loss on the check sample at the end.
    pub end_loss: f32,
    /// Precision at the end.
    pub end_accuracy: f32,
    /// Applied structural operations.
    pub applied: usize,
    /// Whether stopped by user.
    pub stopped: bool,
}

fn sample(seed: u64, class: usize) -> Vec<[f32; INPUTS]> {
    let mut rng = Rng64::new(seed);
    (0..STEPS)
        .map(|_| {
            let mut x = [0.0; INPUTS];
            for (c, v) in x.iter_mut().enumerate() {
                let p = if c < 4 && c / 2 == class {
                    0.6
                } else if c >= 4 {
                    0.1
                } else {
                    0.0
                };
                if rng.chance(p) {
                    *v = 1.0;
                }
            }
            x
        })
        .collect()
}

/// Control sample without supervision and without plasticity: `(loss, accuracy)`.
fn evaluate(net: &DynamicNetwork) -> (f32, f32) {
    let mut p = net.probe_copy();
    let (mut loss, mut correct) = (0.0f32, 0usize);
    for i in 0..EVAL_SAMPLES {
        let class = i % 2;
        p.begin_sample();
        let mut count = [0u32; 2];
        for x in sample(900_000 + i as u64, class) {
            if let Ok(o) = p.step(&x) {
                for (k, s) in o.output_spikes.iter().enumerate().take(2) {
                    count[k] += *s as u32;
                }
            }
        }
        let margin = count[class] as f32 - count[1 - class] as f32;
        loss += (1.0 - margin / 2.0).clamp(0.0, 2.0);
        correct += (margin > 0.0) as usize;
    }
    (
        loss / EVAL_SAMPLES as f32,
        correct as f32 / EVAL_SAMPLES as f32,
    )
}

#[allow(clippy::field_reassign_with_default)]
/// Controller configuration for this task.
pub fn structural_config(seed: u64) -> StructuralConfig {
    let mut c = StructuralConfig::default();
    c.seed = seed;
    c.topology.max_neurons = 16;
    c.topology.max_synapses = 40;
    c.schedule.min_interval_steps = 400;
    c.schedule.max_ops_per_interval = 6;
    c.growth.synapses.max_new_per_interval = 4;
    c.growth.synapses.min_score = 100.0;
    c.growth.synapses.init_weight = 0.12;
    c.growth.target_loss = Some(0.1);
    c.pruning.min_age_steps = 800;
    c.pruning.min_observed_samples = 20;
    c.pruning.reeval_interval_steps = 400;
    c.pruning.grace_steps = 400;
    c.pruning.min_hidden = 0;
    c.homeostasis.enabled = false;
    c
}

/// The initial (“unfinished”) network of the task.
pub fn build_network(cfg: &StructuralConfig) -> DynamicNetwork {
    let mut t = Topology::new(cfg.topology, cfg.params);
    let b = Birth::default();
    let inputs: Vec<NeuronId> = (0..INPUTS)
        .map(|_| {
            t.add_neuron(NeuronKind::Input, LifParams::default(), Origin::Initial, b)
                .expect("input")
        })
        .collect();
    let mut lif = |kind| {
        t.add_neuron(kind, LifParams::new(0.9, 1.0), Origin::Initial, b)
            .expect("neuron")
    };
    let (o0, o1, dead) = (
        lif(NeuronKind::Output),
        lif(NeuronKind::Output),
        lif(NeuronKind::Hidden),
    );
    for (src, dst, w) in [
        (inputs[0], o0, 0.10),
        (inputs[2], o1, 0.10),
        (inputs[4], o0, 0.05),
        (inputs[5], o1, 0.05),
        (dead, o0, 0.3),
    ] {
        t.add_synapse(src, dst, w, 0, Origin::Initial, b)
            .expect("synapse");
    }
    let mut ec = EngineConfig::default();
    ec.stdp.mode = PlasticityMode::Immediate;
    ec.stdp.a_plus = 0.0005;
    ec.stdp.a_minus = 0.0003;
    ec.stdp.bound = WeightBound::Hard {
        w_min: -0.15,
        w_max: 0.15,
    };
    DynamicNetwork::new(t, ec).expect("network")
}

/// Output spike frame: one sample of each class in a row to make the outputs look
/// the way they actually behave (everyone is silent in “someone else’s” class).
fn output_frame(net: &DynamicNetwork, epoch: usize) -> SpikeFrame {
    let mut p = net.probe_copy();
    let units = p.topology().outputs().len().max(1);
    let mut bits = Vec::with_capacity(2 * STEPS * units);
    for class in 0..2 {
        p.begin_sample();
        for x in sample(910_000 + epoch as u64 * 2 + class as u64, class) {
            let spikes = p.step(&x).map(|o| o.output_spikes).unwrap_or_default();
            bits.extend((0..units).map(|k| spikes.get(k).copied().unwrap_or(false)));
        }
    }
    SpikeFrame::from_bits(2 * STEPS, units, bits)
}

/// Run the demo in the current thread. Causes `tx.start()` and `tx.finish()`;
/// pause and stop from the GUI are handled at the epoch boundary ([`TrainTx::checkpoint`]).
pub fn run(tx: &TrainTx, cfg: &DemoConfig) -> DemoSummary {
    let sc = structural_config(cfg.seed);
    let mut net = build_network(&sc);
    let mut ctl = StructuralPlasticityController::new(sc).expect("configuration");
    let (start_loss, _) = evaluate(&net);
    let (mut end_loss, mut end_accuracy, mut applied, mut stopped) =
        (start_loss, 0.0, 0usize, false);

    tx.start();
    for epoch in 0..cfg.epochs {
        let outs = net.topology().outputs();
        for i in 0..TRAIN_PER_EPOCH {
            let class = i % 2;
            net.begin_sample();
            for x in sample((epoch * 1000 + i + 1) as u64, class) {
                let _ = net.step_with_current(&x, &[(outs[class], 1.5)]);
            }
        }
        let (loss, acc) = evaluate(&net);
        let progress = Progress {
            epoch,
            epochs: cfg.epochs,
            batch: 0,
            batches: 1,
        };
        tx.step(StepMetrics {
            progress,
            loss,
            accuracy: Some(acc),
            grad_norm: None,
        });
        if tx.wants_frame() {
            tx.spike_frame("output", progress, output_frame(&net, epoch));
        }

        let mut eval = |p: &mut DynamicNetwork| evaluate(p).0;
        match ctl.adapt(
            &mut net,
            Metrics {
                task_loss: loss,
                reward: None,
                accuracy: Some(acc),
            },
            Some(&mut eval),
        ) {
            Ok(r) => {
                applied += r.applied;
                let snapshot = cfg
                    .snapshot
                    .then_some((SNAPSHOT_NEURONS, SNAPSHOT_SYNAPSES));
                tx.structural(StructureReport::from_adapt(
                    progress,
                    &r,
                    net.topology(),
                    snapshot,
                ));
            }
            Err(_) => break,
        }
        (end_loss, end_accuracy) = (loss, acc);
        if !cfg.pace.is_zero() {
            std::thread::sleep(cfg.pace);
        }
        if tx.checkpoint() {
            stopped = true;
            break;
        }
    }
    tx.finish();
    DemoSummary {
        start_loss,
        end_loss,
        end_accuracy,
        applied,
        stopped,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::telemetry::{ChannelConfig, TrainEvent, channel};

    fn run_collect(cfg: DemoConfig) -> (DemoSummary, Vec<TrainEvent>, u64) {
        let (tx, rx) = channel(
            ChannelConfig::new("demo")
                .with_capacity(4096)
                .with_frame_interval(Duration::ZERO),
        );
        let s = run(&tx, &cfg);
        let events: Vec<_> = std::iter::from_fn(|| rx.events.try_recv().ok()).collect();
        (s, events, rx.control().dropped())
    }

    #[test]
    fn demo_streams_real_structural_changes() {
        let cfg = DemoConfig {
            pace: Duration::ZERO,
            ..Default::default()
        };
        let (s, events, dropped) = run_collect(cfg.clone());
        assert_eq!(dropped, 0);
        assert!(matches!(events.first(), Some(TrainEvent::Started { .. })));
        assert!(matches!(
            events.last(),
            Some(TrainEvent::Finished { stopped: false })
        ));
        let reports: Vec<_> = events
            .iter()
            .filter_map(|e| {
                if let TrainEvent::Structure(r) = e {
                    Some(r.clone())
                } else {
                    None
                }
            })
            .collect();
        assert_eq!(
            reports.len(),
            cfg.epochs,
            "according to the report for the era"
        );
        // Events come in a row, without holes or repetitions.
        let seqs: Vec<u64> = reports
            .iter()
            .flat_map(|r| r.events.iter().map(|e| e.seq))
            .collect();
        assert!(!seqs.is_empty());
        assert!(seqs.windows(2).all(|w| w[1] == w[0] + 1), "seq: {seqs:?}");
        // The structure really changed and the graph in the snapshot coincides with the counters.
        let last = reports.last().unwrap();
        let snap = last.topology.as_ref().expect("shot of the graph");
        assert_eq!(
            (snap.total_neurons, snap.total_synapses),
            (last.stats.neurons, last.stats.synapses)
        );
        assert!(!snap.truncated);
        // The result is the same as the example: the network solved the problem.
        assert!(s.end_loss < 0.2 && s.end_accuracy >= 0.95, "{s:?}");
        assert!(s.end_loss < s.start_loss - 0.5);
        assert!(s.applied >= 3);
        assert!(
            events
                .iter()
                .any(|e| matches!(e, TrainEvent::Spikes { .. }))
        );
    }

    #[test]
    fn stop_request_ends_the_run_early() {
        let (tx, rx) = channel(ChannelConfig::new("demo").with_capacity(4096));
        rx.control().stop();
        let s = run(
            &tx,
            &DemoConfig {
                pace: Duration::ZERO,
                ..Default::default()
            },
        );
        assert!(s.stopped);
        let events: Vec<_> = std::iter::from_fn(|| rx.events.try_recv().ok()).collect();
        let steps = events
            .iter()
            .filter(|e| matches!(e, TrainEvent::Step(_)))
            .count();
        assert_eq!(steps, 1);
        assert!(matches!(
            events.last(),
            Some(TrainEvent::Finished { stopped: true })
        ));
    }
}
