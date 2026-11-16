//! Demo training of two-layer SNN on a synthetic problem (feature `demo`).
//!
//! This cycle is shared by the GUI (`neo-gui`, Training tab: Start button) and
//! the console example `train_telemetry`. The loop doesn't draw anything: it
//! writes to [`TrainTx`] and obeys [`TrainTx::checkpoint`].
//!
//! Shows exactly what telemetry is needed for: you can see NOT ONLY how the loss is falling, but also what
//! this happens with spikes - has the hidden layer become silent, is it spiked at each
//! step, whether the population is synchronous.

use burn::optim::{AdamConfig, GradientsParams};
use burn::tensor::loss::cross_entropy_with_logits;
use burn::tensor::{Device, Tensor};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use crate::encoding::rate_encode;
use crate::functional::grad_norm;
use crate::neuron::RLeaky;
use crate::state::{NeuronState, ResetMode};
use crate::surrogate::FastSigmoid;
use crate::telemetry::{Progress, StepMetrics, TrainTx};

pub const NUM_STEPS: usize = 24;
pub const FEATURES: usize = 8;
pub const HIDDEN: usize = 24;
pub const CLASSES: usize = 2;

/// Names of layers and weight matrices in telemetry (the dashboard uses them to find its panels).
pub const LAYER_INPUT: &str = "input";
pub const LAYER_HIDDEN: &str = "hidden";
pub const LAYER_OUTPUT: &str = "output";
pub const WEIGHTS_INPUT_HIDDEN: &str = "input→hidden";
pub const WEIGHTS_HIDDEN_OUTPUT: &str = "hidden→output";

/// Demo training options.
#[derive(Debug, Clone)]
pub struct DemoConfig {
    pub epochs: usize,
    pub batches_per_epoch: usize,
    pub batch_size: usize,
    pub lr: f64,
    pub seed: u64,
    /// Artificial pause after each step, ms. On the CPU, this network seems faster than the eye
    /// manages to follow; `0` - no slowdown.
    pub step_delay_ms: u64,
}

impl Default for DemoConfig {
    fn default() -> Self {
        Self {
            epochs: 60,
            batches_per_epoch: 12,
            batch_size: 32,
            lr: 5e-3,
            seed: 42,
            step_delay_ms: 15,
        }
    }
}

/// How did the training end?
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DemoSummary {
    pub steps_done: u64,
    pub stopped: bool,
    pub last_loss: f32,
    pub last_accuracy: f32,
}

#[derive(burn::module::Module, Debug)]
pub struct TwoLayerSnn {
    layer1: RLeaky,
    layer2: RLeaky,
}

/// Network output: logits, spikes of hidden and output layers `[batch, steps, units]`.
///
/// The spikes come back out for the sake of telemetry: they are not needed for learning, but looking at them is
/// the only way to notice layer degeneration before it affects accuracy.
pub struct Outputs {
    pub logits: Tensor<2>,
    pub hidden_spikes: Tensor<3>,
    pub output_spikes: Tensor<3>,
}

impl TwoLayerSnn {
    pub fn new(device: &Device) -> Self {
        Self {
            layer1: RLeaky::new(device, FEATURES, HIDDEN, 0.9, 1.0, ResetMode::Subtract),
            layer2: RLeaky::new(device, HIDDEN, CLASSES, 0.9, 1.0, ResetMode::Subtract),
        }
    }

    pub fn forward(&self, x: Tensor<3>, surrogate: &FastSigmoid) -> Outputs {
        let device = x.device();
        let batch = x.dims()[0];

        let mut state1 = NeuronState::zeros_mem_only(&device, batch, HIDDEN);
        let mut state2 = NeuronState::zeros_mem_only(&device, batch, CLASSES);
        let mut spike1_prev: Tensor<2> = Tensor::zeros([batch, HIDDEN], &device);
        let mut spike2_prev: Tensor<2> = Tensor::zeros([batch, CLASSES], &device);
        let mut logits_accum: Tensor<2> = Tensor::zeros([batch, CLASSES], &device);
        let mut hidden = Vec::with_capacity(NUM_STEPS);
        let mut output = Vec::with_capacity(NUM_STEPS);

        // `split_time`, not `slice` in the loop: backward for `slice` is O(T²), see `neuron::split_time`.
        for input_t in crate::neuron::split_time(x) {
            let (s1, new_state1) =
                self.layer1
                    .step(input_t, spike1_prev.clone(), &state1, surrogate);
            state1 = new_state1;
            spike1_prev = s1.clone();
            hidden.push(s1.clone());

            let (s2, new_state2) = self
                .layer2
                .step(s1, spike2_prev.clone(), &state2, surrogate);
            state2 = new_state2;
            output.push(s2.clone());
            spike2_prev = s2;

            logits_accum = logits_accum + state2.mem.clone();
        }

        Outputs {
            logits: logits_accum.div_scalar(NUM_STEPS as f32),
            hidden_spikes: Tensor::stack::<3>(hidden, 1),
            output_spikes: Tensor::stack::<3>(output, 1),
        }
    }
}

fn make_batch(rng: &mut StdRng, batch: usize) -> (Vec<f32>, Vec<usize>) {
    let mut feats = Vec::with_capacity(batch * FEATURES);
    let mut labels = Vec::with_capacity(batch);
    for _ in 0..batch {
        let label = if rng.r#gen::<f32>() < 0.5 { 0 } else { 1 };
        let center = if label == 0 { 0.2 } else { 0.8 };
        for _ in 0..FEATURES {
            let noise = (rng.r#gen::<f32>() - 0.5) * 0.3;
            feats.push((center + noise).clamp(0.0, 1.0));
        }
        labels.push(label);
    }
    (feats, labels)
}

fn one_hot(labels: &[usize], num_classes: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; labels.len() * num_classes];
    for (i, &l) in labels.iter().enumerate() {
        out[i * num_classes + l] = 1.0;
    }
    out
}

/// Train the network by reporting progress to `tx`. `device` is an autodiff device
/// (`Device::flex().autodiff()`).
///
/// Blocks the thread until the end of training or until `Stop`: run in a separate thread.
pub fn run(device: &Device, cfg: &DemoConfig, tx: &TrainTx) -> DemoSummary {
    let mut rng = StdRng::seed_from_u64(cfg.seed);
    let mut model = TwoLayerSnn::new(device);
    let mut optim = AdamConfig::new().init();
    let surrogate = FastSigmoid::default();

    tx.start();
    let mut summary = DemoSummary {
        steps_done: 0,
        stopped: false,
        last_loss: f32::NAN,
        last_accuracy: 0.0,
    };

    'training: for epoch in 0..cfg.epochs {
        for batch in 0..cfg.batches_per_epoch {
            let progress = Progress {
                epoch,
                epochs: cfg.epochs,
                batch,
                batches: cfg.batches_per_epoch,
            };
            let (feats, labels) = make_batch(&mut rng, cfg.batch_size);

            let images = Tensor::<1>::from_floats(feats.as_slice(), device)
                .reshape([cfg.batch_size, FEATURES]);
            let spike_train = rate_encode(images, NUM_STEPS, device);
            // Input spikes for telemetry: a tensor clone is cheap, it will only go to the host in a frame
            let input_spikes = spike_train.clone();
            let target = Tensor::<1>::from_floats(one_hot(&labels, CLASSES).as_slice(), device)
                .reshape([cfg.batch_size, CLASSES]);

            let out = model.forward(spike_train, &surrogate);

            // Logit accuracy, before backpass.
            let logits_data = out.logits.clone().into_data();
            let logits_vec = logits_data
                .as_slice::<f32>()
                .expect("expected backend with element f32");
            let correct = labels
                .iter()
                .enumerate()
                .filter(|&(i, &label)| {
                    let row = &logits_vec[i * CLASSES..(i + 1) * CLASSES];
                    (row[1] > row[0]) as usize == label
                })
                .count();
            let accuracy = correct as f32 / labels.len() as f32;

            // Summary of logits and hidden layer spikes ([`functional::probe`]) - before
            // `loss` will take `out.logits`. Spikes are read from the device only when the picture is complete
            // will still go to the screen.
            let want_frame = tx.wants_frame();
            if want_frame {
                tx.activity("logits", out.logits.clone().detach());
            }

            let loss = cross_entropy_with_logits(out.logits, target);
            let loss_value = loss
                .clone()
                .into_data()
                .as_slice::<f32>()
                .expect("expected backend with element f32")[0];

            let grads = loss.backward();

            // Norm of the gradient of the input weight of the first layer: it is this layer that is the first to go numb if
            // the network stops learning - the gradient before it passes through both layers and both surrogates.
            let grad_norm_value = model
                .layer1
                .linear_in
                .weight
                .grad(&grads)
                .map(grad_norm)
                .unwrap_or(0.0);

            let grads = GradientsParams::from_grads(grads, &model);
            model = optim.step(cfg.lr, model, grads);

            tx.step(StepMetrics {
                progress,
                loss: loss_value,
                accuracy: Some(accuracy),
                grad_norm: Some(grad_norm_value),
            });
            if want_frame {
                // A tensor with autodiff is given as is: the image is taken to the host once,
                // the graph is not affected. The hidden layer comes first: it gets into the first
                // a list of dashboard layers, and it is based on this that a verdict on the health of the network is made.
                tx.spikes(LAYER_HIDDEN, progress, out.hidden_spikes, 0);
                tx.spikes(LAYER_INPUT, progress, input_spikes, 0);
                tx.spikes(LAYER_OUTPUT, progress, out.output_spikes, 0);
                // Weights after the optimizer step: weight maps show what the network has learned
                tx.weights(
                    WEIGHTS_INPUT_HIDDEN,
                    progress,
                    model.layer1.linear_in.weight.val().detach(),
                );
                tx.weights(
                    WEIGHTS_HIDDEN_OUTPUT,
                    progress,
                    model.layer2.linear_in.weight.val().detach(),
                );
            }

            summary.steps_done += 1;
            summary.last_loss = loss_value;
            summary.last_accuracy = accuracy;

            if cfg.step_delay_ms > 0 {
                std::thread::sleep(std::time::Duration::from_millis(cfg.step_delay_ms));
            }
            if tx.checkpoint() {
                summary.stopped = true;
                break 'training;
            }
        }
    }

    tx.finish();
    summary
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::telemetry::{ChannelConfig, TrainEvent, channel};

    fn quick() -> DemoConfig {
        DemoConfig {
            epochs: 3,
            batches_per_epoch: 4,
            batch_size: 8,
            step_delay_ms: 0,
            ..Default::default()
        }
    }

    #[test]
    fn demo_streams_metrics_spikes_and_finishes() {
        let device = Device::flex().autodiff();
        let (tx, rx) =
            channel(ChannelConfig::new("demo").with_frame_interval(std::time::Duration::ZERO));
        let s = run(&device, &quick(), &tx);
        assert_eq!(s.steps_done, 12);
        assert!(!s.stopped);

        let (mut steps, mut frames, mut activity, mut started, mut finished) =
            (0, 0, 0, false, false);
        let mut weight_events = 0;
        let mut last_step = None;
        while let Ok(e) = rx.events.try_recv() {
            match e {
                TrainEvent::Started { .. } => started = true,
                TrainEvent::Step(m) => {
                    assert!(m.loss.is_finite() && m.grad_norm.is_some() && m.accuracy.is_some());
                    if let Some(prev) = last_step {
                        assert!(m.progress.step() > prev, "steps are moving forward");
                    }
                    last_step = Some(m.progress.step());
                    steps += 1;
                }
                TrainEvent::Spikes { layer, frame, .. } => {
                    let units = match layer.as_str() {
                        LAYER_HIDDEN => HIDDEN,
                        LAYER_INPUT => FEATURES,
                        LAYER_OUTPUT => CLASSES,
                        other => panic!("unexpected layer {other}"),
                    };
                    assert_eq!((frame.steps, frame.units), (NUM_STEPS, units));
                    frames += 1;
                }
                TrainEvent::Links { .. }
                | TrainEvent::Regions { .. }
                | TrainEvent::Structure(_)
                | TrainEvent::Evolution(_) => {}
                TrainEvent::Weights {
                    layer,
                    rows,
                    cols,
                    values,
                    ..
                } => {
                    let expect = match layer.as_str() {
                        WEIGHTS_INPUT_HIDDEN => (FEATURES, HIDDEN),
                        WEIGHTS_HIDDEN_OUTPUT => (HIDDEN, CLASSES),
                        other => panic!("unexpected weights {other}"),
                    };
                    assert_eq!((rows, cols), expect);
                    assert_eq!(values.len(), rows * cols);
                    assert!(values.iter().all(|v| v.is_finite()));
                    weight_events += 1;
                }
                TrainEvent::Activity { stats, .. } => {
                    assert!(stats.abs_max.is_finite());
                    activity += 1;
                }
                TrainEvent::Finished { stopped } => finished = !stopped,
            }
        }
        assert_eq!(steps, 12);
        assert!(started && finished);
        assert!(frames > 0 && activity > 0 && weight_events > 0);
    }

    #[test]
    fn stop_interrupts_training_at_a_step_boundary() {
        let device = Device::flex().autodiff();
        let (tx, rx) = channel(ChannelConfig::new("demo"));
        rx.control().stop();
        let s = run(&device, &quick(), &tx);
        assert!(s.stopped);
        assert_eq!(
            s.steps_done, 1,
            "learning is interrupted at the edge of a step, not in the middle of it"
        );
    }
}
