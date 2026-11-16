//! Example on synthetic data: without downloading the dataset and without `ring`.
//!
//! Problem: two classes in 8-dimensional space. Signs of class 0 on average
//! low (about 0.2), class 1 - high (about 0.8). Input is encoded
//! Poisson rate-coding, the network is two-layer on RLeaky neurons.
//! The purpose of the example is to quickly verify that the entire chain (coding → neurons →
//! surrogate gradient → optimizer) learns: loss should fall, and
//! accuracy increases to 100%.
//!
//! Launch: `cargo run --example synthetic_snn --release`

use burn::optim::{AdamConfig, GradientsParams};
use burn::tensor::Device;
use burn::tensor::Tensor;
use burn::tensor::loss::cross_entropy_with_logits;

use neo_snn::encoding::rate_encode;
use neo_snn::neuron::RLeaky;
use neo_snn::state::{NeuronState, ResetMode};
use neo_snn::surrogate::FastSigmoid;

const NUM_STEPS: usize = 12;
const FEATURES: usize = 8;
const HIDDEN: usize = 32;
const CLASSES: usize = 2;
const EPOCHS: usize = 30;
const BATCH_SIZE: usize = 32;
const BATCHES_PER_EPOCH: usize = 8;
const LR: f64 = 5e-3;

#[derive(burn::module::Module, Debug)]
pub struct TwoLayerSnn {
    layer1: RLeaky,
    layer2: RLeaky,
}

impl TwoLayerSnn {
    pub fn new(device: &Device) -> Self {
        Self {
            layer1: RLeaky::new(device, FEATURES, HIDDEN, 0.9, 1.0, ResetMode::Subtract),
            layer2: RLeaky::new(device, HIDDEN, CLASSES, 0.9, 1.0, ResetMode::Subtract),
        }
    }

    /// `x`: `[batch, num_steps, features]` → time-averaged membrane
    /// output layer `[batch, classes]` (used as logits).
    pub fn forward(&self, x: Tensor<3>, surrogate: &FastSigmoid) -> Tensor<2> {
        let device = x.device();
        let batch = x.dims()[0];

        let mut state1 = NeuronState::zeros_mem_only(&device, batch, HIDDEN);
        let mut state2 = NeuronState::zeros_mem_only(&device, batch, CLASSES);
        let mut spike1_prev: Tensor<2> = Tensor::zeros([batch, HIDDEN], &device);
        let mut spike2_prev: Tensor<2> = Tensor::zeros([batch, CLASSES], &device);
        let mut logits_accum: Tensor<2> = Tensor::zeros([batch, CLASSES], &device);

        // `split_time`, not `slice` in the loop: backward for `slice` is O(T²), see `neuron::split_time`.
        for input_t in neo_snn::neuron::split_time(x.clone()) {
            let (s1, new_state1) =
                self.layer1
                    .step(input_t, spike1_prev.clone(), &state1, surrogate);
            state1 = new_state1;
            spike1_prev = s1.clone();

            let (s2, new_state2) = self
                .layer2
                .step(s1, spike2_prev.clone(), &state2, surrogate);
            state2 = new_state2;
            spike2_prev = s2;

            logits_accum = logits_accum + state2.mem.clone();
        }

        logits_accum.div_scalar(NUM_STEPS as f32)
    }
}

/// Simple deterministic generator (LCG) - without dependence on `rand`.
struct Lcg(u64);

impl Lcg {
    fn next_f32(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 40) as f32) / ((1u64 << 24) as f32)
    }
}

/// Generates a batch: `(features [batch*features], tags)`.
/// Class 0 → features about 0.2, class 1 → about 0.8, noise ±0.15.
fn make_batch(rng: &mut Lcg, batch: usize) -> (Vec<f32>, Vec<usize>) {
    let mut feats = Vec::with_capacity(batch * FEATURES);
    let mut labels = Vec::with_capacity(batch);
    for _ in 0..batch {
        let label = if rng.next_f32() < 0.5 { 0 } else { 1 };
        let center = if label == 0 { 0.2 } else { 0.8 };
        for _ in 0..FEATURES {
            let noise = (rng.next_f32() - 0.5) * 0.3;
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

fn main() {
    let device = Device::flex().autodiff();
    let mut rng = Lcg(42);

    let mut model = TwoLayerSnn::new(&device);
    let mut optim = AdamConfig::new().init();
    let surrogate = FastSigmoid::default();

    println!("Synthetic problem: {CLASSES} class, {FEATURES} features, {NUM_STEPS} time steps");

    for epoch in 0..EPOCHS {
        let mut total_loss = 0.0f32;
        let mut correct = 0usize;
        let mut seen = 0usize;

        for _ in 0..BATCHES_PER_EPOCH {
            let (feats, labels) = make_batch(&mut rng, BATCH_SIZE);

            let images =
                Tensor::<1>::from_floats(feats.as_slice(), &device).reshape([BATCH_SIZE, FEATURES]);
            let spike_train = rate_encode(images, NUM_STEPS, &device);

            let target = Tensor::<1>::from_floats(one_hot(&labels, CLASSES).as_slice(), &device)
                .reshape([BATCH_SIZE, CLASSES]);

            let logits = model.forward(spike_train, &surrogate);

            // We calculate the accuracy to backward, using logit values.
            let logits_data = logits.clone().into_data();
            let logits_vec = logits_data.as_slice::<f32>().unwrap();
            for (i, &label) in labels.iter().enumerate() {
                let row = &logits_vec[i * CLASSES..(i + 1) * CLASSES];
                let pred = if row[1] > row[0] { 1 } else { 0 };
                if pred == label {
                    correct += 1;
                }
                seen += 1;
            }

            let loss = cross_entropy_with_logits(logits, target);
            total_loss += loss.clone().into_data().as_slice::<f32>().unwrap()[0];

            let grads = loss.backward();
            let grads = GradientsParams::from_grads(grads, &model);
            model = optim.step(LR, model, grads);
        }

        if epoch % 5 == 0 || epoch + 1 == EPOCHS {
            println!(
                "Epoch {:>2}: loss = {:.4}, accuracy = {:.1}%",
                epoch + 1,
                total_loss / BATCHES_PER_EPOCH as f32,
                100.0 * correct as f32 / seen as f32
            );
        }
    }

    println!("Done.");
}
