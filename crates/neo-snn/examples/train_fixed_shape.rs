//! Structural plasticity over Burn layers of a fixed shape (`RLeaky`).
//!
//! Two-layer SNN as in `demo`, but the hidden layer has a capacity of 24 neurons, 12 of which are active,
//! the rest are reserve; the `linear_in` matrix of layer 1 is sparse (70%). Between batches
//! `FixedShapeController` turns on connections/backup neurons, turns off unimportant ones and rolls back
//! changes that worsened loss.
//!
//! Launch: `cargo run -p neo-snn --example train_fixed_shape --features demo --release`

use burn::optim::{AdamConfig, GradientsParams};
use burn::tensor::loss::cross_entropy_with_logits;
use burn::tensor::{Device, Tensor};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use neo_snn::encoding::rate_encode;
use neo_snn::fixed_shape::{
    MaskProjector, apply_blob, apply_layer_update, load_bundle, named_tensor, observe_linear_grads,
    read_linear_weights, save_bundle, tensor_from_named,
};
use neo_snn::neuron::RLeaky;
use neo_snn::state::{NeuronState, ResetMode};
use neo_snn::structural::fixed::*;
use neo_snn::surrogate::FastSigmoid;
use neo_snn::telemetry::{ChannelConfig, Progress, StructureReport, channel};

const STEPS: usize = 24;
const FEATURES: usize = 8;
const HIDDEN: usize = 24;
const CLASSES: usize = 2;
const BATCH: usize = 32;

#[derive(burn::module::Module, Debug)]
struct Snn {
    layer1: RLeaky,
    layer2: RLeaky,
}

struct Out {
    logits: Tensor<2>,
    hidden: Tensor<3>,
    output: Tensor<3>,
}

impl Snn {
    fn forward(&self, x: Tensor<3>, sg: &FastSigmoid) -> Out {
        let device = x.device();
        let batch = x.dims()[0];
        let mut st1 = NeuronState::zeros_mem_only(&device, batch, HIDDEN);
        let mut st2 = NeuronState::zeros_mem_only(&device, batch, CLASSES);
        let mut sp1: Tensor<2> = Tensor::zeros([batch, HIDDEN], &device);
        let mut sp2: Tensor<2> = Tensor::zeros([batch, CLASSES], &device);
        let mut acc: Tensor<2> = Tensor::zeros([batch, CLASSES], &device);
        let (mut h, mut o) = (Vec::with_capacity(STEPS), Vec::with_capacity(STEPS));
        // `split_time`, not `slice` in the loop: backward for `slice` is O(T²), see `neuron::split_time`.
        for xt in neo_snn::neuron::split_time(x.clone()) {
            let (s1, n1) = self.layer1.step(xt, sp1.clone(), &st1, sg);
            st1 = n1;
            sp1 = s1.clone();
            h.push(s1.clone());
            let (s2, n2) = self.layer2.step(s1, sp2.clone(), &st2, sg);
            st2 = n2;
            sp2 = s2.clone();
            o.push(s2);
            acc = acc + st2.mem.clone();
        }
        Out {
            logits: acc.div_scalar(STEPS as f32),
            hidden: Tensor::stack::<3>(h, 1),
            output: Tensor::stack::<3>(o, 1),
        }
    }
}

fn rates(t: Tensor<3>, units: usize) -> Vec<f32> {
    let b = t.dims()[0];
    t.detach()
        .mean_dim(1)
        .reshape([b, units])
        .into_data()
        .try_to_vec::<f32>()
        .expect("f32")
}

fn make_batch(rng: &mut StdRng) -> (Vec<f32>, Vec<usize>) {
    let (mut f, mut l) = (Vec::new(), Vec::new());
    for _ in 0..BATCH {
        let label = usize::from(rng.r#gen::<f32>() >= 0.5);
        let c = if label == 0 { 0.2 } else { 0.8 };
        for _ in 0..FEATURES {
            f.push((c + (rng.r#gen::<f32>() - 0.5) * 0.3).clamp(0.0, 1.0));
        }
        l.push(label);
    }
    (f, l)
}

fn main() {
    let device = Device::flex().autodiff();
    let mut rng = StdRng::seed_from_u64(42);
    let (tx, rx) = channel(ChannelConfig::new("fixed shape RLeaky"));

    let mut cfg = FixedShapeConfig::default();
    cfg.schedule.min_interval_steps = 40;
    cfg.pruning.min_age_steps = 80;
    cfg.rollback.verify_after_calls = 10;
    cfg.neurons.min_loss = 0.05;
    let specs = vec![
        LayerSpec::dense("hidden.in", FEATURES, HIDDEN, LayerRole::Hidden)
            .with_initial_active(12)
            .with_density(0.7),
        LayerSpec::dense("out.in", HIDDEN, CLASSES, LayerRole::Output).fed_by(0),
    ];
    let mut ctrl = FixedShapeController::new(cfg, specs).expect("configuration");
    let cap = ctrl.capacity();

    let mut model = Snn {
        layer1: RLeaky::new(&device, FEATURES, HIDDEN, 0.9, 1.0, ResetMode::Subtract),
        layer2: RLeaky::new(&device, HIDDEN, CLASSES, 0.9, 1.0, ResetMode::Subtract),
    };
    let mut optim = AdamConfig::new().init();
    let sg = FastSigmoid::default();
    let mut proj = MaskProjector::new(&ctrl, &device);
    model = project_all(model, &proj);

    let (epochs, per_epoch) = (40usize, 12usize);
    let mut ema = f32::NAN;
    for epoch in 0..epochs {
        for batch in 0..per_epoch {
            let step = (epoch * per_epoch + batch) as u64;
            let progress = Progress {
                epoch,
                epochs,
                batch,
                batches: per_epoch,
            };
            let (feats, labels) = make_batch(&mut rng);
            let x = Tensor::<1>::from_floats(feats.as_slice(), &device).reshape([BATCH, FEATURES]);
            let train = rate_encode(x, STEPS, &device);
            let mut oh = vec![0.0f32; BATCH * CLASSES];
            for (i, &l) in labels.iter().enumerate() {
                oh[i * CLASSES + l] = 1.0;
            }
            let target = Tensor::<1>::from_floats(oh.as_slice(), &device).reshape([BATCH, CLASSES]);

            let in_rate = rates(train.clone(), FEATURES);
            let out = model.forward(train, &sg);
            let (h_rate, o_rate) = (rates(out.hidden, HIDDEN), rates(out.output, CLASSES));
            let loss = cross_entropy_with_logits(out.logits, target);
            let lv = loss.clone().into_data().try_to_vec::<f32>().expect("f32")[0];
            ema = if ema.is_nan() {
                lv
            } else {
                ema + 0.1 * (lv - ema)
            };

            let grads = loss.backward();
            let g1 = model.layer1.linear_in.weight.val().grad(&grads);
            let g2 = model.layer2.linear_in.weight.val().grad(&grads);
            observe_linear_grads(&mut ctrl, 0, &model.layer1.linear_in, g1)
                .expect("layer 0 gradients");
            observe_linear_grads(&mut ctrl, 1, &model.layer2.linear_in, g2)
                .expect("layer 1 gradients");
            ctrl.observe_activity(0, &in_rate, &h_rate, BATCH)
                .expect("activity 0");
            ctrl.observe_activity(1, &h_rate, &o_rate, BATCH)
                .expect("activity 1");

            let gp = GradientsParams::from_grads(grads, &model);
            model = optim.step(5e-3, model, gp);
            model = project_all(model, &proj);

            // The structural step is strictly between batches, outside the autodiff graph.
            let w = read_linear_weights(&[&model.layer1.linear_in, &model.layer2.linear_in])
                .expect("weight");
            let refs: Vec<&[f32]> = w.iter().map(|v| v.as_slice()).collect();
            let rep = ctrl
                .adapt(
                    AdaptInput {
                        step,
                        sample: step,
                        loss: Some(ema),
                    },
                    &refs,
                )
                .expect("adapt");
            if !rep.updates.is_empty() {
                for u in &rep.updates {
                    match u.layer {
                        0 => {
                            model.layer1.linear_in =
                                apply_layer_update(model.layer1.linear_in, u).expect("entry 0")
                        }
                        _ => {
                            model.layer2.linear_in =
                                apply_layer_update(model.layer2.linear_in, u).expect("entry 1")
                        }
                    }
                }
                proj.refresh(&ctrl);
                model = project_all(model, &proj);
            }
            if !rep.events.is_empty() || rep.rolled_back {
                tx.structural(StructureReport::from_fixed(progress, &rep, cap));
            }
        }
        let s = ctrl.stats(AdaptInput::default(), None);
        println!(
            "era {:>2} loss {:.4} hidden {} connections {}",
            epoch + 1,
            ema,
            s.hidden,
            s.active_synapses
        );
    }
    println!("events in the log: {}", ctrl.log().len());

    // One file: masks + weights of managed layers + recurrent matrix.
    let path = std::env::temp_dir().join("train_fixed_shape.nfxb");
    let rec =
        named_tensor("layer1.rec", model.layer1.linear_rec.weight.val()).expect("recurrent matrix");
    save_bundle(
        &ctrl,
        &[&model.layer1.linear_in, &model.layer2.linear_in],
        vec![rec],
        b"epochs=40".to_vec(),
        &path,
    )
    .expect("recording");
    // Loading into a fresh controller and a fresh model (as when resuming training).
    let mut ctrl2 = FixedShapeController::new(
        ctrl.config().clone(),
        ctrl.layers().iter().map(|l| l.spec().clone()).collect(),
    )
    .expect("controller");
    let loaded = load_bundle(&mut ctrl2, &path).expect("reading");
    let mut model2 = Snn {
        layer1: RLeaky::new(&device, FEATURES, HIDDEN, 0.9, 1.0, ResetMode::Subtract),
        layer2: RLeaky::new(&device, HIDDEN, CLASSES, 0.9, 1.0, ResetMode::Subtract),
    };
    model2.layer1.linear_in =
        apply_blob(model2.layer1.linear_in, &loaded.layers[0]).expect("layer 0");
    model2.layer2.linear_in =
        apply_blob(model2.layer2.linear_in, &loaded.layers[1]).expect("layer 1");
    let rec2: Tensor<2> = tensor_from_named(&loaded.tensors[0], &device).expect("rec");
    model2.layer1.linear_rec.weight = model2.layer1.linear_rec.weight.map(|_| rec2);
    let proj2 = MaskProjector::new(&ctrl2, &device);
    let _restored = project_all(model2, &proj2);
    println!(
        "saved and restored: {} (extra = {})",
        path.display(),
        String::from_utf8_lossy(&loaded.extra)
    );
    drop(rx);
}

fn project_all(mut m: Snn, p: &MaskProjector) -> Snn {
    m.layer1.linear_in = p.project_linear(0, m.layer1.linear_in);
    // Recurrent matrix: outputs are the same neurons as layer1.linear_in.
    m.layer1.linear_rec = p.project_tied(0, m.layer1.linear_rec);
    m.layer2.linear_in = p.project_linear(1, m.layer2.linear_in);
    m
}
