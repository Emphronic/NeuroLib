//! Dynamic SNN with structural plasticity on a small repeatable problem.
//!
//! Launch: `cargo run --release --example dynamic_snn -p neo-snn`
//!
//! Problem: 6 inputs, 2 outputs. Class A - spikes on inputs 0-1, class B - on 2-3;
//! inputs 4–5 - noise. The readable result is which output spikes more often. The initial
//! the network is deliberately “unfinished”: for each class there is a connection from only one
//! of its two inputs, and the weight is limited so that one input is not enough for the output;
//! in addition, one hidden neuron has no inputs at all. STDP is not a task alone
//! solve - you need to grow the missing connections. Training takes place with a “teacher” (external current
//! to the correct output) and pair-STDP; the structure is changed by `StructuralPlasticityController`:
//! grows connections between jointly active neurons, removes useless and
//! rolls back changes that degrade quality on the control sample.
#![allow(clippy::field_reassign_with_default)]

use neo_snn::structural::*;

const INPUTS: usize = 6;
const STEPS: usize = 20;
const TRAIN_PER_EPOCH: usize = 20;
const EPOCHS: usize = 14;

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
    let outs = p.topology().outputs();
    let (mut loss, mut correct, n) = (0.0f32, 0usize, 20usize);
    for i in 0..n {
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
        let _ = &outs;
    }
    (loss / n as f32, correct as f32 / n as f32)
}

fn build(cfg: &StructuralConfig) -> DynamicNetwork {
    let mut t = Topology::new(cfg.topology, cfg.params);
    let b = Birth::default();
    let inputs: Vec<_> = (0..INPUTS)
        .map(|_| {
            t.add_neuron(NeuronKind::Input, LifParams::default(), Origin::Initial, b)
                .unwrap()
        })
        .collect();
    let o0 = t
        .add_neuron(
            NeuronKind::Output,
            LifParams::new(0.9, 1.0),
            Origin::Initial,
            b,
        )
        .unwrap();
    let o1 = t
        .add_neuron(
            NeuronKind::Output,
            LifParams::new(0.9, 1.0),
            Origin::Initial,
            b,
        )
        .unwrap();
    let dead = t
        .add_neuron(
            NeuronKind::Hidden,
            LifParams::new(0.9, 1.0),
            Origin::Initial,
            b,
        )
        .unwrap();
    // Too weak connections: the outputs themselves do not spike.
    t.add_synapse(inputs[0], o0, 0.10, 0, Origin::Initial, b)
        .unwrap();
    t.add_synapse(inputs[2], o1, 0.10, 0, Origin::Initial, b)
        .unwrap();
    t.add_synapse(inputs[4], o0, 0.05, 0, Origin::Initial, b)
        .unwrap();
    t.add_synapse(inputs[5], o1, 0.05, 0, Origin::Initial, b)
        .unwrap();
    t.add_synapse(dead, o0, 0.3, 0, Origin::Initial, b).unwrap(); // neuron without inputs
    let mut ec = EngineConfig::default();
    ec.stdp.mode = PlasticityMode::Immediate;
    ec.stdp.a_plus = 0.0005;
    ec.stdp.a_minus = 0.0003;
    ec.stdp.bound = neo_snn::functional::plasticity::WeightBound::Hard {
        w_min: -0.15,
        w_max: 0.15,
    };
    DynamicNetwork::new(t, ec).unwrap()
}

fn config() -> StructuralConfig {
    let mut c = StructuralConfig::default();
    c.seed = 7;
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

fn train_epoch(net: &mut DynamicNetwork, epoch: usize) {
    let outs = net.topology().outputs();
    for i in 0..TRAIN_PER_EPOCH {
        let class = i % 2;
        net.begin_sample();
        for x in sample((epoch * 1000 + i + 1) as u64, class) {
            let _ = net.step_with_current(&x, &[(outs[class], 1.5)]);
        }
    }
}

fn main() {
    let cfg = config();
    let mut net = build(&cfg);
    let mut ctl = StructuralPlasticityController::new(cfg).unwrap();
    let (l0, a0) = evaluate(&net);
    println!(
        "start: neurons {}, synapses {}, loss {l0:.3}, accuracy {a0:.2}",
        net.topology().neuron_count(),
        net.topology().synapse_count()
    );
    for epoch in 0..EPOCHS {
        train_epoch(&mut net, epoch);
        let (loss, acc) = evaluate(&net);
        let mut eval = |p: &mut DynamicNetwork| evaluate(p).0;
        let r = ctl
            .adapt(
                &mut net,
                Metrics {
                    task_loss: loss,
                    reward: None,
                    accuracy: Some(acc),
                },
                Some(&mut eval),
            )
            .unwrap();
        let t = net.topology();
        println!(
            "epoch {epoch:2}: loss {loss:.3} acc {acc:.2} | N={} S={} (active {}) | applied {} rollback {} skip {:?}",
            t.neuron_count(),
            t.synapse_count(),
            t.active_synapse_count(),
            r.applied,
            r.rolled_back,
            r.skipped
        );
        for e in &r.events {
            println!("    {e}");
        }
    }
    let (l, a) = evaluate(&net);
    println!(
        "total: loss {l:.3} accuracy {a:.2}; events in the log: {}",
        ctl.log().len()
    );
    let bytes = save_checkpoint(&net, &ctl, b"optimizer-state-if-any");
    let ck = load_checkpoint(&bytes).unwrap();
    assert_eq!(ck.network.topology(), net.topology());
    println!(
        "checkpoint: {} bytes, topology restored identically",
        bytes.len()
    );
}
