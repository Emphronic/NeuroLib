//! Rough measurement of the step speed of a dynamic network and the cost of structural operations
//! with increasing graph size. Launch: `cargo run --release --example bench_structural -p neo-snn`
//! (this is a guideline, not a strict benchmark: one run, without warming up and statistics).
#![allow(clippy::field_reassign_with_default)]

use std::time::Instant;

use neo_snn::structural::*;

fn build(n_hidden: usize, fan_out: usize) -> DynamicNetwork {
    let mut cfg = StructuralConfig::default();
    cfg.topology.max_neurons = 1_000_000;
    cfg.topology.max_synapses = 100_000_000;
    cfg.topology.max_in_degree = 10_000;
    cfg.topology.max_out_degree = 10_000;
    let mut t = Topology::new(cfg.topology, cfg.params);
    let b = Birth::default();
    let ins: Vec<_> = (0..32)
        .map(|_| {
            t.add_neuron(NeuronKind::Input, LifParams::default(), Origin::Initial, b)
                .unwrap()
        })
        .collect();
    let hid: Vec<_> = (0..n_hidden)
        .map(|_| {
            t.add_neuron(
                NeuronKind::Hidden,
                LifParams::new(0.9, 1.0),
                Origin::Initial,
                b,
            )
            .unwrap()
        })
        .collect();
    let out = t
        .add_neuron(
            NeuronKind::Output,
            LifParams::new(0.9, 1.0),
            Origin::Initial,
            b,
        )
        .unwrap();
    let mut rng = Rng64::new(1);
    for (k, i) in ins.iter().enumerate() {
        for _ in 0..fan_out {
            let h = hid[rng.below(hid.len())];
            let _ = t.add_synapse(*i, h, 0.5, 0, Origin::Initial, b);
            let _ = k;
        }
    }
    for h in &hid {
        for _ in 0..fan_out {
            let d = hid[rng.below(hid.len())];
            let _ = t.add_synapse(*h, d, 0.3, 0, Origin::Initial, b);
        }
        let _ = t.add_synapse(*h, out, 0.1, 0, Origin::Initial, b);
    }
    DynamicNetwork::new(t, EngineConfig::default()).unwrap()
}

fn main() {
    println!(
        "{:>8} {:>10} {:>12} {:>14} {:>16}",
        "neurons", "synapses", "steps/s", "split, μs", "remove_neuron, μs"
    );
    for n in [100usize, 1_000, 5_000, 20_000] {
        let mut net = build(n, 8);
        let inputs: Vec<f32> = (0..32)
            .map(|i| if i % 3 == 0 { 1.0 } else { 0.0 })
            .collect();
        let steps = 200;
        let t0 = Instant::now();
        for _ in 0..steps {
            net.step(&inputs).unwrap();
        }
        let sps = steps as f64 / t0.elapsed().as_secs_f64();
        let id = net.topology().synapses().next().map(|s| s.id).unwrap();
        let mut rng = Rng64::new(3);
        let t1 = Instant::now();
        apply_action(
            &mut net,
            &StructuralAction::SplitSynapse {
                id,
                init: SplitInit::default(),
            },
            &mut rng,
        )
        .unwrap();
        let split_us = t1.elapsed().as_micros();
        let h = net.topology().hidden()[0];
        let t2 = Instant::now();
        apply_action(
            &mut net,
            &StructuralAction::RemoveNeuron { id: h },
            &mut rng,
        )
        .unwrap();
        let rm_us = t2.elapsed().as_micros();
        println!(
            "{:>8} {:>10} {:>12.0} {:>14} {:>16}",
            net.topology().neuron_count(),
            net.topology().synapse_count(),
            sps,
            split_us,
            rm_us
        );
    }
}
