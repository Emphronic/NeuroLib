//! Online E-prop training of a tiny `DynamicNetwork` (Input → Output LIF).
//!
//! `EpropTrainer` holds the eligibility traces (`e`), learning signals (`d`)
//! and drives weight updates online — `net.step(&inputs)?; trainer.step(&mut net)?;`
//! — no backprop through time, O(N + S) memory.
//!
//! Run: `cargo run --example eprop_dynamic -p neo-snn`

use neo_snn::structural::{
    Birth, DynamicNetwork, EngineConfig, EpropConfig, EpropTrainer, LifParams, NeuronRole, Origin,
    Topology, TopologyLimits,
};

fn main() {
    // Network: Input → Output (LIF/LIF), trainable synapse.
    let mut topo = Topology::new(TopologyLimits::default(), Default::default());
    let b = Birth::default();
    let i = topo
        .add_neuron(
            NeuronRole::Input,
            LifParams::new(0.9, 1.0),
            Origin::Initial,
            b,
        )
        .unwrap();
    let o = topo
        .add_neuron(
            NeuronRole::Output,
            LifParams::new(0.9, 0.4),
            Origin::Initial,
            b,
        )
        .unwrap();
    let sid = topo.add_synapse(i, o, 0.4, 0, Origin::Initial, b).unwrap();
    let mut net = DynamicNetwork::new(topo, EngineConfig::default()).unwrap();

    // Trainer: LocalError readout (outputs: d_j = spike − target), η = 0.1.
    let mut cfg = EpropConfig::default();
    cfg.learning_rate = 0.1;
    cfg.trace_decay = 0.95;
    let mut trainer = EpropTrainer::new(&net, cfg).unwrap();
    trainer.set_target(o, 0.0); // the output should stop spiking

    let w0 = net.topology().synapse(sid).unwrap().weight;
    for step in 0..12u32 {
        net.step(&[1.0]).unwrap();
        let rep = trainer.step(&mut net).unwrap();
        println!(
            "step {:>2}: w = {:.3}, learning_signals = {}, weights_updated = {}, traces = {}",
            step,
            net.topology().synapse(sid).unwrap().weight,
            rep.learning_signals,
            rep.weights_updated,
            trainer.state().synapse_trace_count()
        );
    }
    let w1 = net.topology().synapse(sid).unwrap().weight;
    println!("w: {w0:.3} -> {w1:.3}");
}
