//! Learnable per-synapse delays on `DynamicNetwork` (Input → Output).
//!
//! With `EngineConfig::learnable_delay` the engine uses per-synapse
//! [`TrainableDelayState`] (`delay`, `elig`): at each spike dispatch the
//! three-factor rule shifts the delay by `learning_rate · elig`, where
//! `elig` accumulates the teaching signal of the postsynaptic neuron
//! (`set_teaching_signal`/`set_teaching_signals`).
//!
//! Run: `cargo run --example learnable_delay -p neo-snn`

use neo_snn::structural::{
    Birth, DynamicNetwork, EngineConfig, LearnableDelayConfig, LifParams, NeuronRole, Origin,
    Topology, TopologyLimits,
};

fn main() {
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
    let sid = topo.add_synapse(i, o, 0.5, 0, Origin::Initial, b).unwrap();

    let mut cfg = EngineConfig::default();
    cfg.learnable_delay = Some(LearnableDelayConfig {
        enabled: true,
        min_delay: 0.0,
        max_delay: 4.0,
        initial: 0.5,
        learning_rate: 0.1,
        elig_decay: 0.9,
    });
    let mut net = DynamicNetwork::new(topo, cfg).unwrap();

    for step in 0..10u32 {
        net.set_teaching_signal(o, 1.0); // positive teaching signal → delay grows
        net.step(&[1.0]).unwrap();
        println!(
            "step {:>2}: learnable delay = {:.3}",
            step + 1,
            net.learnable_delay(sid).unwrap_or(0.0)
        );
    }
}
