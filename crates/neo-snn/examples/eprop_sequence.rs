//! Online sequence training without backprop: the dynamic network learns on
//! the fly, one local weight update per step, from eligibility traces and
//! readout errors — no BPTT, no surrogate gradients, no reward modulations.
//!
//! Run: `cargo run --example eprop_sequence -p neo-snn`
//!
//! Design: `i0 → h → o` (LIF, β = 0.9, θ = 0.5), target = stay quiet
//! (readout error `d_o = ŷ_o − 0`), input = periodic bursts. During bursts
//! the network fires; the trainer depresses the recurrent weights, so the
//! output rate should drop over the run.

use neo_snn::structural::eprop::{EpropConfig, EpropTrainer};
use neo_snn::structural::{
    Birth, DynamicNetwork, EngineConfig, LifParams, NeuronRole, Origin, ParamLimits, Topology,
    TopologyLimits,
};

fn main() {
    let mut topo = Topology::new(TopologyLimits::default(), ParamLimits::default());
    let b = Birth::default();
    let i0 = topo
        .add_neuron(
            NeuronRole::Input,
            LifParams::new(0.9, 0.5),
            Origin::Initial,
            b,
        )
        .unwrap();
    let h = topo
        .add_neuron(
            NeuronRole::Hidden,
            LifParams::new(0.9, 0.5),
            Origin::Manual,
            b,
        )
        .unwrap();
    let o = topo
        .add_neuron(
            NeuronRole::Output,
            LifParams::new(0.9, 0.5),
            Origin::Initial,
            b,
        )
        .unwrap();
    topo.add_synapse(i0, h, 1.0, 0, Origin::Initial, b).unwrap();
    topo.add_synapse(h, o, 1.0, 0, Origin::Initial, b).unwrap();
    let mut net = DynamicNetwork::new(topo, EngineConfig::default()).unwrap();
    let order: Vec<_> = net.topology().neurons().map(|n| n.id).collect();
    let idx_o = order.iter().position(|&n| n == o).unwrap();

    let mut trainer = EpropTrainer::new(
        &net,
        EpropConfig {
            learning_rate: 0.05,
            trace_decay: 0.95,
            ..Default::default()
        },
    )
    .unwrap();

    let steps = 600;
    let (mut fired, mut window) = (0u32, 0u32);
    let (mut rate_first, mut rate_last) = (0.0f32, 0.0f32);
    let mut first_done = false;
    for t in 0..steps {
        let s = ((t / 3) % 2 == 0) as u8 as f32; // bursts
        net.step(&[s]).unwrap();
        trainer.set_target(o, 0.0);
        trainer.step(&mut net).unwrap();
        fired += if net.last_spikes()[idx_o] { 1 } else { 0 };
        window += 1;
        if t % 100 == 99 {
            let rate = fired as f32 / window as f32;
            if !first_done && t < 200 {
                rate_first = rate;
                first_done = true;
            }
            if t >= 500 {
                rate_last = rate;
            }
            println!("step {t:>4}: output rate = {rate:.3}");
            fired = 0;
            window = 0;
        }
    }
    assert_eq!(net.topology().validate(), Ok(()));
    println!(
        "trained steps: {}, updates applied: {}, trainer traces: {}",
        trainer.state().steps(),
        trainer.state().updates(),
        trainer.state().synapse_trace_count()
    );
    println!("output rate: first {rate_first:.3} -> last {rate_last:.3}");
    assert!(rate_first > 0.0, "the network should fire during bursts");
    assert!(
        rate_last < rate_first,
        "the rate should drop as the trainer depresses the weights"
    );
}
