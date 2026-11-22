//! Unified per-synapse plasticity pipeline: assignments of rules (PairStdp,
//! Clopath, Vogels inhibitory STDP, BCM) to specific synapses — engine STDP off,
//! one hook per step in `DynamicNetwork::step`.
//!
//! Run: `cargo run --example plasticity_pipeline -p neo-snn`
//!
//! Design: `Input → H(inhibitory) → Output` (LIF), mixed assignment:
//! * default `PairStdp` on excitatory links (`Input→Output`, `H→Output`);
//! * `InhibitoryStdp` (Vogels) on the inhibitory-source link (class rule);
//! * `Clopath` on the `Input→Output` link (explicit per-synapse assignment).

use neo_snn::structural::{
    Birth, DynamicNetwork, EngineConfig, LifParams, NeuronPolarity, NeuronRole, Origin,
    ParamLimits, PlasticityPipeline, PlasticityPipelineConfig, RuleClass, RuleConfig, RuleEntry,
    RuleId, Topology, TopologyLimits,
};

fn main() {
    let mut topo = Topology::new(TopologyLimits::default(), ParamLimits::default());
    let b = Birth::default();
    let i = topo
        .add_neuron(
            NeuronRole::Input,
            LifParams::new(0.9, 0.4),
            Origin::Initial,
            b,
        )
        .unwrap();
    let h = topo
        .add_neuron_with(
            NeuronRole::Hidden,
            NeuronPolarity::Inhibitory,
            LifParams::new(0.9, 1.0),
            Origin::Initial,
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
    let s_io = topo.add_synapse(i, o, 0.4, 0, Origin::Initial, b).unwrap();
    let s_ih = topo.add_synapse(i, h, 0.5, 0, Origin::Initial, b).unwrap();
    topo.add_synapse(h, o, -0.5, 0, Origin::Initial, b).unwrap();

    // Pipeline config: default PairStdp; class rule on inhibitory sources
    // (Vogels); per-synapse Clopath on Input→Output.
    let mut pl_cfg = PlasticityPipelineConfig::default();
    pl_cfg.default_rules = vec![RuleEntry::new(RuleId::PairStdp)];
    pl_cfg.class_rules.push((
        RuleClass::ByPolarity {
            src: Some(NeuronPolarity::Inhibitory),
            dst: None,
        },
        vec![RuleEntry::new(RuleId::InhibitoryStdp)],
    ));
    pl_cfg
        .synapse_rules
        .insert(s_io, vec![RuleEntry::new(RuleId::Clopath)]);
    pl_cfg.rule_configs.insert(
        RuleId::Clopath,
        RuleConfig::Clopath {
            a_ltp: 0.05,
            a_dep: 0.01,
            theta_ltp: 0.5,
            theta_dep: 0.5,
            tau_v: 0.95,
        },
    );

    let mut net = DynamicNetwork::new(topo, EngineConfig::default()).unwrap();
    net.set_plasticity_pipeline(Some(PlasticityPipeline::new(pl_cfg)))
        .unwrap();

    let w0_io = net.topology().synapse(s_io).unwrap().weight;
    let w0_ih = net.topology().synapse(s_ih).unwrap().weight;
    for step in 0..20 {
        net.step(&[1.0]).unwrap();
        if step % 5 == 4 {
            println!(
                "step {:>2}: w(Input→Output, Clopath) = {:.3}, w(Input→H, inhibitory) = {:.3}",
                step + 1,
                net.topology().synapse(s_io).unwrap().weight,
                net.topology().synapse(s_ih).unwrap().weight
            );
        }
    }
    println!(
        "weights moved: io {:.3} -> {:.3}, ih {:.3} -> {:.3}",
        w0_io,
        net.topology().synapse(s_io).unwrap().weight,
        w0_ih,
        net.topology().synapse(s_ih).unwrap().weight
    );
}
