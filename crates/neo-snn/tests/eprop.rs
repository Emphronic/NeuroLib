//! Integration tests for the online plasticity system (`structural::eprop`):
//! eligibility recurrence for LIF and AdaptiveLIF (independent reference
//! simulators), capability errors, scope, external signals, memory growth,
//! a supervised toy task, Dale/bounds, checkpoint-resume, STDP regression.
//!
//! Run: `cargo test -p neo-snn --test eprop`

use neo_snn::structural::eprop::{
    EpropConfig, EpropScope, EpropTrainer, FeedbackScheme, ReadoutMode,
};
use neo_snn::structural::{
    AdaptiveLifParams, Birth, DynamicNetwork, EngineConfig, IzhikevichParams, LifParams, NeuronId,
    NeuronModel, NeuronRole, OnlineEpropFamily, Origin, ParamLimits, StdpConfig, StructuralError,
    Topology, TopologyLimits,
};
use neo_snn::structural::{
    PlasticityMode, StructuralConfig, StructuralPlasticityController, load_checkpoint,
    save_checkpoint,
};

// ---------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------

fn order_of(net: &DynamicNetwork) -> Vec<NeuronId> {
    net.topology().neurons().map(|n| n.id).collect()
}

fn idx_of(order: &[NeuronId], id: NeuronId) -> usize {
    order.iter().position(|x| *x == id).unwrap()
}

fn weight_of(net: &DynamicNetwork, src: NeuronId, dst: NeuronId) -> f32 {
    let s = net.topology().find_synapse(src, dst).unwrap();
    net.topology().synapse(s).unwrap().weight
}

fn weights_of(net: &DynamicNetwork) -> Vec<f32> {
    net.topology().synapses().map(|s| s.weight).collect()
}

fn cfg() -> EpropConfig<f32> {
    EpropConfig {
        learning_rate: 0.0,
        trace_decay: 0.95,
        ..Default::default()
    }
}

/// Network: `i0` (Input) -> `o` (Output), LIF/LIF.
fn net_i0_to_o(beta: f32, theta: f32, w_io: f32) -> (DynamicNetwork, NeuronId, NeuronId) {
    let mut topo = Topology::new(TopologyLimits::default(), ParamLimits::default());
    let b = Birth::default();
    let i0 = topo
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
            LifParams::new(beta, theta),
            Origin::Initial,
            b,
        )
        .unwrap();
    topo.add_synapse(i0, o, w_io, 0, Origin::Initial, b)
        .unwrap();
    (
        DynamicNetwork::new(topo, EngineConfig::default()).unwrap(),
        i0,
        o,
    )
}

// ---------------------------------------------------------------------
// independent reference simulators (LIF, ALIF)
// ---------------------------------------------------------------------

struct RefLif {
    u: f32,
    beta: f32,
    theta: f32,
}
impl RefLif {
    fn step(&mut self, current: f32) -> bool {
        self.u = self.beta * self.u + current;
        let fired = self.u >= self.theta;
        if fired {
            self.u -= self.theta;
        }
        fired
    }
}

struct RefAlif {
    u: f32,
    a: f32,
    beta: f32,
    rho: f32,
    a_inc: f32,
    theta: f32,
}
impl RefAlif {
    fn step(&mut self, current: f32) -> bool {
        self.u = self.beta * self.u + current - self.a;
        let fired = self.u >= self.theta;
        if fired {
            self.u -= self.theta;
            self.a = self.rho * self.a + self.a_inc;
        } else {
            self.a *= self.rho;
        }
        fired
    }
}

/// Eligibility recurrence shared by both reference loops:
/// `e(t) = γ·e(t−1) + fired(t)·x(t)`, `x(t)` = the input spike delivered
/// from the previous step (delay-0 delivery).
///
/// `reference` steps the independent simulator with `current = w·x`.
fn assert_reference_matches(
    net: &mut DynamicNetwork,
    trainer: &mut EpropTrainer,
    inputs: &[f32],
    mut reference: impl FnMut(f32) -> bool,
) {
    let order = order_of(net);
    let o = *net.topology().outputs().first().expect("output");
    let idx_o = idx_of(&order, o);
    let (i0, sid) = {
        let s = net.topology().synapses().next().unwrap();
        (s.src, s.id)
    };
    let mut e = 0.0f32;
    let mut prev_input = false;
    for &s in inputs {
        let w = weight_of(net, i0, o);
        net.step(&[s]).unwrap();
        trainer.step(net).unwrap();
        let x = prev_input as u8 as f32;
        let fired = reference(w * x);
        let fired_f = if fired { 1.0 } else { 0.0 };
        e = 0.95 * e + fired_f * x;
        assert_eq!(
            net.last_spikes()[idx_o],
            fired,
            "engine vs reference at {s}"
        );
        assert_eq!(
            trainer.state().synapse_trace(sid).unwrap_or(0.0),
            e,
            "eligibility trace vs reference at {s}"
        );
        prev_input = s >= 0.5;
    }
}

// ---------------------------------------------------------------------
// tests
// ---------------------------------------------------------------------

/// LIF eligibility reference: no spike, single spike, repeated spikes, reset.
#[test]
fn lif_eligibility_reference() {
    // no spike: input 0 -> output silent, trace stays 0
    let (net, _i0, _o) = net_i0_to_o(0.9, 0.5, 1.0);
    let mut net_c = net;
    let mut trainer = EpropTrainer::new(&net_c, cfg()).unwrap();
    let mut ref_lif = RefLif {
        u: 0.0,
        beta: 0.9,
        theta: 0.5,
    };
    assert_reference_matches(&mut net_c, &mut trainer, &[0.0, 0.0, 0.0, 0.0], |c| {
        ref_lif.step(c)
    });

    // single spike: one delivery -> trace becomes positive
    let (net, i0, o) = net_i0_to_o(0.9, 0.5, 1.0);
    let mut net_c = net;
    let mut trainer = EpropTrainer::new(&net_c, cfg()).unwrap();
    let mut ref_lif = RefLif {
        u: 0.0,
        beta: 0.9,
        theta: 0.5,
    };
    assert_reference_matches(&mut net_c, &mut trainer, &[1.0, 0.0, 0.0], |c| {
        ref_lif.step(c)
    });
    let sid = net_c.topology().find_synapse(i0, o).unwrap();
    assert!(
        trainer.state().synapse_trace(sid).is_some(),
        "a spike must create a trace"
    );
    let _ = o;

    // repeated spikes: trace accumulates with decay
    let (net, _i0, _o) = net_i0_to_o(0.9, 0.5, 1.0);
    let mut net_c = net;
    let mut trainer = EpropTrainer::new(&net_c, cfg()).unwrap();
    let mut ref_lif = RefLif {
        u: 0.0,
        beta: 0.9,
        theta: 0.5,
    };
    assert_reference_matches(&mut net_c, &mut trainer, &[1.0, 1.0, 1.0, 1.0], |c| {
        ref_lif.step(c)
    });
}

/// ALIF eligibility reference.
#[test]
fn alif_eligibility_reference() {
    let mut topo = Topology::new(TopologyLimits::default(), ParamLimits::default());
    let b = Birth::default();
    let i0 = topo
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
            AdaptiveLifParams {
                beta: 0.9,
                rho: 0.9,
                adapt_increment: 0.5,
                threshold: 0.5,
                ..Default::default()
            },
            Origin::Initial,
            b,
        )
        .unwrap();
    topo.add_synapse(i0, o, 1.0, 0, Origin::Initial, b).unwrap();
    let net = DynamicNetwork::new(topo, EngineConfig::default()).unwrap();
    let mut net_c = net;
    let mut trainer = EpropTrainer::new(&net_c, cfg()).unwrap();
    let mut ref_alif = RefAlif {
        u: 0.0,
        a: 0.0,
        beta: 0.9,
        rho: 0.9,
        a_inc: 0.5,
        theta: 0.5,
    };
    assert_reference_matches(&mut net_c, &mut trainer, &[1.0, 1.0, 0.0, 1.0, 1.0], |c| {
        ref_alif.step(c)
    });
}

/// Unsupported dynamics: typed error for non-LIF/ALIF models; capability interface.
#[test]
fn unsupported_dynamics_is_typed_error() {
    let mut topo = Topology::new(TopologyLimits::default(), ParamLimits::default());
    let b = Birth::default();
    let _i = topo
        .add_neuron(
            NeuronRole::Input,
            LifParams::new(0.9, 1.0),
            Origin::Initial,
            b,
        )
        .unwrap();
    topo.add_neuron(
        NeuronRole::Output,
        IzhikevichParams::regular_spiking(),
        Origin::Initial,
        b,
    )
    .unwrap();
    let net = DynamicNetwork::new(topo, EngineConfig::default()).unwrap();
    let err = EpropTrainer::new(&net, EpropConfig::default()).unwrap_err();
    assert!(
        matches!(err, StructuralError::UnsupportedEpropDynamics(_)),
        "got {err:?}"
    );
    assert!(
        NeuronModel::<f32>::Izhikevich(IzhikevichParams::regular_spiking())
            .online_eprop_family()
            .is_none()
    );
    assert_eq!(
        NeuronModel::<f32>::Lif(LifParams::default()).online_eprop_family(),
        Some(OnlineEpropFamily::Lif)
    );
    assert_eq!(
        NeuronModel::<f32>::AdaptiveLif(AdaptiveLifParams::default()).online_eprop_family(),
        Some(OnlineEpropFamily::AdaptiveLif)
    );
}

/// Scope: disabled synapses and allow-lists.
#[test]
fn scope_disables_and_allows_synapses() {
    let (net, i0, o) = net_i0_to_o(0.9, 0.5, 1.0);
    let sid = net.topology().find_synapse(i0, o).unwrap();
    let w0 = weight_of(&net, i0, o);
    let mut net = net;

    // disabled synapse: no weight change
    let mut trainer = EpropTrainer::new(
        &net,
        EpropConfig {
            scope: EpropScope {
                disabled_synapses: [sid].into(),
                ..Default::default()
            },
            ..Default::default()
        },
    )
    .unwrap();
    assert!(!trainer.is_enabled_synapse(&net, sid));
    for _ in 0..10 {
        net.step(&[1.0]).unwrap();
        trainer.step(&mut net).unwrap();
    }
    assert_eq!(
        weight_of(&net, i0, o),
        w0,
        "disabled synapse must not change"
    );

    // empty allow-list: nothing enabled
    let mut trainer = EpropTrainer::new(
        &net,
        EpropConfig {
            scope: EpropScope {
                only_synapses: Some(std::iter::empty().collect()),
                ..Default::default()
            },
            ..Default::default()
        },
    )
    .unwrap();
    for _ in 0..10 {
        net.step(&[1.0]).unwrap();
        trainer.step(&mut net).unwrap();
    }
    assert_eq!(weight_of(&net, i0, o), w0);
}

/// External learning signals (RL/hybrid): stored and used.
#[test]
fn external_signals_are_stored_and_used() {
    let mut topo = Topology::new(TopologyLimits::default(), ParamLimits::default());
    let b = Birth::default();
    let i0 = topo
        .add_neuron(
            NeuronRole::Input,
            LifParams::new(0.9, 1.0),
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
    let net = DynamicNetwork::new(topo, EngineConfig::default()).unwrap();
    let mut trainer = EpropTrainer::new(
        &net,
        EpropConfig {
            readout_mode: ReadoutMode::External,
            feedback: FeedbackScheme::None,
            ..Default::default()
        },
    )
    .unwrap();
    trainer.set_external_signal(h, 0.7);
    trainer.set_external_signal(o, -0.3);
    assert_eq!(trainer.state().external_signals().get(&h), Some(&0.7));
    assert_eq!(trainer.state().external_signals().get(&o), Some(&-0.3));
    let mut net = net;
    net.step(&[1.0]).unwrap();
    let report = trainer.step(&mut net).unwrap();
    assert!(report.learning_signals >= 1);
}

/// Online memory: the trainer state is bounded by O(S) — traces never grow
/// with the sequence length.
#[test]
fn online_memory_does_not_grow_with_sequence() {
    let (net, _i0, _o) = net_i0_to_o(0.9, 0.5, 1.0);
    let mut net = net;
    let mut trainer = EpropTrainer::new(&net, EpropConfig::default()).unwrap();
    let mut trace_counts = Vec::new();
    for _ in 0..300 {
        net.step(&[1.0]).unwrap();
        trainer.step(&mut net).unwrap();
        if net.step_count() % 50 == 0 {
            trace_counts.push(trainer.state().synapse_trace_count());
        }
    }
    assert_eq!(trainer.state().steps(), 300);
    let s_count = net.topology().active_synapse_count();
    assert!(
        trace_counts.iter().all(|&c| c <= s_count),
        "trace counts {trace_counts:?} > {s_count}"
    );
}

/// Supervised toy task: output should stay quiet; e-prop changes weights and
/// the output rate improves relative to the start.
#[test]
fn supervised_toy_task_trains_without_bptt() {
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
    let order = order_of(&net);
    let idx_o = idx_of(&order, o);
    let mut trainer = EpropTrainer::new(
        &net,
        EpropConfig {
            learning_rate: 0.05,
            trace_decay: 0.95,
            ..Default::default()
        },
    )
    .unwrap();
    let (mut rate_first, mut rate_last) = (0.0f32, 0.0f32);
    for t in 0..300u32 {
        let s = ((t / 3) % 2 == 0) as u8 as f32; // bursts
        net.step(&[s]).unwrap();
        trainer.set_target(o, 0.0);
        trainer.step(&mut net).unwrap();
        let r = net.last_spikes()[idx_o] as u8 as f32;
        if t < 60 {
            rate_first += r;
        }
        if t >= 240 {
            rate_last += r;
        }
    }
    rate_first /= 60.0;
    rate_last /= 60.0;
    assert!(
        rate_first > 0.0,
        "the network should fire during bursts initially: {rate_first}"
    );
    assert!(
        rate_last < rate_first,
        "loss should improve: rate {rate_first:.3} -> {rate_last:.3}"
    );
    assert!(
        trainer.state().updates() > 0,
        "the trainer should have updated weights"
    );
    assert_eq!(net.topology().validate(), Ok(()));
}

/// Dale's principle + weight bounds are respected after training.
#[test]
fn dale_and_weight_bounds_respected() {
    let mut topo = Topology::new(
        TopologyLimits::default(),
        ParamLimits {
            weight_abs_max: 0.05,
            ..Default::default()
        },
    );
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
    topo.add_synapse(i0, h, 0.05, 0, Origin::Initial, b)
        .unwrap();
    topo.add_synapse(h, o, 0.05, 0, Origin::Initial, b).unwrap();
    let mut net: DynamicNetwork<f32> = DynamicNetwork::new(topo, EngineConfig::default()).unwrap();
    let mut trainer = EpropTrainer::new(&net, EpropConfig::default()).unwrap();
    for _ in 0..50 {
        net.step(&[1.0]).unwrap();
        trainer.step(&mut net).unwrap();
    }
    assert_eq!(net.topology().validate(), Ok(()));
    for s in net.topology().synapses() {
        assert!(
            s.weight.abs() <= 0.05 + 1e-6,
            "weight {} out of bounds",
            s.weight
        );
        assert!(s.weight >= -1e-6, "Dale violated: weight {}", s.weight);
    }
}

/// Checkpoint-resume: saving the trainer state and continuing on a restored
/// copy produces identical weights/traces; the full network checkpoint works.
#[test]
fn checkpoint_resume_preserves_state() {
    let (net, _i0, _o) = net_i0_to_o(0.9, 0.5, 1.0);
    let mut net = net;
    let cfg = EpropConfig::default();
    let mut trainer = EpropTrainer::new(&net, cfg.clone()).unwrap();
    for _ in 0..30 {
        net.step(&[1.0]).unwrap();
        trainer.step(&mut net).unwrap();
    }
    let bytes = trainer.save_state().unwrap();
    assert!(!bytes.is_empty());

    let mut net2 = net.clone();
    let mut trainer2 = EpropTrainer::new(&net2, cfg).unwrap();
    trainer2.load_state(&bytes).unwrap();

    for _ in 0..20 {
        net.step(&[1.0]).unwrap();
        trainer.step(&mut net).unwrap();
        net2.step(&[1.0]).unwrap();
        trainer2.step(&mut net2).unwrap();
    }
    assert_eq!(weights_of(&net), weights_of(&net2));
    assert_eq!(trainer.state().steps(), 50);
    assert_eq!(trainer.state(), trainer2.state());

    // Full network checkpoint: restores identically.
    let ctl = StructuralPlasticityController::new(StructuralConfig::default()).unwrap();
    let ck = load_checkpoint(&save_checkpoint(&net, &ctl, b"").unwrap()).unwrap();
    assert_eq!(weights_of(&ck.network), weights_of(&net));
}

/// Hybrid composition: STDP without a trainer behaves as before; the trainer
/// with an empty scope is a no-op.
#[test]
fn stdp_without_trainer_and_disabled_trainer_match() {
    let mut topo1 = Topology::new(TopologyLimits::default(), ParamLimits::default());
    let mut topo2 = Topology::new(TopologyLimits::default(), ParamLimits::default());
    let b = Birth::default();
    for topo in [&mut topo1, &mut topo2] {
        let i0 = topo
            .add_neuron(
                NeuronRole::Input,
                LifParams::new(0.9, 0.5),
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
        topo.add_synapse(i0, o, 1.0, 0, Origin::Initial, b).unwrap();
    }
    let engine_cfg = EngineConfig {
        stdp: StdpConfig {
            mode: PlasticityMode::Immediate,
            ..Default::default()
        },
        ..Default::default()
    };
    let mut net1 = DynamicNetwork::new(topo1, engine_cfg.clone()).unwrap();
    let mut net2 = DynamicNetwork::new(topo2, engine_cfg).unwrap();
    let mut trainer = EpropTrainer::new(
        &net2,
        EpropConfig {
            scope: EpropScope::none(),
            ..Default::default()
        },
    )
    .unwrap();
    let init = weights_of(&net1);
    for _ in 0..20 {
        net1.step(&[1.0]).unwrap();
        net2.step(&[1.0]).unwrap();
        trainer.step(&mut net2).unwrap();
    }
    assert_eq!(
        weights_of(&net1),
        weights_of(&net2),
        "trainer with empty scope must not change weights"
    );
    assert_ne!(
        init,
        weights_of(&net1),
        "STDP should change weights on its own"
    );
}

/// Feedback schemes: symmetric (mirrored weights) and random-fixed.
#[test]
fn feedback_schemes() {
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
    topo.add_synapse(h, o, 0.7, 0, Origin::Initial, b).unwrap();
    let net = DynamicNetwork::new(topo, EngineConfig::default()).unwrap();

    let t = EpropTrainer::new(
        &net,
        EpropConfig {
            feedback: FeedbackScheme::Symmetric,
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!(
        t.feedback_weights().get(&h).and_then(|m| m.get(&o)),
        Some(&0.7)
    );

    let fb = |t: EpropTrainer<f32>| {
        t.feedback_weights()
            .get(&h)
            .and_then(|m| m.get(&o))
            .copied()
    };
    let v1 = fb(EpropTrainer::new(
        &net,
        EpropConfig {
            feedback: FeedbackScheme::RandomFixed {
                seed: 42,
                magnitude: 0.2,
            },
            ..Default::default()
        },
    )
    .unwrap());
    let v2 = fb(EpropTrainer::new(
        &net,
        EpropConfig {
            feedback: FeedbackScheme::RandomFixed {
                seed: 42,
                magnitude: 0.2,
            },
            ..Default::default()
        },
    )
    .unwrap());
    assert_eq!(v1, v2, "same seed -> same feedback");
    assert!(v1.is_some_and(|v: f32| v.abs() <= 0.2 + 1e-6));
}

/// `begin_sample`: counter +1 and detach (traces cleared).
#[test]
fn begin_sample_detaches_traces() {
    let (net, _i0, _o) = net_i0_to_o(0.9, 0.5, 1.0);
    let mut net = net;
    let mut trainer = EpropTrainer::new(&net, EpropConfig::default()).unwrap();
    // two steps: input spike (t=0) is delivered to the output at t=1
    net.step(&[1.0]).unwrap();
    trainer.step(&mut net).unwrap();
    net.step(&[1.0]).unwrap();
    trainer.step(&mut net).unwrap();
    assert!(
        trainer.state().synapse_trace_count() > 0,
        "the output should have spiked, making a trace"
    );
    trainer.begin_sample(&mut net);
    assert_eq!(trainer.state().samples(), 1);
    assert_eq!(trainer.state().synapse_trace_count(), 0);
    assert_eq!(net.sample_count(), 1);
}

/// Stale state: `trainer.step` before `net.step` is an error.
#[test]
fn step_before_net_step_is_an_error() {
    let (net, _i0, _o) = net_i0_to_o(0.9, 0.5, 1.0);
    let mut net = net;
    let mut trainer = EpropTrainer::new(&net, EpropConfig::default()).unwrap();
    let err = trainer.step(&mut net).unwrap_err();
    assert!(
        matches!(err, StructuralError::InvalidParameter { .. }),
        "got {err:?}"
    );
}

/// `EpropState` roundtrip through `to_bytes`/`from_bytes` via the trainer.
#[test]
fn state_bytes_roundtrip() {
    let (net, _i0, _o) = net_i0_to_o(0.9, 0.5, 1.0);
    let mut net = net;
    let mut trainer = EpropTrainer::new(&net, EpropConfig::default()).unwrap();
    net.step(&[1.0]).unwrap();
    trainer.step(&mut net).unwrap();
    let bytes = trainer.state().to_bytes().unwrap();
    let mut trainer2 = EpropTrainer::new(&net, EpropConfig::default()).unwrap();
    trainer2.load_state(&bytes).unwrap();
    assert_eq!(trainer2.state().steps(), trainer.state().steps());
    assert_eq!(trainer2.state(), trainer.state());
}
