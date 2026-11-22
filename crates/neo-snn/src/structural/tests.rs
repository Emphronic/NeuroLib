//! Unit tests of structural plasticity.
#![allow(clippy::field_reassign_with_default)]

use std::collections::BTreeSet;

use super::*;
use crate::functional::plasticity::WeightBound;

fn b() -> Birth {
    Birth::default()
}

fn topo(limits: TopologyLimits) -> Topology {
    Topology::new(limits, ParamLimits::default())
}

fn inp(t: &mut Topology) -> NeuronId {
    t.add_neuron(
        NeuronKind::Input,
        LifParams::default(),
        Origin::Initial,
        b(),
    )
    .unwrap()
}

fn hid(t: &mut Topology, beta: f32, th: f32) -> NeuronId {
    t.add_neuron(
        NeuronKind::Hidden,
        LifParams::new(beta, th),
        Origin::Initial,
        b(),
    )
    .unwrap()
}

fn out(t: &mut Topology, beta: f32, th: f32) -> NeuronId {
    t.add_neuron(
        NeuronKind::Output,
        LifParams::new(beta, th),
        Origin::Initial,
        b(),
    )
    .unwrap()
}

fn syn(t: &mut Topology, s: NeuronId, d: NeuronId, w: f32) -> SynapseId {
    t.add_synapse(s, d, w, 0, Origin::Initial, b()).unwrap()
}

fn net(t: Topology) -> DynamicNetwork {
    DynamicNetwork::new(t, EngineConfig::default()).unwrap()
}

fn stdp_net(t: Topology, mode: PlasticityMode) -> DynamicNetwork {
    let mut cfg = EngineConfig::default();
    cfg.stdp = StdpConfig {
        mode,
        a_plus: 0.1,
        a_minus: 0.1,
        trace_decay: 0.9,
        bound: WeightBound::Hard {
            w_min: -1.0,
            w_max: 1.0,
        },
        elig_decay: 0.9,
        reward_lr: 1.0,
    };
    DynamicNetwork::new(t, cfg).unwrap()
}

/// Run a sequence of inputs (one input neuron).
fn run(net: &mut DynamicNetwork, pattern: &[f32]) -> Vec<Vec<bool>> {
    pattern
        .iter()
        .map(|x| net.step(&[*x]).unwrap().output_spikes)
        .collect()
}

// ──────────────────────── Topology ─────────────────────────

#[test]
fn add_and_remove_neuron_removes_all_its_synapses() {
    let mut t = topo(TopologyLimits::default());
    let i = inp(&mut t);
    let h = hid(&mut t, 0.9, 1.0);
    let o = out(&mut t, 0.9, 1.0);
    syn(&mut t, i, h, 1.0);
    syn(&mut t, h, o, 1.0);
    syn(&mut t, i, o, 0.5);
    let removed = t.remove_neuron(h).unwrap();
    assert_eq!(removed.synapses.len(), 2);
    assert_eq!(t.synapse_count(), 1);
    assert!(t.neuron(h).is_none());
    t.validate().unwrap();
}

#[test]
fn self_loop_removed_with_neuron_exactly_once() {
    let mut lim = TopologyLimits::default();
    lim.allow_self_loops = true;
    let mut t = topo(lim);
    let h = hid(&mut t, 0.9, 1.0);
    syn(&mut t, h, h, 0.3);
    assert_eq!(t.remove_neuron(h).unwrap().synapses.len(), 1);
    t.validate().unwrap();
}

#[test]
fn input_and_output_neurons_cannot_be_removed_or_cloned() {
    let mut t = topo(TopologyLimits::default());
    let i = inp(&mut t);
    let o = out(&mut t, 0.9, 1.0);
    assert!(matches!(
        t.remove_neuron(i),
        Err(StructuralError::WrongKind { .. })
    ));
    assert!(matches!(
        t.remove_neuron(o),
        Err(StructuralError::WrongKind { .. })
    ));
    let mut rng = Rng64::new(1);
    assert!(matches!(
        t.clone_neuron(o, 0.0, &mut rng, b()),
        Err(StructuralError::WrongKind { .. })
    ));
    assert!(matches!(
        t.remove_neuron(NeuronId(99)),
        Err(StructuralError::NoSuchNeuron(_))
    ));
}

#[test]
fn ids_are_never_reused() {
    let mut t = topo(TopologyLimits::default());
    let h1 = hid(&mut t, 0.9, 1.0);
    t.remove_neuron(h1).unwrap();
    let h2 = hid(&mut t, 0.9, 1.0);
    assert_ne!(h1, h2);
}

#[test]
fn synapse_rules_are_enforced() {
    let mut lim = TopologyLimits::default();
    lim.max_synapses = 3;
    lim.max_in_degree = 2;
    let mut t = topo(lim);
    let i = inp(&mut t);
    let i2 = inp(&mut t);
    let h = hid(&mut t, 0.9, 1.0);
    let o = out(&mut t, 0.9, 1.0);
    syn(&mut t, i, o, 1.0);
    assert!(matches!(
        t.add_synapse(i, o, 1.0, 0, Origin::Manual, b()),
        Err(StructuralError::DuplicateSynapse { .. })
    ));
    assert!(matches!(
        t.add_synapse(h, h, 1.0, 0, Origin::Manual, b()),
        Err(StructuralError::SelfLoopForbidden(_))
    ));
    assert!(matches!(
        t.add_synapse(h, i, 1.0, 0, Origin::Manual, b()),
        Err(StructuralError::InputCannotReceive(_))
    ));
    assert!(matches!(
        t.add_synapse(i, o, f32::NAN, 1, Origin::Manual, b()),
        Err(StructuralError::InvalidParameter { .. })
    ));
    assert!(matches!(
        t.add_synapse(i2, o, 1.0, 99, Origin::Manual, b()),
        Err(StructuralError::InvalidParameter { .. })
    ));
    syn(&mut t, i2, o, 1.0);
    assert!(matches!(
        t.add_synapse(h, o, 1.0, 0, Origin::Manual, b()),
        Err(StructuralError::LimitExceeded {
            what: "max_in_degree",
            ..
        })
    ));
    syn(&mut t, i, h, 1.0);
    assert!(matches!(
        t.add_synapse(i2, h, 1.0, 0, Origin::Manual, b()),
        Err(StructuralError::LimitExceeded {
            what: "max_synapses",
            ..
        })
    ));
    // weight is limited by physical limit
    let s = t.synapses().next().unwrap().id;
    assert_eq!(
        t.set_synapse_weight(s, 100.0).unwrap(),
        ParamLimits::<f32>::default().weight_abs_max
    );
    t.validate().unwrap();
}

#[test]
fn recurrent_cycles_allowed_or_forbidden_by_config() {
    let mut t = topo(TopologyLimits::default());
    let a = hid(&mut t, 0.9, 1.0);
    let c = hid(&mut t, 0.9, 1.0);
    syn(&mut t, a, c, 1.0);
    syn(&mut t, c, a, 1.0); // recurrence is allowed by default
    t.validate().unwrap();

    let mut lim = TopologyLimits::default();
    lim.allow_recurrent = false;
    let mut t = topo(lim);
    let a = hid(&mut t, 0.9, 1.0);
    let c = hid(&mut t, 0.9, 1.0);
    syn(&mut t, a, c, 1.0);
    assert!(matches!(
        t.add_synapse(c, a, 1.0, 0, Origin::Manual, b()),
        Err(StructuralError::RecurrenceForbidden { .. })
    ));
}

#[test]
fn rewire_keeps_id_and_weight_and_checks_rules() {
    let mut t = topo(TopologyLimits::default());
    let i = inp(&mut t);
    let h = hid(&mut t, 0.9, 1.0);
    let o = out(&mut t, 0.9, 1.0);
    let s = syn(&mut t, i, h, 0.7);
    syn(&mut t, i, o, 0.2);
    assert!(matches!(
        t.rewire_synapse(s, None, Some(o)),
        Err(StructuralError::DuplicateSynapse { .. })
    ));
    assert_eq!(t.synapse(s).unwrap().dst, h); // hasn't changed
    t.rewire_synapse(s, Some(h), Some(o)).unwrap();
    let y = t.synapse(s).unwrap();
    assert_eq!((y.src, y.dst, y.weight), (h, o, 0.7));
    assert!(t.out_synapses(i).iter().all(|x| *x != s));
    assert!(t.in_synapses(o).contains(&s));
    t.validate().unwrap();
}

#[test]
fn transaction_rolls_back_on_error_and_on_invalid_result() {
    let mut t = topo(TopologyLimits::default());
    let i = inp(&mut t);
    let o = out(&mut t, 0.9, 1.0);
    syn(&mut t, i, o, 1.0);
    let before = t.clone();
    let r: StructuralResult<()> = t.transaction(|t| {
        let h = t.add_neuron(
            NeuronKind::Hidden,
            LifParams::default(),
            Origin::Manual,
            b(),
        )?;
        t.add_synapse(i, h, 1.0, 0, Origin::Manual, b())?;
        t.add_synapse(i, h, 1.0, 0, Origin::Manual, b())?; // duplicate -> error
        Ok(())
    });
    assert!(r.is_err());
    assert_eq!(t, before);
    // the closure “successfully” breaks the invariant - the transaction is also rolled back
    let r: StructuralResult<()> = t.transaction(|t| {
        let s = t.synapses().next().unwrap().id;
        t.synapses.get_mut(&s).unwrap().dst = NeuronId(777);
        Ok(())
    });
    assert!(r.is_err());
    assert_eq!(t, before);
}

#[test]
fn validate_detects_corruption() {
    let mut t = topo(TopologyLimits::default());
    let i = inp(&mut t);
    let o = out(&mut t, 0.9, 1.0);
    let s = syn(&mut t, i, o, 1.0);
    t.validate().unwrap();
    let mut bad = t.clone();
    bad.synapses.get_mut(&s).unwrap().weight = f32::NAN;
    assert!(bad.validate().is_err());
    let mut bad = t.clone();
    bad.out_adj.get_mut(&i).unwrap().clear();
    assert!(bad.validate().is_err());
    let mut bad = t.clone();
    bad.neurons.get_mut(&o).unwrap().model = NeuronModel::Lif(LifParams {
        beta: 2.0,
        ..LifParams::default()
    });
    assert!(bad.validate().is_err());
}

#[test]
fn many_random_operations_keep_invariants() {
    let mut lim = TopologyLimits::default();
    lim.allow_self_loops = true;
    lim.max_neurons = 40;
    lim.max_synapses = 200;
    lim.max_in_degree = 12;
    lim.max_out_degree = 12;
    let mut t = topo(lim);
    for _ in 0..3 {
        inp(&mut t);
    }
    for _ in 0..2 {
        out(&mut t, 0.9, 1.0);
    }
    let mut rng = Rng64::new(42);
    let mut crng = Rng64::new(7);
    let mut seen_n = BTreeSet::new();
    let mut seen_s = BTreeSet::new();
    for step in 0..4000u64 {
        let ids: Vec<NeuronId> = t.neurons().map(|n| n.id).collect();
        let sids: Vec<SynapseId> = t.synapses().map(|s| s.id).collect();
        let pick_n = |r: &mut Rng64| ids[r.below(ids.len())];
        let birth = Birth {
            step,
            sample: step / 10,
        };
        match rng.below(8) {
            0 => {
                let _ = t.add_neuron(
                    NeuronKind::Hidden,
                    LifParams::new(0.5 + 0.4 * rng.next_f32(), 0.5 + rng.next_f32()),
                    Origin::Manual,
                    birth,
                );
            }
            1 | 2 => {
                let (s, d) = (pick_n(&mut rng), pick_n(&mut rng));
                let _ = t.add_synapse(
                    s,
                    d,
                    rng.symmetric() * 2.0,
                    rng.below(4) as u16,
                    Origin::Manual,
                    birth,
                );
            }
            3 if !sids.is_empty() => {
                let _ = t.remove_synapse(sids[rng.below(sids.len())]);
            }
            4 => {
                let _ = t.remove_neuron(pick_n(&mut rng));
            }
            5 if !sids.is_empty() => {
                let (s, d) = (pick_n(&mut rng), pick_n(&mut rng));
                let _ = t.rewire_synapse(sids[rng.below(sids.len())], Some(s), Some(d));
            }
            6 => {
                let _ = t.clone_neuron(pick_n(&mut rng), 0.1, &mut crng, birth);
            }
            7 if !sids.is_empty() => {
                let _ = t.split_synapse(sids[rng.below(sids.len())], SplitInit::default(), birth);
            }
            _ => {}
        }
        t.validate().unwrap_or_else(|e| panic!("step {step}: {e}"));
        for n in t.neurons() {
            seen_n.insert(n.id);
        }
        for s in t.synapses() {
            seen_s.insert(s.id);
        }
    }
    assert!(
        seen_n.len() > 20 && seen_s.len() > 100,
        "the operations were actually carried out"
    );
}

#[test]
fn mutate_neuron_clamps_and_supports_inputs() {
    // Loose weights: This test tests legacy mode without the Dale principle.
    let mut t = topo(TopologyLimits {
        enforce_dale: false,
        ..TopologyLimits::default()
    });
    let i = inp(&mut t);
    let h = hid(&mut t, 0.9, 1.0);
    let r = t
        .mutate_neuron(
            h,
            NeuronMutation {
                beta: Some(5.0),
                threshold: Some(0.001),
                ..Default::default()
            },
        )
        .unwrap();
    assert!(r.clamped);
    assert_eq!(
        r.new.beta().unwrap(),
        ParamLimits::<f32>::default().beta_max
    );
    assert_eq!(
        r.new.threshold(),
        ParamLimits::<f32>::default().threshold_min
    );
    // The input neuron is an ordinary dynamic neuron: its parameters are also
    // limited rather than rejected.
    let ri = t
        .mutate_neuron(
            i,
            NeuronMutation {
                beta: Some(0.5),
                threshold: Some(2.0),
                ..Default::default()
            },
        )
        .unwrap();
    assert!(!ri.clamped);
    assert_eq!(ri.new.beta(), Some(0.5));
    assert_eq!(ri.new.threshold(), 2.0);
    assert!(matches!(
        t.mutate_neuron(
            h,
            NeuronMutation {
                beta: Some(f32::NAN),
                ..Default::default()
            }
        ),
        Err(StructuralError::InvalidParameter { .. })
    ));
    // synapse mutation
    let o = out(&mut t, 0.9, 1.0);
    let s = syn(&mut t, h, o, 0.5);
    t.mutate_synapse(
        s,
        SynapseMutation {
            weight: Some(-0.25),
            delay: Some(3),
        },
    )
    .unwrap();
    assert_eq!(
        (t.synapse(s).unwrap().weight, t.synapse(s).unwrap().delay),
        (-0.25, 3)
    );
    assert!(
        t.mutate_synapse(
            s,
            SynapseMutation {
                delay: Some(200),
                ..Default::default()
            }
        )
        .is_err()
    );
}

#[test]
fn state_migration_policies_convert_explicitly() {
    let lif = NeuronModel::Lif(LifParams::default());
    let alif = NeuronModel::AdaptiveLif(AdaptiveLifParams::slow_adaptation());
    let izh = NeuronModel::Izhikevich(IzhikevichParams::fast_spiking());
    let source = NeuronState::Lif(LifState { mem: 0.42 });

    let reset = source.clone().migrate_to(&izh, StateMigration::Reset);
    assert!(matches!(
        reset,
        NeuronState::Izhikevich(s)
            if s.v == IzhikevichParams::<f32>::fast_spiking().c && s.u == 0.0
    ));
    let membrane = source.clone().migrate_to(&izh, StateMigration::Membrane);
    assert!(matches!(
        membrane,
        NeuronState::Izhikevich(s) if s.v == 0.42 && s.u == 0.0
    ));
    // Different families: Compatible drops rather than drags dimensionless
    // membrane LIF to millivolts Izhikevich.
    assert_eq!(
        source.clone().migrate_to(&izh, StateMigration::Compatible),
        izh.init_state()
    );
    // One family: membrane is transferred, adaptation is initialized.
    let same = source.clone().migrate_to(&alif, StateMigration::Compatible);
    assert!(matches!(
        same,
        NeuronState::AdaptiveLif(s) if s.mem == 0.42 && s.adaptation == 0.0
    ));
    // The same model: Compatible saves everything, Reset resets everything.
    assert_eq!(
        source.clone().migrate_to(&lif, StateMigration::Compatible),
        source
    );
    assert_eq!(
        source.migrate_to(&lif, StateMigration::Reset),
        NeuronState::Lif(LifState::default())
    );
    // Izhikevich → AdEx: both in mV, Compatible retains membrane.
    let izh_state = NeuronState::Izhikevich(IzhikevichState { v: -55.0, u: 2.0 });
    let adex = NeuronModel::AdEx(AdExParams::default());
    assert_eq!(
        izh_state
            .migrate_to(&adex, StateMigration::Compatible)
            .mem(),
        -55.0
    );
}

#[test]
fn change_neuron_model_migrates_state_and_is_an_action() {
    let mut t = topo(TopologyLimits::default());
    let i = inp(&mut t);
    let h = hid(&mut t, 0.9, 1.0);
    let o = out(&mut t, 0.9, 1.0);
    syn(&mut t, i, h, 1.0);
    syn(&mut t, h, o, 1.0);
    t.neurons.get_mut(&h).unwrap().state = NeuronState::Lif(LifState { mem: 0.5 });

    let switch = t
        .change_neuron_model(
            h,
            AdaptiveLifParams::slow_adaptation(),
            StateMigration::Compatible,
        )
        .unwrap();
    assert!(switch.changed);
    assert_eq!(switch.old_state, NeuronState::Lif(LifState { mem: 0.5 }));
    assert!(matches!(
        switch.new_state,
        NeuronState::AdaptiveLif(s) if s.mem == 0.5 && s.adaptation == 0.0
    ));
    t.validate().unwrap();

    // Repeated change to the same model is no-op, the state does not migrate.
    let again = t
        .change_neuron_model(
            h,
            AdaptiveLifParams::slow_adaptation(),
            StateMigration::Reset,
        )
        .unwrap();
    assert!(!again.changed);
    assert_eq!(again.new_state, again.old_state);

    // The input neuron can also be switched: the role does not limit the model.
    let input_switch = t
        .change_neuron_model(
            i,
            IzhikevichParams::regular_spiking(),
            StateMigration::Reset,
        )
        .unwrap();
    assert!(input_switch.changed);
    assert!(matches!(
        t.neuron(i).unwrap().model,
        NeuronModel::Izhikevich(_)
    ));

    let mut n = net(t);
    let mut rng = Rng64::new(1);
    apply_action(
        &mut n,
        &StructuralAction::ChangeNeuronModel {
            id: h,
            model: NeuronModel::Izhikevich(IzhikevichParams::fast_spiking()),
            migration: StateMigration::Membrane,
        },
        &mut rng,
    )
    .unwrap();
    let nn = n.topology().neuron(h).unwrap();
    assert!(matches!(nn.model, NeuronModel::Izhikevich(_)));
    assert_eq!(nn.state.mem(), 0.5);
    n.topology().validate().unwrap();
}

#[test]
fn model_selection_proposes_specialization_and_controller_applies_it() {
    let (mut t, _, h, _) = chain();
    {
        let n = t.neurons.get_mut(&h).unwrap();
        n.stats.steps = 1000;
        // A frequency of 0.1 suits Izhikevich RS equally well (target 0.08) and
        // HH (target 0.1), so the choice should be decided by the cheaper Izhikevich.
        n.stats.rate_ema = 0.1;
    }
    let mut n = net(t);
    let selection = ModelSelectionConfig {
        enabled: true,
        min_observed_steps: 100,
        max_switches_per_interval: 1,
        min_improvement: 0.25,
        ..Default::default()
    };
    let proposals = propose_model_switches(&n, &selection);
    assert_eq!(proposals.len(), 1);
    let StructuralAction::ChangeNeuronModel {
        id,
        model,
        migration,
    } = &proposals[0].0
    else {
        panic!("model switch expected");
    };
    assert_eq!(*id, h);
    assert!(matches!(model, NeuronModel::Izhikevich(_)));
    assert_eq!(*migration, StateMigration::Compatible);

    let mut cfg = quick_cfg();
    cfg.growth.enabled = false;
    cfg.model_selection = selection;
    let mut ctl = StructuralPlasticityController::new(cfg).unwrap();
    let report = ctl.adapt(&mut n, Metrics::loss(1.0), None).unwrap();
    assert_eq!(report.applied, 1);
    assert_eq!(report.events[0].kind, ActionKind::ChangeNeuronModel);
    assert!(report.events[0].reason.contains("model selection"));
    assert!(matches!(
        n.topology().neuron(h).unwrap().model,
        NeuronModel::Izhikevich(_)
    ));
}

#[test]
fn current_input_mode_makes_input_neurons_dynamic() {
    let mut t = topo(TopologyLimits::default());
    let i = inp(&mut t);
    let o = out(&mut t, 0.9, 1.0);
    syn(&mut t, i, o, 1.2);
    let mut cfg = EngineConfig::default();
    cfg.input_mode = InputMode::Current;
    let mut n = DynamicNetwork::new(t, cfg).unwrap();

    // 0.6 < threshold 1.0: the first step accumulates the membrane, there is no spike.
    let r1 = n.step(&[0.6]).unwrap();
    assert_eq!(r1.total_spikes, 0);
    assert!(matches!(
        n.topology().neuron(i).unwrap().state,
        NeuronState::Lif(s) if (s.mem - 0.6).abs() < 1e-6
    ));
    // 0.9·0.6 + 0.6 = 1.14 ≥ 1.0: the input spikes as in a usual LIF.
    let r2 = n.step(&[0.6]).unwrap();
    assert_eq!(r2.total_spikes, 1);
    assert_eq!(r2.output_spikes, vec![false]);
    // The input spike is delivered to the output in the next step.
    let r3 = n.step(&[0.0]).unwrap();
    assert_eq!(r3.output_spikes, vec![true]);
}

#[test]
fn normalize_connectivity_drops_least_useful_first() {
    let mut t = topo(TopologyLimits::default());
    let o = out(&mut t, 0.9, 1.0);
    let srcs: Vec<NeuronId> = (0..5).map(|_| inp(&mut t)).collect();
    let ids: Vec<SynapseId> = srcs
        .iter()
        .enumerate()
        .map(|(k, s)| syn(&mut t, *s, o, 0.1 * (k + 1) as f32))
        .collect();
    t.disable_synapse(ids[4], 0).unwrap(); // the most “valuable” by weight, but disabled
    let removed = t
        .normalize_connectivity(
            &DegreePolicy {
                max_in: 3,
                max_out: 10,
                max_density: None,
            },
            &|s| s.weight.abs(),
        )
        .unwrap();
    assert_eq!(removed.len(), 2);
    assert!(removed.contains(&ids[4]), "disabled ones are deleted first");
    assert!(removed.contains(&ids[0]));
    assert_eq!(t.in_synapses(o).len(), 3);
    t.validate().unwrap();
    // density
    let removed = t
        .normalize_connectivity(
            &DegreePolicy {
                max_in: 10,
                max_out: 10,
                max_density: Some(0.0),
            },
            &|_| 0.0,
        )
        .unwrap();
    assert_eq!(removed.len(), 3);
    assert_eq!(t.synapse_count(), 0);
}

// clone and split preserve behavior ─────────────────────

fn chain() -> (Topology, NeuronId, NeuronId, NeuronId) {
    let mut t = topo(TopologyLimits::default());
    let i = inp(&mut t);
    let h = hid(&mut t, 0.0, 1.0);
    let o = out(&mut t, 0.9, 1.0);
    syn(&mut t, i, h, 1.5);
    syn(&mut t, h, o, 1.2);
    (t, i, h, o)
}

const PATTERN: [f32; 12] = [1.0, 0.0, 1.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0, 0.0, 0.0, 0.0];

#[test]
fn clone_neuron_preserves_network_behaviour() {
    let (t, _, h, _) = chain();
    let mut original = net(t.clone());
    let mut cloned_topo = t;
    let mut rng = Rng64::new(3);
    let c = cloned_topo.clone_neuron(h, 0.0, &mut rng, b()).unwrap();
    assert_eq!(cloned_topo.neuron(c).unwrap().meta.origin, Origin::Clone(h));
    assert_eq!(cloned_topo.neuron_count(), 4);
    assert_eq!(cloned_topo.synapse_count(), 4);
    cloned_topo.validate().unwrap();
    let mut cloned = net(cloned_topo);
    assert_eq!(run(&mut original, &PATTERN), run(&mut cloned, &PATTERN));
}

#[test]
fn clone_with_jitter_diverges_only_in_incoming_weights() {
    let (mut t, _, h, _) = chain();
    let mut rng = Rng64::new(9);
    let c = t.clone_neuron(h, 0.2, &mut rng, b()).unwrap();
    let w_orig = t.synapse(t.in_synapses(h)[0]).unwrap().weight;
    let w_clone = t.synapse(t.in_synapses(c)[0]).unwrap().weight;
    assert!(w_orig != w_clone && (w_clone / w_orig - 1.0).abs() <= 0.2 + 1e-6);
}

#[test]
fn split_synapse_relays_spikes_with_one_step_delay() {
    let mut t = topo(TopologyLimits::default());
    let i = inp(&mut t);
    let o = out(&mut t, 0.9, 1.0);
    let s = syn(&mut t, i, o, 1.2);
    let mut original = net(t.clone());
    let r = t.split_synapse(s, SplitInit::default(), b()).unwrap();
    assert!(t.synapse(s).is_none());
    assert_eq!(t.neuron(r.neuron).unwrap().meta.origin, Origin::Split(s));
    assert_eq!(t.synapse(r.outgoing).unwrap().weight, 1.2);
    t.validate().unwrap();
    let mut split = net(t);
    let a = run(&mut original, &PATTERN);
    let c = run(&mut split, &PATTERN);
    assert!(a.iter().any(|x| x[0]));
    assert_eq!(
        &a[..a.len() - 1],
        &c[1..],
        "the spikes are the same, shifted by 1 step"
    );
}

#[test]
fn split_keeps_inhibitory_sign_and_rejects_disabled() {
    // Legacy mode: negative weight of the excitatory source is acceptable.
    let mut t = topo(TopologyLimits {
        enforce_dale: false,
        ..TopologyLimits::default()
    });
    let i = inp(&mut t);
    let o = out(&mut t, 0.9, 1.0);
    let s = syn(&mut t, i, o, -0.8);
    let r = t.split_synapse(s, SplitInit::default(), b()).unwrap();
    assert_eq!(t.synapse(r.outgoing).unwrap().weight, -0.8);
    t.disable_synapse(r.incoming, 0).unwrap();
    assert!(
        t.split_synapse(r.incoming, SplitInit::default(), b())
            .is_err()
    );
    assert!(
        t.split_synapse(
            r.outgoing,
            SplitInit {
                relay_gain: 0.1,
                ..Default::default()
            },
            b()
        )
        .is_err()
    );
}

// engine ─────────────────────────

#[test]
fn synaptic_delay_is_respected() {
    let mut t = topo(TopologyLimits::default());
    let i = inp(&mut t);
    let o = out(&mut t, 0.9, 1.0);
    t.add_synapse(i, o, 2.0, 2, Origin::Initial, b()).unwrap();
    let mut n = net(t);
    let spikes = run(&mut n, &[1.0, 0.0, 0.0, 0.0, 0.0, 0.0]);
    let at: Vec<usize> = spikes
        .iter()
        .enumerate()
        .filter(|(_, s)| s[0])
        .map(|(k, _)| k)
        .collect();
    assert_eq!(at, vec![3]);
}

#[test]
fn disabled_synapse_does_not_transmit() {
    let mut t = topo(TopologyLimits::default());
    let i = inp(&mut t);
    let o = out(&mut t, 0.9, 1.0);
    let s = syn(&mut t, i, o, 2.0);
    t.disable_synapse(s, 0).unwrap();
    let mut n = net(t);
    assert!(run(&mut n, &PATTERN).iter().all(|x| !x[0]));
}

#[test]
fn wrong_input_size_is_a_typed_error() {
    let (t, ..) = chain();
    let mut n = net(t);
    assert!(matches!(
        n.step(&[0.0, 0.0]),
        Err(StructuralError::InputSize {
            expected: 1,
            actual: 2,
            ..
        })
    ));
}

#[test]
fn network_survives_structure_change_between_steps() {
    let (t, _, h, _) = chain();
    let mut n = net(t);
    run(&mut n, &PATTERN[..4]);
    let mut rng = Rng64::new(1);
    apply_action(
        &mut n,
        &StructuralAction::CloneNeuron { id: h, jitter: 0.0 },
        &mut rng,
    )
    .unwrap();
    run(&mut n, &PATTERN[4..8]);
    apply_action(&mut n, &StructuralAction::RemoveNeuron { id: h }, &mut rng).unwrap();
    run(&mut n, &PATTERN[8..]); // the story refers to deleted neurons - it's safe
    n.topology().validate().unwrap();
}

fn pair_net(mode: PlasticityMode, w: f32) -> (DynamicNetwork, NeuronId, NeuronId, SynapseId) {
    let mut t = topo(TopologyLimits::default());
    let i = inp(&mut t);
    let o = out(&mut t, 0.9, 1.0);
    let s = syn(&mut t, i, o, w);
    (stdp_net(t, mode), i, o, s)
}

#[test]
fn stdp_potentiation_matches_the_pair_formula() {
    // the post spiked at step 0 (trace 1), at step 2 the spike pre arrives; post trace before update = 0.9: Δw = a+ · 0.9
    let (mut n, _, o, s) = pair_net(PlasticityMode::Immediate, 0.1);
    n.step_with_current(&[0.0], &[(o, 2.0)]).unwrap();
    n.step(&[1.0]).unwrap();
    n.step(&[0.0]).unwrap();
    let w = n.topology().synapse(s).unwrap().weight;
    assert!((w - (0.1 + 0.1 * 0.9)).abs() < 1e-6, "w = {w}");
}

#[test]
fn stdp_depression_matches_the_pair_formula() {
    // pre spiked at step 0 (its spike will arrive at step 1 with a weak current), post - at step 2:
    // trace before updating at step 2 = 0.9: Δw = −a− · 0.9
    let (mut n, _, o, s) = pair_net(PlasticityMode::Immediate, 0.5);
    n.step(&[1.0]).unwrap();
    n.step(&[0.0]).unwrap();
    n.step_with_current(&[0.0], &[(o, 2.0)]).unwrap();
    let w = n.topology().synapse(s).unwrap().weight;
    assert!((w - (0.5 - 0.1 * 0.9)).abs() < 1e-6, "w = {w}");
}

#[test]
fn stdp_respects_weight_bounds_and_plasticity_switch() {
    let (mut n, _, o, s) = pair_net(PlasticityMode::Immediate, 0.99);
    n.set_plasticity_enabled(false);
    n.step_with_current(&[0.0], &[(o, 2.0)]).unwrap();
    n.step(&[1.0]).unwrap();
    n.step(&[0.0]).unwrap();
    assert_eq!(n.topology().synapse(s).unwrap().weight, 0.99);
    n.set_plasticity_enabled(true);
    n.reset_state();
    for _ in 0..30 {
        n.step_with_current(&[1.0], &[(o, 0.3)]).unwrap();
        n.step(&[0.0]).unwrap();
        let w = n.topology().synapse(s).unwrap().weight;
        assert!((-1.0..=1.0).contains(&w));
    }
}

#[test]
fn reward_modulated_updates_only_on_reward_with_its_sign() {
    for (reward, positive) in [(1.0f32, true), (-1.0, false)] {
        let (mut n, _, o, s) = pair_net(PlasticityMode::RewardModulated, 0.1);
        n.step_with_current(&[0.0], &[(o, 2.0)]).unwrap();
        n.step(&[1.0]).unwrap();
        n.step(&[0.0]).unwrap();
        assert_eq!(
            n.topology().synapse(s).unwrap().weight,
            0.1,
            "the weight does not change until the reward"
        );
        assert!(n.topology().synapse(s).unwrap().elig > 0.0);
        n.apply_reward(reward);
        let w = n.topology().synapse(s).unwrap().weight;
        assert_eq!(w > 0.1, positive, "w = {w}");
    }
}

#[test]
fn credit_goes_to_recently_active_synapses_only() {
    let mut t = topo(TopologyLimits::default());
    let a = inp(&mut t);
    let c = inp(&mut t);
    let o = out(&mut t, 0.9, 1.0);
    let sa = syn(&mut t, a, o, 0.2);
    let sc = syn(&mut t, c, o, 0.2);
    let mut n = net(t);
    n.step(&[1.0, 0.0]).unwrap();
    n.step(&[0.0, 0.0]).unwrap(); // spike a delivered by sa
    n.assign_credit(1.0);
    assert!(n.topology().synapse(sa).unwrap().stats.credit_ema > 0.0);
    assert_eq!(n.topology().synapse(sc).unwrap().stats.credit_ema, 0.0);
}

// ───────────────────────── pruning ─────────────────────────

fn prune_cfg() -> PruningConfig {
    PruningConfig {
        min_age_steps: 20,
        min_observed_samples: 5,
        reeval_interval_steps: 0,
        grace_steps: 30,
        drive_rel_threshold: 0.05,
        ..PruningConfig::default()
    }
}

/// `a` spikes in every sample, `rare` in rare ones, `silent` never.
fn prune_world() -> (DynamicNetwork, [SynapseId; 4]) {
    let mut t = topo(TopologyLimits::default());
    let (a, rare, silent, tiny) = (inp(&mut t), inp(&mut t), inp(&mut t), inp(&mut t));
    let o = out(&mut t, 0.9, 5.0);
    let ids = [
        syn(&mut t, a, o, 0.3),
        syn(&mut t, rare, o, 0.3),
        syn(&mut t, silent, o, 0.3),
        syn(&mut t, tiny, o, 0.001),
    ];
    let mut n = net(t);
    for k in 0..60 {
        n.begin_sample();
        let r = if k % 25 == 0 { 1.0 } else { 0.0 };
        for _ in 0..3 {
            n.step(&[1.0, r, 0.0, 1.0]).unwrap();
            n.step(&[0.0, 0.0, 0.0, 0.0]).unwrap();
        }
    }
    (n, ids)
}

#[test]
fn unused_synapse_pruned_but_used_tiny_weight_synapse_is_not() {
    let (mut n, [a, rare, silent, tiny]) = prune_world();
    let disabled = prune_synapses(&mut n, &prune_cfg(), 8).unwrap();
    assert_eq!(
        disabled,
        vec![silent],
        "only a synapse that did not transmit a single spike"
    );
    for id in [a, rare, tiny] {
        assert!(
            n.topology().synapse(id).unwrap().is_active(),
            "{id} must stay"
        );
    }
}

#[test]
fn young_or_unobserved_synapses_are_never_evaluated() {
    let (n, _) = prune_world();
    let mut young = prune_cfg();
    young.min_age_steps = 1_000_000;
    assert!(
        synapse_verdicts(&n, &young)
            .iter()
            .all(|v| !v.evaluable && !v.candidate)
    );
    let mut few = prune_cfg();
    few.min_observed_samples = 10_000;
    assert!(synapse_verdicts(&n, &few).iter().all(|v| !v.candidate));
}

#[test]
fn two_stage_pruning_disables_then_removes_after_grace() {
    let (mut n, [_, _, silent, _]) = prune_world();
    let cfg = prune_cfg();
    prune_synapses(&mut n, &cfg, 8).unwrap();
    assert!(!n.topology().synapse(silent).unwrap().is_active());
    // until the end of the wait, the synapse physically remains
    prune_synapses(&mut n, &cfg, 8).unwrap();
    assert!(n.topology().synapse(silent).is_some());
    for _ in 0..cfg.grace_steps {
        n.step(&[0.0; 4]).unwrap();
    }
    let removed = prune_synapses(&mut n, &cfg, 8).unwrap();
    assert!(removed.contains(&silent));
    assert!(n.topology().synapse(silent).is_none());
    n.topology().validate().unwrap();
}

#[test]
fn rare_but_useful_synapse_is_protected_by_credit() {
    let (mut n, [_, rare, ..]) = prune_world();
    let mut cfg = prune_cfg();
    cfg.drive_rel_threshold = 5.0; // “everything below 5 medians” is a candidate; protection must work
    let before = synapse_verdicts(&n, &cfg);
    assert!(before.iter().find(|v| v.id == rare).unwrap().candidate);
    // reward immediately after a rare spike → positive contribution
    n.begin_sample();
    n.step(&[0.0, 1.0, 0.0, 0.0]).unwrap();
    n.step(&[0.0; 4]).unwrap();
    n.assign_credit(1.0);
    let after = synapse_verdicts(&n, &cfg);
    let v = after.iter().find(|v| v.id == rare).unwrap();
    assert!(v.protected && !v.candidate);
    assert!(
        !propose_prune_synapses(&n, &cfg, &after, 8)
            .iter()
            .any(|(a, _)| *a == StructuralAction::DisableSynapse { id: rare })
    );
}

#[test]
fn pruning_respects_minimum_connectivity() {
    let mut t = topo(TopologyLimits::default());
    let i = inp(&mut t);
    let o = out(&mut t, 0.9, 5.0);
    let s = syn(&mut t, i, o, 0.3);
    let mut n = net(t);
    for _ in 0..60 {
        n.begin_sample();
        n.step(&[0.0]).unwrap();
    }
    let cfg = PruningConfig {
        min_age_steps: 0,
        min_observed_samples: 1,
        reeval_interval_steps: 0,
        ..PruningConfig::default()
    };
    assert!(synapse_verdicts(&n, &cfg)[0].candidate);
    assert!(
        prune_synapses(&mut n, &cfg, 8).unwrap().is_empty(),
        "the last input of the output neuron is protected"
    );
    assert!(n.topology().synapse(s).unwrap().is_active());
    let mut cfg2 = cfg.clone();
    cfg2.min_output_in_degree = 0;
    cfg2.min_synapses = 1;
    assert!(
        prune_synapses(&mut n, &cfg2, 8).unwrap().is_empty(),
        "min_synapses"
    );
}

#[test]
fn dead_and_silent_neurons_removed_respecting_min_hidden() {
    let mut t = topo(TopologyLimits::default());
    let i = inp(&mut t);
    let o = out(&mut t, 0.9, 1.0);
    let live = hid(&mut t, 0.0, 0.5);
    let dead_end = hid(&mut t, 0.9, 1.0); // there is an input, there is no output
    let isolated = hid(&mut t, 0.9, 1.0);
    syn(&mut t, i, live, 1.0);
    syn(&mut t, live, o, 1.0);
    syn(&mut t, i, dead_end, 1.0);
    let mut n = net(t);
    for _ in 0..40 {
        n.begin_sample();
        n.step(&[1.0]).unwrap();
        n.step(&[0.0]).unwrap();
    }
    let cfg = PruningConfig {
        min_age_steps: 0,
        min_observed_samples: 1,
        max_neuron_prunes: 5,
        ..PruningConfig::default()
    };
    let plan = propose_prune_neurons(&n, &cfg);
    let ids: BTreeSet<NeuronId> = plan
        .iter()
        .filter_map(|(a, _)| {
            if let StructuralAction::RemoveNeuron { id } = a {
                Some(*id)
            } else {
                None
            }
        })
        .collect();
    assert_eq!(ids, BTreeSet::from([dead_end, isolated]));
    let mut keep_one = cfg.clone();
    keep_one.min_hidden = 2;
    assert_eq!(
        propose_prune_neurons(&n, &keep_one).len(),
        1,
        "minimum hidden neurons"
    );
    let removed = prune_neurons(&mut n, &cfg).unwrap();
    assert_eq!(removed.len(), 2);
    assert!(n.topology().neuron(live).is_some());
    n.topology().validate().unwrap();
}

// ───────────────────────── growth ─────────────────────────

fn growth_net() -> (DynamicNetwork, NeuronId, NeuronId) {
    let mut t = topo(TopologyLimits::default());
    let i = inp(&mut t);
    let o = out(&mut t, 0.9, 1.0);
    let mut n = net(t);
    n.enable_coactivity(4, 128, 32);
    (n, i, o)
}

#[test]
fn coactive_pair_grows_a_synapse_and_never_a_duplicate() {
    let (mut n, i, o) = growth_net();
    for _ in 0..20 {
        n.step_with_current(&[1.0], &[]).unwrap();
        n.step_with_current(&[0.0], &[(o, 2.0)]).unwrap(); // the output spikes after the input
    }
    let cfg = GrowthConfig::default();
    let mut rng = Rng64::new(5);
    let plan = propose_grow_synapses(&n, &cfg, &mut rng, 4);
    assert_eq!(plan.len(), 1);
    assert!(
        matches!(plan[0].0, StructuralAction::AddSynapse { src, dst, .. } if src == i && dst == o)
    );
    assert!(matches!(plan[0].1, Reason::CoActivity { .. }));
    let ids = grow_synapses(&mut n, &cfg, &mut rng, 4).unwrap();
    assert_eq!(ids.len(), 1);
    assert!(n.topology().connected(i, o));
    assert!(
        propose_grow_synapses(&n, &cfg, &mut rng, 4).is_empty(),
        "the couple is already connected"
    );
}

#[test]
fn synapse_growth_respects_limits_and_threshold() {
    let (mut n, _, o) = growth_net();
    for _ in 0..20 {
        n.step(&[1.0]).unwrap();
        n.step_with_current(&[0.0], &[(o, 2.0)]).unwrap();
    }
    let mut cfg = GrowthConfig::default();
    cfg.synapses.min_score = 1e9;
    let mut rng = Rng64::new(5);
    assert!(propose_grow_synapses(&n, &cfg, &mut rng, 4).is_empty());
    cfg.synapses.min_score = 1.0;
    cfg.enabled = false;
    assert!(propose_grow_synapses(&n, &cfg, &mut rng, 4).is_empty());
    cfg.enabled = true;
    let mut lim = *n.topology().limits();
    lim.max_synapses = 0;
    n.topology_mut().limits = lim;
    assert!(
        propose_grow_synapses(&n, &cfg, &mut rng, 4).is_empty(),
        "synapse limit"
    );
}

#[test]
fn neuron_growth_strategies_produce_valid_actions() {
    let (t, _, h, _) = chain();
    let mut n = net(t);
    for _ in 0..10 {
        run(&mut n, &PATTERN);
    }
    let mut rng = Rng64::new(2);
    for strategy in [
        GrowthStrategy::SplitSynapse,
        GrowthStrategy::CloneNeuron,
        GrowthStrategy::NewNeuron,
    ] {
        let mut cfg = NeuronGrowthConfig::default();
        cfg.strategies = vec![strategy];
        let mut copy = n.clone();
        let created = grow_neurons(&mut copy, &cfg, &mut rng, 1).unwrap();
        assert_eq!(created.len(), 1, "{strategy:?}");
        copy.topology().validate().unwrap();
        assert_eq!(
            copy.topology().neuron_count(),
            n.topology().neuron_count() + 1
        );
    }
    let _ = h;
    // pause operation type eliminates strategy
    let mut cfg = NeuronGrowthConfig::default();
    cfg.strategies = vec![GrowthStrategy::CloneNeuron];
    let skip = BTreeSet::from([ActionKind::CloneNeuron]);
    assert!(propose_grow_neuron(&n, &cfg, &mut rng, &skip, 3).is_none());
}

#[test]
fn neuron_limit_stops_growth() {
    let (t, ..) = chain();
    let mut n = net(t);
    for _ in 0..5 {
        run(&mut n, &PATTERN);
    }
    n.topology_mut().limits.max_neurons = 3;
    let mut rng = Rng64::new(2);
    assert!(
        grow_neurons(&mut n, &NeuronGrowthConfig::default(), &mut rng, 5)
            .unwrap()
            .is_empty()
    );
}

// Controller ─────────────────────────

fn quick_cfg() -> StructuralConfig {
    let mut c = StructuralConfig::default();
    c.schedule.min_interval_steps = 0;
    c.pruning.enabled = false;
    c.homeostasis.enabled = false;
    c.growth.synapses.max_new_per_interval = 0;
    c.growth.neurons.plateau_patience = 1;
    c.growth.neurons.strategies = vec![GrowthStrategy::SplitSynapse];
    c.rollback.cooldown_steps = 100;
    c
}

fn warmed(t: Topology) -> DynamicNetwork {
    let mut n = net(t);
    for _ in 0..5 {
        run(&mut n, &PATTERN);
    }
    n
}

#[test]
fn plateau_triggers_neuron_growth_with_logged_reason() {
    let (t, ..) = chain();
    let mut n = warmed(t);
    let mut ctl = StructuralPlasticityController::new(quick_cfg()).unwrap();
    let r0 = ctl.adapt(&mut n, Metrics::loss(1.0), None).unwrap();
    assert_eq!(r0.applied, 0, "the first rating is not a plateau");
    let r1 = ctl.adapt(&mut n, Metrics::loss(1.0), None).unwrap();
    assert_eq!(r1.applied, 1);
    assert_eq!(n.topology().neuron_count(), 4);
    let e = &r1.events[0];
    assert_eq!(e.kind, ActionKind::SplitSynapse);
    assert!(e.reason.contains("plateau"));
    assert_eq!((e.neurons_before, e.neurons_after), (3, 4));
    assert_eq!(ctl.log().len(), 1);
    assert_eq!(ctl.log().totals()[&ActionKind::SplitSynapse].applied, 1);
}

#[test]
fn improving_loss_does_not_grow() {
    let (t, ..) = chain();
    let mut n = warmed(t);
    let mut ctl = StructuralPlasticityController::new(quick_cfg()).unwrap();
    for k in 0..10 {
        let r = ctl
            .adapt(&mut n, Metrics::loss(1.0 - 0.05 * k as f32), None)
            .unwrap();
        assert_eq!(r.applied, 0);
    }
    assert_eq!(n.topology().neuron_count(), 3);
}

#[test]
fn growth_is_bounded_by_limits_even_under_permanent_plateau() {
    let (t, ..) = chain();
    let mut n = warmed(t);
    let mut cfg = quick_cfg();
    cfg.topology.max_neurons = 8;
    cfg.topology.max_synapses = 30;
    cfg.rollback.enabled = false;
    n.topology_mut().limits = cfg.topology;
    let mut ctl = StructuralPlasticityController::new(cfg).unwrap();
    for _ in 0..300 {
        ctl.adapt(&mut n, Metrics::loss(1.0), None).unwrap();
        run(&mut n, &PATTERN[..2]);
    }
    assert!(n.topology().neuron_count() <= 8);
    assert!(n.topology().synapse_count() <= 30);
    assert_eq!(
        n.topology().neuron_count(),
        8,
        "growth reaches a limit and stops"
    );
    n.topology().validate().unwrap();
}

#[test]
fn memory_budget_blocks_growth() {
    let (t, ..) = chain();
    let mut n = warmed(t);
    let mut cfg = quick_cfg();
    cfg.max_state_bytes = Some(1);
    let mut ctl = StructuralPlasticityController::new(cfg).unwrap();
    for _ in 0..5 {
        assert_eq!(
            ctl.adapt(&mut n, Metrics::loss(1.0), None).unwrap().applied,
            0
        );
    }
    assert_eq!(n.topology().neuron_count(), 3);
}

#[test]
fn min_interval_and_analysis_cadence_are_respected() {
    let (t, ..) = chain();
    let mut n = warmed(t);
    let mut cfg = quick_cfg();
    cfg.schedule.min_interval_steps = 1000;
    cfg.rollback.enabled = false; // without a deferred check, the reason for the pause is precisely the interval
    let mut ctl = StructuralPlasticityController::new(cfg).unwrap();
    ctl.adapt(&mut n, Metrics::loss(1.0), None).unwrap();
    assert_eq!(
        ctl.adapt(&mut n, Metrics::loss(1.0), None).unwrap().applied,
        1
    );
    let r = ctl.adapt(&mut n, Metrics::loss(1.0), None).unwrap();
    assert_eq!((r.applied, r.skipped), (0, Some(SkipReason::TooSoon)));
    let mut cfg = quick_cfg();
    cfg.schedule.analyze_every = 3;
    let mut ctl = StructuralPlasticityController::new(cfg).unwrap();
    let r = ctl.adapt(&mut n, Metrics::loss(1.0), None).unwrap();
    assert_eq!(r.skipped, Some(SkipReason::NotDue));
    assert_eq!(
        ctl.adapt(&mut n, Metrics::loss(f32::NAN), None)
            .unwrap()
            .skipped,
        Some(SkipReason::InvalidMetrics)
    );
}

#[test]
fn evaluator_rolls_back_a_harmful_change_and_pauses_that_operation() {
    let (t, ..) = chain();
    let mut n = warmed(t);
    let mut ctl = StructuralPlasticityController::new(quick_cfg()).unwrap();
    ctl.adapt(&mut n, Metrics::loss(1.0), None).unwrap();
    let snapshot = n.topology().clone();
    // the score “gets worse” as the number of neurons increases
    let mut eval = |p: &mut DynamicNetwork| 1.0 + p.topology().neuron_count() as f32;
    let r = ctl
        .adapt(&mut n, Metrics::loss(1.0), Some(&mut eval))
        .unwrap();
    assert!(r.rolled_back);
    assert_eq!(r.applied, 0);
    assert!(r.loss_after.unwrap() > r.loss_before.unwrap());
    assert_eq!(
        *n.topology(),
        snapshot,
        "the topology has been completely restored"
    );
    assert_eq!(r.events[0].outcome, Outcome::RolledBack);
    assert_eq!(ctl.log().totals()[&ActionKind::SplitSynapse].rolled_back, 1);
    // the same operation on pause
    let again = ctl
        .adapt(&mut n, Metrics::loss(1.0), Some(&mut eval))
        .unwrap();
    assert_eq!(again.skipped, Some(SkipReason::NothingToDo));
}

#[test]
fn evaluator_keeps_a_helpful_change() {
    let (t, ..) = chain();
    let mut n = warmed(t);
    let mut ctl = StructuralPlasticityController::new(quick_cfg()).unwrap();
    ctl.adapt(&mut n, Metrics::loss(1.0), None).unwrap();
    let mut eval = |p: &mut DynamicNetwork| 1.0 - 0.1 * p.topology().neuron_count() as f32;
    let r = ctl
        .adapt(&mut n, Metrics::loss(1.0), Some(&mut eval))
        .unwrap();
    assert!(!r.rolled_back);
    assert_eq!(r.applied, 1);
    assert_eq!(n.topology().neuron_count(), 4);
}

#[test]
fn evaluator_does_not_disturb_the_live_network() {
    let (t, ..) = chain();
    let mut n = warmed(t);
    let mut ctl = StructuralPlasticityController::new(quick_cfg()).unwrap();
    ctl.adapt(&mut n, Metrics::loss(1.0), None).unwrap();
    let steps = n.step_count();
    let mut eval = |p: &mut DynamicNetwork| {
        for _ in 0..50 {
            p.step(&[1.0]).unwrap();
        }
        0.5
    };
    ctl.adapt(&mut n, Metrics::loss(1.0), Some(&mut eval))
        .unwrap();
    assert_eq!(n.step_count(), steps);
}

#[test]
fn deferred_check_rolls_back_when_training_metrics_get_worse() {
    let (t, ..) = chain();
    let mut n = warmed(t);
    let mut cfg = quick_cfg();
    cfg.rollback.verify_after_calls = 2;
    let mut ctl = StructuralPlasticityController::new(cfg).unwrap();
    ctl.adapt(&mut n, Metrics::loss(1.0), None).unwrap();
    let snapshot = n.topology().clone();
    assert_eq!(
        ctl.adapt(&mut n, Metrics::loss(1.0), None).unwrap().applied,
        1
    );
    assert_eq!(n.topology().neuron_count(), 4);
    ctl.adapt(&mut n, Metrics::loss(3.0), None).unwrap();
    let r = ctl.adapt(&mut n, Metrics::loss(3.0), None).unwrap();
    assert!(r.rolled_back);
    assert_eq!(*n.topology(), snapshot);
    let totals = ctl.log().totals()[&ActionKind::SplitSynapse];
    assert_eq!((totals.applied, totals.rolled_back), (0, 1));
}

#[test]
fn deferred_check_keeps_change_when_metrics_do_not_degrade() {
    let (t, ..) = chain();
    let mut n = warmed(t);
    let mut cfg = quick_cfg();
    cfg.rollback.verify_after_calls = 2;
    let mut ctl = StructuralPlasticityController::new(cfg).unwrap();
    ctl.adapt(&mut n, Metrics::loss(1.0), None).unwrap();
    ctl.adapt(&mut n, Metrics::loss(1.0), None).unwrap();
    for _ in 0..3 {
        assert!(
            !ctl.adapt(&mut n, Metrics::loss(0.9), None)
                .unwrap()
                .rolled_back
        );
    }
    assert!(n.topology().neuron_count() >= 4);
}

#[test]
fn controller_handles_empty_and_minimal_networks() {
    let mut ctl = StructuralPlasticityController::new(StructuralConfig::default()).unwrap();
    let mut empty = net(topo(TopologyLimits::default()));
    for _ in 0..5 {
        ctl.adapt(&mut empty, Metrics::loss(1.0), None).unwrap();
    }
    empty.step(&[]).unwrap();
    let mut t = topo(TopologyLimits::default());
    let i = inp(&mut t);
    let o = out(&mut t, 0.9, 1.0);
    syn(&mut t, i, o, 1.0);
    let mut minimal = net(t);
    let mut cfg = StructuralConfig::default();
    cfg.schedule.min_interval_steps = 0;
    cfg.pruning.min_age_steps = 0;
    cfg.pruning.min_observed_samples = 0;
    let mut ctl = StructuralPlasticityController::new(cfg).unwrap();
    for _ in 0..20 {
        minimal.begin_sample();
        minimal.step(&[0.0]).unwrap();
        ctl.adapt(&mut minimal, Metrics::loss(1.0), None).unwrap();
        minimal.topology().validate().unwrap();
    }
    assert!(minimal.topology().connected(i, o) || minimal.topology().synapse_count() == 0);
}

#[test]
fn homeostasis_raises_threshold_of_overactive_neurons_within_limits() {
    let mut t = topo(TopologyLimits::default());
    let i = inp(&mut t);
    let o = out(&mut t, 0.0, 0.5);
    syn(&mut t, i, o, 1.0);
    let mut n = net(t);
    for _ in 0..400 {
        n.step(&[1.0]).unwrap();
    }
    let mut cfg = StructuralConfig::default();
    cfg.schedule.min_interval_steps = 0;
    cfg.pruning.enabled = false;
    cfg.growth.enabled = false;
    cfg.homeostasis.rate_max = 0.5;
    cfg.homeostasis.min_observed_steps = 10;
    let mut ctl = StructuralPlasticityController::new(cfg).unwrap();
    let before = n.topology().neuron(o).unwrap().threshold();
    let r = ctl.adapt(&mut n, Metrics::loss(1.0), None).unwrap();
    assert_eq!(r.applied, 1);
    let after = n.topology().neuron(o).unwrap().threshold();
    assert!(after > before && after <= ParamLimits::default().threshold_max);
    assert!(r.events[0].reason.contains("overload"));
}

#[test]
fn best_topology_snapshot_can_be_restored() {
    let (t, ..) = chain();
    let mut n = warmed(t);
    let mut ctl = StructuralPlasticityController::new(quick_cfg()).unwrap();
    ctl.adapt(&mut n, Metrics::loss(0.2), None).unwrap();
    assert_eq!(ctl.best().unwrap().topology.neuron_count(), 3);
    ctl.adapt(&mut n, Metrics::loss(0.2), None).unwrap(); // plateau → growth
    assert_eq!(n.topology().neuron_count(), 4);
    assert!(ctl.restore_best(&mut n));
    assert_eq!(n.topology().neuron_count(), 3);
}

#[test]
fn invalid_configs_are_rejected() {
    let mut c = StructuralConfig::default();
    c.params.beta_max = 1.0;
    assert!(StructuralPlasticityController::new(c).is_err());
    let mut c = StructuralConfig::<f32>::default();
    c.schedule.analyze_every = 0;
    assert!(c.validate().is_err());
    let mut c = StructuralConfig::default();
    c.pruning.max_fraction_per_interval = 2.0;
    assert!(c.validate().is_err());
    assert!(StructuralConfig::<f32>::default().validate().is_ok());
}

#[test]
fn custom_cost_and_linear_cost() {
    struct OnlyNeurons;
    impl StructureCost for OnlyNeurons {
        fn cost(&self, loss: f32, s: &StructuralStats) -> f32 {
            loss + s.neurons as f32
        }
    }
    let (t, ..) = chain();
    let n = warmed(t);
    let stats = n.stats(0.001, 0.5);
    assert_eq!(
        (
            stats.neurons,
            stats.synapses,
            stats.inputs,
            stats.hidden,
            stats.outputs
        ),
        (3, 2, 1, 1, 1)
    );
    let ctl = StructuralPlasticityController::new(StructuralConfig::default())
        .unwrap()
        .with_cost(Box::new(OnlyNeurons));
    assert_eq!(ctl.cost(1.0, &stats), 4.0);
    let lin = LinearCost {
        lambda_neurons: 1.0,
        lambda_synapses: 10.0,
        lambda_activity: 0.0,
        lambda_state_bytes: 0.0,
    };
    assert_eq!(
        lin.cost(0.5, &stats),
        0.5 + stats.active_neurons as f32 + 20.0
    );
}

// ───────────────────────── checkpoint ─────────────────────────

fn trained() -> (DynamicNetwork, StructuralPlasticityController) {
    let (t, ..) = chain();
    let mut n = stdp_net(t, PlasticityMode::Immediate);
    n.enable_coactivity(4, 64, 16);
    let mut cfg = quick_cfg();
    cfg.rollback.verify_after_calls = 4;
    cfg.growth.synapses.max_new_per_interval = 1;
    cfg.seed = 11;
    let mut ctl = StructuralPlasticityController::new(cfg).unwrap();
    for k in 0..6 {
        n.begin_sample();
        run(&mut n, &PATTERN);
        ctl.adapt(&mut n, Metrics::loss(1.0), None).unwrap();
        let _ = k;
    }
    (n, ctl)
}

#[test]
fn checkpoint_roundtrip_restores_everything() {
    let (n, ctl) = trained();
    assert!(!ctl.log().is_empty());
    let bytes = save_checkpoint(&n, &ctl, b"optimizer-state").unwrap();
    let c = load_checkpoint(&bytes).unwrap();
    assert_eq!(*c.network.topology(), *n.topology());
    assert_eq!(c.network.step_count(), n.step_count());
    assert_eq!(c.network.sample_count(), n.sample_count());
    assert_eq!(c.network.config(), n.config());
    assert_eq!(c.network.tracker(), n.tracker());
    assert_eq!(c.controller.log(), ctl.log());
    assert_eq!(c.controller.config(), ctl.config());
    assert_eq!(
        c.controller.plateau_evaluations(),
        ctl.plateau_evaluations()
    );
    assert_eq!(c.controller.best(), ctl.best());
    assert_eq!(c.extra, b"optimizer-state");
    assert_eq!(
        save_checkpoint(&c.network, &c.controller, &c.extra).unwrap(),
        bytes,
        "re-saving is identical"
    );
}

#[test]
fn restored_run_continues_identically_to_uninterrupted_run() {
    let (mut a, mut ca) = trained();
    let bytes = save_checkpoint(&a, &ca, &[]).unwrap();
    let Checkpoint {
        network: mut bnet,
        controller: mut cb,
        ..
    } = load_checkpoint(&bytes).unwrap();
    for _ in 0..8 {
        a.begin_sample();
        bnet.begin_sample();
        let (oa, ob) = (run(&mut a, &PATTERN), run(&mut bnet, &PATTERN));
        assert_eq!(oa, ob);
        let ra = ca.adapt(&mut a, Metrics::loss(1.0), None).unwrap();
        let rb = cb.adapt(&mut bnet, Metrics::loss(1.0), None).unwrap();
        assert_eq!(
            (ra.applied, ra.rolled_back, ra.events.len()),
            (rb.applied, rb.rolled_back, rb.events.len())
        );
        assert_eq!(a.topology(), bnet.topology());
    }
    assert_eq!(ca.log(), cb.log());
}

#[test]
fn checkpoint_rejects_corruption_truncation_and_future_versions() {
    let (n, ctl) = trained();
    let bytes = save_checkpoint(&n, &ctl, &[]).unwrap();
    let mut flipped = bytes.clone();
    flipped[bytes.len() / 2] ^= 0x55;
    assert!(
        matches!(load_checkpoint(&flipped), Err(StructuralError::Checkpoint(m)) if m.contains("checksum"))
    );
    assert!(load_checkpoint(&bytes[..bytes.len() - 20]).is_err());
    assert!(load_checkpoint(&bytes[..10]).is_err());
    assert!(load_checkpoint(b"").is_err());
    let mut wrong_magic = bytes.clone();
    wrong_magic[0] = b'X';
    assert!(load_checkpoint(&wrong_magic).is_err());
    // newer version of the format (with correct checksum)
    let mut future = bytes.clone();
    future[4..8].copy_from_slice(&(FORMAT_VERSION_F64 + 1).to_le_bytes());
    let body = future.len() - 8;
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for byte in &future[..body] {
        h ^= *byte as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    future[body..].copy_from_slice(&h.to_le_bytes());
    assert!(
        matches!(load_checkpoint(&future), Err(StructuralError::Checkpoint(m)) if m.contains("version"))
    );
}

#[test]
fn checkpoint_file_roundtrip() {
    let (n, ctl) = trained();
    let dir = std::env::temp_dir().join(format!("neo_snn_ckpt_{}", std::process::id()));
    std::fs::create_dir_all(&dir).unwrap();
    let path = dir.join("net.nstr");
    save_to_file(&path, &n, &ctl, b"x").unwrap();
    let c = load_from_file(&path).unwrap();
    assert_eq!(*c.network.topology(), *n.topology());
    assert!(load_from_file(dir.join("missing.nstr")).is_err());
    std::fs::remove_dir_all(&dir).unwrap();
}

#[test]
fn batch_is_all_or_nothing() {
    let (t, i, h, o) = chain();
    let mut n = net(t);
    let before = n.topology().clone();
    let mut rng = Rng64::new(1);
    let bad = [
        StructuralAction::CloneNeuron { id: h, jitter: 0.0 },
        StructuralAction::AddSynapse {
            src: i,
            dst: o,
            weight: 0.5,
            delay: 0,
        },
        StructuralAction::AddSynapse {
            src: i,
            dst: o,
            weight: 0.5,
            delay: 0,
        }, // duplicate
    ];
    assert!(apply_batch(&mut n, &bad, &mut rng).is_err());
    assert_eq!(*n.topology(), before);
    assert_eq!(rng, Rng64::new(1), "the RNG state is also rolled back");
    let ok = &bad[..2];
    assert_eq!(apply_batch(&mut n, ok, &mut rng).unwrap().len(), 2);
    assert_eq!(n.topology().neuron_count(), 4);
}

#[test]
fn rng_is_deterministic_and_resumable() {
    let mut a = Rng64::new(5);
    let mut b2 = Rng64::from_state(a.state());
    assert!((0..100).all(|_| a.next_u64() == b2.next_u64()));
    let mut r = Rng64::new(1);
    assert!((0..1000).all(|_| (0.0..1.0).contains(&r.next_f32())));
}

#[test]
fn no_new_changes_while_previous_one_is_being_verified() {
    let (t, ..) = chain();
    let mut n = warmed(t);
    let mut cfg = quick_cfg();
    cfg.rollback.verify_after_calls = 3;
    let mut ctl = StructuralPlasticityController::new(cfg).unwrap();
    ctl.adapt(&mut n, Metrics::loss(1.0), None).unwrap();
    assert_eq!(
        ctl.adapt(&mut n, Metrics::loss(1.0), None).unwrap().applied,
        1
    );
    let r = ctl.adapt(&mut n, Metrics::loss(1.0), None).unwrap();
    assert_eq!((r.applied, r.skipped), (0, Some(SkipReason::Verifying)));
    assert_eq!(n.topology().neuron_count(), 4);
}

// ───────────── growth: target loss, lateral outputs, plan conflicts ─────────────

#[test]
fn lateral_output_links_are_not_grown_unless_allowed() {
    let mut t = topo(TopologyLimits::default());
    let o1 = out(&mut t, 0.9, 1.0);
    let o2 = out(&mut t, 0.9, 1.0);
    let mut n = net(t);
    n.enable_coactivity(4, 128, 32);
    for _ in 0..20 {
        n.step_with_current(&[], &[(o1, 2.0)]).unwrap();
        n.step_with_current(&[], &[(o2, 2.0)]).unwrap();
    }
    let mut cfg = GrowthConfig::default();
    cfg.synapses.min_score = 1.0;
    let mut rng = Rng64::new(1);
    assert!(
        propose_grow_synapses(&n, &cfg, &mut rng, 4).is_empty(),
        "by default output→output disabled"
    );
    cfg.allow_output_to_output = true;
    let plan = propose_grow_synapses(&n, &cfg, &mut rng, 4);
    assert!(plan.iter().any(|(a, _)| matches!(a, StructuralAction::AddSynapse { src, dst, .. } if *src == o1 && *dst == o2)));
}

#[test]
fn growth_waits_until_loss_is_above_target() {
    let (mut n, _, o) = growth_net();
    let mut cfg = quick_cfg();
    cfg.rollback.enabled = false;
    cfg.growth.synapses.max_new_per_interval = 2;
    cfg.growth.synapses.min_score = 1.0;
    cfg.growth.neurons.enabled = false;
    cfg.growth.target_loss = Some(0.5);
    let mut ctl = StructuralPlasticityController::new(cfg).unwrap();
    ctl.adapt(&mut n, Metrics::loss(0.1), None).unwrap(); // turns on the drive
    for _ in 0..20 {
        n.step(&[1.0]).unwrap();
        n.step_with_current(&[0.0], &[(o, 2.0)]).unwrap();
    }
    let r = ctl.adapt(&mut n, Metrics::loss(0.1), None).unwrap();
    assert_eq!(
        r.applied, 0,
        "the problem is solved - the network does not become more complicated"
    );
    let r = ctl.adapt(&mut n, Metrics::loss(0.9), None).unwrap();
    assert_eq!(r.applied, 1, "loss above the target - growth is allowed");
}

#[test]
fn plan_does_not_touch_synapses_of_neurons_it_removes() {
    let mut t = topo(TopologyLimits::default());
    let i = inp(&mut t);
    let h = hid(&mut t, 0.9, 1.0);
    let o = out(&mut t, 0.9, 1.0);
    syn(&mut t, i, o, 0.2);
    syn(&mut t, h, o, 0.3); // h without inputs: “dead” neuron; its synapse is also a candidate for shutdown
    let mut n = net(t);
    for _ in 0..3 {
        n.begin_sample();
        run(&mut n, &PATTERN);
    }
    let mut cfg = quick_cfg();
    cfg.pruning.enabled = true;
    cfg.pruning.min_age_steps = 0;
    cfg.pruning.min_observed_samples = 0;
    cfg.pruning.reeval_interval_steps = 0;
    cfg.rollback.enabled = false;
    cfg.growth.enabled = false;
    let mut ctl = StructuralPlasticityController::new(cfg).unwrap();
    let r = ctl.adapt(&mut n, Metrics::loss(1.0), None).unwrap();
    assert_eq!(r.failed, 0, "events: {:?}", r.events);
    assert!(
        r.events
            .iter()
            .all(|e| !matches!(e.outcome, Outcome::Failed(_)))
    );
    assert!(n.topology().neuron(h).is_none());
}

#[test]
fn deferred_check_also_guards_changes_validated_by_an_evaluator() {
    let (t, ..) = chain();
    let mut n = warmed(t);
    let mut cfg = quick_cfg();
    cfg.rollback.verify_after_calls = 2;
    let mut ctl = StructuralPlasticityController::new(cfg).unwrap();
    ctl.adapt(&mut n, Metrics::loss(1.0), None).unwrap();
    let snapshot = n.topology().clone();
    // Static evaluation change approves...
    let mut eval = |p: &mut DynamicNetwork| 1.0 - 0.1 * p.topology().neuron_count() as f32;
    let r = ctl
        .adapt(&mut n, Metrics::loss(1.0), Some(&mut eval))
        .unwrap();
    assert_eq!((r.applied, r.rolled_back), (1, false));
    // ...but the actual learning metrics then deteriorate—the change is rolled back.
    ctl.adapt(&mut n, Metrics::loss(3.0), None).unwrap();
    let r = ctl.adapt(&mut n, Metrics::loss(3.0), None).unwrap();
    assert!(r.rolled_back);
    assert_eq!(*n.topology(), snapshot);
}

// ───────────────────────── heterogeneous neurons ─────────────────────────

#[test]
fn inhibitory_polarity_inverts_the_weight_sign() {
    let mut t = topo(TopologyLimits::default());
    let i = t
        .add_neuron_with(
            NeuronRole::Input,
            NeuronPolarity::Inhibitory,
            LifParams::default(),
            Origin::Initial,
            b(),
        )
        .unwrap();
    let o = t
        .add_neuron_with(
            NeuronRole::Output,
            NeuronPolarity::Excitatory,
            LifParams::new(0.0, 1.0),
            Origin::Initial,
            b(),
        )
        .unwrap();
    syn(&mut t, i, o, 1.0);
    let mut n = net(t);
    assert!(!n.step(&[1.0]).unwrap().output_spikes[0]);
    let out = n.step(&[0.0]).unwrap();
    assert!(!out.output_spikes[0]);
    assert!(n.topology().neuron(o).unwrap().mem() < 0.0);

    // Control: excitatory input with the same weight ignites the output.
    let mut t = topo(TopologyLimits::default());
    let i = inp(&mut t);
    let o = t
        .add_neuron_with(
            NeuronRole::Output,
            NeuronPolarity::Excitatory,
            LifParams::new(0.0, 1.0),
            Origin::Initial,
            b(),
        )
        .unwrap();
    syn(&mut t, i, o, 1.0);
    let mut n = net(t);
    n.step(&[1.0]).unwrap();
    assert!(n.step(&[0.0]).unwrap().output_spikes[0]);
}

fn modulatory_net(weight: f32) -> (DynamicNetwork, NeuronId) {
    let mut t = topo(TopologyLimits::default());
    let e = inp(&mut t);
    let m = t
        .add_neuron_with(
            NeuronRole::Input,
            NeuronPolarity::Modulatory,
            LifParams::default(),
            Origin::Initial,
            b(),
        )
        .unwrap();
    let o = t
        .add_neuron_with(
            NeuronRole::Output,
            NeuronPolarity::Excitatory,
            LifParams::new(0.0, 1.0),
            Origin::Initial,
            b(),
        )
        .unwrap();
    syn(&mut t, e, o, 0.6);
    syn(&mut t, m, o, weight);
    (net(t), o)
}

#[test]
fn modulatory_polarity_scales_current_and_decays() {
    let (mut weak, _) = modulatory_net(0.5);
    weak.step(&[1.0, 1.0]).unwrap();
    assert!(!weak.step(&[0.0, 0.0]).unwrap().output_spikes[0]);

    let (mut strong, o) = modulatory_net(1.0);
    strong.step(&[1.0, 1.0]).unwrap();
    assert!(strong.step(&[0.0, 0.0]).unwrap().output_spikes[0]);
    let gain = strong.topology().neuron(o).unwrap().modulation;
    assert!((gain - 0.9).abs() < 1e-6, "modulation decays: {gain}");
    strong.step(&[0.0, 0.0]).unwrap();
    let gain = strong.topology().neuron(o).unwrap().modulation;
    assert!((gain - 0.81).abs() < 1e-6, "modulation decays: {gain}");
}

#[test]
fn heterogeneous_models_run_in_one_network() {
    let mut t = topo(TopologyLimits::default());
    let i = inp(&mut t);
    let alif = t
        .add_neuron_with(
            NeuronRole::Hidden,
            NeuronPolarity::Excitatory,
            AdaptiveLifParams::slow_adaptation(),
            Origin::Initial,
            b(),
        )
        .unwrap();
    let izh = t
        .add_neuron_with(
            NeuronRole::Hidden,
            NeuronPolarity::Excitatory,
            IzhikevichParams::regular_spiking(),
            Origin::Initial,
            b(),
        )
        .unwrap();
    let hh = t
        .add_neuron_with(
            NeuronRole::Hidden,
            NeuronPolarity::Excitatory,
            HodgkinHuxleyParams::default(),
            Origin::Initial,
            b(),
        )
        .unwrap();
    let o = t
        .add_neuron_with(
            NeuronRole::Output,
            NeuronPolarity::Excitatory,
            LifParams::new(0.9, 1.0),
            Origin::Initial,
            b(),
        )
        .unwrap();
    syn(&mut t, i, alif, 1.0);
    syn(&mut t, i, izh, 10.0);
    syn(&mut t, i, hh, 10.0);
    syn(&mut t, alif, o, 1.0);
    syn(&mut t, izh, o, 0.1);
    syn(&mut t, hh, o, 0.1);
    let mut n = net(t);
    for _ in 0..20 {
        n.step(&[1.0]).unwrap();
        n.topology().validate().unwrap();
    }
    for id in [alif, izh, hh] {
        let neuron = n.topology().neuron(id).unwrap();
        assert!(
            neuron.state.is_finite(),
            "{}: status",
            neuron.model.as_str()
        );
        assert!(matches!(
            (&neuron.model, &neuron.state),
            (NeuronModel::AdaptiveLif(_), NeuronState::AdaptiveLif(_))
                | (NeuronModel::Izhikevich(_), NeuronState::Izhikevich(_))
                | (NeuronModel::HodgkinHuxley(_), NeuronState::HodgkinHuxley(_))
        ));
    }
    assert!(n.topology().neuron(alif).unwrap().stats.spikes > 0);
}

#[test]
fn growth_can_create_specialized_alif_neuron() {
    let mut t = topo(TopologyLimits::default());
    let i = inp(&mut t);
    let o = out(&mut t, 0.9, 1.0);
    syn(&mut t, i, o, 1.0);
    let mut n = net(t);
    n.step(&[1.0]).unwrap();

    let mut cfg = NeuronGrowthConfig::default();
    cfg.strategies = vec![GrowthStrategy::NewNeuron];
    cfg.new_neuron_model = Some(NeuronModel::AdaptiveLif(
        AdaptiveLifParams::slow_adaptation(),
    ));
    cfg.new_neuron_polarity = NeuronPolarity::Inhibitory;
    cfg.new_neuron_fan = 1;
    let mut rng = Rng64::new(3);
    let new = grow_neurons(&mut n, &cfg, &mut rng, 1).unwrap();
    assert_eq!(new.len(), 1);
    let neuron = n.topology().neuron(new[0]).unwrap();
    assert!(matches!(neuron.model, NeuronModel::AdaptiveLif(_)));
    assert_eq!(neuron.polarity, NeuronPolarity::Inhibitory);
    assert!(matches!(neuron.state, NeuronState::AdaptiveLif(_)));
}

#[test]
fn dense_executor_supports_alif_inhibition_and_rejects_others() {
    // ALIF is supported now.
    let mut t = topo(TopologyLimits::default());
    inp(&mut t);
    t.add_neuron_with(
        NeuronRole::Hidden,
        NeuronPolarity::Excitatory,
        AdaptiveLifParams::slow_adaptation(),
        Origin::Initial,
        b(),
    )
    .unwrap();
    out(&mut t, 0.9, 1.0);
    assert!(super::dense::DenseNet::compile(&t).is_ok());

    // Inhibition is supported (signs of weights).
    let mut t = topo(TopologyLimits::default());
    inp(&mut t);
    t.add_neuron_with(
        NeuronRole::Hidden,
        NeuronPolarity::Inhibitory,
        LifParams::default(),
        Origin::Initial,
        b(),
    )
    .unwrap();
    out(&mut t, 0.9, 1.0);
    assert!(super::dense::DenseNet::compile(&t).is_ok());

    // Unsupported models are still rejected.
    let mut t = topo(TopologyLimits::default());
    inp(&mut t);
    t.add_neuron_with(
        NeuronRole::Hidden,
        NeuronPolarity::Excitatory,
        IzhikevichParams::regular_spiking(),
        Origin::Initial,
        b(),
    )
    .unwrap();
    out(&mut t, 0.9, 1.0);
    assert!(super::dense::DenseNet::compile(&t).is_err());
}

#[test]
fn scaled_migration_preserves_position_between_rest_and_threshold() {
    let lif = NeuronModel::Lif(LifParams::new(0.9, 1.0));
    let izh = NeuronModel::Izhikevich(IzhikevichParams::regular_spiking());
    let adex = NeuronModel::AdEx(AdExParams::default());
    let hh = NeuronModel::HodgkinHuxley(HodgkinHuxleyParams::default());

    // LIF halfway to the threshold.
    let source: NeuronState = NeuronState::Lif(LifState { mem: 0.5 });
    for target in [&izh, &adex, &hh] {
        let (rest, fire) = target.membrane_range().unwrap();
        let next = source
            .clone()
            .migrate_between(&lif, target, StateMigration::Scaled);
        assert!(next.matches(target));
        assert!(next.is_finite());
        let expected = rest + 0.5 * (fire - rest);
        assert!((next.mem() - expected).abs() < 1e-3, "{target:?}");
    }
    // Fast variables are in equilibrium, not in “alien” units.
    let NeuronState::Izhikevich(s) =
        source
            .clone()
            .migrate_between(&lif, &izh, StateMigration::Scaled)
    else {
        panic!("expected status Izhikevich");
    };
    assert!((s.u - 0.2 * s.v).abs() < 1e-4);

    // Reverse: mV -> dimensionless scale, the value remains below the threshold.
    let back = NeuronState::AdEx(AdExState { v: -60.0, w: 5.0 }).migrate_between(
        &adex,
        &lif,
        StateMigration::Scaled,
    );
    assert!(matches!(back, NeuronState::Lif(s) if s.mem > 0.0 && s.mem < 1.0));

    // The membrane above the threshold does not cause an instantaneous spike in the new model.
    let hot = NeuronState::Lif(LifState { mem: 7.0 });
    let next = hot.migrate_between(&lif, &izh, StateMigration::Scaled);
    assert!(next.mem() < izh.threshold());

    // The same model - the state is not affected; Compatible/Membrane does not change.
    assert_eq!(
        source
            .clone()
            .migrate_between(&lif, &lif, StateMigration::Scaled),
        source
    );
    assert_eq!(
        source
            .clone()
            .migrate_between(&lif, &izh, StateMigration::Compatible),
        izh.init_state()
    );
}

#[test]
fn scaled_migration_applies_through_action_and_checkpoint_code() {
    assert_eq!(
        StateMigration::from_code(StateMigration::Scaled.code()),
        Some(StateMigration::Scaled)
    );
    assert_eq!(StateMigration::Scaled.as_str(), "scaled");

    let (t, _, h, _) = chain();
    let source_mem = t.neuron(h).unwrap().state.mem();
    let source_model = t.neuron(h).unwrap().model.clone();
    let mut n = net(t);
    let mut rng = Rng64::new(1);
    let target = NeuronModel::AdEx(AdExParams::default());
    apply_action(
        &mut n,
        &StructuralAction::ChangeNeuronModel {
            id: h,
            model: target.clone(),
            migration: StateMigration::Scaled,
        },
        &mut rng,
    )
    .unwrap();
    let nn = n.topology().neuron(h).unwrap();
    let (src_rest, src_fire) = source_model.membrane_range().unwrap();
    let (rest, fire) = target.membrane_range().unwrap();
    let fraction = (source_mem - src_rest) / (src_fire - src_rest);
    assert!((nn.state.mem() - (rest + fraction * (fire - rest))).abs() < 1e-3);
    n.topology().validate().unwrap();
}

// Dale principle and E/I network ─────────────────────────

fn set_polarity(t: &mut Topology, id: NeuronId, p: NeuronPolarity) {
    t.mutate_neuron(
        id,
        NeuronMutation {
            polarity: Some(p),
            ..Default::default()
        },
    )
    .unwrap();
}

#[test]
fn dale_is_on_by_default_and_new_synapses_take_source_polarity() {
    let mut t = topo(TopologyLimits::default());
    assert!(t.dale());
    let e = inp(&mut t);
    let i = inp(&mut t);
    let m = inp(&mut t);
    let o = out(&mut t, 0.9, 1.0);
    set_polarity(&mut t, i, NeuronPolarity::Inhibitory);
    set_polarity(&mut t, m, NeuronPolarity::Modulatory);
    // The calling code can pass any character: the polarity of the source decides.
    let se = syn(&mut t, e, o, -0.5);
    let si = syn(&mut t, i, o, 0.5);
    let sm = syn(&mut t, m, o, -0.4);
    assert_eq!(t.synapse(se).unwrap().weight, 0.5);
    assert_eq!(t.synapse(si).unwrap().weight, -0.5);
    assert_eq!(t.synapse(sm).unwrap().weight, 0.4);
    t.validate().unwrap();
    // The change in weight cannot pass through zero: the sign is preserved.
    assert_eq!(t.set_synapse_weight(se, -2.0).unwrap(), 0.0);
    assert_eq!(t.set_synapse_weight(si, 2.0).unwrap(), 0.0);
    assert_eq!(t.set_synapse_weight(si, -0.7).unwrap(), -0.7);
    t.mutate_synapse(
        se,
        SynapseMutation {
            weight: Some(-1.0),
            delay: None,
        },
    )
    .unwrap();
    assert_eq!(t.synapse(se).unwrap().weight, 0.0);
    t.validate().unwrap();
}

#[test]
fn dale_violation_is_rejected_by_validate_and_legacy_mode_allows_free_signs() {
    let mut t = topo(TopologyLimits::default());
    let i = inp(&mut t);
    let o = out(&mut t, 0.9, 1.0);
    let s = syn(&mut t, i, o, 0.5);
    t.synapse_mut(s).unwrap().weight = -0.5;
    assert!(matches!(
        t.validate(),
        Err(StructuralError::InvariantViolation(_))
    ));
    assert_eq!(t.ei_stats().dale_violations, 1);
    // Conversion of a legacy network into E/I: signs are set, magnitudes are preserved.
    assert!(!{
        let mut legacy = topo(TopologyLimits {
            enforce_dale: false,
            ..TopologyLimits::default()
        });
        let a = inp(&mut legacy);
        let b = out(&mut legacy, 0.9, 1.0);
        let sid = syn(&mut legacy, a, b, -0.3);
        assert_eq!(legacy.synapse(sid).unwrap().weight, -0.3);
        legacy.validate().unwrap();
        assert_eq!(legacy.set_enforce_dale(true), 1);
        legacy.validate().unwrap();
        legacy.synapse(sid).unwrap().weight < 0.0
    });
}

#[test]
fn polarity_change_rewire_and_clone_keep_dale() {
    let mut t = topo(TopologyLimits::default());
    let a = inp(&mut t);
    let b = inp(&mut t);
    let h = hid(&mut t, 0.9, 1.0);
    let o = out(&mut t, 0.9, 1.0);
    syn(&mut t, a, h, 0.9);
    let s = syn(&mut t, h, o, 0.6);
    // Changing the polarity reverses the outgoing weights, the magnitude remains.
    set_polarity(&mut t, h, NeuronPolarity::Inhibitory);
    assert_eq!(t.synapse(s).unwrap().weight, -0.6);
    t.validate().unwrap();
    // Redirection to the excitatory source returns a positive sign.
    t.rewire_synapse(s, Some(b), None).unwrap();
    assert_eq!(t.synapse(s).unwrap().weight, 0.6);
    t.validate().unwrap();
    // The clone retains its polarity, and its connections are the sign of the source.
    set_polarity(&mut t, h, NeuronPolarity::Inhibitory);
    let o2 = out(&mut t, 0.9, 1.0);
    syn(&mut t, h, o2, 0.5);
    let mut rng = Rng64::new(3);
    let c = t.clone_neuron(h, 0.3, &mut rng, b_birth()).unwrap();
    assert_eq!(t.neuron(c).unwrap().polarity, NeuronPolarity::Inhibitory);
    assert!(
        t.out_synapses(c)
            .iter()
            .all(|id| t.synapse(*id).unwrap().weight <= 0.0)
    );
    t.validate().unwrap();
}

fn b_birth() -> Birth {
    b()
}

#[test]
fn inhibitory_neuron_suppresses_its_target_with_dale() {
    let mut t = topo(TopologyLimits::default());
    let drive = inp(&mut t);
    let brake = inp(&mut t);
    let o = out(&mut t, 0.9, 1.0);
    set_polarity(&mut t, brake, NeuronPolarity::Inhibitory);
    syn(&mut t, drive, o, 0.6);
    syn(&mut t, brake, o, 0.6);
    let mut n = net(t);
    n.step(&[1.0, 1.0]).unwrap();
    n.step(&[0.0, 0.0]).unwrap();
    let mem_with_brake = n.topology().neuron(o).unwrap().state.mem();
    // The same current without the inhibitory input produces a positive membrane.
    let mut t2 = topo(TopologyLimits::default());
    let d2 = inp(&mut t2);
    let o2 = out(&mut t2, 0.9, 1.0);
    syn(&mut t2, d2, o2, 0.6);
    let mut n2 = net(t2);
    n2.step(&[1.0]).unwrap();
    n2.step(&[0.0]).unwrap();
    let mem_alone = n2.topology().neuron(o2).unwrap().state.mem();
    assert!(mem_alone > 0.4, "{mem_alone}");
    assert!(
        mem_with_brake.abs() < 1e-5,
        "E and I cancel each other: {mem_with_brake}"
    );
    let ex = n.topology().neuron(o).unwrap().stats.clone();
    assert!(ex.exc_drive_ema > 0.0 && ex.inh_drive_ema > 0.0);
}

fn ei_pair(init: f32, eta: f32, target_rate: f32) -> (DynamicNetwork, SynapseId, NeuronId) {
    let mut t = topo(TopologyLimits::default());
    let brake = inp(&mut t);
    let drive = inp(&mut t);
    let o = out(&mut t, 0.9, 1.0);
    set_polarity(&mut t, brake, NeuronPolarity::Inhibitory);
    let s = syn(&mut t, brake, o, init);
    syn(&mut t, drive, o, 1.2);
    let mut cfg = EngineConfig::default();
    cfg.inhibitory = InhibitoryPlasticityConfig {
        enabled: true,
        eta,
        target_rate,
    };
    let mut n = DynamicNetwork::new(t, cfg).unwrap();
    n.set_plasticity_enabled(true);
    (n, s, o)
}

#[test]
fn inhibitory_stdp_strengthens_when_target_is_exceeded_and_weakens_below_it() {
    // The receiver accelerates, the target is 0 → inhibition increases in magnitude.
    let (mut n, s, _) = ei_pair(-0.1, 0.01, 0.0);
    for _ in 0..200 {
        n.step(&[1.0, 1.0]).unwrap();
    }
    let w = n.topology().synapse(s).unwrap().weight;
    assert!((-5.0..-0.1).contains(&w), "{w}");
    n.topology().validate().unwrap();

    // Target is higher than actual frequency → inhibition is reduced to zero, but does not
    // change sign.
    let (mut n, s, _) = ei_pair(-0.4, 0.01, 0.9);
    for _ in 0..400 {
        n.step(&[1.0, 1.0]).unwrap();
    }
    let w = n.topology().synapse(s).unwrap().weight;
    assert!(w > -0.4 && w <= 0.0, "{w}");
    n.topology().validate().unwrap();

    // A disabled rule does not change anything.
    let (n, s, _) = ei_pair(-0.1, 0.01, 0.0);
    let mut cfg = n.config().clone();
    cfg.inhibitory.enabled = false;
    let mut t = n.topology().clone();
    t.synapse_mut(s).unwrap().weight = -0.1;
    let mut n = DynamicNetwork::new(t, cfg).unwrap();
    n.set_plasticity_enabled(true);
    for _ in 0..50 {
        n.step(&[1.0, 1.0]).unwrap();
    }
    assert_eq!(n.topology().synapse(s).unwrap().weight, -0.1);
}

#[test]
fn excitatory_stdp_cannot_flip_sign_and_ignores_inhibitory_synapses() {
    let mut t = topo(TopologyLimits::default());
    let a = inp(&mut t);
    let brake = inp(&mut t);
    let o = out(&mut t, 0.9, 1.0);
    set_polarity(&mut t, brake, NeuronPolarity::Inhibitory);
    let se = syn(&mut t, a, o, 0.05);
    let si = syn(&mut t, brake, o, 0.3);
    let mut n = stdp_net(t, PlasticityMode::Immediate);
    for k in 0..300 {
        // Output active without presynaptic spike a: pure depression.
        n.step(&[if k % 7 == 0 { 1.0 } else { 0.0 }, 1.0]).unwrap();
    }
    let we = n.topology().synapse(se).unwrap().weight;
    assert!(
        we >= 0.0,
        "the excitatory weight has passed through zero: {we}"
    );
    // Ordinary STDP does not touch inhibitory synapses (they have their own rule).
    assert_eq!(n.topology().synapse(si).unwrap().weight, -0.3);
    n.topology().validate().unwrap();
}

#[test]
fn balance_inhibition_scales_incoming_inhibition_locally() {
    let build = |exc: f32, inh: f32| {
        let mut t = topo(TopologyLimits::default());
        let a = inp(&mut t);
        let brake = inp(&mut t);
        let o = out(&mut t, 0.9, 1.0);
        set_polarity(&mut t, brake, NeuronPolarity::Inhibitory);
        syn(&mut t, a, o, 0.5);
        let s = syn(&mut t, brake, o, 0.5);
        let st = &mut t.neurons.get_mut(&o).unwrap().stats;
        st.steps = 1000;
        st.exc_drive_ema = exc;
        st.inh_drive_ema = inh;
        (net(t), s)
    };
    let cfg = EiBalanceConfig {
        enabled: true,
        ..Default::default()
    };
    // There is little inhibition (inh/exc = 0.2 with a target of 1.0) → increases.
    let (mut n, s) = build(1.0, 0.2);
    let report = n.balance_inhibition(&cfg);
    assert_eq!(report.adjusted, 1);
    assert!((report.mean_ratio - 0.2).abs() < 1e-6);
    let w = n.topology().synapse(s).unwrap().weight;
    assert!((w + 0.54).abs() < 1e-5, "{w}");
    // There is a lot of inhibition → weakened (the step is limited by max_step).
    let (mut n, s) = build(1.0, 3.0);
    n.balance_inhibition(&cfg);
    let w = n.topology().synapse(s).unwrap().weight;
    assert!((w + 0.45).abs() < 1e-5, "{w}");
    n.topology().validate().unwrap();
    // Without observations, the neuron is not regulated.
    let (mut n, s) = build(1.0, 0.2);
    n.topology_mut()
        .neurons
        .get_mut(&NeuronId(2))
        .unwrap()
        .stats
        .steps = 5;
    assert_eq!(n.balance_inhibition(&cfg).adjusted, 0);
    assert_eq!(n.topology().synapse(s).unwrap().weight, -0.5);
}

#[test]
fn ei_stats_report_populations_weights_and_balance() {
    let mut t = topo(TopologyLimits::default());
    let a = inp(&mut t);
    let brake = inp(&mut t);
    let h = hid(&mut t, 0.9, 1.0);
    let o = out(&mut t, 0.9, 1.0);
    set_polarity(&mut t, brake, NeuronPolarity::Inhibitory);
    syn(&mut t, a, h, 0.4);
    syn(&mut t, a, o, 0.6);
    syn(&mut t, brake, o, 0.3);
    {
        let st = &mut t.neurons.get_mut(&h).unwrap().stats;
        st.rate_ema = 0.1;
        st.exc_drive_ema = 0.5;
    }
    {
        let st = &mut t.neurons.get_mut(&o).unwrap().stats;
        st.rate_ema = 0.3;
        st.exc_drive_ema = 1.5;
        st.inh_drive_ema = 1.0;
    }
    let e = t.ei_stats();
    assert_eq!((e.excitatory, e.inhibitory, e.modulatory), (3, 1, 0));
    assert!((e.inhibitory_fraction - 0.25).abs() < 1e-6);
    assert_eq!((e.excitatory_synapses, e.inhibitory_synapses), (2, 1));
    assert!((e.mean_excitatory_weight - 0.5).abs() < 1e-6);
    assert!((e.mean_inhibitory_weight - 0.3).abs() < 1e-6);
    assert!((e.excitatory_rate - 0.2).abs() < 1e-6);
    assert!((e.excitatory_drive - 1.0).abs() < 1e-6);
    assert!((e.inhibitory_drive - 0.5).abs() < 1e-6);
    assert!((e.ei_ratio - 0.5).abs() < 1e-6);
    assert_eq!(e.dale_violations, 0);
}

#[test]
fn checkpoint_v5_roundtrips_dale_ei_state_and_configs() {
    let (mut n, s, o) = ei_pair(-0.2, 0.01, 0.05);
    let mut cfg = n.config().clone();
    cfg.balance = EiBalanceConfig {
        enabled: true,
        target_ratio: 0.8,
        rate: 0.2,
        max_step: 0.3,
        min_drive: 0.01,
        min_observed_steps: 10,
    };
    n = DynamicNetwork::new(n.topology().clone(), cfg.clone()).unwrap();
    for _ in 0..30 {
        n.step(&[1.0, 1.0]).unwrap();
    }
    let ctl = StructuralPlasticityController::new(StructuralConfig::default()).unwrap();
    let c = load_checkpoint(&save_checkpoint(&n, &ctl, b"").unwrap()).unwrap();
    assert!(c.network.topology().dale());
    assert_eq!(c.network.config().balance, cfg.balance);
    assert_eq!(c.network.config().inhibitory, cfg.inhibitory);
    let (a, b) = (
        n.topology().neuron(o).unwrap().stats.clone(),
        c.network.topology().neuron(o).unwrap().stats.clone(),
    );
    assert!(a.exc_drive_ema > 0.0);
    assert_eq!(a, b);
    assert_eq!(
        n.topology().synapse(s).unwrap().weight,
        c.network.topology().synapse(s).unwrap().weight
    );
}

#[test]
fn ei_stats_support_f64_and_custom_scalars_with_f64_accumulation() {
    let mut t = topo(TopologyLimits::default());
    let a = inp(&mut t);
    let h = hid(&mut t, 0.9, 1.0);
    let o = out(&mut t, 0.9, 1.0);
    syn(&mut t, a, h, 0.1);
    syn(&mut t, a, o, 0.2);
    for id in [h, o] {
        let st = &mut t.neurons.get_mut(&id).unwrap().stats;
        st.rate_ema = 0.1;
        st.exc_drive_ema = 0.3;
        st.inh_drive_ema = 0.1;
    }
    let f32s: EiStats = t.ei_stats();
    let f64s: EiStats<f64> = t.ei_stats_as::<f64>();
    assert_eq!(f32s, f64s.cast::<f32>());
    assert!((f64s.mean_excitatory_weight - f64::from(0.15f32)).abs() < 1e-7);
    assert!((f64s.ei_ratio - f64::from(0.1f32) / f64::from(0.3f32)).abs() < 1e-7);

    // Custom narrow type (imitation f16: 8 significant mantissa bits).
    #[derive(Debug, Clone, Copy, PartialEq)]
    struct Coarse(f32);
    impl StatsScalar for Coarse {
        fn from_f64(v: f64) -> Self {
            Coarse((v * 256.0).round() as f32 / 256.0)
        }
        fn to_f64(self) -> f64 {
            f64::from(self.0)
        }
    }
    let c = t.ei_stats_as::<Coarse>();
    assert_eq!(c.excitatory, f32s.excitatory);
    assert!((c.ei_ratio.to_f64() - f64s.ei_ratio).abs() < 1.0 / 256.0);
}

// generalized scalar type ─────────────────────────

fn simulate_model<T: crate::scalar::Scalar>(
    model: NeuronModel<T>,
    current: T,
    steps: usize,
) -> usize {
    let mut state = model.init_state();
    let ctx = crate::structural::custom::NeuronStepContext::manual(NeuronId(0), 0, 0);
    (0..steps)
        .filter(|_| model.step(&mut state, current, &ctx).unwrap_or(false))
        .count()
}

#[test]
fn neuron_models_run_in_f32_and_f64_with_matching_spike_counts() {
    // For each model, the number of spikes at f32 and f64 coincides over a short window.
    let cases: [(NeuronModel<f64>, f64, usize); 4] = [
        (NeuronModel::Lif(LifParams::new(0.9, 1.0)), 0.3, 200),
        (
            NeuronModel::Izhikevich(IzhikevichParams::regular_spiking()),
            10.0,
            200,
        ),
        (NeuronModel::AdEx(AdExParams::default()), 1000.0, 200),
        (
            NeuronModel::HodgkinHuxley(HodgkinHuxleyParams::default()),
            10.0,
            200,
        ),
    ];
    for (m64, i64_, n) in cases {
        let m32: NeuronModel<f32> = match &m64 {
            NeuronModel::Lif(p) => {
                NeuronModel::Lif(LifParams::new(p.beta as f32, p.threshold as f32))
            }
            NeuronModel::Izhikevich(_) => {
                NeuronModel::Izhikevich(IzhikevichParams::regular_spiking())
            }
            NeuronModel::AdEx(_) => NeuronModel::AdEx(AdExParams::default()),
            NeuronModel::HodgkinHuxley(_) => {
                NeuronModel::HodgkinHuxley(HodgkinHuxleyParams::default())
            }
            NeuronModel::AdaptiveLif(_) | NeuronModel::Custom(_) => unreachable!(),
        };
        let a = simulate_model(m32, i64_ as f32, n);
        let b = simulate_model(m64, i64_, n);
        assert!(a > 0 && b > 0, "the model is silent: f32={a}, f64={b}");
        assert!(a.abs_diff(b) <= 1, "f32={a}, f64={b}");
    }
}

#[test]
fn topology_works_in_f64_with_dale_and_scaled_migration() {
    let mut t: Topology<f64> = Topology::new(TopologyLimits::default(), ParamLimits::default());
    let a = t
        .add_neuron(
            NeuronRole::Input,
            LifParams::<f64>::default(),
            Origin::Initial,
            b(),
        )
        .unwrap();
    let h = t
        .add_neuron(
            NeuronRole::Hidden,
            LifParams::<f64>::new(0.8, 1.0),
            Origin::Initial,
            b(),
        )
        .unwrap();
    let o = t
        .add_neuron(
            NeuronRole::Output,
            LifParams::<f64>::default(),
            Origin::Initial,
            b(),
        )
        .unwrap();
    t.mutate_neuron(
        h,
        NeuronMutation {
            polarity: Some(NeuronPolarity::Inhibitory),
            ..Default::default()
        },
    )
    .unwrap();
    let s1 = t
        .add_synapse(a, h, -0.123456789012345, 0, Origin::Initial, b())
        .unwrap();
    let s2 = t.add_synapse(h, o, 0.5, 0, Origin::Initial, b()).unwrap();
    // Dale's principle in f64: the sign specifies the source, the modulus is preserved without rounding to f32.
    assert_eq!(t.synapse(s1).unwrap().weight, 0.123456789012345);
    assert_eq!(t.synapse(s2).unwrap().weight, -0.5);
    t.validate().unwrap();
    assert_eq!(t.set_synapse_weight(s2, 1.0).unwrap(), 0.0);
    // Changing the model with unit conversion.
    t.change_neuron_model(
        o,
        NeuronModel::AdEx(AdExParams::<f64>::default()),
        StateMigration::Scaled,
    )
    .unwrap();
    t.validate().unwrap();
    let mut rng = Rng64::new(2);
    let c = t.clone_neuron(h, 0.1, &mut rng, b()).unwrap();
    assert_eq!(t.neuron(c).unwrap().polarity, NeuronPolarity::Inhibitory);
}

#[test]
fn structural_actions_and_batches_run_on_an_f64_network() {
    let mut t: Topology<f64> = Topology::new(TopologyLimits::default(), ParamLimits::default());
    let a = t
        .add_neuron(
            NeuronRole::Input,
            LifParams::<f64>::default(),
            Origin::Initial,
            b(),
        )
        .unwrap();
    let o = t
        .add_neuron(
            NeuronRole::Output,
            LifParams::<f64>::new(0.9, 1.0),
            Origin::Initial,
            b(),
        )
        .unwrap();
    let mut n: DynamicNetwork<f64> =
        DynamicNetwork::new(t, EngineConfig::<f64>::default()).unwrap();
    let mut rng = Rng64::new(11);
    let eff = apply_action(
        &mut n,
        &StructuralAction::AddSynapse {
            src: a,
            dst: o,
            weight: 0.8f64,
            delay: 0,
        },
        &mut rng,
    )
    .unwrap();
    let sid = eff.synapses[0];
    // Split relay neuron and model change - both in f64.
    let split = apply_batch(
        &mut n,
        &[StructuralAction::SplitSynapse {
            id: sid,
            init: SplitInit::<f64>::default(),
        }],
        &mut rng,
    )
    .unwrap();
    let relay = split[0].neurons[0];
    apply_action(
        &mut n,
        &StructuralAction::ChangeNeuronModel {
            id: relay,
            model: NeuronModel::Izhikevich(IzhikevichParams::<f64>::regular_spiking()),
            migration: StateMigration::Scaled,
        },
        &mut rng,
    )
    .unwrap();
    n.topology().validate().unwrap();
    assert_eq!(
        n.topology().neuron(relay).unwrap().model.as_str(),
        "izhikevich"
    );
    // The erroneous packet is rolled back entirely.
    let before = n.topology().clone();
    let bad = apply_batch(
        &mut n,
        &[
            StructuralAction::AddSynapse {
                src: a,
                dst: o,
                weight: 0.1f64,
                delay: 0,
            },
            StructuralAction::RemoveSynapse {
                id: SynapseId(9_999),
            },
        ],
        &mut rng,
    );
    assert!(bad.is_err());
    assert_eq!(n.topology().synapse_count(), before.synapse_count());
    let _: EiStats<f64> = n.topology().ei_stats_as::<f64>();
}

#[test]
fn controller_runs_end_to_end_on_an_f64_network() {
    let mut t: Topology<f64> = Topology::new(TopologyLimits::default(), ParamLimits::default());
    let a = t
        .add_neuron(
            NeuronRole::Input,
            LifParams::<f64>::default(),
            Origin::Initial,
            b(),
        )
        .unwrap();
    let h = t
        .add_neuron(
            NeuronRole::Hidden,
            LifParams::<f64>::new(0.8, 0.5),
            Origin::Initial,
            b(),
        )
        .unwrap();
    let o = t
        .add_neuron(
            NeuronRole::Output,
            LifParams::<f64>::new(0.8, 0.5),
            Origin::Initial,
            b(),
        )
        .unwrap();
    t.add_synapse(a, h, 1.2, 0, Origin::Initial, b()).unwrap();
    t.add_synapse(h, o, 1.2, 0, Origin::Initial, b()).unwrap();
    let mut net: DynamicNetwork<f64> =
        DynamicNetwork::new(t, EngineConfig::<f64>::default()).unwrap();
    let mut cfg = StructuralConfig::<f64>::default();
    cfg.schedule.min_interval_steps = 0;
    cfg.growth.neurons.plateau_patience = 2;
    cfg.growth.neurons.plateau_min_delta = 1e-3;
    cfg.pruning.enabled = false;
    cfg.homeostasis.enabled = false;
    let mut ctl = StructuralPlasticityController::<f64>::new(cfg).unwrap();
    let neurons_before = net.topology().neuron_count();
    let mut eval = |n: &mut DynamicNetwork<f64>| -> f64 {
        let mut spikes = 0.0;
        for _ in 0..20 {
            spikes += n.step(&[1.0]).unwrap().total_spikes as f64;
        }
        1.0 / (1.0 + spikes)
    };
    let mut last: Option<AdaptReport<f64>> = None;
    for _ in 0..8 {
        for _ in 0..50 {
            net.step(&[1.0]).unwrap();
        }
        // Constant loss is a plateau, after which the controller grows the network.
        last = Some(
            ctl.adapt(&mut net, Metrics::<f64>::loss(1.0), Some(&mut eval))
                .unwrap(),
        );
    }
    assert!(last.is_some());
    assert!(
        net.topology().neuron_count() > neurons_before,
        "a plateau should cause growth"
    );
    net.topology().validate().unwrap();
    let st: StructuralStats<f64> = net.stats(1e-3, 0.5);
    assert!(st.mean_rate.is_finite());
    assert!(ctl.cost(1.0, &st).is_finite());
    assert!(ctl.best().is_some());
}

fn f64_net() -> (DynamicNetwork<f64>, StructuralPlasticityController<f64>) {
    let mut t: Topology<f64> = Topology::new(TopologyLimits::default(), ParamLimits::default());
    let a = t
        .add_neuron(
            NeuronRole::Input,
            LifParams::<f64>::default(),
            Origin::Initial,
            b(),
        )
        .unwrap();
    let o = t
        .add_neuron(
            NeuronRole::Output,
            LifParams::<f64>::new(0.8, 0.5),
            Origin::Initial,
            b(),
        )
        .unwrap();
    // weight that doesn't fit into f32 without loss
    t.add_synapse(a, o, 1.000_000_000_123_456_7, 0, Origin::Initial, b())
        .unwrap();
    let mut net = DynamicNetwork::new(t, EngineConfig::<f64>::default()).unwrap();
    let ctl =
        StructuralPlasticityController::<f64>::new(StructuralConfig::<f64>::default()).unwrap();
    for _ in 0..30 {
        net.step(&[1.0]).unwrap();
    }
    (net, ctl)
}

#[test]
fn f64_checkpoint_roundtrips_without_precision_loss_and_uses_version_6() {
    let (net, ctl) = f64_net();
    let bytes = save_checkpoint(&net, &ctl, b"extra").unwrap();
    assert_eq!(&bytes[4..8], &FORMAT_VERSION_F64.to_le_bytes());
    let back = load_checkpoint_as::<f64>(&bytes).unwrap();
    assert_eq!(back.extra, b"extra");
    let w0 = net.topology().synapses().next().unwrap().weight;
    let w1 = back.network.topology().synapses().next().unwrap().weight;
    assert_eq!(w0, w1, "f64 lossless");
    assert_eq!(back.network.topology(), net.topology());
    // saving again gives the same bytes
    assert_eq!(
        save_checkpoint(&back.network, &back.controller, b"extra").unwrap(),
        bytes
    );
    // f64 → f32 read with rounding
    let narrow = load_checkpoint(&bytes).unwrap();
    let wn = narrow.network.topology().synapses().next().unwrap().weight;
    assert_eq!(wn, w0 as f32);
    // trimming and spoilage are still caught
    assert!(load_checkpoint_as::<f64>(&bytes[..bytes.len() - 9]).is_err());
}

#[test]
fn f32_checkpoint_stays_current_version_and_widens_losslessly_to_f64() {
    let mut t: Topology = Topology::new(TopologyLimits::default(), ParamLimits::default());
    let a = t
        .add_neuron(
            NeuronRole::Input,
            LifParams::default(),
            Origin::Initial,
            b(),
        )
        .unwrap();
    let o = t
        .add_neuron(
            NeuronRole::Output,
            LifParams::default(),
            Origin::Initial,
            b(),
        )
        .unwrap();
    t.add_synapse(a, o, 0.75, 0, Origin::Initial, b()).unwrap();
    let net = DynamicNetwork::new(t, EngineConfig::default()).unwrap();
    let ctl = StructuralPlasticityController::new(StructuralConfig::default()).unwrap();
    let bytes = save_checkpoint(&net, &ctl, b"").unwrap();
    assert_eq!(&bytes[4..8], &FORMAT_VERSION.to_le_bytes());
    let wide = load_checkpoint_as::<f64>(&bytes).unwrap();
    assert_eq!(
        wide.network.topology().synapses().next().unwrap().weight,
        0.75f64
    );
}

// ---------------------------------------------------------------------
// Short-term plasticity (STP, Tsodyks–Markram)
// ---------------------------------------------------------------------

fn stp_net(stp: Option<StpConfig>) -> DynamicNetwork {
    let mut t = topo(TopologyLimits::default());
    let i = inp(&mut t);
    let o = out(&mut t, 0.9, 0.4);
    t.add_synapse(i, o, 0.5, 0, Origin::Initial, b()).unwrap();
    let mut cfg = EngineConfig::default();
    cfg.stp = stp;
    DynamicNetwork::new(t, cfg).unwrap()
}

#[test]
fn stp_state_facilitation_close_spikes_release_higher() {
    // Facilitation preset: τ_fac = 50 (slow utilization revert), τ_rec = 5.
    let cfg = StpConfig::preset(StpMode::Facilitation);
    let mut st = StpState::new(cfg);
    let rel: Vec<f32> = (0..3).map(|_| st.release(cfg)).collect();
    assert!(rel[1] > rel[0] + 1e-3, "facilitation: {rel:?}");
    assert!(rel[2] > rel[1] + 1e-3, "facilitation: {rel:?}");
}

#[test]
fn stp_state_depression_burst_depletes_release() {
    // Depression preset: τ_rec = 50 (slow recovery), τ_fac = 5.
    let cfg = StpConfig::preset(StpMode::Depression);
    let mut st = StpState::new(cfg);
    let rel: Vec<f32> = (0..3).map(|_| st.release(cfg)).collect();
    assert!(rel[1] < rel[0] - 1e-3, "depression: {rel:?}");
    assert!(rel[2] < rel[1] - 1e-3, "depression: {rel:?}");
}

#[test]
fn stp_state_baseline_return_after_pause() {
    let cfg: StpConfig = StpConfig::preset(StpMode::Depression);
    let mut st = StpState::new(cfg);
    st.release(cfg); // deplete x.
    for _ in 0..100 {
        st.update_between(1000, cfg);
    } // long pause
    assert!(st.x.abs() > 0.99, "x recovered toward 1: {}", st.x);
    assert!((st.u - cfg.u).abs() < 0.01, "u reverts toward U: {}", st.u);
    let y: f32 = st.release(cfg);
    assert!(
        (y - cfg.u).abs() < 1e-3,
        "release returns to baseline U: {y}"
    );
}

#[test]
fn stp_telemetry_accessors() {
    let mut net = stp_net(Some(StpConfig::preset(StpMode::Mixed)));
    assert_eq!(net.stp_config(), Some(StpConfig::preset(StpMode::Mixed)));
    net.step(&[1.0]).unwrap();
    net.step(&[1.0]).unwrap(); // first arrival happens on step 2.
    let sid = net.topology().synapses().next().unwrap().id;
    assert!(net.stp_state(sid).is_some());
    assert!(net.stp_states().contains_key(&sid));
    let mut net2 = stp_net(None);
    assert_eq!(net2.stp_config(), None);
    net2.set_stp(Some(StpConfig::preset(StpMode::Mixed)));
    assert_eq!(net2.stp_config(), Some(StpConfig::preset(StpMode::Mixed)));
    net2.set_stp(None);
    assert_eq!(net2.stp_config(), None);
}

#[test]
fn stp_static_mode_is_legacy_parity() {
    // `StpMode::Static` ⇒ y ≡ 1: identical delivery to a network without STP.
    let mut a = stp_net(Some(StpConfig::preset(StpMode::Static)));
    let mut b = stp_net(None);
    for _ in 0..5 {
        a.step(&[1.0]).unwrap();
        b.step(&[1.0]).unwrap();
    }
    assert_eq!(
        a.topology().synapses().next().unwrap().weight,
        b.topology().synapses().next().unwrap().weight
    );
    assert!(a.stp_states().is_empty());
}

#[test]
fn stp_composes_with_pair_stdp() {
    let mut t = topo(TopologyLimits::default());
    let i = inp(&mut t);
    let o = out(&mut t, 0.9, 0.4);
    t.add_synapse(i, o, 0.5, 0, Origin::Initial, b()).unwrap();
    let mut cfg = EngineConfig::default();
    cfg.stdp.mode = PlasticityMode::Immediate;
    // U = 1.0: release y ≡ 1 (no STP depression), delivery = 0.5·1 = 0.5.
    cfg.stp = Some(StpConfig::new(StpMode::Mixed, 1.0, 10.0, 10.0));
    let mut net = DynamicNetwork::new(t, cfg).unwrap();
    for _ in 0..5 {
        net.step(&[10.0]).unwrap();
    }
    assert_ne!(net.topology().synapses().next().unwrap().weight, 0.5);
    assert!(!net.stp_states().is_empty());
}

#[test]
fn stp_checkpoint_roundtrip() {
    let mut net = stp_net(Some(StpConfig::preset(StpMode::Mixed)));
    for _ in 0..3 {
        net.step(&[1.0]).unwrap();
    }
    let sid = net.topology().synapses().next().unwrap().id;
    let (x0, u0, rel0) = {
        let st = net.stp_state(sid).unwrap();
        (st.x, st.u, st.released)
    };
    assert!(rel0 > 0.0);
    let ctl = StructuralPlasticityController::new(StructuralConfig::default()).unwrap();
    let bytes = save_checkpoint(&net, &ctl, b"").unwrap();
    let c = load_checkpoint(&bytes).unwrap();
    let st = c.network.stp_state(sid).unwrap();
    assert_eq!((st.x, st.u), (x0, u0));
    assert_eq!(st.released, rel0);
}

#[test]
fn stp_set_none_disables_and_clears() {
    let mut net = stp_net(Some(StpConfig::preset(StpMode::Mixed)));
    net.step(&[1.0]).unwrap();
    net.step(&[1.0]).unwrap(); // arrival on step 2.
    assert!(!net.stp_states().is_empty());
    net.set_stp(None);
    assert!(net.stp_states().is_empty());
    assert_eq!(net.stp_config(), None);
}

// ---------------------------------------------------------------------
// Conductance-based synapses (AMPA/NMDA/GABA) — PROMPT 3
// ---------------------------------------------------------------------

fn g_of(net: &DynamicNetwork, sid: SynapseId) -> f32 {
    net.cond_states().get(&sid).map(|s| s.g).unwrap_or(0.0)
}

#[test]
fn amp_pulse_decays_with_tau() {
    let mut t = topo(TopologyLimits::default());
    let i = inp(&mut t);
    let o = out(&mut t, 0.9, 0.4);
    t.add_synapse_with_model(i, o, 0.5, 0, SynapseModel::Ampa, Origin::Initial, b())
        .unwrap();
    let mut net = DynamicNetwork::new(t, EngineConfig::default()).unwrap();
    for _ in 0..5 {
        net.step(&[1.0]).unwrap();
    }
    let sid = net.topology().synapses().next().unwrap().id;
    let g5 = g_of(&net, sid);
    assert!(g5 > 0.0, "AMPA conductance after spikes: {g5}");
    for _ in 0..10 {
        net.step(&[0.0]).unwrap();
    }
    let g_last = g_of(&net, sid);
    assert!(
        g_last < g5,
        "AMPA conductance decays between steps: {g5} -> {g_last}"
    );
}

#[test]
fn nmda_block_changes_with_voltage() {
    let p = ConductanceParams::<f32>::nmda();
    let b0 = p.nmda_block(0.0);
    let b_half = p.nmda_block(-0.4);
    let b_deep = p.nmda_block(-0.8);
    assert!(b0 > b_half, "block opens at V=0: {b0} vs {b_half}");
    assert!(
        b_half > b_deep,
        "block closes at V=-0.8: {b_half} vs {b_deep}"
    );
    assert!(b_deep < 0.01, "block nearly closed at V=-0.8: {b_deep}");
}

#[test]
fn reversal_potential_direction_vs_voltage() {
    let p = ConductanceParams::<f32>::amp();
    assert_eq!(p.current(0.5, 0.0), 0.0);
    assert!(p.current(0.5, 0.5) < 0.0);
    assert!(p.current(0.5, -0.5) > 0.0);
}

#[test]
fn amp_nmda_independent_states() {
    let mut t = topo(TopologyLimits::default());
    let i = inp(&mut t);
    let o = out(&mut t, 0.9, 0.4);
    t.add_synapse_with_model(i, o, 0.5, 0, SynapseModel::Ampa, Origin::Initial, b())
        .unwrap();
    t.add_synapse_with_model(i, o, 0.5, 0, SynapseModel::Nmda, Origin::Initial, b())
        .unwrap();
    let mut net = DynamicNetwork::new(t, EngineConfig::default()).unwrap();
    for _ in 0..5 {
        net.step(&[1.0]).unwrap();
    }
    assert_eq!(
        net.cond_states().len(),
        2,
        "AMPA + NMDA multiedge has two independent states"
    );
    assert!(net.topology().validate().is_ok());
}

#[test]
fn gaba_reduces_drive_and_dale_invariant_holds() {
    // GABA multiedge adds negative current (I = g·(E_rev − V), E_rev < 0) on top of a
    // legacy +0.5 current-based synapse — the target drive must decrease.
    let drive = |with_gaba: bool| -> f32 {
        let mut t = topo(TopologyLimits::default());
        let i = inp(&mut t);
        // theta high: no output spikes anywhere (compare membrane levels).
        let o = out(&mut t, 0.9, 1.0);
        t.add_synapse(i, o, 0.5, 0, Origin::Initial, b()).unwrap(); // legacy: +0.5 current
        if with_gaba {
            t.add_synapse_with_model(i, o, 0.5, 0, SynapseModel::Gaba, Origin::Initial, b())
                .unwrap();
        }
        let mut net = DynamicNetwork::new(t, EngineConfig::default()).unwrap();
        net.step(&[1.0]).unwrap();
        net.step(&[1.0]).unwrap();
        net.topology().neuron(o).unwrap().state.mem()
    };
    // drive(false) = 0.5 (legacy), drive(true) = 0.5 + g·(E_rev − V) = 0.5 − 0.21.
    assert!(
        drive(true) < drive(false),
        "GABA must reduce the target drive: {} vs {}",
        drive(true),
        drive(false)
    );
    assert!(ConductanceParams::<f32>::gaba().current(0.3, 0.0) < 0.0);
}

#[test]
fn current_based_unchanged_parity() {
    // Legacy current-based delivery: weight is the current — unchanged behavior.
    let mut t = topo(TopologyLimits::default());
    let i = inp(&mut t);
    let o = out(&mut t, 0.9, 0.4);
    let sid = t.add_synapse(i, o, 0.5, 0, Origin::Initial, b()).unwrap();
    let mut net = DynamicNetwork::new(t, EngineConfig::default()).unwrap();
    for _ in 0..5 {
        net.step(&[1.0]).unwrap();
    }
    assert_eq!(
        net.topology().synapse(sid).unwrap().model,
        SynapseModel::CurrentBased
    );
    assert!(!net.cond_states().contains_key(&sid));
}

#[test]
fn conductance_f32_f64() {
    let build = |t: &mut Topology| {
        let i = inp(t);
        let o = out(t, 0.9, 0.4);
        t.add_synapse_with_model(i, o, 0.5, 0, SynapseModel::Ampa, Origin::Initial, b())
            .unwrap();
    };
    let mut t = topo(TopologyLimits::default());
    build(&mut t);
    let mut net = DynamicNetwork::new(t, EngineConfig::default()).unwrap();
    net.step(&[1.0]).unwrap();
    net.step(&[1.0]).unwrap(); // first arrival happens on step 2
    assert_eq!(net.cond_states().len(), 1);
    // f64 variant.
    let mut t = Topology::<f64>::new(TopologyLimits::default(), ParamLimits::default());
    let i = t
        .add_neuron(
            NeuronKind::Input,
            LifParams::default(),
            Origin::Initial,
            b(),
        )
        .unwrap();
    let o = t
        .add_neuron(
            NeuronKind::Output,
            LifParams::new(0.9, 0.4),
            Origin::Initial,
            b(),
        )
        .unwrap();
    t.add_synapse_with_model(i, o, 0.5, 0, SynapseModel::Ampa, Origin::Initial, b())
        .unwrap();
    let mut net = DynamicNetwork::new(t, EngineConfig::<f64>::default()).unwrap();
    net.step(&[1.0]).unwrap();
    net.step(&[1.0]).unwrap();
    let sid = net.topology().synapses().next().unwrap().id;
    assert_eq!(
        net.topology().synapse(sid).unwrap().model,
        SynapseModel::Ampa
    );
    assert!(net.cond_states().contains_key(&sid));
}

#[test]
fn conductance_numeric_param_validation() {
    assert!(ConductanceParams::<f32>::amp().validate().is_ok());
    assert!(ConductanceParams::<f32>::nmda().validate().is_ok());
    assert!(ConductanceParams::<f32>::gaba().validate().is_ok());
    assert!(
        ConductanceParams::<f32> {
            g_peak: -0.1,
            ..ConductanceParams::<f32>::amp()
        }
        .validate()
        .is_err()
    );
    assert!(
        ConductanceParams::<f32> {
            tau: 0.0,
            ..ConductanceParams::<f32>::amp()
        }
        .validate()
        .is_err()
    );
    assert!(
        ConductanceParams::<f32> {
            e_rev: f32::NAN,
            ..ConductanceParams::<f32>::amp()
        }
        .validate()
        .is_err()
    );
    assert!(
        ConductanceParams::<f32> {
            delta_v: 0.0,
            ..ConductanceParams::<f32>::nmda()
        }
        .validate()
        .is_err()
    );
}

#[test]
fn conductance_checkpoint_roundtrip() {
    let mut t = topo(TopologyLimits::default());
    let i = inp(&mut t);
    let o = out(&mut t, 0.9, 0.4);
    let sid = t
        .add_synapse_with_model(i, o, 0.5, 0, SynapseModel::Ampa, Origin::Initial, b())
        .unwrap();
    let mut net = DynamicNetwork::new(t, EngineConfig::default()).unwrap();
    for _ in 0..3 {
        net.step(&[1.0]).unwrap();
    }
    let g0 = g_of(&net, sid);
    assert!(g0 > 0.0);
    let ctl = StructuralPlasticityController::new(StructuralConfig::default()).unwrap();
    let bytes = save_checkpoint(&net, &ctl, b"").unwrap();
    let c = load_checkpoint(&bytes).unwrap();
    assert_eq!(
        c.network.topology().synapse(sid).unwrap().model,
        SynapseModel::Ampa
    );
    assert_eq!(g_of(&c.network, sid), g0);
}

#[test]
fn conductance_topology_mutations() {
    // Remove → state gone.
    let mut t = topo(TopologyLimits::default());
    let i = inp(&mut t);
    let o = out(&mut t, 0.9, 0.4);
    let sid = t
        .add_synapse_with_model(i, o, 0.5, 0, SynapseModel::Ampa, Origin::Initial, b())
        .unwrap();
    let mut net = DynamicNetwork::new(t, EngineConfig::default()).unwrap();
    for _ in 0..3 {
        net.step(&[1.0]).unwrap();
    }
    assert!(net.cond_states().contains_key(&sid));
    net.topology_mut().remove_synapse(sid).unwrap();
    net.step(&[1.0]).unwrap(); // cond state pruned on the next step
    assert!(!net.cond_states().contains_key(&sid));
    // Disable → state pruned on the next step.
    let mut t = topo(TopologyLimits::default());
    let i = inp(&mut t);
    let o = out(&mut t, 0.9, 0.4);
    let sid = t
        .add_synapse_with_model(i, o, 0.5, 0, SynapseModel::Ampa, Origin::Initial, b())
        .unwrap();
    let mut net = DynamicNetwork::new(t, EngineConfig::default()).unwrap();
    for _ in 0..3 {
        net.step(&[1.0]).unwrap();
    }
    assert!(net.cond_states().contains_key(&sid));
    net.topology_mut().disable_synapse(sid, 100).unwrap();
    net.step(&[1.0]).unwrap();
    assert!(!net.cond_states().contains_key(&sid));
}

// ---------------------------------------------------------------------
// E-prop on DynamicNetwork (PROMPT 1 — verification/docs)
// ---------------------------------------------------------------------

#[test]
fn eprop_on_dynamic_network_trains_readout() {
    // Input→Output LIF: `EpropTrainer` drives online weight updates
    // (`W ← W − η·d·e`) after each `net.step()` — no backprop through time.
    let mut t = topo(TopologyLimits::default());
    let i = inp(&mut t);
    let o = out(&mut t, 0.9, 0.4);
    let sid = t.add_synapse(i, o, 0.4, 0, Origin::Initial, b()).unwrap();
    let mut net = DynamicNetwork::new(t, EngineConfig::default()).unwrap();
    let mut cfg = EpropConfig::<f32>::default();
    cfg.learning_rate = 0.1;
    let mut trainer = EpropTrainer::new(&net, cfg).unwrap();
    trainer.set_target(o, 1.0);
    for _ in 0..12 {
        net.step(&[1.0]).unwrap();
        trainer.step(&mut net).unwrap();
    }
    assert_eq!(trainer.state().steps(), 12);
    assert!(trainer.state().synapse_trace_count() > 0);
    assert!(net.topology().synapse(sid).is_some());
}

// ---------------------------------------------------------------------
// Learnable delays (PROMPT 5)
// ---------------------------------------------------------------------

fn ld_net(k: u16) -> (DynamicNetwork, SynapseId) {
    let mut t = topo(TopologyLimits::default());
    let i = inp(&mut t);
    let o = out(&mut t, 0.9, 0.4);
    let sid = t.add_synapse(i, o, 0.5, k, Origin::Initial, b()).unwrap();
    (
        DynamicNetwork::new(t, EngineConfig::default()).unwrap(),
        sid,
    )
}

fn ld_net_cfg(cfg: LearnableDelayConfig) -> DynamicNetwork {
    let mut t = topo(TopologyLimits::default());
    let i = inp(&mut t);
    let o = out(&mut t, 0.9, 0.4);
    t.add_synapse(i, o, 0.5, 0, Origin::Initial, b()).unwrap();
    let mut ec = EngineConfig::default();
    ec.learnable_delay = Some(cfg);
    DynamicNetwork::new(t, ec).unwrap()
}

fn ld_cfg(initial: f32, lr: f32, max: f32) -> LearnableDelayConfig {
    LearnableDelayConfig {
        enabled: true,
        min_delay: 0.0,
        max_delay: max,
        initial,
        learning_rate: lr,
        elig_decay: 0.9,
    }
}

/// Arrival step of the first spike on `sid` (scan up to 8 steps).
fn arrival_step(net: &mut DynamicNetwork, sid: SynapseId) -> u64 {
    for _ in 0..8 {
        net.step(&[1.0]).unwrap();
        if net.last_arrivals().contains(&sid) {
            return net.step_count();
        }
    }
    panic!("no arrival within 8 steps");
}

#[test]
fn learnable_delay_integer_parity_with_legacy() {
    for k in 0..=2u16 {
        let (mut net, sid) = ld_net(k);
        let legacy = arrival_step(&mut net, sid);
        let mut net2 = ld_net_cfg(ld_cfg(k as f32, 0.0, 4.0));
        let sid2 = net2.topology().synapses().next().unwrap().id;
        let learnable = arrival_step(&mut net2, sid2);
        assert_eq!(
            legacy, learnable,
            "integer parity for delay {k}: legacy={legacy} learnable={learnable}"
        );
    }
}

#[test]
fn learnable_delay_fractional_split() {
    let mut net = ld_net_cfg(ld_cfg(1.5, 0.0, 3.0));
    let sid = net.topology().synapses().next().unwrap().id;
    net.step(&[1.0]).unwrap(); // spike of the input during step 1
    net.step(&[1.0]).unwrap(); // dispatch pass
    assert_eq!(net.learnable_delay(sid), Some(1.5));
    let mut due: Vec<u64> = net.frac_deliveries().keys().copied().collect();
    due.sort_unstable();
    assert_eq!(due.len(), 2, "two bins: {due:?}");
    assert_eq!(due[1] - due[0], 1, "adjacent bins: {due:?}");
    let amt: f32 = net
        .frac_deliveries()
        .values()
        .flatten()
        .map(|(_, a)| *a)
        .sum();
    assert!((amt - 1.0).abs() < 1e-6, "charge preserved: {amt}");
    for _ in 0..3 {
        net.step(&[1.0]).unwrap();
    }
    assert!(net.last_arrivals().contains(&sid));
}

#[test]
fn learnable_delay_training_moves_delay() {
    // Positive teaching signal → delay grows.
    let mut net = ld_net_cfg(ld_cfg(0.5, 0.1, 3.0));
    let sid = net.topology().synapses().next().unwrap().id;
    let o = net.topology().synapses().next().unwrap().dst;
    net.set_teaching_signal(o, 1.0);
    for _ in 0..5 {
        net.step(&[1.0]).unwrap();
    }
    assert!(net.learnable_delay(sid).unwrap() > 0.5 + 1e-6);
    // Negative → shrinks toward min_delay.
    let mut net = ld_net_cfg(ld_cfg(0.5, 0.1, 3.0));
    let sid = net.topology().synapses().next().unwrap().id;
    let o = net.topology().synapses().next().unwrap().dst;
    net.set_teaching_signal(o, -1.0);
    for _ in 0..5 {
        net.step(&[1.0]).unwrap();
    }
    assert!(net.learnable_delay(sid).unwrap() < 0.5 - 1e-6);
    // Zero signal → unchanged.
    let mut net = ld_net_cfg(ld_cfg(0.5, 0.1, 3.0));
    let sid = net.topology().synapses().next().unwrap().id;
    for _ in 0..5 {
        net.step(&[1.0]).unwrap();
    }
    assert!((net.learnable_delay(sid).unwrap() - 0.5).abs() < 1e-6);
}

#[test]
fn learnable_delay_checkpoint_roundtrip() {
    let mut net = ld_net_cfg(ld_cfg(0.75, 0.1, 3.0));
    let sid = net.topology().synapses().next().unwrap().id;
    let o = net.topology().synapses().next().unwrap().dst;
    net.set_teaching_signal(o, 0.5);
    for _ in 0..3 {
        net.step(&[1.0]).unwrap();
    }
    let d0 = net.learnable_delay(sid).unwrap();
    assert!(d0 > 0.75);
    let ctl = StructuralPlasticityController::new(StructuralConfig::default()).unwrap();
    let bytes = save_checkpoint(&net, &ctl, b"").unwrap();
    let c = load_checkpoint(&bytes).unwrap();
    assert_eq!(c.network.learnable_delay(sid), Some(d0));
    assert_eq!(
        c.network.learnable_delay_elig(sid),
        net.learnable_delay_elig(sid)
    );
    assert!(c.network.config().learnable_delay.is_some());
    // Legacy (no learnable) — empty and roundtrips as `None`.
    let (mut net2, sid2) = ld_net(1);
    for _ in 0..2 {
        net2.step(&[1.0]).unwrap();
    }
    assert!(net2.learnable_delays().is_empty());
    let ctl2 = StructuralPlasticityController::new(StructuralConfig::default()).unwrap();
    let bytes2 = save_checkpoint(&net2, &ctl2, b"").unwrap();
    let c2 = load_checkpoint(&bytes2).unwrap();
    assert_eq!(c2.network.learnable_delay(sid2), None);
    assert!(c2.network.config().learnable_delay.is_none());
}

#[test]
fn learnable_delay_config_validation() {
    let mut cfg = LearnableDelayConfig::default();
    assert!(cfg.validate().is_ok());
    cfg.min_delay = -0.1;
    assert!(cfg.validate().is_err());
    cfg.min_delay = 0.0;
    cfg.max_delay = -1.0;
    assert!(cfg.validate().is_err());
    cfg.max_delay = 3.0;
    cfg.initial = 5.0;
    assert!(cfg.validate().is_err());
    cfg.initial = 1.5;
    cfg.learning_rate = f32::NAN;
    assert!(cfg.validate().is_err());
    cfg.learning_rate = 0.05;
    cfg.elig_decay = 1.0;
    assert!(cfg.validate().is_err());
    cfg.elig_decay = 0.9;
    assert!(cfg.validate().is_ok());
}

#[test]
fn active_input_list_records_spikes() {
    // Active-input lists — the canonical sparse input lists of the engine.
    let mut t = topo(TopologyLimits::default());
    let i = inp(&mut t);
    let o = out(&mut t, 0.9, 0.4);
    t.add_synapse(i, o, 0.5, 0, Origin::Initial, b()).unwrap();
    let mut net = DynamicNetwork::new(t, EngineConfig::default()).unwrap();
    for _ in 0..4 {
        net.step(&[1.0]).unwrap();
    }
    // The input spiked (fed 1.0 each step) — it is in the last active-input list.
    let last = net.last_active_inputs();
    assert!(last.contains(&i), "input in last_active_inputs: {last:?}");
    // The window is bounded (like `history`) and time-ordered.
    assert!(net.active_inputs().len() <= net.topology().param_limits().delay_max as usize + 1);
    let steps: Vec<u64> = net.active_inputs().iter().map(|&(s, _)| s).collect();
    assert!(
        steps.windows(2).all(|w| w[0] <= w[1]),
        "time-ordered: {steps:?}"
    );
    // Empty when nothing spiked.
    let mut t2 = topo(TopologyLimits::default());
    let i2 = inp(&mut t2);
    let o2 = out(&mut t2, 0.9, 0.4);
    t2.add_synapse(i2, o2, 0.5, 0, Origin::Initial, b())
        .unwrap();
    let mut net2 = DynamicNetwork::new(t2, EngineConfig::default()).unwrap();
    for _ in 0..3 {
        net2.step(&[0.0]).unwrap();
    }
    assert!(net2.last_active_inputs().is_empty());
}

#[test]
fn synapse_binary_flag_and_setter() {
    let mut t = topo(TopologyLimits::default());
    let i = inp(&mut t);
    let o = out(&mut t, 0.9, 0.4);
    let sid = t.add_synapse(i, o, 0.5, 0, Origin::Initial, b()).unwrap();
    assert!(!t.synapse(sid).unwrap().binary);
    t.set_synapse_binary(sid, true).unwrap();
    assert!(t.synapse(sid).unwrap().binary);
    t.set_synapse_binary(sid, false).unwrap();
    assert!(!t.synapse(sid).unwrap().binary);
    assert!(t.set_synapse_binary(SynapseId(999), true).is_err());
}

#[test]
fn count_scores_counts_active_inputs() {
    let mut t = topo(TopologyLimits::default());
    let i = inp(&mut t);
    let o = out(&mut t, 0.9, 0.4);
    t.add_synapse(i, o, 0.5, 0, Origin::Initial, b()).unwrap();
    let net = DynamicNetwork::new(t, EngineConfig::default()).unwrap();
    // Paper Eq. `c_j = Σ_{i∈Q} M_ij`: active input `i` → 1 counted synapse of `o`.
    let scores = net.count_scores(&[i]);
    assert_eq!(scores.get(&o), Some(&1u32));
    assert!(net.count_scores(&[]).is_empty());
}

#[test]
fn count_scores_skips_zero_weight_and_inactive() {
    let mut t = topo(TopologyLimits::default());
    let i = inp(&mut t);
    let o = out(&mut t, 0.9, 0.4);
    let sid = t.add_synapse(i, o, 0.5, 0, Origin::Initial, b()).unwrap();
    let mut net = DynamicNetwork::new(t, EngineConfig::default()).unwrap();
    assert_eq!(net.count_scores(&[i]).get(&o), Some(&1u32));
    // Zero weight → M_ij = 0.
    net.topology_mut().set_synapse_weight(sid, 0.0).unwrap();
    assert!(net.count_scores(&[i]).get(&o).is_none());
    // Disabled → not counted.
    net.topology_mut().set_synapse_weight(sid, 0.5).unwrap();
    net.topology_mut().disable_synapse(sid, 100).unwrap();
    net.step(&[1.0]).unwrap();
    assert!(net.count_scores(&[i]).get(&o).is_none());
}

#[test]
fn binary_flag_checkpoint_roundtrip() {
    let mut t = topo(TopologyLimits::default());
    let i = inp(&mut t);
    let o = out(&mut t, 0.9, 0.4);
    let sid = t.add_synapse(i, o, 0.5, 0, Origin::Initial, b()).unwrap();
    t.set_synapse_binary(sid, true).unwrap();
    let net = DynamicNetwork::new(t, EngineConfig::default()).unwrap();
    let ctl = StructuralPlasticityController::new(StructuralConfig::default()).unwrap();
    let bytes = save_checkpoint(&net, &ctl, b"").unwrap();
    let c = load_checkpoint(&bytes).unwrap();
    assert!(c.network.topology().synapse(sid).unwrap().binary);
    // Legacy: no binary flags — roundtrips as `false`.
    let mut t2 = topo(TopologyLimits::default());
    let i2 = inp(&mut t2);
    let o2 = out(&mut t2, 0.9, 0.4);
    let sid2 = t2
        .add_synapse(i2, o2, 0.5, 0, Origin::Initial, b())
        .unwrap();
    let net2 = DynamicNetwork::new(t2, EngineConfig::default()).unwrap();
    let bytes2 = save_checkpoint(&net2, &ctl, b"").unwrap();
    let c2 = load_checkpoint(&bytes2).unwrap();
    assert!(!c2.network.topology().synapse(sid2).unwrap().binary);
}
