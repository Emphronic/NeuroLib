//! Integration tests of structural plasticity on a small deterministic
//! problem (the same as in the example `dynamic_snn`).
//!
//! Problem: 6 inputs, 2 outputs; class A - spikes on inputs 0–1, class B - on 2–3,
//! inputs 4–5 - noise. The weight of the synapse is limited (`WeightBound::Hard 0.15`), so
//! one input per class is not enough; the initial network contains one
//! connections per class and one hidden neuron without inputs. STDP alone cannot solve the problem.
//!
//! The tests do NOT claim that structural plasticity improves any task:
//! It is this specially built one that is being tested.
#![allow(clippy::field_reassign_with_default)]

use neo_snn::functional::plasticity::WeightBound;
use neo_snn::structural::*;

const INPUTS: usize = 6;
const STEPS: usize = 20;
const TRAIN_PER_EPOCH: usize = 20;

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
    let (mut loss, mut correct, n) = (0.0f32, 0usize, 20usize);
    for i in 0..n {
        let class = i % 2;
        p.begin_sample();
        let mut count = [0u32; 2];
        for x in sample(900_000 + i as u64, class) {
            let o = p.step(&x).expect("evaluation step");
            for (k, s) in o.output_spikes.iter().enumerate().take(2) {
                count[k] += *s as u32;
            }
        }
        let margin = count[class] as f32 - count[1 - class] as f32;
        loss += (1.0 - margin / 2.0).clamp(0.0, 2.0);
        correct += (margin > 0.0) as usize;
    }
    (loss / n as f32, correct as f32 / n as f32)
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
    c.homeostasis.enabled = false;
    c
}

fn build(cfg: &StructuralConfig) -> (DynamicNetwork, NeuronId) {
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
    t.add_synapse(inputs[0], o0, 0.10, 0, Origin::Initial, b)
        .unwrap();
    t.add_synapse(inputs[2], o1, 0.10, 0, Origin::Initial, b)
        .unwrap();
    t.add_synapse(inputs[4], o0, 0.05, 0, Origin::Initial, b)
        .unwrap();
    t.add_synapse(inputs[5], o1, 0.05, 0, Origin::Initial, b)
        .unwrap();
    t.add_synapse(dead, o0, 0.3, 0, Origin::Initial, b).unwrap();
    let mut ec = EngineConfig::default();
    ec.stdp.mode = PlasticityMode::Immediate;
    ec.stdp.a_plus = 0.0005;
    ec.stdp.a_minus = 0.0003;
    ec.stdp.bound = WeightBound::Hard {
        w_min: -0.15,
        w_max: 0.15,
    };
    (DynamicNetwork::new(t, ec).unwrap(), dead)
}

fn train_epoch(net: &mut DynamicNetwork, epoch: usize) {
    let outs = net.topology().outputs();
    for i in 0..TRAIN_PER_EPOCH {
        let class = i % 2;
        net.begin_sample();
        for x in sample((epoch * 1000 + i + 1) as u64, class) {
            net.step_with_current(&x, &[(outs[class], 1.5)])
                .expect("learning step");
        }
    }
}

/// One era: training, assessment, structural solution.
fn epoch(
    net: &mut DynamicNetwork,
    ctl: &mut StructuralPlasticityController,
    e: usize,
) -> (f32, AdaptReport) {
    train_epoch(net, e);
    let (loss, acc) = evaluate(net);
    let mut eval = |p: &mut DynamicNetwork| evaluate(p).0;
    let r = ctl
        .adapt(
            net,
            Metrics {
                task_loss: loss,
                reward: None,
                accuracy: Some(acc),
            },
            Some(&mut eval),
        )
        .expect("adapt");
    (loss, r)
}

#[test]
fn stdp_alone_cannot_solve_the_task() {
    // Control: the same learning without structural plasticity remains poor.
    let cfg = config();
    let (mut net, _) = build(&cfg);
    for e in 0..8 {
        train_epoch(&mut net, e);
    }
    let (loss, _) = evaluate(&net);
    assert!(loss > 0.6, "without structure loss = {loss}");
}

#[test]
fn structure_fixes_what_stdp_alone_cannot() {
    let cfg = config();
    let (mut net, dead) = build(&cfg);
    let mut ctl = StructuralPlasticityController::new(cfg.clone()).unwrap();
    let (start_loss, _) = evaluate(&net);
    let (n0, s0) = (
        net.topology().neuron_count(),
        net.topology().synapse_count(),
    );

    let mut first_change: Option<(usize, f32)> = None;
    let mut losses = Vec::new();
    for e in 0..8 {
        let (loss, r) = epoch(&mut net, &mut ctl, e);
        losses.push(loss);
        if first_change.is_none() && r.applied > 0 {
            first_change = Some((e, loss));
        }
        // Limits and invariants are respected at each boundary.
        net.topology().validate().expect("invariants");
        assert!(net.topology().neuron_count() <= cfg.topology.max_neurons);
        assert!(net.topology().synapse_count() <= cfg.topology.max_synapses);
    }
    let (end_loss, end_acc) = evaluate(&net);

    // 1–2. The network trained, the controller received metrics and made decisions.
    assert!(!ctl.log().is_empty(), "the log is empty");
    // 3. The topology has really changed.
    let totals = ctl.log().totals();
    let applied = |k: ActionKind| totals.get(&k).map_or(0, |c| c.applied);
    assert!(
        applied(ActionKind::AddSynapse) >= 2,
        "connections have not grown: {totals:?}"
    );
    assert!(
        applied(ActionKind::RemoveNeuron) >= 1,
        "dead neuron is not removed"
    );
    assert!(net.topology().neuron(dead).is_none());
    assert!(net.topology().neuron_count() < n0);
    assert!(net.topology().synapse_count() > s0 - 1);
    // 4. Each change is recorded with a reason and effect.
    for ev in ctl.log().iter() {
        assert!(!ev.reason.is_empty() && !ev.action.is_empty());
    }
    // 5. Learning continues after changes: after the first change, loss drops.
    let (e_change, loss_at_change) = first_change.expect("there were no changes");
    let after = losses[(e_change + 1)..]
        .iter()
        .cloned()
        .fold(f32::INFINITY, f32::min);
    assert!(
        after < loss_at_change - 0.3,
        "after changing loss {after}, there was {loss_at_change}"
    );
    // 7–8. The result is noticeably better than both the starting network and control without structure.
    assert!(
        end_loss < 0.2 && end_acc >= 0.95,
        "total: loss {end_loss}, acc {end_acc}"
    );
    assert!(end_loss < start_loss - 0.5);
}

#[test]
fn checkpoint_restores_structure_and_resumes_identically() {
    let cfg = config();
    let (mut net, _) = build(&cfg);
    let mut ctl = StructuralPlasticityController::new(cfg).unwrap();
    for e in 0..3 {
        epoch(&mut net, &mut ctl, e);
    }
    let bytes = save_checkpoint(&net, &ctl, b"extra-optimizer-state");
    let Checkpoint {
        network: mut net2,
        controller: mut ctl2,
        extra,
    } = load_checkpoint(&bytes).unwrap();
    assert_eq!(extra, b"extra-optimizer-state");
    // 6. Topology and state restored.
    assert_eq!(net.topology(), net2.topology());
    assert_eq!(net.step_count(), net2.step_count());
    assert_eq!(ctl.log(), ctl2.log());

    // Continuing from a checkpoint is indistinguishable from a continuous run.
    for e in 3..8 {
        let (la, ra) = epoch(&mut net, &mut ctl, e);
        let (lb, rb) = epoch(&mut net2, &mut ctl2, e);
        assert_eq!(la, lb, "era {e}: loss diverges");
        assert_eq!(ra.applied, rb.applied);
        assert_eq!(ra.rolled_back, rb.rolled_back);
    }
    assert_eq!(net.topology(), net2.topology());
    assert_eq!(ctl.log(), ctl2.log());
    assert_eq!(evaluate(&net), evaluate(&net2));
}

#[test]
fn growth_is_bounded_under_permanent_plateau() {
    // Worst case: quality never improves, growth is always allowed.
    let mut cfg = config();
    cfg.growth.target_loss = None;
    cfg.growth.neurons.plateau_patience = 1;
    cfg.topology.max_neurons = 12;
    cfg.topology.max_synapses = 30;
    cfg.schedule.min_interval_steps = 0;
    cfg.rollback.enabled = false;
    let (mut net, _) = build(&cfg);
    let mut ctl = StructuralPlasticityController::new(cfg.clone()).unwrap();
    for e in 0..40 {
        train_epoch(&mut net, e);
        ctl.adapt(&mut net, Metrics::loss(1.0), None).unwrap();
        net.topology().validate().unwrap();
        assert!(
            net.topology().neuron_count() <= 12,
            "neurons {}",
            net.topology().neuron_count()
        );
        assert!(
            net.topology().synapse_count() <= 30,
            "synapses {}",
            net.topology().synapse_count()
        );
    }
}

#[test]
fn bad_change_is_rolled_back_and_network_keeps_working() {
    // Dividing a synapse into a plateau for values outside the STDP boundaries breaks the relay
    // (weight 1.0 is trimmed to 0.15); The controller must notice this and roll back.
    let mut cfg = config();
    cfg.growth.target_loss = None;
    cfg.growth.synapses.max_new_per_interval = 0;
    cfg.pruning.enabled = false;
    cfg.growth.neurons.plateau_patience = 1;
    cfg.growth.neurons.strategies = vec![GrowthStrategy::SplitSynapse];
    cfg.schedule.min_interval_steps = 0;
    let (mut net, _) = build(&cfg);
    for e in 0..4 {
        train_epoch(&mut net, e);
    }
    let mut ctl = StructuralPlasticityController::new(cfg).unwrap();
    let mut rolled = false;
    for e in 4..14 {
        let (_, r) = epoch(&mut net, &mut ctl, e);
        rolled |= r.rolled_back;
        net.topology().validate().unwrap();
    }
    assert!(rolled, "not a single rollback during the entire run");
    let rolled_events = ctl
        .log()
        .iter()
        .filter(|e| e.outcome == Outcome::RolledBack)
        .count();
    assert!(rolled_events >= 1);
}
