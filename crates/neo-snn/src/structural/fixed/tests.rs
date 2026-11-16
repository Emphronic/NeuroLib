#![allow(
    clippy::needless_range_loop,
    clippy::field_reassign_with_default,
    clippy::identity_op
)]
use super::*;
use crate::structural::error::StructuralError;
use crate::structural::mutation::{ActionKind, Outcome};

fn fast_cfg() -> FixedShapeConfig {
    let mut c = FixedShapeConfig::default();
    c.schedule.min_interval_steps = 0;
    c.pruning.min_age_steps = 0;
    c.pruning.min_observed_samples = 1;
    c.pruning.rel_threshold = 0.5;
    c.pruning.max_fraction_per_interval = 1.0;
    c.neurons.min_age_steps = 0;
    c.neurons.min_observed_samples = 1;
    c.rollback.verify_after_calls = 1;
    c
}

fn refs(w: &[Vec<f32>]) -> Vec<&[f32]> {
    w.iter().map(|v| v.as_slice()).collect()
}

fn inp(step: u64, loss: Option<f32>) -> AdaptInput {
    AdaptInput {
        step,
        sample: step,
        loss,
    }
}

fn apply(c: &FixedShapeController, w: &mut [Vec<f32>], rep: &FixedAdaptReport) {
    for u in &rep.updates {
        u.write_weights(&mut w[u.layer]).unwrap();
    }
    for l in 0..w.len() {
        c.project_weights(l, &mut w[l]).unwrap();
    }
}

fn one_layer(inp_n: usize, out_n: usize) -> FixedShapeController {
    FixedShapeController::new(
        fast_cfg(),
        vec![LayerSpec::dense("fc", inp_n, out_n, LayerRole::Output)],
    )
    .unwrap()
}

#[test]
fn config_validation_is_typed() {
    let mut c = FixedShapeConfig::default();
    c.ema = 0.0;
    assert!(matches!(
        c.validate(),
        Err(StructuralError::InvalidConfig(_))
    ));
    let mut c = FixedShapeConfig::default();
    c.loss_window = 1;
    assert!(matches!(
        c.validate(),
        Err(StructuralError::InvalidConfig(_))
    ));
}

#[test]
fn spec_mismatch_is_rejected() {
    let specs = vec![
        LayerSpec::dense("a", 4, 6, LayerRole::Hidden),
        LayerSpec::dense("b", 5, 2, LayerRole::Output).fed_by(0),
    ];
    let e = FixedShapeController::new(fast_cfg(), specs).unwrap_err();
    assert!(matches!(e, StructuralError::InvalidConfig(_)), "{e}");
    let e = FixedShapeController::new(
        fast_cfg(),
        vec![LayerSpec::dense("a", 4, 6, LayerRole::Output).with_initial_active(3)],
    )
    .unwrap_err();
    assert!(matches!(e, StructuralError::InvalidConfig(_)));
}

#[test]
fn reserve_neurons_start_inactive_and_masked() {
    let c = FixedShapeController::new(
        fast_cfg(),
        vec![
            LayerSpec::dense("h", 3, 6, LayerRole::Hidden)
                .with_initial_active(4)
                .with_density(0.5),
        ],
    )
    .unwrap();
    let l = &c.layers()[0];
    assert_eq!(l.active_neurons(), 4);
    let m = c.effective_mask(0);
    for o in 4..6 {
        assert!(
            (0..3).all(|i| !m[i * 6 + o]),
            "the reserve should not have connections"
        );
    }
    for o in 0..4 {
        assert!(
            (0..3).any(|i| m[i * 6 + o]),
            "an active neuron has an input"
        );
    }
    let mut w = vec![1.0; 18];
    c.project_weights(0, &mut w).unwrap();
    assert!(w.iter().zip(&m).all(|(x, &mm)| (*x != 0.0) == mm));
    let mut bad = vec![0.0; 3];
    assert!(matches!(
        c.project_weights(0, &mut bad),
        Err(StructuralError::InputSize { .. })
    ));
}

#[test]
fn prune_disables_low_saliency_and_keeps_min_in_degree() {
    let mut c = one_layer(4, 2);
    let mut w = vec![vec![1.0f32; 8]];
    // Importance: connections (i,0) are small, (i,1) are large.
    let mut sal = vec![0.0f32; 8];
    for i in 0..4 {
        sal[i * 2] = 0.001;
        sal[i * 2 + 1] = 1.0;
    }
    let g: Vec<f32> = sal.clone();
    c.observe_grads(0, &w[0], &g).unwrap();
    let rep = c.adapt(inp(10, Some(1.0)), &refs(&w)).unwrap();
    assert!(rep.skipped.is_none(), "{:?}", rep.skipped);
    apply(&c, &mut w, &rep);
    // Neuron 0 loses connections, but at least one remains.
    let m = c.effective_mask(0);
    let indeg0 = (0..4).filter(|&i| m[i * 2]).count();
    let indeg1 = (0..4).filter(|&i| m[i * 2 + 1]).count();
    assert_eq!(indeg0, 1, "protection min_in_degree");
    assert_eq!(indeg1, 4);
    assert!(
        rep.events
            .iter()
            .all(|e| e.kind == ActionKind::DisableSynapse && e.outcome == Outcome::Applied)
    );
    assert_eq!(rep.events.len(), 3);
    for (idx, x) in w[0].iter().enumerate() {
        assert_eq!(*x != 0.0, m[idx]);
    }
}

#[test]
fn prune_respects_age() {
    let mut cfg = fast_cfg();
    cfg.pruning.min_age_steps = 100;
    let mut c =
        FixedShapeController::new(cfg, vec![LayerSpec::dense("fc", 4, 2, LayerRole::Output)])
            .unwrap();
    let w = vec![vec![1.0f32; 8]];
    let mut g = vec![1.0f32; 8];
    g[0] = 0.0;
    c.observe_grads(0, &w[0], &g).unwrap();
    let rep = c.adapt(inp(10, None), &refs(&w)).unwrap();
    assert_eq!(rep.skipped, Some(FixedSkip::NothingToDo));
}

#[test]
fn growth_picks_most_coactive_pair() {
    let mut cfg = fast_cfg();
    cfg.pruning.enabled = false;
    cfg.growth.max_new_per_interval = 1;
    let mut c = FixedShapeController::new(
        cfg,
        vec![LayerSpec::dense("fc", 3, 3, LayerRole::Output).with_density(0.34)],
    )
    .unwrap();
    let flags = c.layers()[0].synapse_flags().to_vec();
    let free: Vec<usize> = (0..9).filter(|&i| !flags[i]).collect();
    assert!(free.len() >= 2);
    let target = free[1];
    // The target pair is always active at the same time.
    let (ti, to) = (target / 3, target % 3);
    let mut pre = vec![0.0f32; 3];
    let mut post = vec![0.0f32; 3];
    pre[ti] = 1.0;
    post[to] = 1.0;
    c.observe_activity(0, &pre, &post, 1).unwrap();
    let mut w = vec![vec![0.0f32; 9]];
    c.init_weights(0, &mut w[0], |_, _| 0.5).unwrap();
    let rep = c.adapt(inp(5, None), &refs(&w)).unwrap();
    assert_eq!(rep.events.len(), 1);
    assert_eq!(rep.events[0].kind, ActionKind::AddSynapse);
    apply(&c, &mut w, &rep);
    assert!(c.layers()[0].synapse_flags()[target]);
    assert!(
        w[0][target] > 0.05 && w[0][target] < 0.16,
        "initial weight {}",
        w[0][target]
    );
}

#[test]
fn growth_respects_density_cap() {
    let mut cfg = fast_cfg();
    cfg.pruning.enabled = false;
    cfg.growth.max_density = 0.5;
    let mut c = FixedShapeController::new(
        cfg,
        vec![LayerSpec::dense("fc", 4, 2, LayerRole::Output).with_density(0.5)],
    )
    .unwrap();
    let have = c.layers()[0].synapse_flags().iter().filter(|&&f| f).count();
    c.observe_activity(0, &[1.0; 4], &[1.0; 2], 1).unwrap();
    let mut w = vec![vec![0.0f32; 8]];
    c.init_weights(0, &mut w[0], |_, _| 0.5).unwrap();
    let rep = c.adapt(inp(5, None), &refs(&w)).unwrap();
    let now = c.effective_mask(0).iter().filter(|&&f| f).count();
    // The initial density may already be above the limit: growth is then prohibited, but it does not take away either.
    assert!(now <= have.max(4), "was {have}, became {now}, limit 4");
    assert!(
        rep.events.iter().all(|e| e.kind != ActionKind::AddSynapse),
        "{:?}",
        rep.events
    );
}

fn hidden_net() -> FixedShapeController {
    let mut cfg = fast_cfg();
    cfg.pruning.enabled = false;
    cfg.growth.enabled = false;
    cfg.neurons.plateau_patience = 4;
    cfg.neurons.new_fan_in = 2;
    cfg.neurons.new_fan_out = 2;
    FixedShapeController::new(
        cfg,
        vec![
            LayerSpec::dense("h", 4, 5, LayerRole::Hidden).with_initial_active(3),
            LayerSpec::dense("o", 5, 2, LayerRole::Output).fed_by(0),
        ],
    )
    .unwrap()
}

#[test]
fn plateau_activates_reserve_neuron_with_wiring() {
    let mut c = hidden_net();
    let mut w: Vec<Vec<f32>> = vec![vec![0.0; 20], vec![0.0; 10]];
    for l in 0..2 {
        c.init_weights(l, &mut w[l], |_, _| 0.3).unwrap();
    }
    c.observe_activity(0, &[1.0, 0.5, 0.0, 0.0], &[0.2, 0.2, 0.2, 0.0, 0.0], 1)
        .unwrap();
    // Loss stands still.
    let mut rep = None;
    for s in 0..4u64 {
        let r = c.adapt(inp(s, Some(1.0)), &refs(&w)).unwrap();
        if !r.events.is_empty() {
            rep = Some(r);
            break;
        }
    }
    let rep = rep.expect("at the plateau the reserve should turn on");
    assert_eq!(rep.events[0].kind, ActionKind::AddNeuron);
    apply(&c, &mut w, &rep);
    assert_eq!(c.layers()[0].active_neurons(), 4);
    let o = 3;
    let m0 = c.effective_mask(0);
    assert_eq!((0..4).filter(|&i| m0[i * 5 + o]).count(), 2);
    let m1 = c.effective_mask(1);
    assert_eq!((0..2).filter(|&o2| m1[o * 2 + o2]).count(), 2);
    // The most active inputs are selected first.
    assert!(m0[o] && m0[5 + o]);
}

#[test]
fn deactivating_silent_neuron_clears_all_links() {
    let mut cfg = fast_cfg();
    cfg.pruning.enabled = false;
    cfg.growth.enabled = false;
    let mut c = FixedShapeController::new(
        cfg,
        vec![
            LayerSpec::dense("h", 3, 3, LayerRole::Hidden),
            LayerSpec::dense("o", 3, 2, LayerRole::Output).fed_by(0),
        ],
    )
    .unwrap();
    let mut w: Vec<Vec<f32>> = vec![vec![0.0; 9], vec![0.0; 6]];
    for l in 0..2 {
        c.init_weights(l, &mut w[l], |_, _| 0.4).unwrap();
    }
    c.observe_activity(0, &[1.0; 3], &[0.3, 0.0, 0.3], 1)
        .unwrap();
    let rep = c.adapt(inp(10, Some(1.0)), &refs(&w)).unwrap();
    assert_eq!(rep.events.len(), 1, "{:?}", rep.events);
    assert_eq!(rep.events[0].kind, ActionKind::RemoveNeuron);
    apply(&c, &mut w, &rep);
    assert!(!c.layers()[0].neuron_active()[1]);
    // Column 1 of the first layer and row 1 of the second are zeros.
    assert!((0..3).all(|i| w[0][i * 3 + 1] == 0.0));
    assert!((0..2).all(|o| w[1][2 + o] == 0.0));
    // An effective mask of the next layer does not allow dead input to pass through.
    let m1 = c.effective_mask(1);
    assert!((0..2).all(|o| !m1[2 + o]));
}

#[test]
fn rollback_restores_masks_and_weights() {
    let mut c = one_layer(4, 2);
    let mut w = vec![vec![0.7f32; 8]];
    let mut g = vec![1.0f32; 8];
    for i in 0..4 {
        g[i * 2] = 0.0;
    }
    c.observe_grads(0, &w[0], &g).unwrap();
    let flags_before = c.layers()[0].synapse_flags().to_vec();
    let w_before = w[0].clone();
    let rep = c.adapt(inp(10, Some(1.0)), &refs(&w)).unwrap();
    assert!(!rep.events.is_empty());
    apply(&c, &mut w, &rep);
    assert_ne!(c.layers()[0].synapse_flags(), &flags_before[..]);
    assert!(c.is_verifying());
    // Loss deteriorated sharply → rollback.
    let rep2 = c.adapt(inp(11, Some(5.0)), &refs(&w)).unwrap();
    assert!(rep2.rolled_back);
    apply(&c, &mut w, &rep2);
    assert_eq!(c.layers()[0].synapse_flags(), &flags_before[..]);
    assert_eq!(
        w[0], w_before,
        "weights of disconnected connections are restored"
    );
    assert!(c.log().iter().all(|e| e.outcome == Outcome::RolledBack));
    assert!(!c.is_verifying());
    // Pause after rollback: Immediately retrying does not disconnect communications.
    c.observe_grads(0, &w[0], &g).unwrap();
    let rep3 = c.adapt(inp(12, Some(1.0)), &refs(&w)).unwrap();
    assert!(
        rep3.events
            .iter()
            .all(|e| e.kind != ActionKind::DisableSynapse)
    );
}

#[test]
fn accepted_change_is_kept_when_loss_ok() {
    let mut c = one_layer(4, 2);
    let mut w = vec![vec![0.7f32; 8]];
    let mut g = vec![1.0f32; 8];
    for i in 0..4 {
        g[i * 2] = 0.0;
    }
    c.observe_grads(0, &w[0], &g).unwrap();
    let rep = c.adapt(inp(10, Some(1.0)), &refs(&w)).unwrap();
    apply(&c, &mut w, &rep);
    let flags = c.layers()[0].synapse_flags().to_vec();
    let rep2 = c.adapt(inp(11, Some(0.9)), &refs(&w)).unwrap();
    assert!(!rep2.rolled_back);
    assert_eq!(c.layers()[0].synapse_flags(), &flags[..]);
    assert!(c.log().iter().all(|e| e.outcome == Outcome::Applied));
}

#[test]
fn revert_report_undoes_failed_apply() {
    let mut c = one_layer(4, 2);
    let w = vec![vec![0.7f32; 8]];
    let mut g = vec![1.0f32; 8];
    g[0] = 0.0;
    c.observe_grads(0, &w[0], &g).unwrap();
    let before = c.layers()[0].synapse_flags().to_vec();
    let rep = c.adapt(inp(10, None), &refs(&w)).unwrap();
    assert!(!rep.events.is_empty());
    c.revert_report(10);
    assert_eq!(c.layers()[0].synapse_flags(), &before[..]);
    assert!(!c.is_verifying());
}

#[test]
fn wrong_weight_shapes_are_typed_errors() {
    let mut c = one_layer(4, 2);
    let w = vec![vec![0.0f32; 7]];
    assert!(matches!(
        c.adapt(inp(1, None), &refs(&w)),
        Err(StructuralError::InputSize { .. })
    ));
    assert!(matches!(
        c.adapt(inp(1, None), &[]),
        Err(StructuralError::InputSize { .. })
    ));
    assert!(matches!(
        c.observe_activity(0, &[1.0; 3], &[1.0; 2], 1),
        Err(StructuralError::InputSize { .. })
    ));
}

#[test]
fn masks_roundtrip_and_detect_corruption() {
    let mut a = hidden_net();
    let bytes = a.save_masks();
    let mut b = FixedShapeController::new(
        a.config().clone(),
        a.layers().iter().map(|l| l.spec().clone()).collect(),
    )
    .unwrap();
    // Let's change b, then load masks a.
    let mut w: Vec<Vec<f32>> = vec![vec![0.0; 20], vec![0.0; 10]];
    for l in 0..2 {
        b.init_weights(l, &mut w[l], |_, _| 0.3).unwrap();
    }
    b.load_masks(&bytes).unwrap();
    for l in 0..2 {
        assert_eq!(a.effective_mask(l), b.effective_mask(l));
    }
    let mut bad = bytes.clone();
    bad[20] ^= 1;
    assert!(matches!(
        b.load_masks(&bad),
        Err(StructuralError::Checkpoint(_))
    ));
    assert!(matches!(
        b.load_masks(&bytes[..10]),
        Err(StructuralError::Checkpoint(_))
    ));
    let mut other = one_layer(4, 2);
    assert!(other.load_masks(&bytes).is_err());
    let _ = &mut a;
}

#[test]
fn same_seed_same_decisions() {
    let run = || {
        let mut c = hidden_net();
        let mut w: Vec<Vec<f32>> = vec![vec![0.0; 20], vec![0.0; 10]];
        for l in 0..2 {
            c.init_weights(l, &mut w[l], |_, _| 0.3).unwrap();
        }
        c.observe_activity(0, &[1.0, 0.5, 0.2, 0.1], &[0.2, 0.2, 0.2, 0.0, 0.0], 1)
            .unwrap();
        for s in 0..6u64 {
            let r = c.adapt(inp(s, Some(1.0)), &refs(&w)).unwrap();
            apply(&c, &mut w, &r);
        }
        (c.save_masks(), w)
    };
    assert_eq!(run(), run());
}

/// End-to-end loop on a toy problem without Burn: linear regression `y = x·W`
/// with true sparse `W*`, SGD + projection + `adapt` between steps.
#[test]
fn toy_training_loop_keeps_projection_invariant_and_learns() {
    let (n_in, n_out) = (8usize, 2usize);
    let mut cfg = fast_cfg();
    cfg.schedule.min_interval_steps = 20;
    cfg.pruning.min_age_steps = 50;
    cfg.pruning.min_observed_samples = 10;
    cfg.pruning.rel_threshold = 0.2;
    cfg.growth.enabled = false;
    cfg.rollback.verify_after_calls = 2;
    cfg.rollback.rel_tolerance = 0.5;
    let mut c = FixedShapeController::new(
        cfg,
        vec![LayerSpec::dense("fc", n_in, n_out, LayerRole::Output)],
    )
    .unwrap();
    // True: output 0 depends on inputs 0,1; output 1 - from 2.
    let mut wt = vec![0.0f32; n_in * n_out];
    wt[0] = 1.0;
    wt[1 * 2] = -1.0;
    wt[2 * 2 + 1] = 2.0;
    let mut rng = crate::structural::Rng64::new(7);
    let mut w = vec![vec![0.0f32; n_in * n_out]];
    c.init_weights(0, &mut w[0], |_, _| 0.1 * rng.symmetric())
        .unwrap();
    let lr = 0.05f32;
    let mut loss_ema = 1.0f32;
    let mut first_loss = None;
    for step in 0..600u64 {
        let x: Vec<f32> = (0..n_in).map(|_| rng.symmetric()).collect();
        let mut y = vec![0.0f32; n_out];
        let mut t = vec![0.0f32; n_out];
        for o in 0..n_out {
            for i in 0..n_in {
                y[o] += x[i] * w[0][i * n_out + o];
                t[o] += x[i] * wt[i * n_out + o];
            }
        }
        let err: Vec<f32> = y.iter().zip(&t).map(|(a, b)| a - b).collect();
        let loss: f32 = err.iter().map(|e| e * e).sum::<f32>() / n_out as f32;
        loss_ema = if step == 0 {
            loss
        } else {
            loss_ema + 0.05 * (loss - loss_ema)
        };
        first_loss.get_or_insert(loss_ema);
        let mut grad = vec![0.0f32; n_in * n_out];
        for i in 0..n_in {
            for o in 0..n_out {
                grad[i * n_out + o] = 2.0 * err[o] * x[i] / n_out as f32;
            }
        }
        c.observe_grads(0, &w[0], &grad).unwrap();
        c.observe_activity(
            0,
            &x.iter().map(|v| v.abs()).collect::<Vec<_>>(),
            &y.iter().map(|v| v.abs()).collect::<Vec<_>>(),
            1,
        )
        .unwrap();
        for (a, g) in w[0].iter_mut().zip(&grad) {
            *a -= lr * g;
        }
        c.project_weights(0, &mut w[0]).unwrap();
        let rep = c.adapt(inp(step, Some(loss_ema)), &refs(&w)).unwrap();
        apply(&c, &mut w, &rep);
        // Invariant: outside the effective mask - exactly zero.
        let m = c.effective_mask(0);
        assert!(
            w[0].iter().zip(&m).all(|(x, &mm)| mm || *x == 0.0),
            "step {step}"
        );
    }
    assert!(
        loss_ema < first_loss.unwrap() * 0.2,
        "loss {} → {}",
        first_loss.unwrap(),
        loss_ema
    );
    let kept = c.effective_mask(0).iter().filter(|&&m| m).count();
    assert!(
        kept < n_in * n_out,
        "at least one outage was expected, {kept} remained"
    );
    assert!(
        c.effective_mask(0)[0] && c.effective_mask(0)[2] && c.effective_mask(0)[5],
        "true connections should not be disabled"
    );
}
