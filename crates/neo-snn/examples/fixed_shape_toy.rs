//! Structural plasticity of layers of a fixed shape - without Burn (on flat sections).
//!
//! Toy task: two-layer linear network with spare neurons learns from a target with
//! sparse true connections. Shows the full cycle: SGD → projection onto mask → `adapt`
//! between steps → record changes → report to telemetry. Burn-cycle variant (same controller)
//! described in `neo_snn::fixed_shape`.
//!
//! Launch: `cargo run -p neo-snn --example fixed_shape_toy --release`

#![allow(clippy::needless_range_loop)]

use neo_snn::structural::Rng64;
use neo_snn::structural::fixed::*;
use neo_snn::telemetry::{ChannelConfig, Progress, StructureReport, TrainEvent, channel};

const N_IN: usize = 8;
const HID: usize = 6; // capacity; active first 2
const N_OUT: usize = 2;

fn forward(w1: &[f32], w2: &[f32], x: &[f32]) -> (Vec<f32>, Vec<f32>) {
    let mut h = vec![0.0f32; HID];
    for i in 0..N_IN {
        for o in 0..HID {
            h[o] += x[i] * w1[i * HID + o];
        }
    }
    let mut y = vec![0.0f32; N_OUT];
    for i in 0..HID {
        for o in 0..N_OUT {
            y[o] += h[i] * w2[i * N_OUT + o];
        }
    }
    (h, y)
}

fn main() {
    let (tx, rx) = channel(ChannelConfig::new("fixed shape masks"));
    let mut cfg = FixedShapeConfig::default();
    cfg.schedule.min_interval_steps = 150;
    cfg.pruning.min_age_steps = 300;
    cfg.pruning.rel_threshold = 0.1;
    cfg.neurons.plateau_patience = 8;
    cfg.neurons.plateau_min_delta = 5e-3;
    cfg.neurons.min_age_steps = 600;
    cfg.neurons.min_loss = 0.02; // With almost zero loss, we do not increase the capacity
    cfg.rollback.verify_after_calls = 40;
    cfg.rollback.rel_tolerance = 0.3;
    cfg.loss_window = 64;
    let specs = vec![
        LayerSpec::dense("hidden", N_IN, HID, LayerRole::Hidden)
            .with_initial_active(2)
            .with_density(0.6),
        LayerSpec::dense("out", HID, N_OUT, LayerRole::Output).fed_by(0),
    ];
    let mut ctrl = FixedShapeController::new(cfg, specs).expect("configuration");
    let cap = ctrl.capacity();

    // The goal is the composition of two hidden paths (you need ≥ 2 neurons, preferably 3).
    let mut rng = Rng64::new(11);
    let mut t1 = vec![0.0f32; N_IN * HID];
    let mut t2 = vec![0.0f32; HID * N_OUT];
    for (h, ins) in [(0usize, vec![0usize, 1]), (1, vec![2, 3]), (2, vec![4, 5])] {
        for i in ins {
            t1[i * HID + h] = 1.0;
        }
        t2[h * N_OUT + (h % N_OUT)] = 1.0;
    }
    let mut w = [vec![0.0f32; N_IN * HID], vec![0.0f32; HID * N_OUT]];
    for l in 0..2 {
        let mut r = Rng64::new(100 + l as u64);
        ctrl.init_weights(l, &mut w[l], |_, _| 0.3 * r.symmetric())
            .unwrap();
    }

    let steps = 6000u64;
    let lr = 0.02f32;
    let mut ema = 0.0f32;
    for step in 0..steps {
        let x: Vec<f32> = (0..N_IN).map(|_| rng.symmetric()).collect();
        let (h, y) = forward(&w[0], &w[1], &x);
        let (_, tgt) = forward(&t1, &t2, &x);
        let err: Vec<f32> = y.iter().zip(&tgt).map(|(a, b)| a - b).collect();
        let loss: f32 = err.iter().map(|e| e * e).sum::<f32>() / N_OUT as f32;
        ema = if step == 0 {
            loss
        } else {
            ema + 0.02 * (loss - ema)
        };

        // Gradients.
        let mut g2 = vec![0.0f32; HID * N_OUT];
        let mut dh = [0.0f32; HID];
        for i in 0..HID {
            for o in 0..N_OUT {
                g2[i * N_OUT + o] = 2.0 * err[o] * h[i] / N_OUT as f32;
                dh[i] += 2.0 * err[o] * w[1][i * N_OUT + o] / N_OUT as f32;
            }
        }
        let mut g1 = vec![0.0f32; N_IN * HID];
        for i in 0..N_IN {
            for o in 0..HID {
                g1[i * HID + o] = dh[o] * x[i];
            }
        }
        ctrl.observe_grads(0, &w[0], &g1).unwrap();
        ctrl.observe_grads(1, &w[1], &g2).unwrap();
        let ax: Vec<f32> = x.iter().map(|v| v.abs()).collect();
        let ah: Vec<f32> = h.iter().map(|v| v.abs()).collect();
        let ay: Vec<f32> = y.iter().map(|v| v.abs()).collect();
        ctrl.observe_activity(0, &ax, &ah, 1).unwrap();
        ctrl.observe_activity(1, &ah, &ay, 1).unwrap();

        // Optimizer step and projection.
        for (a, g) in w[0].iter_mut().zip(&g1) {
            *a -= lr * g;
        }
        for (a, g) in w[1].iter_mut().zip(&g2) {
            *a -= lr * g;
        }
        for l in 0..2 {
            ctrl.project_weights(l, &mut w[l]).unwrap();
        }

        // Between steps.
        let refs: Vec<&[f32]> = w.iter().map(|v| v.as_slice()).collect();
        let rep = ctrl
            .adapt(
                AdaptInput {
                    step,
                    sample: step,
                    loss: Some(ema),
                },
                &refs,
            )
            .unwrap();
        if !rep.updates.is_empty() {
            for u in &rep.updates {
                u.write_weights(&mut w[u.layer]).unwrap();
            }
            for l in 0..2 {
                ctrl.project_weights(l, &mut w[l]).unwrap();
            }
        }
        if !rep.events.is_empty() || rep.rolled_back {
            tx.structural(StructureReport::from_fixed(
                Progress {
                    epoch: 0,
                    epochs: 1,
                    batch: step as usize,
                    batches: steps as usize,
                },
                &rep,
                cap,
            ));
        }
        if step % 1000 == 0 {
            let s = ctrl.stats(
                AdaptInput {
                    step,
                    sample: step,
                    loss: None,
                },
                None,
            );
            println!(
                "step {step:>5} loss {ema:.4} neurons {} (hidden {}) connections {}",
                s.neurons, s.hidden, s.active_synapses
            );
        }
    }
    drop(tx);
    let mut shown = 0;
    while let Ok(ev) = rx.events.try_recv() {
        if let TrainEvent::Structure(r) = ev {
            for e in &r.events {
                if shown < 12 {
                    println!("  {e}");
                    shown += 1;
                }
            }
        }
    }
    let s = ctrl.stats(
        AdaptInput {
            step: steps,
            sample: steps,
            loss: None,
        },
        Some(&w.iter().map(|v| v.as_slice()).collect::<Vec<_>>()),
    );
    println!(
        "total: loss {ema:.4}, hidden neurons {}, connections {}, density {:.2}",
        s.hidden, s.active_synapses, s.density
    );
    println!("events in the log: {}", ctrl.log().len());
}
