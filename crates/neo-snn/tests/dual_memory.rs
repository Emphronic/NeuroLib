//! Integration tests for `neo_snn::temporal::memory` — dual fast/slow memory
//! (paper: Sun et al., "Algorithm–hardware co-design of neuromorphic networks
//! with dual memory pathways", Nature Machine Intelligence, 2026):
//!
//! * `state_matrices_hand_checked` — the Legendre state-space matrices match
//!   the paper formulas (hand-checked values);
//! * `exact_zoh_discretization_hand_checked` — the exact ZOH discretization
//!   `Ā = e^{A·dt}`, `B̄ = A⁻¹(e^{A·dt} − I)B` gives hand-checked values on
//!   `d = 1` (`A = [[−1]]`, `B = [1]` → `Ā = e^{−1}`, `B̄ = 1 − e^{−1}`);
//! * `slow_step_matches_paper_recurrence` — `SlowMemory::step` runs the paper
//!   recurrence `m[k] = Ā·m[k−1] + B̄·x[k]`;
//! * `slow_drive_shapes_and_zero` — the scalar drive is `[batch, 1]`, zero on
//!   zero spikes with the default zero bias;
//! * `dual_layer_zero_input_zero_output` — zero input → zero output, both
//!   states stay zero;
//! * `dual_layer_populates_both_paths` — a live run: spikes ∈ {0, 1}, the
//!   slow state is populated, memory feedback is nonzero;
//! * `dual_layer_forward_seq_from` — `SpikingLayer` sequence mode runs and
//!   the state detaches.
//!
//! Run: `cargo test -p neo-snn --test dual_memory`

use burn::module::Param;
use burn::tensor::{Device, Tensor};
use neo_snn::prelude::*;
use neo_snn::temporal::memory::{SlowMemory, SlowMemoryConfig, SlowMemoryState};

fn dev() -> Device {
    Device::flex()
}

fn t2(data: &[f32], shape: [usize; 2]) -> Tensor<2> {
    Tensor::<1>::from_floats(data, &dev()).reshape(shape)
}

fn v2(t: Tensor<2>) -> Vec<f32> {
    t.into_data().try_to_vec::<f32>().unwrap()
}

/// `m[j] = Σ_i a[j][i]·x[i]` — the `[d, d]` matrix `a` as a flat row-major vec.
fn mat_vec(a: &[f32], x: &[f32], d: usize) -> Vec<f32> {
    (0..d)
        .map(|j| (0..d).map(|i| a[j * d + i] * x[i]).sum())
        .collect()
}

#[test]
fn state_matrices_hand_checked() {
    // Paper formulas: for d = 2, θ = 2 → r = [0.5, 1.5] →
    // A = [[−0.5, −0.5], [1.5, −1.5]], B = [0.5, −1.5].
    let (a, b) = SlowMemory::state_matrices(2, 2.0);
    let expect_a = [[-0.5, -0.5], [1.5, -1.5]];
    let expect_b = [0.5, -1.5];
    for i in 0..2 {
        for j in 0..2 {
            assert!((a[i][j] - expect_a[i][j]).abs() < 1e-12, "a[{i}][{j}]");
        }
    }
    for i in 0..2 {
        assert!((b[i] - expect_b[i]).abs() < 1e-12, "b[{i}]");
    }
    // d = 1, θ = 1 → A = [[−1]], B = [1].
    let (a, b) = SlowMemory::state_matrices(1, 1.0);
    assert!((a[0][0] + 1.0).abs() < 1e-12);
    assert!((b[0] - 1.0).abs() < 1e-12);
}

#[test]
fn exact_zoh_discretization_hand_checked() {
    // d = 1, θ = 1, dt = 1: A = [[−1]], B = [1] → Ā = e^{−1}, B̄ = 1 − e^{−1}.
    let slow = SlowMemory::new(SlowMemoryConfig::new(1, 1.0), 4, &dev());
    let a_hat = v2(slow.a_hat.clone());
    let b_hat = v2(slow.b_hat.clone());
    assert_eq!(a_hat.len(), 1);
    assert_eq!(b_hat.len(), 1);
    assert!(
        (a_hat[0] - (-1.0f32).exp()).abs() < 1e-5,
        "a_hat = {a_hat:?}"
    );
    assert!(
        (b_hat[0] - (1.0f32 - (-1.0f32).exp())).abs() < 1e-5,
        "b_hat = {b_hat:?}"
    );
}

#[test]
fn slow_step_matches_paper_recurrence() {
    // m[k] = Ā·m[k−1] + B̄·x[k] — two steps from zero state, constant drive x.
    let d = 3;
    let slow = SlowMemory::new(SlowMemoryConfig::new(d, 5.0), 6, &dev());
    let a_hat = v2(slow.a_hat.clone());
    let b_hat = v2(slow.b_hat.clone());
    let x = 0.7f32;
    let x_t = t2(&[x], [1, 1]);

    let state = SlowMemoryState::zeros(1, d, &dev());
    let m1 = slow.step(&x_t, &state);
    let m1v = v2(m1.m.clone());
    for j in 0..d {
        let expect = x * b_hat[j];
        assert!(
            (m1v[j] - expect).abs() < 1e-4,
            "m1[{j}] = {} vs {expect}",
            m1v[j]
        );
    }

    let m2 = slow.step(&x_t, &m1);
    let m2v = v2(m2.m.clone());
    let sum = mat_vec(&a_hat, &m1v, d);
    for j in 0..d {
        let expect = sum[j] + x * b_hat[j];
        assert!(
            (m2v[j] - expect).abs() < 1e-4,
            "m2[{j}] = {} vs {expect}",
            m2v[j]
        );
    }
}

#[test]
fn slow_drive_shapes_and_zero() {
    // Drive is `[batch, 1]` and is zero on zero spikes with the default zero bias.
    let slow = SlowMemory::new(SlowMemoryConfig::new(2, 4.0), 5, &dev());
    let drive = slow.drive(&Tensor::zeros([3, 5], &dev()));
    assert_eq!(drive.dims(), [3, 1]);
    for v in v2(drive) {
        assert_eq!(v, 0.0);
    }
}

#[test]
fn dual_layer_zero_input_zero_output() {
    let device = dev();
    let layer = DualMemoryLayer::new(DualMemoryConfig::new(2, 10.0), 4, 3, &device);
    let surrogate = ATan::default();
    let state = layer.init_state(2, &device);
    assert_eq!(state.slow.m.dims(), [2, 2]);
    assert_eq!(state.fast.mem.dims(), [2, 4]);

    let (out, state) = layer.step(t2(&[0.0; 6], [2, 3]), state, &surrogate);
    assert_eq!(out.dims(), [2, 4]);
    for v in v2(out) {
        assert_eq!(v, 0.0);
    }
    for v in v2(state.slow.m.clone()) {
        assert_eq!(v, 0.0);
    }
    for v in v2(state.fast.mem.clone()) {
        assert_eq!(v, 0.0);
    }
}

#[test]
fn dual_layer_populates_both_paths() {
    let device = dev();
    let mut layer = DualMemoryLayer::new(DualMemoryConfig::new(2, 10.0), 4, 3, &device);
    let surrogate = ATan::default();
    // Deterministic dynamics: unit slow-drive weights (drive = 3 per step)
    // and unit fast-path input weights (so the fast membrane is nonzero and
    // spikes are guaranteed with the default threshold 1.0).
    layer.slow.w_x = Param::from_tensor(Tensor::ones([1, 3], &device));
    layer.w_f = Param::from_tensor(Tensor::ones([4, 3], &device));
    let mut state = layer.init_state(2, &device);
    let spikes_in = t2(&[1.0; 6], [2, 3]);

    let mut saw_spike = false;
    for _ in 0..20 {
        let (out, ns) = layer.step(spikes_in.clone(), state, &surrogate);
        assert_eq!(out.dims(), [2, 4]);
        for v in v2(out.clone()) {
            assert!(v == 0.0 || v == 1.0, "spike value {v} is not binary");
            saw_spike |= v == 1.0;
        }
        state = ns;
    }

    assert!(
        !v2(state.slow.m.clone()).iter().all(|v| *v == 0.0),
        "slow state must be populated"
    );
    let fb = layer.memory_feedback.feedback(&state.slow.m);
    assert_eq!(fb.dims(), [2, 4]);
    assert!(
        v2(fb).iter().any(|v| *v != 0.0),
        "memory feedback must be nonzero when the slow state is populated"
    );
    assert!(saw_spike, "fast path should produce spikes over 20 steps");
}

#[test]
fn dual_layer_forward_seq_from() {
    let device = dev();
    let layer = DualMemoryLayer::new(DualMemoryConfig::new(2, 10.0), 4, 3, &device);
    let surrogate = ATan::default();
    let xs: Vec<Tensor<2>> = (0..5)
        .map(|_| t2(&[1.0, 0.0, 1.0, 1.0, 0.0, 1.0], [2, 3]))
        .collect();
    let (outs, state) = layer.forward_seq_from(xs, layer.init_state(2, &device), &surrogate);
    assert_eq!(outs.len(), 5);
    state.detach();
}
