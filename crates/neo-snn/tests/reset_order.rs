//! Order "reset ↔ attenuation" (`ResetOrder`): `AfterDecay` = snnTorch dynamics,
//! `U[t+1] = β·U[t] − θ·S[t] + I[t+1]`, `BeforeDecay` applies the reset before decay,
//! `U[t+1] = β·(U[t] − θ·S[t]) + I[t+1]`.
//!
//! Expected values are computed in f32; the minimum gap |U − θ| ≈ 1.6e-3 for
//! `after_decay_matches_snntorch_recurrence` keeps the comparison away from
//! threshold ties. Launch: `cargo test -p neo-snn --test reset_order`.

use burn::tensor::{Device, Tensor};
use neo_snn::prelude::*;

fn dev() -> Device {
    Device::flex()
}

fn t2(data: &[f32], shape: [usize; 2]) -> Tensor<2> {
    Tensor::<1>::from_floats(data, &dev()).reshape(shape)
}

fn t3(data: &[f32], shape: [usize; 3]) -> Tensor<3> {
    Tensor::<1>::from_floats(data, &dev()).reshape(shape)
}

fn v2(t: Tensor<2>) -> Vec<f32> {
    t.into_data().try_to_vec::<f32>().unwrap()
}

fn v3(t: Tensor<3>) -> Vec<f32> {
    t.into_data().try_to_vec::<f32>().unwrap()
}

fn close(a: &[f32], b: &[f32], tol: f32, what: &str) {
    assert_eq!(a.len(), b.len(), "{what}: different lengths");
    for (i, (x, y)) in a.iter().zip(b).enumerate() {
        assert!((x - y).abs() <= tol, "{what}[{i}]: {x} vs {y}");
    }
}

/// Running one neuron `[1, 1]`: returns (spikes, membrane from the state after the step).
fn run_leaky(config: &LeakyConfig, xs: &[f32]) -> (Vec<f32>, Vec<f32>) {
    let device = dev();
    let layer = Leaky::new(config, &device);
    let surrogate = ATan::default();
    let mut state = NeuronState::zeros_mem_only(&device, 1, 1);
    let (mut spikes, mut mems) = (vec![], vec![]);
    for &x in xs {
        let (s, ns) = layer.step(t2(&[x], [1, 1]), &state, &surrogate);
        spikes.push(v2(s)[0]);
        mems.push(v2(ns.mem.clone())[0]);
        state = ns;
    }
    (spikes, mems)
}

fn inputs(n: usize) -> Vec<f32> {
    (0..n)
        .map(|t| ((t * 37) % 11) as f32 / 10.0 + 0.05)
        .collect()
}

#[test]
fn default_order_is_before_decay() {
    assert_eq!(ResetOrder::default(), ResetOrder::BeforeDecay);
    assert_eq!(LeakyConfig::default().reset_order, ResetOrder::BeforeDecay);
}

/// β = 0.5, θ = 1, input 1.5 at the first step, then zero: computed by hand.
#[test]
fn closed_form_single_neuron() {
    let xs = [1.5, 0.0, 0.0, 0.0];

    // BeforeDecay: the membrane is in the post-reset state, and the threshold decays along with it.
    let before = LeakyConfig::new(0.5).with_threshold(1.0);
    let (spikes, mems) = run_leaky(&before, &xs);
    assert_eq!(spikes, vec![1.0, 0.0, 0.0, 0.0]);
    close(&mems, &[0.5, 0.25, 0.125, 0.0625], 1e-6, "before mem");

    // AfterDecay: in state - membrane before reset; at the 2nd step 0.5·1.5 − 1 = −0.25.
    let after = before.with_reset_order(ResetOrder::AfterDecay);
    let (spikes, mems) = run_leaky(&after, &xs);
    assert_eq!(spikes, vec![1.0, 0.0, 0.0, 0.0]);
    close(&mems, &[1.5, -0.25, -0.125, -0.0625], 1e-6, "after mem");
}

/// General recurrent snnTorch at 60 steps (membrane and spikes), and make sure that the orders
/// really give different rasters (otherwise the test would not check anything).
#[test]
fn after_decay_matches_snntorch_recurrence() {
    let xs = inputs(60);
    let beta = 0.8f32;
    let th = 1.0f32;

    let (mut u, mut s_prev) = (0.0f32, 0.0f32);
    let (mut ref_spikes, mut ref_mems) = (vec![], vec![]);
    for &x in &xs {
        u = beta * u + x - th * s_prev;
        s_prev = if u > th { 1.0 } else { 0.0 };
        ref_spikes.push(s_prev);
        ref_mems.push(u);
    }

    let after = LeakyConfig::new(beta)
        .with_threshold(th)
        .with_reset_order(ResetOrder::AfterDecay);
    let (spikes, mems) = run_leaky(&after, &xs);
    assert_eq!(spikes, ref_spikes, "raster AfterDecay");
    close(&mems, &ref_mems, 1e-4, "AfterDecay membrane");

    // BeforeDecay: reset first, U' = β·U_reset + I.
    let (mut u, mut before_ref) = (0.0f32, vec![]);
    for &x in &xs {
        let un = beta * u + x;
        let s = if un > th { 1.0 } else { 0.0 };
        u = un - th * s;
        before_ref.push(s);
    }
    let before = LeakyConfig::new(beta).with_threshold(th);
    let (spikes_before, _) = run_leaky(&before, &xs);
    assert_eq!(spikes_before, before_ref, "raster BeforeDecay");
    assert_ne!(
        spikes_before, ref_spikes,
        "orders should give different rasters"
    );
}

/// `Zero` and `None` do not depend on order (zeroing commutes with scalar β).
#[test]
fn zero_and_none_modes_are_order_independent() {
    let xs = inputs(60);
    for mode in [ResetMode::Zero, ResetMode::None] {
        let base = LeakyConfig::new(0.8)
            .with_threshold(1.0)
            .with_reset_mode(mode);
        let (a, _) = run_leaky(&base, &xs);
        let (b, _) = run_leaky(&base.with_reset_order(ResetOrder::AfterDecay), &xs);
        assert_eq!(a, b, "{mode:?}");
    }
}

/// `MultiStepLeaky` (and thus `LeakyParallel` for Subtract/Zero) must match
/// with an explicit cycle `Leaky::step` - including winner-take-all.
#[test]
fn multistep_matches_explicit_loop_after_decay() {
    let device = dev();
    let (batch, steps, feats) = (2usize, 15usize, 3usize);
    let data: Vec<f32> = (0..batch * steps * feats)
        .map(|i| ((i * 37) % 101) as f32 / 60.0)
        .collect();
    let surrogate = ATan::default();

    for inhibition in [false, true] {
        let config = LeakyConfig::new(0.85)
            .with_threshold(1.0)
            .with_reset_order(ResetOrder::AfterDecay)
            .with_inhibition(inhibition);

        let multi = MultiStepLeaky::new(&config, &device);
        let (out, final_state) =
            multi.forward_with_state(t3(&data, [batch, steps, feats]), &surrogate);

        let layer = Leaky::new(&config, &device);
        let input = t3(&data, [batch, steps, feats]);
        let mut state = NeuronState::zeros_mem_only(&device, batch, feats);
        let mut outs = vec![];
        for t in 0..steps {
            let x_t = input
                .clone()
                .slice([0..batch, t..t + 1, 0..feats])
                .squeeze_dims::<2>(&[1]);
            let (s, ns) = layer.step(x_t, &state, &surrogate);
            state = ns;
            outs.push(s);
        }
        let explicit = Tensor::stack::<3>(outs, 1);

        close(&v3(out), &v3(explicit), 1e-5, "spikes");
        close(&v2(final_state.mem), &v2(state.mem), 1e-4, "final membrane");
    }
}

#[test]
fn leaky_parallel_matches_explicit_loop_after_decay() {
    let device = dev();
    let (batch, steps, feats) = (2usize, 12usize, 3usize);
    let data: Vec<f32> = (0..batch * steps * feats)
        .map(|i| ((i * 29) % 97) as f32 / 55.0)
        .collect();
    let surrogate = ATan::default();
    let config = LeakyConfig::new(0.9)
        .with_threshold(1.0)
        .with_reset_order(ResetOrder::AfterDecay);

    let par = LeakyParallel::new(&config, &device);
    let out = par.forward(t3(&data, [batch, steps, feats]), &surrogate);

    let layer = Leaky::new(&config, &device);
    let input = t3(&data, [batch, steps, feats]);
    let mut state = NeuronState::zeros_mem_only(&device, batch, feats);
    let mut outs = vec![];
    for t in 0..steps {
        let x_t = input
            .clone()
            .slice([0..batch, t..t + 1, 0..feats])
            .squeeze_dims::<2>(&[1]);
        let (s, ns) = layer.step(x_t, &state, &surrogate);
        state = ns;
        outs.push(s);
    }
    close(&v3(out), &v3(Tensor::stack::<3>(outs, 1)), 1e-5, "spikes");
}

#[test]
fn rleaky_multistep_matches_explicit_loop_after_decay() {
    let device = dev();
    let (batch, steps, inf, outf) = (2usize, 20usize, 2usize, 3usize);
    let data: Vec<f32> = (0..batch * steps * inf)
        .map(|i| ((i * 37) % 101) as f32 / 40.0)
        .collect();
    let surrogate = ATan::default();

    let layer = RLeaky::new(&device, inf, outf, 0.85, 0.5, ResetMode::Subtract)
        .with_reset_order(ResetOrder::AfterDecay);
    let multi = MultiStepRLeaky::from_layer(layer.clone());
    let out = multi.forward(t3(&data, [batch, steps, inf]), &surrogate);

    let input = t3(&data, [batch, steps, inf]);
    let mut state = NeuronState::zeros_mem_only(&device, batch, outf);
    let mut spike_prev = Tensor::<2>::zeros([batch, outf], &device);
    let mut outs = vec![];
    for t in 0..steps {
        let x_t = input
            .clone()
            .slice([0..batch, t..t + 1, 0..inf])
            .squeeze_dims::<2>(&[1]);
        let (s, ns) = layer.step(x_t, spike_prev.clone(), &state, &surrogate);
        state = ns;
        spike_prev = s.clone();
        outs.push(s);
    }
    close(&v3(out), &v3(Tensor::stack::<3>(outs, 1)), 1e-5, "spikes");
}

/// Synaptic: `syn' = α·syn + I`, `U' = β·U + syn' − θ·S_prev`.
#[test]
fn synaptic_after_decay_matches_reference() {
    let device = dev();
    let xs: Vec<f32> = inputs(60).iter().map(|v| 0.35 * v + 0.1).collect();
    let (beta, alpha, th) = (0.8f32, 0.6f32, 1.0f32);

    let (mut syn, mut u, mut s_prev) = (0.0f32, 0.0f32, 0.0f32);
    let mut ref_spikes = vec![];
    for &x in &xs {
        syn = alpha * syn + x;
        u = beta * u + syn - th * s_prev;
        s_prev = if u > th { 1.0 } else { 0.0 };
        ref_spikes.push(s_prev);
    }

    let layer = Synaptic::new(&device, beta, alpha, th, ResetMode::Subtract)
        .with_reset_order(ResetOrder::AfterDecay);
    let surrogate = ATan::default();
    let mut state = NeuronState::zeros(&device, 1, 1);
    let mut spikes = vec![];
    for &x in &xs {
        let (s, ns) = layer.step(t2(&[x], [1, 1]), &state, &surrogate);
        spikes.push(v2(s)[0]);
        state = ns;
    }
    assert_eq!(spikes, ref_spikes);
}
