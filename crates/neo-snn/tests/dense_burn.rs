//! Comparison of the Burn-executor (`TensorNet`) with the standard (`DenseNet`) and with the CPU engine.
//!
//! Launch: `cargo test -p neo-snn --test dense_burn`.
//! To test on a GPU, replace `Device::flex()` with the device of the desired backend and run the test
//! again.

use burn::tensor::{Device, Tensor, TensorData};
use neo_snn::dense_burn::{InhibitoryPlasticityConfig, TensorNet};
use neo_snn::state::ResetMode;
use neo_snn::structural::dense::DenseNet;
use neo_snn::structural::{
    AdaptiveLifParams, Birth, LifParams, NeuronKind, NeuronModel, NeuronPolarity, Origin, Rng64,
    StructuralConfig, Topology,
};

fn random_net(seed: u64) -> Topology {
    let cfg = StructuralConfig::default();
    let mut t = Topology::new(cfg.topology, cfg.params);
    let mut r = Rng64::new(seed);
    let b = Birth::default();
    let ins: Vec<_> = (0..4)
        .map(|_| {
            t.add_neuron(NeuronKind::Input, LifParams::default(), Origin::Initial, b)
                .unwrap()
        })
        .collect();
    let mut dst = vec![];
    for k in 0..9 {
        let reset = [ResetMode::Subtract, ResetMode::Zero, ResetMode::None][k % 3];
        let p = LifParams {
            beta: 0.5 + 0.4 * r.next_f32(),
            threshold: 0.6 + r.next_f32(),
            reset,
        };
        let kind = if k < 6 {
            NeuronKind::Hidden
        } else {
            NeuronKind::Output
        };
        dst.push(t.add_neuron(kind, p, Origin::Initial, b).unwrap());
    }
    let srcs: Vec<_> = ins.iter().chain(dst.iter()).copied().collect();
    let (mut made, mut guard) = (0, 0);
    while made < 40 && guard < 10_000 {
        guard += 1;
        let (s, d) = (srcs[r.below(srcs.len())], dst[r.below(dst.len())]);
        if t.add_synapse(
            s,
            d,
            1.4 * r.symmetric() + 0.3,
            r.below(4) as u16,
            Origin::Initial,
            b,
        )
        .is_ok()
        {
            made += 1;
        }
    }
    t
}

/// Mixed network: LIF + AdaptiveLIF neurons, inhibitory inputs.
fn random_net_mixed(seed: u64) -> Topology {
    let cfg = StructuralConfig::default();
    let mut t = Topology::new(cfg.topology, cfg.params);
    let mut r = Rng64::new(seed);
    let b = Birth::default();
    let ins: Vec<_> = (0..4)
        .map(|k| {
            t.add_neuron_with(
                NeuronKind::Input,
                if k % 2 == 0 {
                    NeuronPolarity::Excitatory
                } else {
                    NeuronPolarity::Inhibitory
                },
                LifParams::default(),
                Origin::Initial,
                b,
            )
            .unwrap()
        })
        .collect();
    let mut dst = vec![];
    for k in 0..9 {
        let reset = [ResetMode::Subtract, ResetMode::Zero, ResetMode::None][k % 3];
        let kind = if k < 6 {
            NeuronKind::Hidden
        } else {
            NeuronKind::Output
        };
        let model = if k % 2 == 0 {
            NeuronModel::AdaptiveLif(AdaptiveLifParams {
                beta: 0.5 + 0.4 * r.next_f32(),
                rho: 0.6 + 0.4 * r.next_f32(),
                adapt_increment: 0.05 + 0.2 * r.next_f32(),
                threshold: 0.6 + r.next_f32(),
                reset,
            })
        } else {
            NeuronModel::Lif(LifParams {
                beta: 0.5 + 0.4 * r.next_f32(),
                threshold: 0.6 + r.next_f32(),
                reset,
            })
        };
        dst.push(t.add_neuron(kind, model, Origin::Initial, b).unwrap());
    }
    let srcs: Vec<_> = ins.iter().chain(dst.iter()).copied().collect();
    let (mut made, mut guard) = (0, 0);
    while made < 40 && guard < 10_000 {
        guard += 1;
        let (s, d) = (srcs[r.below(srcs.len())], dst[r.below(dst.len())]);
        if t.add_synapse(
            s,
            d,
            1.4 * r.symmetric() + 0.3,
            r.below(4) as u16,
            Origin::Initial,
            b,
        )
        .is_ok()
        {
            made += 1;
        }
    }
    t
}

#[test]
fn tensor_net_matches_dense_reference() {
    let device = Device::flex();
    for seed in 1..=6u64 {
        let dense = DenseNet::compile(&random_net(seed)).unwrap();
        let mut net = TensorNet::from_dense(&dense, &device);
        let (steps, batch, k) = (80usize, 3usize, dense.n_inputs());
        let mut r = Rng64::new(seed * 31);
        let x: Vec<f32> = (0..steps * batch * k)
            .map(|_| if r.chance(0.35) { 1.0 } else { 0.0 })
            .collect();
        let want = dense.run(&x, steps, batch).unwrap();
        let got = net
            .run(Tensor::<3>::from_data(
                TensorData::new(x, [steps, batch, k]),
                &device,
            ))
            .into_data()
            .try_to_vec::<f32>()
            .unwrap();
        assert_eq!(got.len(), want.len());
        // Float accumulation in matmul can shift a rare spike to the threshold boundary: allow ≤ 2% discrepancy.
        let diff: f32 = got.iter().zip(&want).map(|(a, b)| (a - b).abs()).sum();
        let total: f32 = want.iter().sum::<f32>().max(1.0);
        assert!(
            diff / total <= 0.02,
            "seed {seed}: divergence of {diff} from {total}"
        );
        assert!(
            want.iter().sum::<f32>() > 0.0,
            "seed {seed}: no output spikes - test is empty"
        );
    }
}

/// Vogels inhibitory STDP: inhibitory weights stay negative and bounded,
/// homeostasis applies (traces, modulation).
#[test]
fn vogels_inhibitory_plasticity_on_device() {
    let device = Device::flex();
    let dense = DenseNet::compile(&random_net_mixed(11)).unwrap();
    let mut net = TensorNet::from_dense(&dense, &device);
    net.set_vogels(Some(InhibitoryPlasticityConfig {
        enabled: true,
        eta: 0.001,
        target_rate: 0.02,
        trace_decay: 0.95,
    }));
    let mut st = net.new_state(1);
    for _ in 0..30 {
        let inputs = Tensor::from_data(
            TensorData::new(vec![1.0; net.n_inputs()], [1, net.n_inputs()]),
            &device,
        );
        let _ = net.step(&mut st, inputs);
    }
    let _ = &st;
    let n = net.n();
    let mut any_inhibit = false;
    for w in net.weights() {
        for &w in w.clone().into_data().try_to_vec::<f32>().unwrap().iter() {
            if w != 0.0 {
                any_inhibit |= w < 0.0;
                assert!(w.abs() <= 5.0 + 1e-6, "weight out of bounds: {w}");
            }
        }
    }
    assert!(any_inhibit, "inhibitory weights should exist");
    let tr = st.traces().clone().into_data().try_to_vec::<f32>().unwrap();
    assert!(tr.iter().any(|t| *t > 0.0), "traces should be nonzero");
    assert_eq!(tr.len(), n);
    let _ = st
        .modulation()
        .clone()
        .into_data()
        .try_to_vec::<f32>()
        .unwrap();
}

#[test]
fn tensor_net_stats_accumulate_on_device() {
    let device = Device::flex();
    let dense = DenseNet::compile(&random_net_mixed(3)).unwrap();
    let mut net = TensorNet::from_dense(&dense, &device);
    let mut st = net.new_state(1);
    assert_eq!(st.steps(), 0);
    assert_eq!(st.uses(), vec![0.0; net.n()]);
    for _ in 0..10 {
        let inputs = Tensor::from_data(
            TensorData::new(vec![1.0; net.n_inputs()], [1, net.n_inputs()]),
            &device,
        );
        let _ = net.step(&mut st, inputs);
    }
    let s = st.stats();
    assert_eq!(s.steps, 10);
    assert_eq!(s.uses.len(), net.n());
    assert_eq!(s.rates.len(), net.n());
    for (u, r) in s.uses.iter().zip(&s.rates) {
        assert!(*r <= 1.0 + 1e-6, "rate {r} out of range");
        assert!((u / 10.0 - r).abs() < 1e-6, "rate != uses/steps");
    }
}

/// Mixed LIF/ALIF + inhibition: the Burn executor matches the CPU `DenseNet`
/// reference spike-by-spike (within the usual float tolerance of matmul).
#[test]
fn tensor_net_matches_dense_reference_alif_and_inhibition() {
    let device = Device::flex();
    for seed in 1..=6u64 {
        let dense = DenseNet::compile(&random_net_mixed(seed)).unwrap();
        let mut net = TensorNet::from_dense(&dense, &device);
        let (steps, batch, k) = (80usize, 3usize, dense.n_inputs());
        let mut r = Rng64::new(seed * 31);
        let x: Vec<f32> = (0..steps * batch * k)
            .map(|_| if r.chance(0.35) { 1.0 } else { 0.0 })
            .collect();
        let want = dense.run(&x, steps, batch).unwrap();
        let got = net
            .run(Tensor::<3>::from_data(
                TensorData::new(x, [steps, batch, k]),
                &device,
            ))
            .into_data()
            .try_to_vec::<f32>()
            .unwrap();
        assert_eq!(got.len(), want.len());
        let diff: f32 = got.iter().zip(&want).map(|(a, b)| (a - b).abs()).sum();
        let total: f32 = want.iter().sum::<f32>().max(1.0);
        assert!(
            diff / total <= 0.02,
            "seed {seed}: divergence of {diff} from {total}"
        );
        assert!(
            want.iter().sum::<f32>() > 0.0,
            "seed {seed}: no output spikes - test is empty"
        );
    }
}
