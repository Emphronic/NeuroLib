//! Comparison of the Burn-executor (`TensorNet`) with the standard (`DenseNet`) and with the CPU engine.
//!
//! Launch: `cargo test -p neo-snn --test dense_burn`.
//! To test on a GPU, replace `Device::flex()` with the device of the desired backend and run the test
//! again.

use burn::tensor::{Device, Tensor, TensorData};
use neo_snn::dense_burn::TensorNet;
use neo_snn::state::ResetMode;
use neo_snn::structural::dense::DenseNet;
use neo_snn::structural::{
    Birth, LifParams, NeuronKind, Origin, Rng64, StructuralConfig, Topology,
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

#[test]
fn tensor_net_matches_dense_reference() {
    let device = Device::flex();
    for seed in 1..=6u64 {
        let dense = DenseNet::compile(&random_net(seed)).unwrap();
        let net = TensorNet::from_dense(&dense, &device);
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
