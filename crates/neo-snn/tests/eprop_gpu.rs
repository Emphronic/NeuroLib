//! Integration tests for the GPU e-prop trainer (`GpuEpropTrainer` on `TensorNet`).
//!
//! Run: `cargo test -p neo-snn --test eprop_gpu`

use burn::tensor::{Device, Tensor, TensorData};
use neo_snn::dense_burn::TensorNet;
use neo_snn::structural::dense::DenseNet;
use neo_snn::structural::eprop_gpu::{GpuEpropConfig, GpuEpropTrainer};
use neo_snn::structural::{Birth, LifParams, NeuronRole, Origin, StructuralConfig, Topology};

fn dev() -> Device {
    Device::flex()
}

fn vec_of(t: &Tensor<2>) -> Vec<f32> {
    t.clone().into_data().try_to_vec::<f32>().unwrap()
}

/// Small net: 2 inputs -> 1 output (LIF), `W = 1.0` everywhere.
fn small_net() -> DenseNet {
    let cfg = StructuralConfig::default();
    let mut t = Topology::new(cfg.topology, cfg.params);
    let b = Birth::default();
    let i0 = t
        .add_neuron(
            NeuronRole::Input,
            LifParams::new(0.9, 1.0),
            Origin::Initial,
            b,
        )
        .unwrap();
    let o = t
        .add_neuron(
            NeuronRole::Output,
            LifParams::new(0.9, 0.7),
            Origin::Initial,
            b,
        )
        .unwrap();
    t.add_synapse(i0, o, 1.0, 0, Origin::Initial, b).unwrap();
    DenseNet::compile(&t).unwrap()
}

fn inputs() -> Tensor<2> {
    // small_net has 1 input neuron (i0).
    Tensor::from_data(TensorData::new(vec![1.0], [1, 1]), &dev())
}

#[test]
fn gpu_eprop_traces_nonzero_and_weights_change() {
    let dense = small_net();
    let n = dense.n;
    let mut trainer = GpuEpropTrainer::new(
        TensorNet::from_dense(&dense, &dev()),
        GpuEpropConfig {
            learning_rate: 0.1,
            trace_decay: 0.95,
            targets: vec![0.0; n],
            feedback: Tensor::zeros([0, 0], &dev()),
            external_signals: None,
            use_external: false,
        },
    )
    .unwrap();
    let mut st = trainer.net().new_state(1);
    for _ in 0..2 {
        trainer.step(&mut st, inputs()).unwrap();
    }
    assert_eq!(trainer.steps(), 2);
    // Traces: e[i,j] = γ·e + spikes[j]·x[i,j] — nonzero once output spiked (step 1).
    let traces = vec_of(trainer.traces());
    assert_eq!(traces.len(), n * n);
    assert!(
        traces.iter().any(|t| *t != 0.0),
        "traces should accumulate on spikes"
    );
    // Weights change when d_j != 0 (output overfired: d_o = 1).
    let w0 = vec_of(&trainer.net().weights()[0]);
    trainer.step(&mut st, inputs()).unwrap();
    let w1 = vec_of(&trainer.net().weights()[0]);
    assert_ne!(w0, w1, "weights should change");
    assert_eq!(trainer.updates(), 3);
}

#[test]
fn gpu_eprop_traces_follow_eligibility() {
    // After 2 steps: the synapse i0→o has trace e[0,1] = spikes_o(1)·W = 1 (θ_o=0.7 fires).
    let dense = small_net();
    let n = dense.n;
    let mut trainer = GpuEpropTrainer::new(
        TensorNet::from_dense(&dense, &dev()),
        GpuEpropConfig {
            learning_rate: 0.1,
            trace_decay: 0.95,
            targets: vec![0.0; n],
            feedback: Tensor::zeros([0, 0], &dev()),
            external_signals: None,
            use_external: false,
        },
    )
    .unwrap();
    let mut st = trainer.net().new_state(1);
    trainer.step(&mut st, inputs()).unwrap(); // step 0: delivery delayed
    trainer.step(&mut st, inputs()).unwrap(); // step 1: delivery, o fires
    let traces = vec_of(trainer.traces());
    assert!(
        traces[1] > 0.0,
        "synapse i0→o trace should be positive, got {}",
        traces[1]
    );
}

#[test]
fn gpu_eprop_external_signals() {
    let dense = small_net();
    let n = dense.n;
    let mut trainer = GpuEpropTrainer::new(
        TensorNet::from_dense(&dense, &dev()),
        GpuEpropConfig {
            learning_rate: 0.1,
            trace_decay: 0.95,
            targets: vec![0.0; n],
            feedback: Tensor::zeros([0, 0], &dev()),
            external_signals: None,
            use_external: true,
        },
    )
    .unwrap();
    trainer.set_external_signals(Tensor::from_data(
        TensorData::new(vec![0.3, 0.1], [1, 2]),
        &dev(),
    ));
    let mut st = trainer.net().new_state(1);
    let _ = trainer.step(&mut st, Tensor::zeros([1, 1], &dev()));
    assert_eq!(trainer.steps(), 1);
    // Zero-input step with external signals: traces = γ·0 + spikes·0 = 0 (no delivery).
    assert!(
        trainer
            .traces()
            .clone()
            .into_data()
            .try_to_vec::<f32>()
            .unwrap()
            .iter()
            .all(|t| *t == 0.0)
    );
}
