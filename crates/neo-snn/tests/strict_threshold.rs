//! Strict threshold (`mem > threshold`, as in snnTorch), `step_with_beta` and
//! scalar amplitude factor of `Leaky`.
//!
//! Launch: `cargo test -p neo-snn --test strict_threshold`

use burn::tensor::Device;
use burn::tensor::Tensor;

use neo_snn::neuron::{Leaky, LeakyConfig};
use neo_snn::state::NeuronState;
use neo_snn::surrogate::Heaviside;

fn dev() -> Device {
    Device::flex()
}

fn t2(data: Vec<f32>, shape: [usize; 2]) -> Tensor<2> {
    Tensor::<1>::from_floats(data.as_slice(), &dev()).reshape(shape)
}

fn v(t: Tensor<2>) -> Vec<f32> {
    t.into_data().try_to_vec::<f32>().unwrap()
}

/// The membrane does not produce a spike exactly at the threshold (`input > 0` in snnTorch).
#[test]
fn membrane_exactly_at_threshold_does_not_spike() {
    let layer: Leaky = LeakyConfig::new(0.9).with_threshold(1.0).init(&dev());
    let state = NeuronState::zeros_mem_only(&dev(), 1, 3);
    // mem_new = 0·β + x = x, that is, 1.0 is exactly at the threshold.
    let (spike, next) = layer.step(t2(vec![1.0, 1.5, 0.5], [1, 3]), &state, &Heaviside);
    assert_eq!(v(spike), vec![0.0, 1.0, 0.0]);
    // Without a spike, the membrane does not reset.
    assert_eq!(v(next.mem), vec![1.0, 0.5, 0.5]);
}

/// `step_with_beta` with pre-calculated β must give the same as `step`.
#[test]
fn step_with_prepared_beta_matches_step() {
    let layer: Leaky = LeakyConfig::new(0.8).with_threshold(0.5).init(&dev());
    let beta = layer.beta_tensor([2, 2]);
    let mut a = NeuronState::zeros_mem_only(&dev(), 2, 2);
    let mut b = a.clone();
    for x in [
        vec![0.3, 0.6, 0.2, 0.9],
        vec![0.4, 0.1, 0.7, 0.0],
        vec![0.5, 0.5, 0.5, 0.5],
    ] {
        let (ya, na) = layer.step(t2(x.clone(), [2, 2]), &a, &Heaviside);
        let (yb, nb) = layer.step_with_beta(t2(x, [2, 2]), &b, &beta, &Heaviside);
        assert_eq!(v(ya), v(yb));
        assert_eq!(v(na.mem.clone()), v(nb.mem.clone()));
        a = na;
        b = nb;
    }
}

/// A fixed amplitude multiplier is applied by a scalar.
#[test]
fn fixed_graded_factor_scales_output() {
    let layer: Leaky = LeakyConfig::new(0.9)
        .with_threshold(1.0)
        .with_graded_spikes_factor(2.5)
        .init(&dev());
    let state = NeuronState::zeros_mem_only(&dev(), 1, 2);
    let (spike, _) = layer.step(t2(vec![1.5, 0.2], [1, 2]), &state, &Heaviside);
    assert_eq!(v(spike), vec![2.5, 0.0]);
}

/// The threshold being trained follows a tensor path and is also strict: with softplus(raw) = 1.0
/// membrane 1.0 does not produce a spike.
#[test]
fn learnable_threshold_path_is_also_strict() {
    let layer: Leaky = LeakyConfig::new(0.9)
        .with_threshold(1.0)
        .with_learn_threshold(true)
        .init(&dev());
    let state = NeuronState::zeros_mem_only(&dev(), 1, 2);
    let (spike, _) = layer.step(t2(vec![1.5, 0.5], [1, 2]), &state, &Heaviside);
    assert_eq!(v(spike), vec![1.0, 0.0]);
}
