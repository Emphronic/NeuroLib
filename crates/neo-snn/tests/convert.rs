//! ANN → SNN via public API (`neo_snn::prelude`).

use burn::module::Param;
use burn::nn::{Linear, LinearConfig};
use burn::tensor::{Device, Tensor};
use neo_snn::prelude::*;

fn dev() -> Device {
    Device::flex()
}

fn linear(w: &[f32], b: &[f32], i: usize, o: usize) -> Linear {
    let mut l = LinearConfig::new(i, o).init(&dev());
    l.weight = Param::from_tensor(Tensor::<1>::from_floats(w, &dev()).reshape([i, o]));
    l.bias = Some(Param::from_tensor(Tensor::<1>::from_floats(b, &dev())));
    l
}

/// Two-class problem “which of two inputs is greater”: the network is specified manually
/// (no training required), conversion is required to save the solution.
#[test]
fn hand_made_classifier_keeps_its_decisions_after_conversion() {
    // hidden = relu([a - b, b - a]); out = hidden (class 0: a > b, class 1: b > a).
    let fc1 = linear(&[1.0, -1.0, -1.0, 1.0], &[0.0, 0.0], 2, 2);
    let fc2 = linear(&[1.0, 0.0, 0.0, 1.0], &[0.0, 0.0], 2, 2);
    let ann = AnnNet::new().linear(&fc1).relu().linear(&fc2);

    let xs = [0.9f32, 0.1, 0.2, 0.8, 0.7, 0.3, 0.4, 0.6];
    let x = Act::D2(Tensor::<1>::from_floats(xs.as_slice(), &dev()).reshape([4, 2]));
    let labels = [0usize, 1, 0, 1];

    let snn = ann_to_snn(&ann, &[x.clone()], &ConvertConfig::default()).unwrap();
    let out = snn.run(&x, 100).unwrap();

    let acc = out.accuracy_over_time(&labels).unwrap();
    assert_eq!(acc.len(), 100);
    assert_eq!(*acc.last().unwrap(), 1.0);
    // The values are close to ANN: a - b = 0.8, 0.6, 0.4, 0.2 (accurate to 1/T).
    let got = out.output.to_vec("test").unwrap();
    let want = [0.8, 0.0, 0.0, 0.6, 0.4, 0.0, 0.0, 0.2];
    for (g, w) in got.iter().zip(want) {
        assert!((g - w).abs() < 0.05, "{g} vs {w}");
    }
}
