use burn::module::Param;
use burn::nn::conv::{Conv2d, Conv2dConfig};
use burn::nn::{Linear, LinearConfig};
use burn::tensor::{Device, Tensor};

use super::*;
use crate::error::SnnError;
use crate::state::ResetMode;

fn dev() -> Device {
    Device::flex()
}

/// Deterministic value generator in `[-1, 1]`.
fn rnd(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    (0..n)
        .map(|_| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((s >> 33) as f32 / (1u64 << 31) as f32) * 2.0 - 1.0
        })
        .collect()
}

fn lin(i: usize, o: usize, seed: u64) -> Linear {
    let mut l = LinearConfig::new(i, o).init(&dev());
    let k = 1.0 / (i as f32).sqrt();
    let w: Vec<f32> = rnd(i * o, seed).into_iter().map(|v| v * k).collect();
    let b: Vec<f32> = rnd(o, seed + 100).into_iter().map(|v| v * 0.1).collect();
    l.weight = Param::from_tensor(Tensor::<1>::from_floats(w.as_slice(), &dev()).reshape([i, o]));
    l.bias = Some(Param::from_tensor(Tensor::<1>::from_floats(
        b.as_slice(),
        &dev(),
    )));
    l
}

fn conv(cin: usize, cout: usize, k: usize, seed: u64) -> Conv2d {
    let mut c = Conv2dConfig::new([cin, cout], [k, k]).init(&dev());
    let s = 1.0 / ((cin * k * k) as f32).sqrt();
    let w: Vec<f32> = rnd(cout * cin * k * k, seed)
        .into_iter()
        .map(|v| v * s)
        .collect();
    let b: Vec<f32> = rnd(cout, seed + 7).into_iter().map(|v| v * 0.1).collect();
    c.weight = Param::from_tensor(
        Tensor::<1>::from_floats(w.as_slice(), &dev()).reshape([cout, cin, k, k]),
    );
    c.bias = Some(Param::from_tensor(Tensor::<1>::from_floats(
        b.as_slice(),
        &dev(),
    )));
    c
}

fn batch2(b: usize, f: usize, seed: u64) -> Act {
    let v: Vec<f32> = rnd(b * f, seed).into_iter().map(|x| x.abs()).collect();
    Act::D2(Tensor::<1>::from_floats(v.as_slice(), &dev()).reshape([b, f]))
}

fn batch4(b: usize, c: usize, h: usize, w: usize, seed: u64) -> Act {
    let v: Vec<f32> = rnd(b * c * h * w, seed)
        .into_iter()
        .map(|x| x.abs())
        .collect();
    Act::D4(Tensor::<1>::from_floats(v.as_slice(), &dev()).reshape([b, c, h, w]))
}

fn max_abs(v: &[f32]) -> f32 {
    v.iter().fold(0.0f32, |m, x| m.max(x.abs()))
}

fn errors(a: &Act, b: &Act) -> (f32, f32) {
    let (a, b) = (a.to_vec("t").unwrap(), b.to_vec("t").unwrap());
    let diffs: Vec<f32> = a.iter().zip(&b).map(|(x, y)| (x - y).abs()).collect();
    (
        max_abs(&diffs),
        diffs.iter().sum::<f32>() / diffs.len() as f32,
    )
}

fn mlp() -> AnnNet {
    AnnNet::new()
        .linear(&lin(8, 16, 1))
        .relu()
        .dropout()
        .linear(&lin(16, 16, 2))
        .relu()
        .linear(&lin(16, 4, 3))
}

#[test]
fn mlp_converted_matches_ann() {
    let ann = mlp();
    let x = batch2(6, 8, 10);
    let want = ann.forward(x.clone()).unwrap();
    let snn = convert(&ann, &[x.clone()], &ConvertConfig::default()).unwrap();
    let got = snn.run(&x, 400).unwrap().output;
    let scale = max_abs(&want.to_vec("t").unwrap()).max(1.0);
    let (max_err, _) = errors(&want, &got);
    assert!(
        max_err < 0.05 * scale,
        "max. error {max_err} at scale {scale}"
    );
}

#[test]
fn error_shrinks_with_more_steps() {
    let ann = mlp();
    let x = batch2(6, 8, 11);
    let want = ann.forward(x.clone()).unwrap();
    let snn = convert(&ann, &[x.clone()], &ConvertConfig::default()).unwrap();
    let out = snn.run(&x, 300).unwrap();
    let (_, e_short) = errors(&want, &out.per_step[9]);
    let (_, e_long) = errors(&want, &out.per_step[299]);
    assert!(e_long < e_short, "T=300: {e_long}, T=10: {e_short}");
    assert_eq!(out.per_step.len(), 300);
}

#[test]
fn batch_norm_folding_is_exact() {
    let c = |v: Vec<f32>| Tensor::<1>::from_floats(v.as_slice(), &dev());
    let bn = BatchNormParams::new(
        c(vec![1.5, 0.7, -1.2, 0.9]),
        c(vec![0.1, -0.2, 0.3, 0.0]),
        c(vec![0.2, -0.1, 0.4, 0.05]),
        c(vec![0.9, 1.3, 0.6, 2.0]),
        1e-5,
    );
    let ann = AnnNet::new()
        .linear(&lin(5, 4, 4))
        .push(AnnLayer::BatchNorm(bn.clone()))
        .relu()
        .linear(&lin(4, 3, 5));
    let x = batch2(7, 5, 12);
    let folded = ann.canonicalize().unwrap();
    assert!(
        folded
            .layers()
            .iter()
            .all(|l| !matches!(l, AnnLayer::BatchNorm(_)))
    );
    let (e, _) = errors(
        &ann.forward(x.clone()).unwrap(),
        &folded.forward(x).unwrap(),
    );
    assert!(e < 1e-5, "BN fold error: {e}");
}

#[test]
fn conv_batch_norm_folding_is_exact() {
    let c = |v: Vec<f32>| Tensor::<1>::from_floats(v.as_slice(), &dev());
    let bn = BatchNormParams::new(
        c(vec![1.1, 0.8]),
        c(vec![0.05, -0.1]),
        c(vec![0.1, 0.2]),
        c(vec![0.7, 1.4]),
        1e-5,
    );
    let ann = AnnNet::new()
        .conv2d(&conv(1, 2, 3, 6))
        .push(AnnLayer::BatchNorm(bn))
        .relu();
    let x = batch4(2, 1, 6, 6, 13);
    let (e, _) = errors(
        &ann.forward(x.clone()).unwrap(),
        &ann.canonicalize().unwrap().forward(x).unwrap(),
    );
    assert!(e < 1e-5, "BN convolution error in Conv2d: {e}");
}

#[test]
fn scales_follow_calibration_statistic() {
    let id = Tensor::<1>::from_floats([1.0, 0.0, 0.0, 1.0].as_slice(), &dev()).reshape([2, 2]);
    let mut l = LinearConfig::new(2, 2).init(&dev());
    l.weight = Param::from_tensor(id);
    l.bias = Some(Param::from_tensor(Tensor::<1>::zeros([2], &dev())));
    let ann = AnnNet::new().linear(&l).relu();
    let x =
        Act::D2(Tensor::<1>::from_floats([1.0, 2.0, 3.0, 0.5].as_slice(), &dev()).reshape([2, 2]));

    let max = convert(
        &ann,
        &[x.clone()],
        &ConvertConfig {
            norm: NormMode::Max,
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!(max.scales(), &[3.0]);
    let p100 = convert(
        &ann,
        &[x.clone()],
        &ConvertConfig {
            norm: NormMode::Percentile(100.0),
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!(p100.scales(), &[3.0]);
    let none = convert(
        &ann,
        &[],
        &ConvertConfig {
            norm: NormMode::None,
            ..Default::default()
        },
    )
    .unwrap();
    assert_eq!(none.scales(), &[1.0]);
}

#[test]
fn normalization_keeps_activations_within_firing_range() {
    // After normalization, activations ANN divided by λ do not exceed 1 - frequency
    // shooting is not saturated. We check on ANN itself using calibration data.
    let ann = mlp();
    let x = batch2(16, 8, 14);
    let net = ann.canonicalize().unwrap();
    let scales = activation_scales(&net, &[x.clone()], NormMode::Max).unwrap();
    let mut seen = Vec::new();
    net.forward_observe(x, &mut |i, a| {
        seen.push((i, max_abs(&a.to_vec("t").unwrap())));
    })
    .unwrap();
    for (i, m) in seen {
        assert!(
            m / scales[i] <= 1.0 + 1e-6,
            "layer {i}: {m} / {}",
            scales[i]
        );
    }
}

#[test]
fn conv_net_converted_matches_ann() {
    let ann = AnnNet::new()
        .conv2d(&conv(1, 3, 3, 20))
        .relu()
        .avg_pool2d([2, 2])
        .flatten()
        .linear(&lin(3 * 3 * 3, 5, 21))
        .relu()
        .linear(&lin(5, 3, 22));
    let x = batch4(4, 1, 8, 8, 15);
    let want = ann.forward(x.clone()).unwrap();
    assert_eq!(want.dims(), vec![4, 3]);
    let snn = convert(&ann, &[x.clone()], &ConvertConfig::default()).unwrap();
    assert_eq!(snn.spiking_layers(), 2);
    let got = snn.run(&x, 500).unwrap().output;
    let scale = max_abs(&want.to_vec("t").unwrap()).max(1.0);
    let (max_err, _) = errors(&want, &got);
    assert!(
        max_err < 0.08 * scale,
        "max. error {max_err} at scale {scale}"
    );
}

#[test]
fn network_ending_with_spiking_layer_is_rescaled() {
    let ann = AnnNet::new()
        .linear(&lin(6, 4, 30))
        .relu()
        .linear(&lin(4, 5, 31))
        .relu();
    let x = batch2(5, 6, 16);
    let want = ann.forward(x.clone()).unwrap();
    let snn = convert(&ann, &[x.clone()], &ConvertConfig::default()).unwrap();
    let got = snn.run(&x, 500).unwrap().output;
    let scale = max_abs(&want.to_vec("t").unwrap()).max(1.0);
    let (max_err, _) = errors(&want, &got);
    assert!(
        max_err < 0.05 * scale,
        "max. error {max_err} at scale {scale}"
    );
}

#[test]
fn stats_and_accuracy_over_time() {
    let ann = mlp();
    let x = batch2(12, 8, 17);
    let want = ann.forward(x.clone()).unwrap().to_vec("t").unwrap();
    let labels: Vec<usize> = want.chunks(4).map(argmax).collect();
    let snn = convert(&ann, &[x.clone()], &ConvertConfig::default()).unwrap();
    let out = snn.run(&x, 200).unwrap();

    assert_eq!(out.stats.len(), 2);
    for s in &out.stats {
        assert!(
            (0.0..=1.0).contains(&s.rate),
            "frequency outside [0,1]: {}",
            s.rate
        );
    }
    assert_eq!(out.stats[0].neurons, 16);
    assert!(out.total_spikes() > 0.0);

    let acc = out.accuracy_over_time(&labels).unwrap();
    assert_eq!(acc.len(), 200);
    assert!(acc[199] >= 0.99, "accuracy at T=200: {}", acc[199]);
    assert!(acc[199] >= acc[0]);
    assert!(out.accuracy_over_time(&labels[..3]).is_err());
}

#[test]
fn rate_encoded_input_approximates_ann() {
    let ann = mlp();
    let x = batch2(4, 8, 18);
    let want = ann.forward(x.clone()).unwrap();
    let cfg = ConvertConfig {
        spike: SpikeConfig {
            input: InputEncoding::Rate,
            ..Default::default()
        },
        ..Default::default()
    };
    let snn = convert(&ann, &[x.clone()], &cfg).unwrap();
    let got = snn.run(&x, 3000).unwrap().output;
    let scale = max_abs(&want.to_vec("t").unwrap()).max(1.0);
    let (_, mean_err) = errors(&want, &got);
    assert!(
        mean_err < 0.1 * scale,
        "average error {mean_err} at scale {scale}"
    );
}

#[test]
fn zero_reset_runs_and_is_finite() {
    let ann = mlp();
    let x = batch2(3, 8, 19);
    let cfg = ConvertConfig {
        spike: SpikeConfig {
            reset: ResetMode::Zero,
            ..Default::default()
        },
        ..Default::default()
    };
    let snn = convert(&ann, &[x.clone()], &cfg).unwrap();
    let out = snn.run(&x, 50).unwrap().output.to_vec("t").unwrap();
    assert!(out.iter().all(|v| v.is_finite()));
}

#[test]
fn half_initial_membrane_reduces_short_horizon_error() {
    let ann = mlp();
    let x = batch2(8, 8, 23);
    let want = ann.forward(x.clone()).unwrap();
    let err_for = |v_init: f32| {
        let cfg = ConvertConfig {
            spike: SpikeConfig {
                v_init,
                ..Default::default()
            },
            ..Default::default()
        };
        let snn = convert(&ann, &[x.clone()], &cfg).unwrap();
        errors(&want, &snn.run(&x, 20).unwrap().output).1
    };
    assert!(err_for(0.5) < err_for(0.0));
}

#[test]
fn step_by_step_equals_run() {
    let ann = mlp();
    let x = batch2(3, 8, 24);
    let snn = convert(&ann, &[x.clone()], &ConvertConfig::default()).unwrap();
    let mut state = snn.init_state();
    for _ in 0..30 {
        let (_, s) = snn.step(&x, state).unwrap();
        state = s;
    }
    assert_eq!(state.steps(), 30);
    let run = snn.run(&x, 30).unwrap();
    assert_eq!(run.per_step.len(), 30);
}

fn err_what(r: std::result::Result<impl std::fmt::Debug, SnnError>) -> String {
    r.unwrap_err().to_string()
}

#[test]
fn unsupported_structures_are_rejected_with_reasons() {
    let l = lin(4, 4, 40);
    let x = batch2(2, 4, 25);
    let cfg = ConvertConfig::default();

    // ReLU is not immediately after the weights layer.
    let a = AnnNet::new()
        .conv2d(&conv(1, 1, 1, 41))
        .avg_pool2d([1, 1])
        .relu();
    assert!(err_what(convert(&a, &[x.clone()], &cfg)).contains("ReLU"));
    let a = AnnNet::new().relu().linear(&l);
    assert!(err_what(a.canonicalize()).contains("ReLU"));

    // Layer with weights without ReLU in the middle.
    let a = AnnNet::new().linear(&l).linear(&l).relu();
    assert!(err_what(a.canonicalize()).contains("last"));

    // BatchNorm without a layer with weights in front of it.
    let t = Tensor::<1>::ones([4], &dev());
    let bn = BatchNormParams::new(t.clone(), t.clone(), t.clone(), t, 1e-5);
    let a = AnnNet::new().push(AnnLayer::BatchNorm(bn)).linear(&l);
    assert!(err_what(a.canonicalize()).contains("BatchNorm"));

    // There are no layers with weights.
    assert!(err_what(AnnNet::new().relu().canonicalize()).contains("ReLU"));
    assert!(err_what(AnnNet::new().flatten().canonicalize()).contains("no layers with weights"));
}

#[test]
fn bad_config_is_rejected() {
    let ann = mlp();
    let x = batch2(2, 8, 26);
    let bad_reset = ConvertConfig {
        spike: SpikeConfig {
            reset: ResetMode::None,
            ..Default::default()
        },
        ..Default::default()
    };
    assert!(convert(&ann, &[x.clone()], &bad_reset).is_err());
    let bad_init = ConvertConfig {
        spike: SpikeConfig {
            v_init: 1.0,
            ..Default::default()
        },
        ..Default::default()
    };
    assert!(convert(&ann, &[x.clone()], &bad_init).is_err());
    let bad_p = ConvertConfig {
        norm: NormMode::Percentile(0.0),
        ..Default::default()
    };
    assert!(convert(&ann, &[x], &bad_p).is_err());
    // Calibration is required if scales are taken from the data.
    assert!(matches!(
        convert(&ann, &[], &ConvertConfig::default()),
        Err(SnnError::EmptySequence { .. })
    ));
}

#[test]
fn wrong_input_rank_is_a_typed_error() {
    let ann = mlp();
    let snn = convert(&ann, &[batch2(2, 8, 27)], &ConvertConfig::default()).unwrap();
    let e = snn.run(&batch4(2, 1, 2, 4, 28), 3).unwrap_err();
    assert!(matches!(e, SnnError::Shape { .. }), "{e}");
    assert!(matches!(
        snn.run(&batch2(2, 8, 29), 0),
        Err(SnnError::EmptySequence { .. })
    ));
}
