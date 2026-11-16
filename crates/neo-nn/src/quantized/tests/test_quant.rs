//! `QuantizedLinear` tests: fused-path versus dequant-path, versus regular
//! `burn::nn::Linear`, weight layout, bias, policies, autodiff, size.

use burn::nn::{Linear, LinearConfig};
use burn::prelude::*;
use burn::tensor::{Device, Distribution};
use neo_quantization::config::RoundingMode;
use neo_quantization::dtype::DType;
use neo_quantization::pack::Calibration;
use neo_quantization::scheme::{PackedScheme, ScaleGranularity};
use neo_quantization::{QuantizeConfig, quantize};
use neo_synk::quant::KernelKind;

use crate::quantized::{ExecutionPolicy, QuantizedLinear, QuantizedLinearError};

// ───────────────────────── assistants ─────────────────────────

fn dev() -> Device {
    Device::flex()
}

/// Device with autodiff - for gradient tests.
fn adev() -> Device {
    Device::flex().autodiff()
}

fn randn<const D: usize>(shape: [usize; D]) -> Tensor<D> {
    Tensor::<D>::random(shape, Distribution::Normal(0.0, 1.0), &dev())
}

fn to_vec<const D: usize>(t: Tensor<D>) -> Vec<f32> {
    t.into_data().try_to_vec::<f32>().unwrap()
}

fn cfg(scheme: PackedScheme) -> QuantizeConfig {
    QuantizeConfig {
        scheme,
        rounding: RoundingMode::Nearest,
        symmetric: true,
        outlier_clip_percentile: 0.0,
    }
}

fn weights(len: usize, seed: u32) -> Vec<f32> {
    let mut s = seed.wrapping_mul(2654435761).wrapping_add(12345);
    (0..len)
        .map(|_| {
            s = s.wrapping_mul(1664525).wrapping_add(1013904223);
            ((s >> 8) as f32 / (1u32 << 24) as f32 - 0.5) * 2.0
        })
        .collect()
}

fn max_abs_diff(a: &[f32], b: &[f32]) -> f32 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f32::max)
}

fn rel_l2(got: &[f32], want: &[f32]) -> f32 {
    let num: f32 = got.iter().zip(want).map(|(a, b)| (a - b) * (a - b)).sum();
    let den: f32 = want.iter().map(|b| b * b).sum();
    (num / den.max(1e-12)).sqrt()
}

/// Enumeration of schemes: (name, scheme, in_features).
fn schemes() -> Vec<(&'static str, PackedScheme, usize)> {
    let g32 = ScaleGranularity::PerGroup { group_size: 32 };
    vec![
        (
            "int4 sym",
            PackedScheme::Int4(g32, Calibration::Symmetric),
            64,
        ),
        (
            "int4 asym",
            PackedScheme::Int4(g32, Calibration::Asymmetric),
            64,
        ),
        (
            "int8 per-channel",
            PackedScheme::Int8(ScaleGranularity::PerChannel, Calibration::Symmetric),
            64,
        ),
        (
            "int3 asym",
            PackedScheme::Int3(g32, Calibration::Asymmetric),
            64,
        ),
        ("int1", PackedScheme::Int1(ScaleGranularity::PerChannel), 64),
        (
            "nf4",
            PackedScheme::Nf4(ScaleGranularity::PerGroup { group_size: 64 }),
            64,
        ),
        ("ternary", PackedScheme::Ternary(g32), 64),
        ("float", PackedScheme::Float, 64),
        ("Q4_0", PackedScheme::Gguf(DType::Q4_0), 64),
        ("Q8_0", PackedScheme::Gguf(DType::Q8_0), 64),
        ("Q4_K", PackedScheme::Gguf(DType::Q4_K), 256),
        (
            "IQ4_NL (dense fallback)",
            PackedScheme::Gguf(DType::IQ4_NL),
            64,
        ),
    ]
}

// ───────────────────────── construction ─────────────────────────

#[test]
fn new_validates_weight_size() {
    let w = quantize(&weights(10, 1), &cfg(PackedScheme::Float), 0);
    let err = QuantizedLinear::new(w, 4, 3, None).unwrap_err();
    assert_eq!(
        err,
        QuantizedLinearError::WeightSize {
            expected: 12,
            got: 10
        }
    );
}

#[test]
fn new_validates_bias_size() {
    let w = quantize(&weights(12, 1), &cfg(PackedScheme::Float), 0);
    let bias = Tensor::<1>::zeros([5], &dev());
    let err = QuantizedLinear::new(w, 4, 3, Some(bias)).unwrap_err();
    assert_eq!(
        err,
        QuantizedLinearError::BiasSize {
            expected: 3,
            got: 5
        }
    );
}

#[test]
fn new_rejects_gguf_raw() {
    let w = neo_quantization::pack::PackedBuffer::GgufRaw {
        bytes: vec![0; 18],
        dtype: DType::Q4_0,
        numel: 32,
    };
    assert!(matches!(
        QuantizedLinear::new(w, 32, 1, None),
        Err(QuantizedLinearError::Unsupported(_))
    ));
}

#[test]
fn from_weights_validates_length() {
    let err = QuantizedLinear::from_weights(&weights(10, 1), 4, 3, None, &cfg(PackedScheme::Float))
        .unwrap_err();
    assert!(matches!(err, QuantizedLinearError::WeightSize { .. }));
}

// ───────────────────────── fused == dequantize ─────────────────────────

/// Main test: fused core and “deploy + matmul backend” give the same result
/// on all formats (the only difference is the float summation order).
#[test]
fn fused_matches_dequantize_for_every_scheme() {
    let (n, m) = (12, 5);
    for (name, scheme, k) in schemes() {
        let w = weights(n * k, 3);
        let fused = QuantizedLinear::from_weights(&w, k, n, None, &cfg(scheme))
            .unwrap()
            .with_policy(ExecutionPolicy::Fused);
        let deq = fused.clone().with_policy(ExecutionPolicy::Dequantize);

        let x = randn([m, k]);
        let a = to_vec(fused.forward(x.clone()));
        let b = to_vec(deq.forward(x));

        let scale = b.iter().fold(1e-3f32, |s, v| s.max(v.abs()));
        assert!(
            max_abs_diff(&a, &b) <= 5e-4 * scale + 1e-5,
            "{name}: fused vs dequantize differ by {} (scale {scale})",
            max_abs_diff(&a, &b)
        );
    }
}

#[test]
fn forward_keeps_leading_dims_for_any_rank() {
    let (k, n) = (64, 10);
    let q = QuantizedLinear::from_weights(
        &weights(n * k, 4),
        k,
        n,
        None,
        &QuantizeConfig::int4_per_group_32(),
    )
    .unwrap();

    assert_eq!(q.forward(randn([k])).dims(), [n]);
    assert_eq!(q.forward(randn([3, k])).dims(), [3, n]);
    assert_eq!(q.forward(randn([2, 3, k])).dims(), [2, 3, n]);
    assert_eq!(q.forward(randn([2, 3, 4, k])).dims(), [2, 3, 4, n]);
}

/// Running a 3D batch is equal to running the same matrix flattened in 2D.
#[test]
fn leading_dims_are_just_flattened_rows() {
    let (k, n) = (64, 8);
    let q = QuantizedLinear::from_weights(
        &weights(n * k, 5),
        k,
        n,
        None,
        &QuantizeConfig::int4_per_group_32(),
    )
    .unwrap();
    let x = randn([2, 3, k]);
    let a = to_vec(q.forward(x.clone()));
    let b = to_vec(q.forward(x.reshape([6, k])));
    assert!(max_abs_diff(&a, &b) < 1e-6);
}

#[test]
fn empty_batch_gives_empty_output() {
    let (k, n) = (32, 4);
    let q = QuantizedLinear::from_weights(
        &weights(n * k, 6),
        k,
        n,
        None,
        &QuantizeConfig::int8_per_group_32(),
    )
    .unwrap();
    let y = q.forward(Tensor::<2>::zeros([0, k], &dev()));
    assert_eq!(y.dims(), [0, n]);
}

#[test]
#[should_panic(expected = "in_features")]
fn forward_rejects_wrong_input_width() {
    let q = QuantizedLinear::from_weights(
        &weights(4 * 32, 7),
        32,
        4,
        None,
        &QuantizeConfig::int8_per_group_32(),
    )
    .unwrap();
    let _ = q.forward(randn([2, 31]));
}

// ───────────────────────── vs burn::nn::Linear ─────────────────────────

fn make_linear(d_in: usize, d_out: usize, bias: bool) -> Linear {
    LinearConfig::new(d_in, d_out).with_bias(bias).init(&dev())
}

/// Checks the layout: `Linear.weight` - `[in, out]`, and we quantize `[out, in]`.
/// If the transposition is mixed up, the result will differ significantly.
#[test]
fn from_linear_int8_matches_float_linear() {
    let (d_in, d_out) = (96, 40); // d_in != d_out to catch axis confusion
    let linear = make_linear(d_in, d_out, true);
    let q = QuantizedLinear::from_linear(&linear, &QuantizeConfig::int8_per_group_32());

    let x = randn([7, d_in]);
    let want = to_vec(linear.forward(x.clone()));
    let got = to_vec(q.forward(x));

    assert!(
        rel_l2(&got, &want) < 0.03,
        "relative L2 error {}",
        rel_l2(&got, &want)
    );
}

#[test]
fn from_linear_int4_error_is_bounded_and_larger_than_int8() {
    let (d_in, d_out) = (128, 48);
    let linear = make_linear(d_in, d_out, false);
    let x = randn([9, d_in]);
    let want = to_vec(linear.forward(x.clone()));

    let e8 = rel_l2(
        &to_vec(
            QuantizedLinear::from_linear(&linear, &QuantizeConfig::int8_per_group_32())
                .forward(x.clone()),
        ),
        &want,
    );
    let e4 = rel_l2(
        &to_vec(
            QuantizedLinear::from_linear(&linear, &QuantizeConfig::int4_per_group_32()).forward(x),
        ),
        &want,
    );
    assert!(e4 < 0.25, "int4 error {e4}");
    assert!(
        e8 < e4,
        "int8 ({e8}) must be more accurate than int4 ({e4})"
    );
}

#[test]
fn from_linear_without_bias_has_no_bias() {
    let linear = make_linear(32, 8, false);
    let q = QuantizedLinear::from_linear(&linear, &QuantizeConfig::int8_per_group_32());
    // zero input → zero output
    let y = to_vec(q.forward(Tensor::<2>::zeros([3, 32], &dev())));
    assert!(y.iter().all(|v| v.abs() < 1e-7));
}

#[test]
fn bias_is_added_after_matmul() {
    let (k, n) = (32, 4);
    let bias = Tensor::<1>::from_floats([1.0, -2.0, 3.0, 0.5], &dev());
    let q = QuantizedLinear::from_weights(
        &weights(n * k, 8),
        k,
        n,
        Some(bias),
        &QuantizeConfig::int8_per_group_32(),
    )
    .unwrap();

    // Zero input → exactly bias, in each line.
    let y = to_vec(q.forward(Tensor::<2>::zeros([3, k], &dev())));
    for row in 0..3 {
        assert_eq!(&y[row * n..(row + 1) * n], &[1.0, -2.0, 3.0, 0.5]);
    }
}

// ───────────────────────── policies ─────────────────────────

#[test]
fn auto_policy_picks_fused_or_dequantize() {
    let (k, n) = (64, 4);
    let build = |scheme| {
        QuantizedLinear::from_weights(&weights(n * k, 9), k, n, None, &cfg(scheme)).unwrap()
    };

    let fused = build(PackedScheme::Int4(
        ScaleGranularity::PerGroup { group_size: 32 },
        Calibration::Symmetric,
    ));
    assert_eq!(fused.kernel_kind(), KernelKind::IntDomain);
    assert_eq!(fused.effective_policy(), ExecutionPolicy::Fused);

    let dense = build(PackedScheme::Gguf(DType::IQ4_NL));
    assert_eq!(dense.kernel_kind(), KernelKind::DenseFallback);
    assert_eq!(dense.effective_policy(), ExecutionPolicy::Dequantize);

    // Explicit policy is not overridden.
    assert_eq!(
        dense.with_policy(ExecutionPolicy::Fused).effective_policy(),
        ExecutionPolicy::Fused
    );
}

// ───────────────────────── autodiff ─────────────────────────

/// `Dequantize` is the learning path: the gradient along `x` is equal to `Σ_o W[o, :]`.
#[test]
fn dequantize_policy_passes_gradient_to_input() {
    let device: Device = adev();
    let (m, k, n) = (3, 32, 5);
    let q = QuantizedLinear::from_weights(
        &weights(n * k, 10),
        k,
        n,
        None,
        &QuantizeConfig::int8_per_group_32(),
    )
    .unwrap()
    .with_policy(ExecutionPolicy::Dequantize);

    let x = Tensor::<2>::random([m, k], Distribution::Normal(0.0, 1.0), &device).require_grad();
    let grads = q.forward(x.clone()).sum().backward();
    let gx = to_vec(x.grad(&grads).expect("x must receive a gradient"));

    // dL/dx[i, c] = Σ_o W_hat[o, c]
    let w_hat = q.packed().to_f32();
    for c in 0..k {
        let expected: f32 = (0..n).map(|o| w_hat[o * k + c]).sum();
        for i in 0..m {
            assert!((gx[i * k + c] - expected).abs() < 1e-4, "i={i} c={c}");
        }
    }
}

/// `Fused` works on the Autodiff backend (the result is a graph sheet).
#[test]
fn fused_works_on_autodiff_backend() {
    let device: Device = Default::default();
    let (k, n) = (64, 6);
    let w = weights(n * k, 11);
    let q = QuantizedLinear::from_weights(&w, k, n, None, &QuantizeConfig::int4_per_group_32())
        .unwrap()
        .with_policy(ExecutionPolicy::Fused);
    let q_ref = QuantizedLinear::from_weights(&w, k, n, None, &QuantizeConfig::int4_per_group_32())
        .unwrap();

    let x = weights(2 * k, 12);
    let a = to_vec(q.forward(Tensor::<2>::from_data(
        TensorData::new(x.clone(), [2, k]),
        &device,
    )));
    let b = to_vec(q_ref.forward(Tensor::<2>::from_data(TensorData::new(x, [2, k]), &dev())));
    assert!(max_abs_diff(&a, &b) < 1e-6);
}

// ───────────────────────── size and Module ─────────────────────────

#[test]
fn int4_is_about_six_times_smaller_than_f32() {
    let (k, n) = (256, 64);
    let q = QuantizedLinear::from_weights(
        &weights(n * k, 13),
        k,
        n,
        None,
        &QuantizeConfig::int4_per_group_32(),
    )
    .unwrap();

    // 4 bits of codes + 32 bits of scale for 32 weights = 5 bits/weight.
    assert!(
        (q.bits_per_weight() - 5.0).abs() < 0.05,
        "{}",
        q.bits_per_weight()
    );
    assert!(q.compression_ratio() > 6.0, "{}", q.compression_ratio());
    assert!(q.weight_bytes() < n * k * 4 / 6);
}

#[test]
fn clone_shares_packed_weights() {
    let q = QuantizedLinear::from_weights(
        &weights(4 * 32, 14),
        32,
        4,
        None,
        &QuantizeConfig::int8_per_group_32(),
    )
    .unwrap();
    let c = q.clone();
    assert!(
        std::ptr::eq(q.packed(), c.packed()),
        "Arc<PackedBuffer> must be shared"
    );
}

#[test]
fn dequantize_weight_has_out_in_shape_and_matches_packed() {
    let (k, n) = (32, 6);
    let q = QuantizedLinear::from_weights(
        &weights(n * k, 15),
        k,
        n,
        None,
        &QuantizeConfig::int8_per_group_32(),
    )
    .unwrap();
    let w = q.dequantize_weight(&dev());
    assert_eq!(w.dims(), [n, k]);
    assert_eq!(to_vec(w), q.packed().to_f32());
}

/// The layer can be embedded in the `#[derive(Module)]` model.
#[test]
fn can_be_nested_in_a_module() {
    #[derive(Module, Debug)]
    struct Mlp {
        up: QuantizedLinear,
        down: QuantizedLinear,
    }

    let (d, h) = (32, 64);
    let mk = |i, o, seed| {
        QuantizedLinear::from_weights(
            &weights(i * o, seed),
            i,
            o,
            None,
            &QuantizeConfig::int8_per_group_32(),
        )
        .unwrap()
    };
    let mlp = Mlp {
        up: mk(d, h, 20),
        down: mk(h, d, 21),
    };
    let y = mlp.down.forward(mlp.up.forward(randn([3, d])));
    assert_eq!(y.dims(), [3, d]);
}
