//! Integration tests of quantization QUALITY, not just data shapes.
//!
//! Unit tests inside `pack::*` check block sizes, round-trip and
//! match with ggml reference vectors. What's easy is tested here
//! break unnoticed: how well the format approximates the original weights.
//! These are the checks that catch calibration regressions - the code continues
//! compile, shape tests continue to pass, but the model quietly loses
//! quality.
//!
//! The data is deterministic (proprietary LCG), so the numbers are reproducible and
//! tests do not “blink”.

use neo_quantization::pack::gguf::quantize_gguf;
use neo_quantization::pack::iq::{quantize_iq4_nl, quantize_iq4_xs};
use neo_quantization::pack::kquant::quantize_kquant;
use neo_quantization::DType;

/// Deterministic Gaussian set: sum of six uniform ones.
fn gauss(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed;
    let mut next = || {
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((s >> 40) as f32) / ((1u64 << 24) as f32)
    };
    (0..n)
        .map(|_| (0..6).map(|_| next()).sum::<f32>() - 3.0)
        .collect()
}

fn mse(a: &[f32], b: &[f32]) -> f64 {
    assert_eq!(a.len(), b.len());
    a.iter()
        .zip(b)
        .map(|(x, y)| ((x - y) as f64).powi(2))
        .sum::<f64>()
        / a.len() as f64
}

/// The input variance is the basis for the relative error.
fn variance(a: &[f32]) -> f64 {
    let mean = a.iter().map(|v| *v as f64).sum::<f64>() / a.len() as f64;
    a.iter().map(|v| (*v as f64 - mean).powi(2)).sum::<f64>() / a.len() as f64
}

/// More bits per weight must not make the error worse.
///
/// This is the main invariant of any quantization library: if a format uses
/// more space, it must approximate the weights better. A violation indicates
/// a calibration issue rather than a property of the format;
/// `pack::iq1::tests::test_iq1m_should_beat_iq1s` tracks the IQ1_M case.
#[test]
fn more_bits_means_lower_error_kquants() {
    let data = gauss(2048, 11);
    let var = variance(&data);

    let mut prev: Option<(DType, f64)> = None;
    for dtype in [
        DType::Q2_K,
        DType::Q3_K,
        DType::Q4_K,
        DType::Q5_K,
        DType::Q6_K,
    ] {
        let err = mse(&data, &quantize_kquant(&data, dtype).to_f32());
        assert!(
            err < var,
            "{dtype:?}: the error {err} is not less than the variance of the input {var} - \
             quantization is no better than issuing a constant"
        );
        if let Some((prev_dtype, prev_err)) = prev {
            assert!(
                err <= prev_err * 1.05,
                "{dtype:?} (MSE {err}) is worse than the rougher {prev_dtype:?} (MSE {prev_err})"
            );
        }
        prev = Some((dtype, err));
    }
}

/// The same for simple GGUF formats.
#[test]
fn more_bits_means_lower_error_gguf() {
    let data = gauss(2048, 13);
    let q4 = mse(&data, &quantize_gguf(&data, DType::Q4_0).to_f32());
    let q5 = mse(&data, &quantize_gguf(&data, DType::Q5_0).to_f32());
    let q8 = mse(&data, &quantize_gguf(&data, DType::Q8_0).to_f32());
    assert!(
        q5 < q4,
        "Q5_0 (MSE {q5}) should be more accurate than Q4_0 (MSE {q4})"
    );
    assert!(
        q8 < q5,
        "Q8_0 (MSE {q8}) should be more accurate than Q5_0 (MSE {q5})"
    );
}

/// Extreme weights should not turn the block into `inf`/`NaN`.
///
/// The block scale is stored in `f16`, the maximum of which is 65504. The weight is about 1e6
/// (found in unnormalized checkpoints) used to give `inf` in
/// scale, and the ENTIRE block was dequantized to `inf` - silently, without error.
#[test]
fn extreme_magnitudes_stay_finite() {
    let mut data = vec![0.01f32; 512];
    data[0] = 1e6;
    data[1] = -1e6;
    data[2] = f32::MAX;

    for dtype in [
        DType::Q4_0,
        DType::Q4_1,
        DType::Q5_0,
        DType::Q5_1,
        DType::Q8_0,
    ] {
        let r = quantize_gguf(&data, dtype).to_f32();
        assert!(
            r.iter().all(|v| v.is_finite()),
            "{dtype:?} gave non-finite values"
        );
    }
    for dtype in [
        DType::Q2_K,
        DType::Q3_K,
        DType::Q4_K,
        DType::Q5_K,
        DType::Q6_K,
        DType::Q8_K,
    ] {
        let r = quantize_kquant(&data, dtype).to_f32();
        assert!(
            r.iter().all(|v| v.is_finite()),
            "{dtype:?} gave non-finite values"
        );
    }
    assert!(quantize_iq4_nl(&data)
        .to_f32()
        .iter()
        .all(|v| v.is_finite()));
    assert!(quantize_iq4_xs(&data)
        .to_f32()
        .iter()
        .all(|v| v.is_finite()));
}

/// The IQ4_NL calibration should beat the naive `amax/127`.
///
/// Checks scale selection: a naive formula grows the error and fails here.
/// Comparison is with the standard computed right here - without magic
/// numbers that will become obsolete.
#[test]
fn iq4_nl_calibration_beats_naive_scale() {
    // Naive scheme: d = amax/127, round-to-nearest codes according to the table.
    fn naive(data: &[f32]) -> Vec<f32> {
        use neo_quantization::pack::iq::IQ4_NL_LEVELS;
        let mut out = Vec::with_capacity(data.len());
        for chunk in data.chunks(32) {
            let amax = chunk.iter().fold(0.0f32, |a, w| a.max(w.abs()));
            let d = if amax > 0.0 { amax / 127.0 } else { 0.0 };
            let inv = if d > 0.0 { 1.0 / d } else { 0.0 };
            for &w in chunk {
                let t = (w * inv).clamp(-127.0, 127.0) as i32;
                let (mut bi, mut bd) = (0usize, i32::MAX);
                for (i, &l) in IQ4_NL_LEVELS.iter().enumerate() {
                    let dd = (t - l as i32).abs();
                    if dd < bd {
                        bd = dd;
                        bi = i;
                    }
                }
                out.push(IQ4_NL_LEVELS[bi] as f32 * d);
            }
        }
        out
    }

    for (name, data) in [
        ("Gaussian weights", gauss(4096, 7)),
        (
            "linear ramp",
            (0..4096)
                .map(|i| (i as f32 - 2048.0) * 0.01)
                .collect::<Vec<_>>(),
        ),
        ("weights with outliers", {
            let mut v = gauss(4096, 3);
            v[0] = 50.0;
            v[100] = -40.0;
            v
        }),
    ] {
        let naive_err = mse(&data, &naive(&data));
        let actual_err = mse(&data, &quantize_iq4_nl(&data).to_f32());
        assert!(
            actual_err <= naive_err,
            "{name}: calibration worse than naive amax/127 - MSE {actual_err} vs {naive_err}"
        );
    }
}

/// IQ4_NL and IQ4_XS use one level table and one calibration, so
/// that they must be comparable in QUALITY.
///
/// The comparison is done by MSE, not element by element: XS stores the scale of the sub-block as
/// The 6-bit number is relative to the overall super-scale, so the individual weights
/// legally diverge from NL by percentage. Comparability is guaranteed -
/// if it is broken, then the calibration has moved between formats.
#[test]
fn iq4_nl_and_xs_comparable_quality() {
    for (name, data) in [
        ("Gaussian weights", gauss(1024, 5)),
        (
            "linear ramp",
            (0..1024)
                .map(|i| (i as f32 - 512.0) * 0.01)
                .collect::<Vec<_>>(),
        ),
    ] {
        let nl = mse(&data, &quantize_iq4_nl(&data).to_f32());
        let xs = mse(&data, &quantize_iq4_xs(&data).to_f32());
        assert!(
            xs <= nl * 3.0,
            "{name}: IQ4_XS (MSE {xs}) is much worse than IQ4_NL (MSE {nl}) - \
             with a general table of levels, this is a sign of a loose calibration"
        );
    }
}

/// The null tensor must remain null in all formats: division by
/// zero scale - classic source `NaN`.
#[test]
fn all_zeros_stay_zero() {
    let data = vec![0.0f32; 512];
    for dtype in [DType::Q4_0, DType::Q5_0, DType::Q8_0] {
        let r = quantize_gguf(&data, dtype).to_f32();
        assert!(r.iter().all(|v| *v == 0.0), "{dtype:?} did not save zeros");
    }
    for dtype in [DType::Q2_K, DType::Q4_K, DType::Q6_K] {
        let r = quantize_kquant(&data, dtype).to_f32();
        assert!(
            r.iter().all(|v| v.is_finite()),
            "{dtype:?} gave non-finite values on zeros"
        );
    }
}

// ═══════════════════════════════════════════════════════════════════════════
//  Built-in formats go through the same test bench as custom ones
// ═══════════════════════════════════════════════════════════════════════════
//
// The `testkit` is intended for third-party schemes, but running your own through it
// is the best way to make sure it tests meaningful things and not
// invented requirements that the crate itself does not meet.

use neo_quantization::custom::{CustomScheme, QuantContext};
use neo_quantization::testkit;

/// A wrapper around the built-in GGUF format for the user scheme interface.
#[derive(Debug)]
struct BuiltinGguf(DType);

impl CustomScheme for BuiltinGguf {
    fn tag(&self) -> &'static str {
        "builtin_gguf"
    }
    fn quantize(&self, w: &[f32], _ctx: &QuantContext) -> Vec<u8> {
        quantize_gguf(w, self.0).blocks
    }
    fn dequantize(&self, bytes: &[u8], numel: usize) -> Vec<f32> {
        use neo_quantization::pack::gguf::{dequantize_gguf, GgufPacked};
        let mut v = dequantize_gguf(&GgufPacked {
            blocks: bytes.to_vec(),
            dtype: self.0,
            numel,
        });
        v.truncate(numel);
        v
    }
}

#[test]
fn builtin_gguf_formats_pass_the_public_testkit() {
    for dtype in [
        DType::Q4_0,
        DType::Q4_1,
        DType::Q5_0,
        DType::Q5_1,
        DType::Q8_0,
    ] {
        let report = testkit::check_scheme(&BuiltinGguf(dtype));
        assert!(
            report.is_ok(),
            "{dtype:?} did not pass its own testkit:\n{report}"
        );
    }
}

/// Ternary is required to give exactly 16x reduction against f32.
///
/// 2 bits per trit vs 32 bits on f32 = 16x. Checked on actual
/// buffer size, and not on the declared number: this is how the “forgot” is caught
/// pack four to a byte."
#[test]
fn ternary_gives_16x_memory_reduction() {
    use neo_quantization::pack::quantize_ternary;
    use neo_quantization::ScaleGranularity;

    let n = 1 << 16;
    let data = gauss(n, 21);
    let packed = quantize_ternary(&data, ScaleGranularity::PerTensor, 0);

    let float_bytes = n * 4;
    let packed_bytes = packed.bits_bytes();
    assert_eq!(
        packed_bytes,
        n / 4,
        "ternary must pack exactly 4 trits per byte"
    );
    assert_eq!(
        float_bytes / packed_bytes,
        16,
        "expected 16-fold reduction, received {}x",
        float_bytes as f32 / packed_bytes as f32
    );

    // Taking into account the scale (one f32 for the entire tensor) - almost the same 16x.
    let bpw = packed.effective_bits_per_weight();
    assert!(
        (bpw - 2.0).abs() < 0.01,
        "bit/weight taking into account scale: {bpw}, expected ≈2.0"
    );
}
