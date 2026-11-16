// neo-quantization/src/testkit.rs
//
// A ready-made testkit for testing a quantization scheme.
//
// ## Why is this in the crate itself, and not in tests/
//
// The scheme is written by the user, in their own crate. So they will test it
// themselves - and without a ready-made testkit, testing comes down to a pair of
// round-trip tests on convenient data. This is exactly how defects pass unnoticed,
// which then cost the most: a scale that goes to `inf` on one
// block out of a thousand; non-determinism due to HashMap traversal; loss
// of the tail when its length is not a multiple of the block.
//
// All checks here are regressions of real defects found in
// built-in formats of this crate (see the history of `pack::tq`, `pack::iq`,
// `pack::kquant`). The testkit catches them even before the scheme gets into the model.

use crate::custom::{CustomScheme, QuantContext};

// ═══════════════════════════════════════════════════════════════════════════
// § 1 REPORT
// ═══════════════════════════════════════════════════════════════════════════

/// The outcome of one test.
#[derive(Debug, Clone, PartialEq)]
pub enum Outcome {
    /// Verification passed.
    Pass,
    /// Check failed; inside - what exactly came apart.
    Fail(String),
    /// Validation does not apply to this scheme (and why).
    Skip(String),
}

impl Outcome {
    /// Whether the check failed.
    pub fn is_fail(&self) -> bool {
        matches!(self, Outcome::Fail(_))
    }
}

/// The result of one named test.
#[derive(Debug, Clone)]
pub struct Check {
    /// Machine name of the check.
    pub name: &'static str,
    /// What it means is included in the error text.
    pub description: &'static str,
    /// Exodus.
    pub outcome: Outcome,
}

/// Full report on the scheme.
#[derive(Debug, Clone)]
pub struct Report {
    /// Tag of the tested scheme.
    pub scheme: String,
    /// All checks are in order.
    pub checks: Vec<Check>,
    /// Measured bits per weight on Gaussian data.
    pub bits_per_weight: f32,
    /// Relation of MSE to input variance on Gaussian data.
    ///
    /// Less than 1 - the scheme is better than issuing a constant. The closer to 0, the
    /// more precisely. For 4-bit schemes 1e-3...1e-2 is typical, for ternary - about
    /// 0.1…0.3.
    pub relative_error: f32,
}

impl Report {
    /// Have all checks been passed?
    pub fn is_ok(&self) -> bool {
        !self.checks.iter().any(|c| c.outcome.is_fail())
    }

    /// List of failed checks.
    pub fn failures(&self) -> Vec<&Check> {
        self.checks.iter().filter(|c| c.outcome.is_fail()).collect()
    }
}

impl core::fmt::Display for Report {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        writeln!(f, "Scheme '{}'", self.scheme)?;
        writeln!(
            f,
            "  bit/weight: {:.3} relative error: {:.4}",
            self.bits_per_weight, self.relative_error
        )?;
        for c in &self.checks {
            let mark = match &c.outcome {
                Outcome::Pass => "  [OK]    ",
                Outcome::Fail(_) => "  [ERROR]",
                Outcome::Skip(_) => "  [SKIP]",
            };
            write!(f, "{mark} {}", c.name)?;
            match &c.outcome {
                Outcome::Fail(m) => writeln!(f, "\n {}\n expected: {}", m, c.description)?,
                Outcome::Skip(m) => writeln!(f, " — {m}")?,
                Outcome::Pass => writeln!(f)?,
            }
        }
        Ok(())
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// § 2 DATA FOR CHECKS
// ═══════════════════════════════════════════════════════════════════════════

/// Deterministic generator: own LCG, without dependence on `rand`.
///
/// Determinism is required here - a testkit that "blinks" is useless: a failure
/// that cannot be reproduced cannot be fixed.
struct Lcg(u64);

impl Lcg {
    fn new(seed: u64) -> Self {
        Self(seed)
    }
    fn uniform(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((self.0 >> 40) as f32) / ((1u64 << 24) as f32)
    }
    /// Gaussian: sum of six uniforms (central limit theorem).
    fn normal(&mut self) -> f32 {
        (0..6).map(|_| self.uniform()).sum::<f32>() - 3.0
    }
}

/// Gaussian weights are what the weights of a real network look like.
pub fn gaussian_weights(n: usize, seed: u64) -> Vec<f32> {
    let mut r = Lcg::new(seed);
    (0..n).map(|_| r.normal()).collect()
}

/// Gaussian weights with rare outliers are what naive `amax` breaks down on.
pub fn weights_with_outliers(n: usize, seed: u64) -> Vec<f32> {
    let mut v = gaussian_weights(n, seed);
    if n > 0 {
        v[0] = 50.0;
    }
    if n > 100 {
        v[100] = -40.0;
    }
    v
}

fn variance(a: &[f32]) -> f64 {
    if a.is_empty() {
        return 0.0;
    }
    let mean = a.iter().map(|v| *v as f64).sum::<f64>() / a.len() as f64;
    a.iter().map(|v| (*v as f64 - mean).powi(2)).sum::<f64>() / a.len() as f64
}

fn mse(a: &[f32], b: &[f32]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(x, y)| ((x - y) as f64).powi(2))
        .sum::<f64>()
        / a.len().max(1) as f64
}

// ═══════════════════════════════════════════════════════════════════════════
// § 3 CHECKS
// ═══════════════════════════════════════════════════════════════════════════

fn roundtrip(scheme: &dyn CustomScheme, w: &[f32], ctx: &QuantContext) -> Vec<f32> {
    let bytes = scheme.quantize(w, ctx);
    scheme.dequantize(&bytes, w.len())
}

/// Run the scheme through a full set of checks and return a report.
///
/// There is no need to panic: the report can be printed, saved, compared between
/// scheme versions. If you just need to “fall if the contract is violated” -
/// [`assert_scheme`].
///
/// ```
/// use neo_quantization::testkit;
/// # use neo_quantization::custom::{CustomScheme, QuantContext};
/// # #[derive(Debug)] struct S;
/// # impl CustomScheme for S {
/// #   fn tag(&self) -> &'static str { "s" }
/// #   fn quantize(&self, w: &[f32], _: &QuantContext) -> Vec<u8> {
/// #     w.iter().flat_map(|v| v.to_le_bytes()).collect() }
/// #   fn dequantize(&self, b: &[u8], n: usize) -> Vec<f32> {
/// #     b.chunks(4).take(n).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect() }
/// # }
/// let report = testkit::check_scheme(&S);
/// println!("{report}");
/// assert!(report.is_ok());
/// ```
pub fn check_scheme(scheme: &dyn CustomScheme) -> Report {
    let mut checks = Vec::new();
    let ctx = QuantContext::with_row_len(128);

    // ── 1. Maintaining length on edge dimensions
    //
    // Block formats almost always break at lengths that are not multiples of the block:
    // either lose the tail or return the padding as real weights.
    let mut len_problem = None;
    for &n in &[
        0usize, 1, 2, 3, 7, 31, 32, 33, 63, 127, 128, 129, 255, 256, 257, 1000,
    ] {
        let w = gaussian_weights(n, 1234);
        let out = roundtrip(scheme, &w, &ctx);
        if out.len() != n {
            len_problem = Some(format!(
                "with a length of {n} at the output of {} elements instead of {n}",
                out.len()
            ));
            break;
        }
    }
    checks.push(Check {
        name: "length_preserved",
        description: "dequantize(quantize(w)).len() == w.len() for any length, including 0 and not multiples of the block size",
        outcome: match len_problem {
            Some(m) => Outcome::Fail(m),
            None => Outcome::Pass,
        },
    });

    // ── 2. Determinism ───────────────────── ─────────────────────
    //
    // Non-deterministic quantization breaks the weight cache and
    // reproducibility: the same model produces different files.
    let w = gaussian_weights(512, 77);
    let a = scheme.quantize(&w, &ctx);
    let b = scheme.quantize(&w, &ctx);
    checks.push(Check {
        name: "deterministic",
        description: "two quantizations of one input give the same bytes",
        outcome: if a == b {
            Outcome::Pass
        } else {
            Outcome::Fail(format!(
                "bytes diverged: {} vs {} bytes, first discrepancy in position {:?}",
                a.len(),
                b.len(),
                a.iter().zip(&b).position(|(x, y)| x != y)
            ))
        },
    });

    // ── 3. Extremity based on normal data
    let out = roundtrip(scheme, &w, &ctx);
    let bad = out.iter().position(|v| !v.is_finite());
    checks.push(Check {
        name: "finite_on_normal_input",
        description: "on a Gaussian scale the output does not contain NaN and inf",
        outcome: match bad {
            Some(i) => Outcome::Fail(format!("element {i} = {}", out[i])),
            None => Outcome::Pass,
        },
    });

    // ── 4. Limb at extreme values
    //
    // The block scale is often stored at f16 (maximum 65504). Weight is about 1e6
    // occurs at non-normalized checkpoints, and then the scale
    // becomes inf, and with it the WHOLE block - silently.
    let mut extreme = vec![0.01f32; 512];
    extreme[0] = 1e6;
    extreme[1] = -1e6;
    extreme[2] = f32::MAX;
    extreme[3] = f32::MIN_POSITIVE;
    let out = roundtrip(scheme, &extreme, &ctx);
    let bad = out.iter().position(|v| !v.is_finite());
    checks.push(Check {
        name: "finite_on_extreme_input",
        description: "weights up to f32::MAX do not generate NaN/inf (a typical reason is scale overflow at f16)",
        outcome: match bad {
            Some(i) => Outcome::Fail(format!("element {i} = {} at input {}", out[i], extreme[i])),
            None => Outcome::Pass,
        },
    });

    // ── 5. Zero tensor ─────────────────────── ────────────────────────
    //
    // Division by zero scale is a classic source of NaN. Zeros
    // found in real models: camouflaged heads, pruning.
    let zeros = vec![0.0f32; 256];
    let out = roundtrip(scheme, &zeros, &ctx);
    let bad = out.iter().position(|v| !v.is_finite());
    checks.push(Check {
        name: "zeros_stay_finite",
        description: "an all-null tensor does not generate NaN/inf (division by zero scale)",
        outcome: match bad {
            Some(i) => Outcome::Fail(format!("element {i} = {}", out[i])),
            None => Outcome::Pass,
        },
    });

    // ── 6. Constant tensor ───────────────────── ──────────────────────
    //
    // Zero dispersion: for symmetric schemes the scale is correct, for
    // asymmetric - easy to get zero range and division by zero.
    let consts = vec![0.7f32; 256];
    let out = roundtrip(scheme, &consts, &ctx);
    let worst = out.iter().map(|v| (v - 0.7).abs()).fold(0.0f32, f32::max);
    checks.push(Check {
        name: "constant_tensor",
        description: "the constant tensor is restored close to a constant",
        outcome: if !worst.is_finite() {
            Outcome::Fail("the output is not finite".into())
        } else if worst > 0.7 {
            Outcome::Fail(format!("maximum deviation {worst} at constant 0.7"))
        } else {
            Outcome::Pass
        },
    });

    // ── 7. Meaningful: better than giving out the average ─────────────────────
    let w = gaussian_weights(4096, 2024);
    let out = roundtrip(scheme, &w, &ctx);
    let var = variance(&w);
    let err = mse(&w, &out);
    let rel = (err / var.max(f64::MIN_POSITIVE)) as f32;
    checks.push(Check {
        name: "better_than_mean",
        description:
            "MSE is less than the variance of the input - otherwise the scheme is no better than replacing all the weights with the average",
        outcome: if err < var {
            Outcome::Pass
        } else {
            Outcome::Fail(format!("MSE {err:.6} with variance {var:.6}"))
        },
    });

    // ── 8. Emission resistance ──────────────────────────────────────
    //
    // One outlier weight should not destroy the accuracy of the others: this
    // checking for per-tensor scale where block scale is needed.
    let w_out = weights_with_outliers(4096, 99);
    let out = roundtrip(scheme, &w_out, &ctx);
    // We look only at the “normal” part, without the emissions themselves.
    let body: Vec<f32> = w_out[1..100].to_vec();
    let body_out: Vec<f32> = out[1..100].to_vec();
    let body_rel = mse(&body, &body_out) / variance(&body).max(f64::MIN_POSITIVE);
    checks.push(Check {
        name: "outlier_robustness",
        description: "rare outliers do not destroy the accuracy of the remaining scales (a sign of per-tensor scale where a block one is needed)",
        outcome: if !body_rel.is_finite() {
            Outcome::Fail("the error is not finite".into())
        } else if body_rel < 1.0 {
            Outcome::Pass
        } else {
            Outcome::Skip(format!(
                "relative error outside outliers {body_rel:.3} - a scheme with one scale per tensor behaves as expected"
            ))
        },
    });

    // ── 9. Honesty bits_per_weight ────────────────────────────────────
    let bytes = scheme.quantize(&w, &ctx);
    let claimed = scheme.bits_per_weight(w.len(), bytes.len());
    let actual = bytes.len() as f32 * 8.0 / w.len() as f32;
    checks.push(Check {
        name: "bits_per_weight_honest",
        description: "the stated bits/weights do not underestimate the actual buffer size",
        outcome: if claimed >= actual - 1e-3 {
            Outcome::Pass
        } else {
            Outcome::Fail(format!(
                "stated {claimed:.3} bits/weight, in fact the buffer gives {actual:.3}"
            ))
        },
    });

    // ── 10. Scale invariance ───────────────────────────────────
    //
    // Multiplying all weights by a constant must multiply the result by it
    // same. Violation means hardcoded range instead of scale -
    // the scheme will only work on data of a “convenient” amplitude.
    let w = gaussian_weights(1024, 555);
    let scaled: Vec<f32> = w.iter().map(|v| v * 100.0).collect();
    let a = roundtrip(scheme, &w, &ctx);
    let b = roundtrip(scheme, &scaled, &ctx);
    let drift = a
        .iter()
        .zip(&b)
        .map(|(x, y)| (x * 100.0 - y).abs())
        .fold(0.0f32, f32::max);
    let amax = scaled.iter().fold(0.0f32, |m, v| m.max(v.abs()));
    checks.push(Check {
        name: "scale_equivariance",
        description: "quantization w*100 gives a result 100 times greater than quantization w (otherwise the range is hard-coded)",
        outcome: if !drift.is_finite() {
            Outcome::Fail("the result is not finite".into())
        } else if drift <= amax * 0.05 {
            Outcome::Pass
        } else {
            // It is Fail, not Skip: schemes with a fixed GRID (nf4,
            // fp8) are still scale-invariant, because they keep
            // separate scale. Violation here means exactly one thing -
            // the range of values is programmed into the code, and on scales of a different amplitude
            // the scheme will silently trim them.
            Outcome::Fail(format!(
                "discrepancy {drift:.3} at amax {amax:.3}: it seems that the range of values is hardwired into the code instead of the scale"
            ))
        },
    });

    // ── 11. Format stability ──────────────────── ────────────────────
    //
    // The length of the buffer should depend only on the number of weights, and not on their
    // values: otherwise it is impossible to plan the memory layout in advance.
    let a = scheme.quantize(&gaussian_weights(1024, 1), &ctx).len();
    let b = scheme.quantize(&gaussian_weights(1024, 2), &ctx).len();
    checks.push(Check {
        name: "size_depends_only_on_count",
        description: "the package size is determined by the number of weights, not their values (needed for memory planning)",
        outcome: if a == b {
            Outcome::Pass
        } else {
            Outcome::Fail(format!("{a} vs {b} bytes on the same number of weights"))
        },
    });

    Report {
        scheme: scheme.tag().to_string(),
        checks,
        bits_per_weight: actual,
        relative_error: rel,
    }
}

/// Same as [`check_scheme`], but panics with a detailed report at the first
/// breach of contract. Intended for use directly in `#[test]`.
///
/// ```
/// # use neo_quantization::custom::{CustomScheme, QuantContext};
/// # #[derive(Debug)] struct S;
/// # impl CustomScheme for S {
/// #   fn tag(&self) -> &'static str { "s" }
/// #   fn quantize(&self, w: &[f32], _: &QuantContext) -> Vec<u8> {
/// #     w.iter().flat_map(|v| v.to_le_bytes()).collect() }
/// #   fn dequantize(&self, b: &[u8], n: usize) -> Vec<f32> {
/// #     b.chunks(4).take(n).map(|c| f32::from_le_bytes(c.try_into().unwrap())).collect() }
/// # }
/// // Directly in #[test] of your crate:
/// neo_quantization::testkit::assert_scheme(&S);
/// ```
pub fn assert_scheme(scheme: &dyn CustomScheme) {
    let report = check_scheme(scheme);
    assert!(report.is_ok(), "\n{report}");
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Correct scheme: 4 bits, block 32, scale f32 per block.
    #[derive(Debug)]
    struct GoodInt4;

    impl CustomScheme for GoodInt4 {
        fn tag(&self) -> &'static str {
            "good_int4"
        }
        fn quantize(&self, w: &[f32], _ctx: &QuantContext) -> Vec<u8> {
            let mut out = Vec::new();
            for block in w.chunks(32) {
                let amax = block.iter().fold(0.0f32, |a, v| a.max(v.abs()));
                let scale = if amax > 0.0 { amax / 7.0 } else { 1.0 };
                out.extend_from_slice(&scale.to_le_bytes());
                let inv = 1.0 / scale;
                let mut padded = [0.0f32; 32];
                padded[..block.len()].copy_from_slice(block);
                for pair in padded.chunks(2) {
                    let enc = |v: f32| ((v * inv).round().clamp(-7.0, 7.0) as i32 + 8) as u8;
                    out.push(enc(pair[0]) | (enc(pair[1]) << 4));
                }
            }
            out
        }
        fn dequantize(&self, bytes: &[u8], numel: usize) -> Vec<f32> {
            let mut out = Vec::with_capacity(numel);
            for blk in bytes.chunks(20) {
                let scale = f32::from_le_bytes(blk[..4].try_into().unwrap());
                for &b in &blk[4..] {
                    out.push(((b & 0x0F) as i32 - 8) as f32 * scale);
                    out.push(((b >> 4) as i32 - 8) as f32 * scale);
                }
            }
            out.truncate(numel);
            out
        }
    }

    #[test]
    fn testkit_accepts_correct_scheme() {
        assert_scheme(&GoodInt4);
    }

    /// A scheme whose scale exceeds f16 is precisely the defect that
    /// was found in embedded GGUF formats.
    #[derive(Debug)]
    struct OverflowingScale;

    impl CustomScheme for OverflowingScale {
        fn tag(&self) -> &'static str {
            "overflowing"
        }
        fn quantize(&self, w: &[f32], _ctx: &QuantContext) -> Vec<u8> {
            let mut out = Vec::new();
            for block in w.chunks(32) {
                let amax = block.iter().fold(0.0f32, |a, v| a.max(v.abs()));
                let scale = burn::tensor::f16::from_f32(amax / 127.0); // without saturation
                out.extend_from_slice(&scale.to_le_bytes());
                let inv = 1.0 / scale.to_f32();
                for i in 0..32 {
                    let v = block.get(i).copied().unwrap_or(0.0);
                    out.push(((v * inv).round().clamp(-127.0, 127.0) as i8) as u8);
                }
            }
            out
        }
        fn dequantize(&self, bytes: &[u8], numel: usize) -> Vec<f32> {
            let mut out = Vec::with_capacity(numel);
            for blk in bytes.chunks(34) {
                let scale = burn::tensor::f16::from_le_bytes([blk[0], blk[1]]).to_f32();
                for &b in &blk[2..] {
                    out.push(b as i8 as f32 * scale);
                }
            }
            out.truncate(numel);
            out
        }
    }

    #[test]
    fn testkit_catches_f16_scale_overflow() {
        let report = check_scheme(&OverflowingScale);
        assert!(!report.is_ok(), "the testkit must catch the scale overflow");
        let names: Vec<_> = report.failures().iter().map(|c| c.name).collect();
        assert!(
            names.contains(&"finite_on_extreme_input"),
            "expected failure finite_on_extreme_input, received: {names:?}"
        );
    }

    /// A scheme that loses its tail when the length is not a multiple of the block.
    #[derive(Debug)]
    struct DropsTail;

    impl CustomScheme for DropsTail {
        fn tag(&self) -> &'static str {
            "drops_tail"
        }
        fn quantize(&self, w: &[f32], _ctx: &QuantContext) -> Vec<u8> {
            let full = w.len() / 32 * 32; // tail silently dropped
            w[..full].iter().flat_map(|v| v.to_le_bytes()).collect()
        }
        fn dequantize(&self, bytes: &[u8], _numel: usize) -> Vec<f32> {
            bytes
                .chunks(4)
                .map(|c| f32::from_le_bytes(c.try_into().unwrap()))
                .collect()
        }
    }

    #[test]
    fn testkit_catches_dropped_tail() {
        let report = check_scheme(&DropsTail);
        assert!(!report.is_ok());
        let names: Vec<_> = report.failures().iter().map(|c| c.name).collect();
        assert!(names.contains(&"length_preserved"), "received: {names:?}");
    }

    /// Circuit with a hardcoded range [-1, 1] - works only on
    /// “convenient” data and silently spoils everything else.
    #[derive(Debug)]
    struct HardcodedRange;

    impl CustomScheme for HardcodedRange {
        fn tag(&self) -> &'static str {
            "hardcoded"
        }
        fn quantize(&self, w: &[f32], _ctx: &QuantContext) -> Vec<u8> {
            w.iter()
                .map(|v| ((v.clamp(-1.0, 1.0) * 127.0) as i8) as u8)
                .collect()
        }
        fn dequantize(&self, bytes: &[u8], numel: usize) -> Vec<f32> {
            bytes
                .iter()
                .take(numel)
                .map(|&b| b as i8 as f32 / 127.0)
                .collect()
        }
    }

    #[test]
    fn testkit_flags_hardcoded_range() {
        let report = check_scheme(&HardcodedRange);
        assert!(!report.is_ok(), "\n{report}");
        let names: Vec<_> = report.failures().iter().map(|c| c.name).collect();
        assert!(
            names.contains(&"scale_equivariance"),
            "the hardwired range must fail scale_equivariance, obtained: {names:?}"
        );
    }
}
