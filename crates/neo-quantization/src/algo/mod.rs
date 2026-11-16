// neo-quantization/src/algo/mod.rs
//
// Quantization algorithms WITH CALIBRATION - GPTQ and AWQ.
//
// ## How are they different from `pack::*`
//
// Everything in `pack::*` is CODING: how to decompose numbers into bits. Such
// the schemes look only at the weights themselves and minimize the error across the weights.
//
// Here are SELECTION ALGORITHMS: what exactly to code to minimize
// error AT THE LAYER OUTPUT. This requires real activations: without them
// it is unknown which inputs are important and which have little or no effect on the outcome.
// That is why the same 4-bit format gives noticeably different
// quality depends on what it is filled with.
//
// Both algorithms are layer-by-layer and do not require additional training: you only need a run
// calibration set (typically 128 text fragments) to collect inputs
// each linear layer.
//
// ## What to choose
//
// * [`gptq_quantize`] - more precisely at low bit depths (2-3 bits), more expensive
//   by calculations: inversion of the matrix `in_features × in_features`.
// * [`awq_quantize`] - cheaper and simpler, works through scaling
//   channels; on 4 bits is usually comparable to GPTQ, and on models with
//   pronounced spikes in activations are often better.
//
// You can compare directly using your data: both results have
// `weight_mse`, and the output error is reported as [`output_mse`].

mod awq;
mod gptq;
mod linalg;

pub use awq::{awq_quantize, AwqConfig, AwqError, AwqResult};
pub use gptq::{gptq_quantize, GptqConfig, GptqError, GptqResult};
pub use linalg::{hessian_from_activations, Matrix};

use std::vec::Vec;

// ═══════════════════════════════════════════════════════════════════════════
//  General group quantizer
// ═══════════════════════════════════════════════════════════════════════════

/// Quantization parameters for one weight group: asymmetric mesh
/// `w ≈ scale·q + zero`, where `q ∈ [0, 2^bits − 1]`.
///
/// Asymmetrical, not symmetrical: both GPTQ and AWQ in reference
/// implementations use it. Distribution of weights within a group of
/// 128 elements are almost never symmetrical about zero, and
/// a separate zero point pays for its bits.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GroupQuant {
    /// Grid pitch.
    pub scale: f64,
    /// Value corresponding to code 0.
    pub zero: f64,
}

/// Select a grid for a group of scales.
///
/// Degenerate cases (empty group, all weights equal) give `scale = 0` −
/// and then [`dequantize_group`] returns `zero` for any code. This
/// correct: constant group and represents a single number.
pub fn quantize_group(values: &[f64], bits: u8) -> GroupQuant {
    if values.is_empty() {
        return GroupQuant {
            scale: 0.0,
            zero: 0.0,
        };
    }
    let levels = ((1u32 << bits) - 1) as f64;

    let mut min = f64::INFINITY;
    let mut max = f64::NEG_INFINITY;
    for &v in values {
        if v.is_finite() {
            min = min.min(v);
            max = max.max(v);
        }
    }
    if !min.is_finite() || !max.is_finite() {
        return GroupQuant {
            scale: 0.0,
            zero: 0.0,
        };
    }

    let range = max - min;
    if range <= 0.0 {
        // All weights are the same: no mesh is needed, the zero point is enough.
        return GroupQuant {
            scale: 0.0,
            zero: min,
        };
    }
    GroupQuant {
        scale: range / levels,
        zero: min,
    }
}

/// Expand the code back to a real value.
#[inline]
pub fn dequantize_group(code: i32, gq: &GroupQuant) -> f64 {
    gq.zero + gq.scale * code as f64
}

/// Error at the OUTPUT of the layer between the original and quantized weights.
///
/// It is this that GPTQ and AWQ minimize, and it is by this that they should be compared
/// among themselves - not by mistake on the scales. The scheme may distort the weights more,
/// but in those directions where the activations are small, and then the output will suffer
/// less.
///
/// * `weights`, `quantized` — `[out_features, in_features]`;
/// * `activations` — `[n_samples, in_features]`.
pub fn output_mse(
    weights: &[f32],
    quantized: &[f32],
    activations: &[f32],
    out_features: usize,
    in_features: usize,
) -> f64 {
    assert_eq!(weights.len(), quantized.len(), "scale sizes do not match");
    assert_eq!(
        activations.len() % in_features,
        0,
        "activations are not multiples of in_features"
    );

    let n_samples = activations.len() / in_features;
    if n_samples == 0 || out_features == 0 {
        return 0.0;
    }

    let mut acc = 0.0;
    for s in 0..n_samples {
        let row = &activations[s * in_features..(s + 1) * in_features];
        for r in 0..out_features {
            let mut diff = 0.0;
            for c in 0..in_features {
                let dw =
                    weights[r * in_features + c] as f64 - quantized[r * in_features + c] as f64;
                diff += dw * row[c] as f64;
            }
            acc += diff * diff;
        }
    }
    acc / (n_samples * out_features) as f64
}

/// Simple group rounding without calibration - baseline.
///
/// Needed to have something to compare GPTQ and AWQ with: the statement “algorithm
/// improves quality" is testable only against a specific alternative.
pub fn round_to_nearest(
    weights: &[f32],
    out_features: usize,
    in_features: usize,
    bits: u8,
    group_size: usize,
) -> Vec<f32> {
    let group = if group_size == 0 {
        in_features
    } else {
        group_size
    };
    let levels = (1i32 << bits) - 1;
    let mut out = vec![0.0f32; weights.len()];

    for r in 0..out_features {
        for gs in (0..in_features).step_by(group) {
            let end = (gs + group).min(in_features);
            let slice: Vec<f64> = (gs..end)
                .map(|c| weights[r * in_features + c] as f64)
                .collect();
            let gq = quantize_group(&slice, bits);
            for c in gs..end {
                let v = weights[r * in_features + c] as f64;
                let code = if gq.scale == 0.0 {
                    0
                } else {
                    (((v - gq.zero) / gq.scale).round() as i32).clamp(0, levels)
                };
                out[r * in_features + c] = dequantize_group(code, &gq) as f32;
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_group_quant_roundtrip_covers_range() {
        let values: Vec<f64> = (0..128).map(|i| (i as f64 - 64.0) * 0.03).collect();
        let gq = quantize_group(&values, 4);

        // Extreme values must fall into extreme codes: otherwise part
        // no mesh is used and precision is wasted.
        let code_min = ((values[0] - gq.zero) / gq.scale).round() as i32;
        let code_max = ((values[127] - gq.zero) / gq.scale).round() as i32;
        assert_eq!(code_min, 0);
        assert_eq!(code_max, 15);

        for &v in &values {
            let code = (((v - gq.zero) / gq.scale).round() as i32).clamp(0, 15);
            let back = dequantize_group(code, &gq);
            assert!((v - back).abs() <= gq.scale / 2.0 + 1e-12, "{v} vs {back}");
        }
    }

    #[test]
    fn test_group_quant_degenerate_cases() {
        // Empty group.
        let gq = quantize_group(&[], 4);
        assert_eq!(gq.scale, 0.0);

        // All weights are equal: scale = 0, but zero stores the value itself.
        let gq = quantize_group(&[0.7; 16], 4);
        assert_eq!(gq.scale, 0.0);
        assert!((dequantize_group(0, &gq) - 0.7).abs() < 1e-12);
        assert!((dequantize_group(15, &gq) - 0.7).abs() < 1e-12);

        // Non-numeric values should not generate NaN in the grid.
        let gq = quantize_group(&[f64::NAN, f64::INFINITY], 4);
        assert!(gq.scale.is_finite() && gq.zero.is_finite());
    }

    #[test]
    fn test_output_mse_is_zero_for_identical_weights() {
        let w: Vec<f32> = (0..32).map(|i| i as f32 * 0.1).collect();
        let x: Vec<f32> = (0..160).map(|i| (i % 7) as f32).collect();
        assert_eq!(output_mse(&w, &w, &x, 4, 8), 0.0);
    }

    #[test]
    fn test_round_to_nearest_more_bits_is_better() {
        let w: Vec<f32> = (0..256)
            .map(|i| ((i * 31 % 97) as f32 - 48.0) / 48.0)
            .collect();
        let mut prev = f64::INFINITY;
        for bits in [2u8, 4, 8] {
            let q = round_to_nearest(&w, 8, 32, bits, 32);
            let mse: f64 = w
                .iter()
                .zip(&q)
                .map(|(a, b)| ((a - b) as f64).powi(2))
                .sum::<f64>()
                / w.len() as f64;
            assert!(mse < prev, "{bits} bit is no better than the previous one");
            prev = mse;
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
//  Bridge to custom API
// ═══════════════════════════════════════════════════════════════════════════

use crate::custom::{CustomScheme, QuantContext};

/// GPTQ and AWQ as normal schemes [`CustomScheme`].
///
/// It is necessary that calibrated algorithms move along the same pipeline as
/// and everything else: `WeightRegistry`, `convert_checkpoint`, statistics on
/// bits. At the same time, this is a check that [`QuantContext`] is designed
/// enough: the fields `row_len` and `calibration` are enough to express
/// layer-by-layer algorithm without finishing anything.
///
/// The packaging here is f32 (the algorithms return already dequantized weights),
/// therefore, this wrapper does not provide any gain in memory: its task is to obtain
/// BETTER QUALITY WEIGHTS, which are then encoded by any 4-bit
/// format from `pack::*`. The separation is intentional: selection of meanings and their
/// coding are different tasks.
///
/// ```
/// use neo_quantization::algo::{CalibratedScheme, GptqConfig};
/// use neo_quantization::custom::QuantContext;
///
/// let in_features = 16;
/// let weights: Vec<f32> = (0..8 * in_features).map(|i| (i % 13) as f32 * 0.1 - 0.6).collect();
/// let activations: Vec<f32> = (0..256 * in_features).map(|i| (i % 17) as f32 * 0.1 - 0.8).collect();
///
/// let scheme = CalibratedScheme::gptq(GptqConfig::int4_g128().with_group_size(16));
/// let ctx = QuantContext::with_row_len(in_features).with_calibration(&activations);
///
/// use neo_quantization::custom::CustomScheme;
/// let packed = scheme.quantize(&weights, &ctx);
/// let restored = scheme.dequantize(&packed, weights.len());
/// assert_eq!(restored.len(), weights.len());
/// ```
#[derive(Debug, Clone)]
pub enum CalibratedScheme {
    /// GPTQ with a given configuration.
    Gptq(GptqConfig),
    /// AWQ with a given configuration.
    Awq(AwqConfig),
}

impl CalibratedScheme {
    /// GPTQ based scheme.
    pub fn gptq(config: GptqConfig) -> Self {
        Self::Gptq(config)
    }

    /// AWQ based scheme.
    pub fn awq(config: AwqConfig) -> Self {
        Self::Awq(config)
    }

    /// The number of bits the algorithm uses when guessing values.
    pub fn bits(&self) -> u8 {
        match self {
            Self::Gptq(c) => c.bits,
            Self::Awq(c) => c.bits,
        }
    }
}

impl CustomScheme for CalibratedScheme {
    fn tag(&self) -> &'static str {
        match self {
            Self::Gptq(_) => "gptq",
            Self::Awq(_) => "awq",
        }
    }

    /// Select weights and save them as f32.
    ///
    /// If there is no calibration or the shape is not specified, a ROLLback to
    /// normal rounding of the same bit depth - and this is the only place
    /// where such a rollback is acceptable. Calling `gptq_quantize` directly will return
    /// an error because the user there specifically asked for GPTQ; right here
    /// the scheme is called by a generalized code (`convert_checkpoint`, etc.),
    /// which must return something for each tensor. Bit depth at
    /// this is observed, so the size of the model does not change - it changes
    /// only quality, and there is no silent format substitution.
    fn quantize(&self, weights: &[f32], ctx: &QuantContext) -> Vec<u8> {
        let in_features = ctx.row_len;
        let usable =
            in_features > 0 && !weights.is_empty() && weights.len().is_multiple_of(in_features);

        let values: Vec<f32> = if !usable {
            fallback(weights, self.bits())
        } else {
            let out_features = weights.len() / in_features;
            match (self, ctx.calibration) {
                (Self::Gptq(cfg), Some(act)) if !act.is_empty() => {
                    match gptq_quantize(weights, out_features, in_features, act, cfg) {
                        Ok(r) => r.weights,
                        Err(_) => round_to_nearest(
                            weights,
                            out_features,
                            in_features,
                            cfg.bits,
                            cfg.group_size,
                        ),
                    }
                }
                (Self::Awq(cfg), Some(act)) if !act.is_empty() => {
                    match awq_quantize(weights, out_features, in_features, act, cfg) {
                        Ok(r) => r.weights,
                        Err(_) => round_to_nearest(
                            weights,
                            out_features,
                            in_features,
                            cfg.bits,
                            cfg.group_size,
                        ),
                    }
                }
                _ => {
                    let (bits, group) = match self {
                        Self::Gptq(c) => (c.bits, c.group_size),
                        Self::Awq(c) => (c.bits, c.group_size),
                    };
                    round_to_nearest(weights, out_features, in_features, bits, group)
                }
            }
        };

        values.iter().flat_map(|v| v.to_le_bytes()).collect()
    }

    fn dequantize(&self, bytes: &[u8], numel: usize) -> Vec<f32> {
        bytes
            .as_chunks::<4>()
            .0
            .iter()
            .take(numel)
            .map(|c| f32::from_le_bytes(*c))
            .collect()
    }
}

/// Rounding a flat vector without a known shape - one group for everything.
fn fallback(weights: &[f32], bits: u8) -> Vec<f32> {
    if weights.is_empty() {
        return Vec::new();
    }
    round_to_nearest(weights, 1, weights.len(), bits, weights.len())
}

#[cfg(test)]
mod bridge_tests {
    use super::*;

    fn lcg(n: usize, seed: u64) -> Vec<f32> {
        let mut s = seed;
        (0..n)
            .map(|_| {
                s = s
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                ((s >> 40) as f32) / ((1u64 << 24) as f32) - 0.5
            })
            .collect()
    }

    #[test]
    fn calibrated_schemes_pass_the_public_testkit() {
        // The testkit invokes the scheme WITHOUT calibration and without a shape - we check that
        // the rollback is correct and the contract is still respected.
        crate::testkit::assert_scheme(&CalibratedScheme::gptq(
            GptqConfig::default().with_group_size(32),
        ));
        crate::testkit::assert_scheme(&CalibratedScheme::awq(
            AwqConfig::default().with_group_size(32),
        ));
    }

    #[test]
    fn calibration_through_context_actually_improves_quality() {
        let (out_f, in_f) = (16, 64);
        let w = lcg(out_f * in_f, 71);
        let base = lcg(512 * in_f, 72);
        let mut x = vec![0.0f32; 512 * in_f];
        for s in 0..512 {
            for c in 0..in_f {
                let prev = if c == 0 { 0.0 } else { x[s * in_f + c - 1] };
                x[s * in_f + c] = 0.7 * prev + base[s * in_f + c];
            }
        }

        let scheme = CalibratedScheme::gptq(GptqConfig::default().with_group_size(32));

        let with_calib = QuantContext::with_row_len(in_f).with_calibration(&x);
        let without = QuantContext::with_row_len(in_f);

        let a = scheme.dequantize(&scheme.quantize(&w, &with_calib), w.len());
        let b = scheme.dequantize(&scheme.quantize(&w, &without), w.len());

        let err_a = output_mse(&w, &a, &x, out_f, in_f);
        let err_b = output_mse(&w, &b, &x, out_f, in_f);
        assert!(
            err_a < err_b,
            "calibration via QuantContext had no effect: {err_a:.3e} vs {err_b:.3e}"
        );
    }

    #[test]
    fn packed_buffer_carries_calibrated_scheme() {
        use crate::custom::CustomPacked;
        use crate::PackedBuffer;
        use std::sync::Arc;

        let in_f = 32;
        let w = lcg(4 * in_f, 81);
        let x = lcg(256 * in_f, 82);
        let ctx = QuantContext::with_row_len(in_f).with_calibration(&x);

        let packed = PackedBuffer::Custom(CustomPacked::new(
            Arc::new(CalibratedScheme::awq(
                AwqConfig::default().with_group_size(32),
            )),
            &w,
            &ctx,
        ));

        assert_eq!(packed.numel(), w.len());
        assert_eq!(packed.scheme_tag(), "awq");
        assert_eq!(packed.to_f32().len(), w.len());
        assert!(packed.to_f32().iter().all(|v| v.is_finite()));
    }
}
