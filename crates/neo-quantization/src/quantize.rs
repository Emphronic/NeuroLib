// neo-quantization/src/quantize.rs
//
// Single entry point: `float slice + QuantizeConfig → PackedBuffer`.
// Dispatches to specific pack::* modules by `config.scheme`.
//
// Outlier clipping (config.outlier_clip_percentile) is applied here BEFORE
// passing data to pack::* so that the pack modules themselves remain simple
// "pure" implementations without knowledge of calibration strategies - the same
// pack::quantize_ternary/int1/int8/int4 are also used directly (without
// QuantizeConfig) where calibration is not needed.
//
// Stochastic rounding (RoundingMode::Stochastic) - EXACT (unbiased:
// up with the probability of the fractional part) for schemes with a uniform grid, where
// RNG will be stretched inside the package: Ternary and Int2..Int8 (symmetric and
// asymmetric), see `quantize_exact_stochastic`. For other schemes (Int1,
// Nf4, GGUF-block formats) the grid is uneven/internal, so stochastic
// rounding uses an approximation - multiplicative jitter before round-to-nearest
// (see `prepare`).

use crate::config::{QuantizeConfig, RoundingMode};
use crate::pack::intn::{self, IntNPacked};
use crate::pack::{self, Calibration, PackedBuffer};
use crate::rng::XorShift64;
use crate::scheme::{PackedScheme, ScaleGranularity};

// ═══════════════════════════════════════════════════════════════════════════
// § 1 MAIN FUNCTION
// ═══════════════════════════════════════════════════════════════════════════

/// Quantize one tensor according to `config`.
///
/// `row_len` — length of the row (in_features) for `ScaleGranularity::PerChannel`;
/// ignored for `PerTensor`/`PerGroup`. Pass `0` if it is not
/// applicable (for example, for `PerTensor` schemes).
pub fn quantize(data: &[f32], config: &QuantizeConfig, row_len: usize) -> PackedBuffer {
    // Stochastic rounding: for schemes where the RNG is extended inside the package,
    // it is accurate, and an approximate jitter on top of it is not needed (there would be a double
    // noise); for other schemes `exact_stochastic` - `false`, and `prepare`
    // applies jitter as before.
    let stochastic_seed = match config.rounding {
        RoundingMode::Stochastic { seed } => Some(seed),
        RoundingMode::Nearest => None,
    };
    let exact_stochastic = stochastic_seed.is_some() && supports_exact_stochastic(config.scheme);

    let prepared = prepare(data, config, exact_stochastic);
    let data: &[f32] = prepared.as_deref().unwrap_or(data);

    if let (true, Some(seed)) = (exact_stochastic, stochastic_seed) {
        if let Some(buf) = quantize_exact_stochastic(data, config.scheme, row_len, seed) {
            return buf;
        }
    }

    match config.scheme {
        PackedScheme::Float => PackedBuffer::Float(data.to_vec()),

        PackedScheme::Ternary(g) => PackedBuffer::Ternary(pack::quantize_ternary(data, g, row_len)),

        PackedScheme::Int1(g) => PackedBuffer::Int1(pack::quantize_int1(data, g, row_len)),

        PackedScheme::Int2(g, Calibration::Symmetric) => {
            PackedBuffer::Int2(pack::quantize_int2(data, g, row_len))
        }
        PackedScheme::Int2(g, Calibration::Asymmetric) => {
            PackedBuffer::Int2(pack::quantize_int2_asymmetric(data, g, row_len))
        }

        PackedScheme::Int3(g, Calibration::Symmetric) => {
            PackedBuffer::Int3(pack::quantize_int3(data, g, row_len))
        }
        PackedScheme::Int3(g, Calibration::Asymmetric) => {
            PackedBuffer::Int3(pack::quantize_int3_asymmetric(data, g, row_len))
        }

        PackedScheme::Int4(g, Calibration::Symmetric) => {
            PackedBuffer::Int4(pack::quantize_int4(data, g, row_len))
        }
        PackedScheme::Int4(g, Calibration::Asymmetric) => {
            PackedBuffer::Int4(pack::quantize_int4_asymmetric(data, g, row_len))
        }

        PackedScheme::Int5(g, Calibration::Symmetric) => {
            PackedBuffer::Int5(pack::quantize_int5(data, g, row_len))
        }
        PackedScheme::Int5(g, Calibration::Asymmetric) => {
            PackedBuffer::Int5(pack::quantize_int5_asymmetric(data, g, row_len))
        }

        PackedScheme::Int6(g, Calibration::Symmetric) => {
            PackedBuffer::Int6(pack::quantize_int6(data, g, row_len))
        }
        PackedScheme::Int6(g, Calibration::Asymmetric) => {
            PackedBuffer::Int6(pack::quantize_int6_asymmetric(data, g, row_len))
        }

        PackedScheme::Int7(g, Calibration::Symmetric) => {
            PackedBuffer::Int7(pack::quantize_int7(data, g, row_len))
        }
        PackedScheme::Int7(g, Calibration::Asymmetric) => {
            PackedBuffer::Int7(pack::quantize_int7_asymmetric(data, g, row_len))
        }

        PackedScheme::Int8(g, Calibration::Symmetric) => {
            PackedBuffer::Int8(pack::quantize_int8(data, g, row_len))
        }
        PackedScheme::Int8(g, Calibration::Asymmetric) => {
            PackedBuffer::Int8(pack::quantize_int8_asymmetric(data, g, row_len))
        }

        PackedScheme::Nf4(g) => PackedBuffer::Nf4(pack::quantize_nf4(data, g, row_len)),

        PackedScheme::Gguf(dtype) => {
            // Simple block formats (Q4_0/Q4_1/Q5_0/Q5_1/Q8_0) - via
            // pack::gguf. K-quant super-block formats (Q2_K..Q8_K) - via
            // pack::kquant. IQ-series (non-linear IQ4_NL/IQ4_XS, codebook
            // IQ2_XXS/IQ2_XS/IQ3_XXS/IQ3_S/IQ1_S/IQ1_M) - via pack::iq
            // (see pack::iq::is_iq_series/is_supported). TQ1_0/TQ2_0 - via
            // pack::tq; other GGUF dtypes fall through to the final panic below.
            match pack::gguf_backend(dtype) {
                Some(pack::GgufBackend::Simple) => {
                    PackedBuffer::Gguf(pack::gguf::quantize_gguf(data, dtype))
                }
                Some(pack::GgufBackend::KQuant) => {
                    PackedBuffer::KQuant(pack::kquant::quantize_kquant(data, dtype))
                }
                Some(pack::GgufBackend::Iq) => PackedBuffer::Iq(pack::iq::quantize_iq(data, dtype)),
                Some(pack::GgufBackend::Tq) => PackedBuffer::Tq(pack::tq::quantize_tq(data, dtype)),
                None => panic!(
                    "quantize(): PackedScheme::Gguf({dtype:?}) is not GGUF-block\
                     format implemented in neo-quantization (see pack::gguf_backend) - \
                     use PackedBuffer::GgufRaw as a byte container or writer neo-gguf"
                ),
            }
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// § 2 DATA PREPARATION (outlier clip + stochastic rounding jitter)
// ═══════════════════════════════════════════════════════════════════════════

/// Apply the preprocessing required by the config to a copy of the data.
///
/// Returns `None` if no conversion is required (fast path is not
/// allocate extra Vec when outlier_clip_percentile==0.0 and no jitter
/// needed).
///
/// `exact_stochastic` - stochastic rounding of this scheme is performed
/// inside the package (see `quantize_exact_stochastic`), and jitter is not needed.
fn prepare(data: &[f32], config: &QuantizeConfig, exact_stochastic: bool) -> Option<Vec<f32>> {
    let needs_clip = config.outlier_clip_percentile > 0.0;
    let needs_jitter =
        matches!(config.rounding, RoundingMode::Stochastic { .. }) && !exact_stochastic;

    if !needs_clip && !needs_jitter {
        return None;
    }

    let mut buf = data.to_vec();

    if needs_clip {
        clip_outliers_inplace(&mut buf, config.outlier_clip_percentile);
    }

    // Jitter - only for schemes without precise stochastic rounding: with
    // `exact_stochastic` (and the required clip) we get here only for the sake of the clip, and
    // noise on top of exact rounding would produce double the noise.
    if let (true, RoundingMode::Stochastic { seed }) = (needs_jitter, config.rounding) {
        // Fallback for schemes WITHOUT exact stochastic rounding (Int1, Nf4,
        // GGUF-block formats - see `supports_exact_stochastic`): step
        // quantization is unknown here (depends on alpha/grid calculated
        // inside pack::*), so stochastic rounding is approaching noise
        // scale ~0.5% of local |w| BEFORE round-to-nearest. This
        // conscious approximation rather than unbiased rounding; exact - for
        // Ternary and Int2..Int8, where the RNG will be pulled inside the package.
        apply_jitter_inplace(&mut buf, seed);
    }

    Some(buf)
}

/// Trim `percentile` the share of the largest modulo values, replacing them
/// to the threshold - reduces the impact of single emissions on mean/max-abs calibration.
fn clip_outliers_inplace(data: &mut [f32], percentile: f32) {
    if data.is_empty() {
        return;
    }

    let mut abs_sorted: Vec<f32> = data.iter().map(|w| w.abs()).collect();
    abs_sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    let cutoff_idx =
        (((1.0 - percentile) * abs_sorted.len() as f32) as usize).min(abs_sorted.len() - 1);
    let threshold = abs_sorted[cutoff_idx];

    for w in data.iter_mut() {
        if w.abs() > threshold {
            *w = threshold.copysign(*w);
        }
    }
}

fn apply_jitter_inplace(data: &mut [f32], seed: u64) {
    let mut rng = XorShift64::new(seed);
    for w in data.iter_mut() {
        let jitter_scale = w.abs() * 0.005; // 0.5% local noise
        let noise = (rng.next_f32() - 0.5) * jitter_scale;
        *w += noise;
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// § 2b EXACT STOCHASTIC ROUNDING (Ternary, Int2..Int8)
// ═══════════════════════════════════════════════════════════════════════════

/// Does the scheme have precise stochastic rounding (RNG inside the package):
/// Ternary and Int2..Int8 - a uniform grid, the step of which is (alpha/scale)
/// known only within `pack::*`. Int1 (sign only), Nf4
/// (uneven grid) and GGUF-block formats do not.
fn supports_exact_stochastic(scheme: PackedScheme) -> bool {
    matches!(
        scheme,
        PackedScheme::Ternary(_)
            | PackedScheme::Int2(..)
            | PackedScheme::Int3(..)
            | PackedScheme::Int4(..)
            | PackedScheme::Int5(..)
            | PackedScheme::Int6(..)
            | PackedScheme::Int7(..)
            | PackedScheme::Int8(..)
    )
}

/// intN engine with stochastic rounding: symmetric - max-abs
/// calibration (same as `pack::quantize_intN`), asymmetric - min/max.
fn intn_stochastic(
    data: &[f32],
    bits: u8,
    calibration: Calibration,
    granularity: ScaleGranularity,
    row_len: usize,
    seed: u64,
) -> IntNPacked {
    match calibration {
        Calibration::Symmetric => {
            intn::quantize_maxabs_stochastic(data, bits, granularity, row_len, seed)
        }
        Calibration::Asymmetric => {
            intn::quantize_stochastic(data, bits, calibration, granularity, row_len, seed)
        }
    }
}

/// Quantize with STOCHASTIC rounding if the scheme supports it
/// (`supports_exact_stochastic`); otherwise `None`. The scale is the same as
/// round-to-nearest paths in `quantize` (calibration does not change, it changes
/// only rounding of codes), RNG is deterministic by `seed`.
fn quantize_exact_stochastic(
    data: &[f32],
    scheme: PackedScheme,
    row_len: usize,
    seed: u64,
) -> Option<PackedBuffer> {
    Some(match scheme {
        PackedScheme::Ternary(g) => PackedBuffer::Ternary(
            pack::ternary::quantize_ternary_stochastic(data, g, row_len, seed),
        ),
        PackedScheme::Int2(g, c) => PackedBuffer::Int2(pack::Int2Packed::from_inner(
            intn_stochastic(data, 2, c, g, row_len, seed),
        )),
        PackedScheme::Int3(g, c) => PackedBuffer::Int3(pack::Int3Packed::from_inner(
            intn_stochastic(data, 3, c, g, row_len, seed),
        )),
        PackedScheme::Int4(g, c) => PackedBuffer::Int4(pack::Int4Packed::from_inner(
            intn_stochastic(data, 4, c, g, row_len, seed),
        )),
        PackedScheme::Int5(g, c) => PackedBuffer::Int5(pack::Int5Packed::from_inner(
            intn_stochastic(data, 5, c, g, row_len, seed),
        )),
        PackedScheme::Int6(g, c) => PackedBuffer::Int6(pack::Int6Packed::from_inner(
            intn_stochastic(data, 6, c, g, row_len, seed),
        )),
        PackedScheme::Int7(g, c) => PackedBuffer::Int7(pack::Int7Packed::from_inner(
            intn_stochastic(data, 7, c, g, row_len, seed),
        )),
        PackedScheme::Int8(g, c) => PackedBuffer::Int8(pack::Int8Packed::from_inner(
            intn_stochastic(data, 8, c, g, row_len, seed),
        )),
        _ => return None,
    })
}

// ═══════════════════════════════════════════════════════════════════════════
// § 3 TESTS
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::QuantizeConfig;

    #[test]
    fn test_quantize_ternary_via_config() {
        let data = vec![0.9, -0.9, 0.1, -0.1];
        let cfg = QuantizeConfig::ternary_per_tensor();
        let buf = quantize(&data, &cfg, 0);
        assert!(matches!(buf, PackedBuffer::Ternary(_)));
        assert_eq!(buf.numel(), 4);
    }

    #[test]
    fn test_quantize_float_passthrough() {
        let data = vec![1.0, 2.0, 3.0];
        let cfg = QuantizeConfig::float();
        let buf = quantize(&data, &cfg, 0);
        match buf {
            PackedBuffer::Float(v) => assert_eq!(v, data),
            _ => panic!("expected Float passthrough"),
        }
    }

    #[test]
    fn test_outlier_clip_reduces_max_abs() {
        let mut data = vec![1.0f32; 99];
        data.push(1000.0); // 1 outlier in 100
        let cfg = QuantizeConfig::ternary_per_tensor().with_outlier_clip(0.02);
        // with clip 2% the top overshoot should be trimmed to ~1.0 before alpha calibration
        let buf = quantize(&data, &cfg, 0);
        if let PackedBuffer::Ternary(p) = buf {
            // alpha should not be inflated by an outlier of 1000
            assert!(
                p.scales[0] < 10.0,
                "alpha not protected from outlier: {}",
                p.scales[0]
            );
        } else {
            panic!("expected ternary");
        }
    }

    #[test]
    fn test_stochastic_rounding_is_deterministic_given_seed() {
        let data = vec![0.33f32; 50];
        let cfg = QuantizeConfig::ternary_per_tensor()
            .with_rounding(RoundingMode::Stochastic { seed: 7 });
        let buf1 = quantize(&data, &cfg, 0);
        let buf2 = quantize(&data, &cfg, 0);
        assert_eq!(buf1, buf2, "same seed must produce identical output");
    }

    #[test]
    fn test_gguf_simple_block_format_dispatch() {
        // Q4_0 - a simple block format, packaged via pack::gguf
        // (see quantize.rs § 1).
        let data = vec![1.0f32; 32];
        let cfg = QuantizeConfig {
            scheme: PackedScheme::Gguf(crate::dtype::DType::Q4_0),
            rounding: RoundingMode::Nearest,
            symmetric: true,
            outlier_clip_percentile: 0.0,
        };
        let buf = quantize(&data, &cfg, 0);
        assert!(matches!(buf, PackedBuffer::Gguf(_)));
        assert_eq!(buf.numel(), 32);
    }

    #[test]
    fn test_gguf_kquant_dispatch() {
        // Q4_K - K-quant super-block format, packed via pack::kquant
        // (see quantize.rs § 1).
        let data: Vec<f32> = (0..256).map(|i| (i as f32 - 128.0) * 0.01).collect();
        let cfg = QuantizeConfig {
            scheme: PackedScheme::Gguf(crate::dtype::DType::Q4_K),
            rounding: RoundingMode::Nearest,
            symmetric: true,
            outlier_clip_percentile: 0.0,
        };
        let buf = quantize(&data, &cfg, 0);
        assert!(matches!(buf, PackedBuffer::KQuant(_)));
        assert_eq!(buf.numel(), 256);
    }

    #[test]
    fn test_gguf_iq4_nl_dispatch() {
        // IQ4_NL - non-linear 4-bit, packed via pack::iq
        // (see quantize.rs § 1).
        let data: Vec<f32> = (0..32).map(|i| (i as f32 - 16.0) * 0.05).collect();
        let cfg = QuantizeConfig {
            scheme: PackedScheme::Gguf(crate::dtype::DType::IQ4_NL),
            rounding: RoundingMode::Nearest,
            symmetric: true,
            outlier_clip_percentile: 0.0,
        };
        let buf = quantize(&data, &cfg, 0);
        assert!(matches!(buf, PackedBuffer::Iq(_)));
        assert_eq!(buf.numel(), 32);
    }

    #[test]
    fn test_gguf_iq_codebook_dispatch() {
        // IQ2_XXS - codebook/lattice-based; `pack::iq_codebook`
        // implements quantize_row_iq2_xxs_impl from ggml-quants.c (see
        // iq_codebook.rs title) and dispatches through the same path as the
        // other GGUF formats above.
        let data: Vec<f32> = (0..256).map(|i| (i as f32 - 128.0) * 0.01).collect();
        let cfg = QuantizeConfig {
            scheme: PackedScheme::Gguf(crate::dtype::DType::IQ2_XXS),
            rounding: RoundingMode::Nearest,
            symmetric: true,
            outlier_clip_percentile: 0.0,
        };
        let buf = quantize(&data, &cfg, 0);
        assert!(matches!(buf, PackedBuffer::Iq(_)));
        assert_eq!(buf.numel(), 256);
    }

    #[test]
    fn test_gguf_iq1_s_dispatch() {
        // IQ1_S is dispatched through the IQ series like the other
        // codebook formats (see quantize.rs § 1 / pack::iq::quantize_iq).
        let data: Vec<f32> = (0..256).map(|i| (i as f32 - 128.0) * 0.01).collect();
        let cfg = QuantizeConfig {
            scheme: PackedScheme::Gguf(crate::dtype::DType::IQ1_S),
            rounding: RoundingMode::Nearest,
            symmetric: true,
            outlier_clip_percentile: 0.0,
        };
        let buf = quantize(&data, &cfg, 0);
        assert!(matches!(buf, PackedBuffer::Iq(_)));
        assert_eq!(buf.numel(), 256);
        assert_eq!(buf.scheme_tag(), "IQ1_S");
    }

    #[test]
    fn test_gguf_iq1_m_dispatch() {
        let data: Vec<f32> = (0..256).map(|i| (i as f32 - 128.0) * 0.01).collect();
        let cfg = QuantizeConfig {
            scheme: PackedScheme::Gguf(crate::dtype::DType::IQ1_M),
            rounding: RoundingMode::Nearest,
            symmetric: true,
            outlier_clip_percentile: 0.0,
        };
        let buf = quantize(&data, &cfg, 0);
        assert!(matches!(buf, PackedBuffer::Iq(_)));
        assert_eq!(buf.numel(), 256);
        assert_eq!(buf.scheme_tag(), "IQ1_M");
    }

    #[test]
    fn test_gguf_tq1_0_dispatch() {
        // TQ1_0 is dispatched through the TQ packer to PackedBuffer/quantize().
        let data: Vec<f32> = (0..256).map(|i| (i as f32 - 128.0) * 0.01).collect();
        let cfg = QuantizeConfig {
            scheme: PackedScheme::Gguf(crate::dtype::DType::TQ1_0),
            rounding: RoundingMode::Nearest,
            symmetric: true,
            outlier_clip_percentile: 0.0,
        };
        let buf = quantize(&data, &cfg, 0);
        assert!(matches!(buf, PackedBuffer::Tq(_)));
        assert_eq!(buf.numel(), 256);
        assert_eq!(buf.scheme_tag(), "TQ1_0");
    }

    #[test]
    fn test_gguf_tq2_0_now_works() {
        let data: Vec<f32> = (0..256).map(|i| (i as f32 - 128.0) * 0.01).collect();
        let cfg = QuantizeConfig {
            scheme: PackedScheme::Gguf(crate::dtype::DType::TQ2_0),
            rounding: RoundingMode::Nearest,
            symmetric: true,
            outlier_clip_percentile: 0.0,
        };
        let buf = quantize(&data, &cfg, 0);
        assert!(matches!(buf, PackedBuffer::Tq(_)));
        assert_eq!(buf.numel(), 256);
        assert_eq!(buf.scheme_tag(), "TQ2_0");
    }

    #[test]
    fn test_tq_dequantize_into_via_wildcard_path() {
        // dequantize_into (see dequantize.rs § 1) does not have a bulk branch for
        // Tq - falls into the general point-lookup fallback via get_f32.
        // Explicitly cover it with a test, using the same principle that already exists for Iq
        // (see test_dequantize_into_handles_iq4_nl_via_wildcard_path in
        // dequantize.rs).
        let data: Vec<f32> = (0..256).map(|i| (i as f32 - 128.0) * 0.01).collect();
        let cfg = QuantizeConfig {
            scheme: PackedScheme::Gguf(crate::dtype::DType::TQ1_0),
            rounding: RoundingMode::Nearest,
            symmetric: true,
            outlier_clip_percentile: 0.0,
        };
        let buf = quantize(&data, &cfg, 0);
        let via_alloc = buf.to_f32();
        let mut via_inplace = vec![0.0f32; buf.numel()];
        crate::dequantize::dequantize_into(&buf, &mut via_inplace);
        assert_eq!(via_alloc, via_inplace);
    }

    // ── Exact stochastic rounding (Ternary, Int2..Int8) ─────────────────

    fn cfg_with(scheme: PackedScheme, rounding: RoundingMode) -> QuantizeConfig {
        QuantizeConfig {
            scheme,
            rounding,
            symmetric: true,
            outlier_clip_percentile: 0.0,
        }
    }

    /// One element 7.0 (for int4 `max_code == 7`, so max-abs scale
    /// exactly 1.0) and a dense cluster of values 2.3 between codes 2 and 3.
    fn dense_cluster(n: usize) -> Vec<f32> {
        let mut v = vec![2.3f32; n];
        v[0] = 7.0;
        v
    }

    fn mean_of_tail(v: &[f32]) -> f32 {
        v[1..].iter().sum::<f32>() / (v.len() - 1) as f32
    }

    #[test]
    fn test_exact_stochastic_int4_is_unbiased_while_nearest_is_biased() {
        let data = dense_cluster(20_001);
        let scheme = PackedScheme::Int4(ScaleGranularity::PerTensor, Calibration::Symmetric);

        let nearest = quantize(&data, &cfg_with(scheme, RoundingMode::Nearest), 0).to_f32();
        let stochastic = quantize(
            &data,
            &cfg_with(scheme, RoundingMode::Stochastic { seed: 11 }),
            0,
        )
        .to_f32();

        // Nearest systematically underestimates the cluster: 2.3 → 2.0.
        assert!((mean_of_tail(&nearest) - 2.0).abs() < 1e-6);
        // Stochastic rounding unbiased: average ≈ 2.3.
        let mean = mean_of_tail(&stochastic);
        assert!((mean - 2.3).abs() < 0.03, "stochastic mean={mean}");
        // And each weight fell on exactly one of the two adjacent codes.
        assert!(stochastic[1..].iter().all(|&v| v == 2.0 || v == 3.0));
        assert!(stochastic[1..].iter().any(|&v| v == 2.0));
        assert!(stochastic[1..].iter().any(|&v| v == 3.0));
    }

    #[test]
    fn test_exact_stochastic_is_deterministic_and_seed_sensitive() {
        let data = dense_cluster(5_000);
        let scheme = PackedScheme::Int4(ScaleGranularity::PerTensor, Calibration::Symmetric);
        let run = |seed: u64| {
            quantize(
                &data,
                &cfg_with(scheme, RoundingMode::Stochastic { seed }),
                0,
            )
        };
        assert_eq!(run(1), run(1));
        assert_ne!(run(1), run(2));
    }

    #[test]
    fn test_exact_stochastic_ternary_is_unbiased() {
        // alpha = mean(|w|) = 0.3; 0.2/alpha = 0.667 → code 1 with probability
        // 0.667, expectation = 0.2. Nearest always gives 1 → 0.3 (offset).
        let data: Vec<f32> = (0..20_000)
            .map(|i| if i % 2 == 0 { 0.2 } else { 0.4 })
            .collect();
        let scheme = PackedScheme::Ternary(ScaleGranularity::PerTensor);

        let nearest = quantize(&data, &cfg_with(scheme, RoundingMode::Nearest), 0).to_f32();
        let stochastic = quantize(
            &data,
            &cfg_with(scheme, RoundingMode::Stochastic { seed: 5 }),
            0,
        )
        .to_f32();

        let mean_of_02 = |v: &[f32]| -> f32 {
            let vals: Vec<f32> = v.iter().step_by(2).copied().collect();
            vals.iter().sum::<f32>() / vals.len() as f32
        };
        assert!(
            mean_of_02(&nearest) > 0.29,
            "nearest={}",
            mean_of_02(&nearest)
        );
        assert!(
            (mean_of_02(&stochastic) - 0.2).abs() < 0.02,
            "stochastic={}",
            mean_of_02(&stochastic)
        );
    }

    #[test]
    fn test_exact_stochastic_asymmetric_keeps_zero_exact() {
        let data: Vec<f32> = (0..999)
            .map(|i| {
                if i % 3 == 0 {
                    0.0
                } else {
                    i as f32 * 0.01 - 5.0
                }
            })
            .collect();
        let scheme = PackedScheme::Int8(ScaleGranularity::PerTensor, Calibration::Asymmetric);
        let out = quantize(
            &data,
            &cfg_with(scheme, RoundingMode::Stochastic { seed: 3 }),
            0,
        )
        .to_f32();
        assert_eq!(out.len(), data.len());
        for (w, r) in data.iter().zip(&out) {
            if *w == 0.0 {
                assert_eq!(*r, 0.0, "zero must stay exactly representable");
            }
        }
    }

    #[test]
    fn test_exact_stochastic_covers_every_intn_width() {
        let data: Vec<f32> = (0..256).map(|i| (i as f32 - 128.0) * 0.013).collect();
        let g = ScaleGranularity::PerGroup { group_size: 64 };
        let schemes = [
            PackedScheme::Ternary(g),
            PackedScheme::Int2(g, Calibration::Symmetric),
            PackedScheme::Int3(g, Calibration::Asymmetric),
            PackedScheme::Int4(g, Calibration::Symmetric),
            PackedScheme::Int5(g, Calibration::Asymmetric),
            PackedScheme::Int6(g, Calibration::Symmetric),
            PackedScheme::Int7(g, Calibration::Asymmetric),
            PackedScheme::Int8(g, Calibration::Symmetric),
        ];
        for scheme in schemes {
            assert!(supports_exact_stochastic(scheme), "{scheme:?}");
            let cfg = cfg_with(scheme, RoundingMode::Stochastic { seed: 9 });
            let buf = quantize(&data, &cfg, 0);
            assert_eq!(buf.numel(), 256, "{scheme:?}");
            // The scheme should not change depending on the rounding method.
            let nearest = quantize(&data, &cfg_with(scheme, RoundingMode::Nearest), 0);
            assert_eq!(buf.scheme_tag(), nearest.scheme_tag(), "{scheme:?}");
        }
    }

    #[test]
    fn test_stochastic_without_exact_support_falls_back_to_jitter() {
        let data: Vec<f32> = (0..64).map(|i| (i as f32 - 32.0) * 0.05).collect();
        for scheme in [
            PackedScheme::Nf4(ScaleGranularity::PerTensor),
            PackedScheme::Int1(ScaleGranularity::PerTensor),
        ] {
            assert!(!supports_exact_stochastic(scheme), "{scheme:?}");
            let cfg = cfg_with(scheme, RoundingMode::Stochastic { seed: 4 });
            let a = quantize(&data, &cfg, 0);
            assert_eq!(a.numel(), 64);
            assert_eq!(
                a,
                quantize(&data, &cfg, 0),
                "fallback must stay deterministic"
            );
        }
    }

    #[test]
    fn test_exact_stochastic_with_clip_adds_no_jitter_noise() {
        // clip + exact SR: data before packaging is only clip, without
        // jitter on top (otherwise you would get double noise).
        let data = dense_cluster(2_001);
        let scheme = PackedScheme::Int4(ScaleGranularity::PerTensor, Calibration::Symmetric);
        let mut clipped_cfg = cfg_with(scheme, RoundingMode::Stochastic { seed: 21 });
        clipped_cfg.outlier_clip_percentile = 0.001;
        let got = quantize(&data, &clipped_cfg, 0);

        let mut clipped = data.clone();
        clip_outliers_inplace(&mut clipped, 0.001);
        let expected = quantize(
            &clipped,
            &cfg_with(scheme, RoundingMode::Stochastic { seed: 21 }),
            0,
        );
        assert_eq!(got, expected);
    }
}
