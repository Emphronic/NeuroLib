// neo-quantization/src/config.rs
//
// `QuantizeConfig` is all `quantize.rs` needs to turn a slice
// f32 into the packed weights of the given scheme. Separated from `PackedScheme` (see
// scheme.rs), because the same scheme can be quantized with different
// rounding/clipping strategies is the orthogonal axis.
//
// `PackedScheme::Int2..Int8` carry `Calibration` directly in the enum variant
// (see scheme.rs); the presets below provide the symmetric default plus
// explicit `*_asymmetric` options.
//
// Presets cover Int2/Int3/Int5/Int6/Int7 and Nf4.
//
// `symmetric: bool` is retained for API compatibility (some existing code may
// rely on it for Int4/Int8), but the source of truth is `Calibration` inside
// `scheme`; both must be consistent, and every preset below guarantees that.

use crate::pack::Calibration;
use crate::scheme::{PackedScheme, ScaleGranularity};

// ═══════════════════════════════════════════════════════════════════════════
// § 1  ROUNDING / CLIPPING
// ═══════════════════════════════════════════════════════════════════════════

/// Rounding strategy for quantization.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum RoundingMode {
    /// Round to nearest (standard for most schemes).
    #[default]
    Nearest,
    /// Stochastic rounding - up with fractional part probability
    /// `w / scale`, down with extra. Unbiased estimate
    /// (`E[rounded] = w`), as opposed to round-to-nearest, whose error
    /// systematic on dense clusters of values, so useful in
    /// QAT-like scenarios. The result is reproduced using `seed`.
    ///
    /// EXACT rounding - for `Ternary` and `Int2..Int8` (symmetric and
    /// asymmetric): RNG is pulled inside the package, the quantization step is known.
    /// For other schemes (`Int1`, `Nf4`, GGUF-block formats) this is only
    /// approximation: multiplicative noise ~0.5% of |w| before
    /// round-to-nearest (see quantize.rs), rather than unbiased rounding.
    Stochastic {
        /// Seed for PRNG (xorshift64 without external dependencies, see rng.rs).
        seed: u64,
    },
}

// ═══════════════════════════════════════════════════════════════════════════
// § 2  QUANTIZE CONFIG
// ═══════════════════════════════════════════════════════════════════════════

/// Configuration of one quantization pass: float slice → packed bytes.
#[derive(Debug, Clone, PartialEq)]
pub struct QuantizeConfig {
    /// Target packaging scheme.
    pub scheme: PackedScheme,

    /// Rounding strategy.
    pub rounding: RoundingMode,

    /// Symmetric quantization (without zero point/offset). Reflects that
    /// the same as `Calibration` inside `scheme` - kept separate
    /// field for the sake of API backward compatibility; all designers/builders
    /// below keep it consistent with `scheme` automatically.
    pub symmetric: bool,

    /// Limit extreme outliers before calculating scale
    /// (percentile clipping), 0.0 = disabled. For example, 0.001 will cut off
    /// 0.1% of the largest absolute weights before calibration - reduces
    /// influence of single emissions on alpha.
    pub outlier_clip_percentile: f32,
}

impl QuantizeConfig {
    /// Ternary with per-tensor scale (classic BitNet b1.58:
    /// `alpha = mean(|W|)`, `round(clip(W/alpha, -1, 1))`).
    pub fn ternary_per_tensor() -> Self {
        QuantizeConfig {
            scheme: PackedScheme::Ternary(ScaleGranularity::PerTensor),
            rounding: RoundingMode::Nearest,
            symmetric: true,
            outlier_clip_percentile: 0.0,
        }
    }

    /// Ternary with per-group scale of a given group size - more precisely,
    /// but requires storing `numel / group_size` additional. float scalars.
    pub fn ternary_per_group(group_size: usize) -> Self {
        QuantizeConfig {
            scheme: PackedScheme::Ternary(ScaleGranularity::PerGroup { group_size }),
            rounding: RoundingMode::Nearest,
            symmetric: true,
            outlier_clip_percentile: 0.0,
        }
    }

    /// Strict 1-bit, per-channel scale (typical choice for honest 1-bit:
    /// one alpha per row of weights gives noticeably better quality than
    /// per-tensor, almost without extra overhead).
    pub fn int1_per_channel() -> Self {
        QuantizeConfig {
            scheme: PackedScheme::Int1(ScaleGranularity::PerChannel),
            rounding: RoundingMode::Nearest,
            symmetric: true,
            outlier_clip_percentile: 0.0,
        }
    }

    /// Int2 symmetric, per-group(32).
    pub fn int2_per_group_32() -> Self {
        Self::intn_preset(
            2,
            ScaleGranularity::PerGroup { group_size: 32 },
            Calibration::Symmetric,
        )
    }
    /// Int2 asymmetric (GPTQ/AWQ style), per-group(32).
    pub fn int2_per_group_32_asymmetric() -> Self {
        Self::intn_preset(
            2,
            ScaleGranularity::PerGroup { group_size: 32 },
            Calibration::Asymmetric,
        )
    }

    /// Int3 symmetric, per-group(32).
    pub fn int3_per_group_32() -> Self {
        Self::intn_preset(
            3,
            ScaleGranularity::PerGroup { group_size: 32 },
            Calibration::Symmetric,
        )
    }
    /// Int3 asymmetric, per-group(32).
    pub fn int3_per_group_32_asymmetric() -> Self {
        Self::intn_preset(
            3,
            ScaleGranularity::PerGroup { group_size: 32 },
            Calibration::Asymmetric,
        )
    }

    /// Int4 is symmetrical, per-group(32) is similar to Q4_0 in spirit.
    pub fn int4_per_group_32() -> Self {
        Self::intn_preset(
            4,
            ScaleGranularity::PerGroup { group_size: 32 },
            Calibration::Symmetric,
        )
    }
    /// Int4 asymmetric (GPTQ/AWQ style), per-group(32).
    pub fn int4_per_group_32_asymmetric() -> Self {
        Self::intn_preset(
            4,
            ScaleGranularity::PerGroup { group_size: 32 },
            Calibration::Asymmetric,
        )
    }

    /// Int4 symmetric, per-group(N) with arbitrary group size -
    /// parameterizable variant of `int4_per_group_32()`, the same pattern as
    /// `ternary_per_group(group_size)` above. Needed by the calling code
    /// which is tied to a specific group_size other than 32 (for example,
    /// maintaining the same `group_size=128` behavior from the old one
    /// neo-synx-cpu::quant_kernel::quantize_b_int4(b, k, n, 128), up to
    /// combining an int4 path with this crate).
    pub fn int4_per_group(group_size: usize) -> Self {
        Self::intn_preset(
            4,
            ScaleGranularity::PerGroup { group_size },
            Calibration::Symmetric,
        )
    }
    /// Int4 asymmetric, per-group(N) with arbitrary group size -
    /// parameterizable version of `int4_per_group_32_asymmetric()`.
    pub fn int4_per_group_asymmetric(group_size: usize) -> Self {
        Self::intn_preset(
            4,
            ScaleGranularity::PerGroup { group_size },
            Calibration::Asymmetric,
        )
    }

    /// Int5 symmetric, per-group(32).
    pub fn int5_per_group_32() -> Self {
        Self::intn_preset(
            5,
            ScaleGranularity::PerGroup { group_size: 32 },
            Calibration::Symmetric,
        )
    }
    /// Int5 asymmetric, per-group(32).
    pub fn int5_per_group_32_asymmetric() -> Self {
        Self::intn_preset(
            5,
            ScaleGranularity::PerGroup { group_size: 32 },
            Calibration::Asymmetric,
        )
    }

    /// Int6 symmetric, per-group(32).
    pub fn int6_per_group_32() -> Self {
        Self::intn_preset(
            6,
            ScaleGranularity::PerGroup { group_size: 32 },
            Calibration::Symmetric,
        )
    }
    /// Int6 asymmetric, per-group(32).
    pub fn int6_per_group_32_asymmetric() -> Self {
        Self::intn_preset(
            6,
            ScaleGranularity::PerGroup { group_size: 32 },
            Calibration::Asymmetric,
        )
    }

    /// Int7 symmetric, per-group(32).
    pub fn int7_per_group_32() -> Self {
        Self::intn_preset(
            7,
            ScaleGranularity::PerGroup { group_size: 32 },
            Calibration::Symmetric,
        )
    }
    /// Int7 asymmetric, per-group(32).
    pub fn int7_per_group_32_asymmetric() -> Self {
        Self::intn_preset(
            7,
            ScaleGranularity::PerGroup { group_size: 32 },
            Calibration::Asymmetric,
        )
    }

    /// Int8 is symmetrical, per-group(32) is similar to Q8_0 in spirit, but with its own
    /// packaging (see pack::int8), not tied to GGUF block layout.
    pub fn int8_per_group_32() -> Self {
        Self::intn_preset(
            8,
            ScaleGranularity::PerGroup { group_size: 32 },
            Calibration::Symmetric,
        )
    }
    /// Int8 asymmetric (GPTQ/AWQ style), per-group(32).
    pub fn int8_per_group_32_asymmetric() -> Self {
        Self::intn_preset(
            8,
            ScaleGranularity::PerGroup { group_size: 32 },
            Calibration::Asymmetric,
        )
    }

    /// NF4 (QLoRA-style), per-group(64) - typical choice of group_size for
    /// double-quantization QLoRA (see pack/nf4.rs).
    pub fn nf4_per_group_64() -> Self {
        QuantizeConfig {
            scheme: PackedScheme::Nf4(ScaleGranularity::PerGroup { group_size: 64 }),
            rounding: RoundingMode::Nearest,
            symmetric: true,
            outlier_clip_percentile: 0.0,
        }
    }

    /// Without quantization - passthrough (for API uniformity, when the scheme
    /// selected dynamically from the configuration file/CLI).
    pub fn float() -> Self {
        QuantizeConfig {
            scheme: PackedScheme::Float,
            rounding: RoundingMode::Nearest,
            symmetric: true,
            outlier_clip_percentile: 0.0,
        }
    }

    /// General constructor for intN presets (N=2..8) - builds the required
    /// option `PackedScheme::IntN(granularity, calibration)` by numeric
    /// `bits`, instead of eight copies of the same function body.
    fn intn_preset(bits: u8, granularity: ScaleGranularity, calibration: Calibration) -> Self {
        let scheme = match bits {
            2 => PackedScheme::Int2(granularity, calibration),
            3 => PackedScheme::Int3(granularity, calibration),
            4 => PackedScheme::Int4(granularity, calibration),
            5 => PackedScheme::Int5(granularity, calibration),
            6 => PackedScheme::Int6(granularity, calibration),
            7 => PackedScheme::Int7(granularity, calibration),
            8 => PackedScheme::Int8(granularity, calibration),
            _ => panic!("QuantizeConfig::intn_preset: bits must be 2..=8, got {bits}"),
        };
        QuantizeConfig {
            scheme,
            rounding: RoundingMode::Nearest,
            symmetric: calibration == Calibration::Symmetric,
            outlier_clip_percentile: 0.0,
        }
    }

    /// Builder style: change rounding.
    pub fn with_rounding(mut self, rounding: RoundingMode) -> Self {
        self.rounding = rounding;
        self
    }

    /// Builder style: enable outlier clipping.
    pub fn with_outlier_clip(mut self, percentile: f32) -> Self {
        debug_assert!(
            (0.0..1.0).contains(&percentile),
            "percentile must be in [0, 1)"
        );
        self.outlier_clip_percentile = percentile;
        self
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// § 3 TESTS
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_presets_are_symmetric() {
        assert!(QuantizeConfig::ternary_per_tensor().symmetric);
        assert!(QuantizeConfig::int1_per_channel().symmetric);
        assert!(QuantizeConfig::int8_per_group_32().symmetric);
        assert!(QuantizeConfig::int4_per_group_32().symmetric);
    }

    #[test]
    fn test_asymmetric_presets_flag_and_scheme_agree() {
        let cfg = QuantizeConfig::int4_per_group_32_asymmetric();
        assert!(!cfg.symmetric);
        assert!(cfg.scheme.is_asymmetric());
    }

    #[test]
    fn test_builder_chain() {
        let cfg = QuantizeConfig::ternary_per_tensor()
            .with_rounding(RoundingMode::Stochastic { seed: 42 })
            .with_outlier_clip(0.001);
        assert_eq!(cfg.rounding, RoundingMode::Stochastic { seed: 42 });
        assert!((cfg.outlier_clip_percentile - 0.001).abs() < 1e-9);
    }

    #[test]
    fn test_group_size_roundtrip() {
        let cfg = QuantizeConfig::ternary_per_group(64);
        match cfg.scheme {
            PackedScheme::Ternary(ScaleGranularity::PerGroup { group_size }) => {
                assert_eq!(group_size, 64);
            }
            _ => panic!("expected ternary per-group scheme"),
        }
    }

    #[test]
    fn test_all_intn_presets_build_expected_scheme() {
        assert!(matches!(
            QuantizeConfig::int2_per_group_32().scheme,
            PackedScheme::Int2(_, Calibration::Symmetric)
        ));
        assert!(matches!(
            QuantizeConfig::int3_per_group_32().scheme,
            PackedScheme::Int3(_, Calibration::Symmetric)
        ));
        assert!(matches!(
            QuantizeConfig::int5_per_group_32().scheme,
            PackedScheme::Int5(_, Calibration::Symmetric)
        ));
        assert!(matches!(
            QuantizeConfig::int6_per_group_32().scheme,
            PackedScheme::Int6(_, Calibration::Symmetric)
        ));
        assert!(matches!(
            QuantizeConfig::int7_per_group_32().scheme,
            PackedScheme::Int7(_, Calibration::Symmetric)
        ));
        assert!(matches!(
            QuantizeConfig::int2_per_group_32_asymmetric().scheme,
            PackedScheme::Int2(_, Calibration::Asymmetric)
        ));
        assert!(matches!(
            QuantizeConfig::int8_per_group_32_asymmetric().scheme,
            PackedScheme::Int8(_, Calibration::Asymmetric)
        ));
    }

    #[test]
    fn test_nf4_preset() {
        let cfg = QuantizeConfig::nf4_per_group_64();
        match cfg.scheme {
            PackedScheme::Nf4(ScaleGranularity::PerGroup { group_size }) => {
                assert_eq!(group_size, 64)
            }
            _ => panic!("expected nf4 per-group scheme"),
        }
    }

    #[test]
    fn test_int4_per_group_arbitrary_size() {
        // The parameterizable preset must accept any group_size (not
        // only 32) - in particular the old default group_size=128 from
        // quant_kernel::quantize_b_int4 before merging with this crate.
        let cfg = QuantizeConfig::int4_per_group(128);
        match cfg.scheme {
            PackedScheme::Int4(
                ScaleGranularity::PerGroup { group_size },
                Calibration::Symmetric,
            ) => {
                assert_eq!(group_size, 128);
            }
            _ => panic!("expected int4 per-group(128) symmetric scheme"),
        }
        assert!(cfg.symmetric);

        let asym = QuantizeConfig::int4_per_group_asymmetric(128);
        match asym.scheme {
            PackedScheme::Int4(
                ScaleGranularity::PerGroup { group_size },
                Calibration::Asymmetric,
            ) => {
                assert_eq!(group_size, 128);
            }
            _ => panic!("expected int4 per-group(128) asymmetric scheme"),
        }
        assert!(!asym.symmetric);
    }
}
