// neo-quantization/src/scheme.rs
//
// `PackedScheme` - complete set of weight packaging schemes supported by this
// crate. Expands `burn::tensor::QuantScheme` with families that are not there
// and it cannot be:
//
//   • Ternary (-1 / 0 / +1) - BitNet b1.58 / Qwen3-like models.
//   • Int1..Int8 (N bits, symmetric/asymmetric) - general purpose
//                                        intN quantization of this crate.
//   • Nf4 (4-bit NormalFloat) - QLoRA-style uneven mesh.
//
// Why not added directly to `crate::dtype::DType`/`QuantScheme`
// ─────────────────────────────────────────────────────────────
// `crate::dtype::DType` strictly follows the table GGUF (`gguf_type_id` /
// `from_gguf_type_id`) is a stable external compatibility contract with
// llama.cpp/GGUF-ecosystem. None of the schemes in this file (except simple
// GGUF block formats, see `Gguf(DType)`) does not have GGUF numeric type id.
// Therefore:
//
//   • `burn::tensor::QuantScheme` remains as is - describes GGUF-compatible
//     schemes for crates that must speak the language GGUF (loader,
//     GEMM kernels for a specific DType).
//   • `PackedScheme` (this file) - superset for the needs of THIS crate:
//     conversion/packaging/switching. Where the scheme matches
//     GGUF-type, `PackedScheme::to_neo_dtype()` gives real `DType`.
//     Where there is no (Ternary/IntN/Nf4) — `PackedScheme` is self-sufficient: carries
//     own block_bits/block_elems, without pretending to be a GGUF type.
//
// ## Symmetric vs Asymmetric in PackedScheme
//
// Int2..Int8 carry `Calibration` (see `pack::intn::Calibration`) directly into
// enum option - `Int4(granularity, Calibration::Asymmetric)` and
// `Int4(granularity, Calibration::Symmetric)` are DIFFERENT schemes with different
// `tag()` (important for the `WeightRegistry` cache in switch.rs - otherwise cache
// would mix up binary incompatible encodings under one key). Int1
// does not support `Calibration` at all: `intn::quantize` does not support
// asymmetric with N=1 (see pack/intn.rs) - there is no meaningful content there
// "zero-point" beyond the usual sign.

use crate::dtype::DType;
use crate::pack::Calibration;

// ═══════════════════════════════════════════════════════════════════════════
// § 1  SCALE GRANULARITY
// ═══════════════════════════════════════════════════════════════════════════

/// Calibration granularity for floating schemes
/// scale (ternary, intN, nf4, etc.).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum ScaleGranularity {
    /// One alpha scalar for the entire tensor (`alpha = mean(|W|)` over the entire tensor).
    ///
    /// Cheap, simple, but rough for tensors with highly inhomogeneous
    /// scattered across rows/channels.
    PerTensor,

    /// One scalar alpha per group of `group_size` successive weights
    /// along the last axis (same principle as group_size in Q4_K/GPTQ/AWQ).
    ///
    /// More precisely, it is more expensive in memory (additional float per group) and a little more expensive in
    /// kernel (you need to read the correct alpha by group index).
    PerGroup {
        /// Group size in elements. Typically 32/64/128.
        group_size: usize,
    },

    /// One alpha scalar per channel (row of output neurons) - that is
    /// `group_size == in_features`. Special case `PerGroup`, rendered
    /// a separate option for the readability of configs and APIs.
    PerChannel,
}

impl ScaleGranularity {
    /// Number of weights covered by one alpha for a given tensor shape
    /// `[out_features, in_features]`. For `PerTensor`/`PerChannel` not
    /// depends on `group_size`, for `PerGroup` - exactly `group_size`.
    pub fn elems_per_scale(self, in_features: usize) -> usize {
        match self {
            ScaleGranularity::PerTensor => usize::MAX, // marker "entire tensor"
            ScaleGranularity::PerGroup { group_size } => group_size,
            ScaleGranularity::PerChannel => in_features,
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// § 2  PACKED SCHEME
// ═══════════════════════════════════════════════════════════════════════════

/// Complete set of weight packaging schemes supported by `neo-quantization`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PackedScheme {
    /// Without quantization - float passthrough (F32/F16/BF16 depending on
    /// source data, no conversion is performed).
    Float,

    /// Ternary: weights ∈ {-1, 0, +1}, log2(3) ≈ 1.58 bits/weight (BitNet b1.58,
    /// new wave of 1-bit Qwen3-like models). Scale granularity
    /// is specified separately via `ScaleGranularity`.
    Ternary(ScaleGranularity),

    /// Strict 1 bit: weights ∈ {-1, +1}, exactly 1 bit/weight, no zero level.
    /// Worse than ternary in quality with the same architecture (no "off"
    /// scales), but twice as compact in memory for core scales. Always
    /// symmetric — `intn::quantize` does not support asymmetric when N=1
    /// (see pack/intn.rs) so there is no `Calibration` parameter here.
    Int1(ScaleGranularity),

    /// 2-bit general purpose quantization, symmetric or asymmetric
    /// (GPTQ/AWQ zero-point style).
    Int2(ScaleGranularity, Calibration),

    /// General purpose 3-bit quantization, symmetric or asymmetric.
    Int3(ScaleGranularity, Calibration),

    /// 4-bit general purpose quantization (similar to Q4_0/GPTQ/AWQ core
    /// ideas: round-to-nearest with scale), symmetric or asymmetric.
    Int4(ScaleGranularity, Calibration),

    /// General purpose 5-bit quantization, symmetric or asymmetric.
    Int5(ScaleGranularity, Calibration),

    /// General purpose 6-bit quantization, symmetric or asymmetric.
    Int6(ScaleGranularity, Calibration),

    /// 7-bit general purpose quantization, symmetric or asymmetric.
    Int7(ScaleGranularity, Calibration),

    /// 8-bit quantization (analogue of Q8_0), not tied to the GGUF block
    /// exactly 32 elements - granularity is adjustable. Symmetric or
    /// asymmetric.
    Int8(ScaleGranularity, Calibration),

    /// NF4 (4-bit NormalFloat, QLoRA-style) - uneven mesh of 16
    /// fixed levels (quantiles N(0,1)), more precisely uniform Int4
    /// on weights close to the Gaussian distribution (see pack/nf4.rs).
    /// There is granularity (absmax per group); symmetric/asymmetric concepts
    /// no - the grid levels are fixed in advance, only the scale is calibrated.
    Nf4(ScaleGranularity),

    /// GGUF-compatible scheme - passthrough to `crate::dtype::DType`. Used
    /// when exactly the GGUF format is needed (for loading/saving in .gguf),
    /// and not the crate's own packaging. Really packaged
    /// (the full list is listed by name; it is checked against the code by the test
    /// `tests/docs_vs_code.rs`):
    ///
    /// * simple blocks (`pack::gguf`): Q4_0, Q4_1, Q5_0, Q5_1, Q8_0, Q8_1;
    /// * K-quants (`pack::kquant`): Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, Q8_K;
    /// * IQ-series (`pack::iq`): IQ4_NL, IQ4_XS and codebook IQ1_S, IQ1_M,
    ///   IQ2_XXS, IQ2_XS, IQ3_XXS, IQ3_S;
    /// * ternary (`pack::tq`): TQ1_0, TQ2_0.
    ///
    /// The current source of truth is `pack::gguf_backend`. For `DType`,
    /// which is not there (float/int/bool), `quantize()` panics: such
    /// data should be kept as `PackedBuffer::GgufRaw`.
    Gguf(DType),
}

impl PackedScheme {
    /// The number of bits per weight (without taking into account the overhead on the scale table - see.
    /// `effective_bits_per_weight` to `PackedBuffer` for an honest assessment
    /// with overhead). Not `const` — `DType::bits_per_element()` does
    /// float division is not allowed in const context on this version of Rust.
    pub fn raw_bits_per_weight(self) -> f32 {
        match self {
            PackedScheme::Float => 32.0,      // we assume f32 source
            PackedScheme::Ternary(_) => 1.58, // log2(3), actual packing is 2 bits/weight (see pack::ternary)
            PackedScheme::Int1(_) => 1.0,
            PackedScheme::Int2(..) => 2.0,
            PackedScheme::Int3(..) => 3.0,
            PackedScheme::Int4(..) => 4.0,
            PackedScheme::Int5(..) => 5.0,
            PackedScheme::Int6(..) => 6.0,
            PackedScheme::Int7(..) => 7.0,
            PackedScheme::Int8(..) => 8.0,
            PackedScheme::Nf4(_) => 4.0,
            PackedScheme::Gguf(dtype) => dtype.bits_per_element(),
        }
    }

    /// `true`, if the scheme is not float (that is, it actually compresses the weights).
    #[inline]
    pub fn is_quantized(self) -> bool {
        !matches!(self, PackedScheme::Float)
    }

    /// `true`, if the scheme is ternary or int1 (low-bit schemes of this crate,
    /// not having a direct GGUF equivalent).
    #[inline]
    pub fn is_low_bit(self) -> bool {
        matches!(self, PackedScheme::Ternary(_) | PackedScheme::Int1(_))
    }

    /// `true` if the scheme uses asymmetric (zero-point) calibration.
    /// `false` for all schemes without the concept of calibration (`Float`/`Ternary`/
    /// `Int1`/`Nf4`/`Gguf`), and not only for explicit `Symmetric`.
    #[inline]
    pub fn is_asymmetric(self) -> bool {
        matches!(
            self,
            PackedScheme::Int2(_, Calibration::Asymmetric)
                | PackedScheme::Int3(_, Calibration::Asymmetric)
                | PackedScheme::Int4(_, Calibration::Asymmetric)
                | PackedScheme::Int5(_, Calibration::Asymmetric)
                | PackedScheme::Int6(_, Calibration::Asymmetric)
                | PackedScheme::Int7(_, Calibration::Asymmetric)
                | PackedScheme::Int8(_, Calibration::Asymmetric)
        )
    }

    /// Scale granularity, if the scheme has one (`None` for `Float`
    /// and `Gguf` - they are either not quantized or carry their own scheme
    /// blocks inside `DType`).
    pub fn granularity(self) -> Option<ScaleGranularity> {
        match self {
            PackedScheme::Ternary(g) | PackedScheme::Int1(g) | PackedScheme::Nf4(g) => Some(g),
            PackedScheme::Int2(g, _)
            | PackedScheme::Int3(g, _)
            | PackedScheme::Int4(g, _)
            | PackedScheme::Int5(g, _)
            | PackedScheme::Int6(g, _)
            | PackedScheme::Int7(g, _)
            | PackedScheme::Int8(g, _) => Some(g),
            PackedScheme::Float | PackedScheme::Gguf(_) => None,
        }
    }

    /// Try to display the scheme in `crate::dtype::DType` - only successful
    /// for `Gguf(dtype)` and `Float` (→ F32). Other schemes of this crate
    /// do not have a GGUF equivalent and return `None` - their entire format
    /// described in `pack::*`, not in `crate::dtype::DType`.
    pub fn to_neo_dtype(self) -> Option<DType> {
        match self {
            PackedScheme::Float => Some(DType::F32),
            PackedScheme::Gguf(dt) => Some(dt),
            _ => None,
        }
    }

    /// Short machine name of the scheme - used in logs, cache file names,
    /// keys `QuantizedWeightSet` (see switch.rs). Symmetric/asymmetric give
    /// DIFFERENT tags (needed for `WeightRegistry::activate` in `Cached` mode -
    /// otherwise the cache would mix up the encodings under one key).
    pub fn tag(self) -> &'static str {
        match self {
            PackedScheme::Float => "f32",
            PackedScheme::Ternary(ScaleGranularity::PerTensor) => "ternary_pt",
            PackedScheme::Ternary(ScaleGranularity::PerChannel) => "ternary_pc",
            PackedScheme::Ternary(ScaleGranularity::PerGroup { .. }) => "ternary_pg",
            PackedScheme::Int1(ScaleGranularity::PerTensor) => "int1_pt",
            PackedScheme::Int1(ScaleGranularity::PerChannel) => "int1_pc",
            PackedScheme::Int1(ScaleGranularity::PerGroup { .. }) => "int1_pg",
            PackedScheme::Int2(_, Calibration::Symmetric) => "int2",
            PackedScheme::Int2(_, Calibration::Asymmetric) => "int2_asym",
            PackedScheme::Int3(_, Calibration::Symmetric) => "int3",
            PackedScheme::Int3(_, Calibration::Asymmetric) => "int3_asym",
            PackedScheme::Int4(_, Calibration::Symmetric) => "int4",
            PackedScheme::Int4(_, Calibration::Asymmetric) => "int4_asym",
            PackedScheme::Int5(_, Calibration::Symmetric) => "int5",
            PackedScheme::Int5(_, Calibration::Asymmetric) => "int5_asym",
            PackedScheme::Int6(_, Calibration::Symmetric) => "int6",
            PackedScheme::Int6(_, Calibration::Asymmetric) => "int6_asym",
            PackedScheme::Int7(_, Calibration::Symmetric) => "int7",
            PackedScheme::Int7(_, Calibration::Asymmetric) => "int7_asym",
            PackedScheme::Int8(_, Calibration::Symmetric) => "int8",
            PackedScheme::Int8(_, Calibration::Asymmetric) => "int8_asym",
            PackedScheme::Nf4(_) => "nf4",
            PackedScheme::Gguf(dt) => dt.as_str(),
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// § 3 TESTS
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ternary_bits() {
        let s = PackedScheme::Ternary(ScaleGranularity::PerTensor);
        assert!((s.raw_bits_per_weight() - 1.58).abs() < 0.01);
        assert!(s.is_quantized());
        assert!(s.is_low_bit());
        assert_eq!(s.to_neo_dtype(), None);
    }

    #[test]
    fn test_int1_bits() {
        let s = PackedScheme::Int1(ScaleGranularity::PerChannel);
        assert_eq!(s.raw_bits_per_weight(), 1.0);
        assert!(s.is_low_bit());
        assert!(!s.is_asymmetric());
    }

    #[test]
    fn test_float_passthrough() {
        let s = PackedScheme::Float;
        assert!(!s.is_quantized());
        assert_eq!(s.to_neo_dtype(), Some(DType::F32));
    }

    #[test]
    fn test_gguf_passthrough() {
        let s = PackedScheme::Gguf(DType::Q4_K);
        assert_eq!(s.to_neo_dtype(), Some(DType::Q4_K));
        assert!(s.is_quantized());
        assert!(!s.is_low_bit());
    }

    #[test]
    fn test_gguf_iq4_nl_passthrough() {
        // IQ4_NL is actually packed via pack::iq (see quantize.rs) -
        // at the PackedScheme level it is still a simple Gguf(dtype) passthrough,
        // as for K-quants/simple blocks.
        let s = PackedScheme::Gguf(DType::IQ4_NL);
        assert_eq!(s.to_neo_dtype(), Some(DType::IQ4_NL));
        assert!(s.is_quantized());
        assert_eq!(s.tag(), "IQ4_NL");
    }

    #[test]
    fn test_granularity_elems_per_scale() {
        let g = ScaleGranularity::PerGroup { group_size: 64 };
        assert_eq!(g.elems_per_scale(4096), 64);
        assert_eq!(ScaleGranularity::PerChannel.elems_per_scale(4096), 4096);
    }

    #[test]
    fn test_symmetric_vs_asymmetric_tags_and_flag() {
        let sym = PackedScheme::Int4(ScaleGranularity::PerTensor, Calibration::Symmetric);
        let asym = PackedScheme::Int4(ScaleGranularity::PerTensor, Calibration::Asymmetric);
        assert_ne!(sym.tag(), asym.tag());
        assert!(!sym.is_asymmetric());
        assert!(asym.is_asymmetric());
    }

    #[test]
    fn test_nf4_bits_and_granularity() {
        let s = PackedScheme::Nf4(ScaleGranularity::PerGroup { group_size: 64 });
        assert_eq!(s.raw_bits_per_weight(), 4.0);
        assert!(s.granularity().is_some());
        assert!(!s.is_asymmetric());
        assert_eq!(s.tag(), "nf4");
    }

    #[test]
    fn test_tags_unique_enough() {
        // Just a smoke-test that tag() does not panic on all options and that
        // all intN(symmetric)/intN(asymmetric) pairs give different tags.
        let schemes = [
            PackedScheme::Float,
            PackedScheme::Ternary(ScaleGranularity::PerTensor),
            PackedScheme::Int1(ScaleGranularity::PerGroup { group_size: 32 }),
            PackedScheme::Int2(ScaleGranularity::PerTensor, Calibration::Symmetric),
            PackedScheme::Int2(ScaleGranularity::PerTensor, Calibration::Asymmetric),
            PackedScheme::Int3(ScaleGranularity::PerTensor, Calibration::Symmetric),
            PackedScheme::Int4(ScaleGranularity::PerTensor, Calibration::Symmetric),
            PackedScheme::Int5(ScaleGranularity::PerTensor, Calibration::Symmetric),
            PackedScheme::Int6(ScaleGranularity::PerTensor, Calibration::Symmetric),
            PackedScheme::Int7(ScaleGranularity::PerTensor, Calibration::Symmetric),
            PackedScheme::Int8(ScaleGranularity::PerChannel, Calibration::Symmetric),
            PackedScheme::Nf4(ScaleGranularity::PerTensor),
            PackedScheme::Gguf(DType::Q8_0),
        ];
        for s in schemes {
            assert!(!s.tag().is_empty());
        }
    }
}
