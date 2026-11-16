// neo-quantization/src/pack/mod.rs
//
// Weight packaging formats + single enum `PackedBuffer` erasing a specific
// type (`TernaryPacked`/`Int1Packed`/...) behind one crate-wide abstraction.
// Needed for `switch.rs`, where `WeightRegistry` should be able to hold
// mixed different schemes under the same tensor name without a generic parameter
// at every point of use.
//
// Format coverage:
//   • `intn` - a common engine for int1..int8 (symmetric/asymmetric);
//     int1/int4/int8 are thin public wrappers over it, and int2/int3/int5/int6/int7
//     are exposed through the same engine.
//   • `nf4` - NF4 (4-bit NormalFloat, QLoRA-style), uneven
//     quantization grid under a Gaussian distribution of weights.
//   • `PackedBuffer` carries variants for Int1..Int8 and Nf4.
//   • GGUF formats (bitwise compatible with llama.cpp): simple blocks -
//     `pack::gguf`, K-quants — `pack::kquant`, IQ-series — `pack::iq`
//     (+ `iq1`, `iq_codebook`), ternary – `pack::tq`. Which DType is packed
//     by which backend - `pack::gguf_backend`.

use burn::tensor::f16;

// ═══════════════════════════════════════════════════════════════════════════
//  General scale recording helpers
// ═══════════════════════════════════════════════════════════════════════════

/// Convert the scale to `f16` with SATURATION instead of overflowing to `±inf`.
///
/// `f16::from_f32` returns infinity for anything greater
/// `f16::MAX` (65504). All GGUF formats store the block scale exactly in
/// `f16`, so one weight is of the order of 1e6 - and such ones are found in
/// non-normalized checkpoints and in outlier tensors - turned the scale
/// in `inf`, and the WHOLE block during dequantization became `inf`/`NaN`, silently and
/// without a single mistake. The poisoned block further spreads throughout the network.
///
/// Saturation limits the damage to one block: the weights in it will be
/// pressed to `f16::MAX`, which is still an inevitable loss for such inputs,
/// but the rest of the model remains a number, not `NaN`.
#[inline]
pub(crate) fn f16_saturating(x: f32) -> f16 {
    if x.is_nan() {
        return f16::from_f32(0.0);
    }
    let max = f16::MAX.to_f32();
    f16::from_f32(x.clamp(-max, max))
}

/// BF16 (bfloat16) - 1s 8e 7m, same exponential range as f32.
/// Self-contained float (without scale/group), unlike intN/nf4.
pub mod bf16;
/// FP16 (IEEE half-precision) - 1s 5e 10m, narrower exponential range than
/// f32/bf16 - requires fair overflow/subnormal processing when converting.
pub mod fp16;
/// FP8 (E4M3/E5M2) - 8-bit float with scale (range is too narrow for
/// direct storage of weights without normalization): tensor-wide (`Fp8Packed`) or
/// per group/channel (`Fp8GroupPacked`).
pub mod fp8;
/// GGUF-compatible (llama.cpp) block packaging: Q4_0/Q4_1/Q5_0/Q5_1/Q8_0
/// and (in the future) K-quants/IQ-series.
pub mod gguf;
/// The strict 1-bit format {-1,+1} is a thin wrapper over `intn`.
pub mod int1;
/// The 2-bit format is a thin wrapper over `intn`.
pub mod int2;
/// The 3-bit format is a thin wrapper over `intn`.
pub mod int3;
/// The 4-bit format is a thin wrapper over `intn`.
pub mod int4;
/// The 5-bit format is a thin wrapper over `intn`.
pub mod int5;
/// The 6-bit format is a thin wrapper over `intn`.
pub mod int6;
/// The 7-bit format is a thin wrapper over `intn`.
pub mod int7;
/// The 8-bit format is a thin wrapper over `intn`.
pub mod int8;
/// General intN-quantization engine (N=1..8), symmetric and asymmetric.
pub mod intn;
/// IQ series (importance-weighted quants): `IQ4_NL`/`IQ4_XS` – non-linear
/// 4-bit with a fixed 16-level grid; `IQ2_XXS`/`IQ2_XS`/
/// `IQ3_XXS`/`IQ3_S` - dispatched to `pack::iq_codebook`
/// (codebook/lattice-quantization, see this module for details).
pub mod iq;
/// Codebook quantization `IQ1_S`/`IQ1_M` - packing and unpacking based on
/// `quantize_row_iq1_s_impl`/`_iq1_m_impl` from `ggml-quants.c`. Dispatched
/// through `IqPacked`/`quantize_iq` (see `pack::iq`) as part of the IQ series.
pub mod iq1;
/// Codebook/lattice-quantization `IQ2_XXS`/`IQ2_XS`/`IQ3_XXS`/`IQ3_S` —
/// implementation `quantize_row_iq*_ref`/`dequantize_row_iq*` from `ggml-quants.c`.
pub mod iq_codebook;
/// Grid/sign lookup tables for codebook-based IQ formats, matching
/// `ggml/src/ggml-common.h` byte for byte (see iq_grids.rs).
pub mod iq_grids;
/// K-quant super-block packaging (llama.cpp): Q2_K..Q8_K (`QK_K=256`,
/// two-level scale). IQ-series (codebook-based) is not here.
pub mod kquant;
/// NF4 (4-bit NormalFloat, QLoRA style) - uneven mesh.
pub mod nf4;
/// Ternary {-1,0,+1} packaging (BitNet b1.58-style), 2 bits/weight.
pub mod ternary;
/// `TQ1_0`/`TQ2_0` - GGUF-compatible (llama.cpp) ternary super-block
/// packaging, separate from this crate's `pack::ternary::TernaryPacked`
/// (see tq.rs § 1 for differences in calibration and layout).
pub mod tq;

pub use bf16::{bf16_to_f32, dequantize_bf16, f32_to_bf16, quantize_bf16, Bf16Packed};
pub use fp16::{dequantize_fp16, f32_to_fp16, fp16_to_f32, quantize_fp16, Fp16Packed};
pub use fp8::{
    dequantize_fp8, dequantize_fp8_grouped, f32_to_fp8_e4m3, f32_to_fp8_e5m2, fp8_e4m3_to_f32,
    fp8_e5m2_to_f32, quantize_fp8, quantize_fp8_grouped, Fp8GroupPacked, Fp8Packed, Fp8Variant,
};
pub use gguf::GgufPacked;
pub use int1::{dequantize_int1, quantize_int1, Int1Packed};
pub use int2::{dequantize_int2, quantize_int2, quantize_int2_asymmetric, Int2Packed};
pub use int3::{dequantize_int3, quantize_int3, quantize_int3_asymmetric, Int3Packed};
pub use int4::{dequantize_int4, quantize_int4, quantize_int4_asymmetric, Int4Packed};
pub use int5::{dequantize_int5, quantize_int5, quantize_int5_asymmetric, Int5Packed};
pub use int6::{dequantize_int6, quantize_int6, quantize_int6_asymmetric, Int6Packed};
pub use int7::{dequantize_int7, quantize_int7, quantize_int7_asymmetric, Int7Packed};
pub use int8::{dequantize_int8, quantize_int8, quantize_int8_asymmetric, Int8Packed};
pub use intn::{Calibration, IntNPacked};
pub use iq::{dequantize_iq, quantize_iq, Iq4NlPacked, Iq4XsPacked, IqPacked};
pub use iq1::{
    quantize_iq1_m, quantize_iq1_s, quantize_iq1_s_uncalibrated, Iq1MPacked, Iq1SPacked,
};
pub use kquant::KQuantPacked;
pub use nf4::{dequantize_nf4, quantize_nf4, Nf4Packed, NF4_LEVELS};
pub use ternary::{dequantize_ternary, quantize_ternary, TernaryPacked};
/// Which `pack`-module packs this `DType` at
/// `PackedScheme::Gguf(dtype)`. The only source of truth about what
/// GGUF formats are actually implemented: they are used by `quantize()`,
/// `dequantize()` and the “documentation versus code” test
/// (`tests/docs_vs_code.rs`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GgufBackend {
    /// Simple blocks Q4_0..Q8_1 - [`gguf`].
    Simple,
    /// K-quant super-block Q2_K..Q8_K — [`kquant`].
    KQuant,
    /// IQ series (IQ4_NL/IQ4_XS and codebook IQ1/IQ2/IQ3) - [`iq`].
    Iq,
    /// Ternary TQ1_0/TQ2_0 - [`tq`].
    Tq,
}

/// Is `dtype` implemented as a GGUF packaging format, and if so, in what way?
/// module. `None` is not a GGUF block type (float/int/bool) or the format is not
/// implemented; `quantize()` panics at this.
pub fn gguf_backend(dtype: crate::dtype::DType) -> Option<GgufBackend> {
    if gguf::is_supported(dtype) {
        Some(GgufBackend::Simple)
    } else if kquant::is_supported(dtype) {
        Some(GgufBackend::KQuant)
    } else if iq::is_supported(dtype) {
        Some(GgufBackend::Iq)
    } else if tq::is_tq_series(dtype) {
        Some(GgufBackend::Tq)
    } else {
        None
    }
}

pub use tq::{
    dequantize_tq, is_tq_series, quantize_tq, quantize_tq1_0, quantize_tq2_0, Tq1_0Packed,
    Tq2_0Packed, TqPacked,
};

/// Parse raw GGUF blocks into typed [`PackedBuffer`] of the desired
/// format (`Gguf`/`KQuant`/`Iq`/`Tq` - according to [`gguf_backend`]).
///
/// This is the only place that knows how `PackedBuffer::GgufRaw` turns
/// into a buffer that can decode `to_f32`/`get_f32`/`dequantize*`.
/// The block layout is already bitwise compatible with llama.cpp, so the bytes
/// reused as is, without repackaging.
///
/// `numel` may not be divisible by the block size: the last block is then incomplete
/// (padded with zeros during quantization, as in `quantize_*`). Length `bytes`
/// must be exactly `ceil(numel / block_elems) * block_bytes`.
///
/// # Bugs
///
/// - `dtype` is not a GGUF-block format from among those implemented (see.
///   [`gguf_backend`]);
/// - length `bytes` does not correspond to `numel`.
pub fn gguf_raw_to_typed(
    bytes: Vec<u8>,
    dtype: DType,
    numel: usize,
) -> Result<PackedBuffer, String> {
    let backend = gguf_backend(dtype).ok_or_else(|| {
        format!(
            "{dtype:?} is not the GGUF-block format implemented in \
             neo-quantization (see pack::gguf_backend)"
        )
    })?;
    let block_elems = dtype.block_elems();
    let block_bytes = dtype.block_bytes();
    let expected = numel.div_ceil(block_elems) * block_bytes;
    if bytes.len() != expected {
        return Err(format!(
            "{dtype:?}: for {numel} scales you need {expected} B\
             (block: {block_elems} element = {block_bytes} B), received {} B",
            bytes.len()
        ));
    }
    let blocks = bytes;
    let mismatch =
        || format!("{dtype:?}: the type does not match the selected backend {backend:?}");
    Ok(match backend {
        GgufBackend::Simple => PackedBuffer::Gguf(GgufPacked {
            dtype,
            blocks,
            numel,
        }),
        GgufBackend::KQuant => PackedBuffer::KQuant(KQuantPacked {
            dtype,
            blocks,
            numel,
        }),
        GgufBackend::Iq => PackedBuffer::Iq(match dtype {
            DType::IQ4_NL => IqPacked::Iq4Nl(Iq4NlPacked { blocks, numel }),
            DType::IQ4_XS => IqPacked::Iq4Xs(Iq4XsPacked { blocks, numel }),
            DType::IQ2_XXS => IqPacked::Iq2Xxs(iq_codebook::Iq2XxsPacked { blocks, numel }),
            DType::IQ2_XS => IqPacked::Iq2Xs(iq_codebook::Iq2XsPacked { blocks, numel }),
            DType::IQ3_XXS => IqPacked::Iq3Xxs(iq_codebook::Iq3XxsPacked { blocks, numel }),
            DType::IQ3_S => IqPacked::Iq3S(iq_codebook::Iq3SPacked { blocks, numel }),
            DType::IQ1_S => IqPacked::Iq1S(Iq1SPacked { blocks, numel }),
            DType::IQ1_M => IqPacked::Iq1M(Iq1MPacked { blocks, numel }),
            _ => return Err(mismatch()),
        }),
        GgufBackend::Tq => PackedBuffer::Tq(match dtype {
            DType::TQ1_0 => TqPacked::Tq1_0(Tq1_0Packed { blocks, numel }),
            DType::TQ2_0 => TqPacked::Tq2_0(Tq2_0Packed { blocks, numel }),
            _ => return Err(mismatch()),
        }),
    })
}

use crate::dtype::DType;

// ═══════════════════════════════════════════════════════════════════════════
// § 1  PACKED BUFFER (type-erased)
// ═══════════════════════════════════════════════════════════════════════════

/// Packed weights of a single tensor, independent of the specific design.
///
/// This crate-wide "sum type" on top of all pack::* formats is what
/// is actually stored in the `WeightRegistry` registry (see switch.rs) and that
/// returns `convert.rs` for one model tensor.
#[derive(Debug, Clone, PartialEq)]
pub enum PackedBuffer {
    /// Unquantized data (passthrough).
    Float(Vec<f32>),
    /// Ternary {-1,0,+1}, 2 bits/weight.
    Ternary(TernaryPacked),
    /// Strict 1 bit {-1,+1}.
    Int1(Int1Packed),
    /// 2-bit quantization (symmetric max-abs or asymmetric).
    Int2(Int2Packed),
    /// 3-bit quantization.
    Int3(Int3Packed),
    /// 4-bit quantization.
    Int4(Int4Packed),
    /// 5-bit quantization.
    Int5(Int5Packed),
    /// 6-bit quantization.
    Int6(Int6Packed),
    /// 7-bit quantization.
    Int7(Int7Packed),
    /// 8-bit quantization.
    Int8(Int8Packed),
    /// NF4 (4-bit NormalFloat, QLoRA-style) - uneven mesh for
    /// Gaussian distribution of weights, see nf4.rs.
    Nf4(Nf4Packed),
    /// BF16 (bfloat16) - 1s 8e 7m, same exponential range as f32.
    Bf16(Bf16Packed),
    /// FP16 (IEEE half-precision) — 1s 5e 10m.
    Fp16(Fp16Packed),
    /// FP8 (E4M3 or E5M2) - 8-bit float, the only tensor-wide scale.
    Fp8(Fp8Packed),
    /// GGUF-compatible packaging (Q4_0/Q4_1/Q5_0/Q5_1/Q8_0/K-quants/IQ),
    /// bitwise identical to llama.cpp - see pack::gguf.
    Gguf(GgufPacked),
    /// K-quant super-block packaging (Q2_K..Q8_K), bit identical
    /// llama.cpp - see pack::kquant.
    KQuant(KQuantPacked),
    /// IQ-series - all eight DTypes are implemented: IQ4_NL/IQ4_XS (fixed
    /// non-linear grid), IQ2_XXS/IQ2_XS/IQ3_XXS/IQ3_S (codebook/lattice,
    /// see pack::iq_codebook) and IQ1_S/IQ1_M (codebook, see pack::iq1;
    /// fix - was not connected to PackedBuffer before).
    Iq(IqPacked),
    /// TQ1_0/TQ2_0 — GGUF-compatible ternary super-block packaging,
    /// SEPARATE from `PackedBuffer::Ternary` (see pack::tq § 1 for difference).
    /// Fix - Wasn't connected to PackedBuffer before.
    Tq(TqPacked),
    /// Custom scheme (see [`crate::custom`]). Dynamic
    /// dispatching: the crate does not know its format, but knows how to ask
    /// it has the size, bits/weight and expand back to f32.
    Custom(crate::custom::CustomPacked),
    /// Raw bytes in GGUF-compatible format WITHOUT parsing the structure -
    /// container for blocks read from the file as is. Decoded
    /// “on the fly”: `to_f32`/`get_f32`/`dequantize*` parse bytes into
    /// typed buffer via [`gguf_raw_to_typed`]. For `dtype`, which
    /// not in [`gguf_backend`], or if the length of `bytes` does not match with `numel`
    /// decoding panics; You can check this yourself in advance
    /// [`gguf_raw_to_typed`], it returns `Result`.
    GgufRaw {
        /// Raw bytes in the original GGUF block-layout of the given `dtype`.
        bytes: Vec<u8>,
        /// GGUF-type defining the structure `bytes` (block size, layout).
        dtype: DType,
        /// The initial number of weights that these bytes represent.
        numel: usize,
    },
}

impl PackedBuffer {
    /// The number of source weights (elements) that the buffer represents.
    pub fn numel(&self) -> usize {
        match self {
            PackedBuffer::Float(v) => v.len(),
            PackedBuffer::Ternary(p) => p.numel,
            PackedBuffer::Int1(p) => p.numel(),
            PackedBuffer::Int2(p) => p.numel(),
            PackedBuffer::Int3(p) => p.numel(),
            PackedBuffer::Int4(p) => p.numel(),
            PackedBuffer::Int5(p) => p.numel(),
            PackedBuffer::Int6(p) => p.numel(),
            PackedBuffer::Int7(p) => p.numel(),
            PackedBuffer::Int8(p) => p.numel(),
            PackedBuffer::Nf4(p) => p.numel,
            PackedBuffer::Bf16(p) => p.numel,
            PackedBuffer::Fp16(p) => p.numel,
            PackedBuffer::Fp8(p) => p.numel,
            PackedBuffer::Gguf(p) => p.numel(),
            PackedBuffer::KQuant(p) => p.numel(),
            PackedBuffer::Iq(p) => p.numel(),
            PackedBuffer::Tq(p) => p.numel(),
            PackedBuffer::Custom(p) => p.numel(),
            PackedBuffer::GgufRaw { numel, .. } => *numel,
        }
    }

    /// General intN engine if the buffer is one of the `Int1..Int8` schemes.
    ///
    /// Need fused GEMM cores: they read codes/scales/zero-points directly and
    /// do not turn the weights at f32.
    pub fn as_intn(&self) -> Option<&IntNPacked> {
        match self {
            PackedBuffer::Int1(p) => Some(p.as_intn()),
            PackedBuffer::Int2(p) => Some(p.as_intn()),
            PackedBuffer::Int3(p) => Some(p.as_intn()),
            PackedBuffer::Int4(p) => Some(p.as_intn()),
            PackedBuffer::Int5(p) => Some(p.as_intn()),
            PackedBuffer::Int6(p) => Some(p.as_intn()),
            PackedBuffer::Int7(p) => Some(p.as_intn()),
            PackedBuffer::Int8(p) => Some(p.as_intn()),
            _ => None,
        }
    }

    /// The size in bytes actually occupied by the data (without taking into account scales -
    /// use `effective_bits_per_weight` for a fair estimate).
    pub fn data_bytes(&self) -> usize {
        match self {
            PackedBuffer::Float(v) => v.len() * 4,
            PackedBuffer::Ternary(p) => p.bits_bytes(),
            PackedBuffer::Int1(p) => p.bits_bytes(),
            PackedBuffer::Int2(p) => p.bytes_len(),
            PackedBuffer::Int3(p) => p.bytes_len(),
            PackedBuffer::Int4(p) => p.bytes_len(),
            PackedBuffer::Int5(p) => p.bytes_len(),
            PackedBuffer::Int6(p) => p.bytes_len(),
            PackedBuffer::Int7(p) => p.bytes_len(),
            PackedBuffer::Int8(p) => p.data_len(),
            PackedBuffer::Nf4(p) => p.bytes_len(),
            PackedBuffer::Bf16(p) => p.bytes_len(),
            PackedBuffer::Fp16(p) => p.bytes_len(),
            PackedBuffer::Fp8(p) => p.bytes_len(),
            PackedBuffer::Gguf(p) => p.bytes_len(),
            PackedBuffer::KQuant(p) => p.bytes_len(),
            PackedBuffer::Iq(p) => p.bytes_len(),
            PackedBuffer::Tq(p) => p.bytes_len(),
            PackedBuffer::Custom(p) => p.bytes_len(),
            PackedBuffer::GgufRaw { bytes, .. } => bytes.len(),
        }
    }

    /// Actual bits/weight taking into account overhead on the scale table (where
    /// applicable). For `GgufRaw` uses `DType::bits_per_element`.
    pub fn effective_bits_per_weight(&self) -> f32 {
        match self {
            PackedBuffer::Float(_) => 32.0,
            PackedBuffer::Ternary(p) => p.effective_bits_per_weight(),
            PackedBuffer::Int1(p) => p.effective_bits_per_weight(),
            PackedBuffer::Int2(p) => p.effective_bits_per_weight(),
            PackedBuffer::Int3(p) => p.effective_bits_per_weight(),
            PackedBuffer::Int4(p) => p.effective_bits_per_weight(),
            PackedBuffer::Int5(p) => p.effective_bits_per_weight(),
            PackedBuffer::Int6(p) => p.effective_bits_per_weight(),
            PackedBuffer::Int7(p) => p.effective_bits_per_weight(),
            PackedBuffer::Int8(p) => p.effective_bits_per_weight(),
            PackedBuffer::Nf4(p) => p.effective_bits_per_weight(),
            PackedBuffer::Bf16(p) => p.effective_bits_per_weight(),
            PackedBuffer::Fp16(p) => p.effective_bits_per_weight(),
            PackedBuffer::Fp8(p) => p.effective_bits_per_weight(),
            PackedBuffer::Gguf(p) => p.dtype.bits_per_element(),
            PackedBuffer::KQuant(p) => p.effective_bits_per_weight(),
            // IqPacked does not store overhead separately from the block (like Gguf/KQuant
            // super-blocks) - bits_per_element() from crate::dtype::DType already
            // correctly reflects the block_bytes/block_elems contract (see.
            // pack::iq § 2 for the warning about IQ4_XS internally-
            // consistent layout).
            PackedBuffer::Iq(p) => p.dtype().bits_per_element(),
            // TqPacked - same principle as Iq above: block_bytes/block_elems
            // from crate::dtype::DType already completely describe the overhead block.
            PackedBuffer::Tq(p) => p.dtype().bits_per_element(),
            PackedBuffer::Custom(p) => p.effective_bits_per_weight(),
            PackedBuffer::GgufRaw { dtype, .. } => dtype.bits_per_element(),
        }
    }

    /// Fully expand the buffer in `Vec<f32>` - single entry point for
    /// any fallback path (verification, lack of specialized
    /// kernels for this backend). `GgufRaw` is decoded via
    /// [`gguf_raw_to_typed`] (with byte copy); panics if `dtype` doesn't
    /// supported or byte length does not match `numel`.
    pub fn to_f32(&self) -> Vec<f32> {
        match self {
            PackedBuffer::Float(v) => v.clone(),
            PackedBuffer::Ternary(p) => dequantize_ternary(p),
            PackedBuffer::Int1(p) => dequantize_int1(p),
            PackedBuffer::Int2(p) => dequantize_int2(p),
            PackedBuffer::Int3(p) => dequantize_int3(p),
            PackedBuffer::Int4(p) => dequantize_int4(p),
            PackedBuffer::Int5(p) => dequantize_int5(p),
            PackedBuffer::Int6(p) => dequantize_int6(p),
            PackedBuffer::Int7(p) => dequantize_int7(p),
            PackedBuffer::Int8(p) => dequantize_int8(p),
            PackedBuffer::Nf4(p) => dequantize_nf4(p),
            PackedBuffer::Bf16(p) => dequantize_bf16(p),
            PackedBuffer::Fp16(p) => dequantize_fp16(p),
            PackedBuffer::Fp8(p) => dequantize_fp8(p),
            PackedBuffer::Gguf(p) => p.to_f32(),
            PackedBuffer::KQuant(p) => p.to_f32(),
            PackedBuffer::Iq(p) => p.to_f32(),
            PackedBuffer::Tq(p) => p.to_f32(),
            PackedBuffer::Custom(p) => p.to_f32(),
            PackedBuffer::GgufRaw {
                bytes,
                dtype,
                numel,
            } => match gguf_raw_to_typed(bytes.clone(), *dtype, *numel) {
                Ok(typed) => typed.to_f32(),
                Err(e) => panic!("PackedBuffer::to_f32: GgufRaw: {e}"),
            },
        }
    }

    /// Dequantize one element by flat index `idx` without full
    /// unpacking the entire buffer - a single entry point for
    /// `dequantize_into`/`dequantize_range` (dequantize.rs) so that they don’t
    /// duplicated match for all options manually (and did not forget new ones
    /// schemes when expanding `PackedBuffer` - see the history of this match).
    /// On `Gguf`/`KQuant`/`Iq`/`Tq`/`GgufRaw` it is not effective: now it is
    /// decodes the entire buffer (for `GgufRaw` it also parses the bytes), and
    /// not only containing a block - acceptable, because this function is intended
    /// for intN/nf4/ternary point-lookup, not bulk paths. For ranges and
    /// for the entire buffer use `dequantize_range`/`dequantize_into`.
    /// `GgufRaw` panics for the same reasons as `to_f32`.
    #[inline]
    pub fn get_f32(&self, idx: usize) -> f32 {
        match self {
            PackedBuffer::Float(v) => v[idx],
            PackedBuffer::Ternary(p) => p.get_f32(idx),
            PackedBuffer::Int1(p) => p.get_f32(idx),
            PackedBuffer::Int2(p) => p.get_f32(idx),
            PackedBuffer::Int3(p) => p.get_f32(idx),
            PackedBuffer::Int4(p) => p.get_f32(idx),
            PackedBuffer::Int5(p) => p.get_f32(idx),
            PackedBuffer::Int6(p) => p.get_f32(idx),
            PackedBuffer::Int7(p) => p.get_f32(idx),
            PackedBuffer::Int8(p) => p.get_f32(idx),
            PackedBuffer::Nf4(p) => p.get_f32(idx),
            PackedBuffer::Bf16(p) => p.get_f32(idx),
            PackedBuffer::Fp16(p) => p.get_f32(idx),
            PackedBuffer::Fp8(p) => p.get_f32(idx),
            PackedBuffer::Gguf(p) => p.to_f32()[idx],
            PackedBuffer::KQuant(p) => p.to_f32()[idx],
            PackedBuffer::Iq(p) => p.to_f32()[idx],
            // TqPacked - like Gguf/KQuant/Iq above: super-block requires
            // decode the entire containing block, acceptable for point-lookup.
            PackedBuffer::Tq(p) => p.to_f32()[idx],
            PackedBuffer::Custom(p) => p.to_f32()[idx],
            PackedBuffer::GgufRaw {
                bytes,
                dtype,
                numel,
            } => match gguf_raw_to_typed(bytes.clone(), *dtype, *numel) {
                Ok(typed) => typed.to_f32()[idx],
                Err(e) => panic!("PackedBuffer::get_f32: GgufRaw: {e}"),
            },
        }
    }

    /// Circuit machine tag - matches `PackedScheme::tag()`, but is output
    /// from already packed data (useful for logging/debugging when the original
    /// `PackedScheme` is not saved nearby).
    pub fn scheme_tag(&self) -> &'static str {
        match self {
            PackedBuffer::Float(_) => "f32",
            PackedBuffer::Ternary(_) => "ternary",
            PackedBuffer::Int1(_) => "int1",
            PackedBuffer::Int2(p) => {
                if p.is_asymmetric() {
                    "int2_asym"
                } else {
                    "int2"
                }
            }
            PackedBuffer::Int3(p) => {
                if p.is_asymmetric() {
                    "int3_asym"
                } else {
                    "int3"
                }
            }
            PackedBuffer::Int4(p) => {
                if p.is_asymmetric() {
                    "int4_asym"
                } else {
                    "int4"
                }
            }
            PackedBuffer::Int5(p) => {
                if p.is_asymmetric() {
                    "int5_asym"
                } else {
                    "int5"
                }
            }
            PackedBuffer::Int6(p) => {
                if p.is_asymmetric() {
                    "int6_asym"
                } else {
                    "int6"
                }
            }
            PackedBuffer::Int7(p) => {
                if p.is_asymmetric() {
                    "int7_asym"
                } else {
                    "int7"
                }
            }
            PackedBuffer::Int8(p) => {
                if p.is_asymmetric() {
                    "int8_asym"
                } else {
                    "int8"
                }
            }
            PackedBuffer::Nf4(_) => "nf4",
            PackedBuffer::Bf16(_) => "bf16",
            PackedBuffer::Fp16(_) => "fp16",
            PackedBuffer::Fp8(p) => match p.variant {
                Fp8Variant::E4M3 => "fp8_e4m3",
                Fp8Variant::E5M2 => "fp8_e5m2",
            },
            PackedBuffer::Gguf(p) => p.dtype.as_str(),
            PackedBuffer::KQuant(p) => p.dtype.as_str(),
            PackedBuffer::Iq(p) => p.dtype().as_str(),
            PackedBuffer::Tq(p) => p.dtype().as_str(),
            PackedBuffer::Custom(p) => p.scheme.tag(),
            PackedBuffer::GgufRaw { dtype, .. } => dtype.as_str(),
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// § 2 TESTS
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use crate::scheme::ScaleGranularity;

    #[test]
    fn test_packed_buffer_numel_consistency() {
        let data = vec![0.5f32; 40];
        let t = quantize_ternary(&data, ScaleGranularity::PerTensor, 0);
        let buf = PackedBuffer::Ternary(t);
        assert_eq!(buf.numel(), 40);
    }

    #[test]
    fn test_packed_buffer_roundtrip_via_to_f32() {
        let data = vec![0.9, -0.9, 0.9, -0.9];
        let packed = quantize_int1(&data, ScaleGranularity::PerTensor, 0);
        let buf = PackedBuffer::Int1(packed);
        let restored = buf.to_f32();
        assert_eq!(restored.len(), 4);
    }

    #[test]
    fn test_float_passthrough_bits() {
        let buf = PackedBuffer::Float(vec![1.0, 2.0, 3.0]);
        assert_eq!(buf.effective_bits_per_weight(), 32.0);
        assert_eq!(buf.data_bytes(), 12);
    }

    /// Pass the data through typed packaging and return the same bytes
    /// as `GgufRaw` - decoding by both methods must coincide.
    fn raw_of(buf: &PackedBuffer, dtype: DType) -> PackedBuffer {
        let (bytes, numel) = match buf {
            PackedBuffer::Gguf(p) => (p.blocks.clone(), p.numel),
            PackedBuffer::KQuant(p) => (p.blocks.clone(), p.numel),
            _ => panic!("raw_of: test helper supports Gguf/KQuant only"),
        };
        PackedBuffer::GgufRaw {
            bytes,
            dtype,
            numel,
        }
    }

    #[test]
    fn test_gguf_raw_decodes_like_typed_simple_block() {
        let data: Vec<f32> = (0..64).map(|i| (i as f32 - 32.0) * 0.05).collect();
        let typed = PackedBuffer::Gguf(crate::pack::gguf::quantize_gguf(&data, DType::Q4_0));
        let raw = raw_of(&typed, DType::Q4_0);
        assert_eq!(raw.to_f32(), typed.to_f32());
        assert_eq!(raw.get_f32(5), typed.to_f32()[5]);
        assert_eq!(raw.numel(), 64);
    }

    #[test]
    fn test_gguf_raw_decodes_like_typed_kquant() {
        let data: Vec<f32> = (0..256).map(|i| (i as f32 - 128.0) * 0.01).collect();
        let typed = PackedBuffer::KQuant(crate::pack::kquant::quantize_kquant(&data, DType::Q4_K));
        let raw = raw_of(&typed, DType::Q4_K);
        assert_eq!(raw.to_f32(), typed.to_f32());
    }

    #[test]
    fn test_gguf_raw_all_zero_block_decodes_to_zeros() {
        // Q4_0: 32 weights = 18 bytes; zero bytes - zero scale f16 → zeros.
        let buf = PackedBuffer::GgufRaw {
            bytes: vec![0; 18],
            dtype: DType::Q4_0,
            numel: 32,
        };
        let out = buf.to_f32();
        assert_eq!(out.len(), 32);
        assert!(out.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_gguf_raw_to_typed_picks_backend_and_checks_length() {
        let data: Vec<f32> = (0..256).map(|i| (i as f32 - 128.0) * 0.01).collect();
        let bytes = crate::pack::kquant::quantize_kquant(&data, DType::Q4_K).blocks;
        let typed = gguf_raw_to_typed(bytes.clone(), DType::Q4_K, 256).unwrap();
        assert!(matches!(typed, PackedBuffer::KQuant(_)));

        // Incorrect length is a mistake, not quietly ruined weights.
        let err =
            gguf_raw_to_typed(bytes[..bytes.len() - 1].to_vec(), DType::Q4_K, 256).unwrap_err();
        assert!(err.contains("need"), "{err}");

        // Not GGUF-block dtype.
        assert!(gguf_raw_to_typed(vec![0; 4], DType::F32, 1).is_err());
    }

    #[test]
    #[should_panic(expected = "GgufRaw")]
    fn test_gguf_raw_with_wrong_length_panics_on_decode() {
        let buf = PackedBuffer::GgufRaw {
            bytes: vec![0; 17], // need 18
            dtype: DType::Q4_0,
            numel: 32,
        };
        let _ = buf.to_f32();
    }

    #[test]
    fn test_kquant_packed_buffer_roundtrips() {
        let data: Vec<f32> = (0..256).map(|i| (i as f32 - 128.0) * 0.01).collect();
        let packed = crate::pack::kquant::quantize_kquant(&data, DType::Q4_K);
        let buf = PackedBuffer::KQuant(packed);
        assert_eq!(buf.numel(), 256);
        let restored = buf.to_f32();
        assert_eq!(restored.len(), 256);
        assert_eq!(buf.scheme_tag(), "Q4_K");
    }

    #[test]
    fn test_all_intn_variants_roundtrip_through_packed_buffer() {
        let data: Vec<f32> = (0..64).map(|i| (i as f32 - 32.0) * 0.05).collect();

        let buffers = vec![
            PackedBuffer::Int2(quantize_int2(&data, ScaleGranularity::PerTensor, 0)),
            PackedBuffer::Int3(quantize_int3(&data, ScaleGranularity::PerTensor, 0)),
            PackedBuffer::Int4(quantize_int4(&data, ScaleGranularity::PerTensor, 0)),
            PackedBuffer::Int5(quantize_int5(&data, ScaleGranularity::PerTensor, 0)),
            PackedBuffer::Int6(quantize_int6(&data, ScaleGranularity::PerTensor, 0)),
            PackedBuffer::Int7(quantize_int7(&data, ScaleGranularity::PerTensor, 0)),
            PackedBuffer::Int8(quantize_int8(&data, ScaleGranularity::PerTensor, 0)),
            PackedBuffer::Nf4(quantize_nf4(&data, ScaleGranularity::PerTensor, 0)),
        ];

        for buf in &buffers {
            assert_eq!(buf.numel(), 64, "tag={}", buf.scheme_tag());
            let restored = buf.to_f32();
            assert_eq!(restored.len(), 64, "tag={}", buf.scheme_tag());
        }

        // The rounding error should monotonically decrease with increasing bit depth
        // (except for NF4, which has a different, uneven mesh).
        let err_of = |b: &PackedBuffer| -> f32 {
            b.to_f32()
                .iter()
                .zip(data.iter())
                .map(|(a, x)| (a - x).abs())
                .fold(0.0, f32::max)
        };
        let e2 = err_of(&buffers[0]);
        let e4 = err_of(&buffers[2]);
        let e8 = err_of(&buffers[6]);
        assert!(e4 <= e2 + 1e-3, "int4 err={e4} should be <= int2 err={e2}");
        assert!(e8 <= e4 + 1e-3, "int8 err={e8} should be <= int4 err={e4}");
    }

    #[test]
    fn test_iq_packed_buffer_roundtrips() {
        let data: Vec<f32> = (0..64).map(|i| (i as f32 - 32.0) * 0.05).collect();
        let packed = crate::pack::iq::quantize_iq(&data, DType::IQ4_NL);
        let buf = PackedBuffer::Iq(packed);
        assert_eq!(buf.numel(), 64);
        let restored = buf.to_f32();
        assert_eq!(restored.len(), 64);
        assert_eq!(buf.scheme_tag(), "IQ4_NL");
    }

    #[test]
    fn test_packed_buffer_bf16_fp16_fp8_roundtrip() {
        let data: Vec<f32> = (0..48).map(|i| (i as f32 - 24.0) * 0.25).collect();

        let bf16 = PackedBuffer::Bf16(quantize_bf16(&data));
        assert_eq!(bf16.numel(), 48);
        assert_eq!(bf16.data_bytes(), 96);
        assert_eq!(bf16.effective_bits_per_weight(), 16.0);
        assert_eq!(bf16.to_f32().len(), 48);
        assert_eq!(bf16.scheme_tag(), "bf16");

        let fp16 = PackedBuffer::Fp16(quantize_fp16(&data));
        assert_eq!(fp16.numel(), 48);
        assert_eq!(fp16.data_bytes(), 96);
        assert_eq!(fp16.to_f32().len(), 48);
        assert_eq!(fp16.scheme_tag(), "fp16");

        let fp8 = PackedBuffer::Fp8(quantize_fp8(&data, Fp8Variant::E4M3));
        assert_eq!(fp8.numel(), 48);
        assert_eq!(fp8.data_bytes(), 48);
        assert!(fp8.effective_bits_per_weight() >= 8.0);
        assert_eq!(fp8.to_f32().len(), 48);
        assert_eq!(fp8.get_f32(0), fp8.to_f32()[0]);
        assert_eq!(fp8.scheme_tag(), "fp8_e4m3");

        let fp8_e5m2 = PackedBuffer::Fp8(quantize_fp8(&data, Fp8Variant::E5M2));
        assert_eq!(fp8_e5m2.scheme_tag(), "fp8_e5m2");
    }

    #[test]
    fn test_asymmetric_scheme_tags_distinguish_from_symmetric() {
        let data: Vec<f32> = (0..32).map(|i| 5.0 + i as f32 * 0.1).collect();
        let sym = PackedBuffer::Int4(quantize_int4(&data, ScaleGranularity::PerTensor, 0));
        let asym = PackedBuffer::Int4(quantize_int4_asymmetric(
            &data,
            ScaleGranularity::PerTensor,
            0,
        ));
        assert_eq!(sym.scheme_tag(), "int4");
        assert_eq!(asym.scheme_tag(), "int4_asym");
        assert_ne!(sym.scheme_tag(), asym.scheme_tag());
    }

    // ── PackedBuffer::Tq + IQ1_S/IQ1_M via PackedBuffer::Iq ──────
    // Each accessor is covered explicitly for the Tq variant and for
    // IQ1_S/IQ1_M through Iq (numel/data_bytes/effective_bits_per_weight/
    // to_f32/get_f32/scheme_tag).

    #[test]
    fn test_packed_buffer_tq_all_accessors() {
        let data: Vec<f32> = (0..256).map(|i| (i as f32 - 128.0) * 0.01).collect();
        let buf = PackedBuffer::Tq(crate::pack::tq::quantize_tq(&data, DType::TQ1_0));

        assert_eq!(buf.numel(), 256);
        assert_eq!(buf.data_bytes(), 54); // TQ1_0_BLOCK_BYTES, 1 block
        assert!(buf.effective_bits_per_weight() > 0.0);
        assert_eq!(buf.to_f32().len(), 256);
        assert_eq!(buf.get_f32(0), buf.to_f32()[0]);
        assert_eq!(buf.scheme_tag(), "TQ1_0");
    }

    #[test]
    fn test_packed_buffer_tq2_0_all_accessors() {
        let data: Vec<f32> = (0..256).map(|i| (i as f32 - 128.0) * 0.01).collect();
        let buf = PackedBuffer::Tq(crate::pack::tq::quantize_tq(&data, DType::TQ2_0));

        assert_eq!(buf.numel(), 256);
        assert_eq!(buf.data_bytes(), 66); // TQ2_0_BLOCK_BYTES, 1 block
        assert_eq!(buf.to_f32().len(), 256);
        assert_eq!(buf.scheme_tag(), "TQ2_0");
    }

    #[test]
    fn test_packed_buffer_iq1_s_all_accessors() {
        let data: Vec<f32> = (0..256).map(|i| (i as f32 - 128.0) * 0.01).collect();
        let buf = PackedBuffer::Iq(quantize_iq(&data, DType::IQ1_S));

        assert_eq!(buf.numel(), 256);
        assert_eq!(buf.data_bytes(), 50); // IQ1_S block_bytes, 1 block
        assert!(buf.effective_bits_per_weight() > 0.0);
        assert_eq!(buf.to_f32().len(), 256);
        assert_eq!(buf.get_f32(0), buf.to_f32()[0]);
        assert_eq!(buf.scheme_tag(), "IQ1_S");
    }

    #[test]
    fn test_packed_buffer_iq1_m_all_accessors() {
        let data: Vec<f32> = (0..256).map(|i| (i as f32 - 128.0) * 0.01).collect();
        let buf = PackedBuffer::Iq(quantize_iq(&data, DType::IQ1_M));

        assert_eq!(buf.numel(), 256);
        assert_eq!(buf.data_bytes(), 56); // IQ1_M block_bytes, 1 block
        assert_eq!(buf.to_f32().len(), 256);
        assert_eq!(buf.scheme_tag(), "IQ1_M");
    }
}
