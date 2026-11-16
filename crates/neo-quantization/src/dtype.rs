// neo-quantization/src/dtype.rs
//
// `DType` — GGUF-compatible tensor element type.
//
// ## Why does this guy live here and not come from burn
//
// Moved from `neo-core` as part of the transition of the crate to burn. Direct
// There is NO equivalent in burn, and it cannot be replaced:
//
//   • `burn::tensor::DType` only knows F64/F32/Flex32/F16/BF16, integers, Bool
//     and one generalized `QFloat(QuantScheme)`;
//   • `burn::tensor::QuantScheme` (originally from `cubecl_common::quant::scheme`)
//     describes quantization parametrically - `QuantValue` (Q8S/Q4S/Q2S,
//     E4M3/E5M2/E2M1), scale levels (`per_tensor`/`per_block`), `QuantStore`. This
//     "symmetric/block N-bit" model, and it is IMPOSSIBLE to express
//     neither super-blocks nor K-quants (Q4_K: 144 bytes for 256 elements, two
//     scale level d/dmin + 6-bit sub-scale), nor codebook formats
//     IQ-series, no base-5 packaging TQ1_0.
//
// Therefore, the GGUF table (block_elems / block_bytes / gguf_type_id) remains
// own type of this crate - it is the on-disk format representation.
// Compatibility with burn is ensured by explicit conversions in §4:
// `to_burn` / `from_burn` for unquantized types and `to_burn_quant_scheme`
// for those quantized who have an honest parametric analogue.
//
// Numeric constants form an external contract: they must match GGUF files on disk.

use core::fmt;

// ═══════════════════════════════════════════════════════════════════════════
// § 1  DTYPE
// ═══════════════════════════════════════════════════════════════════════════

/// Tensor data type.
///
/// Lightweight `Copy`-enum without payload - cheap to compare and store.
#[allow(non_camel_case_types)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DType {
    // ── Real ──────────────────────────── ────────────────────────────
    /// 64-bit IEEE float.
    F64,
    /// 32-bit IEEE float.
    F32,
    /// 16-bit brain float.
    BF16,
    /// 16-bit IEEE float (half).
    F16,

    // ── Whole ─────────────────────────────── ────────────────────────────────
    /// Signed 64-bit integer.
    I64,
    /// Signed 32-bit integer.
    I32,
    /// Signed 16-bit integer.
    I16,
    /// Signed 8-bit integer.
    I8,
    /// Unsigned 64-bit integer.
    U64,
    /// Unsigned 32-bit integer.
    U32,
    /// Unsigned 16-bit integer.
    U16,
    /// Unsigned 8-bit integer.
    U8,

    // ── Boolean ─────────────────────────────── ───────────────────────────────
    /// Boolean (1 byte per element).
    Bool,

    // ── Quantized (GGUF-compatible) ─────────────────────────────────────
    /// 4-bit quantization, 32 elems/block.
    Q4_0,
    /// 4-bit quantization with min, 32 elems/block.
    Q4_1,
    /// 5-bit quantization, 32 elems/block.
    Q5_0,
    /// 5-bit quantization with min, 32 elems/block.
    Q5_1,
    /// 8-bit quantization, 32 elems/block.
    Q8_0,
    /// 8-bit quantization with explicit sum, 32 elems/block.
    ///
    /// Intermediate llama.cpp format: `d`+`s` (scale + row-sum) instead
    /// `d`+`m` (scale + min) for Q8_0. Mainly used in upstream
    /// as `vec_dot_type` for Q4_1/Q5_1 (operand for dot product, not
    /// format for storing weights on disk) - but can also be found as valid
    /// GGUF tensor type (`GGML_TYPE_Q8_1 = 9`), therefore included along with
    /// the rest. See the Q8_1 layout block below.
    Q8_1,

    // K-quants (256 elem/block)
    /// 2-bit K-quant.
    Q2_K,
    /// 3-bit K-quant.
    Q3_K,
    /// 4-bit K-quant.
    Q4_K,
    /// 5-bit K-quant.
    Q5_K,
    /// 6-bit K-quant.
    Q6_K,
    /// 8-bit K-quant.
    Q8_K,

    // IQ (importance-weighted quants)
    /// IQ 2-bit XXS.
    IQ2_XXS,
    /// IQ 2-bit XS.
    IQ2_XS,
    /// IQ 3-bit XXS.
    IQ3_XXS,
    /// IQ 3-bit S.
    IQ3_S,
    /// IQ 4-bit NL.
    IQ4_NL,
    /// IQ 4-bit XS.
    IQ4_XS,
    /// IQ 1-bit S (codebook, super-block scale `d` + grid `qs`/`qh`).
    ///
    /// Part of the IQ1/TQ family described below; all constants are checked
    /// against upstream ggml.
    IQ1_S,
    /// IQ 1-bit M (codebook, per-sub-block scales, WITHOUT separate
    /// super-block `d` - the scale is encoded inside `scales`/`iq1m_scale_t`,
    /// see `block_bytes` for layout details).
    IQ1_M,
    /// Ternary Q1 (BitNet-style, GGUF-compatible super-block layout;
    /// NOT the same as `neo_quantization::pack::ternary::TernaryPacked`
    /// - it uses its own 2-bit/weight format, not GGUF-compatible).
    TQ1_0,
    /// Ternary Q2 (same meaning as `TQ1_0`, but 2 bits/element, without
    /// base-5 packaging - easier to decode, a little more space).
    TQ2_0,
}

// ═══════════════════════════════════════════════════════════════════════════
// IQ1/TQ family: IQ1_S, IQ1_M, TQ1_0, TQ2_0
//
// These GGUF-compatible types complete the DType set. `pack::ternary` in this
// crate solves a similar task in its own, non-GGUF-compatible format.
//
// Numeric constants (gguf_type_id, block_bytes) are checked against upstream
// ggml (`ggml/include/ggml.h`, `ggml/src/ggml-common.h`,
// https://github.com/ggml-org/llama.cpp):
//
//   GGML_TYPE_IQ1_S = 19, GGML_TYPE_IQ1_M = 29,
//   GGML_TYPE_TQ1_0 = 34, GGML_TYPE_TQ2_0 = 35
//
//   block_iq1_s: `{ ggml_half d; uint8_t qs[QK_K/8]; uint16_t qh[QK_K/32]; }`
//                = 2 + 32 + 16 = 50 bytes (QK_K=256; the static assertion
//                in ggml-common.h is `sizeof(ggml_half) + QK_K/8 + QK_K/16`).
//
//   block_iq1_m: `{ uint8_t qs[QK_K/8]; uint8_t qh[QK_K/16]; uint8_t
//                scales[QK_K/32]; }` = 32 + 16 + 8 = 56 bytes. There is no
//                separate `ggml_half d` at struct level: the scale is encoded
//                inside `scales` through `iq1m_scale_t` (a f16/u16 union that
//                reuses the most significant bits of the scale bytes).
//                `block_bytes()`/`block_elems()` describe the physical layout
//                needed for `storage_bytes` and GGUF headers; decoding
//                `iq1m_scale_t` belongs to `pack::iq1`.
//
//   block_tq1_0: `{ uint8_t qs[QK_K/64]; uint8_t qs2[(QK_K-4*QK_K/64)/5];
//                ggml_half d; }` = 4 + 48 + 2 = 54 bytes (~1.6875 bits/weight;
//                the base-5 packing of ternary values is intentional).
//                The layout does not match `neo_quantization::pack::ternary`.
//
//   block_tq2_0: `{ uint8_t qs[QK_K/4]; ggml_half d; }` = 64 + 2 = 66 bytes
//                (2 bits/element directly, without base-5 packing).
//
// Type metadata (id, block size, elements) is sufficient for `DType`, GGUF
// headers and `storage_bytes()`. Packing and unpacking live in
// `pack::iq1` (IQ1_S/IQ1_M) and `pack::tq` (TQ1_0/TQ2_0).
// ═══════════════════════════════════════════════════════════════════════════

// ═══════════════════════════════════════════════════════════════════════════
// Q8_1 layout
// ═══════════════════════════════════════════════════════════════════════════
//
// Q8_1 mirrors the GGUF `block_q8_1` layout:
//
//   struct { union { struct { ggml_half d; ggml_half s; }; ggml_half2 ds; };
//            int8_t qs[QK8_1]; }
//   QK8_1 = 32, so the block is 2 + 2 + 32 = 36 bytes.
//
// `d` is the scale and `s` is the row sum used by quantized dot products.
// Both are f16 (`ggml_half`), not f32. The numeric id is
// `GGML_TYPE_Q8_1 = 9`.
// ═══════════════════════════════════════════════════════════════════════════

impl DType {
    // ── Categories ────────────────────────────── ──────────────────────────────

    /// Whether the type is a float.
    #[inline]
    pub const fn is_float(self) -> bool {
        matches!(self, DType::F64 | DType::F32 | DType::BF16 | DType::F16)
    }

    /// Whether the type is a signed integer.
    #[inline]
    pub const fn is_signed_int(self) -> bool {
        matches!(self, DType::I64 | DType::I32 | DType::I16 | DType::I8)
    }

    /// Whether the type is an unsigned integer.
    #[inline]
    pub const fn is_unsigned_int(self) -> bool {
        matches!(self, DType::U64 | DType::U32 | DType::U16 | DType::U8)
    }

    /// Whether the type is an integer (signed or unsigned).
    #[inline]
    pub const fn is_int(self) -> bool {
        self.is_signed_int() || self.is_unsigned_int()
    }

    /// Whether the type is quantized.
    #[inline]
    pub const fn is_quantized(self) -> bool {
        !self.is_float() && !self.is_int() && !matches!(self, DType::Bool)
    }

    /// Whether the type is numeric (float or int, but not bool, not quantized).
    #[inline]
    pub const fn is_numeric(self) -> bool {
        self.is_float() || self.is_int()
    }

    /// ALL variants of `DType` in the order of announcement - the only list,
    /// from which the “documentation versus code” tests are based. Completeness
    /// holds test `all_lists_every_variant_exactly_once`: `ordinal`
    /// written without `_ =>` and stops compiling as soon as enum
    /// an option appears that is not in the match.
    pub const ALL: [DType; 35] = [
        DType::F64,
        DType::F32,
        DType::BF16,
        DType::F16,
        DType::I64,
        DType::I32,
        DType::I16,
        DType::I8,
        DType::U64,
        DType::U32,
        DType::U16,
        DType::U8,
        DType::Bool,
        DType::Q4_0,
        DType::Q4_1,
        DType::Q5_0,
        DType::Q5_1,
        DType::Q8_0,
        DType::Q8_1,
        DType::Q2_K,
        DType::Q3_K,
        DType::Q4_K,
        DType::Q5_K,
        DType::Q6_K,
        DType::Q8_K,
        DType::IQ2_XXS,
        DType::IQ2_XS,
        DType::IQ3_XXS,
        DType::IQ3_S,
        DType::IQ4_NL,
        DType::IQ4_XS,
        DType::IQ1_S,
        DType::IQ1_M,
        DType::TQ1_0,
        DType::TQ2_0,
    ];

    /// The serial number of the variant in [`DType::ALL`]. Without `_ =>`: new
    /// the enum option will not compile until it is added here.
    #[doc(hidden)]
    pub const fn ordinal(self) -> usize {
        match self {
            DType::F64 => 0,
            DType::F32 => 1,
            DType::BF16 => 2,
            DType::F16 => 3,
            DType::I64 => 4,
            DType::I32 => 5,
            DType::I16 => 6,
            DType::I8 => 7,
            DType::U64 => 8,
            DType::U32 => 9,
            DType::U16 => 10,
            DType::U8 => 11,
            DType::Bool => 12,
            DType::Q4_0 => 13,
            DType::Q4_1 => 14,
            DType::Q5_0 => 15,
            DType::Q5_1 => 16,
            DType::Q8_0 => 17,
            DType::Q8_1 => 18,
            DType::Q2_K => 19,
            DType::Q3_K => 20,
            DType::Q4_K => 21,
            DType::Q5_K => 22,
            DType::Q6_K => 23,
            DType::Q8_K => 24,
            DType::IQ2_XXS => 25,
            DType::IQ2_XS => 26,
            DType::IQ3_XXS => 27,
            DType::IQ3_S => 28,
            DType::IQ4_NL => 29,
            DType::IQ4_XS => 30,
            DType::IQ1_S => 31,
            DType::IQ1_M => 32,
            DType::TQ1_0 => 33,
            DType::TQ2_0 => 34,
        }
    }

    /// The effective number of bits per element (for quantized ones - fractional!).
    pub fn bits_per_element(self) -> f32 {
        if self.is_quantized() {
            let bytes = self.block_bytes() as f32;
            let elems = self.block_elems() as f32;
            bytes * 8.0 / elems
        } else {
            self.element_size_bytes().unwrap_or(1) as f32 * 8.0
        }
    }

    // ── Dimensions ─────────────────────────────── ───────────────────────────────

    /// The size of one **element** in bytes for non-quantized types.
    ///
    /// For quantized it returns `None` - use `block_bytes` / `block_elems`.
    pub const fn element_size_bytes(self) -> Option<usize> {
        match self {
            DType::F64 | DType::I64 | DType::U64 => Some(8),
            DType::F32 | DType::I32 | DType::U32 => Some(4),
            DType::BF16 | DType::F16 | DType::I16 | DType::U16 => Some(2),
            DType::I8 | DType::U8 | DType::Bool => Some(1),
            _ => None,
        }
    }

    /// The number of elements in one quantum block.
    ///
    /// For unquantized, it returns `1`.
    pub const fn block_elems(self) -> usize {
        match self {
            DType::Q4_0 | DType::Q4_1 | DType::Q5_0 | DType::Q5_1 | DType::Q8_0 | DType::Q8_1 => 32,
            DType::Q2_K | DType::Q3_K | DType::Q4_K | DType::Q5_K | DType::Q6_K | DType::Q8_K => {
                256
            }
            DType::IQ2_XXS | DType::IQ2_XS | DType::IQ3_XXS | DType::IQ3_S => 256,
            // IQ4_XS is a super-block format (QK_K=256, 8 sub-blocks of 32
            // with a common super-scale `d`), unlike the simple 32-element
            // IQ4_NL block. See `block_iq4_xs` in ggml-common.h.
            DType::IQ4_XS => 256,
            DType::IQ4_NL => 32,
            // IQ1_S/IQ1_M use a codebook super-block with QK_K=256.
            DType::IQ1_S | DType::IQ1_M => 256,
            // TQ1_0/TQ2_0 use the same QK_K=256 super-block as K-quants.
            DType::TQ1_0 | DType::TQ2_0 => 256,
            _ => 1,
        }
    }

    /// The size of one block in bytes.
    ///
    /// For non-quantized ones it coincides with `element_size_bytes`.
    pub const fn block_bytes(self) -> usize {
        match self {
            // 32 elem/block
            DType::Q4_0 => 2 + 16,         // f16 scale + 16 bytes (4bit×32)
            DType::Q4_1 => 2 + 2 + 16,     // f16 scale + f16 min + qs
            DType::Q5_0 => 2 + 4 + 16,     // f16 + qh(4) + qs(16)
            DType::Q5_1 => 2 + 2 + 4 + 16, // f16×2 + qh(4) + qs(16)
            DType::Q8_0 => 2 + 32,         // f16 scale + 32 i8
            // Q8_1: d(f16) + s(f16) + qs[32] = 2 + 2 + 32 = 36 bytes.
            // Both fields are `ggml_half`; the static assertion in
            // ggml-common.h is
            //   sizeof(block_q8_1) == 2*sizeof(ggml_half) + QK8_1.
            // `test_block_sizes_match_ggml_upstream` verifies the layout.
            DType::Q8_1 => 2 + 2 + 32,

            // 256 elem/block (K-quants)
            // block_q2_K = scales[16] + qs[QK_K/4=64] + d(f16,2) +
            // dmin(f16,2) = 16+64+2+2 = 84 (ggml-common.h:
            // sizeof(block_q2_K) == 2*sizeof(ggml_half) + QK_K/16 + QK_K/4).
            DType::Q2_K => 84,
            DType::Q3_K => 110,
            DType::Q4_K => 144,
            DType::Q5_K => 176,
            DType::Q6_K => 210,
            DType::Q8_K => 292,

            // IQ-quants
            DType::IQ2_XXS => 66,
            // IQ2_XS: d(f16,2) + qs[QK_K/8=32]*u16(2) + scales[QK_K/32=8]
            // = 2 + 64 + 8 = 74. `block_iq2_xs` has an additional `scales`
            // array that IQ2_XXS does not.
            DType::IQ2_XS => 2 + 64 + 8,
            DType::IQ3_XXS => 98,
            DType::IQ3_S => 110,
            DType::IQ4_NL => 2 + 16,
            // `block_iq4_xs` is a super-block (QK_K=256): d(f16,2) +
            // scales_h(u16,2) + scales_l[QK_K/64=4] + qs[QK_K/2=128]
            // = 2+2+4+128 = 136 bytes. Eight 32-element sub-blocks with a
            // 6-bit signed scale code (4 low bits in scales_l, 2 high bits in
            // scales_h) and one f16 super-scale `d`.
            DType::IQ4_XS => 2 + 2 + 4 + 128,

            // IQ1_S: ggml_half d(2) + qs[QK_K/8=32] + qh[QK_K/32=8]*u16(2)
            // = 2 + 32 + 16 = 50. See the IQ1/TQ family block above.
            DType::IQ1_S => 2 + 32 + 16,
            // IQ1_M: NO separate `d` at struct level (scale lives
            // inside scales via iq1m_scale_t) — qs[32] + qh[16] +
            // scales[QK_K/32=8] = 56. See block comment above.
            DType::IQ1_M => 32 + 16 + 8,
            // TQ1_0: qs[QK_K/64=4] + qs2[(QK_K-4*QK_K/64)/5=48] + d(2) = 54.
            DType::TQ1_0 => 4 + 48 + 2,
            // TQ2_0: qs[QK_K/4=64] + d(2) = 66.
            DType::TQ2_0 => 64 + 2,

            // Unquantized
            DType::F64 | DType::I64 | DType::U64 => 8,
            DType::F32 | DType::I32 | DType::U32 => 4,
            DType::BF16 | DType::F16 | DType::I16 | DType::U16 => 2,
            DType::I8 | DType::U8 | DType::Bool => 1,
        }
    }

    /// Number of bytes to store `n_elems` elements.
    ///
    /// Correctly handles odd `n_elems` for quantized types.
    pub fn storage_bytes(self, n_elems: usize) -> usize {
        let block_elems = self.block_elems();
        let n_blocks = n_elems.div_ceil(block_elems);
        n_blocks * self.block_bytes()
    }

    // ── GGUF ─────────────────────────────────────────────────────────────────

    /// GGUF numeric type id (compatible with llama.cpp).
    ///
    /// The ids match `ggml/include/ggml.h` (upstream `ggml-org/llama.cpp`).
    /// Every `DType` with a GGUF equivalent round-trips through
    /// [`from_gguf_type_id`](Self::from_gguf_type_id). `IQ2_S` (upstream id 22)
    /// has no variant in this enum.
    pub const fn gguf_type_id(self) -> u32 {
        match self {
            DType::F32 => 0,
            DType::F16 => 1,
            DType::Q4_0 => 2,
            DType::Q4_1 => 3,
            DType::Q5_0 => 6,
            DType::Q5_1 => 7,
            DType::Q8_0 => 8,
            // GGML_TYPE_Q8_1.
            DType::Q8_1 => 9,
            DType::Q8_K => 15,
            DType::Q2_K => 10,
            DType::Q3_K => 11,
            DType::Q4_K => 12,
            DType::Q5_K => 13,
            DType::Q6_K => 14,
            DType::IQ2_XXS => 16,
            DType::IQ2_XS => 17,
            DType::IQ3_XXS => 18,
            // GGML_TYPE_IQ1_S.
            DType::IQ1_S => 19,
            DType::IQ4_NL => 20,
            // GGML_TYPE_IQ3_S.
            DType::IQ3_S => 21,
            // GGML_TYPE_IQ4_XS.
            DType::IQ4_XS => 23,
            DType::I8 => 24,
            DType::I16 => 25,
            DType::I32 => 26,
            DType::I64 => 27,
            DType::F64 => 28,
            DType::IQ1_M => 29,
            // GGML_TYPE_BF16.
            DType::BF16 => 30,
            DType::TQ1_0 => 34,
            DType::TQ2_0 => 35,
            _ => u32::MAX,
        }
    }

    /// Recover `DType` from a GGUF type id.
    ///
    /// Inverse of [`gguf_type_id`](Self::gguf_type_id) for every variant that
    /// has a GGUF equivalent.
    pub const fn from_gguf_type_id(id: u32) -> Option<Self> {
        match id {
            0 => Some(DType::F32),
            1 => Some(DType::F16),
            2 => Some(DType::Q4_0),
            3 => Some(DType::Q4_1),
            6 => Some(DType::Q5_0),
            7 => Some(DType::Q5_1),
            8 => Some(DType::Q8_0),
            // GGML_TYPE_Q8_1.
            9 => Some(DType::Q8_1),
            10 => Some(DType::Q2_K),
            11 => Some(DType::Q3_K),
            12 => Some(DType::Q4_K),
            13 => Some(DType::Q5_K),
            14 => Some(DType::Q6_K),
            15 => Some(DType::Q8_K),
            16 => Some(DType::IQ2_XXS),
            17 => Some(DType::IQ2_XS),
            18 => Some(DType::IQ3_XXS),
            // GGML_TYPE_IQ1_S.
            19 => Some(DType::IQ1_S),
            20 => Some(DType::IQ4_NL),
            // GGML_TYPE_IQ3_S.
            21 => Some(DType::IQ3_S),
            // GGML_TYPE_IQ4_XS.
            23 => Some(DType::IQ4_XS),
            24 => Some(DType::I8),
            25 => Some(DType::I16),
            26 => Some(DType::I32),
            27 => Some(DType::I64),
            28 => Some(DType::F64),
            29 => Some(DType::IQ1_M),
            // GGML_TYPE_BF16.
            30 => Some(DType::BF16),
            34 => Some(DType::TQ1_0),
            35 => Some(DType::TQ2_0),
            _ => None,
        }
    }

    // ── Safetensors dtype string ─────────────────────────────────────────────

    /// Type string in safetensors/HuggingFace format.
    pub const fn as_str(self) -> &'static str {
        match self {
            DType::F64 => "F64",
            DType::F32 => "F32",
            DType::BF16 => "BF16",
            DType::F16 => "F16",
            DType::I64 => "I64",
            DType::I32 => "I32",
            DType::I16 => "I16",
            DType::I8 => "I8",
            DType::U64 => "U64",
            DType::U32 => "U32",
            DType::U16 => "U16",
            DType::U8 => "U8",
            DType::Bool => "BOOL",
            DType::Q4_0 => "Q4_0",
            DType::Q4_1 => "Q4_1",
            DType::Q5_0 => "Q5_0",
            DType::Q5_1 => "Q5_1",
            DType::Q8_0 => "Q8_0",
            DType::Q8_1 => "Q8_1",
            DType::Q2_K => "Q2_K",
            DType::Q3_K => "Q3_K",
            DType::Q4_K => "Q4_K",
            DType::Q5_K => "Q5_K",
            DType::Q6_K => "Q6_K",
            DType::Q8_K => "Q8_K",
            DType::IQ2_XXS => "IQ2_XXS",
            DType::IQ2_XS => "IQ2_XS",
            DType::IQ3_XXS => "IQ3_XXS",
            DType::IQ3_S => "IQ3_S",
            DType::IQ4_NL => "IQ4_NL",
            DType::IQ4_XS => "IQ4_XS",
            DType::IQ1_S => "IQ1_S",
            DType::IQ1_M => "IQ1_M",
            DType::TQ1_0 => "TQ1_0",
            DType::TQ2_0 => "TQ2_0",
        }
    }

    /// Parse from the line safetensors/GGUF.
    ///
    /// Returns `Option`, not `Result`, so this is not an implementation
    /// [`std::str::FromStr`]; the name has been retained for API compatibility.
    #[allow(clippy::should_implement_trait)]
    pub fn from_str(s: &str) -> Option<Self> {
        match s {
            "F64" | "f64" => Some(DType::F64),
            "F32" | "f32" => Some(DType::F32),
            "BF16" | "bf16" => Some(DType::BF16),
            "F16" | "f16" => Some(DType::F16),
            "I64" | "i64" => Some(DType::I64),
            "I32" | "i32" => Some(DType::I32),
            "I16" | "i16" => Some(DType::I16),
            "I8" | "i8" => Some(DType::I8),
            "U64" | "u64" => Some(DType::U64),
            "U32" | "u32" => Some(DType::U32),
            "U16" | "u16" => Some(DType::U16),
            "U8" | "u8" => Some(DType::U8),
            "BOOL" | "bool" => Some(DType::Bool),
            "Q4_0" => Some(DType::Q4_0),
            "Q4_1" => Some(DType::Q4_1),
            "Q5_0" => Some(DType::Q5_0),
            "Q5_1" => Some(DType::Q5_1),
            "Q8_0" => Some(DType::Q8_0),
            "Q8_1" => Some(DType::Q8_1),
            "Q2_K" => Some(DType::Q2_K),
            "Q3_K" => Some(DType::Q3_K),
            "Q4_K" => Some(DType::Q4_K),
            "Q5_K" => Some(DType::Q5_K),
            "Q6_K" => Some(DType::Q6_K),
            "Q8_K" => Some(DType::Q8_K),
            "IQ2_XXS" => Some(DType::IQ2_XXS),
            "IQ2_XS" => Some(DType::IQ2_XS),
            "IQ3_XXS" => Some(DType::IQ3_XXS),
            "IQ3_S" => Some(DType::IQ3_S),
            "IQ4_NL" => Some(DType::IQ4_NL),
            "IQ4_XS" => Some(DType::IQ4_XS),
            "IQ1_S" => Some(DType::IQ1_S),
            "IQ1_M" => Some(DType::IQ1_M),
            "TQ1_0" => Some(DType::TQ1_0),
            "TQ2_0" => Some(DType::TQ2_0),
            _ => None,
        }
    }
}

impl fmt::Display for DType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(self.as_str())
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// § 2 TESTS
// ═══════════════════════════════════════════════════════════════════════════

// ═══════════════════════════════════════════════════════════════════════════
// § 4 INTEROP WITH BURN
// ═══════════════════════════════════════════════════════════════════════════
//
// Bridge between the GGUF table above and the burn types. Intentionally partial and
// completely explicit: a conversion that would lose format returns `None`,
// rather than the “nearest similar” type. Silent replacement of Q4_K with a generalized one
// A 4-bit block would spoil the weights, and this would only be discovered by
// quality of the model.

use burn::tensor::quantization::{QuantScheme, QuantValue, ScaleDtype};
use burn::tensor::BoolStore;
use burn::tensor::DType as BurnDType;

impl DType {
    /// The corresponding `burn::tensor::DType` is only for NON-quantized types.
    ///
    /// For quantized it returns `None`: `burn::tensor::DType::QFloat`
    /// requires `QuantScheme`, which does not describe GGUF formats entirely
    /// (see [`DType::to_burn_quant_scheme`]).
    ///
    /// `Bool` is displayed in `BurnDType::Bool(BoolStore::Native)`: in burn
    /// the boolean type is parameterized by the storage method (Native/U8/U32), for GGUF
    /// there is no such difference, and `Native` is the only option for which
    /// `block_bytes() == 1` is the same as `BurnDType::size()`.
    pub fn to_burn(self) -> Option<BurnDType> {
        Some(match self {
            DType::F64 => BurnDType::F64,
            DType::F32 => BurnDType::F32,
            DType::BF16 => BurnDType::BF16,
            DType::F16 => BurnDType::F16,
            DType::I64 => BurnDType::I64,
            DType::I32 => BurnDType::I32,
            DType::I16 => BurnDType::I16,
            DType::I8 => BurnDType::I8,
            DType::U64 => BurnDType::U64,
            DType::U32 => BurnDType::U32,
            DType::U16 => BurnDType::U16,
            DType::U8 => BurnDType::U8,
            DType::Bool => BurnDType::Bool(BoolStore::Native),
            _ => return None,
        })
    }

    /// Reverse conversion from `burn::tensor::DType`.
    ///
    /// `None` for `QFloat` (there is no inverse mapping of the generalized scheme to
    /// specific GGUF-type) and for `Flex32` - which has no analogue in
    /// GGUF-table at all.
    pub fn from_burn(dtype: BurnDType) -> Option<Self> {
        Some(match dtype {
            BurnDType::F64 => DType::F64,
            BurnDType::F32 => DType::F32,
            BurnDType::BF16 => DType::BF16,
            BurnDType::F16 => DType::F16,
            BurnDType::I64 => DType::I64,
            BurnDType::I32 => DType::I32,
            BurnDType::I16 => DType::I16,
            BurnDType::I8 => DType::I8,
            BurnDType::U64 => DType::U64,
            BurnDType::U32 => DType::U32,
            BurnDType::U16 => DType::U16,
            BurnDType::U8 => DType::U8,
            BurnDType::Bool(_) => DType::Bool,
            BurnDType::Flex32 | BurnDType::QFloat(_) => return None,
        })
    }

    /// Parametric burn scheme for those GGUF-types that are expressed in its
    /// terms WITHOUT loss of format.
    ///
    /// Successful for Q8_0 and Q4_0 only: symmetric N-bit block
    /// quantization with one scale per block of 32 elements is exactly what
    /// which describes `QuantScheme`.
    ///
    /// For every other type - `None`, because the format genuinely does not fit:
    ///
    ///   • Q4_1/Q5_1/Q8_1 - asymmetric (scale + min/sum), while
    ///     `QuantMode` is symmetric only;
    ///   • K-quants - two scale levels (super-block d/dmin plus
    ///     6-bit sub-scale for every 16/32 elements);
    ///   • IQ-series - codebook: bits index the lattice table instead of
    ///     encoding the value directly;
    ///   • TQ1_0 - base-5 packing of five ternary weights into one byte.
    ///
    /// None of this is expressed by `QuantValue`/scale levels, and a
    /// “roughly similar” scheme would be worse than returning `None`.
    pub fn to_burn_quant_scheme(self) -> Option<QuantScheme> {
        let value = match self {
            DType::Q8_0 => QuantValue::Q8S,
            DType::Q4_0 => QuantValue::Q4S,
            _ => return None,
        };
        // burn/cubecl 0.22: `QuantLevel`/`QuantParam` replaced by levels
        // scale in the scheme itself; `per_block` = one f32 scale per block.
        Some(
            QuantScheme::default()
                .with_value(value)
                .per_block([self.block_elems() as u8], ScaleDtype::F32),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── Interop with burn (§ 4) ──────────────────────── ────────────────────────

    #[test]
    fn test_burn_roundtrip_unquantized() {
        // Every non-quantized type must survive DType → burn → DType.
        for dt in [
            DType::F64,
            DType::F32,
            DType::BF16,
            DType::F16,
            DType::I64,
            DType::I32,
            DType::I16,
            DType::I8,
            DType::U64,
            DType::U32,
            DType::U16,
            DType::U8,
            DType::Bool,
        ] {
            let burn = dt.to_burn().expect("an unquantized type must be converted");
            assert_eq!(
                DType::from_burn(burn),
                Some(dt),
                "round-trip is broken for {dt}"
            );
        }
    }

    #[test]
    fn test_burn_size_agrees_for_unquantized() {
        // A discrepancy in size would mean that the types “match” only in
        // name - then the conversion would silently spoil the buffer markup.
        for dt in [
            DType::F64,
            DType::F32,
            DType::F16,
            DType::BF16,
            DType::I8,
            DType::U32,
        ] {
            let burn = dt.to_burn().unwrap();
            assert_eq!(dt.block_bytes(), burn.size(), "size mismatch for {dt}");
        }
    }

    #[test]
    fn test_burn_conversion_refuses_quantized() {
        // Quantized GGUF types do NOT have a representation in burn::tensor::DType —
        // The conversion must refuse, and not select a similar type.
        for dt in [
            DType::Q4_0,
            DType::Q4_K,
            DType::IQ2_XXS,
            DType::IQ1_S,
            DType::TQ1_0,
        ] {
            assert_eq!(
                dt.to_burn(),
                None,
                "{dt} should not be converted to burn DType"
            );
        }
    }

    #[test]
    fn test_burn_quant_scheme_only_for_symmetric_block() {
        use burn::tensor::quantization::QuantValue;

        // Q8_0/Q4_0 are symmetrical 32-element blocks, this is what burn expresses.
        let q8 = DType::Q8_0
            .to_burn_quant_scheme()
            .expect("Q8_0 is expressible");
        assert_eq!(q8.value, QuantValue::Q8S);
        assert_eq!(
            q8.block_size().map(|b| b.as_slice().to_vec()),
            Some(vec![32])
        );

        let q4 = DType::Q4_0
            .to_burn_quant_scheme()
            .expect("Q4_0 is expressible");
        assert_eq!(q4.value, QuantValue::Q4S);

        // The rest are not: super-block, codebook, asymmetry, base-5.
        for dt in [
            DType::Q4_1,    // asymmetrical (scale + min)
            DType::Q4_K,    // super-block, two scale levels
            DType::IQ2_XXS, // codebook
            DType::IQ1_M,   // codebook, scale inside scales
            DType::TQ1_0,   // base-5 packaging
        ] {
            assert_eq!(
                dt.to_burn_quant_scheme(),
                None,
                "{dt} is not expressed by the burn scheme without loss of format"
            );
        }
    }

    #[test]
    fn test_float_sizes() {
        assert_eq!(DType::F32.element_size_bytes(), Some(4));
        assert_eq!(DType::F16.element_size_bytes(), Some(2));
        assert_eq!(DType::F64.element_size_bytes(), Some(8));
    }

    #[test]
    fn test_quantized_has_no_elem_size() {
        assert_eq!(DType::Q4_0.element_size_bytes(), None);
        assert!(DType::Q4_0.is_quantized());
    }

    #[test]
    fn test_q4_0_storage() {
        assert_eq!(DType::Q4_0.storage_bytes(32), 18); // 1 block
        assert_eq!(DType::Q4_0.storage_bytes(64), 36); // 2 blocks
        assert_eq!(DType::Q4_0.storage_bytes(33), 36); // ceil(33/32)=2
    }

    #[test]
    fn test_iq4_xs_is_super_block_not_simple_block() {
        // IQ4_XS - super-block format (QK_K=256, like K-quants), NOT
        // a simple 32-elem block like IQ4_NL - see block_iq4_xs in
        // ggml-common.h.
        assert_eq!(DType::IQ4_XS.block_elems(), 256);
        assert_eq!(DType::IQ4_XS.block_bytes(), 136);
        assert_eq!(DType::IQ4_NL.block_elems(), 32);
        assert_eq!(DType::IQ4_NL.block_bytes(), 18);
        assert_ne!(DType::IQ4_XS.block_elems(), DType::IQ4_NL.block_elems());
    }

    #[test]
    fn test_iq4_xs_storage_bytes() {
        assert_eq!(DType::IQ4_XS.storage_bytes(256), 136); // 1 super-block
        assert_eq!(DType::IQ4_XS.storage_bytes(512), 272); // 2 super-block
        assert_eq!(DType::IQ4_XS.storage_bytes(257), 272); // ceil(257/256)=2
    }

    #[test]
    fn test_gguf_roundtrip() {
        for dtype in [
            DType::F32,
            DType::Q4_0,
            DType::Q6_K,
            DType::IQ4_NL,
            DType::Q8_1,
        ] {
            let id = dtype.gguf_type_id();
            assert_ne!(id, u32::MAX);
            assert_eq!(DType::from_gguf_type_id(id), Some(dtype));
        }
    }

    /// `Q8_1` uses id 9, between `Q8_0` (8) and `Q2_K` (10).
    #[test]
    fn test_q8_1_id_between_neighbours() {
        assert_eq!(DType::Q8_0.gguf_type_id(), 8);
        assert_eq!(DType::Q8_1.gguf_type_id(), 9);
        assert_eq!(DType::Q2_K.gguf_type_id(), 10);
        assert_eq!(DType::from_gguf_type_id(9), Some(DType::Q8_1));
    }

    #[test]
    fn test_q8_1_block_layout() {
        // Values according to `block_q8_1` from ggml-common.h:
        // d(f16) + s(f16) + 32×i8 = 36 bytes.
        assert_eq!(DType::Q8_1.block_elems(), 32);
        assert_eq!(DType::Q8_1.block_bytes(), 36);
        assert_eq!(DType::Q8_1.storage_bytes(32), 36);
        assert_eq!(DType::Q8_1.storage_bytes(64), 72);
        // 36 bytes for 32 weights = 9 bits/weight: 8 for data + 1 for d and s.
        assert!((DType::Q8_1.bits_per_element() - 9.0).abs() < 1e-6);
    }

    #[test]
    fn test_str_roundtrip() {
        for dtype in [
            DType::F32,
            DType::BF16,
            DType::Q4_K,
            DType::IQ3_S,
            DType::Q8_1,
        ] {
            let s = dtype.as_str();
            assert_eq!(DType::from_str(s), Some(dtype));
        }
    }

    #[test]
    fn test_categories() {
        assert!(DType::F32.is_float());
        assert!(DType::I32.is_int());
        assert!(DType::Q4_K.is_quantized());
        assert!(!DType::F32.is_quantized());
        assert!(DType::U8.is_unsigned_int());
        assert!(!DType::U8.is_signed_int());
    }

    // ── Tests IQ1_S/IQ1_M/TQ1_0/TQ2_0 ──────

    #[test]
    fn test_iq1_tq_family_are_quantized_super_blocks() {
        for dtype in [DType::IQ1_S, DType::IQ1_M, DType::TQ1_0, DType::TQ2_0] {
            assert!(dtype.is_quantized(), "{dtype:?} must be quantized");
            assert!(!dtype.is_float());
            assert!(!dtype.is_int());
            assert_eq!(
                dtype.block_elems(),
                256,
                "{dtype:?} is a QK_K=256 super-block"
            );
        }
    }

    #[test]
    fn test_iq1_s_block_bytes_matches_upstream() {
        // block_iq1_s = ggml_half d(2) + qs[QK_K/8=32] + qh[QK_K/32=8]*u16(2)
        // = 2 + 32 + 16 = 50, static_assert - confirmed in ggml-common.h.
        assert_eq!(DType::IQ1_S.block_bytes(), 50);
        assert_eq!(DType::IQ1_S.storage_bytes(256), 50);
        assert_eq!(DType::IQ1_S.storage_bytes(257), 100); // ceil(257/256)=2
    }

    #[test]
    fn test_iq1_m_block_bytes_matches_upstream() {
        // block_iq1_m = qs[32] + qh[16] + scales[8] = 56, WITHOUT separate
        // fields `d` (scale inside scales via iq1m_scale_t) - see.
        // doc comment on enum variant IQ1_M.
        assert_eq!(DType::IQ1_M.block_bytes(), 56);
    }

    #[test]
    fn test_tq1_0_tq2_0_block_bytes_match_upstream() {
        // block_tq1_0 = qs[4] + qs2[48] + d(2) = 54 (~1.6875 bits/weight).
        assert_eq!(DType::TQ1_0.block_bytes(), 54);
        // block_tq2_0 = qs[64] + d(2) = 66 (2.0625 bits/weight).
        assert_eq!(DType::TQ2_0.block_bytes(), 66);
        // TQ2_0 stores 2 bits/element directly (without base-5 packing TQ1_0),
        // therefore, with an equal number of elements, it takes up more space.
        assert!(DType::TQ2_0.block_bytes() > DType::TQ1_0.block_bytes());
    }

    #[test]
    fn test_iq1_tq_family_str_roundtrip() {
        for dtype in [DType::IQ1_S, DType::IQ1_M, DType::TQ1_0, DType::TQ2_0] {
            let s = dtype.as_str();
            assert_eq!(
                DType::from_str(s),
                Some(dtype),
                "as_str/from_str roundtrip for {dtype:?}"
            );
        }
    }

    #[test]
    fn test_iq1_tq_family_gguf_id_roundtrip() {
        // All four family variants have valid GGUF ids and round-trip.
        for dtype in [DType::IQ1_S, DType::IQ1_M, DType::TQ1_0, DType::TQ2_0] {
            let id = dtype.gguf_type_id();
            assert_ne!(id, u32::MAX, "{dtype:?} must have a valid gguf id");
            assert_eq!(DType::from_gguf_type_id(id), Some(dtype));
        }
    }

    #[test]
    fn test_iq1_s_has_upstream_id() {
        // IQ1_S uses GGML_TYPE_IQ1_S = 19.
        assert_eq!(DType::IQ1_S.gguf_type_id(), 19);
        assert_eq!(DType::from_gguf_type_id(19), Some(DType::IQ1_S));
    }

    #[test]
    fn test_iq3_s_iq4_xs_bf16_ids_match_upstream() {
        // IQ3_S/IQ4_XS/BF16 use upstream ids 21/23/30.
        assert_eq!(DType::IQ3_S.gguf_type_id(), 21);
        assert_eq!(DType::from_gguf_type_id(21), Some(DType::IQ3_S));

        assert_eq!(DType::IQ4_XS.gguf_type_id(), 23);
        assert_eq!(DType::from_gguf_type_id(23), Some(DType::IQ4_XS));

        assert_eq!(DType::BF16.gguf_type_id(), 30);
        assert_eq!(DType::from_gguf_type_id(30), Some(DType::BF16));
    }

    #[test]
    fn test_no_gguf_id_collisions_across_all_variants() {
        // Bijectivity of gguf_type_id/from_gguf_type_id: no two variants
        // may share an id.
        let all = [
            DType::F64,
            DType::F32,
            DType::BF16,
            DType::F16,
            DType::I64,
            DType::I32,
            DType::I16,
            DType::I8,
            DType::U64,
            DType::U32,
            DType::U16,
            DType::U8,
            DType::Bool,
            DType::Q4_0,
            DType::Q4_1,
            DType::Q5_0,
            DType::Q5_1,
            DType::Q8_0,
            DType::Q2_K,
            DType::Q3_K,
            DType::Q4_K,
            DType::Q5_K,
            DType::Q6_K,
            DType::Q8_K,
            DType::IQ2_XXS,
            DType::IQ2_XS,
            DType::IQ3_XXS,
            DType::IQ3_S,
            DType::IQ4_NL,
            DType::IQ4_XS,
            DType::IQ1_S,
            DType::IQ1_M,
            DType::TQ1_0,
            DType::TQ2_0,
        ];
        let mut seen_ids: Vec<u32> = Vec::new();
        for dtype in all {
            let id = dtype.gguf_type_id();
            if id == u32::MAX {
                continue; // allowed: DType without GGUF equivalent (U64/U32/U16/U8/Bool)
            }
            assert!(
                !seen_ids.contains(&id),
                "duplicate gguf_type_id {id} involving {dtype:?}"
            );
            seen_ids.push(id);
        }
    }

    /// Block size formulas directly from `ggml-common.h`.
    ///
    /// The formulas mirror the upstream `static_assert`s, so each value is
    /// traceable to its layout definition.
    #[test]
    fn test_block_sizes_match_ggml_upstream() {
        const QK_K: usize = 256;
        const K_SCALE_SIZE: usize = 12;
        const IQ3S_N_SCALE: usize = QK_K / 64;
        const HALF: usize = 2; // ggml_half
        const U16: usize = 2;
        const U32: usize = 4;
        const F32: usize = 4;
        const I16: usize = 2;

        let expected: &[(DType, usize)] = &[
            (DType::Q4_0, HALF + 32 / 2),
            (DType::Q4_1, 2 * HALF + 32 / 2),
            (DType::Q5_0, HALF + U32 + 32 / 2),
            (DType::Q5_1, 2 * HALF + U32 + 32 / 2),
            (DType::Q8_0, HALF + 32),
            (DType::Q8_1, 2 * HALF + 32),
            (
                DType::TQ1_0,
                HALF + QK_K / 64 + (QK_K - 4 * (QK_K / 64)) / 5,
            ),
            (DType::TQ2_0, HALF + QK_K / 4),
            (DType::Q2_K, 2 * HALF + QK_K / 16 + QK_K / 4),
            (DType::Q3_K, HALF + QK_K / 4 + QK_K / 8 + 12),
            (DType::Q4_K, 2 * HALF + K_SCALE_SIZE + QK_K / 2),
            (DType::Q5_K, 2 * HALF + K_SCALE_SIZE + QK_K / 2 + QK_K / 8),
            (DType::Q6_K, HALF + QK_K / 16 + 3 * QK_K / 4),
            (DType::Q8_K, F32 + QK_K + (QK_K / 16) * I16),
            (DType::IQ2_XXS, HALF + (QK_K / 8) * U16),
            (DType::IQ2_XS, HALF + (QK_K / 8) * U16 + QK_K / 32),
            (DType::IQ3_XXS, HALF + 3 * (QK_K / 8)),
            (DType::IQ3_S, HALF + 13 * (QK_K / 32) + IQ3S_N_SCALE),
            (DType::IQ1_S, HALF + QK_K / 8 + QK_K / 16),
            (DType::IQ1_M, QK_K / 8 + QK_K / 16 + QK_K / 32),
            (DType::IQ4_NL, HALF + 32 / 2),
            (DType::IQ4_XS, HALF + U16 + QK_K / 64 + QK_K / 2),
        ];

        for &(dtype, upstream) in expected {
            assert_eq!(
                dtype.block_bytes(),
                upstream,
                "{dtype:?}: block_bytes() = {}, ggml gives {upstream}",
                dtype.block_bytes()
            );
        }
    }

    /// Numeric id types GGUF - external contract with files on disk.
    /// Values from `enum ggml_type` (ggml.h).
    #[test]
    fn test_gguf_type_ids_match_ggml_upstream() {
        let expected: &[(DType, u32)] = &[
            (DType::F32, 0),
            (DType::F16, 1),
            (DType::Q4_0, 2),
            (DType::Q4_1, 3),
            (DType::Q5_0, 6),
            (DType::Q5_1, 7),
            (DType::Q8_0, 8),
            (DType::Q8_1, 9),
            (DType::Q2_K, 10),
            (DType::Q3_K, 11),
            (DType::Q4_K, 12),
            (DType::Q5_K, 13),
            (DType::Q6_K, 14),
            (DType::Q8_K, 15),
            (DType::IQ2_XXS, 16),
            (DType::IQ2_XS, 17),
            (DType::IQ3_XXS, 18),
            (DType::IQ1_S, 19),
            (DType::IQ4_NL, 20),
            (DType::IQ3_S, 21),
            (DType::IQ4_XS, 23),
            (DType::I8, 24),
            (DType::I16, 25),
            (DType::I32, 26),
            (DType::I64, 27),
            (DType::F64, 28),
            (DType::IQ1_M, 29),
            (DType::BF16, 30),
            (DType::TQ1_0, 34),
            (DType::TQ2_0, 35),
        ];
        for &(dtype, id) in expected {
            assert_eq!(dtype.gguf_type_id(), id, "{dtype:?}");
            assert_eq!(
                DType::from_gguf_type_id(id),
                Some(dtype),
                "reverse mapping id {id}"
            );
        }
    }

    #[test]
    fn all_lists_every_variant_exactly_once() {
        for (i, d) in DType::ALL.iter().enumerate() {
            assert_eq!(d.ordinal(), i, "{d:?} is in the wrong place in ALL");
        }
    }

    #[test]
    fn test_bits_per_element() {
        assert_eq!(DType::F32.bits_per_element(), 32.0);
        assert_eq!(DType::F16.bits_per_element(), 16.0);
        // Q4_0: 18 bytes / 32 elems * 8 = 4.5 bpw
        assert!((DType::Q4_0.bits_per_element() - 4.5).abs() < 0.01);
    }
}
