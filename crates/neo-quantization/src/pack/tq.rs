// neo-quantization/src/pack/tq.rs
//
// Portions of this file are derived from ggml/llama.cpp
// (`ggml/src/ggml-quants.c`), Copyright (c) the ggml/llama.cpp authors,
// used here under the MIT License.
//
// TQ1_0 / TQ2_0 - GGUF-compatible ternary packaging (llama.cpp), following
// `quantize_row_tq1_0_ref`/`_tq2_0_ref` and `dequantize_row_tq1_0`/`_tq2_0`
// from `ggml/src/ggml-quants.c`.
//
// ═══════════════════════════════════════════════════════════════════════
// § 1 RELATIONSHIP TO pack::ternary
// ═══════════════════════════════════════════════════════════════════════
//
// This crate already contains `pack::ternary::TernaryPacked` - its own,
// NOT GGUF-compatible {-1,0,+1} format at 2 bits/weight (byte-aligned,
// 4 weights/byte, alpha = mean(|W|)). TQ1_0/TQ2_0 here are SEPARATE formats:
// they exist to decode GGUF files actually written by llama.cpp
// with these DTypes, so that they can be read/written bitwise identically, and
// not because they are better than this crate's own design. Differences:
//
//   • Calibration: TQ1_0/TQ2_0 use `alpha = max(|W|)` (amplitude),
//     `pack::ternary` — `alpha = mean(|W|)` (see `pack::ternary.rs`
//     heading for the rationale for mean for BitNet b1.58-style models).
//   • Layout: TQ1_0 - REAL base-3 packaging (5 trits/byte, closer to
//     theoretical limit log2(3)≈1.585 bits/weight, here 1.6875 due to
//     block padding), TQ2_0 - 2 bits/element directly (2.0625 bpw).
//     `pack::ternary` is always 2.0 bpw exactly, without base-3 options at all
//     (see justification in ternary.rs § "Storage format" - the same
//     compromise "simple unpack at the cost of not reaching the theoretical limit",
//     which is here for TQ2_0 but NOT for TQ1_0 where ggml pays the price
//     base-3 arithmetic for the sake of 1.6875 versus 2.0625 bpw).
//   • Super-block: TQ1_0/TQ2_0 — QK_K=256 (one scale for the entire block of 256
//     elements), `pack::ternary` supports arbitrary
//     `ScaleGranularity` (per-tensor/per-channel/per-group).
//
// Both formats coexist in the crate intentionally - `pack::ternary` for
// internal path (when granularity flexibility and mean calibration are important
// more precisely for BitNet-style trained models), TQ1_0/TQ2_0 for
// GGUF round-trip compatibility with existing files on disk.
//
// ═══════════════════════════════════════════════════════════════════════
// § 2 BASE-3 PACKING TQ1_0 (implementation `(q*3)>>8` trick)
// ═══════════════════════════════════════════════════════════════════════
//
// TQ1_0 encodes 5 trits `{0,1,2}` (with an offset: the real weights -1,0,1
// are stored as xi = round(w/alpha)+1 ∈ {0,1,2}) into ONE byte via
// `q = q*3 + xi` (classic base-3 accumulation, most significant trit first),
// then “stretches” the resulting number `q ∈ [0, 242]` (`3^5-1=242`) to
// the entire byte range via `ceil(q * 256 / 243)` — this is NOT the same as
// `q` directly (`242 < 256`, but each base-3 value must, after
// rounding, map to DIFFERENT byte values so that reverse
// decoding `((byte * pow3[n]) as u16 * 3) >> 8` restores
// the original `xi` losslessly — division by 243 via an 8-bit shift
// (`>>8` ≈ `/256`) is approximate but accurate enough, because
// the original multiplication by `256/243` has already stretched the range with a margin).
// It is implementation bitwise (the same arithmetic in `u8`/`u16`, the same order
// of operations), and is not reinvented through `f32`, because the resulting
// bytes must match byte-for-byte the files actually written by
// llama.cpp.
//
// ═══════════════════════════════════════════════════════════════════════

use burn::tensor::f16;

// ═══════════════════════════════════════════════════════════════════════
// § 3 TQ1_0 - 54 bytes / 256 weights (1.6875 bpw)
// ═══════════════════════════════════════════════════════════════════════
//
// Layout (block_tq1_0 in ggml-common.h):
//   qs: [u8; 32] - 5 trits/byte (32 = ceil((256 - 4*4)/5)), two passes:
//                      32-byte groups (156 triples? see below) and one
//                      16-byte tail
//   qh: [u8; 4] — 4 trits/byte in base-3, separate “fast” section
//   d:   f16        — super-block scale (amax)
//
// Upstream implementation literally: cycle through `j` in steps of 32 (as much as fits entirely),
// then 16 each (remainder), then a separate section `qh` in steps of 4.

/// Packed TQ1_0 super-block tensor.
#[derive(Debug, Clone, PartialEq)]
pub struct Tq1_0Packed {
    /// Raw super-blocks (54 bytes each).
    pub blocks: Vec<u8>,
    /// Initial number of scales.
    pub numel: usize,
}

impl Tq1_0Packed {
    /// Initial number of scales.
    pub fn numel(&self) -> usize {
        self.numel
    }
    /// Size in bytes.
    pub fn bytes_len(&self) -> usize {
        self.blocks.len()
    }
    /// Expand to `Vec<f32>`.
    pub fn to_f32(&self) -> Vec<f32> {
        dequantize_tq1_0(self)
    }
}

const TQ1_0_BLOCK_BYTES: usize = 54; // 4 + 48 + 2, see crate::dtype::DType::TQ1_0.block_bytes()
const TQ1_0_QS_LEN: usize = 48; // (QK_K - 4*QK_K/64)/5 = (256-16)/5 = 48
const TQ1_0_QH_LEN: usize = 4; // QK_K/64 = 4

/// `pow3` upstream: powers of three for base-3 decoding of a single byte.
/// Powers of three in `u8` - for a decoder where multiplication is REQUIRED
/// overflow modulo 256 (see `extract_trit`).
const POW3_U8: [u8; 6] = [1, 3, 9, 27, 81, 243];

/// Extract one trit from a base-3 packed byte.
///
/// # Why multiplication is 8-bit
///
/// In `dequantize_row_tq1_0` (ggml) the intermediate product is declared as
/// `uint8_t q = qs[j] * pow3[n];` - that is, it is CUTTED to 8 bits, and this is not
/// an accident, but part of the scheme: coding does
/// `q = (q*256 + 242)/243`, but it turns out that the most significant bits
/// are discarded, and the desired trit is moved to the upper two bits, from where it
/// takes `(q*3) >> 8`.
///
/// Here the intermediate value was kept in `u16`, no truncation occurred,
/// and `xi` went outside the trit range `0..=2`. The output produced weights
/// of the form `(xi-1)*d` with `xi` up to 5 - that is, up to five times the block
/// amax, with an arbitrary sign. This did not show up at all scales, but only
/// where the high trit was non-zero, so ordinary roundtrip tests with
/// mean error missed the bug; its own sign test caught it
/// (`-4` was restored as `+38.39` at amax = 12.8).
#[inline]
fn extract_trit(byte: u8, n: usize) -> i16 {
    let q = byte.wrapping_mul(POW3_U8[n]);
    ((q as u16 * 3) >> 8) as i16
}

/// implementation `dequantize_row_tq1_0`.
pub fn dequantize_tq1_0(packed: &Tq1_0Packed) -> Vec<f32> {
    let n_blocks = packed.blocks.len() / TQ1_0_BLOCK_BYTES;
    let mut out = Vec::with_capacity(n_blocks * 256);

    for b in 0..n_blocks {
        let blk = &packed.blocks[b * TQ1_0_BLOCK_BYTES..(b + 1) * TQ1_0_BLOCK_BYTES];
        let qs = &blk[0..TQ1_0_QS_LEN];
        let qh = &blk[TQ1_0_QS_LEN..TQ1_0_QS_LEN + TQ1_0_QH_LEN];
        let d = f16::from_le_bytes([
            blk[TQ1_0_QS_LEN + TQ1_0_QH_LEN],
            blk[TQ1_0_QS_LEN + TQ1_0_QH_LEN + 1],
        ])
        .to_f32();

        // qs: 32-byte groups, then 16-byte tail (48 = 32 + 16)
        for j in (0..32).step_by(32) {
            for n in 0..5 {
                for m in 0..32 {
                    let xi = extract_trit(qs[j + m], n);
                    out.push((xi - 1) as f32 * d);
                }
            }
        }
        for j in (32..TQ1_0_QS_LEN).step_by(16) {
            for n in 0..5 {
                for m in 0..16 {
                    let xi = extract_trit(qs[j + m], n);
                    out.push((xi - 1) as f32 * d);
                }
            }
        }
        // qh: separate 4-byte section, step 4 (n=0..4, not 0..5 - qh
        // stores only 4 of 5 "slots" on n, missing fifth contribution
        // the above is already covered through qs-cycles; loop implementation
        // `for n in 0..4 { for j in 0..sizeof(qh) { ... } }` upstream).
        for n in 0..4 {
            for j in 0..TQ1_0_QH_LEN {
                let xi = extract_trit(qh[j], n);
                out.push((xi - 1) as f32 * d);
            }
        }
    }

    out.truncate(packed.numel);
    out
}

/// Quantize `data` into TQ1_0.
///
/// `alpha = max(|W|)` for a block of 256 weights (not `mean`, unlike
/// `pack::ternary` - see § 1 in the module title).
///
/// #Panic
///
/// Panics if `data.len()` is not a multiple of 256 (`QK_K`) - just like upstream.
pub fn quantize_tq1_0(data: &[f32]) -> Tq1_0Packed {
    assert_eq!(
        data.len() % 256,
        0,
        "quantize_tq1_0: len must be a multiple of QK_K=256"
    );
    let n_blocks = data.len() / 256;
    let mut blocks = vec![0u8; n_blocks * TQ1_0_BLOCK_BYTES];

    for ibl in 0..n_blocks {
        let mut xbl = &data[ibl * 256..(ibl + 1) * 256];
        let amax = xbl.iter().fold(0.0f32, |a, v| a.max(v.abs()));
        let d = amax;
        let id = if d != 0.0 { 1.0 / d } else { 0.0 };

        let blk = &mut blocks[ibl * TQ1_0_BLOCK_BYTES..(ibl + 1) * TQ1_0_BLOCK_BYTES];
        let (qs, rest) = blk.split_at_mut(TQ1_0_QS_LEN);
        let (qh, d_slot) = rest.split_at_mut(TQ1_0_QH_LEN);

        // 32-byte group (m=0..32, n=0..5): consumes 5*32=160 elements
        for j in (0..32).step_by(32) {
            for m in 0..32 {
                let mut q: u8 = 0;
                for n in 0..5 {
                    let xi = (nearest_int(xbl[m + n * 32] * id) + 1).clamp(0, 2) as u8;
                    q = q.wrapping_mul(3).wrapping_add(xi);
                }
                q = (q as u16 * 256).div_ceil(243) as u8;
                qs[j + m] = q;
            }
            xbl = &xbl[5 * 32..];
        }
        // 16-byte tail (m=0..16, n=0..5): consumes 5*16=80 elements
        for j in (32..TQ1_0_QS_LEN).step_by(16) {
            for m in 0..16 {
                let mut q: u8 = 0;
                for n in 0..5 {
                    let xi = (nearest_int(xbl[m + n * 16] * id) + 1).clamp(0, 2) as u8;
                    q = q.wrapping_mul(3).wrapping_add(xi);
                }
                q = (q as u16 * 256).div_ceil(243) as u8;
                qs[j + m] = q;
            }
            xbl = &xbl[5 * 16..];
        }
        // qh (4 bytes, m=0..4, n=0..4): consumes 4*4=16 elements, exactly
        // remainder 256 - 160 - 80 = 16.
        for j in 0..TQ1_0_QH_LEN {
            let mut q: u8 = 0;
            for m in 0..4 {
                let xi = (nearest_int(xbl[j + m * TQ1_0_QH_LEN] * id) + 1).clamp(0, 2) as u8;
                q = q.wrapping_mul(3).wrapping_add(xi);
            }
            // "shift the first value to the most significant trit" - add.
            // multiplying by 3 before the final ceiling division (upstream).
            q = q.wrapping_mul(3);
            q = (q as u16 * 256).div_ceil(243) as u8;
            qh[j] = q;
        }
        xbl = &xbl[4 * TQ1_0_QH_LEN..];
        debug_assert!(xbl.is_empty(), "tq1_0: block must be fully consumed");

        d_slot.copy_from_slice(&crate::pack::f16_saturating(d).to_le_bytes());
    }

    Tq1_0Packed {
        blocks,
        numel: data.len(),
    }
}

// ═══════════════════════════════════════════════════════════════════════
// § 4 TQ2_0 - 66 bytes / 256 weights (2.0625 bpw)
// ═══════════════════════════════════════════════════════════════════════
//
// Layout (block_tq2_0 in ggml-common.h): qs: [u8; 64] (2 bits/element
// directly, 4 weights/byte), d: f16. The simplest of all ternary formats
// This crate is direct bitwise encoding, without base-3.

/// Packed TQ2_0 super-block tensor.
#[derive(Debug, Clone, PartialEq)]
pub struct Tq2_0Packed {
    /// Raw super-blocks (66 bytes each).
    pub blocks: Vec<u8>,
    /// Initial number of scales.
    pub numel: usize,
}

impl Tq2_0Packed {
    /// Initial number of scales.
    pub fn numel(&self) -> usize {
        self.numel
    }
    /// Size in bytes.
    pub fn bytes_len(&self) -> usize {
        self.blocks.len()
    }
    /// Expand to `Vec<f32>`.
    pub fn to_f32(&self) -> Vec<f32> {
        dequantize_tq2_0(self)
    }
}

const TQ2_0_BLOCK_BYTES: usize = 66; // 64 + 2, see crate::dtype::DType::TQ2_0.block_bytes()
const TQ2_0_QS_LEN: usize = 64; // QK_K/4 = 64

/// implementation `dequantize_row_tq2_0`.
pub fn dequantize_tq2_0(packed: &Tq2_0Packed) -> Vec<f32> {
    let n_blocks = packed.blocks.len() / TQ2_0_BLOCK_BYTES;
    let mut out = Vec::with_capacity(n_blocks * 256);

    for b in 0..n_blocks {
        let blk = &packed.blocks[b * TQ2_0_BLOCK_BYTES..(b + 1) * TQ2_0_BLOCK_BYTES];
        let qs = &blk[0..TQ2_0_QS_LEN];
        let d = f16::from_le_bytes([blk[TQ2_0_QS_LEN], blk[TQ2_0_QS_LEN + 1]]).to_f32();

        for j in (0..TQ2_0_QS_LEN).step_by(32) {
            for l in 0..4 {
                for m in 0..32 {
                    let q = ((qs[j + m] >> (l * 2)) & 3) as i8;
                    out.push((q - 1) as f32 * d);
                }
            }
        }
    }

    out.truncate(packed.numel);
    out
}

/// Quantize `data` into TQ2_0.
///
/// `alpha = max(|W|)` per block of 256 weights, direct 2-bit/element
/// encoding (`xi = round(w/alpha)+1 ∈ {0,1,2}`, packed 4
/// element/byte via `xi << 2n`).
///
/// #Panic
///
/// Panics if `data.len()` is not a multiple of 256 (`QK_K`).
pub fn quantize_tq2_0(data: &[f32]) -> Tq2_0Packed {
    assert_eq!(
        data.len() % 256,
        0,
        "quantize_tq2_0: len must be a multiple of QK_K=256"
    );
    let n_blocks = data.len() / 256;
    let mut blocks = vec![0u8; n_blocks * TQ2_0_BLOCK_BYTES];

    for ibl in 0..n_blocks {
        let mut xbl = &data[ibl * 256..(ibl + 1) * 256];
        let amax = xbl.iter().fold(0.0f32, |a, v| a.max(v.abs()));
        let d = amax;
        let id = if d != 0.0 { 1.0 / d } else { 0.0 };

        let blk = &mut blocks[ibl * TQ2_0_BLOCK_BYTES..(ibl + 1) * TQ2_0_BLOCK_BYTES];
        let (qs, d_slot) = blk.split_at_mut(TQ2_0_QS_LEN);

        for j in (0..TQ2_0_QS_LEN).step_by(32) {
            for m in 0..32 {
                let mut q: u8 = 0;
                for n in 0..4 {
                    let xi = (nearest_int(xbl[m + n * 32] * id) + 1).clamp(0, 2) as u8;
                    q += (xi & 3) << (2 * n);
                }
                qs[j + m] = q;
            }
            xbl = &xbl[4 * 32..];
        }
        debug_assert!(xbl.is_empty(), "tq2_0: block must be fully consumed");

        d_slot.copy_from_slice(&crate::pack::f16_saturating(d).to_le_bytes());
    }

    Tq2_0Packed {
        blocks,
        numel: data.len(),
    }
}

// ═══════════════════════════════════════════════════════════════════════
// § 5 GENERAL HELPERS
// ═══════════════════════════════════════════════════════════════════════

/// `lroundf`-equivalent to upstream (round-half-away-from-zero) - TQ1_0/TQ2_0
/// use `lroundf`, NOT `nearest_int`-bit-trick, unlike
/// `pack::iq_codebook`/`pack::iq1` (which implementation `nearest_int` from
/// `ggml-quants.c` for IQ series). Different upstream functions for different
/// formats - saved as is, not artificially unified.
#[inline]
fn nearest_int(x: f32) -> i32 {
    x.round() as i32
}

// ═══════════════════════════════════════════════════════════════════════
// § 5.5 TQ PACKED (type-erased) - single entry point
// ═══════════════════════════════════════════════════════════════════════
//
// Before this block, `Tq1_0Packed`/`Tq2_0Packed` existed as separate,
// unrelated types - there was no common wrapper (analogous to `IqPacked` in iq.rs),
// no dispatching via `DType`, no connection to `PackedBuffer` in
// `pack::mod`. This made TQ1_0/TQ2_0 unreachable via `quantize()`/
// `WeightRegistry` are the only official entry points for the crate. Below -
// the same pattern as `IqPacked`/`quantize_iq` in iq.rs, for the sake of consistency
// API between all "GGUF native" formats (Gguf/KQuant/Iq/Tq).

use crate::dtype::DType;

/// `true`, if DType is one of TQ1_0/TQ2_0 (GGUF-compatible ternary
/// super-block formats, see file header § 1 for differences from
/// `pack::ternary::TernaryPacked`).
pub fn is_tq_series(dtype: DType) -> bool {
    matches!(dtype, DType::TQ1_0 | DType::TQ2_0)
}

/// Packed result TQ1_0 or TQ2_0 - type-erased wrapper, the same
/// pattern that `IqPacked` in `pack::iq`.
#[derive(Debug, Clone, PartialEq)]
pub enum TqPacked {
    /// TQ1_0 - base-3 packaging, 1.6875 bits/weight, see § 2/§ 3.
    Tq1_0(Tq1_0Packed),
    /// TQ2_0 - direct 2-bit/element packing, 2.0625 bit/weight, see § 4.
    Tq2_0(Tq2_0Packed),
}

impl TqPacked {
    /// GGUF DType of this buffer.
    pub fn dtype(&self) -> DType {
        match self {
            TqPacked::Tq1_0(_) => DType::TQ1_0,
            TqPacked::Tq2_0(_) => DType::TQ2_0,
        }
    }
    /// Initial number of scales.
    pub fn numel(&self) -> usize {
        match self {
            TqPacked::Tq1_0(p) => p.numel(),
            TqPacked::Tq2_0(p) => p.numel(),
        }
    }
    /// Size in bytes actually occupied by blocks.
    pub fn bytes_len(&self) -> usize {
        match self {
            TqPacked::Tq1_0(p) => p.bytes_len(),
            TqPacked::Tq2_0(p) => p.bytes_len(),
        }
    }
    /// Expand to `Vec<f32>`.
    pub fn to_f32(&self) -> Vec<f32> {
        match self {
            TqPacked::Tq1_0(p) => p.to_f32(),
            TqPacked::Tq2_0(p) => p.to_f32(),
        }
    }
}

/// Quantize `data` into TQ format `dtype` (`TQ1_0` or `TQ2_0`).
///
/// #Panic
///
/// If `dtype` is not part of the TQ-series (see `is_tq_series`), or if
/// `data.len()` is not a multiple of 256 (`QK_K`) - see `quantize_tq1_0`/`quantize_tq2_0`.
pub fn quantize_tq(data: &[f32], dtype: DType) -> TqPacked {
    match dtype {
        DType::TQ1_0 => TqPacked::Tq1_0(quantize_tq1_0(data)),
        DType::TQ2_0 => TqPacked::Tq2_0(quantize_tq2_0(data)),
        _ => panic!("pack::tq::quantize_tq: {dtype:?} is not a TQ-series DType"),
    }
}

/// Completely expand the TQ package back to `Vec<f32>`.
pub fn dequantize_tq(packed: &TqPacked) -> Vec<f32> {
    packed.to_f32()
}

// ═══════════════════════════════════════════════════════════════════════
// § 6 TESTS
// ═══════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    fn ramp(n: usize) -> Vec<f32> {
        (0..n)
            .map(|i| ((i as f32) - (n as f32) / 2.0) * 0.05)
            .collect()
    }

    #[test]
    fn test_tq1_0_block_size_matches_dtype() {
        assert_eq!(TQ1_0_BLOCK_BYTES, crate::dtype::DType::TQ1_0.block_bytes());
    }

    #[test]
    fn test_tq2_0_block_size_matches_dtype() {
        assert_eq!(TQ2_0_BLOCK_BYTES, crate::dtype::DType::TQ2_0.block_bytes());
    }

    #[test]
    fn test_tq1_0_roundtrip_shape() {
        let data = ramp(256);
        let packed = quantize_tq1_0(&data);
        assert_eq!(packed.numel(), 256);
        assert_eq!(packed.bytes_len(), TQ1_0_BLOCK_BYTES);
        assert_eq!(packed.to_f32().len(), 256);
    }

    #[test]
    fn test_tq2_0_roundtrip_shape() {
        let data = ramp(256);
        let packed = quantize_tq2_0(&data);
        assert_eq!(packed.numel(), 256);
        assert_eq!(packed.bytes_len(), TQ2_0_BLOCK_BYTES);
        assert_eq!(packed.to_f32().len(), 256);
    }

    /// Checking features for ternary format.
    ///
    /// The threshold here is RELATIVE (a share of the amax block), and this is important:
    /// an absolute bound is unrelated to the block scale, whereas
    /// round-to-nearest ternary quantization sends everything to zero
    /// which in modulus is less than half the block scale, that is, with amax = 12.8
    /// everything is reset to zero up to |w|=6.4. A value of -6.35 → 0 is
    /// exactly what the format is supposed to do. Zero has no sign, so
    /// there is only one meaningful statement: non-zero results have a sign
    /// coincides with the original one, and the weights near amax must be non-zero.
    fn assert_ternary_signs(data: &[f32], restored: &[f32], block: usize) {
        let mut checked = 0;
        for (bi, (din, dout)) in data.chunks(block).zip(restored.chunks(block)).enumerate() {
            let amax = din.iter().fold(0.0f32, |a, v| a.max(v.abs()));
            for (i, (&a, &b)) in din.iter().zip(dout.iter()).enumerate() {
                if b != 0.0 {
                    assert_eq!(
                        a.is_sign_negative(),
                        b.is_sign_negative(),
                        "block {bi}, element {i}: sign not saved, {a} -> {b}"
                    );
                    checked += 1;
                }
                // Weights from 0.75 amax are required to survive quantization by non-zero:
                // otherwise the block scale is selected incorrectly.
                if a.abs() >= 0.75 * amax && amax > 0.0 {
                    assert_ne!(
                        b, 0.0,
                        "block {bi}, element {i}: |{a}| >= 0.75 amax({amax}), but went to 0"
                    );
                }
            }
        }
        assert!(checked > 0, "the test didn't check anything");
    }

    #[test]
    fn test_tq1_0_roundtrip_signs_preserved() {
        let data = ramp(512);
        let restored = quantize_tq1_0(&data).to_f32();
        assert_ternary_signs(&data, &restored, 256);
    }

    #[test]
    fn test_tq2_0_roundtrip_signs_preserved() {
        let data = ramp(512);
        let restored = quantize_tq2_0(&data).to_f32();
        assert_ternary_signs(&data, &restored, 256);
    }

    #[test]
    fn test_tq2_0_uses_more_bits_than_tq1_0() {
        let bpw_1 = TQ1_0_BLOCK_BYTES as f32 * 8.0 / 256.0;
        let bpw_2 = TQ2_0_BLOCK_BYTES as f32 * 8.0 / 256.0;
        assert!((bpw_1 - 1.6875).abs() < 1e-6);
        assert!((bpw_2 - 2.0625).abs() < 1e-6);
        assert!(bpw_2 > bpw_1);
    }

    #[test]
    fn test_all_zero_block_is_safe() {
        let data = vec![0.0f32; 256];
        let p1 = quantize_tq1_0(&data);
        assert!(p1.to_f32().iter().all(|&v| v == 0.0));
        let p2 = quantize_tq2_0(&data);
        assert!(p2.to_f32().iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_extreme_values_map_to_plus_minus_amax() {
        let mut data = vec![0.0f32; 256];
        data[0] = 10.0;
        data[1] = -10.0;
        let p1 = quantize_tq1_0(&data);
        let r1 = p1.to_f32();
        assert!((r1[0] - 10.0).abs() < 1e-2);
        assert!((r1[1] + 10.0).abs() < 1e-2);

        let p2 = quantize_tq2_0(&data);
        let r2 = p2.to_f32();
        assert!((r2[0] - 10.0).abs() < 1e-2);
        assert!((r2[1] + 10.0).abs() < 1e-2);
    }

    #[test]
    #[should_panic(expected = "must be a multiple of QK_K")]
    fn test_tq1_0_rejects_non_multiple_of_256() {
        let _ = quantize_tq1_0(&vec![0.0f32; 100]);
    }

    #[test]
    #[should_panic(expected = "must be a multiple of QK_K")]
    fn test_tq2_0_rejects_non_multiple_of_256() {
        let _ = quantize_tq2_0(&vec![0.0f32; 100]);
    }

    #[test]
    fn test_multi_block_independent() {
        // 2 blocks of 256 - the second block with a much smaller amplitude is not
        // should influence the scale of the first one (each blk has its own super-scale).
        let mut data = vec![0.0f32; 512];
        for i in 0..256 {
            data[i] = if i % 2 == 0 { 8.0 } else { -8.0 };
        }
        for i in 256..512 {
            data[i] = if i % 2 == 0 { 0.1 } else { -0.1 };
        }

        let packed = quantize_tq1_0(&data);
        let restored = packed.to_f32();
        assert!((restored[0] - 8.0).abs() < 0.5);
        assert!((restored[256] - 0.1).abs() < 0.05);
    }

    // ── TqPacked / quantize_tq (single entry point, see § 5.5) ──────────

    #[test]
    fn test_is_tq_series() {
        assert!(is_tq_series(DType::TQ1_0));
        assert!(is_tq_series(DType::TQ2_0));
        assert!(!is_tq_series(DType::Q4_0));
        assert!(!is_tq_series(DType::IQ1_S));
    }

    #[test]
    fn test_quantize_tq_dispatches_both_variants() {
        let data = ramp(256);

        let p1 = quantize_tq(&data, DType::TQ1_0);
        assert!(matches!(p1, TqPacked::Tq1_0(_)));
        assert_eq!(p1.dtype(), DType::TQ1_0);
        assert_eq!(p1.numel(), 256);
        assert_eq!(p1.bytes_len(), TQ1_0_BLOCK_BYTES);

        let p2 = quantize_tq(&data, DType::TQ2_0);
        assert!(matches!(p2, TqPacked::Tq2_0(_)));
        assert_eq!(p2.dtype(), DType::TQ2_0);
        assert_eq!(p2.numel(), 256);
        assert_eq!(p2.bytes_len(), TQ2_0_BLOCK_BYTES);
    }

    #[test]
    fn test_quantize_tq_matches_direct_call() {
        let data = ramp(512);
        let via_dispatch = quantize_tq(&data, DType::TQ1_0);
        let via_direct = quantize_tq1_0(&data);
        assert_eq!(via_dispatch.to_f32(), via_direct.to_f32());
    }

    #[test]
    fn test_dequantize_tq_matches_to_f32() {
        let data = ramp(256);
        let packed = quantize_tq(&data, DType::TQ2_0);
        assert_eq!(dequantize_tq(&packed), packed.to_f32());
    }

    #[test]
    #[should_panic(expected = "is not a TQ-series DType")]
    fn test_quantize_tq_rejects_non_tq_dtype() {
        let _ = quantize_tq(&vec![0.0f32; 256], DType::Q4_0);
    }
}
