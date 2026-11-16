// neo-quantization/src/pack/gguf.rs
//
// Portions of this file implement GGUF formats (llama.cpp), Copyright (c)
// the ggml/llama.cpp authors, used here under the MIT License.
//
// GGUF-compatible packaging - bitwise identical to llama.cpp.
//
// This file implements the simple block formats (Q4_0/Q4_1/Q5_0/Q5_1/Q8_0).
// K-quants live in `pack::kquant`, the IQ series in `pack::iq`/`pack::iq1`/
// `pack::iq_codebook`, and TQ1_0/TQ2_0 in `pack::tq` (see
// `pack::gguf_backend`). An unsupported DType panics explicitly rather than
// producing a silent incorrect result.
//
// ## Why these formats first
//
// Q4_0/Q4_1/Q5_0/Q5_1/Q8_0 - the simplest GGUF blocks: fixed
// block_elems=32, one float16 scale (+ optional min for _1 options)
// per block, without nested hierarchy. Exactly the case where "GGUF-compatible
// "packing" comes down to an exact repetition of the byte layout structure
// block - in contrast to K-quants, where the layout of the block itself is
// several sub-blocks with their own truncated scale codes.
//
// ## Layout (little-endian, as in GGUF/llama.cpp)
//
//   Q4_0 block (18 bytes, 32 weights):
//     d: f16 (2 bytes) — scale
//     qs: [u8; 16] — 32 nibble codes, signed-biased
//                                          (code - 8), 2 weights/byte
//     W_hat = (nibble - 8) * d
//
//   Q8_0 block (34 bytes, 32 weights):
//     d: f16 (2 bytes) — scale
//     qs: [i8; 32] - signed code directly
//     W_hat = qs[i] * d
//
//   Q4_1 block (20 bytes, 32 weights) - asymmetric with explicit min:
//     d: f16, m: f16 (4 bytes) — scale, min
//     qs: [u8; 16] - 32 nibble codes, unsigned [0.15]
//     W_hat = qs[i] * d + m
//
//   Q5_0/Q5_1 - 5-bit options with an additional field `qh: u32`,
//   carrying the most significant (5th) bit of each of the 32 codes separately from the main one
//   nibble-array `qs`. Implemented according to the same scheme as Q4_0/Q4_1, with
//   this additional field.

use burn::tensor::f16;

use crate::dtype::DType;

// ═══════════════════════════════════════════════════════════════════════════
// § 1  PACKED CONTAINER
// ═══════════════════════════════════════════════════════════════════════════

/// GGUF-compatible packed tensor - raw bytes in `dtype` format,
/// bitwise identical to what llama.cpp would write/read.
#[derive(Debug, Clone, PartialEq)]
pub struct GgufPacked {
    /// Specific GGUF DType (defines block layout).
    pub dtype: DType,
    /// Raw blocks concatenated in a row (without tensor header -
    /// title/name/shape is `neo-gguf`'s concern, not this file's).
    pub blocks: Vec<u8>,
    /// Initial number of weights (last block may be incomplete if
    /// `numel % block_elems != 0` - padded with zeros during quantization).
    pub numel: usize,
}

impl GgufPacked {
    /// The initial number of weights that the buffer represents.
    pub fn numel(&self) -> usize {
        self.numel
    }
    /// The size in bytes actually occupied by packed blocks.
    pub fn bytes_len(&self) -> usize {
        self.blocks.len()
    }

    /// Expand to `Vec<f32>`. Panics for DType whose layout is not included
    /// in `is_supported` (K-quants/IQ/TQ - other modules).
    pub fn to_f32(&self) -> Vec<f32> {
        dequantize_gguf(self)
    }
}

/// `true`, if `pack::gguf` can pack/unpack this DType
/// (simple block formats Q4_0..Q8_1). K-quants/IQ/TQ are packaged
/// modules `pack::kquant`/`pack::iq`/`pack::tq` - single point of choice:
/// `pack::gguf_backend`.
pub fn is_supported(dtype: DType) -> bool {
    matches!(
        dtype,
        DType::Q4_0 | DType::Q4_1 | DType::Q5_0 | DType::Q5_1 | DType::Q8_0 | DType::Q8_1
    )
}

// ═══════════════════════════════════════════════════════════════════════════
// § 2  QUANTIZE
// ═══════════════════════════════════════════════════════════════════════════

/// Quantize `data` into GGUF-compatible block format `dtype`.
///
/// #Panic
///
/// If `dtype` is not included in `is_supported` (K-quants/IQ/TQ are implemented in
/// other modules - see `pack::gguf_backend`).
pub fn quantize_gguf(data: &[f32], dtype: DType) -> GgufPacked {
    assert!(
        is_supported(dtype),
        "pack::gguf::quantize_gguf: {dtype:?} block layout is not implemented yet \
         (K-quants - pack::kquant, IQ - pack::iq, TQ - pack::tq; see pack::gguf_backend)"
    );

    let block_elems = dtype.block_elems();
    let n_blocks = data.len().div_ceil(block_elems);
    let mut blocks = Vec::with_capacity(n_blocks * dtype.block_bytes());

    for b in 0..n_blocks {
        let start = b * block_elems;
        let end = (start + block_elems).min(data.len());
        let mut chunk = [0.0f32; 32];
        chunk[..end - start].copy_from_slice(&data[start..end]);
        let chunk = &chunk[..block_elems.min(32)];

        match dtype {
            DType::Q4_0 => encode_q4_0(chunk, &mut blocks),
            DType::Q4_1 => encode_q4_1(chunk, &mut blocks),
            DType::Q5_0 => encode_q5_0(chunk, &mut blocks),
            DType::Q5_1 => encode_q5_1(chunk, &mut blocks),
            DType::Q8_0 => encode_q8_0(chunk, &mut blocks),
            DType::Q8_1 => encode_q8_1(chunk, &mut blocks),
            _ => unreachable!("checked by is_supported above"),
        }
    }

    GgufPacked {
        dtype,
        blocks,
        numel: data.len(),
    }
}

fn encode_q4_0(chunk: &[f32], out: &mut Vec<u8>) {
    // ggml semantics (quantize_row_q4_0_ref): find index where |w| maximum
    // and take the value AT THIS POINT SIGNED (not abs!) as `max`. Then
    // d = max / -8.0 - the minus sign here specifically gives d the correct sign
    // so that the range of nibbles comes out symmetrically [-8,7] without a separate
    // clamp on both sides. Divide by -8, NOT by 7 from amax.
    let mut max = 0.0f32;
    let mut amax = 0.0f32;
    for &w in chunk {
        if w.abs() > amax {
            amax = w.abs();
            max = w;
        }
    }
    let d = max / -8.0;
    let d_safe = if d == 0.0 { 1.0 } else { d };
    let id = 1.0 / d_safe;

    out.extend_from_slice(&crate::pack::f16_saturating(d).to_le_bytes());
    for pair in chunk.chunks(2) {
        let q0 = quantize_nibble_signed(pair[0], id);
        let q1 = if pair.len() > 1 {
            quantize_nibble_signed(pair[1], id)
        } else {
            8
        };
        out.push(q0 | (q1 << 4));
    }
}

fn encode_q4_1(chunk: &[f32], out: &mut Vec<u8>) {
    // min/max strictly from the block data - DO NOT expand to include zero
    // (unlike pack::intn::quantize Asymmetric, where you need exact 0
    // for padding scales). This is llama.cpp Q4_1 semantics: a block of weights,
    // far from zero (typical for some bias tensors), should
    // get a narrow, precise mesh - expansion to 0 without the need
    // tripled the quantization step on such data (see bug history:
    // test_q4_1_handles_skewed_data).
    let min = chunk.iter().cloned().fold(f32::INFINITY, f32::min);
    let max = chunk.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let d = (max - min) / 15.0;
    let d_safe = if d == 0.0 { 1.0 } else { d };

    out.extend_from_slice(&crate::pack::f16_saturating(d_safe).to_le_bytes());
    out.extend_from_slice(&crate::pack::f16_saturating(min).to_le_bytes());
    for pair in chunk.chunks(2) {
        let q0 = quantize_nibble_unsigned(pair[0], d_safe, min);
        let q1 = if pair.len() > 1 {
            quantize_nibble_unsigned(pair[1], d_safe, min)
        } else {
            0
        };
        out.push(q0 | (q1 << 4));
    }
}

fn encode_q5_0(chunk: &[f32], out: &mut Vec<u8>) {
    // Same bug/fix as in encode_q4_0: ggml takes signed `max` at point
    // maximum module (not amax), divides by -16 (not amax/15).
    let mut max = 0.0f32;
    let mut amax = 0.0f32;
    for &w in chunk {
        if w.abs() > amax {
            amax = w.abs();
            max = w;
        }
    }
    let d = max / -16.0;
    let d_safe = if d == 0.0 { 1.0 } else { d };
    let id = 1.0 / d_safe;

    let mut qh: u32 = 0;
    let mut nibbles = [0u8; 32];
    for (i, &w) in chunk.iter().enumerate() {
        let q5 = quantize_5bit_signed(w, id); // 0..=31
        nibbles[i] = (q5 & 0x0F) as u8;
        if q5 & 0x10 != 0 {
            qh |= 1 << i;
        }
    }

    out.extend_from_slice(&crate::pack::f16_saturating(d).to_le_bytes());
    out.extend_from_slice(&qh.to_le_bytes());
    for pair in nibbles.chunks(2) {
        out.push(pair[0] | (pair[1] << 4));
    }
}

fn encode_q5_1(chunk: &[f32], out: &mut Vec<u8>) {
    // min/max strictly from block data - see the comment in encode_q4_1 above
    // (the range is not expanded to include zero, matching llama.cpp semantics).
    let min = chunk.iter().cloned().fold(f32::INFINITY, f32::min);
    let max = chunk.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
    let d = (max - min) / 31.0;
    let d_safe = if d == 0.0 { 1.0 } else { d };

    let mut qh: u32 = 0;
    let mut nibbles = [0u8; 32];
    for (i, &w) in chunk.iter().enumerate() {
        let q5 = (((w - min) / d_safe).round().clamp(0.0, 31.0)) as u32;
        nibbles[i] = (q5 & 0x0F) as u8;
        if q5 & 0x10 != 0 {
            qh |= 1 << i;
        }
    }

    out.extend_from_slice(&crate::pack::f16_saturating(d_safe).to_le_bytes());
    out.extend_from_slice(&crate::pack::f16_saturating(min).to_le_bytes());
    out.extend_from_slice(&qh.to_le_bytes());
    for pair in nibbles.chunks(2) {
        out.push(pair[0] | (pair[1] << 4));
    }
}

fn encode_q8_0(chunk: &[f32], out: &mut Vec<u8>) {
    let amax = chunk.iter().fold(0.0f32, |acc, w| acc.max(w.abs()));
    let d = amax / 127.0;
    let d_safe = if d == 0.0 { 1.0 } else { d };

    out.extend_from_slice(&crate::pack::f16_saturating(d).to_le_bytes());
    for &w in chunk {
        let q = (w / d_safe).round().clamp(-127.0, 127.0) as i8;
        out.push(q as u8);
    }
    // We pad with zeros if the chunk is shorter than 32 (the last incomplete block).
    for _ in chunk.len()..32 {
        out.push(0);
    }
}

/// ggml Q8_1: same 8-bit symmetric quantization as Q8_0, but block
/// additionally carries `s = d · Σqs` - the pre-summed contribution of the block.
///
/// # Why is the format needed at all?
///
/// Q8_1 is almost never used as a format for STORING weights: in upstream it is
/// first of all, `vec_dot_type` for Q4_1/Q5_1 is the shape in which
/// activation before dot product with asymmetrically quantized
/// weight. The asymmetrical weight unfolds as `w = d·q + m`, so
/// the inner product splits into `d·Σ(q_w·q_x)` and `m·Σq_x`.
/// The second term does not depend on the weights - it is stored in the `s` field so that
/// The kernel did not recalculate the sum of activations for each block of weights.
///
/// However, `GGML_TYPE_Q8_1 = 9` is a valid tensor type in GGUF, so it is
/// declared in [`crate::DType`] and packed/unpacked by the code below.
///
/// # Traversal order
///
/// Upstream (`quantize_row_q8_1_ref`) goes along the block in TWO halves
/// simultaneously: `j` and `QK8_1/2 + j`. It doesn't matter to the result -
/// each element is processed exactly once and independently - but the order
/// saved so that row-by-row comparison with the upstream remains possible.
fn encode_q8_1(chunk: &[f32], out: &mut Vec<u8>) {
    let mut padded = [0.0f32; 32];
    padded[..chunk.len()].copy_from_slice(chunk);

    let amax = padded.iter().fold(0.0f32, |acc, w| acc.max(w.abs()));
    let d = amax / 127.0;
    let id = if d != 0.0 { 1.0 / d } else { 0.0 };

    let mut qs = [0i8; 32];
    let mut sum: i32 = 0;
    for j in 0..16 {
        let v0 = padded[j] * id;
        let v1 = padded[16 + j] * id;
        // `roundf` upstream - rounding to the nearest, half from zero;
        // `f32::round` behaves the same way in Rust.
        qs[j] = v0.round().clamp(-127.0, 127.0) as i8;
        qs[16 + j] = v1.round().clamp(-127.0, 127.0) as i8;
        sum += qs[j] as i32 + qs[16 + j] as i32;
    }

    out.extend_from_slice(&crate::pack::f16_saturating(d).to_le_bytes());
    out.extend_from_slice(&crate::pack::f16_saturating(sum as f32 * d).to_le_bytes());
    for &q in &qs {
        out.push(q as u8);
    }
}

/// ggml Q4_0 quantization: `q = trunc(w*id + 8.5)`, clamped to the unsigned
/// nibble range `[0, 15]`. NB: takes `id` (reciprocal of `d`), not `d` — the
/// ggml reference computes `id = 1/d` once per block and multiplies, rather
/// than dividing per-element. `trunc(x + 8.5)` is intentionally NOT the same
/// as `(w/d).round()` shifted by +8: they only agree except exactly on
/// half-integer boundaries of `w*id`.
#[inline]
fn quantize_nibble_signed(w: f32, id: f32) -> u8 {
    let x = (w * id + 8.5).trunc();
    x.clamp(0.0, 15.0) as u8
}

#[inline]
fn quantize_nibble_unsigned(w: f32, d: f32, min: f32) -> u8 {
    // (w - min)/d is always >= 0 here, so trunc(x+0.5) == round() — no
    // change needed vs. ggml (see analysis: Q4_1/Q5_1 formulas already
    // match).
    (((w - min) / d).round().clamp(0.0, 15.0)) as u8
}

/// ggml Q5_0 quantization: `q = trunc(w*id + 16.5)`, clamped to the
/// unsigned 5-bit range `[0, 31]`. Takes `id` (reciprocal of `d`), matching
/// `quantize_nibble_signed` above.
#[inline]
fn quantize_5bit_signed(w: f32, id: f32) -> u32 {
    let x = (w * id + 16.5).trunc();
    x.clamp(0.0, 31.0) as u32
}

// ═══════════════════════════════════════════════════════════════════════════
// § 3  DEQUANTIZE
// ═══════════════════════════════════════════════════════════════════════════

/// Completely expand the GGUF-compatible packaging back to `Vec<f32>`.
///
/// #Panic
///
/// If `packed.dtype` is not included in `is_supported` (K-quants/IQ).
pub fn dequantize_gguf(packed: &GgufPacked) -> Vec<f32> {
    assert!(
        is_supported(packed.dtype),
        "pack::gguf::dequantize_gguf: {:?} block layout is not implemented yet",
        packed.dtype
    );

    let block_elems = packed.dtype.block_elems();
    let block_bytes = packed.dtype.block_bytes();
    let n_blocks = packed.blocks.len() / block_bytes;
    let mut out = Vec::with_capacity(n_blocks * block_elems);

    for b in 0..n_blocks {
        let block = &packed.blocks[b * block_bytes..(b + 1) * block_bytes];
        match packed.dtype {
            DType::Q4_0 => decode_q4_0(block, &mut out),
            DType::Q4_1 => decode_q4_1(block, &mut out),
            DType::Q5_0 => decode_q5_0(block, &mut out),
            DType::Q5_1 => decode_q5_1(block, &mut out),
            DType::Q8_0 => decode_q8_0(block, &mut out),
            DType::Q8_1 => decode_q8_1(block, &mut out),
            _ => unreachable!("checked by is_supported above"),
        }
    }

    out.truncate(packed.numel);
    out
}

fn decode_q4_0(block: &[u8], out: &mut Vec<f32>) {
    let d = f16::from_le_bytes([block[0], block[1]]).to_f32();
    for &byte in &block[2..18] {
        out.push(((byte & 0x0F) as i32 - 8) as f32 * d);
        out.push(((byte >> 4) as i32 - 8) as f32 * d);
    }
}

fn decode_q4_1(block: &[u8], out: &mut Vec<f32>) {
    let d = f16::from_le_bytes([block[0], block[1]]).to_f32();
    let m = f16::from_le_bytes([block[2], block[3]]).to_f32();
    for &byte in &block[4..20] {
        out.push((byte & 0x0F) as f32 * d + m);
        out.push((byte >> 4) as f32 * d + m);
    }
}

fn decode_q5_0(block: &[u8], out: &mut Vec<f32>) {
    let d = f16::from_le_bytes([block[0], block[1]]).to_f32();
    let qh = u32::from_le_bytes([block[2], block[3], block[4], block[5]]);
    let qs = &block[6..22];
    for i in 0..32 {
        let byte = qs[i / 2];
        let low = if i % 2 == 0 { byte & 0x0F } else { byte >> 4 };
        let high_bit = ((qh >> i) & 1) as u8;
        let q5 = (low | (high_bit << 4)) as i32 - 16;
        out.push(q5 as f32 * d);
    }
}

fn decode_q5_1(block: &[u8], out: &mut Vec<f32>) {
    let d = f16::from_le_bytes([block[0], block[1]]).to_f32();
    let m = f16::from_le_bytes([block[2], block[3]]).to_f32();
    let qh = u32::from_le_bytes([block[4], block[5], block[6], block[7]]);
    let qs = &block[8..24];
    for i in 0..32 {
        let byte = qs[i / 2];
        let low = if i % 2 == 0 { byte & 0x0F } else { byte >> 4 };
        let high_bit = ((qh >> i) & 1) as u8;
        let q5 = (low | (high_bit << 4)) as u32;
        out.push(q5 as f32 * d + m);
    }
}

fn decode_q8_0(block: &[u8], out: &mut Vec<f32>) {
    let d = f16::from_le_bytes([block[0], block[1]]).to_f32();
    for &byte in &block[2..34] {
        out.push(byte as i8 as f32 * d);
    }
}

/// Expand block Q8_1. The `s` field is not needed for dequantization: this
/// derived quantity (`d·Σqs`), useful only to the kernels of the scalar
/// works. We skip it, but check it in consistency tests.
fn decode_q8_1(block: &[u8], out: &mut Vec<f32>) {
    let d = f16::from_le_bytes([block[0], block[1]]).to_f32();
    for &byte in &block[4..36] {
        out.push(byte as i8 as f32 * d);
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// § 4 TESTS
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_q4_0_roundtrip_length_and_bytes() {
        let data: Vec<f32> = (0..32).map(|i| (i as f32 - 16.0) * 0.1).collect();
        let packed = quantize_gguf(&data, DType::Q4_0);
        assert_eq!(packed.bytes_len(), 18); // 1 block exactly
        let restored = packed.to_f32();
        assert_eq!(restored.len(), 32);
        // ggml formula: max (signed, at the maximum point |w|) = -1.6 (at i=0),
        // d = max / -8 = 0.2 → maximum rounding error d/2 = 0.1.
        // The threshold of 0.13 gives a small margin above this theoretical
        // maximum (plus f16 quantization of d itself contributes ~0.1% from above).
        for (a, b) in data.iter().zip(restored.iter()) {
            assert!((a - b).abs() < 0.13, "{a} vs {b}");
        }
    }

    #[test]
    fn test_q8_0_roundtrip_more_accurate_than_q4_0() {
        let data: Vec<f32> = (0..32).map(|i| (i as f32 - 16.0) * 0.1).collect();
        let q4 = quantize_gguf(&data, DType::Q4_0);
        let q8 = quantize_gguf(&data, DType::Q8_0);
        let err = |p: &GgufPacked| -> f32 {
            p.to_f32()
                .iter()
                .zip(data.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0, f32::max)
        };
        assert!(err(&q8) <= err(&q4), "q8={} q4={}", err(&q8), err(&q4));
        assert_eq!(q8.bytes_len(), 34);
    }

    #[test]
    fn test_q4_1_handles_skewed_data() {
        let data: Vec<f32> = (0..32).map(|i| 10.0 + i as f32 * 0.1).collect();
        let packed = quantize_gguf(&data, DType::Q4_1);
        let restored = packed.to_f32();
        for (a, b) in data.iter().zip(restored.iter()) {
            assert!((a - b).abs() < 0.2, "{a} vs {b}");
        }
    }

    #[test]
    fn test_q5_0_more_accurate_than_q4_0() {
        let data: Vec<f32> = (0..32).map(|i| (i as f32 - 16.0) * 0.1).collect();
        let q4 = quantize_gguf(&data, DType::Q4_0);
        let q5 = quantize_gguf(&data, DType::Q5_0);
        let err = |p: &GgufPacked| -> f32 {
            p.to_f32()
                .iter()
                .zip(data.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0, f32::max)
        };
        assert!(err(&q5) <= err(&q4), "q5={} q4={}", err(&q5), err(&q4));
    }

    #[test]
    fn test_q5_1_handles_skewed_data() {
        let data: Vec<f32> = (0..32).map(|i| 10.0 + i as f32 * 0.1).collect();
        let packed = quantize_gguf(&data, DType::Q5_1);
        let restored = packed.to_f32();
        for (a, b) in data.iter().zip(restored.iter()) {
            assert!((a - b).abs() < 0.1, "{a} vs {b}");
        }
    }

    #[test]
    fn test_partial_last_block_padded() {
        // 40 weights = 1 full block (32) + 1 incomplete (8) → 2 blocks, but
        // to_f32() should return exactly 40, not 64.
        let data: Vec<f32> = (0..40).map(|i| (i as f32 - 20.0) * 0.05).collect();
        let packed = quantize_gguf(&data, DType::Q4_0);
        assert_eq!(packed.bytes_len(), 36); // 2 blocks * 18 bytes
        let restored = packed.to_f32();
        assert_eq!(restored.len(), 40);
    }

    #[test]
    #[should_panic(expected = "block layout is not implemented yet")]
    fn test_k_quant_not_supported_yet() {
        let data = vec![1.0f32; 256];
        let _ = quantize_gguf(&data, DType::Q4_K);
    }

    #[test]
    fn test_is_supported_matches_implemented_set() {
        assert!(is_supported(DType::Q4_0));
        assert!(is_supported(DType::Q8_0));
        assert!(!is_supported(DType::Q4_K));
        assert!(!is_supported(DType::IQ4_NL));
    }

    // ─────────────────────────────────────────────────────────────────────
    // Tests against the ggml reference formula (see analysis neo-quant.txt):
    // d = max(signed, at the maximum point |w|) / -N,
    // rounding = trunc(x*id + N + 0.5).
    // ─────────────────────────────────────────────────────────────────────

    #[test]
    fn test_q4_0_matches_ggml_reference_codes() {
        // data = (i-16)*0.1, i in 0..32 → range [-1.6, 1.5].
        // max |w| = 1.6 at i=0, signed value = -1.6.
        // ggml: d = -1.6 / -8 = 0.2, id = 5.0.
        // Codes (standard, calculated manually using the formula trunc(w*id + 8.5)):
        let data: Vec<f32> = (0..32).map(|i| (i as f32 - 16.0) * 0.1).collect();
        let packed = quantize_gguf(&data, DType::Q4_0);
        let d = f16::from_le_bytes([packed.blocks[0], packed.blocks[1]]).to_f32();
        assert!(
            (d - 0.2).abs() < 1e-4,
            "d={d}, expected 0.2 (ggml max/-8, not amax/7)"
        );

        // The vector is computed in f32, not exact arithmetic: `(i-16)*0.1`
        // for i=3 gives -1.3000001 rather than -1.3, so `w*id + 8.5`
        // is 1.9999995 and trunc gives 1, not 2. ggml produces the same 1
        // on these inputs because it performs the same operations in f32 and
        // in the same order. Boundary cases of the form "exactly .0 after
        // multiplication" depend on the input representation; matching ggml is
        // the contract here. The test below (`..._away_from_boundaries`) checks
        // the same formula on data without such boundaries.
        let expected_codes: [u8; 32] = [
            0, 1, 1, 1, 2, 3, 3, 4, 4, 5, 5, 6, 6, 7, 7, 8, 8, 9, 9, 10, 10, 11, 11, 12, 12, 13,
            13, 14, 14, 15, 15, 15,
        ];
        let mut actual_codes = Vec::with_capacity(32);
        for &byte in &packed.blocks[2..18] {
            actual_codes.push(byte & 0x0F);
            actual_codes.push(byte >> 4);
        }
        assert_eq!(actual_codes, expected_codes);
    }

    #[test]
    fn test_q4_0_matches_ggml_reference_codes_away_from_boundaries() {
        // The same contract (d = max_signed / -8, code = trunc(w*id + 8.5)),
        // but the input is shifted so that `w*id + 8.5` nowhere falls close to
        // to the whole. Here the result is independent of the input representation error,
        // Therefore, the test pins exactly the FORMULA, and not successful rounding.
        let data: Vec<f32> = (0..32).map(|i| (i as f32 - 16.0) * 0.1 + 0.03).collect();
        let packed = quantize_gguf(&data, DType::Q4_0);
        let d = f16::from_le_bytes([packed.blocks[0], packed.blocks[1]]).to_f32();

        let mut max = 0.0f32;
        let mut amax = 0.0f32;
        for &w in &data {
            if w.abs() > amax {
                amax = w.abs();
                max = w;
            }
        }
        let d_expected = max / -8.0;
        assert!(
            (d - d_expected).abs() < 1e-3,
            "d={d}, expected {d_expected}"
        );

        let id = 1.0 / d;
        for (i, &w) in data.iter().enumerate() {
            let expected = (w * id + 8.5).trunc().clamp(0.0, 15.0) as u8;
            let byte = packed.blocks[2 + i / 2];
            let actual = if i % 2 == 0 { byte & 0x0F } else { byte >> 4 };
            assert_eq!(actual, expected, "the code went to position {i} (w={w})");
        }
    }

    // ── Q8_1 ────────────────────────────────────────────────────────────

    #[test]
    fn test_q8_1_block_layout_and_roundtrip() {
        let data: Vec<f32> = (0..64).map(|i| (i as f32 - 32.0) * 0.1).collect();
        let packed = quantize_gguf(&data, DType::Q8_1);

        assert_eq!(packed.dtype, DType::Q8_1);
        assert_eq!(packed.bytes_len(), 2 * DType::Q8_1.block_bytes());
        assert_eq!(packed.bytes_len(), 72, "2 blocks of 36 bytes");

        let restored = packed.to_f32();
        assert_eq!(restored.len(), 64);
        // 8 bits symmetrically: the error is no more than half the quantization step.
        for (a, b) in data.iter().zip(&restored) {
            let amax = 3.2f32;
            assert!((a - b).abs() <= amax / 127.0 / 2.0 + 1e-4, "{a} vs {b}");
        }
    }

    #[test]
    fn test_q8_1_sum_field_matches_definition() {
        // `s` must be equal to d·Σqs - the kernels of the scalar
        // products for Q4_1/Q5_1, and a divergence here would give an incorrect
        // the result of matrix multiplication with externally correct weights.
        let data: Vec<f32> = (0..32).map(|i| ((i * 37) % 53) as f32 - 26.0).collect();
        let packed = quantize_gguf(&data, DType::Q8_1);

        let d = f16::from_le_bytes([packed.blocks[0], packed.blocks[1]]).to_f32();
        let s = f16::from_le_bytes([packed.blocks[2], packed.blocks[3]]).to_f32();
        let sum: i32 = packed.blocks[4..36].iter().map(|&b| b as i8 as i32).sum();

        let expected = sum as f32 * d;
        assert!(
            (s - expected).abs() <= expected.abs() * 1e-2 + 1e-3,
            "s = {s}, expected d·Σqs = {expected}"
        );
    }

    #[test]
    fn test_q8_1_matches_q8_0_quantization_of_same_data() {
        // The codes for Q8_0 and Q8_1 are obtained using the same formula (amax/127,
        // round-to-nearest), the only difference is the presence of the `s` field. If the codes
        // If they diverge, it means one of the coders has deviated from the upstream.
        let data: Vec<f32> = (0..32).map(|i| (i as f32 - 16.0) * 0.13).collect();
        let q0 = quantize_gguf(&data, DType::Q8_0);
        let q1 = quantize_gguf(&data, DType::Q8_1);

        assert_eq!(
            &q0.blocks[2..34],
            &q1.blocks[4..36],
            "codes Q8_0 and Q8_1 must match on the same data"
        );
    }

    #[test]
    fn test_q8_1_is_supported_and_reaches_bit_budget() {
        assert!(is_supported(DType::Q8_1));
        // 36 bytes for 32 weights = 9 bits/weight, versus 8.5 for Q8_0: extra
        // half a bit is the `s` field.
        assert!((DType::Q8_1.bits_per_element() - 9.0).abs() < 1e-6);
        assert!((DType::Q8_0.bits_per_element() - 8.5).abs() < 1e-6);
    }

    #[test]
    fn test_q5_0_matches_ggml_reference_codes() {
        // Same data, N=16: d = -1.6 / -16 = 0.1, id = 10.0.
        // codes[i] = trunc(w_i * 10.0 + 16.5), clamp(0,31) — monotonic
        // 0..31 for this linear ramp.
        let data: Vec<f32> = (0..32).map(|i| (i as f32 - 16.0) * 0.1).collect();
        let packed = quantize_gguf(&data, DType::Q5_0);
        let d = f16::from_le_bytes([packed.blocks[0], packed.blocks[1]]).to_f32();
        assert!(
            (d - 0.1).abs() < 1e-4,
            "d={d}, expected 0.1 (ggml max/-16, not amax/15)"
        );

        let qh = u32::from_le_bytes([
            packed.blocks[2],
            packed.blocks[3],
            packed.blocks[4],
            packed.blocks[5],
        ]);
        let qs = &packed.blocks[6..22];
        let mut actual_codes = Vec::with_capacity(32);
        for i in 0..32 {
            let byte = qs[i / 2];
            let low = if i % 2 == 0 { byte & 0x0F } else { byte >> 4 };
            let high = ((qh >> i) & 1) as u8;
            actual_codes.push(low | (high << 4));
        }
        let expected_codes: Vec<u8> = (0..32).collect(); // 0..31 monotonic
        assert_eq!(actual_codes, expected_codes);
    }

    #[test]
    fn test_q4_0_half_integer_boundary_uses_trunc_not_round() {
        // Construct a block where w*id lands exactly on a k.5 boundary,
        // to distinguish trunc(x+8.5) from Rust's .round() (round-half-
        // away-from-zero). With d=1.0 (id=1.0), w=2.5 → x = 2.5+8.5=11.0
        // → trunc=11 (no ambiguity at the trunc step itself, since 11.0
        // is exact); the real divergence vs the old code is in `d`
        // (max/-8 vs amax/7) and the +8.5 trunc vs separate round()+8
        // for values whose scaled position sits at a genuine tie. This
        // test locks the block-level scale-selection behavior: the
        // block's own extremum must be used with its sign, not abs().
        let mut chunk = [0.0f32; 32];
        chunk[0] = -8.0; // most negative -> should become code 0 (min)
        chunk[1] = 7.0 * (8.0 / 8.0); // arbitrary interior value
        let d_expected = -8.0 / -8.0; // max=-8.0 (signed, most negative wins abs) / -8 = 1.0
        let mut out = Vec::new();
        encode_q4_0(&chunk, &mut out);
        let d = f16::from_le_bytes([out[0], out[1]]).to_f32();
        assert!(
            (d - d_expected).abs() < 1e-4,
            "d={d}, expected {d_expected}"
        );
        // code for w=-8.0: x = -8.0*1.0 + 8.5 = 0.5 -> trunc = 0
        let byte0 = out[2];
        assert_eq!(
            byte0 & 0x0F,
            0,
            "w=-8.0 should quantize to code 0 via trunc(0.5)=0"
        );
    }
}
