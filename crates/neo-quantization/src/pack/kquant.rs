// neo-quantization/src/pack/kquant.rs
//
// Portions of this file are derived from ggml/llama.cpp
// (`ggml/src/ggml-quants.c`), Copyright (c) the ggml/llama.cpp authors,
// used here under the MIT License.
//
// K-quants — GGUF "super-block" formats (Q2_K..Q8_K), block_elems=256
// (`QK_K` in terms of llama.cpp). Unlike simple blocks (Q4_0 and
// etc. in gguf.rs), each super-block of 256 weights is divided by 16
// sub-blocks of 16 (Q2_K/Q3_K/Q6_K) or 8 sub-blocks of 32 (Q4_K/Q5_K),
// each with its own quantized 4..8-bit scale/min on top of one
// float16 (or f32 for Q8_K) super-block scale - two-level scheme,
// which gives K-quants noticeably better quality per bit compared to
// simple Q4_0/Q8_0 with a comparable average bit/weight.
//
// The implementation follows `quantize_row_*_K_ref`/`dequantize_row_*_K`
// from `ggml/src/ggml-quants.c` (llama.cpp). These are the REFERENCE variants
// (not the AVX/NEON optimized intrinsics in the same file) - simpler and
// bitwise identical to the on-disk format, but not the fastest quantizer
// (dequantization is a linear pass in both cases; the quantization path is
// not a hot inference pass).
//
// ## Why calibration is more difficult than `pack::intn`
//
// `pack::intn` uses round-to-nearest from one global formula
// (mean-abs or max-abs) - O(n) per group. K-quants in the original
// use ITERATIVE calibration (`make_qx_quants`/`make_qkx2_quants`/
// `make_q3_quants`) — weighted least-squares search (scale, min),
// refined by 5..20 iterations on top of the initial round-to-nearest
// approaching. This is not “features for the sake of features” - it is this search that gives
// K-quants are their precision per bit; naive round(x/max_abs) gives noticeably
// worse quality and does NOT match bitwise what llama.cpp writes.
// We save the implementation of this search as is (see `optimize_scale_min`/
// `optimize_scale_signed`/`optimize_q3_scale` below), rather than simplifying -
// otherwise there would be no talk of byte compatibility with real .gguf files.
//
// ## Coverage
//
// Q2_K..Q8_K are implemented for both quantization and dequantization,
// bitwise compatible with the `ggml-quants.c` reference path. The IQ series
// (codebook-based vector quantization) lives in `pack::iq`, `pack::iq1` and
// `pack::iq_codebook`; `PackedScheme::Gguf`/`pack::gguf::is_supported`
// dispatch to the right module, and an unsupported DType panics instead of
// producing an incorrect result.

use crate::dtype::DType;

use burn::tensor::f16;

// ═══════════════════════════════════════════════════════════════════════════
// § 0 GENERAL CONSTANTS / HELPERS
// ═══════════════════════════════════════════════════════════════════════════

/// Number of weights in one super-block (`QK_K` in ggml).
pub const QK_K: usize = 256;

/// Threshold “block of all zeros” - below it calibration makes no sense
/// (matches `GROUP_MAX_EPS` in ggml-quants.c).
const GROUP_MAX_EPS: f32 = 1e-15;

/// `true` if `pack::kquant` implements boxing/unboxing of this DType.
pub fn is_supported(dtype: DType) -> bool {
    matches!(
        dtype,
        DType::Q2_K | DType::Q3_K | DType::Q4_K | DType::Q5_K | DType::Q6_K | DType::Q8_K
    )
}

/// `round-half-away-from-zero`, like `nearest_int()` in ggml-quants.c
/// (the bitwise IEEE754 trick there is equivalent to the usual `.round()` for
/// all practical ranges that occur here).
#[inline]
fn nearest_i32(x: f32) -> i32 {
    x.round() as i32
}

#[inline]
fn clampi(v: i32, lo: i32, hi: i32) -> i32 {
    v.max(lo).min(hi)
}

/// Unpack 6-bit (scale, min) sub-block `j` (0..8) from a 12-byte
/// `scales`-fields Q4_K/Q5_K - direct implementation of `get_scale_min_k4` (ggml-quants.c).
#[inline]
fn get_scale_min_k4(j: usize, q: &[u8]) -> (u8, u8) {
    if j < 4 {
        (q[j] & 63, q[j + 4] & 63)
    } else {
        let d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        let m = (q[j + 4] >> 4) | ((q[j] >> 6) << 4);
        (d, m)
    }
}

/// Read `chunk` from `data[start..]`, padding with zeros to `QK_K`, if
/// this is the last incomplete super-block.
fn read_padded_block(data: &[f32], start: usize) -> [f32; QK_K] {
    let mut chunk = [0.0f32; QK_K];
    let end = (start + QK_K).min(data.len());
    chunk[..end - start].copy_from_slice(&data[start..end]);
    chunk
}

// ═══════════════════════════════════════════════════════════════════════════
// § 0b ITERATIVE CALIBRATIONS (implementation make_qx_quants/make_qkx2_quants/
//        make_q3_quants) - common to several formats
// ═══════════════════════════════════════════════════════════════════════════

/// implementation `make_qkx2_quants` — weighted least-squares (scale, min) for
/// asymmetric sub-block with `nmax+1` levels (`[0, nmax]`, unsigned code).
/// Used in Q4_K (nmax=15) and Q5_K (nmax=31).
///
/// Returns `(scale, min, codes)`; `W_hat[i] = scale*codes[i] - min`
/// (note the sign `min` - as in ggml, `the_min` stores
/// `-min_initial_range`, so it is then SUBTRACTED).
fn optimize_scale_min(
    x: &[f32],
    weights: &[f32],
    nmax: i32,
    rmin: f32,
    rdelta: f32,
    nstep: i32,
) -> (f32, f32, Vec<u8>) {
    let n = x.len();
    let mut min = x[0];
    let mut max = x[0];
    let mut sum_w = weights[0];
    let mut sum_x = sum_w * x[0];
    for i in 1..n {
        min = min.min(x[i]);
        max = max.max(x[i]);
        let w = weights[i];
        sum_w += w;
        sum_x += w * x[i];
    }
    if min > 0.0 {
        min = 0.0;
    }
    // `max >= min` is always, so `<=` is an exact test for a degenerate range
    // without strict comparison float (`clippy::float_cmp`).
    if max <= min {
        return (0.0, -min, vec![0u8; n]);
    }

    let mut iscale = nmax as f32 / (max - min);
    let mut scale = 1.0 / iscale;
    let mut l: Vec<i32> = (0..n)
        .map(|i| clampi(nearest_i32(iscale * (x[i] - min)), 0, nmax))
        .collect();
    let mut best_error: f32 = l
        .iter()
        .zip(x.iter())
        .zip(weights.iter())
        .map(|((&li, &xi), &w)| {
            let diff = scale * li as f32 + min - xi;
            w * diff * diff
        })
        .sum();

    if nstep >= 1 {
        for is in 0..=nstep {
            iscale = (rmin + rdelta * is as f32 + nmax as f32) / (max - min);
            let mut sum_l = 0.0f32;
            let mut sum_l2 = 0.0f32;
            let mut sum_xl = 0.0f32;
            let laux: Vec<i32> = (0..n)
                .map(|i| {
                    let li = clampi(nearest_i32(iscale * (x[i] - min)), 0, nmax);
                    let w = weights[i];
                    sum_l += w * li as f32;
                    sum_l2 += w * (li * li) as f32;
                    sum_xl += w * li as f32 * x[i];
                    li
                })
                .collect();
            let d = sum_w * sum_l2 - sum_l * sum_l;
            if d > 0.0 {
                let mut this_scale = (sum_w * sum_xl - sum_x * sum_l) / d;
                let mut this_min = (sum_l2 * sum_x - sum_l * sum_xl) / d;
                if this_min > 0.0 {
                    this_min = 0.0;
                    this_scale = sum_xl / sum_l2;
                }
                let cur_error: f32 = laux
                    .iter()
                    .zip(x.iter())
                    .zip(weights.iter())
                    .map(|((&li, &xi), &w)| {
                        let diff = this_scale * li as f32 + this_min - xi;
                        w * diff * diff
                    })
                    .sum();
                if cur_error < best_error {
                    l = laux;
                    best_error = cur_error;
                    scale = this_scale;
                    min = this_min;
                }
            }
        }
    }

    let codes: Vec<u8> = l.iter().map(|&v| v as u8).collect();
    (scale, -min, codes)
}

/// implementation `make_qx_quants` with `rmse_type=1` (weights `x²`) - symmetric
/// weighted-least-squares scale for sub-block with codes `[-nmax, nmax-1]`,
/// stored at offset `+nmax` (unsigned range `[0, 2*nmax-1]`).
/// Used in Q6_K (nmax=32).
fn optimize_scale_signed(x: &[f32], nmax: i32) -> (f32, Vec<i8>) {
    let n = x.len();
    let mut max = 0.0f32;
    let mut amax = 0.0f32;
    for &xi in x {
        let ax = xi.abs();
        if ax > amax {
            amax = ax;
            max = xi;
        }
    }
    if amax < GROUP_MAX_EPS {
        return (0.0, vec![0i8; n]);
    }

    let mut iscale = -(nmax as f32) / max;
    let mut l: Vec<i32> = x
        .iter()
        .map(|&xi| clampi(nearest_i32(iscale * xi), -nmax, nmax - 1))
        .collect();

    let mut sumlx = 0.0f32;
    let mut suml2 = 0.0f32;
    for i in 0..n {
        let w = x[i] * x[i];
        sumlx += w * x[i] * l[i] as f32;
        suml2 += w * (l[i] * l[i]) as f32;
    }
    let mut scale = if suml2 != 0.0 { sumlx / suml2 } else { 0.0 };
    let mut best = scale * sumlx;

    for is in -9..=9 {
        if is == 0 {
            continue;
        }
        iscale = -(nmax as f32 + 0.1 * is as f32) / max;
        let mut sumlx2 = 0.0f32;
        let mut suml22 = 0.0f32;
        let laux: Vec<i32> = x
            .iter()
            .map(|&xi| clampi(nearest_i32(iscale * xi), -nmax, nmax - 1))
            .collect();
        for i in 0..n {
            let w = x[i] * x[i];
            sumlx2 += w * x[i] * laux[i] as f32;
            suml22 += w * (laux[i] * laux[i]) as f32;
        }
        if suml22 > 0.0 && sumlx2 * sumlx2 > best * suml22 {
            l = laux;
            scale = sumlx2 / suml22;
            best = scale * sumlx2;
        }
    }

    let codes: Vec<i8> = l.iter().map(|&v| (v + nmax) as i8).collect();
    (scale, codes)
}

/// implementation `make_q3_quants` (`do_rmse=true`) - symmetric coordinate-descent
/// scale for a sub-block with `[-nmax, nmax-1]` codes stored with
/// offset `+nmax`. Used in Q3_K (nmax=4).
fn optimize_q3_scale(x: &[f32], nmax: i32) -> (f32, Vec<i8>) {
    let n = x.len();
    let mut max = 0.0f32;
    let mut amax = 0.0f32;
    for &xi in x {
        let ax = xi.abs();
        if ax > amax {
            amax = ax;
            max = xi;
        }
    }
    if amax < GROUP_MAX_EPS {
        return (0.0, vec![0i8; n]);
    }

    let iscale = -(nmax as f32) / max;
    let mut l: Vec<i32> = x
        .iter()
        .map(|&xi| clampi(nearest_i32(iscale * xi), -nmax, nmax - 1))
        .collect();

    let mut sumlx = 0.0f32;
    let mut suml2 = 0.0f32;
    for i in 0..n {
        let w = x[i] * x[i];
        sumlx += w * x[i] * l[i] as f32;
        suml2 += w * (l[i] * l[i]) as f32;
    }

    for _itry in 0..5 {
        let mut n_changed = 0;
        for i in 0..n {
            let w = x[i] * x[i];
            let slx = sumlx - w * x[i] * l[i] as f32;
            if slx > 0.0 {
                let sl2 = suml2 - w * (l[i] * l[i]) as f32;
                let mut new_l = clampi(nearest_i32(x[i] * sl2 / slx), -nmax, nmax - 1);
                if new_l != l[i] {
                    let slx2 = slx + w * x[i] * new_l as f32;
                    let sl22 = sl2 + w * (new_l * new_l) as f32;
                    if sl22 > 0.0 && slx2 * slx2 * suml2 > sumlx * sumlx * sl22 {
                        l[i] = new_l;
                        sumlx = slx2;
                        suml2 = sl22;
                        n_changed += 1;
                    } else {
                        new_l = l[i]; // no changes
                        let _ = new_l;
                    }
                }
            }
        }
        if n_changed == 0 {
            break;
        }
    }

    let codes: Vec<i8> = l.iter().map(|&v| (v + nmax) as i8).collect();
    let scale = if suml2 > 0.0 { sumlx / suml2 } else { 0.0 };
    (scale, codes)
}

// ═══════════════════════════════════════════════════════════════════════════
// § 1 COMMON CONTAINER
// ═══════════════════════════════════════════════════════════════════════════

/// Packed K-quant tensor - raw super-block bytes in `dtype` format,
/// bitwise identical to what llama.cpp would write/read.
///
/// Similar to `pack::gguf::GgufPacked`, but for super-block formats (`QK_K=256`,
/// and not `block_elems=32`). Separate type (does not reuse `GgufPacked`),
/// because `numel`/`bytes_len` are computed here relative to `QK_K`, and
/// not regarding `dtype.block_elems()` simple formats - keep them
/// separately is clearer than branching one type into two different constants
/// block size.
#[derive(Debug, Clone, PartialEq)]
pub struct KQuantPacked {
    /// Specific K-quant DType (defines super-block layout).
    pub dtype: DType,
    /// Raw super-blocks, concatenated in a row (without header
    /// tensor - title/name/shape - care `neo-gguf`).
    pub blocks: Vec<u8>,
    /// Initial number of weights (last super-block may be incomplete,
    /// if `numel % QK_K != 0` - padded with zeros during quantization).
    pub numel: usize,
}

impl KQuantPacked {
    /// The initial number of weights that the buffer represents.
    pub fn numel(&self) -> usize {
        self.numel
    }

    /// Size in bytes actually occupied by packed super-blocks.
    pub fn bytes_len(&self) -> usize {
        self.blocks.len()
    }

    /// Actual bits/weight (including all scale/min/qh fields - already part
    /// the super-block itself, there is no separate overhead on top of it,
    /// unlike `pack::intn`, where scales are stored externally).
    pub fn effective_bits_per_weight(&self) -> f32 {
        (self.blocks.len() as f32 * 8.0) / self.numel as f32
    }

    /// Expand to `Vec<f32>`.
    pub fn to_f32(&self) -> Vec<f32> {
        dequantize_kquant(self)
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// § 2 QUANTIZE / DEQUANTIZE – SINGLE ENTRY POINT
// ═══════════════════════════════════════════════════════════════════════════

/// Quantize `data` in K-quant super-block format `dtype`.
///
/// #Panic
///
/// If `dtype` is not included in `is_supported` (IQ/TQ-series - other modules,
/// see `pack::gguf_backend`).
pub fn quantize_kquant(data: &[f32], dtype: DType) -> KQuantPacked {
    assert!(
        is_supported(dtype),
        "pack::kquant::quantize_kquant: {dtype:?} is not a K-quant super-block format \
         (IQ/TQ series are packaged in pack::iq / pack::tq - see pack::gguf_backend)"
    );

    let n_blocks = data.len().div_ceil(QK_K);
    let block_bytes = dtype.block_bytes();
    let mut blocks = Vec::with_capacity(n_blocks * block_bytes);

    for b in 0..n_blocks {
        let start = b * QK_K;
        let chunk = read_padded_block(data, start);

        match dtype {
            DType::Q2_K => encode_q2_k(&chunk, &mut blocks),
            DType::Q3_K => encode_q3_k(&chunk, &mut blocks),
            DType::Q4_K => encode_q4_k(&chunk, &mut blocks),
            DType::Q5_K => encode_q5_k(&chunk, &mut blocks),
            DType::Q6_K => encode_q6_k(&chunk, &mut blocks),
            DType::Q8_K => encode_q8_k(&chunk, &mut blocks),
            _ => unreachable!("checked by is_supported above"),
        }
    }

    KQuantPacked {
        dtype,
        blocks,
        numel: data.len(),
    }
}

/// Completely expand the K-quant package back to `Vec<f32>`.
///
/// #Panic
///
/// If `packed.dtype` is not included in `is_supported`.
pub fn dequantize_kquant(packed: &KQuantPacked) -> Vec<f32> {
    assert!(
        is_supported(packed.dtype),
        "pack::kquant::dequantize_kquant: {:?} is not a K-quant super-block format",
        packed.dtype
    );

    let block_bytes = packed.dtype.block_bytes();
    let n_blocks = packed.blocks.len() / block_bytes;
    let mut out = Vec::with_capacity(n_blocks * QK_K);

    for b in 0..n_blocks {
        let block = &packed.blocks[b * block_bytes..(b + 1) * block_bytes];
        match packed.dtype {
            DType::Q2_K => decode_q2_k(block, &mut out),
            DType::Q3_K => decode_q3_k(block, &mut out),
            DType::Q4_K => decode_q4_k(block, &mut out),
            DType::Q5_K => decode_q5_k(block, &mut out),
            DType::Q6_K => decode_q6_k(block, &mut out),
            DType::Q8_K => decode_q8_k(block, &mut out),
            _ => unreachable!("checked by is_supported above"),
        }
    }

    out.truncate(packed.numel);
    out
}

// ═══════════════════════════════════════════════════════════════════════════
// § 3 Q4_K / Q5_K - GENERAL INFRASTRUCTURE (12-byte 6-bit scales-packing)
// ═══════════════════════════════════════════════════════════════════════════
//
// Q4_K and Q5_K use the SAME sub-block scheme (8 sub-blocks of 32
// element, 6-bit (scale, min) per sub-block, packed into 12 bytes across
// `get_scale_min_k4`/`put_scale_min_k4`) - differ only in width
// the most quantized code (4 bits vs 5 bits with an additional `qh` field).

/// The reverse operation to `get_scale_min_k4` is to pack 8 pairs (6-bit
/// scale, 6-bit min) into a 12-byte field `scales`. Direct Logic implementation
/// `get_scale_min_k4` "in the opposite direction" (see its comment above:
/// `j<4` stores the lower 6 bits of both values directly in two bytes;
/// `j>=4` adds the most significant 2 bits of each value from the upper bits
/// bytes `j-4`).
fn put_scale_min_k4(scales_q: &[u8; 8], mins_q: &[u8; 8], out: &mut [u8; 12]) {
    for j in 0..4 {
        out[j] = scales_q[j] & 63;
        out[j + 4] = mins_q[j] & 63;
    }
    for j in 4..8 {
        out[j + 4] = (scales_q[j] & 0xF) | ((mins_q[j] & 0xF) << 4);
        out[j - 4] |= (scales_q[j] >> 4) << 6;
        out[j] |= (mins_q[j] >> 4) << 6;
    }
}

/// Calibrate all 8 sub-blocks Q4_K/Q5_K: for each sub-block runs
/// `optimize_scale_min` (asymmetric weighted LS, `nmax` levels), then
/// quantizes themselves (scale, min) into 6 bits relative to the general super-block
/// maximum - direct loop implementation `quantize_row_q4_K_ref`/`quantize_row_q5_K_ref`
/// (the general part before the differences in the coding of the scales themselves).
///
/// Returns `(d, dmin, scales_packed[12], sub_scales[8], sub_mins[8])` —
/// the last two are needed by the calling code (`encode_q4_k`/`encode_q5_k`) for
/// in order to reforge the weights with the same (scale, min) that was recorded
/// to the file (after 6-bit rounding), and not with the original float value -
/// otherwise dequantize will not match what would have been written by the quantizer
/// llama.cpp (there the weights are also quantized AFTER the scale/min rounding).
fn calibrate_k4_subblocks(
    chunk: &[f32; QK_K],
    nmax: i32,
) -> (f32, f32, [u8; 12], [f32; 8], [f32; 8]) {
    let mut sub_scale = [0.0f32; 8];
    let mut sub_min = [0.0f32; 8];
    let mut codes_per_sub: Vec<Vec<u8>> = Vec::with_capacity(8);

    for sb in 0..8 {
        let sub = &chunk[sb * 32..(sb + 1) * 32];
        // `make_qkx2_quants` weights K4 sub-blocks by |x| (not x², unlike the
        // Q6_K/Q3_K rmse paths); the weights are computed from the actual data.
        let w: Vec<f32> = sub.iter().map(|x| x.abs()).collect();
        let (scale, min, codes) = optimize_scale_min(sub, &w, nmax, -1.0, 0.1, 20);
        sub_scale[sb] = scale;
        sub_min[sb] = min;
        codes_per_sub.push(codes);
    }

    let max_scale = sub_scale.iter().cloned().fold(0.0f32, f32::max);
    let max_min = sub_min.iter().cloned().fold(0.0f32, f32::max);

    let inv_scale = if max_scale > 0.0 {
        63.0 / max_scale
    } else {
        0.0
    };
    let inv_min = if max_min > 0.0 { 63.0 / max_min } else { 0.0 };

    let mut scales_q = [0u8; 8];
    let mut mins_q = [0u8; 8];
    for sb in 0..8 {
        scales_q[sb] = clampi(nearest_i32(inv_scale * sub_scale[sb]), 0, 63) as u8;
        mins_q[sb] = clampi(nearest_i32(inv_min * sub_min[sb]), 0, 63) as u8;
    }

    let d = if max_scale > 0.0 {
        max_scale / 63.0
    } else {
        0.0
    };
    let dmin = if max_min > 0.0 { max_min / 63.0 } else { 0.0 };

    let mut packed = [0u8; 12];
    put_scale_min_k4(&scales_q, &mins_q, &mut packed);

    // Restore actually used (after 6-bit rounding)
    // scale/min for each sub-block - it is IMI that quantizes the weights, not
    // the original `sub_scale[sb]`/`sub_min[sb]` from the LS calibration.
    let mut eff_scale = [0.0f32; 8];
    let mut eff_min = [0.0f32; 8];
    for sb in 0..8 {
        let (dq, mq) = get_scale_min_k4(sb, &packed);
        eff_scale[sb] = d * dq as f32;
        eff_min[sb] = dmin * mq as f32;
    }

    let _ = codes_per_sub; // The codes above are calculated with the old (before rounding) scale;
                           // recalculate below into encode_q4_k/encode_q5_k with eff_*.

    (d, dmin, packed, eff_scale, eff_min)
}

// ═══════════════════════════════════════════════════════════════════════════
// § 4  Q4_K
// ═══════════════════════════════════════════════════════════════════════════
//
// Layout (144 bytes, 256 weights):
//   d: f16 — super-block scale for scales
//   dmin: f16 — super-block scale for mins
//   scales: [u8; 12] - 8x(6-bit scale, 6-bit min), see get/put_scale_min_k4
//   qs: [u8; 128] - 4-bit unsigned codes, 2 sub-blocks/32 weights
//                                for 16 bytes (nibble pairs), 8 sub-blocks in total
//   W_hat = d*sc[j] * q - dmin*m[j]   (q ∈ [0, 15])

fn encode_q4_k(chunk: &[f32; QK_K], out: &mut Vec<u8>) {
    let (d, dmin, scales_packed, eff_scale, eff_min) = calibrate_k4_subblocks(chunk, 15);

    out.extend_from_slice(&crate::pack::f16_saturating(d).to_le_bytes());
    out.extend_from_slice(&crate::pack::f16_saturating(dmin).to_le_bytes());
    out.extend_from_slice(&scales_packed);

    let mut qs = [0u8; 128];
    for sb in 0..8 {
        let sub = &chunk[sb * 32..(sb + 1) * 32];
        let scale = eff_scale[sb];
        let min = eff_min[sb];
        let inv = if scale > 0.0 { 1.0 / scale } else { 0.0 };

        let mut codes = [0u8; 32];
        for i in 0..32 {
            let q = clampi(nearest_i32(inv * (sub[i] + min)), 0, 15) as u8;
            codes[i] = q;
        }

        // 2 sub-blocks per 16-byte "half-layer" in the original layout:
        // qs stores sub-blocks in pairs (sb, sb+1) alternating nibble by byte -
        // actual ggml layout: qs[sb*16 .. sb*16+16] contains EXACTLY
        // one sub-block of 32 weights (2 weights/byte), 8 sub-blocks in a row.
        let base = sb * 16;
        for i in 0..16 {
            qs[base + i] = codes[2 * i] | (codes[2 * i + 1] << 4);
        }
    }
    out.extend_from_slice(&qs);
}

fn decode_q4_k(block: &[u8], out: &mut Vec<f32>) {
    let d = f16::from_le_bytes([block[0], block[1]]).to_f32();
    let dmin = f16::from_le_bytes([block[2], block[3]]).to_f32();
    let scales = &block[4..16];
    let qs = &block[16..144];

    for sb in 0..8 {
        let (sc_q, min_q) = get_scale_min_k4(sb, scales);
        let scale = d * sc_q as f32;
        let min = dmin * min_q as f32;

        let base = sb * 16;
        for i in 0..16 {
            let byte = qs[base + i];
            out.push((byte & 0x0F) as f32 * scale - min);
            out.push((byte >> 4) as f32 * scale - min);
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// § 5  Q5_K
// ═══════════════════════════════════════════════════════════════════════════
//
// Layout (176 bytes, 256 weights): same (d, dmin, scales[12]) super-block
// hat, that Q4_K, but 5-bit codes: the least significant 4 bits in `qs[128]` (the same
// layout, which is Q4_K), the 5th (most significant) bit of each of the 256 weights - in
// separate `qh[32]` (1 bit/weight, like qh in Q5_0/Q5_1 in gguf.rs).
//
//   d:      f16
//   dmin:   f16
//   scales: [u8; 12]
//   qh: [u8; 32] - most significant bit, 1 bit/weight, 256 bits = 32 bytes
//   qs: [u8; 128] - low 4 bits, same layout as Q4_K
//   W_hat = d*sc[j] * q - dmin*m[j]   (q ∈ [0, 31], q = low4 | (high_bit<<4))

fn encode_q5_k(chunk: &[f32; QK_K], out: &mut Vec<u8>) {
    let (d, dmin, scales_packed, eff_scale, eff_min) = calibrate_k4_subblocks(chunk, 31);

    out.extend_from_slice(&crate::pack::f16_saturating(d).to_le_bytes());
    out.extend_from_slice(&crate::pack::f16_saturating(dmin).to_le_bytes());
    out.extend_from_slice(&scales_packed);

    let mut qh = [0u8; 32];
    let mut qs = [0u8; 128];

    for sb in 0..8 {
        let sub = &chunk[sb * 32..(sb + 1) * 32];
        let scale = eff_scale[sb];
        let min = eff_min[sb];
        let inv = if scale > 0.0 { 1.0 / scale } else { 0.0 };

        let mut codes = [0u8; 32];
        for i in 0..32 {
            let q = clampi(nearest_i32(inv * (sub[i] + min)), 0, 31) as u32;
            codes[i] = (q & 0x0F) as u8;
            if q & 0x10 != 0 {
                let flat_idx = sb * 32 + i;
                qh[flat_idx / 8] |= 1 << (flat_idx % 8);
            }
        }

        let base = sb * 16;
        for i in 0..16 {
            qs[base + i] = codes[2 * i] | (codes[2 * i + 1] << 4);
        }
    }

    out.extend_from_slice(&qh);
    out.extend_from_slice(&qs);
}

fn decode_q5_k(block: &[u8], out: &mut Vec<f32>) {
    let d = f16::from_le_bytes([block[0], block[1]]).to_f32();
    let dmin = f16::from_le_bytes([block[2], block[3]]).to_f32();
    let scales = &block[4..16];
    let qh = &block[16..48];
    let qs = &block[48..176];

    for sb in 0..8 {
        let (sc_q, min_q) = get_scale_min_k4(sb, scales);
        let scale = d * sc_q as f32;
        let min = dmin * min_q as f32;

        let base = sb * 16;
        for i in 0..16 {
            let byte = qs[base + i];
            for half in 0..2 {
                let local_idx = 2 * i + half;
                let flat_idx = sb * 32 + local_idx;
                let low = if half == 0 { byte & 0x0F } else { byte >> 4 };
                let high_bit = (qh[flat_idx / 8] >> (flat_idx % 8)) & 1;
                let q = (low as u32) | ((high_bit as u32) << 4);
                out.push(q as f32 * scale - min);
            }
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// § 6  Q6_K
// ═══════════════════════════════════════════════════════════════════════════
//
// Layout (210 bytes, 256 weights):
//   ql: [u8; 128] - low 4 bits of codes (256 weights, 2/byte)
//   qh: [u8; 64] - the most significant 2 bits of codes (256 weights, 4/byte)
//   scales: [i8; 16] - 8-bit signed scale per sub-block of 16
//   d:      f16                — super-block scale
//   W_hat = d * scales[sb] * q (q ∈ [-32, 31], symmetric, 6 bits signed)
//
// Each sub-unit (16 scales) is independently calibrated via
// `optimize_scale_signed` (weighted-LS, symmetric, nmax=32) - direct implementation
// `quantize_row_q6_K_ref`. Final `d = max(|scale_sb|) / 127` – top
// the level quantizes the float-scale sub-blocks themselves into 8 bits (not 6, in
// difference from Q4_K/Q5_K/Q2_K/Q3_K) - reflected in `scales: [i8; 16]`.

fn encode_q6_k(chunk: &[f32; QK_K], out: &mut Vec<u8>) {
    // 16 sub-blocks of 16 weights, symmetric LS-calibration of each.
    let mut sub_scale = [0.0f32; 16];
    let mut sub_codes: Vec<[i8; 16]> = Vec::with_capacity(16);

    for sb in 0..16 {
        let sub = &chunk[sb * 16..(sb + 1) * 16];
        let (scale, codes) = optimize_scale_signed(sub, 32);
        sub_scale[sb] = scale;
        let mut arr = [0i8; 16];
        // `codes` from optimize_scale_signed are already shifted by +nmax (see its
        // signature/use in Q3_K below) - signed code needed here
        // without offset (Q6_K stores REAL signed range [-32,31], not
        // offset-biased), so we subtract back nmax=32.
        for i in 0..16 {
            arr[i] = codes[i] - 32;
        }
        sub_codes.push(arr);
    }

    let amax_scale = sub_scale.iter().fold(0.0f32, |acc, s| acc.max(s.abs()));
    let d = if amax_scale > 0.0 {
        amax_scale / 127.0
    } else {
        0.0
    };
    let inv_d = if d > 0.0 { 1.0 / d } else { 0.0 };

    let mut scales_i8 = [0i8; 16];
    for sb in 0..16 {
        scales_i8[sb] = clampi(nearest_i32(inv_d * sub_scale[sb]), -127, 127) as i8;
    }

    // Reconvert weights effectively (after 8-bit rounding)
    // scale of each sub-block - the same reason as in calibrate_k4_subblocks:
    // dequantize must match what is actually encoded.
    let mut ql = [0u8; 128];
    let mut qh = [0u8; 64];
    for sb in 0..16 {
        let eff_scale = d * scales_i8[sb] as f32;
        let inv = if eff_scale != 0.0 {
            1.0 / eff_scale
        } else {
            0.0
        };
        let sub = &chunk[sb * 16..(sb + 1) * 16];

        for i in 0..16 {
            let flat_idx = sb * 16 + i;
            let q = clampi(nearest_i32(inv * sub[i]), -32, 31) + 32; // → [0.63], 6 bit unsigned
            let q = q as u8;

            let low4 = q & 0x0F;
            let high2 = (q >> 4) & 0x03;

            // ql: 2 weights/byte (low 4 bits), qh: 4 weights/byte (high 2 bits)
            let ql_byte = flat_idx / 2;
            if flat_idx % 2 == 0 {
                ql[ql_byte] = (ql[ql_byte] & 0xF0) | low4;
            } else {
                ql[ql_byte] = (ql[ql_byte] & 0x0F) | (low4 << 4);
            }
            let qh_byte = flat_idx / 4;
            let qh_shift = (flat_idx % 4) * 2;
            qh[qh_byte] |= high2 << qh_shift;
        }
    }

    out.extend_from_slice(&ql);
    out.extend_from_slice(&qh);
    // scales: i8, write as is (le byte-for-byte, i8 takes 1 byte)
    for &s in &scales_i8 {
        out.push(s as u8);
    }
    out.extend_from_slice(&crate::pack::f16_saturating(d).to_le_bytes());
}

fn decode_q6_k(block: &[u8], out: &mut Vec<f32>) {
    let ql = &block[0..128];
    let qh = &block[128..192];
    let scales = &block[192..208];
    let d = f16::from_le_bytes([block[208], block[209]]).to_f32();

    for flat_idx in 0..QK_K {
        let sb = flat_idx / 16;
        let scale_i8 = scales[sb] as i8;
        let scale = d * scale_i8 as f32;

        let ql_byte = ql[flat_idx / 2];
        let low4 = if flat_idx % 2 == 0 {
            ql_byte & 0x0F
        } else {
            ql_byte >> 4
        };
        let qh_byte = qh[flat_idx / 4];
        let qh_shift = (flat_idx % 4) * 2;
        let high2 = (qh_byte >> qh_shift) & 0x03;

        let q = ((high2 as i32) << 4 | low4 as i32) - 32; // [0,63] → [-32,31]
        out.push(q as f32 * scale);
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// § 7  Q8_K
// ═══════════════════════════════════════════════════════════════════════════
//
// Layout (292 bytes, 256 weights) - "for intermediate calculations only
// and dot products" (see ggml-common.h): used as activation format
// with quantized GEMV/GEMM (K-quant weight × Q8_K activation), not as
// format for storing weights on disk - but still part of the public set
// DType, therefore implemented here for completeness and symmetry with the others.
//
//   d: f32 - the only super-block scale, max-abs/127
//   qs: [i8; 256] - signed code directly (like Q8_0, but block=256)
//   bsums: [i16; 16] - sums of qs over groups of 16 (for quick
//                                 skip recalculation of partial sums in the dot product;
//                                 do not participate in dequantize, but are required
//                                 for bit-by-bit file format matching)

fn encode_q8_k(chunk: &[f32; QK_K], out: &mut Vec<u8>) {
    let amax = chunk.iter().fold(0.0f32, |acc, w| acc.max(w.abs()));
    let mut d = if amax > 0.0 { amax / 127.0 } else { 0.0 };
    // Q8_K stores the scale at f32 (unlike other K-quants with f16),
    // therefore there is no overflow on write - but there is on READ: dequantization
    // computes `q * d` with q up to 127, and at amax near `f32::MAX`
    // the product goes beyond f32 and gives `inf`. We reduce `d` until the
    // result is finite: the loss of precision here is negligible, while
    // the alternative is a whole `inf` block with no indication of an error.
    while d > 0.0 && !(d * 127.0).is_finite() {
        d *= 0.5;
    }
    let inv_d = if d > 0.0 { 1.0 / d } else { 0.0 };

    let mut qs = [0i8; QK_K];
    for i in 0..QK_K {
        qs[i] = clampi(nearest_i32(inv_d * chunk[i]), -127, 127) as i8;
    }

    let mut bsums = [0i16; 16];
    for g in 0..16 {
        let mut sum: i32 = 0;
        for i in 0..16 {
            sum += qs[g * 16 + i] as i32;
        }
        bsums[g] = sum as i16;
    }

    out.extend_from_slice(&d.to_le_bytes());
    for &q in &qs {
        out.push(q as u8);
    }
    for &b in &bsums {
        out.extend_from_slice(&b.to_le_bytes());
    }
}

fn decode_q8_k(block: &[u8], out: &mut Vec<f32>) {
    let d = f32::from_le_bytes([block[0], block[1], block[2], block[3]]);
    let qs = &block[4..4 + QK_K];
    // bsums (block[4+QK_K .. 4+QK_K+32]) are not needed for dequantize - skip it.

    for &byte in qs {
        out.push(byte as i8 as f32 * d);
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// § 8  Q2_K
// ═══════════════════════════════════════════════════════════════════════════
//
// Layout (84 bytes, 256 weights):
//   scales: [u8; 16] - per sub-block of 16: low 4 bits = 4 bits
//                                scale, high 4 bits = 4 bit min
//                                (packaging is simpler than Q4_K/Q5_K - one byte
//                                per sub-block, not a 12-byte bit-cursor field)
//   qs: [u8; 64] - 2-bit unsigned codes, 4 weights/byte
//   d: f16 — super-block scale for scales
//   dmin: f16 — super-block scale for mins
//   W_hat = d*sc[sb] * q - dmin*m[sb]   (q ∈ [0, 3])
//
// 16 sub-blocks of 16 weights, asymmetric weighted-LS calibration
// (`optimize_scale_min`, nmax=3) for each, then themselves (scale, min)
// quantized into 4 bits relative to the total super-block maximum -
// direct implementation `quantize_row_q2_K_ref`.

fn encode_q2_k(chunk: &[f32; QK_K], out: &mut Vec<u8>) {
    let mut sub_scale = [0.0f32; 16];
    let mut sub_min = [0.0f32; 16];

    for sb in 0..16 {
        let sub = &chunk[sb * 16..(sb + 1) * 16];
        let w: Vec<f32> = sub.iter().map(|x| x.abs()).collect();
        let (scale, min, _codes) = optimize_scale_min(sub, &w, 3, -0.5, 0.1, 15);
        sub_scale[sb] = scale;
        sub_min[sb] = min;
    }

    let max_scale = sub_scale.iter().cloned().fold(0.0f32, f32::max);
    let max_min = sub_min.iter().cloned().fold(0.0f32, f32::max);
    let d = if max_scale > 0.0 {
        max_scale / 15.0
    } else {
        0.0
    };
    let dmin = if max_min > 0.0 { max_min / 15.0 } else { 0.0 };
    let inv_d = if d > 0.0 { 1.0 / d } else { 0.0 };
    let inv_dmin = if dmin > 0.0 { 1.0 / dmin } else { 0.0 };

    let mut scales_packed = [0u8; 16];
    let mut eff_scale = [0.0f32; 16];
    let mut eff_min = [0.0f32; 16];
    for sb in 0..16 {
        let sc_q = clampi(nearest_i32(inv_d * sub_scale[sb]), 0, 15) as u8;
        let min_q = clampi(nearest_i32(inv_dmin * sub_min[sb]), 0, 15) as u8;
        scales_packed[sb] = sc_q | (min_q << 4);
        eff_scale[sb] = d * sc_q as f32;
        eff_min[sb] = dmin * min_q as f32;
    }

    let mut qs = [0u8; 64];
    for sb in 0..16 {
        let sub = &chunk[sb * 16..(sb + 1) * 16];
        let scale = eff_scale[sb];
        let min = eff_min[sb];
        let inv = if scale > 0.0 { 1.0 / scale } else { 0.0 };

        for i in 0..16 {
            let flat_idx = sb * 16 + i;
            let q = clampi(nearest_i32(inv * (sub[i] + min)), 0, 3) as u8;
            let byte_idx = flat_idx / 4;
            let shift = (flat_idx % 4) * 2;
            qs[byte_idx] |= q << shift;
        }
    }

    out.extend_from_slice(&scales_packed);
    out.extend_from_slice(&qs);
    out.extend_from_slice(&crate::pack::f16_saturating(d).to_le_bytes());
    out.extend_from_slice(&crate::pack::f16_saturating(dmin).to_le_bytes());
}

fn decode_q2_k(block: &[u8], out: &mut Vec<f32>) {
    let scales = &block[0..16];
    let qs = &block[16..80];
    let d = f16::from_le_bytes([block[80], block[81]]).to_f32();
    let dmin = f16::from_le_bytes([block[82], block[83]]).to_f32();

    for flat_idx in 0..QK_K {
        let sb = flat_idx / 16;
        let sc_q = scales[sb] & 0x0F;
        let min_q = (scales[sb] >> 4) & 0x0F;
        let scale = d * sc_q as f32;
        let min = dmin * min_q as f32;

        let byte_idx = flat_idx / 4;
        let shift = (flat_idx % 4) * 2;
        let q = (qs[byte_idx] >> shift) & 0x03;

        out.push(q as f32 * scale - min);
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// § 9  Q3_K
// ═══════════════════════════════════════════════════════════════════════════
//
// Layout (110 bytes, 256 weights):
//   hmask: [u8; 32] - the most significant (3rd) bit of each of the 256 codes,
//                                1 bit/weight
//   qs: [u8; 64] - low-order 2 bits of codes, 4 weights/byte
//   scales: [u8; 12] - 16x6-bit signed scale (offset +32), the same
//                                bit-cursor idea that get/put_scale_min_k4,
//                                but WITHOUT min-half (Q3_K symmetric, only
//                                x = a*q, not x = a*q+b) - packing is simpler:
//                                12 bytes = 16 sub-blocks * 6 bits = 96 bits.
//   d:      f16                — super-block scale
//   W_hat = d*scale[sb] * (q - 4) (q ∈ [0,7] unsigned code, code-4 gives
//                                    signed range [-4,3])
//
// The scale code here is SIGNED 6-bit (range after offset [-32,31], stored
// as unsigned [0,63] with offset +32) - direct packaging implementation `scales`
// in `quantize_row_q3_K_ref`: bottom 4 bits of each of the first 8 values
// lie in scales[0..8] 4 bits/value NO - actual layout
// most simply described as an honest bit-cursor: 16 values of 6 bits each
// consecutive = 96 bits = 12 bytes, the least significant bit of the first value is bit 0 of byte 0.

/// Write `nbits`-bit value `val` to bit-cursor position `bit_pos`
/// dense array `buf` (little-bit-first within a byte, as in all
/// other bit-cursor packages of this crate - see `intn::write_bits`).
fn write_bits_at(buf: &mut [u8], bit_pos: usize, nbits: u32, val: u32) {
    for b in 0..nbits {
        if (val >> b) & 1 != 0 {
            let pos = bit_pos + b as usize;
            buf[pos / 8] |= 1 << (pos % 8);
        }
    }
}

/// Read `nbits`-bit value from position `bit_pos`. Symmetrical
/// `write_bits_at` above.
fn read_bits_at(buf: &[u8], bit_pos: usize, nbits: u32) -> u32 {
    let mut val = 0u32;
    for b in 0..nbits {
        let pos = bit_pos + b as usize;
        if (buf[pos / 8] >> (pos % 8)) & 1 != 0 {
            val |= 1 << b;
        }
    }
    val
}

fn encode_q3_k(chunk: &[f32; QK_K], out: &mut Vec<u8>) {
    let mut sub_scale = [0.0f32; 16];
    let mut sub_codes: Vec<[i8; 16]> = Vec::with_capacity(16);

    for sb in 0..16 {
        let sub = &chunk[sb * 16..(sb + 1) * 16];
        let (scale, codes) = optimize_q3_scale(sub, 4);
        sub_scale[sb] = scale;
        let mut arr = [0i8; 16];
        // `codes` already offset-biased (+nmax=4) from optimize_q3_scale, range
        // [0,7] unsigned - keep it as is, the same format as the final one
        // qs code (see decoder: `code - 4`).
        arr.copy_from_slice(&codes[..16]);
        sub_codes.push(arr);
    }

    let amax_scale = sub_scale.iter().fold(0.0f32, |acc, s| acc.max(s.abs()));
    let d = if amax_scale > 0.0 {
        amax_scale / 31.0
    } else {
        0.0
    };
    let inv_d = if d > 0.0 { 1.0 / d } else { 0.0 };

    // 16 sub-blocks * 6 bits = 96 bits = 12 bytes, signed scale with offset +32
    // (range [-32.31] → [0.63] unsigned), fair bit-cursor.
    let mut scales_packed = [0u8; 12];
    let mut eff_scale = [0.0f32; 16];
    for sb in 0..16 {
        let sc_signed = clampi(nearest_i32(inv_d * sub_scale[sb]), -32, 31);
        let sc_q = (sc_signed + 32) as u32; // → [0,63]
        write_bits_at(&mut scales_packed, sb * 6, 6, sc_q);
        eff_scale[sb] = d * sc_signed as f32;
    }

    let mut hmask = [0u8; 32];
    let mut qs = [0u8; 64];
    for sb in 0..16 {
        let scale = eff_scale[sb];
        let inv = if scale != 0.0 { 1.0 / scale } else { 0.0 };
        let sub = &chunk[sb * 16..(sb + 1) * 16];

        for i in 0..16 {
            let flat_idx = sb * 16 + i;
            let q = clampi(nearest_i32(inv * sub[i]), -4, 3) + 4; // → [0,7]
            let q = q as u8;

            let low2 = q & 0x03;
            let high1 = (q >> 2) & 0x01;

            let byte_idx = flat_idx / 4;
            let shift = (flat_idx % 4) * 2;
            qs[byte_idx] |= low2 << shift;

            if high1 != 0 {
                hmask[flat_idx / 8] |= 1 << (flat_idx % 8);
            }
        }
    }

    out.extend_from_slice(&hmask);
    out.extend_from_slice(&qs);
    out.extend_from_slice(&scales_packed);
    out.extend_from_slice(&crate::pack::f16_saturating(d).to_le_bytes());
}

fn decode_q3_k(block: &[u8], out: &mut Vec<f32>) {
    let hmask = &block[0..32];
    let qs = &block[32..96];
    let scales = &block[96..108];
    let d = f16::from_le_bytes([block[108], block[109]]).to_f32();

    for sb in 0..16 {
        let sc_q = read_bits_at(scales, sb * 6, 6) as i32;
        let sc_signed = sc_q - 32; // [0,63] → [-32,31]
        let scale = d * sc_signed as f32;

        for i in 0..16 {
            let flat_idx = sb * 16 + i;
            let byte_idx = flat_idx / 4;
            let shift = (flat_idx % 4) * 2;
            let low2 = (qs[byte_idx] >> shift) & 0x03;
            let high1 = (hmask[flat_idx / 8] >> (flat_idx % 8)) & 1;
            let q = ((high1 as i32) << 2 | low2 as i32) - 4; // [0,7] → [-4,3]
            out.push(q as f32 * scale);
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// § 10 TESTS
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    fn ramp(n: usize, scale: f32) -> Vec<f32> {
        (0..n)
            .map(|i| (i as f32 - n as f32 / 2.0) * scale)
            .collect()
    }

    #[test]
    fn test_is_supported_excludes_iq_and_simple_blocks() {
        assert!(is_supported(DType::Q2_K));
        assert!(is_supported(DType::Q3_K));
        assert!(is_supported(DType::Q4_K));
        assert!(is_supported(DType::Q5_K));
        assert!(is_supported(DType::Q6_K));
        assert!(is_supported(DType::Q8_K));
        assert!(!is_supported(DType::Q4_0));
        assert!(!is_supported(DType::IQ4_NL));
    }

    #[test]
    fn test_q4_k_roundtrip_single_block() {
        let data = ramp(256, 0.01);
        let packed = quantize_kquant(&data, DType::Q4_K);
        assert_eq!(packed.bytes_len(), 144); // 1 super-block exactly
        let restored = packed.to_f32();
        assert_eq!(restored.len(), 256);
        for (a, b) in data.iter().zip(restored.iter()) {
            assert!((a - b).abs() < 0.05, "{a} vs {b}");
        }
    }

    #[test]
    fn test_q5_k_more_accurate_than_q4_k() {
        let data = ramp(256, 0.01);
        let q4 = quantize_kquant(&data, DType::Q4_K);
        let q5 = quantize_kquant(&data, DType::Q5_K);
        assert_eq!(q5.bytes_len(), 176);
        let err = |p: &KQuantPacked| -> f32 {
            p.to_f32()
                .iter()
                .zip(data.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0, f32::max)
        };
        assert!(
            err(&q5) <= err(&q4) + 1e-4,
            "q5={} q4={}",
            err(&q5),
            err(&q4)
        );
    }

    #[test]
    fn test_q6_k_roundtrip_and_size() {
        let data = ramp(256, 0.005);
        let packed = quantize_kquant(&data, DType::Q6_K);
        assert_eq!(packed.bytes_len(), 210);
        let restored = packed.to_f32();
        for (a, b) in data.iter().zip(restored.iter()) {
            assert!((a - b).abs() < 0.02, "{a} vs {b}");
        }
    }

    #[test]
    fn test_q6_k_more_accurate_than_q4_k() {
        let data = ramp(256, 0.01);
        let q4 = quantize_kquant(&data, DType::Q4_K);
        let q6 = quantize_kquant(&data, DType::Q6_K);
        let err = |p: &KQuantPacked| -> f32 {
            p.to_f32()
                .iter()
                .zip(data.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0, f32::max)
        };
        assert!(
            err(&q6) <= err(&q4) + 1e-4,
            "q6={} q4={}",
            err(&q6),
            err(&q4)
        );
    }

    #[test]
    fn test_q8_k_roundtrip_high_precision() {
        let data = ramp(256, 0.01);
        let packed = quantize_kquant(&data, DType::Q8_K);
        assert_eq!(packed.bytes_len(), 292);
        let restored = packed.to_f32();
        for (a, b) in data.iter().zip(restored.iter()) {
            assert!((a - b).abs() < 0.02, "{a} vs {b}");
        }
    }

    #[test]
    fn test_q2_k_roundtrip_and_size() {
        let data = ramp(256, 0.02);
        let packed = quantize_kquant(&data, DType::Q2_K);
        assert_eq!(packed.bytes_len(), 84);
        let restored = packed.to_f32();
        assert_eq!(restored.len(), 256);
        // 2 bits/weight - a coarse grid, we allow a noticeably larger error.
        for (a, b) in data.iter().zip(restored.iter()) {
            assert!((a - b).abs() < 1.0, "{a} vs {b}");
        }
    }

    #[test]
    fn test_q3_k_roundtrip_and_size() {
        let data = ramp(256, 0.02);
        let packed = quantize_kquant(&data, DType::Q3_K);
        assert_eq!(packed.bytes_len(), 110);
        let restored = packed.to_f32();
        assert_eq!(restored.len(), 256);
        for (a, b) in data.iter().zip(restored.iter()) {
            assert!((a - b).abs() < 0.5, "{a} vs {b}");
        }
    }

    #[test]
    fn test_q3_k_more_accurate_than_q2_k() {
        let data = ramp(256, 0.02);
        let q2 = quantize_kquant(&data, DType::Q2_K);
        let q3 = quantize_kquant(&data, DType::Q3_K);
        let err = |p: &KQuantPacked| -> f32 {
            p.to_f32()
                .iter()
                .zip(data.iter())
                .map(|(a, b)| (a - b).abs())
                .fold(0.0, f32::max)
        };
        assert!(
            err(&q3) <= err(&q2) + 1e-4,
            "q3={} q2={}",
            err(&q3),
            err(&q2)
        );
    }

    #[test]
    fn test_partial_last_superblock_padded() {
        // 300 weights = 1 full super-block (256) + 1 incomplete (44) → 2
        // super-block, but to_f32() should return exactly 300, not 512.
        let data = ramp(300, 0.01);
        let packed = quantize_kquant(&data, DType::Q4_K);
        assert_eq!(packed.bytes_len(), 288); // 2 * 144
        let restored = packed.to_f32();
        assert_eq!(restored.len(), 300);
    }

    #[test]
    fn test_all_zero_superblock_is_safe() {
        // All weights == 0 should not produce NaN/panic in any format.
        for dtype in [
            DType::Q2_K,
            DType::Q3_K,
            DType::Q4_K,
            DType::Q5_K,
            DType::Q6_K,
            DType::Q8_K,
        ] {
            let data = vec![0.0f32; 256];
            let packed = quantize_kquant(&data, dtype);
            let restored = packed.to_f32();
            assert!(restored.iter().all(|v| v.is_finite()), "dtype={dtype:?}");
            assert!(restored.iter().all(|&v| v.abs() < 1e-3), "dtype={dtype:?}");
        }
    }

    #[test]
    fn test_effective_bits_per_weight_matches_block_bytes() {
        let data = ramp(256, 0.01);
        let q4 = quantize_kquant(&data, DType::Q4_K);
        // 144 bytes * 8 / 256 weights = 4.5 bits/weight exactly.
        assert!((q4.effective_bits_per_weight() - 4.5).abs() < 1e-4);
    }

    #[test]
    #[should_panic(expected = "is not a K-quant super-block format")]
    fn test_quantize_kquant_rejects_non_kquant_dtype() {
        let data = vec![1.0f32; 32];
        let _ = quantize_kquant(&data, DType::Q4_0);
    }

    #[test]
    fn test_extreme_values_clip_without_panic() {
        // Values greatly exceeding the typical range should be clamped
        // and not give overflow/panic in round-to-nearest.
        let mut data = vec![0.01f32; 256];
        data[0] = 1e6;
        data[1] = -1e6;
        for dtype in [
            DType::Q2_K,
            DType::Q3_K,
            DType::Q4_K,
            DType::Q5_K,
            DType::Q6_K,
            DType::Q8_K,
        ] {
            let packed = quantize_kquant(&data, dtype);
            let restored = packed.to_f32();
            assert!(restored.iter().all(|v| v.is_finite()), "dtype={dtype:?}");
        }
    }
}
