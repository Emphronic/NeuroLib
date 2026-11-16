// neo-quantization/src/pack/iq.rs
//
// Portions of this file are derived from ggml/llama.cpp
// (`ggml/src/ggml-quants.c`, `ggml-common.h`), Copyright (c) the ggml/llama.cpp
// authors, used here under the MIT License.
//
// IQ-series — GGUF "importance-weighted quants" (codebook / non-linear
// quantization). Six formats supported by `crate::dtype::DType`:
//
//   IQ4_NL, IQ4_XS - non-linear 4-bit, fixed 16-level
//                       table (similar to NF4 in spirit - see pack/nf4.rs),
//                       block structure like simple/K-quant blocks.
//                       Implemented COMPLETELY in this file.
//
//   IQ2_XXS, IQ2_XS,
//   IQ3_XXS, IQ3_S — codebook-based VECTOR quantization: each
//                       a group of 4/8 weights is coded by an index in
//                       precomputed (offline, k-means/lattice-search
//                       on an E8-like lattice) codebook on
//                       256..1024+ records, with separately packaged
//                       signed bit per element via parity-trick
//                       (`ksigns_iq2xs`/`kmask_iq2xs` in ggml). IMPLEMENTED
//                       in `pack::iq_codebook` (see § "IQ2/IQ3 status" below
//                       for details and division of responsibility of files).
//
// ═══════════════════════════════════════════════════════════════════════════
// § IQ2/IQ3 status - IMPLEMENTED via pack::iq_codebook
// ═══════════════════════════════════════════════════════════════════════════
//
// `IQ2_XXS`/`IQ2_XS`/`IQ3_XXS`/`IQ3_S` require byte-for-byte accurate grid tables
// from `ggml/src/ggml-common.h` (llama.cpp) - hundreds and thousands of 8/16-bit
// records per format (`iq2xxs_grid[256]`, `iq2xs_grid[512]`,
// `iq3xxs_grid[256]`, `iq3s_grid[512]`), plus auxiliary tables for
// sign bit packing (`ksigns_iq2xs[128]`, `kmask_iq2xs`). These are NOT
// formulas derived from the format architecture but specific numbers
// obtained offline by grid search by the llama.cpp developers and fixed as
// part of the binary format contract.
//
// Round-to-nearest or any other procedure WITHOUT the exact values of these
// tables would produce plausible-looking but different bytes - silent
// corruption of data under the guise of the real format. The tables are
// therefore kept byte-for-byte, not approximately:
//
//   • `pack::iq_grids` — `iq2xxs_grid`/`iq2xs_grid`/`iq3xxs_grid`/
//     `iq3s_grid`/`ksigns_iq2xs`/`kmask_iq2xs`, matching
//     `ggml/src/ggml-common.h` token-for-token (see iq_grids.rs).
//   • `pack::iq_codebook` — `quantize_row_iq2_xxs_ref`/
//     `quantize_row_iq2_xs_ref`/`quantize_row_iq3_xxs_ref`/
//     `quantize_row_iq3_s_ref` and paired `dequantize_row_iq*` from
//     `ggml/src/ggml-quants.c`. Instead of the runtime-built lookup structures
//     (`kmap`/`kneighbours`, built once during initialization
//     to speed up reverse search) here we use direct
//     searching through the entire grid table for a miss - gives an identical result
//     (`kneighbours` upstream is a pure speed optimization that does not change
//     which point is selected), acceptable for offline conversion of weights
//     (not hot path inference).
//
// `quantize_iq_codebook` in this file (§ 3) dispatches to
// `pack::iq_codebook` by `DType`; dequantization goes through
// `IqPacked::to_f32()` - each option carries its own `to_f32()`, a separate
// dequantize dispatcher is not needed (see `IqPacked` in § 4, options
// `Iq2Xxs`/`Iq2Xs`/`Iq3Xxs`/`Iq3S` on par with `Iq4Nl`/`Iq4Xs`).

use burn::tensor::f16;

use crate::dtype::DType;

// ═══════════════════════════════════════════════════════════════════════════
// § 0 GENERAL HELPERS (same style as pack::kquant/pack::intn)
// ═══════════════════════════════════════════════════════════════════════════

#[inline]
fn nearest_i32(x: f32) -> i32 {
    x.round() as i32
}

#[inline]
fn clampi(v: i32, lo: i32, hi: i32) -> i32 {
    v.max(lo).min(hi)
}

/// `true`, if `pack::iq` can pack/unpack this DType.
///
/// All six IQ-DTypes: `IQ4_NL`/`IQ4_XS` are implemented directly in this file
/// (§ 1/§ 2), `IQ2_XXS`/`IQ2_XS`/`IQ3_XXS`/`IQ3_S` - dispatched in
/// `pack::iq_codebook` (see file header).
pub fn is_supported(dtype: DType) -> bool {
    is_iq_series(dtype)
}

/// `true` if `DType` belongs to the IQ series.
///
/// The eight variants are handled by three modules: `IQ4_NL`/`IQ4_XS` in
/// this file, `IQ1_S`/`IQ1_M` in `pack::iq1`, and the IQ2/IQ3 codebook
/// formats in `pack::iq_codebook`.
pub fn is_iq_series(dtype: DType) -> bool {
    matches!(
        dtype,
        DType::IQ2_XXS
            | DType::IQ2_XS
            | DType::IQ3_XXS
            | DType::IQ3_S
            | DType::IQ4_NL
            | DType::IQ4_XS
            | DType::IQ1_S
            | DType::IQ1_M
    )
}

// ═══════════════════════════════════════════════════════════════════════════
// § 1 IQ4_NL - 32 elem/block, fixed 16-level nonlinear grid
// ═══════════════════════════════════════════════════════════════════════════
//
// Layout (18 bytes, 32 weights) - a direct analogue of Q4_0, but with uneven
// levels instead of uniform nibble code:
//
//   d: f16 (2 bytes) — scale
//   qs: [u8; 16] - 32 nibble indexes in IQ4_NL_LEVELS, 2/byte
//
//   W_hat[i] = IQ4_NL_LEVELS[nibble[i]] * d

/// 16 fixed non-linear levels IQ4_NL (same role as
/// `NF4_LEVELS` to `pack::nf4`, but different, separately calibrated for
/// llama.cpp grid - integer codes in the range [-127,127],
/// normalized through `d` rather than N(0,1) quantiles). Values - standard
/// table `kvalues_iq4nl` from ggml-common.h.
pub const IQ4_NL_LEVELS: [i8; 16] = [
    -127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113,
];

#[inline]
fn nearest_iq4nl_index(x: i32) -> u8 {
    let mut best_idx = 0usize;
    let mut best_dist = i32::MAX;
    for (i, &lvl) in IQ4_NL_LEVELS.iter().enumerate() {
        let d = (x - lvl as i32).abs();
        if d < best_dist {
            best_dist = d;
            best_idx = i;
        }
    }
    best_idx as u8
}

/// Packed IQ4_NL tensor.
#[derive(Debug, Clone, PartialEq)]
pub struct Iq4NlPacked {
    /// Raw blocks (18 bytes/block), concatenated in a row.
    pub blocks: Vec<u8>,
    /// Initial number of scales.
    pub numel: usize,
}

impl Iq4NlPacked {
    /// Initial number of scales.
    pub fn numel(&self) -> usize {
        self.numel
    }
    /// Size in bytes actually occupied by blocks.
    pub fn bytes_len(&self) -> usize {
        self.blocks.len()
    }
    /// Expand to `Vec<f32>`.
    pub fn to_f32(&self) -> Vec<f32> {
        dequantize_iq4_nl(self)
    }
}

/// Quantize `data` in IQ4_NL.
///
/// Calibration: `d = amax / 127` (maximum modulus table code -
/// `127`, see `IQ4_NL_LEVELS`), then round-to-nearest table index by
/// `w / d`. This is a direct analogue of `encode_q4_0` in `pack::gguf`, it is different
/// only non-linear lookup instead of `signed + 8` coding.
pub fn quantize_iq4_nl(data: &[f32]) -> Iq4NlPacked {
    let block_elems = DType::IQ4_NL.block_elems(); // 32
    let n_blocks = data.len().div_ceil(block_elems);
    let mut blocks = Vec::with_capacity(n_blocks * DType::IQ4_NL.block_bytes());

    for b in 0..n_blocks {
        let start = b * block_elems;
        let end = (start + block_elems).min(data.len());
        let mut chunk = [0.0f32; 32];
        chunk[..end - start].copy_from_slice(&data[start..end]);

        // Both IQ4_NL and IQ4_XS use `calibrate_iq4_block` (see its doc
        // comment): with one shared level table, a single calibration rule
        // keeps their codes consistent for the same 32 weights.
        let (d, codes) = calibrate_iq4_block(&chunk);

        // The scale is stored in f16, and the codes were selected for the f32 value.
        // We re-select the codes for the ACTUALLY recorded scale, otherwise
        // dequantization will follow a different d path than the one that minimized
        // error - the same logic as in step 3 for IQ4_XS below.
        let d_stored = crate::pack::f16_saturating(d);
        blocks.extend_from_slice(&d_stored.to_le_bytes());
        let d_eff = d_stored.to_f32();
        let inv_d = if d_eff > 0.0 { 1.0 / d_eff } else { 0.0 };

        let mut final_codes = [0u8; 32];
        for (i, &w) in chunk.iter().enumerate() {
            final_codes[i] = if inv_d > 0.0 {
                nearest_iq4nl_index(clampi(nearest_i32(inv_d * w), -127, 127))
            } else {
                codes[i]
            };
        }
        for pair in final_codes.chunks(2) {
            blocks.push(pair[0] | (pair[1] << 4));
        }
    }

    Iq4NlPacked {
        blocks,
        numel: data.len(),
    }
}

/// Completely expand the IQ4_NL package back to `Vec<f32>`.
pub fn dequantize_iq4_nl(packed: &Iq4NlPacked) -> Vec<f32> {
    let block_bytes = DType::IQ4_NL.block_bytes(); // 18
    let n_blocks = packed.blocks.len() / block_bytes;
    let mut out = Vec::with_capacity(n_blocks * 32);

    for b in 0..n_blocks {
        let block = &packed.blocks[b * block_bytes..(b + 1) * block_bytes];
        let d = f16::from_le_bytes([block[0], block[1]]).to_f32();
        for &byte in &block[2..18] {
            out.push(IQ4_NL_LEVELS[(byte & 0x0F) as usize] as f32 * d);
            out.push(IQ4_NL_LEVELS[(byte >> 4) as usize] as f32 * d);
        }
    }

    out.truncate(packed.numel);
    out
}

// ═══════════════════════════════════════════════════════════════════════════
// § 2  IQ4_XS — 256 elem/super-block (QK_K), IQ4_NL levels +
//        K-quant-style two-level 6-bit signed scale/sub-block
// ═══════════════════════════════════════════════════════════════════════════
//
// Direct implementation of `block_iq4_xs` (ggml-common.h) /
// `quantize_row_iq4_xs_ref` (ggml-quants.c) - bitwise-compatible layout,
// not an approximation. The type contract is
// `DType::IQ4_XS.block_elems() == 256`, `block_bytes() == 136`.
//
// ## Layout (136 bytes, 256 weights, 8 sub-blocks of 32)
//
//   d: f16 (2 bytes) — super-block scale
//   scales_h: u16 (2 bytes) - the most significant 2 bits of each of 8
//                                       sub-block 6-bit signed codes
//                                       (bits `2*sb .. 2*sb+2`)
//   scales_l: [u8; 4] - the least significant 4 bits of each of 8
//                                       6-bit codes, nibble-packed
//                                       (2 sub-blocks/byte)
//   qs: [u8; 128] - 256 nibble indexes in
//                                       `IQ4_NL_LEVELS`, 2 weights/byte,
//                                       sub-blocks sequentially (one
//                                       same layout as qs in Q4_K)
//
//   sub_scale_code[sb] ∈ [0, 63] = scales_l_nibble[sb] | (scales_h_2bit[sb] << 4)
//   sub_scale[sb] = d * (sub_scale_code[sb] - 32)      — signed, offset +32
//   W_hat[sb*32 + i] = IQ4_NL_LEVELS[qs_nibble[i]] * sub_scale[sb]
//
// Unlike Q4_K/Q5_K (see `pack::kquant`), there is NO separate `min`
// (`dmin`/`get_scale_min_k4`-pair field) - only one signed scale per
// sub-block around a fixed non-linear table `IQ4_NL_LEVELS`,
// so packing 6-bit codes is simpler: 4 bits are nibble-packed into
// `scales_l[4]`, 2 most significant bits - separately in `scales_h` (u16, 2
// bits × 8 sub-blocks = 16 bits exactly).

/// Number of weights in one IQ4_XS super-block (QK_K=256 in llama.cpp,
/// 8 sub-blocks of 32). Same as `DType::IQ4_XS.block_elems()` -
/// declared as a separate constant for the sake of readability of the code below (`chunk:
/// [f32; 256]`, etc. otherwise you would have to write it as a non-const expression).
const QK_IQ4_XS: usize = 256;

/// Unpack the 6-bit signed scale code of the sub-block `sb` (0..8) from
/// `scales_l`/`scales_h` block fields. Returns an unsigned code
/// `[0, 63]` (the caller subtracts offset `32` to get
/// signed values - see Layout section above).
#[inline]
fn get_iq4_xs_scale(sb: usize, scales_l: &[u8; 4], scales_h: u16) -> u8 {
    let low = if sb.is_multiple_of(2) {
        scales_l[sb / 2] & 0x0F
    } else {
        (scales_l[sb / 2] >> 4) & 0x0F
    };
    let high = ((scales_h >> (sb * 2)) & 0x03) as u8;
    low | (high << 4)
}

/// Pack the 6-bit unsigned scale code of the sub-block `sb` into `scales_l`/
/// `scales_h`. Reverse operation to `get_iq4_xs_scale`.
#[inline]
fn put_iq4_xs_scale(sb: usize, code: u8, scales_l: &mut [u8; 4], scales_h: &mut u16) {
    let low = code & 0x0F;
    let high = (code >> 4) & 0x03;
    if sb.is_multiple_of(2) {
        scales_l[sb / 2] = (scales_l[sb / 2] & 0xF0) | low;
    } else {
        scales_l[sb / 2] = (scales_l[sb / 2] & 0x0F) | (low << 4);
    }
    *scales_h |= (high as u16) << (sb * 2);
}

/// Packed IQ4_XS super-block tensor - bitwise compatible with
/// `block_iq4_xs` from llama.cpp (see Layout above).
#[derive(Debug, Clone, PartialEq)]
pub struct Iq4XsPacked {
    /// Raw super-blocks (136 bytes each), concatenated in a row.
    pub blocks: Vec<u8>,
    /// Initial number of scales.
    pub numel: usize,
}

impl Iq4XsPacked {
    /// Initial number of scales.
    pub fn numel(&self) -> usize {
        self.numel
    }
    /// Size in bytes actually occupied by blocks.
    pub fn bytes_len(&self) -> usize {
        self.blocks.len()
    }
    /// Expand to `Vec<f32>`.
    pub fn to_f32(&self) -> Vec<f32> {
        dequantize_iq4_xs(self)
    }
}

/// Calibrate one sub-block (32 weights) for a fixed table
/// `IQ4_NL_LEVELS`: Finds `scale` that minimizes rounding error
/// `Σ(w_i - level[code_i]·scale)²` by simple search of candidates
/// `amax / |level|` for each non-zero table level (same in spirit
/// reception that `optimize_scale_signed` in `pack::kquant`, adapted
/// for a nonlinear, rather than uniform, code grid - there is a candidate scale
/// one, here there are up to 8 mismatched ones, because the table is asymmetrical
/// value between adjacent levels).
///
/// Returns `(scale, codes[32])`, `codes[i] ∈ [0, 15]` - index in
/// Calibration of one 32-element IQ4 block: selection of scale and codes.
///
/// Common for IQ4_NL and for each sub-block IQ4_XS - XS used to sort through
/// grid of scales, and NL took `amax/127` “head-on”, which is why on the same
/// On the same 32 scales, two formats gave DIFFERENT codes, although they have a table of levels
/// alone.
///
/// Two stages, as in `quantize_row_iq4_nl_impl` (ggml):
///
///  1. Enumeration of candidates of scale around `amax/127`. Needed because
///     the `IQ4_NL_LEVELS` grid is NOT uniform: shifting the scale changes not only
///     rounding error, but also what levels will be
///     are involved, and the optimum does not have to lie in `amax/127`.
///
///  2. For the codes of each candidate - the exact scale using the method of least
///     squares: with fixed codes `q_j` minimum `Σ(x_j − d·q_j)²`
///     is achieved at `d = Σ(q_j·x_j) / Σ(q_j²)`. This gives the best
///     possible scale for a given set of codes free of charge, without
///     refinement of the search grid.
///
/// Returns `(scale, codes)` - an always matched pair: both
/// taken from one winning candidate.
fn calibrate_iq4_block(sub: &[f32]) -> (f32, [u8; 32]) {
    debug_assert_eq!(sub.len(), 32);

    let amax = sub.iter().fold(0.0f32, |acc, w| acc.max(w.abs()));
    if amax < 1e-15 {
        // The table does not contain zero; the level closest to zero is index 8
        // (value 1). At scale = 0, dequantization gives exactly zeros.
        return (0.0, [8u8; 32]);
    }

    let base_scale = amax / 127.0;
    let mut best_scale = base_scale;
    let mut best_err = f32::INFINITY;
    let mut best_codes = [8u8; 32];

    for step in -10i32..=10 {
        let cand_scale = base_scale * (1.0 + 0.01 * step as f32);
        if cand_scale <= 0.0 {
            continue;
        }
        let inv = 1.0 / cand_scale;

        let mut codes = [0u8; 32];
        let mut sum_qx = 0.0f32;
        let mut sum_q2 = 0.0f32;
        for (i, &w) in sub.iter().enumerate() {
            let target = (w * inv).clamp(-127.0, 127.0) as i32;
            let code = nearest_iq4nl_index(target);
            codes[i] = code;
            let q = IQ4_NL_LEVELS[code as usize] as f32;
            sum_qx += q * w;
            sum_q2 += q * q;
        }

        // Stage 2: optimal scale for already selected codes.
        let refined = if sum_q2 > 0.0 {
            sum_qx / sum_q2
        } else {
            cand_scale
        };
        // A negative scale would mean inverting the features of the entire block -
        // for such a set of codes it is not optimal, but a degenerate case
        // (all weights are zero, etc.); then we remain with the candidate.
        let scale = if refined > 0.0 { refined } else { cand_scale };

        let mut err = 0.0f32;
        for (i, &w) in sub.iter().enumerate() {
            let diff = w - IQ4_NL_LEVELS[codes[i] as usize] as f32 * scale;
            err += diff * diff;
        }

        if err < best_err {
            best_err = err;
            best_scale = scale;
            best_codes = codes;
        }
    }

    (best_scale, best_codes)
}

/// Quantize `data` in IQ4_XS (see Layout in the title of § 2).
pub fn quantize_iq4_xs(data: &[f32]) -> Iq4XsPacked {
    let block_elems = QK_IQ4_XS; // 256
    let block_bytes = DType::IQ4_XS.block_bytes(); // 136
    let n_blocks = data.len().div_ceil(block_elems);
    let mut blocks = Vec::with_capacity(n_blocks * block_bytes);

    for b in 0..n_blocks {
        let start = b * block_elems;
        let end = (start + block_elems).min(data.len());
        let mut chunk = [0.0f32; 256];
        chunk[..end - start].copy_from_slice(&data[start..end]);

        // ── Step 1: Calibrate each of the 8 sub-blocks independently ──────
        let mut sub_scale = [0.0f32; 8];
        let mut sub_codes: Vec<[u8; 32]> = Vec::with_capacity(8);
        for sb in 0..8 {
            let sub = &chunk[sb * 32..(sb + 1) * 32];
            let (scale, codes) = calibrate_iq4_block(sub);
            sub_scale[sb] = scale;
            sub_codes.push(codes);
        }

        // ── Step 2: quantize the sub_scale itself into 6-bit signed (offset +32) ──
        let max_scale = sub_scale.iter().fold(0.0f32, |acc, s| acc.max(s.abs()));
        let d = if max_scale > 0.0 {
            max_scale / 31.0
        } else {
            0.0
        };
        let inv_d = if d > 0.0 { 1.0 / d } else { 0.0 };

        let mut scales_l = [0u8; 4];
        let mut scales_h: u16 = 0;
        let mut eff_scale = [0.0f32; 8];
        for sb in 0..8 {
            let signed = clampi(nearest_i32(inv_d * sub_scale[sb]), -32, 31);
            let code = (signed + 32) as u8; // → [0,63]
            put_iq4_xs_scale(sb, code, &mut scales_l, &mut scales_h);
            eff_scale[sb] = d * signed as f32;
        }

        // ── Step 3: reforge the weights effectively (after 6-bit
        //    rounding) scale - the same principle as `calibrate_k4_subblocks`
        //    in pack::kquant (see its doc comment): dequantization MUST
        //    to coincide with what a real quantizer would write means
        //    weights need to be reforged with a rounded scale, and not with the one
        //    that came out of `calibrate_iq4_block` before rounding.
        let mut qs = [0u8; 128];
        for sb in 0..8 {
            let sub = &chunk[sb * 32..(sb + 1) * 32];
            let scale = eff_scale[sb];
            let inv = if scale != 0.0 { 1.0 / scale } else { 0.0 };

            let mut codes = [0u8; 32];
            for i in 0..32 {
                let target = clampi(nearest_i32(inv * sub[i]), -127, 127);
                codes[i] = nearest_iq4nl_index(target);
            }

            let base = sb * 16;
            for i in 0..16 {
                qs[base + i] = codes[2 * i] | (codes[2 * i + 1] << 4);
            }
        }
        let _ = sub_codes; // calculated at the calibration stage, final codes from Step 3

        blocks.extend_from_slice(&crate::pack::f16_saturating(d).to_le_bytes());
        blocks.extend_from_slice(&scales_h.to_le_bytes());
        blocks.extend_from_slice(&scales_l);
        blocks.extend_from_slice(&qs);
    }

    debug_assert_eq!(blocks.len(), n_blocks * block_bytes);
    Iq4XsPacked {
        blocks,
        numel: data.len(),
    }
}

/// Completely expand the IQ4_XS package back to `Vec<f32>`.
pub fn dequantize_iq4_xs(packed: &Iq4XsPacked) -> Vec<f32> {
    let block_bytes = DType::IQ4_XS.block_bytes(); // 136
    let n_blocks = packed.blocks.len() / block_bytes;
    let mut out = Vec::with_capacity(n_blocks * QK_IQ4_XS);

    for b in 0..n_blocks {
        let block = &packed.blocks[b * block_bytes..(b + 1) * block_bytes];
        let d = f16::from_le_bytes([block[0], block[1]]).to_f32();
        let scales_h = u16::from_le_bytes([block[2], block[3]]);
        let scales_l: [u8; 4] = [block[4], block[5], block[6], block[7]];
        let qs = &block[8..136];

        for sb in 0..8 {
            let code = get_iq4_xs_scale(sb, &scales_l, scales_h);
            let scale = d * (code as i32 - 32) as f32;

            let base = sb * 16;
            for i in 0..16 {
                let byte = qs[base + i];
                out.push(IQ4_NL_LEVELS[(byte & 0x0F) as usize] as f32 * scale);
                out.push(IQ4_NL_LEVELS[(byte >> 4) as usize] as f32 * scale);
            }
        }
    }

    out.truncate(packed.numel);
    out
}

// ═══════════════════════════════════════════════════════════════════════════
// § 3 IQ2_XXS / IQ2_XS / IQ3_XXS / IQ3_S - codebook quantization
// ═══════════════════════════════════════════════════════════════════════════
//
// NOT IMPLEMENTED - see file header. The functions below exist only
// to give a clear error and a single entry point, symmetrical
// `pack::gguf`/`pack::kquant` (same pattern `is_supported` +
// `quantize_*`/`dequantize_*` with an explicit panic for the rest of the crate).

/// Quantize into one of the codebook-based IQ formats.
///
/// Dispatch at `pack::iq_codebook` by `dtype`; calls the corresponding
/// `quantize_iq{2,3}_*` with `quant_weights = None` (fallback `weight[i] =
/// x[i]²` - see § "quant_weights" in `pack::iq_codebook` for when this
/// bitwise coincides with upstream, and when it is simply an equivalent principle).
/// The calling code that has a true importance matrix must
/// call `pack::iq_codebook::quantize_iq{2,3}_*` directly from `Some(imatrix)`.
///
/// #Panic
///
/// If `dtype` is not included in the codebook subset of the IQ series (checked
/// calling `quantize_iq` before calling this function via `is_iq_series`).
pub fn quantize_iq_codebook(data: &[f32], dtype: DType) -> IqPacked {
    use crate::pack::iq_codebook as cb;
    match dtype {
        DType::IQ2_XXS => IqPacked::Iq2Xxs(cb::quantize_iq2_xxs(data, None)),
        DType::IQ2_XS => IqPacked::Iq2Xs(cb::quantize_iq2_xs(data, None)),
        DType::IQ3_XXS => IqPacked::Iq3Xxs(cb::quantize_iq3_xxs(data, None)),
        DType::IQ3_S => IqPacked::Iq3S(cb::quantize_iq3_s(data, None)),
        _ => unreachable!(
            "pack::iq::quantize_iq_codebook: {dtype:?} is not a codebook-based IQ DType \
             (caller must check is_iq_series before dispatching here)"
        ),
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// § 4 SINGLE ENTRY POINT (symmetrically pack::gguf::quantize_gguf /
//        pack::kquant::quantize_kquant)
// ═══════════════════════════════════════════════════════════════════════════

use crate::pack::iq1::{Iq1MPacked, Iq1SPacked};
use crate::pack::iq_codebook::{Iq2XsPacked, Iq2XxsPacked, Iq3SPacked, Iq3XxsPacked};

/// Packed result of any IQ format - type-erased over all eight packed types.
/// `PackedBuffer` in `pack::mod` wraps this type the same way as
/// `GgufPacked`/`KQuantPacked`.
#[derive(Debug, Clone, PartialEq)]
pub enum IqPacked {
    /// IQ4_NL - non-linear 4-bit, fixed 16-level grid, see § 1.
    Iq4Nl(Iq4NlPacked),
    /// IQ4_XS - 256-elem super-block format (`block_iq4_xs`), bitwise
    /// compatible with llama.cpp/GGUF - see Layout in § 2.
    Iq4Xs(Iq4XsPacked),
    /// IQ2_XXS - codebook/lattice quantization, see `pack::iq_codebook`.
    Iq2Xxs(Iq2XxsPacked),
    /// IQ2_XS - codebook/lattice quantization, see `pack::iq_codebook`.
    Iq2Xs(Iq2XsPacked),
    /// IQ3_XXS - codebook/lattice quantization, see `pack::iq_codebook`.
    Iq3Xxs(Iq3XxsPacked),
    /// IQ3_S - codebook/lattice quantization, see `pack::iq_codebook`.
    Iq3S(Iq3SPacked),
    /// IQ1_S - 1-bit codebook quantization, see `pack::iq1`.
    Iq1S(Iq1SPacked),
    /// IQ1_M - 1-bit codebook quantization with per-sub-block scale, see.
    /// `pack::iq1`.
    Iq1M(Iq1MPacked),
}

impl IqPacked {
    /// GGUF DType of this buffer.
    pub fn dtype(&self) -> DType {
        match self {
            IqPacked::Iq4Nl(_) => DType::IQ4_NL,
            IqPacked::Iq4Xs(_) => DType::IQ4_XS,
            IqPacked::Iq2Xxs(_) => DType::IQ2_XXS,
            IqPacked::Iq2Xs(_) => DType::IQ2_XS,
            IqPacked::Iq3Xxs(_) => DType::IQ3_XXS,
            IqPacked::Iq3S(_) => DType::IQ3_S,
            IqPacked::Iq1S(_) => DType::IQ1_S,
            IqPacked::Iq1M(_) => DType::IQ1_M,
        }
    }
    /// Initial number of scales.
    pub fn numel(&self) -> usize {
        match self {
            IqPacked::Iq4Nl(p) => p.numel(),
            IqPacked::Iq4Xs(p) => p.numel(),
            IqPacked::Iq2Xxs(p) => p.numel(),
            IqPacked::Iq2Xs(p) => p.numel(),
            IqPacked::Iq3Xxs(p) => p.numel(),
            IqPacked::Iq3S(p) => p.numel(),
            IqPacked::Iq1S(p) => p.numel(),
            IqPacked::Iq1M(p) => p.numel(),
        }
    }
    /// Size in bytes actually occupied by blocks.
    pub fn bytes_len(&self) -> usize {
        match self {
            IqPacked::Iq4Nl(p) => p.bytes_len(),
            IqPacked::Iq4Xs(p) => p.bytes_len(),
            IqPacked::Iq2Xxs(p) => p.bytes_len(),
            IqPacked::Iq2Xs(p) => p.bytes_len(),
            IqPacked::Iq3Xxs(p) => p.bytes_len(),
            IqPacked::Iq3S(p) => p.bytes_len(),
            IqPacked::Iq1S(p) => p.bytes_len(),
            IqPacked::Iq1M(p) => p.bytes_len(),
        }
    }
    /// Expand to `Vec<f32>`.
    pub fn to_f32(&self) -> Vec<f32> {
        match self {
            IqPacked::Iq4Nl(p) => p.to_f32(),
            IqPacked::Iq4Xs(p) => p.to_f32(),
            IqPacked::Iq2Xxs(p) => p.to_f32(),
            IqPacked::Iq2Xs(p) => p.to_f32(),
            IqPacked::Iq3Xxs(p) => p.to_f32(),
            IqPacked::Iq3S(p) => p.to_f32(),
            IqPacked::Iq1S(p) => p.to_f32(),
            IqPacked::Iq1M(p) => p.to_f32(),
        }
    }
}

/// Quantize `data` into IQ format `dtype`.
///
/// For `IQ2_XXS`/`IQ2_XS`/`IQ3_XXS`/`IQ3_S` uses `quant_weights =
/// None` (see `quantize_iq_codebook` for details); calling code with
/// The real importance matrix should be called `pack::iq_codebook` directly.
///
/// #Panic
///
/// If `dtype` is not included in the IQ series at all (see `is_iq_series`).
pub fn quantize_iq(data: &[f32], dtype: DType) -> IqPacked {
    use crate::pack::iq1;
    match dtype {
        DType::IQ4_NL => IqPacked::Iq4Nl(quantize_iq4_nl(data)),
        DType::IQ4_XS => IqPacked::Iq4Xs(quantize_iq4_xs(data)),
        // IQ1_S/IQ1_M: do not go through quantize_iq_codebook (that function
        // covers only `quant_weights: Option<&[f32]>` - IQ1_S requires
        // `&[f32]` directly, see pack::iq1::quantize_iq1_s). We use
        // uncalibrated option (fallback quant_weights = x[i]²) - the same
        // the principle is that `quant_weights = None` for IQ2/IQ3 is higher.
        DType::IQ1_S => IqPacked::Iq1S(iq1::quantize_iq1_s_uncalibrated(data)),
        DType::IQ1_M => IqPacked::Iq1M(iq1::quantize_iq1_m(data, None)),
        _ => {
            assert!(
                is_iq_series(dtype),
                "pack::iq::quantize_iq: {dtype:?} is not an IQ-series DType"
            );
            quantize_iq_codebook(data, dtype)
        }
    }
}

/// Completely expand the IQ package back to `Vec<f32>`.
pub fn dequantize_iq(packed: &IqPacked) -> Vec<f32> {
    packed.to_f32()
}

// ═══════════════════════════════════════════════════════════════════════════
// § 5 TESTS
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_qk_iq4_xs_matches_dtype_contract() {
        // Local `QK_IQ4_XS` must remain equal to `DType::IQ4_XS.block_elems()`
        // (256, super-block), not the value of IQ4_NL (32).
        assert_eq!(QK_IQ4_XS, DType::IQ4_XS.block_elems());
        assert_eq!(QK_IQ4_XS, 256);
    }

    #[test]
    fn test_is_supported_covers_all_eight() {
        assert!(is_supported(DType::IQ4_NL));
        assert!(is_supported(DType::IQ4_XS));
        assert!(is_supported(DType::IQ2_XXS));
        assert!(is_supported(DType::IQ2_XS));
        assert!(is_supported(DType::IQ3_XXS));
        assert!(is_supported(DType::IQ3_S));
        // IQ1_S/IQ1_M also belong to the IQ series; see is_iq_series.
        assert!(is_supported(DType::IQ1_S));
        assert!(is_supported(DType::IQ1_M));
    }

    #[test]
    fn test_is_iq_series_covers_all_eight() {
        for dt in [
            DType::IQ2_XXS,
            DType::IQ2_XS,
            DType::IQ3_XXS,
            DType::IQ3_S,
            DType::IQ4_NL,
            DType::IQ4_XS,
            DType::IQ1_S,
            DType::IQ1_M,
        ] {
            assert!(is_iq_series(dt), "{dt:?} should be IQ-series");
        }
        assert!(!is_iq_series(DType::Q4_K));
        assert!(!is_iq_series(DType::F32));
        assert!(!is_iq_series(DType::TQ1_0));
    }

    #[test]
    fn test_levels_sorted_and_in_i8_range() {
        for w in IQ4_NL_LEVELS.windows(2) {
            assert!(w[0] < w[1], "IQ4_NL_LEVELS must be strictly increasing");
        }
        for &l in IQ4_NL_LEVELS.iter() {
            assert!((-127..=127).contains(&(l as i32)));
        }
    }

    #[test]
    fn test_iq4_nl_roundtrip_single_block() {
        let data: Vec<f32> = (0..32).map(|i| (i as f32 - 16.0) * 0.1).collect();
        let packed = quantize_iq4_nl(&data);
        assert_eq!(packed.bytes_len(), 18); // 1 block exactly (2 + 16)
        let restored = packed.to_f32();
        assert_eq!(restored.len(), 32);
        for (a, b) in data.iter().zip(restored.iter()) {
            // An uneven grid gives different steps across the range; threshold
            // selected with a margin relative to the worst table step
            // (max. difference between adjacent levels IQ4_NL_LEVELS = 24, amplitude
            // here amax=1.6 → d≈0.0126 → in units of weight the scatter is small).
            assert!((a - b).abs() < 0.35, "{a} vs {b}");
        }
    }

    #[test]
    fn test_iq4_nl_partial_last_block_padded() {
        let data: Vec<f32> = (0..40).map(|i| (i as f32 - 20.0) * 0.05).collect();
        let packed = quantize_iq4_nl(&data);
        assert_eq!(packed.bytes_len(), 36); // 2 blocks * 18 bytes
        let restored = packed.to_f32();
        assert_eq!(restored.len(), 40);
    }

    #[test]
    fn test_iq4_nl_zero_tensor_safe() {
        let data = vec![0.0f32; 32];
        let packed = quantize_iq4_nl(&data);
        let restored = packed.to_f32();
        assert!(restored.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_iq4_nl_extreme_values_map_to_extreme_level() {
        let mut data = vec![0.0f32; 32];
        data[0] = 100.0;
        data[1] = -100.0;
        let packed = quantize_iq4_nl(&data);
        let restored = packed.to_f32();
        // amax=100 → d=100/127; level 100 should be mapped to
        // the maximum table code is (113), and -100 is the minimum (-127).
        assert!(restored[0] > 0.0);
        assert!(restored[1] < 0.0);
        assert!((restored[0] - 100.0).abs() < 5.0);
        assert!((restored[1] + 100.0).abs() < 5.0);
    }

    #[test]
    fn test_iq4_xs_roundtrip_and_size() {
        let data: Vec<f32> = (0..32).map(|i| (i as f32 - 16.0) * 0.1).collect();
        let packed = quantize_iq4_xs(&data);
        assert_eq!(packed.bytes_len(), DType::IQ4_XS.block_bytes());
        let restored = packed.to_f32();
        assert_eq!(restored.len(), 32);
        for (a, b) in data.iter().zip(restored.iter()) {
            assert!((a - b).abs() < 0.35, "{a} vs {b}");
        }
    }

    #[test]
    fn test_iq4_xs_matches_iq4_nl_precision() {
        // Data shorter than QK_IQ4_XS (256) occupies only the first sub-block
        // one super-block; the remaining 7 sub-blocks quantize the padding
        // zeros and have no effect on the first 32 elements. IQ4_NL quantizes the same
        // 32 weights as your only block with the same level table -
        // with an exact calibration match (amax is the same) codes
        // must match one to one.
        let data: Vec<f32> = (0..32).map(|i| (i as f32 - 16.0) * 0.07).collect();
        let nl = quantize_iq4_nl(&data).to_f32();
        let xs = quantize_iq4_xs(&data).to_f32();

        // The codes must match: since both formats are calibrated
        // via `calibrate_iq4_block`, they choose the same levels.
        // But the dequantized values cannot match EXACTLY by design:
        // NL stores the block scale as f16 directly, XS stores it as a
        // 6-bit sub-scale relative to the overall super-scale. Different
        // representations of one number give a difference of the order of 1e-4.
        let max_diff = nl
            .iter()
            .zip(xs.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        let amax = data.iter().fold(0.0f32, |a, v| a.max(v.abs()));
        assert!(
            max_diff < amax * 1e-3,
            "IQ4_XS and IQ4_NL diverged more than the 6-bit sub-scale allows: \
             max_diff={max_diff}, threshold={}",
            amax * 1e-3
        );
    }

    #[test]
    fn test_iq2_xxs_quantize_roundtrips_via_dispatch() {
        let data: Vec<f32> = (0..256).map(|i| (i as f32 - 128.0) * 0.01).collect();
        let packed = quantize_iq(&data, DType::IQ2_XXS);
        assert!(matches!(packed, IqPacked::Iq2Xxs(_)));
        assert_eq!(packed.numel(), 256);
        assert_eq!(packed.to_f32().len(), 256);
    }

    #[test]
    fn test_iq3_s_quantize_roundtrips_via_dispatch() {
        let data: Vec<f32> = (0..256).map(|i| (i as f32 - 128.0) * 0.01).collect();
        let packed = quantize_iq(&data, DType::IQ3_S);
        assert!(matches!(packed, IqPacked::Iq3S(_)));
        assert_eq!(packed.numel(), 256);
        assert_eq!(packed.to_f32().len(), 256);
    }

    #[test]
    fn test_quantize_iq_dispatches_all_eight() {
        let small = vec![1.0f32; 32];
        let block = vec![1.0f32; 256];
        assert!(matches!(
            quantize_iq(&small, DType::IQ4_NL),
            IqPacked::Iq4Nl(_)
        ));
        assert!(matches!(
            quantize_iq(&small, DType::IQ4_XS),
            IqPacked::Iq4Xs(_)
        ));
        assert!(matches!(
            quantize_iq(&block, DType::IQ2_XXS),
            IqPacked::Iq2Xxs(_)
        ));
        assert!(matches!(
            quantize_iq(&block, DType::IQ2_XS),
            IqPacked::Iq2Xs(_)
        ));
        assert!(matches!(
            quantize_iq(&block, DType::IQ3_XXS),
            IqPacked::Iq3Xxs(_)
        ));
        assert!(matches!(
            quantize_iq(&block, DType::IQ3_S),
            IqPacked::Iq3S(_)
        ));
        assert!(matches!(
            quantize_iq(&block, DType::IQ1_S),
            IqPacked::Iq1S(_)
        ));
        assert!(matches!(
            quantize_iq(&block, DType::IQ1_M),
            IqPacked::Iq1M(_)
        ));
    }

    #[test]
    fn test_packed_dtype_roundtrip() {
        let data = vec![0.5f32; 32];
        let p = quantize_iq(&data, DType::IQ4_NL);
        assert_eq!(p.dtype(), DType::IQ4_NL);
        assert_eq!(p.numel(), 32);
        assert_eq!(p.to_f32().len(), 32);
    }

    // ── IQ1_S/IQ1_M through a single entry point (see fix on IqPacked/is_iq_series) ──

    #[test]
    fn test_iq1_s_quantize_roundtrips_via_dispatch() {
        let data: Vec<f32> = (0..256).map(|i| (i as f32 - 128.0) * 0.01).collect();
        let packed = quantize_iq(&data, DType::IQ1_S);
        assert!(matches!(packed, IqPacked::Iq1S(_)));
        assert_eq!(packed.dtype(), DType::IQ1_S);
        assert_eq!(packed.numel(), 256);
        assert_eq!(packed.to_f32().len(), 256);
    }

    #[test]
    fn test_iq1_m_quantize_roundtrips_via_dispatch() {
        let data: Vec<f32> = (0..256).map(|i| (i as f32 - 128.0) * 0.01).collect();
        let packed = quantize_iq(&data, DType::IQ1_M);
        assert!(matches!(packed, IqPacked::Iq1M(_)));
        assert_eq!(packed.dtype(), DType::IQ1_M);
        assert_eq!(packed.numel(), 256);
        assert_eq!(packed.to_f32().len(), 256);
    }

    #[test]
    fn test_dequantize_iq_matches_to_f32_for_iq1() {
        let data: Vec<f32> = (0..256).map(|i| (i as f32 - 128.0) * 0.01).collect();
        let packed = quantize_iq(&data, DType::IQ1_S);
        assert_eq!(dequantize_iq(&packed), packed.to_f32());
    }
}
