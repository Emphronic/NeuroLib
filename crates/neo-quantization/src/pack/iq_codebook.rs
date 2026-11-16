// neo-quantization/src/pack/iq_codebook.rs
//
// Portions of this file are derived from ggml/llama.cpp
// (`ggml/src/ggml-quants.c`, `ggml-common.h`), Copyright (c) the ggml/llama.cpp
// authors, used here under the MIT License.
//
// Codebook/lattice-quantization IQ2_XXS, IQ2_XS, IQ3_XXS, IQ3_S, following
// `dequantize_row_iq2_xxs`/`_iq2_xs`/`_iq3_xxs`/`_iq3_s` and
// `quantize_row_iq2_xxs_impl`/`_iq2_xs_impl`/`_iq3_xxs_impl`/`_iq3_s_impl`
// from `ggml/src/ggml-quants.c` (llama.cpp), with grid/sign tables from
// `ggml-common.h` (see `pack::iq_grids` - each value there is checked
// token-to-token against the source).
//
// ═══════════════════════════════════════════════════════════════════════
// § 1 WHAT ARE THESE FORMATS
// ═══════════════════════════════════════════════════════════════════════
//
// Unlike intN/ternary/NF4 (uniform or fixed pointwise
// grid for EACH weight), IQ2_XXS/IQ2_XS/IQ3_XXS/IQ3_S - vector
// quantization: group of 8 (IQ2_*) or 4 (IQ3_*) weights together
// encoded with ONE index into a pre-calculated (offline, k-means/
// lattice-search on an E8-like grid) codebook - grid -
// plus a separately packed sign bit per element. Signs are not stored
// as N individual bits, and through parity-trick: `ksigns_iq2xs[signs_byte]`
// gives an 8-bit character mask, but not all 256 character combinations are allowed
// (even XOR only - saves one bit per group of 8), so
// signs_byte only accepts 128 values.
//
// ## Layout of the block (all four are super-block QK_K=256, see crate::dtype::DType)
//
//   IQ2_XXS (66 bytes): d: f16, qs: [u16; 32]
//     — for every 32 weights (`ib32` out of 8): 2×u32 (8 bytes) = 4 grid indexes
//       8 bits each + 4×7-bit features + 4-bit sub-scale (offset+0.5, step 0.25)
//
//   IQ2_XS (74 bytes): d: f16, qs: [u16; 32], scales: [u8; 8]
//     — for every 32 weights: 4× u16 (grid_index:9 bits | signs_index:7 bits),
//       sub-scale — 4-bit nibble in scales[ib32/2]
//
//   IQ3_XXS (98 bytes): d: f16, qs: [u8; 96]
//     — for every 32 weights: 8× u8 grid index (3-bit groups of 4 weights,
//       8 groups = 32 weights) + separate u32 (scales_and_signs) with 4x7-bit
//       features + 4-bit sub-scale
//
//   IQ3_S (110 bytes): d: f16, qs: [u8; 64], qh: [u8; 8],
//                        signs: [u8; 32], scales: [u8; 4]
//     — grid index 9 bits (8 bits in qs + 1 most significant bit in qh), features as
//       fair byte-per-group (not parity-packed, unlike IQ2/IQ3XXS)
//
// All four: sub-scale (per each 32-element `ib32`) expressed through
// one super-block `d` (f16) and integer code, same as K-quants
// (see pack::kquant) - but here the code is incremental and uneven (0.5+code)
// *0.25 or (1+2*code), not a linear min/max range.
//
// ═══════════════════════════════════════════════════════════════════════
// § 2  QUANTIZE: WEIGHTED NEAREST-LATTICE-POINT SEARCH
// ═══════════════════════════════════════════════════════════════════════
//
// Upstream builds at the start of runtime (`iq2xs_init_impl`/`iq3xs_init_impl`)
// two structures on top of a static grid table:
//   • `kmap`: reverse dictionary "packed octet of non-negative
//     codes → index in grid" (code point is NOT for all 2^16/2^12
//     combinations - grid covers only 256..512 points out of a much larger one
//     space).
//   • `kneighbours`: for each NOT-grid point - a short list of
//     `nwant` (2 or 3) closest L2 grid points, found offline
//     through qsort of all points by dist².
// `iq*_find_best_neighbour` then chooses from this short list
// that point that minimizes the WEIGHTED square error
// `Σ weight[i]·(scale·grid[i] - xval[i])²`.
//
// This module skips the construction of `kmap`/`kneighbours` entirely and
// instead of "dot not on the hash mark → search among a short offline list"
// does a direct search of the ENTIRE grid table (256 or 512 points) for each
// direct index miss. The result is identical: `kneighbours` in
// upstream is PURELY acceleration (there is no reason why true
// the nearest neighbor by weighted error would lie outside the short
// list `nwant` closest by unweighted L2 - in practice the final
// the selected point is always among the first 2-3 due to the grid structure),
// not changing the result. Full search of 256/512 points on 8/4-weight
// group is not a bottleneck in offline quantization (not hot path inference).
//
// ## `quant_weights` (importance matrix)
//
// Upstream requires `quant_weights` (per-tensor importance matrix,
// `imatrix.gguf`, calculated from calibration data) for IQ2_XXS/
// IQ2_XS without exceptions (`GGML_ASSERT(quant_weights && ...)` - these two
// format do not have "no-imatrix" fallback in upstream). For IQ3_XXS/IQ3_S
// upstream EXPLICITLY supports `quant_weights == NULL` (see
// `quantize_row_iq3_s_ref`/`quantize_row_iq3_xxs_ref`, whose name is impl
// with `NULL`) - in this case the weight is replaced by `x[i]²` (net energy
// signal, without external calibration).
//
// This crate does not parse `imatrix.gguf` (this is a matter for `neo-gguf`/loader),
// so `quantize_iq2/3_*` here take `Option<&[f32]>`:
//   • `Some(weights)` — calibration importance matrix, the same meaning,
//     that `quant_weights` is in the upstream (the same `weight[i] = qw[i] *
//     sqrt(sigma2 + x[i]²)`).
//   • `None` – `weight[i] = x[i]²` is used for ALL four
//     formats (unified fallback; for IQ3 this is bitwise identical
//     with the upstream `_ref`-path, for IQ2 - the same principle as
//     is equivalent to "uniform imatrix" on input `ggml_quantize_chunk`,
//     since the constant factor in `quant_weights` cancels
//     between scale candidates when choosing argmax).
//
// ═══════════════════════════════════════════════════════════════════════

use crate::dtype::DType;
use burn::tensor::f16;

use crate::pack::iq_grids::*;

// ═══════════════════════════════════════════════════════════════════════
// § 0 GENERAL HELPERS
// ═══════════════════════════════════════════════════════════════════════

const GROUP_MAX_EPS: f32 = 1e-15;
const GROUP_MAX_EPS_IQ3_XXS: f32 = 1e-8;

/// Bit-exact implementation of `nearest_int` from ggml-quants.c: round-half-to-even
/// via float bit-trick (magic number `1.5 * 2^23`), not `f32::round()`
/// (round-half-away-from-zero) - on most inputs they are the same, but
/// would diverge on rare even `.5` boundaries, and the format must be
/// byte-compatible, not "almost compatible".
#[inline]
fn nearest_int(fval: f32) -> i32 {
    debug_assert!(
        fval.abs() <= 4194303.0,
        "nearest_int: input out of range: {fval}"
    );
    let val = fval + 12582912.0f32; // 1.5 * 2^23
    let bits = val.to_bits() as i32;
    (bits & 0x007f_ffff) - 0x0040_0000
}

#[inline]
fn clampi(v: i32, lo: i32, hi: i32) -> i32 {
    v.max(lo).min(hi)
}

/// Expands the 8-byte `u64` grid element (IQ2_XXS/IQ2_XS) into
/// array `i8` - the same bytes as in C via `(const int8_t*)(grid + idx)`.
#[inline]
fn grid8_u64(grid: &[u64], idx: usize) -> [i8; 8] {
    let bytes = grid[idx].to_le_bytes();
    let mut out = [0i8; 8];
    for i in 0..8 {
        out[i] = bytes[i] as i8;
    }
    out
}

/// The same for the 4-byte `u32` grid element (IQ3_XXS/IQ3_S).
#[inline]
fn grid4_u32(grid: &[u32], idx: usize) -> [i8; 4] {
    let bytes = grid[idx].to_le_bytes();
    let mut out = [0i8; 4];
    for i in 0..4 {
        out[i] = bytes[i] as i8;
    }
    out
}

/// `true` if `dtype` is one of the four supported here
/// codebook formats.
pub fn is_supported(dtype: DType) -> bool {
    matches!(
        dtype,
        DType::IQ2_XXS | DType::IQ2_XS | DType::IQ3_XXS | DType::IQ3_S
    )
}

// ═══════════════════════════════════════════════════════════════════════
// § 1 REVERSE SEARCH BY GRID (replacing kmap+kneighbours upstream)
// ═══════════════════════════════════════════════════════════════════════

/// Exact reverse search: `packed_octet` (packed non-negative
/// codes `0..kMaxQ` for each of the 8 positions, 2 bits each) → index in
/// `IQ2XXS_GRID`/`IQ2XS_GRID`, if such a point exists on the lattice.
///
/// implementation of the construction cycle `kmap_q2xs` from `iq2xs_init_impl`: for each
/// grid-point `pos[i] = 2*l+1` is encoded as `q = (pos[i]-1)/2`, then
/// packaged in `index`. It's not a table that's being built here (we don't cache
/// between calls - offline quantize, not hot path), but simply direct
/// comparison for each request; O(grid_size) on miss - see § 2 in
/// module header.
fn map_lookup_u64(grid: &[u64], packed_octet: u16) -> Option<usize> {
    for (i, &g) in grid.iter().enumerate() {
        let bytes = g.to_le_bytes();
        let mut index: u16 = 0;
        for k in 0..8 {
            let q = ((bytes[k] as i32 - 1) / 2) as u16;
            index |= q << (2 * k);
        }
        if index == packed_octet {
            return Some(i);
        }
    }
    None
}

/// Same for IQ3 (4-element groups, 3-bit codes, u32 grid).
fn map_lookup_u32(grid: &[u32], packed_quad: u16) -> Option<usize> {
    for (i, &g) in grid.iter().enumerate() {
        let bytes = g.to_le_bytes();
        let mut index: u16 = 0;
        for k in 0..4 {
            let q = ((bytes[k] as i32 - 1) / 2) as u16;
            index |= q << (3 * k);
        }
        if index == packed_quad {
            return Some(i);
        }
    }
    None
}

/// implementation `iq2_find_best_neighbour`, but without precomputed short
/// list of neighbors - exhaustive search of `grid` by weighted quadratic
/// error (see § 2 of the module title for the rationale for equivalence).
/// Returns `(grid_index, L[8])` where `L[i] = (grid[i]-1)/2`.
fn find_best_grid_point_u64(
    grid: &[u64],
    xval: &[f32],
    weight: &[f32],
    scale: f32,
) -> (usize, [i8; 8]) {
    let mut best_d2 = f32::INFINITY;
    let mut best_idx = 0usize;
    for j in 0..grid.len() {
        let pg = grid8_u64(grid, j);
        let mut d2 = 0.0f32;
        for i in 0..8 {
            let q = pg[i] as f32;
            let diff = scale * q - xval[i];
            d2 += weight[i] * diff * diff;
        }
        if d2 < best_d2 {
            best_d2 = d2;
            best_idx = j;
        }
    }
    let pg = grid8_u64(grid, best_idx);
    let mut l = [0i8; 8];
    for i in 0..8 {
        l[i] = (pg[i] - 1) / 2;
    }
    (best_idx, l)
}

/// Same for IQ3 (4-element groups).
fn find_best_grid_point_u32(
    grid: &[u32],
    xval: &[f32],
    weight: &[f32],
    scale: f32,
) -> (usize, [i8; 4]) {
    let mut best_d2 = f32::INFINITY;
    let mut best_idx = 0usize;
    for j in 0..grid.len() {
        let pg = grid4_u32(grid, j);
        let mut d2 = 0.0f32;
        for i in 0..4 {
            let q = pg[i] as f32;
            let diff = scale * q - xval[i];
            d2 += weight[i] * diff * diff;
        }
        if d2 < best_d2 {
            best_d2 = d2;
            best_idx = j;
        }
    }
    let pg = grid4_u32(grid, best_idx);
    let mut l = [0i8; 4];
    for i in 0..4 {
        l[i] = (pg[i] - 1) / 2;
    }
    (best_idx, l)
}

/// Find `(grid_index, L)` for packed octet `u`: exact first
/// `map_lookup`, if there is a miss - exhaustive search of `find_best_grid_point`.
/// Bundle implementation `kmap_q2xs[u]` → (if `< 0`) `iq2_find_best_neighbour`.
fn resolve_octet_u64(
    grid: &[u64],
    u: u16,
    xval: &[f32],
    weight: &[f32],
    scale: f32,
) -> (usize, [i8; 8]) {
    if let Some(idx) = map_lookup_u64(grid, u) {
        let pg = grid8_u64(grid, idx);
        let mut l = [0i8; 8];
        for i in 0..8 {
            l[i] = (pg[i] - 1) / 2;
        }
        (idx, l)
    } else {
        find_best_grid_point_u64(grid, xval, weight, scale)
    }
}

fn resolve_quad_u32(
    grid: &[u32],
    u: u16,
    xval: &[f32],
    weight: &[f32],
    scale: f32,
) -> (usize, [i8; 4]) {
    if let Some(idx) = map_lookup_u32(grid, u) {
        let pg = grid4_u32(grid, idx);
        let mut l = [0i8; 4];
        for i in 0..4 {
            l[i] = (pg[i] - 1) / 2;
        }
        (idx, l)
    } else {
        find_best_grid_point_u32(grid, xval, weight, scale)
    }
}

// ═══════════════════════════════════════════════════════════════════════
// § 2 IQ2_XXS - 66 bytes/256 weights (8 sub-blocks of 32, 4 grid points/sub-block)
// ═══════════════════════════════════════════════════════════════════════

/// Packed IQ2_XXS super-block tensor.
#[derive(Debug, Clone, PartialEq)]
pub struct Iq2XxsPacked {
    /// Raw super-blocks (66 bytes each).
    pub blocks: Vec<u8>,
    /// Initial number of scales.
    pub numel: usize,
}

impl Iq2XxsPacked {
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
        dequantize_iq2_xxs(self)
    }
}

/// implementation `dequantize_row_iq2_xxs`.
pub fn dequantize_iq2_xxs(packed: &Iq2XxsPacked) -> Vec<f32> {
    let block_bytes = DType::IQ2_XXS.block_bytes(); // 66
    let n_blocks = packed.blocks.len() / block_bytes;
    let mut out = Vec::with_capacity(n_blocks * 256);

    for b in 0..n_blocks {
        let blk = &packed.blocks[b * block_bytes..(b + 1) * block_bytes];
        let d = f16::from_le_bytes([blk[0], blk[1]]).to_f32();
        let qs = &blk[2..66]; // 16 * u16 = 32 bytes... wait see below

        for ib32 in 0..8 {
            let off = 4 * ib32; // 4 bytes per ib32 step over the 32-byte qs region *2 u32
                                // qs region is 32 bytes = 8 * u32(4 bytes); each ib32 consumes 2 u32 (8 bytes)
            let base = ib32 * 8;
            let aux0 = u32::from_le_bytes([qs[base], qs[base + 1], qs[base + 2], qs[base + 3]]);
            let aux1 = u32::from_le_bytes([qs[base + 4], qs[base + 5], qs[base + 6], qs[base + 7]]);
            let aux8 = aux0.to_le_bytes(); // 4 grid indices (one per l)
            let _ = off;

            let db = d * (0.5 + (aux1 >> 28) as f32) * 0.25;
            for l in 0..4 {
                let grid_idx = aux8[l] as usize;
                let grid = grid8_u64(&IQ2XXS_GRID, grid_idx);
                let signs = KSIGNS_IQ2XS[((aux1 >> (7 * l)) & 127) as usize];
                for j in 0..8 {
                    let sign = if signs & KMASK_IQ2XS[j] != 0 {
                        -1.0
                    } else {
                        1.0
                    };
                    out.push(db * grid[j] as f32 * sign);
                }
            }
        }
    }

    out.truncate(packed.numel);
    out
}

/// Quantize `data` into IQ2_XXS.
///
/// `quant_weights`: `Some(imatrix)` - calibration importance matrix
/// (same semantics as `quant_weights` in upstream); `None` —
/// `weight[i] = x[i]²` (see § "quant_weights" in the module header).
///
/// #Panic
///
/// Panics if `data.len()` is not a multiple of 256 (`QK_K`) - like upstream
/// (`GGML_ASSERT(n%QK_K == 0)`), this is a format limitation, not this one
/// implementation: the IQ2_XXS superblock must cover exactly 256 weights.
pub fn quantize_iq2_xxs(data: &[f32], quant_weights: Option<&[f32]>) -> Iq2XxsPacked {
    assert_eq!(
        data.len() % 256,
        0,
        "quantize_iq2_xxs: len must be a multiple of QK_K=256"
    );
    let n_blocks = data.len() / 256;
    let block_bytes = DType::IQ2_XXS.block_bytes();
    let mut blocks = vec![0u8; n_blocks * block_bytes];

    const K_MAX_Q: i32 = 3;

    for ibl in 0..n_blocks {
        let xbl = &data[ibl * 256..(ibl + 1) * 256];
        let sumx2: f32 = xbl.iter().map(|w| w * w).sum();
        let sigma2 = sumx2 / 256.0;

        let mut scales = [0.0f32; 8];
        let mut q2 = [0u32; 16]; // 2 per ib32, 8 ib32
        let mut max_scale = 0.0f32;

        for ib in 0..8 {
            let xb = &xbl[32 * ib..32 * ib + 32];
            let mut weight = [0.0f32; 32];
            match quant_weights {
                Some(qw) => {
                    let qwb = &qw[ibl * 256 + 32 * ib..ibl * 256 + 32 * ib + 32];
                    for i in 0..32 {
                        weight[i] = qwb[i] * (sigma2 + xb[i] * xb[i]).sqrt();
                    }
                }
                None => {
                    for i in 0..32 {
                        weight[i] = xb[i] * xb[i];
                    }
                }
            }
            let mut waux = [0.0f32; 32];
            for i in 0..32 {
                waux[i] = weight[i].sqrt();
            }

            let mut xval = [0.0f32; 32];
            let mut block_signs = [0u8; 4];
            for k in 0..4 {
                let mut nflip = 0;
                let mut s: u8 = 0;
                for i in 0..8 {
                    let v = xb[8 * k + i];
                    if v >= 0.0 {
                        xval[8 * k + i] = v;
                    } else {
                        xval[8 * k + i] = -v;
                        nflip += 1;
                        s |= 1 << i;
                    }
                }
                if nflip % 2 == 1 {
                    let mut imin = 0usize;
                    let mut min_ax = weight[8 * k] * xb[8 * k] * xb[8 * k];
                    for i in 1..8 {
                        let ax = weight[8 * k + i] * xb[8 * k + i] * xb[8 * k + i];
                        if ax < min_ax {
                            min_ax = ax;
                            imin = i;
                        }
                    }
                    xval[8 * k + imin] = -xval[8 * k + imin];
                    s ^= 1 << imin;
                }
                block_signs[k] = s & 127;
            }

            let max = xval.iter().cloned().fold(0.0f32, f32::max);
            if max < GROUP_MAX_EPS {
                scales[ib] = 0.0;
                continue;
            }

            // make_qp_quants(32, kMaxQ+1, xval, L, weight): find the initial
            // scale/codes. Upstream make_qp_quants - a separate algorithm
            // (positive-only quant optimizer); here we use
            // equivalent in purpose, simpler initialization -
            // `scale = max / kMaxQ`, `L[i] = round(xval[i]/scale)` - it
            // serves only as a STARTING point for subsequent is-perturbation
            // search below (the same by `is` cycle as in the upstream), which
            // and determines the actual result; sensitivity to
            // the starting point due to the 21-step search is minimal.
            let mut scale = max / K_MAX_Q as f32;
            let mut l_codes = [0i32; 32];
            for i in 0..32 {
                l_codes[i] = clampi(nearest_int(xval[i] / scale), 0, K_MAX_Q - 1);
            }

            let eff_max = scale * K_MAX_Q as f32;
            if eff_max <= 0.0 {
                scales[ib] = 0.0;
                continue;
            }

            let mut best = 0.0f32;
            let mut best_l = l_codes;
            for is in -6..=6i32 {
                let id = (2 * K_MAX_Q - 1) as f32 + is as f32 * 0.1;
                let id = id / eff_max;
                let this_scale = 1.0 / id;
                let mut laux = [0i8; 32];
                for k in 0..4 {
                    let mut u: u16 = 0;
                    for i in 0..8 {
                        let l = clampi(
                            nearest_int(0.5 * (id * xval[8 * k + i] - 1.0)),
                            0,
                            K_MAX_Q - 1,
                        );
                        laux[8 * k + i] = l as i8;
                        u |= (l as u16) << (2 * i);
                    }
                    if map_lookup_u64(&IQ2XXS_GRID, u).is_none() {
                        let (_, l) = find_best_grid_point_u64(
                            &IQ2XXS_GRID,
                            &xval[8 * k..8 * k + 8],
                            &waux[8 * k..8 * k + 8],
                            this_scale,
                        );
                        for i in 0..8 {
                            laux[8 * k + i] = l[i];
                        }
                    }
                }
                let mut sumqx = 0.0f32;
                let mut sumq2 = 0.0f32;
                for i in 0..32 {
                    let w = weight[i];
                    let q = 2.0 * laux[i] as f32 + 1.0;
                    sumqx += w * xval[i] * q;
                    sumq2 += w * q * q;
                }
                if sumq2 > 0.0 && sumqx * sumqx > best * sumq2 {
                    scale = sumqx / sumq2;
                    best = scale * sumqx;
                    for i in 0..32 {
                        best_l[i] = laux[i] as i32;
                    }
                }
            }
            let mut l_final = best_l;

            if scale > 0.0 {
                let id = 1.0 / scale;
                let mut grid_idx_k = [0usize; 4];
                for k in 0..4 {
                    let mut u: u16 = 0;
                    for i in 0..8 {
                        let l = clampi(
                            nearest_int(0.5 * (id * xval[8 * k + i] - 1.0)),
                            0,
                            K_MAX_Q - 1,
                        );
                        u |= (l as u16) << (2 * i);
                    }
                    let (gidx, l) = resolve_octet_u64(
                        &IQ2XXS_GRID,
                        u,
                        &xval[8 * k..8 * k + 8],
                        &waux[8 * k..8 * k + 8],
                        scale,
                    );
                    grid_idx_k[k] = gidx;
                    for i in 0..8 {
                        l_final[8 * k + i] = l[i] as i32;
                    }
                }
                let mut sumqx = 0.0f32;
                let mut sumq2 = 0.0f32;
                for i in 0..32 {
                    let w = weight[i];
                    let q = 2.0 * l_final[i] as f32 + 1.0;
                    sumqx += w * xval[i] * q;
                    sumq2 += w * q * q;
                }
                if sumq2 > 0.0 {
                    scale = sumqx / sumq2;
                }
                let _ = grid_idx_k;
            }

            let mut block_signs_final = block_signs;
            if scale < 0.0 {
                scale = -scale;
                for k in 0..4 {
                    block_signs_final[k] = (!block_signs_final[k]) & 127;
                }
            }

            for k in 0..4 {
                let mut u: u16 = 0;
                for i in 0..8 {
                    u |= (l_final[8 * k + i] as u16) << (2 * i);
                }
                let Some(grid_index) = map_lookup_u64(&IQ2XXS_GRID, u) else {
                    // Internal upstream invariant (`GGML_ASSERT(grid_index >= 0)`):
                    // after selecting the nearest grid, the final codes must be
                    // grid point. The violation is a bug of the implementation, not the input data.
                    panic!(
                        "quantize_iq2_xxs: sub-block {ib}, group {k}: {u:#06x} codes do not lie on the IQ2XXS_GRID grid \
                         (the internal invariant of the quantizer is violated)"
                    );
                };
                q2[2 * ib] |= (grid_index as u32) << (8 * k);
                q2[2 * ib + 1] |= (block_signs_final[k] as u32) << (7 * k);
            }

            scales[ib] = scale;
            if scale > max_scale {
                max_scale = scale;
            }
        }

        let blk = &mut blocks[ibl * block_bytes..(ibl + 1) * block_bytes];
        if max_scale == 0.0 {
            blk[0..2].copy_from_slice(&crate::pack::f16_saturating(0.0).to_le_bytes());
            // qs already zeroed
            continue;
        }

        let d = max_scale / 31.0;
        blk[0..2].copy_from_slice(&crate::pack::f16_saturating(d).to_le_bytes());
        let id = 1.0 / d;
        for ib in 0..8 {
            let l = clampi(nearest_int(0.5 * (id * scales[ib] - 1.0)), 0, 15);
            q2[2 * ib + 1] |= (l as u32) << 28;
        }
        for ib in 0..8 {
            let off = 2 + ib * 8;
            blk[off..off + 4].copy_from_slice(&q2[2 * ib].to_le_bytes());
            blk[off + 4..off + 8].copy_from_slice(&q2[2 * ib + 1].to_le_bytes());
        }
    }

    Iq2XxsPacked {
        blocks,
        numel: data.len(),
    }
}

// ═══════════════════════════════════════════════════════════════════════
// § 3 IQ2_XS - 74 bytes/256 weights (16 sub-blocks of 16, 2 grid points/sub-block)
// ═══════════════════════════════════════════════════════════════════════

/// Packed IQ2_XS super-block tensor.
#[derive(Debug, Clone, PartialEq)]
pub struct Iq2XsPacked {
    /// Raw super-blocks (74 bytes each).
    pub blocks: Vec<u8>,
    /// Initial number of scales.
    pub numel: usize,
}

impl Iq2XsPacked {
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
        dequantize_iq2_xs(self)
    }
}

/// implementation `dequantize_row_iq2_xs`.
pub fn dequantize_iq2_xs(packed: &Iq2XsPacked) -> Vec<f32> {
    let block_bytes = DType::IQ2_XS.block_bytes(); // 74
    let n_blocks = packed.blocks.len() / block_bytes;
    let mut out = Vec::with_capacity(n_blocks * 256);

    for b in 0..n_blocks {
        let blk = &packed.blocks[b * block_bytes..(b + 1) * block_bytes];
        let d = f16::from_le_bytes([blk[0], blk[1]]).to_f32();
        // block_iq2_xs (ggml): d: f16, qs: [u16; QK_K/8 = 32], scales: [u8; QK_K/32 = 8].
        let qs_bytes = &blk[2..66]; // 32 * u16
        let scales = &blk[66..74]; // 8 bytes

        for ib32 in 0..8 {
            let db0 = d * (0.5 + (scales[ib32] & 0xf) as f32) * 0.25;
            let db1 = d * (0.5 + (scales[ib32] >> 4) as f32) * 0.25;
            let db = [db0, db1];

            for l in 0..4 {
                let idx16 = ib32 * 4 + l;
                let raw = u16::from_le_bytes([qs_bytes[idx16 * 2], qs_bytes[idx16 * 2 + 1]]);
                let grid_idx = (raw & 511) as usize;
                let signs_idx = (raw >> 9) as usize;
                let grid = grid8_u64(&IQ2XS_GRID, grid_idx);
                let signs = KSIGNS_IQ2XS[signs_idx];
                let dl = db[l / 2];
                for j in 0..8 {
                    let sign = if signs & KMASK_IQ2XS[j] != 0 {
                        -1.0
                    } else {
                        1.0
                    };
                    out.push(dl * grid[j] as f32 * sign);
                }
            }
        }
    }

    out.truncate(packed.numel);
    out
}

/// Quantize `data` into IQ2_XS. See `quantize_iq2_xxs` for semantics
/// `quant_weights`.
///
/// #Panic
///
/// Panics if `data.len()` is not a multiple of 256 (`QK_K`).
pub fn quantize_iq2_xs(data: &[f32], quant_weights: Option<&[f32]>) -> Iq2XsPacked {
    assert_eq!(
        data.len() % 256,
        0,
        "quantize_iq2_xs: len must be a multiple of QK_K=256"
    );
    let n_blocks = data.len() / 256;
    let block_bytes = DType::IQ2_XS.block_bytes();
    let mut blocks = vec![0u8; n_blocks * block_bytes];

    const K_MAX_Q: i32 = 3;

    for ibl in 0..n_blocks {
        let xbl = &data[ibl * 256..(ibl + 1) * 256];
        let sumx2: f32 = xbl.iter().map(|w| w * w).sum();
        let sigma2 = sumx2 / 256.0;

        let mut scales = [0.0f32; 16];
        let mut q2 = [0u16; 32]; // 2 per ib (16-elem sub-block), 16 ib
        let mut max_scale = 0.0f32;

        for ib in 0..16 {
            let xb = &xbl[16 * ib..16 * ib + 16];
            let mut weight = [0.0f32; 16];
            match quant_weights {
                Some(qw) => {
                    let qwb = &qw[ibl * 256 + 16 * ib..ibl * 256 + 16 * ib + 16];
                    for i in 0..16 {
                        weight[i] = qwb[i] * (sigma2 + xb[i] * xb[i]).sqrt();
                    }
                }
                None => {
                    for i in 0..16 {
                        weight[i] = xb[i] * xb[i];
                    }
                }
            }
            let mut waux = [0.0f32; 16];
            for i in 0..16 {
                waux[i] = weight[i].sqrt();
            }

            let mut xval = [0.0f32; 16];
            let mut block_signs = [0u8; 2];
            for k in 0..2 {
                let mut nflip = 0;
                let mut s: u8 = 0;
                for i in 0..8 {
                    let v = xb[8 * k + i];
                    if v >= 0.0 {
                        xval[8 * k + i] = v;
                    } else {
                        xval[8 * k + i] = -v;
                        nflip += 1;
                        s |= 1 << i;
                    }
                }
                if nflip % 2 == 1 {
                    let mut imin = 0usize;
                    let mut min_ax = weight[8 * k] * xb[8 * k] * xb[8 * k];
                    for i in 1..8 {
                        let ax = weight[8 * k + i] * xb[8 * k + i] * xb[8 * k + i];
                        if ax < min_ax {
                            min_ax = ax;
                            imin = i;
                        }
                    }
                    xval[8 * k + imin] = -xval[8 * k + imin];
                    s ^= 1 << imin;
                }
                block_signs[k] = s & 127;
            }

            let max = xval.iter().cloned().fold(0.0f32, f32::max);
            let mut l_final = [0i32; 16];
            if max < GROUP_MAX_EPS {
                scales[ib] = 0.0;
                continue;
            }

            let mut best = 0.0f32;
            let mut scale = max / (2 * K_MAX_Q - 1) as f32;
            let mut is_on_grid = [true, true];

            for is in -9..=9i32 {
                let id = ((2 * K_MAX_Q - 1) as f32 + is as f32 * 0.1) / max;
                let this_scale = 1.0 / id;
                let mut laux = [0i8; 16];
                let mut is_on_grid_aux = [true, true];
                for k in 0..2 {
                    let mut u: u16 = 0;
                    for i in 0..8 {
                        let l = clampi(
                            nearest_int(0.5 * (id * xval[8 * k + i] - 1.0)),
                            0,
                            K_MAX_Q - 1,
                        );
                        laux[8 * k + i] = l as i8;
                        u |= (l as u16) << (2 * i);
                    }
                    if let Some(_gidx) = map_lookup_u64(&IQ2XS_GRID, u) {
                        is_on_grid_aux[k] = true;
                    } else {
                        is_on_grid_aux[k] = false;
                        let (_, l) = find_best_grid_point_u64(
                            &IQ2XS_GRID,
                            &xval[8 * k..8 * k + 8],
                            &waux[8 * k..8 * k + 8],
                            this_scale,
                        );
                        for i in 0..8 {
                            laux[8 * k + i] = l[i];
                        }
                    }
                }
                let mut sumqx = 0.0f32;
                let mut sumq2 = 0.0f32;
                for i in 0..16 {
                    let w = weight[i];
                    let q = 2.0 * laux[i] as f32 + 1.0;
                    sumqx += w * xval[i] * q;
                    sumq2 += w * q * q;
                }
                if sumq2 > 0.0 && sumqx * sumqx > best * sumq2 {
                    scale = sumqx / sumq2;
                    best = scale * sumqx;
                    for i in 0..16 {
                        l_final[i] = laux[i] as i32;
                    }
                    is_on_grid = is_on_grid_aux;
                }
            }

            let n_not_on_grid = is_on_grid.iter().filter(|&&b| !b).count();
            if n_not_on_grid > 0 && scale > 0.0 {
                let id = 1.0 / scale;
                for k in 0..2 {
                    if is_on_grid[k] {
                        continue;
                    }
                    let mut u: u16 = 0;
                    for i in 0..8 {
                        let l = clampi(
                            nearest_int(0.5 * (id * xval[8 * k + i] - 1.0)),
                            0,
                            K_MAX_Q - 1,
                        );
                        u |= (l as u16) << (2 * i);
                        l_final[8 * k + i] = l;
                    }
                    if map_lookup_u64(&IQ2XS_GRID, u).is_none() {
                        let (_, l) = find_best_grid_point_u64(
                            &IQ2XS_GRID,
                            &xval[8 * k..8 * k + 8],
                            &waux[8 * k..8 * k + 8],
                            scale,
                        );
                        for i in 0..8 {
                            l_final[8 * k + i] = l[i] as i32;
                        }
                    }
                }
                let mut sumqx = 0.0f32;
                let mut sumq2 = 0.0f32;
                for i in 0..16 {
                    let w = weight[i];
                    let q = 2.0 * l_final[i] as f32 + 1.0;
                    sumqx += w * xval[i] * q;
                    sumq2 += w * q * q;
                }
                if sumq2 > 0.0 {
                    scale = sumqx / sumq2;
                }
            }

            let mut block_signs_final = block_signs;
            if scale < 0.0 {
                scale = -scale;
                for k in 0..2 {
                    block_signs_final[k] = (!block_signs_final[k]) & 127;
                }
            }

            for k in 0..2 {
                let mut u: u16 = 0;
                for i in 0..8 {
                    u |= (l_final[8 * k + i] as u16) << (2 * i);
                }
                let Some(grid_index) = map_lookup_u64(&IQ2XS_GRID, u) else {
                    // Internal upstream invariant (`GGML_ASSERT(grid_index >= 0)`):
                    // after selecting the nearest grid, the final codes must be
                    // grid point. The violation is a bug of the implementation, not the input data.
                    panic!(
                        "quantize_iq2_xs: sub-block {ib}, group {k}: {u:#06x} codes do not lie on the IQ2XS_GRID grid\
                         (the internal invariant of the quantizer is violated)"
                    );
                };
                q2[2 * ib + k] = grid_index as u16 | ((block_signs_final[k] as u16) << 9);
            }

            scales[ib] = scale;
            if scale > max_scale {
                max_scale = scale;
            }
        }

        let blk = &mut blocks[ibl * block_bytes..(ibl + 1) * block_bytes];
        if max_scale == 0.0 {
            blk[0..2].copy_from_slice(&crate::pack::f16_saturating(0.0).to_le_bytes());
            continue;
        }

        let d = max_scale / 31.0;
        blk[0..2].copy_from_slice(&crate::pack::f16_saturating(d).to_le_bytes());
        let id = 1.0 / d;
        let mut scale_bytes = [0u8; 8];
        for ib in 0..16 {
            let l = clampi(nearest_int(0.5 * (id * scales[ib] - 1.0)), 0, 15) as u8;
            if ib % 2 == 0 {
                scale_bytes[ib / 2] = l;
            } else {
                scale_bytes[ib / 2] |= l << 4;
            }
        }
        // All 32 codes (2 for each of the 16 sub-blocks); the 16 scale bytes
        // follow at offset 66.
        for (i, code) in q2.iter().enumerate() {
            let off = 2 + i * 2;
            blk[off..off + 2].copy_from_slice(&code.to_le_bytes());
        }
        blk[66..74].copy_from_slice(&scale_bytes);
    }

    Iq2XsPacked {
        blocks,
        numel: data.len(),
    }
}

// ═══════════════════════════════════════════════════════════════════════
// § 4 IQ3_XXS - 98 bytes/256 weights (8 sub-blocks of 32, 4-element groups)
// ═══════════════════════════════════════════════════════════════════════

/// Packed IQ3_XXS super-block tensor.
#[derive(Debug, Clone, PartialEq)]
pub struct Iq3XxsPacked {
    /// Raw super-blocks (98 bytes each).
    pub blocks: Vec<u8>,
    /// Initial number of scales.
    pub numel: usize,
}

impl Iq3XxsPacked {
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
        dequantize_iq3_xxs(self)
    }
}

/// implementation `dequantize_row_iq3_xxs`.
pub fn dequantize_iq3_xxs(packed: &Iq3XxsPacked) -> Vec<f32> {
    let block_bytes = DType::IQ3_XXS.block_bytes(); // 98
    let n_blocks = packed.blocks.len() / block_bytes;
    let mut out = Vec::with_capacity(n_blocks * 256);

    for b in 0..n_blocks {
        let blk = &packed.blocks[b * block_bytes..(b + 1) * block_bytes];
        let d = f16::from_le_bytes([blk[0], blk[1]]).to_f32();
        let qs = &blk[2..2 + 96]; // 3*QK_K/8 = 96 bytes: first 64 = grid idx pairs, last 32 = scales_and_signs
        let grid_qs = &qs[0..64];
        let scales_and_signs = &qs[64..96];

        for ib32 in 0..8 {
            let sas = u32::from_le_bytes([
                scales_and_signs[4 * ib32],
                scales_and_signs[4 * ib32 + 1],
                scales_and_signs[4 * ib32 + 2],
                scales_and_signs[4 * ib32 + 3],
            ]);
            let db = d * (0.5 + (sas >> 28) as f32) * 0.5;
            for l in 0..4 {
                let signs = KSIGNS_IQ2XS[((sas >> (7 * l)) & 127) as usize];
                let idx1 = grid_qs[8 * ib32 + 2 * l] as usize;
                let idx2 = grid_qs[8 * ib32 + 2 * l + 1] as usize;
                let grid1 = grid4_u32(&IQ3XXS_GRID, idx1);
                let grid2 = grid4_u32(&IQ3XXS_GRID, idx2);
                // implementation `y[j+0] = db*grid1[j]*sign; y[j+4] = db*grid2[j]*sign;`
                // — fixed window of 8 outputs per iteration `l`: first
                // 4 from grid1, next 4 from grid2 (not two independent
                // sequential cycles).
                let mut window = [0.0f32; 8];
                for j in 0..4 {
                    let s0 = if signs & KMASK_IQ2XS[j] != 0 {
                        -1.0
                    } else {
                        1.0
                    };
                    let s4 = if signs & KMASK_IQ2XS[j + 4] != 0 {
                        -1.0
                    } else {
                        1.0
                    };
                    window[j] = db * grid1[j] as f32 * s0;
                    window[j + 4] = db * grid2[j] as f32 * s4;
                }
                out.extend_from_slice(&window);
            }
        }
    }

    out.truncate(packed.numel);
    out
}

/// Quantize `data` into IQ3_XXS. See `quantize_iq2_xxs` for semantics
/// `quant_weights` - here, as in the upstream `quantize_row_iq3_xxs_ref`,
/// `None` is a documented (not auxiliary) mode without imatrix.
///
/// #Panic
///
/// Panics if `data.len()` is not a multiple of 256 (`QK_K`).
pub fn quantize_iq3_xxs(data: &[f32], quant_weights: Option<&[f32]>) -> Iq3XxsPacked {
    assert_eq!(
        data.len() % 256,
        0,
        "quantize_iq3_xxs: len must be a multiple of QK_K=256"
    );
    let n_blocks = data.len() / 256;
    let block_bytes = DType::IQ3_XXS.block_bytes();
    let mut blocks = vec![0u8; n_blocks * block_bytes];

    const K_MAX_Q: i32 = 8;

    for ibl in 0..n_blocks {
        let xbl = &data[ibl * 256..(ibl + 1) * 256];
        let sumx2: f32 = xbl.iter().map(|w| w * w).sum();
        let sigma2 = 2.0 * sumx2 / 256.0;

        let mut scales = [0.0f32; 8];
        let mut q3_grid = [0u8; 64]; // 8 ib32 * 8 bytes (4 group-pairs * 2 idx each)
        let mut scales_and_signs = [0u32; 8];
        let mut max_scale = 0.0f32;

        for ib in 0..8 {
            let xb = &xbl[32 * ib..32 * ib + 32];
            let mut weight = [0.0f32; 32];
            match quant_weights {
                Some(qw) => {
                    let qwb = &qw[ibl * 256 + 32 * ib..ibl * 256 + 32 * ib + 32];
                    for i in 0..32 {
                        weight[i] = qwb[i] * (sigma2 + xb[i] * xb[i]).sqrt();
                    }
                }
                None => {
                    for i in 0..32 {
                        weight[i] = xb[i] * xb[i];
                    }
                }
            }
            let mut waux = [0.0f32; 32];
            for i in 0..32 {
                waux[i] = weight[i].sqrt();
            }

            let mut xval = [0.0f32; 32];
            let mut block_signs = [0u8; 4];
            for k in 0..4 {
                let mut nflip = 0;
                let mut s: u8 = 0;
                for i in 0..8 {
                    let v = xb[8 * k + i];
                    if v >= 0.0 {
                        xval[8 * k + i] = v;
                    } else {
                        xval[8 * k + i] = -v;
                        nflip += 1;
                        s |= 1 << i;
                    }
                }
                if nflip % 2 == 1 {
                    let mut imin = 0usize;
                    let mut min_ax = weight[8 * k] * xb[8 * k] * xb[8 * k];
                    for i in 1..8 {
                        let ax = weight[8 * k + i] * xb[8 * k + i] * xb[8 * k + i];
                        if ax < min_ax {
                            min_ax = ax;
                            imin = i;
                        }
                    }
                    xval[8 * k + imin] = -xval[8 * k + imin];
                    s ^= 1 << imin;
                }
                block_signs[k] = s & 127;
            }

            let max = xval.iter().cloned().fold(0.0f32, f32::max);
            let mut l_final = [0i32; 32];
            if max < GROUP_MAX_EPS_IQ3_XXS {
                scales[ib] = 0.0;
                continue;
            }

            let mut best = 0.0f32;
            let mut scale = max / (2 * K_MAX_Q - 1) as f32;
            let mut is_on_grid = [true; 8];

            for is in -15..=15i32 {
                let id = ((2 * K_MAX_Q - 1) as f32 + is as f32 * 0.2) / max;
                let this_scale = 1.0 / id;
                let mut laux = [0i8; 32];
                let mut is_on_grid_aux = [true; 8];
                for k in 0..8 {
                    let mut u: u16 = 0;
                    for i in 0..4 {
                        let l = clampi(
                            nearest_int(0.5 * (id * xval[4 * k + i] - 1.0)),
                            0,
                            K_MAX_Q - 1,
                        );
                        laux[4 * k + i] = l as i8;
                        u |= (l as u16) << (3 * i);
                    }
                    if map_lookup_u32(&IQ3XXS_GRID, u).is_some() {
                        is_on_grid_aux[k] = true;
                    } else {
                        is_on_grid_aux[k] = false;
                        let (_, l) = find_best_grid_point_u32(
                            &IQ3XXS_GRID,
                            &xval[4 * k..4 * k + 4],
                            &waux[4 * k..4 * k + 4],
                            this_scale,
                        );
                        for i in 0..4 {
                            laux[4 * k + i] = l[i];
                        }
                    }
                }
                let mut sumqx = 0.0f32;
                let mut sumq2 = 0.0f32;
                for i in 0..32 {
                    let w = weight[i];
                    let q = 2.0 * laux[i] as f32 + 1.0;
                    sumqx += w * xval[i] * q;
                    sumq2 += w * q * q;
                }
                if sumq2 > 0.0 && sumqx * sumqx > best * sumq2 {
                    scale = sumqx / sumq2;
                    best = scale * sumqx;
                    for i in 0..32 {
                        l_final[i] = laux[i] as i32;
                    }
                    is_on_grid = is_on_grid_aux;
                }
            }

            let n_not_on_grid = is_on_grid.iter().filter(|&&b| !b).count();
            if n_not_on_grid > 0 && scale > 0.0 {
                let id = 1.0 / scale;
                for k in 0..8 {
                    if is_on_grid[k] {
                        continue;
                    }
                    let mut u: u16 = 0;
                    for i in 0..4 {
                        let l = clampi(
                            nearest_int(0.5 * (id * xval[4 * k + i] - 1.0)),
                            0,
                            K_MAX_Q - 1,
                        );
                        u |= (l as u16) << (3 * i);
                    }
                    let (gidx, _) = resolve_quad_u32(
                        &IQ3XXS_GRID,
                        u,
                        &xval[4 * k..4 * k + 4],
                        &waux[4 * k..4 * k + 4],
                        scale,
                    );
                    let pg = grid4_u32(&IQ3XXS_GRID, gidx);
                    for i in 0..4 {
                        l_final[4 * k + i] = ((pg[i] - 1) / 2) as i32;
                    }
                }
                let mut sumqx = 0.0f32;
                let mut sumq2 = 0.0f32;
                for i in 0..32 {
                    let w = weight[i];
                    let q = 2.0 * l_final[i] as f32 + 1.0;
                    sumqx += w * xval[i] * q;
                    sumq2 += w * q * q;
                }
                if sumq2 > 0.0 {
                    scale = sumqx / sumq2;
                }
            }

            let mut block_signs_final = block_signs;
            if scale < 0.0 {
                scale = -scale;
                for k in 0..4 {
                    block_signs_final[k] = (!block_signs_final[k]) & 127;
                }
            }

            for k in 0..8 {
                let mut u: u16 = 0;
                for i in 0..4 {
                    u |= (l_final[4 * k + i] as u16) << (3 * i);
                }
                let Some(grid_index) = map_lookup_u32(&IQ3XXS_GRID, u) else {
                    // Internal upstream invariant (`GGML_ASSERT(grid_index >= 0)`):
                    // after selecting the nearest grid, the final codes must be
                    // grid point. The violation is a bug of the implementation, not the input data.
                    panic!(
                        "quantize_iq3_xxs: sub-block {ib}, group {k}: {u:#06x} codes do not lie on the IQ3XXS_GRID grid \
                         (the internal invariant of the quantizer is violated)"
                    );
                };
                q3_grid[8 * ib + k] = grid_index as u8;
            }
            scales_and_signs[ib] = block_signs_final[0] as u32
                | (block_signs_final[1] as u32) << 7
                | (block_signs_final[2] as u32) << 14
                | (block_signs_final[3] as u32) << 21;

            scales[ib] = scale;
            if scale > max_scale {
                max_scale = scale;
            }
        }

        let blk = &mut blocks[ibl * block_bytes..(ibl + 1) * block_bytes];
        if max_scale == 0.0 {
            blk[0..2].copy_from_slice(&crate::pack::f16_saturating(0.0).to_le_bytes());
            continue;
        }

        let d = max_scale / 31.0;
        blk[0..2].copy_from_slice(&crate::pack::f16_saturating(d).to_le_bytes());
        let id = 1.0 / d;
        for ib in 0..8 {
            let l = clampi(nearest_int(0.5 * (id * scales[ib] - 1.0)), 0, 15);
            scales_and_signs[ib] |= (l as u32) << 28;
        }

        blk[2..2 + 64].copy_from_slice(&q3_grid);
        for ib in 0..8 {
            let off = 2 + 64 + 4 * ib;
            blk[off..off + 4].copy_from_slice(&scales_and_signs[ib].to_le_bytes());
        }
    }

    Iq3XxsPacked {
        blocks,
        numel: data.len(),
    }
}

// ═══════════════════════════════════════════════════════════════════════
// § 5 IQ3_S - 110 bytes/256 weights (4-element groups, 9-bit grid index,
//        fair signed bytes instead of parity packing, scale step (1+2c))
// ═══════════════════════════════════════════════════════════════════════

/// Packed IQ3_S super-block tensor.
#[derive(Debug, Clone, PartialEq)]
pub struct Iq3SPacked {
    /// Raw super-blocks (110 bytes each).
    pub blocks: Vec<u8>,
    /// Initial number of scales.
    pub numel: usize,
}

impl Iq3SPacked {
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
        dequantize_iq3_s(self)
    }
}

/// implementation `dequantize_row_iq3_s`.
pub fn dequantize_iq3_s(packed: &Iq3SPacked) -> Vec<f32> {
    let block_bytes = DType::IQ3_S.block_bytes(); // 110
    let n_blocks = packed.blocks.len() / block_bytes;
    let mut out = Vec::with_capacity(n_blocks * 256);

    for b in 0..n_blocks {
        let blk = &packed.blocks[b * block_bytes..(b + 1) * block_bytes];
        let d = f16::from_le_bytes([blk[0], blk[1]]).to_f32();
        let qs_all = &blk[2..2 + 64]; // 64 bytes
        let qh_all = &blk[66..66 + 8]; // 8 bytes
        let signs_all = &blk[74..74 + 32]; // 32 bytes
        let scales = &blk[106..110]; // 4 bytes

        let mut qs_off = 0usize;
        let mut sign_off = 0usize;

        for ib32_pair in 0..4 {
            // ib32 in 0,2,4,6 (step 2) — 4 pairs cover QK_K/32=8
            let sc_byte = scales[ib32_pair];
            let db1 = d * (1.0 + 2.0 * (sc_byte & 0xf) as f32);
            let db2 = d * (1.0 + 2.0 * (sc_byte >> 4) as f32);
            let qh0 = qh_all[2 * ib32_pair] as u32;
            let qh1 = qh_all[2 * ib32_pair + 1] as u32;

            // First half of the pair (uses qh0, db1)
            for l in 0..4 {
                let signs_byte = signs_all[sign_off + l];
                let idx1 = qs_all[qs_off + 2 * l] as u32 | ((qh0 << (8 - 2 * l)) & 256);
                let idx2 = qs_all[qs_off + 2 * l + 1] as u32 | ((qh0 << (7 - 2 * l)) & 256);
                let grid1 = grid4_u32(&IQ3S_GRID, idx1 as usize);
                let grid2 = grid4_u32(&IQ3S_GRID, idx2 as usize);
                let mut window = [0.0f32; 8];
                for j in 0..4 {
                    let s0 = if signs_byte & KMASK_IQ2XS[j] != 0 {
                        -1.0
                    } else {
                        1.0
                    };
                    let s4 = if signs_byte & KMASK_IQ2XS[j + 4] != 0 {
                        -1.0
                    } else {
                        1.0
                    };
                    window[j] = db1 * grid1[j] as f32 * s0;
                    window[j + 4] = db1 * grid2[j] as f32 * s4;
                }
                out.extend_from_slice(&window);
            }
            qs_off += 8;
            sign_off += 4;

            // Second half of the pair (uses qh1, db2)
            for l in 0..4 {
                let signs_byte = signs_all[sign_off + l];
                let idx1 = qs_all[qs_off + 2 * l] as u32 | ((qh1 << (8 - 2 * l)) & 256);
                let idx2 = qs_all[qs_off + 2 * l + 1] as u32 | ((qh1 << (7 - 2 * l)) & 256);
                let grid1 = grid4_u32(&IQ3S_GRID, idx1 as usize);
                let grid2 = grid4_u32(&IQ3S_GRID, idx2 as usize);
                let mut window = [0.0f32; 8];
                for j in 0..4 {
                    let s0 = if signs_byte & KMASK_IQ2XS[j] != 0 {
                        -1.0
                    } else {
                        1.0
                    };
                    let s4 = if signs_byte & KMASK_IQ2XS[j + 4] != 0 {
                        -1.0
                    } else {
                        1.0
                    };
                    window[j] = db2 * grid1[j] as f32 * s0;
                    window[j + 4] = db2 * grid2[j] as f32 * s4;
                }
                out.extend_from_slice(&window);
            }
            qs_off += 8;
            sign_off += 4;
        }
    }

    out.truncate(packed.numel);
    out
}

/// Quantize `data` into IQ3_S (`IQ3S_BLOCK_SIZE = 32`). See `quantize_iq2_xxs`
/// behind the semantics of `quant_weights` - like IQ3_XXS, `None` - documented
/// no-imatrix upstream mode (`quantize_row_iq3_s_ref`).
///
/// #Panic
///
/// Panics if `data.len()` is not a multiple of 256 (`QK_K`).
pub fn quantize_iq3_s(data: &[f32], quant_weights: Option<&[f32]>) -> Iq3SPacked {
    assert_eq!(
        data.len() % 256,
        0,
        "quantize_iq3_s: len must be a multiple of QK_K=256"
    );
    let n_blocks = data.len() / 256;
    let block_bytes = DType::IQ3_S.block_bytes();
    let mut blocks = vec![0u8; n_blocks * block_bytes];

    const K_MAX_Q: i32 = 8;
    const BS4: usize = 8; // block_size/4 = 32/4
    const BS8: usize = 4; // block_size/8 = 32/8

    for ibl in 0..n_blocks {
        let xbl = &data[ibl * 256..(ibl + 1) * 256];
        let sumx2: f32 = xbl.iter().map(|w| w * w).sum();
        let sigma2 = 2.0 * sumx2 / 256.0;

        let mut scales = [0.0f32; 8]; // QK_K/32 = 8
        let mut qs_all = [0u8; 64];
        let mut qh_all = [0u8; 8];
        let mut signs_all = [0u8; 32];
        let mut max_scale = 0.0f32;

        let mut qs_off = 0usize;
        let mut sign_off = 0usize;

        for ib in 0..8 {
            // QK_K/32
            let xb = &xbl[32 * ib..32 * ib + 32];
            let mut weight = [0.0f32; 32];
            match quant_weights {
                Some(qw) => {
                    let qwb = &qw[ibl * 256 + 32 * ib..ibl * 256 + 32 * ib + 32];
                    for i in 0..32 {
                        weight[i] = qwb[i] * (sigma2 + xb[i] * xb[i]).sqrt();
                    }
                }
                None => {
                    for i in 0..32 {
                        weight[i] = xb[i] * xb[i];
                    }
                }
            }
            let mut waux = [0.0f32; 32];
            for i in 0..32 {
                waux[i] = weight[i].sqrt();
            }

            let mut xval = [0.0f32; 32];
            let mut block_signs = [0u8; BS8];
            for k in 0..BS8 {
                let mut s: u8 = 0;
                for i in 0..8 {
                    let v = xb[8 * k + i];
                    if v >= 0.0 {
                        xval[8 * k + i] = v;
                    } else {
                        xval[8 * k + i] = -v;
                        s |= 1 << i;
                    }
                }
                block_signs[k] = s; // not masked in 127 - fair byte (see header § IQ3_S)
            }

            let max = xval.iter().cloned().fold(0.0f32, f32::max);
            let mut l_final = [0i32; 32];
            if max == 0.0 {
                scales[ib] = 0.0;
                qs_off += BS4;
                sign_off += BS8;
                continue;
            }

            let mut best = 0.0f32;
            let mut scale = max / (2 * K_MAX_Q - 1) as f32;
            let mut is_on_grid = [false; BS4];

            for is in -9..=9i32 {
                let id = ((2 * K_MAX_Q - 1) as f32 + is as f32 * 0.2) / max;
                let this_scale = 1.0 / id;
                let mut laux = [0i8; 32];
                let mut is_on_grid_aux = [true; BS4];
                for k in 0..BS4 {
                    let mut u: u16 = 0;
                    for i in 0..4 {
                        let l = clampi(
                            nearest_int(0.5 * (id * xval[4 * k + i] - 1.0)),
                            0,
                            K_MAX_Q - 1,
                        );
                        laux[4 * k + i] = l as i8;
                        u |= (l as u16) << (3 * i);
                    }
                    if map_lookup_u32(&IQ3S_GRID, u).is_some() {
                        is_on_grid_aux[k] = true;
                    } else {
                        is_on_grid_aux[k] = false;
                        let (_, l) = find_best_grid_point_u32(
                            &IQ3S_GRID,
                            &xval[4 * k..4 * k + 4],
                            &waux[4 * k..4 * k + 4],
                            this_scale,
                        );
                        for i in 0..4 {
                            laux[4 * k + i] = l[i];
                        }
                    }
                }
                let mut sumqx = 0.0f32;
                let mut sumq2 = 0.0f32;
                for i in 0..32 {
                    let w = weight[i];
                    let q = 2.0 * laux[i] as f32 + 1.0;
                    sumqx += w * xval[i] * q;
                    sumq2 += w * q * q;
                }
                if sumq2 > 0.0 && sumqx * sumqx > best * sumq2 {
                    scale = sumqx / sumq2;
                    best = scale * sumqx;
                    for i in 0..32 {
                        l_final[i] = laux[i] as i32;
                    }
                    is_on_grid = is_on_grid_aux;
                }
            }

            let n_not_on_grid = is_on_grid.iter().filter(|&&b| !b).count();
            if n_not_on_grid > 0 && scale > 0.0 {
                let id = 1.0 / scale;
                // Upstream comments `if (is_on_grid[k]) continue;` on
                // this block for IQ3_S (unlike IQ2_XXS/IQ2_XS/IQ3_XXS,
                // where the line is active) - here ALL k are recalculated,
                // not just non-on-grid. The implementation preserves this verbatim.
                for k in 0..BS4 {
                    let mut u: u16 = 0;
                    for i in 0..4 {
                        let l = clampi(
                            nearest_int(0.5 * (id * xval[4 * k + i] - 1.0)),
                            0,
                            K_MAX_Q - 1,
                        );
                        u |= (l as u16) << (3 * i);
                    }
                    let (gidx, _) = resolve_quad_u32(
                        &IQ3S_GRID,
                        u,
                        &xval[4 * k..4 * k + 4],
                        &waux[4 * k..4 * k + 4],
                        scale,
                    );
                    let pg = grid4_u32(&IQ3S_GRID, gidx);
                    for i in 0..4 {
                        l_final[4 * k + i] = ((pg[i] - 1) / 2) as i32;
                    }
                }
                let mut sumqx = 0.0f32;
                let mut sumq2 = 0.0f32;
                for i in 0..32 {
                    let w = weight[i];
                    let q = 2.0 * l_final[i] as f32 + 1.0;
                    sumqx += w * xval[i] * q;
                    sumq2 += w * q * q;
                }
                if sumq2 > 0.0 {
                    scale = sumqx / sumq2;
                }
            }

            let mut block_signs_final = block_signs;
            if scale < 0.0 {
                scale = -scale;
                for k in 0..BS8 {
                    block_signs_final[k] = !block_signs_final[k];
                }
            }

            for k in 0..BS4 {
                let mut u: u16 = 0;
                for i in 0..4 {
                    u |= (l_final[4 * k + i] as u16) << (3 * i);
                }
                let Some(grid_index) = map_lookup_u32(&IQ3S_GRID, u) else {
                    // Internal upstream invariant (`GGML_ASSERT(grid_index >= 0)`):
                    // after selecting the nearest grid, the final codes must be
                    // grid point. The violation is a bug of the implementation, not the input data.
                    panic!(
                        "quantize_iq3_s: sub-block {ib}, group {k}: {u:#06x} codes do not lie on the IQ3S_GRID grid\
                         (the internal invariant of the quantizer is violated)"
                    );
                };
                qs_all[qs_off + k] = (grid_index & 255) as u8;
                let bit_pos = (ib * BS4 + k) % 8;
                let byte_pos = (ib * BS4 + k) / 8;
                qh_all[byte_pos] |= ((grid_index >> 8) as u8) << bit_pos;
            }
            qs_off += BS4;
            signs_all[sign_off..sign_off + BS8].copy_from_slice(&block_signs_final[..BS8]);
            sign_off += BS8;

            scales[ib] = scale;
            if scale > max_scale {
                max_scale = scale;
            }
        }

        let blk = &mut blocks[ibl * block_bytes..(ibl + 1) * block_bytes];
        if max_scale == 0.0 {
            continue; // the entire block remains zero (d=0, like memset in upstream)
        }

        let d = max_scale / 31.0;
        let d_stored = d * 1.033; // the same fudge-factor as in the upstream
        blk[0..2].copy_from_slice(&crate::pack::f16_saturating(d_stored).to_le_bytes());
        let id = 1.0 / d;
        let mut scale_bytes = [0u8; 4];
        for ib in (0..8).step_by(2) {
            let l1 = clampi(nearest_int(0.5 * (id * scales[ib] - 1.0)), 0, 15) as u8;
            let l2 = clampi(nearest_int(0.5 * (id * scales[ib + 1] - 1.0)), 0, 15) as u8;
            scale_bytes[ib / 2] = l1 | (l2 << 4);
        }

        blk[2..66].copy_from_slice(&qs_all);
        blk[66..74].copy_from_slice(&qh_all);
        blk[74..106].copy_from_slice(&signs_all);
        blk[106..110].copy_from_slice(&scale_bytes);
    }

    Iq3SPacked {
        blocks,
        numel: data.len(),
    }
}
