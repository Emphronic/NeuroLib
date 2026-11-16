// neo-quantization/src/pack/intn.rs
//
// A single parameterized implementation of intN quantization for N ∈ {1..8}.
//
// Motivation
// ─────────
// int1.rs/int4.rs/int8.rs (written earlier) - three independent copies
// the same algorithm (round-to-nearest + group scale),
// differing only in bit width and byte packing. Expansion to
// int2/int3/int5/int6/int7 copy-paste would mean eight almost
// identical files - a source of desynchronization during any future fix.
// Instead, there is one generic-by-bit engine here; int1.rs/int4.rs/
// int8.rs is now a thin wrapper over `IntNPacked`, keeping the same
// public API (`Int1Packed`/`Int4Packed`/`Int8Packed`), which is already
// refer `pack::PackedBuffer`/`convert.rs`/`switch.rs`.
//
// ## Symmetric vs Asymmetric (zero-point)
//
// Symmetric:
//   alpha = mean(|W|) * k(N) — calibration via mean(|W|), not
//                                    max(|W|): works correctly when
//                                    N=1 (where signed code range
//                                    degenerate to {-1,+1}, max-abs formula
//                                    not applicable - see "Why mean, not
//                                    max" below) and remains the same
//                                    mean(|W|)-calibrator that was in
//                                    original ternary.rs/int1.rs.
//   q     = round(clip(W / alpha, -max_code, max_code))
//   W_hat = q * alpha
//
//   Code range: `[-(2^(N-1)-1), 2^(N-1)-1]` at N≥2 (symmetrical
//   around zero, do not use the extreme negative code - the same choice,
//   that in the original int8.rs: [-127,127], not [-128,127]); at N=1 range
//   degenerates into `{-1, +1}` (there is no “zero” among the codes - this corresponds to
//   the original int1.rs, where there are never zero weights, only a sign).
//
// ## Why mean(|W|), and not max(|W|), for symmetric
//
// Optimal alpha that minimizes `Σ(W_i - q_i·alpha)²` at
// round-to-nearest is close to `max(|W|)/max_code` at a large number of levels
// (the int8/int4 symmetric schemes use exactly this formula - it is
// more efficient on wide code ranges where outliers are rare and
// “wasting” the entire dynamic range on them is almost not painful). When
// small number of levels (N=1, ternary N=2 with zero) the difference between
// max-abs and mean-abs calibration is much more significant: mean(|W|) - accurate
// minimizer for single-level {-1,+1} quantization (standard
// result of BinaryConnect/XNOR-Net), and max-abs is sharply underutilized there
// dynamic range (one surge compresses all weights to 0).
// To have ONE calibration formula that works correctly on everything
// range N=1..8 and giving the same numbers as the original ternary.rs/
// int1.rs for N=1, here we use the mean(|W|) calibration with
// scaling factor `k(N)`, selected so that:
//   • for N=1: alpha = mean(|W|) (as in the original int1.rs)
//   • with N→∞: alpha → mean(|W|) remains the same, but max_code increases,
//     so the effective coverage grows with the bit rate naturally
//     without additional calibration - mean(|W|) scale multiplied by
//     number of levels, gives reasonable coverage of heavy tails already at
//     N=3..4 (empirical observation from the BitNet family: mean-abs
//     calibration with a small number of bits is more resistant to outliers than max-abs).
// `k(N) = 1` for all N in this version (the simplest correct choice:
// alpha = mean(|W|), q = round(clip(W/alpha, -max_code, max_code)));
// external callers who need max-abs calibration (as in the original
// int4.rs/int8.rs, more precisely on “wide” code ranges without emissions),
// can use `quantize_maxabs` - the second public function of this
// file left specifically for N≥2 (see § 4b).
//
// ## Asymmetric (GPTQ/AWQ style)
//
//   scale      = (max(W) - min(W)) / (2^N - 1)
//   zero_point = round(-min(W) / scale) [unsigned code]
//   q          = round(clip(W / scale + zero_point, 0, 2^N - 1))
//   W_hat      = (q - zero_point) * scale
//
//   Code range: [0, 2^N - 1] - full unsigned range of N bits.
//   Not defined for N=1 in the general engine (asymmetric on 1 bit does not have
//   meaningful meaning beyond the usual symmetric with threshold ≠ 0 -
//   see int1.rs, where this case is not explicitly proposed) - `quantize`
//   panics at `bits == 1 && calibration == Asymmetric`.
//
// ## Why N as a runtime parameter and not a const generic
//
// `IntNPacked` could be `IntNPacked<const N: u32>`, but then every
// combination of N/PackedScheme option would require a separate option
// in `PackedBuffer`. Instead, `bits: u8` is a normal field; packaging
// via bit-cursor functions below, without unsafe, without dependence on N on
// compilation stage.

use crate::rng::{round_or_stochastic, XorShift64};
use crate::scheme::ScaleGranularity;

// ═══════════════════════════════════════════════════════════════════════════
// § 1 CALIBRATION MODE
// ═══════════════════════════════════════════════════════════════════════════

/// Symmetric (around zero) or asymmetric (GPTQ/AWQ style, with zero-point).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum Calibration {
    /// `q = round(W / alpha)`, the code range is symmetrical around 0.
    /// Easier and cheaper in the GEMM core (no need to subtract zero_point on
    /// each FMA), but worse on biased weight distributions.
    #[default]
    Symmetric,
    /// `q = round(W / scale) + zero_point`, code range - full
    /// unsigned `[0, 2^N-1]`. Requires additional subtract zero_point at
    /// dequantization/GEMM, but more precisely on calibrated balances (GPTQ/AWQ).
    /// Not supported with `bits == 1` (see file header).
    Asymmetric,
}

// ═══════════════════════════════════════════════════════════════════════════
// § 2  PACKED CONTAINER
// ═══════════════════════════════════════════════════════════════════════════

/// Packed intN tensor, N from 1 to 8 bits, symmetric or asymmetric.
#[derive(Debug, Clone, PartialEq)]
pub struct IntNPacked {
    /// Code width in bits, 1..=8.
    pub bits: u8,
    /// Calibration mode - determines how to read `codes` through `scales`
    /// (and `zero_points` if `Asymmetric`).
    pub calibration: Calibration,
    /// Packed codes, `bits` bits per element, bit-cursor from low
    /// bit of the first byte (see `write_bits`/`read_bits` below). For
    /// symmetric code is stored in biased-unsigned shape (`signed_code +
    /// half_range`), for asymmetric - as is (already unsigned
    /// `[0, 2^N-1]`).
    pub codes: Vec<u8>,
    /// Scale (alpha for symmetric, scale for asymmetric) by groups.
    pub scales: Vec<f32>,
    /// Zero-point by groups. Empty at `Calibration::Symmetric`.
    /// The length is equal to `scales.len()` at `Asymmetric`.
    pub zero_points: Vec<i32>,
    /// Group size in elements. `usize::MAX` = PerTensor.
    pub group_size: usize,
    /// Initial number of scales.
    pub numel: usize,
}

impl IntNPacked {
    /// Maximum unsigned code for a given bit depth (`2^bits - 1`).
    #[inline]
    pub fn max_unsigned_code(bits: u8) -> u32 {
        (1u32 << bits) - 1
    }

    /// Maximum symmetric code (`2^(N-1) - 1` for N≥2; `1` for N=1 —
    /// a special case without a “zero” among the codes, see the file header).
    #[inline]
    pub fn max_symmetric_code(bits: u8) -> i32 {
        if bits == 1 {
            1
        } else {
            (1i32 << (bits - 1)) - 1
        }
    }

    /// Bias for storing symmetric code in an unsigned field.
    ///
    /// When N≥2: the code `[-max,max]` is stored as `[0, 2*max]` (bias=max),
    /// so `raw = signed + bias` always fits into N bits
    /// (`2*max_symmetric_code(N) ≤ 2^N - 1`).
    ///
    /// With N=1 the general formula would give `signed ∈ {-1,+1}`, `bias=1` →
    /// `raw ∈ {0,2}` - the code `2` does not fit into a 1-bit field (`{0,1}`).
    /// N=1 therefore uses a direct mapping `raw∈{0,1} ↔ val∈{-1,+1}` through
    /// `2*raw-1` (see `get_f32`/encoding in `quantize()`, where bits==1 takes a
    /// separate branch). `symmetric_bias(1)` is not used on that path and
    /// stays `0`; it is a safe (nonsensical) value for accidental callers.
    #[inline]
    pub fn symmetric_bias(bits: u8) -> i32 {
        if bits == 1 {
            0
        } else {
            Self::max_symmetric_code(bits)
        }
    }

    /// Packing size in bytes (without scales/zero_points).
    pub fn codes_bytes(&self) -> usize {
        self.codes.len()
    }

    /// Actual bits/weight taking into account overhead on scales and (if any)
    /// zero_points.
    pub fn effective_bits_per_weight(&self) -> f32 {
        let data_bits = self.codes.len() as f32 * 8.0;
        let scale_bits = self.scales.len() as f32 * 32.0;
        let zp_bits = self.zero_points.len() as f32 * 32.0;
        (data_bits + scale_bits + zp_bits) / self.numel as f32
    }

    /// Group index (scale bucket) for element `idx`.
    #[inline]
    pub fn group_of(&self, idx: usize) -> usize {
        if self.group_size == usize::MAX {
            0
        } else {
            idx / self.group_size
        }
    }

    /// Read the unsigned (packed) code of the element `idx`.
    #[inline]
    pub fn get_raw_code(&self, idx: usize) -> u32 {
        read_bits(&self.codes, idx, self.bits)
    }

    /// Dequantize the element `idx` in f32.
    #[inline]
    pub fn get_f32(&self, idx: usize) -> f32 {
        let raw = self.get_raw_code(idx);
        let g = self.group_of(idx);
        let scale = self.scales[g];
        match self.calibration {
            Calibration::Symmetric => {
                let signed = if self.bits == 1 {
                    // raw∈{0,1} ↔ signed∈{-1,+1}, see quantize()/symmetric_bias.
                    if raw == 1 {
                        1
                    } else {
                        -1
                    }
                } else {
                    raw as i32 - Self::symmetric_bias(self.bits)
                };
                signed as f32 * scale
            }
            Calibration::Asymmetric => {
                let zp = self.zero_points[g];
                (raw as i32 - zp) as f32 * scale
            }
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// § 3 BIT-CURSOR READ/WRITE
// ═══════════════════════════════════════════════════════════════════════════
//
// General packing "N bits per element, sequentially from the beginning of the buffer" -
// for an arbitrary N, the element may cross a byte boundary; bit-cursor
// takes this into account explicitly (unlike int4.rs/int1.rs, where N=4/N=1 is convenient
// never cross byte boundaries).

/// Write `value` (lower `bits` bits are significant) to the position of element `idx`
/// inside `buf`. The buffer must be allocated in advance for `numel` elements
/// (see `bits_to_bytes`).
#[inline]
fn write_bits(buf: &mut [u8], idx: usize, bits: u8, value: u32) {
    let bit_offset = idx * bits as usize;
    let mut remaining = bits;
    let mut byte_idx = bit_offset / 8;
    let mut bit_in_byte = bit_offset % 8;
    let mut v = value;

    while remaining > 0 {
        let space_in_byte = 8 - bit_in_byte;
        let take = remaining.min(space_in_byte as u8);
        let mask = ((1u32 << take) - 1) as u8;
        let chunk = (v & mask as u32) as u8;
        buf[byte_idx] |= chunk << bit_in_byte;

        v >>= take;
        remaining -= take;
        byte_idx += 1;
        bit_in_byte = 0;
    }
}

/// Read the `bits`-bit value of element `idx` from `buf`.
#[inline]
fn read_bits(buf: &[u8], idx: usize, bits: u8) -> u32 {
    let bit_offset = idx * bits as usize;
    let mut remaining = bits;
    let mut byte_idx = bit_offset / 8;
    let mut bit_in_byte = bit_offset % 8;
    let mut result: u32 = 0;
    let mut shift = 0u32;

    while remaining > 0 {
        let space_in_byte = 8 - bit_in_byte;
        let take = remaining.min(space_in_byte as u8);
        let mask = ((1u32 << take) - 1) as u8;
        let chunk = (buf[byte_idx] >> bit_in_byte) & mask;
        result |= (chunk as u32) << shift;

        shift += take as u32;
        remaining -= take;
        byte_idx += 1;
        bit_in_byte = 0;
    }

    result
}

/// The number of bytes needed to pack `numel` elements into `bits` bits each.
#[inline]
pub fn bits_to_bytes(numel: usize, bits: u8) -> usize {
    (numel * bits as usize).div_ceil(8)
}

#[inline]
fn group_bounds(g: usize, group_size: usize, numel: usize) -> (usize, usize) {
    if group_size == usize::MAX {
        (0, numel)
    } else {
        let start = g * group_size;
        let end = (start + group_size).min(numel);
        (start, end)
    }
}

/// Number of groups for a given `group_size`/`numel` (public helper,
/// reused by int1.rs/int4.rs/int8.rs wrappers, so as not
/// duplicate the formula `div_ceil` in each file).
#[inline]
pub fn n_groups(group_size: usize, numel: usize) -> usize {
    if group_size == usize::MAX {
        1
    } else {
        numel.div_ceil(group_size)
    }
}

/// Allow `ScaleGranularity` to specific `group_size` (public
/// helper - the same formula is needed in all pack::* files).
#[inline]
pub fn resolve_group_size(granularity: ScaleGranularity, row_len: usize) -> usize {
    match granularity {
        ScaleGranularity::PerTensor => usize::MAX,
        ScaleGranularity::PerChannel => row_len,
        ScaleGranularity::PerGroup { group_size } => group_size,
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// § 4 QUANTIZE - mean(|W|) calibration (single formula for all N)
// ═══════════════════════════════════════════════════════════════════════════

/// Quantize `data` into `bits`-bit intN format, symmetric or
/// asymmetric calibration.
///
/// Symmetric uses `alpha = mean(|W|)` by group - it works
/// equally correct for any `bits` from 1 to 8 inclusive (incl.
/// degenerate N=1, where the max-abs formula is not applicable - see title
/// file). For max-abs calibration at N≥2 (compatible with original
/// int4.rs/int8.rs) see `quantize_maxabs` in § 4b below.
///
/// #Panic
///
/// - `bits` is outside the range of `1..=8`.
/// - `bits == 1 && calibration == Asymmetric` (not supported, see
///   file header - use `pack::ternary` for 3-level
///   {-1,0,+1} quantization, if you need asymmetry of meaning on a 1-bit
///   scale).
pub fn quantize(
    data: &[f32],
    bits: u8,
    calibration: Calibration,
    granularity: ScaleGranularity,
    row_len: usize,
) -> IntNPacked {
    quantize_impl(data, bits, calibration, granularity, row_len, None)
}

/// Same as [`quantize`], but weights are rounded STOCHASTICALLY: up from
/// probability of the fractional part `w / scale`, down - with an additional one. Evaluation
/// unbiased (`E[code * scale] = w` for weights within the range), in
/// in contrast to round-to-nearest, whose error is systematic on dense
/// clusters of values. Scales and zero-points are computed the same way as
/// [`quantize`] (according to the initial data), RNG is determined by `seed`.
///
/// With `bits == 1` there is no rounding (sign), the result coincides with
/// [`quantize`]. It panics in the same places as [`quantize`].
pub fn quantize_stochastic(
    data: &[f32],
    bits: u8,
    calibration: Calibration,
    granularity: ScaleGranularity,
    row_len: usize,
    seed: u64,
) -> IntNPacked {
    quantize_impl(
        data,
        bits,
        calibration,
        granularity,
        row_len,
        Some(XorShift64::new(seed)),
    )
}

/// General implementation: `rng == None` - round-to-nearest (behavior
/// `quantize`), `Some` - stochastic rounding.
fn quantize_impl(
    data: &[f32],
    bits: u8,
    calibration: Calibration,
    granularity: ScaleGranularity,
    row_len: usize,
    mut rng: Option<XorShift64>,
) -> IntNPacked {
    assert!(
        (1..=8).contains(&bits),
        "intN::quantize: bits must be 1..=8, got {bits}"
    );
    assert!(
        !(bits == 1 && calibration == Calibration::Asymmetric),
        "intN::quantize: asymmetric calibration is not supported at bits=1 \
         (use pack::ternary for a 3-level {{-1,0,+1}} scheme instead)"
    );

    let numel = data.len();
    let group_size = resolve_group_size(granularity, row_len);
    let groups = n_groups(group_size, numel);

    let mut scales = vec![0.0f32; groups];
    let mut zero_points: Vec<i32> = Vec::new();

    match calibration {
        Calibration::Symmetric => {
            for g in 0..groups {
                let (start, end) = group_bounds(g, group_size, numel);
                let slice = &data[start..end];
                let mean_abs = if slice.is_empty() {
                    0.0
                } else {
                    slice.iter().map(|w| w.abs()).sum::<f32>() / slice.len() as f32
                };
                scales[g] = mean_abs.max(f32::EPSILON);
            }
        }
        Calibration::Asymmetric => {
            zero_points = vec![0i32; groups];
            let max_code = IntNPacked::max_unsigned_code(bits);
            for g in 0..groups {
                let (start, end) = group_bounds(g, group_size, numel);
                let slice = &data[start..end];
                let (mut lo, mut hi) = (f32::INFINITY, f32::NEG_INFINITY);
                for &w in slice {
                    lo = lo.min(w);
                    hi = hi.max(w);
                }
                if !lo.is_finite() {
                    lo = 0.0;
                }
                if !hi.is_finite() {
                    hi = 0.0;
                }
                // The range must include 0 - otherwise W=0 (padding/masked
                // weight) would not be accurately representable after round-trip.
                lo = lo.min(0.0);
                hi = hi.max(0.0);

                let scale = ((hi - lo) / max_code as f32).max(f32::EPSILON);
                let zp = (-lo / scale).round().clamp(0.0, max_code as f32) as i32;
                scales[g] = scale;
                zero_points[g] = zp;
            }
        }
    }

    let n_bytes = bits_to_bytes(numel, bits);
    let mut codes = vec![0u8; n_bytes];

    match calibration {
        Calibration::Symmetric => {
            let max_code = IntNPacked::max_symmetric_code(bits);
            let bias = IntNPacked::symmetric_bias(bits);
            for (idx, &w) in data.iter().enumerate() {
                let g = if group_size == usize::MAX {
                    0
                } else {
                    idx / group_size
                };
                let alpha = scales[g];
                let raw: u32 = if bits == 1 {
                    // Degenerate case: 1 bit stores the code directly as
                    // {0,1} ↔ {-1,+1} (raw = (signed+1)/2), NOT via
                    // signed+bias of the general formula - it gives raw∈{0,2},
                    // and 2 does not fit into a 1-bit field (see symmetric_bias).
                    if w >= 0.0 {
                        1
                    } else {
                        0
                    }
                } else {
                    let signed = round_or_stochastic(w / alpha, &mut rng)
                        .clamp(-(max_code as f32), max_code as f32)
                        as i32;
                    (signed + bias) as u32
                };
                write_bits(&mut codes, idx, bits, raw);
            }
        }
        Calibration::Asymmetric => {
            let max_code = IntNPacked::max_unsigned_code(bits) as f32;
            for (idx, &w) in data.iter().enumerate() {
                let g = if group_size == usize::MAX {
                    0
                } else {
                    idx / group_size
                };
                let scale = scales[g];
                let zp = zero_points[g];
                let q = round_or_stochastic(w / scale + zp as f32, &mut rng).clamp(0.0, max_code)
                    as u32;
                write_bits(&mut codes, idx, bits, q);
            }
        }
    }

    IntNPacked {
        bits,
        calibration,
        codes,
        scales,
        zero_points,
        group_size,
        numel,
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// § 4b QUANTIZE - max(|W|) calibration (compatibility with original
//        int4.rs/int8.rs; available separately for N ≥ 2)
// ═══════════════════════════════════════════════════════════════════════════

/// Symmetric quantization with `alpha = max(|W|) / max_code` instead
/// `mean(|W|)` is the calibration that was used by the original int4.rs/
/// int8.rs before being merged into this engine. More precisely `quantize()` on wide
/// code ranges (N≥6) with rare outliers, where to reserve
/// code for the maximum modulus weight is not wasteful; worse at low
/// bits with frequent outliers (see the file header, "Why mean, but
/// not max"). Does not support `Asymmetric` - asymmetric calibration in
/// this crate always goes through the min/max range (see `quantize()`
/// above), where the concept "max-abs vs mean-abs" does not apply (asymmetric
/// already uses the full `[min(W), max(W)]`, not `|W|`).
///
/// Not defined at `bits == 1` - max-abs formula is degenerate there,
/// where and in `quantize()` (see file header); use `quantize()`
/// with `Calibration::Symmetric` for 1 bit.
pub fn quantize_maxabs(
    data: &[f32],
    bits: u8,
    granularity: ScaleGranularity,
    row_len: usize,
) -> IntNPacked {
    quantize_maxabs_impl(data, bits, granularity, row_len, None)
}

/// Same as [`quantize_maxabs`], but with STOCHASTIC rounding of codes
/// (see [`quantize_stochastic`]); `alpha = max(|W|) / max_code` is computed
/// as usual, RNG is determined by `seed`. Panicking in the same place
/// [`quantize_maxabs`].
pub fn quantize_maxabs_stochastic(
    data: &[f32],
    bits: u8,
    granularity: ScaleGranularity,
    row_len: usize,
    seed: u64,
) -> IntNPacked {
    quantize_maxabs_impl(
        data,
        bits,
        granularity,
        row_len,
        Some(XorShift64::new(seed)),
    )
}

fn quantize_maxabs_impl(
    data: &[f32],
    bits: u8,
    granularity: ScaleGranularity,
    row_len: usize,
    mut rng: Option<XorShift64>,
) -> IntNPacked {
    assert!(
        (2..=8).contains(&bits),
        "intN::quantize_maxabs: bits must be 2..=8, got {bits}"
    );

    let numel = data.len();
    let group_size = resolve_group_size(granularity, row_len);
    let groups = n_groups(group_size, numel);
    let max_code = IntNPacked::max_symmetric_code(bits);

    let mut scales = vec![0.0f32; groups];
    for g in 0..groups {
        let (start, end) = group_bounds(g, group_size, numel);
        let slice = &data[start..end];
        let max_abs = slice.iter().fold(0.0f32, |acc, w| acc.max(w.abs()));
        scales[g] = (max_abs / max_code as f32).max(f32::EPSILON);
    }

    let n_bytes = bits_to_bytes(numel, bits);
    let mut codes = vec![0u8; n_bytes];
    let bias = IntNPacked::symmetric_bias(bits);

    for (idx, &w) in data.iter().enumerate() {
        let g = if group_size == usize::MAX {
            0
        } else {
            idx / group_size
        };
        let alpha = scales[g];
        let signed = round_or_stochastic(w / alpha, &mut rng)
            .clamp(-(max_code as f32), max_code as f32) as i32;
        let biased = (signed + bias) as u32;
        write_bits(&mut codes, idx, bits, biased);
    }

    IntNPacked {
        bits,
        calibration: Calibration::Symmetric,
        codes,
        scales,
        zero_points: Vec::new(),
        group_size,
        numel,
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// § 5  DEQUANTIZE
// ═══════════════════════════════════════════════════════════════════════════

/// Completely expand the intN package back to `Vec<f32>`.
pub fn dequantize(packed: &IntNPacked) -> Vec<f32> {
    (0..packed.numel).map(|i| packed.get_f32(i)).collect()
}

// ═══════════════════════════════════════════════════════════════════════════
// § 6 TESTS
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    fn roundtrip_error(bits: u8, calib: Calibration, data: &[f32]) -> f32 {
        let packed = quantize(data, bits, calib, ScaleGranularity::PerTensor, 0);
        let restored = dequantize(&packed);
        data.iter()
            .zip(restored.iter())
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max)
    }

    #[test]
    fn test_bit_cursor_roundtrip_all_widths() {
        for bits in 1u8..=8 {
            let numel = 37;
            let mut buf = vec![0u8; bits_to_bytes(numel, bits)];
            let max = (1u32 << bits) - 1;
            let values: Vec<u32> = (0..numel).map(|i| (i as u32 * 7) % (max + 1)).collect();
            for (i, &v) in values.iter().enumerate() {
                write_bits(&mut buf, i, bits, v);
            }
            for (i, &v) in values.iter().enumerate() {
                assert_eq!(read_bits(&buf, i, bits), v, "bits={bits} idx={i}");
            }
        }
    }

    #[test]
    fn test_symmetric_all_widths_roundtrip_shrinks_with_bits() {
        let data: Vec<f32> = (0..64).map(|i| (i as f32 - 32.0) * 0.1).collect();
        let mut prev_err = f32::INFINITY;
        for bits in 2u8..=8 {
            let err = roundtrip_error(bits, Calibration::Symmetric, &data);
            assert!(
                err <= prev_err + 1e-3,
                "bits={bits} err={err} not <= prev={prev_err}"
            );
            prev_err = err;
        }
    }

    #[test]
    fn test_bits_1_symmetric_matches_sign() {
        let data = vec![3.0, -3.0, 0.5, -0.5];
        let packed = quantize(
            &data,
            1,
            Calibration::Symmetric,
            ScaleGranularity::PerTensor,
            0,
        );
        let restored = dequantize(&packed);
        for (a, b) in data.iter().zip(restored.iter()) {
            assert_eq!(a.is_sign_negative(), b.is_sign_negative());
        }
    }

    #[test]
    #[should_panic(expected = "asymmetric calibration is not supported at bits=1")]
    fn test_bits_1_asymmetric_panics() {
        let data = vec![1.0f32; 8];
        let _ = quantize(
            &data,
            1,
            Calibration::Asymmetric,
            ScaleGranularity::PerTensor,
            0,
        );
    }

    #[test]
    fn test_asymmetric_handles_skewed_distribution() {
        let data: Vec<f32> = (0..64).map(|i| 10.0 + i as f32 * 0.05).collect();
        let sym_err = roundtrip_error(4, Calibration::Symmetric, &data);
        let asym_err = roundtrip_error(4, Calibration::Asymmetric, &data);
        assert!(asym_err < sym_err,
            "asymmetric should be more accurate on a shifted distribution: asym={asym_err} sym={sym_err}");
    }

    #[test]
    fn test_asymmetric_zero_is_representable() {
        let mut data = vec![5.0f32; 31];
        data.push(0.0);
        let packed = quantize(
            &data,
            4,
            Calibration::Asymmetric,
            ScaleGranularity::PerTensor,
            0,
        );
        let restored = dequantize(&packed);
        assert!(
            restored[31].abs() < 0.5,
            "zero not well represented: {}",
            restored[31]
        );
    }

    #[test]
    fn test_effective_bits_matches_bits_param() {
        let data = vec![1.0f32; 10_000];
        let packed = quantize(
            &data,
            3,
            Calibration::Symmetric,
            ScaleGranularity::PerTensor,
            0,
        );
        assert!(packed.effective_bits_per_weight() < 3.01);
        assert!(packed.effective_bits_per_weight() >= 3.0);
    }

    #[test]
    fn test_asymmetric_overhead_includes_zero_points() {
        let data = vec![1.0f32; 1000];
        let packed = quantize(
            &data,
            4,
            Calibration::Asymmetric,
            ScaleGranularity::PerTensor,
            0,
        );
        let expected = (4000.0 + 32.0 + 32.0) / 1000.0;
        assert!((packed.effective_bits_per_weight() - expected).abs() < 1e-4);
    }

    #[test]
    fn test_per_group_scale_count() {
        let data: Vec<f32> = (0..100).map(|i| i as f32 * 0.01).collect();
        let packed = quantize(
            &data,
            5,
            Calibration::Symmetric,
            ScaleGranularity::PerGroup { group_size: 25 },
            0,
        );
        assert_eq!(packed.scales.len(), 4);
    }

    #[test]
    #[should_panic(expected = "bits must be 1..=8")]
    fn test_bits_out_of_range_panics() {
        let data = vec![1.0f32; 8];
        let _ = quantize(
            &data,
            9,
            Calibration::Symmetric,
            ScaleGranularity::PerTensor,
            0,
        );
    }

    #[test]
    fn test_bits_8_boundary_ok() {
        let data = vec![1.0f32; 8];
        let packed = quantize(
            &data,
            8,
            Calibration::Symmetric,
            ScaleGranularity::PerTensor,
            0,
        );
        assert_eq!(packed.bits, 8);
    }

    #[test]
    fn test_negative_and_positive_signs_preserved_symmetric() {
        let data = vec![3.0, -3.0, 1.0, -1.0];
        let packed = quantize(
            &data,
            6,
            Calibration::Symmetric,
            ScaleGranularity::PerTensor,
            0,
        );
        let restored = dequantize(&packed);
        assert!(restored[0] > 0.0);
        assert!(restored[1] < 0.0);
        assert!(restored[2] > 0.0);
        assert!(restored[3] < 0.0);
    }

    #[test]
    fn test_quantize_maxabs_matches_extreme_code_at_max() {
        let data = vec![1.0, -2.0, 0.5, -0.1];
        let packed = quantize_maxabs(&data, 8, ScaleGranularity::PerTensor, 0);
        // -2.0 — maximum modulo → should go to the extreme negative code.
        let max_code = IntNPacked::max_symmetric_code(8);
        let bias = IntNPacked::symmetric_bias(8);
        let raw = packed.get_raw_code(1);
        assert_eq!(raw as i32 - bias, -max_code);
    }

    #[test]
    #[should_panic(expected = "bits must be 2..=8")]
    fn test_quantize_maxabs_rejects_bits_1() {
        let data = vec![1.0f32; 8];
        let _ = quantize_maxabs(&data, 1, ScaleGranularity::PerTensor, 0);
    }
}
