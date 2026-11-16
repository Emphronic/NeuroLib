// neo-quantization/src/pack/ternary.rs
//
// Ternary packing: weights ∈ {-1, 0, +1} (BitNet b1.58 / Qwen3-like models).
//
// ## Storage format
//
//   The theoretical limit is log2(3) ≈ 1.58 bits/weight, but the packing is exactly 1.58
//   bits" (base-3 arithmetic of 5 trits per byte) makes the GEMV/GEMM core
//   noticeably more expensive (you need to unpack base-3 numbers before each FMA).
//   Instead, we use 2 bits/weight - a simple byte-aligned encoding:
//
//     00 =  0
//     01 = +1
//     10 = -1
//     11 = unused (reserved)
//
//   4 weights per byte. Gives 2.0 bits/weight actually (not 1.58) - compromise
//   "almost theoretical limit" in exchange for trivial unpack logic in
//   GEMM core (two shifts and a mask, no division/multiplication by 3). The same
//   trade-off does most practical BitNet runtimes (bitnet.cpp
//   also does not use pure base-3 packaging in the hot path).
//
// ## Scale (alpha)
//
//   Classic BitNet b1.58: `alpha = mean(|W|)` by range covered
//   (entire tensor for PerTensor, one group for PerGroup/PerChannel).
//   Quantization: `q = round(clip(W / alpha, -1, 1))`, dequantization:
//   `W_hat = q * alpha`.
//
// ## Layout `TernaryPacked`
//
//   bits: Vec<u8>, packed 2-bit codes, 4 weights/byte
//   scales: Vec<f32>, alpha by groups (len==1 for PerTensor)
//   group_size: usize, elements per 1 scale (usize::MAX = whole tensor)
//   numel: usize the initial number of weights (for last byte padding)

use crate::rng::{round_or_stochastic, XorShift64};
use crate::scheme::ScaleGranularity;

// ═══════════════════════════════════════════════════════════════════════════
// § 1 CODES
// ═══════════════════════════════════════════════════════════════════════════

const CODE_ZERO: u8 = 0b00;
const CODE_POS: u8 = 0b01;
const CODE_NEG: u8 = 0b10;

#[inline(always)]
const fn code_to_val(code: u8) -> i8 {
    match code {
        CODE_ZERO => 0,
        CODE_POS => 1,
        CODE_NEG => -1,
        _ => 0, // 0b11 reserved - treated as 0 (safe default)
    }
}

#[inline(always)]
const fn val_to_code(v: i8) -> u8 {
    match v {
        0 => CODE_ZERO,
        1 => CODE_POS,
        -1 => CODE_NEG,
        _ => CODE_ZERO, // should not happen after clip(-1,1)
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// § 2  PACKED CONTAINER
// ═══════════════════════════════════════════════════════════════════════════

/// Packed ternary tensor.
#[derive(Debug, Clone, PartialEq)]
pub struct TernaryPacked {
    /// 2-bit codes, 4 weights per byte, little-endian on bit pairs
    /// (element `i` lives in `bits[i/4]`, bits `(i%4)*2 .. (i%4)*2+2`).
    pub bits: Vec<u8>,
    /// Alpha scales by group. `scales.len() == 1` means per-tensor.
    pub scales: Vec<f32>,
    /// Group size in elements covered by one `scales[i]`.
    /// `usize::MAX` — marker “the entire tensor in one group” (PerTensor).
    pub group_size: usize,
    /// Initial number of weights (last byte may be partially filled
    /// zero padding if `numel % 4 != 0`).
    pub numel: usize,
}

impl TernaryPacked {
    /// The size in bytes that the package actually takes up (without taking into account scales).
    pub fn bits_bytes(&self) -> usize {
        self.bits.len()
    }

    /// Actual number of bits per weight, taking into account overhead on scales
    /// (`4 * scales.len() * 8` bits are distributed across all weights).
    pub fn effective_bits_per_weight(&self) -> f32 {
        let bits_total = self.bits.len() as f32 * 8.0;
        let scale_bits = self.scales.len() as f32 * 32.0;
        (bits_total + scale_bits) / self.numel as f32
    }

    /// Read one code by flat index without full unpacking.
    #[inline]
    pub fn get_code(&self, idx: usize) -> u8 {
        let byte = self.bits[idx / 4];
        let shift = (idx % 4) * 2;
        (byte >> shift) & 0b11
    }

    /// The group (scale-bucket) index for this element.
    #[inline]
    pub fn group_of(&self, idx: usize) -> usize {
        if self.group_size == usize::MAX {
            0
        } else {
            idx / self.group_size
        }
    }

    /// Dequantize one element in f32.
    #[inline]
    pub fn get_f32(&self, idx: usize) -> f32 {
        let code = self.get_code(idx);
        let alpha = self.scales[self.group_of(idx)];
        code_to_val(code) as f32 * alpha
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// § 3  QUANTIZE (float → ternary)
// ═══════════════════════════════════════════════════════════════════════════

/// Quantize the f32 slice into ternary packaging.
///
/// `granularity` determines how much alpha will be calculated:
///   • `PerTensor` → 1 alpha for the entire `data`
///   • `PerChannel` → 1 alpha for every `row_len` elements (calling
///     the code must pass `row_len == in_features` via `PerGroup`)
///   • `PerGroup{group_size}` → 1 alpha for every `group_size` elements
pub fn quantize_ternary(
    data: &[f32],
    granularity: ScaleGranularity,
    row_len: usize,
) -> TernaryPacked {
    quantize_ternary_impl(data, granularity, row_len, None)
}

/// Same as [`quantize_ternary`], but `W / alpha` is rounded to
/// `{-1, 0, +1}` STOCHASTICALLY: up with fractional part probability, down -
/// with additional (estimate unbiased within `[-1, 1]`). `alpha` is computed
/// as usual (`mean(|W|)` by group), RNG is determined by `seed`.
pub fn quantize_ternary_stochastic(
    data: &[f32],
    granularity: ScaleGranularity,
    row_len: usize,
    seed: u64,
) -> TernaryPacked {
    quantize_ternary_impl(data, granularity, row_len, Some(XorShift64::new(seed)))
}

/// General implementation: `rng == None` - round-to-nearest (`quantize_ternary`),
/// `Some` - stochastic rounding.
fn quantize_ternary_impl(
    data: &[f32],
    granularity: ScaleGranularity,
    row_len: usize,
    mut rng: Option<XorShift64>,
) -> TernaryPacked {
    let numel = data.len();
    let group_size = match granularity {
        ScaleGranularity::PerTensor => usize::MAX,
        ScaleGranularity::PerChannel => row_len,
        ScaleGranularity::PerGroup { group_size } => group_size,
    };

    let n_groups = if group_size == usize::MAX {
        1
    } else {
        numel.div_ceil(group_size)
    };

    let mut scales = vec![0.0f32; n_groups];

    // ── Step 1: alpha = mean(|W|) for each group ──────────────────────
    for g in 0..n_groups {
        let (start, end) = group_bounds(g, group_size, numel);
        let slice = &data[start..end];
        let mean_abs: f32 = if slice.is_empty() {
            0.0
        } else {
            slice.iter().map(|w| w.abs()).sum::<f32>() / slice.len() as f32
        };
        // Protection against null/degenerate tensor (all weights == 0):
        // alpha=0 would invalidate dequantization (0/0). Clump to
        // small epsilon preserves the correctness of round-trip for
        // borderline case of an all-zero block.
        scales[g] = mean_abs.max(f32::EPSILON);
    }

    // ── Step 2: round(clip(W/alpha, -1, 1)) → 2-bit code ───────────────
    let n_bytes = numel.div_ceil(4);
    let mut bits = vec![0u8; n_bytes];

    for (idx, &w) in data.iter().enumerate() {
        let g = if group_size == usize::MAX {
            0
        } else {
            idx / group_size
        };
        let alpha = scales[g];
        let scaled = (w / alpha).clamp(-1.0, 1.0);
        // Nearest (`rng == None`) or stochastic rounding; result
        // ∈ {-1, 0, 1}, because `scaled` is already clamped into [-1, 1].
        let q = round_or_stochastic(scaled, &mut rng) as i8;
        let code = val_to_code(q);

        let byte_idx = idx / 4;
        let shift = (idx % 4) * 2;
        bits[byte_idx] |= code << shift;
    }

    TernaryPacked {
        bits,
        scales,
        group_size,
        numel,
    }
}

/// Boundaries of the group `g` in the flat index space `[0, numel)`.
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

// ═══════════════════════════════════════════════════════════════════════════
// § 4  DEQUANTIZE (ternary → float)
// ═══════════════════════════════════════════════════════════════════════════

/// Completely expand the ternary package back to `Vec<f32>`.
///
/// Used to verify round-trip errors and as a fallback path to
/// backends that do not yet have a specialized ternary GEMM core
/// (more expensive than fused kernel, but always correct).
pub fn dequantize_ternary(packed: &TernaryPacked) -> Vec<f32> {
    (0..packed.numel).map(|i| packed.get_f32(i)).collect()
}

// ═══════════════════════════════════════════════════════════════════════════
// § 5 TESTS
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pack_unpack_roundtrip_per_tensor() {
        let data = vec![0.9, -0.9, 0.05, -0.05, 1.5, -1.5, 0.0, 0.3];
        let packed = quantize_ternary(&data, ScaleGranularity::PerTensor, 0);
        assert_eq!(packed.scales.len(), 1);
        let restored = dequantize_ternary(&packed);
        assert_eq!(restored.len(), data.len());
        // Values must be multiples of alpha and lie in {-alpha, 0, +alpha}
        let alpha = packed.scales[0];
        for &v in &restored {
            let ratio = (v / alpha).round();
            assert!(
                ratio == -1.0 || ratio == 0.0 || ratio == 1.0,
                "unexpected ternary value: {v}"
            );
        }
    }

    #[test]
    fn test_large_values_clip_to_plus_minus_one() {
        // Extremely large weights should be coded as +1/-1, not as "garbage".
        let data = vec![100.0, -100.0];
        let packed = quantize_ternary(&data, ScaleGranularity::PerTensor, 0);
        let restored = dequantize_ternary(&packed);
        let alpha = packed.scales[0];
        assert!((restored[0] - alpha).abs() < 1e-5);
        assert!((restored[1] + alpha).abs() < 1e-5);
    }

    #[test]
    fn test_all_zero_tensor_is_safe() {
        let data = vec![0.0f32; 16];
        let packed = quantize_ternary(&data, ScaleGranularity::PerTensor, 0);
        let restored = dequantize_ternary(&packed);
        assert!(restored.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_per_group_scales_count() {
        let data: Vec<f32> = (0..100).map(|i| (i as f32 - 50.0) * 0.03).collect();
        let packed = quantize_ternary(&data, ScaleGranularity::PerGroup { group_size: 32 }, 0);
        // ceil(100/32) = 4 groups
        assert_eq!(packed.scales.len(), 4);
        assert_eq!(packed.group_size, 32);
    }

    #[test]
    fn test_per_channel_scales_count() {
        // 3 "rows" of 10 elements
        let data: Vec<f32> = (0..30).map(|i| (i as f32) * 0.1 - 1.5).collect();
        let packed = quantize_ternary(&data, ScaleGranularity::PerChannel, 10);
        assert_eq!(packed.scales.len(), 3);
    }

    #[test]
    fn test_bits_packing_density() {
        // 8 weights → ceil(8/4) = 2 bytes
        let data = vec![0.5f32; 8];
        let packed = quantize_ternary(&data, ScaleGranularity::PerTensor, 0);
        assert_eq!(packed.bits_bytes(), 2);
    }

    #[test]
    fn test_non_multiple_of_4_numel() {
        // 7 weights are not a multiple of 4 - the last byte is partially filled.
        let data = vec![0.9, -0.9, 0.9, -0.9, 0.9, -0.9, 0.9];
        let packed = quantize_ternary(&data, ScaleGranularity::PerTensor, 0);
        assert_eq!(packed.bits_bytes(), 2); // ceil(7/4) = 2
        let restored = dequantize_ternary(&packed);
        assert_eq!(restored.len(), 7);
    }

    #[test]
    fn test_effective_bits_per_weight_near_2() {
        let data = vec![0.9f32; 1000];
        let packed = quantize_ternary(&data, ScaleGranularity::PerTensor, 0);
        // ~2 bits/weight + negligible overhead from one scale per 1000 weights
        assert!(packed.effective_bits_per_weight() < 2.1);
        assert!(packed.effective_bits_per_weight() > 2.0);
    }

    #[test]
    fn test_get_code_matches_dequantize() {
        let data = vec![1.0, -1.0, 0.0, 1.0, -1.0];
        let packed = quantize_ternary(&data, ScaleGranularity::PerTensor, 0);
        for i in 0..data.len() {
            let via_code =
                code_to_val(packed.get_code(i)) as f32 * packed.scales[packed.group_of(i)];
            assert_eq!(via_code, packed.get_f32(i));
        }
    }
}
