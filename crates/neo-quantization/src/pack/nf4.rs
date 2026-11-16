// neo-quantization/src/pack/nf4.rs
//
// NF4 (4-bit NormalFloat) - QLoRA-style non-uniform quantization.
//
// Motivation
// ─────────
// All intN schemes of this crate (int1..int8, ternary) use UNIFORM
// code grid: `q * scale`, codes equally spaced. Pretrained weights
// neural networks are empirically distributed close to N(0, σ) (see Dettmers et
// al., "QLoRA: Efficient Finetuning of Quantized LLMs", 2023) - under
// Gaussian distribution uniform grid wastes resolution
// near the tails and not enough near zero, where the density of the weights is higher
// everything. NF4 instead uses a fixed, precomputed
// Uneven grid of 16 levels - quantiles of the standard normal
// distribution - so that each level covers an equal PROBABILITY
// mass, not an equal numerical interval. This is the only scheme in the crate
// without the “running time rounding formula” — 16 level-table constants
// calculated once offline and hardcoded (same approach as in
// original bitsandbytes implementation).
//
// ## Format
//
//   Quantization: for each weight, find the CLOSEST level from
//   NF4_LEVELS (after normalization to `absmax` group), write it down
//   4-bit index (0..15).
//   Dequantization: `W_hat = NF4_LEVELS[code] * absmax`.
//
//   `absmax = max(|W|)` by group (not mean - NF4_LEVELS itself already carries
//   "shape" of the Gaussian distribution, all that remains is to calibrate
//   maximum scale, exactly like the original QLoRA double quantization
//   first level).
//
// ## Difference from double quantization QLoRA
//
// The original QLoRA additionally quantizes the `absmax` scalars themselves
// (8-bit, "double quantization") to save memory on blockwise
// scales with a very small group_size (usually 64). Here `scales`
// stored as normal `Vec<f32>` (not quantized) - the same choice
// simplicity/overhead trade-off, as in all other pack::*
// files of this crate (neither ternary.rs nor int8.rs quantize their
// scales). If double quantization is needed, this is separate,
// orthogonal expansion on top of `scales: Vec<f32>`, without changing
// NF4Packed format.

use crate::scheme::ScaleGranularity;

// ═══════════════════════════════════════════════════════════════════════════
// § 1 TABLE OF LEVELS (16 quantiles N(0,1), normalized to [-1,1])
// ═══════════════════════════════════════════════════════════════════════════

/// 16 fixed NF4 levels - the same values as the original
/// implementation bitsandbytes (asymmetric relative to zero: 8 negative
/// levels, 1 zero, 7 positive - because 0 is already for sure
/// Let's imagine one of the levels, saving the second code under "almost zero"
/// not necessary). Values are N(0,1) quantiles normalized by division by
/// maximum absolute quantile so that `NF4_LEVELS` fits
/// in `[-1, 1]`.
pub const NF4_LEVELS: [f32; 16] = [
    -1.0,
    -0.696_192_8,
    -0.525_073_05,
    -0.394_917_5,
    -0.284_441_38,
    -0.184_773_43,
    -0.091_050_036,
    0.0,
    0.079_580_3,
    0.160_930_2,
    0.246_112_3,
    0.337_915_24,
    0.440_709_83,
    0.562_617,
    0.722_956_84,
    1.0,
];

/// Find the index of the closest level in `NF4_LEVELS` to the normalized one
/// value `x ∈ [-1, 1]` (linear search - table of only 16 elements,
/// a binary search wouldn't pay off at that length; called one at a time
/// times per weight during quantization).
#[inline]
fn nearest_level(x: f32) -> u8 {
    let mut best_idx = 0usize;
    let mut best_dist = f32::INFINITY;
    for (i, &level) in NF4_LEVELS.iter().enumerate() {
        let dist = (x - level).abs();
        if dist < best_dist {
            best_dist = dist;
            best_idx = i;
        }
    }
    best_idx as u8
}

// ═══════════════════════════════════════════════════════════════════════════
// § 2  PACKED CONTAINER
// ═══════════════════════════════════════════════════════════════════════════

/// Packed NF4 tensor.
#[derive(Debug, Clone, PartialEq)]
pub struct Nf4Packed {
    /// 4-bit indexes in `NF4_LEVELS`, 2 weights per byte (same nibble-
    /// packing that Int4Packed: even index → low nibble).
    pub codes: Vec<u8>,
    /// `absmax` by groups.
    pub scales: Vec<f32>,
    /// Group size in elements. `usize::MAX` = PerTensor.
    pub group_size: usize,
    /// Initial number of scales.
    pub numel: usize,
}

impl Nf4Packed {
    /// Size in bytes occupied by code packing (without scales).
    pub fn bytes_len(&self) -> usize {
        self.codes.len()
    }

    /// Actual bits/weight taking into account overhead on scales.
    pub fn effective_bits_per_weight(&self) -> f32 {
        let data_bits = self.codes.len() as f32 * 8.0;
        let scale_bits = self.scales.len() as f32 * 32.0;
        (data_bits + scale_bits) / self.numel as f32
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

    /// Read the 4-bit index into `NF4_LEVELS` for element `idx`.
    #[inline]
    pub fn get_code(&self, idx: usize) -> u8 {
        let byte = self.codes[idx / 2];
        if idx.is_multiple_of(2) {
            byte & 0x0F
        } else {
            (byte >> 4) & 0x0F
        }
    }

    /// Dequantize one element by flat index in f32.
    #[inline]
    pub fn get_f32(&self, idx: usize) -> f32 {
        let code = self.get_code(idx) as usize;
        let level = NF4_LEVELS[code];
        let absmax = self.scales[self.group_of(idx)];
        level * absmax
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// § 3  QUANTIZE / DEQUANTIZE
// ═══════════════════════════════════════════════════════════════════════════

/// Quantize slice f32 in NF4.
///
/// `absmax = max(|W|)` by group, each weight is normalized by `absmax`
/// and is replaced by the nearest level `NF4_LEVELS`.
pub fn quantize_nf4(data: &[f32], granularity: ScaleGranularity, row_len: usize) -> Nf4Packed {
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

    for g in 0..n_groups {
        let (start, end) = group_bounds(g, group_size, numel);
        let slice = &data[start..end];
        let absmax = slice.iter().fold(0.0f32, |acc, w| acc.max(w.abs()));
        scales[g] = absmax.max(f32::EPSILON);
    }

    let n_bytes = numel.div_ceil(2);
    let mut codes = vec![0u8; n_bytes];

    for (idx, &w) in data.iter().enumerate() {
        let g = if group_size == usize::MAX {
            0
        } else {
            idx / group_size
        };
        let absmax = scales[g];
        let normalized = (w / absmax).clamp(-1.0, 1.0);
        let code = nearest_level(normalized);

        let byte_idx = idx / 2;
        if idx.is_multiple_of(2) {
            codes[byte_idx] = (codes[byte_idx] & 0xF0) | code;
        } else {
            codes[byte_idx] = (codes[byte_idx] & 0x0F) | (code << 4);
        }
    }

    Nf4Packed {
        codes,
        scales,
        group_size,
        numel,
    }
}

/// Completely expand the NF4 package back to `Vec<f32>`.
pub fn dequantize_nf4(packed: &Nf4Packed) -> Vec<f32> {
    (0..packed.numel).map(|i| packed.get_f32(i)).collect()
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

// ═══════════════════════════════════════════════════════════════════════════
// § 4 TESTS
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_levels_are_sorted() {
        for w in NF4_LEVELS.windows(2) {
            assert!(w[0] < w[1], "NF4_LEVELS must be strictly increasing");
        }
    }

    #[test]
    fn test_levels_bounded_in_unit_range() {
        for &l in NF4_LEVELS.iter() {
            assert!((-1.0..=1.0).contains(&l));
        }
    }

    #[test]
    fn test_zero_is_exact_level() {
        assert!(NF4_LEVELS.contains(&0.0));
    }

    #[test]
    fn test_packing_density() {
        let data = vec![1.0f32; 10];
        let packed = quantize_nf4(&data, ScaleGranularity::PerTensor, 0);
        assert_eq!(packed.bytes_len(), 5); // ceil(10/2)
    }

    #[test]
    fn test_gaussian_data_low_error() {
        // Approximately Gaussian data (Box-Muller without external rand dependency
        // in the core path - a simple deterministic sequence).
        let data: Vec<f32> = (1..200)
            .map(|i| {
                let u1 = (i as f32) / 200.0;
                let u2 = ((i * 37) % 200) as f32 / 200.0;
                (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos()
            })
            .collect();

        let nf4_packed = quantize_nf4(&data, ScaleGranularity::PerTensor, 0);
        let nf4_restored = dequantize_nf4(&nf4_packed);
        let nf4_err: f32 = data
            .iter()
            .zip(nf4_restored.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f32>()
            / data.len() as f32;

        // Compare with equivalent-bit uniform int4 (max-abs).
        let int4_packed =
            crate::pack::intn::quantize_maxabs(&data, 4, ScaleGranularity::PerTensor, 0);
        let int4_restored = crate::pack::intn::dequantize(&int4_packed);
        let int4_err: f32 = data
            .iter()
            .zip(int4_restored.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f32>()
            / data.len() as f32;

        // NF4 should be no worse (usually noticeably better) on Gaussian data.
        assert!(nf4_err <= int4_err * 1.5,
            "NF4 err={nf4_err} should be competitive with uniform int4 err={int4_err} on gaussian-like data");
    }

    #[test]
    fn test_roundtrip_length() {
        let data: Vec<f32> = (0..77).map(|i| (i as f32 - 38.0) * 0.03).collect();
        let packed = quantize_nf4(&data, ScaleGranularity::PerGroup { group_size: 16 }, 0);
        let restored = dequantize_nf4(&packed);
        assert_eq!(restored.len(), 77);
    }

    #[test]
    fn test_extreme_values_clip_to_endpoints() {
        let data = vec![1000.0, -1000.0];
        let packed = quantize_nf4(&data, ScaleGranularity::PerTensor, 0);
        let restored = dequantize_nf4(&packed);
        assert!((restored[0] - 1000.0).abs() < 1e-2);
        assert!((restored[1] + 1000.0).abs() < 1e-2);
    }
}
