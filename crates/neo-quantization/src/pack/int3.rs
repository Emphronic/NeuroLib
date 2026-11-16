// neo-quantization/src/pack/int3.rs
//
// 3-bit quantization, symmetric and asymmetric. Thin wrapper over
// `intn::IntNPacked` (bits=3) - see int2.rs/int4.rs for the same pattern.
//
// The only intN variant where the elements systematically intersect
// byte boundaries for ANY group_size (3 does not divide 8 completely under any circumstances
// alignment - 8 elements = 24 bits = exactly 3 bytes, but separate
// the element can start on any bit within this triple). It's exactly
// the case for which `intn::write_bits`/`read_bits` were implemented
// as an honest bit-cursor, and not as a byte-aligned nibble/bit logic (see.
// int4.rs/int1.rs, where N=4/N=1 make alignment trivial).

use crate::pack::intn::{self, Calibration, IntNPacked};
use crate::scheme::ScaleGranularity;

/// A packed 3-bit tensor is a thin wrapper over `IntNPacked`.
#[derive(Debug, Clone, PartialEq)]
pub struct Int3Packed {
    inner: IntNPacked,
}

impl Int3Packed {
    /// Wrap a ready-made intN engine (for quantization paths that build
    /// `IntNPacked` directly - for example, with stochastic rounding).
    pub(crate) fn from_inner(inner: IntNPacked) -> Self {
        Self { inner }
    }

    /// Access to the general intN engine (bits, codes, scales, zero-points) - needed
    /// fused GEMM cores (`neo-synk::quant`) to read packing without dequantization.
    pub fn as_intn(&self) -> &IntNPacked {
        &self.inner
    }

    /// Size in bytes occupied by code packing (without scales/zero_points).
    pub fn bytes_len(&self) -> usize {
        self.inner.codes_bytes()
    }

    /// Actual bits/weight taking into account overhead on scales (and zero_points,
    /// if the packaging is asymmetric).
    pub fn effective_bits_per_weight(&self) -> f32 {
        self.inner.effective_bits_per_weight()
    }

    /// Group index (scale bucket) for element `idx`.
    #[inline]
    pub fn group_of(&self, idx: usize) -> usize {
        self.inner.group_of(idx)
    }

    /// Dequantize one element by flat index in f32.
    #[inline]
    pub fn get_f32(&self, idx: usize) -> f32 {
        self.inner.get_f32(idx)
    }

    /// `true` if the package uses asymmetric calibration (zero-point).
    pub fn is_asymmetric(&self) -> bool {
        self.inner.calibration == Calibration::Asymmetric
    }

    /// Zero-point group `g` (only for asymmetric; `0` for symmetric).
    pub fn zero_point_of_group(&self, g: usize) -> i32 {
        self.inner.zero_points.get(g).copied().unwrap_or(0)
    }

    /// Scales by group.
    pub fn scales(&self) -> &[f32] {
        &self.inner.scales
    }
    /// Group size in elements. `usize::MAX` = PerTensor.
    pub fn group_size(&self) -> usize {
        self.inner.group_size
    }
    /// Initial number of scales.
    pub fn numel(&self) -> usize {
        self.inner.numel
    }
}

/// Symmetric 3-bit quantization, `alpha = max(|W|) / 3` by group.
pub fn quantize_int3(data: &[f32], granularity: ScaleGranularity, row_len: usize) -> Int3Packed {
    Int3Packed {
        inner: intn::quantize_maxabs(data, 3, granularity, row_len),
    }
}

/// Asymmetric 3-bit quantization (GPTQ/AWQ style).
pub fn quantize_int3_asymmetric(
    data: &[f32],
    granularity: ScaleGranularity,
    row_len: usize,
) -> Int3Packed {
    Int3Packed {
        inner: intn::quantize(data, 3, Calibration::Asymmetric, granularity, row_len),
    }
}

/// Completely expand the Int3-packing back to `Vec<f32>`.
pub fn dequantize_int3(packed: &Int3Packed) -> Vec<f32> {
    intn::dequantize(&packed.inner)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_packing_density() {
        // 3 bits * 8 weights = 24 bits = 3 bytes exactly
        let data = vec![1.0f32; 8];
        let packed = quantize_int3(&data, ScaleGranularity::PerTensor, 0);
        assert_eq!(packed.bytes_len(), 3);
    }

    #[test]
    fn test_non_byte_aligned_count() {
        // 5 weights * 3 bits = 15 bits -> 2 bytes, element 4 crosses the boundary
        let data = vec![1.0, -1.0, 0.5, -0.5, 0.25];
        let packed = quantize_int3(&data, ScaleGranularity::PerTensor, 0);
        assert_eq!(packed.bytes_len(), 2);
        let restored = dequantize_int3(&packed);
        assert_eq!(restored.len(), 5);
    }

    #[test]
    fn test_effective_bits_near_3() {
        let data = vec![0.5f32; 3000];
        let packed = quantize_int3(&data, ScaleGranularity::PerTensor, 0);
        // 3000*3=9000 bits -> ceil to byte 9000 bits (multiples of 8) + 32 bits
        // scale overhead = 9032/3000 = 3.0107 bits/weight. Threshold 3.02 leaves
        // a small margin instead of the exact 3.01, which did not pass due to
        // this byte padding.
        assert!(
            packed.effective_bits_per_weight() < 3.02,
            "effective_bits_per_weight={}",
            packed.effective_bits_per_weight()
        );
        assert!(packed.effective_bits_per_weight() >= 3.0);
    }

    #[test]
    fn test_asymmetric_variant_roundtrips() {
        let data: Vec<f32> = (0..40).map(|i| 4.0 + i as f32 * 0.1).collect();
        let packed = quantize_int3_asymmetric(&data, ScaleGranularity::PerTensor, 0);
        assert!(packed.is_asymmetric());
        let restored = dequantize_int3(&packed);
        assert_eq!(restored.len(), data.len());
    }
}
