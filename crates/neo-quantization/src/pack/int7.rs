// neo-quantization/src/pack/int7.rs
//
// 7-bit quantization, symmetric and asymmetric. Thin wrapper over
// `intn::IntNPacked` (bits=7) - the same pattern as int2.rs/int6.rs.
// Rarely used in practice (7 bits are not aligned to either byte or
// to powers of two for typical vector ISAs), but included for completeness
// coverage "all N from 1 to 8" - the same bit-cursor engine processes it
// Without special code, it costs nothing to keep this option in the API.

use crate::pack::intn::{self, Calibration, IntNPacked};
use crate::scheme::ScaleGranularity;

/// The packed 7-bit tensor is a thin wrapper over `IntNPacked`.
#[derive(Debug, Clone, PartialEq)]
pub struct Int7Packed {
    inner: IntNPacked,
}

impl Int7Packed {
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

/// Symmetric 7-bit quantization, `alpha = max(|W|) / 63` by group.
pub fn quantize_int7(data: &[f32], granularity: ScaleGranularity, row_len: usize) -> Int7Packed {
    Int7Packed {
        inner: intn::quantize_maxabs(data, 7, granularity, row_len),
    }
}

/// Asymmetric 7-bit quantization (GPTQ/AWQ style).
pub fn quantize_int7_asymmetric(
    data: &[f32],
    granularity: ScaleGranularity,
    row_len: usize,
) -> Int7Packed {
    Int7Packed {
        inner: intn::quantize(data, 7, Calibration::Asymmetric, granularity, row_len),
    }
}

/// Completely expand the Int7 package back to `Vec<f32>`.
pub fn dequantize_int7(packed: &Int7Packed) -> Vec<f32> {
    intn::dequantize(&packed.inner)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_packing_density() {
        // 8 weights * 7 bits = 56 bits = 7 bytes
        let data = vec![1.0f32; 8];
        let packed = quantize_int7(&data, ScaleGranularity::PerTensor, 0);
        assert_eq!(packed.bytes_len(), 7);
    }

    #[test]
    fn test_effective_bits_near_7() {
        let data = vec![0.5f32; 7000];
        let packed = quantize_int7(&data, ScaleGranularity::PerTensor, 0);
        assert!(packed.effective_bits_per_weight() < 7.01);
        assert!(packed.effective_bits_per_weight() >= 7.0);
    }

    #[test]
    fn test_roundtrip_error_small() {
        let data: Vec<f32> = (0..64).map(|i| (i as f32 - 32.0) * 0.1).collect();
        let packed = quantize_int7(&data, ScaleGranularity::PerGroup { group_size: 32 }, 0);
        let restored = dequantize_int7(&packed);
        for (a, b) in data.iter().zip(restored.iter()) {
            assert!(
                (a - b).abs() < 0.06,
                "int7 roundtrip error too large: {a} vs {b}"
            );
        }
    }

    #[test]
    fn test_asymmetric_variant_roundtrips() {
        let data: Vec<f32> = (0..70).map(|i| 9.0 + i as f32 * 0.1).collect();
        let packed = quantize_int7_asymmetric(&data, ScaleGranularity::PerTensor, 0);
        assert!(packed.is_asymmetric());
        let restored = dequantize_int7(&packed);
        assert_eq!(restored.len(), data.len());
    }
}
