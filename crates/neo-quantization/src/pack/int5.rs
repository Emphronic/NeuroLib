// neo-quantization/src/pack/int5.rs
//
// 5-bit quantization, symmetric and asymmetric. Thin wrapper over
// `intn::IntNPacked` (bits=5) - the same pattern as int2.rs/int3.rs.
// Intermediate point between Q4-class (int4.rs) and Q6/Q8 - practical
// analogous to GGUF Q5_0/Q5_1 in spirit (not in byte layout - the same
// reproduced separately in pack::gguf, see next step).

use crate::pack::intn::{self, Calibration, IntNPacked};
use crate::scheme::ScaleGranularity;

/// A packed 5-bit tensor is a thin wrapper over `IntNPacked`.
#[derive(Debug, Clone, PartialEq)]
pub struct Int5Packed {
    inner: IntNPacked,
}

impl Int5Packed {
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

/// Symmetric 5-bit quantization, `alpha = max(|W|) / 15` by group.
pub fn quantize_int5(data: &[f32], granularity: ScaleGranularity, row_len: usize) -> Int5Packed {
    Int5Packed {
        inner: intn::quantize_maxabs(data, 5, granularity, row_len),
    }
}

/// Asymmetric 5-bit quantization (GPTQ/AWQ style).
pub fn quantize_int5_asymmetric(
    data: &[f32],
    granularity: ScaleGranularity,
    row_len: usize,
) -> Int5Packed {
    Int5Packed {
        inner: intn::quantize(data, 5, Calibration::Asymmetric, granularity, row_len),
    }
}

/// Completely expand the Int5-packing back to `Vec<f32>`.
pub fn dequantize_int5(packed: &Int5Packed) -> Vec<f32> {
    intn::dequantize(&packed.inner)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_packing_density() {
        // 8 weights * 5 bits = 40 bits = 5 bytes
        let data = vec![1.0f32; 8];
        let packed = quantize_int5(&data, ScaleGranularity::PerTensor, 0);
        assert_eq!(packed.bytes_len(), 5);
    }

    #[test]
    fn test_effective_bits_near_5() {
        let data = vec![0.5f32; 5000];
        let packed = quantize_int5(&data, ScaleGranularity::PerTensor, 0);
        assert!(packed.effective_bits_per_weight() < 5.01);
        assert!(packed.effective_bits_per_weight() >= 5.0);
    }

    #[test]
    fn test_roundtrip_error_bounded() {
        let data: Vec<f32> = (0..64).map(|i| (i as f32 - 32.0) * 0.15).collect();
        let packed = quantize_int5(&data, ScaleGranularity::PerGroup { group_size: 32 }, 0);
        let restored = dequantize_int5(&packed);
        for (a, b) in data.iter().zip(restored.iter()) {
            assert!(
                (a - b).abs() < 0.2,
                "int5 roundtrip error too large: {a} vs {b}"
            );
        }
    }

    #[test]
    fn test_asymmetric_variant_roundtrips() {
        let data: Vec<f32> = (0..50).map(|i| 6.0 + i as f32 * 0.1).collect();
        let packed = quantize_int5_asymmetric(&data, ScaleGranularity::PerTensor, 0);
        assert!(packed.is_asymmetric());
        let restored = dequantize_int5(&packed);
        assert_eq!(restored.len(), data.len());
    }
}
