// neo-quantization/src/pack/int2.rs
//
// 2-bit quantization, symmetric (`alpha = max(|W|) / 1`) and asymmetric
// (GPTQ/AWQ style). Thin wrapper over `intn::IntNPacked` (bits=2) —
// see int4.rs/int8.rs for the same pattern and intn.rs for the same engine.
//
// Code symmetric for N=2: range `[-1, 1]` (3 values out of 4 possible
// are used - `max_symmetric_code(2) = (2^1)-1 = 1`), i.e. actually
// {-1, 0, +1} - the same code power as `ternary.rs`, but with DIFFERENT
// calibration (max-abs here, mean-abs there) and OTHER byte layout
// (4 weights/byte is the same as ternary.rs, but ternary.rs uses
// special codes 00/01/10, not biased-arithmetic). Both schemes are left
// in the crate as different points on the "calibration × layout" curve - `Ternary`
// remains the canonical scheme for BitNet b1.58-style models (see.
// scheme.rs), `Int2` - general purpose option, consistent in spirit
// with Int4/Int8/... the rest intN.

use crate::pack::intn::{self, Calibration, IntNPacked};
use crate::scheme::ScaleGranularity;

/// A packed 2-bit tensor is a thin wrapper over `IntNPacked`.
#[derive(Debug, Clone, PartialEq)]
pub struct Int2Packed {
    inner: IntNPacked,
}

impl Int2Packed {
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

/// Symmetric 2-bit quantization, `alpha = max(|W|) / 1` by group.
pub fn quantize_int2(data: &[f32], granularity: ScaleGranularity, row_len: usize) -> Int2Packed {
    Int2Packed {
        inner: intn::quantize_maxabs(data, 2, granularity, row_len),
    }
}

/// Asymmetric 2-bit quantization (GPTQ/AWQ style).
pub fn quantize_int2_asymmetric(
    data: &[f32],
    granularity: ScaleGranularity,
    row_len: usize,
) -> Int2Packed {
    Int2Packed {
        inner: intn::quantize(data, 2, Calibration::Asymmetric, granularity, row_len),
    }
}

/// Completely expand the Int2-packing back to `Vec<f32>`.
pub fn dequantize_int2(packed: &Int2Packed) -> Vec<f32> {
    intn::dequantize(&packed.inner)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_packing_density() {
        // 2 bits/weight → 4 weights/byte
        let data = vec![1.0f32; 16];
        let packed = quantize_int2(&data, ScaleGranularity::PerTensor, 0);
        assert_eq!(packed.bytes_len(), 4);
    }

    #[test]
    fn test_effective_bits_near_2() {
        let data = vec![0.5f32; 4000];
        let packed = quantize_int2(&data, ScaleGranularity::PerTensor, 0);
        assert!(packed.effective_bits_per_weight() < 2.01);
        assert!(packed.effective_bits_per_weight() >= 2.0);
    }

    #[test]
    fn test_roundtrip_signs() {
        let data = vec![5.0, -5.0, 2.0, -2.0];
        let packed = quantize_int2(&data, ScaleGranularity::PerTensor, 0);
        let restored = dequantize_int2(&packed);
        assert!(restored[0] > 0.0);
        assert!(restored[1] < 0.0);
    }

    #[test]
    fn test_asymmetric_variant_roundtrips() {
        let data: Vec<f32> = (0..32).map(|i| 3.0 + i as f32 * 0.1).collect();
        let packed = quantize_int2_asymmetric(&data, ScaleGranularity::PerTensor, 0);
        assert!(packed.is_asymmetric());
        let restored = dequantize_int2(&packed);
        assert_eq!(restored.len(), data.len());
    }
}
