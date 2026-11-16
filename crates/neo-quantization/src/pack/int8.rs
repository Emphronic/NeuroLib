// neo-quantization/src/pack/int8.rs
//
// Symmetrical 8-bit quantization (compatible with original
// implementation - `alpha = max(|W|) / 127`) + new asymmetric option
// (GPTQ/AWQ-style, zero-point). Thin wrapper over `intn::IntNPacked`
// (bits=8) - see int4.rs (same wrapper pattern) and intn.rs (common engine).
//
// IMPORTANT: the public field `data: Vec<i8>` of the original `Int8Packed` has been replaced
// to private `inner: IntNPacked` + method `.data()`, returning
// equivalent to `Vec<i8>` - the direct field would have to be either materialized
// again every time it is called (expensive if the calling code does this in
// cycle), or store redundantly next to `codes` (memory duplication).
// `pack::mod.rs` is the only external place that read `p.data.len()`
// directly (see `data_bytes()`); updated by `p.data_len()`.

use crate::pack::intn::{self, Calibration, IntNPacked};
use crate::scheme::ScaleGranularity;

/// The packed 8-bit tensor is a thin wrapper over `IntNPacked`.
#[derive(Debug, Clone, PartialEq)]
pub struct Int8Packed {
    inner: IntNPacked,
}

impl Int8Packed {
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

    /// Equivalent to the original `data.len()` - data size in bytes
    /// (1 byte/weight for 8-bit packing, so same as numel).
    pub fn data_len(&self) -> usize {
        self.inner.codes_bytes()
    }

    /// Materialize data in `Vec<i8>` (only makes sense for
    /// symmetric versions - asymmetric code is unsigned `[0,255]`, which is not
    /// displayed correctly in `i8`; panics if called on
    /// asymmetric packaging - use `get_f32`/`get_raw_code`).
    pub fn to_i8_vec(&self) -> Vec<i8> {
        assert_eq!(self.inner.calibration, Calibration::Symmetric,
            "Int8Packed::to_i8_vec: asymmetric codes are unsigned, use get_f32/get_raw_code instead");
        let bias = IntNPacked::symmetric_bias(8);
        (0..self.inner.numel)
            .map(|i| (self.inner.get_raw_code(i) as i32 - bias) as i8)
            .collect()
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

/// Symmetric 8-bit quantization, `alpha = max(|W|) / 127` by group -
/// behavior is identical to the original int8.rs (before merging with intn.rs).
pub fn quantize_int8(data: &[f32], granularity: ScaleGranularity, row_len: usize) -> Int8Packed {
    let inner = intn::quantize_maxabs(data, 8, granularity, row_len);
    Int8Packed { inner }
}

/// Asymmetric 8-bit quantization (GPTQ/AWQ style).
pub fn quantize_int8_asymmetric(
    data: &[f32],
    granularity: ScaleGranularity,
    row_len: usize,
) -> Int8Packed {
    let inner = intn::quantize(data, 8, Calibration::Asymmetric, granularity, row_len);
    Int8Packed { inner }
}

/// Completely expand the Int8 package back to `Vec<f32>`.
pub fn dequantize_int8(packed: &Int8Packed) -> Vec<f32> {
    intn::dequantize(&packed.inner)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_max_abs_element_maps_to_extreme_code() {
        let data = vec![1.0, -2.0, 0.5, -0.1];
        let packed = quantize_int8(&data, ScaleGranularity::PerTensor, 0);
        let codes = packed.to_i8_vec();
        // -2.0 — maximum modulo → should go to code -127
        assert_eq!(codes[1], -127);
    }

    #[test]
    fn test_roundtrip_error_small() {
        let data: Vec<f32> = (0..64).map(|i| (i as f32 - 32.0) * 0.1).collect();
        let packed = quantize_int8(&data, ScaleGranularity::PerGroup { group_size: 32 }, 0);
        let restored = dequantize_int8(&packed);
        for (a, b) in data.iter().zip(restored.iter()) {
            assert!(
                (a - b).abs() < 0.05,
                "int8 roundtrip error too large: {a} vs {b}"
            );
        }
    }

    #[test]
    fn test_effective_bits_near_8() {
        let data = vec![0.5f32; 1000];
        let packed = quantize_int8(&data, ScaleGranularity::PerTensor, 0);
        assert!(packed.effective_bits_per_weight() < 8.1);
    }

    #[test]
    fn test_asymmetric_more_accurate_on_skewed_data() {
        let data: Vec<f32> = (0..128).map(|i| 50.0 + i as f32 * 0.1).collect();
        let sym = quantize_int8(&data, ScaleGranularity::PerTensor, 0);
        let asym = quantize_int8_asymmetric(&data, ScaleGranularity::PerTensor, 0);
        let sym_err: f32 = dequantize_int8(&sym)
            .iter()
            .zip(data.iter())
            .map(|(a, b)| (a - b).abs())
            .sum();
        let asym_err: f32 = dequantize_int8(&asym)
            .iter()
            .zip(data.iter())
            .map(|(a, b)| (a - b).abs())
            .sum();
        assert!(asym_err <= sym_err, "asym={asym_err} sym={sym_err}");
    }

    #[test]
    #[should_panic(expected = "asymmetric codes are unsigned")]
    fn test_to_i8_vec_panics_on_asymmetric() {
        let data = vec![1.0f32; 8];
        let packed = quantize_int8_asymmetric(&data, ScaleGranularity::PerTensor, 0);
        let _ = packed.to_i8_vec();
    }
}
