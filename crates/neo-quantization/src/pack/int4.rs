// neo-quantization/src/pack/int4.rs
//
// Symmetrical 4-bit quantization (`alpha = max(|W|) / 8`) with an asymmetric
// option (GPTQ/AWQ-style, zero-point). Thin wrapper over `intn::IntNPacked`
// (bits=4) - see intn.rs for the general engine and the merge motivation.
//
// Public API (`Int4Packed`, `quantize_int4`, `dequantize_int4`, methods
// `bytes_len`/`effective_bits_per_weight`/`group_of`/`get_code`/`get_f32`):
// the symmetric max-abs path produces bit-identical numbers to the original
// 4-bit implementation. `quantize_int4_asymmetric`/`get_zero_point`/
// `is_asymmetric` are additive.

use crate::pack::intn::{self, Calibration, IntNPacked};
use crate::scheme::ScaleGranularity;

/// A packed 4-bit tensor is a thin wrapper over `IntNPacked`.
#[derive(Debug, Clone, PartialEq)]
pub struct Int4Packed {
    inner: IntNPacked,
}

impl Int4Packed {
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

    /// Signed code of the element `idx` (`-8..=7`) - only for the symmetric version.
    /// For asymmetric, returns the unsigned code directly (`0..=15`) without
    /// subtracting zero_point so as not to change the return signature (`i8`) -
    /// use `get_f32` for the correct value in both modes.
    #[inline]
    pub fn get_code(&self, idx: usize) -> i8 {
        match self.inner.calibration {
            Calibration::Symmetric => {
                let raw = self.inner.get_raw_code(idx) as i32;
                (raw - IntNPacked::symmetric_bias(4)) as i8
            }
            Calibration::Asymmetric => self.inner.get_raw_code(idx) as i8,
        }
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

/// Symmetric 4-bit quantization, `alpha = max(|W|) / 8` by group -
/// behavior is identical to the original int4.rs (before merging with intn.rs).
pub fn quantize_int4(data: &[f32], granularity: ScaleGranularity, row_len: usize) -> Int4Packed {
    let inner = intn::quantize_maxabs(data, 4, granularity, row_len);
    Int4Packed { inner }
}

/// Asymmetric 4-bit quantization (GPTQ/AWQ-style): `scale = (max(W) -
/// min(W)) / 15`, `zero_point = round(-min(W)/scale)`. More precisely on
/// shifted weight distributions (see intn.rs § "Asymmetric").
pub fn quantize_int4_asymmetric(
    data: &[f32],
    granularity: ScaleGranularity,
    row_len: usize,
) -> Int4Packed {
    let inner = intn::quantize(data, 4, Calibration::Asymmetric, granularity, row_len);
    Int4Packed { inner }
}

/// Completely expand the Int4-packing back to `Vec<f32>`.
pub fn dequantize_int4(packed: &Int4Packed) -> Vec<f32> {
    intn::dequantize(&packed.inner)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_code_range() {
        let data = vec![100.0, -100.0, 0.0, 50.0];
        let packed = quantize_int4(&data, ScaleGranularity::PerTensor, 0);
        for i in 0..data.len() {
            let c = packed.get_code(i);
            assert!((-8..=7).contains(&c), "code out of range: {c}");
        }
    }

    #[test]
    fn test_nibble_packing_density() {
        let data = vec![1.0f32; 10];
        let packed = quantize_int4(&data, ScaleGranularity::PerTensor, 0);
        assert_eq!(packed.bytes_len(), 5); // ceil(10*4/8) = 5
    }

    #[test]
    fn test_odd_even_nibble_independence() {
        let data = vec![7.0, -8.0, 3.0, -3.0];
        let packed = quantize_int4(&data, ScaleGranularity::PerTensor, 0);
        let restored = dequantize_int4(&packed);
        assert!(restored[0] > 0.0);
        assert!(restored[1] < 0.0);
        assert!(restored[2] > 0.0);
        assert!(restored[3] < 0.0);
    }

    #[test]
    fn test_roundtrip_error_bounded() {
        let data: Vec<f32> = (0..32).map(|i| (i as f32 - 16.0) * 0.2).collect();
        let packed = quantize_int4(&data, ScaleGranularity::PerGroup { group_size: 32 }, 0);
        let restored = dequantize_int4(&packed);
        for (a, b) in data.iter().zip(restored.iter()) {
            assert!(
                (a - b).abs() < 0.3,
                "int4 roundtrip error too large: {a} vs {b}"
            );
        }
    }

    #[test]
    fn test_asymmetric_more_accurate_on_skewed_data() {
        let data: Vec<f32> = (0..64).map(|i| 20.0 + i as f32 * 0.1).collect();
        let sym = quantize_int4(&data, ScaleGranularity::PerTensor, 0);
        let asym = quantize_int4_asymmetric(&data, ScaleGranularity::PerTensor, 0);
        let sym_err: f32 = dequantize_int4(&sym)
            .iter()
            .zip(data.iter())
            .map(|(a, b)| (a - b).abs())
            .sum();
        let asym_err: f32 = dequantize_int4(&asym)
            .iter()
            .zip(data.iter())
            .map(|(a, b)| (a - b).abs())
            .sum();
        assert!(asym_err < sym_err, "asym={asym_err} sym={sym_err}");
        assert!(asym.is_asymmetric());
        assert!(!sym.is_asymmetric());
    }
}
