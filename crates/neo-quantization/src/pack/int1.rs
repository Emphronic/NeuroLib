// neo-quantization/src/pack/int1.rs
//
// Fair 1-bit packing: weights ∈ {-1, +1}. A thin wrapper over
// `intn::IntNPacked` (bits=1, `Calibration::Symmetric`, `alpha = mean(|W|)`) -
// see intn.rs § "Why mean, not max" for why intN schemes share one engine and
// why mean(|W|) is the correct sizing formula in this degenerate (without
// “zero” code) case N=1.
//
// Public API (`Int1Packed`, `quantize_int1`, `dequantize_int1`, methods
// `bits_bytes`/`effective_bits_per_weight`/`get_bit`/`group_of`/`get_f32`,
// field `scales`) is stable: `pack::PackedBuffer`, `dequantize.rs`,
// `switch.rs` use it unchanged.

use crate::pack::intn::{self, Calibration, IntNPacked};
use crate::scheme::ScaleGranularity;

// ═══════════════════════════════════════════════════════════════════════════
// § 1  PACKED CONTAINER
// ═══════════════════════════════════════════════════════════════════════════

/// A packed strictly 1-bit tensor is a thin wrapper over `IntNPacked`.
#[derive(Debug, Clone, PartialEq)]
pub struct Int1Packed {
    inner: IntNPacked,
}

impl Int1Packed {
    /// Access to the general intN engine (bits, codes, scales, zero-points) - needed
    /// fused GEMM cores (`neo-synk::quant`) to read packing without dequantization.
    pub fn as_intn(&self) -> &IntNPacked {
        &self.inner
    }

    /// Size in bytes occupied by bit packing (without scales).
    pub fn bits_bytes(&self) -> usize {
        self.inner.codes_bytes()
    }

    /// Actual bits/weight taking into account overhead on scales.
    pub fn effective_bits_per_weight(&self) -> f32 {
        self.inner.effective_bits_per_weight()
    }

    /// `true` if the element `idx` is encoded as `+1` (aka `-1`).
    #[inline]
    pub fn get_bit(&self, idx: usize) -> bool {
        self.inner.get_raw_code(idx) == 1
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

    /// Alpha scales by group.
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

// ═══════════════════════════════════════════════════════════════════════════
// § 2  QUANTIZE / DEQUANTIZE
// ═══════════════════════════════════════════════════════════════════════════

/// Quantize the f32 slice into strictly 1-bit packaging.
///
/// `q = sign(W)` (`W >= 0.0` → `+1`, otherwise `-1`), `alpha = mean(|W|)` by
/// group - optimal alpha to minimize `||W - alpha*sign(W)||²`
/// (standard result of BinaryConnect/XNOR-Net).
pub fn quantize_int1(data: &[f32], granularity: ScaleGranularity, row_len: usize) -> Int1Packed {
    let inner = intn::quantize(data, 1, Calibration::Symmetric, granularity, row_len);
    Int1Packed { inner }
}

/// Completely expand the Int1-packing back to `Vec<f32>`.
pub fn dequantize_int1(packed: &Int1Packed) -> Vec<f32> {
    intn::dequantize(&packed.inner)
}

// ═══════════════════════════════════════════════════════════════════════════
// § 3 TESTS
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_roundtrip_sign_preserved() {
        let data = vec![0.9, -0.9, 0.05, -0.05, 3.0, -3.0];
        let packed = quantize_int1(&data, ScaleGranularity::PerTensor, 0);
        let restored = dequantize_int1(&packed);
        for (orig, got) in data.iter().zip(restored.iter()) {
            assert_eq!(
                orig.is_sign_negative(),
                got.is_sign_negative(),
                "sign mismatch for {orig} -> {got}"
            );
        }
    }

    #[test]
    fn test_zero_goes_positive() {
        let data = vec![0.0f32, -0.0f32];
        let packed = quantize_int1(&data, ScaleGranularity::PerTensor, 0);
        assert!(packed.get_bit(0));
        assert!(packed.get_bit(1));
    }

    #[test]
    fn test_bit_density() {
        // 16 weights → exactly 2 bytes (8 bits/byte, 1 bit/weight)
        let data = vec![1.0f32; 16];
        let packed = quantize_int1(&data, ScaleGranularity::PerTensor, 0);
        assert_eq!(packed.bits_bytes(), 2);
    }

    #[test]
    fn test_effective_bits_near_1() {
        let data = vec![0.7f32; 2000];
        let packed = quantize_int1(&data, ScaleGranularity::PerTensor, 0);
        assert!(packed.effective_bits_per_weight() < 1.1);
        assert!(packed.effective_bits_per_weight() >= 1.0);
    }

    #[test]
    fn test_alpha_is_mean_abs() {
        let data = vec![2.0, -2.0, 2.0, -2.0];
        let packed = quantize_int1(&data, ScaleGranularity::PerTensor, 0);
        assert!((packed.scales()[0] - 2.0).abs() < 1e-6);
    }

    #[test]
    fn test_per_group_matches_ternary_group_count() {
        let data: Vec<f32> = (0..100).map(|i| (i as f32 - 50.0) * 0.02).collect();
        let packed = quantize_int1(&data, ScaleGranularity::PerGroup { group_size: 25 }, 0);
        assert_eq!(packed.scales().len(), 4);
    }
}
