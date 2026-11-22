// neo-quantization/src/pack/bf16.rs
//
// BF16 (bfloat16) - 1 sign + 8 exponent + 7 mantissa bits. The same
// dynamic range that f32 (matching 8 bit exponent), but
// truncated mantissa - conversion f32→bf16 is reduced to rounding the upper
// 16 bits with the bottom 16 discarded, no special cases
// overflows/denormals (unlike fp16.rs/fp8.rs). That's why
// this scheme receives its own file, and is not started via intn.rs -
// this is not an integer grid of codes with scale/zero-point, but a truncated float,
// there is no physical analogue among intN.
//
// ## Origin
//
// Based on `neo-synx-cpu/src/cpu_jit/quant_kernel.rs`
// (`f32_to_bf16`/`bf16_to_f32`/`convert_to_bf16`/`dequantize_bf16`) - that one
// same algorithm round-to-nearest-even, number-for-number identical output.
// neo-synx-cpu now depends on this file instead of duplicating logic
// (See neo-synx-cpu/src/cpu_jit/quant_kernel.rs § "Bridge to neo-quantization").
//
// ## Space in PackedBuffer
//
// Unlike intN/ternary/nf4, BF16 does not carry scale/group_size - each
// the value is self-contained (it's still a float, just narrower).
// `Bf16Packed` therefore does not wrap `IntNPacked` - trivial
// `Vec<u16>` plus `numel`.

// ═══════════════════════════════════════════════════════════════════════════
// § 1 CONVERSION OF ELEMENTS
// ═══════════════════════════════════════════════════════════════════════════

/// Convert f32 → BF16 (round-to-nearest-even mantissa truncation).
#[inline(always)]
pub fn f32_to_bf16(x: f32) -> u16 {
    let bits = x.to_bits();
    // Round-to-nearest-even for the lower 16 bits: add 0x7FFF plus LSB
    // bit retained (bit 16) before truncation - standard formula
    // RNE for truncate-by-16-bits.
    let rounding = 0x7FFFu32 + ((bits >> 16) & 1);
    ((bits + rounding) >> 16) as u16
}

/// Convert BF16 -> f32 (exact extension - upper 16 bits of f32
/// coincide with bf16 bitwise, the lower 16 are zero).
#[inline(always)]
pub fn bf16_to_f32(x: u16) -> f32 {
    f32::from_bits((x as u32) << 16)
}

// ═══════════════════════════════════════════════════════════════════════════
// § 2  PACKED CONTAINER
// ═══════════════════════════════════════════════════════════════════════════

/// The packed BF16 tensor is flat `Vec<u16>`, without scale/group (each
/// the value is self-sufficient, it is a float, not an integer code).
#[derive(Debug, Clone, PartialEq)]
pub struct Bf16Packed {
    /// BF16 values, one `u16` per weight.
    pub data: Vec<u16>,
    /// Initial number of weights (same as `data.len()`, stored separately
    /// for the sake of API consistency with other pack::* containers).
    pub numel: usize,
}

impl Bf16Packed {
    /// Size in bytes occupied by packaging (2 bytes/weight).
    pub fn bytes_len(&self) -> usize {
        self.data.len() * 2
    }

    /// Exactly 16.0 - BF16 does not carry overhead on scales.
    pub fn effective_bits_per_weight(&self) -> f32 {
        16.0
    }

    /// Dequantize one element by flat index in f32.
    #[inline]
    pub fn get_f32(&self, idx: usize) -> f32 {
        bf16_to_f32(self.data[idx])
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// § 3 QUANTIZE / DEQUANTIZE (full tensor)
// ═══════════════════════════════════════════════════════════════════════════

/// Convert f32 slice to BF16 package.
pub fn quantize_bf16(data: &[f32]) -> Bf16Packed {
    Bf16Packed {
        data: data.iter().map(|&x| f32_to_bf16(x)).collect(),
        numel: data.len(),
    }
}

/// Completely expand the BF16 package back to `Vec<f32>`.
pub fn dequantize_bf16(packed: &Bf16Packed) -> Vec<f32> {
    packed.data.iter().map(|&q| bf16_to_f32(q)).collect()
}

// ═══════════════════════════════════════════════════════════════════════════
// § 4 TESTS
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_roundtrip_exact_values() {
        // Values that are exactly representable in bf16 (little significant bits of the mantissa).
        let data = vec![1.0f32, -1.0, 0.5, -0.5, 0.0, 2.0, 100.0];
        let packed = quantize_bf16(&data);
        let restored = dequantize_bf16(&packed);
        assert_eq!(data, restored);
    }

    #[test]
    fn test_rounding_reduces_precision() {
        // A value with a large number of significant mantissa bits must lose
        // precision when truncated to 7 bits of mantissa.
        let x = 1.0f32 + 1.0 / 3.0;
        let q = f32_to_bf16(x);
        let restored = bf16_to_f32(q);
        assert!(
            (restored - x).abs() < 0.01,
            "bf16 error too large: {restored} vs {x}"
        );
        assert_ne!(
            restored.to_bits(),
            x.to_bits(),
            "bf16 must actually truncate mantissa"
        );
    }

    #[test]
    fn test_bytes_len_and_bits() {
        let packed = quantize_bf16(&[1.0f32; 10]);
        assert_eq!(packed.bytes_len(), 20);
        assert_eq!(packed.effective_bits_per_weight(), 16.0);
    }

    #[test]
    fn test_infinity_and_nan_preserved() {
        let data = vec![f32::INFINITY, f32::NEG_INFINITY];
        let packed = quantize_bf16(&data);
        let restored = dequantize_bf16(&packed);
        assert!(restored[0].is_infinite() && restored[0] > 0.0);
        assert!(restored[1].is_infinite() && restored[1] < 0.0);

        let nan_packed = quantize_bf16(&[f32::NAN]);
        assert!(dequantize_bf16(&nan_packed)[0].is_nan());
    }

    #[test]
    fn test_get_f32_matches_dequantize() {
        let data: Vec<f32> = (0..32).map(|i| (i as f32 - 16.0) * 0.3).collect();
        let packed = quantize_bf16(&data);
        let restored = dequantize_bf16(&packed);
        for i in 0..data.len() {
            assert_eq!(packed.get_f32(i), restored[i]);
        }
    }
}
