// neo-quantization/src/pack/fp16.rs
//
// FP16 (IEEE 754 half-precision) - 1 sign + 5 exponent + 10 mantissa bits.
// Unlike bf16.rs (same exponential range as f32), FP16 has
// the range is noticeably narrower (5-bit exponent vs 8-bit) - conversion requires
// fair processing of overflow→infinity, subnormals and NaN/Inf pass-through,
// not just bit truncation. See bf16.rs for the opposite (trivial)
// by chance.
//
// ## Origin
//
// Based on `neo-synx-cpu/src/cpu_jit/quant_kernel.rs`
// (`f32_to_fp16`/`fp16_to_f32`/`convert_to_fp16`/`dequantize_fp16`) —
// same bit algorithm, number-to-number identical output. neo-synx-cpu
// now depends on this file instead of duplicating logic.

// ═══════════════════════════════════════════════════════════════════════════
// § 1 CONVERSION OF ELEMENTS
// ═══════════════════════════════════════════════════════════════════════════

/// Convert f32 → FP16 (IEEE half-precision).
#[inline(always)]
pub fn f32_to_fp16(x: f32) -> u16 {
    let bits = x.to_bits();
    let sign = ((bits >> 16) & 0x8000) as u16;
    let exp = ((bits >> 23) & 0xFF) as i32;
    let mant = bits & 0x7FFFFF;

    if exp == 255 {
        // Infinity or NaN - we save it through the most significant bit of the mantissa for NaN.
        let m16 = if mant != 0 { 0x200u16 } else { 0 };
        return sign | 0x7C00 | m16;
    }
    let exp16 = exp - 127 + 15;
    if exp16 <= 0 {
        // Subnormal or zero.
        if exp16 < -10 {
            return sign;
        }
        let m = (mant | 0x800000) >> (1 - exp16);
        return sign | (m >> 13) as u16;
    }
    if exp16 >= 31 {
        return sign | 0x7C00; // Overflow → Infinity
    }
    sign | ((exp16 as u16) << 10) | ((mant >> 13) as u16)
}

/// Convert FP16 → f32.
#[inline(always)]
pub fn fp16_to_f32(x: u16) -> f32 {
    let sign = (x as u32 & 0x8000) << 16;
    let exp = (x as u32 >> 10) & 0x1F;
    let mant = (x as u32) & 0x3FF;

    if exp == 0 {
        if mant == 0 {
            return f32::from_bits(sign);
        }
        // Subnormal FP16: value = mant * 2^(-24).
        // Let's normalize: msb position = 31 - leading_zeros(mant).
        // f32_exp = 127 + (msb - 24) = 103 + msb = 103 + (31 - lz) = 134 - lz.
        // New mantissa: remove implicit 1, shift by (23 - msb) = lz - 8.
        let lz = mant.leading_zeros();
        let shift = lz - 8;
        let f32_exp = 134 - lz;
        return f32::from_bits(sign | (f32_exp << 23) | ((mant << shift) & 0x7FFFFF));
    }
    if exp == 31 {
        return f32::from_bits(sign | 0x7F800000 | (mant << 13));
    }
    f32::from_bits(sign | ((exp + 127 - 15) << 23) | (mant << 13))
}

// ═══════════════════════════════════════════════════════════════════════════
// § 2  PACKED CONTAINER
// ═══════════════════════════════════════════════════════════════════════════

/// The packed FP16 tensor is flat `Vec<u16>`, without scale/group.
#[derive(Debug, Clone, PartialEq)]
pub struct Fp16Packed {
    /// FP16 values, one `u16` per weight.
    pub data: Vec<u16>,
    /// Initial number of scales.
    pub numel: usize,
}

impl Fp16Packed {
    /// Size in bytes occupied by packaging (2 bytes/weight).
    pub fn bytes_len(&self) -> usize {
        self.data.len() * 2
    }

    /// Exactly 16.0 - FP16 does not carry overhead on scales.
    pub fn effective_bits_per_weight(&self) -> f32 {
        16.0
    }

    /// Dequantize one element by flat index in f32.
    #[inline]
    pub fn get_f32(&self, idx: usize) -> f32 {
        fp16_to_f32(self.data[idx])
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// § 3 QUANTIZE / DEQUANTIZE (full tensor)
// ═══════════════════════════════════════════════════════════════════════════

/// Convert f32 slice to FP16 package.
pub fn quantize_fp16(data: &[f32]) -> Fp16Packed {
    Fp16Packed {
        data: data.iter().map(|&x| f32_to_fp16(x)).collect(),
        numel: data.len(),
    }
}

/// Completely expand the FP16 package back to `Vec<f32>`.
pub fn dequantize_fp16(packed: &Fp16Packed) -> Vec<f32> {
    packed.data.iter().map(|&q| fp16_to_f32(q)).collect()
}

// ═══════════════════════════════════════════════════════════════════════════
// § 4 TESTS
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_roundtrip_exact_values() {
        let data = vec![1.0f32, -1.0, 0.5, -0.5, 0.0, 2.0, 100.0, 0.25];
        let packed = quantize_fp16(&data);
        let restored = dequantize_fp16(&packed);
        assert_eq!(data, restored);
    }

    #[test]
    fn test_overflow_saturates_to_infinity() {
        // FP16 max normal ≈ 65504; 100000.0 should overflow in Infinity.
        let q = f32_to_fp16(100000.0);
        let restored = fp16_to_f32(q);
        assert!(restored.is_infinite() && restored > 0.0);
    }

    #[test]
    fn test_subnormal_roundtrip() {
        // Low value in the subnormal range FP16 (< 2^-14).
        let x = 0.00003f32;
        let q = f32_to_fp16(x);
        let restored = fp16_to_f32(q);
        assert!(
            (restored - x).abs() < 0.00001,
            "subnormal roundtrip error: {restored} vs {x}"
        );
    }

    #[test]
    fn test_infinity_and_nan_preserved() {
        let data = vec![f32::INFINITY, f32::NEG_INFINITY];
        let packed = quantize_fp16(&data);
        let restored = dequantize_fp16(&packed);
        assert!(restored[0].is_infinite() && restored[0] > 0.0);
        assert!(restored[1].is_infinite() && restored[1] < 0.0);

        let nan_packed = quantize_fp16(&[f32::NAN]);
        assert!(dequantize_fp16(&nan_packed)[0].is_nan());
    }

    #[test]
    fn test_bytes_len_and_bits() {
        let packed = quantize_fp16(&[1.0f32; 10]);
        assert_eq!(packed.bytes_len(), 20);
        assert_eq!(packed.effective_bits_per_weight(), 16.0);
    }

    #[test]
    fn test_get_f32_matches_dequantize() {
        let data: Vec<f32> = (0..32).map(|i| (i as f32 - 16.0) * 0.3).collect();
        let packed = quantize_fp16(&data);
        let restored = dequantize_fp16(&packed);
        for i in 0..data.len() {
            assert_eq!(packed.get_f32(i), restored[i]);
        }
    }
}
