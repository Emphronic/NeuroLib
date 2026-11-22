//! Conversions f32 ↔ f16 / bf16 (round to nearest even) without dependencies.

/// f32 → IEEE-754 binary16 bits.
pub fn f32_to_f16_bits(x: f32) -> u16 {
    let b = x.to_bits();
    let sign = ((b >> 16) & 0x8000) as u16;
    let exp = ((b >> 23) & 0xff) as i32;
    let man = b & 0x007f_ffff;
    if exp == 0xff {
        // inf / NaN (NaN retains the sign and the most significant bits of the mantissa)
        return if man != 0 {
            sign | 0x7c00 | 0x0200 | (man >> 13) as u16
        } else {
            sign | 0x7c00
        };
    }
    let e = exp - 127 + 15;
    if e >= 0x1f {
        return sign | 0x7c00; // overflow → inf
    }
    if e <= 0 {
        if e < -10 {
            return sign; // too little even for a subnormal
        }
        let m = man | 0x0080_0000;
        let shift = (14 - e) as u32;
        let half = m >> shift;
        let rem = m & ((1u32 << shift) - 1);
        let halfway = 1u32 << (shift - 1);
        let mut r = half;
        if rem > halfway || (rem == halfway && (half & 1) == 1) {
            r += 1;
        }
        return sign | r as u16;
    }
    let mut r = ((e as u32) << 10) | (man >> 13);
    let rem = man & 0x1fff;
    if rem > 0x1000 || (rem == 0x1000 && (r & 1) == 1) {
        r += 1; // transfer to exponent is correct (including up to inf)
    }
    sign | r as u16
}

/// Bits binary16 → f32.
pub fn f16_bits_to_f32(h: u16) -> f32 {
    let sign = ((h & 0x8000) as u32) << 16;
    let exp = ((h >> 10) & 0x1f) as u32;
    let man = (h & 0x03ff) as u32;
    let bits = if exp == 0 {
        if man == 0 {
            sign
        } else {
            let mut e = 113u32;
            let mut m = man;
            while m & 0x400 == 0 {
                m <<= 1;
                e -= 1;
            }
            sign | (e << 23) | ((m & 0x3ff) << 13)
        }
    } else if exp == 0x1f {
        sign | 0x7f80_0000 | (man << 13)
    } else {
        sign | ((exp + 112) << 23) | (man << 13)
    };
    f32::from_bits(bits)
}

/// f32 → bfloat16 bits.
pub fn f32_to_bf16_bits(x: f32) -> u16 {
    let b = x.to_bits();
    if x.is_nan() {
        return ((b >> 16) | 0x0040) as u16;
    }
    let lsb = (b >> 16) & 1;
    (b.wrapping_add(0x7fff + lsb) >> 16) as u16
}

/// Bits bfloat16 → f32.
pub fn bf16_bits_to_f32(h: u16) -> f32 {
    f32::from_bits((h as u32) << 16)
}

#[cfg(test)]
#[allow(clippy::excessive_precision)] // exact f16/bf16 boundary values are the point of these tests
mod tests {
    use super::*;

    #[test]
    fn f16_exact_values() {
        for &v in &[
            0.0f32,
            1.0,
            -1.0,
            0.5,
            2.0,
            65504.0,
            -65504.0,
            6.103515625e-5,
        ] {
            assert_eq!(f16_bits_to_f32(f32_to_f16_bits(v)), v, "{v}");
        }
        assert_eq!(f32_to_f16_bits(1.0), 0x3c00);
        assert_eq!(f32_to_f16_bits(-2.0), 0xc000);
        assert_eq!(f32_to_f16_bits(65520.0), 0x7c00); // rounding to inf
        assert_eq!(f32_to_f16_bits(f32::INFINITY), 0x7c00);
        assert!(f16_bits_to_f32(f32_to_f16_bits(f32::NAN)).is_nan());
    }

    #[test]
    fn f16_subnormals() {
        let smallest = 5.9604644775390625e-8f32; // 2^-24
        assert_eq!(f32_to_f16_bits(smallest), 1);
        assert_eq!(f16_bits_to_f32(1), smallest);
        assert_eq!(f32_to_f16_bits(1e-10), 0);
    }

    #[test]
    fn f16_roundtrip_all_finite_bits() {
        for h in 0u16..=0xffff {
            let f = f16_bits_to_f32(h);
            if f.is_nan() {
                continue;
            }
            assert_eq!(f32_to_f16_bits(f), h, "bits {h:#06x}");
        }
    }

    #[test]
    fn bf16_basic() {
        assert_eq!(f32_to_bf16_bits(1.0), 0x3f80);
        assert_eq!(bf16_bits_to_f32(0x3f80), 1.0);
        assert!(bf16_bits_to_f32(f32_to_bf16_bits(f32::NAN)).is_nan());
        // 1.00390625 = 1 + 2^-8 lies exactly in the middle → rounding to even (1.0)
        assert_eq!(f32_to_bf16_bits(1.003_906_25), 0x3f80);
    }
}
