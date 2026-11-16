// neo-quantization/src/pack/fp8.rs
//
// FP8 - two options according to the IEEE-adjacent convention (OCP/NVIDIA):
//   E4M3: 1 sign + 4 exp + 3 mantissa, max ±448.0 - more precisely, already a range.
//   E5M2: 1 sign + 5 exp + 2 mantissa, max ±57344.0 - wider range, coarser.
//
// Unlike bf16/fp16 (self-sufficient float, no scale), FP8
// has too narrow a range to store typical neural network weights
// directly without preliminary normalization - therefore `Fp8Packed` carries
// one tensor-wide `scale: f32` (the same compromise “one scale for everything”,
// that namespace - but chose nf4.rs/ternary.rs for PerTensor granularity).
//
// When one scale is not enough for the entire tensor (wide activations, weights
// with very different spread across rows), use `Fp8GroupPacked` /
// `quantize_fp8_grouped` (§ 5): the same FP8 code, but with a separate scale on
// group/channel - similar to `int8::quantize_int8` and `Nf4Packed`.
// `Fp8Packed` did not change (`scale` remained only `f32`).
//
// ## Origin
//
// Based on `neo-synx-cpu/src/cpu_jit/quant_kernel.rs`
// (`f32_to_fp8_e4m3`/`fp8_e4m3_to_f32`/`f32_to_fp8_e5m2`/`fp8_e5m2_to_f32`/
// `convert_to_fp8_e4m3`/`convert_to_fp8_e5m2`/`dequantize_fp8`) - the same
// bit algorithm and the same formula scale (`max_abs / MAX_FP8`).

use crate::pack::intn;
use crate::scheme::ScaleGranularity;

// ═══════════════════════════════════════════════════════════════════════════
// § 1 OPTION FP8
// ═══════════════════════════════════════════════════════════════════════════

/// Distinguishes E4M3 (more precisely, narrower range) from E5M2 (wider range, coarser).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Fp8Variant {
    /// 1s 4e 3m, saturate at ±448.0.
    E4M3,
    /// 1s 5e 2m, saturate at ±57344.0.
    E5M2,
}

impl Fp8Variant {
    /// The maximum modulo representable value for this option.
    #[inline]
    pub fn max_abs(self) -> f32 {
        match self {
            Fp8Variant::E4M3 => 448.0,
            Fp8Variant::E5M2 => 57344.0,
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// § 2 CONVERSION OF ELEMENTS
// ═══════════════════════════════════════════════════════════════════════════

/// Convert f32 → FP8 E4M3 (saturate at ±448.0). The input should already
/// be in the range of representable values (after dividing by scale) -
/// This function clamps, but does not scale.
#[inline(always)]
pub fn f32_to_fp8_e4m3(x: f32) -> u8 {
    const MAX_FP8: f32 = 448.0;
    let clamped = x.clamp(-MAX_FP8, MAX_FP8);
    if clamped == 0.0 {
        return 0;
    }

    let bits = clamped.to_bits();
    let sign = (bits >> 31) as u8;
    let exp = ((bits >> 23) & 0xFF) as i32;
    let mant = (bits & 0x7FFFFF) >> 20; // 3-bit mantissa

    let exp8 = (exp - 127 + 7).clamp(0, 15) as u8;
    (sign << 7) | (exp8 << 3) | (mant as u8 & 0x7)
}

/// Convert FP8 E4M3 → f32.
#[inline(always)]
pub fn fp8_e4m3_to_f32(x: u8) -> f32 {
    if x & 0x7F == 0 {
        return if x >> 7 == 1 { -0.0 } else { 0.0 };
    }
    let sign = (x >> 7) as u32;
    let exp8 = ((x >> 3) & 0xF) as u32;
    let mant = (x & 0x7) as u32;

    let exp32 = exp8 + 127 - 7;
    let bits32 = (sign << 31) | (exp32 << 23) | (mant << 20);
    f32::from_bits(bits32)
}

/// Convert f32 → FP8 E5M2 (saturate at ±57344.0).
#[inline(always)]
pub fn f32_to_fp8_e5m2(x: f32) -> u8 {
    const MAX_FP8: f32 = 57344.0;
    let clamped = x.clamp(-MAX_FP8, MAX_FP8);
    if clamped == 0.0 {
        return 0;
    }

    let bits = clamped.to_bits();
    let sign = (bits >> 31) as u8;
    let exp = ((bits >> 23) & 0xFF) as i32;
    let mant = (bits & 0x7FFFFF) >> 21; // 2-bit mantissa

    let exp8 = (exp - 127 + 15).clamp(0, 31) as u8;
    (sign << 7) | (exp8 << 2) | (mant as u8 & 0x3)
}

/// Convert FP8 E5M2 → f32.
#[inline(always)]
pub fn fp8_e5m2_to_f32(x: u8) -> f32 {
    if x & 0x7F == 0 {
        return if x >> 7 == 1 { -0.0 } else { 0.0 };
    }
    let sign = (x >> 7) as u32;
    let exp8 = ((x >> 2) & 0x1F) as u32;
    let mant = (x & 0x3) as u32;

    let exp32 = exp8 + 127 - 15;
    let bits32 = (sign << 31) | (exp32 << 23) | (mant << 21);
    f32::from_bits(bits32)
}

#[inline]
fn f32_to_fp8(x: f32, variant: Fp8Variant) -> u8 {
    match variant {
        Fp8Variant::E4M3 => f32_to_fp8_e4m3(x),
        Fp8Variant::E5M2 => f32_to_fp8_e5m2(x),
    }
}

#[inline]
fn fp8_to_f32(x: u8, variant: Fp8Variant) -> f32 {
    match variant {
        Fp8Variant::E4M3 => fp8_e4m3_to_f32(x),
        Fp8Variant::E5M2 => fp8_e5m2_to_f32(x),
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// § 3  PACKED CONTAINER
// ═══════════════════════════════════════════════════════════════════════════

/// Packaged FP8 tensor with a single tensor-wide scale. Option with
/// scale per group/channel - [`Fp8GroupPacked`] (§ 5).
#[derive(Debug, Clone, PartialEq)]
pub struct Fp8Packed {
    /// FP8 codes, one `u8` per weight.
    pub data: Vec<u8>,
    /// Single scale (tensor-wide): `W ≈ fp8_to_f32(code) * scale`.
    pub scale: f32,
    /// E4M3 or E5M2.
    pub variant: Fp8Variant,
    /// Initial number of scales.
    pub numel: usize,
}

impl Fp8Packed {
    /// Size in bytes occupied by packaging (1 byte/weight, excluding
    /// single scale-float - negligible on any real
    /// tensor, the same choice as PerTensor in the rest of the pack::*).
    pub fn bytes_len(&self) -> usize {
        self.data.len()
    }

    /// 8.0 plus the negligible overhead of one scale for the entire tensor.
    pub fn effective_bits_per_weight(&self) -> f32 {
        if self.numel == 0 {
            return 8.0;
        }
        8.0 + (32.0 / self.numel as f32)
    }

    /// Dequantize one element by flat index in f32.
    #[inline]
    pub fn get_f32(&self, idx: usize) -> f32 {
        fp8_to_f32(self.data[idx], self.variant) * self.scale
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// § 4 QUANTIZE / DEQUANTIZE (full tensor, tensor-wide scale)
// ═══════════════════════════════════════════════════════════════════════════

/// Quantize slice f32 in FP8 with tensor-wide scale: `scale = max(|W|) /
/// variant.max_abs()`, each weight is normalized by `scale` before
/// conversion to FP8 code.
pub fn quantize_fp8(data: &[f32], variant: Fp8Variant) -> Fp8Packed {
    let max_abs = data.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
    let scale = if max_abs > 0.0 {
        max_abs / variant.max_abs()
    } else {
        1.0
    };
    let inv_scale = 1.0 / scale;
    Fp8Packed {
        data: data
            .iter()
            .map(|&x| f32_to_fp8(x * inv_scale, variant))
            .collect(),
        scale,
        variant,
        numel: data.len(),
    }
}

/// Completely expand the FP8 package back to `Vec<f32>`.
pub fn dequantize_fp8(packed: &Fp8Packed) -> Vec<f32> {
    (0..packed.numel).map(|i| packed.get_f32(i)).collect()
}

// ═══════════════════════════════════════════════════════════════════════════
// § 5  PER-GROUP / PER-CHANNEL SCALE
// ═══════════════════════════════════════════════════════════════════════════
//
// The same FP8 code, but scale is calculated not over the entire tensor, but over the group
// (`ScaleGranularity::PerGroup`) or by row (`PerChannel`): one outlier
// spoils only its group, and not the dynamic range of the entire tensor.
// The layout repeats `Nf4Packed` (`scales: Vec<f32>` + `group_size`, where
// `usize::MAX` means PerTensor), and the selection rule `group_size` is the same,
// what's in `intn::resolve_group_size`.

/// Packaged FP8 tensor with a separate scale per weight group.
///
/// `W[i] ≈ fp8_to_f32(data[i]) * scales[i / group_size]`. When
/// `group_size == usize::MAX` (PerTensor) there is exactly one element in `scales`.
#[derive(Debug, Clone, PartialEq)]
pub struct Fp8GroupPacked {
    /// FP8 codes, one `u8` per weight.
    pub data: Vec<u8>,
    /// The scale of each group is `max(|W_group|) / variant.max_abs()`.
    pub scales: Vec<f32>,
    /// Group size in elements. `usize::MAX` = PerTensor.
    pub group_size: usize,
    /// E4M3 or E5M2.
    pub variant: Fp8Variant,
    /// Initial number of scales.
    pub numel: usize,
}

impl Fp8GroupPacked {
    /// Size in bytes occupied by codes (without scales).
    pub fn bytes_len(&self) -> usize {
        self.data.len()
    }

    /// Actual bits/weight taking into account overhead on scales.
    pub fn effective_bits_per_weight(&self) -> f32 {
        if self.numel == 0 {
            return 8.0;
        }
        (self.data.len() as f32 * 8.0 + self.scales.len() as f32 * 32.0) / self.numel as f32
    }

    /// Group index (scale bucket) for element `idx`.
    #[inline]
    pub fn group_of(&self, idx: usize) -> usize {
        if self.group_size == usize::MAX {
            0
        } else {
            idx / self.group_size
        }
    }

    /// Dequantize one element by flat index in f32.
    #[inline]
    pub fn get_f32(&self, idx: usize) -> f32 {
        fp8_to_f32(self.data[idx], self.variant) * self.scales[self.group_of(idx)]
    }
}

/// Quantize f32 slice in FP8 with scale per group.
///
/// `granularity` and `row_len` are treated as in the other `pack::quantize_*`
/// (`PerChannel` → group of length `row_len`; `PerTensor` → one group).
/// Group scale - `max(|W|) / variant.max_abs()`; for completely zero
/// group it is equal to `1.0`, as in [`quantize_fp8`].
///
/// #Panic
///
/// `group_size == 0` (including `PerChannel` with `row_len == 0` on non-empty
/// data - for tensors without a row structure, use `PerTensor`).
pub fn quantize_fp8_grouped(
    data: &[f32],
    variant: Fp8Variant,
    granularity: ScaleGranularity,
    row_len: usize,
) -> Fp8GroupPacked {
    let numel = data.len();
    let group_size = intn::resolve_group_size(granularity, row_len);
    assert!(
        group_size > 0 || numel == 0,
        "quantize_fp8_grouped: group_size must be > 0 (PerChannel requires row_len > 0)"
    );

    let n_groups = if numel == 0 {
        0
    } else if group_size == usize::MAX {
        1
    } else {
        numel.div_ceil(group_size)
    };

    let mut scales = vec![1.0f32; n_groups];
    let mut codes = vec![0u8; numel];

    for g in 0..n_groups {
        let (start, end) = if group_size == usize::MAX {
            (0, numel)
        } else {
            (g * group_size, ((g + 1) * group_size).min(numel))
        };
        let slice = &data[start..end];
        let max_abs = slice.iter().map(|x| x.abs()).fold(0.0f32, f32::max);
        let scale = if max_abs > 0.0 {
            max_abs / variant.max_abs()
        } else {
            1.0
        };
        let inv_scale = 1.0 / scale;
        scales[g] = scale;
        for (code, &x) in codes[start..end].iter_mut().zip(slice) {
            *code = f32_to_fp8(x * inv_scale, variant);
        }
    }

    Fp8GroupPacked {
        data: codes,
        scales,
        group_size,
        variant,
        numel,
    }
}

/// Completely expand the per-group FP8 package back to `Vec<f32>`.
pub fn dequantize_fp8_grouped(packed: &Fp8GroupPacked) -> Vec<f32> {
    (0..packed.numel).map(|i| packed.get_f32(i)).collect()
}

// ═══════════════════════════════════════════════════════════════════════════
// § 6 TESTS
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_e4m3_roundtrip_small_values() {
        let data = vec![1.0f32, -1.0, 0.5, -0.5, 0.0];
        let packed = quantize_fp8(&data, Fp8Variant::E4M3);
        let restored = dequantize_fp8(&packed);
        for (a, b) in data.iter().zip(restored.iter()) {
            assert!(
                (a - b).abs() < 0.2,
                "E4M3 roundtrip error too large: {a} vs {b}"
            );
        }
    }

    #[test]
    fn test_e5m2_wider_range_than_e4m3() {
        assert!(Fp8Variant::E5M2.max_abs() > Fp8Variant::E4M3.max_abs());
    }

    #[test]
    fn test_scale_normalizes_to_max_abs() {
        let data = vec![10.0f32, -20.0, 5.0];
        let packed = quantize_fp8(&data, Fp8Variant::E4M3);
        // max_abs=20.0 → scale = 20.0 / 448.0
        assert!((packed.scale - 20.0 / 448.0).abs() < 1e-6);
    }

    #[test]
    fn test_all_zero_tensor_is_safe() {
        let data = vec![0.0f32; 8];
        let packed = quantize_fp8(&data, Fp8Variant::E4M3);
        assert_eq!(packed.scale, 1.0);
        let restored = dequantize_fp8(&packed);
        assert!(restored.iter().all(|&v| v == 0.0));
    }

    #[test]
    fn test_bytes_len_and_bits() {
        let packed = quantize_fp8(&vec![1.0f32; 100], Fp8Variant::E4M3);
        assert_eq!(packed.bytes_len(), 100);
        assert!(packed.effective_bits_per_weight() < 8.5);
        assert!(packed.effective_bits_per_weight() >= 8.0);
    }

    #[test]
    fn test_get_f32_matches_dequantize() {
        let data: Vec<f32> = (0..32).map(|i| (i as f32 - 16.0) * 0.3).collect();
        let packed = quantize_fp8(&data, Fp8Variant::E5M2);
        let restored = dequantize_fp8(&packed);
        for i in 0..data.len() {
            assert_eq!(packed.get_f32(i), restored[i]);
        }
    }

    #[test]
    fn test_grouped_scale_isolates_outlier() {
        // One group with an outlier should not spoil the accuracy of the neighboring one.
        let mut data: Vec<f32> = (0..64).map(|i| (i as f32 - 32.0) * 0.001).collect();
        data.extend((0..64).map(|i| (i as f32 - 32.0) * 100.0));
        let g = ScaleGranularity::PerGroup { group_size: 64 };
        let grouped = quantize_fp8_grouped(&data, Fp8Variant::E4M3, g, 0);
        let global = quantize_fp8(&data, Fp8Variant::E4M3);
        assert_eq!(grouped.scales.len(), 2);

        let rg = dequantize_fp8_grouped(&grouped);
        let rt = dequantize_fp8(&global);
        let err = |r: &[f32]| -> f32 {
            data[..64]
                .iter()
                .zip(&r[..64])
                .map(|(a, b)| (a - b).abs())
                .sum::<f32>()
        };
        assert!(
            err(&rg) < err(&rt),
            "per-group scale must protect the small-magnitude group: grouped={} global={}",
            err(&rg),
            err(&rt)
        );
    }

    #[test]
    fn test_grouped_per_tensor_matches_tensor_wide() {
        let data: Vec<f32> = (0..50).map(|i| (i as f32 - 25.0) * 0.37).collect();
        let grouped = quantize_fp8_grouped(&data, Fp8Variant::E5M2, ScaleGranularity::PerTensor, 0);
        let global = quantize_fp8(&data, Fp8Variant::E5M2);
        assert_eq!(grouped.group_size, usize::MAX);
        assert_eq!(grouped.scales, vec![global.scale]);
        assert_eq!(grouped.data, global.data);
        assert_eq!(dequantize_fp8_grouped(&grouped), dequantize_fp8(&global));
    }

    #[test]
    fn test_grouped_per_channel_uses_row_len() {
        let data: Vec<f32> = (0..24).map(|i| i as f32 + 1.0).collect();
        let packed = quantize_fp8_grouped(&data, Fp8Variant::E4M3, ScaleGranularity::PerChannel, 8);
        assert_eq!(packed.group_size, 8);
        assert_eq!(packed.scales.len(), 3);
    }

    #[test]
    fn test_grouped_tail_group_and_get_f32() {
        // numel is not a multiple of group_size - the last group is shorter.
        let data: Vec<f32> = (0..37).map(|i| (i as f32 - 18.0) * 0.21).collect();
        let g = ScaleGranularity::PerGroup { group_size: 16 };
        let packed = quantize_fp8_grouped(&data, Fp8Variant::E4M3, g, 0);
        assert_eq!(packed.scales.len(), 3);
        let restored = dequantize_fp8_grouped(&packed);
        assert_eq!(restored.len(), 37);
        for i in 0..data.len() {
            assert_eq!(packed.get_f32(i), restored[i]);
            assert!((data[i] - restored[i]).abs() <= data[i].abs() * 0.13 + 1e-3);
        }
    }

    #[test]
    fn test_grouped_zero_group_is_safe_and_bits_overhead() {
        let mut data = vec![0.0f32; 32];
        data.extend((0..32).map(|i| i as f32));
        let g = ScaleGranularity::PerGroup { group_size: 32 };
        let packed = quantize_fp8_grouped(&data, Fp8Variant::E4M3, g, 0);
        assert_eq!(packed.scales[0], 1.0);
        assert!(dequantize_fp8_grouped(&packed)[..32]
            .iter()
            .all(|&v| v == 0.0));
        assert_eq!(packed.bytes_len(), 64);
        // 8 bits + 2 scale * 32 bits / 64 weights = 9.0
        assert!((packed.effective_bits_per_weight() - 9.0).abs() < 1e-6);
    }

    #[test]
    fn test_grouped_empty_input() {
        let packed = quantize_fp8_grouped(&[], Fp8Variant::E4M3, ScaleGranularity::PerTensor, 0);
        assert_eq!(packed.numel, 0);
        assert!(packed.scales.is_empty());
        assert_eq!(packed.effective_bits_per_weight(), 8.0);
        assert!(dequantize_fp8_grouped(&packed).is_empty());
    }

    #[test]
    #[should_panic(expected = "group_size must be > 0")]
    fn test_grouped_per_channel_zero_row_len_panics() {
        let _ = quantize_fp8_grouped(
            &[1.0, 2.0],
            Fp8Variant::E4M3,
            ScaleGranularity::PerChannel,
            0,
        );
    }
}
