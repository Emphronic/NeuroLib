// neo-quantization/src/dequantize.rs
//
// Entry point for unpacking `PackedBuffer → f32`. The basic logic is already
// lives in `PackedBuffer::to_f32()` (pack/mod.rs) - this file adds what
// what is deliberately not there: dequant WITHOUT new allocation (reuse
// existing `&mut [f32]` buffer, important in hot path inference, where
// dequant is called every forward and should not touch the allocator).

use crate::pack::PackedBuffer;

// ═══════════════════════════════════════════════════════════════════════════
// § 1 DEQUANT INTO AN EXISTING BUFFER
// ═══════════════════════════════════════════════════════════════════════════

/// Unpack `packed` into the already allocated buffer `out` without allocation.
///
/// #Panic
///
/// Panics if `out.len() != packed.numel()` - calling code must
/// allocate a buffer of the exact size in advance (usually once at startup
/// inference, reused on each forward).
pub fn dequantize_into(packed: &PackedBuffer, out: &mut [f32]) {
    assert_eq!(
        out.len(),
        packed.numel(),
        "dequantize_into: buffer size mismatch (out={}, packed.numel={})",
        out.len(),
        packed.numel()
    );

    match packed {
        // Fast paths with bulk copying - avoid unnecessary calls
        // get_f32() where there is a ready adjacent Vec<f32>.
        PackedBuffer::Float(v) => out.copy_from_slice(v),
        PackedBuffer::Gguf(p) => out.copy_from_slice(&p.to_f32()),
        // Raw GGUF blocks: parsed into a typed buffer once and decoded
        // entirely (rather than element by element through `get_f32`, which
        // would re-parse the bytes for every index). Panics on unsupported
        // dtype/invalid length - see `gguf_raw_to_typed`.
        PackedBuffer::GgufRaw { .. } => out.copy_from_slice(&packed.to_f32()),
        // All intN/ternary/nf4 schemes - through a single point-lookup on
        // PackedBuffer (see pack/mod.rs::PackedBuffer::get_f32). One match
        // here, and not a separate implementation for each scheme - new PackedBuffer
        // options do not require editing this file.
        _ => {
            for (i, slot) in out.iter_mut().enumerate() {
                *slot = packed.get_f32(i);
            }
        }
    }
}

/// The allocating variant is a thin wrapper over `PackedBuffer::to_f32()`,
/// added here for the sake of API symmetry (`quantize()` in quantize.rs has
/// paired `dequantize()` is here, and not scattered as a method on enum).
pub fn dequantize(packed: &PackedBuffer) -> Vec<f32> {
    packed.to_f32()
}

// ═══════════════════════════════════════════════════════════════════════════
// § 2 PARTIAL UNPACKING (range)
// ═══════════════════════════════════════════════════════════════════════════

/// Unpack only the range `[start, end)` of elements.
///
/// Useful for large embedding tables where only one row is needed
/// (`token_id`th row), and not the entire table - avoids unpacking
/// millions of extra weights for every lookup.
pub fn dequantize_range(packed: &PackedBuffer, start: usize, end: usize) -> Vec<f32> {
    assert!(
        start <= end && end <= packed.numel(),
        "dequantize_range: invalid range"
    );

    match packed {
        PackedBuffer::Float(v) => v[start..end].to_vec(),
        // We decode the whole thing once; element-by-element `get_f32` would parse
        // bytes per index (O(n²)).
        PackedBuffer::GgufRaw { .. } => packed.to_f32()[start..end].to_vec(),
        _ => (start..end).map(|i| packed.get_f32(i)).collect(),
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// § 3 TESTS
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;
    use crate::config::QuantizeConfig;
    use crate::quantize::quantize;

    #[test]
    fn test_dequantize_into_matches_dequantize() {
        let data = vec![0.9, -0.9, 0.1, -0.1, 0.5, -0.5, 0.0, 1.0];
        let cfg = QuantizeConfig::ternary_per_tensor();
        let packed = quantize(&data, &cfg, 0);

        let via_alloc = dequantize(&packed);
        let mut via_inplace = vec![0.0f32; packed.numel()];
        dequantize_into(&packed, &mut via_inplace);

        assert_eq!(via_alloc, via_inplace);
    }

    #[test]
    #[should_panic(expected = "buffer size mismatch")]
    fn test_dequantize_into_wrong_size_panics() {
        let data = vec![1.0f32; 8];
        let cfg = QuantizeConfig::int1_per_channel();
        let packed = quantize(&data, &cfg, 8);
        let mut wrong_size = vec![0.0f32; 4];
        dequantize_into(&packed, &mut wrong_size);
    }

    #[test]
    fn test_dequantize_range_matches_full() {
        let data: Vec<f32> = (0..64).map(|i| (i as f32 - 32.0) * 0.05).collect();
        let cfg = QuantizeConfig::int8_per_group_32();
        let packed = quantize(&data, &cfg, 0);

        let full = dequantize(&packed);
        let partial = dequantize_range(&packed, 10, 20);
        assert_eq!(partial, full[10..20]);
    }

    #[test]
    fn test_dequantize_range_full_span() {
        let data = vec![0.5f32; 16];
        let cfg = QuantizeConfig::ternary_per_tensor();
        let packed = quantize(&data, &cfg, 0);
        let all = dequantize_range(&packed, 0, 16);
        assert_eq!(all.len(), 16);
    }

    #[test]
    fn test_dequantize_into_handles_iq4_nl_via_wildcard_path() {
        // PackedBuffer::Iq does not have a special bulk branch in
        // dequantize_into (see § 1 - falls into the general point-lookup
        // fallback via get_f32), so it's important to explicitly cover it with test:
        // new PackedBuffer variants should not silently break this path.
        let data: Vec<f32> = (0..32).map(|i| (i as f32 - 16.0) * 0.05).collect();
        let cfg = QuantizeConfig {
            scheme: crate::scheme::PackedScheme::Gguf(crate::dtype::DType::IQ4_NL),
            rounding: crate::config::RoundingMode::Nearest,
            symmetric: true,
            outlier_clip_percentile: 0.0,
        };
        let packed = quantize(&data, &cfg, 0);
        assert!(matches!(packed, crate::pack::PackedBuffer::Iq(_)));

        let via_alloc = dequantize(&packed);
        let mut via_inplace = vec![0.0f32; packed.numel()];
        dequantize_into(&packed, &mut via_inplace);
        assert_eq!(via_alloc, via_inplace);
    }

    #[test]
    fn test_dequantize_handles_gguf_raw_like_typed_buffer() {
        use crate::pack::PackedBuffer;

        let data: Vec<f32> = (0..96).map(|i| (i as f32 - 48.0) * 0.03).collect();
        let typed = PackedBuffer::Gguf(crate::pack::gguf::quantize_gguf(
            &data,
            crate::dtype::DType::Q8_0,
        ));
        let raw = match &typed {
            PackedBuffer::Gguf(p) => PackedBuffer::GgufRaw {
                bytes: p.blocks.clone(),
                dtype: p.dtype,
                numel: p.numel,
            },
            _ => unreachable!(),
        };

        let expected = dequantize(&typed);
        assert_eq!(dequantize(&raw), expected);

        let mut out = vec![0.0f32; raw.numel()];
        dequantize_into(&raw, &mut out);
        assert_eq!(out, expected);

        assert_eq!(dequantize_range(&raw, 10, 70), expected[10..70]);
    }
}
