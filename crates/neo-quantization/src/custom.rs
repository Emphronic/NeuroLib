// neo-quantization/src/custom.rs
//
// Extension point: how to connect YOUR quantization scheme without
// touching a single line inside `pack::*`.
//
// ## Why a separate mechanism
//
// `PackedBuffer` is a private enum: adding a format means adding a variant,
// updating seven match expressions, and rebuilding the entire crate. For
// built-in GGUF formats this is correct (their set is fixed by the
// specification), but a research scheme cannot be connected this way.
//
// Here the same pipeline is exposed via dynamic dispatch:
// the user implements one trait [`CustomScheme`], and the result
// moves through the crate together with the built-in formats -
// `WeightRegistry`, `convert_checkpoint`, `dequantize`, and bit statistics
// for weights work without changes.
//
// ## Price
//
// The call to `quantize`/`dequantize` goes through a vtable. For quantization this
// is unimportant: it happens once during the lifetime of the model, and inside
// one call processes the entire tensor. For guaranteed
// monomorphized code on the hot path, use `pack::*` directly.

use core::fmt;
use std::sync::Arc;

use crate::config::QuantizeConfig;

// ═══════════════════════════════════════════════════════════════════════════
// § 1 CONTEXT
// ═══════════════════════════════════════════════════════════════════════════

/// What the scheme knows about the tensor besides the weights themselves.
///
/// Sent to [`CustomScheme::quantize`]. The fields are intentionally descriptive and
/// not “tuning”: the configuration of the scheme itself lives in its own
/// structure, and here are properties of the data that the scheme cannot infer from
/// the slice `&[f32]`.
#[derive(Debug, Clone, Default)]
pub struct QuantContext<'a> {
    /// Length of the weight matrix row in elements, `0` - if unknown.
    ///
    /// Needed for schemes with per-channel scale: `&[f32]` comes in flat, and
    /// without the row length it is impossible to tell where one output channel ends.
    pub row_len: usize,
    /// The name of the tensor if the call came from `convert_checkpoint`.
    ///
    /// The scheme can decide by name (do not quantize `lm_head`, take another
    /// granularity for `attn.*`). Empty when calling `quantize` directly.
    pub tensor_name: &'a str,
    /// The general quantization configuration, if specified by the calling code.
    ///
    /// The scheme is free to ignore it - but `rounding`/`clip` usually come from there
    /// and should be observed, otherwise the behavior will diverge from the built-in schemes.
    pub config: Option<&'a QuantizeConfig>,
    /// Calibration activations `[n_samples, row_len]`, if any.
    ///
    /// This is what distinguishes data-aware algorithms (GPTQ, AWQ - see below).
    /// [`crate::algo`]) from the usual round-to-nearest. `None` - calibration
    /// is not provided, the scheme must work without it.
    pub calibration: Option<&'a [f32]>,
}

impl<'a> QuantContext<'a> {
    /// Context with row length only is the most common case.
    pub fn with_row_len(row_len: usize) -> Self {
        Self {
            row_len,
            ..Default::default()
        }
    }

    /// Add calibration activations `[n_samples, row_len]`.
    pub fn with_calibration(mut self, activations: &'a [f32]) -> Self {
        self.calibration = Some(activations);
        self
    }

    /// Add tensor name.
    pub fn with_name(mut self, name: &'a str) -> Self {
        self.tensor_name = name;
        self
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// § 2 SCHEME TRAIT
// ═══════════════════════════════════════════════════════════════════════════

/// Custom quantization scheme.
///
/// Implement it - and the scheme works wherever the built-in ones work.
///
/// # Contract
///
/// An implementation MUST follow four rules. They are all checked
/// automatically: see [`crate::testkit::check_scheme`].
///
/// 1. **Determinism.** Quantizing the same input twice must
///    produce the same bytes. Otherwise weight caches break and
///    training reproducibility suffers.
/// 2. **Length preservation.** `dequantize(quantize(w), w.len()).len() == w.len()`
///    for ANY length, including 0 and lengths that are not multiples of the block size.
/// 3. **Finiteness.** For any finite input the output is finite. `NaN` or
///    `inf` is the most common defect: the scale goes to zero or
///    infinity, and the error shows up through hundreds of layers.
/// 4. **Meaningfulness.** The error is less than the variance of the input, that is, the scheme is
///    better than “just output the average.” Formally a weak requirement, but it
///    catches confused indices and broken scales.
///
/// # Example: 4-bit scheme with one scale per tensor
///
/// ```
/// use neo_quantization::custom::{CustomScheme, QuantContext};
///
/// #[derive(Debug)]
/// struct MyInt4;
///
/// impl CustomScheme for MyInt4 {
///     fn tag(&self) -> &'static str { "my_int4" }
///
///     fn quantize(&self, w: &[f32], _ctx: &QuantContext) -> Vec<u8> {
///         let amax = w.iter().fold(0.0f32, |a, v| a.max(v.abs()));
///         let mut scale = if amax > 0.0 { amax / 7.0 } else { 1.0 };
///         // `8 * scale` is the extreme value on the reverse pass; it must
///         // remain finite even for scales around f32::MAX.
///         while !(scale * 8.0).is_finite() { scale *= 0.5; }
///         let inv = 1.0 / scale;
///
///         let mut out = scale.to_le_bytes().to_vec();
///         for pair in w.chunks(2) {
///             let enc = |v: f32| ((v * inv).round().clamp(-7.0, 7.0) as i8 + 8) as u8;
///             let lo = enc(pair[0]);
///             let hi = if pair.len() > 1 { enc(pair[1]) } else { 8 };
///             out.push(lo | (hi << 4));
///         }
///         out
///     }
///
///     fn dequantize(&self, bytes: &[u8], numel: usize) -> Vec<f32> {
///         let scale = f32::from_le_bytes(bytes[..4].try_into().unwrap());
///         let mut out = Vec::with_capacity(numel);
///         for &b in &bytes[4..] {
///             out.push(((b & 0x0F) as i32 - 8) as f32 * scale);
///             out.push(((b >> 4) as i32 - 8) as f32 * scale);
///         }
///         out.truncate(numel);
///         out
///     }
/// }
///
/// // Check the contract in one line:
/// neo_quantization::testkit::assert_scheme(&MyInt4);
/// ```
pub trait CustomScheme: fmt::Debug + Send + Sync {
    /// Short machine name of the scheme: logs, cache keys, file names.
    ///
    /// Must be unique among schemes used together:
    /// `WeightRegistry` distinguishes cached options based on it.
    fn tag(&self) -> &'static str;

    /// Pack weights into bytes.
    ///
    /// All that is needed for the inverse transformation is scales, offsets, and
    /// tables - the scheme puts them in the same bytes. The crate does not look into them.
    fn quantize(&self, weights: &[f32], ctx: &QuantContext) -> Vec<u8>;

    /// Expand the bytes back into `numel` weights.
    ///
    /// `numel` is passed separately rather than stored in the bytes, because
    /// the calling code already knows the length, and duplicating it in every tensor is
    /// expensive. If your scheme needs it inside, store it yourself.
    fn dequantize(&self, bytes: &[u8], numel: usize) -> Vec<f32>;

    /// Actual bits by weight.
    ///
    /// By default it is calculated from the buffer size - an honest estimate
    /// that includes all overhead. Override only if part of the bytes
    /// is shared between tensors.
    fn bits_per_weight(&self, numel: usize, bytes_len: usize) -> f32 {
        if numel == 0 {
            0.0
        } else {
            bytes_len as f32 * 8.0 / numel as f32
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// § 3 PACKED RESULT
// ═══════════════════════════════════════════════════════════════════════════

/// The result of the custom scheme.
///
/// Stores bytes together with a scheme that can read them, so
/// `to_f32()` works without an external format registry.
#[derive(Debug, Clone)]
pub struct CustomPacked {
    /// Packed bytes are entirely at the discretion of the scheme.
    pub bytes: Vec<u8>,
    /// Initial number of scales.
    pub numel: usize,
    /// The scheme that created these bytes; it also reads them.
    pub scheme: Arc<dyn CustomScheme>,
}

impl CustomPacked {
    /// Pack the weights according to the given scheme.
    pub fn new(scheme: Arc<dyn CustomScheme>, weights: &[f32], ctx: &QuantContext) -> Self {
        let bytes = scheme.quantize(weights, ctx);
        Self {
            bytes,
            numel: weights.len(),
            scheme,
        }
    }

    /// Initial number of scales.
    pub fn numel(&self) -> usize {
        self.numel
    }

    /// Package size in bytes.
    pub fn bytes_len(&self) -> usize {
        self.bytes.len()
    }

    /// Actual bits by weight.
    pub fn effective_bits_per_weight(&self) -> f32 {
        self.scheme.bits_per_weight(self.numel, self.bytes.len())
    }

    /// Expand back to `Vec<f32>`.
    pub fn to_f32(&self) -> Vec<f32> {
        self.scheme.dequantize(&self.bytes, self.numel)
    }
}

/// Comparison by scheme tag and bytes.
///
/// Trait objects cannot be compared, and `PackedBuffer` requires `PartialEq`
/// (used in tests and in `WeightRegistry`). Tag plus content -
/// a complete description of the packaging in terms of the crate.
impl PartialEq for CustomPacked {
    fn eq(&self, other: &Self) -> bool {
        self.numel == other.numel
            && self.scheme.tag() == other.scheme.tag()
            && self.bytes == other.bytes
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[derive(Debug)]
    struct Int8PerTensor;

    impl CustomScheme for Int8PerTensor {
        fn tag(&self) -> &'static str {
            "test_int8"
        }
        fn quantize(&self, w: &[f32], _ctx: &QuantContext) -> Vec<u8> {
            let amax = w.iter().fold(0.0f32, |a, v| a.max(v.abs()));
            let scale = if amax > 0.0 { amax / 127.0 } else { 1.0 };
            let inv = 1.0 / scale;
            let mut out = scale.to_le_bytes().to_vec();
            out.extend(
                w.iter()
                    .map(|v| ((v * inv).round().clamp(-127.0, 127.0) as i8) as u8),
            );
            out
        }
        fn dequantize(&self, bytes: &[u8], numel: usize) -> Vec<f32> {
            let scale = f32::from_le_bytes(bytes[..4].try_into().unwrap());
            bytes[4..]
                .iter()
                .take(numel)
                .map(|&b| b as i8 as f32 * scale)
                .collect()
        }
    }

    #[test]
    fn test_custom_packed_roundtrip() {
        let w: Vec<f32> = (0..100).map(|i| (i as f32 - 50.0) * 0.1).collect();
        let packed = CustomPacked::new(Arc::new(Int8PerTensor), &w, &QuantContext::default());

        assert_eq!(packed.numel(), 100);
        assert_eq!(packed.to_f32().len(), 100);
        // 4 bytes of scale per 100 weights + one byte per weight.
        assert!((packed.effective_bits_per_weight() - 8.32).abs() < 0.01);
        for (a, b) in w.iter().zip(packed.to_f32()) {
            assert!((a - b).abs() < 0.05, "{a} vs {b}");
        }
    }

    #[test]
    fn test_custom_packed_eq_uses_tag_and_bytes() {
        let w = vec![1.0f32, 2.0, 3.0];
        let ctx = QuantContext::default();
        let a = CustomPacked::new(Arc::new(Int8PerTensor), &w, &ctx);
        let b = CustomPacked::new(Arc::new(Int8PerTensor), &w, &ctx);
        assert_eq!(a, b);

        let c = CustomPacked::new(Arc::new(Int8PerTensor), &[1.0, 2.0, 4.0], &ctx);
        assert_ne!(a, c);
    }
}
