// neo-quantization/src/lib.rs
//
// neo-quantization - conversion of float weights to quantized formats and
// back, with fast scheme switching while the application is running.
//
// ┌────────────────────────────────────────────────────────────────────────┐
// │                      Place in the workspace                            │
// │                                                                        │
// │  burn-std   -- burn's DType/QuantScheme (interop only)                 │
// │      |                                                                 │
// │      +-- neo-quantization  (this crate)                                │
// │              |                                                         │
// │              +-- dtype.rs       DType - GGUF table (block_bytes        │
// │              |                  /gguf_type_id); moved here             │
// │              |                  from neo-core, no burn analogue        │
// │              +-- scheme.rs      PackedScheme (+ Ternary/Int1,          │
// │              |                  absent from the GGUF table)            │
// │              +-- config.rs      QuantizeConfig (rounding/clip)         │
// │              +-- pack/*         float <-> packed per format            │
// │              |     +-- gguf.rs    Q4_0/Q4_1/Q5_0/Q5_1/Q8_0             │
// │              |     +-- kquant.rs  Q2_K..Q8_K (super-block)             │
// │              |     +-- iq.rs      IQ4_NL/IQ4_XS; dispatcher            │
// │              |     |              over iq_codebook (IQ2/IQ3)           │
// │              |     |              and iq1 (IQ1_S/IQ1_M): all 8         │
// │              |     |              IQ formats implemented               │
// │              |     +-- tq.rs      TQ1_0/TQ2_0 (ternary)                │
// │              +-- quantize.rs    single entry point: quantize()         │
// │              +-- dequantize.rs  single entry: dequantize()             │
// │              +-- convert.rs     convert the WHOLE checkpoint           │
// │              +-- switch.rs      WeightRegistry - runtime switch        │
// │                                                                        │
// │  Does not depend on neo-synx-cpu - each backend (CPU/CUDA/wgpu)        │
// │  decides how to use PackedBuffer in its own GEMM kernels.              │
// │  Does not parse model files (GGUF/safetensors) - that is neo-gguf.     │
// └────────────────────────────────────────────────────────────────────────┘
//
// ## Quick start
//
// (a working example is in the doc comment below; it is checked by `cargo test`)
//
// ## Converting an entire checkpoint
//
// ```rust,ignore
// use neo_quantization::{ConversionPlan, convert_checkpoint, compute_stats};
//
// let plan = ConversionPlan::bitnet_style(128); // embed/lm_head/norm stay float
// let converted = convert_checkpoint(&tensors, &plan);
// let stats = compute_stats(&converted);
// println!("{:.2}x compression, {:.2} bits/weight", stats.compression_ratio(), stats.avg_bits_per_weight());
// ```
//
// ## Fast scheme switching at runtime
//
// ```rust,ignore
// use neo_quantization::{WeightRegistry, SwitchMode, QuantizeConfig};
//
// let registry = WeightRegistry::new(SwitchMode::Cached);
// registry.register("layer0.weight", weights, 4096);
// registry.activate("layer0.weight", &QuantizeConfig::ternary_per_tensor());
// // ... later, without recalculation (already cached):
// registry.activate("layer0.weight", &QuantizeConfig::int8_per_group_32());
// ```

#![deny(missing_docs)]
// The crate does bit packing and arithmetic; the temptation to reach for
// `unsafe` for speed is constant. This is a prohibition, not a warning: any
// invalid packing in this code produces silently corrupted weights rather
// than a panic, and `unsafe` would turn that into UB as well.
#![forbid(unsafe_code)]
#![deny(clippy::float_cmp)]
#![warn(clippy::all)]
#![allow(clippy::module_inception)]
// Block indices, bit shifts, and level tables are hard-coded only with the
// "magic" numbers from the GGUF specification - they are documented next to
// every use.
#![allow(clippy::too_many_arguments, clippy::needless_range_loop)]

//! Quantization, dequantization and runtime switching of quantization schemes
//! weights of neural networks - from float to ternary/1-bit (BitNet b1.58-style).
//!
//! # Example
//!
//! ```
//! use neo_quantization::{QuantizeConfig, quantize, dequantize};
//!
//! let weights: Vec<f32> = (0..1024).map(|i| (i as f32 - 512.0) * 0.01).collect();
//!
//! // BitNet b1.58-style: 2 bits per weight + one scale per group of 128.
//! let cfg = QuantizeConfig::ternary_per_group(128);
//! let packed = quantize(&weights, &cfg, 1024);
//!
//! assert!(packed.effective_bits_per_weight() < 3.0);
//!
//! // Dequantization - for verification and for the fallback path of the backend.
//! let restored = dequantize(&packed);
//! assert_eq!(restored.len(), weights.len());
//! ```
//!
//! # Own quantization scheme
//!
//! The main extension point of the crate. Implement [`CustomScheme`] - and yours
//! the scheme works wherever the built-in ones work, including `WeightRegistry`
//! and `convert_checkpoint`. The contract is verified with one line:
//!
//! ```
//! use neo_quantization::custom::{CustomScheme, QuantContext};
//!
//! #[derive(Debug)]
//! struct MyScheme;
//!
//! impl CustomScheme for MyScheme {
//!     fn tag(&self) -> &'static str { "my_scheme" }
//!     fn quantize(&self, w: &[f32], _: &QuantContext) -> Vec<u8> {
//!         let amax = w.iter().fold(0.0f32, |a, v| a.max(v.abs()));
//!         let mut scale = if amax > 0.0 { amax / 127.0 } else { 1.0 };
//!         // The inverse multiplication `127 * scale` must remain finite.
//!         // Without this line the scheme gives inf for scales around f32::MAX - and
//!         // the testkit below catches it here.
//!         while !(scale * 127.0).is_finite() { scale *= 0.5; }
//!         let mut out = scale.to_le_bytes().to_vec();
//!         out.extend(w.iter().map(|v| ((v / scale).round().clamp(-127.0, 127.0) as i8) as u8));
//!         out
//!     }
//!     fn dequantize(&self, b: &[u8], n: usize) -> Vec<f32> {
//!         let scale = f32::from_le_bytes(b[..4].try_into().unwrap());
//!         b[4..].iter().take(n).map(|&x| x as i8 as f32 * scale).collect()
//!     }
//! }
//!
//! // Eleven checks: preserving length on edge dimensions,
//! // determinism, finitude on extreme scales, behavior on
//! // zeros and constants, resistance to outliers, scale
//! // invariance, fairness bits_per_weight.
//! neo_quantization::testkit::assert_scheme(&MyScheme);
//! ```
//!
//! A detailed analysis of each check is in [`testkit::check_scheme`], also known as
//! returns a report instead of panic.
//!
//! # Quantization with calibration
//!
//! [`algo`] contains GPTQ and AWQ - algorithms that minimize error
//! AT THE OUTPUT OF THE LAYER, not on the weights. They need real layer activations:
//!
//! ```no_run
//! use neo_quantization::algo::{gptq_quantize, output_mse, round_to_nearest, GptqConfig};
//! # let (out_features, in_features) = (32usize, 128usize);
//! # let weights = vec![0.1f32; out_features * in_features];
//! # let activations = vec![0.2f32; 1024 * in_features];
//!
//! let result = gptq_quantize(
//!     &weights, out_features, in_features, &activations,
//!     &GptqConfig::int4_g128(),
//! )?;
//!
//! // Comparison with regular rounding of the same bit depth:
//! let baseline = round_to_nearest(&weights, out_features, in_features, 4, 128);
//! let e_gptq = output_mse(&weights, &result.weights, &activations, out_features, in_features);
//! let e_base = output_mse(&weights, &baseline, &activations, out_features, in_features);
//! assert!(e_gptq < e_base);
//! # Ok::<(), neo_quantization::algo::GptqError>(())
//! ```
//!
//! Both algorithms select the VALUES of the weights; they are further encoded by any
//! format from `pack::*`. The separation is intentional - selection and coding
//! solve different problems.
//!
//! Format selection: `pack::gguf` (Q4_0…Q8_0) and `pack::kquant` (Q2_K…Q8_K) —
//! bitwise compatible with llama.cpp; `pack::intn`, `pack::ternary`,
//! `pack::nf4` - own crate schemes with arbitrary granularity
//! scale. GGUF format correspondence table is in [`DType`].

/// GGUF-compatible element type (`DType`) + conversions to burn types.
pub mod dtype;

/// Custom quantization schemes: connecting your own format.
pub mod custom;

/// A ready-made testkit for checking that a quantization scheme complies with the contract.
pub mod testkit;

/// Quantization algorithms with calibration: GPTQ and AWQ.
pub mod algo;

/// Weight packing schemes (`PackedScheme`, `ScaleGranularity`).
pub mod scheme;

/// Quantization process configuration (`QuantizeConfig`, `RoundingMode`).
pub mod config;

/// Specific packaging formats: ternary, int1, int8, int4 + `PackedBuffer`.
pub mod pack;

/// Entry point: float → `PackedBuffer`.
pub mod quantize;

/// Entry point: `PackedBuffer` → float.
pub mod dequantize;

/// Converting an entire checkpoint (a set of named tensors).
pub mod convert;

/// Runtime register of scales with fast switching of the active scheme.
pub mod switch;

// Internal PRNG for stochastic rounding (not part of the public API).
mod rng;

// ─── Re-exports ─────────────────────────────────────────────────────────────

// `DType` moved here from neo-core (there is no equivalent GGUF table in burn -
// see dtype.rs § 4). Consumer crates only need to change
// `use neo_core::DType;` to `use neo_quantization::DType;` - type name and that’s it
// the options are the same.
pub use algo::{awq_quantize, gptq_quantize, AwqConfig, GptqConfig};
pub use config::{QuantizeConfig, RoundingMode};
pub use convert::{
    compute_stats, convert_checkpoint, convert_checkpoint_map, ConversionPlan, ConversionStats,
    ConvertedTensor, RawTensor,
};
pub use custom::{CustomPacked, CustomScheme, QuantContext};
pub use dequantize::{dequantize, dequantize_into, dequantize_range};
pub use dtype::DType;
pub use pack::{
    GgufPacked, Int1Packed, Int4Packed, Int8Packed, Iq1MPacked, Iq1SPacked, Iq4NlPacked,
    Iq4XsPacked, IqPacked, KQuantPacked, PackedBuffer, TernaryPacked, Tq1_0Packed, Tq2_0Packed,
    TqPacked,
};
pub use quantize::quantize;
pub use scheme::{PackedScheme, ScaleGranularity};
pub use switch::{SwitchMode, WeightRegistry};

#[cfg(test)]
mod tests {
    use super::*;

    /// End-to-end smoke test: all the way float → ternary → dequant, with
    /// switching via WeightRegistry - covers the integration of all
    /// crate modules at once.
    #[test]
    fn test_end_to_end_ternary_flow() {
        let weights: Vec<f32> = (0..256).map(|i| ((i as f32) - 128.0) * 0.02).collect();

        // Direct path.
        let cfg = QuantizeConfig::ternary_per_group(64);
        let packed = quantize(&weights, &cfg, 0);
        // 256 weights, group_size=64 → 4 groups: (256*2 + 4*32) / 256 = 2.5 bits/weight
        // (2 data bits/weight + overhead 4 scale-float for 256 weights). Threshold 2.6
        // leaves a margin, but still catches regressions of the main formula.
        assert!(
            packed.effective_bits_per_weight() < 2.6,
            "effective_bits_per_weight={}",
            packed.effective_bits_per_weight()
        );
        let restored = dequantize(&packed);
        assert_eq!(restored.len(), weights.len());

        // Through the registry with fast switching.
        let registry = WeightRegistry::new(SwitchMode::Cached);
        registry.register("w", weights.clone(), 64);
        assert!(registry.activate("w", &cfg));
        let via_registry = registry.get_active("w").unwrap();
        assert_eq!(via_registry.numel(), weights.len());

        // Switching to int8 and back.
        registry.activate("w", &QuantizeConfig::int8_per_group_32());
        assert!(matches!(
            registry.get_active("w").unwrap(),
            PackedBuffer::Int8(_)
        ));
        registry.activate("w", &cfg);
        assert!(matches!(
            registry.get_active("w").unwrap(),
            PackedBuffer::Ternary(_)
        ));
    }
}
