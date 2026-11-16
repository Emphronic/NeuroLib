// neo-quantization/src/convert.rs
//
// Converting the WHOLE checkpoint - a set of named float tensors of the model
// (`{"model.layers.0.self_attn.q_proj.weight": [f32; ...], ...}`) - in
// target scheme. Doesn't parse file formats (safetensors/GGUF/PyTorch) -
// this is a matter for `neo-gguf`/separate bootloader; there's only orchestration here
// quantization itself over a set of tensors, with parallelization through rayon
// (mentioned in Cargo.toml precisely for this purpose - Loop3-style parallelism
// tensors, analogous to what is already used in PackedWeights in
// neo-synx-cpu for ic parallelism).
//
// ## Naming rule (per-tensor policy)
//
//   Different tensors of the model are differently sensitive to aggressive
//   quantization - typical practice (including BitNet b1.58/Qwen3): matrices
//   attention/FFN are quantized aggressively (ternary/int1), and embedding/lm_head/
//   layer-norm weights remain at higher precision (int8/float).
//   `ConversionPlan` formalizes this through `TensorNamePolicy`.

use std::collections::HashMap;

use rayon::prelude::*;

use crate::config::QuantizeConfig;
use crate::pack::PackedBuffer;
use crate::quantize::quantize;
use crate::scheme::PackedScheme;

// ═══════════════════════════════════════════════════════════════════════════
// § 1 INPUT / OUTPUT TYPES
// ═══════════════════════════════════════════════════════════════════════════

/// One model tensor before conversion.
#[derive(Debug, Clone)]
pub struct RawTensor {
    /// The full name of the tensor (for example, `"model.layers.0.mlp.down_proj.weight"`).
    pub name: String,
    /// Data in f32 (conversion from BF16/F16 to f32 - responsibility
    /// calling code/bootloader, here always f32 at the input).
    pub data: Vec<f32>,
    /// `in_features` - needed for `ScaleGranularity::PerChannel`. For
    /// tensors without an expressed row/col structure (bias, layer-norm gamma)
    /// pass `0` - then PerChannel will degenerate into PerTensor).
    pub row_len: usize,
}

/// The result of converting one tensor.
#[derive(Debug, Clone)]
pub struct ConvertedTensor {
    /// Full name of the tensor (same as in `RawTensor::name`).
    pub name: String,
    /// Packed weights after quantization.
    pub buffer: PackedBuffer,
    /// The scheme that was actually applied (after permission from policy).
    pub scheme: PackedScheme,
}

// ═══════════════════════════════════════════════════════════════════════════
// § 2 POLICY: WHICH SCHEME FOR WHICH TENSOR
// ═══════════════════════════════════════════════════════════════════════════

/// The rule for choosing a quantization config based on the tensor name.
///
/// Checked in order; The first rule that matches wins.
/// `default` is applied if no rules match.
pub struct ConversionPlan {
    /// `(name_substring, config)` — if the tensor name contains a substring,
    /// the corresponding config is applied.
    rules: Vec<(String, QuantizeConfig)>,
    /// Default config if no rule worked.
    default: QuantizeConfig,
}

impl ConversionPlan {
    /// A new plan with a given default config (usually the target
    /// aggressive scheme, for example ternary for the bulk of scales).
    pub fn new(default: QuantizeConfig) -> Self {
        ConversionPlan {
            rules: Vec::new(),
            default,
        }
    }

    /// Add rule: if tensor name contains `substring`, use
    /// `config` instead of `default`.
    pub fn with_rule(mut self, substring: impl Into<String>, config: QuantizeConfig) -> Self {
        self.rules.push((substring.into(), config));
        self
    }

    /// Ready-made plan "BitNet b1.58-style": the bulk of the weights are ternary
    /// per-group(128), and embedding/lm_head/norm remain at their original precision
    /// (empirically, these layers are the most sensitive to aggressive
    /// quantization, so most ternary models do not touch them).
    pub fn bitnet_style(group_size: usize) -> Self {
        ConversionPlan::new(QuantizeConfig::ternary_per_group(group_size))
            .with_rule("embed_tokens", QuantizeConfig::float())
            .with_rule("lm_head", QuantizeConfig::float())
            .with_rule("norm", QuantizeConfig::float())
            .with_rule("bias", QuantizeConfig::float())
    }

    /// Allow config for specific tensor name.
    pub fn resolve(&self, tensor_name: &str) -> &QuantizeConfig {
        for (substring, config) in &self.rules {
            if tensor_name.contains(substring.as_str()) {
                return config;
            }
        }
        &self.default
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// § 3 CONVERSION
// ═══════════════════════════════════════════════════════════════════════════

/// Convert the entire set of tensors according to plan, in parallel according to
/// tensors (rayon `par_iter`, each tensor is an independent unit of work,
/// without a shared mutable state between them - safe for
/// data-parallel processing). The order of the results matches the order
/// `tensors` (rayon stores the indices at `collect` in `Vec`), and the result
/// does not depend on the number of threads: each tensor is quantized independently, and
/// `RoundingMode::Stochastic` sows RNG onto the tensor from `config`, not from
/// general condition.
pub fn convert_checkpoint(tensors: &[RawTensor], plan: &ConversionPlan) -> Vec<ConvertedTensor> {
    tensors
        .par_iter()
        .map(|t| {
            let config = plan.resolve(&t.name);
            let buffer = quantize(&t.data, config, t.row_len);
            ConvertedTensor {
                name: t.name.clone(),
                buffer,
                scheme: config.scheme,
            }
        })
        .collect()
}

/// A convenient option that returns `HashMap<name, ConvertedTensor>` - like this
/// looks more natural to consumers searching for a tensor by name (loading
/// into specific layers of the model).
pub fn convert_checkpoint_map(
    tensors: &[RawTensor],
    plan: &ConversionPlan,
) -> HashMap<String, ConvertedTensor> {
    convert_checkpoint(tensors, plan)
        .into_iter()
        .map(|t| (t.name.clone(), t))
        .collect()
}

/// Summary conversion statistics - useful for log/report after
/// checkpoint conversion ("was 54GB, now 7.2GB, average 1.9 bits/weight").
#[derive(Debug, Clone, Copy, Default)]
pub struct ConversionStats {
    /// Number of converted tensors.
    pub total_tensors: usize,
    /// The total number of weights (parameters) for all tensors.
    pub total_params: usize,
    /// The size in bytes that all the weights in f32 (baseline) would occupy.
    pub original_bytes: usize,
    /// Actual size in bytes after packing all tensors.
    pub packed_bytes: usize,
}

impl ConversionStats {
    /// The weighted average number of bits per weight across the entire checkpoint.
    pub fn avg_bits_per_weight(&self) -> f32 {
        if self.total_params == 0 {
            return 0.0;
        }
        (self.packed_bytes as f32 * 8.0) / self.total_params as f32
    }

    /// Compression ratio relative to the original f32.
    pub fn compression_ratio(&self) -> f32 {
        if self.packed_bytes == 0 {
            return 0.0;
        }
        self.original_bytes as f32 / self.packed_bytes as f32
    }
}

/// Calculate statistics based on the result `convert_checkpoint`.
pub fn compute_stats(converted: &[ConvertedTensor]) -> ConversionStats {
    let mut stats = ConversionStats::default();
    for t in converted {
        stats.total_tensors += 1;
        stats.total_params += t.buffer.numel();
        stats.original_bytes += t.buffer.numel() * 4; // baseline: f32
        stats.packed_bytes += t.buffer.data_bytes();
    }
    stats
}

// ═══════════════════════════════════════════════════════════════════════════
// § 4 TESTS
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    fn make_tensor(name: &str, n: usize, row_len: usize) -> RawTensor {
        RawTensor {
            name: name.to_string(),
            data: (0..n).map(|i| (i as f32 - n as f32 / 2.0) * 0.01).collect(),
            row_len,
        }
    }

    #[test]
    fn test_bitnet_plan_protects_embeddings() {
        let plan = ConversionPlan::bitnet_style(128);
        let cfg = plan.resolve("model.embed_tokens.weight");
        assert_eq!(cfg.scheme, PackedScheme::Float);

        let cfg2 = plan.resolve("model.layers.0.mlp.down_proj.weight");
        assert!(cfg2.scheme.is_quantized());
    }

    #[test]
    fn test_convert_checkpoint_parallel_matches_serial() {
        let tensors = vec![
            make_tensor("model.layers.0.mlp.up_proj.weight", 4096, 1024),
            make_tensor("model.layers.0.mlp.down_proj.weight", 4096, 1024),
            make_tensor("model.embed_tokens.weight", 1000, 100),
        ];
        let plan = ConversionPlan::bitnet_style(64);
        let converted = convert_checkpoint(&tensors, &plan);

        assert_eq!(converted.len(), 3);
        let embed = converted
            .iter()
            .find(|t| t.name.contains("embed_tokens"))
            .unwrap();
        assert!(matches!(embed.buffer, PackedBuffer::Float(_)));

        let mlp = converted
            .iter()
            .find(|t| t.name.contains("up_proj"))
            .unwrap();
        assert!(matches!(mlp.buffer, PackedBuffer::Ternary(_)));

        // The parallel result must coincide with the sequential one
        // quantization of the same tensors - both in content and in order.
        for (raw, conv) in tensors.iter().zip(converted.iter()) {
            let cfg = plan.resolve(&raw.name);
            assert_eq!(conv.name, raw.name);
            assert_eq!(conv.scheme, cfg.scheme);
            assert_eq!(conv.buffer, quantize(&raw.data, cfg, raw.row_len));
        }
    }

    #[test]
    fn test_convert_checkpoint_map_lookup() {
        let tensors = vec![make_tensor("a.weight", 100, 10)];
        let plan = ConversionPlan::new(QuantizeConfig::int8_per_group_32());
        let map = convert_checkpoint_map(&tensors, &plan);
        assert!(map.contains_key("a.weight"));
    }

    #[test]
    fn test_stats_compression_ratio() {
        let tensors = vec![make_tensor("w", 10_000, 100)];
        let plan = ConversionPlan::new(QuantizeConfig::ternary_per_tensor());
        let converted = convert_checkpoint(&tensors, &plan);
        let stats = compute_stats(&converted);

        assert_eq!(stats.total_params, 10_000);
        assert_eq!(stats.original_bytes, 40_000);
        // ternary: 2 bits/weight → ~4x fewer bytes than f32 → ratio ~ 16x
        assert!(
            stats.compression_ratio() > 10.0,
            "ratio={}",
            stats.compression_ratio()
        );
        assert!(stats.avg_bits_per_weight() < 3.0);
    }

    #[test]
    fn test_custom_rule_overrides_default() {
        let plan = ConversionPlan::new(QuantizeConfig::ternary_per_tensor())
            .with_rule("special", QuantizeConfig::int8_per_group_32());
        let cfg = plan.resolve("layer.special.weight");
        assert_eq!(
            cfg.scheme,
            PackedScheme::Int8(
                crate::scheme::ScaleGranularity::PerGroup { group_size: 32 },
                crate::pack::Calibration::Symmetric,
            )
        );
    }
}
