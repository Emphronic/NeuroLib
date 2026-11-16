//! Paths without panic: `try_init` / `try_forward*` return `NnError`, and
//! the `init`/`forward*` wrappers panic with the same message.

use burn::prelude::*;
use burn::tensor::{Distribution, Int, TensorData};

use crate::NnError;
use crate::attention::{
    GqaAttentionConfig, LinearAttentionConfig, QkNormConfig, SlidingWindowConfig,
    SparseAttentionConfig, SparsePattern,
};
use crate::embedding::{AdaptiveEmbeddingConfig, AdaptiveTierConfig};
use crate::ffn::{MoeConfig, OverflowPolicy};
use crate::norm::{
    AdaptiveRmsNormConfig, ConditionalRmsNormConfig, LayerScaleConfig, ScaleNormConfig,
};
use crate::transformer::{KvCache, LlmBlockConfig, MoeBlockConfig};

fn dev() -> Device {
    Device::flex()
}

fn randn<const D: usize>(shape: [usize; D]) -> Tensor<D> {
    Tensor::<D>::random(shape, Distribution::Normal(0.0, 1.0), &dev())
}

fn is_config_err<T>(r: Result<T, NnError>, layer: &str) -> bool {
    matches!(r, Err(NnError::InvalidConfig { layer: l, .. }) if l == layer)
}

#[test]
fn try_init_reports_invalid_configs() {
    let d = dev();
    assert!(is_config_err(
        GqaAttentionConfig::new(16, 3, 1).try_init(&d),
        "GqaAttentionConfig"
    ));
    assert!(is_config_err(
        LinearAttentionConfig::new(16, 3).try_init(&d),
        "LinearAttentionConfig"
    ));
    assert!(is_config_err(
        QkNormConfig::new(0).try_init(&d),
        "QkNormConfig"
    ));
    assert!(is_config_err(
        SlidingWindowConfig::new(0, GqaAttentionConfig::new(16, 2, 1)).try_init(&d),
        "SlidingWindowConfig"
    ));
    assert!(is_config_err(
        SparseAttentionConfig::new(
            SparsePattern::BlockLocal {
                block_size: 0,
                blocks_per_query: 1
            },
            GqaAttentionConfig::new(16, 2, 1)
        )
        .try_init(&d),
        "SparseAttentionConfig"
    ));
    assert!(is_config_err(
        ConditionalRmsNormConfig::new(0).try_init(&d),
        "ConditionalRmsNormConfig"
    ));
    assert!(is_config_err(
        AdaptiveRmsNormConfig::new(0, 4).try_init(&d),
        "AdaptiveRmsNormConfig"
    ));
    assert!(is_config_err(
        LayerScaleConfig::new(0).try_init(&d),
        "LayerScaleConfig"
    ));
    assert!(is_config_err(
        ScaleNormConfig::new(0).try_init(&d),
        "ScaleNormConfig"
    ));
    assert!(is_config_err(
        MoeConfig::new(8, 16, 4).with_top_k(5).try_init(&d),
        "Moe"
    ));
    assert!(is_config_err(
        LlmBlockConfig::new(64, 4, 3, 128).try_init(&d),
        "LlmBlockConfig"
    ));
}

#[test]
fn try_init_accepts_valid_configs() {
    let d = dev();
    assert!(GqaAttentionConfig::new(16, 2, 1).try_init(&d).is_ok());
    assert!(LayerScaleConfig::new(8).try_init(&d).is_ok());
    assert!(MoeConfig::new(8, 16, 4).try_init(&d).is_ok());
    assert!(LlmBlockConfig::new(64, 4, 2, 128).try_init(&d).is_ok());
}

#[test]
fn moe_block_without_moe_field_is_an_error_not_a_panic() {
    let cfg = LlmBlockConfig::new(16, 2, 1, 24).with_ffn_kind(crate::transformer::FfnKind::Moe);
    assert!(is_config_err(cfg.try_init(&dev()), "LlmBlockConfig"));
}

#[test]
fn moe_block_try_paths_work() {
    let base = LlmBlockConfig::new(16, 2, 1, 24).with_max_seq_len(16);
    let block = MoeBlockConfig::new(base, MoeConfig::new(16, 24, 4))
        .try_init(&dev())
        .expect("valid config");
    assert_eq!(block.try_moe().map(|m| m.num_experts()).ok(), Some(4));
    let (y, aux) = block
        .try_forward_with_aux(randn([2, 4, 16]), None)
        .expect("forward");
    assert_eq!(y.dims(), [2, 4, 16]);
    assert_eq!(aux.expert_indices.dims(), [8, 2]);
}

#[test]
fn moe_empty_batch_returns_empty_stats() {
    let moe = MoeConfig::new(8, 16, 4).init(&dev());
    let (y, aux) = moe
        .try_forward_tokens(Tensor::<2>::zeros([0, 8], &dev()))
        .expect("empty batch");
    assert_eq!(y.dims(), [0, 8]);
    assert_eq!(aux.dropped_assignments, 0);
    assert_eq!(aux.expert_load.dims(), [4]);
}

#[test]
fn moe_shared_expert_and_capacity_on_canonical_moe() {
    let moe = MoeConfig::new(8, 16, 4)
        .with_top_k(2)
        .with_shared_expert(true)
        .with_capacity_factor(0.5)
        .with_overflow_policy(OverflowPolicy::Drop)
        .init(&dev());
    assert!(moe.shared.is_some());
    let (_, aux) = moe.try_forward_tokens(randn([16, 8])).expect("forward");
    assert!(aux.dropped_assignments > 0);
}

#[test]
fn adaptive_routed_reports_out_of_range_id_as_error() {
    let cfg = AdaptiveEmbeddingConfig::new(10, 4, vec![AdaptiveTierConfig::new(0, 10, 4)]);
    let emb = cfg.init(&dev());
    let ids = Tensor::<2, Int>::from_data(TensorData::new(vec![20i64], [1, 1]), &dev());
    assert!(matches!(
        emb.try_forward_routed(ids),
        Err(NnError::InvalidInput { .. })
    ));
}

#[test]
fn kv_cache_append_and_get_returns_accumulated_tensors() {
    let mut cache = KvCache::new(8);
    let (k1, v1) = cache.append_and_get(randn([1, 1, 2, 4]), randn([1, 1, 2, 4]));
    assert_eq!((k1.dims(), v1.dims()), ([1, 1, 2, 4], [1, 1, 2, 4]));
    let (k2, v2) = cache.append_and_get(randn([1, 1, 3, 4]), randn([1, 1, 3, 4]));
    assert_eq!((k2.dims(), v2.dims()), ([1, 1, 5, 4], [1, 1, 5, 4]));
    assert_eq!(cache.len(), 5);
    assert!(cache.get().is_some());
}
