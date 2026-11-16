//! Tests of the transformer module (NdArray; autodiff - for the router gradient).

use burn::module::Param;
use burn::prelude::*;
use burn::tensor::{Distribution, Int};

use crate::ffn::MoeConfig;
use crate::transformer::*;

fn dev() -> Device {
    Device::flex()
}

/// Device with autodiff - for gradient tests.
fn adev() -> Device {
    Device::flex().autodiff()
}

fn randn<const D: usize>(shape: [usize; D]) -> Tensor<D> {
    Tensor::<D>::random(shape, Distribution::Normal(0.0, 1.0), &dev())
}

fn to_vec<const D: usize>(t: Tensor<D>) -> Vec<f32> {
    t.into_data().try_to_vec::<f32>().unwrap()
}

fn max_diff<const D: usize>(a: Tensor<D>, b: Tensor<D>) -> f32 {
    to_vec(a)
        .iter()
        .zip(to_vec(b))
        .map(|(x, y)| (x - y).abs())
        .fold(0.0, f32::max)
}

fn all_finite<const D: usize>(t: Tensor<D>) -> bool {
    to_vec(t).iter().all(|v| v.is_finite())
}

fn base_cfg() -> LlmBlockConfig {
    LlmBlockConfig::new(64, 4, 2, 128).with_max_seq_len(32)
}

/// Full pass == flow one token at a time == prefill + chunk.
fn check_cache<F, G>(x: Tensor<3>, mut cache: KvCache, full: F, mut step: G, tol: f32)
where
    F: Fn(Tensor<3>) -> Tensor<3>,
    G: FnMut(Tensor<3>, &mut KvCache, usize) -> Tensor<3>,
{
    let [b, t, d] = x.dims();
    let expected = full(x.clone());

    let mut outs = Vec::new();
    for pos in 0..t {
        outs.push(step(
            x.clone().slice([0..b, pos..pos + 1, 0..d]),
            &mut cache,
            pos,
        ));
    }
    assert_eq!(cache.len(), t);
    let got = Tensor::cat(outs, 1);
    let diff = max_diff(got, expected.clone());
    assert!(diff < tol, "token-by-token: max diff {diff}");

    cache.clear();
    let split = t * 5 / 8;
    let head = step(x.clone().slice([0..b, 0..split, 0..d]), &mut cache, 0);
    let tail = step(x.slice([0..b, split..t, 0..d]), &mut cache, split);
    let diff = max_diff(Tensor::cat(vec![head, tail], 1), expected);
    assert!(diff < tol, "prefill+chunk: max diff {diff}");
}

// ───────────────────────── LlmBlock ─────────────────────────

#[test]
fn block_keeps_shape_and_is_finite() {
    let block = base_cfg().init(&dev());
    let y = block.forward(randn([2, 8, 64]));
    assert_eq!(y.dims(), [2, 8, 64]);
    assert!(all_finite(y));
}

#[test]
fn qkv_shapes() {
    let block = base_cfg().init(&dev());
    let (q, k, v) = block.attention.project_qkv(randn([2, 5, 64]), 0);
    assert_eq!(q.dims(), [2, 4, 5, 16]);
    assert_eq!(k.dims(), [2, 2, 5, 16]);
    assert_eq!(v.dims(), [2, 2, 5, 16]);
}

#[test]
fn gqa_mqa_mha_variants_run() {
    for (h, kv, kind) in [
        (32, 8, AttentionKind::Gqa),
        (16, 4, AttentionKind::Gqa),
        (8, 1, AttentionKind::Mqa),
        (8, 8, AttentionKind::Mha),
    ] {
        let cfg = LlmBlockConfig::new(h * 4, h, kv, 16)
            .with_max_seq_len(16)
            .with_attention_kind(kind);
        let block = cfg.init(&dev());
        let x = randn([1, 4, h * 4]);
        let (q, k, v) = block.attention.project_qkv(x.clone(), 0);
        assert_eq!(q.dims(), [1, h, 4, 4]);
        assert_eq!(k.dims(), [1, kv, 4, 4]);
        assert_eq!(v.dims(), [1, kv, 4, 4]);
        assert_eq!(block.forward(x).dims(), [1, 4, h * 4]);
    }
}

/// Grouping heads without copying K/V is equal to naive repeat_interleave.
#[test]
fn gqa_matches_naive_repeat() {
    let cfg = LlmBlockConfig::new(32, 8, 2, 16).with_max_seq_len(16);
    let attn = cfg.init(&dev()).attention;
    let x = randn([2, 6, 32]);
    let (q, k, v) = attn.project_qkv(x.clone(), 0);
    let (b, hq, t, dh) = (2, 8, 6, 4);
    let g = hq / 2;
    let rep = |y: Tensor<4>| {
        y.unsqueeze_dim::<5>(2)
            .expand([b, 2, g, t, dh])
            .reshape([b, hq, t, dh])
    };
    let scores = q.matmul(rep(k).swap_dims(2, 3)).div_scalar(2.0) + causal_mask(t, t, 0, &dev());
    let y = burn::tensor::activation::softmax(scores, 3)
        .matmul(rep(v))
        .swap_dims(1, 2)
        .reshape([b, t, hq * dh]);
    let expected = attn.o_proj.forward(y);
    assert!(max_diff(attn.forward(x, None), expected) < 1e-5);
}

#[test]
fn invalid_configs_are_rejected() {
    let bad = [
        (LlmBlockConfig::new(64, 0, 2, 128), "num_attention_heads"),
        (LlmBlockConfig::new(64, 4, 0, 128), "num_kv_heads"),
        (LlmBlockConfig::new(64, 4, 3, 128), "num_kv_heads"),
        (LlmBlockConfig::new(66, 4, 2, 128), "d_model"),
        (LlmBlockConfig::new(12, 4, 2, 128), "head_dim"),
        (LlmBlockConfig::new(0, 4, 2, 128), "d_model"),
        (LlmBlockConfig::new(64, 4, 2, 0), "d_ff"),
        (
            LlmBlockConfig::new(64, 4, 2, 128).with_max_seq_len(0),
            "max_seq_len",
        ),
        (
            LlmBlockConfig::new(64, 4, 2, 128).with_rms_norm_eps(0.0),
            "rms_norm_eps",
        ),
        (
            LlmBlockConfig::new(64, 4, 2, 128).with_attention_kind(AttentionKind::Mha),
            "Mha",
        ),
        (
            LlmBlockConfig::new(64, 4, 2, 128).with_attention_kind(AttentionKind::Mqa),
            "Mqa",
        ),
        (
            LlmBlockConfig::new(64, 4, 2, 128).with_ffn_kind(FfnKind::Moe),
            "moe",
        ),
    ];
    for (cfg, needle) in bad {
        let err = cfg.validate().expect_err(needle);
        assert!(err.contains(needle), "'{err}' does not contain '{needle}'");
    }
}

#[test]
#[should_panic(expected = "num_kv_heads")]
fn init_panics_on_invalid_config() {
    let _ = LlmBlockConfig::new(64, 4, 3, 128).init(&dev());
}

#[test]
fn causal_mask_values() {
    // 2 requests over 3 cache tokens: line 0 sees j<=3, line 1 sees everything.
    let m = to_vec(causal_mask(2, 5, 3, &dev()));
    let neg: Vec<bool> = m.iter().map(|v| *v == f32::NEG_INFINITY).collect();
    assert_eq!(
        neg,
        [
            false, false, false, false, true, false, false, false, false, false
        ]
    );
    assert!(m.iter().all(|v| *v == 0.0 || *v == f32::NEG_INFINITY));
}

#[test]
fn future_token_does_not_affect_past() {
    let block = base_cfg().init(&dev());
    let x1 = randn([2, 3, 64]);
    let tail = randn([2, 1, 64]);
    let x2 = Tensor::cat(vec![x1.clone().slice([0..2, 0..2, 0..64]), tail], 1);
    let (y1, y2) = (block.forward(x1), block.forward(x2));
    assert!(
        max_diff(
            y1.clone().slice([0..2, 0..2, 0..64]),
            y2.clone().slice([0..2, 0..2, 0..64])
        ) < 1e-5
    );
    assert!(max_diff(y1.slice([0..2, 2..3, 0..64]), y2.slice([0..2, 2..3, 0..64])) > 1e-4);
}

#[test]
fn explicit_mask_replaces_causal() {
    let block = base_cfg().init(&dev());
    let x = randn([1, 4, 64]);
    let open = Tensor::<4>::zeros([1, 1, 4, 4], &dev());
    let a = block.forward_with_mask(x.clone(), Some(open));
    let b = block.forward(x);
    assert!(max_diff(a.slice([0..1, 0..1, 0..64]), b.slice([0..1, 0..1, 0..64])) > 1e-5);
}

#[test]
fn all_ffn_kinds_run() {
    for kind in [FfnKind::SwiGlu, FfnKind::GeGlu, FfnKind::ReGlu] {
        let block = base_cfg().with_ffn_kind(kind).init(&dev());
        assert_eq!(block.forward(randn([1, 3, 64])).dims(), [1, 3, 64]);
    }
}

// ───────────────────────── RoPE ─────────────────────────

fn layouts() -> [RopeLayout; 2] {
    [RopeLayout::Interleaved, RopeLayout::HalfSplit]
}

#[test]
fn rope_preserves_shape_norm_and_position_zero() {
    for layout in layouts() {
        let rope = RotaryEmbedding::new(10_000.0, 16, 8).with_layout(layout);
        let x = randn([2, 3, 6, 8]);
        let y = rope.apply(x.clone(), 0);
        assert_eq!(y.dims(), [2, 3, 6, 8]);
        assert!(all_finite(y.clone()));
        let n = |t: Tensor<4>| t.powf_scalar(2.0).sum_dim(3);
        assert!(max_diff(n(x.clone()), n(y)) < 1e-4);
        // position 0: cos = 1, sin = 0
        let one = x.slice([0..2, 0..3, 0..1, 0..8]);
        assert!(max_diff(rope.apply(one.clone(), 0), one) < 1e-6);
    }
}

#[test]
fn rope_is_relative_and_start_pos_consistent() {
    for layout in layouts() {
        let rope = RotaryEmbedding::new(10_000.0, 16, 8).with_layout(layout);
        let q0 = randn([1, 1, 1, 8]);
        let k0 = randn([1, 1, 1, 8]);
        let q = rope.apply(q0.clone().repeat_dim(2, 8), 0);
        let k = rope.apply(k0.repeat_dim(2, 8), 0);
        let s = to_vec(q.clone().matmul(k.swap_dims(2, 3)));
        assert!(
            (s[3 * 8 + 1] - s[5 * 8 + 3]).abs() < 1e-4,
            "depends not only on m-n"
        );

        let full = rope.apply(q0.clone().repeat_dim(2, 8), 0);
        let one = rope.apply(q0, 5);
        assert!(max_diff(full.slice([0..1, 0..1, 5..6, 0..8]), one) < 1e-5);
    }
}

#[test]
fn rope_not_applied_to_v() {
    let attn = base_cfg().init(&dev()).attention;
    let x = randn([2, 5, 64]);
    let (_, _, v) = attn.project_qkv(x.clone(), 3);
    let expected = attn
        .v_proj
        .forward(x)
        .reshape([2, 5, 2, 16])
        .swap_dims(1, 2);
    assert!(max_diff(v, expected) < 1e-6);
}

#[test]
#[should_panic(expected = "head_dim")]
fn rope_rejects_odd_head_dim() {
    let _ = RotaryEmbedding::new(10_000.0, 8, 7);
}

// ───────────────────────── KV-cache ─────────────────────────

#[test]
fn kv_cache_append_get_clear() {
    let mut c = KvCache::new(8);
    assert!(c.is_empty() && c.get().is_none());
    c.append(randn([1, 2, 3, 4]), randn([1, 2, 3, 4]));
    c.append(randn([1, 2, 1, 4]), randn([1, 2, 1, 4]));
    let (k, v) = c.get().unwrap();
    assert_eq!(
        (k.dims(), v.dims(), c.len()),
        ([1, 2, 4, 4], [1, 2, 4, 4], 4)
    );
    c.clear();
    assert!(c.is_empty() && c.get().is_none());
}

#[test]
#[should_panic(expected = "overflow")]
fn kv_cache_overflow_panics() {
    let mut c = KvCache::new(2);
    c.append(randn([1, 1, 3, 4]), randn([1, 1, 3, 4]));
}

#[test]
#[should_panic(expected = "position")]
fn forward_cached_checks_position() {
    let block = base_cfg().init(&dev());
    let mut cache = block.new_cache();
    let _ = block.forward_cached(randn([1, 1, 64]), &mut cache, 3);
}

#[test]
fn cached_equals_full_for_every_layout() {
    for layout in layouts() {
        let block = base_cfg().with_rope_layout(layout).init(&dev());
        check_cache(
            randn([2, 8, 64]),
            block.new_cache(),
            |x| block.forward(x),
            |x, c, p| block.forward_cached(x, c, p),
            1e-4,
        );
    }
}

// ───────────────────────── Llama / Qwen ─────────────────────────

#[test]
fn llama_block() {
    let block = LlamaBlockConfig::new(64, 4, 2, 128)
        .with_max_seq_len(16)
        .init(&dev());
    let x = randn([2, 8, 64]);
    let y = block.forward(x.clone());
    assert_eq!(y.dims(), [2, 8, 64]);
    assert!(all_finite(y.clone()));
    assert!(
        max_diff(y, x.clone()) > 1e-3,
        "the block degenerates into the identity"
    );
    check_cache(
        x,
        block.new_cache(),
        |x| block.forward(x),
        |x, c, p| block.forward_cached(x, c, p),
        1e-4,
    );
}

#[test]
fn qwen_block_with_and_without_qk_norm() {
    for qk in [false, true] {
        let block = QwenBlockConfig::new(64, 4, 2, 128)
            .with_max_seq_len(16)
            .with_use_qk_norm(qk)
            .init(&dev());
        assert_eq!(block.block.attention.q_norm.is_some(), qk);
        assert_eq!(block.block.attention.k_norm.is_some(), qk);
        let x = randn([2, 8, 64]);
        let y = block.forward(x.clone());
        assert_eq!(y.dims(), [2, 8, 64]);
        assert!(all_finite(y));
        check_cache(
            x,
            block.new_cache(),
            |x| block.forward(x),
            |x, c, p| block.forward_cached(x, c, p),
            1e-4,
        );
    }
}

#[test]
fn qwen_qkv_bias_and_geglu() {
    let block = QwenBlockConfig::new(64, 4, 2, 128)
        .with_max_seq_len(16)
        .with_qkv_bias(true)
        .with_ffn_kind(FfnKind::GeGlu)
        .init(&dev());
    let a = &block.block.attention;
    assert!(a.q_proj.bias.is_some() && a.k_proj.bias.is_some() && a.v_proj.bias.is_some());
    assert!(a.o_proj.bias.is_none());
    assert_eq!(block.forward(randn([1, 4, 64])).dims(), [1, 4, 64]);
}

#[test]
fn qwen_rejects_moe_kind() {
    assert!(
        QwenBlockConfig::new(64, 4, 2, 128)
            .with_ffn_kind(FfnKind::Moe)
            .validate()
            .is_err()
    );
}

// ───────────────────────── MoE ─────────────────────────

/// Experts without bias (as in Llama/Qwen-MoE); the rest are default values.
fn moe_cfg(e: usize, k: usize) -> MoeConfig {
    MoeConfig::new(16, 24, e).with_top_k(k).with_bias(false)
}

#[test]
fn moe_routing_is_valid() {
    for k in [1, 2] {
        let moe = moe_cfg(4, k).init(&dev());
        let (probs, w, idx) = moe.route(randn([10, 16]));
        assert_eq!(
            (probs.dims(), w.dims(), idx.dims()),
            ([10, 4], [10, k], [10, k])
        );
        let idx: Vec<i64> = idx
            .into_data()
            .convert::<i64>()
            .try_to_vec::<i64>()
            .unwrap();
        assert!(idx.iter().all(|&i| (0..4).contains(&i)));
        let w = to_vec(w);
        assert!(w.iter().all(|v| v.is_finite() && *v >= 0.0));
        for row in w.chunks(k) {
            assert!((row.iter().sum::<f32>() - 1.0).abs() < 1e-5);
        }
    }
}

#[test]
fn moe_without_normalization_keeps_raw_probs() {
    let moe = moe_cfg(4, 2).with_normalize_top_k(false).init(&dev());
    let (_, w, _) = moe.route(randn([6, 16]));
    assert!(
        to_vec(w)
            .chunks(2)
            .all(|r| r.iter().sum::<f32>() < 1.0 + 1e-6)
    );
}

#[test]
fn moe_sparse_matches_dense_reference() {
    for shared in [false, true] {
        for k in [1, 2] {
            let moe = moe_cfg(4, k).with_shared_expert(shared).init(&dev());
            let x = randn([13, 16]);
            let (sparse, _) = moe.forward_tokens(x.clone());
            assert!(
                max_diff(sparse, moe.forward_dense(x)) < 1e-5,
                "k={k} shared={shared}"
            );
        }
    }
}

#[test]
fn moe_controlled_experts_combine_exactly() {
    let mut moe = moe_cfg(2, 2).with_router_bias(true).init(&dev());
    moe.router.weight = moe.router.weight.map(|w| w.zeros_like());
    let logits = [0.25f32.ln(), 0.75f32.ln()];
    moe.router.bias = Some(Param::from_tensor(Tensor::<1>::from_floats(logits, &dev())));
    // Expert 1 = exactly 2 · expert 0.
    // Param is initialized lazily: the clone would receive
    // other random weights, so we copy them via val().
    let lin = |l: &burn::nn::Linear, k: f32| burn::nn::Linear {
        weight: Param::from_tensor(l.weight.val().mul_scalar(k)),
        bias: None,
    };
    let e0 = &moe.experts[0];
    let e1 = crate::ffn::GatedFfn {
        gate: lin(&e0.gate, 1.0),
        value: lin(&e0.value, 1.0),
        out: lin(&e0.out, 2.0),
        activation: crate::ffn::GateActivation::Silu,
    };
    moe.experts[1] = e1;
    let probe = randn([3, 16]);
    assert!(
        max_diff(
            moe.experts[1].forward(probe.clone()),
            moe.experts[0].forward(probe).mul_scalar(2.0)
        ) < 1e-5,
        "Expert 1 must be exactly 2x Expert 0"
    );

    let x = randn([2, 5, 16]);
    let y = moe.forward(x.clone());
    let flat = x.reshape([10, 16]);
    let manual = moe.experts[0].forward(flat.clone()).mul_scalar(0.25)
        + moe.experts[1].forward(flat.clone()).mul_scalar(0.75);
    let scaled = moe.experts[0].forward(flat).mul_scalar(1.75);
    let y = y.reshape([10, 16]);
    assert!(max_diff(y.clone(), manual) < 1e-5);
    assert!(max_diff(y, scaled) < 1e-5);
}

#[test]
fn moe_capacity_drops_overflow() {
    let cfg = moe_cfg(4, 2).with_capacity_factor(0.5);
    assert_eq!(cfg.capacity(16), 4);
    let moe = cfg.init(&dev());
    let (y, aux) = moe.forward_tokens(randn([16, 16]));
    assert!(all_finite(y));
    let expected: usize = to_vec(aux.expert_load.clone())
        .iter()
        .map(|f| ((f * 32.0).round() as usize).saturating_sub(4))
        .sum();
    assert_eq!(aux.dropped_assignments, expected);
    assert!(aux.dropped_assignments >= 16);
}

#[test]
fn moe_default_capacity_never_drops() {
    let (_, aux) = moe_cfg(8, 2).init(&dev()).forward_tokens(randn([32, 16]));
    assert_eq!(aux.dropped_assignments, 0);
}

#[test]
fn moe_uniform_router_has_unit_balance_loss() {
    let mut moe = moe_cfg(4, 2).init(&dev());
    moe.router.weight = moe.router.weight.map(|w| w.zeros_like());
    let (_, aux) = moe.forward_with_aux(randn([2, 6, 16]));
    assert!((to_vec(moe.load_balance_loss(&aux))[0] - 1.0).abs() < 1e-5);
    assert!((to_vec(aux.expert_load).iter().sum::<f32>() - 1.0).abs() < 1e-5);
}

#[test]
fn moe_router_receives_gradient() {
    let moe = moe_cfg(4, 2).init(&adev());
    let x = Tensor::<3>::random([2, 4, 16], Distribution::Normal(0.0, 1.0), &adev());
    let (y, aux) = moe.forward_with_aux(x);
    let loss = y.powf_scalar(2.0).mean() + aux.load_balance_loss().mul_scalar(0.01);
    let grads = loss.backward();
    let g = moe
        .router
        .weight
        .val()
        .grad(&grads)
        .expect("router gradient");
    let g: Vec<f32> = g.into_data().try_to_vec::<f32>().unwrap();
    assert!(g.iter().all(|v| v.is_finite()) && g.iter().any(|v| v.abs() > 0.0));
}

#[test]
fn moe_block_shape_aux_and_cache() {
    let base = LlmBlockConfig::new(16, 2, 1, 24).with_max_seq_len(16);
    let block = MoeBlockConfig::new(base, moe_cfg(4, 2).with_shared_expert(true)).init(&dev());
    let x = randn([2, 8, 16]);
    let (y, aux) = block.forward_with_aux(x.clone(), None);
    assert_eq!(y.dims(), [2, 8, 16]);
    assert_eq!(aux.router_probs.dims(), [16, 4]);
    assert_eq!(aux.expert_indices.dims(), [16, 2]);
    assert!(all_finite(y));
    let moe = block.moe();
    assert_eq!(moe.num_experts(), 4);
    assert!(to_vec(block.load_balance_loss(&aux))[0] > 0.0);
    let mut cache = block.block.new_cache();
    check_cache(
        x,
        std::mem::replace(&mut cache, block.block.new_cache()),
        |x| block.forward(x),
        |x, c, p| block.forward_cached(x, c, p),
        1e-4,
    );
}

#[test]
fn moe_config_validation() {
    assert!(moe_cfg(0, 1).validate().is_err());
    assert!(moe_cfg(4, 0).validate().is_err());
    assert!(moe_cfg(4, 5).validate().is_err());
    assert!(moe_cfg(4, 2).with_capacity_factor(0.0).validate().is_err());
    assert!(moe_cfg(4, 2).validate().is_ok());
}

#[allow(dead_code)]
fn _int_type_check(_: Tensor<2, Int>) {}
