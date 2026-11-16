//! Attention tests: GQA, MQA, Sliding, Sparse, Linear, QK-Norm.

use burn::nn::Linear;
use burn::prelude::*;
use burn::tensor::Distribution;
use burn::tensor::activation::softmax;

use crate::attention::*;
use crate::transformer::{LlmAttention, LlmBlockConfig, causal_mask};

fn dev() -> Device {
    Device::flex()
}

/// Device with autodiff - for gradient tests.
fn adev() -> Device {
    Device::flex().autodiff()
}

fn v<const D: usize>(t: Tensor<D>) -> Vec<f32> {
    t.into_data().try_to_vec::<f32>().unwrap()
}

fn close(a: &[f32], b: &[f32], tol: f32) {
    assert_eq!(a.len(), b.len(), "different lengths");
    for (i, (x, y)) in a.iter().zip(b).enumerate() {
        assert!(x.is_finite() && (x - y).abs() <= tol, "[{i}] {x} vs {y}");
    }
}

fn rnd_on<const D: usize>(device: &Device, shape: [usize; D]) -> Tensor<D> {
    Tensor::random(shape, Distribution::Normal(0.0, 1.0), device)
}

fn rnd<const D: usize>(shape: [usize; D]) -> Tensor<D> {
    rnd_on(&dev(), shape)
}

/// Burn initializes the weights lazily: we force clone() to divide the values.
fn force(l: &Linear) {
    let _ = l.weight.val();
    let _ = l.bias.as_ref().map(|b| b.val());
}

fn force_attn(a: &GqaAttention) {
    for l in [&a.q_proj, &a.k_proj, &a.v_proj, &a.o_proj] {
        force(l);
    }
}

fn copy_weights(dst: &mut GqaAttention, src: &GqaAttention) {
    force_attn(src);
    dst.q_proj = src.q_proj.clone();
    dst.k_proj = src.k_proj.clone();
    dst.v_proj = src.v_proj.clone();
    dst.o_proj = src.o_proj.clone();
}

fn gqa_cfg(h: usize, kv: usize) -> GqaAttentionConfig {
    GqaAttentionConfig::new(16, h, kv)
}

fn gqa_on(device: &Device, h: usize, kv: usize) -> GqaAttention {
    let a = gqa_cfg(h, kv).init(device);
    force_attn(&a);
    a
}

fn gqa(h: usize, kv: usize) -> GqaAttention {
    gqa_on(&dev(), h, kv)
}

/// Naive standard: K/V are clearly repeated on the group, softmax on the head.
fn naive(q: Tensor<4>, k: Tensor<4>, val: Tensor<4>, mask: Option<Tensor<4>>) -> Tensor<4> {
    let [b, hq, _, dh] = q.dims();
    let [_, hkv, tk, _] = k.dims();
    let g = hq / hkv;
    let rep = |t: Tensor<4>| {
        t.unsqueeze_dim::<5>(2)
            .expand([b, hkv, g, tk, dh])
            .reshape([b, hq, tk, dh])
    };
    let mut s = q
        .matmul(rep(k).swap_dims(2, 3))
        .div_scalar((dh as f32).sqrt());
    if let Some(m) = mask {
        s = s + m;
    }
    softmax(s, 3).matmul(rep(val))
}

// ───────────────────────── QK-Norm ─────────────────────────

#[test]
fn qknorm_known_numeric_result() {
    let n = QkNormConfig::new(2).with_eps(1e-12).init(&dev());
    let q = Tensor::<4>::from_data(TensorData::new(vec![3.0, 4.0], [1, 1, 1, 2]), &dev());
    let y = v(n.forward_q(q));
    let rms = (12.5f32).sqrt();
    close(&y, &[3.0 / rms, 4.0 / rms], 1e-5);
}

#[test]
fn qknorm_shapes_zero_input_and_independence() {
    let mut n = QkNormConfig::new(8).init(&dev());
    let (q, k) = n.forward(rnd::<4>([2, 4, 3, 8]), rnd::<4>([2, 2, 3, 8]));
    assert_eq!(q.dims(), [2, 4, 3, 8]);
    assert_eq!(k.dims(), [2, 2, 3, 8]);
    assert!(
        v(n.forward_k(Tensor::<4>::zeros([1, 2, 3, 8], &dev())))
            .iter()
            .all(|a| *a == 0.0)
    );

    // the weight of Q does not affect K
    let kx = rnd::<4>([1, 2, 3, 8]);
    let before = v(n.forward_k(kx.clone()));
    n.q_norm.gamma = burn::module::Param::from_tensor(Tensor::full([8], 5.0, &dev()));
    close(&v(n.forward_k(kx)), &before, 0.0);
    assert_eq!(QkNormConfig::new(8).parameter_count(), 16);
}

#[test]
fn qknorm_gradient() {
    let n = QkNormConfig::new(8).init(&adev());
    let q = rnd_on(&adev(), [1, 2, 3, 8]).require_grad();
    let w = rnd_on(&adev(), [1, 2, 3, 8]);
    let grads = (n.forward_q(q.clone()) * w).sum().backward();
    assert!(q.grad(&grads).is_some());
    assert!(n.q_norm.gamma.val().grad(&grads).is_some());
    assert!(n.k_norm.gamma.val().grad(&grads).is_none());
}

// ───────────────────────── GQA ─────────────────────────

#[test]
fn gqa_matches_naive_reference_for_mha_gqa_mqa() {
    for (h, kv) in [(4, 4), (4, 2), (4, 1), (8, 2)] {
        let a = gqa(h, kv);
        let x = rnd::<3>([2, 5, 16]);
        let (q, k, val) = a.project_qkv(x.clone(), 0);
        let mask = causal_mask(5, 5, 0, &dev());
        let expected = a
            .o_proj
            .forward(a.merge_heads(naive(q, k, val, Some(mask))));
        close(&v(a.forward(x, None)), &v(expected), 1e-5);
    }
}

#[test]
fn gqa_shapes() {
    let a = gqa(4, 2);
    let (q, k, val) = a.project_qkv(rnd::<3>([2, 5, 16]), 0);
    assert_eq!(q.dims(), [2, 4, 5, 4]);
    assert_eq!(k.dims(), [2, 2, 5, 4]);
    assert_eq!(val.dims(), [2, 2, 5, 4]);
    assert_eq!(a.forward(rnd::<3>([2, 5, 16]), None).dims(), [2, 5, 16]);
}

#[test]
fn gqa_causal_has_no_future_leakage() {
    let a = gqa(4, 2);
    let x = rnd::<3>([1, 6, 16]);
    let y1 = v(a.forward(x.clone(), None));
    // change the last token
    let head = x.clone().narrow(1, 0, 5);
    let tail = rnd::<3>([1, 1, 16]) * 10.0;
    let y2 = v(a.forward(Tensor::cat(vec![head, tail], 1), None));
    close(&y1[..5 * 16], &y2[..5 * 16], 1e-6);
    assert!(
        y1[5 * 16..]
            .iter()
            .zip(&y2[5 * 16..])
            .any(|(p, q)| (p - q).abs() > 1e-4)
    );
}

#[test]
fn gqa_non_causal_sees_future() {
    let a = gqa_cfg(4, 2).with_causal(false).init(&dev());
    force_attn(&a);
    let x = rnd::<3>([1, 4, 16]);
    let y1 = v(a.forward(x.clone(), None));
    let tail = rnd::<3>([1, 1, 16]) * 10.0;
    let y2 = v(a.forward(Tensor::cat(vec![x.narrow(1, 0, 3), tail], 1), None));
    assert!(
        y1[..16]
            .iter()
            .zip(&y2[..16])
            .any(|(p, q)| (p - q).abs() > 1e-5)
    );
}

#[test]
fn gqa_gradients() {
    let a = gqa_on(&adev(), 4, 2);
    let x = rnd_on(&adev(), [2, 4, 16]).require_grad();
    let grads = a.forward(x.clone(), None).sum().backward();
    assert!(x.grad(&grads).is_some());
    for l in [&a.q_proj, &a.k_proj, &a.v_proj, &a.o_proj] {
        assert!(l.weight.val().grad(&grads).is_some());
    }
}

#[test]
fn gqa_parity_with_llm_attention() {
    for (h, kv, qk) in [(4, 4, false), (4, 2, false), (4, 1, true), (4, 2, true)] {
        let cfg = LlmBlockConfig::new(16, h, kv, 32).with_use_qk_norm(qk);
        let old = LlmAttention::new(&cfg, cfg.build_rotary(), &dev());
        for l in [&old.q_proj, &old.k_proj, &old.v_proj, &old.o_proj] {
            force(l);
        }
        let mut new = gqa_cfg(h, kv).with_use_qk_norm(qk).init(&dev());
        new.q_proj = old.q_proj.clone();
        new.k_proj = old.k_proj.clone();
        new.v_proj = old.v_proj.clone();
        new.o_proj = old.o_proj.clone();

        let x = rnd::<3>([2, 6, 16]);
        close(
            &v(new.forward(x.clone(), None)),
            &v(old.forward(x.clone(), None)),
            1e-5,
        );

        // incremental mode
        let mut c_old = crate::transformer::KvCache::new(32);
        let mut c_new = new.new_cache();
        let first = x.clone().narrow(1, 0, 4);
        close(
            &v(new.forward_cached(first.clone(), &mut c_new, 0)),
            &v(old.forward_cached(first, &mut c_old, 0)),
            1e-5,
        );
        for t in 4..6 {
            let xt = x.clone().narrow(1, t, 1);
            close(
                &v(new.forward_cached(xt.clone(), &mut c_new, t)),
                &v(old.forward_cached(xt, &mut c_old, t)),
                1e-5,
            );
        }
    }
}

#[test]
fn gqa_cached_equals_full() {
    let a = gqa(4, 2);
    let x = rnd::<3>([1, 6, 16]);
    let full = v(a.forward(x.clone(), None));
    let mut cache = a.new_cache();
    let mut outs = Vec::new();
    for t in 0..6 {
        outs.extend(v(a.forward_cached(
            x.clone().narrow(1, t, 1),
            &mut cache,
            t,
        )));
    }
    close(&outs, &full, 1e-5);
}

#[test]
fn gqa_config_validation() {
    assert!(GqaAttentionConfig::new(16, 4, 3).validate().is_err());
    assert!(GqaAttentionConfig::new(15, 4, 2).validate().is_err());
    assert!(GqaAttentionConfig::new(16, 4, 0).validate().is_err());
    assert!(GqaAttentionConfig::new(12, 4, 2).validate().is_err()); // head_dim 3 odd + RoPE
    assert!(
        GqaAttentionConfig::new(12, 4, 2)
            .with_rope(false)
            .validate()
            .is_ok()
    );
    assert!(
        GqaAttentionConfig::new(16, 4, 2)
            .with_dropout(1.0)
            .validate()
            .is_err()
    );
}

// ───────────────────────── MQA ─────────────────────────

#[test]
fn mqa_equals_gqa_with_one_kv_head() {
    let mqa = MqaAttentionConfig::new(16, 4)
        .with_use_qk_norm(true)
        .init(&dev());
    force_attn(&mqa.inner);
    let mut g = gqa_cfg(4, 1).with_use_qk_norm(true).init(&dev());
    copy_weights(&mut g, &mqa.inner);
    assert_eq!(mqa.inner.num_kv_heads, 1);
    let x = rnd::<3>([2, 5, 16]);
    close(
        &v(mqa.forward(x.clone(), None)),
        &v(g.forward(x, None)),
        1e-6,
    );
}

#[test]
fn mqa_gradients_and_cache() {
    let m = MqaAttentionConfig::new(16, 4).init(&adev());
    let x = rnd_on(&adev(), [1, 4, 16]).require_grad();
    let grads = m.forward(x.clone(), None).sum().backward();
    assert!(x.grad(&grads).is_some());
    assert!(m.inner.k_proj.weight.val().grad(&grads).is_some());

    let m = MqaAttentionConfig::new(16, 4).init(&adev());
    let mut c = m.new_cache();
    let _ = m.forward_cached(rnd_on(&adev(), [1, 3, 16]), &mut c, 0);
    assert_eq!(c.get().unwrap().0.dims(), [1, 1, 3, 4]);
}

// ───────────────────────── Sliding ─────────────────────────

fn sliding(w: usize) -> SlidingWindowAttention {
    let a = SlidingWindowConfig::new(w, gqa_cfg(4, 2)).init(&adev());
    force_attn(&a.inner);
    a
}

#[test]
fn sliding_large_window_equals_dense() {
    let s = sliding(100);
    let x = rnd::<3>([2, 7, 16]);
    close(&v(s.forward(x.clone())), &v(s.inner.forward(x, None)), 1e-6);
}

#[test]
fn sliding_window_one_attends_only_to_self() {
    let s = sliding(1);
    let x = rnd::<3>([2, 5, 16]);
    let (_, _, val) = s.inner.project_qkv(x.clone(), 0);
    // each query head sees only its “parent” V → output = o_proj(V extended)
    let [b, hkv, t, dh] = val.dims();
    let g = 4 / hkv;
    let expanded = val
        .unsqueeze_dim::<5>(2)
        .expand([b, hkv, g, t, dh])
        .reshape([b, 4, t, dh]);
    let expected = s.inner.o_proj.forward(s.inner.merge_heads(expanded));
    close(&v(s.forward(x)), &v(expected), 1e-5);
}

#[test]
fn sliding_no_future_leakage_and_window_limit() {
    let s = sliding(2);
    let x = rnd::<3>([1, 6, 16]);
    let y1 = v(s.forward(x.clone()));

    // the future does not affect
    let tail = rnd::<3>([1, 1, 16]) * 10.0;
    let y2 = v(s.forward(Tensor::cat(vec![x.clone().narrow(1, 0, 5), tail], 1)));
    close(&y1[..5 * 16], &y2[..5 * 16], 1e-6);

    // token 0 is outside the window for position 3 (sees 2,3), and for all positions ≥ 2
    let head = rnd::<3>([1, 1, 16]) * 10.0;
    let y3 = v(s.forward(Tensor::cat(vec![head, x.narrow(1, 1, 5)], 1)));
    close(&y1[2 * 16..], &y3[2 * 16..], 1e-5);
    assert!(
        y1[..16]
            .iter()
            .zip(&y3[..16])
            .any(|(p, q)| (p - q).abs() > 1e-4)
    );
}

#[test]
fn sliding_cached_equals_full_and_partial_windows() {
    for w in [1, 2, 3, 6] {
        let s = sliding(w);
        let x = rnd::<3>([1, 6, 16]);
        let full = v(s.forward(x.clone()));
        let mut c = s.new_cache();
        let mut outs = Vec::new();
        for t in 0..6 {
            outs.extend(v(s.forward_cached(x.clone().narrow(1, t, 1), &mut c, t)));
        }
        close(&outs, &full, 1e-5);
    }
}

#[test]
fn sliding_validation_and_gradient() {
    assert!(
        SlidingWindowConfig::new(0, gqa_cfg(4, 2))
            .validate()
            .is_err()
    );
    let s = SlidingWindowConfig::new(2, gqa_cfg(4, 2)).init(&adev());
    let x = rnd_on(&adev(), [1, 5, 16]).require_grad();
    let grads = s.forward(x.clone()).sum().backward();
    assert!(x.grad(&grads).is_some());
    assert!(s.inner.q_proj.weight.val().grad(&grads).is_some());
}

// ───────────────────────── Sparse ─────────────────────────

fn sparse(p: SparsePattern) -> SparseAttention {
    let a = SparseAttentionConfig::new(p, gqa_cfg(4, 2)).init(&adev());
    force_attn(&a.inner);
    a
}

#[test]
fn sparse_pattern_blocks() {
    let p = SparsePattern::BlockLocal {
        block_size: 2,
        blocks_per_query: 2,
    };
    assert_eq!(
        p.allowed_blocks(4, 4),
        vec![vec![0], vec![0, 1], vec![1, 2], vec![2, 3]]
    );
    let e = SparsePattern::Explicit {
        block_size: 2,
        pairs: vec![(0, 0), (1, 0), (1, 1), (1, 0), (2, 9)],
    };
    assert_eq!(e.allowed_blocks(3, 3), vec![vec![0], vec![0, 1], vec![]]);
    assert!(
        SparsePattern::BlockLocal {
            block_size: 0,
            blocks_per_query: 1
        }
        .validate()
        .is_err()
    );
    assert!(
        SparsePattern::Explicit {
            block_size: 2,
            pairs: vec![]
        }
        .validate()
        .is_err()
    );
}

#[test]
fn sparse_full_pattern_equals_dense() {
    let s = sparse(SparsePattern::BlockLocal {
        block_size: 2,
        blocks_per_query: 100,
    });
    let x = rnd::<3>([2, 6, 16]);
    close(&v(s.forward(x.clone())), &v(s.inner.forward(x, None)), 1e-6);
}

#[test]
fn sparse_single_block_equals_dense() {
    let s = sparse(SparsePattern::BlockLocal {
        block_size: 8,
        blocks_per_query: 1,
    });
    let x = rnd::<3>([1, 6, 16]);
    close(&v(s.forward(x.clone())), &v(s.inner.forward(x, None)), 1e-6);
}

#[test]
fn sparse_forbidden_blocks_do_not_influence_output() {
    // block 1 sees only itself ⇒ changing block 0 does not affect the outputs of block 1
    let s = sparse(SparsePattern::BlockLocal {
        block_size: 3,
        blocks_per_query: 1,
    });
    let x = rnd::<3>([1, 6, 16]);
    let y1 = v(s.forward(x.clone()));
    let head = rnd::<3>([1, 3, 16]) * 10.0;
    let y2 = v(s.forward(Tensor::cat(vec![head, x.narrow(1, 3, 3)], 1)));
    close(&y1[3 * 16..], &y2[3 * 16..], 1e-5);
    assert!(
        y1[..3 * 16]
            .iter()
            .zip(&y2[..3 * 16])
            .any(|(p, q)| (p - q).abs() > 1e-4)
    );
}

#[test]
fn sparse_explicit_matches_naive_masked_attention() {
    // block 0 → {0}; block 1 → {0, 1}; block 2 → {0, 2} (skip block 1)
    let pairs = vec![(0, 0), (1, 0), (1, 1), (2, 0), (2, 2)];
    let s = sparse(SparsePattern::Explicit {
        block_size: 2,
        pairs,
    });
    let x = rnd::<3>([2, 6, 16]);
    let (q, k, val) = s.inner.project_qkv(x.clone(), 0);

    let allowed = |i: usize, j: usize| {
        let (qb, kb) = (i / 2, j / 2);
        j <= i && matches!((qb, kb), (0, 0) | (1, 0) | (1, 1) | (2, 0) | (2, 2))
    };
    let mut m = Vec::new();
    for i in 0..6 {
        for j in 0..6 {
            m.push(if allowed(i, j) {
                0.0
            } else {
                f32::NEG_INFINITY
            });
        }
    }
    let mask = Tensor::<4>::from_data(TensorData::new(m, [1, 1, 6, 6]), &dev());
    let expected = s
        .inner
        .o_proj
        .forward(s.inner.merge_heads(naive(q, k, val, Some(mask))));
    close(&v(s.forward(x)), &v(expected), 1e-5);
}

#[test]
fn sparse_causal_and_cached() {
    let s = sparse(SparsePattern::BlockLocal {
        block_size: 2,
        blocks_per_query: 2,
    });
    let x = rnd::<3>([1, 6, 16]);
    let y1 = v(s.forward(x.clone()));
    let tail = rnd::<3>([1, 1, 16]) * 10.0;
    let y2 = v(s.forward(Tensor::cat(vec![x.clone().narrow(1, 0, 5), tail], 1)));
    close(&y1[..5 * 16], &y2[..5 * 16], 1e-6);

    let mut c = s.new_cache();
    let mut outs = Vec::new();
    for t in 0..6 {
        outs.extend(v(s.forward_cached(x.clone().narrow(1, t, 1), &mut c, t)));
    }
    close(&outs, &y1, 1e-5);
}

#[test]
#[should_panic(expected = "completely masked")]
fn sparse_fully_masked_row_panics() {
    // block 1 sees nothing
    let s = sparse(SparsePattern::Explicit {
        block_size: 2,
        pairs: vec![(0, 0)],
    });
    s.forward(rnd::<3>([1, 4, 16]));
}

#[test]
fn sparse_gradient() {
    let s = SparseAttentionConfig::new(
        SparsePattern::BlockLocal {
            block_size: 2,
            blocks_per_query: 2,
        },
        gqa_cfg(4, 2),
    )
    .init(&adev());
    let x = rnd_on(&adev(), [1, 6, 16]).require_grad();
    let grads = s.forward(x.clone()).sum().backward();
    assert!(x.grad(&grads).is_some());
}

// ───────────────────────── Linear ─────────────────────────

fn lin_attn_on(device: &Device, causal: bool) -> LinearAttention {
    let a = LinearAttentionConfig::new(16, 4)
        .with_causal(causal)
        .init(device);
    for l in [&a.q_proj, &a.k_proj, &a.v_proj, &a.o_proj] {
        force(l);
    }
    a
}

fn lin_attn(causal: bool) -> LinearAttention {
    lin_attn_on(&adev(), causal)
}

#[test]
fn linear_shape_and_finite() {
    for causal in [true, false] {
        let y = v(lin_attn(causal).forward(rnd::<3>([2, 7, 16])));
        assert_eq!(y.len(), 2 * 7 * 16);
        assert!(y.iter().all(|a| a.is_finite()));
    }
}

#[test]
fn linear_causal_property() {
    let a = lin_attn(true);
    let x = rnd::<3>([1, 6, 16]);
    let y1 = v(a.forward(x.clone()));
    let tail = rnd::<3>([1, 1, 16]) * 10.0;
    let y2 = v(a.forward(Tensor::cat(vec![x.narrow(1, 0, 5), tail], 1)));
    close(&y1[..5 * 16], &y2[..5 * 16], 1e-5);
    assert!(
        y1[5 * 16..]
            .iter()
            .zip(&y2[5 * 16..])
            .any(|(p, q)| (p - q).abs() > 1e-4)
    );
}

#[test]
fn linear_non_causal_sees_future() {
    let a = lin_attn(false);
    let x = rnd::<3>([1, 4, 16]);
    let y1 = v(a.forward(x.clone()));
    let tail = rnd::<3>([1, 1, 16]) * 10.0;
    let y2 = v(a.forward(Tensor::cat(vec![x.narrow(1, 0, 3), tail], 1)));
    assert!(
        y1[..16]
            .iter()
            .zip(&y2[..16])
            .any(|(p, q)| (p - q).abs() > 1e-5)
    );
}

#[test]
fn linear_matches_quadratic_reference() {
    // y_t = Σ_{j≤t} (φq_t·φk_j) v_j / (Σ_{j≤t} φq_t·φk_j + eps)
    let a = lin_attn(true);
    let x = rnd::<3>([1, 5, 16]);
    let [b, t, _] = x.dims();
    let heads = |y: Tensor<3>| y.reshape([b, t, 4, 4]).swap_dims(1, 2);
    let phi = |y: Tensor<4>| y.clone().clamp_max(0.0).exp() + y.clamp_min(0.0);
    let q = phi(heads(a.q_proj.forward(x.clone())));
    let k = phi(heads(a.k_proj.forward(x.clone())));
    let val = heads(a.v_proj.forward(x.clone()));
    let mask = causal_mask(t, t, 0, &dev())
        .clamp_min(-1e30)
        .add_scalar(0.0);
    let keep = mask.equal_elem(0.0).float(); // 1 where possible
    let w = q.matmul(k.swap_dims(2, 3)) * keep;
    let den = w.clone().sum_dim(3).add_scalar(1e-5);
    let y = w.matmul(val) / den;
    let expected = a.o_proj.forward(y.swap_dims(1, 2).reshape([b, t, 16]));
    close(&v(a.forward(x)), &v(expected), 1e-4);
}

#[test]
fn linear_stateful_equals_full() {
    let a = lin_attn(true);
    let x = rnd::<3>([2, 7, 16]);
    let full = v(a.forward(x.clone()));

    // token by token
    let mut st = a.new_state(2, &dev());
    let mut parts = Vec::new();
    for t in 0..7 {
        parts.push(a.forward_stateful(x.clone().narrow(1, t, 1), &mut st));
    }
    close(&v(Tensor::cat(parts, 1)), &full, 1e-4);

    // chunks 3 + 4
    let mut st = a.new_state(2, &dev());
    let y1 = a.forward_stateful(x.clone().narrow(1, 0, 3), &mut st);
    let y2 = a.forward_stateful(x.narrow(1, 3, 4), &mut st);
    close(&v(Tensor::cat(vec![y1, y2], 1)), &full, 1e-4);
}

#[test]
fn linear_gradients() {
    let a = lin_attn_on(&adev(), true);
    let x = rnd_on(&adev(), [1, 5, 16]).require_grad();
    let grads = a.forward(x.clone()).sum().backward();
    assert!(x.grad(&grads).is_some());
    for l in [&a.q_proj, &a.k_proj, &a.v_proj, &a.o_proj] {
        assert!(l.weight.val().grad(&grads).is_some());
    }
}

#[test]
fn linear_validation() {
    assert!(LinearAttentionConfig::new(15, 4).validate().is_err());
    assert!(LinearAttentionConfig::new(16, 0).validate().is_err());
    assert!(
        LinearAttentionConfig::new(16, 4)
            .with_eps(0.0)
            .validate()
            .is_err()
    );
}

#[test]
#[should_panic]
fn linear_stateful_requires_causal() {
    let a = lin_attn(false);
    let mut st = a.new_state(1, &dev());
    a.forward_stateful(rnd::<3>([1, 2, 16]), &mut st);
}
