//! Tests of embedding layers: FactorizedEmbedding, AdaptiveEmbedding, FourierEmbedding.

use burn::prelude::*;
use burn::tensor::{Device, Distribution, Int, TensorData};

use crate::embedding::*;

// ───────────────────────── auxiliary ─────────────────────────

fn dev() -> Device {
    Device::flex()
}

/// Device with autodiff - for gradient tests.
fn adev() -> Device {
    Device::flex().autodiff()
}

/// Int-tensor `[B, T]` from the flat list id.
fn ids_on(device: &Device, data: Vec<i64>, shape: [usize; 2]) -> Tensor<2, Int> {
    Tensor::from_data(TensorData::new(data, shape), device)
}

fn ids(data: Vec<i64>, shape: [usize; 2]) -> Tensor<2, Int> {
    ids_on(&dev(), data, shape)
}

fn to_vec<const D: usize>(t: Tensor<D>) -> Vec<f32> {
    t.into_data().try_to_vec::<f32>().unwrap()
}

fn assert_close(got: &[f32], expected: &[f32], tol: f32) {
    assert_eq!(got.len(), expected.len(), "different lengths");
    for (i, (g, e)) in got.iter().zip(expected).enumerate() {
        assert!((g - e).abs() <= tol, "[{i}] got {g}, expected {e}");
    }
}

// ───────────────────────── FactorizedEmbedding ─────────────────────────

fn fact_cfg() -> FactorizedEmbeddingConfig {
    FactorizedEmbeddingConfig::new(50, 6, 16)
}

#[test]
fn factorized_forward_maps_bt_to_btd() {
    let emb = fact_cfg().init(&dev());
    let y = emb.forward(ids(vec![1, 2, 3, 4, 5, 6], [2, 3]));
    assert_eq!(y.dims(), [2, 3, 16]);
}

#[test]
fn factorized_lookup_has_rank_dimension() {
    let emb = fact_cfg().init(&dev());
    assert_eq!(emb.lookup(ids(vec![1, 2, 3, 4], [2, 2])).dims(), [2, 2, 6]);
}

/// The same ids give the same vectors - both before and after the projection.
#[test]
fn factorized_repeated_tokens_give_identical_vectors() {
    let emb = fact_cfg().init(&dev());
    let x = ids(vec![5, 5, 5], [1, 3]);

    let z = to_vec(emb.lookup(x.clone()));
    assert_close(&z[0..6], &z[6..12], 0.0);
    assert_close(&z[0..6], &z[12..18], 0.0);

    let y = to_vec(emb.forward(x));
    assert_close(&y[0..16], &y[16..32], 1e-6);
    assert_close(&y[0..16], &y[32..48], 1e-6);
}

/// The result coincides with the manual calculation on the host: `E1[token] · E2`.
#[test]
fn factorized_matches_manual_matrix_product() {
    let emb = fact_cfg().init(&dev());
    let tokens = [3_i64, 0, 49, 3];
    let y = to_vec(emb.forward(ids(tokens.to_vec(), [2, 2])));

    let e1 = to_vec(emb.token_embedding.weight.val()); // [50, 6]
    let w = to_vec(emb.projection.weight.val()); // [6, 16]

    let mut expected = Vec::new();
    for &tok in &tokens {
        for d in 0..16 {
            let mut s = 0.0f32;
            for r in 0..6 {
                s += e1[tok as usize * 6 + r] * w[r * 16 + d];
            }
            expected.push(s);
        }
    }
    assert_close(&y, &expected, 1e-5);
}

#[test]
fn factorized_batched_equals_per_token() {
    let emb = fact_cfg().init(&dev());
    let tokens = vec![7_i64, 1, 42, 1, 0, 49];
    let full = to_vec(emb.forward(ids(tokens.clone(), [2, 3])));
    for (i, &tok) in tokens.iter().enumerate() {
        let single = to_vec(emb.forward(ids(vec![tok], [1, 1])));
        assert_close(&full[i * 16..(i + 1) * 16], &single, 1e-5);
    }
}

#[test]
fn factorized_gradients_reach_both_matrices() {
    let emb = fact_cfg().init(&adev());
    let y = emb.forward(ids_on(&adev(), vec![1, 2, 1, 3], [2, 2]));
    let grads = y.sum().backward();

    let g_tok = emb
        .token_embedding
        .weight
        .val()
        .grad(&grads)
        .expect("gradient E1");
    let g_proj = emb
        .projection
        .weight
        .val()
        .grad(&grads)
        .expect("gradient E2");

    let g_tok = to_vec(g_tok);
    let g_proj = to_vec(g_proj);
    assert!(g_proj.iter().any(|v| v.abs() > 0.0));

    // The lines of used tokens (1, 2, 3) are non-zero, the lines of unused tokens (0) are zero.
    let row = |t: usize| &g_tok[t * 6..(t + 1) * 6];
    assert!(row(1).iter().any(|v| v.abs() > 0.0));
    assert!(row(2).iter().any(|v| v.abs() > 0.0));
    assert!(row(3).iter().any(|v| v.abs() > 0.0));
    assert!(row(0).iter().all(|v| *v == 0.0));
}

#[test]
fn factorized_parameter_count_is_smaller_than_dense() {
    let c = FactorizedEmbeddingConfig::new(1000, 16, 64);
    assert_eq!(c.parameter_count(), 1000 * 16 + 16 * 64);
    assert!(c.parameter_count() < c.dense_parameter_count());
    assert_eq!(
        c.clone().with_bias(true).parameter_count(),
        1000 * 16 + 16 * 64 + 64
    );

    // Example from the documentation.
    let big = FactorizedEmbeddingConfig::new(200_000, 256, 4096);
    assert_eq!(big.dense_parameter_count(), 819_200_000);
    assert_eq!(big.parameter_count(), 52_248_576);
}

#[test]
fn factorized_weight_shapes_match_documentation() {
    let emb = fact_cfg().init(&dev());
    assert_eq!(emb.token_embedding.weight.val().dims(), [50, 6]);
    assert_eq!(emb.projection.weight.val().dims(), [6, 16]);
}

#[test]
fn factorized_project_to_vocab_matches_associated_product() {
    let emb = fact_cfg().init(&dev());
    let hidden = Tensor::<3>::random(
        [2, 3, 16],
        burn::tensor::Distribution::Normal(0.0, 1.0),
        &dev(),
    );

    let logits = emb.project_to_vocab(hidden.clone());
    assert_eq!(logits.dims(), [2, 3, 50]);

    // (h · Wᵀ) · E1ᵀ == h · (Wᵀ · E1ᵀ)
    let w = emb.projection.weight.val(); // [6, 16]
    let e = emb.token_embedding.weight.val(); // [50, 6]
    let combined = w.transpose().matmul(e.transpose()); // [16, 50]
    let expected = hidden.reshape([6, 16]).matmul(combined).reshape([2, 3, 50]);
    assert_close(&to_vec(logits), &to_vec(expected), 1e-4);
}

#[test]
fn factorized_validate_rejects_zero_sizes() {
    assert!(
        FactorizedEmbeddingConfig::new(0, 4, 8)
            .validate()
            .unwrap_err()
            .contains("vocab_size")
    );
    assert!(
        FactorizedEmbeddingConfig::new(10, 0, 8)
            .validate()
            .unwrap_err()
            .contains("rank")
    );
    assert!(
        FactorizedEmbeddingConfig::new(10, 4, 0)
            .validate()
            .unwrap_err()
            .contains("d_model")
    );
    assert!(
        FactorizedEmbeddingConfig::new(10, 4, 8)
            .try_init(&dev())
            .is_ok()
    );
}

#[test]
#[should_panic(expected = "rank")]
fn factorized_init_panics_on_invalid_config() {
    let _ = FactorizedEmbeddingConfig::new(10, 0, 8).init(&dev());
}

// ───────────────────────── AdaptiveEmbedding ─────────────────────────

/// 20 tokens: [0,5) dim 8 (no projection), [5,12) dim 4, [12,20) dim 2; D = 8.
fn small_cfg() -> AdaptiveEmbeddingConfig {
    AdaptiveEmbeddingConfig::new(
        20,
        8,
        vec![
            AdaptiveTierConfig::new(0, 5, 8),
            AdaptiveTierConfig::new(5, 12, 4),
            AdaptiveTierConfig::new(12, 20, 2),
        ],
    )
}

/// Same boundaries as in the example (50k / 150k / 200k), but narrow dimensions -
/// so that tests on real ids are cheap. D = 4.
fn spec_cfg() -> AdaptiveEmbeddingConfig {
    AdaptiveEmbeddingConfig::new(
        200_000,
        4,
        vec![
            AdaptiveTierConfig::new(0, 50_000, 4),
            AdaptiveTierConfig::new(50_000, 150_000, 2),
            AdaptiveTierConfig::new(150_000, 200_000, 1),
        ],
    )
}

/// Manual calculation of one token on the host (bias = false).
fn reference_token(emb: &AdaptiveEmbedding, id: usize) -> Vec<f32> {
    let tier = &emb.tiers[emb.tier_index(id).unwrap()];
    let local = id - tier.start;
    let table = to_vec(tier.embedding.weight.val()); // [size, dim]
    let row = table[local * tier.dim..(local + 1) * tier.dim].to_vec();
    match &tier.projection {
        None => row,
        Some(p) => {
            let w = to_vec(p.weight.val()); // [dim, out]
            (0..emb.output_dim)
                .map(|d| {
                    (0..tier.dim)
                        .map(|r| row[r] * w[r * emb.output_dim + d])
                        .sum()
                })
                .collect()
        }
    }
}

fn sample_ids() -> Vec<i64> {
    (0..24).map(|i| (i * 7 + 3) % 20).collect()
}

// ───────────── routing of tokens by tier ─────────────

#[test]
fn adaptive_token_to_tier_mapping_at_boundaries() {
    let cfg = AdaptiveEmbeddingConfig::from_cutoffs(200_000, 4096, &[50_000, 150_000], 2);
    assert_eq!(cfg.tier_of(0), Some(0));
    assert_eq!(cfg.tier_of(49_999), Some(0));
    assert_eq!(cfg.tier_of(50_000), Some(1));
    assert_eq!(cfg.tier_of(149_999), Some(1));
    assert_eq!(cfg.tier_of(150_000), Some(2));
    assert_eq!(cfg.tier_of(199_999), Some(2));
    assert_eq!(cfg.tier_of(200_000), None);
}

#[test]
fn adaptive_local_index_is_global_minus_start() {
    let cfg = spec_cfg();
    assert_eq!(cfg.locate(50_000), Some((1, 0)));
    assert_eq!(cfg.locate(50_017), Some((1, 17)));
    assert_eq!(cfg.locate(150_003), Some((2, 3)));
    assert_eq!(cfg.locate(7), Some((0, 7)));
}

#[test]
fn adaptive_module_and_config_agree_on_tier_index() {
    let emb = small_cfg().init(&dev());
    let cfg = small_cfg();
    for id in 0..21 {
        assert_eq!(emb.tier_index(id), cfg.tier_of(id), "id {id}");
    }
}

#[test]
fn adaptive_from_cutoffs_builds_geometric_dims() {
    let cfg = AdaptiveEmbeddingConfig::from_cutoffs(200_000, 4096, &[50_000, 150_000], 2);
    let got: Vec<_> = cfg.tiers.iter().map(|t| (t.start, t.end, t.dim)).collect();
    assert_eq!(
        got,
        vec![
            (0, 50_000, 4096),
            (50_000, 150_000, 2048),
            (150_000, 200_000, 1024)
        ]
    );
    assert!(cfg.validate().is_ok());
}

// ───────────── shape and structure ─────────────

#[test]
fn adaptive_output_shape_is_btd_for_both_modes() {
    let emb = small_cfg().init(&dev());
    let x = ids(sample_ids(), [4, 6]);
    assert_eq!(emb.forward(x.clone()).dims(), [4, 6, 8]);
    assert_eq!(emb.forward_routed(x).dims(), [4, 6, 8]);
}

#[test]
fn adaptive_projection_exists_only_when_dim_differs() {
    let emb = small_cfg().init(&dev());
    assert!(emb.tiers[0].projection.is_none());
    assert!(emb.tiers[1].projection.is_some());
    assert!(emb.tiers[2].projection.is_some());
    assert_eq!(emb.tiers[1].embedding.weight.val().dims(), [7, 4]);
    assert_eq!(emb.tiers[2].embedding.weight.val().dims(), [8, 2]);
}

#[test]
fn adaptive_parameter_count_matches_formula() {
    // 5·8 + (7·4 + 4·8) + (8·2 + 2·8) = 40 + 60 + 32
    assert_eq!(small_cfg().parameter_count(), 132);
    assert!(small_cfg().parameter_count() < small_cfg().dense_parameter_count());
    // With bias, output_dim is added to each projection (two projections).
    assert_eq!(small_cfg().with_bias(true).parameter_count(), 132 + 2 * 8);
}

// ───────────── correctness of values ─────────────

/// The result of both modes is equal to manual per-token calculation, including tier boundaries.
#[test]
fn adaptive_matches_manual_per_token_reference_at_boundaries() {
    let emb = spec_cfg().init(&dev());
    let tokens = [0_i64, 49_999, 50_000, 149_999, 150_000, 199_999];

    let mut expected = Vec::new();
    for &t in &tokens {
        expected.extend(reference_token(&emb, t as usize));
    }
    let x = ids(tokens.to_vec(), [2, 3]);
    assert_close(&to_vec(emb.forward(x.clone())), &expected, 1e-5);
    assert_close(&to_vec(emb.forward_routed(x)), &expected, 1e-5);
}

/// Critical test: batched lookup == concatenation of individual forwards.
/// Catches routing / scatter / token order errors.
#[test]
fn adaptive_batched_forward_equals_single_token_forwards() {
    let emb = spec_cfg().init(&dev());
    let tokens = vec![5_i64, 100_000, 7, 170_000];

    let dense = to_vec(emb.forward(ids(tokens.clone(), [1, 4])));
    let routed = to_vec(emb.forward_routed(ids(tokens.clone(), [1, 4])));

    let mut concat = Vec::new();
    for &t in &tokens {
        concat.extend(to_vec(emb.forward(ids(vec![t], [1, 1]))));
    }
    assert_close(&dense, &concat, 1e-6);
    assert_close(&routed, &concat, 1e-6);
}

#[test]
fn adaptive_dense_and_routed_agree_and_keep_token_order() {
    let emb = small_cfg().init(&dev());
    let x = ids(sample_ids(), [4, 6]);
    assert_close(
        &to_vec(emb.forward(x.clone())),
        &to_vec(emb.forward_routed(x)),
        1e-6,
    );
}

#[test]
fn adaptive_repeated_tokens_give_identical_vectors() {
    let emb = small_cfg().init(&dev());
    let y = to_vec(emb.forward(ids(vec![9, 9, 9], [1, 3])));
    assert_close(&y[0..8], &y[8..16], 1e-6);
    assert_close(&y[0..8], &y[16..24], 1e-6);
}

#[test]
fn adaptive_out_of_range_id_gives_zero_vector_in_dense_mode() {
    let emb = small_cfg().init(&dev());
    let y = to_vec(emb.forward(ids(vec![20, 3], [1, 2])));
    assert!(y[0..8].iter().all(|v| *v == 0.0));
    assert!(y[8..16].iter().any(|v| v.abs() > 0.0));
}

#[test]
#[should_panic(expected = "AdaptiveEmbedding")]
fn adaptive_routed_mode_rejects_out_of_range_id() {
    let emb = small_cfg().init(&dev());
    let _ = emb.forward_routed(ids(vec![20], [1, 1]));
}

// ───────────── gradients ─────────────

#[test]
fn adaptive_gradients_reach_every_used_tier() {
    let emb = small_cfg().init(&adev());
    let y = emb.forward(ids_on(&adev(), vec![0, 6, 13, 1, 7, 14], [2, 3]));
    let grads = y.sum().backward();

    for (k, tier) in emb.tiers.iter().enumerate() {
        let g = to_vec(
            tier.embedding
                .weight
                .val()
                .grad(&grads)
                .expect("gradient table"),
        );
        assert!(
            g.iter().any(|v| v.abs() > 0.0),
            "tier {k}: zero table gradient"
        );
        if let Some(p) = &tier.projection {
            let g = to_vec(p.weight.val().grad(&grads).expect("projection gradient"));
            assert!(
                g.iter().any(|v| v.abs() > 0.0),
                "tier {k}: zero projection gradient"
            );
        }
    }
}

/// Tier without tokens in the batch does not receive a non-zero gradient (mask).
#[test]
fn adaptive_unused_tier_gets_no_gradient() {
    let emb = small_cfg().init(&adev());
    let y = emb.forward(ids_on(&adev(), vec![0, 1, 2, 3], [1, 4])); // only tier 0
    let grads = y.sum().backward();

    for k in [1, 2] {
        if let Some(g) = emb.tiers[k].embedding.weight.val().grad(&grads) {
            assert!(
                to_vec(g).iter().all(|v| *v == 0.0),
                "tier {k}: gradient must be zero"
            );
        }
    }
}

#[test]
fn adaptive_dense_and_routed_produce_same_gradients() {
    let emb = small_cfg().init(&adev());
    let x = ids_on(&adev(), sample_ids(), [4, 6]);

    let g_dense = emb.forward(x.clone()).sum().backward();
    let g_routed = emb.forward_routed(x).sum().backward();

    for tier in emb.tiers.iter().take(3) {
        let a = to_vec(tier.embedding.weight.val().grad(&g_dense).expect("dense"));
        let b = to_vec(tier.embedding.weight.val().grad(&g_routed).expect("routed"));
        assert_close(&a, &b, 1e-5);
    }
}

// ───────────── configuration validation ─────────────

#[test]
fn adaptive_validate_rejects_bad_tier_layouts() {
    let t = AdaptiveTierConfig::new;

    let empty = AdaptiveEmbeddingConfig::new(10, 4, vec![]);
    assert!(empty.validate().unwrap_err().contains("empty"));

    let not_zero = AdaptiveEmbeddingConfig::new(10, 4, vec![t(1, 10, 4)]);
    assert!(not_zero.validate().unwrap_err().contains("from 0"));

    let gap = AdaptiveEmbeddingConfig::new(10, 4, vec![t(0, 4, 4), t(5, 10, 2)]);
    assert!(gap.validate().unwrap_err().contains("gap"));

    let overlap = AdaptiveEmbeddingConfig::new(10, 4, vec![t(0, 6, 4), t(5, 10, 2)]);
    assert!(overlap.validate().unwrap_err().contains("overlap"));

    let short = AdaptiveEmbeddingConfig::new(10, 4, vec![t(0, 8, 4)]);
    assert!(short.validate().unwrap_err().contains("vocab_size"));

    let zero_dim = AdaptiveEmbeddingConfig::new(10, 4, vec![t(0, 10, 0)]);
    assert!(zero_dim.validate().unwrap_err().contains("dim"));

    let empty_range = AdaptiveEmbeddingConfig::new(10, 4, vec![t(0, 0, 4), t(0, 10, 4)]);
    assert!(empty_range.validate().unwrap_err().contains("start"));

    assert!(
        AdaptiveEmbeddingConfig::new(10, 4, vec![t(0, 10, 4)])
            .try_init(&dev())
            .is_ok()
    );
}

#[test]
#[should_panic(expected = "AdaptiveEmbeddingConfig")]
fn adaptive_init_panics_on_invalid_config() {
    let _ = AdaptiveEmbeddingConfig::new(10, 4, vec![]).init(&dev());
}

// ───────────────────────── FourierEmbedding ─────────────────────────

fn randn<const D: usize>(shape: [usize; D]) -> Tensor<D> {
    Tensor::<D>::random(shape, Distribution::Normal(0.0, 1.0), &dev())
}

// ───────────── frequencies ─────────────

#[test]
fn fourier_frequency_scales_produce_expected_values() {
    let log2 = FourierEmbeddingConfig::new(1, 4);
    assert_close(&log2.frequency_values(), &[1.0, 2.0, 4.0, 8.0], 1e-6);

    let linear = FourierEmbeddingConfig::new(1, 4)
        .with_scale(FrequencyScale::Linear)
        .with_max_frequency(4.0);
    assert_close(&linear.frequency_values(), &[1.0, 2.0, 3.0, 4.0], 1e-6);

    let geometric = FourierEmbeddingConfig::new(1, 4)
        .with_scale(FrequencyScale::Geometric)
        .with_max_frequency(8.0);
    assert_close(&geometric.frequency_values(), &[1.0, 2.0, 4.0, 8.0], 1e-5);

    // One frequency is always min, without dividing by F − 1 = 0.
    for scale in [
        FrequencyScale::Linear,
        FrequencyScale::Log2,
        FrequencyScale::Geometric,
    ] {
        let one = FourierEmbeddingConfig::new(1, 1)
            .with_scale(scale)
            .with_min_frequency(3.0);
        assert_close(&one.frequency_values(), &[3.0], 1e-6);
    }
}

#[test]
fn fourier_fourier_dim_follows_documented_formula() {
    assert_eq!(FourierEmbeddingConfig::new(1, 8).fourier_dim(), 16);
    assert_eq!(
        FourierEmbeddingConfig::new(1, 8)
            .with_include_input(true)
            .fourier_dim(),
        17
    );
    assert_eq!(FourierEmbeddingConfig::new(3, 4).fourier_dim(), 24);
    assert_eq!(FourierEmbeddingConfig::new(4096, 16).fourier_dim(), 131_072);
    assert_eq!(
        FourierEmbeddingConfig::new(3, 4)
            .with_output_dim(10)
            .output_width(),
        10
    );
}

// ───────────── shapes and meanings ─────────────

#[test]
fn fourier_output_shapes_for_all_modes() {
    let plain = FourierEmbeddingConfig::new(3, 4).init(&dev());
    assert_eq!(plain.forward(randn([2, 5, 3])).dims(), [2, 5, 24]);
    assert_eq!(plain.forward(randn([5, 3])).dims(), [5, 24]);
    assert_eq!(plain.forward(randn([3])).dims(), [24]);

    let with_input = FourierEmbeddingConfig::new(3, 4)
        .with_include_input(true)
        .init(&dev());
    assert_eq!(with_input.forward(randn([2, 5, 3])).dims(), [2, 5, 27]);

    let projected = FourierEmbeddingConfig::new(3, 4)
        .with_output_dim(10)
        .init(&dev());
    assert_eq!(projected.forward(randn([2, 5, 3])).dims(), [2, 5, 10]);
}

#[test]
fn fourier_output_is_finite_and_deterministic() {
    let fe = FourierEmbeddingConfig::new(2, 6)
        .with_include_input(true)
        .init(&dev());
    let x = randn([4, 7, 2]);
    let a = to_vec(fe.forward(x.clone()));
    let b = to_vec(fe.forward(x));
    assert!(a.iter().all(|v| v.is_finite()));
    assert_eq!(a, b);
}

/// Manual calculation of sin/cos on the host, checking the layout of features.
#[test]
fn fourier_matches_manual_sin_cos_and_layout() {
    let cfg = FourierEmbeddingConfig::new(2, 3)
        .with_scale(FrequencyScale::Linear)
        .with_min_frequency(0.5)
        .with_max_frequency(2.0)
        .with_include_input(true);
    let fe = cfg.init(&dev());
    let freqs = cfg.frequency_values();

    let xs = [0.3_f32, -1.2, 0.75, 2.0]; // 2 points × 2 coordinates
    let x = Tensor::<2>::from_data(TensorData::new(xs.to_vec(), [2, 2]), &dev());
    let got = to_vec(fe.forward(x));

    let mut expected = Vec::new();
    for point in xs.chunks(2) {
        expected.extend_from_slice(point); // include_input
        for &xd in point {
            for &f in &freqs {
                expected.push(((std::f64::consts::TAU * f as f64) * xd as f64).sin() as f32);
            }
            for &f in &freqs {
                expected.push(((std::f64::consts::TAU * f as f64) * xd as f64).cos() as f32);
            }
        }
    }
    assert_eq!(got.len(), 2 * cfg.fourier_dim());
    assert_close(&got, &expected, 1e-4);
}

/// `enc(x) ≈ enc(x + 1/f)` for a single frequency.
#[test]
fn fourier_encoding_is_periodic_in_inverse_frequency() {
    let f = 4.0_f32;
    let fe = FourierEmbeddingConfig::new(1, 1)
        .with_min_frequency(f)
        .init(&dev());

    for x in [0.1_f32, 0.37, -0.8] {
        let a = Tensor::<1>::from_floats([x], &dev());
        let b = Tensor::<1>::from_floats([x + 1.0 / f], &dev());
        assert_close(
            &to_vec(fe.encode_scalars_1d(a)),
            &to_vec(fe.encode_scalars_1d(b)),
            1e-4,
        );
    }
}

#[test]
fn fourier_scalar_helpers_match_feature_axis_forward() {
    let fe = FourierEmbeddingConfig::new(1, 4)
        .with_output_dim(6)
        .init(&dev());

    let v = randn([5]);
    let direct = to_vec(fe.forward(v.clone().reshape([5, 1])));
    assert_close(&to_vec(fe.encode_scalars_1d(v)), &direct, 1e-6);

    let m = randn([2, 3]);
    let direct = to_vec(fe.forward(m.clone().reshape([2, 3, 1])));
    let got = fe.encode_scalars_2d(m);
    assert_eq!(got.dims(), [2, 3, 6]);
    assert_close(&to_vec(got), &direct, 1e-6);
}

#[test]
fn fourier_position_encoding_rows_and_offset() {
    let fe = FourierEmbeddingConfig::new(1, 3).init(&dev());

    let pe = fe.position_encoding(5, &dev());
    assert_eq!(pe.dims(), [5, 6]);

    // Position 0: sin = 0, cos = 1.
    let all = to_vec(pe.clone());
    assert_close(&all[0..3], &[0.0, 0.0, 0.0], 1e-6);
    assert_close(&all[3..6], &[1.0, 1.0, 1.0], 1e-6);

    // Shift: positions 2..5 coincide with rows 2..5 of the full table.
    let tail = to_vec(fe.position_encoding_at(2, 3, &dev()));
    assert_close(&tail, &all[2 * 6..5 * 6], 1e-5);
}

// ───────────── autodiff ─────────────

#[test]
fn fourier_gradients_reach_projection_and_input() {
    let fe = FourierEmbeddingConfig::new(2, 3)
        .with_output_dim(8)
        .init(&adev());
    let x = Tensor::<3>::random([2, 4, 2], Distribution::Normal(0.0, 1.0), &adev()).require_grad();

    let grads = fe.forward(x.clone()).sum().backward();

    let g_proj = to_vec(
        fe.projection
            .as_ref()
            .unwrap()
            .weight
            .val()
            .grad(&grads)
            .expect("projection gradient"),
    );
    assert!(g_proj.iter().any(|v| v.abs() > 0.0));

    let g_x = to_vec(x.grad(&grads).expect("input gradient via sin/cos"));
    assert!(g_x.iter().all(|v| v.is_finite()) && g_x.iter().any(|v| v.abs() > 0.0));
}

// ───────────── validation ─────────────

#[test]
fn fourier_validate_rejects_bad_configs() {
    assert!(
        FourierEmbeddingConfig::new(0, 4)
            .validate()
            .unwrap_err()
            .contains("input_dim")
    );
    assert!(
        FourierEmbeddingConfig::new(1, 0)
            .validate()
            .unwrap_err()
            .contains("num_frequencies")
    );
    assert!(
        FourierEmbeddingConfig::new(1, 4)
            .with_min_frequency(0.0)
            .validate()
            .unwrap_err()
            .contains("min_frequency")
    );
    assert!(
        FourierEmbeddingConfig::new(1, 4)
            .with_scale(FrequencyScale::Geometric)
            .with_min_frequency(8.0)
            .with_max_frequency(1.0)
            .validate()
            .unwrap_err()
            .contains("max_frequency")
    );
    assert!(
        FourierEmbeddingConfig::new(1, 4)
            .with_output_dim(0)
            .validate()
            .unwrap_err()
            .contains("output_dim")
    );
    // 2^200 doesn't fit at f32.
    assert!(
        FourierEmbeddingConfig::new(1, 201)
            .validate()
            .unwrap_err()
            .contains("f32")
    );
    assert!(FourierEmbeddingConfig::new(1, 4).try_init(&dev()).is_ok());
}

#[test]
#[should_panic(expected = "last axis")]
fn fourier_forward_rejects_wrong_last_axis() {
    let fe = FourierEmbeddingConfig::new(3, 2).init(&dev());
    let _ = fe.forward(randn([2, 4]));
}

#[test]
#[should_panic(expected = "input_dim = 1")]
fn fourier_position_encoding_requires_scalar_input() {
    let fe = FourierEmbeddingConfig::new(2, 2).init(&dev());
    let _ = fe.position_encoding(4, &dev());
}
