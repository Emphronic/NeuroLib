//! Tests of FFN layers: GEGLU, ReGLU, universal GatedFfn and MoE.

use burn::nn::Linear;
use burn::prelude::*;
use burn::tensor::activation::{gelu, relu, sigmoid, silu, softmax};
use burn::tensor::{Device, Distribution};

use crate::ffn::*;

// ───────────────────────── auxiliary ─────────────────────────

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

fn idx_vec(t: Tensor<2, Int>) -> Vec<i64> {
    t.into_data().convert::<i64>().try_to_vec::<i64>().unwrap()
}

fn assert_close(got: &[f32], expected: &[f32], tol: f32) {
    assert_eq!(got.len(), expected.len(), "different lengths");
    for (i, (g, e)) in got.iter().zip(expected).enumerate() {
        assert!((g - e).abs() < tol, "[{i}] got {g}, expected {e}");
    }
}

/// The parameters Burn are initialized lazily: `clone()` before the first call gives
/// a copy with INDEPENDENTLY initialized weights. Before copying layers
/// (to collect the “same” layer of a different type) you need to materialize the weights.
fn materialize(l: &Linear) {
    let _ = l.weight.val();
    if let Some(b) = &l.bias {
        let _ = b.val();
    }
}

// ───────────────────────── GateActivation ─────────────────────────

#[test]
fn activations_match_known_values() {
    let x = Tensor::<2>::from_floats([[-1.0, 0.0, 2.0]], &dev());

    assert_close(
        &to_vec(GateActivation::Relu.apply(x.clone())),
        &[0.0, 0.0, 2.0],
        1e-6,
    );
    // sigmoid(-1), sigmoid(0), sigmoid(2)
    assert_close(
        &to_vec(GateActivation::Sigmoid.apply(x.clone())),
        &[0.268_941_4, 0.5, 0.880_797_1],
        1e-4,
    );
    assert_close(
        &to_vec(GateActivation::Tanh.apply(x.clone())),
        &[-0.761_594_2, 0.0, 0.964_027_6],
        1e-4,
    );
    // silu(x) = x · sigmoid(x)
    assert_close(
        &to_vec(GateActivation::Silu.apply(x.clone())),
        &[-0.268_941_4, 0.0, 1.761_594_2],
        1e-4,
    );
    // gelu(x) = x · Φ(x)
    assert_close(
        &to_vec(GateActivation::Gelu.apply(x)),
        &[-0.158_655_3, 0.0, 1.954_499_7],
        1e-3,
    );
}

// ───────────────────────── GEGLU ─────────────────────────

#[test]
fn geglu_keeps_shape_for_3d_input() {
    let ffn = GegluConfig::new(8, 20).init(&dev());
    assert_eq!(ffn.forward(randn([2, 5, 8])).dims(), [2, 5, 8]);
}

#[test]
fn geglu_keeps_shape_for_2d_input() {
    let ffn = GegluConfig::new(8, 20).init(&dev());
    assert_eq!(ffn.forward(randn([3, 8])).dims(), [3, 8]);
}

/// The result should match the manual calculation using the formula.
#[test]
fn geglu_matches_manual_formula() {
    let ffn = GegluConfig::new(6, 12).init(&dev());
    let x = randn([4, 6]);

    let a = ffn.linear_a.forward(x.clone());
    let b = ffn.linear_b.forward(x.clone());
    let expected = ffn.linear_out.forward(gelu(a).mul(b));

    assert_close(&to_vec(ffn.forward(x)), &to_vec(expected), 1e-6);
}

/// Zero input and bias = false give zero output.
#[test]
fn geglu_zero_input_without_bias_gives_zero() {
    let ffn = GegluConfig::new(6, 12).with_bias(false).init(&dev());
    let x = Tensor::<2>::zeros([3, 6], &dev());
    assert!(to_vec(ffn.forward(x)).iter().all(|v| v.abs() < 1e-7));
}

/// Running a batch is equal to running each row separately.
#[test]
fn geglu_rows_are_independent() {
    let ffn = GegluConfig::new(6, 12).init(&dev());
    let x = randn([3, 6]);

    let full = to_vec(ffn.forward(x.clone()));
    for i in 0..3 {
        let single = to_vec(ffn.forward(x.clone().slice([i..i + 1, 0..6])));
        assert_close(&full[i * 6..(i + 1) * 6], &single, 1e-5);
    }
}

#[test]
fn geglu_param_parity_hidden_size() {
    let cfg = GegluConfig::with_param_parity(512, 2048);
    // 2/3 · 2048 = 1365.33 → 1365
    assert_eq!(cfg.d_hidden, 1365);
    assert!(cfg.bias);
}

// ───────────────────────── ReGLU ─────────────────────────

#[test]
fn reglu_keeps_shape() {
    let ffn = RegluConfig::new(8, 20).init(&dev());
    assert_eq!(ffn.forward(randn([2, 5, 8])).dims(), [2, 5, 8]);
    assert_eq!(ffn.forward(randn([3, 8])).dims(), [3, 8]);
}

#[test]
fn reglu_matches_manual_formula() {
    let ffn = RegluConfig::new(6, 12).init(&dev());
    let x = randn([4, 6]);

    let a = ffn.linear_a.forward(x.clone());
    let b = ffn.linear_b.forward(x.clone());
    let expected = ffn.linear_out.forward(relu(a).mul(b));

    assert_close(&to_vec(ffn.forward(x)), &to_vec(expected), 1e-6);
}

#[test]
fn reglu_zero_input_without_bias_gives_zero() {
    let ffn = RegluConfig::new(6, 12).with_bias(false).init(&dev());
    let x = Tensor::<2>::zeros([3, 6], &dev());
    assert!(to_vec(ffn.forward(x)).iter().all(|v| v.abs() < 1e-7));
}

#[test]
fn reglu_param_parity_hidden_size() {
    assert_eq!(RegluConfig::with_param_parity(512, 2048).d_hidden, 1365);
}

// ───────────────────────── GatedFfn ─────────────────────────

#[test]
fn gated_matches_manual_formula_for_every_activation() {
    // Independent “reference” activation for each option.
    let cases: Vec<(GateActivation, fn(Tensor<2>) -> Tensor<2>)> = vec![
        (GateActivation::Gelu, |t| gelu(t)),
        (GateActivation::Relu, |t| relu(t)),
        (GateActivation::Silu, |t| silu(t)),
        (GateActivation::Sigmoid, |t| sigmoid(t)),
        (GateActivation::Tanh, |t| t.tanh()),
    ];

    for (act, reference) in cases {
        let ffn = GatedFfnConfig::new(6, 12, act.clone()).init(&dev());
        let x = randn([4, 6]);

        let gate = reference(ffn.gate.forward(x.clone()));
        let value = ffn.value.forward(x.clone());
        let expected = ffn.out.forward(gate.mul(value));

        assert_close(&to_vec(ffn.forward(x)), &to_vec(expected), 1e-6);
        assert_eq!(ffn.activation, act);
    }
}

#[test]
fn gated_keeps_shape_for_any_rank() {
    let ffn = GatedFfnConfig::swiglu(8, 20).init(&dev());
    assert_eq!(ffn.forward(randn([8])).dims(), [8]);
    assert_eq!(ffn.forward(randn([3, 8])).dims(), [3, 8]);
    assert_eq!(ffn.forward(randn([2, 3, 8])).dims(), [2, 3, 8]);
    assert_eq!(ffn.forward(randn([2, 3, 4, 8])).dims(), [2, 3, 4, 8]);
}

/// GatedFfn(Gelu) with the same weights is exactly GEGLU.
#[test]
fn gated_gelu_equals_geglu() {
    let geglu = GegluConfig::new(6, 12).init(&dev());
    materialize(&geglu.linear_a);
    materialize(&geglu.linear_b);
    materialize(&geglu.linear_out);
    let gated = GatedFfn {
        gate: geglu.linear_a.clone(),
        value: geglu.linear_b.clone(),
        out: geglu.linear_out.clone(),
        activation: GateActivation::Gelu,
    };

    let x = randn([5, 6]);
    assert_close(
        &to_vec(gated.forward(x.clone())),
        &to_vec(geglu.forward(x)),
        1e-6,
    );
}

/// GatedFfn(Relu) with the same weights is exactly ReGLU.
#[test]
fn gated_relu_equals_reglu() {
    let reglu = RegluConfig::new(6, 12).init(&dev());
    materialize(&reglu.linear_a);
    materialize(&reglu.linear_b);
    materialize(&reglu.linear_out);
    let gated = GatedFfn {
        gate: reglu.linear_a.clone(),
        value: reglu.linear_b.clone(),
        out: reglu.linear_out.clone(),
        activation: GateActivation::Relu,
    };

    let x = randn([5, 6]);
    assert_close(
        &to_vec(gated.forward(x.clone())),
        &to_vec(reglu.forward(x)),
        1e-6,
    );
}

#[test]
fn gated_zero_input_without_bias_gives_zero_for_every_activation() {
    for act in [
        GateActivation::Gelu,
        GateActivation::Relu,
        GateActivation::Silu,
        GateActivation::Sigmoid,
        GateActivation::Tanh,
    ] {
        let ffn = GatedFfnConfig::new(6, 12, act)
            .with_bias(false)
            .init(&dev());
        let x = Tensor::<2>::zeros([3, 6], &dev());
        assert!(to_vec(ffn.forward(x)).iter().all(|v| v.abs() < 1e-7));
    }
}

#[test]
fn gated_named_constructors_pick_activation() {
    assert_eq!(GatedFfnConfig::geglu(4, 8).activation, GateActivation::Gelu);
    assert_eq!(GatedFfnConfig::reglu(4, 8).activation, GateActivation::Relu);
    assert_eq!(
        GatedFfnConfig::swiglu(4, 8).activation,
        GateActivation::Silu
    );
    assert!(GatedFfnConfig::swiglu(4, 8).bias);
}

#[test]
fn gated_param_parity_hidden_size() {
    let cfg = GatedFfnConfig::with_param_parity(512, 2048, GateActivation::Silu);
    assert_eq!(cfg.d_hidden, 1365);
}

// ───────────────────────── MoE ─────────────────────────

const D_MODEL: usize = 8;

fn make_moe(num_experts: usize, top_k: usize) -> Moe {
    MoeConfig::new(D_MODEL, 16, num_experts)
        .with_top_k(top_k)
        .init(&dev())
}

#[test]
fn moe_keeps_shape_for_any_rank() {
    let moe = make_moe(4, 2);
    assert_eq!(moe.forward(randn([2, 5, D_MODEL])).dims(), [2, 5, D_MODEL]);
    assert_eq!(moe.forward(randn([7, D_MODEL])).dims(), [7, D_MODEL]);
}

#[test]
fn moe_builds_requested_number_of_experts() {
    let moe = make_moe(5, 2);
    assert_eq!(moe.num_experts(), 5);
    assert_eq!(moe.experts.len(), 5);
    assert_eq!(moe.top_k, 2);
}

/// The sparse passage must match the dense reference.
#[test]
fn moe_sparse_matches_dense_reference() {
    // (num_experts, top_k)
    for (e, k) in [(4, 2), (8, 1), (5, 5), (3, 2), (6, 3)] {
        let moe = make_moe(e, k);
        let x = randn([2, 7, D_MODEL]);

        let sparse = to_vec(moe.forward(x.clone()));
        let dense = to_vec(moe.forward_dense(x));
        assert_close(&sparse, &dense, 1e-4);
    }
}

/// The weights of the selected experts are probabilities: the sum per token is 1,
/// indices lie in a range and are not repeated.
#[test]
fn moe_route_gives_normalised_weights_and_distinct_experts() {
    let (e, k) = (6, 3);
    let moe = make_moe(e, k);
    let tokens = 10;

    let (_probs, weights, indices) = moe.route(randn([tokens, D_MODEL]));
    assert_eq!(weights.dims(), [tokens, k]);
    assert_eq!(indices.dims(), [tokens, k]);

    let w = to_vec(weights);
    let idx = idx_vec(indices);
    for t in 0..tokens {
        let row_w = &w[t * k..(t + 1) * k];
        let row_i = &idx[t * k..(t + 1) * k];

        assert!((row_w.iter().sum::<f32>() - 1.0).abs() < 1e-5, "token {t}");
        assert!(row_w.iter().all(|&v| v > 0.0));
        assert!(row_i.iter().all(|&i| (0..e as i64).contains(&i)));

        let mut sorted = row_i.to_vec();
        sorted.sort_unstable();
        sorted.dedup();
        assert_eq!(sorted.len(), k, "token {t}: experts repeat themselves");
    }
}

/// At `top_k = num_experts` all experts are selected and
/// `y = Σ softmax(W_r·x)_e · Expert_e(x)`.
#[test]
fn moe_matches_manual_weighted_sum() {
    let moe = make_moe(2, 2);
    let x = randn([5, D_MODEL]);

    let w = softmax(moe.router.forward(x.clone()), 1);
    let w0 = w.clone().slice([0..5, 0..1]);
    let w1 = w.slice([0..5, 1..2]);
    let expected = moe.experts[0]
        .forward(x.clone())
        .mul(w0)
        .add(moe.experts[1].forward(x.clone()).mul(w1));

    assert_close(&to_vec(moe.forward(x)), &to_vec(expected), 1e-5);
}

/// One expert and `top_k = 1`: MoE turns into a regular FFN.
#[test]
fn moe_with_single_expert_is_plain_ffn() {
    let moe = make_moe(1, 1);
    let x = randn([4, D_MODEL]);

    let got = to_vec(moe.forward(x.clone()));
    let expected = to_vec(moe.experts[0].forward(x));
    assert_close(&got, &expected, 1e-5);
}

/// When `top_k = 1` the weight is 1, and the entire token goes to one expert -
/// the one with the maximum logit.
#[test]
fn moe_top1_sends_token_to_argmax_expert() {
    let (e, tokens) = (4, 6);
    let moe = make_moe(e, 1);
    let x = randn([tokens, D_MODEL]);

    let logits = to_vec(moe.router.forward(x.clone()));
    let got = to_vec(moe.forward(x.clone()));

    for t in 0..tokens {
        let row = &logits[t * e..(t + 1) * e];
        let best = row
            .iter()
            .enumerate()
            .max_by(|a, b| a.1.partial_cmp(b.1).unwrap())
            .map(|(i, _)| i)
            .unwrap();

        let x_t = x.clone().slice([t..t + 1, 0..D_MODEL]);
        let expected = to_vec(moe.experts[best].forward(x_t));
        assert_close(&got[t * D_MODEL..(t + 1) * D_MODEL], &expected, 1e-5);
    }
}

/// Tokens are processed independently: a batch is equal to a row-by-row run.
#[test]
fn moe_tokens_are_independent() {
    let moe = make_moe(4, 2);
    let tokens = 5;
    let x = randn([tokens, D_MODEL]);

    let full = to_vec(moe.forward(x.clone()));
    for t in 0..tokens {
        let single = to_vec(moe.forward(x.clone().slice([t..t + 1, 0..D_MODEL])));
        assert_close(&full[t * D_MODEL..(t + 1) * D_MODEL], &single, 1e-4);
    }
}

/// With one expert, the load is obviously “ideal”: the loss is 1.
#[test]
fn moe_aux_loss_is_one_for_single_expert() {
    let moe = make_moe(1, 1);
    let (_y, stats) = moe.forward_with_aux(randn([6, D_MODEL]));
    let aux = stats.load_balance_loss();
    assert_eq!(aux.dims(), [1]);
    assert!((to_vec(aux)[0] - 1.0).abs() < 1e-5);
}

/// The loss is finite, positive and does not exceed the number of experts
/// (with `f` and `P` - distributions above `N` it cannot rise).
#[test]
fn moe_aux_loss_is_bounded() {
    let e = 4;
    let moe = make_moe(e, 2);
    let (y, stats) = moe.forward_with_aux(randn([3, 6, D_MODEL]));
    assert_eq!(y.dims(), [3, 6, D_MODEL]);

    let aux = to_vec(stats.load_balance_loss())[0];
    assert!(aux.is_finite());
    assert!(aux > 0.0 && aux <= e as f32 + 1e-4, "aux = {aux}");
}

/// The gradient reaches the router and at least some of the experts.
#[test]
fn moe_gradients_reach_router_and_experts() {
    let device = adev();
    let moe = MoeConfig::new(D_MODEL, 16, 4).with_top_k(2).init(&device);
    let x = Tensor::<3>::random([2, 6, D_MODEL], Distribution::Normal(0.0, 1.0), &device);

    let (y, stats) = moe.forward_with_aux(x);
    let loss = y.sum().add(stats.load_balance_loss().mul_scalar(0.01));
    let grads = loss.backward();

    assert!(
        moe.router.weight.val().grad(&grads).is_some(),
        "the router did not receive the gradient"
    );
    assert!(
        moe.experts
            .iter()
            .any(|ex| ex.out.weight.val().grad(&grads).is_some()),
        "no expert received the gradient"
    );
}

#[test]
#[should_panic(expected = "top_k")]
fn moe_rejects_zero_top_k() {
    let _ = MoeConfig::new(D_MODEL, 16, 4).with_top_k(0).init(&dev());
}

#[test]
#[should_panic(expected = "top_k")]
fn moe_rejects_top_k_above_num_experts() {
    let _ = MoeConfig::new(D_MODEL, 16, 4).with_top_k(5).init(&dev());
}

#[test]
#[should_panic(expected = "num_experts")]
fn moe_rejects_zero_experts() {
    let _ = MoeConfig::new(D_MODEL, 16, 0).init(&dev());
}
