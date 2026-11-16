//! Tests norm: ScaleNorm, LayerScale, ConditionalRmsNorm, AdaptiveRmsNorm.

use burn::module::Param;
use burn::prelude::*;
use burn::tensor::Distribution;

use crate::norm::*;

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

fn nonzero<const D: usize>(t: Tensor<D>) -> bool {
    v(t).iter().any(|x| x.abs() > 0.0)
}

fn t1(data: Vec<f32>) -> Tensor<1> {
    let n = data.len();
    Tensor::from_data(TensorData::new(data, [n]), &dev())
}

fn t2(data: Vec<f32>, shape: [usize; 2]) -> Tensor<2> {
    Tensor::from_data(TensorData::new(data, shape), &dev())
}

// ───────────────────────── ScaleNorm ─────────────────────────

#[test]
fn scalenorm_shape_preserved() {
    let n = ScaleNormConfig::new(8).init(&dev());
    assert_eq!(n.forward(rnd::<2>([3, 8])).dims(), [3, 8]);
    assert_eq!(n.forward(rnd::<3>([2, 3, 8])).dims(), [2, 3, 8]);
    assert_eq!(n.forward(rnd::<4>([2, 2, 3, 8])).dims(), [2, 2, 3, 8]);
}

#[test]
fn scalenorm_known_numeric_result() {
    // x = [3, 4] → ‖x‖ = 5, g = 2 → [1.2, 1.6]
    let n = ScaleNormConfig::new(2)
        .with_eps(1e-12)
        .with_init_scale(2.0)
        .init(&dev());
    let y = v(n.forward(t2(vec![3.0, 4.0], [1, 2])));
    close(&y, &[1.2, 1.6], 1e-5);
}

#[test]
fn scalenorm_zero_input_is_finite() {
    let n = ScaleNormConfig::new(4).init(&dev());
    let y = v(n.forward(Tensor::<2>::zeros([2, 4], &dev())));
    assert!(y.iter().all(|a| *a == 0.0));
}

#[test]
fn scalenorm_scale_invariance() {
    let n = ScaleNormConfig::new(8).with_eps(1e-12).init(&dev());
    let x = rnd::<3>([2, 3, 8]);
    close(&v(n.forward(x.clone() * 7.5)), &v(n.forward(x)), 1e-4);
}

#[test]
fn scalenorm_gradients() {
    let n = ScaleNormConfig::new(8).init(&adev());
    let x = rnd_on(&adev(), [2, 3, 8]).require_grad();
    // sum(y) for a normalized vector is not constant, the gradients are non-zero
    let w = rnd_on(&adev(), [2, 3, 8]);
    let grads = (n.forward(x.clone()) * w).sum().backward();
    assert!(x.grad(&grads).unwrap().abs().sum().into_scalar::<f32>() > 0.0);
    assert!(
        n.scale
            .val()
            .grad(&grads)
            .unwrap()
            .abs()
            .sum()
            .into_scalar::<f32>()
            > 0.0
    );
}

#[test]
fn scalenorm_parameter_count() {
    assert_eq!(ScaleNormConfig::new(512).parameter_count(), 1);
    let n = ScaleNormConfig::new(512).init(&dev());
    assert_eq!(n.parameter_count(), 1);
    assert_eq!(n.scale.dims(), [1]);
}

#[test]
fn scalenorm_validate() {
    assert!(ScaleNormConfig::new(0).validate().is_err());
    assert!(ScaleNormConfig::new(4).with_eps(0.0).validate().is_err());
    assert!(
        ScaleNormConfig::new(4)
            .with_init_scale(f32::NAN)
            .validate()
            .is_err()
    );
}

#[test]
#[should_panic]
fn scalenorm_rejects_wrong_last_dim() {
    ScaleNormConfig::new(8)
        .init(&dev())
        .forward(rnd::<2>([2, 7]));
}

// ───────────────────────── LayerScale ─────────────────────────

#[test]
fn layerscale_shape_and_broadcast() {
    let l = LayerScaleConfig::new(8).init(&dev());
    assert_eq!(l.forward(rnd::<2>([3, 8])).dims(), [3, 8]);
    assert_eq!(l.forward(rnd::<3>([2, 3, 8])).dims(), [2, 3, 8]);
}

#[test]
fn layerscale_init_one_is_identity() {
    let l = LayerScaleConfig::new(8).init(&dev());
    let x = rnd::<3>([2, 3, 8]);
    close(&v(l.forward(x.clone())), &v(x), 0.0);
}

#[test]
fn layerscale_init_zero_is_zero() {
    let l = LayerScaleConfig::new(8).with_init_value(0.0).init(&dev());
    assert!(!nonzero(l.forward(rnd::<3>([2, 3, 8]))));
}

#[test]
fn layerscale_known_gamma_result() {
    let mut l = LayerScaleConfig::new(3).init(&dev());
    l.gamma = Param::from_tensor(t1(vec![2.0, -1.0, 0.5]));
    let y = v(l.forward(t2(vec![1.0, 2.0, 4.0, -1.0, -2.0, -4.0], [2, 3])));
    close(&y, &[2.0, -2.0, 2.0, -2.0, 2.0, -2.0], 1e-6);
}

#[test]
fn layerscale_gradients() {
    let l = LayerScaleConfig::new(8).with_init_value(0.1).init(&adev());
    let x = rnd_on(&adev(), [2, 3, 8]).require_grad();
    let grads = l.forward(x.clone()).sum().backward();
    assert!(x.grad(&grads).is_some());
    assert!(l.gamma.val().grad(&grads).is_some());
}

#[test]
fn layerscale_parameter_count_and_validate() {
    assert_eq!(LayerScaleConfig::new(64).parameter_count(), 64);
    assert_eq!(LayerScaleConfig::new(64).init(&dev()).parameter_count(), 64);
    assert!(LayerScaleConfig::new(0).validate().is_err());
    assert!(
        LayerScaleConfig::new(4)
            .with_init_value(f32::INFINITY)
            .validate()
            .is_err()
    );
}

// ───────────────────── ConditionalRmsNorm ─────────────────────

fn cond() -> ConditionalRmsNorm {
    ConditionalRmsNormConfig::new(8).init(&dev())
}

#[test]
fn conditional_zero_modulation_equals_rmsnorm() {
    let c = cond();
    let x = rnd::<3>([2, 4, 8]);
    let z = Tensor::<3>::zeros([2, 1, 8], &dev());
    close(
        &v(c.forward_with_modulation(x.clone(), z.clone(), z)),
        &v(c.norm.forward(x)),
        1e-6,
    );
}

#[test]
fn conditional_known_scale_and_shift() {
    let c = cond();
    let x = rnd::<2>([3, 8]);
    let n = c.norm.forward(x.clone());
    let scale = Tensor::<2>::full([3, 8], 1.0, &dev());
    let shift = Tensor::<2>::full([3, 8], 0.5, &dev());
    close(
        &v(c.forward_with_modulation(x, scale, shift)),
        &v(n * 2.0 + 0.5),
        1e-5,
    );
}

#[test]
fn conditional_broadcast_batch_condition() {
    let c = cond();
    let x = rnd::<3>([2, 4, 8]);
    let sc = rnd::<3>([2, 1, 8]);
    let sh = rnd::<3>([2, 1, 8]);
    let y = c.forward_with_modulation(x.clone(), sc.clone(), sh.clone());
    assert_eq!(y.dims(), [2, 4, 8]);
    // same as explicit stretching along T
    let y_full = c.forward_with_modulation(x, sc.expand([2, 4, 8]), sh.expand([2, 4, 8]));
    close(&v(y), &v(y_full), 1e-6);
}

#[test]
fn conditional_per_token() {
    let c = cond();
    let x = rnd::<3>([2, 4, 8]);
    let sc = rnd::<3>([2, 4, 8]);
    let sh = rnd::<3>([2, 4, 8]);
    let y = v(c.forward_with_modulation(x.clone(), sc.clone(), sh.clone()));
    let expected = v(c.norm.forward(x) * (sc + 1.0) + sh);
    close(&y, &expected, 1e-5);
}

#[test]
fn conditional_gradients() {
    let c = ConditionalRmsNormConfig::new(8).init(&adev());
    let x = rnd_on(&adev(), [2, 4, 8]).require_grad();
    let sc = rnd_on(&adev(), [2, 1, 8]).require_grad();
    let sh = rnd_on(&adev(), [2, 1, 8]).require_grad();
    let w = rnd_on(&adev(), [2, 4, 8]);
    let grads = (c.forward_with_modulation(x.clone(), sc.clone(), sh.clone()) * w)
        .sum()
        .backward();
    assert!(x.grad(&grads).is_some());
    assert!(sc.grad(&grads).is_some());
    assert!(sh.grad(&grads).is_some());
}

#[test]
fn conditional_finite_on_zero_input() {
    let c = cond();
    let x = Tensor::<3>::zeros([2, 4, 8], &dev());
    let y = v(c.forward_with_modulation(x, rnd::<3>([2, 1, 8]), rnd::<3>([2, 1, 8])));
    assert!(y.iter().all(|a| a.is_finite()));
}

#[test]
fn conditional_parameter_count_is_norm_weight_only() {
    assert_eq!(ConditionalRmsNormConfig::new(8).parameter_count(), 8);
}

#[test]
#[should_panic]
fn conditional_rejects_wrong_modulation_dim() {
    cond().forward_with_modulation(
        rnd::<3>([2, 4, 8]),
        rnd::<3>([2, 1, 7]),
        rnd::<3>([2, 1, 8]),
    );
}

// ───────────────────── AdaptiveRmsNorm ─────────────────────

fn ada_on(device: &Device, use_shift: bool, zero_init: bool) -> AdaptiveRmsNorm {
    AdaptiveRmsNormConfig::new(8, 5)
        .with_use_shift(use_shift)
        .with_zero_init(zero_init)
        .init(device)
}

fn ada(use_shift: bool, zero_init: bool) -> AdaptiveRmsNorm {
    ada_on(&dev(), use_shift, zero_init)
}

/// Non-zero projection so that the condition actually has an effect.
fn trained(a: &mut AdaptiveRmsNorm) {
    let [i, o] = a.modulation.weight.dims();
    a.modulation.weight = Param::from_tensor(rnd_on(&a.modulation.weight.device(), [i, o]));
}

#[test]
fn adaptive_zero_init_preserves_rmsnorm_for_any_condition() {
    let a = ada(true, true);
    let x = rnd::<3>([2, 4, 8]);
    let expected = v(a.conditional.norm.forward(x.clone()));
    for _ in 0..2 {
        let c = rnd::<2>([2, 5]);
        close(&v(a.forward_global(x.clone(), c)), &expected, 1e-6);
    }
}

#[test]
fn adaptive_condition_changes_output_after_training() {
    let mut a = ada(true, true);
    trained(&mut a);
    let x = rnd::<3>([2, 4, 8]);
    let ca = rnd::<2>([2, 5]);
    let cb = rnd::<2>([2, 5]);
    let ya = v(a.forward_global(x.clone(), ca.clone()));
    let yb = v(a.forward_global(x.clone(), cb));
    assert!(ya.iter().zip(&yb).any(|(p, q)| (p - q).abs() > 1e-4));
    // same condition → same result
    close(&ya, &v(a.forward_global(x, ca)), 0.0);
}

#[test]
fn adaptive_matches_manual_reference() {
    let mut a = ada(true, true);
    trained(&mut a);
    let x = rnd::<3>([2, 4, 8]);
    let c = rnd::<2>([2, 5]);

    let m = a.modulation.forward(c.clone()); // [2, 16]
    let gamma = m.clone().narrow(1, 0, 8).reshape([2, 1, 8]);
    let beta = m.narrow(1, 8, 8).reshape([2, 1, 8]);
    let expected = a.conditional.norm.forward(x.clone()) * (gamma + 1.0) + beta;
    close(&v(a.forward_global(x, c)), &v(expected), 1e-4);
}

#[test]
fn adaptive_global_equals_explicit_broadcast_and_per_token() {
    let mut a = ada(true, true);
    trained(&mut a);
    let x = rnd::<3>([2, 4, 8]);
    let c = rnd::<2>([2, 5]);
    let y_global = v(a.forward_global(x.clone(), c.clone()));

    let c3 = c.clone().reshape([2, 1, 5]);
    close(&y_global, &v(a.forward(x.clone(), c3)), 1e-6);

    // per-token with the same condition on all positions = global
    let c_tok = c.reshape([2, 1, 5]).expand([2, 4, 5]);
    close(&y_global, &v(a.forward(x, c_tok)), 1e-5);
}

#[test]
fn adaptive_per_token_differs_across_tokens() {
    let mut a = ada(true, true);
    trained(&mut a);
    let x = Tensor::<3>::ones([1, 2, 8], &dev());
    let c = rnd::<3>([1, 2, 5]);
    let y = v(a.forward(x, c));
    assert!(
        y[0..8]
            .iter()
            .zip(&y[8..16])
            .any(|(p, q)| (p - q).abs() > 1e-4)
    );
}

#[test]
fn adaptive_without_shift() {
    let mut a = ada(false, true);
    assert_eq!(a.modulation.weight.dims(), [5, 8]);
    trained(&mut a);
    let x = rnd::<3>([2, 4, 8]);
    let c = rnd::<2>([2, 5]);
    let g = a.modulation.forward(c.clone()).reshape([2, 1, 8]);
    let expected = a.conditional.norm.forward(x.clone()) * (g + 1.0);
    close(&v(a.forward_global(x, c)), &v(expected), 1e-4);
}

#[test]
fn adaptive_2d_input() {
    let mut a = ada(true, true);
    trained(&mut a);
    let y = a.forward_global(rnd::<2>([3, 8]), rnd::<2>([3, 5]));
    assert_eq!(y.dims(), [3, 8]);
}

#[test]
fn adaptive_gradients() {
    let mut a = ada_on(&adev(), true, true);
    trained(&mut a);
    let x = rnd_on(&adev(), [2, 4, 8]).require_grad();
    let c = rnd_on(&adev(), [2, 5]).require_grad();
    let w = rnd_on(&adev(), [2, 4, 8]);
    let grads = (a.forward_global(x.clone(), c.clone()) * w)
        .sum()
        .backward();
    assert!(x.grad(&grads).is_some());
    assert!(c.grad(&grads).is_some());
    assert!(a.modulation.weight.val().grad(&grads).is_some());
    assert!(
        a.modulation
            .bias
            .as_ref()
            .unwrap()
            .val()
            .grad(&grads)
            .is_some()
    );
    assert!(a.conditional.norm.gamma.val().grad(&grads).is_some());
}

#[test]
fn adaptive_zero_init_projection_still_gets_gradient() {
    let a = ada_on(&adev(), true, true);
    let x = rnd_on(&adev(), [2, 4, 8]);
    let c = rnd_on(&adev(), [2, 5]);
    let w = rnd_on(&adev(), [2, 4, 8]);
    let grads = (a.forward_global(x, c) * w).sum().backward();
    let g = a.modulation.weight.val().grad(&grads).unwrap();
    assert!(g.abs().sum().into_scalar::<f32>() > 0.0);
}

#[test]
fn adaptive_no_nan_with_zero_input() {
    let mut a = ada(true, false);
    trained(&mut a);
    let y = v(a.forward_global(Tensor::<3>::zeros([2, 4, 8], &dev()), rnd::<2>([2, 5])));
    assert!(y.iter().all(|p| p.is_finite()));
}

#[test]
fn adaptive_parameter_count_and_validate() {
    // norm 8 + weight 5·16 + bias 16
    let c = AdaptiveRmsNormConfig::new(8, 5);
    assert_eq!(c.parameter_count(), 8 + 5 * 16 + 16);
    assert_eq!(
        AdaptiveRmsNormConfig::new(8, 5)
            .with_use_shift(false)
            .parameter_count(),
        8 + 5 * 8 + 8
    );
    assert!(AdaptiveRmsNormConfig::new(8, 0).validate().is_err());
    assert!(AdaptiveRmsNormConfig::new(0, 5).validate().is_err());
    assert!(
        AdaptiveRmsNormConfig::new(8, 5)
            .with_eps(-1.0)
            .validate()
            .is_err()
    );
}

#[test]
#[should_panic]
fn adaptive_rejects_wrong_condition_dim() {
    ada(true, true).forward_global(rnd::<3>([2, 4, 8]), rnd::<2>([2, 4]));
}
