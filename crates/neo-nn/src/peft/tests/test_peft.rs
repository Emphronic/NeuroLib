//! PEFT tests: LoRA, DoRA, IA³.

use burn::nn::LinearConfig;
use burn::prelude::*;
use burn::tensor::Distribution;

use crate::peft::*;

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
    assert_eq!(a.len(), b.len());
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

fn lin(i: usize, o: usize, bias: bool) -> burn::nn::Linear {
    let l: burn::nn::Linear = LinearConfig::new(i, o).with_bias(bias).init(&dev());
    // Burn initializes the weights lazily: force it, otherwise clone() will give different values.
    let _ = l.weight.val();
    let _ = l.bias.as_ref().map(|b| b.val());
    l
}

fn nonzero(t: Tensor<2>) -> bool {
    v(t).iter().any(|x| x.abs() > 0.0)
}

// ───────────────────────── LoRA ─────────────────────────

fn lora_cfg() -> LoraConfig {
    LoraConfig::new(8, 6, 4).with_alpha(8.0)
}

/// Substitute a non-zero B to make the adapter do something.
fn lora_trained_on(device: &Device) -> LoraLinear {
    let mut l = lora_cfg().init(device);
    l.lora_b.weight = burn::module::Param::from_tensor(rnd_on(device, [4, 6]));
    l
}

fn lora_trained() -> LoraLinear {
    lora_trained_on(&dev())
}

#[test]
fn lora_shapes() {
    let l = lora_cfg().init(&dev());
    assert_eq!(l.lora_a.weight.dims(), [8, 4]);
    assert_eq!(l.lora_b.weight.dims(), [4, 6]);
    assert_eq!(l.base.weight.dims(), [8, 6]);
}

#[test]
fn lora_zero_init_equals_base() {
    let base = lin(8, 6, true);
    let l = lora_cfg().init_from_linear(base.clone());
    assert!(!nonzero(l.lora_b.weight.val()));
    let x = rnd::<3>([2, 5, 8]);
    close(&v(l.forward(x.clone())), &v(base.forward(x)), 1e-6);
}

#[test]
fn lora_scaling() {
    let c = LoraConfig::new(4, 4, 16).with_alpha(32.0);
    assert_eq!(c.scaling(), 2.0);
}

#[test]
fn lora_rank_polymorphic() {
    let l = lora_trained();
    assert_eq!(l.forward(rnd::<2>([3, 8])).dims(), [3, 6]);
    assert_eq!(l.forward(rnd::<4>([2, 2, 3, 8])).dims(), [2, 2, 3, 6]);
}

#[test]
fn lora_matches_manual_reference() {
    let l = lora_trained();
    let x = rnd::<2>([3, 8]);
    let expected = x.clone().matmul(l.base.weight())
        + l.base.bias().unwrap().unsqueeze::<2>()
        + x.clone()
            .matmul(l.lora_a.weight.val())
            .matmul(l.lora_b.weight.val())
            .mul_scalar(2.0);
    close(&v(l.forward(x)), &v(expected), 1e-4);
}

#[test]
fn lora_gradient_isolation() {
    let l = lora_trained_on(&adev());
    let x = rnd_on(&adev(), [2, 3, 8]).require_grad();
    let grads = l.forward(x.clone()).sum().backward();

    assert!(l.base.weight.val().grad(&grads).is_none());
    assert!(l.base.bias.as_ref().unwrap().val().grad(&grads).is_none());
    assert!(l.lora_a.weight.val().grad(&grads).is_some());
    assert!(l.lora_b.weight.val().grad(&grads).is_some());
    assert!(x.grad(&grads).is_some());
}

#[test]
fn lora_merge_unmerge() {
    let mut l = lora_trained();
    let x = rnd::<3>([2, 3, 8]);
    let before = v(l.forward(x.clone()));

    l.merge();
    l.merge(); // idempotent
    assert!(l.is_merged() && !l.is_active());
    close(&v(l.forward(x.clone())), &before, 1e-4);

    l.unmerge();
    l.unmerge();
    assert!(!l.is_merged() && l.is_active());
    close(&v(l.forward(x)), &before, 0.0);
}

#[test]
fn lora_parameter_count() {
    let c = LoraConfig::new(4096, 4096, 16);
    assert_eq!(c.parameter_count(), 131_072);
    assert!(c.parameter_count() < 4096 * 4096);
    let l = lora_cfg().init(&dev());
    assert_eq!(l.adapter_parameter_count(), l_count());
    fn l_count() -> usize {
        8 * 4 + 4 * 6
    }
}

#[test]
#[should_panic]
fn lora_rejects_zero_rank() {
    LoraConfig::new(4, 4, 0).init(&dev());
}

// ───────────────────────── DoRA ─────────────────────────

fn dora_cfg() -> DoraConfig {
    DoraConfig::new(8, 6, 4).with_alpha(8.0)
}

fn dora_trained_on(device: &Device) -> DoraLinear {
    let mut l = dora_cfg().init(device);
    l.lora_b.weight = burn::module::Param::from_tensor(rnd_on(device, [4, 6]));
    l
}

fn dora_trained() -> DoraLinear {
    dora_trained_on(&dev())
}

#[test]
fn dora_initial_output_equals_base() {
    let base = lin(8, 6, true);
    let l = dora_cfg().init_from_linear(base.clone());
    let x = rnd::<3>([2, 5, 8]);
    close(&v(l.forward(x.clone())), &v(base.forward(x)), 1e-5);
}

#[test]
fn dora_initial_magnitude_is_row_norms() {
    let base = lin(8, 6, true);
    let w = v(base.weight.val());
    let l = dora_cfg().init_from_linear(base);
    let expected: Vec<f32> = (0..6)
        .map(|o| ((0..8).map(|i| w[i * 6 + o].powi(2)).sum::<f32>() + 1e-6).sqrt())
        .collect();
    close(&v(l.magnitude.val()), &expected, 1e-5);
}

#[test]
fn dora_matches_explicit_weight() {
    let l = dora_trained();
    let x = rnd::<2>([3, 8]);
    let w = l.merged_weight();
    let expected = x.clone().matmul(w) + l.base.bias().unwrap().unsqueeze::<2>();
    close(&v(l.forward(x)), &v(expected), 1e-4);
}

#[test]
fn dora_gradients() {
    let l = dora_trained_on(&adev());
    let x = rnd_on(&adev(), [2, 3, 8]).require_grad();
    let grads = l.forward(x.clone()).sum().backward();

    assert!(l.base.weight.val().grad(&grads).is_none());
    assert!(l.base.bias.as_ref().unwrap().val().grad(&grads).is_none());
    assert!(l.magnitude.val().grad(&grads).is_some());
    assert!(l.lora_a.weight.val().grad(&grads).is_some());
    assert!(l.lora_b.weight.val().grad(&grads).is_some());
    assert!(x.grad(&grads).is_some());
}

#[test]
fn dora_merge_unmerge() {
    let mut l = dora_trained();
    let x = rnd::<3>([2, 3, 8]);
    let before = v(l.forward(x.clone()));
    l.merge();
    l.merge();
    assert!(l.is_merged());
    close(&v(l.forward(x.clone())), &before, 1e-4);
    l.unmerge();
    assert!(!l.is_merged());
    close(&v(l.forward(x)), &before, 0.0);
}

#[test]
fn dora_zero_weight_columns_stay_finite() {
    let mut base = lin(8, 6, true);
    let mut w = v(base.weight.val());
    for i in 0..8 {
        w[i * 6 + 2] = 0.0; // output 2 is completely zero
    }
    base.weight =
        burn::module::Param::from_tensor(Tensor::from_data(TensorData::new(w, [8, 6]), &dev()));
    let mut l = dora_cfg().init_from_linear(base.clone());
    let x = rnd::<2>([3, 8]);
    let y = v(l.forward(x.clone()));
    assert!(y.iter().all(|a| a.is_finite()));
    close(&y, &v(base.forward(x.clone())), 1e-4);
    l.merge();
    assert!(v(l.forward(x)).iter().all(|a| a.is_finite()));
}

#[test]
fn dora_scaling_and_count() {
    assert_eq!(DoraConfig::new(4, 4, 8).with_alpha(16.0).scaling(), 2.0);
    let l = dora_cfg().init(&dev());
    assert_eq!(l.adapter_parameter_count(), 8 * 4 + 4 * 6 + 6);
}

// ───────────────────────── IA³ ─────────────────────────

fn ia3_on(device: &Device, mode: Ia3Mode, bias: bool) -> Ia3Linear {
    let mut l = Ia3Config::new(8, 6)
        .with_mode(mode)
        .with_bias(bias)
        .init(device);
    let n = l.scale.dims()[0];
    l.scale = burn::module::Param::from_tensor(rnd_on(device, [n]));
    l
}

fn ia3(mode: Ia3Mode, bias: bool) -> Ia3Linear {
    ia3_on(&dev(), mode, bias)
}

#[test]
fn ia3_ones_equals_base() {
    for mode in [Ia3Mode::Input, Ia3Mode::Output] {
        let base = lin(8, 6, true);
        let l = Ia3Config::new(8, 6)
            .with_mode(mode)
            .init_from_linear(base.clone());
        let x = rnd::<3>([2, 4, 8]);
        close(&v(l.forward(x.clone())), &v(base.forward(x)), 1e-6);
    }
}

#[test]
fn ia3_input_manual_reference() {
    let l = ia3(Ia3Mode::Input, true);
    let x = rnd::<2>([3, 8]);
    let s = l.scale.val().unsqueeze::<2>();
    let expected =
        (x.clone() * s).matmul(l.base.weight()) + l.base.bias().unwrap().unsqueeze::<2>();
    close(&v(l.forward(x)), &v(expected), 1e-5);
}

#[test]
fn ia3_output_manual_reference_scales_bias() {
    let l = ia3(Ia3Mode::Output, true);
    let x = rnd::<2>([3, 8]);
    let s = l.scale.val().unsqueeze::<2>();
    let expected =
        (x.clone().matmul(l.base.weight()) + l.base.bias().unwrap().unsqueeze::<2>()) * s;
    close(&v(l.forward(x)), &v(expected), 1e-5);
}

#[test]
fn ia3_gradients() {
    for mode in [Ia3Mode::Input, Ia3Mode::Output] {
        let l = ia3_on(&adev(), mode, true);
        let x = rnd_on(&adev(), [2, 3, 8]).require_grad();
        let grads = l.forward(x.clone()).sum().backward();
        assert!(l.base.weight.val().grad(&grads).is_none());
        assert!(l.base.bias.as_ref().unwrap().val().grad(&grads).is_none());
        assert!(l.scale.val().grad(&grads).is_some());
        assert!(x.grad(&grads).is_some());
    }
}

#[test]
fn ia3_merge_unmerge_all_modes() {
    for mode in [Ia3Mode::Input, Ia3Mode::Output] {
        for bias in [true, false] {
            let mut l = ia3(mode.clone(), bias);
            let x = rnd::<3>([2, 3, 8]);
            let before = v(l.forward(x.clone()));
            l.merge();
            l.merge();
            assert!(l.is_merged());
            close(&v(l.forward(x.clone())), &before, 1e-4);
            l.unmerge();
            assert!(!l.is_merged());
            close(&v(l.forward(x)), &before, 0.0);
        }
    }
}

#[test]
fn ia3_broadcast_ranks_and_count() {
    let l = ia3(Ia3Mode::Input, false);
    assert_eq!(l.forward(rnd::<2>([5, 8])).dims(), [5, 6]);
    assert_eq!(l.forward(rnd::<3>([2, 5, 8])).dims(), [2, 5, 6]);
    assert_eq!(l.adapter_parameter_count(), 8);
    assert_eq!(ia3(Ia3Mode::Output, true).adapter_parameter_count(), 6);
}

// ───────────────────────── general ─────────────────────────

#[test]
fn all_adapters_start_equal_to_base() {
    let base = lin(8, 6, true);
    let x = rnd::<3>([2, 3, 8]);
    let y0 = v(base.forward(x.clone()));

    close(
        &v(lora_cfg().init_from_linear(base.clone()).forward(x.clone())),
        &y0,
        1e-5,
    );
    close(
        &v(dora_cfg().init_from_linear(base.clone()).forward(x.clone())),
        &y0,
        1e-5,
    );
    close(
        &v(Ia3Config::new(8, 6).init_from_linear(base).forward(x)),
        &y0,
        1e-5,
    );
}
