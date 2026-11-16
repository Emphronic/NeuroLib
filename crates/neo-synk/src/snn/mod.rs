//! Kernels for `neo-snn`'s spiking layers.
//!
//! First (and currently only) kernel: a fused forward pass for
//! `LeakyParallel`'s membrane recurrence
//! `u[t] = beta * u[t-1] + input[t]`, `u[-1] = mem0`.
//!
//! The GPU path works: it is verified on wgpu (discrete GPU, Vulkan) against
//! [`leaky_scan_reference`] (max|diff| ≈ 7e-7) and is 16×–570× faster than the
//! CPU reference on the `leaky_scan_bench` sizes. Reproduce with
//! `cargo run --release -p neo-synk --features wgpu,fusion --example leaky_scan_bench`.
//!
//! * [`leaky_scan_forward`] — the entry point `neo-snn` calls. On a CubeCL
//!   device (wgpu / CUDA / ROCm; features `wgpu`, `cuda`, `rocm`) it launches
//!   the fused kernel from [`gpu`] (one launch instead of the O(log T) tensor
//!   ops of the reference path, with a hand-written backward for autodiff).
//!   On any other device (Flex CPU, ...) it falls back to
//!   [`leaky_scan_reference`].
//! * [`leaky_scan_reference`] is a correct (if slow) implementation built
//!   purely from ordinary Burn tensor ops, usable on any device.

use burn::tensor::Tensor;

#[cfg(feature = "flex")]
pub mod spike;
#[cfg(feature = "flex")]
pub use spike::{SurrogateKind, spike_fused_supported, spike_surrogate};

#[cfg(any(feature = "wgpu", feature = "cuda", feature = "rocm"))]
pub mod gpu;

/// Fused forward pass for the leaky-membrane scan.
///
/// * `input` — `[batch, steps, features]`.
/// * `mem0` — `[batch, features]`, the membrane state just before `t = 0`.
/// * `beta` — `[1]`, already passed through `sigmoid` by the caller (this
///   function does not know or care how `beta` is parameterized upstream).
///
/// Returns `[batch, steps, features]`, where
/// `output[b, t, f] = beta^(t+1) * mem0[b, f] + sum_{k=0}^{t} beta^(t-k) *
/// input[b, k, f]`.
///
/// burn 0.22 has no backend type parameter on `Tensor`: the backend is the
/// tensor's [`Device`](burn::tensor::Device). CubeCL devices run the fused
/// kernel (autodiff works through its manual backward); other devices go
/// through [`leaky_scan_reference`] (ordinary differentiable tensor ops).
pub fn leaky_scan_forward(input: Tensor<3>, mem0: Tensor<2>, beta: Tensor<1>) -> Tensor<3> {
    #[cfg(any(feature = "wgpu", feature = "cuda", feature = "rocm"))]
    {
        if input.dims()[1] > 0 && gpu::is_cube_device(&input.device()) {
            return gpu::leaky_scan_fused(input, mem0, beta);
        }
    }
    leaky_scan_reference(input, mem0, beta)
}

/// Reference implementation: the same recurrence, computed with ordinary
/// Burn tensor ops (a doubling / Hillis-Steele-style scan over an affine
/// monoid `(beta, x) : y -> beta*y + x`, associative under composition).
///
/// This is intentionally backend-agnostic (works on any [`Device`](burn::tensor::Device))
/// and is used both as the default path for devices without a fused kernel,
/// and as the ground truth that `neo-synk`'s CubeCL kernel is checked against
/// in tests.
///
/// * `input` — `[batch, steps, features]`.
/// * `mem0` — `[batch, features]`.
/// * `beta` — `[1]`.
pub fn leaky_scan_reference(input: Tensor<3>, mem0: Tensor<2>, beta: Tensor<1>) -> Tensor<3> {
    let [batch, steps, features] = input.dims();
    let device = input.device();

    if steps == 0 {
        return input;
    }

    // Doubling scan over the affine monoid (a, x) representing
    // `y -> a*y + x`. Composing `(a2, x2) . (a1, x1) = (a2*a1, a2*x1 + x2)`
    // (apply (a1,x1) first, then (a2,x2)) lets a Hillis-Steele-style
    // doubling pass compute all T prefix compositions in O(log T) rounds
    // instead of running the T-step sequential recurrence directly.
    //
    // Seed: a[t] = beta for every t (each step multiplies the running
    // state by beta once), x[t] = input[t].
    let mut a: Tensor<3> = beta.reshape([1, 1, 1]).expand([batch, steps, features]);
    let mut x = input;

    let mut d = 1usize;
    while d < steps {
        // Shift (a, x) by d steps along the time axis, padding the front
        // with the identity element (a=1, x=0) so steps with no
        // predecessor at this offset are untouched by composition.
        let ones_pad: Tensor<3> = Tensor::ones([batch, d, features], &device);
        let zeros_pad: Tensor<3> = Tensor::zeros([batch, d, features], &device);

        let a_hi = a.clone().slice([0..batch, 0..(steps - d), 0..features]);
        let x_hi = x.clone().slice([0..batch, 0..(steps - d), 0..features]);

        let a_shift = Tensor::cat(vec![ones_pad, a_hi], 1);
        let x_shift = Tensor::cat(vec![zeros_pad, x_hi], 1);

        // Compose the CURRENT transform after the shifted PREFIX:
        // `(a, x) . (a_shift, x_shift)`. For `y -> a*y + x`, this is
        // `y -> a*(a_shift*y + x_shift) + x`, hence
        // `new_x = a*x_shift + x`.
        //
        // This order is essential: the recurrence is
        // `u[t] = beta*u[t-1] + input[t]`, so at distance `d` the
        // current input contribution must remain unscaled while the
        // earlier prefix is multiplied by the current transform's `a`.
        let new_x = a.clone().mul(x_shift) + x;
        let new_a = a.mul(a_shift);

        a = new_a;
        x = new_x;
        d *= 2;
    }

    // mem0 contributes beta^(t+1) * mem0[b, f] to every output[b, t, f];
    // `a[b, t, f]` after the scan above already equals beta^(t+1) (the
    // product of t+1 copies of beta), so reuse it directly instead of
    // recomputing powers of beta separately.
    let mem0_contrib = a.mul(
        mem0.reshape([batch, 1, features])
            .expand([batch, steps, features]),
    );

    x + mem0_contrib
}

#[cfg(test)]
mod reference_tests {
    use super::*;
    use burn::tensor::Device;

    fn dev() -> Device {
        Device::flex()
    }

    fn t1(data: Vec<f32>) -> Tensor<1> {
        Tensor::<1>::from_floats(data.as_slice(), &dev())
    }

    fn t2(data: Vec<f32>, shape: [usize; 2]) -> Tensor<2> {
        Tensor::<1>::from_floats(data.as_slice(), &dev()).reshape(shape)
    }

    fn t3(data: Vec<f32>, shape: [usize; 3]) -> Tensor<3> {
        Tensor::<1>::from_floats(data.as_slice(), &dev()).reshape(shape)
    }

    fn vec3(t: Tensor<3>) -> Vec<f32> {
        t.into_data().try_to_vec::<f32>().unwrap()
    }

    /// Hand-computed closed form for (batch=1, features=1), mem0=0: must
    /// match the direct sum `sum_k beta^(t-k) * x[k]`.
    #[test]
    fn reference_matches_hand_computed_closed_form_zero_mem0() {
        let beta = 0.8f32;
        let x = vec![1.0f32, 2.0, 0.5, -1.0, 3.0];
        let n = x.len();

        let mut expected = Vec::with_capacity(n);
        for t in 0..n {
            let mut s = 0.0f32;
            for k in 0..=t {
                s += beta.powi((t - k) as i32) * x[k];
            }
            expected.push(s);
        }

        let out = leaky_scan_reference(t3(x, [1, n, 1]), t2(vec![0.0], [1, 1]), t1(vec![beta]));
        let got = vec3(out);

        for (i, (&g, &e)) in got.iter().zip(expected.iter()).enumerate() {
            assert!((g - e).abs() < 1e-4, "t={i}: got {g}, expected {e}");
        }
    }

    /// `mem0 != 0` must enter as `beta^(t+1) * mem0`.
    #[test]
    fn reference_applies_mem0_with_correct_power() {
        let beta = 0.5f32;
        let mem0 = 4.0f32;
        let x = vec![0.0f32, 0.0, 0.0]; // pure input is zero: only mem0 shows

        let out = leaky_scan_reference(
            t3(x.clone(), [1, x.len(), 1]),
            t2(vec![mem0], [1, 1]),
            t1(vec![beta]),
        );
        let got = vec3(out);

        for (t, &g) in got.iter().enumerate() {
            let expected = beta.powi((t + 1) as i32) * mem0;
            assert!(
                (g - expected).abs() < 1e-4,
                "t={t}: got {g}, expected {expected}"
            );
        }
    }

    /// Non-power-of-two length exercises the doubling scan's padding path.
    #[test]
    fn reference_handles_non_power_of_two_length() {
        let beta = 0.7f32;
        let x: Vec<f32> = (0..17).map(|i| (i as f32 - 8.0) / 4.0).collect();
        let n = x.len();

        let mut expected = Vec::with_capacity(n);
        for t in 0..n {
            let mut s = 0.0f32;
            for k in 0..=t {
                s += beta.powi((t - k) as i32) * x[k];
            }
            expected.push(s);
        }

        let out = leaky_scan_reference(t3(x, [1, n, 1]), t2(vec![0.0], [1, 1]), t1(vec![beta]));
        let got = vec3(out);

        for (i, (&g, &e)) in got.iter().zip(expected.iter()).enumerate() {
            assert!((g - e).abs() < 1e-3, "t={i}: got {g}, expected {e}");
        }
    }

    /// Batch and feature lanes must not leak into each other.
    #[test]
    fn reference_keeps_batch_and_feature_independent() {
        let beta = 0.6f32;
        // batch=2, steps=4, features=2; batch 0 all zeros, batch 1 has data,
        // feature 0 vs feature 1 use different values.
        let x = vec![
            0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, // batch 0
            1.0, -1.0, 2.0, 0.5, -0.5, 1.5, 0.25, -0.25, // batch 1
        ];
        let out = leaky_scan_reference(
            t3(x, [2, 4, 2]),
            t2(vec![0.0, 0.0, 0.0, 0.0], [2, 2]),
            t1(vec![beta]),
        );
        let got = vec3(out);

        // Batch 0 must stay all zeros.
        for v in &got[0..8] {
            assert!(v.abs() < 1e-6, "batch 0 leaked nonzero: {v}");
        }
        // Batch 1, feature 0 direct check against hand closed form.
        let x1f0 = [1.0f32, 2.0, -0.5, 0.25];
        for t in 0..4 {
            let mut expected = 0.0f32;
            for k in 0..=t {
                expected += beta.powi((t - k) as i32) * x1f0[k];
            }
            let g = got[8 + t * 2];
            assert!(
                (g - expected).abs() < 1e-3,
                "batch1/f0 t={t}: got {g}, expected {expected}"
            );
        }
    }

    /// Empty sequence must return the (empty) input unchanged, not panic.
    #[test]
    fn reference_handles_empty_sequence() {
        // `from_floats(&[]).reshape([1, 0, 1])` panics in burn 0.21 (reshape
        // zero size inside cubecl-zspace), so empty input
        // is created directly.
        let empty = Tensor::<3>::zeros([1, 0, 1], &dev());
        let out = leaky_scan_reference(empty, t2(vec![0.0], [1, 1]), t1(vec![0.5]));
        assert_eq!(out.dims(), [1, 0, 1]);
    }

    /// Single-step sequence must skip the doubling loop body entirely and
    /// still be correct (`d=1` is not `< steps=1`).
    #[test]
    fn reference_handles_single_step() {
        let beta = 0.9f32;
        let mem0 = 2.0f32;
        let x = vec![3.0f32];

        let out = leaky_scan_reference(
            t3(x.clone(), [1, 1, 1]),
            t2(vec![mem0], [1, 1]),
            t1(vec![beta]),
        );
        let got = vec3(out);
        let expected = beta * mem0 + x[0];
        assert!(
            (got[0] - expected).abs() < 1e-5,
            "got {}, expected {expected}",
            got[0]
        );
    }
}

// ───────────────────── GPU (CubeCL) ─────────────────────
// `cargo test -p neo-synk --features wgpu gpu_` (without GPU - software Vulkan).
#[cfg(all(test, any(feature = "wgpu", feature = "cuda", feature = "rocm")))]
mod gpu_tests {
    use super::*;
    use burn::tensor::{Device, TensorData};

    fn device() -> Device {
        #[cfg(feature = "cuda")]
        return Device::cuda(0usize);
        #[cfg(all(feature = "wgpu", not(feature = "cuda")))]
        return Device::wgpu(burn::tensor::DeviceKind::DefaultDevice);
        #[cfg(all(feature = "rocm", not(any(feature = "cuda", feature = "wgpu"))))]
        return Device::rocm(0usize);
    }

    fn data(len: usize, seed: u32) -> Vec<f32> {
        let mut s = seed;
        (0..len)
            .map(|_| {
                s = s.wrapping_mul(1664525).wrapping_add(1013904223);
                ((s >> 8) as f32 / (1u32 << 24) as f32) - 0.5
            })
            .collect()
    }

    fn t3(v: Vec<f32>, shape: [usize; 3], d: &Device) -> Tensor<3> {
        Tensor::<3>::from_data(TensorData::new(v, shape), d)
    }
    fn t2(v: Vec<f32>, shape: [usize; 2], d: &Device) -> Tensor<2> {
        Tensor::<2>::from_data(TensorData::new(v, shape), d)
    }

    fn close(a: &[f32], b: &[f32], what: &str) {
        assert_eq!(a.len(), b.len(), "{what}");
        for (i, (x, y)) in a.iter().zip(b).enumerate() {
            assert!(
                (x - y).abs() <= 1e-4 * (1.0 + y.abs()),
                "{what}[{i}]: gpu={x} ref={y}"
            );
        }
    }

    #[test]
    fn gpu_leaky_scan_forward_matches_reference() {
        let d = device();
        let (b, t, f) = (3, 37, 5);
        let x = data(b * t * f, 1);
        let m0 = data(b * f, 2);
        let beta = Tensor::<1>::from_data(TensorData::new(vec![0.83f32], [1]), &d);
        let got = gpu::leaky_scan_fused(
            t3(x.clone(), [b, t, f], &d),
            t2(m0.clone(), [b, f], &d),
            beta.clone(),
        );
        let want = leaky_scan_reference(t3(x, [b, t, f], &d), t2(m0, [b, f], &d), beta);
        close(
            &got.into_data().try_to_vec::<f32>().unwrap(),
            &want.into_data().try_to_vec::<f32>().unwrap(),
            "forward",
        );
    }

    #[test]
    fn gpu_leaky_scan_entry_point_uses_gpu_and_handles_empty() {
        let d = device();
        let beta = Tensor::<1>::from_data(TensorData::new(vec![0.5f32], [1]), &d);
        let out = leaky_scan_forward(
            Tensor::<3>::zeros([2, 0, 4], &d),
            Tensor::<2>::zeros([2, 4], &d),
            beta,
        );
        assert_eq!(out.dims(), [2, 0, 4]);
    }

    #[test]
    fn gpu_leaky_scan_gradients_match_reference() {
        let d = device().autodiff();
        let (b, t, f) = (2, 9, 4);
        let x = data(b * t * f, 3);
        let m0 = data(b * f, 4);
        let w = data(b * t * f, 5);
        let run = |fused: bool| {
            let xi = t3(x.clone(), [b, t, f], &d).require_grad();
            let mi = t2(m0.clone(), [b, f], &d).require_grad();
            let bt = Tensor::<1>::from_data(TensorData::new(vec![0.7f32], [1]), &d).require_grad();
            let out = if fused {
                gpu::leaky_scan_fused(xi.clone(), mi.clone(), bt.clone())
            } else {
                leaky_scan_reference(xi.clone(), mi.clone(), bt.clone())
            };
            let loss = (out * t3(w.clone(), [b, t, f], &d)).sum();
            let g = loss.backward();
            let v = |t: Tensor<3>| t.into_data().try_to_vec::<f32>().unwrap();
            (
                v(xi.grad(&g).unwrap()),
                mi.grad(&g)
                    .unwrap()
                    .into_data()
                    .try_to_vec::<f32>()
                    .unwrap(),
                bt.grad(&g)
                    .unwrap()
                    .into_data()
                    .try_to_vec::<f32>()
                    .unwrap(),
            )
        };
        let (gx, gm, gb) = run(true);
        let (rx, rm, rb) = run(false);
        close(&gx, &rx, "d input");
        close(&gm, &rm, "d mem0");
        close(&gb, &rb, "d beta");
    }

    /// Chunked path (few pairs, long chain): forward vs reference.
    #[test]
    fn gpu_leaky_scan_chunked_matches_reference() {
        let d = device();
        let beta = Tensor::<1>::from_data(TensorData::new(vec![0.93f32], [1]), &d);
        // lengths are not a multiple of the number of pieces; fewer and more features CHUNK_FX; empty tail pieces
        for (b, t, f) in [
            (1usize, 1000usize, 5usize),
            (2, 513, 3),
            (1, 300, 1),
            (3, 257, 9),
            (1, 256, 8),
        ] {
            let x = data(b * t * f, 11);
            let m0 = data(b * f, 12);
            let got = gpu::leaky_scan_fused(
                t3(x.clone(), [b, t, f], &d),
                t2(m0.clone(), [b, f], &d),
                beta.clone(),
            );
            let want = leaky_scan_reference(t3(x, [b, t, f], &d), t2(m0, [b, f], &d), beta.clone());
            close(
                &got.into_data().try_to_vec::<f32>().unwrap(),
                &want.into_data().try_to_vec::<f32>().unwrap(),
                &format!("chunked forward {b}x{t}x{f}"),
            );
        }
    }

    /// Gradients via chunked path (backward uses the same kernel).
    #[test]
    fn gpu_leaky_scan_chunked_gradients_match_reference() {
        let d = device().autodiff();
        let (b, t, f) = (1, 300, 3);
        let x = data(b * t * f, 13);
        let m0 = data(b * f, 14);
        let w = data(b * t * f, 15);
        let run = |fused: bool| {
            let xi = t3(x.clone(), [b, t, f], &d).require_grad();
            let mi = t2(m0.clone(), [b, f], &d).require_grad();
            let bt = Tensor::<1>::from_data(TensorData::new(vec![0.9f32], [1]), &d).require_grad();
            let out = if fused {
                gpu::leaky_scan_fused(xi.clone(), mi.clone(), bt.clone())
            } else {
                leaky_scan_reference(xi.clone(), mi.clone(), bt.clone())
            };
            let g = (out * t3(w.clone(), [b, t, f], &d)).sum().backward();
            (
                xi.grad(&g)
                    .unwrap()
                    .into_data()
                    .try_to_vec::<f32>()
                    .unwrap(),
                mi.grad(&g)
                    .unwrap()
                    .into_data()
                    .try_to_vec::<f32>()
                    .unwrap(),
                bt.grad(&g)
                    .unwrap()
                    .into_data()
                    .try_to_vec::<f32>()
                    .unwrap(),
            )
        };
        let (gx, gm, gb) = run(true);
        let (rx, rm, rb) = run(false);
        close(&gx, &rx, "d input");
        close(&gm, &rm, "d mem0");
        // β-gradient - sum over 900 terms: relative tolerance, wider
        for (a, r) in gb.iter().zip(&rb) {
            assert!(
                (a - r).abs() <= 1e-3 * (1.0 + r.abs()),
                "d beta gpu={a} ref={r}"
            );
        }
    }
}
