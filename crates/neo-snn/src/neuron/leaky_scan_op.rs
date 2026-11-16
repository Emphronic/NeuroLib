// File: crates/neo-snn/src/neuron/leaky_scan_op.rs
//
// Thin wrapper over `neo_synk::snn::leaky_scan_forward` for cutless
// membrane recurrent LeakyParallel:
//
//     u_t = β · u_{t-1} + x_t,   u_{-1} = mem0
//
// In Burn 0.22, the backend is selected at runtime by the device.
// `neo_synk::snn::leaky_scan_forward` itself chooses the path along `Device`:
//
//   * device CubeCL (wgpu / CUDA / ROCm; features `wgpu`, `cuda`, `rocm`
//     neo-synk) - fused core CubeCL with manual backward (`neo_synk::snn::gpu`),
//     autodiff works through it;
//   * any other device (Flex CPU, ...) - `leaky_scan_reference` on
//     ordinary tensor operations, autodiff is automatic.
//
// GPU path checked on wgpu (discrete GPU, Vulkan): the result matches
// CPU reference (max|diff| ≈ 7e-7), acceleration versus `ref-cpu` ×16…×570
// (`cargo run --release -p neo-synk --features wgpu,fusion --example leaky_scan_bench`).
// Your own `Backward` is not needed in neo-snn.

use burn::tensor::Tensor;

use neo_synk::snn::{leaky_scan_forward, leaky_scan_reference};

/// Fused scan path: `input` `[batch, steps, features]`, `mem0` `[batch, features]`,
/// `beta` `[1]`.
pub fn leaky_scan_fused(input: Tensor<3>, mem0: Tensor<2>, beta: Tensor<1>) -> Tensor<3> {
    leaky_scan_forward(input, mem0, beta)
}

/// Reference path for common tensor operations (tests).
pub fn leaky_scan(input: Tensor<3>, mem0: Tensor<2>, beta: Tensor<1>) -> Tensor<3> {
    leaky_scan_reference(input, mem0, beta)
}

#[cfg(test)]
mod leaky_scan_grad_tests {
    use super::*;
    use burn::tensor::Device;

    fn dev() -> Device {
        Device::flex().autodiff()
    }

    fn t1(data: Vec<f32>) -> Tensor<1> {
        Tensor::<1>::from_floats(data.as_slice(), &dev()).require_grad()
    }

    fn t2(data: Vec<f32>, shape: [usize; 2]) -> Tensor<2> {
        Tensor::<1>::from_floats(data.as_slice(), &dev())
            .reshape(shape)
            .require_grad()
    }

    fn t3(data: Vec<f32>, shape: [usize; 3]) -> Tensor<3> {
        Tensor::<1>::from_floats(data.as_slice(), &dev())
            .reshape(shape)
            .require_grad()
    }

    /// Finite-difference check of the gradient w.r.t. `beta`, via
    /// `leaky_scan_reference` — at this stage that's the only path on
    /// NdArray, but it's the source of truth
    /// for the analytic formula anyway: the manual VJP is checked against
    /// finite differences, not against Burn's own autodiff graph directly
    /// (which would be circular if the formula had just been copied off
    /// that graph).
    ///
    /// Run for T = 1, 2, 17, 64, 257.
    #[test]
    fn beta_gradient_matches_finite_difference_across_lengths() {
        for &steps in &[1usize, 2, 17, 64, 257] {
            let beta_val = 0.85f32;
            let batch = 2usize;
            let features = 3usize;

            let x: Vec<f32> = (0..batch * steps * features)
                .map(|i| ((i * 37 % 101) as f32 - 50.0) / 50.0)
                .collect();
            let mem0_v: Vec<f32> = (0..batch * features)
                .map(|i| ((i * 13 % 17) as f32 - 8.0) / 8.0)
                .collect();

            let loss_at = |beta: f32| -> f32 {
                let input = t3(x.clone(), [batch, steps, features]);
                let mem0 = t2(mem0_v.clone(), [batch, features]);
                let beta_t = t1(vec![beta]);
                let out = leaky_scan_reference(input, mem0, beta_t);
                out.sum().into_data().try_to_vec::<f32>().unwrap()[0]
            };

            let eps = 1e-3f32;
            let numeric = (loss_at(beta_val + eps) - loss_at(beta_val - eps)) / (2.0 * eps);

            let input = t3(x.clone(), [batch, steps, features]);
            let mem0 = t2(mem0_v.clone(), [batch, features]);
            let beta_t = t1(vec![beta_val]);
            let out = leaky_scan_reference(input, mem0, beta_t.clone());
            let loss = out.sum();
            let grads = loss.backward();
            let analytic = beta_t
                .grad(&grads)
                .expect("beta must have a gradient")
                .into_data()
                .try_to_vec::<f32>()
                .unwrap()[0];

            assert!(
                (analytic - numeric).abs() < 5e-2,
                "T={steps}: analytical dβ={analytic}, numerical={numeric}"
            );
        }
    }

    /// Same, for `mem0[0, 0]` — checks `∂L/∂mem0 = r_0 · β` indirectly
    /// (via comparison with Burn's autodiff over the reference graph,
    /// which for mem0 is just an ordinary op graph, no manual VJP, so
    /// this isn't circular relative to the formula being derived for the
    /// fused path).
    #[test]
    fn mem0_gradient_matches_finite_difference_across_lengths() {
        for &steps in &[1usize, 2, 17, 64, 257] {
            let beta_val = 0.7f32;
            let batch = 1usize;
            let features = 1usize;

            let x: Vec<f32> = (0..steps).map(|t| (t as f32 * 0.3).sin()).collect();

            let loss_at = |mem0: f32| -> f32 {
                let input = t3(x.clone(), [batch, steps, features]);
                let mem0_t = t2(vec![mem0], [batch, features]);
                let beta_t = t1(vec![beta_val]);
                let out = leaky_scan_reference(input, mem0_t, beta_t);
                out.sum().into_data().try_to_vec::<f32>().unwrap()[0]
            };

            let eps = 1e-3f32;
            let numeric = (loss_at(0.5 + eps) - loss_at(0.5 - eps)) / (2.0 * eps);

            let input = t3(x.clone(), [batch, steps, features]);
            let mem0_t = t2(vec![0.5], [batch, features]);
            let beta_t = t1(vec![beta_val]);
            let out = leaky_scan_reference(input, mem0_t.clone(), beta_t);
            let loss = out.sum();
            let grads = loss.backward();
            let analytic = mem0_t
                .grad(&grads)
                .expect("mem0 must have a gradient")
                .into_data()
                .try_to_vec::<f32>()
                .unwrap()[0];

            assert!(
                (analytic - numeric).abs() < 5e-2,
                "T={steps}: analytical dmem0={analytic}, numerical={numeric}"
            );
        }
    }
}
