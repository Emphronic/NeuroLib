//! Slow pathway state: the compact memory `m` of a dual memory layer.
//!
//! Implements the paper's slow pathway (Methods): an LMU-style linear
//! state-space driven by a scalar input — `m[k] = Ā·m[k−1] + B̄·x[k]` — with the
//! Legendre state-space matrices of the original implementation of the paper
//! ([`Dual_memory_pathways`](https://github.com/sunpengfei1122/Dual_memory_pathways),
//! file `dula_memory_pathway`), exactly ZOH-discretized:
//! `Ā = e^{A·Δt}`, `B̄ = A⁻¹(e^{A·Δt} − I)B`.
//!
//! `m` is `[batch, d]` — compact (d ≪ N), shared per layer, and learnable
//! through the feedback weights of [`MemoryFeedback`](super::MemoryFeedback).

use burn::module::{Module, Param};
use burn::tensor::activation::relu;
use burn::tensor::{Device, Distribution, Tensor, TensorData};

use crate::stateful::Detach;

// ---------------------------------------------------------------------
// Discrete slow-memory state
// ---------------------------------------------------------------------

/// Slow pathway state: `m`, `[batch, d]` — the compact context of the recent
/// activity (paper: "compact low-dimensional state that summarizes recent
/// activity").
#[derive(Debug, Clone)]
pub struct SlowMemoryState {
    /// `m`: `[batch, d]`.
    pub m: Tensor<2>,
}

impl SlowMemoryState {
    /// Zero state: `m = 0` of shape `[batch, d]`.
    pub fn zeros(batch: usize, d: usize, device: &Device) -> Self {
        Self {
            m: Tensor::zeros([batch, d], device),
        }
    }
}

impl Detach for SlowMemoryState {
    fn detach(self) -> Self {
        Self { m: self.m.detach() }
    }
}

// ---------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------

/// Config of the slow pathway.
#[derive(Debug, Clone, Copy)]
pub struct SlowMemoryConfig {
    /// Memory dimension `d` (≪ layer width; the paper uses 5–10% of the
    /// hidden width).
    pub d: usize,
    /// State buffer length `θ` (paper: "the delay network orthogonalizes the
    /// input signal over a sliding window of length θ").
    pub theta: f32,
    /// Per-step time resolution `Δt` of the exact ZOH discretization (the
    /// paper experiments use 1.0).
    pub dt: f32,
}

impl SlowMemoryConfig {
    /// New config with `dt = 1.0` (the paper's setting).
    pub fn new(d: usize, theta: f32) -> Self {
        Self { d, theta, dt: 1.0 }
    }

    /// Per-step time resolution of the exact ZOH discretization.
    pub fn with_dt(mut self, dt: f32) -> Self {
        self.dt = dt;
        self
    }
}

// ---------------------------------------------------------------------
// Slow memory (LMU-style state-space)
// ---------------------------------------------------------------------

/// Slow pathway: LMU-style state-space memory `m' = Ā·m + B̄·x`.
///
/// `Ā` and `B̄` are the exact ZOH discretization of the Legendre state-space
/// matrices from [`SlowMemory::state_matrices`] and are frozen during
/// training (as in the paper); the drive weights `w_x`/`b_x` are trainable.
/// The drive `x = ReLU(W_x·s + b_x)` compresses the presynaptic spikes `s`
/// of the layer input ("compress presynaptic spikes into a scalar drive",
/// paper Methods).
#[derive(Module, Debug)]
pub struct SlowMemory {
    /// Memory dimension `d`.
    pub d: usize,
    /// State buffer length `θ`.
    pub theta: f32,
    /// `Ā = e^{A·Δt}`: discrete transition, `[d, d]` (frozen, paper).
    #[module(skip)]
    pub a_hat: Tensor<2>,
    /// `B̄ = A⁻¹(e^{A·Δt} − I)B`: discrete input, `[d, 1]` (frozen, paper).
    #[module(skip)]
    pub b_hat: Tensor<2>,
    /// `W_x` `[1, input_dim]`: scalar-drive compression of presynaptic spikes.
    pub w_x: Param<Tensor<2>>,
    /// `b_x` `[1]`: drive bias.
    pub b_x: Param<Tensor<1>>,
}

impl SlowMemory {
    /// Create the slow pathway.
    ///
    /// `input_dim` is the number of presynaptic features the drive compresses
    /// (the layer input width). `Ā`/`B̄` are computed by [`Self::state_matrices`]
    /// and exact ZOH discretization; the drive weights are initialized
    /// uniformly `±1/√input_dim`, `b_x` with zeros.
    pub fn new(config: SlowMemoryConfig, input_dim: usize, device: &Device) -> Self {
        assert!(config.d > 0, "SlowMemory: d must be at least 1");
        assert!(input_dim > 0, "SlowMemory: input_dim must be at least 1");
        assert!(config.theta > 0.0, "SlowMemory: theta must be positive");
        assert!(config.dt > 0.0, "SlowMemory: dt must be positive");
        let d = config.d;
        let (a, b) = Self::state_matrices(d, config.theta);
        let (a_hat, b_hat) = zoh_discretize(&a, &b, d, config.dt);
        let bound = (1.0 / input_dim as f64).sqrt();
        Self {
            d,
            theta: config.theta,
            a_hat: mat_to_tensor(&a_hat, device),
            b_hat: vec_to_col(&b_hat, device),
            w_x: Param::from_tensor(Tensor::random(
                [1, input_dim],
                Distribution::Uniform(-bound, bound),
                device,
            )),
            b_x: Param::from_tensor(Tensor::zeros([1], device)),
        }
    }

    /// Continuous-time state-space matrices `(A, B)` of the LMU slow pathway,
    /// per the paper's Methods and the original implementation: with
    /// `r[i] = (2i + 1)/θ`,
    ///
    /// * `A[i][j] = r[i]·(−1)` if `j > i`, else `r[i]·(−1)^{i−j+1}`
    /// * `B[i] = r[i]·(−1)^i`
    ///
    /// (`A` is `[d, d]`, `B` is `[d]`, the paper's `[d, 1]` column.)
    pub fn state_matrices(d: usize, theta: f32) -> (Vec<Vec<f64>>, Vec<f64>) {
        assert!(d > 0, "state_matrices: d must be at least 1");
        let theta = theta as f64;
        assert!(theta > 0.0, "state_matrices: theta must be positive");
        let mut a = vec![vec![0.0; d]; d];
        let mut b = vec![0.0; d];
        for i in 0..d {
            let r = (2 * i + 1) as f64 / theta;
            for j in 0..d {
                let sign = if i < j {
                    -1.0
                } else {
                    (-1.0f64).powi((i - j + 1) as i32)
                };
                a[i][j] = r * sign;
            }
            b[i] = r * (-1.0f64).powi(i as i32);
        }
        (a, b)
    }

    /// Scalar drive `x = ReLU(W_x·s + b_x)`, `[batch, 1]` — the "scalar drive"
    /// that the slow pathway receives (paper Methods).
    pub fn drive(&self, spikes: &Tensor<2>) -> Tensor<2> {
        let [batch, _] = spikes.dims();
        let drive = spikes.clone().matmul(self.w_x.val().transpose())
            + self.b_x.val().reshape([1, 1]).expand([batch, 1]);
        relu(drive)
    }

    /// Slow state update: `m' = Ā·m + B̄·x` — the paper's
    /// `m[k] = Ā·m[k−1] + B̄·x[k]` (`drive` is the scalar drive of step `k`).
    pub fn step(&self, drive: &Tensor<2>, state: &SlowMemoryState) -> SlowMemoryState {
        let m = state.m.clone().matmul(self.a_hat.clone().transpose())
            + drive.clone().matmul(self.b_hat.clone().transpose());
        SlowMemoryState { m }
    }
}

// ---------------------------------------------------------------------
// Host-side linear algebra (f64) for the exact ZOH discretization
// ---------------------------------------------------------------------

fn mat_mul(a: &[Vec<f64>], b: &[Vec<f64>]) -> Vec<Vec<f64>> {
    let n = a.len();
    let p = a[0].len();
    let m = b[0].len();
    assert_eq!(b.len(), p, "mat_mul: inner dimensions differ");
    let mut out = vec![vec![0.0; m]; n];
    for i in 0..n {
        for l in 0..p {
            let a_il = a[i][l];
            if a_il == 0.0 {
                continue;
            }
            for j in 0..m {
                out[i][j] += a_il * b[l][j];
            }
        }
    }
    out
}

fn mat_identity(n: usize) -> Vec<Vec<f64>> {
    (0..n)
        .map(|i| {
            (0..n)
                .map(|j| if i == j { 1.0 } else { 0.0 })
                .collect::<Vec<f64>>()
        })
        .collect()
}

fn mat_scale(a: &[Vec<f64>], s: f64) -> Vec<Vec<f64>> {
    a.iter()
        .map(|r| r.iter().map(|v| v * s).collect())
        .collect()
}

fn mat_add(a: &[Vec<f64>], b: &[Vec<f64>]) -> Vec<Vec<f64>> {
    a.iter()
        .zip(b)
        .map(|(ra, rb)| ra.iter().zip(rb).map(|(x, y)| x + y).collect())
        .collect()
}

/// Matrix exponential via scaling-and-squaring with a Taylor series (f64).
fn expm(a: &[Vec<f64>], n: usize) -> Vec<Vec<f64>> {
    // Infinity norm.
    let norm = a
        .iter()
        .map(|row| row.iter().map(|v| v.abs()).fold(0.0, f64::max))
        .fold(0.0, f64::max);
    // Scale so the Taylor series converges quickly: ||A/2^s||∞ ≤ 0.5.
    let mut squarings = 0usize;
    let mut scaled = a.to_vec();
    let mut scaled_norm = norm;
    while scaled_norm > 0.5 {
        squarings += 1;
        scaled = mat_scale(&scaled, 0.5);
        scaled_norm *= 0.5;
    }
    // Taylor: E = I + A_s + A_s²/2! + A_s³/3! + ...
    let mut e = mat_identity(n);
    let mut term = mat_identity(n);
    for k in 1..=100usize {
        term = mat_mul(&term, &scaled);
        term = mat_scale(&term, 1.0 / k as f64);
        let mut tiny = true;
        for row in &term {
            for v in row {
                if v.abs() > 1e-18 {
                    tiny = false;
                }
            }
        }
        e = mat_add(&e, &term);
        if tiny {
            break;
        }
    }
    // Undo the scaling: e = e^{A}.
    for _ in 0..squarings {
        e = mat_mul(&e, &e);
    }
    e
}

/// Exact ZOH discretization of the paper: `Ā = e^{A·Δt}` and
/// `B̄ = A⁻¹(e^{A·Δt} − I)B`.
///
/// Implemented via the augmented system `M = [[A·Δt, B], [0, 0]]` (size `d+1`):
/// `e^M = [[e^{A·Δt}, A⁻¹(e^{A·Δt} − I)B/Δt], [0, 1]]`, so the discrete input
/// is the top-right block scaled by `Δt`. No explicit inverse/solve is needed.
fn zoh_discretize(a: &[Vec<f64>], b: &[f64], d: usize, dt: f32) -> (Vec<Vec<f64>>, Vec<f64>) {
    let n = d + 1;
    let mut m = vec![vec![0.0; n]; n];
    for i in 0..d {
        for j in 0..d {
            m[i][j] = a[i][j] * dt as f64;
        }
        m[i][d] = b[i];
    }
    let em = expm(&m, n);
    let a_hat: Vec<Vec<f64>> = (0..d).map(|i| (0..d).map(|j| em[i][j]).collect()).collect();
    let b_hat: Vec<f64> = (0..d).map(|i| em[i][d] * dt as f64).collect();
    (a_hat, b_hat)
}

fn mat_to_tensor(m: &[Vec<f64>], device: &Device) -> Tensor<2> {
    let (d0, d1) = (m.len(), m[0].len());
    let data: Vec<f32> = m
        .iter()
        .flat_map(|row| row.iter())
        .map(|&v| v as f32)
        .collect();
    Tensor::from_data(TensorData::new(data, [d0, d1]), device)
}

fn vec_to_col(v: &[f64], device: &Device) -> Tensor<2> {
    let data: Vec<f32> = v.iter().map(|&x| x as f32).collect();
    Tensor::from_data(TensorData::new(data, [v.len(), 1]), device)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn state_matrices_hand_checked() {
        // Paper formulas: for d = 2, θ = 2 → r = [0.5, 1.5] →
        // A = [[−0.5, −0.5], [1.5, −1.5]], B = [0.5, −1.5].
        let (a, b) = SlowMemory::state_matrices(2, 2.0);
        let expect_a = [[-0.5, -0.5], [1.5, -1.5]];
        let expect_b = [0.5, -1.5];
        for i in 0..2 {
            for j in 0..2 {
                assert!((a[i][j] - expect_a[i][j]).abs() < 1e-12, "a[{i}][{j}]");
            }
        }
        for i in 0..2 {
            assert!((b[i] - expect_b[i]).abs() < 1e-12, "b[{i}]");
        }
        // d = 1, θ = 1 → A = [[−1]], B = [1].
        let (a, b) = SlowMemory::state_matrices(1, 1.0);
        assert!((a[0][0] + 1.0).abs() < 1e-12);
        assert!((b[0] - 1.0).abs() < 1e-12);
    }

    #[test]
    fn exact_zoh_discretization_d1() {
        // d = 1, θ = 1, dt = 1: A = [[−1]], B = [1] → Ā = e^{−1},
        // B̄ = A⁻¹(Ā − I)B = 1 − e^{−1}.
        let slow = SlowMemory::new(SlowMemoryConfig::new(1, 1.0), 4, &Device::flex());
        let a_hat: Vec<f32> = slow.a_hat.into_data().try_to_vec::<f32>().unwrap();
        let b_hat: Vec<f32> = slow.b_hat.into_data().try_to_vec::<f32>().unwrap();
        assert!(
            (a_hat[0] - (-1.0f32).exp()).abs() < 1e-5,
            "a_hat = {a_hat:?}"
        );
        assert!(
            (b_hat[0] - (1.0f32 - (-1.0f32).exp())).abs() < 1e-5,
            "b_hat = {b_hat:?}"
        );
    }
}
