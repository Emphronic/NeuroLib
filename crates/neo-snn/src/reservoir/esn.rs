//! Echo State Network (Jaeger, 2001): an analog reservoir of `tanh` neurons
//! with leakage and a prescribed spectral radius of the recurrent matrix.
//!
//! ```text
//! x[t] = (1 − α)·x[t−1] + α·tanh(u[t]·W_in + x[t−1]·W_rec + b)
//! ```
//!
//! `α` is the leak rate (`leak_rate`, 1 = no state memory), `W_rec` is
//! scaled so that its spectral radius equals
//! [`spectral_radius`](EchoStateNetworkConfig::spectral_radius) (usually
//! `< 1` — the "echo property": the influence of past input decays). States
//! are read by [`RidgeReadout`](super::RidgeReadout), as in the LSM.

use burn::tensor::{Device, Tensor, TensorData};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use crate::error::{Result, SnnError};

/// Estimation of the spectral radius of a square matrix `n × n` (row by row).
///
/// Power method: the average logarithmic growth of the norm `‖Mᵏv‖` over `k` gives
/// `ln ρ`. Unlike a naive norm-based estimate, it also works with a complex
/// conjugate pair of dominant eigenvalues. Approximate estimate
/// (for random matrices - units of percent) and deterministic.
pub fn estimate_spectral_radius(matrix: &[f32], n: usize) -> f32 {
    assert_eq!(
        matrix.len(),
        n * n,
        "estimate_spectral_radius: non-n×n matrix"
    );
    if n == 0 {
        return 0.0;
    }
    let mut rng = StdRng::seed_from_u64(0x5eed_5eed);
    let mut v: Vec<f64> = (0..n).map(|_| rng.r#gen::<f64>() - 0.5).collect();
    let norm0 = v.iter().map(|x| x * x).sum::<f64>().sqrt().max(1e-30);
    v.iter_mut().for_each(|x| *x /= norm0);

    let (warmup, iters) = (60usize, 400usize);
    let (mut log_sum, mut count) = (0.0f64, 0usize);
    let mut u = vec![0.0f64; n];
    for k in 0..iters {
        for (i, ui) in u.iter_mut().enumerate() {
            let row = &matrix[i * n..(i + 1) * n];
            *ui = row.iter().zip(&v).map(|(m, x)| *m as f64 * x).sum();
        }
        let norm = u.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm < 1e-30 {
            return 0.0;
        }
        if k >= warmup {
            log_sum += norm.ln();
            count += 1;
        }
        for (vi, ui) in v.iter_mut().zip(&u) {
            *vi = ui / norm;
        }
    }
    (log_sum / count as f64).exp() as f32
}

/// Configuration [`EchoStateNetwork`].
#[derive(Debug, Clone)]
pub struct EchoStateNetworkConfig {
    /// Number of input channels.
    pub inputs: usize,
    /// Number of neurons.
    pub neurons: usize,
    /// Target spectral radius `W_rec`.
    pub spectral_radius: f32,
    /// Leak rate `α ∈ (0, 1]`.
    pub leak_rate: f32,
    /// Maximum modulus of input weight.
    pub input_scale: f32,
    /// The scale of random displacement of neurons (0 - no displacement).
    pub bias_scale: f32,
    /// Probability of connection in `W_rec`.
    pub connection_prob: f32,
    /// Weight generator LED.
    pub seed: u64,
}

impl EchoStateNetworkConfig {
    /// Default configuration.
    pub fn new(inputs: usize, neurons: usize) -> Self {
        Self {
            inputs,
            neurons,
            spectral_radius: 0.9,
            leak_rate: 1.0,
            input_scale: 1.0,
            bias_scale: 0.0,
            connection_prob: 0.1,
            seed: 0,
        }
    }

    /// Set seed.
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    /// Set the spectral radius.
    pub fn with_spectral_radius(mut self, rho: f32) -> Self {
        self.spectral_radius = rho;
        self
    }

    /// Set the leak rate.
    pub fn with_leak_rate(mut self, alpha: f32) -> Self {
        self.leak_rate = alpha;
        self
    }

    fn validate(&self) -> Result<()> {
        let bad = |reason: String| {
            Err(SnnError::InvalidConfig {
                what: "EchoStateNetworkConfig",
                reason,
            })
        };
        if self.inputs == 0 || self.neurons == 0 {
            return bad(format!(
                "inputs and neurons must be > 0 ({}, {})",
                self.inputs, self.neurons
            ));
        }
        if !(self.leak_rate > 0.0 && self.leak_rate <= 1.0) {
            return bad(format!("leak_rate ∈ (0, 1], obtained {}", self.leak_rate));
        }
        if self.spectral_radius.is_nan() || self.spectral_radius <= 0.0 {
            return bad(format!(
                "spectral_radius must be > 0, obtained {}",
                self.spectral_radius
            ));
        }
        if !(0.0..=1.0).contains(&self.connection_prob) {
            return bad(format!(
                "connection_prob ∈ [0, 1], obtained {}",
                self.connection_prob
            ));
        }
        Ok(())
    }
}

/// Analog tank.
#[derive(Debug, Clone)]
pub struct EchoStateNetwork {
    config: EchoStateNetworkConfig,
    w_in: Tensor<2>,
    w_rec: Tensor<2>,
    bias: Tensor<2>,
}

impl EchoStateNetwork {
    /// Builds a reservoir; `W_rec` is reduced to a given spectral radius.
    pub fn new(config: &EchoStateNetworkConfig, device: &Device) -> Result<Self> {
        config.validate()?;
        let n = config.neurons;
        let mut rng = StdRng::seed_from_u64(config.seed);

        let mut w_rec = vec![0.0f32; n * n];
        for w in w_rec.iter_mut() {
            if rng.r#gen::<f32>() < config.connection_prob {
                *w = rng.r#gen::<f32>() * 2.0 - 1.0;
            }
        }
        let estimate = estimate_spectral_radius(&w_rec, n);
        if estimate > 1e-8 {
            let scale = config.spectral_radius / estimate;
            w_rec.iter_mut().for_each(|w| *w *= scale);
        }

        let w_in: Vec<f32> = (0..config.inputs * n)
            .map(|_| (rng.r#gen::<f32>() * 2.0 - 1.0) * config.input_scale)
            .collect();
        let bias: Vec<f32> = (0..n)
            .map(|_| (rng.r#gen::<f32>() * 2.0 - 1.0) * config.bias_scale)
            .collect();

        Ok(Self {
            config: config.clone(),
            w_in: Tensor::from_data(TensorData::new(w_in, [config.inputs, n]), device),
            w_rec: Tensor::from_data(TensorData::new(w_rec, [n, n]), device),
            bias: Tensor::from_data(TensorData::new(bias, [1, n]), device),
        })
    }

    /// Configuration
    pub fn config(&self) -> &EchoStateNetworkConfig {
        &self.config
    }

    /// Recurrent weights `[n, n]`.
    pub fn recurrent_weights(&self) -> Tensor<2> {
        self.w_rec.clone()
    }

    /// Zero state `[batch, neurons]`.
    pub fn init_state(&self, batch: usize, device: &Device) -> Tensor<2> {
        Tensor::zeros([batch, self.config.neurons], device)
    }

    /// One step: `x` - `[batch, inputs]`, `state` - `[batch, neurons]`.
    pub fn step(&self, x: Tensor<2>, state: Tensor<2>) -> Tensor<2> {
        let [batch, n] = state.dims();
        let pre = x.matmul(self.w_in.clone())
            + state.clone().matmul(self.w_rec.clone())
            + self.bias.clone().expand([batch, n]);
        let alpha = self.config.leak_rate;
        state.mul_scalar(1.0 - alpha) + pre.tanh().mul_scalar(alpha)
    }

    /// Run `[batch, time, inputs]` → state `[batch, time, neurons]`.
    pub fn run(&self, x: Tensor<3>) -> Result<Tensor<3>> {
        let [batch, steps, inputs] = x.dims();
        if inputs != self.config.inputs {
            return Err(SnnError::Channels {
                op: "EchoStateNetwork::run",
                expected: self.config.inputs,
                actual: inputs,
            });
        }
        if steps == 0 {
            return Err(SnnError::EmptySequence {
                op: "EchoStateNetwork::run",
            });
        }
        let mut state = self.init_state(batch, &x.device());
        let mut frames = Vec::with_capacity(steps);
        for t in 0..steps {
            let xt = x
                .clone()
                .slice([0..batch, t..t + 1, 0..inputs])
                .reshape([batch, inputs]);
            state = self.step(xt, state);
            frames.push(state.clone());
        }
        Ok(Tensor::stack::<3>(frames, 1))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::tensor_to_vec;
    use crate::reservoir::{RidgeReadout, flatten_time, nmse};

    fn dev() -> Device {
        Device::flex()
    }

    #[test]
    fn spectral_radius_of_known_matrices() {
        // Diagonal: radius = max |λ| = 0.9.
        let diag = [0.3, 0.0, 0.0, 0.0, 0.9, 0.0, 0.0, 0.0, 0.5];
        let r = estimate_spectral_radius(&diag, 3);
        assert!((r - 0.9).abs() < 0.01, "diag: {r}");
        // Rotated and scaled: eigenvalues ±0.7i.
        let rot = [0.0, -0.7, 0.7, 0.0];
        let r = estimate_spectral_radius(&rot, 2);
        assert!((r - 0.7).abs() < 0.01, "rot: {r}");
        // Zero matrix.
        assert_eq!(estimate_spectral_radius(&[0.0; 4], 2), 0.0);
    }

    /// After scaling, the estimated radius `W_rec` is equal to the target radius.
    #[test]
    fn recurrent_matrix_is_scaled_to_target_radius() {
        let c = EchoStateNetworkConfig::new(1, 80)
            .with_seed(5)
            .with_spectral_radius(0.8);
        let esn = EchoStateNetwork::new(&c, &dev()).unwrap();
        let w = tensor_to_vec(esn.recurrent_weights(), "t").unwrap();
        let r = estimate_spectral_radius(&w, 80);
        assert!((r - 0.8).abs() < 0.02, "radius {r}");
    }

    /// The states are bounded by `tanh` and are finite even with a large input.
    #[test]
    fn states_are_bounded() {
        let esn = EchoStateNetwork::new(&EchoStateNetworkConfig::new(1, 50), &dev()).unwrap();
        let x = Tensor::<3>::zeros([2, 100, 1], &dev()).add_scalar(50.0);
        let s = tensor_to_vec(esn.run(x).unwrap(), "t").unwrap();
        assert!(s.iter().all(|v| v.is_finite() && v.abs() <= 1.0 + 1e-6));
    }

    /// Memory task: restore an input from three steps ago.
    /// A linear readout on tank states should cope.
    #[test]
    fn recalls_input_from_three_steps_ago() {
        let device = dev();
        let esn = EchoStateNetwork::new(&EchoStateNetworkConfig::new(1, 100).with_seed(1), &device)
            .unwrap();
        let mut rng = StdRng::seed_from_u64(9);
        let (batch, steps, delay) = (4usize, 200usize, 3usize);
        let u: Vec<f32> = (0..batch * steps)
            .map(|_| rng.r#gen::<f32>() * 2.0 - 1.0)
            .collect();
        let mut target = vec![0.0f32; batch * steps];
        for b in 0..batch {
            for t in delay..steps {
                target[b * steps + t] = u[b * steps + t - delay];
            }
        }
        let x: Tensor<3> = Tensor::from_data(TensorData::new(u, [batch, steps, 1]), &device);
        let y: Tensor<3> = Tensor::from_data(TensorData::new(target, [batch, steps, 1]), &device);

        let states = esn.run(x).unwrap();
        let washout = 20;
        let feats = flatten_time(states, washout);
        let tgt = flatten_time(y, washout);
        let rows = feats.dims()[0];
        let split = rows * 3 / 4;
        let slice = |t: Tensor<2>, a: usize, b: usize| {
            let k = t.dims()[1];
            t.slice([a..b, 0..k])
        };
        let readout = RidgeReadout::fit_tensor(
            slice(feats.clone(), 0, split),
            slice(tgt.clone(), 0, split),
            1e-4,
        )
        .unwrap();
        let pred = readout.predict(slice(feats, split, rows)).unwrap();
        let err = nmse(pred, slice(tgt, split, rows)).unwrap();
        assert!(err < 0.05, "NMSE on memory task: {err}");
    }

    #[test]
    fn rejects_bad_config() {
        let mut c = EchoStateNetworkConfig::new(1, 10);
        c.leak_rate = 0.0;
        assert!(EchoStateNetwork::new(&c, &dev()).is_err());
        let mut c = EchoStateNetworkConfig::new(1, 10);
        c.spectral_radius = -1.0;
        assert!(EchoStateNetwork::new(&c, &dev()).is_err());
        let esn = EchoStateNetwork::new(&EchoStateNetworkConfig::new(2, 10), &dev()).unwrap();
        assert!(esn.run(Tensor::<3>::zeros([1, 5, 3], &dev())).is_err());
    }
}
