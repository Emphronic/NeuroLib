//! GPU trainer for the Burn dynamic executor ([`TensorNet`](crate::dense_burn::TensorNet)):
//! the online local-eligibility rule of [`structural::eprop`](super::eprop),
//! on tensors. The trainer owns the network, stores synapse eligibility
//! traces (`Tensor<2> [n, n]`, `traces[i*n + j]` = trace of synapse `i → j`),
//! and writes weight updates online.
//!
//! Math (LIF/ALIF derivations — see [`structural::eprop`](super::eprop) for
//! the full theory; `x[i,j]` is the delivered current of synapse `i → j`):
//!
//! ```text
//! e[i, j](t) = γ·e[i, j](t−1) + κ_j(t)·x[i, j](t)   // κ_j = postsynaptic spike
//! W_d[i, j](t) ← W_d[i, j](t−1) − η·d_j(t)·e[i, j](t)
//! ```
//!
//! where `x[i,j](t) = Σ_d hist_d[i]·W_d[i,j]` — the delivery of synapse `i→j`
//! at step `t`. Supports batch size 1.
//!
//! # Learning signals
//!
//! * `ReadoutMode::LocalError` (default): `d_j(t) = ŷ_j(t) − targets[j]` for
//!   output neurons (targets from [`GpuEpropConfig::targets`]); hidden neurons:
//!   `d_h = Σ_k F[h][k]·d_k(t)` via [`GpuEpropConfig::feedback`]
//!   (shape `[n_h, n_o]`).
//! * `GpuEpropConfig::use_external` + [`GpuEpropTrainer::set_external_signals`]:
//!   `d = external_signals` for every neuron (RL/hybrid systems).

use burn::tensor::{Tensor, TensorData};

use crate::dense_burn::{TensorNet, TensorState};

use super::error::{StructuralError, StructuralResult};

fn vec_of(t: &Tensor<2>) -> Vec<f32> {
    t.clone()
        .into_data()
        .try_to_vec::<f32>()
        .unwrap_or_default()
}

/// Configuration of the GPU e-prop trainer.
#[derive(Debug, Clone)]
pub struct GpuEpropConfig {
    /// Learning rate `η` of the weight update.
    pub learning_rate: f32,
    /// Trace decay `γ ∈ [0, 1)`.
    pub trace_decay: f32,
    /// Readout error targets per output neuron index (`d_j = ŷ_j − targets[j]`;
    /// missing = 0). Ignored when `use_external` is true.
    pub targets: Vec<f32>,
    /// Feedback matrix for hidden neurons, shape `[n_h, n_o]`
    /// (`d_h = Σ_k F[h][k]·d_k(t)`).
    pub feedback: Tensor<2>,
    /// Use external learning signals for all neurons instead of readout+feedback.
    pub use_external: bool,
    /// External learning signals: `d = external_signals` (when `use_external`).
    pub external_signals: Option<Tensor<2>>,
}

impl GpuEpropConfig {
    /// Validate the config values (finite, ranges, `n`-sized targets).
    pub fn validate(&self, n: usize) -> StructuralResult<()> {
        if !(self.learning_rate.is_finite() && self.learning_rate >= 0.0) {
            return Err(StructuralError::param(
                "learning_rate",
                "must be finite and >= 0",
            ));
        }
        if !(self.trace_decay.is_finite() && (0.0..1.0).contains(&self.trace_decay)) {
            return Err(StructuralError::param(
                "trace_decay",
                "must be finite and in [0, 1)",
            ));
        }
        if self.targets.len() < n {
            return Err(StructuralError::param(
                "targets",
                format!("need >= {n} targets, got {}", self.targets.len()),
            ));
        }
        Ok(())
    }
}

/// GPU e-prop trainer (batch 1). See the module docs for the math.
pub struct GpuEpropTrainer {
    net: TensorNet,
    cfg: GpuEpropConfig,
    /// Eligibility traces `e[i, j]` — `[n, n]` (src × dst).
    traces: Tensor<2>,
    steps: u64,
    updates: u64,
}

impl GpuEpropTrainer {
    /// Wrap a network and create a trainer (`cfg.validate` + zero traces).
    pub fn new(net: TensorNet, cfg: GpuEpropConfig) -> StructuralResult<Self> {
        let n = net.n();
        cfg.validate(n)?;
        let traces = Tensor::zeros([n, n], net.device());
        Ok(Self {
            net,
            cfg,
            traces,
            steps: 0,
            updates: 0,
        })
    }

    /// The network (read-only; weights via [`TensorNet::weights`]).
    pub fn net(&self) -> &TensorNet {
        &self.net
    }

    /// Config.
    pub fn config(&self) -> &GpuEpropConfig {
        &self.cfg
    }

    /// Step counter.
    pub fn steps(&self) -> u64 {
        self.steps
    }

    /// Number of applied weight updates.
    pub fn updates(&self) -> u64 {
        self.updates
    }

    /// Eligibility traces `[n, n]` (`traces[i * n + j]` = trace of synapse `i → j`).
    pub fn traces(&self) -> &Tensor<2> {
        &self.traces
    }

    /// Set external learning signals (`d = external_signals` when `use_external`).
    pub fn set_external_signals(&mut self, s: Tensor<2>) {
        self.cfg.external_signals = Some(s);
    }

    /// The core step: `net.step_and_spikes`, then:
    /// 1. learning signals `d_j(t)` (readout errors for outputs, feedback
    ///    cascade for hidden, external when configured);
    /// 2. eligibility update `e[i,j] = γ·e[i,j] + κ_j·x[i,j]`;
    /// 3. weight update `W_d[i,j] -= η·d_j·e[i,j]`.
    pub fn step(&mut self, st: &mut TensorState, inputs: Tensor<2>) -> StructuralResult<()> {
        if st.batch() != 1 {
            return Err(StructuralError::param(
                "GpuEpropTrainer::step",
                format!("supports batch 1 (got {})", st.batch()),
            ));
        }
        let n = self.net.n();
        // 1. Dynamics.
        let (_out, all) = self.net.step_and_spikes(st, inputs);
        let spikes: Vec<f32> = all.into_data().try_to_vec::<f32>().unwrap_or_default();
        if spikes.len() != n {
            return Err(StructuralError::param(
                "GpuEpropTrainer::step",
                "bad spikes shape",
            ));
        }
        // 2. Learning signals `d[j]`.
        let mut d: Vec<f32> = vec![0.0; n];
        if self.cfg.use_external {
            if let Some(s) = &self.cfg.external_signals {
                d = vec_of(s);
            }
        } else {
            for j in 0..n {
                d[j] = spikes[j] - self.cfg.targets.get(j).copied().unwrap_or(0.0);
            }
            // Hidden: d_h[h] = Σ_k F[h][k]·d[k] (F is [n_h, n_o]).
            let fv = vec_of(&self.cfg.feedback);
            let (nh, no) = {
                let dims = self.cfg.feedback.dims();
                (dims[0], dims[1])
            };
            if nh > 0 && no > 0 {
                for h in 0..nh {
                    let mut acc = 0.0;
                    for k in 0..no {
                        acc += fv[h * no + k] * d[k];
                    }
                    d[h] = acc;
                }
            }
        }
        // 3. Eligibility update: e[i,j] = γ·e[i,j] + κ_j·x[i,j],
        //    x[i,j] = Σ_d hist_d[i]·W_d[i,j] (delivery of synapse i→j).
        let gamma = self.cfg.trace_decay;
        let mut x: Vec<f32> = vec![0.0; n * n];
        for (wd, w) in self.net.weights().iter().enumerate() {
            let wv = vec_of(w);
            let hv = st.hist().get(wd).map(vec_of).unwrap_or_default();
            for i in 0..n {
                if i >= hv.len() {
                    break;
                }
                if hv[i] == 0.0 {
                    continue;
                }
                for j in 0..n {
                    x[i * n + j] += hv[i] * wv[i * n + j];
                }
            }
        }
        let mut e: Vec<f32> = vec_of(&self.traces);
        if e.len() != n * n {
            e = vec![0.0; n * n];
        }
        for j in 0..n {
            let kappa = spikes[j];
            for i in 0..n {
                e[i * n + j] = gamma * e[i * n + j] + kappa * x[i * n + j];
            }
        }
        self.traces = Tensor::from_data(TensorData::new(e.clone(), [n, n]), self.net.device());
        // 4. Weight update: W_d[i,j] ← W_d[i,j] − η·d_j·e[i,j].
        let eta = self.cfg.learning_rate;
        let mut new_ws: Vec<Tensor<2>> = self.net.weights().to_vec();
        for w in new_ws.iter_mut() {
            let mut wv: Vec<f32> = vec_of(w);
            if wv.len() != n * n {
                wv = vec![0.0; n * n];
            }
            for j in 0..n {
                for i in 0..n {
                    wv[i * n + j] -= eta * d[j] * e[i * n + j];
                }
            }
            *w = Tensor::from_data(TensorData::new(wv, [n, n]), self.net.device());
        }
        self.net.set_weight_matrices(new_ws);
        self.steps += 1;
        self.updates += 1;
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::Device;

    #[test]
    fn gpu_eprop_config_validation() {
        let d = Device::flex();
        let cfg = GpuEpropConfig {
            learning_rate: 0.0,
            trace_decay: 0.95,
            targets: vec![0.0; 4],
            feedback: Tensor::zeros([0, 0], &d),
            external_signals: None,
            use_external: false,
        };
        assert!(cfg.validate(4).is_ok());
        assert!(cfg.validate(5).is_err()); // targets too short
        let bad = GpuEpropConfig {
            trace_decay: 1.0,
            ..cfg.clone()
        };
        assert!(bad.validate(4).is_err());
    }
}
