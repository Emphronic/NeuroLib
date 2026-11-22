//! Liquid State Machine (Maass, Natschläger, Markram, 2002): random
//! recurrent reservoir of spike LIF neurons with fixed weights.
//!
//! * Neurons are divided into excitatory and inhibitory (Dale's principle): ALL
//!   the outgoing weights of the neuron have the same sign. Proportion of stimulants -
//!   [`excitatory_fraction`](LiquidStateMachineConfig::excitatory_fraction)
//!   (default 0.8), inhibitory connections are stronger in
//!   [`inhibitory_gain`](LiquidStateMachineConfig::inhibitory_gain) times
//!   (balanced network in the spirit of Brunel).
//! * Connectivity is random with probability
//!   [`connection_prob`](LiquidStateMachineConfig::connection_prob), without
//!   self-connections.
//! * Dynamics of one step: `U ← β·U + x·W_in + s_prev·W_rec`, spike at
//!   `U > θ`, reset by subtracting `U ← U − θ·s`.
//! * "Fluid state" for readout - exponentially filtered spikes
//!   `r ← decay·r + s` ([`LiquidFeature::Trace`]), i.e. what a
//!   postsynaptic neuron would see; raw spikes and the membrane are also available.
//!
//! The weights are set by the seed ([`seed`](LiquidStateMachineConfig::seed)) and
//! reproducible. The reservoir does not learn: it learns on top of states
//! [`RidgeReadout`](super::RidgeReadout).

use burn::tensor::{Device, Tensor, TensorData};
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

use crate::error::{Result, SnnError};

/// State of the reservoir passed to the readout.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum LiquidFeature {
    /// Exponentially filtered spikes (default).
    #[default]
    Trace,
    /// Raw spikes 0/1 of the current step.
    Spikes,
    /// Membrane potential after reset.
    Membrane,
}

/// Configuration [`LiquidStateMachine`].
#[derive(Debug, Clone)]
pub struct LiquidStateMachineConfig {
    /// Number of input channels.
    pub inputs: usize,
    /// Number of reservoir neurons.
    pub neurons: usize,
    /// Proportion of excitatory neurons.
    pub excitatory_fraction: f32,
    /// Probability of connection `pre → post`.
    pub connection_prob: f32,
    /// The base value of the excitatory weight (actual weight - uniformly in
    /// `[0.5, 1.5] ·` this value).
    pub recurrent_scale: f32,
    /// How many times are inhibitory weights stronger than excitatory ones?
    pub inhibitory_gain: f32,
    /// Maximum modulus of input weight.
    pub input_scale: f32,
    /// The fraction of neurons receiving each input channel.
    pub input_density: f32,
    /// Membrane damping β ∈ [0, 1).
    pub beta: f32,
    /// Spike threshold θ > 0.
    pub threshold: f32,
    /// Spike filter attenuation for [`LiquidFeature::Trace`], `[0, 1)`.
    pub trace_decay: f32,
    /// Weight generator LED.
    pub seed: u64,
}

impl LiquidStateMachineConfig {
    /// Default configuration for `inputs` inputs and `neurons` neurons.
    pub fn new(inputs: usize, neurons: usize) -> Self {
        Self {
            inputs,
            neurons,
            excitatory_fraction: 0.8,
            connection_prob: 0.1,
            recurrent_scale: 0.15,
            inhibitory_gain: 4.0,
            input_scale: 0.5,
            input_density: 0.3,
            beta: 0.9,
            threshold: 1.0,
            trace_decay: 0.9,
            seed: 0,
        }
    }

    /// Set seed.
    pub fn with_seed(mut self, seed: u64) -> Self {
        self.seed = seed;
        self
    }

    /// Set the probability of the connection.
    pub fn with_connection_prob(mut self, p: f32) -> Self {
        self.connection_prob = p;
        self
    }

    /// Set the scale of recurrent weights.
    pub fn with_recurrent_scale(mut self, scale: f32) -> Self {
        self.recurrent_scale = scale;
        self
    }

    /// Set the scale of the input weights.
    pub fn with_input_scale(mut self, scale: f32) -> Self {
        self.input_scale = scale;
        self
    }

    fn validate(&self) -> Result<()> {
        let bad = |reason: String| {
            Err(SnnError::InvalidConfig {
                what: "LiquidStateMachineConfig",
                reason,
            })
        };
        if self.inputs == 0 || self.neurons == 0 {
            return bad(format!(
                "inputs and neurons must be > 0 ({}, {})",
                self.inputs, self.neurons
            ));
        }
        for (name, v) in [
            ("excitatory_fraction", self.excitatory_fraction),
            ("connection_prob", self.connection_prob),
            ("input_density", self.input_density),
        ] {
            if !(0.0..=1.0).contains(&v) {
                return bad(format!("{name} must lie in [0, 1], resulting in {v}"));
            }
        }
        for (name, v) in [("beta", self.beta), ("trace_decay", self.trace_decay)] {
            if !(0.0..1.0).contains(&v) {
                return bad(format!("{name} must lie in [0, 1), obtained {v}"));
            }
        }
        if self.threshold.is_nan() || self.threshold <= 0.0 {
            return bad(format!(
                "threshold must be > 0, obtained {}",
                self.threshold
            ));
        }
        Ok(())
    }
}

/// State of the liquid.
#[derive(Debug, Clone)]
pub struct LiquidState {
    /// Membrane `[batch, neurons]`.
    pub mem: Tensor<2>,
    /// Spikes of the last step `[batch, neurons]`.
    pub spikes: Tensor<2>,
    /// Filtered spikes `[batch, neurons]`.
    pub trace: Tensor<2>,
}

/// Spike reservoir.
#[derive(Debug, Clone)]
pub struct LiquidStateMachine {
    config: LiquidStateMachineConfig,
    /// `[inputs, neurons]`.
    w_in: Tensor<2>,
    /// `[neurons(pre), neurons(post)]`.
    w_rec: Tensor<2>,
    excitatory: Vec<bool>,
}

impl LiquidStateMachine {
    /// Builds a reservoir with random but reproducible (by seed) weights.
    pub fn new(config: &LiquidStateMachineConfig, device: &Device) -> Result<Self> {
        config.validate()?;
        let n = config.neurons;
        let mut rng = StdRng::seed_from_u64(config.seed);

        let excitatory: Vec<bool> = (0..n)
            .map(|_| rng.r#gen::<f32>() < config.excitatory_fraction)
            .collect();

        let mut w_rec = vec![0.0f32; n * n];
        for pre in 0..n {
            for post in 0..n {
                if pre == post || rng.r#gen::<f32>() >= config.connection_prob {
                    continue;
                }
                let magnitude = config.recurrent_scale * (0.5 + rng.r#gen::<f32>());
                w_rec[pre * n + post] = if excitatory[pre] {
                    magnitude
                } else {
                    -magnitude * config.inhibitory_gain
                };
            }
        }

        let mut w_in = vec![0.0f32; config.inputs * n];
        for i in 0..config.inputs {
            for j in 0..n {
                if rng.r#gen::<f32>() < config.input_density {
                    w_in[i * n + j] = (rng.r#gen::<f32>() * 2.0 - 1.0) * config.input_scale;
                }
            }
        }

        Ok(Self {
            config: config.clone(),
            w_in: Tensor::from_data(TensorData::new(w_in, [config.inputs, n]), device),
            w_rec: Tensor::from_data(TensorData::new(w_rec, [n, n]), device),
            excitatory,
        })
    }

    /// Configuration
    pub fn config(&self) -> &LiquidStateMachineConfig {
        &self.config
    }

    /// Which neurons are excitatory?
    pub fn excitatory(&self) -> &[bool] {
        &self.excitatory
    }

    /// Recurrent weights `[pre, post]`.
    pub fn recurrent_weights(&self) -> Tensor<2> {
        self.w_rec.clone()
    }

    /// Input weights `[inputs, neurons]`.
    pub fn input_weights(&self) -> Tensor<2> {
        self.w_in.clone()
    }

    /// Zero state for the batch.
    pub fn init_state(&self, batch: usize, device: &Device) -> LiquidState {
        let z = || Tensor::zeros([batch, self.config.neurons], device);
        LiquidState {
            mem: z(),
            spikes: z(),
            trace: z(),
        }
    }

    /// One step: `x` - `[batch, inputs]`.
    pub fn step(&self, x: Tensor<2>, state: &LiquidState) -> LiquidState {
        let theta = self.config.threshold;
        let drive = x.matmul(self.w_in.clone()) + state.spikes.clone().matmul(self.w_rec.clone());
        let mem = state.mem.clone().mul_scalar(self.config.beta) + drive;
        let spikes = mem.clone().greater_elem(theta).float();
        let mem = mem - spikes.clone().mul_scalar(theta);
        let trace = state.trace.clone().mul_scalar(self.config.trace_decay) + spikes.clone();
        LiquidState { mem, spikes, trace }
    }

    /// Running the sequence `[batch, time, inputs]` from the zero state;
    /// returns `[batch, time, neurons]` of the selected characteristic.
    pub fn run(&self, x: Tensor<3>, feature: LiquidFeature) -> Result<Tensor<3>> {
        let [batch, steps, inputs] = x.dims();
        if inputs != self.config.inputs {
            return Err(SnnError::Channels {
                op: "LiquidStateMachine::run",
                expected: self.config.inputs,
                actual: inputs,
            });
        }
        if steps == 0 {
            return Err(SnnError::EmptySequence {
                op: "LiquidStateMachine::run",
            });
        }
        let mut state = self.init_state(batch, &x.device());
        let mut frames = Vec::with_capacity(steps);
        for t in 0..steps {
            let xt = x
                .clone()
                .slice([0..batch, t..t + 1, 0..inputs])
                .reshape([batch, inputs]);
            state = self.step(xt, &state);
            frames.push(match feature {
                LiquidFeature::Trace => state.trace.clone(),
                LiquidFeature::Spikes => state.spikes.clone(),
                LiquidFeature::Membrane => state.mem.clone(),
            });
        }
        Ok(Tensor::stack::<3>(frames, 1))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::tensor_to_vec;
    use crate::reservoir::{Pooling, RidgeReadout, accuracy, one_hot, pool_time};

    fn dev() -> Device {
        Device::flex()
    }

    /// Sampling `[batch, time, 1]`: impulse in a given window + noise.
    fn pulse_batch(labels: &[usize], steps: usize, rng: &mut StdRng, device: &Device) -> Tensor<3> {
        let mut data = Vec::with_capacity(labels.len() * steps);
        for &label in labels {
            // Class 0: impulse in the first third of the window, class 1: in the last.
            let (a, b) = if label == 0 {
                (2, steps / 3)
            } else {
                (2 * steps / 3, steps - 2)
            };
            for t in 0..steps {
                let noise = (rng.r#gen::<f32>() - 0.5) * 0.2;
                let base = if t >= a && t < b { 1.0 } else { 0.0 };
                data.push((base + noise).clamp(0.0, 1.0));
            }
        }
        Tensor::from_data(TensorData::new(data, [labels.len(), steps, 1]), device)
    }

    #[test]
    fn same_seed_gives_same_weights_and_other_seed_differs() {
        let c = LiquidStateMachineConfig::new(2, 50);
        let a = LiquidStateMachine::new(&c.clone().with_seed(7), &dev()).unwrap();
        let b = LiquidStateMachine::new(&c.clone().with_seed(7), &dev()).unwrap();
        let d = LiquidStateMachine::new(&c.with_seed(8), &dev()).unwrap();
        let v = |m: &LiquidStateMachine| tensor_to_vec(m.recurrent_weights(), "t").unwrap();
        assert_eq!(v(&a), v(&b));
        assert_ne!(v(&a), v(&d));
    }

    /// Dale's principle: each presynaptic neuron is all non-zero
    /// the outgoing weights are of the same sign, and it coincides with the type of neuron.
    #[test]
    fn obeys_dales_law_without_self_connections() {
        let m = LiquidStateMachine::new(&LiquidStateMachineConfig::new(1, 60), &dev()).unwrap();
        let n = 60;
        let w = tensor_to_vec(m.recurrent_weights(), "t").unwrap();
        for pre in 0..n {
            assert_eq!(w[pre * n + pre], 0.0, "self-connection at {pre}");
            for post in 0..n {
                let x = w[pre * n + post];
                if x != 0.0 {
                    assert_eq!(
                        x > 0.0,
                        m.excitatory()[pre],
                        "violated Dale at {pre}→{post}"
                    );
                }
            }
        }
        let frac = m.excitatory().iter().filter(|e| **e).count() as f32 / n as f32;
        assert!(
            (0.6..=0.95).contains(&frac),
            "proportion of excitatory {frac}"
        );
    }

    /// Connectivity is close to specified.
    #[test]
    fn connectivity_matches_probability() {
        let n = 100;
        let c = LiquidStateMachineConfig::new(1, n).with_connection_prob(0.2);
        let m = LiquidStateMachine::new(&c, &dev()).unwrap();
        let w = tensor_to_vec(m.recurrent_weights(), "t").unwrap();
        let p = w.iter().filter(|x| **x != 0.0).count() as f32 / (n * (n - 1)) as f32;
        assert!((p - 0.2).abs() < 0.03, "p = {p}");
    }

    /// With a visible input, the network is neither silent nor saturated.
    #[test]
    fn activity_is_neither_silent_nor_saturated() {
        let m = LiquidStateMachine::new(&LiquidStateMachineConfig::new(1, 100), &dev()).unwrap();
        let x = Tensor::<3>::zeros([4, 60, 1], &dev()).add_scalar(0.8);
        let s = m.run(x, LiquidFeature::Spikes).unwrap();
        let v = tensor_to_vec(s, "t").unwrap();
        let rate = v.iter().sum::<f32>() / v.len() as f32;
        assert!(rate > 0.005, "the tank is silent: rate = {rate}");
        assert!(rate < 0.5, "reservoir is saturated: rate = {rate}");
    }

    /// Without input there is no activity (the membrane starts from scratch).
    #[test]
    fn silent_without_input() {
        let m = LiquidStateMachine::new(&LiquidStateMachineConfig::new(1, 50), &dev()).unwrap();
        let s = m
            .run(
                Tensor::<3>::zeros([2, 30, 1], &dev()),
                LiquidFeature::Spikes,
            )
            .unwrap();
        assert_eq!(tensor_to_vec(s, "t").unwrap().iter().sum::<f32>(), 0.0);
    }

    #[test]
    fn rejects_bad_config_and_shapes() {
        let mut c = LiquidStateMachineConfig::new(1, 10);
        c.beta = 1.0;
        assert!(LiquidStateMachine::new(&c, &dev()).is_err());
        let mut c = LiquidStateMachineConfig::new(1, 10);
        c.threshold = 0.0;
        assert!(LiquidStateMachine::new(&c, &dev()).is_err());
        assert!(LiquidStateMachine::new(&LiquidStateMachineConfig::new(0, 10), &dev()).is_err());

        let m = LiquidStateMachine::new(&LiquidStateMachineConfig::new(2, 10), &dev()).unwrap();
        let bad = Tensor::<3>::zeros([1, 5, 3], &dev());
        assert!(matches!(
            m.run(bad, LiquidFeature::Trace),
            Err(SnnError::Channels { .. })
        ));
        let empty = Tensor::<3>::zeros([1, 0, 2], &dev());
        assert!(matches!(
            m.run(empty, LiquidFeature::Trace),
            Err(SnnError::EmptySequence { .. })
        ));
    }

    /// Full LSM → pooling → ridge pipeline: distinguishes between "early" and "late"
    /// impulse on delayed sampling.
    #[test]
    fn classifies_early_vs_late_pulse() {
        let device = dev();
        let m =
            LiquidStateMachine::new(&LiquidStateMachineConfig::new(1, 120).with_seed(3), &device)
                .unwrap();
        let mut rng = StdRng::seed_from_u64(42);
        let steps = 36;

        let make = |n: usize, rng: &mut StdRng| -> (Tensor<2>, Vec<usize>) {
            let labels: Vec<usize> = (0..n).map(|i| i % 2).collect();
            let x = pulse_batch(&labels, steps, rng, &device);
            let states = m.run(x, LiquidFeature::Trace).unwrap();
            (pool_time(states, Pooling::Segments(3)), labels)
        };
        let (train_x, train_y) = make(60, &mut rng);
        let (test_x, test_y) = make(40, &mut rng);

        let readout = RidgeReadout::fit_tensor(
            train_x.clone(),
            one_hot(&train_y, 2, &device).unwrap(),
            1e-1,
        )
        .unwrap();
        let train_acc = accuracy(&readout.classify(train_x).unwrap(), &train_y);
        let test_acc = accuracy(&readout.classify(test_x).unwrap(), &test_y);
        assert!(train_acc > 0.95, "train accuracy {train_acc}");
        assert!(test_acc > 0.9, "test accuracy {test_acc}");
    }
}
