//! Stochastic threshold neuron — [`StochasticThresholdLeaky`]: LIF with a stochastic (Bayesian)
//! threshold — the paper "Device Stochasticity Directly Implements Bayesian Sampling in Spiking
//! Neurons": the threshold is drawn from a distribution — the software analogue of the MTJ
//! device randomness.
//!
//! Two modes:
//! - **stochastic spike execution** — [`StochasticThresholdLeaky::step`] draws the threshold
//!   (per step or per state) and spikes when `mem > θ_sample`; the same input gives different
//!   spikes — Bayesian sampling at the level of the neuron;
//! - **probabilistic forward** — [`crate::probabilistic`] (Rate Estimation,
//!   [`crate::probabilistic::forward_seq_prob`]) — the probabilities of spikes without
//!   generating any spikes.

use burn::module::{Module, Param};
use burn::tensor::activation::{sigmoid, softplus};
use burn::tensor::{Device, Tensor};

use super::{logit, spike_with_threshold};
use crate::neuron::threshold_distribution::ThresholdDistribution;
use crate::probabilistic::RngSource;
use crate::spiking_layer::{Deferred, SpikingLayer};
use crate::state::{NeuronState, ResetMode, apply_reset_tensor};
use crate::stateful::Detach;
use crate::surrogate::Surrogate;

/// Configuration of the stochastic-threshold LIF.
#[derive(Debug, Clone, Copy)]
pub struct StochasticThresholdLeakyConfig {
    /// Damping β_init ∈ (0, 1) — as in [`LeakyConfig::beta_init`](super::LeakyConfig::beta_init).
    pub beta_init: f32,
    /// Parameters of the threshold distribution — `θ ~ N(mean, std_dev)`.
    pub distribution: ThresholdDistribution,
    /// Re-sample the threshold at every step (`true`, the hardware analogue) or once per state
    /// (`false`).
    pub sample_per_timestep: bool,
    /// Seed for the RNG source — reproducible runs.
    pub seed: Option<u64>,
    /// Reset mode.
    pub reset_mode: ResetMode,
}

impl StochasticThresholdLeakyConfig {
    /// New config with `beta_init` and the default distribution `N(1.0, 0.5)`.
    pub fn new(beta_init: f32) -> Self {
        Self {
            beta_init,
            distribution: ThresholdDistribution::new(1.0, 0.5),
            sample_per_timestep: true,
            seed: None,
            reset_mode: ResetMode::Subtract,
        }
    }
}

/// State of the stochastic-threshold LIF: membrane, the sampled threshold and the RNG source.
#[derive(Debug, Clone)]
pub struct StochasticThresholdState {
    /// Membrane/neural state.
    pub mem: NeuronState,
    /// The sampled threshold of the last step (the "device randomness" draw).
    pub threshold_sample: Tensor<2>,
    /// RNG source — state across steps.
    pub rng: RngSource,
}

impl StochasticThresholdState {
    /// Zero state (membrane only).
    pub fn zeros_mem_only(device: &Device, batch: usize, units: usize, rng: RngSource) -> Self {
        Self {
            mem: NeuronState::zeros_mem_only(device, batch, units),
            threshold_sample: Tensor::zeros([batch, units], device),
            rng,
        }
    }
}

impl Detach for StochasticThresholdState {
    fn detach(self) -> Self {
        Self {
            mem: self.mem.detach(),
            threshold_sample: self.threshold_sample.detach(),
            rng: self.rng,
        }
    }
}

/// LIF with a stochastic threshold — `θ ~ N(mean, std_dev)`, drawn per neuron (or per step).
/// Bayesian sampling at the level of the neuron: the same input may give different spikes,
/// modeling the uncertainty of the threshold.
#[derive(Module, Debug)]
pub struct StochasticThresholdLeaky {
    /// Raw β: `sigmoid(beta)` - damping at (0, 1). shape `[1]`.
    pub beta: Param<Tensor<1>>,
    /// Raw mean of the threshold distribution. shape `[1]`.
    pub threshold_mean: Param<Tensor<1>>,
    /// Raw std of the threshold distribution: `softplus(raw)` (positive). shape `[1]`.
    pub threshold_std: Param<Tensor<1>>,
    /// Re-sample the threshold at every step.
    #[module(skip)]
    pub sample_per_timestep: bool,
    /// Reset mode.
    #[module(skip)]
    pub reset_mode: ResetMode,
    /// Seed for the RNG source.
    #[module(skip)]
    pub seed: Option<u64>,
}

/// Inverse softplus for positive values — same inversion as
/// [`Leaky::threshold_param`](super::Leaky::threshold_param) in [`leaky.rs`](super).
fn inverse_softplus(x: f32) -> f32 {
    if x > 20.0 {
        x
    } else {
        (x.exp() - 1.0).max(1e-6).ln()
    }
}

impl StochasticThresholdLeaky {
    /// Create the layer from the config.
    pub fn new(config: &StochasticThresholdLeakyConfig, device: &Device) -> Self {
        let beta = Param::from_tensor(Tensor::from_floats([logit(config.beta_init)], device));
        let threshold_mean =
            Param::from_tensor(Tensor::from_floats([config.distribution.mean], device));
        let threshold_std = Param::from_tensor(Tensor::from_floats(
            [inverse_softplus(config.distribution.std_dev.max(1e-3))],
            device,
        ));
        Self {
            beta,
            threshold_mean,
            threshold_std,
            sample_per_timestep: config.sample_per_timestep,
            reset_mode: config.reset_mode,
            seed: config.seed,
        }
    }

    /// The trained threshold distribution mean, as a scalar.
    pub fn threshold_mean_value(&self) -> f32 {
        self.threshold_mean
            .val()
            .into_data()
            .try_to_vec::<f32>()
            .unwrap()[0]
    }

    /// The trained threshold distribution std (softplus of the raw), as a scalar.
    pub fn threshold_std_value(&self) -> f32 {
        softplus(self.threshold_std.val(), 1.0)
            .into_data()
            .try_to_vec::<f32>()
            .unwrap()[0]
    }

    /// The threshold distribution as a broadcast tensor of shape `dims` — the posterior mean of
    /// the threshold (see [`crate::probabilistic::bayesian_inference`]).
    pub fn threshold_distribution_tensor(&self, dims: [usize; 2], device: &Device) -> Tensor<2> {
        let mean = self.threshold_mean_value();
        Tensor::<1>::from_floats(std::slice::from_ref(&mean), device)
            .reshape([1, 1])
            .expand(dims)
    }

    /// The threshold distribution std as a broadcast tensor of shape `dims`.
    pub fn threshold_std_tensor(&self, dims: [usize; 2], device: &Device) -> Tensor<2> {
        let std = self.threshold_std_value();
        Tensor::<1>::from_floats(std::slice::from_ref(&std), device)
            .reshape([1, 1])
            .expand(dims)
    }

    /// Damping β = `sigmoid(beta)` as the tensor `dims` (as in
    /// [`Leaky::beta_tensor`](super::Leaky::beta_tensor)).
    pub fn beta_tensor(&self, dims: [usize; 2]) -> Tensor<2> {
        sigmoid(self.beta.val()).reshape([1, 1]).expand(dims)
    }

    /// Draw a threshold sample of shape `dims` — `θ ~ N(mean, std)`, per neuron.
    fn sample_threshold(
        &self,
        dims: [usize; 2],
        device: &Device,
        rng: &mut RngSource,
    ) -> Tensor<2> {
        let mean = self.threshold_mean_value();
        let std_dev = self.threshold_std_value();
        let [batch, units] = dims;
        let mut values = Vec::with_capacity(batch * units);
        for _ in 0..batch * units {
            values.push(rng.sample_normal(mean, std_dev));
        }
        Tensor::<1>::from_floats(values.as_slice(), device).reshape([batch, units])
    }

    /// One step: membrane update + threshold sampling + the stochastic spike.
    pub fn step<S: Surrogate + ?Sized>(
        &self,
        input: Tensor<2>,
        state: &StochasticThresholdState,
        surrogate: &S,
    ) -> (Tensor<2>, StochasticThresholdState) {
        let dims = state.mem.mem.dims();
        let device = state.mem.mem.device();
        let mut state = state.clone();

        // The threshold — re-sample at every step (hardware analogue) or reuse the state's draw.
        let threshold = if self.sample_per_timestep {
            let t = self.sample_threshold(dims, &device, &mut state.rng);
            state.threshold_sample = t.clone();
            t
        } else {
            state.threshold_sample.clone()
        };

        // LIF update — as in `Leaky::step_with_beta` (reset mode — tensor threshold).
        let beta = self.beta_tensor(dims);
        let mem_new = state.mem.mem.clone().mul(beta) + input;

        let (spike, spike_hard) =
            spike_with_threshold(mem_new.clone(), threshold.clone(), surrogate);

        let mem_reset = apply_reset_tensor(mem_new, &spike_hard, &threshold, self.reset_mode);

        (
            spike,
            StochasticThresholdState {
                mem: NeuronState {
                    mem: mem_reset,
                    syn: state.mem.syn.clone(),
                },
                threshold_sample: threshold,
                rng: state.rng,
            },
        )
    }
}

impl SpikingLayer for StochasticThresholdLeaky {
    type Input = Tensor<2>;
    type Output = Tensor<2>;
    type State = Deferred<StochasticThresholdState>;

    fn init_state(&self, batch: usize, _device: &Device) -> Self::State {
        Deferred::pending(batch)
    }

    fn step<S: Surrogate + ?Sized>(
        &self,
        x: Tensor<2>,
        state: Self::State,
        surrogate: &S,
    ) -> (Tensor<2>, Self::State) {
        let [batch, units] = x.dims();
        let device = x.device();
        let state = state.resolve(|b| {
            assert_eq!(
                b,
                batch,
                "StochasticThresholdLeaky: the input batch ({batch}) did not match the init_state batch ({b})"
            );
            StochasticThresholdState::zeros_mem_only(&device, batch, units, RngSource::new(self.seed))
        });
        let (y, next) = StochasticThresholdLeaky::step(self, x, &state, surrogate);
        (y, Deferred::Ready(next))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::surrogate::Triangular;

    fn dev() -> Device {
        Device::flex()
    }

    fn config(seed: Option<u64>) -> StochasticThresholdLeakyConfig {
        StochasticThresholdLeakyConfig {
            seed,
            ..StochasticThresholdLeakyConfig::new(0.9)
        }
    }

    #[test]
    fn same_seed_reproduces_the_run() {
        let device = dev();
        let config = config(Some(7));
        let layer = StochasticThresholdLeaky::new(&config, &device);
        let x = Tensor::<1>::from_floats(vec![2.0].as_slice(), &device).reshape([1, 1]);
        let mut spikes_a = Vec::new();
        let mut spikes_b = Vec::new();
        for _ in 0..10 {
            let state = StochasticThresholdState::zeros_mem_only(
                &device,
                1,
                1,
                RngSource::new(config.seed),
            );
            let (y, _) = layer.step(x.clone(), &state, &Triangular::new(1.0));
            spikes_a.push(y.into_data().try_to_vec::<f32>().unwrap()[0]);
            let state = StochasticThresholdState::zeros_mem_only(
                &device,
                1,
                1,
                RngSource::new(config.seed),
            );
            let (y, _) = layer.step(x.clone(), &state, &Triangular::new(1.0));
            spikes_b.push(y.into_data().try_to_vec::<f32>().unwrap()[0]);
        }
        assert_eq!(spikes_a, spikes_b, "same seed must reproduce the spikes");
    }

    #[test]
    fn threshold_is_sampled_each_step() {
        let device = dev();
        let layer = StochasticThresholdLeaky::new(&config(Some(3)), &device);
        let mut rng = RngSource::new(Some(3));
        let samples: Vec<f32> = (0..16)
            .map(|_| {
                layer
                    .sample_threshold([1, 1], &device, &mut rng)
                    .into_data()
                    .try_to_vec::<f32>()
                    .unwrap()[0]
            })
            .collect();
        assert!(
            samples.iter().any(|s| *s != samples[0]),
            "thresholds should vary per step"
        );
    }

    #[test]
    fn layer_params_match_the_config_distribution() {
        let device = dev();
        let config = StochasticThresholdLeakyConfig {
            distribution: ThresholdDistribution::new(2.5, 0.75),
            ..StochasticThresholdLeakyConfig::new(0.9)
        };
        let layer = StochasticThresholdLeaky::new(&config, &device);
        assert!((layer.threshold_mean_value() - 2.5).abs() < 1e-6);
        let std = layer.threshold_std_value();
        assert!((std - 0.75).abs() < 1e-5, "std {std}");
    }

    #[test]
    fn spikes_are_binary() {
        let device = dev();
        let layer = StochasticThresholdLeaky::new(&config(Some(11)), &device);
        let state =
            StochasticThresholdState::zeros_mem_only(&device, 1, 1, RngSource::new(Some(11)));
        let x = Tensor::<1>::from_floats(vec![0.0].as_slice(), &device).reshape([1, 1]);
        let (y, _) = layer.step(x, &state, &Triangular::new(1.0));
        let v = y.into_data().try_to_vec::<f32>().unwrap();
        assert!(v[0] == 0.0 || v[0] == 1.0);
    }

    #[test]
    fn spiking_layer_trait_is_implemented() {
        // The `SpikingLayer` path — what `Session` and the book examples use.
        let device = dev();
        let layer = StochasticThresholdLeaky::new(&config(Some(5)), &device);
        let x = Tensor::<1>::from_floats(vec![0.0].as_slice(), &device).reshape([1, 1]);
        let state = <StochasticThresholdLeaky as SpikingLayer>::init_state(&layer, 1, &device);
        let (y, next) = <StochasticThresholdLeaky as SpikingLayer>::step(
            &layer,
            x,
            state,
            &Triangular::new(1.0),
        );
        assert_eq!(y.dims(), [1, 1]);
        let _ = next;
    }
}
