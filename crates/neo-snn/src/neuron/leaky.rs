//! Leaky Integrate-and-Fire (LIF): `U' = β·U + I`, Heaviside spike,
//! customizable reset mode.

use burn::module::{Module, Param};
use burn::tensor::Device;
use burn::tensor::Tensor;
use burn::tensor::activation::{sigmoid, softplus};

use super::{apply_graded_spikes, logit, spike_with_scalar_threshold, spike_with_threshold};
use crate::functional::winner_take_all;
use crate::state::{
    NeuronState, ResetMode, ResetOrder, apply_reset, apply_reset_tensor, decay_then_reset,
    decay_then_reset_tensor, hard_spike, hard_spike_tensor,
};
use crate::surrogate::Surrogate;

/// Configuration of the LIF neuron.
#[derive(Debug, Clone, Copy)]
pub struct LeakyConfig {
    /// The initial damping of the membrane is β ∈ (0, 1).
    pub beta_init: f32,
    /// Spike threshold (used when `learn_threshold == false`).
    pub threshold: f32,
    /// Make the threshold trainable (positivity is ensured by softplus).
    pub learn_threshold: bool,
    /// Reset mode after spike.
    pub reset_mode: ResetMode,
    /// For the order of "reset ↔ decay" in a step, see [`ResetOrder`]. Default
    /// `BeforeDecay`; `AfterDecay` matches snnTorch dynamics.
    pub reset_order: ResetOrder,
    /// Winner-take-all inside the layer: only the strongest neuron is spiked.
    ///
    /// Analogue of `inhibition=True` from snntorch. Suppression is applied to the
    /// spikes already produced along the membrane of the SAME step, see
    /// [`winner_take_all`](crate::functional::winner_take_all).
    pub inhibition: bool,
    /// Initial spike amplitude multiplier (graded spikes, snntorch:
    /// `graded_spikes_factor`). `1.0` - normal binary spike, behavior
    /// is no different from not having this option.
    ///
    /// Multiplies a ready-made straight-through spike, NOT a membrane or
    /// threshold: the fact of the spike remains Heaviside, multiplied only
    /// the amount that the layer projects outward. See
    /// [`apply_graded_spikes`](crate::neuron::apply_graded_spikes).
    pub graded_spikes_factor: f32,
    /// Make the amplitude multiplier trainable (positive - the same
    /// softplus, the same as the trainable threshold).
    pub learn_graded_spikes_factor: bool,
}

impl Default for LeakyConfig {
    fn default() -> Self {
        Self {
            beta_init: 0.9,
            threshold: 1.0,
            learn_threshold: false,
            reset_mode: ResetMode::Subtract,
            reset_order: ResetOrder::BeforeDecay,
            inhibition: false,
            graded_spikes_factor: 1.0,
            learn_graded_spikes_factor: false,
        }
    }
}

impl LeakyConfig {
    /// New configuration with given β.
    pub fn new(beta_init: f32) -> Self {
        Self {
            beta_init,
            ..Default::default()
        }
    }

    /// Set the threshold.
    pub fn with_threshold(mut self, threshold: f32) -> Self {
        self.threshold = threshold;
        self
    }

    /// Make the threshold trainable.
    pub fn with_learn_threshold(mut self, learn: bool) -> Self {
        self.learn_threshold = learn;
        self
    }

    /// Set reset mode.
    pub fn with_reset_mode(mut self, mode: ResetMode) -> Self {
        self.reset_mode = mode;
        self
    }

    /// Set the order “reset ↔ attenuation” (see [`ResetOrder`]).
    pub fn with_reset_order(mut self, order: ResetOrder) -> Self {
        self.reset_order = order;
        self
    }

    /// Enable winner-take-all inside the layer.
    pub fn with_inhibition(mut self, inhibition: bool) -> Self {
        self.inhibition = inhibition;
        self
    }

    /// Set the spike amplitude multiplier (graded spikes).
    ///
    /// `factor` is clamped from below onto `1e-3` here, in the setter - and not
    /// only inside softplus inversion for learnable mode. Without this clamp the
    /// value would be copied as is when `learn_graded_spikes_factor == false`,
    /// and `with_graded_spikes_factor(-5.0)` would give a negative spike
    /// amplitude, although [`LeakyConfig::graded_spikes_factor`] and
    /// [`Leaky::graded_factor_tensor`] promise a strictly positive multiplier.
    /// The invariant "the multiplier is always positive" holds independently of
    /// `learn_graded_spikes_factor`.
    pub fn with_graded_spikes_factor(mut self, factor: f32) -> Self {
        self.graded_spikes_factor = factor.max(1e-3);
        self
    }

    /// Make the spike amplitude multiplier trainable.
    pub fn with_learn_graded_spikes_factor(mut self, learn: bool) -> Self {
        self.learn_graded_spikes_factor = learn;
        self
    }

    /// Create a layer.
    pub fn init(&self, device: &Device) -> Leaky {
        Leaky::new(self, device)
    }
}

/// Leaky Integrate-and-Fire layer. Trainable β, optional trainable threshold.
#[derive(Module, Debug)]
pub struct Leaky {
    /// Raw β: `sigmoid(beta)` - damping at (0, 1). shape `[1]`.
    pub beta: Param<Tensor<1>>,
    /// Fixed threshold (when the threshold is not trained).
    pub threshold: f32,
    /// Raw trainable threshold: `softplus(raw)` gives V_th.
    pub threshold_param: Option<Param<Tensor<1>>>,
    /// Reset mode (not included in the learning record).
    #[module(skip)]
    pub reset_mode: ResetMode,
    /// The order is “reset ↔ attenuation” (not included in the trainable record).
    #[module(skip)]
    pub reset_order: ResetOrder,
    /// Winner-take-all inside the layer.
    #[module(skip)]
    pub inhibition: bool,
    /// Fixed spike amplitude multiplier (when not learning).
    #[module(skip)]
    pub graded_spikes_factor: f32,
    /// Raw trainable amplitude multiplier: `softplus(raw)` gives the factor.
    /// `None` with `learn_graded_spikes_factor == false` - the same technique,
    /// same as `threshold_param`.
    pub graded_factor_param: Option<Param<Tensor<1>>>,
}

impl Leaky {
    /// Create a LIF layer from the configuration.
    pub fn new(config: &LeakyConfig, device: &Device) -> Self {
        let beta = Param::from_tensor(Tensor::from_floats([logit(config.beta_init)], device));

        let threshold_param = if config.learn_threshold {
            // softplus(raw) = threshold  =>  raw = ln(exp(threshold) - 1)
            let th = config.threshold.max(1e-3);
            let raw = if th > 20.0 {
                th // softplus ≈ identity for large values
            } else {
                (th.exp() - 1.0).max(1e-6).ln()
            };
            Some(Param::from_tensor(Tensor::from_floats([raw], device)))
        } else {
            None
        };

        let graded_factor_param = if config.learn_graded_spikes_factor {
            // Same softplus inversion as threshold_param above.
            let f = config.graded_spikes_factor.max(1e-3);
            let raw = if f > 20.0 {
                f
            } else {
                (f.exp() - 1.0).max(1e-6).ln()
            };
            Some(Param::from_tensor(Tensor::from_floats([raw], device)))
        } else {
            None
        };

        Self {
            beta,
            threshold: config.threshold,
            threshold_param,
            reset_mode: config.reset_mode,
            reset_order: config.reset_order,
            inhibition: config.inhibition,
            // It is clamped here too, and not only in `with_graded_spikes_factor`:
            // `LeakyConfig` is a simple structure (`Debug, Clone, Copy`), and its
            // fields can be set directly with the literal `LeakyConfig { .. }`,
            // bypassing the setter. `graded_factor_tensor`/`apply_graded_spikes`
            // documented as giving a strictly positive amplitude -
            // the invariant must hold regardless of the method
            // constructing the config.
            graded_spikes_factor: config.graded_spikes_factor.max(1e-3),
            graded_factor_param,
        }
    }

    /// Threshold as a tensor of the shape `dims`.
    ///
    /// The threshold stays a tensor even when it is trainable: reading it on
    /// the CPU via `into_data()` for `ResetMode::Subtract` would force a
    /// GPU→CPU synchronization at every time step.
    ///
    /// `pub` (not only for `Leaky::step`) - the same tensor is needed by
    /// [`LeakyParallel`](crate::neuron::LeakyParallel) for vectorized paths:
    /// it applies the threshold to the entire sequence at once
    /// `[batch·time, features]` per call instead of `T` calls inside `step`,
    /// so it must be built with the same formulas as here rather than
    /// duplicating the softplus parameterization.
    pub fn threshold_tensor(&self, device: &Device, dims: [usize; 2]) -> Tensor<2> {
        match &self.threshold_param {
            Some(p) => softplus(p.val(), 1.0).reshape([1, 1]).expand(dims),
            None => Tensor::<1>::from_floats([self.threshold], device)
                .reshape([1, 1])
                .expand(dims),
        }
    }

    /// Spike amplitude multiplier as a tensor of the shape `[1]`.
    ///
    /// Always positive (see [`LeakyConfig::graded_spikes_factor`]),
    /// like the threshold - parameterization via softplus is the same as for
    /// [`threshold_tensor`](Self::threshold_tensor), but without `expand`:
    /// [`apply_graded_spikes`] itself expands the shape to `[batch, units]`.
    pub fn graded_factor_tensor(&self, device: &Device) -> Tensor<1> {
        match &self.graded_factor_param {
            Some(p) => softplus(p.val(), 1.0),
            None => Tensor::<1>::from_floats([self.graded_spikes_factor], device),
        }
    }

    /// Damping β = `sigmoid(beta)` as the tensor `dims`.
    ///
    /// Does not depend on time: in a loop, step by step, calculate once and
    /// pass to [`step_with_beta`](Self::step_with_beta).
    pub fn beta_tensor(&self, dims: [usize; 2]) -> Tensor<2> {
        sigmoid(self.beta.val()).reshape([1, 1]).expand(dims)
    }

    /// One step in time.
    ///
    /// `input` — `[batch, units]`, `state.mem` — `[batch, units]`.
    /// Returns `(spikes, new state)`.
    ///
    /// The returned spikes are multiplied by the amplitude multiplier (graded
    /// spikes); with default values it is exactly a binary spike.
    ///
    /// For a long sequence in a manual cycle it is more profitable
    /// [`step_with_beta`](Self::step_with_beta) with β computed once.
    ///
    /// # Cutting the input in a manual loop
    ///
    /// Don't cut `[batch, time, features]` through `x.slice(..)` inside a loop: backward for `slice`
    /// constructs a zero tensor of the size of the entire input at EACH step, this is `O(T²)` and at long
    /// sequences more than all other work. Slice the input once:
    ///
    /// ```ignore
    /// let beta = lif.beta_tensor([batch, features]);
    /// for x_t in neo_snn::neuron::split_time(x) {
    ///     let (spike, new_state) = lif.step_with_beta(x_t, &state, &beta, &surrogate);
    ///     state = new_state;
    ///     // ...
    /// }
    /// ```
    ///
    /// If the loop is not needed, [`MultiStepLeaky`](crate::neuron::MultiStepLeaky) does the same thing faster.
    pub fn step<S: Surrogate + ?Sized>(
        &self,
        input: Tensor<2>,
        state: &NeuronState,
        surrogate: &S,
    ) -> (Tensor<2>, NeuronState) {
        let beta = self.beta_tensor(state.mem.dims());
        self.step_with_beta(input, state, &beta, surrogate)
    }

    /// Like [`step`](Self::step), but with ready-made β (`[batch, units]`, see
    /// [`beta_tensor`](Self::beta_tensor)): `sigmoid` is not recalculated
    /// at every step.
    pub fn step_with_beta<S: Surrogate + ?Sized>(
        &self,
        input: Tensor<2>,
        state: &NeuronState,
        beta: &Tensor<2>,
        surrogate: &S,
    ) -> (Tensor<2>, NeuronState) {
        if self.reset_order == ResetOrder::AfterDecay {
            return self.step_decay_then_reset(input, state, beta, surrogate);
        }

        let dims = state.mem.dims();
        let device = state.mem.device();

        let mem_new = state.mem.clone().mul(beta.clone()) + input;

        // The fixed threshold follows a scalar path: none
        // `from_floats` (host→device loading) at each step. Tensor
        // the path is needed only for the trainable threshold.
        let (spike, spike_hard, th) = match &self.threshold_param {
            None => {
                let (s, h) =
                    spike_with_scalar_threshold(mem_new.clone(), self.threshold, surrogate);
                (s, h, None)
            }
            Some(_) => {
                let th = self.threshold_tensor(&device, dims);
                let (s, h) = spike_with_threshold(mem_new.clone(), th.clone(), surrogate);
                (s, h, Some(th))
            }
        };

        // Suppression is applied BEFORE reset: the membrane must be reset
        // only the neuron that actually spiked.
        let (spike, spike_hard) = if self.inhibition {
            (
                winner_take_all(spike, mem_new.clone()),
                winner_take_all(spike_hard, mem_new.clone()),
            )
        } else {
            (spike, spike_hard)
        };

        // Reset - by UNgraded spike_hard.
        let mem_reset = match &th {
            Some(th) => apply_reset_tensor(mem_new, &spike_hard, th, self.reset_mode),
            None => apply_reset(mem_new, &spike_hard, self.threshold, self.reset_mode),
        };

        // Amplitude is the last. Fixed multiplier - scalar
        // (without downloading to the device), trained by a tensor.
        let spike = match &self.graded_factor_param {
            Some(_) => apply_graded_spikes(spike, self.graded_factor_tensor(&device)),
            None if self.graded_spikes_factor == 1.0 => spike,
            None => spike.mul_scalar(self.graded_spikes_factor),
        };

        (
            spike,
            NeuronState {
                mem: mem_reset,
                syn: state.syn.clone(),
            },
        )
    }

    /// Step with order `ResetOrder::AfterDecay`: `U' = β·U − θ·S_prev + I`.
    ///
    /// In `state.mem` lies the membrane BEFORE the reset; reset of the previous spike `S_prev = (U > θ)`
    /// (with winner-take-all at `inhibition`) is applied here, after fading. New
    /// the membrane goes back to the non-reset state. Spike, suppression and amplitude are computed
    /// by the same formulas as in `ResetOrder::BeforeDecay`.
    fn step_decay_then_reset<S: Surrogate + ?Sized>(
        &self,
        input: Tensor<2>,
        state: &NeuronState,
        beta: &Tensor<2>,
        surrogate: &S,
    ) -> (Tensor<2>, NeuronState) {
        let dims = state.mem.dims();
        let device = state.mem.device();

        // The threshold as a tensor is only needed for a trainable threshold (otherwise a scalar path).
        let th = self
            .threshold_param
            .as_ref()
            .map(|_| self.threshold_tensor(&device, dims));

        let mem_decayed = if self.reset_mode == ResetMode::None {
            state.mem.clone().mul(beta.clone())
        } else {
            let pending = match &th {
                Some(th) => hard_spike_tensor(&state.mem, th),
                None => hard_spike(&state.mem, self.threshold),
            };
            // Suppression - along the membrane of the previous step, as with the initial spike:
            // Only the actually spiked neuron is reset.
            let pending = if self.inhibition {
                winner_take_all(pending, state.mem.clone())
            } else {
                pending
            };
            match &th {
                Some(th) => decay_then_reset_tensor(
                    state.mem.clone(),
                    beta.clone(),
                    &pending,
                    th,
                    self.reset_mode,
                ),
                None => decay_then_reset(
                    state.mem.clone(),
                    beta.clone(),
                    &pending,
                    self.threshold,
                    self.reset_mode,
                ),
            }
        };
        let mem_new = mem_decayed + input;

        let (spike, _spike_hard) = match &th {
            Some(th) => spike_with_threshold(mem_new.clone(), th.clone(), surrogate),
            None => spike_with_scalar_threshold(mem_new.clone(), self.threshold, surrogate),
        };
        let spike = if self.inhibition {
            winner_take_all(spike, mem_new.clone())
        } else {
            spike
        };

        let spike = match &self.graded_factor_param {
            Some(_) => apply_graded_spikes(spike, self.graded_factor_tensor(&device)),
            None if self.graded_spikes_factor == 1.0 => spike,
            None => spike.mul_scalar(self.graded_spikes_factor),
        };

        (
            spike,
            NeuronState {
                mem: mem_new,
                syn: state.syn.clone(),
            },
        )
    }
}

#[cfg(test)]
mod graded_spikes_factor_sign_tests {
    use super::*;
    use burn::tensor::Device;

    fn dev() -> Device {
        Device::flex()
    }

    /// A negative `graded_spikes_factor` in fixed mode
    /// (`learn_graded_spikes_factor == false`) must be clamped from below to
    /// `1e-3`, not passed through as is - otherwise the spike amplitude
    /// becomes negative, violating the documented invariant of
    /// `graded_factor_tensor`/`apply_graded_spikes`.
    #[test]
    fn with_graded_spikes_factor_clamps_negative_value() {
        let device = dev();
        let config = LeakyConfig::new(0.9).with_graded_spikes_factor(-5.0);
        let layer = Leaky::new(&config, &device);

        let factor = layer
            .graded_factor_tensor(&device)
            .into_data()
            .try_to_vec::<f32>()
            .unwrap();
        assert!(
            factor[0] > 0.0,
            "the multiplier must remain positive: {}",
            factor[0]
        );
        assert!(
            (factor[0] - 1e-3).abs() < 1e-6,
            "expected clamp at 1e-3, received {}",
            factor[0]
        );
    }

    /// The same invariant must hold even if `LeakyConfig`
    /// constructed directly by a structure literal, bypassing
    /// `with_graded_spikes_factor` - clamping should live in `Leaky::new`,
    /// and not just in the setter.
    #[test]
    fn direct_struct_literal_with_negative_factor_is_also_clamped() {
        let device = dev();
        let config = LeakyConfig {
            graded_spikes_factor: -1.0,
            ..LeakyConfig::new(0.9)
        };
        let layer = Leaky::new(&config, &device);

        let factor = layer
            .graded_factor_tensor(&device)
            .into_data()
            .try_to_vec::<f32>()
            .unwrap();
        assert!(
            factor[0] > 0.0,
            "the multiplier must remain positive: {}",
            factor[0]
        );
    }

    /// Positive values should not be changed by clamping.
    #[test]
    fn positive_factor_is_unaffected() {
        let device = dev();
        let config = LeakyConfig::new(0.9).with_graded_spikes_factor(2.5);
        let layer = Leaky::new(&config, &device);

        let factor = layer
            .graded_factor_tensor(&device)
            .into_data()
            .try_to_vec::<f32>()
            .unwrap();
        assert!(
            (factor[0] - 2.5).abs() < 1e-6,
            "expected 2.5, received {}",
            factor[0]
        );
    }
}
