//! Adaptive Leaky Integrate-and-Fire (ALIF): LIF with an adaptation current
//! (spike-frequency adaptation, SFA) — a biologically plausible
//! reduction of the firing rate under sustained stimulation.
//!
//! Dynamics (forward-Euler discretization, the step `dt` is implicitly folded
//! into β/ρ, as in the other neurons of the crate — see [`super::leaky`]):
//!
//! ```text
//! mem' = β·mem + I - w        (membrane: leak + input − adaptation)
//! w'   = ρ·w + a·spike_hard   (adaptation current: its own leak + jump on a spike)
//! ```
//!
//! where `β, ρ ∈ (0, 1)` are the decay coefficients of the membrane and the
//! adaptation current, and `a ≥ 0` is the adaptation increment per spike. This
//! is the discrete analogue of GLIF2 / "LIF + adaptive threshold current" from
//! computational neuroscience (Brette & Gerstner 2005 §2 without the
//! exponential term; Bellec et al. 2018, "ALIF" in the context of SNN
//! training) and of what `snnTorch` does not provide as a separate model —
//! there adaptation is obtained only manually via `Synaptic`/`RLeaky` with tuned
//! parameters. Here adaptation is an explicit state variable with its own
//! leak and its own increment, which is closer to the neurobiological
//! interpretation ("slow potassium current") and lets the model
//! exhibit spike-frequency adaptation (a drop in the instantaneous rate under
//! constant input) and rebound-like behavior when the input is switched off.
//!
//! # Difference from [`RLeaky`](crate::neuron::RLeaky)
//!
//! `RLeaky` mixes into the input a recurrent connection from its OWN spike
//! of the previous step (`V(t) = β·V(t-1) + I + V_rec·S(t-1)`) — this is
//! an instantaneous contribution to the membrane on the next step, without
//! its own decay dynamics. `AdaptiveLeaky` instead subtracts from the membrane
//! a separate variable `w` with its own (usually much slower) time
//! constant `ρ`, accumulating over spikes — that is, the effect of a firing
//! event is stretched over many steps forward and suppresses rather than excites
//! the membrane. These are different biophysical mechanisms: recurrent
//! excitation/inhibition versus slow potassium adaptation.

use burn::module::{Module, Param};
use burn::tensor::Device;
use burn::tensor::Tensor;
use burn::tensor::activation::{sigmoid, softplus};

use super::{logit, spike_with_scalar_threshold};
use crate::state::{NeuronState, ResetMode, apply_reset};
use crate::surrogate::Surrogate;

/// ALIF neuron configuration.
#[derive(Debug, Clone, Copy)]
pub struct AdaptiveLeakyConfig {
    /// Initial membrane decay β ∈ (0, 1).
    pub beta_init: f32,
    /// Initial adaptation current decay ρ ∈ (0, 1).
    ///
    /// Biologically this is a slow variable — `rho_init` is usually
    /// much closer to 1 than `beta_init` (the adaptation time
    /// constant is on the order of hundreds of steps versus units-tens for the membrane).
    pub rho_init: f32,
    /// Adaptation current increment per spike, `a ≥ 0`.
    ///
    /// Positivity is enforced by softplus in the layer itself (see
    /// [`AdaptiveLeaky::adapt_increment_tensor`]) — here any finite value is
    /// allowed, and clamping/inversion happen when the learnable parameter is
    /// built, similarly to `threshold` of
    /// [`LeakyConfig`](super::LeakyConfig).
    pub adapt_increment: f32,
    /// Make the adaptation increment `a` learnable (positivity via softplus).
    pub learn_adapt_increment: bool,
    /// Spike threshold.
    pub threshold: f32,
    /// Membrane reset mode after a spike.
    pub reset_mode: ResetMode,
}

impl Default for AdaptiveLeakyConfig {
    fn default() -> Self {
        Self {
            beta_init: 0.9,
            rho_init: 0.95,
            adapt_increment: 0.1,
            learn_adapt_increment: false,
            threshold: 1.0,
            reset_mode: ResetMode::Subtract,
        }
    }
}

impl AdaptiveLeakyConfig {
    /// New configuration with the given β (membrane) and ρ (adaptation).
    pub fn new(beta_init: f32, rho_init: f32) -> Self {
        Self {
            beta_init,
            rho_init,
            ..Default::default()
        }
    }

    /// Set the adaptation current increment per spike.
    pub fn with_adapt_increment(mut self, a: f32) -> Self {
        self.adapt_increment = a;
        self
    }

    /// Make the adaptation increment learnable.
    pub fn with_learn_adapt_increment(mut self, learn: bool) -> Self {
        self.learn_adapt_increment = learn;
        self
    }

    /// Set the threshold.
    pub fn with_threshold(mut self, threshold: f32) -> Self {
        self.threshold = threshold;
        self
    }

    /// Set the reset mode.
    pub fn with_reset_mode(mut self, mode: ResetMode) -> Self {
        self.reset_mode = mode;
        self
    }

    /// Create a layer.
    pub fn init(&self, device: &Device) -> AdaptiveLeaky {
        AdaptiveLeaky::new(self, device)
    }
}

/// Softplus inversion: finds `raw` such that `softplus(raw) == y`.
///
/// The same trick as with `threshold_param`/`graded_factor_param` in
/// [`Leaky`](super::Leaky): `raw = ln(exp(y) - 1)`, clamped from below at
/// `1e-3` (so that `y == 0` does not yield `ln(0) == -inf`) and with a shortcut for
/// large `y`, where softplus is numerically indistinguishable from identity.
fn inverse_softplus(y: f32) -> f32 {
    let y = y.max(1e-3);
    if y > 20.0 {
        y
    } else {
        (y.exp() - 1.0).max(1e-6).ln()
    }
}

/// Adaptive LIF layer: LIF with a slow adaptation current that
/// suppresses the membrane in proportion to recent spiking activity.
#[derive(Module, Debug)]
pub struct AdaptiveLeaky {
    /// Raw β: `sigmoid(beta)` — membrane decay in (0, 1).
    pub beta: Param<Tensor<1>>,
    /// Raw ρ: `sigmoid(rho)` — adaptation current decay in (0, 1).
    pub rho: Param<Tensor<1>>,
    /// Fixed adaptation increment per spike (when not learned).
    #[module(skip)]
    pub adapt_increment: f32,
    /// Raw learnable adaptation increment: `softplus(raw)` yields `a ≥ 0`.
    pub adapt_increment_param: Option<Param<Tensor<1>>>,
    /// Spike threshold.
    pub threshold: f32,
    /// Reset mode (not part of the learnable record).
    #[module(skip)]
    pub reset_mode: ResetMode,
}

impl AdaptiveLeaky {
    /// Create an ALIF layer from a configuration.
    pub fn new(config: &AdaptiveLeakyConfig, device: &Device) -> Self {
        let beta = Param::from_tensor(Tensor::from_floats([logit(config.beta_init)], device));
        let rho = Param::from_tensor(Tensor::from_floats([logit(config.rho_init)], device));

        let adapt_increment_param = if config.learn_adapt_increment {
            let raw = inverse_softplus(config.adapt_increment);
            Some(Param::from_tensor(Tensor::from_floats([raw], device)))
        } else {
            None
        };

        Self {
            beta,
            rho,
            // Clamped here as well (not only when building the learnable
            // parameter above), for the same reason as
            // `Leaky::graded_spikes_factor`: `AdaptiveLeakyConfig` can be
            // assembled with a struct literal bypassing the setters, and the
            // invariant "the adaptation increment is non-negative" must hold in any
            // case — otherwise a spike would decrease future adaptation instead of
            // increasing it, which is physically meaningless (the adaptation current
            // is defined as monotonically growing with activity).
            adapt_increment: config.adapt_increment.max(0.0),
            adapt_increment_param,
            threshold: config.threshold,
            reset_mode: config.reset_mode,
        }
    }

    /// Adaptation increment as a tensor of shape `[1]`. Always non-negative.
    pub fn adapt_increment_tensor(&self, device: &Device) -> Tensor<1> {
        match &self.adapt_increment_param {
            Some(p) => softplus(p.val(), 1.0),
            None => Tensor::<1>::from_floats([self.adapt_increment], device),
        }
    }

    /// One time step.
    ///
    /// `input` is `[batch, units]`, `state.mem` is the membrane `[batch, units]`,
    /// and `state.syn` is used as the slot for the adaptation current `w`
    /// (we reuse the existing `NeuronState` field instead of introducing a new
    /// state type: semantically this is a separate decaying variable,
    /// like the synaptic current in [`Synaptic`](super::Synaptic), only with a
    /// different update equation — the increment follows the SPIKE rather than the
    /// input). `None` is treated as zero initial adaptation — the same
    /// contract as `Synaptic::step`.
    ///
    /// Returns `(spikes, new state)`; `syn` of the returned
    /// state holds the updated `w`.
    pub fn step<S: Surrogate + ?Sized>(
        &self,
        input: Tensor<2>,
        state: &NeuronState,
        surrogate: &S,
    ) -> (Tensor<2>, NeuronState) {
        let dims = state.mem.dims();
        let device = state.mem.device();

        let w = state
            .syn
            .clone()
            .unwrap_or_else(|| Tensor::zeros(dims, &device));

        let beta = sigmoid(self.beta.val()).reshape([1, 1]).expand(dims);
        let rho = sigmoid(self.rho.val()).reshape([1, 1]).expand(dims);

        // The membrane is charged by the input and discharged by the adaptation current
        // of the PREVIOUS step — `w` has not yet been updated by the arrival of the current
        // spike, otherwise the effect would be instantaneous (in the same step when
        // the neuron first fired) instead of delayed, as required by the
        // biophysics of the slow potassium current.
        let mem_new = state.mem.clone().mul(beta) + input - w.clone();

        let (spike, spike_hard) =
            spike_with_scalar_threshold(mem_new.clone(), self.threshold, surrogate);
        let mem_reset = apply_reset(mem_new, &spike_hard, self.threshold, self.reset_mode);

        // Adaptation is updated from the HARD (non-differentiable Heaviside)
        // spike, detached from the graph — just as the membrane reset
        // uses `spike_hard` rather than the surrogate `spike`.
        // The learnable gradient with respect to `a` (when `learn_adapt_increment == true`)
        // is not lost: `adapt_increment_tensor` itself
        // remains part of the graph and is multiplied by the already detached `spike_hard`.
        let a = self
            .adapt_increment_tensor(&device)
            .reshape([1, 1])
            .expand(dims);
        let w_new = w.mul(rho) + spike_hard.mul(a);

        (
            spike,
            NeuronState {
                mem: mem_reset,
                syn: Some(w_new),
            },
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::Device;

    fn dev() -> Device {
        Device::flex()
    }

    fn t2(data: Vec<f32>, shape: [usize; 2]) -> Tensor<2> {
        Tensor::<1>::from_floats(data.as_slice(), &dev()).reshape(shape)
    }

    fn vec2(t: Tensor<2>) -> Vec<f32> {
        t.into_data().try_to_vec::<f32>().unwrap()
    }

    /// With no input and zero initial adaptation the neuron must not
    /// spike spontaneously, and the membrane must only decay toward zero.
    #[test]
    fn zero_input_decays_without_spiking() {
        let device = dev();
        let config = AdaptiveLeakyConfig::new(0.9, 0.95).with_threshold(1.0);
        let layer = AdaptiveLeaky::new(&config, &device);

        let mut state = NeuronState::zeros(&device, 1, 1);
        state.mem = t2(vec![0.5], [1, 1]);

        for _ in 0..5 {
            let (spike, new_state) = layer.step(
                t2(vec![0.0], [1, 1]),
                &state,
                &crate::surrogate::ATan::default(),
            );
            state = new_state;
            assert_eq!(vec2(spike), vec![0.0]);
        }
        let mem = vec2(state.mem);
        assert!(
            mem[0] < 0.5 && mem[0] >= 0.0,
            "the membrane must decay monotonically: {}",
            mem[0]
        );
    }

    /// A negative `adapt_increment` supplied through a struct literal,
    /// bypassing the setter, must be clamped to 0 in `new` rather than
    /// passed through as is — otherwise a spike would decrease adaptation.
    #[test]
    fn negative_adapt_increment_is_clamped_to_zero() {
        let device = dev();
        let config = AdaptiveLeakyConfig {
            adapt_increment: -3.0,
            ..AdaptiveLeakyConfig::new(0.9, 0.95)
        };
        let layer = AdaptiveLeaky::new(&config, &device);
        let a = layer
            .adapt_increment_tensor(&device)
            .into_data()
            .try_to_vec::<f32>()
            .unwrap();
        assert!(
            a[0] >= 0.0,
            "the adaptation increment must be non-negative: {}",
            a[0]
        );
    }

    /// A sustained constant input sufficient for repeated firing must
    /// eventually REDUCE the spike rate due to the accumulation of `w`
    /// (spike-frequency adaptation) compared with the rate in the first steps.
    #[test]
    fn sustained_input_reduces_firing_rate_over_time() {
        let device = dev();
        // Large adaptation increment so that the effect is unmistakably visible
        // over the small number of test steps.
        let config = AdaptiveLeakyConfig::new(0.9, 0.95)
            .with_threshold(1.0)
            .with_adapt_increment(0.8);
        let layer = AdaptiveLeaky::new(&config, &device);
        let surrogate = crate::surrogate::ATan::default();

        let mut state = NeuronState::zeros(&device, 1, 1);
        let input = t2(vec![0.5], [1, 1]);

        let mut spikes_first_half = 0.0f32;
        let mut spikes_second_half = 0.0f32;
        let n = 40;
        for i in 0..n {
            let (spike, new_state) = layer.step(input.clone(), &state, &surrogate);
            state = new_state;
            let s = vec2(spike)[0];
            if i < n / 2 {
                spikes_first_half += s;
            } else {
                spikes_second_half += s;
            }
        }

        assert!(
            spikes_second_half <= spikes_first_half,
            "the firing rate in the second half ({spikes_second_half}) must not exceed that in the first ({spikes_first_half}) — adaptation must suppress sustained stimulation"
        );
    }

    /// `learn_adapt_increment=false` (the default) does not create a
    /// learnable parameter beyond the unconditional `beta`/`rho` — the same
    /// contract as `threshold_param`/`graded_factor_param` in [`Leaky`](super::Leaky).
    #[test]
    fn fixed_mode_has_no_extra_learnable_param() {
        let device = dev();
        let config = AdaptiveLeakyConfig::new(0.9, 0.95);
        let layer = AdaptiveLeaky::new(&config, &device);
        assert!(layer.adapt_increment_param.is_none());
    }
}
