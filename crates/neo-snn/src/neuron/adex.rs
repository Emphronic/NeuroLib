//! Adaptive Exponential Integrate-and-Fire neuron (AdEx;
//! Brette & Gerstner, 2005): a two-dimensional bio-plausible model in
//! which the firing nonlinearity is an exponential in the membrane equation
//! and adaptation is a separate slow current `w`.
//!
//! ```text
//! C·dV/dt  = −g_L·(V − E_L) + g_L·Δ_T·exp((V − V_T)/Δ_T) − w + I
//! τ_w·dw/dt = a·(V − E_L) − w
//! if V > V_peak:  V ← V_reset,  w ← w + b
//! ```
//!
//! # Units
//!
//! As in the original paper and in Brian2/NEST: `V` is mV, time is ms, `C` is pF,
//! `g_L` and `a` are nS, and currents (`I`, `w`, `b`) are pA. Then `C·dV/dt` and all currents
//! agree in pA without scaling factors.
//!
//! # Numerical scheme
//!
//! One call to [`AdEx::step`] covers `dt` ms (1 by default), split into
//! [`substeps`](AdExConfig::substeps) sub-steps with forward Euler. Firing is
//! checked and reset on EVERY sub-step, not once at the end of the step: otherwise
//! the exponential manages to "blow up" `V`, and `w` (which depends on `V`) receives a
//! giant jump that can no longer be corrected by the reset. The exponent argument
//! is additionally clipped ([`AdExConfig::exp_clip`]), so in one sub-step
//! `V` can grow by at most a finite amount, but not to `inf`.
//!
//! The step spike is "at least one firing per step": `1 − ∏(1 − sₖ)` over sub-steps.
//! In the forward pass this is exactly 0/1; in the backward pass the gradient goes through the surrogate of
//! each sub-step.
//!
//! # Presets
//!
//! [`AdExPreset::BretteGerstner`] — values from the Brian2 example for the
//! Brette & Gerstner paper (regular firing with adaptation). The other presets
//! are taken from the regime table of Naud et al. (2008); when citing, verify the
//! numbers against the original — here they are given to reproduce the qualitative
//! regimes (tonic, adapting, initial burst, regular bursting).

use burn::module::{Module, Param};
use burn::tensor::Device;
use burn::tensor::Tensor;

use super::spike_with_scalar_threshold;
use crate::surrogate::Surrogate;

/// Named AdEx firing regimes.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum AdExPreset {
    /// Brette & Gerstner (2005) example from Brian2: `C=281 pF, g_L=30 nS,
    /// E_L=−70.6 mV, V_T=−50.4 mV, Δ_T=2 mV, a=4 nS, τ_w=144 ms, b=80.5 pA`.
    BretteGerstner,
    /// Tonic spiking: almost no adaptation (`b = 0`).
    Tonic,
    /// Adapting: the firing rate drops under prolonged stimulation.
    Adapting,
    /// Initial burst: a burst of spikes at the start of the stimulus, then sparse firing.
    InitialBurst,
    /// Regular bursting: regular bursts.
    RegularBursting,
}

impl AdExPreset {
    /// Preset configuration.
    pub fn config(self) -> AdExConfig {
        let base = AdExConfig::default();
        match self {
            Self::BretteGerstner => base,
            Self::Tonic => AdExConfig {
                c: 200.0,
                g_l: 10.0,
                e_l: -70.0,
                v_t: -50.0,
                delta_t: 2.0,
                a: 2.0,
                tau_w: 30.0,
                b: 0.0,
                v_reset: -58.0,
                ..base
            },
            Self::Adapting => AdExConfig {
                c: 200.0,
                g_l: 12.0,
                e_l: -70.0,
                v_t: -50.0,
                delta_t: 2.0,
                a: 2.0,
                tau_w: 300.0,
                b: 60.0,
                v_reset: -58.0,
                ..base
            },
            Self::InitialBurst => AdExConfig {
                c: 130.0,
                g_l: 18.0,
                e_l: -58.0,
                v_t: -50.0,
                delta_t: 2.0,
                a: 4.0,
                tau_w: 150.0,
                b: 120.0,
                v_reset: -50.0,
                ..base
            },
            Self::RegularBursting => AdExConfig {
                c: 200.0,
                g_l: 10.0,
                e_l: -58.0,
                v_t: -50.0,
                delta_t: 2.0,
                a: 2.0,
                tau_w: 120.0,
                b: 100.0,
                v_reset: -46.0,
                ..base
            },
        }
    }
}

/// AdEx neuron configuration (units — see the module documentation).
#[derive(Debug, Clone, Copy)]
pub struct AdExConfig {
    /// Membrane capacitance `C`, pF.
    pub c: f32,
    /// Leak conductance `g_L`, nS.
    pub g_l: f32,
    /// Resting potential `E_L`, mV.
    pub e_l: f32,
    /// Exponential threshold `V_T`, mV.
    pub v_t: f32,
    /// Exponential sharpness `Δ_T`, mV (`Δ_T → 0` yields an ordinary LIF).
    pub delta_t: f32,
    /// Subthreshold adaptation `a`, nS.
    pub a: f32,
    /// Adaptation time constant `τ_w`, ms.
    pub tau_w: f32,
    /// Adaptation current increment per spike `b`, pA.
    pub b: f32,
    /// Reset potential `V_reset`, mV.
    pub v_reset: f32,
    /// Spike detection threshold `V_peak`, mV.
    pub v_peak: f32,
    /// Duration of one `step` call, ms.
    pub dt: f32,
    /// Number of Euler sub-steps per `step` (`dt_sub = dt / substeps`).
    pub substeps: usize,
    /// Upper bound on the exponent argument `(V − V_T)/Δ_T` (protection against
    /// `exp` overflow).
    pub exp_clip: f32,
    /// Whether `a`, `b` and `τ_w` are learnable.
    pub learn_params: bool,
}

impl Default for AdExConfig {
    fn default() -> Self {
        Self {
            c: 281.0,
            g_l: 30.0,
            e_l: -70.6,
            v_t: -50.4,
            delta_t: 2.0,
            a: 4.0,
            tau_w: 144.0,
            b: 80.5,
            v_reset: -70.6,
            v_peak: 0.0,
            dt: 1.0,
            substeps: 4,
            exp_clip: 20.0,
            learn_params: false,
        }
    }
}

impl AdExConfig {
    /// Configuration from a preset.
    pub fn from_preset(preset: AdExPreset) -> Self {
        preset.config()
    }

    /// Set the duration of one step, ms.
    pub fn with_dt(mut self, dt: f32) -> Self {
        self.dt = dt;
        self
    }

    /// Set the number of sub-steps (clamped from below at 1).
    pub fn with_substeps(mut self, substeps: usize) -> Self {
        self.substeps = substeps.max(1);
        self
    }

    /// Make `a`, `b`, `τ_w` learnable.
    pub fn with_learn_params(mut self, learn: bool) -> Self {
        self.learn_params = learn;
        self
    }

    /// Create a layer.
    pub fn init(&self, device: &Device) -> AdEx {
        AdEx::new(self, device)
    }
}

/// AdEx state: membrane `v` (mV) and adaptation current `w` (pA).
#[derive(Debug, Clone)]
pub struct AdExState {
    /// Membrane potential `[batch, units]`, mV.
    pub v: Tensor<2>,
    /// Adaptation current `[batch, units]`, pA.
    pub w: Tensor<2>,
}

impl AdExState {
    /// Rest: `v = e_l`, `w = 0`.
    pub fn resting(device: &Device, batch_size: usize, units: usize, e_l: f32) -> Self {
        Self {
            v: Tensor::zeros([batch_size, units], device).add_scalar(e_l),
            w: Tensor::zeros([batch_size, units], device),
        }
    }

    /// State from ready-made tensors.
    pub fn new(v: Tensor<2>, w: Tensor<2>) -> Self {
        Self { v, w }
    }

    /// Detach from the autodiff graph (truncated BPTT).
    pub fn detach(self) -> Self {
        Self {
            v: self.v.detach(),
            w: self.w.detach(),
        }
    }
}

/// AdEx neuron layer.
#[derive(Module, Debug)]
pub struct AdEx {
    /// Subthreshold adaptation `a` (nS).
    pub a: Param<Tensor<1>>,
    /// Adaptation increment per spike `b` (pA).
    pub b: Param<Tensor<1>>,
    /// `ln τ_w`: `τ_w = exp(log_tau_w)` is strictly positive for any value of the
    /// parameter (just as `sigmoid` keeps β in `(0, 1)` for `Leaky`).
    pub log_tau_w: Param<Tensor<1>>,
    /// Capacitance, pF.
    #[module(skip)]
    pub c: f32,
    /// Leak conductance, nS.
    #[module(skip)]
    pub g_l: f32,
    /// Resting potential, mV.
    #[module(skip)]
    pub e_l: f32,
    /// Exponential threshold, mV.
    #[module(skip)]
    pub v_t: f32,
    /// Exponential sharpness, mV.
    #[module(skip)]
    pub delta_t: f32,
    /// Reset potential, mV.
    #[module(skip)]
    pub v_reset: f32,
    /// Spike detection threshold, mV.
    #[module(skip)]
    pub v_peak: f32,
    /// Step duration, ms.
    #[module(skip)]
    pub dt: f32,
    /// Number of sub-steps.
    #[module(skip)]
    pub substeps: usize,
    /// Exponent argument clip.
    #[module(skip)]
    pub exp_clip: f32,
    /// Whether `a`, `b`, `τ_w` are learnable (when `false` they are detached from the graph).
    #[module(skip)]
    pub learn_params: bool,
}

impl AdEx {
    /// Create a layer from a configuration.
    pub fn new(config: &AdExConfig, device: &Device) -> Self {
        Self {
            a: Param::from_tensor(Tensor::from_floats([config.a], device)),
            b: Param::from_tensor(Tensor::from_floats([config.b], device)),
            log_tau_w: Param::from_tensor(Tensor::from_floats(
                [config.tau_w.max(1e-3).ln()],
                device,
            )),
            c: config.c,
            g_l: config.g_l,
            e_l: config.e_l,
            v_t: config.v_t,
            delta_t: config.delta_t.max(1e-3),
            v_reset: config.v_reset,
            v_peak: config.v_peak,
            dt: config.dt,
            substeps: config.substeps.max(1),
            exp_clip: config.exp_clip,
            learn_params: config.learn_params,
        }
    }

    /// `(a, b, τ_w)` of shape `dims`; when `learn_params == false` they are detached from
    /// the graph before use in `step`.
    fn params(&self, dims: [usize; 2]) -> (Tensor<2>, Tensor<2>, Tensor<2>) {
        let maybe_detach =
            |t: Tensor<1>| -> Tensor<1> { if self.learn_params { t } else { t.detach() } };
        let a = maybe_detach(self.a.val()).reshape([1, 1]).expand(dims);
        let b = maybe_detach(self.b.val()).reshape([1, 1]).expand(dims);
        let tau_w = maybe_detach(self.log_tau_w.val())
            .exp()
            .reshape([1, 1])
            .expand(dims);
        (a, b, tau_w)
    }

    /// One step (`dt` ms). `input` is the current `[batch, units]` in pA.
    /// Returns `(spikes 0/1, new state)`.
    pub fn step<S: Surrogate + ?Sized>(
        &self,
        input: Tensor<2>,
        state: &AdExState,
        surrogate: &S,
    ) -> (Tensor<2>, AdExState) {
        let dims = state.v.dims();
        let device = state.v.device();
        let (a, b, tau_w) = self.params(dims);

        let dt = self.dt / self.substeps as f32;
        let inv_c = 1.0 / self.c;
        let inv_delta = 1.0 / self.delta_t;
        // The surrogate operates on the dimensionless axis "mV / Δ_T"; otherwise, with a
        // difference of tens of mV, its gradient would be practically zero.
        let thr = self.v_peak * inv_delta;

        let mut v = state.v.clone();
        let mut w = state.w.clone();
        let mut no_spike: Tensor<2> = Tensor::ones(dims, &device);

        for _ in 0..self.substeps {
            let exp_arg = v
                .clone()
                .sub_scalar(self.v_t)
                .mul_scalar(inv_delta)
                .clamp_max(self.exp_clip);
            let i_exp = exp_arg.exp().mul_scalar(self.g_l * self.delta_t);
            let i_leak = v.clone().sub_scalar(self.e_l).mul_scalar(-self.g_l);
            let dv = (i_leak + i_exp - w.clone() + input.clone()).mul_scalar(inv_c);
            let dw = (a.clone().mul(v.clone().sub_scalar(self.e_l)) - w.clone()).div(tau_w.clone());

            let v_new = v + dv.mul_scalar(dt);
            let w_new = w + dw.mul_scalar(dt);

            let (spike, hard) =
                spike_with_scalar_threshold(v_new.clone().mul_scalar(inv_delta), thr, surrogate);
            no_spike = no_spike.mul(spike.mul_scalar(-1.0).add_scalar(1.0));

            // Reset by the hard spike, without per-element branching.
            let keep = hard.clone().mul_scalar(-1.0).add_scalar(1.0);
            v = v_new.mul(keep) + hard.clone().mul_scalar(self.v_reset);
            w = w_new + hard.mul(b.clone());
        }

        let spike = no_spike.mul_scalar(-1.0).add_scalar(1.0);
        (spike, AdExState { v, w })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::surrogate::ATan;

    fn dev() -> Device {
        Device::flex()
    }

    fn dev_ad() -> Device {
        Device::flex().autodiff()
    }

    fn t2(data: Vec<f32>, shape: [usize; 2], device: &Device) -> Tensor<2> {
        Tensor::<1>::from_floats(data.as_slice(), device).reshape(shape)
    }

    fn val(t: Tensor<2>) -> f32 {
        t.into_data().try_to_vec::<f32>().unwrap()[0]
    }

    /// Run `steps` steps at constant current; returns the step indices
    /// with spikes and the final state.
    fn run(config: AdExConfig, current: f32, steps: usize) -> (Vec<usize>, AdExState) {
        let device = dev();
        let layer = AdEx::new(&config, &device);
        let surrogate = ATan::default();
        let mut state = AdExState::resting(&device, 1, 1, config.e_l);
        let input = t2(vec![current], [1, 1], &device);
        let mut times = Vec::new();
        for t in 0..steps {
            let (s, next) = layer.step(input.clone(), &state, &surrogate);
            state = next;
            if val(s) > 0.5 {
                times.push(t);
            }
            let v = val(state.v.clone());
            assert!(v.is_finite(), "v diverged at step {t}: {v}");
        }
        (times, state)
    }

    /// Without input the neuron stays at rest and does not fire.
    #[test]
    fn rests_without_input() {
        let config = AdExConfig::default();
        let (times, state) = run(config, 0.0, 200);
        assert!(times.is_empty(), "spontaneous spikes: {times:?}");
        let v = val(state.v);
        assert!((v - config.e_l).abs() < 0.5, "v left rest: {v}");
    }

    /// A suprathreshold current causes firing, and adaptation stretches the intervals:
    /// the last inter-spike interval is longer than the first.
    #[test]
    fn spikes_and_adapts_under_constant_current() {
        let (times, state) = run(AdExConfig::default(), 1000.0, 400);
        assert!(times.len() >= 5, "expected ≥5 spikes, got {times:?}");
        let first = times[1] - times[0];
        let last = times[times.len() - 1] - times[times.len() - 2];
        assert!(
            last > first,
            "no frequency adaptation: ISI {first} → {last} ({times:?})"
        );
        assert!(val(state.w) > 0.0, "adaptation current did not grow");
    }

    /// A preset without adaptation (`b = 0`) fires more evenly than the adapting one.
    #[test]
    fn adapting_preset_slows_down_more_than_tonic() {
        let ratio = |preset: AdExPreset, current: f32| -> f32 {
            let (times, _) = run(preset.config(), current, 600);
            assert!(times.len() >= 4, "{preset:?}: too few spikes {times:?}");
            let first = (times[1] - times[0]) as f32;
            let last = (times[times.len() - 1] - times[times.len() - 2]) as f32;
            last / first
        };
        let tonic = ratio(AdExPreset::Tonic, 500.0);
        let adapting = ratio(AdExPreset::Adapting, 500.0);
        assert!(
            adapting > tonic,
            "adapting ({adapting}) must slow down more than tonic ({tonic})"
        );
    }

    /// After a spike `v` returns to `V_reset`, and `w` receives a jump of `b`.
    #[test]
    fn reset_applies_v_reset_and_w_jump() {
        let device = dev();
        let config = AdExConfig::default();
        let layer = AdEx::new(&config, &device);
        let surrogate = ATan::default();
        // Membrane above the exponential threshold: firing will occur on the first sub-step.
        let state = AdExState::new(
            t2(vec![10.0], [1, 1], &device),
            t2(vec![0.0], [1, 1], &device),
        );
        let (spike, next) = layer.step(t2(vec![0.0], [1, 1], &device), &state, &surrogate);
        assert_eq!(val(spike), 1.0);
        let v = val(next.v);
        let w = val(next.w);
        assert!((v - config.v_reset).abs() < 1.5, "v after reset: {v}");
        assert!((w - config.b).abs() < 5.0, "w after reset: {w}");
    }

    /// A huge current does not lead to NaN/inf.
    #[test]
    fn extreme_input_stays_finite() {
        let (_, state) = run(AdExConfig::default(), 1.0e5, 50);
        assert!(val(state.w).is_finite());
    }

    /// Gradient flows to the learnable `a`, `b`, `τ_w`.
    #[test]
    fn gradient_flows_to_learnable_params() {
        let device = dev_ad();
        let config = AdExConfig::default().with_learn_params(true);
        let layer = AdEx::new(&config, &device);
        let surrogate = ATan::default();
        let mut state = AdExState::resting(&device, 1, 1, config.e_l);
        let input = t2(vec![1000.0], [1, 1], &device);
        let mut loss: Option<Tensor<1>> = None;
        for _ in 0..40 {
            let (s, next) = layer.step(input.clone(), &state, &surrogate);
            state = next;
            let term = s.sum() + state.v.clone().sum().mul_scalar(1e-3);
            loss = Some(match loss {
                Some(l) => l + term,
                None => term,
            });
        }
        let grads = loss.unwrap().backward();
        for (name, g) in [
            ("a", layer.a.grad(&grads)),
            ("b", layer.b.grad(&grads)),
            ("log_tau_w", layer.log_tau_w.grad(&grads)),
        ] {
            let g = g.unwrap_or_else(|| panic!("no gradient for {name}"));
            let g = g.into_data().try_to_vec::<f32>().unwrap()[0];
            assert!(g.is_finite(), "gradient for {name} is not finite: {g}");
        }
    }

    /// `learn_params == false` detaches the parameters from the graph.
    #[test]
    fn frozen_params_get_no_gradient() {
        let device = dev_ad();
        let config = AdExConfig::default();
        let layer = AdEx::new(&config, &device);
        let surrogate = ATan::default();
        let state = AdExState::resting(&device, 1, 1, config.e_l);
        let (s, next) = layer.step(t2(vec![1000.0], [1, 1], &device), &state, &surrogate);
        let grads = (s.sum() + next.v.sum()).backward();
        assert!(layer.a.grad(&grads).is_none());
        assert!(layer.b.grad(&grads).is_none());
    }
}
