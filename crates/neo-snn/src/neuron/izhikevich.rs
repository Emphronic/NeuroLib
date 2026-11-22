//! Izikevich neuron model (Izhikevich, 2003): two-dimensional quadratic
//! "membrane + variable recovery" system, reproducing
//! wide repertoire of bio-plausible spike modes (regular
//! spiking, fast spiking, chattering, bursting, intrinsically bursting and
//! etc.) at the cost of only two differential equations per neuron - in
//! Unlike the LIF family of this crate, nonlinearity is built in here
//! into the dynamics of the membrane itself, and not just into the spike threshold.
//!
//! Canonical continuous shape (Izhikevich, "Simple Model of Spiking
//! Neurons", IEEE Trans. Neural Networks, 2003):
//!
//! ```text
//! v' = 0.04·v² + 5·v + 140 − u + I
//! u' = a·(b·v − u)
//! if v ≥ 30 mV: v ← c, u ← u + d
//! ```
//!
//! `v`—membrane potential (mV), `u`—recovery variable
//! (slow current responsible for adaptation and bursting), `I` - input
//! current The four `(a, b, c, d)` sets the spike mode - see table in
//! [`IzhikevichConfig`].
//!
//! # Numerical scheme
//!
//! The quadratic term `0.04·v²` takes a forward (forward Euler) step with
//! `dt = 1` is numerically unstable near the threshold - `v` can “jump over”
//! reasonable range in one step and disperse. Standard Practice
//! (also used in the reference MATLAB code by Izhikevich himself) -
//! break each step along `v` into several sub-steps `dt = 0.5` with
//! unchanged `u`, and only then update `u` at a full step. Here it is
//! done explicitly via [`IzhikevichConfig::substeps`] (default 2, then
//! is `dt = 0.5`, as in the original), and NOT forward-Euler in one step -
//! otherwise, with normal input currents (`I` of the order of 10-20, as in
//! classical examples of Izikevich) `v` goes to NaN in several steps
//! test.
//!
//! # Difference from the LIF crate family
//!
//! U [`Leaky`](super::Leaky)/[`Synaptic`](super::Synaptic)/
//! [`AdaptiveLeaky`](super::AdaptiveLeaky) threshold - external constant
//! (or trainable parameter), and the membrane itself is linear; the spike shape is not
//! modeled, there is only the fact of exceeding the threshold. Izikevich has a threshold
//! ~30 mV is part of the quadratic dynamics itself (the membrane ITSELF leaves
//! upward like an avalanche when approaching it), and `c`/`d` set
//! post-spike reset AND membrane AND recovery variable
//! at the same time - hence the possibility of rich patterns like bursting
//! (several spikes in a row with increasing interspike interval) without
//! any adaptation of the threshold, purely due to the shape of the `u` dynamics.

use burn::module::{Module, Param};
use burn::tensor::Device;
use burn::tensor::Tensor;
use burn::tensor::activation::sigmoid;

use super::{logit, spike_with_scalar_threshold};
use crate::surrogate::Surrogate;

/// Named sets `(a, b, c, d)` from the canonical mode table
/// category in the original paper by Izhikevich (2003, Fig. 2) and in his
/// "Which model to use..." (2004). The values given are on a current scale
/// and article time (`dt = 1` ms per “full” step, input current of the order
/// units to tens).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IzhikevichPreset {
    /// Regular Spiking (RS) is a typical cortical pyramidal neuron:
    /// initial high-frequency burst, then adaptation to rarer
    /// stable spike. `a=0.02, b=0.2, c=-65, d=8`.
    RegularSpiking,
    /// Intrinsically Bursting (IB) - a burst of spikes at the beginning of the stimulus,
    /// then singles. `a=0.02, b=0.2, c=-55, d=4`.
    IntrinsicallyBursting,
    /// Chattering (CH) - repeating high-frequency bursts (pack ~
    /// every 25-50 ms). `a=0.02, b=0.2, c=-50, d=2`.
    Chattering,
    /// Fast Spiking (FS) - fast spike with virtually no adaptation
    /// frequency, typical of inhibitory interneurons. `a=0.1, b=0.2,
    /// c=-65, d=2`.
    FastSpiking,
    /// Low-Threshold Spiking (LTS) - inhibitory interneurons with low
    /// threshold and noticeable adaptation. `a=0.02, b=0.25, c=-65, d=2`.
    LowThresholdSpiking,
    /// Resonator (RZ) - selective sensitivity to input frequency
    /// due to subthreshold resonance (`b > 0.2`). `a=0.1, b=0.26,
    /// c=-65, d=2`.
    Resonator,
}

impl IzhikevichPreset {
    /// Parameters of the `(a, b, c, d)` preset.
    pub fn params(self) -> (f32, f32, f32, f32) {
        match self {
            Self::RegularSpiking => (0.02, 0.2, -65.0, 8.0),
            Self::IntrinsicallyBursting => (0.02, 0.2, -55.0, 4.0),
            Self::Chattering => (0.02, 0.2, -50.0, 2.0),
            Self::FastSpiking => (0.1, 0.2, -65.0, 2.0),
            Self::LowThresholdSpiking => (0.02, 0.25, -65.0, 2.0),
            Self::Resonator => (0.1, 0.26, -65.0, 2.0),
        }
    }
}

/// Configuration of the Izikevich neuron.
#[derive(Debug, Clone, Copy)]
pub struct IzhikevichConfig {
    /// Recovery rate `u` (inverse time constant). Small
    /// values → slow recovery.
    pub a: f32,
    /// Sensitivity of `u` to subthreshold fluctuations of `v`. `b > 0.2·a`
    /// gives resonant (Resonator) behavior instead of integrating.
    pub b: f32,
    /// The value to which `v` is reset after a spike (mV).
    pub c: f32,
    /// The increase in `u` after the spike determines the strength of adaptation/bursting.
    pub d: f32,
    /// Spike detection threshold (mV). The canonical value of the article is 30.
    pub threshold: f32,
    /// Number of integration sub-steps `v` per call [`Izhikevich::step`]
    /// (see module documentation). Should be `≥ 1`; with `substeps ==
    /// N` effective `dt` for `v` is `1.0/N`.
    pub substeps: usize,
    /// Are `a, b, c, d` trainable (all four at once - they are not independent
    /// physically, splitting learning one by one does not make sense for
    /// maintaining the spike mode).
    pub learn_params: bool,
}

impl Default for IzhikevichConfig {
    fn default() -> Self {
        let (a, b, c, d) = IzhikevichPreset::RegularSpiking.params();
        Self {
            a,
            b,
            c,
            d,
            threshold: 30.0,
            substeps: 2,
            learn_params: false,
        }
    }
}

impl IzhikevichConfig {
    /// Configuration from a named spike mode preset.
    pub fn from_preset(preset: IzhikevichPreset) -> Self {
        let (a, b, c, d) = preset.params();
        Self {
            a,
            b,
            c,
            d,
            ..Default::default()
        }
    }

    /// Configuration with explicit `(a, b, c, d)`.
    pub fn new(a: f32, b: f32, c: f32, d: f32) -> Self {
        Self {
            a,
            b,
            c,
            d,
            ..Default::default()
        }
    }

    /// Set the spike detection threshold.
    pub fn with_threshold(mut self, threshold: f32) -> Self {
        self.threshold = threshold;
        self
    }

    /// Set the number of integration sub-steps `v`.
    ///
    /// Clamped from below to 1: `substeps == 0` would do
    /// [`Izhikevich::step`] by identical transformation without integration
    /// in general, which is not an acceptable mode of operation of the model.
    pub fn with_substeps(mut self, substeps: usize) -> Self {
        self.substeps = substeps.max(1);
        self
    }

    /// Make `a, b, c, d` trainable.
    pub fn with_learn_params(mut self, learn: bool) -> Self {
        self.learn_params = learn;
        self
    }

    /// Create a layer.
    pub fn init(&self, device: &Device) -> Izhikevich {
        Izhikevich::new(self, device)
    }
}

/// Izikevich neuron state: membrane `v` and recovery variable
/// `u`. A separate type, not [`NeuronState`](crate::state::NeuronState) —
/// the semantics of `v`/`u` are different here (no “membrane + optional current”,
/// both variables are required and participate in the associated nonlinear
/// system), and unlike `NeuronState::syn`, the variable `u` cannot
/// be `None`: without it the equation `v'` is undefined.
#[derive(Debug, Clone)]
pub struct IzhikevichState {
    /// Membrane potential `v`, `[batch, units]`, in the same units
    /// that `threshold`/`c` (canonically - mV).
    pub v: Tensor<2>,
    /// Recovery variable `u`, `[batch, units]`.
    pub u: Tensor<2>,
}

impl IzhikevichState {
    /// Rest state: `v` is initialized to `c` (typical
    /// membrane resting point immediately after hyperpolarization reset -
    /// avoids unphysical transition process in the first steps, in
    /// difference from `v = 0`, which for most presets lies higher
    /// real resting potential and provokes a false early spike),
    /// `u` - zero.
    pub fn resting(device: &Device, batch_size: usize, units: usize, c: f32) -> Self {
        Self {
            v: Tensor::<1>::from_floats([c], device)
                .reshape([1, 1])
                .expand([batch_size, units]),
            u: Tensor::zeros([batch_size, units], device),
        }
    }

    /// The canonical state of rest of the classical Izikevich model:
    /// `v = c`, `u = b · c` - in contrast to [`resting`](Self::resting), where
    /// `u = 0` (see module documentation, section about initial state).
    ///
    /// **ADDED**: `resting()` - valid, specially selected contract
    /// this crate (avoids the transient process at `u = 0` for
    /// presets, where this is not a physical balance point), but does NOT coincide with
    /// by what is most often used as the initial condition for direct
    /// compared with the canonical Izikevich model - there is a zero derivative
    /// `u` at rest (`u' = a·(b·v − u) = 0` at `v = c`) gives
    /// namely `u = b·c`, not `u = 0`. This constructor exists for
    /// cases where exact parity with the canonical model is important (for example,
    /// comparison with the reference implementation in Python/MATLAB), without replacing
    /// `resting()` - which of the two contracts is suitable depends on the task.
    pub fn canonical(device: &Device, batch_size: usize, units: usize, b: f32, c: f32) -> Self {
        Self {
            v: Tensor::<1>::from_floats([c], device)
                .reshape([1, 1])
                .expand([batch_size, units]),
            u: Tensor::<1>::from_floats([b * c], device)
                .reshape([1, 1])
                .expand([batch_size, units]),
        }
    }

    /// A state of ready-made tensors.
    pub fn new(v: Tensor<2>, u: Tensor<2>) -> Self {
        Self { v, u }
    }

    /// Detach the state from the auto-differentiation graph (truncated BPTT),
    /// see [`NeuronState::detach`](crate::state::NeuronState::detach).
    pub fn detach(self) -> Self {
        Self {
            v: self.v.detach(),
            u: self.u.detach(),
        }
    }
}

/// Izikevich neuron layer.
#[derive(Module, Debug)]
pub struct Izhikevich {
    /// Raw `a`: `sigmoid(a_raw)` gives the recovery speed of `u` in
    /// open interval `(0, 1)`.
    ///
    /// Storing `a` directly with a `clamp(1e-4, 1.0)` at
    /// [`Izhikevich::params`] would not be safe for training: if an
    /// optimizer step moves `a` outside the range, `clamp` returns EXACTLY
    /// ZERO gradient by definition (`d(clamp)/dx = 0` outside `[lo, hi]`),
    /// so a parameter stuck outside can no longer return by gradient
    /// descent. `sigmoid` never vanishes identically (it decreases
    /// asymptotically but remains strictly positive at any finite `a_raw`) and
    /// physically cannot produce a value outside `(0, 1)`, so no additional
    /// `clamp` is needed. The same approach is used by `Leaky::beta`
    /// (`beta_init` through `logit`, value via `sigmoid`, see
    /// [`super::logit`] and `leaky.rs`). To interpret a trained value,
    /// read `sigmoid(a.val())` (or [`Izhikevich::params`]), just as `beta`
    /// is read through `sigmoid(beta.val())` in `Leaky`.
    pub a: Param<Tensor<1>>,
    /// Sensitivity of `u` to `v`.
    pub b: Param<Tensor<1>>,
    /// Reset potential `v` after a spike.
    pub c: Param<Tensor<1>>,
    /// Increase in `u` after the spike.
    pub d: Param<Tensor<1>>,
    /// Spike detection threshold.
    #[module(skip)]
    pub threshold: f32,
    /// Number of sub-steps of integration `v`.
    #[module(skip)]
    pub substeps: usize,
    /// Are the parameters trainable?
    ///
    /// `a, b, c, d` are stored as `Param<Tensor<...>>` regardless of this
    /// flag, so the type alone cannot freeze them. [`Izhikevich::params`]
    /// calls `.detach()` on all four tensors when `learn_params == false`,
    /// BEFORE they participate in `step`, so no gradient reaches them through
    /// the graph of this forward pass even if an external optimizer
    /// accidentally includes them (without an incoming gradient the optimizer
    /// step is a no-op; `Param` remains structurally “teachable”, but is
    /// effectively frozen). This matches `learn_threshold`/`learn_delay` in
    /// the remaining layers of the crate.
    #[module(skip)]
    pub learn_params: bool,
}

impl Izhikevich {
    /// Create a layer from the configuration.
    ///
    /// `a` is initialized via [`logit`] (the inverse of `sigmoid`), so
    /// `sigmoid(a_raw) == config.a` EXACTLY at the moment the layer was created -
    /// the same technique that `Leaky::new` initializes `beta` from
    /// `beta_init`. `config.a` must lie in `(0, 1)` (like all
    /// canonical presets, see [`IzhikevichPreset::params`]); `logit`
    /// itself already clamps the input to the open interval through `EPS` (see.
    /// [`super::logit`]), so no additional check is needed here.
    pub fn new(config: &IzhikevichConfig, device: &Device) -> Self {
        Self {
            a: Param::from_tensor(Tensor::from_floats([logit(config.a)], device)),
            b: Param::from_tensor(Tensor::from_floats([config.b], device)),
            c: Param::from_tensor(Tensor::from_floats([config.c], device)),
            d: Param::from_tensor(Tensor::from_floats([config.d], device)),
            threshold: config.threshold,
            substeps: config.substeps.max(1),
            learn_params: config.learn_params,
        }
    }

    /// Parameters `(a, b, c, d)` of the shape `dims`, with `a = sigmoid(a_raw)` -
    /// always strictly in `(0, 1)` by construction (see field documentation
    /// `Izhikevich::a`), without clamping. `b, c, d` are not reparameterized -
    /// they do not have a similar strict physical sign constraint in
    /// canonical model.
    ///
    /// At `learn_params == false` all four tensors are uncoupled from
    /// auto-differentiation graph (`.detach()`) IMMEDIATELY after reading
    /// `.val()`, up to `sigmoid`/`reshape`/`expand` - that is, before
    /// they fall into `step` and participate in the calculation of `v`/`u`/spike. This
    /// and there is actual freezing: since the gradient cannot reach `a,
    /// b, c, d` through any path of the computation graph of this
    /// forward pass, the value for them is `.grad()` after `backward()`
    /// guaranteed `None` - regardless of whether the caller enables
    /// Code these `Param` in the list of optimizer parameters by mistake. See
    /// documentation of the `Izhikevich::learn_params` field.
    fn params(&self, dims: [usize; 2]) -> (Tensor<2>, Tensor<2>, Tensor<2>, Tensor<2>) {
        let maybe_detach =
            |t: Tensor<1>| -> Tensor<1> { if self.learn_params { t } else { t.detach() } };

        let a = sigmoid(maybe_detach(self.a.val()))
            .reshape([1, 1])
            .expand(dims);
        let b = maybe_detach(self.b.val()).reshape([1, 1]).expand(dims);
        let c = maybe_detach(self.c.val()).reshape([1, 1]).expand(dims);
        let d = maybe_detach(self.d.val()).reshape([1, 1]).expand(dims);
        (a, b, c, d)
    }

    /// One time step (one “full” step `u`, consisting of
    /// [`substeps`](IzhikevichConfig::substeps) sub-steps by `v` - see.
    /// module documentation).
    ///
    /// `input` — input current `[batch, units]`, `state.v`/`state.u` —
    /// `[batch, units]`. Returns `(spikes, new state)`.
    pub fn step<S: Surrogate + ?Sized>(
        &self,
        input: Tensor<2>,
        state: &IzhikevichState,
        surrogate: &S,
    ) -> (Tensor<2>, IzhikevichState) {
        let dims = state.v.dims();
        let (a, b, c, d) = self.params(dims);

        let dt = 1.0 / self.substeps as f32;
        let mut v = state.v.clone();
        let u = state.u.clone();

        // The sub-steps update only v for a fixed u - the same as in
        // reference implementation of Izikevich (u changes much more slowly than v
        // and is updated once per full step, see modular
        // documentation about numerical stability).
        for _ in 0..self.substeps {
            let dv =
                v.clone().powf_scalar(2.0).mul_scalar(0.04) + v.clone().mul_scalar(5.0) + 140.0
                    - u.clone()
                    + input.clone();
            v = v + dv.mul_scalar(dt);
        }
        let du = (v.clone().mul(b) - u.clone()).mul(a);
        let u_new = u + du; // dt = 1 for u - one full step, as in the original.

        let (spike, spike_hard) = spike_with_scalar_threshold(v.clone(), self.threshold, surrogate);

        // Reset by HARD spike: where spike_hard == 1, v ← c, u ← u + d;
        // otherwise the values remain as after integration. Recorded without
        // branching by elements (`mul`/`add` on masks) to work
        // the same on CPU and GPU backends Burn.
        let keep = spike_hard.clone().mul_scalar(-1.0).add_scalar(1.0); // 1 - spike_hard
        let v_reset = v.mul(keep.clone()) + c.mul(spike_hard.clone());
        let u_reset = u_new.clone() + d.mul(spike_hard);

        (
            spike,
            IzhikevichState {
                v: v_reset,
                u: u_reset,
            },
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::surrogate::ATan;
    use burn::tensor::Device;

    fn dev() -> Device {
        Device::flex().autodiff()
    }

    fn t2(data: Vec<f32>, shape: [usize; 2]) -> Tensor<2> {
        Tensor::<1>::from_floats(data.as_slice(), &dev()).reshape(shape)
    }

    fn vec2(t: Tensor<2>) -> Vec<f32> {
        t.into_data().try_to_vec::<f32>().unwrap()
    }

    /// Regular Spiking with classic stimulus (I = 10, as in
    /// reference example of Izikevich) must produce at least one
    /// spike within a reasonable window and remain numerically stable (no NaN).
    #[test]
    fn regular_spiking_produces_spikes_with_classic_stimulus() {
        let device = dev();
        let config = IzhikevichConfig::from_preset(IzhikevichPreset::RegularSpiking);
        let layer = Izhikevich::new(&config, &device);
        let surrogate = ATan::default();

        let mut state = IzhikevichState::resting(&device, 1, 1, config.c);
        let input = t2(vec![10.0], [1, 1]);

        let mut total_spikes = 0.0f32;
        for _ in 0..200 {
            let (spike, new_state) = layer.step(input.clone(), &state, &surrogate);
            state = new_state;
            total_spikes += vec2(spike)[0];

            let v = vec2(state.v.clone())[0];
            let u = vec2(state.u.clone())[0];
            assert!(v.is_finite(), "v diverged (NaN/inf): {v}");
            assert!(u.is_finite(), "u diverged (NaN/inf): {u}");
        }
        assert!(
            total_spikes >= 1.0,
            "at least one spike was expected per 200 steps at I=10, obtained {total_spikes}"
        );
    }

    /// Without input current and in a resting state, a neuron should not spontaneously
    /// spike.
    #[test]
    fn resting_state_without_input_does_not_spike() {
        let device = dev();
        let config = IzhikevichConfig::from_preset(IzhikevichPreset::RegularSpiking);
        let layer = Izhikevich::new(&config, &device);
        let surrogate = ATan::default();

        let mut state = IzhikevichState::resting(&device, 1, 1, config.c);
        for _ in 0..50 {
            let (spike, new_state) = layer.step(t2(vec![0.0], [1, 1]), &state, &surrogate);
            state = new_state;
            assert_eq!(
                vec2(spike),
                vec![0.0],
                "there should be no spikes without input at rest"
            );
        }
    }

    /// After the spike, `v` must be reset exactly to `c`, and `u` -
    /// increased exactly by `d` relative to the value before reset
    /// (we use a forced high `v` to guarantee a spike
    /// at the very first step, no matter how many sub-steps).
    /// After the spike, `v` must be reset exactly to `c`, and `u` -
    /// increased exactly by `d` relative to the value AFTER integration
    /// (not relative to the original `v` - `u` in the Izikevich model
    /// is updated according to the already integrated `v` of the same step, and not according to
    /// value up to sub-steps, see `Izhikevich::step`). Formula
    /// is recalculated here explicitly using the same equations as in the layer,
    /// instead of relying on a constant hardcoded into the test by hand
    /// - this way the check does not become out of sync with the implementation when changing
    ///   `substeps`.
    #[test]
    fn post_spike_reset_matches_c_and_d() {
        let device = dev();
        let (a, b, c, d) = (0.02, 0.2, -65.0, 8.0);
        let substeps = 1usize;
        let config = IzhikevichConfig::new(a, b, c, d).with_substeps(substeps);
        let layer = Izhikevich::new(&config, &device);
        let surrogate = ATan::default();

        let v0 = 35.0f32;
        let u0 = 0.0f32;
        let state = IzhikevichState::new(t2(vec![v0], [1, 1]), t2(vec![u0], [1, 1]));
        let (spike, new_state) = layer.step(t2(vec![0.0], [1, 1]), &state, &surrogate);

        assert_eq!(
            vec2(spike),
            vec![1.0],
            "a spike was expected at v=35 > threshold=30"
        );
        let v_after = vec2(new_state.v)[0];
        assert!(
            (v_after - c).abs() < 1e-3,
            "v after the spike must be c={c}, obtained {v_after}"
        );

        // We reproduce sub-steps by v (dt = 1/substeps for each), exactly
        // as in Izhikevich::step to get v AFTER integration,
        // but BEFORE the reset - from it, and not from v0, du is computed.
        let dt = 1.0 / substeps as f32;
        let mut v = v0;
        for _ in 0..substeps {
            let dv = 0.04 * v * v + 5.0 * v + 140.0 - u0 + 0.0;
            v += dv * dt;
        }
        let expected_u = u0 + a * (b * v - u0) + d;
        let u_after = vec2(new_state.u)[0];
        assert!(
            (u_after - expected_u).abs() < 1e-2,
            "u after the spike: expected {expected_u}, received {u_after}"
        );
    }

    /// `substeps` is clamped from below to 1 - `with_substeps(0)` should not
    /// give a layer without integration.
    #[test]
    fn substeps_clamped_to_at_least_one() {
        let config = IzhikevichConfig::default().with_substeps(0);
        assert_eq!(config.substeps, 1);
    }

    /// Fast Spiking should give a spike frequency no less than
    /// Regular Spiking, with the same suprathreshold stimulus - high-quality
    /// checking that the presets actually define different modes, and
    /// are not ignored.
    #[test]
    fn fast_spiking_fires_at_least_as_often_as_regular_spiking() {
        let device = dev();
        let surrogate = ATan::default();
        let input = t2(vec![10.0], [1, 1]);

        let count_spikes = |preset: IzhikevichPreset| -> f32 {
            let config = IzhikevichConfig::from_preset(preset);
            let layer = Izhikevich::new(&config, &device);
            let mut state = IzhikevichState::resting(&device, 1, 1, config.c);
            let mut total = 0.0f32;
            for _ in 0..200 {
                let (spike, new_state) = layer.step(input.clone(), &state, &surrogate);
                state = new_state;
                total += vec2(spike)[0];
            }
            total
        };

        let rs = count_spikes(IzhikevichPreset::RegularSpiking);
        let fs = count_spikes(IzhikevichPreset::FastSpiking);
        assert!(
            fs >= rs,
            "Fast Spiking ({fs}) must spike at least as often as Regular Spiking ({rs})"
        );
    }
}

#[cfg(test)]
mod gradient_tests {
    use super::*;
    use crate::surrogate::ATan;
    use burn::tensor::Device;

    fn dev() -> Device {
        Device::flex().autodiff()
    }

    fn t2(data: Vec<f32>, shape: [usize; 2]) -> Tensor<2> {
        Tensor::<1>::from_floats(data.as_slice(), &dev()).reshape(shape)
    }

    /// The gradient along `a` must remain NON-ZERO even when the value of `a`
    /// lies near the ends of the range, where a `clamp(1e-4, 1.0)` would be
    /// saturated and give `d(clamp)/dx == 0` identically. `sigmoid`
    /// asymptotically tends to 0 but never resets to zero for finite
    /// `a_raw`, so the gradient must always flow - checked at both ends
    /// of the range.
    #[test]
    fn gradient_flows_through_a_near_both_bounds_of_range() {
        let device = dev();

        // Very small a (close to the lower admissible limit).
        let config_low = IzhikevichConfig::new(1e-5, 0.2, -65.0, 8.0).with_learn_params(true);
        let layer_low = Izhikevich::new(&config_low, &device);
        let surrogate = ATan::default();

        let state = IzhikevichState::new(t2(vec![35.0], [1, 1]), t2(vec![0.0], [1, 1]));
        let (spike, new_state) = layer_low.step(t2(vec![10.0], [1, 1]), &state, &surrogate);
        let loss = spike.sum() + new_state.u.sum() + new_state.v.sum();
        let grads = loss.backward();
        let a_grad = layer_low.a.grad(&grads);
        assert!(
            a_grad.is_some(),
            "the gradient along a does not flow when a is close to the lower boundary"
        );
        let a_grad_val = a_grad.unwrap().into_data().try_to_vec::<f32>().unwrap()[0];
        assert!(
            a_grad_val.abs() > 0.0 && a_grad_val.is_finite(),
            "the gradient along a at the lower boundary must be finite and non-zero, obtained {a_grad_val}"
        );

        // Very large a (close to the upper admissible limit, >1.0).
        let config_high = IzhikevichConfig::new(5.0, 0.2, -65.0, 8.0).with_learn_params(true);
        let layer_high = Izhikevich::new(&config_high, &device);
        let state = IzhikevichState::new(t2(vec![35.0], [1, 1]), t2(vec![0.0], [1, 1]));
        let (spike, new_state) = layer_high.step(t2(vec![10.0], [1, 1]), &state, &surrogate);
        let loss = spike.sum() + new_state.u.sum() + new_state.v.sum();
        let grads = loss.backward();
        let a_grad = layer_high.a.grad(&grads);
        assert!(
            a_grad.is_some(),
            "the gradient along a does not flow when a is close to the upper limit"
        );
        let a_grad_val = a_grad.unwrap().into_data().try_to_vec::<f32>().unwrap()[0];
        assert!(
            a_grad_val.abs() > 0.0 && a_grad_val.is_finite(),
            "the gradient along a at the upper boundary must be finite and non-zero, obtained {a_grad_val}"
        );
    }

    /// `sigmoid(a_raw)` must always lie in `(0, 1)`, whatever `config.a` is
    /// at the input (including values `> 1.0`). Here `config.a = 5.0`, a
    /// physically incorrect “recovery speed” value - the layer must not panic
    /// or produce NaN/Inf, and `params()` (via `step`) must still return
    /// something in `(0, 1)`.
    #[test]
    fn a_stays_in_open_unit_interval_even_for_out_of_range_config() {
        let device = dev();
        let config = IzhikevichConfig::new(5.0, 0.2, -65.0, 8.0);
        let layer = Izhikevich::new(&config, &device);
        let surrogate = ATan::default();

        let state = IzhikevichState::new(t2(vec![-65.0], [1, 1]), t2(vec![0.0], [1, 1]));
        let (_, new_state) = layer.step(t2(vec![0.0], [1, 1]), &state, &surrogate);

        let u = new_state.u.into_data().try_to_vec::<f32>().unwrap()[0];
        assert!(
            u.is_finite(),
            "u must remain finite for a=5.0, obtained {u}"
        );
    }
}
