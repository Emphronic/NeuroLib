//! Minimal Hodgkin–Huxley model (Hodgkin & Huxley, 1952): sodium and
//! potassium channels with gate variables `m, h, n` and leakage current.
//!
//! ```text
//! C·dV/dt = I − g_Na·m³·h·(V − E_Na) − g_K·n⁴·(V − E_K) − g_L·(V − E_L)
//! dx/dt   = α_x(V)·(1 − x) − β_x(V)·x,        x ∈ {m, h, n}
//! ```
//!
//! Modern sign convention is used (rest ≈ −65 mV, spike - rise
//! `V`) and classic channel speeds:
//!
//! ```text
//! α_m = 0.1·(V+40)/(1 − e^{−(V+40)/10})      β_m = 4·e^{−(V+65)/18}
//! α_h = 0.07·e^{−(V+65)/20}                  β_h = 1/(1 + e^{−(V+35)/10})
//! α_n = 0.01·(V+55)/(1 − e^{−(V+55)/10})     β_n = 0.125·e^{−(V+65)/80}
//! ```
//!
//! # Units
//!
//! `V` - mV, time - ms, current - µA/cm², conductivity - mS/cm², `C` - µF/cm².
//! The default parameters are classic: `C=1, g_Na=120, g_K=36, g_L=0.3,
//! E_Na=50, E_K=−77, E_L=−54.387`.
//!
//! # Numerical scheme
//!
//! The model is rigid: direct Euler for `V` is stable at `dt ≲ 0.05 ms`. That's why
//! one [`HodgkinHuxley::step`] (`dt` ms, default 1) divided by
//! [`substeps`](HodgkinHuxleyConfig::substeps) sub-steps (default 40,
//! `dt_sub = 0.025 ms`). Gate variables are updated exponentially
//! Euler (`x ← x∞ + (x − x∞)·e^{−dt(α+β)}`, is unconditionally stable and does not come out
//! from `[0, 1]`), then `V` - by direct Euler using the already updated gates.
//!
//! Features `α_m` and `α_n` with `V = −40` and `V = −55` (0/0) removed
//! Taylor expansion in the neighborhood of a singular point, so that neither a straight line nor
//! the reverse pass is not given by NaN.
//!
//! # Spike
//!
//! The model does not have a reset: the spike is an UPWARD crossing of the threshold
//! ([`spike_threshold`](HodgkinHuxleyConfig::spike_threshold), 0 mV) per step.
//! The "membrane was above the threshold" flag is stored in the state, so the same
//! a spike does not count twice.

use burn::module::{Module, Param};
use burn::tensor::Device;
use burn::tensor::Tensor;

use super::spike_with_scalar_threshold;
use crate::surrogate::Surrogate;

/// `x / (1 − e^{−x/y})` with the singularity removed at `x → 0` (the limit is `y`).
/// For `|x/y| < 0.02` the decomposition `y·(1 − r/2 + r²/12)`, `r = x/y` is used.
fn vtrap(x: Tensor<2>, y: f32) -> Tensor<2> {
    let r = x.clone().mul_scalar(1.0 / y);
    let small = r.clone().abs().lower_elem(0.02).float().detach();
    let den = r
        .clone()
        .mul_scalar(-1.0)
        .exp()
        .mul_scalar(-1.0)
        .add_scalar(1.0);
    // In the “small” region, the denominator is close to zero: we replace it with 1 so that
    // the unused branch did not give 0/0 in either the forward or backward pass.
    let regular = x.div(den + small.clone());
    let limit = (r.clone().mul_scalar(-0.5) + r.clone().mul(r).mul_scalar(1.0 / 12.0))
        .add_scalar(1.0)
        .mul_scalar(y);
    regular.mul(small.clone().mul_scalar(-1.0).add_scalar(1.0)) + limit.mul(small)
}

/// Channel velocities `(α_m, β_m, α_h, β_h, α_n, β_n)` for the tensor `V`.
fn rates(v: &Tensor<2>) -> [Tensor<2>; 6] {
    let am = vtrap(v.clone().add_scalar(40.0), 10.0).mul_scalar(0.1);
    let bm = v
        .clone()
        .add_scalar(65.0)
        .mul_scalar(-1.0 / 18.0)
        .exp()
        .mul_scalar(4.0);
    let ah = v
        .clone()
        .add_scalar(65.0)
        .mul_scalar(-1.0 / 20.0)
        .exp()
        .mul_scalar(0.07);
    let bh = v
        .clone()
        .add_scalar(35.0)
        .mul_scalar(-1.0 / 10.0)
        .exp()
        .add_scalar(1.0);
    let bh = recip(bh);
    let an = vtrap(v.clone().add_scalar(55.0), 10.0).mul_scalar(0.01);
    let bn = v
        .clone()
        .add_scalar(65.0)
        .mul_scalar(-1.0 / 80.0)
        .exp()
        .mul_scalar(0.125);
    [am, bm, ah, bh, an, bn]
}

/// `1/x` (there is no `recip` in the crate, so through `ones / x`).
fn recip(x: Tensor<2>) -> Tensor<2> {
    Tensor::ones(x.dims(), &x.device()).div(x)
}

/// Exponential Euler for the gate variable: exact solution
/// `dx/dt = α(1−x) − βx` with frozen `V` on the segment `dt`.
fn gate_update(x: Tensor<2>, alpha: Tensor<2>, beta: Tensor<2>, dt: f32) -> Tensor<2> {
    let total = alpha.clone() + beta;
    let x_inf = alpha.div(total.clone());
    let decay = total.mul_scalar(-dt).exp();
    x_inf.clone() + (x - x_inf).mul(decay)
}

/// Scalar `vtrap` (for initial rest state and tests).
fn vtrap_f32(x: f32, y: f32) -> f32 {
    let r = x / y;
    if r.abs() < 0.02 {
        y * (1.0 - r / 2.0 + r * r / 12.0)
    } else {
        x / (1.0 - (-r).exp())
    }
}

/// Scalar channel speeds `(α_m, β_m, α_h, β_h, α_n, β_n)`.
pub fn hh_rates_f32(v: f32) -> [f32; 6] {
    [
        0.1 * vtrap_f32(v + 40.0, 10.0),
        4.0 * (-(v + 65.0) / 18.0).exp(),
        0.07 * (-(v + 65.0) / 20.0).exp(),
        1.0 / (1.0 + (-(v + 35.0) / 10.0).exp()),
        0.01 * vtrap_f32(v + 55.0, 10.0),
        0.125 * (-(v + 65.0) / 80.0).exp(),
    ]
}

/// Stationary values of the gate `(m∞, h∞, n∞)` for a fixed `V`.
pub fn hh_steady_state(v: f32) -> (f32, f32, f32) {
    let [am, bm, ah, bh, an, bn] = hh_rates_f32(v);
    (am / (am + bm), ah / (ah + bh), an / (an + bn))
}

/// `vtrap` in an arbitrary scalar type.
fn vtrap_scalar<T: crate::scalar::Scalar>(x: T, y: T) -> T {
    let r = x / y;
    if r.abs() < T::c(0.02) {
        y * (T::ONE - r / T::c(2.0) + r * r / T::c(12.0))
    } else {
        x / (T::ONE - (-r).exp())
    }
}

/// Channel rates `(α_m, β_m, α_h, β_h, α_n, β_n)` in scalar type `T`
/// (for `f32` coincides with [`hh_rates_f32`]).
pub fn hh_rates<T: crate::scalar::Scalar>(v: T) -> [T; 6] {
    let c = T::c;
    [
        c(0.1) * vtrap_scalar(v + c(40.0), c(10.0)),
        c(4.0) * (-(v + c(65.0)) / c(18.0)).exp(),
        c(0.07) * (-(v + c(65.0)) / c(20.0)).exp(),
        T::ONE / (T::ONE + (-(v + c(35.0)) / c(10.0)).exp()),
        c(0.01) * vtrap_scalar(v + c(55.0), c(10.0)),
        c(0.125) * (-(v + c(65.0)) / c(80.0)).exp(),
    ]
}

/// Stationary values of the gate `(m∞, h∞, n∞)` in the scalar type `T`.
pub fn hh_steady_state_as<T: crate::scalar::Scalar>(v: T) -> (T, T, T) {
    let [am, bm, ah, bh, an, bn] = hh_rates(v);
    (am / (am + bm), ah / (ah + bh), an / (an + bn))
}

/// Configuration of the Hodgkin–Huxley model.
#[derive(Debug, Clone, Copy)]
pub struct HodgkinHuxleyConfig {
    /// Membrane capacity `C`, µF/cm².
    pub c: f32,
    /// Maximum conductance of sodium channels `g_Na`, mS/cm².
    pub g_na: f32,
    /// Maximum conductance of potassium channels `g_K`, mS/cm².
    pub g_k: f32,
    /// Leakage conductivity `g_L`, mS/cm².
    pub g_l: f32,
    /// Equilibrium potential of sodium `E_Na`, mV.
    pub e_na: f32,
    /// Equilibrium potassium potential `E_K`, mV.
    pub e_k: f32,
    /// Leakage potential `E_L`, mV.
    pub e_l: f32,
    /// Spike detection threshold (ascending crossing), mV.
    pub spike_threshold: f32,
    /// Surrogate scale, mV: the surrogate is calculated from `(V − threshold)/scale`.
    pub surrogate_scale: f32,
    /// Duration of one call `step`, ms.
    pub dt: f32,
    /// Number of sub-steps per one `step` (`dt_sub = dt / substeps`; for
    /// stability keep `dt_sub ≤ 0.05 ms`).
    pub substeps: usize,
    /// Are `g_Na`, `g_K`, `g_L` trainable?
    pub learn_params: bool,
}

impl Default for HodgkinHuxleyConfig {
    fn default() -> Self {
        Self {
            c: 1.0,
            g_na: 120.0,
            g_k: 36.0,
            g_l: 0.3,
            e_na: 50.0,
            e_k: -77.0,
            e_l: -54.387,
            spike_threshold: 0.0,
            surrogate_scale: 10.0,
            dt: 1.0,
            substeps: 40,
            learn_params: false,
        }
    }
}

impl HodgkinHuxleyConfig {
    /// Set step duration, ms.
    pub fn with_dt(mut self, dt: f32) -> Self {
        self.dt = dt;
        self
    }

    /// Set the number of sub-steps (clamped from below to 1).
    pub fn with_substeps(mut self, substeps: usize) -> Self {
        self.substeps = substeps.max(1);
        self
    }

    /// Make conductivity trainable.
    pub fn with_learn_params(mut self, learn: bool) -> Self {
        self.learn_params = learn;
        self
    }

    /// Create a layer.
    pub fn init(&self, device: &Device) -> HodgkinHuxley {
        HodgkinHuxley::new(self, device)
    }
}

/// State of the Hodgkin–Huxley neuron.
#[derive(Debug, Clone)]
pub struct HodgkinHuxleyState {
    /// Membrane potential `[batch, units]`, mV.
    pub v: Tensor<2>,
    /// Activation of sodium `m`.
    pub m: Tensor<2>,
    /// Inactivation of sodium `h`.
    pub h: Tensor<2>,
    /// Potassium activation `n`.
    pub n: Tensor<2>,
    /// 1, if at the previous sub-step the membrane was above the spike threshold
    /// (needed to detect only upward crossings).
    pub above: Tensor<2>,
}

impl HodgkinHuxleyState {
    /// Resting state at potential `v_rest` (default −65 mV): gates
    /// start at their steady-state values for this `V`.
    pub fn resting(device: &Device, batch_size: usize, units: usize, v_rest: f32) -> Self {
        let (m, h, n) = hh_steady_state(v_rest);
        let full = |x: f32| Tensor::zeros([batch_size, units], device).add_scalar(x);
        Self {
            v: full(v_rest),
            m: full(m),
            h: full(h),
            n: full(n),
            above: Tensor::zeros([batch_size, units], device),
        }
    }

    /// State from ready tensors (`above = 0`).
    pub fn new(v: Tensor<2>, m: Tensor<2>, h: Tensor<2>, n: Tensor<2>) -> Self {
        let above = Tensor::zeros(v.dims(), &v.device());
        Self { v, m, h, n, above }
    }

    /// Detach from auto-differentiation graph (truncated BPTT).
    pub fn detach(self) -> Self {
        Self {
            v: self.v.detach(),
            m: self.m.detach(),
            h: self.h.detach(),
            n: self.n.detach(),
            above: self.above.detach(),
        }
    }
}

/// Layer of Hodgkin–Huxley neurons.
#[derive(Module, Debug)]
pub struct HodgkinHuxley {
    /// `ln g_Na`: conductivity `exp(·)` is strictly positive.
    pub log_g_na: Param<Tensor<1>>,
    /// `ln g_K`.
    pub log_g_k: Param<Tensor<1>>,
    /// `ln g_L`.
    pub log_g_l: Param<Tensor<1>>,
    /// Capacity, µF/cm².
    #[module(skip)]
    pub c: f32,
    /// `E_Na`, mV.
    #[module(skip)]
    pub e_na: f32,
    /// `E_K`, mV.
    #[module(skip)]
    pub e_k: f32,
    /// `E_L`, mV.
    #[module(skip)]
    pub e_l: f32,
    /// Spike threshold, mV.
    #[module(skip)]
    pub spike_threshold: f32,
    /// Surrogate scale, mV.
    #[module(skip)]
    pub surrogate_scale: f32,
    /// Step duration, ms.
    #[module(skip)]
    pub dt: f32,
    /// Number of sub-steps.
    #[module(skip)]
    pub substeps: usize,
    /// Are conductivities trainable (at `false` they are detached from the graph).
    #[module(skip)]
    pub learn_params: bool,
}

impl HodgkinHuxley {
    /// Create a layer from the configuration.
    pub fn new(config: &HodgkinHuxleyConfig, device: &Device) -> Self {
        let ln = |g: f32| Param::from_tensor(Tensor::from_floats([g.max(1e-6).ln()], device));
        Self {
            log_g_na: ln(config.g_na),
            log_g_k: ln(config.g_k),
            log_g_l: ln(config.g_l),
            c: config.c,
            e_na: config.e_na,
            e_k: config.e_k,
            e_l: config.e_l,
            spike_threshold: config.spike_threshold,
            surrogate_scale: config.surrogate_scale.max(1e-3),
            dt: config.dt,
            substeps: config.substeps.max(1),
            learn_params: config.learn_params,
        }
    }

    /// Conductivity `(g_Na, g_K, g_L)` of the shape `dims`.
    fn conductances(&self, dims: [usize; 2]) -> (Tensor<2>, Tensor<2>, Tensor<2>) {
        let get = |p: &Param<Tensor<1>>| -> Tensor<2> {
            let t = p.val();
            let t = if self.learn_params { t } else { t.detach() };
            t.exp().reshape([1, 1]).expand(dims)
        };
        (get(&self.log_g_na), get(&self.log_g_k), get(&self.log_g_l))
    }

    /// One step (`dt` ms). `input` — current `[batch, units]` in µA/cm².
    /// Returns `(spikes 0/1, new state)`.
    pub fn step<S: Surrogate + ?Sized>(
        &self,
        input: Tensor<2>,
        state: &HodgkinHuxleyState,
        surrogate: &S,
    ) -> (Tensor<2>, HodgkinHuxleyState) {
        let dims = state.v.dims();
        let device = state.v.device();
        let (g_na, g_k, g_l) = self.conductances(dims);

        let dt = self.dt / self.substeps as f32;
        let inv_c = 1.0 / self.c;
        let inv_scale = 1.0 / self.surrogate_scale;
        let thr = self.spike_threshold * inv_scale;

        let mut v = state.v.clone();
        let mut m = state.m.clone();
        let mut h = state.h.clone();
        let mut n = state.n.clone();
        let mut above = state.above.clone();
        let mut no_spike: Tensor<2> = Tensor::ones(dims, &device);

        for _ in 0..self.substeps {
            // Gate - according to V of the beginning of the sub-step (exact solution with frozen V).
            let [am, bm, ah, bh, an, bn] = rates(&v);
            m = gate_update(m, am, bm, dt);
            h = gate_update(h, ah, bh, dt);
            n = gate_update(n, an, bn, dt);

            // Currents - through the updated gates.
            let m3 = m.clone().mul(m.clone()).mul(m.clone());
            let n2 = n.clone().mul(n.clone());
            let n4 = n2.clone().mul(n2);
            let i_na = g_na
                .clone()
                .mul(m3)
                .mul(h.clone())
                .mul(v.clone().sub_scalar(self.e_na));
            let i_k = g_k.clone().mul(n4).mul(v.clone().sub_scalar(self.e_k));
            let i_l = g_l.clone().mul(v.clone().sub_scalar(self.e_l));
            let dv = (input.clone() - i_na - i_k - i_l).mul_scalar(inv_c);
            v = v + dv.mul_scalar(dt);

            // Ascending threshold crossing: it was lower, it became higher.
            let (spike, hard) =
                spike_with_scalar_threshold(v.clone().mul_scalar(inv_scale), thr, surrogate);
            let was_below = above.clone().mul_scalar(-1.0).add_scalar(1.0);
            let crossing = spike.mul(was_below);
            no_spike = no_spike.mul(crossing.mul_scalar(-1.0).add_scalar(1.0));
            above = hard;
        }

        let spike = no_spike.mul_scalar(-1.0).add_scalar(1.0);
        (spike, HodgkinHuxleyState { v, m, h, n, above })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::surrogate::ATan;

    fn dev() -> Device {
        Device::flex()
    }

    fn t2(data: Vec<f32>, shape: [usize; 2], device: &Device) -> Tensor<2> {
        Tensor::<1>::from_floats(data.as_slice(), device).reshape(shape)
    }

    fn val(t: Tensor<2>) -> f32 {
        t.into_data().try_to_vec::<f32>().unwrap()[0]
    }

    /// The tensor `vtrap` coincides with the scalar, including the singular point.
    #[test]
    fn vtrap_matches_scalar_including_singularity() {
        let device = dev();
        let xs = [-30.0f32, -1.0, -0.05, 0.0, 0.05, 1.0, 30.0];
        let t = vtrap(t2(xs.to_vec(), [1, xs.len()], &device), 10.0);
        let got = t.into_data().try_to_vec::<f32>().unwrap();
        for (x, g) in xs.iter().zip(got) {
            let want = vtrap_f32(*x, 10.0);
            assert!(g.is_finite(), "vtrap({x}) is not finite");
            assert!(
                (g - want).abs() < 1e-3 * want.abs().max(1.0),
                "{x}: {g} vs {want}"
            );
        }
        // The limit at zero is y.
        assert!((vtrap_f32(0.0, 10.0) - 10.0).abs() < 1e-6);
    }

    /// Channel speeds at points with known values.
    #[test]
    fn rates_match_known_values_at_rest() {
        let (m, h, n) = hh_steady_state(-65.0);
        assert!((m - 0.0529).abs() < 2e-3, "m∞ = {m}");
        assert!((h - 0.5961).abs() < 2e-3, "h∞ = {h}");
        assert!((n - 0.3177).abs() < 2e-3, "n∞ = {n}");
    }

    /// Tensor velocities coincide with scalar ones.
    #[test]
    fn tensor_rates_match_scalar_rates() {
        let device = dev();
        for v in [-80.0f32, -65.0, -55.0, -40.0, -20.0, 10.0, 40.0] {
            let got = rates(&t2(vec![v], [1, 1], &device));
            let want = hh_rates_f32(v);
            for (i, (g, w)) in got.into_iter().zip(want).enumerate() {
                let g = val(g);
                assert!(
                    (g - w).abs() < 1e-3 * w.abs().max(1.0),
                    "v={v} #{i}: {g} vs {w}"
                );
            }
        }
    }

    /// At rest without input, the neuron does not spike and remains around −65 mV.
    #[test]
    fn rests_without_input() {
        let device = dev();
        let layer = HodgkinHuxleyConfig::default().init(&device);
        let surrogate = ATan::default();
        let mut state = HodgkinHuxleyState::resting(&device, 1, 1, -65.0);
        let input = t2(vec![0.0], [1, 1], &device);
        for _ in 0..30 {
            let (s, next) = layer.step(input.clone(), &state, &surrogate);
            state = next;
            assert_eq!(val(s), 0.0);
        }
        let v = val(state.v);
        assert!((v + 65.0).abs() < 1.5, "v left rest: {v}");
    }

    /// A constant current of 10 µA/cm² gives a periodic spike of the classical
    /// amplitude (peak above +20 mV), the gate remains at [0, 1].
    #[test]
    fn repetitive_firing_under_constant_current() {
        let device = dev();
        let layer = HodgkinHuxleyConfig::default().init(&device);
        let surrogate = ATan::default();
        let mut state = HodgkinHuxleyState::resting(&device, 1, 1, -65.0);
        let input = t2(vec![10.0], [1, 1], &device);
        let (mut spikes, mut v_max) = (0.0f32, f32::MIN);
        for _ in 0..100 {
            let (s, next) = layer.step(input.clone(), &state, &surrogate);
            state = next;
            spikes += val(s);
            let v = val(state.v.clone());
            assert!(v.is_finite(), "v separated: {v}");
            v_max = v_max.max(v);
            for gate in [&state.m, &state.h, &state.n] {
                let x = val(gate.clone());
                assert!((0.0..=1.0).contains(&x), "gate outside [0,1]: {x}");
            }
        }
        assert!(
            (3.0..=12.0).contains(&spikes),
            "in 100 ms at 10 μA/cm² 3..12 spikes were expected, received {spikes}"
        );
        assert!(v_max > 20.0, "spike peak too low: {v_max} mV");
    }

    /// A weak subthreshold current (1 μA/cm²) does not cause spikes.
    #[test]
    fn subthreshold_current_does_not_fire() {
        let device = dev();
        let layer = HodgkinHuxleyConfig::default().init(&device);
        let surrogate = ATan::default();
        let mut state = HodgkinHuxleyState::resting(&device, 1, 1, -65.0);
        let input = t2(vec![1.0], [1, 1], &device);
        let mut spikes = 0.0;
        for _ in 0..60 {
            let (s, next) = layer.step(input.clone(), &state, &surrogate);
            state = next;
            spikes += val(s);
        }
        assert_eq!(spikes, 0.0);
    }

    /// The gradient along `ln g_Na`/`ln g_K`/`ln g_L` is finite and exists.
    #[test]
    fn gradient_flows_to_conductances() {
        let device = Device::flex().autodiff();
        let config = HodgkinHuxleyConfig::default()
            .with_learn_params(true)
            .with_substeps(20);
        let layer = HodgkinHuxley::new(&config, &device);
        let surrogate = ATan::default();
        let mut state = HodgkinHuxleyState::resting(&device, 1, 1, -65.0);
        let input = t2(vec![15.0], [1, 1], &device);
        let mut loss: Option<Tensor<1>> = None;
        for _ in 0..4 {
            let (s, next) = layer.step(input.clone(), &state, &surrogate);
            state = next;
            let term = s.sum() + state.v.clone().sum().mul_scalar(1e-2);
            loss = Some(match loss {
                Some(l) => l + term,
                None => term,
            });
        }
        let grads = loss.unwrap().backward();
        for (name, g) in [
            ("g_na", layer.log_g_na.grad(&grads)),
            ("g_k", layer.log_g_k.grad(&grads)),
            ("g_l", layer.log_g_l.grad(&grads)),
        ] {
            let g = g.unwrap_or_else(|| panic!("no gradient along {name}"));
            let g = g.into_data().try_to_vec::<f32>().unwrap()[0];
            assert!(
                g.is_finite(),
                "the gradient along {name} is not finite: {g}"
            );
        }
    }
}
