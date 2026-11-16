//! Multilevel (“graded”) and ternary spikes.
//!
//! A typical spike carries one bit. Two expansion steps that are now
//! used in spike transformers and detectors:
//!
//! * **Integer (graded) spikes** - Spike Firing Approximation from
//!   Spike-driven Transformer V3 (Yao et al.) and I-LIF from SpikeYOLO: in training
//!   the neuron outputs the integer `k ∈ {0, …, D}`, and on the inference the same value
//!   unfolds in `k` binary spikes in `D` virtual steps. Amount
//!   of unfolded spikes is equal to the original integer, so that learning occurs “in
//!   integers”, and the launch remains spiked (additions only). See
//!   [`expand_levels`].
//! * **Ternary spikes** - `{−α, 0, +α}` (Ternary Spike, Guo et al.,
//!   AAAI 2024; T-LIF in SpikeLM) with trainable amplitude `α`: neuron
//!   spikes both up and down. This is a special case of a graded neuron:
//!   `max_level = 1`, `signed = true` - see [`GradedLeakyConfig::ternary`].
//!
//! # How the level is calculated
//!
//! Level - the number of thresholds passed `θ, 2θ, …, D·θ` (and mirrored down for
//! signed mode):
//!
//! ```text
//! level(U) = Σ_{k=1..D} H(U − kθ)  −  [signed] Σ_{k=1..D} H(−U − kθ)
//! ```
//!
//! where `H` is a Heaviside step with a surrogate gradient (as with a regular
//! spike of the crate). At `D = 1` and `signed = false` this is exactly a normal spike
//! (`U > θ`, strict inequality), therefore the neuron at `D = 1` coincides with
//! [`Leaky`](super::Leaky). Gradient along `U` - the sum of surrogates at each
//! step of the staircase.
//!
//! Default reset - subtraction: `U ← U − level·θ` (membrane loses exactly
//! the charge that went into the spikes). Amplitude `α` (teachable or
//! fixed) multiplies OUTPUT but not reset.
//!
//! Difference from [`apply_graded_spikes`](super::apply_graded_spikes) /
//! `LeakyConfig::graded_spikes_factor`: there the binary spike is simply scaled
//! per multiplier, here the number of levels is more than two.

use burn::module::{Module, Param};
use burn::tensor::Device;
use burn::tensor::Tensor;
use burn::tensor::activation::sigmoid;

use super::{apply_graded_spikes, logit, spike_with_scalar_threshold};
use crate::state::{NeuronState, ResetMode};
use crate::surrogate::Surrogate;

/// Multilevel spike with scalar threshold.
///
/// Returns `(level, level_hard)`:
/// * `level` - straight-through level (in the forward pass the whole, in the reverse pass -
///   sum of stage surrogates);
/// * `level_hard` - the same level, detached from the graph (for reset).
///
/// `max_level` (≥ 1) — number of steps `D`; at `signed` the levels lie in
/// `[−D, D]`, otherwise in `[0, D]`. The condition of each stage is strict: `U > k·θ`.
pub fn multi_level_spike<S: Surrogate + ?Sized>(
    mem: Tensor<2>,
    threshold: f32,
    max_level: usize,
    signed: bool,
    surrogate: &S,
) -> (Tensor<2>, Tensor<2>) {
    let max_level = max_level.max(1);
    let mut level: Option<Tensor<2>> = None;
    let mut hard: Option<Tensor<2>> = None;
    let mut add = |s: Tensor<2>, h: Tensor<2>, sign: f32| {
        let (s, h) = if sign < 0.0 {
            (s.mul_scalar(-1.0), h.mul_scalar(-1.0))
        } else {
            (s, h)
        };
        level = Some(match level.take() {
            Some(l) => l + s,
            None => s,
        });
        hard = Some(match hard.take() {
            Some(l) => l + h,
            None => h,
        });
    };

    for k in 1..=max_level {
        let step_threshold = threshold * k as f32;
        let (s, h) = spike_with_scalar_threshold(mem.clone(), step_threshold, surrogate);
        add(s, h, 1.0);
        if signed {
            let (s, h) = spike_with_scalar_threshold(
                mem.clone().mul_scalar(-1.0),
                step_threshold,
                surrogate,
            );
            add(s, h, -1.0);
        }
    }
    (
        level.expect("max_level >= 1"),
        hard.expect("max_level >= 1"),
    )
}

/// Expands integer levels in `max_level` "virtual steps"
/// binary (for `signed` - ternary `{−1, 0, 1}`) spikes.
///
/// The plane `j` (`j = 1..=D`) is equal to `[level ≥ j] − [level ≤ −j]`. Amount
/// planes is equal to `level` (for `|level| ≤ D`) - this is the equivalence
/// “integer learning ⇄ spike inference” from Spike-driven Transformer
/// V3. Negative planes appear only at sign levels.
pub fn expand_levels(level: Tensor<2>, max_level: usize) -> Vec<Tensor<2>> {
    (1..=max_level.max(1))
        .map(|j| {
            let up = level.clone().greater_equal_elem(j as f32).float();
            let down = level.clone().lower_equal_elem(-(j as f32)).float();
            up - down
        })
        .collect()
}

/// Separates the ternary spikes `{−1, 0, 1}` into two binary maps
/// `(positive, negative)`: `spikes = pos − neg`. So ternary layer
/// executed on two binary (regular) channels.
pub fn split_ternary(spikes: Tensor<2>) -> (Tensor<2>, Tensor<2>) {
    let pos = spikes.clone().greater_elem(0.0).float();
    let neg = spikes.lower_elem(0.0).float();
    (pos, neg)
}

/// Configuration [`GradedLeaky`].
#[derive(Debug, Clone, Copy)]
pub struct GradedLeakyConfig {
    /// The initial damping of the membrane is β ∈ (0, 1).
    pub beta: f32,
    /// Stage threshold `θ`.
    pub threshold: f32,
    /// Maximum level `D` (number of steps, ≥ 1).
    pub max_level: usize,
    /// Sign levels `[−D, D]` (with `D = 1` - ternary spike).
    pub signed: bool,
    /// Initial amplitude `α` (multiplies the output).
    pub amplitude: f32,
    /// Is the amplitude `α` trainable?
    pub learn_amplitude: bool,
    /// Is β decay learnable?
    pub learn_beta: bool,
    /// Membrane reset method.
    pub reset_mode: ResetMode,
}

impl GradedLeakyConfig {
    /// Graduated neuron with levels `{0, …, max_level}`.
    pub fn new(beta: f32, threshold: f32, max_level: usize) -> Self {
        Self {
            beta,
            threshold,
            max_level: max_level.max(1),
            signed: false,
            amplitude: 1.0,
            learn_amplitude: false,
            learn_beta: false,
            reset_mode: ResetMode::Subtract,
        }
    }

    /// Ternary neuron `{−α, 0, +α}` with trainable amplitude (as in Ternary
    /// Spike): `max_level = 1`, `signed = true`, `learn_amplitude = true`.
    pub fn ternary(beta: f32, threshold: f32) -> Self {
        Self {
            signed: true,
            learn_amplitude: true,
            ..Self::new(beta, threshold, 1)
        }
    }

    /// Sign levels `[−D, D]`.
    pub fn with_signed(mut self, signed: bool) -> Self {
        self.signed = signed;
        self
    }

    /// Initial amplitude.
    pub fn with_amplitude(mut self, amplitude: f32) -> Self {
        self.amplitude = amplitude;
        self
    }

    /// Trainable amplitude.
    pub fn with_learn_amplitude(mut self, learn: bool) -> Self {
        self.learn_amplitude = learn;
        self
    }

    /// Learning attenuation.
    pub fn with_learn_beta(mut self, learn: bool) -> Self {
        self.learn_beta = learn;
        self
    }

    /// Reset method.
    pub fn with_reset_mode(mut self, mode: ResetMode) -> Self {
        self.reset_mode = mode;
        self
    }

    /// Create a layer.
    pub fn init(&self, device: &Device) -> GradedLeaky {
        GradedLeaky::new(self, device)
    }
}

/// LIF-neuron with multi-level (graded or ternary) output.
///
/// Condition: normal [`NeuronState`] only with membrane.
#[derive(Module, Debug)]
pub struct GradedLeaky {
    /// `logit(β)`: `β = sigmoid(·)` is always in `(0, 1)`.
    pub beta: Param<Tensor<1>>,
    /// Amplitude `α`.
    pub amplitude: Param<Tensor<1>>,
    /// Step threshold.
    #[module(skip)]
    pub threshold: f32,
    /// Maximum level `D`.
    #[module(skip)]
    pub max_level: usize,
    /// Iconic levels.
    #[module(skip)]
    pub signed: bool,
    /// Reset method.
    #[module(skip)]
    pub reset_mode: ResetMode,
    /// Is the amplitude trainable?
    #[module(skip)]
    pub learn_amplitude: bool,
    /// Is attenuation learnable?
    #[module(skip)]
    pub learn_beta: bool,
}

impl GradedLeaky {
    /// Create a layer from the configuration.
    pub fn new(config: &GradedLeakyConfig, device: &Device) -> Self {
        Self {
            beta: Param::from_tensor(Tensor::from_floats([logit(config.beta)], device)),
            amplitude: Param::from_tensor(Tensor::from_floats([config.amplitude], device)),
            threshold: config.threshold,
            max_level: config.max_level.max(1),
            signed: config.signed,
            reset_mode: config.reset_mode,
            learn_amplitude: config.learn_amplitude,
            learn_beta: config.learn_beta,
        }
    }

    /// Current attenuation β (scalar per host).
    pub fn beta_value(&self) -> f32 {
        let b = sigmoid(self.beta.val().detach());
        b.into_data().try_to_vec::<f32>().expect("beta f32")[0]
    }

    /// Current amplitude `α` (scalar per host).
    pub fn amplitude_value(&self) -> f32 {
        self.amplitude
            .val()
            .detach()
            .into_data()
            .try_to_vec::<f32>()
            .expect("amplitude f32")[0]
    }

    /// One step. Returns `(α · level, new state)`.
    pub fn step<S: Surrogate + ?Sized>(
        &self,
        input: Tensor<2>,
        state: &NeuronState,
        surrogate: &S,
    ) -> (Tensor<2>, NeuronState) {
        let dims = state.mem.dims();
        let beta = if self.learn_beta {
            self.beta.val()
        } else {
            self.beta.val().detach()
        };
        let beta = sigmoid(beta).reshape([1, 1]).expand(dims);
        let mem_new = state.mem.clone().mul(beta) + input;

        let (level, level_hard) = multi_level_spike(
            mem_new.clone(),
            self.threshold,
            self.max_level,
            self.signed,
            surrogate,
        );

        let mem_reset = match self.reset_mode {
            ResetMode::Subtract => mem_new - level_hard.mul_scalar(self.threshold),
            ResetMode::Zero => {
                let fired = level_hard.abs().clamp_max(1.0);
                mem_new.mul(fired.mul_scalar(-1.0).add_scalar(1.0))
            }
            ResetMode::None => mem_new,
        };

        let amplitude = if self.learn_amplitude {
            self.amplitude.val()
        } else {
            self.amplitude.val().detach()
        };
        let out = apply_graded_spikes(level, amplitude);
        (
            out,
            NeuronState {
                mem: mem_reset,
                syn: None,
            },
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::surrogate::ATan;

    fn dev() -> Device {
        Device::flex()
    }

    fn t2(data: Vec<f32>, shape: [usize; 2]) -> Tensor<2> {
        Tensor::<1>::from_floats(data.as_slice(), &dev()).reshape(shape)
    }

    fn vec2(t: Tensor<2>) -> Vec<f32> {
        t.into_data().try_to_vec::<f32>().unwrap()
    }

    /// Ternary spike: ±1 outside the threshold, 0 inside (strict inequality).
    #[test]
    fn ternary_levels() {
        let mem = t2(
            vec![-2.5, -1.2, -1.0, -0.4, 0.0, 0.4, 1.0, 1.2, 2.5],
            [1, 9],
        );
        let (level, hard) = multi_level_spike(mem, 1.0, 1, true, &ATan::default());
        let want = vec![-1.0, -1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0, 1.0];
        assert_eq!(vec2(level), want);
        assert_eq!(vec2(hard), want);
    }

    /// Graduated spike: number of thresholds passed, not more than `D`.
    #[test]
    fn graded_levels_clip_at_max() {
        let mem = t2(vec![-1.0, 0.5, 1.5, 2.5, 3.5, 7.0], [1, 6]);
        let (level, _) = multi_level_spike(mem, 1.0, 3, false, &ATan::default());
        assert_eq!(vec2(level), vec![0.0, 0.0, 1.0, 2.0, 3.0, 3.0]);
    }

    /// Iconic graduated mode: levels in `[−D, D]`.
    #[test]
    fn signed_graded_levels() {
        let mem = t2(vec![-9.0, -2.5, -1.5, 0.0, 1.5, 2.5, 9.0], [1, 7]);
        let (level, _) = multi_level_spike(mem, 1.0, 2, true, &ATan::default());
        assert_eq!(vec2(level), vec![-2.0, -2.0, -1.0, 0.0, 1.0, 2.0, 2.0]);
    }

    /// `D = 1`, unsigned - regular binary spike `U > θ`.
    #[test]
    fn single_level_matches_binary_spike() {
        let mem = t2(vec![0.9, 1.0, 1.1], [1, 3]);
        let (level, _) = multi_level_spike(mem, 1.0, 1, false, &ATan::default());
        assert_eq!(vec2(level), vec![0.0, 0.0, 1.0]);
    }

    /// Reset by subtraction takes exactly `level·θ`; the output is scaled by `α`.
    #[test]
    fn subtract_reset_and_amplitude() {
        let device = dev();
        let config = GradedLeakyConfig::new(0.9, 1.0, 3).with_amplitude(0.5);
        let layer = config.init(&device);
        let state = NeuronState::zeros_mem_only(&device, 1, 1);
        let (out, next) = layer.step(t2(vec![2.5], [1, 1]), &state, &ATan::default());
        assert_eq!(vec2(out), vec![1.0]); // level 2 · α 0.5
        assert!((vec2(next.mem)[0] - 0.5).abs() < 1e-6);
    }

    /// A signed reset returns the negative membrane to zero.
    #[test]
    fn ternary_negative_spike_resets_upwards() {
        let device = dev();
        let layer = GradedLeakyConfig::ternary(0.9, 1.0).init(&device);
        let state = NeuronState::zeros_mem_only(&device, 1, 1);
        let (out, next) = layer.step(t2(vec![-1.4], [1, 1]), &state, &ATan::default());
        assert_eq!(vec2(out), vec![-1.0]);
        assert!((vec2(next.mem)[0] + 0.4).abs() < 1e-6);
    }

    /// Unwrapping to binary spikes preserves the integer value
    /// (equivalence to “integer learning ⇄ spike inference”).
    #[test]
    fn expand_levels_sums_back_to_level() {
        let level = t2(vec![-3.0, -1.0, 0.0, 1.0, 2.0, 3.0], [1, 6]);
        let planes = expand_levels(level.clone(), 3);
        assert_eq!(planes.len(), 3);
        let mut sum = planes[0].clone();
        for p in &planes[1..] {
            sum = sum + p.clone();
        }
        assert_eq!(vec2(sum), vec2(level));
        // Each plane is ternary.
        for p in planes {
            assert!(vec2(p).iter().all(|x| [-1.0, 0.0, 1.0].contains(x)));
        }
    }

    /// `pos − neg` restores the ternary spike.
    #[test]
    fn split_ternary_roundtrip() {
        let s = t2(vec![-1.0, 0.0, 1.0, 1.0, -1.0], [1, 5]);
        let (pos, neg) = split_ternary(s.clone());
        assert_eq!(vec2(pos.clone() - neg.clone()), vec2(s));
        assert_eq!(vec2(pos), vec![0.0, 0.0, 1.0, 1.0, 0.0]);
        assert_eq!(vec2(neg), vec![1.0, 0.0, 0.0, 0.0, 1.0]);
    }

    /// The gradient goes through the surrogate ladder and to the trained amplitude.
    #[test]
    fn gradient_flows_through_levels_and_amplitude() {
        let device = Device::flex().autodiff();
        let layer = GradedLeakyConfig::ternary(0.9, 1.0).init(&device);
        let state = NeuronState::zeros_mem_only(&device, 1, 3);
        let x = Tensor::<1>::from_floats([1.4f32, -1.4, 0.2].as_slice(), &device)
            .reshape([1, 3])
            .require_grad();
        let (out, _) = layer.step(x.clone(), &state, &ATan::default());
        let grads = out.sum().backward();
        let gx = x.grad(&grads).expect("input gradient");
        let gx = gx.into_data().try_to_vec::<f32>().unwrap();
        assert!(gx.iter().all(|g| g.is_finite()));
        // level increases monotonically across the membrane, so all gradients are positive
        // (for a negative stage, the sign “−” at the output and “−” at the input are compensated).
        assert!(gx.iter().all(|g| *g > 0.0), "input gradients: {gx:?}");
        let ga = layer.amplitude.grad(&grads).expect("amplitude gradient");
        let ga = ga.into_data().try_to_vec::<f32>().unwrap()[0];
        assert!(
            (ga - 0.0).abs() < 1e-5,
            "ΣA level = 1 − 1 + 0 = 0, obtained {ga}"
        );
    }

    /// The amplitude is not trained until `learn_amplitude == false`.
    #[test]
    fn frozen_amplitude_has_no_gradient() {
        let device = Device::flex().autodiff();
        let layer = GradedLeakyConfig::new(0.9, 1.0, 2).init(&device);
        let state = NeuronState::zeros_mem_only(&device, 1, 1);
        let x = Tensor::<1>::from_floats([1.5f32].as_slice(), &device)
            .reshape([1, 1])
            .require_grad();
        let (out, _) = layer.step(x, &state, &ATan::default());
        let grads = out.sum().backward();
        assert!(layer.amplitude.grad(&grads).is_none());
    }
}
