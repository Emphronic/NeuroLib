//! `StdpLinear`: fully connected Linear → LIF layer trained by STDP instead
//! backprop - for unsupervised and on-chip training, following the STDP rules
//! of `snntorch.functional.stdp_linear` (see [`crate::functional::plasticity`]
//! for the rules themselves).
//!
//! # How does this differ from other crate layers
//!
//! All other layers [`crate::neuron`]/[`crate::layers`] are trained through
//! Burn `Param` + backprop: weights - part of `Module::Record`, gradient flows
//! through a surrogate spike derivative. `StdpLinear` - intentional
//! exception: its weights are the usual `Tensor<2>`, part of EXPLICIT STATE
//! ([`StdpState`]), not the `Module` parameter. Weight update - local
//! co-activity rule (see module [`crate::functional::plasticity`]),
//! called at each step explicitly, and not by the optimizer.
//!
//! This is how it should be for an on-chip/unsupervised scenario: the goal of STDP is not
//! "solve a specific problem with a known loss function", and "learn
//! structure in unsupervised input correlations", and that is why STDP
//! suitable for neuromorphic hardware, where there is no global graph
//! auto-differentiation, no separate pass “forward, calculate loss,
//! back": synapse weight update depends only on what is already
//! available locally at this synapse right now.
//!
//! `StdpLinear` therefore does NOT implement `burn::module::Module` - wrap
//! it is not needed in the `Module` tree along with backprop layers (there is nothing
//! trust the optimizer), and `forward`/`step` return output AND new
//! state (weights + traces + neuron membrane) is explicit, as in
//! [`crate::layers::SynapticDelay`].
//!
//! # Three rules
//!
//! Selectable via [`StdpRule`]:
//! * `Pair` — classic pair-based STDP;
//! * `Triplet { .. }` — triplet STDP (frequency-dependent plasticity);
//! * `RewardModulated { .. }` — R-STDP: co-activity accumulates into eligibility
//!   trace, weights are updated only by explicitly calling [`StdpLinear::apply_reward`]
//!   when a scalar reward arrives.
//!
//! # Online and sequence modes
//!
//! Like [`SynapticDelay`](crate::layers::SynapticDelay): [`StdpLinear::step`]
//! — one time step with an explicit state (the essence of on-chip
//! training - each step is applied immediately), [`StdpLinear::forward_sequence`]
//! — run through the prepared sequence `[batch, time, in_features]` with
//! accumulation of STDP updates inside, for the convenience of offline experiments
//! and parity tests with `step`.

use burn::tensor::Device;
use burn::tensor::{Distribution, Tensor};

use crate::error::{Result, SnnError, or_panic};
use crate::layers::record::{
    RECORD_VERSION, StdpHyper, StdpLinearRecord, check_dims, check_optional, check_version,
    decode_bound, encode_bound, record_err,
};

use crate::functional::plasticity::{
    WeightBound, apply_reward, eligibility_trace_update, pair_update, triplet_update, update_trace,
};
use crate::neuron::spike_with_scalar_threshold;
use crate::surrogate::Surrogate;

/// Which STDP-rule does [`StdpLinear`] use? See module
/// [`crate::functional::plasticity`] behind the formulas of each rule.
#[derive(Debug, Clone, Copy)]
pub enum StdpRule {
    /// Classic pair-based STDP: `a_plus`/`a_minus` — coefficients
    /// potentiation/depression, `tau_pre`/`tau_post` - constant attenuation
    /// (in steps) pre-/post-synaptic trace.
    Pair {
        a_plus: f32,
        a_minus: f32,
        tau_pre: f32,
        tau_post: f32,
    },
    /// Triplet STDP: to fast traces (`tau_pre`/`tau_post`, same roles,
    /// that in `Pair`) one slow trace is added with each
    /// sides (`tau_pre_slow`/`tau_post_slow`, usually an order of magnitude larger
    /// fast) and triplet coefficients `a3_plus`/`a3_minus`. When
    /// `a3_plus == a3_minus == 0.0` is equivalent to `Pair` with the same
    /// `a2_plus`/`a2_minus`.
    Triplet {
        a2_plus: f32,
        a2_minus: f32,
        a3_plus: f32,
        a3_minus: f32,
        tau_pre: f32,
        tau_post: f32,
        tau_pre_slow: f32,
        tau_post_slow: f32,
    },
    /// Reward-modulated STDP (three-factor rule). Co-activity accumulates in
    /// eligibility trace with attenuation constant `tau_e` (usually much
    /// more `tau_pre`/`tau_post` - reward comes with a delay
    /// relative to the spikes that deserve it). The weights do NOT change by
    /// each step [`StdpLinear::step`], but only by explicitly calling
    /// [`StdpLinear::apply_reward`].
    RewardModulated {
        a_plus: f32,
        a_minus: f32,
        tau_pre: f32,
        tau_post: f32,
        tau_e: f32,
    },
}

impl StdpRule {
    /// Type and parameters of the rule for the record (see [`crate::layers::record`]).
    pub(crate) fn encode(&self) -> (u8, Vec<f32>) {
        match *self {
            StdpRule::Pair {
                a_plus,
                a_minus,
                tau_pre,
                tau_post,
            } => (0, vec![a_plus, a_minus, tau_pre, tau_post]),
            StdpRule::Triplet {
                a2_plus,
                a2_minus,
                a3_plus,
                a3_minus,
                tau_pre,
                tau_post,
                tau_pre_slow,
                tau_post_slow,
            } => (
                1,
                vec![
                    a2_plus,
                    a2_minus,
                    a3_plus,
                    a3_minus,
                    tau_pre,
                    tau_post,
                    tau_pre_slow,
                    tau_post_slow,
                ],
            ),
            StdpRule::RewardModulated {
                a_plus,
                a_minus,
                tau_pre,
                tau_post,
                tau_e,
            } => (2, vec![a_plus, a_minus, tau_pre, tau_post, tau_e]),
        }
    }

    /// Reverse to [`StdpRule::encode`].
    pub(crate) fn decode(kind: u8, p: &[f32]) -> Result<Self> {
        match (kind, p) {
            (0, &[a_plus, a_minus, tau_pre, tau_post]) => Ok(StdpRule::Pair {
                a_plus,
                a_minus,
                tau_pre,
                tau_post,
            }),
            (
                1,
                &[
                    a2_plus,
                    a2_minus,
                    a3_plus,
                    a3_minus,
                    tau_pre,
                    tau_post,
                    tau_pre_slow,
                    tau_post_slow,
                ],
            ) => Ok(StdpRule::Triplet {
                a2_plus,
                a2_minus,
                a3_plus,
                a3_minus,
                tau_pre,
                tau_post,
                tau_pre_slow,
                tau_post_slow,
            }),
            (2, &[a_plus, a_minus, tau_pre, tau_post, tau_e]) => Ok(StdpRule::RewardModulated {
                a_plus,
                a_minus,
                tau_pre,
                tau_post,
                tau_e,
            }),
            _ => Err(record_err(
                "StdpRule",
                format!("invalid rule: type {kind}, parameters {}", p.len()),
            )),
        }
    }

    /// Variant name (for error messages).
    pub(crate) fn name(&self) -> &'static str {
        match self {
            StdpRule::Pair { .. } => "Pair",
            StdpRule::Triplet { .. } => "Triplet",
            StdpRule::RewardModulated { .. } => "RewardModulated",
        }
    }

    pub(crate) fn tau_pre(&self) -> f32 {
        match *self {
            StdpRule::Pair { tau_pre, .. } => tau_pre,
            StdpRule::Triplet { tau_pre, .. } => tau_pre,
            StdpRule::RewardModulated { tau_pre, .. } => tau_pre,
        }
    }

    pub(crate) fn tau_post(&self) -> f32 {
        match *self {
            StdpRule::Pair { tau_post, .. } => tau_post,
            StdpRule::Triplet { tau_post, .. } => tau_post,
            StdpRule::RewardModulated { tau_post, .. } => tau_post,
        }
    }
}

/// `decay = exp(-1/tau)` is the trace decay constant in one step.
/// `tau <= 0` is interpreted as “the trace does not store the past” (`decay = 0`):
/// degenerate but safe case, not panic or NaN.
fn trace_decay(tau: f32) -> f32 {
    if tau <= 0.0 {
        0.0
    } else {
        (-1.0f32 / tau).exp()
    }
}

/// Configuration [`StdpLinear`].
#[derive(Debug, Clone, Copy)]
pub struct StdpLinearConfig {
    pub in_features: usize,
    pub out_features: usize,
    /// Spike threshold of the postsynaptic LIF neuron.
    pub threshold: f32,
    /// Membrane damping β ∈ (0, 1) of the postsynaptic LIF neuron.
    pub beta: f32,
    /// Rule STDP.
    pub rule: StdpRule,
    /// A way to limit weight.
    pub bound: WeightBound,
    /// Range of uniform initialization of weights.
    pub init_range: [f32; 2],
}

impl StdpLinearConfig {
    /// New configuration with mandatory dimensions and rule STDP.
    /// The remaining fields are reasonable defaults (`threshold = 1.0`,
    /// `beta = 0.9`, soft-bound `[0, 1]`, initialization `[0, 0.4]` —
    /// positive weights, as is customary for unsupervised STDP on
    /// excitatory connections).
    pub fn new(in_features: usize, out_features: usize, rule: StdpRule) -> Self {
        Self {
            in_features,
            out_features,
            threshold: 1.0,
            beta: 0.9,
            rule,
            bound: WeightBound::Soft {
                w_min: 0.0,
                w_max: 1.0,
            },
            init_range: [0.0, 0.4],
        }
    }

    pub fn with_threshold(mut self, threshold: f32) -> Self {
        self.threshold = threshold;
        self
    }

    pub fn with_beta(mut self, beta: f32) -> Self {
        self.beta = beta;
        self
    }

    pub fn with_bound(mut self, bound: WeightBound) -> Self {
        self.bound = bound;
        self
    }

    pub fn with_init_range(mut self, range: [f32; 2]) -> Self {
        self.init_range = range;
        self
    }

    /// Create a layer.
    pub fn init(&self) -> StdpLinear {
        StdpLinear::new(self)
    }
}

/// Layer Linear → LIF, trained by STDP. Weights are part of the explicit state
/// ([`StdpState`]), not a `Module` parameter - see the module documentation.
#[derive(Debug, Clone)]
pub struct StdpLinear {
    in_features: usize,
    out_features: usize,
    threshold: f32,
    beta: f32,
    rule: StdpRule,
    bound: WeightBound,
    init_range: [f32; 2],
}

impl StdpLinear {
    /// Create a layer from the configuration.
    ///
    /// The layer itself (unlike [`StdpState`]) does not store tensors and therefore
    /// not parameterized by backend/device - it’s just validated
    /// a set of constants applied at each step. First tensors (weights)
    /// appear only in [`StdpState::zeros`], where `device` and
    /// transmitted - the same order as the other layers of the crate
    /// (`Config::init` creates a layer, `*State::zeros` creates a state),
    /// it’s just that the layer here doesn’t have its own `Param` that it would need
    /// place it on the device directly in the constructor.
    pub fn new(config: &StdpLinearConfig) -> Self {
        or_panic(Self::try_new(config))
    }

    /// Create layer, checking configuration: wrong dimensions, β out
    /// `(0, 1)` and non-positive slow Triplet constants give
    /// [`SnnError::InvalidConfig`].
    pub fn try_new(config: &StdpLinearConfig) -> Result<Self> {
        let bad = |reason: String| SnnError::InvalidConfig {
            what: "StdpLinear",
            reason,
        };
        if config.in_features < 1 {
            return Err(bad("in_features must be >= 1".into()));
        }
        if config.out_features < 1 {
            return Err(bad("out_features must be >= 1".into()));
        }
        if !(config.beta > 0.0 && config.beta < 1.0) {
            return Err(bad(format!(
                "beta must be in (0, 1), obtained {}",
                config.beta
            )));
        }
        if let StdpRule::Triplet {
            tau_pre_slow,
            tau_post_slow,
            ..
        } = config.rule
            && !(tau_pre_slow > 0.0 && tau_post_slow > 0.0)
        {
            return Err(bad("slow triplet constants STDP must be positive".into()));
        }

        Ok(Self {
            in_features: config.in_features,
            out_features: config.out_features,
            threshold: config.threshold,
            beta: config.beta,
            rule: config.rule,
            bound: config.bound,
            init_range: config.init_range,
        })
    }

    /// Spike threshold of the postsynaptic LIF neuron.
    pub fn threshold(&self) -> f32 {
        self.threshold
    }

    /// Membrane attenuation β.
    pub fn beta(&self) -> f32 {
        self.beta
    }

    pub fn in_features(&self) -> usize {
        self.in_features
    }

    pub fn out_features(&self) -> usize {
        self.out_features
    }

    pub fn rule(&self) -> StdpRule {
        self.rule
    }

    /// Recording a layer along with its state: weights, traces, thresholds and
    /// hyperparameters (see [`crate::layers::record`]).
    pub fn to_record(&self, state: &StdpState) -> StdpLinearRecord {
        let (rule_kind, rule_params) = self.rule.encode();
        let (bound_kind, bound_params) = encode_bound(self.bound);
        StdpLinearRecord {
            in_features: self.in_features,
            out_features: self.out_features,
            hyper: StdpHyper {
                version: RECORD_VERSION,
                threshold: self.threshold,
                beta: self.beta,
                rule_kind,
                rule_params,
                bound_kind,
                bound_params,
                init_range: self.init_range,
            },
            state: state.clone(),
        }
    }

    /// Restore layer and state from recording. Checks the format version,
    /// configuration and shapes of all tensors against the geometry of the layer.
    pub fn from_record(record: StdpLinearRecord) -> Result<(Self, StdpState)> {
        const WHAT: &str = "StdpLinear";
        let StdpLinearRecord {
            in_features,
            out_features,
            hyper,
            state,
        } = record;
        check_version(WHAT, &hyper)?;
        let rule = StdpRule::decode(hyper.rule_kind, &hyper.rule_params)?;
        let bound = decode_bound(hyper.bound_kind, &hyper.bound_params)?;
        let layer = Self::try_new(&StdpLinearConfig {
            in_features,
            out_features,
            threshold: hyper.threshold,
            beta: hyper.beta,
            rule,
            bound,
            init_range: hyper.init_range,
        })?;

        let batch = state.mem.dims()[0];
        check_dims(WHAT, "w", state.w.dims(), [out_features, in_features])?;
        check_dims(WHAT, "mem", state.mem.dims(), [batch, out_features])?;
        check_dims(
            WHAT,
            "pre_trace",
            state.pre_trace.dims(),
            [batch, in_features],
        )?;
        check_dims(
            WHAT,
            "post_trace",
            state.post_trace.dims(),
            [batch, out_features],
        )?;
        let is_triplet = matches!(rule, StdpRule::Triplet { .. });
        let is_reward = matches!(rule, StdpRule::RewardModulated { .. });
        check_optional(
            WHAT,
            "pre_trace_slow",
            &state.pre_trace_slow,
            is_triplet,
            rule.name(),
        )?;
        check_optional(
            WHAT,
            "post_trace_slow",
            &state.post_trace_slow,
            is_triplet,
            rule.name(),
        )?;
        check_optional(
            WHAT,
            "eligibility",
            &state.eligibility,
            is_reward,
            rule.name(),
        )?;
        if let Some(t) = &state.pre_trace_slow {
            check_dims(WHAT, "pre_trace_slow", t.dims(), [batch, in_features])?;
        }
        if let Some(t) = &state.post_trace_slow {
            check_dims(WHAT, "post_trace_slow", t.dims(), [batch, out_features])?;
        }
        if let Some(t) = &state.eligibility {
            check_dims(WHAT, "eligibility", t.dims(), [out_features, in_features])?;
        }
        Ok((layer, state))
    }

    /// One time step: computes the output, updates traces and - if
    /// rule NOT `RewardModulated` - applies STDP to weights immediately.
    /// For `RewardModulated`, this step only accumulates the eligibility trace;
    /// weights are changed by a separate call [`StdpLinear::apply_reward`].
    ///
    /// `input`—presynaptic spikes `[batch, in_features]`.
    /// `surrogate` - the surrogate is needed ONLY to calculate the postsynaptic
    /// spike through the same path as the rest of the crate neurons
    /// ([`spike_with_scalar_threshold`]); backward gradient through it
    /// it is not used anywhere (the weights are not in the auto-differentiation graph),
    /// but a direct pass must give the same 0/1 spikes as any other
    /// LIF in the crate - there is no separate “non-differentiable” version of Heaviside,
    /// to avoid duplicating reset/threshold logic.
    pub fn step<S: Surrogate + ?Sized>(
        &self,
        input: Tensor<2>,
        state: &StdpState,
        surrogate: &S,
    ) -> (Tensor<2>, StdpState) {
        let [batch, in_features] = input.dims();
        assert_eq!(
            in_features, self.in_features,
            "StdpLinear::step: input has {in_features} features, layer is configured to {}",
            self.in_features
        );
        assert_eq!(
            state.mem.dims(),
            [batch, self.out_features],
            "StdpLinear::step: state.mem is incompatible with batch input"
        );

        let device = input.device();

        // Linear → LIF, in the same way as regular Leaky::step (see
        // crate::neuron::leaky), but with weights from state, not Param.
        let mem_new = state.mem.clone().mul_scalar(self.beta)
            + input.clone().matmul(state.w.clone().transpose());
        let (post_spike, post_hard) =
            spike_with_scalar_threshold(mem_new.clone(), self.threshold, surrogate);
        let mem_reset = mem_new - post_hard.clone().mul_scalar(self.threshold);

        // pre_spike for trace update - input as is (maybe
        // graded, not just 0/1: STDP primitives do not require binary).
        let pre_trace_new = update_trace(
            state.pre_trace.clone(),
            input.clone(),
            trace_decay(self.rule.tau_pre()),
        );
        let post_trace_new = update_trace(
            state.post_trace.clone(),
            post_hard.clone(),
            trace_decay(self.rule.tau_post()),
        );

        let (w_new, pre_slow_new, post_slow_new, eligibility_new) = match self.rule {
            StdpRule::Pair {
                a_plus, a_minus, ..
            } => {
                let w = pair_update(
                    state.w.clone(),
                    input,
                    post_hard,
                    state.pre_trace.clone(),
                    state.post_trace.clone(),
                    a_plus,
                    a_minus,
                    self.bound,
                );
                (w, None, None, state.eligibility.clone())
            }
            StdpRule::Triplet {
                a2_plus,
                a2_minus,
                a3_plus,
                a3_minus,
                tau_pre_slow,
                tau_post_slow,
                ..
            } => {
                let pre_slow = state
                    .pre_trace_slow
                    .clone()
                    .unwrap_or_else(|| Tensor::zeros([batch, self.in_features], &device));
                let post_slow = state
                    .post_trace_slow
                    .clone()
                    .unwrap_or_else(|| Tensor::zeros([batch, self.out_features], &device));

                let w = triplet_update(
                    state.w.clone(),
                    input.clone(),
                    post_hard.clone(),
                    state.pre_trace.clone(),
                    state.post_trace.clone(),
                    pre_slow.clone(),
                    post_slow.clone(),
                    a2_plus,
                    a2_minus,
                    a3_plus,
                    a3_minus,
                    self.bound,
                );

                let pre_slow_new = update_trace(pre_slow, input, trace_decay(tau_pre_slow));
                let post_slow_new = update_trace(post_slow, post_hard, trace_decay(tau_post_slow));

                (
                    w,
                    Some(pre_slow_new),
                    Some(post_slow_new),
                    state.eligibility.clone(),
                )
            }
            StdpRule::RewardModulated {
                a_plus,
                a_minus,
                tau_e,
                ..
            } => {
                let e = state.eligibility.clone().unwrap_or_else(|| {
                    Tensor::zeros([self.out_features, self.in_features], &device)
                });
                let e_new = eligibility_trace_update(
                    e,
                    input,
                    post_hard,
                    state.pre_trace.clone(),
                    state.post_trace.clone(),
                    a_plus,
                    a_minus,
                    trace_decay(tau_e),
                );
                // We DO NOT touch the weights - we wait for apply_reward.
                (state.w.clone(), None, None, Some(e_new))
            }
        };

        let out_state = StdpState {
            w: w_new,
            mem: mem_reset,
            pre_trace: pre_trace_new,
            post_trace: post_trace_new,
            pre_trace_slow: pre_slow_new.or_else(|| state.pre_trace_slow.clone()),
            post_trace_slow: post_slow_new.or_else(|| state.post_trace_slow.clone()),
            eligibility: eligibility_new,
        };

        (post_spike, out_state)
    }

    /// Running through the prepared sequence `[batch, time, in_features]`.
    /// The equivalent of a cyclic call [`StdpLinear::step`] is STDP
    /// applied at EVERY step inside (unlike backprop layers
    /// crate, there is no separate “forward without training”: offline
    /// running through a ready-made sequence is still trained
    /// online, step by step, exactly like in chunked inference on
    /// neuromorphic hardware).
    ///
    /// For `RewardModulated` the weights do not change during the run (as in
    /// `step`) — eligibility trace only; reward must be applied
    /// separately through [`StdpLinear::apply_reward`] after (or during,
    /// calling `forward_sequence` on subsegments).
    ///
    /// Returns `(spikes [batch, time, out_features], final state)`.
    pub fn forward_sequence<S: Surrogate + ?Sized>(
        &self,
        input: Tensor<3>,
        mut state: StdpState,
        surrogate: &S,
    ) -> (Tensor<3>, StdpState) {
        let [batch, steps, in_features] = input.dims();
        assert!(
            steps > 0,
            "StdpLinear::forward_sequence: num_steps must be > 0"
        );
        assert_eq!(
            in_features, self.in_features,
            "StdpLinear::forward_sequence: input has {in_features} features, layer is configured to {}",
            self.in_features
        );

        let mut outputs = Vec::with_capacity(steps);
        for t in 0..steps {
            let x_t = input
                .clone()
                .slice([0..batch, t..t + 1, 0..in_features])
                .reshape([batch, in_features]);
            let (spike, new_state) = self.step(x_t, &state, surrogate);
            state = new_state;
            outputs.push(spike);
        }

        (Tensor::stack::<3>(outputs, 1), state)
    }

    /// Applies accumulated rewards to the weights through the eligibility
    /// trace (only for [`StdpRule::RewardModulated`] - for the other rules
    /// the weights are already updated inside [`StdpLinear::step`], and calling
    /// this method is a usage error, see `debug_assert`).
    ///
    /// `lr` is the learning-rate multiplier for this particular
    /// application (see [`crate::functional::plasticity::apply_reward`] -
    /// the division of `lr` by the rule weight is intentional, so that the
    /// learning rate can be changed without changing the constants of STDP itself).
    pub fn apply_reward(&self, state: &StdpState, reward: f32, lr: f32) -> StdpState {
        debug_assert!(
            matches!(self.rule, StdpRule::RewardModulated { .. }),
            "StdpLinear::apply_reward only makes sense when StdpRule::RewardModulated; \
             for Pair/Triplet the weights are already updated inside step()"
        );

        let e = state.eligibility.clone().unwrap_or_else(|| {
            Tensor::zeros([self.out_features, self.in_features], &state.w.device())
        });

        let w_new = apply_reward(state.w.clone(), &e, reward, lr, self.bound);

        StdpState {
            w: w_new,
            ..state.clone()
        }
    }
}

/// Explicit state [`StdpLinear`]: weights + traces + postsynaptic membrane
/// LIF + (optional) slow traces for triplet rules and eligibility
/// trace for the reward-modulated rule.
///
/// Unlike the [`NeuronState`](crate::state::NeuronState) weight here -
/// PART of the state, not a module parameter: STDP changes them every step of the themes
/// in the same way as other state tensors (see documentation
/// module [`self`]).
#[derive(Debug, Clone)]
pub struct StdpState {
    /// Weights `[out_features, in_features]`.
    pub w: Tensor<2>,
    /// Membrane of the postsynaptic LIF `[batch, out_features]`.
    pub mem: Tensor<2>,
    /// Fast presynaptic trace `[batch, in_features]`.
    pub pre_trace: Tensor<2>,
    /// Fast postsynaptic trace `[batch, out_features]`.
    pub post_trace: Tensor<2>,
    /// Slow presynaptic trace (only for `Triplet`), `[batch, in_features]`.
    pub pre_trace_slow: Option<Tensor<2>>,
    /// Slow postsynaptic trace (only for `Triplet`), `[batch, out_features]`.
    pub post_trace_slow: Option<Tensor<2>>,
    /// Eligibility trace (only for `RewardModulated`), `[out_features, in_features]`
    /// - the same shape as the weight, not the batch: co-activity is already averaged over
    ///   batch inside [`eligibility_trace_update`](crate::functional::plasticity::eligibility_trace_update),
    ///   as well as updating the weights of `Pair`/`Triplet`.
    pub eligibility: Option<Tensor<2>>,
}

impl StdpState {
    /// Zero state: membrane and all traces are zeros, weights are random
    /// uniform initialization to `init_range` from layer configuration.
    pub fn zeros(layer: &StdpLinear, device: &Device, batch: usize) -> Self {
        let [lo, hi] = layer.init_range;
        let w = Tensor::random(
            [layer.out_features, layer.in_features],
            Distribution::Uniform(lo as f64, hi as f64),
            device,
        );

        let pre_trace_slow = matches!(layer.rule, StdpRule::Triplet { .. })
            .then(|| Tensor::zeros([batch, layer.in_features], device));
        let post_trace_slow = matches!(layer.rule, StdpRule::Triplet { .. })
            .then(|| Tensor::zeros([batch, layer.out_features], device));
        let eligibility = matches!(layer.rule, StdpRule::RewardModulated { .. })
            .then(|| Tensor::zeros([layer.out_features, layer.in_features], device));

        Self {
            w,
            mem: Tensor::zeros([batch, layer.out_features], device),
            pre_trace: Tensor::zeros([batch, layer.in_features], device),
            post_trace: Tensor::zeros([batch, layer.out_features], device),
            pre_trace_slow,
            post_trace_slow,
            eligibility,
        }
    }

    /// A state with explicitly specified weights (for example, for tests on
    /// parity `step`/`forward_sequence`, where both parties are obliged
    /// start with the IDENTICAL weights, and not with two independent random ones
    /// initializations).
    pub fn with_weights(layer: &StdpLinear, device: &Device, batch: usize, w: Tensor<2>) -> Self {
        let mut s = Self::zeros(layer, device, batch);
        s.w = w;
        s
    }

    /// Detach the entire state from the auto-differentiation graph.
    ///
    /// For `StdpLinear`, this is primarily hygiene: weight is already not in
    /// graph (see module documentation), but `pre_trace`/`post_trace`/
    /// `mem` are constructed through `spike_with_scalar_threshold`, whose
    /// straight-through path is CONNECTED to a surrogate gradient - if
    /// The calling code somehow differentiates something on top of the output
    /// `StdpLinear` (for example, mixed backprop+STDP scenario), not
    /// the uncoupled state would accumulate the graph along the entire length
    /// sequences are exactly the same as those of ordinary neurons
    /// (see [`NeuronState::detach`](crate::state::NeuronState::detach)).
    pub fn detach(self) -> Self {
        Self {
            w: self.w.detach(),
            mem: self.mem.detach(),
            pre_trace: self.pre_trace.detach(),
            post_trace: self.post_trace.detach(),
            pre_trace_slow: self.pre_trace_slow.map(Tensor::detach),
            post_trace_slow: self.post_trace_slow.map(Tensor::detach),
            eligibility: self.eligibility.map(Tensor::detach),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::Device;

    use crate::surrogate::FastSigmoid;

    fn dev() -> Device {
        Device::flex()
    }

    fn t2(data: Vec<f32>, shape: [usize; 2]) -> Tensor<2> {
        Tensor::<1>::from_floats(data.as_slice(), &dev()).reshape(shape)
    }

    fn t3(data: Vec<f32>, shape: [usize; 3]) -> Tensor<3> {
        Tensor::<1>::from_floats(data.as_slice(), &dev()).reshape(shape)
    }

    fn vec2(t: Tensor<2>) -> Vec<f32> {
        t.into_data().try_to_vec::<f32>().unwrap()
    }

    fn pair_config(in_f: usize, out_f: usize) -> StdpLinearConfig {
        StdpLinearConfig::new(
            in_f,
            out_f,
            StdpRule::Pair {
                a_plus: 0.05,
                a_minus: 0.03,
                tau_pre: 20.0,
                tau_post: 20.0,
            },
        )
        .with_bound(WeightBound::Hard {
            w_min: 0.0,
            w_max: 1.0,
        })
    }

    /// `step` in a cycle and `forward_sequence` must give the same
    /// result (same spikes, same final state) with the same
    /// inputs and the same starting point - the same contract as
    /// `SynapticDelay` (`step_and_sequence_modes_agree`).
    #[test]
    fn step_and_sequence_modes_agree() {
        let device = dev();
        let config = pair_config(2, 3);
        let layer = StdpLinear::new(&config);
        let surrogate = FastSigmoid::default();

        let init_w = t2(vec![0.1, 0.2, 0.3, 0.15, 0.25, 0.05], [3, 2]);
        let steps: Vec<Vec<f32>> = vec![
            vec![1.0, 0.0],
            vec![0.0, 1.0],
            vec![1.0, 1.0],
            vec![0.0, 0.0],
            vec![1.0, 0.0],
        ];
        let n = steps.len();

        // Step mode.
        let mut state = StdpState::with_weights(&layer, &device, 1, init_w.clone());
        let mut step_out = Vec::with_capacity(n * 3);
        for s in &steps {
            let (spike, new_state) = layer.step(t2(s.clone(), [1, 2]), &state, &surrogate);
            state = new_state;
            step_out.extend(vec2(spike));
        }

        // Sequential mode, same starting point.
        let flat: Vec<f32> = steps.iter().flat_map(|s| s.iter().copied()).collect();
        let seq_state = StdpState::with_weights(&layer, &device, 1, init_w);
        let (seq_spikes, seq_final_state) =
            layer.forward_sequence(t3(flat, [1, n, 2]), seq_state, &surrogate);
        let seq_out = vec2(seq_spikes.reshape([n, 3]));

        for (i, (&a, &b)) in step_out.iter().zip(seq_out.iter()).enumerate() {
            assert!((a - b).abs() < 1e-5, "spike {i}: step={a} vs sequence={b}");
        }

        let w_step = vec2(state.w);
        let w_seq = vec2(seq_final_state.w);
        for (i, (&a, &b)) in w_step.iter().zip(w_seq.iter()).enumerate() {
            assert!((a - b).abs() < 1e-5, "weight {i}: step={a} vs sequence={b}");
        }
    }

    /// A pair of highly correlated presynaptic spikes with
    /// postsynaptic spike IMMEDIATELY AFTER them must potentiate
    /// (increase) the corresponding weight - a basic sanity check that
    /// the layer actually learns in the right direction and not just
    /// doesn't fall.
    #[test]
    fn repeated_causal_pairing_potentiates_weight() {
        let device = dev();
        let config = pair_config(1, 1).with_beta(0.9).with_threshold(0.5);
        let layer = StdpLinear::new(&config);
        let surrogate = FastSigmoid::default();

        let w0 = t2(vec![0.3], [1, 1]);
        let mut state = StdpState::with_weights(&layer, &device, 1, w0.clone());

        // Pre stably welds every step -> the membrane of the post grows and
        // sooner or later exceeds the threshold -> the post will be spiked shortly AFTER
        // pre -> pair-STDP should potentiate the weight.
        for _ in 0..10 {
            let (_, new_state) = layer.step(t2(vec![1.0], [1, 1]), &state, &surrogate);
            state = new_state;
        }

        let w_final = vec2(state.w)[0];
        let w_init = vec2(w0)[0];
        assert!(
            w_final > w_init,
            "the repeated causal pair pre->post must potentiate the weight: {w_init} -> {w_final}"
        );
    }

    /// `RewardModulated` should not change the weights inside `step` - only
    /// collect eligibility trace. The weights must match the initialization
    /// up to and including explicitly calling `apply_reward`.
    ///
    /// The threshold is set to 0.5 deliberately: with the default
    /// (`threshold = 1.0`, `beta = 0.9`) and input `[1.0, 0.0]` the
    /// postsynaptic membrane reaches only ≈0.82 in 5 steps, never crosses
    /// the threshold, `post_hard` stays zero, and `pot`/`dep` in
    /// `eligibility_trace_update` are identically zero at every step
    /// (both depend on `post_trace`/`post_hard`). The test would then check
    /// “not accumulated” even with a broken function. At 0.5 the
    /// postsynaptic neuron spikes within 5 steps and the eligibility trace
    /// accumulates real co-activity pressure.
    #[test]
    fn reward_modulated_step_does_not_touch_weights() {
        let device = dev();
        let config = StdpLinearConfig::new(
            2,
            2,
            StdpRule::RewardModulated {
                a_plus: 0.1,
                a_minus: 0.05,
                tau_pre: 20.0,
                tau_post: 20.0,
                tau_e: 200.0,
            },
        )
        .with_threshold(0.5);
        let layer = StdpLinear::new(&config);
        let surrogate = FastSigmoid::default();

        let w0 = t2(vec![0.2, 0.2, 0.2, 0.2], [2, 2]);
        let mut state = StdpState::with_weights(&layer, &device, 1, w0.clone());

        for _ in 0..5 {
            let (_, new_state) = layer.step(t2(vec![1.0, 0.0], [1, 2]), &state, &surrogate);
            state = new_state;
        }

        let w_after_steps = vec2(state.w.clone());
        let w_init = vec2(w0);
        for (a, b) in w_after_steps.iter().zip(w_init.iter()) {
            assert!(
                (a - b).abs() < 1e-9,
                "weights should not change until apply_reward: {a} vs {b}"
            );
        }

        // The Eligibility trace must have accumulated (otherwise apply_reward
        // will not be able to apply anything).
        let e = state
            .eligibility
            .clone()
            .expect("eligibility must exist for RewardModulated");
        let e_sum: f32 = vec2(e).iter().map(|x| x.abs()).sum();
        assert!(
            e_sum > 0.0,
            "eligibility trace must accumulate non-zero co-activity pressure"
        );

        // Positive rewards are bound to change weights precisely through
        // apply_reward.
        let rewarded = layer.apply_reward(&state, 1.0, 0.1);
        let w_rewarded = vec2(rewarded.w);
        let changed = w_rewarded
            .iter()
            .zip(w_after_steps.iter())
            .any(|(a, b)| (a - b).abs() > 1e-9);
        assert!(changed, "apply_reward must change at least one weight");
    }

    /// The Triplet state must create slow traces; Pair/RewardModulated
    /// - no (otherwise unused tensors would waste memory on
    ///   each state step).
    #[test]
    fn zeros_allocates_slow_traces_only_for_triplet() {
        let device = dev();

        let pair_layer = StdpLinear::new(&pair_config(2, 2));
        let pair_state = StdpState::zeros(&pair_layer, &device, 1);
        assert!(pair_state.pre_trace_slow.is_none());
        assert!(pair_state.post_trace_slow.is_none());
        assert!(pair_state.eligibility.is_none());

        let triplet_config = StdpLinearConfig::new(
            2,
            2,
            StdpRule::Triplet {
                a2_plus: 0.05,
                a2_minus: 0.03,
                a3_plus: 0.02,
                a3_minus: 0.01,
                tau_pre: 20.0,
                tau_post: 20.0,
                tau_pre_slow: 100.0,
                tau_post_slow: 100.0,
            },
        );
        let triplet_layer = StdpLinear::new(&triplet_config);
        let triplet_state = StdpState::zeros(&triplet_layer, &device, 1);
        assert!(triplet_state.pre_trace_slow.is_some());
        assert!(triplet_state.post_trace_slow.is_some());
        assert!(triplet_state.eligibility.is_none());
    }

    /// Panic when the number of input features and layer configuration do not match
    /// - the same contract as the rest of the crate layers (see.
    ///   `SynapticDelay::step`).
    #[test]
    #[should_panic(expected = "features")]
    fn step_panics_on_feature_mismatch() {
        let device = dev();
        let config = pair_config(2, 2);
        let layer = StdpLinear::new(&config);
        let surrogate = FastSigmoid::default();
        let state = StdpState::zeros(&layer, &device, 1);

        let _ = layer.step(t2(vec![0.0, 0.0, 0.0], [1, 3]), &state, &surrogate);
    }
}
