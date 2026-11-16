//! `StdpConv1d`: convolutional (Conv1d) layer → LIF, trained with STDP instead
//! of backprop — the conv1d analog of [`StdpLinear`](super::StdpLinear) for
//! inputs with a SINGLE spatial/temporal/token axis (`[batch, channels,
//! length]`) rather than two (`[batch, channels, h, w]`, as in
//! [`SpikingConv2d`](super::SpikingConv2d)). The typical scenario is STDP over
//! a sliding window along a sequence: time series, audio, token embeddings,
//! etc., where 2D spatial structure does not apply but local correlation
//! (within `kernel_size` neighboring positions) still exists and is worth
//! learning with a co-activity rule rather than only with the fully connected
//! [`StdpLinear`].
//!
//! # How the kernel reduces to the same STDP primitives as [`StdpLinear`]
//!
//! [`crate::functional::plasticity`] works with 2D weights `[out_features,
//! in_features]`. The convolution kernel has shape `[out_channels,
//! in_channels, kernel_size]` — the "in_features" here is not a single input
//! vector but a window of `kernel_size` neighboring positions across all input
//! channels at once. The standard trick (the same principle as unfold/im2col
//! in an ordinary backprop convolution, where the gradient with respect to
//! the SHARED kernel is summed over sliding positions):
//!
//! 1. Unfold the input `[batch, in_ch, length]` into patches `[batch, num_pos,
//!    in_ch * kernel_size]` — one patch per kernel sliding position, with the
//!    same `in_ch`/`kernel` order inside the patch as in the flattened kernel
//!    `[out_ch, in_ch * kernel_size]`.
//! 2. Unfold the postsynaptic spike/trace `[batch, out_ch, num_pos]`
//!    into `[batch * num_pos, out_ch]`.
//! 3. Call [`pair_update`]/[`triplet_update`] with a "batch" equal to
//!    `batch * num_pos` — co-activity is averaged both over the regular batch
//!    and over all sliding positions AT ONCE, exactly as the gradient over the
//!    shared kernel is averaged in an ordinary backprop convolution (weight
//!    sharing).
//! 4. Reshape the flattened result `[out_ch, in_ch * kernel_size]` back into
//!    the kernel `[out_ch, in_ch, kernel_size]`.
//!
//! Like [`StdpLinear`], this layer does NOT implement `burn::module::Module`:
//! the kernel is part of the explicit state ([`StdpConv1dState`]), not a
//! parameter updated by the optimizer — see the `stdp_linear` module
//! documentation for a more detailed explanation of why STDP layers are
//! organized this way.
//!
//! # `RewardModulated`/R-STDP (MSTDPET) for conv1d
//!
//! [`StdpConv1dRule::RewardModulated`] is the same three-factor rule as in
//! [`StdpLinear`]/[`crate::functional::plasticity::eligibility_trace_update`],
//! applied to the unfolded patches in the same way as `Pair`/`Triplet` above:
//! the eligibility trace has the shape of the flattened kernel
//! `[out_channels, in_channels * kernel_size]` and accumulates at every step;
//! weights change only through an explicit [`StdpConv1d::apply_reward`]. This
//! is a direct analog of `mstdpet` (Florian, 2007) — the eligibility trace
//! here ALSO has its own (usually slow) decay constant `tau_e`, as in the
//! classical MSTDPET formulation, rather than vanishing instantly between
//! reward arrivals.

use burn::tensor::Device;
use burn::tensor::{Distribution, Tensor};

use crate::error::{Result, SnnError, or_panic};
use crate::layers::record::{
    RECORD_VERSION, StdpConv1dRecord, StdpHyper, check_dims, check_optional, check_version,
    decode_bound, encode_bound,
};
use crate::layers::stdp_linear::StdpRule;

use crate::functional::plasticity::{
    Identity, NoWeightDependence, WeightBound, apply_reward, eligibility_trace_update,
    pair_update_with_f_and_dep, triplet_update_with_f_and_dep, update_trace,
};
use crate::neuron::spike_with_scalar_threshold;
use crate::surrogate::Surrogate;

/// Which STDP rule [`StdpConv1d`] uses — the conv1d analog of
/// [`StdpRule`](super::stdp_linear::StdpRule). Field meanings are the same;
/// see the [`crate::functional::plasticity`] documentation for the formulas.
#[derive(Debug, Clone, Copy)]
pub enum StdpConv1dRule {
    Pair {
        a_plus: f32,
        a_minus: f32,
        tau_pre: f32,
        tau_post: f32,
    },
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
    /// R-STDP / MSTDPET (Florian, 2007) for conv1d: co-activity over the
    /// unfolded patches accumulates in an eligibility trace with the shape of
    /// the flattened kernel; the kernel changes only through an explicit
    /// [`StdpConv1d::apply_reward`].
    RewardModulated {
        a_plus: f32,
        a_minus: f32,
        tau_pre: f32,
        tau_post: f32,
        tau_e: f32,
    },
}

impl StdpConv1dRule {
    /// Equivalent to [`StdpRule`] (variant fields match one to one).
    fn to_rule(self) -> StdpRule {
        match self {
            StdpConv1dRule::Pair {
                a_plus,
                a_minus,
                tau_pre,
                tau_post,
            } => StdpRule::Pair {
                a_plus,
                a_minus,
                tau_pre,
                tau_post,
            },
            StdpConv1dRule::Triplet {
                a2_plus,
                a2_minus,
                a3_plus,
                a3_minus,
                tau_pre,
                tau_post,
                tau_pre_slow,
                tau_post_slow,
            } => StdpRule::Triplet {
                a2_plus,
                a2_minus,
                a3_plus,
                a3_minus,
                tau_pre,
                tau_post,
                tau_pre_slow,
                tau_post_slow,
            },
            StdpConv1dRule::RewardModulated {
                a_plus,
                a_minus,
                tau_pre,
                tau_post,
                tau_e,
            } => StdpRule::RewardModulated {
                a_plus,
                a_minus,
                tau_pre,
                tau_post,
                tau_e,
            },
        }
    }

    fn from_rule(rule: StdpRule) -> Self {
        match rule {
            StdpRule::Pair {
                a_plus,
                a_minus,
                tau_pre,
                tau_post,
            } => StdpConv1dRule::Pair {
                a_plus,
                a_minus,
                tau_pre,
                tau_post,
            },
            StdpRule::Triplet {
                a2_plus,
                a2_minus,
                a3_plus,
                a3_minus,
                tau_pre,
                tau_post,
                tau_pre_slow,
                tau_post_slow,
            } => StdpConv1dRule::Triplet {
                a2_plus,
                a2_minus,
                a3_plus,
                a3_minus,
                tau_pre,
                tau_post,
                tau_pre_slow,
                tau_post_slow,
            },
            StdpRule::RewardModulated {
                a_plus,
                a_minus,
                tau_pre,
                tau_post,
                tau_e,
            } => StdpConv1dRule::RewardModulated {
                a_plus,
                a_minus,
                tau_pre,
                tau_post,
                tau_e,
            },
        }
    }

    fn tau_pre(&self) -> f32 {
        match *self {
            StdpConv1dRule::Pair { tau_pre, .. } => tau_pre,
            StdpConv1dRule::Triplet { tau_pre, .. } => tau_pre,
            StdpConv1dRule::RewardModulated { tau_pre, .. } => tau_pre,
        }
    }

    fn tau_post(&self) -> f32 {
        match *self {
            StdpConv1dRule::Pair { tau_post, .. } => tau_post,
            StdpConv1dRule::Triplet { tau_post, .. } => tau_post,
            StdpConv1dRule::RewardModulated { tau_post, .. } => tau_post,
        }
    }
}

/// `decay = exp(-1/tau)`; `tau <= 0` -> `decay = 0` (the trace keeps no
/// history) — the same degenerate but safe case as in
/// `stdp_linear::trace_decay`.
fn trace_decay(tau: f32) -> f32 {
    if tau <= 0.0 {
        0.0
    } else {
        (-1.0f32 / tau).exp()
    }
}

/// Configuration of [`StdpConv1d`].
#[derive(Debug, Clone, Copy)]
pub struct StdpConv1dConfig {
    pub in_channels: usize,
    pub out_channels: usize,
    pub kernel_size: usize,
    /// Kernel sliding step along the length axis. `stride > 1` applies the
    /// same position subsampling as in a regular `Conv1d`.
    pub stride: usize,
    /// Spike threshold of the postsynaptic LIF neuron.
    pub threshold: f32,
    /// Membrane decay β ∈ (0, 1) of the postsynaptic LIF neuron.
    pub beta: f32,
    pub rule: StdpConv1dRule,
    pub bound: WeightBound,
    /// Range of the uniform kernel initialization.
    pub init_range: [f32; 2],
}

impl StdpConv1dConfig {
    /// New configuration with the required sizes and the STDP rule.
    /// `stride = 1`; other defaults are as in [`StdpLinearConfig`]
    /// (`threshold = 1.0`, `beta = 0.9`, soft bound `[0, 1]`,
    /// initialization `[0, 0.4]`).
    pub fn new(
        in_channels: usize,
        out_channels: usize,
        kernel_size: usize,
        rule: StdpConv1dRule,
    ) -> Self {
        Self {
            in_channels,
            out_channels,
            kernel_size,
            stride: 1,
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

    pub fn with_stride(mut self, stride: usize) -> Self {
        self.stride = stride;
        self
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

    pub fn init(&self) -> StdpConv1d {
        StdpConv1d::new(self)
    }
}

/// Conv1d → LIF trained with STDP. The kernel is part of the explicit state
/// ([`StdpConv1dState`]), not a `Module` parameter — see the module docs.
#[derive(Debug, Clone)]
pub struct StdpConv1d {
    in_channels: usize,
    out_channels: usize,
    kernel_size: usize,
    stride: usize,
    threshold: f32,
    beta: f32,
    rule: StdpConv1dRule,
    bound: WeightBound,
    init_range: [f32; 2],
}

impl StdpConv1d {
    pub fn new(config: &StdpConv1dConfig) -> Self {
        or_panic(Self::try_new(config))
    }

    /// Create the layer after validating the configuration ([`SnnError::InvalidConfig`]).
    pub fn try_new(config: &StdpConv1dConfig) -> Result<Self> {
        let bad = |reason: String| SnnError::InvalidConfig {
            what: "StdpConv1d",
            reason,
        };
        if config.in_channels < 1 {
            return Err(bad("in_channels must be >= 1".into()));
        }
        if config.out_channels < 1 {
            return Err(bad("out_channels must be >= 1".into()));
        }
        if config.kernel_size < 1 {
            return Err(bad("kernel_size must be >= 1".into()));
        }
        if config.stride < 1 {
            return Err(bad("stride must be >= 1".into()));
        }
        if !(config.beta > 0.0 && config.beta < 1.0) {
            return Err(bad(format!("beta must be in (0, 1), got {}", config.beta)));
        }
        if let StdpConv1dRule::Triplet {
            tau_pre_slow,
            tau_post_slow,
            ..
        } = config.rule
        {
            if !(tau_pre_slow > 0.0 && tau_post_slow > 0.0) {
                return Err(bad(
                    "the slow triplet STDP time constants must be positive".into()
                ));
            }
        }

        Ok(Self {
            in_channels: config.in_channels,
            out_channels: config.out_channels,
            kernel_size: config.kernel_size,
            stride: config.stride,
            threshold: config.threshold,
            beta: config.beta,
            rule: config.rule,
            bound: config.bound,
            init_range: config.init_range,
        })
    }

    /// Record of the layer together with its state: kernel, traces, thresholds
    /// and hyperparameters (see [`crate::layers::record`]).
    pub fn to_record(&self, state: &StdpConv1dState) -> StdpConv1dRecord {
        let (rule_kind, rule_params) = self.rule.to_rule().encode();
        let (bound_kind, bound_params) = encode_bound(self.bound);
        StdpConv1dRecord {
            in_channels: self.in_channels,
            out_channels: self.out_channels,
            kernel_size: self.kernel_size,
            stride: self.stride,
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

    /// Restore the layer and state from a record, validating the version,
    /// configuration and shapes of all tensors.
    pub fn from_record(record: StdpConv1dRecord) -> Result<(Self, StdpConv1dState)> {
        const WHAT: &str = "StdpConv1d";
        let StdpConv1dRecord {
            in_channels,
            out_channels,
            kernel_size,
            stride,
            hyper,
            state,
        } = record;
        check_version(WHAT, &hyper)?;
        let rule = StdpRule::decode(hyper.rule_kind, &hyper.rule_params)?;
        let layer = Self::try_new(&StdpConv1dConfig {
            in_channels,
            out_channels,
            kernel_size,
            stride,
            threshold: hyper.threshold,
            beta: hyper.beta,
            rule: StdpConv1dRule::from_rule(rule),
            bound: decode_bound(hyper.bound_kind, &hyper.bound_params)?,
            init_range: hyper.init_range,
        })?;

        let patch = in_channels * kernel_size;
        let eff = state.mem.dims()[0];
        check_dims(
            WHAT,
            "kernel",
            state.kernel.dims(),
            [out_channels, in_channels, kernel_size],
        )?;
        check_dims(WHAT, "mem", state.mem.dims(), [eff, out_channels])?;
        check_dims(WHAT, "pre_trace", state.pre_trace.dims(), [eff, patch])?;
        check_dims(
            WHAT,
            "post_trace",
            state.post_trace.dims(),
            [eff, out_channels],
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
            check_dims(WHAT, "pre_trace_slow", t.dims(), [eff, patch])?;
        }
        if let Some(t) = &state.post_trace_slow {
            check_dims(WHAT, "post_trace_slow", t.dims(), [eff, out_channels])?;
        }
        if let Some(t) = &state.eligibility {
            check_dims(WHAT, "eligibility", t.dims(), [out_channels, patch])?;
        }
        Ok((layer, state))
    }

    pub fn in_channels(&self) -> usize {
        self.in_channels
    }

    pub fn out_channels(&self) -> usize {
        self.out_channels
    }

    pub fn kernel_size(&self) -> usize {
        self.kernel_size
    }

    pub fn stride(&self) -> usize {
        self.stride
    }

    pub fn rule(&self) -> StdpConv1dRule {
        self.rule
    }

    /// Number of kernel sliding positions for an input of length `length` —
    /// the same calculation as for a `Conv1d` without padding
    /// (`VALID`/same as none): `(length - kernel_size) / stride + 1`.
    pub fn num_positions(&self, length: usize) -> usize {
        assert!(
            length >= self.kernel_size,
            "StdpConv1d: input length {length} is less than kernel_size {}",
            self.kernel_size
        );
        (length - self.kernel_size) / self.stride + 1
    }

    /// Unfolds `[batch, in_channels, length]` into patches
    /// `[batch * num_pos, in_channels * kernel_size]` — one patch per sliding
    /// position, with the order inside a patch: channel first, then offset
    /// within the window (`in_ch * kernel_size`, channel is the outer axis),
    /// CONSISTENT with the flattened kernel layout in
    /// [`StdpConv1dState::flat_kernel`]/[`StdpConv1dState::unflatten_kernel`].
    ///
    /// PERFORMANCE NOTE: with `stride == 1` there is one slice per offset
    /// within the window (`k` slices in total, each of length `num_pos` along
    /// the length axis). With `stride > 1` positions are built one at a time
    /// (`num_pos` slices of length 1, then `Tensor::stack` puts them back along
    /// the length axis) — the only construction primitive already known from
    /// the rest of the crate (`Tensor::stack`; `Tensor::cat` has no confirmed
    /// use anywhere else in the crate, and relying on an unverified signature
    /// is undesirable). For large inputs with `stride > 1` this is more
    /// expensive than a true strided slice would be; if it becomes a
    /// bottleneck, check whether `burn` adds `Tensor::narrow`/`select` with
    /// stride in future versions and switch to it.
    fn unfold_input(&self, input: Tensor<3>, num_pos: usize) -> Tensor<2> {
        let [batch, in_ch, _length] = input.dims();
        let k = self.kernel_size;

        let per_position: Vec<Tensor<3>> = if self.stride == 1 {
            // input[.., .., j..j+num_pos] for j = 0..k -> [batch, in_ch, num_pos] per offset.
            (0..k)
                .map(|j| input.clone().slice([0..batch, 0..in_ch, j..j + num_pos]))
                .collect()
        } else {
            // General case: one slice of length 1 per position, reassembled
            // back into the length axis via stack -> [batch, in_ch, num_pos]
            // per offset within the window.
            (0..k)
                .map(|j| {
                    let cols: Vec<Tensor<2>> = (0..num_pos)
                        .map(|p| {
                            let idx = p * self.stride + j;
                            input
                                .clone()
                                .slice([0..batch, 0..in_ch, idx..idx + 1])
                                .reshape([batch, in_ch])
                        })
                        .collect();
                    Tensor::stack::<3>(cols, 2)
                })
                .collect()
        };

        // Stack along a new last axis -> [batch, in_ch, num_pos, k].
        let stacked = Tensor::stack::<4>(per_position, 3);
        // permute -> [batch, num_pos, in_ch, k], then reshape -> patches.
        stacked
            .permute([0, 2, 1, 3])
            .reshape([batch * num_pos, in_ch * k])
    }

    /// One time step.
    ///
    /// `input`: `[batch, in_channels, length]`; `length` is fixed for the
    /// entire layer runtime (unlike [`StdpConv1d::forward_sequence`], there is
    /// NO separate time axis here — when applied to token/temporal windows, the
    /// calling code decides what counts as "one step": one `step` call sliding
    /// along the sequence for EACH window, or one call for the whole sequence
    /// treated as a spatial axis; see the module documentation).
    ///
    /// Returns `(spikes [batch, out_channels, num_pos], new state)`.
    pub fn step<S: Surrogate + ?Sized>(
        &self,
        input: Tensor<3>,
        state: &StdpConv1dState,
        surrogate: &S,
    ) -> (Tensor<3>, StdpConv1dState) {
        let [batch, in_ch, length] = input.dims();
        assert_eq!(
            in_ch, self.in_channels,
            "StdpConv1d::step: input has {in_ch} channels, layer is configured for {}",
            self.in_channels
        );
        let num_pos = self.num_positions(length);
        assert_eq!(
            state.mem.dims(),
            [batch * num_pos, self.out_channels],
            "StdpConv1d::step: state.mem is incompatible with the input shape \
             (expected [{}, {}], see StdpConv1d::state_zeros)",
            batch * num_pos,
            self.out_channels
        );

        let device = input.device();
        let patches = self.unfold_input(input, num_pos); // [batch*num_pos, in_ch*k]
        let flat_kernel = state.flat_kernel(); // [out_ch, in_ch*k]

        // Linear (over patches) -> LIF, the same path as StdpLinear::step.
        let mem_new = state.mem.clone().mul_scalar(self.beta)
            + patches.clone().matmul(flat_kernel.clone().transpose());
        let (post_spike, post_hard) =
            spike_with_scalar_threshold(mem_new.clone(), self.threshold, surrogate);
        let mem_reset = mem_new - post_hard.clone().mul_scalar(self.threshold);

        let pre_trace_new = update_trace(
            state.pre_trace.clone(),
            patches.clone(),
            trace_decay(self.rule.tau_pre()),
        );
        let post_trace_new = update_trace(
            state.post_trace.clone(),
            post_hard.clone(),
            trace_decay(self.rule.tau_post()),
        );

        let eff_batch = batch * num_pos;
        let patch_dim = self.in_channels * self.kernel_size;

        let (kernel_new, pre_slow_new, post_slow_new, eligibility_new) = match self.rule {
            StdpConv1dRule::Pair {
                a_plus, a_minus, ..
            } => {
                let w = pair_update_with_f_and_dep(
                    flat_kernel,
                    patches,
                    post_hard,
                    state.pre_trace.clone(),
                    state.post_trace.clone(),
                    a_plus,
                    a_minus,
                    self.bound,
                    &Identity,
                    &Identity,
                    &NoWeightDependence,
                );
                (w, None, None, state.eligibility.clone())
            }
            StdpConv1dRule::Triplet {
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
                    .unwrap_or_else(|| Tensor::zeros([eff_batch, patch_dim], &device));
                let post_slow = state
                    .post_trace_slow
                    .clone()
                    .unwrap_or_else(|| Tensor::zeros([eff_batch, self.out_channels], &device));

                let w = triplet_update_with_f_and_dep(
                    flat_kernel,
                    patches.clone(),
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
                    &Identity,
                    &Identity,
                    &NoWeightDependence,
                );

                let pre_slow_new = update_trace(pre_slow, patches, trace_decay(tau_pre_slow));
                let post_slow_new = update_trace(post_slow, post_hard, trace_decay(tau_post_slow));

                (
                    w,
                    Some(pre_slow_new),
                    Some(post_slow_new),
                    state.eligibility.clone(),
                )
            }
            StdpConv1dRule::RewardModulated {
                a_plus,
                a_minus,
                tau_e,
                ..
            } => {
                let e = state
                    .eligibility
                    .clone()
                    .unwrap_or_else(|| Tensor::zeros([self.out_channels, patch_dim], &device));
                let e_new = eligibility_trace_update(
                    e,
                    patches,
                    post_hard,
                    state.pre_trace.clone(),
                    state.post_trace.clone(),
                    a_plus,
                    a_minus,
                    trace_decay(tau_e),
                );
                (flat_kernel, None, None, Some(e_new))
            }
        };

        let out_state = StdpConv1dState {
            kernel: Self::unflatten_kernel(
                kernel_new,
                self.out_channels,
                self.in_channels,
                self.kernel_size,
            ),
            mem: mem_reset,
            pre_trace: pre_trace_new,
            post_trace: post_trace_new,
            pre_trace_slow: pre_slow_new.or_else(|| state.pre_trace_slow.clone()),
            post_trace_slow: post_slow_new.or_else(|| state.post_trace_slow.clone()),
            eligibility: eligibility_new,
        };

        let spikes = post_spike
            .reshape([batch, num_pos, self.out_channels])
            .permute([0, 2, 1]);

        (spikes, out_state)
    }

    /// Run through a sequence of windows `[batch, time, in_channels,
    /// window_length]` is the equivalent of a cyclic call to [`StdpConv1d::step`]
    /// at each time slice, STDP is applied at EVERY step inside
    /// (same contract as [`StdpLinear::forward_sequence`]). Use,
    /// when there is a separate axis "time step" ON TOP of the window along which
    /// the kernel slides - for example, STDP along token windows, where each step
    /// itself already contains several neighboring tokens, and `time` -
    /// this is what is outside sliding along the sequence of tokens further
    /// [`StdpConv1d::step`] is already folding.
    ///
    /// Returns `(spikes [batch, time, out_channels, num_pos], final state)`.
    pub fn forward_sequence<S: Surrogate + ?Sized>(
        &self,
        input: Tensor<4>,
        mut state: StdpConv1dState,
        surrogate: &S,
    ) -> (Tensor<4>, StdpConv1dState) {
        let [batch, steps, in_ch, window_length] = input.dims();
        assert!(
            steps > 0,
            "StdpConv1d::forward_sequence: num_steps must be > 0"
        );
        assert_eq!(
            in_ch, self.in_channels,
            "StdpConv1d::forward_sequence: input has {in_ch} channels, layer is configured to {}",
            self.in_channels
        );

        let mut outputs = Vec::with_capacity(steps);
        for t in 0..steps {
            let x_t = input
                .clone()
                .slice([0..batch, t..t + 1, 0..in_ch, 0..window_length])
                .reshape([batch, in_ch, window_length]);
            let (spike, new_state) = self.step(x_t, &state, surrogate);
            state = new_state;
            outputs.push(spike);
        }

        (Tensor::stack::<4>(outputs, 1), state)
    }

    /// Applies accumulated rewards to the kernel via an eligibility trace
    /// (only for [`StdpConv1dRule::RewardModulated`] - see
    /// [`StdpLinear::apply_reward`] for the same contract on the Linear side).
    pub fn apply_reward(&self, state: &StdpConv1dState, reward: f32, lr: f32) -> StdpConv1dState {
        debug_assert!(
            matches!(self.rule, StdpConv1dRule::RewardModulated { .. }),
            "StdpConv1d::apply_reward only makes sense when StdpConv1dRule::RewardModulated; \
             for Pair/Triplet the kernel is already updated inside step()"
        );

        let patch_dim = self.in_channels * self.kernel_size;
        let e = state.eligibility.clone().unwrap_or_else(|| {
            Tensor::zeros([self.out_channels, patch_dim], &state.kernel.device())
        });

        let flat_kernel = state.flat_kernel();
        let w_new = apply_reward(flat_kernel, &e, reward, lr, self.bound);

        StdpConv1dState {
            kernel: Self::unflatten_kernel(
                w_new,
                self.out_channels,
                self.in_channels,
                self.kernel_size,
            ),
            ..state.clone()
        }
    }

    fn unflatten_kernel(flat: Tensor<2>, out_ch: usize, in_ch: usize, k: usize) -> Tensor<3> {
        flat.reshape([out_ch, in_ch, k])
    }

    /// Zero state for the output shape `(batch, num_pos)` given an input
    /// of a given length - a convenient wrapper over [`StdpConv1dState::zeros`],
    /// parallel [`SpikingConv2d::state_zeros`](super::SpikingConv2d::state_zeros):
    /// itself retrieves `num_pos` from `length` so that the calling code does not need
    /// it was necessary to recalculate it manually before each `zeros`.
    pub fn state_zeros(&self, device: &Device, batch: usize, length: usize) -> StdpConv1dState {
        StdpConv1dState::zeros(self, device, batch * self.num_positions(length))
    }
}

/// Explicit state of [`StdpConv1d`]: kernel + traces + postsynaptic membrane
/// LIF + (optional) slow traces for triplet rules and eligibility
/// trace for the reward-modulated rule. The same principle of "weight is part
/// states, not a module parameter", which is what [`StdpState`](super::stdp_linear::StdpState) has.
#[derive(Debug, Clone)]
pub struct StdpConv1dState {
    /// Kernel `[out_channels, in_channels, kernel_size]`.
    pub kernel: Tensor<3>,
    /// Membrane of the postsynaptic LIF `[batch*num_pos, out_channels]`.
    pub mem: Tensor<2>,
    /// Fast presynaptic trace (along unfolded patches) `[batch*num_pos, in_channels*kernel_size]`.
    pub pre_trace: Tensor<2>,
    /// Fast postsynaptic trace `[batch*num_pos, out_channels]`.
    pub post_trace: Tensor<2>,
    /// Slow presynaptic trace (only for `Triplet`).
    pub pre_trace_slow: Option<Tensor<2>>,
    /// Slow postsynaptic trace (only for `Triplet`).
    pub post_trace_slow: Option<Tensor<2>>,
    /// Eligibility trace (only for `RewardModulated`), flattened shape
    /// kernels `[out_channels, in_channels*kernel_size]`.
    pub eligibility: Option<Tensor<2>>,
}

impl StdpConv1dState {
    /// Flattens the kernel `[out_ch, in_ch, k]` -> `[out_ch, in_ch*k]` — layout,
    /// MATCHED with [`StdpConv1d::unfold_input`] (order: channel -
    /// external axis, displacement inside the window - internal).
    pub fn flat_kernel(&self) -> Tensor<2> {
        let [out_ch, in_ch, k] = self.kernel.dims();
        self.kernel.clone().reshape([out_ch, in_ch * k])
    }

    /// Zero state: membrane and all traces are zeros, kernel is random
    /// uniform initialization to `init_range` from layer configuration.
    /// `eff_batch` - already calculated `batch * num_positions(length)` (see.
    /// [`StdpConv1d::state_zeros`] behind a convenient wrapper that computes
    /// itself from the input length).
    pub fn zeros(layer: &StdpConv1d, device: &Device, eff_batch: usize) -> Self {
        let [lo, hi] = layer.init_range;
        let patch_dim = layer.in_channels * layer.kernel_size;
        let kernel = Tensor::random(
            [layer.out_channels, layer.in_channels, layer.kernel_size],
            Distribution::Uniform(lo as f64, hi as f64),
            device,
        );

        let pre_trace_slow = matches!(layer.rule, StdpConv1dRule::Triplet { .. })
            .then(|| Tensor::zeros([eff_batch, patch_dim], device));
        let post_trace_slow = matches!(layer.rule, StdpConv1dRule::Triplet { .. })
            .then(|| Tensor::zeros([eff_batch, layer.out_channels], device));
        let eligibility = matches!(layer.rule, StdpConv1dRule::RewardModulated { .. })
            .then(|| Tensor::zeros([layer.out_channels, patch_dim], device));

        Self {
            kernel,
            mem: Tensor::zeros([eff_batch, layer.out_channels], device),
            pre_trace: Tensor::zeros([eff_batch, patch_dim], device),
            post_trace: Tensor::zeros([eff_batch, layer.out_channels], device),
            pre_trace_slow,
            post_trace_slow,
            eligibility,
        }
    }

    /// State with an explicit kernel (for parity tests
    /// `step`/`forward_sequence`, see [`StdpState::with_weights`](super::stdp_linear::StdpState::with_weights)).
    pub fn with_kernel(
        layer: &StdpConv1d,
        device: &Device,
        eff_batch: usize,
        kernel: Tensor<3>,
    ) -> Self {
        let mut s = Self::zeros(layer, device, eff_batch);
        s.kernel = kernel;
        s
    }

    /// Detach the entire state from the auto-differentiation graph - see.
    /// [`StdpState::detach`](super::stdp_linear::StdpState::detach) then
    /// the same contract.
    pub fn detach(self) -> Self {
        Self {
            kernel: self.kernel.detach(),
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

    fn t3(data: Vec<f32>, shape: [usize; 3]) -> Tensor<3> {
        Tensor::<1>::from_floats(data.as_slice(), &dev()).reshape(shape)
    }

    fn t4(data: Vec<f32>, shape: [usize; 4]) -> Tensor<4> {
        Tensor::<1>::from_floats(data.as_slice(), &dev()).reshape(shape)
    }

    fn vec2(t: Tensor<2>) -> Vec<f32> {
        t.into_data().try_to_vec::<f32>().unwrap()
    }

    fn vec3(t: Tensor<3>) -> Vec<f32> {
        t.into_data().try_to_vec::<f32>().unwrap()
    }

    fn pair_config(in_ch: usize, out_ch: usize, k: usize) -> StdpConv1dConfig {
        StdpConv1dConfig::new(
            in_ch,
            out_ch,
            k,
            StdpConv1dRule::Pair {
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

    /// `num_positions` must follow the same formula as normal
    /// `Conv1d` without padding: `(length - kernel_size) / stride + 1`.
    #[test]
    fn num_positions_matches_valid_conv_formula() {
        let layer = StdpConv1d::new(&pair_config(1, 1, 3));
        assert_eq!(layer.num_positions(3), 1);
        assert_eq!(layer.num_positions(5), 3);

        let strided = StdpConv1d::new(&pair_config(1, 1, 3).with_stride(2));
        assert_eq!(strided.num_positions(7), 3);
    }

    /// `step` in a loop over time windows and `forward_sequence` must
    /// give the same result (same spikes, same final
    /// state) with the same inputs and the same starting point -
    /// same contract as `StdpLinear::step_and_sequence_modes_agree`.
    #[test]
    fn step_and_sequence_modes_agree() {
        let device = dev();
        let config = pair_config(1, 2, 2); // in_ch=1, out_ch=2, kernel=2
        let layer = StdpConv1d::new(&config);
        let surrogate = FastSigmoid::default();

        let init_kernel = t3(vec![0.1, 0.2, 0.3, 0.15], [2, 1, 2]);
        let window_length = 4usize; // num_pos = 3
        let steps: Vec<Vec<f32>> = vec![
            vec![1.0, 0.0, 1.0, 0.0],
            vec![0.0, 1.0, 0.0, 1.0],
            vec![1.0, 1.0, 0.0, 0.0],
        ];
        let n = steps.len();
        let num_pos = layer.num_positions(window_length);

        // Step mode.
        let mut state =
            StdpConv1dState::with_kernel(&layer, &device, 1 * num_pos, init_kernel.clone());
        let mut step_out = Vec::with_capacity(n);
        for s in &steps {
            let x_t = t3(s.clone(), [1, 1, window_length]);
            let (spike, new_state) = layer.step(x_t, &state, &surrogate);
            state = new_state;
            step_out.push(spike);
        }

        // Sequential mode, same starting point.
        let flat: Vec<f32> = steps.iter().flat_map(|s| s.iter().copied()).collect();
        let seq_state = StdpConv1dState::with_kernel(&layer, &device, 1 * num_pos, init_kernel);
        let (seq_spikes, seq_final_state) =
            layer.forward_sequence(t4(flat, [1, n, 1, window_length]), seq_state, &surrogate);

        for (t, step_spike) in step_out.iter().enumerate() {
            let seq_spike = seq_spikes
                .clone()
                .slice([0..1, t..t + 1, 0..2, 0..num_pos])
                .reshape([2, num_pos]);
            let a = vec3(step_spike.clone().reshape([1, 2, num_pos]));
            let b = vec2(seq_spike);
            for (i, (&x, &y)) in a.iter().zip(b.iter()).enumerate() {
                assert!(
                    (x - y).abs() < 1e-5,
                    "step {t}, element {i}: step={x} vs sequence={y}"
                );
            }
        }

        let k_step = vec3(state.kernel);
        let k_seq = vec3(seq_final_state.kernel);
        for (i, (&a, &b)) in k_step.iter().zip(k_seq.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-5,
                "kernel element {i}: step={a} vs sequence={b}"
            );
        }
    }

    /// Basic sanity check of the training direction: repeatable
    /// causal co-activity (pre stably spikes each step) is obliged
    /// potentiate at least one kernel weight - the same principle as
    /// `repeated_causal_pairing_potentiates_weight` at `StdpLinear`.
    #[test]
    fn repeated_causal_pairing_potentiates_kernel() {
        let device = dev();
        let config = pair_config(1, 1, 2).with_beta(0.9).with_threshold(0.5);
        let layer = StdpConv1d::new(&config);
        let surrogate = FastSigmoid::default();

        let window_length = 3usize; // num_pos = 2
        let num_pos = layer.num_positions(window_length);
        let k0 = t3(vec![0.3, 0.3], [1, 1, 2]);
        let mut state = StdpConv1dState::with_kernel(&layer, &device, num_pos, k0.clone());

        for _ in 0..10 {
            let x = t3(vec![1.0, 1.0, 1.0], [1, 1, window_length]);
            let (_, new_state) = layer.step(x, &state, &surrogate);
            state = new_state;
        }

        let k_final = vec3(state.kernel);
        let k_init = vec3(k0);
        let any_potentiated = k_final.iter().zip(k_init.iter()).any(|(f, i)| f > i);
        assert!(
            any_potentiated,
            "repeated causal co-activity must potentiate at least one kernel weight: {:?} -> {:?}",
            k_init, k_final
        );
    }

    /// `RewardModulated` should not change the kernel inside `step` - only
    /// accumulate eligibility trace, exactly like `StdpLinear`
    /// (`reward_modulated_step_does_not_touch_weights`).
    #[test]
    fn reward_modulated_step_does_not_touch_kernel() {
        let device = dev();
        let config = StdpConv1dConfig::new(
            1,
            1,
            2,
            StdpConv1dRule::RewardModulated {
                a_plus: 0.1,
                a_minus: 0.05,
                tau_pre: 20.0,
                tau_post: 20.0,
                tau_e: 200.0,
            },
        )
        .with_threshold(0.5);
        let layer = StdpConv1d::new(&config);
        let surrogate = FastSigmoid::default();

        let window_length = 3usize;
        let num_pos = layer.num_positions(window_length);
        let k0 = t3(vec![0.2, 0.2], [1, 1, 2]);
        let mut state = StdpConv1dState::with_kernel(&layer, &device, num_pos, k0.clone());

        for _ in 0..5 {
            let x = t3(vec![1.0, 0.0, 1.0], [1, 1, window_length]);
            let (_, new_state) = layer.step(x, &state, &surrogate);
            state = new_state;
        }

        let k_after_steps = vec3(state.kernel.clone());
        let k_init = vec3(k0);
        for (a, b) in k_after_steps.iter().zip(k_init.iter()) {
            assert!(
                (a - b).abs() < 1e-9,
                "kernel should not change before apply_reward: {a} vs {b}"
            );
        }

        let e = state
            .eligibility
            .clone()
            .expect("eligibility must exist for RewardModulated");
        let e_sum: f32 = vec2(e).iter().map(|x| x.abs()).sum();
        assert!(
            e_sum > 0.0,
            "eligibility trace must accumulate non-zero co-activity pressure"
        );

        let rewarded = layer.apply_reward(&state, 1.0, 0.1);
        let k_rewarded = vec3(rewarded.kernel);
        let changed = k_rewarded
            .iter()
            .zip(k_after_steps.iter())
            .any(|(a, b)| (a - b).abs() > 1e-9);
        assert!(
            changed,
            "apply_reward must change at least one kernel weight"
        );
    }

    /// The Triplet state must create slow traces; Pair/RewardModulated
    /// - no (same principle as `zeros_allocates_slow_traces_only_for_triplet`
    /// y `StdpLinear`).
    #[test]
    fn zeros_allocates_slow_traces_only_for_triplet() {
        let device = dev();

        let pair_layer = StdpConv1d::new(&pair_config(2, 2, 2));
        let pair_state = StdpConv1dState::zeros(&pair_layer, &device, 1);
        assert!(pair_state.pre_trace_slow.is_none());
        assert!(pair_state.post_trace_slow.is_none());
        assert!(pair_state.eligibility.is_none());

        let triplet_config = StdpConv1dConfig::new(
            2,
            2,
            2,
            StdpConv1dRule::Triplet {
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
        let triplet_layer = StdpConv1d::new(&triplet_config);
        let triplet_state = StdpConv1dState::zeros(&triplet_layer, &device, 1);
        assert!(triplet_state.pre_trace_slow.is_some());
        assert!(triplet_state.post_trace_slow.is_some());
        assert!(triplet_state.eligibility.is_none());
    }

    /// Panic when the number of input channels and layer configuration does not match
    /// - the same contract as `StdpLinear`/`SynapticDelay`.
    #[test]
    #[should_panic(expected = "channels")]
    fn step_panics_on_channel_mismatch() {
        let device = dev();
        let config = pair_config(2, 2, 2);
        let layer = StdpConv1d::new(&config);
        let surrogate = FastSigmoid::default();
        let state = layer.state_zeros(&device, 1, 3);

        // 3 channel input with layer configured to 2.
        let _ = layer.step(t3(vec![0.0; 9], [1, 3, 3]), &state, &surrogate);
    }

    /// `state_zeros` is required to compute `num_pos` from the input length in the same way as
    /// this makes `step` internally - otherwise `assert_eq!` to the shape `mem`
    /// internally `step` would always panic even on valid calls.
    #[test]
    fn state_zeros_matches_step_expected_mem_shape() {
        let device = dev();
        let config = pair_config(1, 3, 2);
        let layer = StdpConv1d::new(&config);
        let surrogate = FastSigmoid::default();

        let state = layer.state_zeros(&device, 2, 4); // batch=2, length=4, k=2 -> num_pos=3
        assert_eq!(state.mem.dims(), [2 * 3, 3]);

        let x = t3(vec![0.0; 2 * 1 * 4], [2, 1, 4]);
        let (spikes, _) = layer.step(x, &state, &surrogate);
        assert_eq!(spikes.dims(), [2, 3, 3]); // [batch, out_ch, num_pos]
    }
}
