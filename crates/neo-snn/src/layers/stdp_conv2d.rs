//! `StdpConv2d`: Conv2d → LIF, trained by STDP instead of backprop - convolutional
//! analogue of [`crate::layers::StdpLinear`] for unsupervised and on-chip
//! training (see [`crate::functional::plasticity`] for the rules themselves
//! weight updates).
//!
//! # Why is this not `StdpLinear` with reshape
//!
//! [`StdpLinear::step`](crate::layers::StdpLinear::step) treats Δw as one
//! outer product `post_trace^T @ pre_spike` over the ENTIRE vector
//! features. The convolution has the same filter `[c_out, c_in, kh, kw]`
//! applies in MANY positions, and the “pre” of each neuron is only
//! `kh × kw` window below it, not the entire `[C, H, W]`. Therefore the input
//! is expanded ([`unfold`], im2col) into patches
//! `[batch·h'·w', c_in·kh·kw]`, and Δw for the GENERAL filter is the same
//! pair/triplet outer-product, but with averaging over the batch AND over all
//! positions immediately.
//!
//! # Difference from [`SpikingConv2d`](crate::layers::SpikingConv2d)
//!
//! `SpikingConv2d` - `Module`, trained by backprop. `StdpConv2d`, like
//! `StdpLinear`, - intentional exception: weights - regular `Tensor<2>`
//! (flattened `[c_out, (c_in/groups)·kh·kw]`), part of the explicit state
//! ([`StdpConv2dState`]). Saving/loading state - via
//! [`StdpConv2dState::to_record`] / [`StdpConv2dState::from_record`]
//! (module [`crate::layers::record`]).
//!
//! # Rules
//!
//! The choice of rule is the same [`crate::layers::StdpRule`] as
//! `StdpLinear`: `Pair`, `Triplet { .. }`, `RewardModulated { .. }`.
//!
//! # Stride / padding / dilation / groups
//!
//! `stride`, `padding` (null), `dilation` and `groups` are supported.
//! At `groups > 1` the input is divided into `groups` blocks by `c_in/groups`
//! channels, output - to `groups` blocks along `c_out/groups` channels; everyone
//! the block has its own weights and is updated independently (depthwise -
//! `groups == c_in == c_out`). The order of the patch columns is `(c_in, kh, kw)`,
//! therefore, a group patch is a continuous slice of columns.
//!
//! # Bias
//!
//! STDP is a rule about SYNAPSES (pre/post pair), it has no natural
//! "shift updates". Therefore, bias here is the internal current of the neuron
//! (one per output channel), which is NOT changed by synaptic
//! rules. It is stored in state ([`StdpConv2dState::bias`]) and either
//! remains constant or adjusts homeostatically
//! ([`BiasHomeostasis`]): `b ← b + lr·(target_rate − rate)`, where `rate` —
//! the share of positions at which the channel worked in this step. This is local
//! rule of internal plasticity, not STDP.
//!
//! # Bugs
//!
//! All shape and configuration checks return [`SnnError`] from `try_*`
//! methods; methods without a prefix call them and panic with the same text.

use burn::tensor::Device;
use burn::tensor::{Distribution, Tensor};

use crate::error::{Result, SnnError, or_panic, tensor_to_vec};
use crate::functional::plasticity::{
    WeightBound, apply_reward, eligibility_trace_update, pair_update, triplet_update, update_trace,
};
use crate::layers::StdpRule;
use crate::layers::record::{
    RECORD_VERSION, StdpConv2dRecord, StdpHyper, check_dims, check_optional, check_version,
    decode_bound, encode_bound,
};
use crate::neuron::spike_with_scalar_threshold;
use crate::surrogate::Surrogate;

/// `decay = exp(-1/tau)`; `tau <= 0` gives `decay = 0` (trace without memory).
fn trace_decay(tau: f32) -> f32 {
    if tau <= 0.0 {
        0.0
    } else {
        (-1.0f32 / tau).exp()
    }
}

/// Output size along one axis; the formula is the same as
/// `burn::nn::conv::Conv2dConfig`. If the window does not fit into the input -
/// error instead of `usize` overflow.
fn conv_out_dim(
    size: usize,
    kernel: usize,
    stride: usize,
    padding: usize,
    dilation: usize,
) -> Result<usize> {
    let effective_kernel = dilation * (kernel - 1) + 1;
    let padded = size + 2 * padding;
    if padded < effective_kernel {
        return Err(SnnError::InvalidConfig {
            what: "StdpConv2d",
            reason: format!(
                "input size {size} (with padding {padding} on both sides = {padded}) less than \
                 effective kernel {effective_kernel} (kernel={kernel}, dilation={dilation})"
            ),
        });
    }
    Ok((padded - effective_kernel) / stride + 1)
}

fn expect_dims2(op: &'static str, name: &str, got: [usize; 2], want: [usize; 2]) -> Result<()> {
    if got == want {
        Ok(())
    } else {
        Err(SnnError::shape_msg(op, format!("{name} = {want:?}"), got))
    }
}

/// Expands `[batch, c_in, h, w]` into patches
/// `[batch * h' * w', c_in * kh * kw]` (im2col / unfold).
///
/// The order of rows is `(batch, h', w')`, columns are `(c_in, kh, kw)`; the same
/// order of the flattened weights, otherwise `matmul` in [`StdpConv2d::step`]
/// will silently produce a meaningful-looking but incorrect result. Padding
/// zero. Works via `Vec<f32>` on the host: unfold does not participate in
/// auto-differentiation graph (STDP, not backprop).
fn try_unfold(
    input: &Tensor<4>,
    kh: usize,
    kw: usize,
    stride: usize,
    padding: usize,
    dilation: usize,
) -> Result<(Tensor<2>, usize, usize)> {
    let [batch, c_in, h, w] = input.dims();
    let h_out = conv_out_dim(h, kh, stride, padding, dilation)?;
    let w_out = conv_out_dim(w, kw, stride, padding, dilation)?;
    let device = input.device();

    let src = tensor_to_vec(input.clone(), "StdpConv2d::unfold")?;
    let hp = h + 2 * padding;
    let wp = w + 2 * padding;
    let mut padded = vec![0.0f32; batch * c_in * hp * wp];
    for b in 0..batch {
        for c in 0..c_in {
            for i in 0..h {
                let src_row = ((b * c_in + c) * h + i) * w;
                let dst_row = ((b * c_in + c) * hp + (i + padding)) * wp + padding;
                padded[dst_row..dst_row + w].copy_from_slice(&src[src_row..src_row + w]);
            }
        }
    }

    let patch_len = c_in * kh * kw;
    let mut patches = vec![0.0f32; batch * h_out * w_out * patch_len];
    for b in 0..batch {
        for oy in 0..h_out {
            for ox in 0..w_out {
                let row = (b * h_out + oy) * w_out + ox;
                let mut col = 0usize;
                for c in 0..c_in {
                    for ky in 0..kh {
                        for kx in 0..kw {
                            let iy = oy * stride + ky * dilation;
                            let ix = ox * stride + kx * dilation;
                            patches[row * patch_len + col] =
                                padded[((b * c_in + c) * hp + iy) * wp + ix];
                            col += 1;
                        }
                    }
                }
            }
        }
    }

    let out = Tensor::<1>::from_floats(patches.as_slice(), &device)
        .reshape([batch * h_out * w_out, patch_len]);
    Ok((out, h_out, w_out))
}

/// Homeostasis bias: `b ← b + lr·(target_rate − rate)`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BiasHomeostasis {
    /// Target percentage of channel activations per step, `[0, 1]`.
    pub target_rate: f32,
    /// Adjustment speed.
    pub lr: f32,
}

/// Configuration [`StdpConv2d`].
#[derive(Debug, Clone, Copy)]
pub struct StdpConv2dConfig {
    /// Number of input channels.
    pub c_in: usize,
    /// Number of output channels.
    pub c_out: usize,
    /// Kernel size `(kh, kw)`.
    pub kernel_size: (usize, usize),
    /// Convolution step.
    pub stride: usize,
    /// Zero padding on each side.
    pub padding: usize,
    /// Kernel dilation.
    pub dilation: usize,
    /// Number of groups; `c_in` and `c_out` must be divisible by it.
    pub groups: usize,
    /// Is there internal current (bias) for each output channel.
    pub bias: bool,
    /// Homeostatic adjustment bias (makes sense only at `bias`).
    pub bias_homeostasis: Option<BiasHomeostasis>,
    /// Spike threshold of the postsynaptic LIF neuron.
    pub threshold: f32,
    /// Membrane damping β ∈ (0, 1).
    pub beta: f32,
    /// STDP-rule.
    pub rule: StdpRule,
    /// Limiting weights.
    pub bound: WeightBound,
    /// Range of uniform initialization of weights.
    pub init_range: [f32; 2],
}

impl StdpConv2dConfig {
    /// New configuration. Defaults like
    /// [`crate::layers::StdpLinearConfig::new`]: `threshold = 1.0`,
    /// `beta = 0.9`, soft-bound `[0, 1]`, initialization `[0, 0.4]`;
    /// `stride = 1`, `padding = 0`, `dilation = 1`, `groups = 1`, without bias.
    pub fn new(c_in: usize, c_out: usize, kernel_size: (usize, usize), rule: StdpRule) -> Self {
        Self {
            c_in,
            c_out,
            kernel_size,
            stride: 1,
            padding: 0,
            dilation: 1,
            groups: 1,
            bias: false,
            bias_homeostasis: None,
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

    /// Set the convolution step.
    pub fn with_stride(mut self, stride: usize) -> Self {
        self.stride = stride;
        self
    }

    /// Set padding.
    pub fn with_padding(mut self, padding: usize) -> Self {
        self.padding = padding;
        self
    }

    /// Set dilation.
    pub fn with_dilation(mut self, dilation: usize) -> Self {
        self.dilation = dilation;
        self
    }

    /// Set the number of groups.
    pub fn with_groups(mut self, groups: usize) -> Self {
        self.groups = groups;
        self
    }

    /// Enable bias (internal current), zero at start.
    pub fn with_bias(mut self) -> Self {
        self.bias = true;
        self
    }

    /// Enable bias with homeostatic adjustment.
    pub fn with_bias_homeostasis(mut self, target_rate: f32, lr: f32) -> Self {
        self.bias = true;
        self.bias_homeostasis = Some(BiasHomeostasis { target_rate, lr });
        self
    }

    /// Set the threshold.
    pub fn with_threshold(mut self, threshold: f32) -> Self {
        self.threshold = threshold;
        self
    }

    /// Set β.
    pub fn with_beta(mut self, beta: f32) -> Self {
        self.beta = beta;
        self
    }

    /// Set the weight limit.
    pub fn with_bound(mut self, bound: WeightBound) -> Self {
        self.bound = bound;
        self
    }

    /// Set the initialization range.
    pub fn with_init_range(mut self, range: [f32; 2]) -> Self {
        self.init_range = range;
        self
    }

    /// Create a layer; panics when the configuration is incorrect.
    pub fn init(&self) -> StdpConv2d {
        StdpConv2d::new(self)
    }

    /// Create a layer; invalid configuration - [`SnnError::InvalidConfig`].
    pub fn try_init(&self) -> Result<StdpConv2d> {
        StdpConv2d::try_new(self)
    }
}

/// Conv2d → LIF, trained by STDP. Weights - flattened explicit state
/// ([`StdpConv2dState`]), not a `Module` parameter - see the module documentation.
#[derive(Debug, Clone)]
pub struct StdpConv2d {
    c_in: usize,
    c_out: usize,
    kernel_size: (usize, usize),
    stride: usize,
    padding: usize,
    dilation: usize,
    groups: usize,
    bias: bool,
    bias_homeostasis: Option<BiasHomeostasis>,
    threshold: f32,
    beta: f32,
    rule: StdpRule,
    bound: WeightBound,
    init_range: [f32; 2],
}

fn invalid(reason: impl Into<String>) -> SnnError {
    SnnError::InvalidConfig {
        what: "StdpConv2d",
        reason: reason.into(),
    }
}

impl StdpConv2d {
    /// Create a layer; panics when the configuration is incorrect (see [`Self::try_new`]).
    #[track_caller]
    pub fn new(config: &StdpConv2dConfig) -> Self {
        or_panic(Self::try_new(config))
    }

    /// Create a layer by checking the configuration.
    pub fn try_new(config: &StdpConv2dConfig) -> Result<Self> {
        if config.c_in < 1 || config.c_out < 1 {
            return Err(invalid(format!(
                "c_in and c_out must be >= 1 (c_in={}, c_out={})",
                config.c_in, config.c_out
            )));
        }
        if config.kernel_size.0 < 1 || config.kernel_size.1 < 1 {
            return Err(invalid(format!(
                "kernel_size must be >= 1 on both axes, obtained {:?}",
                config.kernel_size
            )));
        }
        if config.stride < 1 || config.dilation < 1 {
            return Err(invalid(format!(
                "stride and dilation must be >= 1 (stride={}, dilation={})",
                config.stride, config.dilation
            )));
        }
        if config.groups < 1
            || config.c_in % config.groups != 0
            || config.c_out % config.groups != 0
        {
            return Err(invalid(format!(
                "groups={} must be >= 1 and divide c_in={} and c_out={}",
                config.groups, config.c_in, config.c_out
            )));
        }
        if !(config.beta > 0.0 && config.beta < 1.0) {
            return Err(invalid(format!(
                "beta must be in (0, 1), obtained {}",
                config.beta
            )));
        }
        if let StdpRule::Triplet {
            tau_pre_slow,
            tau_post_slow,
            ..
        } = config.rule
        {
            if !(tau_pre_slow > 0.0 && tau_post_slow > 0.0) {
                return Err(invalid("slow triplet constants STDP must be positive"));
            }
        }
        if let Some(h) = config.bias_homeostasis {
            if !config.bias {
                return Err(invalid("bias_homeostasis is set, but bias is disabled"));
            }
            if !(0.0..=1.0).contains(&h.target_rate) || h.lr < 0.0 {
                return Err(invalid(format!(
                    "bias_homeostasis: target_rate should be in [0, 1], lr >= 0 (obtained by {h:?})"
                )));
            }
        }

        Ok(Self {
            c_in: config.c_in,
            c_out: config.c_out,
            kernel_size: config.kernel_size,
            stride: config.stride,
            padding: config.padding,
            dilation: config.dilation,
            groups: config.groups,
            bias: config.bias,
            bias_homeostasis: config.bias_homeostasis,
            threshold: config.threshold,
            beta: config.beta,
            rule: config.rule,
            bound: config.bound,
            init_range: config.init_range,
        })
    }

    /// Recording a layer along with its state: weights, traces, bias, thresholds and
    /// hyperparameters (see [`crate::layers::record`]).
    pub fn to_record(&self, state: &StdpConv2dState) -> StdpConv2dRecord {
        let (rule_kind, rule_params) = self.rule.encode();
        let (bound_kind, bound_params) = encode_bound(self.bound);
        StdpConv2dRecord {
            c_in: self.c_in,
            c_out: self.c_out,
            kernel_h: self.kernel_size.0,
            kernel_w: self.kernel_size.1,
            stride: self.stride,
            padding: self.padding,
            dilation: self.dilation,
            groups: self.groups,
            bias: self.bias,
            bias_homeostasis: self
                .bias_homeostasis
                .map(|h| vec![h.target_rate, h.lr])
                .unwrap_or_default(),
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
    pub fn from_record(record: StdpConv2dRecord) -> Result<(Self, StdpConv2dState)> {
        const WHAT: &str = "StdpConv2d";
        let StdpConv2dRecord {
            c_in,
            c_out,
            kernel_h,
            kernel_w,
            stride,
            padding,
            dilation,
            groups,
            bias,
            bias_homeostasis,
            hyper,
            state,
        } = record;
        check_version(WHAT, &hyper)?;
        let rule = StdpRule::decode(hyper.rule_kind, &hyper.rule_params)?;
        let homeostasis = match bias_homeostasis.as_slice() {
            [] => None,
            &[target_rate, lr] => Some(BiasHomeostasis { target_rate, lr }),
            other => {
                return Err(crate::layers::record::record_err(
                    WHAT,
                    format!(
                        "bias_homeostasis: expected 0 or 2 numbers, received {}",
                        other.len()
                    ),
                ));
            }
        };
        let layer = Self::try_new(&StdpConv2dConfig {
            c_in,
            c_out,
            kernel_size: (kernel_h, kernel_w),
            stride,
            padding,
            dilation,
            groups,
            bias,
            bias_homeostasis: homeostasis,
            threshold: hyper.threshold,
            beta: hyper.beta,
            rule,
            bound: decode_bound(hyper.bound_kind, &hyper.bound_params)?,
            init_range: hyper.init_range,
        })?;

        let pg = layer.group_patch_len();
        let full = layer.full_patch_len();
        let n = state.mem.dims()[0];
        check_dims(WHAT, "w", state.w.dims(), [c_out, pg])?;
        check_dims(WHAT, "mem", state.mem.dims(), [n, c_out])?;
        check_dims(WHAT, "pre_trace", state.pre_trace.dims(), [n, full])?;
        check_dims(WHAT, "post_trace", state.post_trace.dims(), [n, c_out])?;
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
            check_dims(WHAT, "pre_trace_slow", t.dims(), [n, full])?;
        }
        if let Some(t) = &state.post_trace_slow {
            check_dims(WHAT, "post_trace_slow", t.dims(), [n, c_out])?;
        }
        if let Some(t) = &state.eligibility {
            check_dims(WHAT, "eligibility", t.dims(), [c_out, pg])?;
        }
        match (&state.bias, bias) {
            (Some(b), true) => check_dims(WHAT, "bias", b.dims(), [c_out])?,
            (None, false) => {}
            (Some(_), false) => {
                return Err(crate::layers::record::record_err(
                    WHAT,
                    "bias is present, but the layer is recorded without bias",
                ));
            }
            (None, true) => {
                return Err(crate::layers::record::record_err(
                    WHAT,
                    "there is no bias, but the layer is recorded with bias",
                ));
            }
        }
        Ok((layer, state))
    }

    /// Number of input channels.
    pub fn c_in(&self) -> usize {
        self.c_in
    }

    /// Number of output channels.
    pub fn c_out(&self) -> usize {
        self.c_out
    }

    /// Kernel size `(kh, kw)`.
    pub fn kernel_size(&self) -> (usize, usize) {
        self.kernel_size
    }

    /// Convolution step.
    pub fn stride(&self) -> usize {
        self.stride
    }

    /// Padding.
    pub fn padding(&self) -> usize {
        self.padding
    }

    /// Dilation.
    pub fn dilation(&self) -> usize {
        self.dilation
    }

    /// Number of groups.
    pub fn groups(&self) -> usize {
        self.groups
    }

    /// Is there bias?
    pub fn has_bias(&self) -> bool {
        self.bias
    }

    /// Spike threshold.
    pub fn threshold(&self) -> f32 {
        self.threshold
    }

    /// Membrane attenuation β.
    pub fn beta(&self) -> f32 {
        self.beta
    }

    /// STDP-rule.
    pub fn rule(&self) -> StdpRule {
        self.rule
    }

    /// Patch length of ONE group `(c_in/groups)·kh·kw` - number of columns
    /// flattened scales `state.w`.
    fn group_patch_len(&self) -> usize {
        let (kh, kw) = self.kernel_size;
        (self.c_in / self.groups) * kh * kw
    }

    /// The length of the complete (for all groups) patch `c_in·kh·kw`.
    fn full_patch_len(&self) -> usize {
        let (kh, kw) = self.kernel_size;
        self.c_in * kh * kw
    }

    fn group_out(&self) -> usize {
        self.c_out / self.groups
    }

    /// Output dimensions for input `h × w`; panics if the kernel is not
    /// placed in the input (see [`Self::try_output_size`]).
    #[track_caller]
    pub fn output_size(&self, h: usize, w: usize) -> (usize, usize) {
        or_panic(self.try_output_size(h, w))
    }

    /// Output dimensions for input `h × w`.
    pub fn try_output_size(&self, h: usize, w: usize) -> Result<(usize, usize)> {
        let (kh, kw) = self.kernel_size;
        Ok((
            conv_out_dim(h, kh, self.stride, self.padding, self.dilation)?,
            conv_out_dim(w, kw, self.stride, self.padding, self.dilation)?,
        ))
    }

    /// Convolution current `[n_positions, c_out]` for ready-made patches and weights
    /// `[c_out, (c_in/groups)·kh·kw]` - one block per group.
    fn drive(&self, patches: &Tensor<2>, w: &Tensor<2>) -> Tensor<2> {
        let n = patches.dims()[0];
        if self.groups == 1 {
            return patches.clone().matmul(w.clone().transpose());
        }
        let pg = self.group_patch_len();
        let go = self.group_out();
        let blocks = (0..self.groups)
            .map(|g| {
                let p = patches.clone().slice([0..n, g * pg..(g + 1) * pg]);
                let wg = w.clone().slice([g * go..(g + 1) * go, 0..pg]);
                p.matmul(wg.transpose())
            })
            .collect::<Vec<_>>();
        Tensor::cat(blocks, 1)
    }

    fn check_state(
        &self,
        op: &'static str,
        state: &StdpConv2dState,
        n_positions: usize,
    ) -> Result<()> {
        let pg = self.group_patch_len();
        let full = self.full_patch_len();
        expect_dims2(op, "state.w", state.w.dims(), [self.c_out, pg])?;
        expect_dims2(op, "state.mem", state.mem.dims(), [n_positions, self.c_out])?;
        expect_dims2(
            op,
            "state.pre_trace",
            state.pre_trace.dims(),
            [n_positions, full],
        )?;
        expect_dims2(
            op,
            "state.post_trace",
            state.post_trace.dims(),
            [n_positions, self.c_out],
        )?;
        if let Some(t) = &state.pre_trace_slow {
            expect_dims2(op, "state.pre_trace_slow", t.dims(), [n_positions, full])?;
        }
        if let Some(t) = &state.post_trace_slow {
            expect_dims2(
                op,
                "state.post_trace_slow",
                t.dims(),
                [n_positions, self.c_out],
            )?;
        }
        if let Some(t) = &state.eligibility {
            expect_dims2(op, "state.eligibility", t.dims(), [self.c_out, pg])?;
        }
        match (&state.bias, self.bias) {
            (Some(b), true) if b.dims() != [self.c_out] => {
                return Err(SnnError::shape_msg(
                    op,
                    format!("state.bias = [{}]", self.c_out),
                    b.dims(),
                ));
            }
            (None, true) => {
                return Err(SnnError::shape_msg(
                    op,
                    format!(
                        "state.bias = Some([{}]) (layer created with bias)",
                        self.c_out
                    ),
                    "None",
                ));
            }
            (Some(_), false) => {
                return Err(SnnError::shape_msg(
                    op,
                    "state.bias = None (layer created without bias)",
                    "Some",
                ));
            }
            _ => {}
        }
        Ok(())
    }

    /// One time step; panics with the text [`SnnError`] when
    /// inconsistent shapes (see [`Self::try_step`]).
    ///
    /// `input`: `[batch, c_in, h, w]`. Returns `(spikes
    /// [batch, c_out, h', w'], new state)`.
    #[track_caller]
    pub fn step<S: Surrogate + ?Sized>(
        &self,
        input: Tensor<4>,
        state: &StdpConv2dState,
        surrogate: &S,
    ) -> (Tensor<4>, StdpConv2dState) {
        or_panic(self.try_step(input, state, surrogate))
    }

    /// One time step with shape verification at the input.
    ///
    /// Like [`crate::layers::StdpLinear::step`]: `surrogate` is needed
    /// only to compute the post-spike in the same way as for the rest LIF
    /// crate; weights are not in the autodifferentiation graph.
    pub fn try_step<S: Surrogate + ?Sized>(
        &self,
        input: Tensor<4>,
        state: &StdpConv2dState,
        surrogate: &S,
    ) -> Result<(Tensor<4>, StdpConv2dState)> {
        const OP: &str = "StdpConv2d::step";
        let [batch, c_in, h, w] = input.dims();
        if c_in != self.c_in {
            return Err(SnnError::Channels {
                op: OP,
                expected: self.c_in,
                actual: c_in,
            });
        }
        let (h_out, w_out) = self.try_output_size(h, w)?;
        let n_positions = batch * h_out * w_out;
        self.check_state(OP, state, n_positions)?;

        let (kh, kw) = self.kernel_size;
        let (patches, _, _) = try_unfold(&input, kh, kw, self.stride, self.padding, self.dilation)?;

        let mut current = self.drive(&patches, &state.w);
        if let Some(b) = &state.bias {
            current = current + b.clone().reshape([1, self.c_out]);
        }
        let mem_new = state.mem.clone().mul_scalar(self.beta) + current;
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

        let device = input.device();
        let pg = self.group_patch_len();
        let go = self.group_out();
        let n_groups = self.groups;

        // Group slices: patch/pre-trace columns and output/post-trace columns.
        let pre_cols =
            |t: &Tensor<2>, g: usize| t.clone().slice([0..n_positions, g * pg..(g + 1) * pg]);
        let post_cols =
            |t: &Tensor<2>, g: usize| t.clone().slice([0..n_positions, g * go..(g + 1) * go]);
        let w_rows = |t: &Tensor<2>, g: usize| t.clone().slice([g * go..(g + 1) * go, 0..pg]);

        let mut w_new = state.w.clone();
        let mut pre_slow_new = state.pre_trace_slow.clone();
        let mut post_slow_new = state.post_trace_slow.clone();
        let mut eligibility_new = state.eligibility.clone();

        match self.rule {
            StdpRule::Pair {
                a_plus, a_minus, ..
            } => {
                // pair_update averages outer-product along axis 0 (here it is
                // batch·h'·w'), that is, both by batch and by positions -
                // correct for a shared filter.
                let blocks = (0..n_groups)
                    .map(|g| {
                        pair_update(
                            w_rows(&state.w, g),
                            pre_cols(&patches, g),
                            post_cols(&post_hard, g),
                            pre_cols(&state.pre_trace, g),
                            post_cols(&state.post_trace, g),
                            a_plus,
                            a_minus,
                            self.bound,
                        )
                    })
                    .collect::<Vec<_>>();
                w_new = Tensor::cat(blocks, 0);
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
                let pre_slow = state.pre_trace_slow.clone().unwrap_or_else(|| {
                    Tensor::zeros([n_positions, self.full_patch_len()], &device)
                });
                let post_slow = state
                    .post_trace_slow
                    .clone()
                    .unwrap_or_else(|| Tensor::zeros([n_positions, self.c_out], &device));

                let blocks = (0..n_groups)
                    .map(|g| {
                        triplet_update(
                            w_rows(&state.w, g),
                            pre_cols(&patches, g),
                            post_cols(&post_hard, g),
                            pre_cols(&state.pre_trace, g),
                            post_cols(&state.post_trace, g),
                            pre_cols(&pre_slow, g),
                            post_cols(&post_slow, g),
                            a2_plus,
                            a2_minus,
                            a3_plus,
                            a3_minus,
                            self.bound,
                        )
                    })
                    .collect::<Vec<_>>();
                w_new = Tensor::cat(blocks, 0);
                pre_slow_new = Some(update_trace(
                    pre_slow,
                    patches.clone(),
                    trace_decay(tau_pre_slow),
                ));
                post_slow_new = Some(update_trace(
                    post_slow,
                    post_hard.clone(),
                    trace_decay(tau_post_slow),
                ));
            }
            StdpRule::RewardModulated {
                a_plus,
                a_minus,
                tau_e,
                ..
            } => {
                let e = state
                    .eligibility
                    .clone()
                    .unwrap_or_else(|| Tensor::zeros([self.c_out, pg], &device));
                let blocks = (0..n_groups)
                    .map(|g| {
                        eligibility_trace_update(
                            w_rows(&e, g),
                            pre_cols(&patches, g),
                            post_cols(&post_hard, g),
                            pre_cols(&state.pre_trace, g),
                            post_cols(&state.post_trace, g),
                            a_plus,
                            a_minus,
                            trace_decay(tau_e),
                        )
                    })
                    .collect::<Vec<_>>();
                eligibility_new = Some(Tensor::cat(blocks, 0));
            }
        }

        let bias_new = match (&state.bias, self.bias_homeostasis) {
            (Some(b), Some(hom)) => {
                // The share of positions where the channel worked in this step.
                let rate = post_hard.clone().detach().mean_dim(0).reshape([self.c_out]);
                Some(
                    (b.clone() + (rate.neg().add_scalar(hom.target_rate)).mul_scalar(hom.lr))
                        .detach(),
                )
            }
            (b, _) => b.clone(),
        };

        let out_state = StdpConv2dState {
            w: w_new,
            mem: mem_reset,
            pre_trace: pre_trace_new,
            post_trace: post_trace_new,
            pre_trace_slow: pre_slow_new,
            post_trace_slow: post_slow_new,
            eligibility: eligibility_new,
            bias: bias_new,
        };

        // [n_positions, c_out] -> [batch, c_out, h', w']: lines went in
        // order (batch, h', w') - the same as unfold.
        let spikes = post_spike
            .reshape([batch, h_out, w_out, self.c_out])
            .permute([0, 3, 1, 2]);

        Ok((spikes, out_state))
    }

    /// Run through the sequence `[batch, time, c_in, h, w]`; panics
    /// in case of error (see [`Self::try_forward_sequence`]).
    #[track_caller]
    pub fn forward_sequence<S: Surrogate + ?Sized>(
        &self,
        input: Tensor<5>,
        state: StdpConv2dState,
        surrogate: &S,
    ) -> (Tensor<5>, StdpConv2dState) {
        or_panic(self.try_forward_sequence(input, state, surrogate))
    }

    /// Running through the prepared sequence `[batch, time, c_in, h, w]`.
    /// Equivalent to a cyclic call [`StdpConv2d::try_step`]; STDP
    /// applied at every step inside.
    pub fn try_forward_sequence<S: Surrogate + ?Sized>(
        &self,
        input: Tensor<5>,
        mut state: StdpConv2dState,
        surrogate: &S,
    ) -> Result<(Tensor<5>, StdpConv2dState)> {
        const OP: &str = "StdpConv2d::forward_sequence";
        let [batch, steps, c_in, h, w] = input.dims();
        if steps == 0 {
            return Err(SnnError::EmptySequence { op: OP });
        }
        if c_in != self.c_in {
            return Err(SnnError::Channels {
                op: OP,
                expected: self.c_in,
                actual: c_in,
            });
        }

        let mut outputs = Vec::with_capacity(steps);
        for t in 0..steps {
            let x_t = input
                .clone()
                .slice([0..batch, t..t + 1, 0..c_in, 0..h, 0..w])
                .reshape([batch, c_in, h, w]);
            let (spike, new_state) = self.try_step(x_t, &state, surrogate)?;
            state = new_state;
            outputs.push(spike);
        }

        Ok((Tensor::stack::<5>(outputs, 1), state))
    }

    /// Applies accumulated rewards to scales via eligibility
    /// trace; panics when there is an error (see [`Self::try_apply_reward`]).
    #[track_caller]
    pub fn apply_reward(&self, state: &StdpConv2dState, reward: f32, lr: f32) -> StdpConv2dState {
        or_panic(self.try_apply_reward(state, reward, lr))
    }

    /// Applies reward to scales. Only for
    /// [`StdpRule::RewardModulated`]; for `Pair`/`Triplet` the weights are already
    /// are updated inside `step`, and the call is a configuration error.
    pub fn try_apply_reward(
        &self,
        state: &StdpConv2dState,
        reward: f32,
        lr: f32,
    ) -> Result<StdpConv2dState> {
        if !matches!(self.rule, StdpRule::RewardModulated { .. }) {
            return Err(invalid(
                "apply_reward only makes sense when StdpRule::RewardModulated; \
                 for Pair/Triplet the weights are already updated inside step()",
            ));
        }
        expect_dims2(
            "StdpConv2d::apply_reward",
            "state.w",
            state.w.dims(),
            [self.c_out, self.group_patch_len()],
        )?;
        let e = state.eligibility.clone().unwrap_or_else(|| {
            Tensor::zeros([self.c_out, self.group_patch_len()], &state.w.device())
        });
        let w_new = apply_reward(state.w.clone(), &e, reward, lr, self.bound);
        Ok(StdpConv2dState {
            w: w_new,
            ..state.clone()
        })
    }
}

/// Explicit state [`StdpConv2d`]: flattened filter weights
/// `[c_out, (c_in/groups)·kh·kw]`, traces (patches and output shapes),
/// membrane, optional slow traces / eligibility / bias.
///
/// Weights are part of the state, not a modulus parameter (like
/// [`crate::layers::StdpState`]); saving - through
/// [`StdpConv2dState::to_record`].
#[derive(Debug, Clone)]
pub struct StdpConv2dState {
    /// Flattened filter weights `[c_out, (c_in/groups)·kh·kw]`.
    pub w: Tensor<2>,
    /// Membrane of the postsynaptic LIF `[batch·h'·w', c_out]`.
    pub mem: Tensor<2>,
    /// Fast presynaptic trace along `[batch·h'·w', c_in·kh·kw]` patches.
    pub pre_trace: Tensor<2>,
    /// Fast postsynaptic trace `[batch·h'·w', c_out]`.
    pub post_trace: Tensor<2>,
    /// Slow presynaptic trace (`Triplet` only).
    pub pre_trace_slow: Option<Tensor<2>>,
    /// Slow postsynaptic trace (`Triplet` only).
    pub post_trace_slow: Option<Tensor<2>>,
    /// Eligibility trace (`RewardModulated` only), weight shape.
    pub eligibility: Option<Tensor<2>>,
    /// Internal current through output channels `[c_out]` (only with `bias`).
    pub bias: Option<Tensor<1>>,
}

impl StdpConv2dState {
    /// Zero state; panics if the kernel does not fit into the input
    /// (see [`Self::try_zeros`]).
    #[track_caller]
    pub fn zeros(layer: &StdpConv2d, device: &Device, batch: usize, h: usize, w: usize) -> Self {
        or_panic(Self::try_zeros(layer, device, batch, h, w))
    }

    /// Zero state for an input with spatial dimensions `h × w`
    /// (BEFORE folding). Membrane and traces are zeros, weights are uniformly distributed
    /// `init_range`, bias (if any) - zeros.
    pub fn try_zeros(
        layer: &StdpConv2d,
        device: &Device,
        batch: usize,
        h: usize,
        w: usize,
    ) -> Result<Self> {
        let (h_out, w_out) = layer.try_output_size(h, w)?;
        let n_positions = batch * h_out * w_out;
        let pg = layer.group_patch_len();
        let full = layer.full_patch_len();
        let [lo, hi] = layer.init_range;

        let w_tensor = Tensor::random(
            [layer.c_out, pg],
            Distribution::Uniform(lo as f64, hi as f64),
            device,
        );

        let is_triplet = matches!(layer.rule, StdpRule::Triplet { .. });
        let pre_trace_slow = is_triplet.then(|| Tensor::zeros([n_positions, full], device));
        let post_trace_slow = is_triplet.then(|| Tensor::zeros([n_positions, layer.c_out], device));
        let eligibility = matches!(layer.rule, StdpRule::RewardModulated { .. })
            .then(|| Tensor::zeros([layer.c_out, pg], device));
        let bias = layer.bias.then(|| Tensor::zeros([layer.c_out], device));

        Ok(Self {
            w: w_tensor,
            mem: Tensor::zeros([n_positions, layer.c_out], device),
            pre_trace: Tensor::zeros([n_positions, full], device),
            post_trace: Tensor::zeros([n_positions, layer.c_out], device),
            pre_trace_slow,
            post_trace_slow,
            eligibility,
            bias,
        })
    }

    /// State with explicitly specified weights `[c_out, (c_in/groups)·kh·kw]`
    /// (parity tests where both sides start with the same weights).
    #[track_caller]
    pub fn with_weights(
        layer: &StdpConv2d,
        device: &Device,
        batch: usize,
        h: usize,
        w: usize,
        weights: Tensor<2>,
    ) -> Self {
        let mut s = Self::zeros(layer, device, batch, h, w);
        assert_eq!(
            weights.dims(),
            s.w.dims(),
            "StdpConv2dState::with_weights: weights shape expected {:?}, received {:?}",
            s.w.dims(),
            weights.dims()
        );
        s.w = weights;
        s
    }

    /// Detach the entire state from the auto-differentiation graph.
    pub fn detach(self) -> Self {
        Self {
            w: self.w.detach(),
            mem: self.mem.detach(),
            pre_trace: self.pre_trace.detach(),
            post_trace: self.post_trace.detach(),
            pre_trace_slow: self.pre_trace_slow.map(Tensor::detach),
            post_trace_slow: self.post_trace_slow.map(Tensor::detach),
            eligibility: self.eligibility.map(Tensor::detach),
            bias: self.bias.map(Tensor::detach),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::Device;
    use burn::tensor::module::conv2d;
    use burn::tensor::ops::ConvOptions;

    use crate::surrogate::FastSigmoid;

    fn dev() -> Device {
        Device::flex()
    }

    fn t4(data: Vec<f32>, shape: [usize; 4]) -> Tensor<4> {
        Tensor::<1>::from_floats(data.as_slice(), &dev()).reshape(shape)
    }

    fn t5(data: Vec<f32>, shape: [usize; 5]) -> Tensor<5> {
        Tensor::<1>::from_floats(data.as_slice(), &dev()).reshape(shape)
    }

    fn t2(data: Vec<f32>, shape: [usize; 2]) -> Tensor<2> {
        Tensor::<1>::from_floats(data.as_slice(), &dev()).reshape(shape)
    }

    fn vec1(t: Tensor<1>) -> Vec<f32> {
        t.into_data().try_to_vec::<f32>().unwrap()
    }

    fn vec2(t: Tensor<2>) -> Vec<f32> {
        t.into_data().try_to_vec::<f32>().unwrap()
    }

    fn vec4(t: Tensor<4>) -> Vec<f32> {
        t.into_data().try_to_vec::<f32>().unwrap()
    }

    fn pseudo(n: usize, seed: usize) -> Vec<f32> {
        (0..n)
            .map(|i| (((i + seed) * 37 % 101) as f32) / 101.0)
            .collect()
    }

    fn pair_rule() -> StdpRule {
        StdpRule::Pair {
            a_plus: 0.05,
            a_minus: 0.03,
            tau_pre: 20.0,
            tau_post: 20.0,
        }
    }

    fn triplet_rule() -> StdpRule {
        StdpRule::Triplet {
            a2_plus: 0.05,
            a2_minus: 0.03,
            a3_plus: 0.02,
            a3_minus: 0.01,
            tau_pre: 20.0,
            tau_post: 20.0,
            tau_pre_slow: 100.0,
            tau_post_slow: 100.0,
        }
    }

    fn reward_rule() -> StdpRule {
        StdpRule::RewardModulated {
            a_plus: 0.1,
            a_minus: 0.05,
            tau_pre: 20.0,
            tau_post: 20.0,
            tau_e: 200.0,
        }
    }

    fn pair_config(c_in: usize, c_out: usize, k: (usize, usize)) -> StdpConv2dConfig {
        StdpConv2dConfig::new(c_in, c_out, k, pair_rule()).with_bound(WeightBound::Hard {
            w_min: 0.0,
            w_max: 1.0,
        })
    }

    fn assert_close(a: &[f32], b: &[f32], tol: f32, what: &str) {
        assert_eq!(a.len(), b.len(), "{what}: different lengths");
        for (i, (&x, &y)) in a.iter().zip(b.iter()).enumerate() {
            assert!((x - y).abs() <= tol, "{what}[{i}]: {x} vs {y}");
        }
    }

    #[test]
    fn output_size_matches_standard_conv_formula() {
        let layer = StdpConv2d::new(&pair_config(1, 1, (3, 3)));
        assert_eq!(layer.output_size(5, 5), (3, 3));

        let strided = StdpConv2d::new(&pair_config(1, 1, (3, 3)).with_stride(2));
        assert_eq!(strided.output_size(7, 7), (3, 3));

        let padded = StdpConv2d::new(&pair_config(1, 1, (3, 3)).with_padding(1));
        assert_eq!(padded.output_size(5, 5), (5, 5));

        let dilated = StdpConv2d::new(&pair_config(1, 1, (3, 3)).with_dilation(2));
        assert_eq!(dilated.output_size(7, 7), (3, 3)); // effective kernel 5

        let rect = StdpConv2d::new(&pair_config(1, 1, (2, 3)));
        assert_eq!(rect.output_size(4, 6), (3, 4));
    }

    #[test]
    fn output_size_reports_kernel_larger_than_input() {
        let layer = StdpConv2d::new(&pair_config(1, 1, (3, 3)));
        let err = layer.try_output_size(2, 5).unwrap_err();
        assert!(matches!(err, SnnError::InvalidConfig { .. }), "{err:?}");
        assert!(err.to_string().contains("less"));
    }

    #[test]
    fn unfold_extracts_correct_receptive_patches() {
        let input = t4((0..9).map(|x| x as f32).collect(), [1, 1, 3, 3]);
        let (patches, h_out, w_out) = try_unfold(&input, 2, 2, 1, 0, 1).unwrap();
        assert_eq!((h_out, w_out), (2, 2));

        let p = vec2(patches)
            .chunks(4)
            .map(|c| c.to_vec())
            .collect::<Vec<_>>();
        assert_eq!(p[0], vec![0.0, 1.0, 3.0, 4.0]);
        assert_eq!(p[1], vec![1.0, 2.0, 4.0, 5.0]);
        assert_eq!(p[2], vec![3.0, 4.0, 6.0, 7.0]);
        assert_eq!(p[3], vec![4.0, 5.0, 7.0, 8.0]);
    }

    #[test]
    fn unfold_pads_with_zeros() {
        // 1x1x2x2 = [[1,2],[3,4]], kernel 2x2, padding 1 -> output 3x3.
        let input = t4(vec![1.0, 2.0, 3.0, 4.0], [1, 1, 2, 2]);
        let (patches, h_out, w_out) = try_unfold(&input, 2, 2, 1, 1, 1).unwrap();
        assert_eq!((h_out, w_out), (3, 3));
        let p = vec2(patches)
            .chunks(4)
            .map(|c| c.to_vec())
            .collect::<Vec<_>>();
        assert_eq!(p[0], vec![0.0, 0.0, 0.0, 1.0]); // top left corner
        assert_eq!(p[4], vec![1.0, 2.0, 3.0, 4.0]); // center
        assert_eq!(p[8], vec![4.0, 0.0, 0.0, 0.0]); // lower right corner
    }

    /// Main correctness check unfold + group matmul: current
    /// should coincide with `burn::tensor::module::conv2d` on the same scales
    /// for different stride/padding/dilation/groups.
    #[test]
    fn drive_matches_burn_conv2d() {
        // (c_in, c_out, kernel, stride, padding, dilation, groups, h, w)
        let cases = [
            (2, 3, (3, 3), 1, 0, 1, 1, 6, 6),
            (2, 3, (2, 3), 2, 1, 1, 1, 7, 8),
            (3, 2, (3, 3), 1, 2, 2, 1, 7, 7),
            (4, 6, (3, 3), 1, 1, 1, 2, 5, 5),
            (4, 4, (2, 2), 2, 0, 1, 4, 6, 6), // depthwise
            (6, 3, (3, 2), 1, 1, 2, 3, 6, 7),
        ];
        for (idx, &(c_in, c_out, k, stride, padding, dilation, groups, h, w)) in
            cases.iter().enumerate()
        {
            let layer = StdpConv2d::new(
                &pair_config(c_in, c_out, k)
                    .with_stride(stride)
                    .with_padding(padding)
                    .with_dilation(dilation)
                    .with_groups(groups),
            );
            let batch = 2;
            let input = t4(pseudo(batch * c_in * h * w, idx), [batch, c_in, h, w]);
            let pg = layer.group_patch_len();
            let weights = t2(pseudo(c_out * pg, idx + 7), [c_out, pg]);

            let (patches, h_out, w_out) =
                try_unfold(&input, k.0, k.1, stride, padding, dilation).unwrap();
            let ours = layer
                .drive(&patches, &weights)
                .reshape([batch, h_out, w_out, c_out])
                .permute([0, 3, 1, 2]);

            let w4 = weights.reshape([c_out, c_in / groups, k.0, k.1]);
            let reference = conv2d(
                input,
                w4,
                None,
                ConvOptions::new(
                    [stride, stride],
                    [padding, padding],
                    [dilation, dilation],
                    groups,
                ),
            );

            assert_eq!(ours.dims(), reference.dims(), "case {idx}: shapes");
            assert_close(&vec4(ours), &vec4(reference), 1e-4, &format!("case {idx}"));
        }
    }

    #[test]
    fn step_and_sequence_modes_agree() {
        let device = dev();
        let layer = StdpConv2d::new(&pair_config(1, 2, (2, 2)));
        let surrogate = FastSigmoid::default();

        let init_w = t2(vec![0.1, 0.2, 0.3, 0.4, 0.05, 0.15, 0.25, 0.35], [2, 4]);

        let steps: Vec<Vec<f32>> = vec![
            (0..9).map(|x| x as f32 * 0.1).collect(),
            (0..9).map(|x| (8 - x) as f32 * 0.1).collect(),
            vec![1.0; 9],
            vec![0.0; 9],
        ];
        let n = steps.len();

        let mut state = StdpConv2dState::with_weights(&layer, &device, 1, 3, 3, init_w.clone());
        let mut step_out = Vec::new();
        for s in &steps {
            let (spike, new_state) = layer.step(t4(s.clone(), [1, 1, 3, 3]), &state, &surrogate);
            state = new_state;
            step_out.extend(vec4(spike));
        }

        let flat: Vec<f32> = steps.iter().flat_map(|s| s.iter().copied()).collect();
        let seq_state = StdpConv2dState::with_weights(&layer, &device, 1, 3, 3, init_w);
        let (seq_spikes, seq_final_state) =
            layer.forward_sequence(t5(flat, [1, n, 1, 3, 3]), seq_state, &surrogate);
        let seq_out = vec4(seq_spikes.reshape([n, 2, 2, 2]));

        assert_close(&step_out, &seq_out, 1e-5, "spikes");
        assert_close(&vec2(state.w), &vec2(seq_final_state.w), 1e-5, "weight");
    }

    #[test]
    fn repeated_causal_pairing_potentiates_filter_weights() {
        let device = dev();
        let layer = StdpConv2d::new(&pair_config(1, 1, (2, 2)).with_beta(0.9).with_threshold(0.5));
        let surrogate = FastSigmoid::default();

        let w0 = t2(vec![0.3; 4], [1, 4]);
        let mut state = StdpConv2dState::with_weights(&layer, &device, 1, 2, 2, w0.clone());
        for _ in 0..10 {
            let (_, new_state) = layer.step(t4(vec![1.0; 4], [1, 1, 2, 2]), &state, &surrogate);
            state = new_state;
        }

        for (i, (&f, &init)) in vec2(state.w).iter().zip(vec2(w0).iter()).enumerate() {
            assert!(
                f > init,
                "the weight of the filter {i} must be potentiated: {init} -> {f}"
            );
        }
    }

    #[test]
    fn reward_modulated_step_does_not_touch_weights() {
        let device = dev();
        let layer = StdpConv2d::new(
            &StdpConv2dConfig::new(1, 1, (2, 2), reward_rule()).with_threshold(0.5),
        );
        let surrogate = FastSigmoid::default();

        let w0 = t2(vec![0.2; 4], [1, 4]);
        let mut state = StdpConv2dState::with_weights(&layer, &device, 1, 2, 2, w0.clone());
        for _ in 0..5 {
            let (_, new_state) = layer.step(t4(vec![1.0; 4], [1, 1, 2, 2]), &state, &surrogate);
            state = new_state;
        }

        let w_after = vec2(state.w.clone());
        assert_close(&w_after, &vec2(w0), 1e-9, "weights to apply_reward");

        let e = state
            .eligibility
            .clone()
            .expect("eligibility must exist for RewardModulated");
        assert!(
            vec2(e).iter().map(|x| x.abs()).sum::<f32>() > 0.0,
            "eligibility must accumulate co-activity"
        );

        let rewarded = layer.apply_reward(&state, 1.0, 0.1);
        assert!(
            vec2(rewarded.w)
                .iter()
                .zip(w_after.iter())
                .any(|(a, b)| (a - b).abs() > 1e-9),
            "apply_reward must change at least one weight"
        );
    }

    #[test]
    fn apply_reward_rejects_non_reward_rule() {
        let device = dev();
        let layer = StdpConv2d::new(&pair_config(1, 1, (2, 2)));
        let state = StdpConv2dState::zeros(&layer, &device, 1, 2, 2);
        let err = layer.try_apply_reward(&state, 1.0, 0.1).unwrap_err();
        assert!(matches!(err, SnnError::InvalidConfig { .. }));
    }

    #[test]
    fn zeros_allocates_slow_traces_only_for_triplet() {
        let device = dev();
        let pair = StdpConv2d::new(&pair_config(1, 1, (2, 2)));
        let s = StdpConv2dState::zeros(&pair, &device, 1, 3, 3);
        assert!(
            s.pre_trace_slow.is_none() && s.post_trace_slow.is_none() && s.eligibility.is_none()
        );

        let triplet = StdpConv2d::new(&StdpConv2dConfig::new(2, 3, (2, 2), triplet_rule()));
        let s = StdpConv2dState::zeros(&triplet, &device, 2, 3, 3);
        assert_eq!(
            s.pre_trace_slow.as_ref().unwrap().dims(),
            [2 * 2 * 2, 2 * 2 * 2]
        );
        assert_eq!(s.post_trace_slow.as_ref().unwrap().dims(), [2 * 2 * 2, 3]);
        assert!(s.eligibility.is_none());

        let reward = StdpConv2d::new(&StdpConv2dConfig::new(2, 3, (2, 2), reward_rule()));
        let s = StdpConv2dState::zeros(&reward, &device, 1, 3, 3);
        assert_eq!(s.eligibility.as_ref().unwrap().dims(), [3, 2 * 2 * 2]);
        assert!(s.pre_trace_slow.is_none());
    }

    #[test]
    fn state_zeros_matches_step_expected_mem_shape() {
        let device = dev();
        let layer = StdpConv2d::new(&pair_config(2, 4, (3, 3)).with_stride(2).with_padding(1));
        let state = StdpConv2dState::zeros(&layer, &device, 3, 7, 9);
        let (h_out, w_out) = layer.output_size(7, 9);
        assert_eq!(state.mem.dims(), [3 * h_out * w_out, 4]);

        let surrogate = FastSigmoid::default();
        let (spikes, next) = layer.step(
            t4(pseudo(3 * 2 * 7 * 9, 1), [3, 2, 7, 9]),
            &state,
            &surrogate,
        );
        assert_eq!(spikes.dims(), [3, 4, h_out, w_out]);
        assert_eq!(next.mem.dims(), state.mem.dims());
        assert_eq!(next.w.dims(), state.w.dims());
    }

    #[test]
    fn triplet_step_updates_slow_traces_and_weights() {
        let device = dev();
        let layer = StdpConv2d::new(
            &StdpConv2dConfig::new(1, 2, (2, 2), triplet_rule())
                .with_threshold(0.5)
                .with_bound(WeightBound::Hard {
                    w_min: 0.0,
                    w_max: 1.0,
                }),
        );
        let surrogate = FastSigmoid::default();
        let w0 = t2(vec![0.3; 8], [2, 4]);
        let mut state = StdpConv2dState::with_weights(&layer, &device, 1, 3, 3, w0.clone());
        for _ in 0..6 {
            let (_, s) = layer.step(t4(vec![1.0; 9], [1, 1, 3, 3]), &state, &surrogate);
            state = s;
        }
        let slow = vec2(state.pre_trace_slow.clone().unwrap());
        assert!(
            slow.iter().any(|&x| x > 0.0),
            "slow pre-trace is bound to accumulate"
        );
        assert!(
            vec2(state.w)
                .iter()
                .zip(vec2(w0).iter())
                .any(|(a, b)| (a - b).abs() > 1e-6)
        );
    }

    // ---------------------------------------------------------------- groups

    /// The layer with `groups = 2` must coincide with two independent layers
    /// `groups = 1`, each on its own half of the channels (both spikes and weights).
    #[test]
    fn grouped_layer_equals_independent_layers_per_group() {
        let device = dev();
        let surrogate = FastSigmoid::default();
        let cfg = |c_in, c_out| pair_config(c_in, c_out, (2, 2)).with_threshold(0.4);

        let grouped = StdpConv2d::new(&cfg(4, 4).with_groups(2));
        let part = StdpConv2d::new(&cfg(2, 2));

        let pg = grouped.group_patch_len();
        assert_eq!(pg, part.group_patch_len());
        let w_all = pseudo(4 * pg, 3);
        let w_a = t2(w_all[..2 * pg].to_vec(), [2, pg]);
        let w_b = t2(w_all[2 * pg..].to_vec(), [2, pg]);
        let w_g = t2(w_all.clone(), [4, pg]);

        let mut sg = StdpConv2dState::with_weights(&grouped, &device, 1, 3, 3, w_g);
        let mut sa = StdpConv2dState::with_weights(&part, &device, 1, 3, 3, w_a);
        let mut sb = StdpConv2dState::with_weights(&part, &device, 1, 3, 3, w_b);

        for step in 0..5 {
            let full = pseudo(4 * 9, step * 5);
            let xa = t4(full[..2 * 9].to_vec(), [1, 2, 3, 3]);
            let xb = t4(full[2 * 9..].to_vec(), [1, 2, 3, 3]);
            let xg = t4(full.clone(), [1, 4, 3, 3]);

            let (yg, ng) = grouped.step(xg, &sg, &surrogate);
            let (ya, na) = part.step(xa, &sa, &surrogate);
            let (yb, nb) = part.step(xb, &sb, &surrogate);
            sg = ng;
            sa = na;
            sb = nb;

            let ref_y = Tensor::cat(vec![ya, yb], 1);
            assert_close(
                &vec4(yg),
                &vec4(ref_y),
                1e-5,
                &format!("spikes, step {step}"),
            );
        }

        let w_ref = [vec2(sa.w), vec2(sb.w)].concat();
        assert_close(&vec2(sg.w), &w_ref, 1e-5, "group weights");
    }

    #[test]
    fn grouped_triplet_and_reward_shapes() {
        let device = dev();
        let surrogate = FastSigmoid::default();

        let tri =
            StdpConv2d::new(&StdpConv2dConfig::new(4, 6, (2, 2), triplet_rule()).with_groups(2));
        let st = StdpConv2dState::zeros(&tri, &device, 1, 3, 3);
        assert_eq!(st.w.dims(), [6, 2 * 2 * 2]);
        assert_eq!(st.pre_trace.dims(), [4, 4 * 2 * 2]);
        let (_, st) = tri.step(t4(pseudo(4 * 9, 2), [1, 4, 3, 3]), &st, &surrogate);
        assert_eq!(st.w.dims(), [6, 8]);

        let rew =
            StdpConv2d::new(&StdpConv2dConfig::new(4, 6, (2, 2), reward_rule()).with_groups(2));
        let sr = StdpConv2dState::zeros(&rew, &device, 1, 3, 3);
        let (_, sr) = rew.step(t4(pseudo(4 * 9, 2), [1, 4, 3, 3]), &sr, &surrogate);
        assert_eq!(sr.eligibility.as_ref().unwrap().dims(), [6, 8]);
    }

    #[test]
    fn depthwise_groups_keep_channels_independent() {
        let device = dev();
        let surrogate = FastSigmoid::default();
        let layer = StdpConv2d::new(&pair_config(2, 2, (2, 2)).with_groups(2).with_threshold(0.3));
        let state =
            StdpConv2dState::with_weights(&layer, &device, 1, 3, 3, t2(vec![0.5; 8], [2, 4]));

        // Channel 0 - ones, channel 1 - zeros.
        let mut x = vec![1.0; 9];
        x.extend(vec![0.0; 9]);
        // Updating weights uses traces BEFORE the current step, so
        // the first step Δw = 0 - at least the second is needed.
        let (y, state) = layer.step(t4(x.clone(), [1, 2, 3, 3]), &state, &surrogate);
        let y = vec4(y);
        assert!(y[..4].iter().all(|&v| v == 1.0), "channel 0 must work");
        assert!(
            y[4..].iter().all(|&v| v == 0.0),
            "channel 1 must remain silent without input"
        );
        let (_, next) = layer.step(t4(x, [1, 2, 3, 3]), &state, &surrogate);

        let w = vec2(next.w);
        assert!(
            w[..4].iter().all(|&v| v != 0.5),
            "channel 0 weights should have changed"
        );
        assert_close(
            &w[4..],
            &[0.5; 4],
            1e-9,
            "channel 1 weights should not change",
        );
    }

    #[test]
    fn invalid_groups_are_rejected() {
        let err = StdpConv2d::try_new(&pair_config(3, 4, (2, 2)).with_groups(2)).unwrap_err();
        assert!(matches!(err, SnnError::InvalidConfig { .. }));
        assert!(err.to_string().contains("groups"));
        assert!(StdpConv2d::try_new(&pair_config(4, 3, (2, 2)).with_groups(2)).is_err());
        assert!(StdpConv2d::try_new(&pair_config(4, 4, (2, 2)).with_groups(0)).is_err());
    }

    // ------------------------------------------------------------------ bias

    #[test]
    fn bias_adds_intrinsic_current() {
        let device = dev();
        let surrogate = FastSigmoid::default();
        let layer = StdpConv2d::new(&pair_config(1, 2, (2, 2)).with_bias().with_threshold(0.5));
        let mut state =
            StdpConv2dState::with_weights(&layer, &device, 1, 2, 2, t2(vec![0.0; 8], [2, 4]));
        state.bias = Some(Tensor::<1>::from_floats([1.0, 0.0], &device));

        let (y, _) = layer.step(t4(vec![0.0; 4], [1, 1, 2, 2]), &state, &surrogate);
        assert_close(
            &vec4(y),
            &[1.0, 0.0],
            0.0,
            "only the channel with bias is triggered at the zero input",
        );
    }

    #[test]
    fn bias_is_not_changed_by_stdp_without_homeostasis() {
        let device = dev();
        let surrogate = FastSigmoid::default();
        let layer = StdpConv2d::new(&pair_config(1, 1, (2, 2)).with_bias().with_threshold(0.5));
        let mut state = StdpConv2dState::zeros(&layer, &device, 1, 2, 2);
        state.bias = Some(Tensor::<1>::from_floats([0.7], &device));
        for _ in 0..4 {
            let (_, s) = layer.step(t4(vec![1.0; 4], [1, 1, 2, 2]), &state, &surrogate);
            state = s;
        }
        assert_close(&vec1(state.bias.unwrap()), &[0.7], 1e-7, "bias");
    }

    #[test]
    fn homeostasis_lowers_bias_of_overactive_channel_and_raises_silent_one() {
        let device = dev();
        let surrogate = FastSigmoid::default();
        let layer = StdpConv2d::new(
            &pair_config(1, 2, (2, 2))
                .with_bias_homeostasis(0.2, 0.1)
                .with_threshold(0.5),
        );
        let mut state =
            StdpConv2dState::with_weights(&layer, &device, 1, 2, 2, t2(vec![0.0; 8], [2, 4]));
        state.bias = Some(Tensor::<1>::from_floats([1.0, 0.0], &device));

        let (_, next) = layer.step(t4(vec![0.0; 4], [1, 1, 2, 2]), &state, &surrogate);
        let b = vec1(next.bias.unwrap());
        // channel 0: rate = 1 -> b = 1 + 0.1·(0.2 − 1) = 0.92; channel 1: rate = 0 -> b = 0.02
        assert_close(&b, &[0.92, 0.02], 1e-6, "bias after homeostasis");
    }

    #[test]
    fn bias_state_mismatch_is_reported() {
        let device = dev();
        let surrogate = FastSigmoid::default();
        let with_bias = StdpConv2d::new(&pair_config(1, 1, (2, 2)).with_bias());
        let plain = StdpConv2d::new(&pair_config(1, 1, (2, 2)));
        let state_plain = StdpConv2dState::zeros(&plain, &device, 1, 2, 2);
        let err = with_bias
            .try_step(t4(vec![0.0; 4], [1, 1, 2, 2]), &state_plain, &surrogate)
            .unwrap_err();
        assert!(err.to_string().contains("bias"), "{err}");

        let state_bias = StdpConv2dState::zeros(&with_bias, &device, 1, 2, 2);
        assert!(
            plain
                .try_step(t4(vec![0.0; 4], [1, 1, 2, 2]), &state_bias, &surrogate)
                .is_err()
        );
    }

    #[test]
    fn homeostasis_without_bias_is_rejected() {
        let mut cfg = pair_config(1, 1, (2, 2));
        cfg.bias_homeostasis = Some(BiasHomeostasis {
            target_rate: 0.1,
            lr: 0.1,
        });
        assert!(StdpConv2d::try_new(&cfg).is_err());
        assert!(
            StdpConv2d::try_new(&pair_config(1, 1, (2, 2)).with_bias_homeostasis(1.5, 0.1))
                .is_err()
        );
    }

    // ---------------------------------------------------------------- errors

    #[test]
    #[should_panic(expected = "channels")]
    fn step_panics_on_channel_mismatch() {
        let device = dev();
        let layer = StdpConv2d::new(&pair_config(1, 1, (2, 2)));
        let surrogate = FastSigmoid::default();
        let state = StdpConv2dState::zeros(&layer, &device, 1, 2, 2);
        let _ = layer.step(t4(vec![0.0; 8], [1, 2, 2, 2]), &state, &surrogate);
    }

    #[test]
    fn try_step_returns_typed_errors() {
        let device = dev();
        let surrogate = FastSigmoid::default();
        let layer = StdpConv2d::new(&pair_config(1, 1, (2, 2)));
        let state = StdpConv2dState::zeros(&layer, &device, 1, 3, 3);

        let err = layer
            .try_step(t4(vec![0.0; 8], [1, 2, 2, 2]), &state, &surrogate)
            .unwrap_err();
        assert_eq!(
            err,
            SnnError::Channels {
                op: "StdpConv2d::step",
                expected: 1,
                actual: 2
            }
        );

        // The state from the input is 3x3, and the input 4x4 -> mem is of the wrong shape.
        let err = layer
            .try_step(t4(vec![0.0; 16], [1, 1, 4, 4]), &state, &surrogate)
            .unwrap_err();
        match err {
            SnnError::Shape { op, expected, .. } => {
                assert_eq!(op, "StdpConv2d::step");
                assert!(expected.contains("state.mem"), "{expected}");
            }
            other => panic!("expected Shape, received {other:?}"),
        }

        // Another batch.
        let err = layer
            .try_step(t4(vec![0.0; 18], [2, 1, 3, 3]), &state, &surrogate)
            .unwrap_err();
        assert!(matches!(err, SnnError::Shape { .. }));
    }

    #[test]
    fn forward_sequence_rejects_empty_and_wrong_channels() {
        let device = dev();
        let surrogate = FastSigmoid::default();
        let layer = StdpConv2d::new(&pair_config(1, 1, (2, 2)));
        let state = StdpConv2dState::zeros(&layer, &device, 1, 3, 3);

        let empty = Tensor::<5>::zeros([1, 0, 1, 3, 3], &device);
        let err = layer
            .try_forward_sequence(empty, state.clone(), &surrogate)
            .unwrap_err();
        assert!(matches!(err, SnnError::EmptySequence { .. }));

        let wrong = Tensor::<5>::zeros([1, 2, 2, 3, 3], &device);
        let err = layer
            .try_forward_sequence(wrong, state, &surrogate)
            .unwrap_err();
        assert!(matches!(err, SnnError::Channels { .. }));
    }

    #[test]
    fn state_too_small_for_kernel_is_an_error_not_overflow() {
        let device = dev();
        let layer = StdpConv2d::new(&pair_config(1, 1, (3, 3)));
        assert!(StdpConv2dState::try_zeros(&layer, &device, 1, 2, 2).is_err());
    }
}
