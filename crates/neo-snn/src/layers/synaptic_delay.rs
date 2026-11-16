//! `SynapticDelay`: per-channel trainable signal transmission delay.
//!
//! # Why
//!
//! Not a single neuron in [`crate::neuron`] delays the signal: the spike/current
//! computed at step `t` reaches the next layer, also at step `t`.
//! Axonal and synaptic delays are a standard feature of biological
//! chains, and, as shown in the work of Hammouamri, Khalfaoui-Hassani &
//! Masquelier (ICLR 2024) about learnable dilated convolutions with learnable
//! spacing, trainable delay is a competitive parameter specifically for
//! where it is important not only WHAT arrived, but also WHEN.
//! This is a natural complement to [`crate::functional::loss::mse_temporal_loss`]
//! and to [`crate::encoding::latency_encode`] - both already treat time
//! as a trainable quantity, but there was no delay anywhere between them.
//!
//! `SynapticDelay` - a simple layer without a membrane and without a threshold: it is placed
//! BETWEEN projection and neuron (`Linear → SynapticDelay → Leaky`), exactly
//! just like the original idea. Each channel is shifted by its own amount
//! `d_c ≥ 0`, fractional and trainable.
//!
//! # How is fractional delay calculated
//!
//! The delay does not have to be an integer number of steps - otherwise the gradient along it would be
//! would be zero everywhere (the shift by an integer number of steps is piecewise constant). Here
//! linear (2-point) interpolation is used between two adjacent
//! whole shifts:
//!
//! ```text
//! d = floor(d) + frac(d)
//! out[t] = (1 − frac(d))·in[t − floor(d)] + frac(d)·in[t − floor(d) − 1]
//! ```
//!
//! That is, the output at moment `t` is a weighted mixture of the input at two adjacent
//! past steps. The weight is linear in `frac(d)`, so the derivative is in `d`
//! exists everywhere and is equal to the difference of these two readings - ordinary
//! linear interpolation, without a straight-through trick: here, unlike in a
//! spike, there is no need to round artificially — the shift itself is a smooth
//! function of the parameter.
//!
//! The kernel is INTENTIONALLY limited to linear (2 samples). With linear interpolation,
//! the support is exactly two adjacent steps, so the gradient along `d` is local and
//! does not blur throughout the entire buffer; this makes fine tuning of the delay
//! (when the start is already close to the goal) stable. A wider kernel (e.g.
//! Gaussian, for training a LARGE delay from scratch via width annealing)
//! is not implemented here - if necessary this is a separate option
//! [`DelayKernel`], which can be added without touching the existing API.
//!
//! # Non-negative delay
//!
//! The delay is parameterized through sigmoid:
//!
//! ```text
//! d = max_delay · σ(raw)
//! ```
//!
//! `σ(raw) ∈ (0, 1)` strictly for ANY finite `raw` is an exact identity
//! of the sigmoid, not the approximate behavior of a composition of two
//! softpluses - therefore `d ∈ (0, max_delay)` strictly, without `clamp` and
//! without the risk of going beyond the border. The extreme points `d = 0`
//! and `d = max_delay` are EXACTLY unattainable (they need `raw = ±∞`), but
//! are approached arbitrarily closely, and both keep a non-zero gradient
//! along `raw` everywhere (unlike `clamp`, where the derivative outside the
//! range is zero) - the same compromise as an additive parameterization, but
//! without violating the invariant.
//!
//! A composition of two softplus functions, each non-negative "in its own
//! direction", does not keep their difference in range:
//! `softplus(x) > x` for any finite `x`, so
//! `max_delay − softplus(max_delay)` is negative (≈ `−0.01815` when
//! `max_delay = 4`). The interpolator in [`interpolate_from_ring`] treats
//! `delay` as a non-negative input of `floor`/`frac`, so a negative delay
//! would be a real correctness bug.
//!
//! The upper bound `max_delay` is still structural, not trainable:
//! it sets the ring buffer size in [`DelayState`] and the convolution range
//! in [`SynapticDelay::forward_sequence`].

use burn::tensor::Device;
use std::collections::VecDeque;

use burn::module::{Module, Param};
use burn::tensor::{Int, Tensor};

/// Delay interpolation kernel.
///
/// Currently only linear is implemented; option reserved for the future
/// extension (see discussion of the Gaussian kernel in [module](self)) to change
/// kernels did not break the [`SynapticDelay`] signature.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum DelayKernel {
    /// Accurate 2-point linear interpolation. Local gradient
    /// suitable for additional training of a delay already close to the target.
    #[default]
    Linear,
}

/// Configuration [`SynapticDelay`].
#[derive(Debug, Clone, Copy)]
pub struct SynapticDelayConfig {
    /// Number of independently delayed channels.
    pub channels: usize,
    /// The upper limit of the delay in steps. Defines the size of the ring
    /// buffers and the range of causal convolution over the sequence.
    pub max_delay: usize,
    /// The initial delay (steps), the same for all channels at the start.
    pub delay_init: f32,
    /// Make delay a trainable parameter.
    ///
    /// At `false` the delay is fixed at `delay_init` and is not included in
    /// `Module::Record` as a trained quantity - the same technique as
    /// `learn_threshold = false` at [`Leaky`](crate::neuron::Leaky).
    pub learn_delay: bool,
    /// Interpolation kernel.
    pub kernel: DelayKernel,
}

impl SynapticDelayConfig {
    /// New configuration with mandatory dimensions.
    pub fn new(channels: usize, max_delay: usize) -> Self {
        Self {
            channels,
            max_delay,
            delay_init: 1.0,
            learn_delay: true,
            kernel: DelayKernel::Linear,
        }
    }

    /// Set the initial delay.
    pub fn with_delay_init(mut self, delay_init: f32) -> Self {
        self.delay_init = delay_init;
        self
    }

    /// Make the delay fixed (not trainable).
    pub fn with_learn_delay(mut self, learn_delay: bool) -> Self {
        self.learn_delay = learn_delay;
        self
    }

    /// Create a layer.
    pub fn init(&self, device: &Device) -> SynapticDelay {
        SynapticDelay::new(self, device)
    }
}

/// Channel-by-channel trainable signal transmission delay.
///
/// Inserted between the projection and the neuron: `Linear → SynapticDelay → Leaky`.
/// Works in two modes - see [`SynapticDelay::step`] (one step at a time,
/// with an explicit state) and [`SynapticDelay::forward_sequence`] (immediately throughout
/// sequences, paired with
/// [`LeakyParallel`](crate::neuron::LeakyParallel)). Both give the same
/// result with the same delay - see
/// `tests/synaptic_delay.rs::step_and_sequence_modes_agree`.
#[derive(Module, Debug)]
pub struct SynapticDelay {
    /// Raw delay: `softplus(raw)` gives `d ≥ 0`. shape `[channels]`.
    ///
    /// `None` when `learn_delay == false`: then used
    /// [`SynapticDelay::fixed_delay`], and the parameter is not created at all -
    /// the same solution as `threshold_param` in
    /// [`Leaky`](crate::neuron::Leaky).
    pub delay_raw: Option<Param<Tensor<1>>>,
    /// Fixed delay when `delay_raw` is missing.
    #[module(skip)]
    fixed_delay: f32,
    /// Number of channels (structural constant).
    #[module(skip)]
    channels: usize,
    /// Upper delay bound/ring buffer size.
    #[module(skip)]
    max_delay: usize,
    /// Interpolation kernel.
    #[module(skip)]
    kernel: DelayKernel,
}

impl SynapticDelay {
    /// Create a layer from the configuration.
    pub fn new(config: &SynapticDelayConfig, device: &Device) -> Self {
        assert!(
            config.max_delay >= 1,
            "SynapticDelay: max_delay must be >= 1"
        );
        assert!(config.channels >= 1, "SynapticDelay: channels must be >= 1");

        // raw = inverse function to delay_tensor(raw) = max_delay * sigmoid(raw),
        // that is, raw = logit(init / max_delay). Composite is inverted
        // the function as a whole (and not some approximation), so
        // delay_tensor() immediately after init() reproduces delay_init exactly
        // (within f32-accuracy logit/sigmoid at the boundaries, see
        // `inverse_delay_tensor`).
        let init = config.delay_init.clamp(0.0, config.max_delay as f32);
        let raw = inverse_delay_tensor(init, config.max_delay as f32);

        let delay_raw = if config.learn_delay {
            Some(Param::from_tensor(Tensor::from_floats(
                vec![raw; config.channels].as_slice(),
                device,
            )))
        } else {
            None
        };

        Self {
            delay_raw,
            fixed_delay: init,
            channels: config.channels,
            max_delay: config.max_delay,
            kernel: config.kernel,
        }
    }

    /// Number of channels.
    pub fn channels(&self) -> usize {
        self.channels
    }

    /// Upper delay bound (ring buffer size).
    pub fn max_delay(&self) -> usize {
        self.max_delay
    }

    /// Interpolation kernel.
    pub fn kernel(&self) -> DelayKernel {
        self.kernel
    }

    /// The current delay of each channel as a tensor `[channels]`,
    /// `d ∈ (0, max_delay)` strictly at teachable delay (see documentation
    /// module), `d ∈ [0, max_delay]` at fixed (there it’s just `clamp`
    /// to the raw value, without parameterization).
    pub fn delay_tensor(&self, device: &Device) -> Tensor<1> {
        use burn::tensor::activation::sigmoid;

        let md = self.max_delay as f32;
        match &self.delay_raw {
            Some(p) => sigmoid(p.val()).mul_scalar(md),
            None => {
                let d = self.fixed_delay.clamp(0.0, md);
                Tensor::from_floats(vec![d; self.channels].as_slice(), device)
            }
        }
    }

    /// One step in time.
    ///
    /// `input` — `[batch, channels]`, current input. `state` stores
    /// last `max_delay` inputs in the ring buffer. Returns
    /// `(delayed output [batch, channels], new state)`.
    pub fn step(&self, input: Tensor<2>, state: &DelayState) -> (Tensor<2>, DelayState) {
        let device = input.device();
        let [_batch, channels] = input.dims();
        assert_eq!(
            channels, self.channels,
            "SynapticDelay::step: input has {channels} channels, layer is configured to {}",
            self.channels
        );

        let d = self.delay_tensor(&device);
        let out = interpolate_from_ring(&state.ring, input.clone(), d, self.max_delay);

        (out, state.clone().push(input))
    }

    /// Direct passage through the entire sequence at once.
    ///
    /// `input` — `[batch, time, channels]`. Causal: the output at step `t`
    /// depends only on the inputs at steps `<= t`; inputs before the start of the
    /// sequence are considered zero (cold start, like
    /// [`DelayState::zeros`]).
    ///
    /// Pair function k
    /// [`LeakyParallel::forward`](crate::neuron::LeakyParallel::forward):
    /// both run the entire layer along the time axis and give the same result,
    /// that is a step by step call [`SynapticDelay::step`] in a loop.
    pub fn forward_sequence(&self, input: Tensor<3>) -> Tensor<3> {
        let device = input.device();
        let [batch, steps, channels] = input.dims();
        assert!(
            steps > 0,
            "SynapticDelay::forward_sequence: num_steps must be > 0"
        );
        assert_eq!(
            channels, self.channels,
            "SynapticDelay::forward_sequence: input has {channels} channels, layer is configured to {}",
            self.channels
        );

        let d = self.delay_tensor(&device);

        let mut state = DelayState::zeros(&device, batch, self.channels, self.max_delay);
        let mut outputs = Vec::with_capacity(steps);
        for x_t in crate::neuron::split_time(input) {
            let out = interpolate_from_ring(&state.ring, x_t.clone(), d.clone(), self.max_delay);
            state = state.push(x_t);
            outputs.push(out);
        }

        Tensor::stack::<3>(outputs, 1)
    }
}

/// Runtime state of [`SynapticDelay`]: ring buffer of the last `max_delay`
/// inputs needed to interpolate the current output.
///
/// The buffer is stored in `VecDeque`, the oldest at the beginning - that is, `ring[0]`
/// corresponds to shift `max_delay`, `ring[len-1]` - shift 1. Such
/// the order coincides with the direction of the flow of time and does not require
/// reindexing elements that remain in the buffer each time
/// [`DelayState::push`].
///
/// **`O(1)` buffer update**: `push` maintains the ring in `VecDeque`, so
/// neither insertion nor eviction shifts the remaining elements -
/// `Vec::remove(0)` would shift ALL remaining elements by one position,
/// i.e. `O(max_delay)` work for each time step, adding up to `O(T·max_delay)`
/// over a sequence. `VecDeque::push_back`/`pop_front` are both amortized
/// `O(1)` (a ring buffer over a slice, no shifting), and indexing `ring[i]`
/// in `interpolate_from_ring` works for `VecDeque` the same way as for `Vec`
/// (via `Index`), so the caller code is unchanged.
#[derive(Debug, Clone)]
pub struct DelayState {
    /// Latest (before `max_delay`) entries, oldest first. Everyone -
    /// `[batch, channels]`.
    ring: VecDeque<Tensor<2>>,
    #[allow(dead_code)]
    max_delay: usize,
}

impl DelayState {
    /// Empty state: buffer filled with zeros - cold start
    /// sequence is equivalent to the fact that there were only zeros before it.
    pub fn zeros(device: &Device, batch: usize, channels: usize, max_delay: usize) -> Self {
        let ring = (0..max_delay)
            .map(|_| Tensor::zeros([batch, channels], device))
            .collect();
        Self { ring, max_delay }
    }

    /// Shift the buffer and add a new input. Returns new state
    /// (the buffer is immutable, like [`NeuronState`](crate::state::NeuronState)).
    ///
    /// `O(1)` depreciated: `pop_front`/`push_back` to `VecDeque` not
    /// require shifting of remaining elements (see structure documentation).
    fn push(mut self, x: Tensor<2>) -> Self {
        if !self.ring.is_empty() {
            self.ring.pop_front();
        }
        self.ring.push_back(x);
        self
    }

    /// Detach the entire state from the autodifferentiation graph
    /// (truncated BPTT) - same reason as
    /// [`NeuronState::detach`](crate::state::NeuronState::detach).
    pub fn detach(self) -> Self {
        Self {
            ring: self.ring.into_iter().map(Tensor::detach).collect(),
            max_delay: self.max_delay,
        }
    }
}

/// Linear interpolation of delayed output from a ring buffer.
///
/// `ring` — `max_delay` of past entries (oldest first, see
/// [`DelayState`]), `current` — input of the current step (shift 0), `delay` —
/// `[1, channels]`, trainable delay of each channel.
///
/// For each channel, two adjacent integer shifts `k = floor(d)` and
/// `k + 1`, and the result is a linear mixture of them with weight `frac(d)`. Shift `s`
/// addresses `current` with `s = 0` and `ring[max_delay - s]` with
/// `1 <= s <= max_delay`; shifts outside the buffer (`d` at the top
/// boundaries, `k + 1 > max_delay`) read the oldest available sample -
/// natural behavior when clamping `d` in [`SynapticDelay::delay_tensor`].
///
/// **Gather-based interpolation**: constructing a Boolean mask for EACH
/// integer shift `s = 0..=max_delay` (`floor_d == s`) across all channels and
/// accumulating a weighted sum would take `max_delay + 1` full tensor
/// operations per [`interpolate_from_ring`] call (and therefore per time step
/// in [`SynapticDelay::forward_sequence`]/[`SynapticDelay::step`]), almost all
/// of which are “missed” for any given channel (the mask is zero almost
/// everywhere). The underlying task is a gather: “for each channel, select ONE
/// sample from `max_delay + 1` available, by an index known only at runtime”.
/// Burn (like PyTorch's `Tensor::gather`/`torch.gather`) provides this
/// primitive; it differentiates by the gathered values, and the mixing weight
/// `frac` is a smooth function of `delay_raw`, so no index differentiation is
/// needed. The implementation below stacks the `max_delay + 1` samples into one
/// `[batch, max_delay+1, channels]` tensor ONCE (this part is inherently
/// `O(max_delay)` because the buffer has that size, but it is a cheap `cat`,
/// not `O(max_delay)` full tensor operations with broadcasting masks), then
/// reads the required two samples per channel with TWO `gather` calls instead
/// of `max_delay + 1` masks, independent of `max_delay`. The caller-visible
/// cost is therefore `O(1)` (two `gather`s) on top of one `O(max_delay)`
/// stack build, which is cheaper than `max_delay` full-tensor mask operations
/// because it is bare concatenation of already existing tensors without
/// arithmetic.
///
/// Real per-channel `O(1)` without ANY construction `[.., max_delay+1,
/// ..]`-stack would require a custom kernel with direct dynamic
/// indexing the ring buffer on the GPU (see discussion in the modular
/// documentation) - `ring` is physically stored here as `VecDeque` separate
/// tensors `[batch, channels]`, and not as one continuous tensor with an axis
/// time, so before `gather` it in any case needs to be materialized in
/// a single tensor through `cat`.
fn interpolate_from_ring(
    ring: &VecDeque<Tensor<2>>,
    current: Tensor<2>,
    delay: Tensor<1>,
    max_delay: usize,
) -> Tensor<2> {
    let dims = current.dims();
    let [batch, channels] = dims;
    let delay = delay.reshape([1, channels]).expand(dims);

    // `floor` for a NON-NEGATIVE tensor: `trunc` in burn rounds to zero,
    // which for x >= 0 coincides with floor. Delay guaranteed in
    // (0, max_delay) by construction delay_tensor (d = max_delay * sigmoid(raw)
    // with a teachable delay, clamp(0, max_delay) with a fixed delay - both
    // are non-negative by construction), so trunc is safe here and not
    // requires a separate floor primitive.
    let floor_d = delay.clone().trunc();
    let frac = delay - floor_d.clone();

    // Stack of all max_delay+1 available samples along NEW axis 1:
    // shifts[:, s, :] = at(s) for s = 0..=max_delay, with the same addressing
    // as the closed function `at` (`s == 0` → `current`,
    // `1 <= s <= max_delay` → `ring[max_delay - s]`), but materialized once
    // instead of called on demand under each mask.
    let mut shift_slices: Vec<Tensor<3>> = Vec::with_capacity(max_delay + 1);
    shift_slices.push(current.clone().reshape([batch, 1, channels]));
    for s in 1..=max_delay {
        shift_slices.push(ring[max_delay - s].clone().reshape([batch, 1, channels]));
    }
    let shifts: Tensor<3> = Tensor::cat(shift_slices, 1); // [batch, max_delay+1, channels]

    // Indices for gather along axis 1: index of channel c in line batch b —
    // the same for all b (delay does not depend on batch), but gather requires
    // index tensor of the same rank and the same batch/channel shape as
    // and data (except for the gather axis itself), so the index expands to
    // [batch, 1, channels] explicitly, without broadcasting inside gather.
    //
    // Clamping the index at [0, max_delay] reproduces the old behavior
    // "shifts outside the buffer read the oldest available sample":
    // `lo = floor_d` is already in this range by construction (delay ∈ [0,
    // max_delay) with a teachable delay, clamp with a fixed delay, - trunc
    // preserves the range), and `hi = floor_d + 1` can reach `max_delay + 1`
    // only when `floor_d == max_delay` - that is, `delay` AS MUCH AS YOU WANT
    // close to its upper limit (see module documentation about
    // strictly unattainable boundaries), and clamping `hi` back to `max_delay`
    // means `lo == hi == max_delay`, which gives `frac`-reference mixing
    // `max_delay` WITH YOURSELF - that is, just this countdown, regardless of
    // values `frac`, which is what was required (READ "oldest available",
    // rather than going beyond the stack).
    let to_idx = |t: Tensor<2>| -> Tensor<3, Int> {
        t.clamp(0.0, max_delay as f32)
            .reshape([batch, 1, channels])
            .int()
    };
    let lo_idx = to_idx(floor_d.clone());
    let hi_idx = to_idx(floor_d.add_scalar(1.0));

    let lo = shifts.clone().gather(1, lo_idx).reshape([batch, channels]);
    let hi = shifts.gather(1, hi_idx).reshape([batch, channels]);

    lo.mul(frac.clone().mul_scalar(-1.0).add_scalar(1.0)) + hi.mul(frac)
}

/// `logit(p) = ln(p / (1 − p))` with protection from NaN/±inf at the boundaries of `p ∈ {0, 1}` —
/// inverse function to `sigmoid`. Clamping `p` in the open interval before
/// by logarithm gives FINITE (albeit large in absolute value) `raw` even for
/// `target` exactly equal to `0` or `max_delay`: `delay_tensor` after that
/// will return a value arbitrarily close to the border, but not the border itself -
/// see the module documentation about `d ∈ (0, max_delay)` strictly.
fn logit(p: f32) -> f32 {
    let p = p.clamp(1e-6, 1.0 - 1e-6);
    (p / (1.0 - p)).ln()
}

/// Inverse function to `delay_tensor`: `target = max_delay · sigmoid(raw)`,
/// solved with respect to `raw`.
///
/// `raw = logit(target / max_delay)`. Unlike the old double
/// softplus inversions, there is one direct function without composition - because
/// that the parameterization itself is now single-layer (`max_delay * sigmoid`,
/// see module documentation) and not `md - softplus(md - softplus(raw))`.
fn inverse_delay_tensor(target: f32, max_delay: f32) -> f32 {
    let target = target.clamp(0.0, max_delay);
    logit(target / max_delay)
}

#[cfg(test)]
mod gather_interpolation_tests {
    use super::*;
    use burn::tensor::Device;

    fn dev() -> Device {
        Device::flex().autodiff()
    }

    fn t2(data: Vec<f32>, shape: [usize; 2]) -> Tensor<2> {
        Tensor::<1>::from_floats(data.as_slice(), &dev()).reshape(shape)
    }

    /// The gradient over the trained delay must flow through
    /// [`interpolate_from_ring`], which mixes the two gathered samples - `gather`
    /// in Burn differentiates by COLLECTED values (here: by buffer samples),
    /// and the mixture between `lo`/`hi` goes through `frac`, which remains
    /// the usual smooth function `delay_raw`, so delay training is
    /// unaffected by the gather-based implementation.
    #[test]
    fn delay_gradient_flows_through_gather_interpolation() {
        let device = dev();
        let config = SynapticDelayConfig::new(2, 4).with_delay_init(1.5);
        let layer = SynapticDelay::new(&config, &device);

        let state = DelayState::zeros(&device, 1, 2, layer.max_delay());
        let input = t2(vec![0.3, -0.7], [1, 2]);
        let (out, _) = layer.step(input, &state);

        let loss = out.sum();
        let grads = loss.backward();

        let delay_grad = layer
            .delay_raw
            .as_ref()
            .expect("learn_delay=true by default in SynapticDelayConfig::new")
            .grad(&grads);
        assert!(
            delay_grad.is_some(),
            "delay_raw gradient does not flow through gather interpolation"
        );
    }

    /// The same gradient check with a delay EXACTLY on the upper boundary
    /// (`delay_init == max_delay`) - the boundary case where `hi_idx` is
    /// clamped back to `max_delay` (see documentation
    /// [`interpolate_from_ring`], paragraph about `lo == hi == max_delay`).
    /// Index clamping is not an index-differentiable operation, but
    /// the gradient along `frac` (and through it - along `delay_raw`) must flow
    /// all the same, since `frac` does not participate in the calculation of indices for
    /// `gather`, only in the final linear mixing of the already selected
    /// values.
    #[test]
    fn delay_gradient_flows_at_upper_boundary() {
        let device = dev();
        let max_delay = 4usize;
        let config = SynapticDelayConfig::new(1, max_delay).with_delay_init(max_delay as f32);
        let layer = SynapticDelay::new(&config, &device);

        let state = DelayState::zeros(&device, 1, 1, layer.max_delay());
        let input = t2(vec![0.42], [1, 1]);
        let (out, _) = layer.step(input, &state);

        let loss = out.sum();
        let grads = loss.backward();

        let delay_grad = layer
            .delay_raw
            .as_ref()
            .expect("learn_delay=true by default")
            .grad(&grads);
        assert!(
            delay_grad.is_some(),
            "the delay_raw gradient does not flow at the upper bound of max_delay"
        );
    }
}
