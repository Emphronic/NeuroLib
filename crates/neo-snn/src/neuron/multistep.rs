//! Multi-step neurons: running through the ENTIRE sequence `[batch, time,
//! features]` in one call, in the spirit of `multi_step_mode` from SpikingJelly.
//!
//! # What does it speed up and what doesn’t
//!
//! Spike recurrence with membrane reset (`ResetMode::Subtract` /
//! `Zero`) does NOT have a closed analytical shape - `U[t]` depends on
//! `spike_hard[t-1] = H(U[t-1] - threshold)`, nonlinear function `U[t-1]`
//! (see documentation [`crate::neuron::parallel`], where for `ResetMode::None`
//! this is decomposed into one matmul precisely BECAUSE there is no reset there).
//! There is a reset here, so the recurrence along the membrane itself remains
//! The `T`-step cycle is a mathematical necessity, not an oversight.
//!
//! Acceleration is of a different kind, but quite real:
//!
//! * Explicit-state API requires `T` SEPARATE calls to `step()` from
//!   calling code - on the GPU backend this is `T` launch for each
//!   operation inside `step` (mul, add, sigmoid, ...), with overhead on
//!   dispatching and synchronization before each launch. Here's a loop through
//!   time goes INSIDE one call to a Rust function - it itself does not go anywhere
//!   disappears (recurrence obliges), but the calling code and the burn graph
//!   are built once for the entire sequence, and not `T` times.
//! * Everything that does NOT depend on the previous step is batched in one pass
//!   `[batch·time, features]` BEFORE the loop, and not recalculated `T` times
//!   inside it:
//!   - `RLeaky`/`RSynaptic`: `linear_in.forward(input)` - the heaviest
//!     operation per step - is computed as ONE matmul for the whole
//!     sequence at once, not `T` small ones. The matrix
//!     `linear_rec.forward(spike_prev)` inevitably remains in the loop:
//!     `spike_prev[t]` physically appears only after the `t` step.
//!   - `Leaky`: `threshold_tensor`/`graded_factor_tensor` do not depend on
//!     time at all (see `Leaky::step`, where they are recomputed at every
//!     step for nothing) - computed once before the loop.
//!
//! Result: real, but not “magic” gain - the number of launches and
//! intermediate allocations for “heavy” linear operations drops from `T` to
//! `1`, and the element-wise LIF-dynamics itself (`mul`/`add`/comparison with threshold)
//! remains `O(T)`, as in SpikingJelly (`multi_step_mode` accelerates there
//! exactly the same thing - linear layers in front of the LIF core, and not the core itself
//! spike recurrence).
//!
//! # Numerical equivalence
//!
//! `MultiStepLeaky::forward` and `MultiStepRLeaky::forward` are obliged to give
//! EXACTLY the same result as explicit-state `step()` in a manual loop -
//! this is not a different, approximate dynamics, but the same one, just without
//! extra work. Tests at the end of the module test this directly against
//! `Leaky::step` / `RLeaky::step`.

use burn::module::Module;
use burn::tensor::Device;
use burn::tensor::Tensor;

use super::{Leaky, LeakyConfig, RLeaky, apply_graded_spikes, spike_with_scalar_threshold};
use crate::state::{
    NeuronState, ResetMode, ResetOrder, apply_reset_tensor, carry_membrane, decay_membrane,
    decay_then_reset_tensor, hard_spike_tensor,
};
use crate::surrogate::Surrogate;

// ---------------------------------------------------------------------
// MultiStepLeaky
// ---------------------------------------------------------------------

/// `Leaky`, run through the entire sequence `[batch, time,
/// features]` in one call.
///
/// Unlike [`LeakyParallel`](crate::neuron::LeakyParallel), NOT
/// requires `ResetMode::None` - works with regular spike LIF
/// (`Subtract`/`Zero`), because it doesn’t try to collapse itself
/// recurrence in matmul, but only removes the overhead of the `step()` call from
/// external code and computes once what does not depend on time.
/// If reset is disabled (`ResetMode::None`) and you need a matrix
/// way - use `LeakyParallel`, it is faster for this case
/// (real `O(1)` launch instead of `O(T)`).
#[derive(Module, Debug)]
pub struct MultiStepLeaky {
    /// The inner LIF layer is the same structure that uses explicit-state
    /// [`Leaky::step`], parameters are fully compatible and portable between
    /// modes (trained in multi-step - use in explicit-state and
    /// conversely, `step`/`forward` read the same fields).
    pub lif: Leaky,
}

impl MultiStepLeaky {
    /// Create a layer.
    pub fn new(config: &LeakyConfig, device: &Device) -> Self {
        Self {
            lif: Leaky::new(config, device),
        }
    }

    /// Wrap an existing [`Leaky`] into multi-step mode without
    /// re-creating parameters - for example, to switch the trained
    /// explicit-state layer on batched forward for inference.
    pub fn from_layer(lif: Leaky) -> Self {
        Self { lif }
    }

    /// Time run. Returns only spikes, `[batch, time,
    /// features]`.
    pub fn forward<S: Surrogate + ?Sized>(&self, input: Tensor<3>, surrogate: &S) -> Tensor<3> {
        self.forward_with_state(input, surrogate).0
    }

    /// Running through time with returning the final state is necessary if
    /// the sequence is processed in chunks (truncated BPTT) or
    /// interfaces with subsequent explicit-state code.
    ///
    /// The initial state is zero; to continue from what is already known
    /// states (real chunked processing) use
    /// [`forward_from_state`](Self::forward_from_state).
    pub fn forward_with_state<S: Surrogate + ?Sized>(
        &self,
        input: Tensor<3>,
        surrogate: &S,
    ) -> (Tensor<3>, NeuronState) {
        let device = input.device();
        let [batch, _num_steps, features] = input.dims();
        let initial_state = NeuronState::zeros_mem_only(&device, batch, features);
        self.forward_from_state(input, &initial_state, surrogate)
    }

    /// Time run that continues an already existing state −
    /// real input for truncated BPTT/chunked processing: following
    /// a piece of the sequence starts EXACTLY from the state at
    /// which ended the previous one, and not always from scratch.
    ///
    /// # inhibition=true
    ///
    /// At `self.lif.inhibition == true` the fast matrix path through the
    /// membrane remains the same (WTA is applied element by element and does not
    /// depend on whether it is an accumulated matrix or a loop), but
    /// winner-take-all is by construction an operation PER STEP (aggregates the
    /// maximum over the channels of ONE point in time), so it must
    /// be executed inside the loop over `t`, not hoisted out of it.
    /// WTA is computed explicitly on each step through [`winner_take_all`], so
    /// the behavior matches `Leaky::step` regardless of build profile
    /// (`debug_assert!` alone would disappear in `--release`).
    pub fn forward_from_state<S: Surrogate + ?Sized>(
        &self,
        input: Tensor<3>,
        initial_state: &NeuronState,
        surrogate: &S,
    ) -> (Tensor<3>, NeuronState) {
        run_leaky_sequence(&self.lif, input, initial_state, surrogate)
    }
}

/// Cuts `[batch, time, features]` into `time` slices `[batch, features]`.
///
/// The reverse pass from `slice` to Burn builds a ZERO tensor of the full size of the input and adds it to
/// a piece of it, and then the gradients of all slices are also added in full-size additions.
/// With `T` slices, this is `O(T²·batch·features)` of memory and time only on the backward input - on
/// in long sequences this is more than all the rest of the layer's work. Therefore, slicing
/// two-level: first `⌈√T⌉` blocks in time, then steps within the block. Full size
/// passes in backward become `≈ √T` instead of `T` (the rest are block sized).
pub fn split_time(input: Tensor<3>) -> Vec<Tensor<2>> {
    let [batch, steps, features] = input.dims();
    if steps <= 4 {
        return (0..steps)
            .map(|t| {
                input
                    .clone()
                    .slice([0..batch, t..t + 1, 0..features])
                    .squeeze_dims::<2>(&[1])
            })
            .collect();
    }
    let block = (steps as f64).sqrt().ceil() as usize;
    let mut out = Vec::with_capacity(steps);
    let mut lo = 0;
    while lo < steps {
        let hi = (lo + block).min(steps);
        let chunk = input.clone().slice([0..batch, lo..hi, 0..features]);
        for t in 0..hi - lo {
            out.push(
                chunk
                    .clone()
                    .slice([0..batch, t..t + 1, 0..features])
                    .squeeze_dims::<2>(&[1]),
            );
        }
        lo = hi;
    }
    out
}

/// Common kernel of `Leaky` time run for [`MultiStepLeaky`] and sequential branch
/// [`LeakyParallel`](crate::neuron::LeakyParallel) (`Subtract`/`Zero`).
///
/// # What's in the loop and what's outside it
///
/// Inside the loop along `t`, only what is truly recurrent remains: decay
/// (`β·mem`), current input, hard spike (`mem > θ`, uncoupled from the graph) and reset.
/// Everything else does not depend on recurrence and is computed ONCE over the entire
/// sequence `[batch·time, units]`:
///
/// * surrogate and straight-through link `hard + (soft − soft.detach())`;
/// * winner-take-all for the output spike (the mask is built row by row over the membrane);
/// * graded factor.
///
/// This is exactly equivalent to step `Leaky::step` because reset uses the UNCLIPPED
/// `spike_hard`: the surrogate gradient flows only from the output to the `mem_new[t]` membrane and never
/// does not return to recurrence. Therefore, the surrogate can be read after a cycle through the stack
/// `mem_new`, and the number of tensor operations per step drops approximately by half, and the number of nodes
/// surrogate in the autodiff graph - from `T` to one.
pub(crate) fn run_leaky_sequence<S: Surrogate + ?Sized>(
    lif: &Leaky,
    input: Tensor<3>,
    initial_state: &NeuronState,
    surrogate: &S,
) -> (Tensor<3>, NeuronState) {
    let device = input.device();
    let [batch, num_steps, features] = input.dims();
    assert!(num_steps > 0, "MultiStepLeaky: num_steps must be > 0");

    // The threshold and graded factor do not depend on time - once, and not T times in a cycle.
    let dims = [batch, features];
    let threshold = lif.threshold_tensor(&device, dims);
    let graded_factor = lif.graded_factor_tensor(&device);

    let beta = burn::tensor::activation::sigmoid(lif.beta.val())
        .reshape([1, 1])
        .expand(dims);

    let after_decay = lif.reset_order == ResetOrder::AfterDecay;
    let resets = lif.reset_mode != ResetMode::None;

    let mut state = initial_state.clone();
    let mut mems = Vec::with_capacity(num_steps);
    // AfterDecay: hard spike of the previous step (after WTA), also known as “pending” reset.
    // The first step is to restore it from its initial state.
    let mut pending: Option<Tensor<2>> = None;

    let steps_in = split_time(input);

    for x_t in steps_in {
        // ResetOrder::AfterDecay: membrane in state - before reset, reset previous
        // the spike is applied here, after the decay (as in `Leaky::step`).
        let mem_decayed = if after_decay && resets {
            let pend = match pending.take() {
                Some(p) => p,
                None => {
                    let p = hard_spike_tensor(&state.mem, &threshold);
                    if lif.inhibition {
                        crate::functional::winner_take_all(p, state.mem.clone())
                    } else {
                        p
                    }
                }
            };
            decay_then_reset_tensor(
                state.mem.clone(),
                beta.clone(),
                &pend,
                &threshold,
                lif.reset_mode,
            )
        } else {
            state.mem.clone().mul(beta.clone())
        };
        let mem_new = mem_decayed + x_t;

        // A hard spike is needed in the loop only for reset; the output is computed after the loop.
        let need_hard = resets;
        let hard = if need_hard {
            // `mem > θ` directly: strictly, without intermediate `mem − θ`.
            let h = mem_new
                .clone()
                .detach()
                .greater(threshold.clone().detach())
                .float();
            // Suppression - BEFORE reset, with the same function and at the same place in the conveyor,
            // as in Leaky::step: only the membrane of the actually spiked neuron is reset.
            Some(if lif.inhibition {
                crate::functional::winner_take_all(h, mem_new.clone())
            } else {
                h
            })
        } else {
            None
        };

        let mem_carry = if after_decay {
            pending = hard;
            mem_new.clone()
        } else {
            match hard {
                Some(h) => apply_reset_tensor(mem_new.clone(), &h, &threshold, lif.reset_mode),
                None => mem_new.clone(),
            }
        };

        mems.push(mem_new);
        state = NeuronState {
            mem: mem_carry,
            syn: state.syn,
        };
    }

    // ---- Anything that is not recurrent: once over [batch·time, units]. ----
    let rows = batch * num_steps;
    let mem_all = Tensor::stack::<3>(mems, 1).reshape([rows, features]);
    let thr_all = threshold
        .reshape([batch, 1, features])
        .expand([batch, num_steps, features])
        .reshape([rows, features]);

    // The same pipeline as `spike_with_threshold`, but over the entire sequence.
    let (spike, _hard) = super::spike_from_x(mem_all.clone() - thr_all, surrogate);
    let spike = if lif.inhibition {
        crate::functional::winner_take_all(spike, mem_all)
    } else {
        spike
    };
    let spike = apply_graded_spikes(spike, graded_factor).reshape([batch, num_steps, features]);

    (spike, state)
}

// ---------------------------------------------------------------------
// MultiStepRLeaky
// ---------------------------------------------------------------------

/// `RLeaky`, run through the entire sequence `[batch, time,
/// features]` in one call.
///
/// `linear_in.forward(input)` - the most difficult operation per step -
/// is computed as ONE matmul for the entire `[batch, time, in_features]` before
/// cycle. `linear_rec.forward(spike_prev)` inevitably remains inside
/// cycle: `spike_prev[t]` appears only after step `t`
/// calculated and cannot be known in advance.
#[derive(Module, Debug)]
pub struct MultiStepRLeaky {
    /// The internal recurrent LIF layer is the same structure as
    /// uses explicit-state [`RLeaky::step`].
    pub rlif: RLeaky,
}

impl MultiStepRLeaky {
    /// Wrap an existing [`RLeaky`] into multi-step mode.
    pub fn from_layer(rlif: RLeaky) -> Self {
        Self { rlif }
    }

    /// Time run. Returns only spikes, `[batch, time,
    /// out_features]`.
    pub fn forward<S: Surrogate + ?Sized>(&self, input: Tensor<3>, surrogate: &S) -> Tensor<3> {
        self.forward_with_state(input, surrogate).0
    }

    /// Time run with return of final state (membrane +
    /// `spike_prev`) - needed for truncated BPTT and for state transfer
    /// in explicit-state/stateful code between chunks.
    ///
    /// Starts from the zero state and zero `spike_prev`; to
    /// continue from an already known state (next chunk
    /// sequences), use
    /// [`forward_from_state`](Self::forward_from_state).
    pub fn forward_with_state<S: Surrogate + ?Sized>(
        &self,
        input: Tensor<3>,
        surrogate: &S,
    ) -> (Tensor<3>, NeuronState, Tensor<2>) {
        let device = input.device();
        let [batch, _num_steps, _in_features] = input.dims();
        let out_features = self.rlif.linear_in.weight.val().dims()[1];
        let initial_state = NeuronState::zeros_mem_only(&device, batch, out_features);
        let initial_spike_prev = Tensor::<2>::zeros([batch, out_features], &device);
        self.forward_from_state(input, &initial_state, initial_spike_prev, surrogate)
    }

    /// Time run that continues an existing state AND
    /// `spike_prev` - real input for truncated BPTT / chunked
    /// recurrent layer processing: next piece of sequence
    /// starts exactly from the recurrent connection where it ended
    /// the previous one, and does not terminate it with zero.
    pub fn forward_from_state<S: Surrogate + ?Sized>(
        &self,
        input: Tensor<3>,
        initial_state: &NeuronState,
        initial_spike_prev: Tensor<2>,
        surrogate: &S,
    ) -> (Tensor<3>, NeuronState, Tensor<2>) {
        let [batch, num_steps, _in_features] = input.dims();
        assert!(num_steps > 0, "MultiStepRLeaky: num_steps must be > 0");
        let out_features = self.rlif.linear_in.weight.val().dims()[1];

        // linear_in is the only part that does not depend on recurrence
        // by spike_prev. We compute the entire sequence with one matmul:
        // burn::nn::Linear::forward takes an arbitrary leading shape
        // (batch-dims), so [batch, time, in] -> [batch, time, out]
        // in one call, without reshape tricks.
        let current_in = self.rlif.linear_in.forward(input); // [batch, time, out]

        let beta = burn::tensor::activation::sigmoid(self.rlif.beta.val())
            .reshape([1, 1])
            .expand([batch, out_features]);

        let mut state = initial_state.clone();
        let mut spike_prev = initial_spike_prev;
        let mut outputs = Vec::with_capacity(num_steps);

        for current_in_t in split_time(current_in) {
            // Recurrent projection is the only part that
            // cannot be taken out of the loop: depends on spike_prev,
            // born at the previous iteration of the same cycle.
            let current = current_in_t + self.rlif.linear_rec.forward(spike_prev.clone());

            let mem_new = decay_membrane(
                state.mem.clone(),
                beta.clone(),
                self.rlif.threshold,
                self.rlif.reset_mode,
                self.rlif.reset_order,
            ) + current;

            let (spike, spike_hard) =
                spike_with_scalar_threshold(mem_new.clone(), self.rlif.threshold, surrogate);
            let mem_reset = carry_membrane(
                mem_new,
                &spike_hard,
                self.rlif.threshold,
                self.rlif.reset_mode,
                self.rlif.reset_order,
            );

            state = NeuronState {
                mem: mem_reset,
                syn: state.syn,
            };
            spike_prev = spike.clone();
            outputs.push(spike);
        }

        (Tensor::stack::<3>(outputs, 1), state, spike_prev)
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

    // Constructing a rank-1 tensor via from_floats and then reshaping it
    // fails for a shape with zero dimensions (empty sequence,
    // [1, 0, 1]) DEEP inside cubecl-zspace(shape.rs:420), before control
    // reaches forward_from_state and its assert!(num_steps > 0, ...).
    // burn::tensor::TensorData::new builds a tensor immediately at the
    // target rank/shape, bypassing the problematic rank-1 -> rank-3 reshape
    // on empty data, and works for both empty and regular shapes.
    fn t3(data: Vec<f32>, shape: [usize; 3]) -> Tensor<3> {
        Tensor::<3>::from_data(burn::tensor::TensorData::new(data, shape), &dev())
    }

    fn vec2(t: Tensor<2>) -> Vec<f32> {
        t.into_data().try_to_vec::<f32>().unwrap()
    }

    fn vec3(t: Tensor<3>) -> Vec<f32> {
        t.into_data().try_to_vec::<f32>().unwrap()
    }

    /// `MultiStepLeaky::forward` must give EXACTLY the same result as
    /// explicit-state `Leaky::step` in the manual cycle, including the mode with
    /// membrane reset (`ResetMode::Subtract`) is the case that
    /// `LeakyParallel` does not support it in principle.
    #[test]
    fn multistep_leaky_matches_explicit_loop_with_reset() {
        use crate::state::ResetMode;
        use crate::surrogate::FastSigmoid;

        let device = dev();
        let surrogate = FastSigmoid::default();
        let config = LeakyConfig::new(0.85)
            .with_threshold(0.5)
            .with_reset_mode(ResetMode::Subtract);

        let steps: Vec<f32> = vec![
            0.3, 0.1, // t=0
            0.6, 0.0, // t=1
            0.1, 0.9, // t=2
            0.0, 0.0, // t=3
        ];
        let input = t3(steps.clone(), [1, 4, 2]);

        // Explicit-state: manual cycle.
        let explicit_layer = config.init(&device);
        let mut explicit_state = NeuronState::zeros_mem_only(&device, 1, 2);
        let mut explicit_out = Vec::new();
        for t in 0..4 {
            let x = t2(steps[t * 2..t * 2 + 2].to_vec(), [1, 2]);
            let (spike, new_state) = explicit_layer.step(x, &explicit_state, &surrogate);
            explicit_state = new_state;
            explicit_out.extend(vec2(spike));
        }

        // Multi-step: one call for the entire sequence.
        let multistep_layer = MultiStepLeaky::from_layer(config.init(&device));
        let multistep_out = vec3(multistep_layer.forward(input, &surrogate));

        for (i, (&a, &b)) in explicit_out.iter().zip(multistep_out.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-5,
                "element {i}: explicit={a} multistep={b}"
            );
        }
    }

    /// The final state returned by `forward_with_state` must
    /// coincide with the state after a manual explicit-state cycle - important
    /// for truncated BPTT, where the next chunk continues from it.
    #[test]
    fn multistep_leaky_final_state_matches_explicit_loop() {
        use crate::state::ResetMode;
        use crate::surrogate::FastSigmoid;

        let device = dev();
        let surrogate = FastSigmoid::default();
        let config = LeakyConfig::new(0.7)
            .with_threshold(0.5)
            .with_reset_mode(ResetMode::Zero);

        let steps: Vec<f32> = vec![0.9, 0.2, 0.1, 0.8, 0.3, 0.3];
        let input = t3(steps.clone(), [1, 3, 2]);

        let explicit_layer = config.init(&device);
        let mut explicit_state = NeuronState::zeros_mem_only(&device, 1, 2);
        for t in 0..3 {
            let x = t2(steps[t * 2..t * 2 + 2].to_vec(), [1, 2]);
            let (_, new_state) = explicit_layer.step(x, &explicit_state, &surrogate);
            explicit_state = new_state;
        }

        let multistep_layer = MultiStepLeaky::from_layer(config.init(&device));
        let (_, final_state) = multistep_layer.forward_with_state(input, &surrogate);

        let want = vec2(explicit_state.mem);
        let got = vec2(final_state.mem);
        for (i, (&a, &b)) in want.iter().zip(got.iter()).enumerate() {
            assert!((a - b).abs() < 1e-5, "mem[{i}]: explicit={a} multistep={b}");
        }
    }

    /// `MultiStepRLeaky::forward` must give EXACTLY the same result
    /// that explicit-state `RLeaky::step` in a manual loop with yourself
    /// dragged `spike_prev` - including the case where linear_in
    /// batched, but linear_rec remains step-by-step.
    #[test]
    fn multistep_rleaky_matches_explicit_loop() {
        use crate::state::ResetMode;
        use crate::surrogate::FastSigmoid;
        use burn::module::Param;
        use burn::nn::LinearConfig;

        let device = dev();
        let surrogate = FastSigmoid::default();

        // We build both layers from the same tensors, and not through
        // RLeaky::clone() - see the history of edits in stateful.rs: in this
        // burn Module::clone() does not guarantee a bitwise match
        // bias between the original and the clone.
        let seed = RLeaky::new(&device, 2, 2, 0.85, 0.5, ResetMode::Subtract);
        let w_in = seed.linear_in.weight.val();
        let b_in = seed.linear_in.bias.as_ref().unwrap().val();
        let w_rec = seed.linear_rec.weight.val();
        let beta_raw = seed.beta.val();

        let build_layer = || -> RLeaky {
            let mut linear_in = LinearConfig::new(2, 2).init(&device);
            linear_in.weight = Param::from_tensor(w_in.clone());
            linear_in.bias = Some(Param::from_tensor(b_in.clone()));
            let mut linear_rec = LinearConfig::new(2, 2).with_bias(false).init(&device);
            linear_rec.weight = Param::from_tensor(w_rec.clone());
            RLeaky {
                linear_in,
                linear_rec,
                beta: Param::from_tensor(beta_raw.clone()),
                threshold: 0.5,
                reset_mode: ResetMode::Subtract,
                reset_order: crate::state::ResetOrder::BeforeDecay,
            }
        };

        let explicit_layer = build_layer();
        let multistep_layer = MultiStepRLeaky::from_layer(build_layer());

        let steps: Vec<f32> = vec![
            0.3, 0.1, // t=0
            0.6, 0.0, // t=1
            0.1, 0.9, // t=2
        ];
        let input = t3(steps.clone(), [1, 3, 2]);

        // Explicit-state: manual loop, spike_prev is dragged manually.
        let mut explicit_state = NeuronState::zeros_mem_only(&device, 1, 2);
        let mut spike_prev = Tensor::<2>::zeros([1, 2], &device);
        let mut explicit_out = Vec::new();
        for t in 0..3 {
            let x = t2(steps[t * 2..t * 2 + 2].to_vec(), [1, 2]);
            let (spike, new_state) =
                explicit_layer.step(x, spike_prev.clone(), &explicit_state, &surrogate);
            explicit_state = new_state;
            spike_prev = spike.clone();
            explicit_out.extend(vec2(spike));
        }

        let multistep_out = vec3(multistep_layer.forward(input, &surrogate));

        for (i, (&a, &b)) in explicit_out.iter().zip(multistep_out.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-5,
                "element {i}: explicit={a} multistep={b}"
            );
        }
    }

    /// `MultiStepLeaky` with `inhibition=true` must give EXACTLY the same
    /// result as `Leaky::step` in the explicit-state cycle (which always
    /// applies WTA) - including the case of TWO simultaneously
    /// suprathreshold neurons. WTA must remain active in `--release`
    /// builds, where `debug_assert!` is compiled out.
    #[test]
    fn multistep_leaky_matches_explicit_loop_with_inhibition() {
        use crate::state::ResetMode;
        use crate::surrogate::FastSigmoid;

        let device = dev();
        let surrogate = FastSigmoid::default();
        let config = LeakyConfig::new(0.9)
            .with_threshold(0.5)
            .with_reset_mode(ResetMode::Subtract)
            .with_inhibition(true);

        // At each step, neurons 0 and 2 are simultaneously suprathreshold -
        // precisely the case where the absence of WTA gives two spikes instead of one.
        let steps: Vec<f32> = vec![
            0.9, 0.1, 1.3, // t=0: neurons 0 and 2 are suprathreshold, 2 is stronger
            0.7, 0.0, 0.6, // t=1: neurons 0 and 2 are suprathreshold, 0 is stronger
        ];
        let input = t3(steps.clone(), [1, 2, 3]);

        let explicit_layer = config.init(&device);
        let mut explicit_state = NeuronState::zeros_mem_only(&device, 1, 3);
        let mut explicit_out = Vec::new();
        for t in 0..2 {
            let x = t2(steps[t * 3..t * 3 + 3].to_vec(), [1, 3]);
            let (spike, new_state) = explicit_layer.step(x, &explicit_state, &surrogate);
            explicit_state = new_state;
            explicit_out.extend(vec2(spike));
        }

        let multistep_layer = MultiStepLeaky::from_layer(config.init(&device));
        let multistep_out = vec3(multistep_layer.forward(input, &surrogate));

        for (i, (&a, &b)) in explicit_out.iter().zip(multistep_out.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-4,
                "element {i}: explicit={a} multistep={b} - WTA dispersed"
            );
        }

        // Verifying that the WTA actually worked (and not just by accident)
        // matched): exactly one winner per step among three neurons.
        for t in 0..2 {
            let row = &multistep_out[t * 3..t * 3 + 3];
            let winners = row.iter().filter(|&&v| v > 0.5).count();
            assert_eq!(
                winners, 1,
                "step {t}: there must be exactly one winner, obtained {row:?}"
            );
        }
    }

    /// `forward_from_state` must continue EXACTLY from the transferred state:
    /// running the piece [0..2) and then continuing [2..4) from its final
    /// state must coincide with running the entire sequence [0..4) in one
    /// pass.
    #[test]
    fn multistep_leaky_forward_from_state_matches_single_pass() {
        use crate::state::ResetMode;
        use crate::surrogate::FastSigmoid;

        let device = dev();
        let surrogate = FastSigmoid::default();
        let config = LeakyConfig::new(0.8)
            .with_threshold(0.5)
            .with_reset_mode(ResetMode::Subtract);

        // `steps` contains 8 elements, but only `steps[0..4]` is used:
        // the chunks below (`chunk1`/`chunk2`) cover the [0..4) range,
        // matching the contract checked here. Using all 8 elements for
        // `full_input` would make `full_out` (8) and `chunked_out` (4)
        // different lengths, and `zip` would silently truncate the
        // comparison to 4 elements.
        let steps: Vec<f32> = vec![0.9, 0.1, 0.05, 0.95, 0.4, 0.4, 0.0, 0.9];
        let full_input = t3(steps[0..4].to_vec(), [1, 4, 1]);

        // Single run.
        let layer = MultiStepLeaky::from_layer(config.init(&device));
        let (full_out, _) = layer.forward_with_state(full_input, &surrogate);
        let full_out = vec3(full_out);

        // The same layer (the same parameters), two chunks in a row.
        let layer2 = MultiStepLeaky::from_layer(config.init(&device));
        let chunk1 = t3(steps[0..2].to_vec(), [1, 2, 1]);
        let chunk2 = t3(steps[2..4].to_vec(), [1, 2, 1]);

        let (out1, state1) = layer2.forward_with_state(chunk1, &surrogate);
        let (out2, _) = layer2.forward_from_state(chunk2, &state1, &surrogate);

        let mut chunked_out = vec3(out1);
        chunked_out.extend(vec3(out2));

        assert_eq!(
            full_out.len(),
            chunked_out.len(),
            "single-pass and chunked outputs must be the same length - otherwise the zip below will silently truncate the comparison"
        );
        for (i, (&a, &b)) in full_out.iter().zip(chunked_out.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-5,
                "element {i}: single-pass={a} chunked={b} - state was not transferred between chunks"
            );
        }
    }

    /// `forward_with_state` must return the correct final
    /// `spike_prev` - needed to continue the sequence through
    /// `StatefulRecurrent` or another multi-step chunk without a break
    /// recurrent communication.
    #[test]
    fn multistep_rleaky_returns_final_spike_prev() {
        use crate::state::ResetMode;
        use crate::surrogate::FastSigmoid;

        let device = dev();
        let surrogate = FastSigmoid::default();
        let layer = RLeaky::new(&device, 1, 1, 0.9, 0.3, ResetMode::Subtract);
        let multistep_layer = MultiStepRLeaky::from_layer(layer);

        let input = t3(vec![0.9, 0.9, 0.9], [1, 3, 1]);
        let (spikes, _, spike_prev) = multistep_layer.forward_with_state(input, &surrogate);

        let spikes = vec3(spikes);
        let last_spike = spikes[2];
        let returned_spike_prev = vec2(spike_prev)[0];

        assert!(
            (last_spike - returned_spike_prev).abs() < 1e-6,
            "spike_prev in the output should be the spike of the last step: {last_spike} vs {returned_spike_prev}"
        );
    }

    /// `MultiStepLeaky` must panic with a clear message on an empty sequence
    /// (`num_steps == 0`), rather than fail deep inside `Tensor::stack` on an
    /// empty `Vec`. Unified contract for all temporal APIs:
    /// `assert!(num_steps > 0, ...)` at the beginning of the function.
    #[test]
    #[should_panic(expected = "num_steps must be > 0")]
    fn multistep_leaky_panics_on_empty_sequence() {
        use crate::surrogate::FastSigmoid;

        let device = dev();
        let surrogate = FastSigmoid::default();
        let layer = MultiStepLeaky::new(&LeakyConfig::new(0.9), &device);
        let input = t3(vec![], [1, 0, 1]);
        let _ = layer.forward(input, &surrogate);
    }

    /// The same contract for `MultiStepRLeaky` - see.
    /// `multistep_leaky_panics_on_empty_sequence`.
    #[test]
    #[should_panic(expected = "num_steps must be > 0")]
    fn multistep_rleaky_panics_on_empty_sequence() {
        use crate::state::ResetMode;
        use crate::surrogate::FastSigmoid;

        let device = dev();
        let surrogate = FastSigmoid::default();
        let layer = RLeaky::new(&device, 1, 1, 0.9, 0.3, ResetMode::Subtract);
        let multistep_layer = MultiStepRLeaky::from_layer(layer);
        let input = t3(vec![], [1, 0, 1]);
        let _ = multistep_layer.forward(input, &surrogate);
    }
}

/// Equivalence of the fast path ([`run_leaky_sequence`]) to the step-by-step `Leaky::step`: the same
/// spikes, the same final state and the same gradients along the input, `β`, threshold and graded factor.
#[cfg(test)]
mod fused_sequence_tests {
    use super::*;
    use crate::surrogate::{ATan, FastSigmoid};
    use burn::tensor::{Distribution, TensorData};

    fn dev() -> Device {
        Device::flex().autodiff()
    }

    fn v<const D: usize>(t: Tensor<D>) -> Vec<f32> {
        t.into_data().try_to_vec::<f32>().unwrap()
    }

    fn assert_close(name: &str, a: &[f32], b: &[f32], tol: f32) {
        assert_eq!(a.len(), b.len(), "{name}: lengths");
        for (i, (x, y)) in a.iter().zip(b).enumerate() {
            assert!(
                (x - y).abs() <= tol * (1.0 + y.abs()),
                "{name}[{i}]: {x} vs {y}"
            );
        }
    }

    fn check(cfg: LeakyConfig, label: &str) {
        let device = dev();
        let (b, t, f) = (3, 9, 5);
        let lif = Leaky::new(&cfg, &device);
        let data: Vec<f32> = (0..b * t * f)
            .map(|i| ((i * 37 % 23) as f32) / 9.0 - 0.3)
            .collect();
        let sg = ATan::default();

        // The initial state is not zero: we check and chunked start.
        let mem0: Vec<f32> = (0..b * f).map(|i| (i % 4) as f32 * 0.4).collect();
        let st0 = || NeuronState {
            mem: Tensor::<2>::from_data(TensorData::new(mem0.clone(), [b, f]), &device),
            syn: None,
        };
        let x = || {
            Tensor::<3>::from_data(TensorData::new(data.clone(), [b, t, f]), &device).require_grad()
        };

        // Reference: Leaky::step in a loop.
        let xr = x();
        let mut st = st0();
        let mut outs = Vec::new();
        for k in 0..t {
            let xt = xr
                .clone()
                .slice([0..b, k..k + 1, 0..f])
                .squeeze_dims::<2>(&[1]);
            let (sp, ns) = lif.step(xt, &st, &sg);
            st = ns;
            outs.push(sp);
        }
        let ref_spike = Tensor::stack::<3>(outs, 1);
        let ref_mem = st.mem.clone();
        let w = Tensor::<3>::random([b, t, f], Distribution::Uniform(0.5, 1.5), &device);
        let g_ref = (ref_spike.clone() * w.clone()).sum().backward();
        let gx_ref = v(xr.grad(&g_ref).unwrap());
        let gb_ref = v(lif.beta.grad(&g_ref).unwrap());
        let gth_ref = lif
            .threshold_param
            .as_ref()
            .map(|p| v(p.grad(&g_ref).unwrap()));

        // Fast way.
        let xf = x();
        let (spike, stf) = run_leaky_sequence(&lif, xf.clone(), &st0(), &sg);
        let g = (spike.clone() * w).sum().backward();
        assert_close(&format!("{label} spike"), &v(spike), &v(ref_spike), 1e-6);
        assert_close(&format!("{label} mem"), &v(stf.mem), &v(ref_mem), 1e-5);
        assert_close(
            &format!("{label} dx"),
            &v(xf.grad(&g).unwrap()),
            &gx_ref,
            1e-4,
        );
        assert_close(
            &format!("{label} dbeta"),
            &v(lif.beta.grad(&g).unwrap()),
            &gb_ref,
            1e-4,
        );
        if let Some(r) = gth_ref {
            let got = v(lif.threshold_param.as_ref().unwrap().grad(&g).unwrap());
            assert_close(&format!("{label} dthr"), &got, &r, 1e-4);
        }
    }

    #[test]
    fn fused_sequence_matches_step_loop_for_all_modes() {
        for mode in [ResetMode::Subtract, ResetMode::Zero, ResetMode::None] {
            for order in [ResetOrder::BeforeDecay, ResetOrder::AfterDecay] {
                for inhibition in [false, true] {
                    let cfg = LeakyConfig::new(0.8)
                        .with_threshold(1.0)
                        .with_learn_threshold(true)
                        .with_reset_mode(mode)
                        .with_reset_order(order)
                        .with_inhibition(inhibition)
                        .with_graded_spikes_factor(1.5);
                    check(cfg, &format!("{mode:?}/{order:?}/inh={inhibition}"));
                }
            }
        }
    }

    #[test]
    fn fused_sequence_other_surrogate_smoke() {
        let device = dev();
        let lif = Leaky::new(&LeakyConfig::new(0.9), &device);
        let x: Tensor<3> = Tensor::random([2, 6, 4], Distribution::Normal(0.5, 1.0), &device);
        let st = NeuronState::zeros_mem_only(&device, 2, 4);
        let (s, _) = run_leaky_sequence(&lif, x, &st, &FastSigmoid::default());
        assert_eq!(s.dims(), [2, 6, 4]);
    }
}
