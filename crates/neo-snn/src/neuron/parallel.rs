// File: crates/neo-snn/src/neuron/parallel.rs

//! `LeakyParallel`: runs an LIF layer across the time axis
//! `[batch, time, features]`.
//!
//! ## Why this isn't just `Leaky::step` in a loop
//!
//! The reset-free LIF membrane is a first-order linear autoregressive
//! filter: `U[t] = β·U[t-1] + I[t]`, which unrolls into the closed form
//!
//! ```text
//! U[t] = Σ_{k=0}^{t} β^(t-k) · I[k]
//! ```
//!
//! This sum has no sequential dependency between timesteps — each `U[t]` is
//! computed directly from the input `I[0..=t]` — so it can be computed in
//! one pass over the whole sequence instead of `T` sequential small
//! kernel launches on the GPU (as a `for t in 0..num_steps` loop calling
//! `Leaky::step` on every iteration would do). This is how `LeakyParallel`
//! is vectorized in snnTorch.
//!
//! ## Why this only works for `ResetMode::None`
//!
//! The unrolling above is only valid if `U[t-1]` in the recurrence is the
//! membrane WITHOUT threshold subtraction/zeroing on spike. As soon as
//! reset is enabled (`Subtract`/`Zero`), `U[t]` starts depending on
//! `spike_hard[t-1] = H(U[t-1] - threshold)` — a NONLINEAR function of
//! `U[t-1]` — and the sum above stops being valid: `U[t]` then really
//! depends not just on the input but on whether the neuron spiked at every
//! previous step. That is a genuine recurrence, and there is no way around
//! it without losing accuracy (i.e. without computing a different,
//! approximate function). snnTorch limits `Leaky_Parallel` to the
//! same case — it's meant for pipelines where the membrane is used as a
//! temporal filter (`reset_mechanism="none"`, rate-coded readout from
//! `mem`), not as a chain of resetting spikes.
//!
//! So there are two paths here:
//! * `reset_mode == ResetMode::None` — the vectorized path
//!   [`forward_with_state`](LeakyParallel::forward_with_state): one fused
//!   scan across time, no per-step Rust/Python loop.
//! * `Subtract` / `Zero` — the sequential path (unchanged): the only
//!   strictly correct way to compute this recurrence element by element.
//!   The layer doesn't pretend to have sped up something that can't be
//!   vectorized without changing the result.
//!
//! ## Winner-take-all (`inhibition`) in the parallel path
//!
//! `forward_parallel` applies [`winner_take_all`](crate::functional::winner_take_all)
//! at the same point in the pipeline as `Leaky::step` (sequential path) —
//! BEFORE the graded factor, on the already-binary spikes — so that
//! `Leaky` and `LeakyParallel` under the same config (`inhibition=true`,
//! `ResetMode::None`) spike identically. `winner_take_all` operates on
//! `[N, units]` and reduces by max over the last axis; here it's called on
//! `[batch*time, features]`, i.e. WTA is still per-(batch, timestep),
//! matching the sequential path's per-batch, per-step WTA. The mask is
//! built from the membrane, detached from the graph — suppression is
//! structural, not learned, same as in `Leaky::step`.
//!
//! ## How the fused scan is computed
//!
//! `Σ_{k=0}^{t} β^(t-k)·I[k]` is a linear associative scan: timestep `t`
//! carries the pair `(A[t], X[t]) = (β, I[t])` under the affine-transform
//! composition `(a2, x2) ∘ (a1, x1) = (a2·a1, a2·x1 + x2)` (associative,
//! since function composition always is), and an inclusive prefix scan of
//! this operation gives exactly `X[t] = Σ_{k=0}^{t} β^(t-k)·I[k]` at
//! position `t`.
//!
//! `forward_parallel` delegates this scan to
//! [`super::leaky_scan_op::leaky_scan_fused`], which calls
//! `neo_snn::leaky_scan_forward`. On a CubeCL device (wgpu / CUDA / ROCm)
//! that runs neo-synk's fused CubeCL kernel with a hand-written backward; on any
//! other device it falls back to plain tensor ops (autodiff works automatically
//! in both cases). The whole `[batch, steps, features]`
//! output — including the `mem0`
//! contribution — is computed in one call, rather than the `O(log T)`
//! sequence of doubling steps `hillis_steele_linear_scan` below performs.
//! That function is kept in this module (not removed) because it's still
//! exercised directly by
//! `scan_result_independent_of_batch_and_feature_broadcast` below as a
//! plain-tensor-op reference implementation; `forward_parallel` itself no
//! longer calls it.
//!
//! `f32` is not associative under rounding, so the fused-kernel path and
//! the doubling-scan path can differ in the last few bits even though both
//! compute the same mathematical recurrence — `parallel_matches_sequential_when_reset_none`
//! and the other regression tests below compare with a `1e-4`/`1e-3`
//! tolerance, not `assert_eq!`, for exactly this reason.

use burn::module::Module;
use burn::tensor::Device;
use burn::tensor::Tensor;
use burn::tensor::activation::sigmoid;

use super::leaky_scan_op::leaky_scan_fused;
use super::{Leaky, LeakyConfig, apply_graded_spikes, spike_with_threshold};
use crate::functional::winner_take_all;
use crate::state::{NeuronState, ResetMode};
use crate::surrogate::Surrogate;

/// Runs an LIF layer across the time axis.
///
/// Input `[batch, time, features]` -> spikes `[batch, time, features]`.
#[derive(Module, Debug)]
pub struct LeakyParallel {
    /// Inner LIF layer.
    pub lif: Leaky,
}

impl LeakyParallel {
    /// Create the layer.
    pub fn new(config: &LeakyConfig, device: &Device) -> Self {
        Self {
            lif: Leaky::new(config, device),
        }
    }

    /// Runs across time. Returns spikes only.
    pub fn forward<S: Surrogate + ?Sized>(&self, input: Tensor<3>, surrogate: &S) -> Tensor<3> {
        self.forward_with_state(input, surrogate).0
    }

    /// Runs across time and returns the final state, starting from a ZERO
    /// initial state — use this when the sequence is processed as a single
    /// chunk.
    ///
    /// The initial state here is always zero; to continue from an already
    /// known state (real chunked processing — the second and later chunks
    /// of truncated BPTT), use
    /// [`forward_from_state`](Self::forward_from_state), which this
    /// function now calls as a thin wrapper — the same pattern as
    /// [`crate::neuron::MultiStepLeaky::forward_with_state`].
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

    /// Runs across time, continuing from an existing state — the real
    /// entry point for truncated BPTT / chunked processing: the next
    /// chunk of the sequence starts exactly where the previous one left
    /// off, not always from zero.
    ///
    /// At `reset_mode == ResetMode::None` uses the fused scan path (see
    /// the module docs and [`forward_parallel`](Self::forward_parallel));
    /// otherwise the sequential path
    /// ([`forward_sequential`](Self::forward_sequential)), because
    /// membrane reset on spike makes the timesteps genuinely recurrent and
    /// they can't be vectorized without losing accuracy.
    pub fn forward_from_state<S: Surrogate + ?Sized>(
        &self,
        input: Tensor<3>,
        initial_state: &NeuronState,
        surrogate: &S,
    ) -> (Tensor<3>, NeuronState) {
        self.forward_from_state_impl(input, initial_state, surrogate)
    }

    fn forward_from_state_impl<S: Surrogate + ?Sized>(
        &self,
        input: Tensor<3>,
        initial_state: &NeuronState,
        surrogate: &S,
    ) -> (Tensor<3>, NeuronState) {
        let [_batch, num_steps, _features] = input.dims();
        assert!(num_steps > 0, "LeakyParallel: num_steps must be > 0");
        if self.lif.reset_mode == ResetMode::None {
            self.forward_parallel(input, initial_state, surrogate)
        } else {
            self.forward_sequential(input, initial_state, surrogate)
        }
    }

    /// Sequential path: `Leaky::step` in a loop over time, starting from
    /// `initial_state`. The only correct approach for
    /// `ResetMode::Subtract`/`Zero`, see module docs.
    fn forward_sequential<S: Surrogate + ?Sized>(
        &self,
        input: Tensor<3>,
        initial_state: &NeuronState,
        surrogate: &S,
    ) -> (Tensor<3>, NeuronState) {
        // Surrogate, WTA and graded factor are computed once over the entire sequence,
        // only the recurrent one with a hard reset remains in the loop (see `run_leaky_sequence`).
        super::multistep::run_leaky_sequence(&self.lif, input, initial_state, surrogate)
    }

    /// Fused path for `ResetMode::None`: the whole sequence's membrane via
    /// [`leaky_scan_forward`](crate::leaky_scan_forward), then
    /// spike threshold, winner-take-all (if `inhibition == true`) and the
    /// graded factor — one pass over the whole
    /// `[batch, time, features]`.
    ///
    /// Unrolling `U[t] = β·U[t-1] + I[t]` for `t = 0..num_steps` with
    /// `U[-1] = mem0` gives
    ///
    /// ```text
    /// U[t] = β^(t+1)·mem0 + Σ_{k=0}^{t} β^(t-k)·I[k]
    /// ```
    ///
    /// — `leaky_scan_forward` computes exactly this, `mem0` contribution
    /// included, in one fused call (see module docs for how the scan
    /// itself works and why it's delegated to `neo-synk`).
    fn forward_parallel<S: Surrogate + ?Sized>(
        &self,
        input: Tensor<3>,
        initial_state: &NeuronState,
        surrogate: &S,
    ) -> (Tensor<3>, NeuronState) {
        self.forward_parallel_impl(input, initial_state, surrogate)
    }

    fn forward_parallel_impl<S: Surrogate + ?Sized>(
        &self,
        input: Tensor<3>,
        initial_state: &NeuronState,
        surrogate: &S,
    ) -> (Tensor<3>, NeuronState) {
        let device = input.device();
        let [batch, num_steps, features] = input.dims();

        let beta = sigmoid(self.lif.beta.val()); // [1], always in (0, 1)
        let mem0 = initial_state.mem.clone(); // [batch, features]

        // Stage 1 fused forward: `mem0` is the recurrence initial condition.
        let mem = leaky_scan_fused(input, mem0, beta);

        // Spike threshold — the same formulas as Leaky::step, applied to
        // the whole [batch, time, features] at once: threshold doesn't
        // depend on time, so a single expand across the whole sequence is
        // correct.
        let mem_2d = mem.clone().reshape([batch * num_steps, features]);
        let th = self
            .lif
            .threshold_tensor(&device, [batch * num_steps, features]);
        let (spike, _spike_hard) = spike_with_threshold(mem_2d.clone(), th, surrogate);

        // Winner-take-all: the same suppression Leaky::step applies at
        // every step — here in one call across the whole flattened
        // [batch*time, features], since winner_take_all takes the max over
        // the last axis independently per row, which is equivalent to
        // per-(batch, timestep) WTA. Applied BEFORE the graded factor, to
        // the already-binary spikes — same as in Leaky::step.
        let spike = if self.lif.inhibition {
            winner_take_all(spike, mem_2d)
        } else {
            spike
        };

        let spike = apply_graded_spikes(spike, self.lif.graded_factor_tensor(&device));
        let spike = spike.reshape([batch, num_steps, features]);

        // Final state — the membrane at the last step, no reset
        // (ResetMode::None); Leaky has no synaptic current.
        let final_mem = mem
            .slice([0..batch, num_steps - 1..num_steps, 0..features])
            .squeeze_dims::<2>(&[1]);
        let state = NeuronState::new(final_mem, None);

        (spike, state)
    }
}

/// Computes `X[t] = Σ_{k=0}^{t} β^(t-k) · input[b, k, f]` for every `(b, f)`
/// via a parallel (Hillis-Steele, inclusive) linear scan, using plain Burn
/// tensor ops rather than an explicit `[T,T]` weight matrix.
///
/// The production path (`forward_parallel`) delegates to
/// [`leaky_scan_forward`](crate::leaky_scan_forward) instead, which runs a fused CubeCL
/// kernel on GPU devices; this function is kept as a backend-agnostic
/// plain-tensor-op scan, and is exercised directly by
/// `scan_result_independent_of_batch_and_feature_broadcast` below.
///
/// # Why this is an associative scan
///
/// `U[t] = β·U[t-1] + I[t]` is a linear (affine) autoregression. Affine
/// transforms `x ↦ a·x + b` form a monoid under composition:
/// `(a2, b2) ∘ (a1, b1) = (a2·a1, a2·b1 + b2)`, which is ASSOCIATIVE
/// (function composition always is, regardless of the specific `a`, `b`).
/// Assigning each step `t` the element `(A[t], X[t]) = (β, I[t])`, an
/// inclusive prefix scan of this monoid operation gives, at position `t`,
/// the composition `(β, I[t]) ∘ (β, I[t-1]) ∘ ... ∘ (β, I[0])`, whose
/// second component is exactly `Σ_{k=0}^{t} β^(t-k)·I[k]` (unrolls by
/// induction on `t`).
///
/// # Algorithm (Hillis-Steele)
///
/// In `⌈log2 T⌉` doubling steps `d = 1, 2, 4, ...`: at distance `d`,
/// replace element `t` (for every `t >= d`) with the composition of the
/// current element `t` with element `t - d`; elements `t < d` are left
/// unchanged (equivalent to composing with the identity `(1, 0)`). After
/// `⌈log2 T⌉` such steps, position `t` holds the composition of ALL
/// elements `0..=t` — exactly the prefix scan. `O(B·T·F·log T)` compute,
/// `O(B·T·F)` peak memory.
#[cfg_attr(not(test), allow(dead_code))]
fn hillis_steele_linear_scan(
    input: Tensor<3>,
    beta: Tensor<1>,
    batch: usize,
    steps: usize,
    features: usize,
) -> Tensor<3> {
    let device = input.device();

    let mut a: Tensor<1> = beta.reshape([1]).expand([steps]);
    let mut x: Tensor<3> = input;

    let mut d = 1usize;
    while d < steps {
        let valid = steps - d;

        #[allow(clippy::single_range_in_vec_init)]
        let a_tail = a.clone().slice([0..valid]);
        let identity_a = Tensor::<1>::ones([d], &device);
        let a_shift: Tensor<1> = Tensor::cat(vec![identity_a, a_tail], 0);

        let x_tail = x.clone().slice([0..batch, 0..valid, 0..features]);
        let identity_x = Tensor::<3>::zeros([batch, d, features], &device);
        let x_shift: Tensor<3> = Tensor::cat(vec![identity_x, x_tail], 1);

        let a_bcast = a
            .clone()
            .reshape([1, steps, 1])
            .expand([batch, steps, features]);
        let new_x = a_bcast.mul(x_shift) + x.clone();
        let new_a = a.clone().mul(a_shift);

        a = new_a;
        x = new_x;
        d *= 2;
    }

    x
}

#[cfg(test)]
mod parallel_tests {
    use super::*;
    fn dev() -> Device {
        Device::flex().autodiff()
    }

    fn t3(data: Vec<f32>, shape: [usize; 3]) -> Tensor<3> {
        Tensor::<3>::from_data(burn::tensor::TensorData::new(data, shape), &dev())
    }

    fn vec3(t: Tensor<3>) -> Vec<f32> {
        t.into_data().try_to_vec::<f32>().unwrap()
    }

    /// Correctness: at `ResetMode::None` the fused path must give EXACTLY
    /// the same result as the sequential `Leaky::step` loop — this is the
    /// same mathematical function, computed two different ways, not two
    /// different approximations.
    #[test]
    fn parallel_matches_sequential_when_reset_none() {
        let device = dev();
        let config = LeakyConfig::new(0.85).with_reset_mode(ResetMode::None);
        let layer = LeakyParallel::new(&config, &device);
        let surrogate = crate::surrogate::FastSigmoid::default();

        let steps: Vec<f32> = vec![
            0.3, -0.2, // t=0, batch=1, features=2
            0.1, 0.4, // t=1
            0.0, 0.0, // t=2
            1.2, -0.5, // t=3
            0.05, 0.9, // t=4
        ];
        let n = 5;
        let input = t3(steps, [1, n, 2]);

        let zero_state = NeuronState::zeros_mem_only(&device, 1, 2);
        let (par_out, par_state) = layer.forward_parallel(input.clone(), &zero_state, &surrogate);
        let (seq_out, seq_state) = layer.forward_sequential(input, &zero_state, &surrogate);

        let par_vec = vec3(par_out);
        let seq_vec = vec3(seq_out);
        for (i, (&a, &b)) in par_vec.iter().zip(seq_vec.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-4,
                "element {i}: parallel={a} vs sequential={b}"
            );
        }

        let par_mem = par_state.mem.into_data().try_to_vec::<f32>().unwrap();
        let seq_mem = seq_state.mem.into_data().try_to_vec::<f32>().unwrap();
        for (i, (&a, &b)) in par_mem.iter().zip(seq_mem.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-4,
                "final membrane {i}: parallel={a} vs sequential={b}"
            );
        }
    }

    /// `inhibition=true` together with `ResetMode::None` must
    /// produce winner-take-all in the parallel path EXACTLY as in the
    /// sequential path.
    #[test]
    fn parallel_matches_sequential_with_inhibition_and_two_suprathreshold_neurons() {
        let device = dev();
        let config = LeakyConfig::new(0.9)
            .with_threshold(0.5)
            .with_reset_mode(ResetMode::None)
            .with_inhibition(true);
        let layer = LeakyParallel::new(&config, &device);
        let surrogate = crate::surrogate::FastSigmoid::default();

        let steps: Vec<f32> = vec![
            0.9, 0.1, 1.3, // t=0: neurons 0 and 2 suprathreshold, 2 stronger
            0.7, 0.0, 0.6, // t=1: neurons 0 and 2 suprathreshold, 0 stronger
        ];
        let input = t3(steps, [1, 2, 3]);

        let zero_state = NeuronState::zeros_mem_only(&device, 1, 3);
        let (par_out, _) = layer.forward_parallel(input.clone(), &zero_state, &surrogate);
        let (seq_out, _) = layer.forward_sequential(input, &zero_state, &surrogate);

        let par_vec = vec3(par_out);
        let seq_vec = vec3(seq_out);
        for (i, (&a, &b)) in par_vec.iter().zip(seq_vec.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-4,
                "element {i}: parallel={a} vs sequential={b} - WTA diverged between paths"
            );
        }

        for t in 0..2 {
            let row = &par_vec[t * 3..t * 3 + 3];
            let winners = row.iter().filter(|&&v| v > 0.5).count();
            assert_eq!(
                winners, 1,
                "step {t}: there must be exactly one winner, obtained {row:?}"
            );
        }
    }

    /// `forward_with_state` must itself pick the path by `reset_mode`.
    #[test]
    fn dispatch_uses_sequential_path_when_reset_is_not_none() {
        let device = dev();
        let config = LeakyConfig::new(0.9)
            .with_threshold(0.5)
            .with_reset_mode(ResetMode::Subtract);
        let layer = LeakyParallel::new(&config, &device);
        let surrogate = crate::surrogate::FastSigmoid::default();

        let input = t3(vec![0.9, 0.9], [1, 2, 1]);
        let (_, state) = layer.forward_with_state(input, &surrogate);

        let zero_state = NeuronState::zeros_mem_only(&device, 1, 1);
        let (_, seq_state) =
            layer.forward_sequential(t3(vec![0.9, 0.9], [1, 2, 1]), &zero_state, &surrogate);
        let got = state.mem.into_data().try_to_vec::<f32>().unwrap();
        let want = seq_state.mem.into_data().try_to_vec::<f32>().unwrap();
        assert!((got[0] - want[0]).abs() < 1e-5);
    }

    /// Gradient through the learnable `beta` must flow through the fused
    /// path — otherwise the optimizer couldn't learn membrane decay in
    /// this mode.
    #[test]
    fn beta_gradient_flows_through_parallel_path() {
        let device = dev();
        let config = LeakyConfig::new(0.7).with_reset_mode(ResetMode::None);
        let layer = LeakyParallel::new(&config, &device);
        let surrogate = crate::surrogate::FastSigmoid::default();

        let input = t3(vec![0.6, 0.6, 0.6, 0.6], [1, 4, 1]);
        let zero_state = NeuronState::zeros_mem_only(&device, 1, 1);
        let (spikes, _) = layer.forward_parallel(input, &zero_state, &surrogate);
        let loss = spikes.sum();
        let grads = loss.backward();

        let beta_grad = layer.lif.beta.grad(&grads);
        assert!(
            beta_grad.is_some(),
            "gradient along beta does not flow through parallel path"
        );
    }

    #[test]
    #[should_panic(expected = "num_steps must be > 0")]
    fn leaky_parallel_panics_on_empty_sequence_reset_none() {
        let device = dev();
        let config = LeakyConfig::new(0.85).with_reset_mode(ResetMode::None);
        let layer = LeakyParallel::new(&config, &device);
        let surrogate = crate::surrogate::FastSigmoid::default();
        let input = t3(vec![], [1, 0, 1]);
        let _ = layer.forward(input, &surrogate);
    }

    #[test]
    #[should_panic(expected = "num_steps must be > 0")]
    fn leaky_parallel_panics_on_empty_sequence_reset_subtract() {
        let device = dev();
        let config = LeakyConfig::new(0.85).with_reset_mode(ResetMode::Subtract);
        let layer = LeakyParallel::new(&config, &device);
        let surrogate = crate::surrogate::FastSigmoid::default();
        let input = t3(vec![], [1, 0, 1]);
        let _ = layer.forward(input, &surrogate);
    }

    /// `forward_parallel` computes `Σ_{k=0}^{t} β^(t-k)·I[k]` through the
    /// fused `leaky_scan_forward` path. This test checks
    /// equivalence independently of `parallel_matches_sequential_when_reset_none`:
    /// that one compares `forward_parallel` against `forward_sequential`
    /// (both could break the same way simultaneously), while here the
    /// expected value is computed by hand, by the same formula documented
    /// on the module, entirely bypassing both implementations. `num_steps
    /// = 17` is deliberately NOT a power of two.
    #[test]
    fn scan_matches_hand_computed_closed_form_for_non_power_of_two_length() {
        let device = dev();
        let beta = 0.6f32;
        let config = LeakyConfig::new(beta).with_reset_mode(ResetMode::None);
        let layer = LeakyParallel::new(&config, &device);
        let surrogate = crate::surrogate::FastSigmoid::default();

        let num_steps = 17usize;
        let raw: Vec<f32> = (0..num_steps).map(|t| ((t as f32) * 0.3).sin()).collect();
        let input = t3(raw.clone(), [1, num_steps, 1]);

        let zero_state = NeuronState::zeros_mem_only(&device, 1, 1);
        let (spikes, par_state) = layer.forward_parallel(input, &zero_state, &surrogate);
        let _ = spikes;

        let mut expected = vec![0.0f64; num_steps];
        #[allow(clippy::needless_range_loop)]
        for t in 0..num_steps {
            let mut acc = 0.0f64;
            #[allow(clippy::needless_range_loop)]
            for k in 0..=t {
                acc += (beta as f64).powi((t - k) as i32) * raw[k] as f64;
            }
            expected[t] = acc;
        }
        let expected_final = expected[num_steps - 1];

        let got_final = par_state.mem.into_data().try_to_vec::<f32>().unwrap()[0] as f64;
        assert!(
            (got_final - expected_final).abs() < 1e-3,
            "scan (t={}): received {got_final}, expected {expected_final}",
            num_steps - 1
        );
    }

    /// At `num_steps = 1`, `hillis_steele_linear_scan`'s
    /// `while d < steps` loop must not execute at all.
    #[test]
    fn scan_handles_single_step_without_any_doubling() {
        let device = dev();
        let config = LeakyConfig::new(0.5).with_reset_mode(ResetMode::None);
        let layer = LeakyParallel::new(&config, &device);
        let surrogate = crate::surrogate::FastSigmoid::default();

        let input = t3(vec![0.42], [1, 1, 1]);
        let zero_state = NeuronState::zeros_mem_only(&device, 1, 1);
        let (_, state) = layer.forward_parallel(input, &zero_state, &surrogate);

        let got = state.mem.into_data().try_to_vec::<f32>().unwrap()[0];
        assert!((got - 0.42).abs() < 1e-5, "received {got}, expected 0.42");
    }

    /// [`hillis_steele_linear_scan`] must give the same result whether `A`
    /// (the β coefficient) is stored as `[steps]` or as
    /// `[batch, steps, features]` - this keeps memory traffic independent of
    /// the batch/feature broadcast.
    #[test]
    fn scan_result_independent_of_batch_and_feature_broadcast() {
        let device = dev();
        let beta_val = 0.73f32;
        let raw: Vec<f32> = vec![0.5, -0.3, 0.8, 0.1, -0.6];
        let steps = raw.len();

        let beta = Tensor::<1>::from_floats([beta_val], &device);

        let single_input = t3(raw.clone(), [1, steps, 1]);
        let single_out = hillis_steele_linear_scan(single_input, beta.clone(), 1, steps, 1);
        let single_vec = vec3(single_out);

        let batch = 3;
        let features = 2;
        let mut dup = Vec::with_capacity(batch * steps * features);
        for _ in 0..batch {
            for &v in &raw {
                for _ in 0..features {
                    dup.push(v);
                }
            }
        }
        let dup_input = t3(dup, [batch, steps, features]);
        let dup_out = hillis_steele_linear_scan(dup_input, beta, batch, steps, features);
        let dup_vec = vec3(dup_out);

        for b in 0..batch {
            #[allow(clippy::needless_range_loop)]
            for t in 0..steps {
                for f in 0..features {
                    let idx = b * steps * features + t * features + f;
                    let expected = single_vec[t];
                    let got = dup_vec[idx];
                    assert!(
                        (got - expected).abs() < 1e-5,
                        "(b={b}, t={t}, f={f}): received {got}, expected {expected} \
                         (match with batch=1,features=1 for the same row)"
                    );
                }
            }
        }
    }
}
