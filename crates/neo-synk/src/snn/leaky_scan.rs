// neo-synk/src/snn/leaky_scan.rs
//
// Fused CubeCL kernel for the LIF leaky-membrane linear recurrence used by
// `neo-snn`'s `LeakyParallel`:
//
//     u[t] = beta * u[t-1] + input[t],   u[-1] = mem0
//
// solved for every (batch, feature) pair and every timestep in one kernel
// launch. Compared with a Hillis-Steele scan expressed as a sequence of
// O(log T) generic Burn tensor ops (temporary [B,T,F] buffers, `cat`,
// broadcast multiplies, ...), fusing removes both the intermediate-tensor
// traffic and the O(log T) launch count.
//
// ## Parallelization strategy
//
// The recurrence is inherently sequential in time, but independent across
// (batch, feature) pairs. So we parallelize over the B*F pairs (one GPU
// thread walks one full chain of length `steps`, in order) rather than
// trying to parallelize across time within a single chain. This keeps the
// kernel simple and branch-free; it is not the theoretically optimal
// work-efficient scan (a chunked/blocked parallel scan across time would
// give better GPU occupancy when batch*features is small relative to the
// number of available threads), but it matches the "first fused kernel"
// scope: no intermediate tensors, one launch. Gradients are computed
// separately on the Rust side (see `neo-snn`'s `leaky_scan_op.rs`) rather
// than inside this kernel.
//
// ## Memory layout
//
// `input` and `output` are `[batch, steps, features]`, row-major
// (C-contiguous) — the caller (`neo-synk`'s backend impl) is responsible for
// calling `into_contiguous` first, since a thread's per-timestep step of
// `+features` elements assumes that exact layout.
//
// `mem0` is `[batch, features]`. `beta` is a single scalar (shared across
// all batches/features) passed as a 1-element tensor, matching how
// `neo-snn::Leaky` stores it (`Param<Tensor<B, 1>>`, sigmoid already applied
// before this kernel ever sees it).

use cubecl::prelude::*;

/// Fused forward kernel: computes `output[b, t, f] = beta^(t+1) * mem0[b, f]
/// + sum_{k=0}^{t} beta^(t-k) * input[b, k, f]` as a running recurrence
/// (`u = beta * u + input[t]` walked forward through time), not by
/// re-summing the closed form at every `t`.
///
/// One thread is launched per `(b, f)` pair; `total_threads = batch *
/// features`. Threads beyond `total_threads` (padding from rounding the
/// launch grid up to a whole number of cubes) exit immediately.
#[cube(launch)]
pub fn leaky_scan_forward_kernel<F: Float>(
    input: &Tensor<F>,
    mem0: &Tensor<F>,
    beta: &Tensor<F>,
    output: &mut Tensor<F>,
    #[comptime] steps: usize,
    #[comptime] features: usize,
) {
    let thread_id = ABSOLUTE_POS;
    let batch = input.shape(0);
    let total_threads = batch * features;

    if thread_id < total_threads {
        // Decompose the flat thread index back into (batch, feature).
        let b = thread_id / features;
        let f = thread_id % features;

        let beta_val = beta[0];

        // Running membrane value; starts at mem0[b, f] and is updated in
        // place as we walk forward through time. This sequential
        // recurrence `u[t] = beta * u[t-1] + input[t]` is exactly what the
        // kernel exists to fuse, so there is no closed-form shortcut here.
        let mut u = mem0[b * features + f];
        let base = b * steps * features;

        for t in 0..steps {
            let idx = base + t * features + f;
            u = beta_val * u + input[idx];
            output[idx] = u;
        }
    }
}
