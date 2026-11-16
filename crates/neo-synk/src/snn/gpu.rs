//! Fused CubeCL leaky-scan kernel for GPU (wgpu / CUDA / ROCm) under Burn 0.22.
//!
//! Burn 0.22 does not parameterize `Tensor` backend: the backend is determined
//! `Device` at runtime. They connect their operation through
//! `#[backend_extension]`: macro generates routing `Dispatch` →
//! specific backend, and implementations are written for `CubeBackend` (one impl per
//! all runtime CubeCL), for `Autodiff<B, C>` (manual backward) and, with the feature
//! `fusion`, for `Fusion<B>` (lazy registration).
//!
//! Recurrent `u[t] = β·u[t-1] + x[t]`, `u[-1] = mem0`. Core: one thread per pair
//! `(batch, feature)`, the time chain runs sequentially.
//!
//! If `batch·features` is small (the GPU is idle), the time chain is cut into
//! `CHUNKS` pieces that are computed in parallel (`leaky_scan_chunked_kernel`):
//! local scan of each piece from scratch, state transfer between pieces
//! `carry' = β^len·carry + tail` and correction `β^(k+1)·carry_in`.
//!
//! Backward - the same recurrent in reverse time, therefore computed by the same
//! fused kernel: `r = flip_t(scan(flip_t(g), 0, β))`, then
//! `dx = r`, `dmem0 = β·r[0]`, `dβ = Σ r·u_prev`.

use burn::backend::{
    Backend, Dispatch, TensorMetadata,
    autodiff::{
        Autodiff, NodeId,
        checkpoint::{base::Checkpointer, strategy::CheckpointStrategy},
        grads::Gradients,
        ops::{Backward, Ops, OpsKind},
    },
    backend_extension,
    cubecl::dtype_to_storage_type,
    tensor::FloatTensor,
};
use burn::tensor::{Shape, Slice, Tensor as BTensor};
use burn_cubecl::{CubeBackend, kernel::into_contiguous, tensor::CubeTensor};
use cubecl::{CubeCount, CubeDim, cube, prelude::*};

/// Threads in workgroup.
const THREADS: u32 = 128;

/// Fused forward: `output[b, t, f] = β^(t+1)·mem0[b, f] + Σ_{k≤t} β^(t-k)·input[b, k, f]`.
///
/// Tensors are continuous (`into_contiguous` calls host code). One thread
/// goes through the entire `(b, f)` chain in time, so the kernel does not need synchronization.
#[cube(launch)]
pub fn leaky_scan_forward_kernel<F: Float>(
    input: &Tensor<F>,
    mem0: &Tensor<F>,
    beta: &Tensor<F>,
    output: &mut Tensor<F>,
    batch: u32,
    steps: u32,
    features: u32,
    #[define(F)] _dtype: ElemType,
) {
    let thread_id = u32::cast_from(ABSOLUTE_POS);
    if thread_id < batch * features {
        let b = thread_id / features;
        let f = thread_id % features;
        let beta_val = beta[0usize];

        let mut u = mem0[usize::cast_from(b * features + f)];
        let base = b * steps * features;
        for t in 0..steps {
            let idx = usize::cast_from(base + t * features + f);
            u = beta_val * u + input[idx];
            output[idx] = u;
        }
    }
}

/// Threads by characteristics in cube for a chunked kernel.
const CHUNK_FX: u32 = 8;
/// Chunks by time (threads by y) in cube for a chunked kernel: `8 × 32 = 256` threads.
const CHUNKS: u32 = 32;
/// The Chunked core is turned on when the `(batch, feature)` pair is less than this threshold...
const CHUNKED_MAX_PAIRS: usize = 2048;
/// ...and there are at least `CHUNKS * CHUNKED_MIN_LEN` steps in the chain.
const CHUNKED_MIN_LEN: usize = 8;

/// Forward with time parallelism. Cube = `CHUNK_FX` features × `CHUNKS`
/// pieces of one batch row; thread `(f, c)` computes its piece of the chain.
///
/// 1. local scan of a piece from zero (written in `output`), `e` is its tail,
///    `p = β^len`;
/// 2. `e` and `p` are placed in shared; the input of the piece `c` is a convolution of the previous ones:
///    `carry = mem0; carry = p[k]·carry + e[k]` for `k < c`;
/// 3. amendment: `output[t] += β^(t-start+1)·carry`.
///
/// Empty pieces (`start >= steps`) are the identity element (`e = 0`, `p = 1`).
#[cube(launch)]
pub fn leaky_scan_chunked_kernel<F: Float>(
    input: &Tensor<F>,
    mem0: &Tensor<F>,
    beta: &Tensor<F>,
    output: &mut Tensor<F>,
    batch: u32,
    steps: u32,
    features: u32,
    #[comptime] fx: u32,
    #[comptime] chunks: u32,
    #[define(F)] _dtype: ElemType,
) {
    let fi = UNIT_POS_X;
    let c = UNIT_POS_Y;
    let fblocks = (features + fx - 1u32) / fx;
    let cube = u32::cast_from(CUBE_POS);
    let b = cube / fblocks;
    let f = (cube % fblocks) * fx + fi;
    let valid = b < batch && f < features;

    let len = (steps + chunks - 1u32) / chunks;
    let start = c * len;
    let end = if start + len < steps {
        start + len
    } else {
        steps
    };
    let beta_val = beta[0usize];
    let base = b * steps * features;

    let mut y = F::new(0.0f32);
    let mut p = F::new(1.0f32);
    if valid {
        for t in start..end {
            let idx = usize::cast_from(base + t * features + f);
            y = beta_val * y + input[idx];
            output[idx] = y;
            p = p * beta_val;
        }
    }

    let mut e_s = Shared::<[F]>::new_slice(comptime!((fx * chunks) as usize));
    let mut p_s = Shared::<[F]>::new_slice(comptime!((fx * chunks) as usize));
    let slot = usize::cast_from(c * fx + fi);
    e_s[slot] = y;
    p_s[slot] = p;
    sync_cube();

    let mut carry = F::new(0.0f32);
    if valid {
        carry = mem0[usize::cast_from(b * features + f)];
    }
    for k in 0..c {
        let s = usize::cast_from(k * fx + fi);
        carry = p_s[s] * carry + e_s[s];
    }

    if valid {
        let mut q = beta_val;
        for t in start..end {
            let idx = usize::cast_from(base + t * features + f);
            output[idx] = output[idx] + q * carry;
            q = q * beta_val;
        }
    }
}

/// Backend extension: fused forward leaky-scan.
///
/// * `input` — `[batch, steps, features]`; `mem0` — `[batch, features]`;
///   `beta` — `[1]`.
#[backend_extension(Cube, Autodiff, Fusion: cfg(feature = "fusion"))]
pub trait LeakyScanBackend: Backend {
    #[fusion(dtype = input, shape = input)]
    fn leaky_scan_forward(
        input: FloatTensor<Self>,
        mem0: FloatTensor<Self>,
        beta: FloatTensor<Self>,
    ) -> FloatTensor<Self>;
}

impl LeakyScanBackend for CubeBackend {
    fn leaky_scan_forward(
        input: FloatTensor<Self>,
        mem0: FloatTensor<Self>,
        beta: FloatTensor<Self>,
    ) -> FloatTensor<Self> {
        input.assert_is_on_same_device(&mem0);
        input.assert_is_on_same_device(&beta);

        let dtype = input.dtype;
        let input = into_contiguous(input);
        let mem0 = into_contiguous(mem0);
        let beta = into_contiguous(beta);
        assert_eq!(
            input.dtype, mem0.dtype,
            "leaky_scan: dtype mem0 != dtype input"
        );
        assert_eq!(
            input.dtype, beta.dtype,
            "leaky_scan: dtype beta != dtype input"
        );

        let shape = input.meta.shape().clone();
        assert_eq!(
            shape.num_dims(),
            3,
            "leaky_scan: input must be [batch, steps, features]"
        );
        let (batch, steps, features) = (shape[0], shape[1], shape[2]);
        assert_eq!(
            mem0.meta.shape().as_slice(),
            &[batch, features],
            "leaky_scan: mem0 must be [batch, features]"
        );
        assert_eq!(
            beta.meta.shape().num_elements(),
            1,
            "leaky_scan: beta must have 1 element"
        );

        let buffer = input.client.empty(shape.num_elements() * dtype.size());
        let output = CubeTensor::new_contiguous(
            input.client.clone(),
            input.device.clone(),
            shape,
            buffer,
            dtype,
        );

        let total = batch * features;
        if total == 0 || steps == 0 {
            return output;
        }
        // Few pairs (batch, feature) with a long chain: we cut the chain according to time.
        if total < CHUNKED_MAX_PAIRS && steps >= CHUNKS as usize * CHUNKED_MIN_LEN {
            let fblocks = features.div_ceil(CHUNK_FX as usize);
            leaky_scan_chunked_kernel::launch(
                &output.client,
                CubeCount::Static((batch * fblocks) as u32, 1, 1),
                CubeDim::new_2d(CHUNK_FX, CHUNKS),
                input.into_tensor_arg(),
                mem0.into_tensor_arg(),
                beta.into_tensor_arg(),
                output.clone().into_tensor_arg(),
                batch as u32,
                steps as u32,
                features as u32,
                CHUNK_FX,
                CHUNKS,
                dtype_to_storage_type(dtype),
            );
            return output;
        }

        let cube_dim = CubeDim::new_1d(THREADS);
        let cube_count = CubeCount::Static(total.div_ceil(THREADS as usize) as u32, 1, 1);

        leaky_scan_forward_kernel::launch(
            &output.client,
            cube_count,
            cube_dim,
            input.into_tensor_arg(),
            mem0.into_tensor_arg(),
            beta.into_tensor_arg(),
            output.clone().into_tensor_arg(),
            batch as u32,
            steps as u32,
            features as u32,
            dtype_to_storage_type(dtype),
        );
        output
    }
}

fn time_slice(from: usize, to: usize) -> [Slice; 3] {
    [
        Slice::new(0, None, 1),
        Slice::new(from as isize, Some(to as isize), 1),
        Slice::new(0, None, 1),
    ]
}

impl<B: LeakyScanBackend, C: CheckpointStrategy> LeakyScanBackend for Autodiff<B, C> {
    fn leaky_scan_forward(
        input: FloatTensor<Self>,
        mem0: FloatTensor<Self>,
        beta: FloatTensor<Self>,
    ) -> FloatTensor<Self> {
        #[derive(Debug)]
        struct LeakyScanBackward;

        impl<B: LeakyScanBackend> Backward<B, 3> for LeakyScanBackward {
            // (mem0, beta) - checkpoints; output `u` is kept in state.
            type State = (NodeId, NodeId, FloatTensor<B>);

            fn backward(
                self,
                ops: Ops<Self::State, 3>,
                grads: &mut Gradients,
                checkpointer: &mut Checkpointer,
            ) {
                let [node_input, node_mem0, node_beta] = ops.parents;
                let grad = grads.consume::<B>(&ops.node);
                let (mem0_state, beta_state, u) = ops.state;
                let mem0: FloatTensor<B> = checkpointer.retrieve_node_output(mem0_state);
                let beta: FloatTensor<B> = checkpointer.retrieve_node_output(beta_state);

                let shape = u.shape();
                let (batch, steps, features) = (shape[0], shape[1], shape[2]);
                let device = u.device();
                let dtype = u.dtype();

                // r_t = g_t + β·r_{t+1}: the same recurrence in reverse time.
                let zeros = B::float_zeros(Shape::new([batch, features]), &device, dtype.into());
                let g_rev = B::float_flip(grad, &[1]);
                let r_rev = B::leaky_scan_forward(g_rev, zeros, beta.clone());
                let r = B::float_flip(r_rev, &[1]);

                // u_prev = [mem0, u[0..T-1]].
                let mem0_row = B::float_reshape(mem0, Shape::new([batch, 1, features]));
                let u_prev = if steps > 1 {
                    let head = B::float_slice(u, &time_slice(0, steps - 1));
                    B::float_cat(vec![mem0_row, head], 1)
                } else {
                    mem0_row
                };

                if let Some(node) = node_input {
                    grads.register::<B>(node.id, r.clone());
                }
                if let Some(node) = node_mem0 {
                    let r0 = B::float_reshape(
                        B::float_slice(r.clone(), &time_slice(0, 1)),
                        Shape::new([batch, features]),
                    );
                    let beta_2d = B::float_expand(
                        B::float_reshape(beta, Shape::new([1, 1])),
                        Shape::new([batch, features]),
                    );
                    grads.register::<B>(node.id, B::float_mul(r0, beta_2d));
                }
                if let Some(node) = node_beta {
                    let g = B::float_sum(B::float_mul(r, u_prev));
                    grads.register::<B>(node.id, B::float_reshape(g, Shape::new([1])));
                }
            }
        }

        match LeakyScanBackward
            .prepare::<C>([input.node(), mem0.node(), beta.node()])
            .compute_bound()
            .stateful()
        {
            OpsKind::Tracked(mut prep) => {
                let mem0_state = prep.checkpoint(&mem0);
                let beta_state = prep.checkpoint(&beta);
                let output = B::leaky_scan_forward(
                    input.primitive().clone(),
                    mem0.primitive().clone(),
                    beta.primitive().clone(),
                );
                prep.finish((mem0_state, beta_state, output.clone()), output)
            }
            OpsKind::UnTracked(prep) => {
                let output = B::leaky_scan_forward(
                    input.into_primitive(),
                    mem0.into_primitive(),
                    beta.into_primitive(),
                );
                prep.finish(output)
            }
        }
    }
}

/// `true`, if the tensor lies on the CubeCL device (wgpu / CUDA / ROCm / cpu-runtime).
pub fn is_cube_device(device: &burn::tensor::Device) -> bool {
    device.identity().is_some()
}

/// Fused path over `Tensor<3>`. Works for both regular tensors and autodiff.
pub fn leaky_scan_fused(input: BTensor<3>, mem0: BTensor<2>, beta: BTensor<1>) -> BTensor<3> {
    BTensor::from_dispatch(Dispatch::leaky_scan_forward(
        input.into_dispatch(),
        mem0.into_dispatch(),
        beta.into_dispatch(),
    ))
}
