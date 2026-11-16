//! GPU quantized GEMM on CubeCL 0.11 / Burn 0.22: `y = x·Wᵀ` directly from packaged codes.
//!
//! Two cores ([`qgemm_kernel`], [`qgemm_tiled_kernel`]) and one implementation
//! [`QuantMatmulBackend`] for `CubeBackend` cover all runtimes that Burn
//! collects via `burn-cubecl`: wgpu (Vulkan/Metal/DX12), CUDA, ROCm.
//!
//! In Burn 0.22 the backend is not visible in the tensor type, so the "weights on the device" are
//! these are the usual tensors Burn ([`QuantWeights`](super::QuantWeights)): codes
//! (`u32`-words), scales, offsets and LUT. They are loaded onto the
//! device once, and the operation is connected as a backend extension
//! (`#[backend_extension]`).
//!
//! # Scheme
//!
//! One workgroup ([`QGEMM_THREADS`] threads) computes one output column `r`
//! (a row of weights) for a tile of `mt` rows of activations (`mt` = 1, 4 or 8 by
//! `m`). The threads split the weight row element by element with step `QGEMM_THREADS`:
//! neighboring threads read adjacent bits of the code stream (coalesced access).
//! Each code is unpacked **once** and immediately applied to all `mt`
//! rows `x` tiles; partial sums are reduced by the tree into shared memory.
//!
//! ```text
//! w = scale_g (a raw + offset_g) // affine mode
//! w = scale_g · lut[raw] // LUT mode (NF4, ternary)
//! acc_i += w x[i, c] // i is the tile row
//! ```
//!
//! The weight matrix in f32 is not created anywhere: only the
//! `bits`-bit codes and one `scale` (and `offset`) per group.
//!
//! The code layout matches the CPU emulator
//! ([`emulate_gpu_matmul`](super::emulate_gpu_matmul)), covered with tests;
//! the order of summation is different (the results coincide within
//! rounding errors f32).
//!
//! Inference only: the gradient along `x` is not implemented.

use burn::backend::{
    Backend, backend_extension,
    cubecl::dtype_to_storage_type,
    tensor::{FloatTensor, IntTensor},
};
use burn::tensor::Shape;
use burn_cubecl::{CubeBackend, kernel::into_contiguous, tensor::CubeTensor};
use cubecl::{CubeCount, CubeDim, cube, prelude::*};

/// Scalar parameters of the quantized matrix (backend-independent).
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct QuantParams {
    /// Bit per code (1..=8).
    pub bits: u32,
    /// LUT mode (NF4, ternary) instead of affine.
    pub use_lut: bool,
    /// Affine mapping multiplier.
    pub a: f32,
    /// Group size by flat weight index (`>= 1`).
    pub group_size: u32,
    /// Number of weight rows (output columns).
    pub n: u32,
    /// Input size.
    pub k: u32,
}

/// Output shape `[m, n]`; checks the shape `x`.
pub fn quant_out_shape(x: &Shape, params: &QuantParams) -> Shape {
    assert_eq!(
        x.num_dims(),
        2,
        "quant_matmul: x must be rank 2, got {:?}",
        x
    );
    assert_eq!(
        x[1] as u32, params.k,
        "quant_matmul: x has {} columns, weights expect {}",
        x[1], params.k
    );
    Shape::new([x[0], params.n as usize])
}

/// Backend that can calculate `y = x·Wᵀ` using packed weights **on the device**.
///
/// * `x` — `[m, k]`, float;
/// * `words` - codes as `u32`-words (+1 word of reserve), `scales`/`offsets` - by
///   group, `lut` - table `raw → value`.
#[backend_extension(Cube, Fusion: cfg(feature = "fusion"))]
pub trait QuantMatmulBackend: Backend {
    #[fusion(dtype = x, shape = quant_out_shape(x, params))]
    fn quant_matmul(
        x: FloatTensor<Self>,
        words: IntTensor<Self>,
        scales: FloatTensor<Self>,
        offsets: FloatTensor<Self>,
        lut: FloatTensor<Self>,
        params: QuantParams,
    ) -> FloatTensor<Self>;
}

/// Number of threads in the workgroup (power of two: reduction proceeds as a tree).
pub const QGEMM_THREADS: u32 = 64;

/// From which `m` the tile core is turned on (for smaller ones - the GEMV core).
const TILED_MIN_M: usize = 32;

/// Row tile `x` on workgroup for given `m`.
fn row_tile(m: usize) -> u32 {
    if m <= 1 {
        1
    } else if m <= 4 {
        4
    } else {
        8
    }
}

/// Kernel quantized GEMM. The codes are read as a `bits`-bit stream of `u32`-words.
///
/// * `x` - `[m, k]`, `out` - `[m, n]`, both row-major;
/// * `words` - codes (+1 word of reserve), `scales`/`offsets` - by group;
/// * `lut` - table `raw → value` (used with `use_lut`);
/// * `a` — affine scheme multiplier;
/// * `group_size` — group size by flat weight index (`>= 1`);
/// * `mt` — number of `x` rows per workgroup, `threads` — threads in the workgroup
///   (must match `CubeDim`).
///
/// The workgroup index `cube` is decomposed as `r = cube % n`,
/// `tile = cube / n`: adjacent workgroups take adjacent rows of weights when
/// the same tile `x`.
#[cube(launch)]
pub fn qgemm_kernel<F: Float>(
    x: &Tensor<F>,
    words: &Tensor<u32>,
    scales: &Tensor<f32>,
    offsets: &Tensor<f32>,
    lut: &Tensor<f32>,
    out: &mut Tensor<F>,
    m: u32,
    n: u32,
    k: u32,
    group_size: u32,
    a: f32,
    #[comptime] bits: u32,
    #[comptime] use_lut: bool,
    #[comptime] mt: u32,
    #[comptime] threads: u32,
    #[define(F)] _dtype: ElemType,
) {
    let t = comptime!(threads as usize);
    let mtile = comptime!(mt as usize);
    let tid = UNIT_POS_X as usize;

    let n_tiles = (m + mt - 1u32) / mt;
    let total = n * n_tiles;
    // Extra workgroup (grid rounding up) computes the last tile
    // and does not write anything: all flows must reach `sync_cube`.
    let cube_raw = u32::cast_from(CUBE_POS);
    let valid = cube_raw < total;
    let cube = if valid { cube_raw } else { total - 1u32 };

    let r = cube % n; // weights row = output column
    let row0 = (cube / n) * mt; // first row x tiles

    let m_us = usize::cast_from(m);
    let n_us = usize::cast_from(n);
    let k_us = usize::cast_from(k);
    let r_us = usize::cast_from(r);
    let row0_us = usize::cast_from(row0);

    let mask = comptime!((1u32 << bits) - 1u32);

    let mut acc = Array::<f32>::new(mtile);
    #[unroll]
    for i in 0..mtile {
        acc[i] = 0.0f32;
    }

    let row_base = r * k;
    let mut c = UNIT_POS_X;
    while c < k {
        let flat = row_base + c;
        let g = flat / group_size;

        // Code: `bits` bit from position `flat·bits` (possibly across a word boundary).
        let bitpos = flat * bits;
        let wi = bitpos / 32u32;
        let sh = bitpos % 32u32;
        let mut v = words[usize::cast_from(wi)] >> sh;
        if sh + bits > 32u32 {
            v = v | (words[usize::cast_from(wi + 1u32)] << (32u32 - sh));
        }
        let raw = v & mask;

        let scale = scales[usize::cast_from(g)];
        let wv = if use_lut {
            scale * lut[usize::cast_from(raw)]
        } else {
            scale * (a * f32::cast_from(raw) + offsets[usize::cast_from(g)])
        };

        let c_us = usize::cast_from(c);
        #[unroll]
        for i in 0..mtile {
            let row = row0_us + i;
            if row < m_us {
                let xv = f32::cast_from(x[row * k_us + c_us]);
                acc[i] = acc[i] + wv * xv;
            }
        }

        c += threads;
    }

    // Reduction by workgroup threads (tree in shared memory).
    let mut red = Shared::<[f32]>::new_slice(t * mtile);
    #[unroll]
    for i in 0..mtile {
        red[i * t + tid] = acc[i];
    }
    sync_cube();

    let mut stride = usize::cast_from(threads / 2u32);
    while stride > 0usize {
        if tid < stride {
            #[unroll]
            for i in 0..mtile {
                red[i * t + tid] = red[i * t + tid] + red[i * t + tid + stride];
            }
        }
        sync_cube();
        stride = stride / 2usize;
    }

    if tid == 0usize {
        if valid {
            #[unroll]
            for i in 0..mtile {
                let row = row0_us + i;
                if row < m_us {
                    out[row * n_us + r_us] = F::cast_from(red[i * t]);
                }
            }
        }
    }
}

/// Threads in the workgroup of the tile core (16×16).
pub const QTILE_THREADS: u32 = 256;
/// Output tile: 64 rows `x` × 64 columns `W` per workgroup.
pub const QTILE_M: u32 = 64;
/// Output tile by columns `W`.
pub const QTILE_N: u32 = 64;

/// Tile quantized GEMM for large `m` (prefill/batch).
///
/// A workgroup of 256 threads computes the output tile `64×64`. Walks along `k` in steps
/// 16 each: the `x` tile and the **unpacked** scales tile are placed in shared memory,
/// then each thread reads its `4×4` block of registers. Every weight
/// unpacks once per `m/64` tiles instead of once per every 8
/// rows `x`, and reads of weights and `x` are coalesced.
///
/// Workgroup index: `tile_n = cube % n_tiles_n`, `tile_m = cube / n_tiles_n`.
#[cube(launch)]
pub fn qgemm_tiled_kernel<F: Float>(
    x: &Tensor<F>,
    words: &Tensor<u32>,
    scales: &Tensor<f32>,
    offsets: &Tensor<f32>,
    lut: &Tensor<f32>,
    out: &mut Tensor<F>,
    m: u32,
    n: u32,
    k: u32,
    group_size: u32,
    a: f32,
    #[comptime] bits: u32,
    #[comptime] use_lut: bool,
    #[define(F)] _dtype: ElemType,
) {
    let tid = UNIT_POS_X;
    let tx = tid % 16u32; // column inside tile (step 16)
    let ty = tid / 16u32; // row inside tile (step 16)

    let n_tiles_n = (n + 63u32) / 64u32;
    let n_tiles_m = (m + 63u32) / 64u32;
    let total = n_tiles_n * n_tiles_m;
    // Extra workgroup (grid rounding) covers the last tile and does not write:
    // All flows must reach `sync_cube`.
    let cube_raw = u32::cast_from(CUBE_POS);
    let valid = cube_raw < total;
    let cube = if valid { cube_raw } else { total - 1u32 };
    let row0 = (cube / n_tiles_n) * 64u32;
    let col0 = (cube % n_tiles_n) * 64u32;

    let mask = comptime!((1u32 << bits) - 1u32);

    let mut xs = Shared::<[f32]>::new_slice(1024usize); // [kk][ml], 16×64
    let mut ws = Shared::<[f32]>::new_slice(1024usize); // [kk][nl], 16×64

    let mut acc = Array::<f32>::new(16usize);
    #[unroll]
    for i in 0..16usize {
        acc[i] = 0.0f32;
    }

    let mut k0 = 0u32;
    while k0 < k {
        // ── tile loading: 1024 elements / 256 threads = 4 per thread ──
        #[unroll]
        for s in 0..4u32 {
            let idx = tid + s * 256u32;
            let l = idx / 16u32; // tile row (m or n)
            let kk = idx % 16u32;
            let c = k0 + kk;

            // x
            let row = row0 + l;
            let mut xv = 0.0f32;
            if row < m && c < k {
                xv = f32::cast_from(x[usize::cast_from(row * k + c)]);
            }
            xs[usize::cast_from(kk * 64u32 + l)] = xv;

            // weight
            let r = col0 + l;
            let mut wv = 0.0f32;
            if r < n && c < k {
                let flat = r * k + c;
                let g = flat / group_size;
                let bitpos = flat * bits;
                let wi = bitpos / 32u32;
                let sh = bitpos % 32u32;
                let mut v = words[usize::cast_from(wi)] >> sh;
                if sh + bits > 32u32 {
                    v = v | (words[usize::cast_from(wi + 1u32)] << (32u32 - sh));
                }
                let raw = v & mask;
                let scale = scales[usize::cast_from(g)];
                if use_lut {
                    wv = scale * lut[usize::cast_from(raw)];
                } else {
                    wv = scale * (a * f32::cast_from(raw) + offsets[usize::cast_from(g)]);
                }
            }
            ws[usize::cast_from(kk * 64u32 + l)] = wv;
        }
        sync_cube();

        // ── FMA 16 k steps, 4×4 block per stream ──
        #[unroll]
        for kk in 0..16u32 {
            let base = kk * 64u32;
            let mut xa = Array::<f32>::new(4usize);
            let mut wb = Array::<f32>::new(4usize);
            #[unroll]
            for i in 0..4u32 {
                xa[i as usize] = xs[usize::cast_from(base + ty + i * 16u32)];
                wb[i as usize] = ws[usize::cast_from(base + tx + i * 16u32)];
            }
            #[unroll]
            for i in 0..4usize {
                #[unroll]
                for j in 0..4usize {
                    acc[i * 4usize + j] = acc[i * 4usize + j] + xa[i] * wb[j];
                }
            }
        }
        sync_cube();

        k0 += 16u32;
    }

    if valid {
        #[unroll]
        for i in 0..4u32 {
            let row = row0 + ty + i * 16u32;
            #[unroll]
            for j in 0..4u32 {
                let col = col0 + tx + j * 16u32;
                if row < m && col < n {
                    out[usize::cast_from(row * n + col)] =
                        F::cast_from(acc[(i * 4u32 + j) as usize]);
                }
            }
        }
    }
}

impl QuantMatmulBackend for CubeBackend {
    fn quant_matmul(
        x: FloatTensor<Self>,
        words: IntTensor<Self>,
        scales: FloatTensor<Self>,
        offsets: FloatTensor<Self>,
        lut: FloatTensor<Self>,
        p: QuantParams,
    ) -> FloatTensor<Self> {
        x.assert_is_on_same_device(&words);
        x.assert_is_on_same_device(&scales);
        x.assert_is_on_same_device(&offsets);
        x.assert_is_on_same_device(&lut);

        let dtype = x.dtype;
        let x = into_contiguous(x);
        let words = into_contiguous(words);
        let scales = into_contiguous(scales);
        let offsets = into_contiguous(offsets);
        let lut = into_contiguous(lut);

        let shape_out = quant_out_shape(x.meta.shape(), &p);
        let (m, n) = (shape_out[0], shape_out[1]);
        let buffer = x.client.empty(shape_out.num_elements() * dtype.size());
        let out = CubeTensor::new_contiguous(
            x.client.clone(),
            x.device.clone(),
            shape_out,
            buffer,
            dtype,
        );
        if m == 0 || n == 0 {
            return out;
        }
        let client = x.client.clone();

        if m >= TILED_MIN_M {
            // Prefill/batch: tiled core with unpacking weights into shared memory.
            let total_cubes = n.div_ceil(QTILE_N as usize) * m.div_ceil(QTILE_M as usize);
            let cube_dim = CubeDim::new_1d(QTILE_THREADS);
            let cube_count = CubeCount::Static(total_cubes as u32, 1, 1);
            qgemm_tiled_kernel::launch(
                &client,
                cube_count,
                cube_dim,
                x.into_tensor_arg(),
                words.into_tensor_arg(),
                scales.into_tensor_arg(),
                offsets.into_tensor_arg(),
                lut.into_tensor_arg(),
                out.clone().into_tensor_arg(),
                m as u32,
                p.n,
                p.k,
                p.group_size,
                p.a,
                p.bits,
                p.use_lut,
                dtype_to_storage_type(dtype),
            );
            return out;
        }

        let mt = row_tile(m);
        let total_cubes = n * m.div_ceil(mt as usize);
        let cube_dim = CubeDim::new_1d(QGEMM_THREADS);
        let cube_count = CubeCount::Static(total_cubes as u32, 1, 1);
        qgemm_kernel::launch(
            &client,
            cube_count,
            cube_dim,
            x.into_tensor_arg(),
            words.into_tensor_arg(),
            scales.into_tensor_arg(),
            offsets.into_tensor_arg(),
            lut.into_tensor_arg(),
            out.clone().into_tensor_arg(),
            m as u32,
            p.n,
            p.k,
            p.group_size,
            p.a,
            p.bits,
            p.use_lut,
            mt,
            QGEMM_THREADS,
            dtype_to_storage_type(dtype),
        );
        out
    }
}
