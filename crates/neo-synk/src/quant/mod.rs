//! Fused quantized GEMM.
//!
//! `y = x · Wᵀ`, where `W` is the packed matrix `[out, in]`
//! ([`neo_quantization::PackedBuffer`]), and `x` are activations of `[m, in]` in f32.
//!
//! The main property: **the complete weight matrix is never created in f32**.
//! The kernel follows the rows of `W`, unpacks a small piece (≤ 1024 weights,
//! which fits in L1) and immediately collapses it with all rows of `x`.
//!
//! * [`cpu`] - implementation for CPU (rayon): three levels, see [`cpu::KernelKind`].
//! * [`gpu`] (features `wgpu` / `cuda` / `rocm`) - CubeCL 0.11 kernel for GPU;
//!   [`quant_matmul_tensor`] itself chooses the GPU core or CPU emulator according to `Device`.
//!   The GPU path works and is tested on wgpu: the result matches the CPU, and the kernel is faster
//!   `packed_matmul` (see `examples/quant_gpu_bench.rs`).

pub mod cpu;
pub mod layout;

#[cfg(any(feature = "wgpu", feature = "cuda", feature = "rocm"))]
pub mod gpu;

pub use layout::{GpuQuantLayout, emulate_gpu_matmul, gpu_layout};

pub use cpu::{KernelKind, QuantGemmError, dequant_matmul_reference, kernel_kind, packed_matmul};

#[cfg(any(feature = "wgpu", feature = "cuda", feature = "rocm"))]
pub use gpu::{QuantMatmulBackend, QuantParams, quant_out_shape};

use burn::tensor::{Device, Tensor, TensorData};
#[cfg(any(feature = "wgpu", feature = "cuda", feature = "rocm"))]
use burn::tensor::{Int, TensorCreationOptions};
use std::sync::Arc;

/// Quantized weights ready for `y = x·Wᵀ` on the selected device.
///
/// On CubeCL device (wgpu/CUDA/ROCm) codes, scales, offsets and LUTs
/// lie as tensors on this device; the clone is cheap. On other devices
/// (Flex CPU, ...) only the host layout for the CPU emulator is stored.
#[derive(Clone, Debug)]
pub struct QuantWeights {
    n: u32,
    k: u32,
    repr: WeightsRepr,
}

#[derive(Clone, Debug)]
enum WeightsRepr {
    #[cfg(any(feature = "wgpu", feature = "cuda", feature = "rocm"))]
    Device {
        words: Tensor<1, Int>,
        scales: Tensor<1>,
        offsets: Tensor<1>,
        lut: Tensor<1>,
        params: QuantParams,
    },
    Host(Arc<GpuQuantLayout>),
}

impl QuantWeights {
    /// Number of weight rows (output columns).
    pub fn n(&self) -> usize {
        self.n as usize
    }
    /// Input size.
    pub fn k(&self) -> usize {
        self.k as usize
    }
}

/// Load the weight layout onto `device` (once; then used by
/// [`quant_matmul_tensor`]).
pub fn upload_weights(l: &GpuQuantLayout, device: &Device) -> QuantWeights {
    #[cfg(any(feature = "wgpu", feature = "cuda", feature = "rocm"))]
    if device.identity().is_some() {
        let f = |v: &[f32]| Tensor::<1>::from_data(TensorData::new(v.to_vec(), [v.len()]), device);
        let words = Tensor::<1, Int>::from_data(
            TensorData::new(l.words.clone(), [l.words.len()]),
            TensorCreationOptions::new(device.clone()).with_dtype(burn::tensor::DType::U32),
        );
        return QuantWeights {
            n: l.n,
            k: l.k,
            repr: WeightsRepr::Device {
                words,
                scales: f(&l.scales),
                offsets: f(&l.offsets),
                lut: f(&l.lut),
                params: QuantParams {
                    bits: l.bits,
                    use_lut: l.use_lut,
                    a: l.a,
                    group_size: l.group_size,
                    n: l.n,
                    k: l.k,
                },
            },
        };
    }
    let _ = device;
    QuantWeights {
        n: l.n,
        k: l.k,
        repr: WeightsRepr::Host(Arc::new(l.clone())),
    }
}

/// `y = x·Wᵀ` by quantized weights. `x` - `[m, k]`.
///
/// GPU device → fused core CubeCL; otherwise - a CPU emulator of the same layout.
/// Inference only (gradient by `x` is not supported).
pub fn quant_matmul_tensor(x: Tensor<2>, w: &QuantWeights) -> Tensor<2> {
    match &w.repr {
        #[cfg(any(feature = "wgpu", feature = "cuda", feature = "rocm"))]
        WeightsRepr::Device {
            words,
            scales,
            offsets,
            lut,
            params,
        } => {
            use burn::backend::Dispatch;
            assert!(
                !x.device().is_autodiff(),
                "quant_matmul_tensor: inference operation, use non-autodiff device"
            );
            Tensor::from_dispatch(Dispatch::quant_matmul(
                x.into_dispatch(),
                words.clone().into_dispatch(),
                scales.clone().into_dispatch(),
                offsets.clone().into_dispatch(),
                lut.clone().into_dispatch(),
                *params,
            ))
        }
        WeightsRepr::Host(l) => {
            let device = x.device();
            let [m, k] = x.dims();
            assert_eq!(
                k as u32, l.k,
                "quant_matmul: x has {k} columns, weights expect {}",
                l.k
            );
            let xs = x.into_data().try_to_vec::<f32>().expect("f32 activations");
            let y = emulate_gpu_matmul(l, &xs, m);
            Tensor::<2>::from_data(TensorData::new(y, [m, l.n as usize]), &device)
        }
    }
}

#[cfg(test)]
mod tests;
