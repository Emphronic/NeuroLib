//! # neo-synk
//!
//! Low-level kernels for NeuroLib. Separate crate from
//! `neo-snn` is intentional: `neo-snn` contains layers independent of
//! devices, `neo-synk` - accelerated implementations for CubeCL devices
//! (wgpu / CUDA / ROCm; features `wgpu`, `cuda`, `rocm`); the GPU path works and is
//! tested by benchmarks on wgpu (`examples/quant_gpu_bench`, `examples/leaky_scan_bench`). Burn 0.22 does not
//! parameterize the `Tensor` backend: kernels are connected via
//! `#[backend_extension]`, and `Device` chooses between "GPU kernel or CPU".
//!
//! ## Modules
//!
//! * [`snn`] - kernels for spike layers `neo-snn`: `snn::leaky_scan_forward`
//!   (fused kernel on GPU, `snn::leaky_scan_reference` on the remaining
//!   devices; GPU part - `snn::gpu`).
//! * [`quant`] - fused quantized GEMM: `y = x·Wᵀ` directly from packed weights
//!   (`neo-quantization::PackedBuffer`), without matrix expansion at f32.
//!   `quant::upload_weights` + `quant::quant_matmul_tensor`.
pub mod quant;
pub mod snn;
