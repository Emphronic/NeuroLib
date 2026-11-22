# neo-synk

Kernel library: fused CubeCL kernels for NeuroLib — quantized GEMM and spiking neural network
layers. Designed to work with `burn` 0.22 backend extensions (`#[backend_extension]`); `Device`
chooses between "GPU kernel or CPU".

## What it provides

| Module | Contents |
|---|---|
| [`quant`](./src/quant) | Fused quantized GEMM — `y = x·Wᵀ` directly from packed weight packages (codes + scales + zero-points), without materializing full f32 matrices (`quant_matmul_tensor`, `upload_weights`, `kernel_kind`, ...) |
| [`snn`](./src/snn) | Fused spike layers — `leaky_scan_forward` (fused kernel on GPU, `snn::leaky_scan_reference` elsewhere; `snn::gpu::leaky_scan_fused`) |

## Features

| Feature | Description |
|---|---|
| `flex` (default) | Burn Flex CPU backend |
| `wgpu` / `cuda` / `rocm` | GPU backends via burn-wgpu / burn-cuda / burn-rocm |
| `fusion` | Lazy kernel registration in burn Fusion |

## Usage

```rust
use neurolib::synk::quant::{quant_matmul_tensor, upload_weights};

// Quantized matmul directly from packed weights
let result = quant_matmul_tensor(&x, &weights, &device)?;
```

## Examples

- `cargo run --release -p neo-synk --features wgpu,fusion --example quant_gpu_bench`
- `cargo run --release -p neo-synk --features wgpu,fusion --example leaky_scan_bench`

## Book

[Quantization](../../neo-book/src/building-blocks/quantization.md),
[Backend](../../neo-book/src/building-blocks/backend.md)

