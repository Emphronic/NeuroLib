# neo-synk

Kernel library: fused CubeCL kernels for quantized GEMM and spiking neural network layers.
Designed to work with `burn` 0.22 backend extensions.

## What it provides

- **Quantized GEMM** — fused `y = x·Wᵀ` directly from packed weight packages (codes + scales + zero-points), without materializing full f32 matrices
- **Leaky scan** — fused recurrent leaky integrator forward/backward with manual autodiff
- **Spike layers** — fused LIF spike computation with surrogate gradients
- **CPU emulation** — all GPU kernels have CPU fallbacks for machines without CubeCL devices

## Features

| Feature | Description |
|---------|-------------|
| `flex` | Burn Flex CPU backend (default) |
| `wgpu` / `cuda` / `rocm` | GPU backend via burn-wgpu / burn-cuda / burn-rocm |
| `fusion` | Lazy kernel registration in burn Fusion |

## Usage

```rust
use neo_synk::quant::{quant_matmul_tensor, upload_weights};

// Quantized matmul directly from packed weights
let result = quant_matmul_tensor(&x, &weights, &device)?;
```

## Cargo

```toml
[dependencies]
neo-synk = { path = "crates/neo-synk", features = ["wgpu"] }
```

## License

MIT. See `LICENSE` for details.
