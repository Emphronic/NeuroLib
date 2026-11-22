# Backend (ComputeTarget)

`ComputeTarget` is the one name for a runtime across the workspace: the CLI flag, app setting, `NEO_RUNTIME`
environment variable all resolve to it. Choosing a target needs no GPU libraries; running work on it needs the
`compute` feature of `neo-core`.

```rust
use neurolib::core::ComputeTarget;

ComputeTarget::ALL;             // [Wgpu, Cuda, Hip, Cpu]
ComputeTarget::parse("rocm");   // alias of Hip
let target = ComputeTarget::from_env()?;  // NEO_RUNTIME, else first compiled-in target
let target = target.checked()?;           // error if not compiled in
target.is_available();
ComputeTarget::available();
```

| Target | Runtime | Features needed |
|---|---|---|
| `Wgpu` | WebGPU: Vulkan / Metal / DX12 | `wgpu` |
| `Cpu` | Multi-threaded CPU (MLIR JIT) | `cpu` (or the `cubecl-cpu` feature of `neo-core`) |
| `Cuda` | NVIDIA CUDA | `cuda` |
| `Hip` | AMD ROCm | `rocm` |

GPU features (`wgpu`, `cuda`, `rocm`) pull `fusion` — Burn's Tensor backend is not parameterized in Burn 0.22, so
kernels plug in via `#[backend_extension]` and the `Device` chooses "GPU kernel or CPU".

The accelerated kernels are in `neo-synk` (see [Spikes & encoding](./spikes.md) for `leaky_scan_forward` and
[Quantization](./quantization.md) for `quant_matmul_tensor`). The `ComputeTarget`-based device is used by
`neo-core`/`neo-view`/`neo-synk` for DSP kernels.
