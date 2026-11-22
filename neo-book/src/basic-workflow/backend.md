# Backend

The compute runtime is one name across the workspace: [`ComputeTarget`](../building-blocks/backend.md):

- `Wgpu` — WebGPU: Vulkan / Metal / DX12;
- `Cpu` — multi-threaded CPU (MLIR JIT);
- `Cuda` — NVIDIA;
- `Hip` — AMD ROCm.

```rust
use neurolib::core::ComputeTarget;

// From NEO_RUNTIME env var (default: first compiled-in target)
let target = ComputeTarget::from_env()?;
// Parse a name: accepts wgpu / cpu / cuda / hip and aliases (webgpu, vulkan, metal, gpu, rocm)
let target = ComputeTarget::parse("cuda")?;
let target = target.checked()?; // error if not compiled in

target.is_available();     // which runtimes can run here
ComputeTarget::available();
```

The accelerated kernels live in `neo-synk` (leaky scan, quantized GEMM, spike ops) — see
[Building Blocks → Backend](../building-blocks/backend.md) for the details and the `fusion` feature.
