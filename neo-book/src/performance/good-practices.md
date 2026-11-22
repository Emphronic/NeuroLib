# Good practices

Writing models and training code that make the most of the hardware. The accelerated paths are the
`neo-synk` kernels on CubeCL devices (`wgpu` / `cuda` / `rocm`): fused quantized GEMM
(`quant::upload_weights` + `quant::quant_matmul_tensor`), the fused `leaky_scan_forward` kernel and spike-layer ops
(`snn::gpu`); Burn 0.22 does not parameterize the `Tensor` backend, kernels connect via `#[backend_extension]`,
and the `Device` chooses between "GPU kernel or CPU" (`ComputeTarget`, see
[Backend](../building-blocks/backend.md)).

- **Async execution**: CubeCL schedules work asynchronously; read/write boundaries (`into_data`/`from_data`,
  host copies) are the synchronization points — keep them few and batched.
- **Kernel fusion**: adjacent element-wise ops fuse on the device — write models in composable small ops,
  not precomputed tensors (the CPU/GPU split is decided by the backend, not by hand).
- **Kernel selection**: keep tensors on-device (`Tensor` on the target `Device`); move data to host only for
  data-dependent control flow.
- **Use `fused` where it pays off** — see [Quantization](../building-blocks/quantization.md): `Fused` runs on the
  host and is free on CPU backends, but `Dequantize` or the GPU cores are better on GPU.
- **STDP layers are host-side**: `StdpLinear` updates run per step on host tensors — do not put STDP loops in the
  training hot path of large models (see [STDP](../building-blocks/stdp.md)).
- **Avoid GPU sync per step**: the fused reset path exists precisely to avoid reading the threshold to the host
  every step (`apply_reset_tensor` vs `apply_reset`).

The workspace keeps track of where work happens in `neo-core` — `ComputeTarget::from_env()` reads `NEO_RUNTIME`,
so one code path serves CLI, app and tests.
