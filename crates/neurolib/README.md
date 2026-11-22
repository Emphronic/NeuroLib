# NeuroLib

Spiking neural networks, quantized layers, fused kernels, a weight format and
tools for neural recordings in Rust, built on [Burn](https://burn.dev) 0.22.

NeuroLib is created and maintained by **Kostyanikov**, lead of the
[Emphronic](https://github.com/Emphronic) organization.

## Crates

| Module in `neurolib` | Crate | What it is |
|---|---|---|
| `snn` | `neo-snn` | Spiking layers on Burn: LIF/ALIF/Izhikevich/AdEx/Hodgkin-Huxley and more, surrogate gradients, STDP, spike attention, ANN to SNN conversion, structural plasticity, reservoir computing, telemetry |
| `nn` | `neo-nn` | Generic layers: quantized linear, MoE, transformer blocks, embeddings |
| `quant` | `neo-quantization` | GGUF Q4-Q8, K-quants, IQ series, NF4, ternary/TQ, INT1-INT8 |
| `synk` | `neo-synk` | Fused CubeCL kernels: quantized GEMM, leaky scan, spike/surrogate operations |
| `store` | `neo-store` | `.syn` weight format: network graph (NIR compatible), checksums, mmap, sharding; `synctl` CLI |
| `core`, `io`, `view`, `synapse` | `neo-core`, `neo-io`, `neo-view`, `neo-synapse` | Recordings: time and buffers, formats (npy/npz, Zarr, NWB, SpikeGLX), viewing, spike metrics |
| - | `neo-gui` | Workbench application (GPUI): Explore, Curation and Training |

`neurolib` is the single entry point: one dependency, everything else is enabled
through features (the same idea as the `burn` crate).

```toml
[dependencies]
neurolib = { version = "0.1", features = ["snn", "wgpu"] }
burn = "0.22"
```

Feature groups: `ml` (default), `signal`, `full`. Backends: `wgpu`, `cuda`,
`rocm` (plus `fusion`), `flex` for the CPU.

## Quick start

Burn 0.22 has no backend type parameter: the backend is chosen by the `Device`.

```rust
use burn::tensor::{Device, Distribution, Tensor};
use neurolib::snn::neuron::{LeakyConfig, MultiStepLeaky};
use neurolib::snn::surrogate::ATan;

let device = Device::flex().autodiff(); // CPU with autodiff; Device::wgpu(..) for GPU
let layer = MultiStepLeaky::new(&LeakyConfig::new(0.9), &device);
let x: Tensor<3> = Tensor::random([8, 50, 128], Distribution::Normal(0.5, 1.0), &device);
let spikes = layer.forward(x, &ATan::default()); // [batch, time, features]
```

Kernels are reachable from the spiking module; the device picks the
implementation (fused CubeCL kernel on wgpu/CUDA/ROCm, reference on the rest):

```rust
use neurolib::snn::leaky_scan_forward;
let u = leaky_scan_forward(input, mem0, beta);
```

## License

Copyright (c) 2026 Kostyanikov (Emphronic).

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))
- MIT license ([LICENSE-MIT](LICENSE-MIT))

at your option. The crates `neo-core`, `neo-nn`, `neo-quantization`, `neo-snn`
and `neo-synk` are licensed under the MIT license only, see their `Cargo.toml`.
`neo-quantization` contains portions derived from ggml / llama.cpp (MIT); see
[THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) and [NOTICE](NOTICE).

### Contribution

Unless you explicitly state otherwise, any contribution intentionally submitted
for inclusion in NeuroLib by you shall be licensed under the same terms as the
crate it is submitted to, without any additional terms or conditions.

## Citation

See [CITATION.cff](CITATION.cff).
