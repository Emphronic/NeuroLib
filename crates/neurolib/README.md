# neurolib

NeuroLib: a single facade crate for the workspace. Add one dependency and enable the parts you
need through Cargo features — an analogue of the `burn` crate for the NeuroLib ecosystem.

## What it provides

A unified entry point to all NeuroLib crates:
- **ML stack** — spiking neural networks (`neo-snn`), conventional layers (`neo-nn`),
  quantization (`neo-quantization`), fused kernels (`neo-synk`), weight storage (`neo-store`)
- **Signal stack** — core primitives (`neo-core`), I/O formats (`neo-io`),
  viewing (`neo-view`), spike metrics (`neo-synapse`)
- **GPU backends** — `wgpu`, `cuda`, `rocm` via `neo-synk`, `neo-core`, `neo-view`

## Usage

```toml
[dependencies]
burn = { version = "0.22.0", features = ["flex"] }
neurolib = { path = "crates/neurolib", features = ["snn"] }
```

```rust
use neurolib::prelude::*;       // neo-snn prelude
use burn::tensor::Tensor;       // burn is declared separately
use neurolib::nn::NnError;      // everything else via modules
```

## Features

| Feature | What it enables |
|---------|-----------------|
| `default` = `ml` | `snn`, `nn`, `quant`, `kernels`, `store` |
| `signal` | `core`, `io`, `view`, `synapse`, `zarr`, `neuro` |
| `full` | `ml` + `signal` + `demo` (without GPU) |
| `snn`, `nn`, `quant`, `kernels`, `store` | `neo-snn`, `neo-nn`, `neo-quantization`, `neo-synk`, `neo-store` |
| `core`, `io`, `view`, `synapse` | `neo-core`, `neo-io`, `neo-view`, `neo-synapse` |
| `zarr`, `neuro` | Zarr/NWB and SpikeGLX/mtscomp formats in `neo-io` |
| `demo` | `neo_snn::demo` (burn/optim) |
| `optim`, `train`, `dataset`, `vision`, `flex` | Burn features of the same name (forwarded) |
| `fusion` | Registers `neo-synk` kernels in Fusion (enabled by GPU backends) |
| `wgpu`, `cuda`, `rocm` | GPU kernels in `neo-synk` + compute layer in `neo-core`/`neo-view` |
| `cubecl-cpu` | CubeCL CPU device for `neo-core`/`neo-view` |

## Assembly

```bash
cargo build -p neurolib                                   # ml, CPU only
cargo build -p neurolib --features wgpu,fusion            # ml + GPU
cargo build -p neurolib --no-default-features --features signal
cargo test  -p neurolib --features full,flex              # verification of re-exports
```

`neo-gui` is a binary crate, not included in `neurolib`.

## License

Dual-licensed under [MIT](LICENSE-MIT) or [Apache-2.0](LICENSE-APACHE).
