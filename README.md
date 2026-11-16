# NeuroLib

NeuroLib is a Rust workspace focused on spiking neural networks (SNN) and support quantized neural network layers,
weight storage, and multi-channel signal processing. It is built on top of [Burn 0.22]
(https://github.com/burn-rs/burn) and provides a modular, crate-by-crate architecture inspired by
Burn's own design.

## Workspace crates

| Crate | Purpose |
|-------|---------|
| [`neurolib`](crates/neurolib) | Facade crate — single entry point, all features behind Cargo features |
| [`neo-core`](crates/neo-core) | Time, buffers, `RecordingSource`, out-of-core window streaming, CubeCL compute layer |
| [`neo-io`](crates/neo-io) | Recording formats: raw `.bin`, `.npy`/`.npz`, Zarr, NWB, SpikeGLX, mtscomp |
| [`neo-view`](crates/neo-view) | Min/max envelopes and pyramids for viewing multi-channel signals |
| [`neo-synapse`](crates/neo-synapse) | Spike-train metrics: correlograms, ISI, firing rate, snippets, PCA |
| [`neo-quantization`](crates/neo-quantization) | Quantization formats: GGUF (Q4_0..Q8_1), K-quants (Q2_K..Q8_K), IQ series, NF4, TQ |
| [`neo-synk`](crates/neo-synk) | Fused CubeCL kernels: quantized GEMM, leaky scan, spike layers |
| [`neo-snn`](crates/neo-snn) | Spiking neural networks: LIF, Izhikevich, STDP, attention, structural plasticity |
| [`neo-nn`](crates/neo-nn) | Conventional NN layers: quantized linear, MoE, transformer blocks, embeddings |
| [`neo-gui`](crates/neo-gui) | GPUI-based workbench: traces, heatmap, channels, timeline, training dashboard |
| [`neo-store`](crates/neo-store) | `.syn` binary weight container with graph, CRC-32C, mmap, sharding |

## Features of the facade crate (`neurolib`)

```toml
[dependencies]
neurolib = { path = "crates/neurolib", features = ["full"] }
```

| Feature group | What it enables |
|---------------|-----------------|
| `ml` (default) | `neo-snn`, `neo-nn`, `neo-quantization`, `neo-synk`, `neo-store` |
| `signal` | `neo-core`, `neo-io`, `neo-view`, `neo-synapse`, `zarr`, `neuro` |
| `full` | `ml` + `signal` + `demo` |
| `wgpu` / `cuda` / `rocm` | GPU kernels in `neo-synk` + compute layer in `neo-core`/`neo-view` |
| `flex` | Burn CPU backend (convenient for tests) |
| `optim` / `train` / `dataset` / `vision` | Burn features forwarded through the dependency graph |

## Quick start

```rust
use neurolib::prelude::*;
use burn::tensor::Tensor;

// A simple LIF spiking layer
let device = Default::default();
let lif = LeakyConfig::new(128).init(&device);

// Quantized linear layer
let quantized = QuantizedLinearConfig::new(784, 128).init(&device);
```

Build and test:

```bash
cargo check --workspace
cargo test --workspace
cargo run -p neo-gui --release -- --synthetic 30s
```

## Project structure

```
neurolib/                     # Workspace root
├── Cargo.toml                # Workspace manifest
├── LICENSE-MIT               # MIT license
├── LICENSE-APACHE            # Apache-2.0 license
├── CHANGES.md                # Change log
└── crates/
    ├── neurolib/             # Facade crate
    ├── neo-core/             # Core primitives
    ├── neo-io/               # I/O and recording formats
    ├── neo-view/             # Signal viewing
    ├── neo-synapse/          # Spike metrics
    ├── neo-quantization/     # Quantization formats
    ├── neo-synk/             # Fused kernels
    ├── neo-snn/              # Spiking neural networks
    ├── neo-nn/               # Neural network layers
    ├── neo-gui/              # GUI workbench
    └── neo-store/            # Weight storage (.syn format)
```

## License

Dual-licensed under [MIT](LICENSE-MIT) or [Apache-2.0](LICENSE-APACHE),
unless noted otherwise in a specific crate directory.
