# Overview

Welcome to **The NeuroLib Book** 🧠 — the documentation for [NeuroLib](https://github.com/Kostyanikov/NeuroLib), the Rust
workspace for spiking neural networks (SNN), quantized neural network layers, weight storage and multi-channel signal
processing.

The book is written in the spirit of the Burn Book: follow the [Basic Workflow](./basic-workflow/README.md) to build a
first model, dive into [Building Blocks](./building-blocks/README.md) for the details, and use
[Advanced](./advanced/README.md) for the machinery around training and tools.

```text
neurolib/                     # Workspace root
├── Cargo.toml                # Workspace manifest
└── crates/
    ├── neurolib/             # Facade crate — one Cargo entry point, everything else via features
    ├── neo-core/             # Time, buffers, RecordingSource, out-of-core window streaming
    ├── neo-io/               # Recording formats: raw .bin, .npy/.npz, Zarr, NWB, SpikeGLX, mtscomp
    ├── neo-view/             # Min/max envelopes and pyramids for viewing multi-channel signals
    ├── neo-synapse/          # Spike-train metrics: correlograms, ISI, firing rate, snippets, PCA
    ├── neo-quantization/     # GGUF, K-quants, IQ, NF4, TQ quantization + runtime switching
    ├── neo-synk/             # Fused CubeCL kernels: quantized GEMM, leaky scan, spike ops
    ├── neo-snn/              # Spiking neural networks: LIF, Izhikevich, STDP, attention, plasticity
    ├── neo-nn/               # Conventional NN layers: quantized linear, MoE, transformers, embeddings
    ├── neo-store/            # .syn weight container (graph, CRC-32C, mmap, sharding)
    └── neo-gui/              # GPUI workbench: traces, heatmap, curation, training dashboard
```

Start with [Getting started](./getting-started.md).
