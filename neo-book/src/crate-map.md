# Crate map

The NeuroLib workspace (see the workspace root `Cargo.toml`):

| Crate | Purpose |
|---|---|
| [`neurolib`](../../crates/neurolib) | Facade crate — one Cargo entry point; everything else via Cargo features |
| [`neo-core`](../../crates/neo-core) | Time, buffers, `RecordingSource`, out-of-core window streaming, CubeCL compute layer |
| [`neo-io`](../../crates/neo-io) | Recording formats: raw `.bin`, `.npy`/`.npz`, Zarr, NWB, SpikeGLX, mtscomp |
| [`neo-view`](../../crates/neo-view) | Min/max envelopes and pyramids for viewing multi-channel signals |
| [`neo-synapse`](../../crates/neo-synapse) | Spike-train metrics: correlograms, ISI, firing rate, snippets, PCA |
| [`neo-quantization`](../../crates/neo-quantization) | Quantization formats: GGUF (Q4_0..Q8_1), K-quants (Q2_K..Q8_K), IQ series, NF4, TQ |
| [`neo-synk`](../../crates/neo-synk) | Fused CubeCL kernels: quantized GEMM, leaky scan, spike layers |
| [`neo-snn`](../../crates/neo-snn) | Spiking neural networks: LIF, Izhikevich, STDP, attention, structural plasticity |
| [`neo-nn`](../../crates/neo-nn) | Conventional NN layers: quantized linear, MoE, transformer blocks, embeddings |
| [`neo-gui`](../../crates/neo-gui) | GPUI workbench: traces, heatmap, channels, timeline, training dashboard |
| [`neo-store`](../../crates/neo-store) | `.syn` weight container: graph, CRC-32C, mmap, sharding |

## How the pieces fit

The facade crate `neurolib` re-exports each library crate as a module behind a Cargo feature:

| Module | Crate | Feature |
|---|---|---|
| `neurolib::snn` | `neo-snn` | `snn` |
| `neurolib::nn` | `neo-nn` | `nn` |
| `neurolib::quant` | `neo-quantization` | `quant` |
| `neurolib::synk` | `neo-synk` | `kernels` |
| `neurolib::store` | `neo-store` | `store` |
| `neurolib::core` | `neo-core` | `core` |
| `neurolib::io` | `neo-io` | `io` (with `zarr`/`neuro` features) |
| `neurolib::view` | `neo-view` | `view` |
| `neurolib::synapse` | `neo-synapse` | `synapse` |
| `neurolib::prelude` | — | everyday names of `neo-snn` |

Group features: `ml` (default — `snn`+`nn`+`quant`+`kernels`+`store`), `signal` (`core`+`io`+`view`+`synapse`+`zarr`+`neuro`),
`full` (`ml`+`signal`+`demo`). GPU features: `wgpu`/`cuda`/`rocm`, `fusion`, `cubecl-cpu`.

`burn` itself is **not** re-exported: declare it separately in the project (`burn = "0.22"`, as in the workspace).
