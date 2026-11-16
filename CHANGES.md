# Changelog

## 0.1.0

Initial public release.

- `neo-core`: exact time and sample rates, buffers, `RecordingSource`, out-of-core window streaming.
- `neo-io`: raw `.bin`, `.npy`/`.npz`, Zarr, NWB, SpikeGLX and mtscomp recording formats.
- `neo-view`: min/max envelopes and pyramids for multi-channel signal viewing.
- `neo-synapse`: correlograms, ISI, firing rate, snippets and PCA for spike curation.
- `neo-quantization`: GGUF Q4_0..Q8_1, K-quants Q2_K..Q8_K, the IQ series, NF4, ternary/TQ and INT1..INT8.
- `neo-synk`: fused CubeCL kernels for quantized GEMM, leaky scan and spike operations.
- `neo-snn`: LIF, Izhikevich, STDP, attention, structural plasticity and telemetry.
- `neo-nn`: quantized linear layers, MoE, transformer blocks and embeddings.
- `neo-store`: `.syn` weight container with graph metadata, CRC-32C, mmap and sharding.
- `neo-gui`: GPUI workbench for traces, heatmaps, curation and training.
- `neurolib`: facade crate that exposes the workspace through Cargo features.
