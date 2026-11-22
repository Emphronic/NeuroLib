# neo-core

Foundational primitives for NeuroLib recordings: exact time and sample rates, buffers, the
`RecordingSource` contract, out-of-core window streaming, and an optional CubeCL compute layer.

## What it provides

| Module | Contents |
|---|---|
| [`time`](./src/time) | `RationalTime`, `SampleRate`, `TimeRange` — exact time and sample-rate handling |
| [`buffer`](./src/buffer) | `SignalChunk`, `BufferLayout`, `MemoryOrder` — chunked sample buffers |
| [`recording`](./src/recording) | `RecordingSource`, `MemoryRecording`, `RecordingInfo`, `ChannelInfo`, `SampleFormat`, `SignalUnit` (`MICROVOLTS_PER_VOLT`), `SlicedRecording` |
| [`window`](./src/window) | `WindowLoader`, `HaloWindow`, `ChunkSchedule` — out-of-core windowed access to large recordings |
| [`device`](./src/device.rs) | `ComputeTarget` — one name for the runtime across the workspace: `wgpu`, `cuda`, `hip`, `cpu` (env var `NEO_RUNTIME`) |
| [`compute`](./src/compute) (feature `compute`) | CubeCL backend — launching kernels on the target device |
| [`mask`](./src/mask) | `ChannelMask` — which channels are active |
| [`progress`](./src/progress.rs) | `ProgressSink`, `NoProgress`, `ProgressEvent` |
| [`error`](./src/error.rs) | `DspError`, `DspResult` |

## Features

| Feature | Description |
|---|---|
| `compute` | Enable the CubeCL compute layer |
| `wgpu` | WebGPU backend |
| `cuda` | NVIDIA CUDA backend |
| `hip` | AMD ROCm backend |
| `cpu` | CubeCL CPU device |

## Usage

```rust
use neurolib::core::{ComputeTarget, MemoryRecording};

// In-memory recording (f32 samples, channel-major)
let rec = MemoryRecording::new("recording", data, num_channels, sample_rate_hz)?;

// Which compute runtime is available
let target = ComputeTarget::from_env()?;
```

## Cargo

```toml
[dependencies]
neo-core = { path = "crates/neo-core", features = ["compute"] }
```

Book: [compute target](../../neo-book/src/building-blocks/backend.md),
[viewing](../../neo-book/src/building-blocks/viewing.md).



