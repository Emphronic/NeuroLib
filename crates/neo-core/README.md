# neo-core

Foundational primitives for signal recordings: exact time and sample rates, buffers, errors,
the `RecordingSource` contract, out-of-core window streaming, and an optional CubeCL compute
layer.

## What it provides

- **Time and sample rates** — type-safe time representations with exact sample-rate handling
- **Buffers** — contiguous and chunked sample buffers for multi-channel recordings
- **RecordingSource** — trait for opening and listing recording sources (files, streams, memory)
- **Window streaming** — out-of-core windowed access to large recordings without loading everything into RAM
- **Compute layer** — optional CubeCL backend (`wgpu`, `cuda`, `hip`, `cpu`) for GPU-accelerated operations
- **Errors** — `ComputeError`, `IoError`, `UnsupportedFormatError`, and typed result aliases

## Features

| Feature | Description |
|---------|-------------|
| `compute` | Enable CubeCL compute layer (off by default) |
| `wgpu` / `cuda` / `cuda` / `hip` | GPU backend for the compute layer |
| `cpu` | CPU CubeCL device |

## Usage

```rust
use neo_core::{RecordingSource, MemoryRecording};

// Create an in-memory recording
let rec = MemoryRecording::new(f32_data, num_channels, sample_rate);

// Open a source from a file
let source = neo_core::open_source("recording.bin")?;
```

## Cargo

```toml
[dependencies]
neo-core = { path = "crates/neo-core" }
# or, within the NeuroLib workspace:
neo-core = { path = "../neo-core" }
```

Enable GPU compute:

```toml
neo-core = { path = "../neo-core", features = ["compute", "wgpu"] }
```

## License

MIT. See [LICENSE](LICENSE) for details.
