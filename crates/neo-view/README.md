# neo-view

Preparing multi-channel signals for viewing: min/max envelopes and pyramids for efficient
rendering of long recordings.

## What it provides

- **Envelopes** — per-window min/max envelopes for waveform overview rendering
- **Pyramids** — multi-resolution pyramids for zoomable time-axis rendering
- **Device support** — optional CubeCL backends (`wgpu`, `cuda`, `hip`, `cpu`) for GPU envelope/pyramid computation
- **Host-only default** — no GPU dependencies by default

## Features

| Feature | Description |
|---------|-------------|
| `device` | Enable CubeCL device envelopes |
| `wgpu` / `cuda` / `hip` / `cpu` | GPU/CPU backend for device envelopes |

## Usage

```rust
use neo_view::{envelope, pyramid};

// Compute min/max envelope
let env = envelope::compute(&samples, window_size)?;

// Build a pyramid for zoomable rendering
let pyr = pyramid::build(&samples, levels)?;
```

## Cargo

```toml
[dependencies]
neo-view = { path = "crates/neo-view", default-features = false, features = ["device", "wgpu"] }
```

## License

Dual-licensed under [MIT](LICENSE-MIT) or [Apache-2.0](LICENSE-APACHE).
