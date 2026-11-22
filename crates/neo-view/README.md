# neo-view

Preparing multi-channel signals for viewing: min/max envelopes and pyramids for efficient
rendering of long recordings. Host-only by default; optional CubeCL backends.

## What it provides

| Module | Contents |
|---|---|
| [`envelope`](./src/envelope) | min/max reduction — `min_max_decimate(samples, target_buckets)` — `[min, max]` per column keeps every spike peak visible at any zoom; `envelope_on_device` (feature `device`) |
| [`pyramid`](./src/pyramid) | `Pyramid`, `PyramidBuilder` — min/max pyramid of a whole recording (in memory `MEMORY_BASE` or in a file `FILE_BASE`), background building with progress |
| [`signal`](./src/signal.rs) | `LocalSignal`, `SignalBackend` — the signal as a viewer sees it (`FILE_PYRAMID_MIN_BYTES`) |
| [`view`](./src/view.rs) | `View` (what a viewer asks for), `Envelope` (what it gets), `RAW_BLOCK_VALUES` |

## Features

| Feature | Description |
|---|---|
| `device` | Envelopes on a CubeCL device |
| `wgpu` / `cuda` / `hip` / `cpu` | GPU/CPU backends for device envelopes |

## Usage

```rust
use neurolib::view::min_max_decimate;

// Per-column min/max envelope: keeps every spike peak visible at any zoom
let env = min_max_decimate(&samples, 4096);
```

## Cargo

```toml
[dependencies]
neo-view = { path = "crates/neo-view", default-features = false, features = ["device", "wgpu"] }
```

## Book

[Viewing](../../neo-book/src/building-blocks/viewing.md)



