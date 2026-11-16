# neo-io

Recording formats and containers for multi-channel neural data: raw `.bin`, `.npy`/`.npz`,
Zarr, with optional NWB, SpikeGLX, mtscomp, probe geometry, and sorting-file support.

## What it provides

- **Raw binary** — `.bin` + `.meta` JSON sidecar, written by `neo_io::write_raw`
- **NumPy** — `.npy` and `.npz` (zip) reading and writing
- **Zarr** — Zarr v2 traces (behind the `zarr` feature)
- **Neuro formats** — NWB, SpikeGLX, mtscomp, probe geometry, synthetic recordings (behind the `neuro` feature)
- **Phy/Kilosort** — sorting files, templates, similar-templates

## Features

| Feature | Description |
|---------|-------------|
| `zarr` | Enable Zarr trace format |
| `neuro` | Enable NWB, SpikeGLX, mtscomp, probe geometry |

## Usage

```rust
use neo_io::{write_raw, Format};

// Write raw recording + sidecar
write_raw("recording.bin", &data, num_channels, sample_rate)?;

// Open any supported format
let fmt = neo_io::open("recording.bin")?;
```

## Cargo

```toml
[dependencies]
neo-io = { path = "crates/neo-io", features = ["zarr", "neuro"] }
```

## License

Dual-licensed under [MIT](LICENSE-MIT) or [Apache-2.0](LICENSE-APACHE).
