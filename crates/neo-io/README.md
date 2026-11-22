# neo-io

Recording formats and containers for multi-channel neural data: raw `.bin`, `.npy`/`.npz`, Zarr
traces — and, behind the `neuro` feature, NWB, SpikeGLX, mtscomp, probe geometry, synthetic
recordings, and Phy/Kilosort sorting files.

Every format implements `RecordingSource` (bounded reads of any recording without knowing how it is
stored) and the `Format` trait; `detect`/`open` pick the reader from the path.

## What it provides

| Module | Contents |
|---|---|
| [`generic::raw`](./src/generic/raw) | `RawRecording`, `RawParams` (JSON `.meta` sidecar), `write_raw` — headerless binary recordings, memory-mapped |
| [`generic::zarr_traces`](./src/generic/zarr_traces) (`zarr`) | `ZarrRecording`, `write_zarr` — Zarr v2 traces |
| [`container::npy`](./src/container/npy) | `.npy`/`.npz` reading and writing |
| [`container::binary`](./src/container/binary) | Shared binary codecs (frames, CRC) |
| [`neuro`](./src/neuro) (`neuro`) | `MtscompRecording` (SpikeGLX `.bin`/`.cbin`), NWB, probe geometry, synthetic recordings |
| [`core`](./src/core) | `Format` trait, `detect`, `open`, `open_source`, `sources`, `CachedRecording` |
| [`registry`](./src/registry.rs) | `formats` — the registry of supported formats |

## Features

| Feature | Description |
|---|---|
| `zarr` (default) | Zarr trace format |
| `neuro` | NWB, SpikeGLX, mtscomp, probe geometry, sorting files |

## Usage

```rust
use std::path::Path;
use neurolib::io::open;

// Open any supported format (the format is detected from the path)
let rec = open(Path::new("recording.nwb"))?;

// List available sources (the main one + extras, e.g. units for NWB)
for source in neurolib::io::sources(Path::new("recording.nwb"))? {
    println!("{:?}", source);
}
```

## Book

[Data](../../neo-book/src/basic-workflow/data.md)



