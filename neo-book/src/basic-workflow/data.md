# Data

NeuroLib reads recordings through `neo-io` (crate `neo_io`): every format implements `RecordingSource`, so callers
read bounded chunks of any recording without knowing how it is stored, and the `Format` trait + `detect`/`open` pick
the reader from the path.

```rust
use neurolib::io::*;
use std::path::Path;

// Open any supported recording
let recording = open(Path::new("traces.nwb"))?;

// List the available sources (the main one + extras, e.g. units for NWB)
for source in sources(Path::new("traces.nwb"))? {
    println!("{:?}", source);
}
```

Formats:

- raw `.bin` + JSON `.meta` sidecar (`RawParams`: channels, `sample_rate_hz`, format, gain, unit µV) — memory-mapped,
  values are `stored · gain + offset`;
- `.npy` / `.npz` — `CachedRecording`;
- Zarr traces — `ZarrRecording` (feature `zarr`);
- NWB — acquisition, sorting units (feature `neuro`);
- SpikeGLX `.bin`/`.cbin` — `MtscompRecording` (feature `neuro`);
- synthetic recordings — `SyntheticRecording` (see `neo-gui --synthetic`).

Buffer/time types come from `neo-core`: `SampleFormat`, `SignalUnit` (with `MICROVOLTS_PER_VOLT`), `MemoryOrder`,
`SampleRate`, `RationalTime`, `TimeRange`, `SlicedRecording`, `WindowLoader` (out-of-core window streaming).

Encode input signals into spikes with the encoders in `neo-snn` — [Spikes & encoding](../building-blocks/spikes.md).
