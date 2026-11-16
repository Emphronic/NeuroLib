# neo-gui

NeuroLib workbench: a GPUI-based window for viewing multi-channel neural activity — traces,
heatmap, channel list, timeline, and training workspaces (Explore, Curation, Training).
"Channels" here are neurons or layers: spikes, membrane potentials, weights in time.

## Launch

```bash
# View a recording file
cargo run -p neo-gui --release -- <record>

# Synthetic data without a file
cargo run -p neo-gui --release -- --synthetic 90s -c 64

# Synthetic data, save PNG snapshot (no window)
cargo run -p neo-gui --release -- --synthetic 30s --snapshot out.png
```

## Recording formats

Raw `.bin` + JSON sidecar `.meta` (written by `neo_io::write_raw`), Zarr, SpikeGLX.

To show network activity, add it to `neo_core::MemoryRecording` (f32 data, number of
channels, frequency) and save via `neo_io::write_raw(...)`.

## Workspaces

| Workspace | Description |
|-----------|-------------|
| Explore | Raw traces, heatmap, channel list, timeline |
| Curating | Spike sorting curation (Phy folder loading, cluster metrics) |
| Training | Dashboard, network activity, brain view, loss/accuracy plots |
| Health | System and recording health monitoring |

## Requirements

- Rust 1.93 or later (`gpui-pre-util` uses `slice::as_array`)
- `gpui-kit` pinned exactly to `=0.7.0` — update only deliberately

## GPU

The GUI layer is host-only (CPU rendering). GPU acceleration for kernels lives in `neo-synk`
and `neo-core`/`neo-view`, selected through `neurolib` features (`wgpu`, `cuda`, `rocm`).

## Known limitations

- The GPUI layer requires rustc >= 1.93 and is not built in environments without it
- `test_synthetic_dataset_generation` allocates memory for a 4-hour 384-channel recording
  and may hit environment memory limits

## Cargo

```toml
[dependencies]
neo-gui = { path = "crates/neo-gui" }
```

## License

Dual-licensed under [MIT](LICENSE-MIT) or [Apache-2.0](LICENSE-APACHE).
