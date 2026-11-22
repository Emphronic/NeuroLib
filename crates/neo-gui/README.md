# neo-gui

NeuroLib workbench — a GPUI window for viewing multi-channel neural activity: traces, heatmap,
channels list and timeline, with training and curation workspaces (Explore, Curation, Training,
Health).

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

Raw `.bin` + JSON sidecar `.meta` (written by `neo_io::write_raw`), Zarr, SpikeGLX — see `neo-io`.

To show network activity: add it to `neo_core::MemoryRecording` (f32 data, number of channels,
frequency) and save via `neo_io::write_raw(...)`.

## Workspaces

| Workspace | Description |
|-----------|-------------|
| Explore | Raw traces, heatmap, channel list, timeline |
| Curation | Spike sorting curation (Phy folder loading, cluster metrics) |
| Training | Dashboard, network activity, brain view, loss/accuracy plots |
| Health | System and recording health monitoring |

## Requirements

- Rust 1.93 or later (`gpui-pre-util` uses `slice::as_array`)
- `gpui-kit` pinned exactly to `=0.7.0` — update only deliberately

## Book

[The GUI workbench](../../neo-book/src/advanced/gui.md)

