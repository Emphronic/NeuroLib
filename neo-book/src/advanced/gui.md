# The GUI workbench (neo-gui)

`neo-gui` — NeuroLib workbench (GPUI + gpui-kit): the **Explore** workspace — traces, heatmap, channels, timeline —
over any multi-channel recording: neuron activity, membrane potentials, spike rasters (a raw `.bin` + JSON `.meta`
sidecar, Zarr, SpikeGLX, or a synthetic one). Layers: `engine` (no UI type: data, timeline, renderers, render
threads) → `store` (shared state and events) → `viewmodels` → `views` / `app` (GPUI).

Usage:

```bash
neo-gui <path>                 # a recording (SpikeGLX .bin/.cbin, raw .bin + JSON .meta, Zarr)
neo-gui --synthetic 90s        # procedural recording instead
neo-gui --synthetic 2h -c 64 -r 10_000
neo-gui --snapshot out.png --heatmap   # render one view to PNG, no window
```

The GUI renders the signal through `neo-view` (envelopes and pyramids — see
[Viewing](../building-blocks/viewing.md)) and training telemetry through `neo-snn::telemetry` (Training tab).
