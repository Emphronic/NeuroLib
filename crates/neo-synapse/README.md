# neo-synapse

Spike-train metrics and sorting-file loading for curation: correlograms, ISI histograms,
firing-rate curves, waveform snippets, PCA, and Phy/Kilosort folder loading.

## What it provides

- **Correlograms** — symmetric auto/cross-correlograms of sorted spike trains (`Correlogram`:
  bin centers, counts, bin size, window)
- **ISI histograms** — inter-spike interval distributions (`isi_histogram`, `compute_isi_violations`)
- **Firing rate** — per-neuron and population firing rates over time (`FiringRateCurve`)
- **Snippets** — waveform snippets around detected spikes (`read_snippets`, `WaveformSnippet`)
- **PCA** — principal component analysis on spike features (`extract_waveform_pca`, `PcaModel`)
- **Sorting files** — Phy/Kilosort loading (`load_spikes`, `fill_similarity`); `.sorting.zarr` /
  NWB `/units` sortings are not read yet
- **Quality criteria** — `QualityCriteria`, `UnitQualityLabel`

## Usage

```rust
use neurolib::synapse::{compute_autocorrelogram, compute_crosscorrelogram, isi_histogram};

// Symmetric auto-correlogram of a spike train (sample indices)
let acg = compute_autocorrelogram(&train, 30_000.0, 1.0, 10.0);

// Cross-correlogram between two trains (lags t_b - t_a)
let ccg = compute_crosscorrelogram(&train_a, &train_b, 30_000.0, 1.0, 10.0);
```

Host-only: no detection, no sorters, no GPU.

## Cargo

```toml
[dependencies]
neo-synapse = { path = "crates/neo-synapse" }
```

## Book

[Spike metrics](../../neo-book/src/building-blocks/spike-metrics.md)

