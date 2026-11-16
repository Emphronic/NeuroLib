# neo-synapse

Spike-train metrics and sorting-file loading for curation: correlograms, ISI histograms,
firing-rate curves, waveform snippets, PCA, and Phy/Kilosort folder loading.

## What it provides

- **Correlograms** — pairwise spike-train correlograms with lag bins
- **ISI histograms** — inter-spike interval distributions
- **Firing rate** — per-neuron and population firing rates over time
- **Snippets** — waveform snippets around detected spikes
- **PCA** — principal component analysis on spike features
- **Sorting files** — Phy/Kilosort `.npy` templates, similar-templates, cluster metadata

## Usage

```rust
use neo_synapse::{correlogram, isi_histogram, firing_rate};

let correl = correlogram(&spike_trains, num_bins, bin_width)?;
let isi = isi_histogram(&spike_train, max_ms, bin_width)?;
let rate = firing_rate(&spike_train, window_size)?;
```

## Cargo

```toml
[dependencies]
neo-synapse = { path = "crates/neo-synapse" }
```

## License

Dual-licensed under [MIT](LICENSE-MIT) or [Apache-2.0](LICENSE-APACHE).
