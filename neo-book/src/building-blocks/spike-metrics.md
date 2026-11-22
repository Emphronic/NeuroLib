# Spike metrics (curation)

`neo-synapse` — what curating a spike sorting needs, on the host: no detection, no sorters, no GPU.
`.sorting.zarr` / NWB `/units` sortings are not read yet (see crate docs).

- `metrics`: `compute_autocorrelogram`, `compute_crosscorrelogram`, `compute_instantaneous_firing_rate`,
  `isi_histogram`, `compute_isi_violations`, `Correlogram`, `FiringRateCurve`, `IsiMetrics`, `QualityCriteria`;
- `snippets` + `storage`: `read_snippets` (`WaveformSnippet` from a recording), `load_spikes`, `fill_similarity` —
  loading spike data and sorting units (a Phy / Kilosort folder);
- `pca`: `extract_waveform_pca` / `PcaModel` — small PCA over waveforms;
- `label`: `UnitQualityLabel`.

Quality-driven curation helpers: assign labels via `storage::load_spikes` + `QualityCriteria` evaluation.
