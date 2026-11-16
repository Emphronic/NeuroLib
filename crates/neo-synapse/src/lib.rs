//! What curating a spike sorting needs, on the host: spike-train metrics (correlograms, ISI,
//! firing rate), waveform snippets read from a recording, a small PCA, and loading of the sorting
//! files (a Phy / Kilosort folder). Host-only: no detection, no sorters, no GPU.
//! `.sorting.zarr` / NWB `/units` sortings are not read yet.

pub mod extract;
pub mod label;
pub mod math;
pub mod metrics;
pub mod pca;
pub mod snippets;
pub mod storage;

pub use extract::read_snippets;
pub use label::UnitQualityLabel;
pub use metrics::{
    Correlogram, FiringRateCurve, IsiMetrics, QualityCriteria, compute_autocorrelogram,
    compute_crosscorrelogram, compute_instantaneous_firing_rate, compute_isi_violations,
    isi_histogram,
};
pub use pca::{PcaModel, extract_waveform_pca};
pub use snippets::WaveformSnippet;
pub use storage::{fill_similarity, load_spikes};
