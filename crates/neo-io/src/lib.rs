//! Recording file formats for NeuroLib.
//!
//! Every format implements [`neo_core::RecordingSource`], so callers read bounded chunks of any
//! recording without knowing how it is stored, and [`Format`], so [`open`] can pick the reader
//! from the path ([`registry::formats`]).
//!
//! - [`core`]: the [`Format`] trait, detection and opening, source listing, chunk cache.
//!   Out-of-core window streaming is `neo_core::WindowLoader`.
//! - [`container`]: storage containers shared by every format (binary codecs, …).
//! - [`generic`]: domain-independent recording formats (raw binary, Zarr `/traces`).
//! - [`neuro`] (feature `neuro`): neural-recording formats (NWB, SpikeGLX, mtscomp), probe
//!   geometry, synthetic recordings.

pub mod container;
pub mod core;
pub mod generic;
#[cfg(feature = "neuro")]
pub mod neuro;
pub mod registry;

pub use crate::core::{
    CachedRecording, Format, SourceEntry, SourceKind, default_source, detect, open, open_source,
    sources,
};
pub use generic::raw::{RawParams, RawRecording, write_raw};
#[cfg(feature = "zarr")]
pub use generic::zarr_traces::{ZarrRecording, write_zarr};
pub use registry::formats;

#[cfg(feature = "neuro")]
pub use neuro::mtscomp::MtscompRecording;
#[cfg(all(feature = "neuro", feature = "zarr"))]
pub use neuro::nwb::NwbZarrRecording;
#[cfg(feature = "neuro")]
pub use neuro::probe::{Position3D, SensorLayout, SensorSite, probe_of};
#[cfg(feature = "neuro")]
pub use neuro::spikeglx::SpikeGlxMeta;
#[cfg(feature = "neuro")]
pub use neuro::synthetic::{SyntheticParams, SyntheticRecording};
