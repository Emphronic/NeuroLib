//! NWB files stored as Zarr v3 (hdmf-zarr layout).

pub mod acquisition;
mod format;
pub mod units;

pub use acquisition::{NwbZarrRecording, SeriesEntry, is_nwb_zarr, list_series};
pub use format::Nwb;
pub use units::{NwbUnitsTable, infer_nwb_sample_rate};
