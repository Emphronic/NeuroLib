//! # NeuroLib
//!
//! Single entry point to the workspace: one crate, one `Cargo.toml` input, everything else is included
//! through features - in the same way that crate `burn` collects its subcrates.
//!
//! ```toml
//! [dependencies]
//! neurolib = { version = "0.1", features = ["snn", "wgpu"] }
//! ```
//!
//! | Module | Crate | Feature |
//! |---|---|---|
//! | [`snn`] | `neo-snn` | `snn` |
//! | [`nn`] | `neo-nn` | `nn` |
//! | [`quant`] | `neo-quantization` | `quant` |
//! | [`kernels`] | `neo-synk` | `kernels` |
//! | [`store`] | `neo-store` | `store` |
//! | [`core`] | `neo-core` | `core` |
//! | [`io`] | `neo-io` | `io` (`zarr`, `neuro`) |
//! | [`view`] | `neo-view` | `view` |
//! | [`synapse`] | `neo-synapse` | `synapse` |
//!
//! Groups: `ml` (default), `signal`, `full`. GPU: `wgpu`, `cuda`, `rocm`, plus `fusion`.
//! `neo-gui` is an application, not a library, so it's not here.
//!
//! `burn` is not re-exported: declare it separately in the project (`burn = "0.22"`, as in workspace).

#[cfg(feature = "snn")]
pub use neo_snn as snn;

#[cfg(feature = "nn")]
pub use neo_nn as nn;

#[cfg(feature = "quant")]
pub use neo_quantization as quant;

#[cfg(feature = "kernels")]
pub use neo_synk as kernels;

#[cfg(feature = "store")]
pub use neo_store as store;

#[cfg(feature = "core")]
pub use neo_core as core;

#[cfg(feature = "io")]
pub use neo_io as io;

#[cfg(feature = "view")]
pub use neo_view as view;

#[cfg(feature = "synapse")]
pub use neo_synapse as synapse;

/// Everyday names: currently prelude `neo-snn`. The remaining crates are imported via
/// own modules (`neurolib::nn::...`) so that glob import does not produce name conflicts.
pub mod prelude {
    #[cfg(feature = "snn")]
    pub use neo_snn::prelude::*;
}
