//! Preparing continuous multi-channel signals for viewing, wherever the viewer runs: a local
//! app next to the recording, or a remote viewer served over the network.
//!
//! - [`envelope`]: the min/max reduction (`[min, max]` per column keeps every spike peak visible
//!   at any zoom), on the host and (feature `device`, on by default with `wgpu`) on a CubeCL
//!   device.
//! - [`view`]: what a viewer asks for ([`View`]) and gets ([`Envelope`]), locally or remotely.
//! - [`pyramid`]: the min/max pyramid of a whole recording (in memory or in a file) and its
//!   background builder.
//! - [`signal`]: a signal as a viewer sees it ([`SignalBackend`]): [`LocalSignal`] (a recording
//!   and its pyramid in this process), or a remote session answering the same views.
//!
//! Rendering and UI belong to the app; file formats to neo-io.

pub mod envelope;
pub mod pyramid;
mod read;
pub mod signal;
pub mod view;

#[cfg(feature = "device")]
pub use envelope::envelope_on_device;
pub use envelope::{Block, Columns, min_max_decimate, min_max_decimate_into};
pub use pyramid::{
    FILE_BASE, MEMORY_BASE, OnProgress, Progress, Pyramid, PyramidBuilder, PyramidIdentity,
    pyramid_path,
};
pub use signal::{FILE_PYRAMID_MIN_BYTES, LocalSignal, SignalBackend};
pub use view::{Envelope, RAW_BLOCK_VALUES, View};
