//! The compute layer (`compute` feature): the single entry point through which every crate reaches
//! a CubeCL runtime.
//!
//! - [`ComputeTask`] / [`ComputeTarget::run`](crate::device::ComputeTarget): run generic work on
//!   the selected runtime.
//! - [`LaunchGeometry`]: cube sizes and counts from the runtime's properties.
//! - [`tune`]: CubeCL autotuning for launch settings with no device-independent best value.
//! - [`mod@bench`]: device-synchronised timing.
//!
//! Algorithm crates take a `Client` and use these; they never inspect or special-case the
//! device.

pub mod bench;
pub mod launch;
pub mod limits;
pub mod pinning;
mod special;
mod target;
pub mod tune;

pub use crate::device::{ComputeError, ComputeTarget, RUNTIME_ENV};
pub use launch::{LaunchGeometry, channel_position, row_position, sample_position};
pub use limits::{MAX_DEVICE_ELEMENTS, device_elements};
pub use pinning::{PinnedChoices, pin_tuned_choices, tuned_choices_pinned};
pub use special::{
    NEGATIVE_INFINITY_BITS, POSITIVE_INFINITY_BITS, negative_infinity, positive_infinity,
};
pub use target::ComputeTask;
pub use target::{open_device, shutdown_devices};
