//! Foundational primitives, timestamps, buffers, and masks for NeuroLib.

pub mod buffer;
#[cfg(feature = "compute")]
pub mod compute;
pub mod device;
pub mod error;
pub mod mask;
pub mod progress;
pub mod recording;
pub mod time;
pub mod window;

pub use buffer::{BufferLayout, MemoryOrder, SignalChunk};
pub use device::{ComputeError, ComputeTarget};
pub use error::{DspError, DspResult};
pub use mask::ChannelMask;
pub use progress::{NoProgress, ProgressEvent, ProgressSink};
pub use recording::{
    ChannelInfo, MICROVOLTS_PER_VOLT, MemoryRecording, RecordingInfo, RecordingSource,
    SampleFormat, SignalUnit, SlicedRecording,
};
pub use time::{RationalTime, SampleRate, TimeRange};
pub use window::{ChunkSchedule, HaloWindow, WindowLoader};
