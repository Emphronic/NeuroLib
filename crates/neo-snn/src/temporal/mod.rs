//! Temporal memory pathways: the dual fast/slow memory of Sun et al.,
//! "Algorithm–hardware co-design of neuromorphic networks with dual memory
//! pathways" (Nature Machine Intelligence, 2026) — see [`memory`].

pub mod memory;

pub use memory::{
    DualMemoryConfig, DualMemoryLayer, DualMemoryState, FastState, MemoryFeedback, SlowMemory,
    SlowMemoryConfig, SlowMemoryState,
};
