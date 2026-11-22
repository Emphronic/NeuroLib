//! Dual memory pathways: fast LIF activity + compact slow memory
//! (paper: Sun et al., "Algorithm–hardware co-design of neuromorphic
//! networks with dual memory pathways", Nature Machine Intelligence, 2026).
//!
//! * [`fast_state`] — fast pathway state: the ordinary LIF membrane
//!   ([`NeuronState`]);
//! * [`slow_state`] — slow pathway state: the LMU-style memory `m [batch, d]`;
//! * [`memory_feedback`] — `Im = W_m·m`: the slow state as a per-neuron
//!   membrane current;
//! * [`dual_memory`] — [`DualMemoryLayer`](dual_memory::DualMemoryLayer):
//!   one layer with both pathways (the paper's DMP-SNN).
//!
//! Reference: the paper's Methods (state-space matrices, exact ZOH
//! discretization, the DMP-SNN step) and the original implementation
//! ([`Dual_memory_pathways`](https://github.com/sunpengfei1122/Dual_memory_pathways),
//! `dula_memory_pathway`).

pub mod dual_memory;
pub mod fast_state;
pub mod memory_feedback;
pub mod slow_state;

pub use dual_memory::{DualMemoryConfig, DualMemoryLayer, DualMemoryState};
pub use fast_state::FastState;
pub use memory_feedback::MemoryFeedback;
pub use slow_state::{SlowMemory, SlowMemoryConfig, SlowMemoryState};
