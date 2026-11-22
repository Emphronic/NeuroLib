//! Fast pathway state: the ordinary LIF membrane state.
//!
//! The fast pathway of a
//! [`DualMemoryLayer`](super::DualMemoryLayer) is the ordinary LIF machinery:
//! membrane + spikes live in the crate's [`NeuronState`], the state type used
//! by [`Leaky`](crate::neuron::Leaky). `FastState` is that state re-exported
//! under the fast-path name, so [`DualMemoryState`](super::DualMemoryState)
//! reads uniformly: fast state + slow state.

use crate::state::NeuronState;

/// Fast pathway state: the LIF membrane (`mem`) of the fast path.
pub type FastState = NeuronState;
