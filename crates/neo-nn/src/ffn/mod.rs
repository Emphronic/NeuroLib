//! Feed-forward layers: gated family (GLU) and Mixture of Experts.
//!
//! * [`GatedFfn`] - general gated FFN with activation selection ([`GateActivation`]);
//! * [`Geglu`], [`Reglu`] - separate layers for GELU and ReLU;
//! * [`Moe`] is the only implementation of Mixture of Experts (top-k, capacity,
//!   shared expert, balancing loss) on top of [`GatedFfn`].

pub mod gated;
pub mod geglu;
pub mod moe;
pub mod reglu;

pub use gated::{GateActivation, GatedFfn, GatedFfnConfig};
pub use geglu::{Geglu, GegluConfig};
pub use moe::{Moe, MoeAuxStats, MoeConfig, OverflowPolicy};
pub use reglu::{Reglu, RegluConfig};

#[cfg(test)]
mod tests;
