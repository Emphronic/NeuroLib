//! # Structural plasticity for fixed-shape layers (Burn)
//!
//! Tensors of `burn::nn::Linear` and of all `neo-snn` layers cannot be grown on the fly
//! without rebuilding the module, its parameters and the optimizer state. Therefore here
//! the graph is "changed" **within a fixed capacity**:
//!
//! * a **connection** exists if its flag in the mask is enabled: growth = enabling the flag
//!   and writing the initial weight, disabling = clearing the flag and zeroing the weight;
//! * a **neuron** is a column of the weight matrix. Layer capacity includes *reserve*
//!   neurons (`initial_active < out_features`); growing a neuron = activating
//!   a reserve one with its inputs/outputs connected, removing = returning it to the reserve;
//! * after every optimizer step the weights are **projected** onto the mask
//!   (`w ← w ⊙ mask`, bias of inactive neurons = 0). Therefore Adam momentum cannot
//!   "revive" a disabled connection, and the layers (`RLeaky`, `Linear`, …) stay
//!   unchanged;
//! * changes are applied only between batches; every change writes a
//!   `StructuralEvent` (the same log/types as in the dynamic engine),
//!   is validated against loss and rolled back on degradation (weights and flags).
//!
//! The core ([`FixedShapeController`]) does not depend on Burn and works with flat
//! `f32` slices in `[in, out]` layout (`idx = i * out + o`). Tensor reading/writing
//! lives in `burn_glue`.
//!
//! Limitations (see also `STRUCTURAL_INTEGRATION.md`): optimizer state
//! for changed elements is not reset (the projection neutralizes the effect, but
//! when a connection is re-enabled Adam starts from the accumulated moments);
//! co-activity statistics are dense `in×out` per layer; the GPU path is not
//! implemented and has not been tested.

mod bundle;
mod config;
mod controller;
mod layer;

#[cfg(test)]
mod tests;

pub use bundle::{BUNDLE_VERSION, Bundle, LayerBlob, NamedTensor};
pub use config::{FixedGrowthConfig, FixedNeuronConfig, FixedPruningConfig, FixedShapeConfig};
pub use controller::{AdaptInput, FixedAdaptReport, FixedShapeController, FixedSkip};
pub use layer::{
    LayerRole, LayerSpec, LayerState, LayerStats, LayerUpdate, NeuronChange, SynChange,
};
