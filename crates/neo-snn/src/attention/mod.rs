//! Spike-based self-attention and transformer block for SNN.
//!
//! Two competing ways to count "attention" entirely in spikes −
//! without softmax and (for SDSA) without matrix multiplications on inference at all:
//!
//! * [`SpikingSelfAttention`] (SSA, Spikformer, Zhou et al. 2023) —
//!   `Q, K, V` — spike tensors (0/1), attention is calculated as
//!   `SN(Q · Kᵀ · V · s)`, that is, ordinary matrix multiplication, but over
//!   binary matrices and without softmax (it is both expensive and does not fit well with
//!   spikes - an exponent from 0/1 is uninformative). Scale `s` - no
//!   `1/√d`, as in ANN-attention, and separate trainable/fixed
//!   constant: the range of values after summation over spikes is different than
//!   for softmax-normalized weights.
//! * [`SpikeDrivenSelfAttention`] (SDSA, Spike-Driven Transformer,
//!   Yao et al. 2023) - without matrix multiplications at all: attention mask
//!   `SN(Σ_c(Q ⊗ K))` is constructed by element-wise (Hadamard) product and
//!   summation over channels, then applied to `V` also element by element.
//!   On neuromorphic hardware these are addition operations, not multiplication -
//!   hence the "spike-driven" in the name.
//!
//! [`SpikingMlp`] — spiking MLP block (`fc → bn → SN → fc → bn → SN`),
//! [`SpikingTransformerBlock`] - attention + MLP with residual connection by
//! MEMBRANE potential (before spike nonlinearity), as in both
//! original works - residual on already sampled spikes
//! It would simply add two binary signals and lose some of the information.
//!
//! # Difference from the rest of the crate: without explicit per-step state
//!
//! Everything else is in `neo-snn` (neurons in [`crate::neuron`], constituent layers
//! in [`crate::layers`]) explicitly rolls the state step by step through
//! `step(input, &state, surrogate) -> (output, new_state)`, and time `T`
//! is expanded by the calling code in a loop.
//!
//! This module is designed differently: both reference attention (SSA and SDSA)
//! process the entire time axis at once, as an additional dimension
//! batch - `[T, B, N, D] → [T·B, N, D]` for BatchNorm and linear
//! projections - rather than recurrently step by step. attention and MLP are not here
//! membrane that would accumulate BETWEEN steps: each call
//! `forward` is completely self-sufficient for all `T` steps at once, so
//! `NeuronState`/`step()` are not needed for this module. The only memory
//! time" inside the block are BatchNorm statistics, and they are already encapsulated
//! in `burn::nn::BatchNorm`.
//!
//! Spike nonlinearity SN(·) is [`crate::neuron::Leaky`] with
//! `reset_mode = None` (resetting the membrane between "steps" is pointless here:
//! there are no steps in the recurrent sense, each call is a pure function of
//! membrane potential at the input).

mod block;
mod mlp;
mod spike_driven_self_attention;
mod spiking_self_attention;

pub use block::{
    AttentionKind, SpikingAttentionVariant, SpikingTransformerBlock, SpikingTransformerBlockConfig,
};
pub use mlp::{SpikingMlp, SpikingMlpConfig};
pub use spike_driven_self_attention::{SpikeDrivenSelfAttention, SpikeDrivenSelfAttentionConfig};
pub use spiking_self_attention::{SpikingSelfAttention, SpikingSelfAttentionConfig};

use burn::tensor::Device;
use burn::tensor::Tensor;

use crate::neuron::{Leaky, LeakyConfig};
use crate::state::ResetMode;
use crate::surrogate::Surrogate;

/// Common input/output of all attention primitives of this module.
///
/// shape `[T, B, N, D]`: `T` - time steps, `B` - batch, `N` - number
/// tokens (patch), `D` is the dimension of the attribute.
pub type SpikeSequence = Tensor<4>;

/// Create SN(·)-nonlinearity for this module: [`Leaky`] without trainee
/// threshold/β-damping and no reset.
///
/// `beta_init = 0.0` (after `sigmoid` inside `Leaky` is `β ≈ 0.5`, but
/// the value is unimportant: [`apply_spiking_nonlinearity`] calls `step` with
/// zero membrane at the input, so that the attenuation `β · 0` does not affect
/// result) and `reset_mode = ResetMode::None`, because there is no
/// steps between which the charge would accumulate - see the module documentation.
pub(crate) fn spiking_nonlinearity(threshold: f32, device: &Device) -> Leaky {
    LeakyConfig::new(0.0)
        .with_threshold(threshold)
        .with_reset_mode(ResetMode::None)
        .init(device)
}

/// Apply SN(·) to a tensor of arbitrary shape `[.., D]`, where `D` is
/// last dimension (number of "units" for [`Leaky::step`]).
///
/// [`Leaky::step`] works with rank 2 (`[batch, units]`), so the input
/// temporarily rotates to `[prod(other axes), D]` and back.
/// The inlet membrane is zero (see [`spiking_nonlinearity`]): SN(·) in
/// Spikformer/SDT is a pure function of membrane potential, not
/// recurrent neuron with state between calls.
pub(crate) fn apply_spiking_nonlinearity<const D: usize, S: Surrogate + ?Sized>(
    lif: &Leaky,
    x: Tensor<D>,
    surrogate: &S,
) -> Tensor<D> {
    let dims = x.dims();
    let features = dims[D - 1];
    let batch: usize = dims[..D - 1].iter().product();

    let flat = x.reshape([batch, features]);
    let state = crate::state::NeuronState::zeros_mem_only(&flat.device(), batch, features);
    let (spike, _) = lif.step(flat, &state, surrogate);
    spike.reshape(dims)
}
