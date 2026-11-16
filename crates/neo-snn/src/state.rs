//! States of neurons and layers (membrane potential, synaptic current, trace).
//!
//! The state is NOT part of `Module`/`Record` - it is transferred to `step()`
//! and returns from it. `ResetMode` specifies how the membrane is reset after
//! spike (snntorch style); [`ResetOrder`] - the order “reset ↔ decay” within the step.

use burn::tensor::Device;
use burn::tensor::Tensor;

/// Method for resetting the membrane potential after a spike (analogous to `reset_mechanism`
/// in snntorch).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ResetMode {
    /// Subtract the threshold from the membrane at the spike sites (default).
    #[default]
    Subtract,
    /// Zero the membrane where the neuron has spiked.
    Zero,
    /// No reset (for example, for rate-coded reading).
    None,
}

/// Membrane reset with SCALAR threshold.
///
/// `mem_new` - membrane after integration, `spike_hard` - hard spike
/// (0/1), `threshold` — threshold (needed only for [`ResetMode::Subtract`]).
pub fn apply_reset(
    mem_new: Tensor<2>,
    spike_hard: &Tensor<2>,
    threshold: f32,
    mode: ResetMode,
) -> Tensor<2> {
    match mode {
        ResetMode::Subtract => mem_new - spike_hard.clone().mul_scalar(threshold),
        ResetMode::Zero => mem_new.mul(spike_hard.clone().mul_scalar(-1.0).add_scalar(1.0)),
        ResetMode::None => mem_new,
    }
}

/// Membrane reset with TENSOR threshold.
///
/// Needed for a trainable threshold: reading the threshold on the CPU
/// through `into_data()` for `Subtract` at EVERY time step would force a GPU
/// synchronization with the host at every step. Here the threshold remains a
/// tensor and there is no synchronization.
pub fn apply_reset_tensor(
    mem_new: Tensor<2>,
    spike_hard: &Tensor<2>,
    threshold: &Tensor<2>,
    mode: ResetMode,
) -> Tensor<2> {
    match mode {
        ResetMode::Subtract => mem_new - spike_hard.clone().mul(threshold.clone()),
        ResetMode::Zero => mem_new.mul(spike_hard.clone().mul_scalar(-1.0).add_scalar(1.0)),
        ResetMode::None => mem_new,
    }
}

/// The order is “reset ↔ membrane decay” (reset / decay) in one time step.
///
/// Affects only `ResetMode::Subtract`: with `Zero` (multiplying by `1 − S`) and `None`
/// the order does not change either the spikes or the dynamics, because the nulling commutes with the scalar β.
///
/// * [`BeforeDecay`](Self::BeforeDecay) - by default, the behavior of the crate before the appearance
///   this switch. The reset is applied immediately in the spike step, so in the next step
///   the already reset membrane is attenuated, including the subtracted threshold:
///   `U[t+1] = β·(U[t] − θ·S[t]) + I[t+1]`. The membrane is in a state AFTER reset.
/// * [`AfterDecay`](Self::AfterDecay) - dynamics of snnTorch (`reset` is calculated by the membrane
///   the previous step and is subtracted AFTER attenuation, the threshold itself does not attenuate):
///   `U[t+1] = β·U[t] − θ·S[t] + I[t+1]`. In the state (and in the returned `NeuronState::mem`)
///   the membrane lies BEFORE the reset, and the reset of the previous spike is applied at the beginning of the next
///   step; `S[t]` is restored from it as `U[t] > θ` (with winner-take-all, if it
///   enabled), so a separate field in the state is not needed.
///
/// To transfer weights trained on a network with "reset after decay", choose `AfterDecay`:
/// otherwise, the subtracted threshold attenuates by a factor β and the spike rasters diverge.
///
/// The parameter is not included in record (like [`ResetMode`]): it must be set in the configuration
/// layer when loading the weights.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ResetOrder {
    /// Reset first, fade in next step: `U' = β·(U − θ·S) + I` (default).
    #[default]
    BeforeDecay,
    /// First fade, then reset: `U' = β·U − θ·S + I`; in the state of the membrane before reset.
    AfterDecay,
}

/// Hard spike `mem > threshold` (0/1) on the finished membrane, uncoupled from the graph.
///
/// The strict condition is the same as for [`spike_with_scalar_threshold`](crate::neuron::spike_with_scalar_threshold).
pub fn hard_spike(mem: &Tensor<2>, threshold: f32) -> Tensor<2> {
    mem.clone()
        .detach()
        .sub_scalar(threshold)
        .greater_elem(0.0)
        .float()
}

/// Same as [`hard_spike`], but with a tensor threshold (teachable threshold).
pub fn hard_spike_tensor(mem: &Tensor<2>, threshold: &Tensor<2>) -> Tensor<2> {
    (mem.clone().detach() - threshold.clone().detach())
        .greater_elem(0.0)
        .float()
}

/// Attenuation with delayed reset (`ResetOrder::AfterDecay`), SCALAR threshold.
///
/// `mem` - membrane of the previous step BEFORE reset, `pending` - hard spike of the previous step
/// (0/1, see [`hard_spike`]). Returns a membrane ready to accept current step current:
/// `Subtract` — `β·mem − θ·pending`; `Zero` — `β·mem·(1 − pending)`; `None` — `β·mem`.
pub fn decay_then_reset(
    mem: Tensor<2>,
    beta: Tensor<2>,
    pending: &Tensor<2>,
    threshold: f32,
    mode: ResetMode,
) -> Tensor<2> {
    match mode {
        ResetMode::Subtract => mem.mul(beta) - pending.clone().mul_scalar(threshold),
        ResetMode::Zero => mem
            .mul(pending.clone().mul_scalar(-1.0).add_scalar(1.0))
            .mul(beta),
        ResetMode::None => mem.mul(beta),
    }
}

/// Same as [`decay_then_reset`], but with a TENSOR threshold.
pub fn decay_then_reset_tensor(
    mem: Tensor<2>,
    beta: Tensor<2>,
    pending: &Tensor<2>,
    threshold: &Tensor<2>,
    mode: ResetMode,
) -> Tensor<2> {
    match mode {
        ResetMode::Subtract => mem.mul(beta) - pending.clone().mul(threshold.clone()),
        ResetMode::Zero => mem
            .mul(pending.clone().mul_scalar(-1.0).add_scalar(1.0))
            .mul(beta),
        ResetMode::None => mem.mul(beta),
    }
}

/// A membrane ready to accept step current for neurons with a scalar threshold.
///
/// `BeforeDecay` - simply `β·mem` (the membrane is already reset);
/// `AfterDecay` - `β·mem` with delayed reset of the spike of the previous step.
pub fn decay_membrane(
    mem: Tensor<2>,
    beta: Tensor<2>,
    threshold: f32,
    mode: ResetMode,
    order: ResetOrder,
) -> Tensor<2> {
    match order {
        ResetOrder::BeforeDecay => mem.mul(beta),
        ResetOrder::AfterDecay if mode == ResetMode::None => mem.mul(beta),
        ResetOrder::AfterDecay => {
            let pending = hard_spike(&mem, threshold);
            decay_then_reset(mem, beta, &pending, threshold, mode)
        }
    }
}

/// The membrane that goes into the state after the step (scalar threshold).
///
/// `BeforeDecay` — with reset applied ([`apply_reset`]); `AfterDecay` - as is: reset
/// will occur at the beginning of the next step ([`decay_membrane`]).
pub fn carry_membrane(
    mem_new: Tensor<2>,
    spike_hard: &Tensor<2>,
    threshold: f32,
    mode: ResetMode,
    order: ResetOrder,
) -> Tensor<2> {
    match order {
        ResetOrder::BeforeDecay => apply_reset(mem_new, spike_hard, threshold, mode),
        ResetOrder::AfterDecay => mem_new,
    }
}

/// Hidden state of one layer of neurons: membrane and optional
/// synaptic current. The shape of each tensor is `[batch, units]`.
#[derive(Debug, Clone)]
pub struct NeuronState {
    /// Membrane potential `[batch, units]`.
    pub mem: Tensor<2>,
    /// Synaptic current/conductance `[batch, units]`.
    /// `None` for a simple LIF.
    pub syn: Option<Tensor<2>>,
}

impl NeuronState {
    /// Zero state with membrane and synaptic current.
    pub fn zeros(device: &Device, batch_size: usize, units: usize) -> Self {
        Self {
            mem: Tensor::zeros([batch_size, units], device),
            syn: Some(Tensor::zeros([batch_size, units], device)),
        }
    }

    /// Zero state with membrane only (`syn = None`).
    /// For Leaky/RLeaky, which do not need a synaptic trace.
    pub fn zeros_mem_only(device: &Device, batch_size: usize, units: usize) -> Self {
        Self {
            mem: Tensor::zeros([batch_size, units], device),
            syn: None,
        }
    }

    /// A state of ready-made tensors.
    pub fn new(mem: Tensor<2>, syn: Option<Tensor<2>>) -> Self {
        Self { mem, syn }
    }

    /// Detach the state from the autodifferentiation graph.
    ///
    /// Needed for truncated BPTT: without this the graph grows to its entire length
    /// sequences and memory consumption are linear in the number of steps.
    pub fn detach(self) -> Self {
        Self {
            mem: self.mem.detach(),
            syn: self.syn.map(|s| s.detach()),
        }
    }
}

/// Extended state of event-driven layers: trace + membrane +
/// optional delta gating buffers.
#[derive(Debug, Clone)]
pub struct LayerState {
    /// Synaptic trace `[batch, in_features]`.
    pub trace: Tensor<2>,
    /// Membrane potential `[batch, out_features]`.
    pub mem: Tensor<2>,
    /// For delta-gated KAN: previous trace.
    pub last_trace: Option<Tensor<2>>,
    /// For delta-gated KAN: previous KAN output.
    pub last_output: Option<Tensor<2>>,
}

impl LayerState {
    /// Zero state of the event-driven layer.
    pub fn zeros(
        device: &Device,
        batch_size: usize,
        in_features: usize,
        out_features: usize,
        with_kan_cache: bool,
    ) -> Self {
        let trace = Tensor::zeros([batch_size, in_features], device);
        let mem = Tensor::zeros([batch_size, out_features], device);
        let (last_trace, last_output) = if with_kan_cache {
            (
                Some(Tensor::zeros([batch_size, in_features], device)),
                Some(Tensor::zeros([batch_size, out_features], device)),
            )
        } else {
            (None, None)
        };
        Self {
            trace,
            mem,
            last_trace,
            last_output,
        }
    }

    /// Detach the state from the autodifferentiation graph (truncated BPTT).
    pub fn detach(self) -> Self {
        Self {
            trace: self.trace.detach(),
            mem: self.mem.detach(),
            last_trace: self.last_trace.map(|t| t.detach()),
            last_output: self.last_output.map(|t| t.detach()),
        }
    }
}
