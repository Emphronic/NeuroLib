//! Optional stateful mode ("init_hidden" in snnTorch terminology)
//! on top of the explicit-state API of this crate.
//!
//! # For new code - [`crate::spiking_layer`]
//!
//! This module is the old way to keep state inside a layer: individually
//! `impl` for each “layer + state type” pair. There is now a single trait
//! [`SpikingLayer`](crate::spiking_layer::SpikingLayer) and universal
//! wrapper [`Session`](crate::spiking_layer::Session), which replaces
//! `Stateful`/`StatefulRecurrent` for any layer at once (including STDP layers with
//! scales able). `Stateful` left unchanged for feedback
//! compatibility.
//!
//! # Why this module
//!
//! Each layer in [`crate::neuron`] and [`crate::layers`] already implements
//! explicit-state API: `step(&self, input, &State, ...) -> (output, State)`.
//! This is architecturally natural for Rust - state is not hidden behind
//! interior mutability, but visible in the signature, and the calling code itself
//! decides what to do with it between steps (detach for truncated BPTT,
//! save/restore, use multiple independent states with
//! the same layer, etc.).
//!
//! This solution has one obvious cost: when migrating one-to-one models
//! from snnTorch, where `init_hidden=True` hides the state inside the module
//! and the layer is called simply as `spk, mem = lif(x)`, the state has to be
//! threaded through every call. For large networks with dozens of layers this
//! is a noticeable amount of boilerplate on each forward.
//!
//! This module does not change or replace the explicit-state API - it adds
//! ABOVE it is a thin wrapper [`Stateful`], which stores the state in
//! internal `RefCell` and updates it itself with each call, exactly like
//! `init_hidden=True` in snnTorch. Both modes coexist: `Stateful<L>`
//! just calls `L::step` and holds the state for you.
//!
//! # Example
//!
//! ```ignore
//! use neo_snn::prelude::*;
//! use neo_snn::stateful::Stateful;
//!
//! let lif = LeakyConfig::new(0.9).init(&device);
//! let mut lif = Stateful::<Leaky, NeuronState>::new_neuron(lif, &device, batch, units);
//!
//! for x_t in input_over_time {
//!     let spike = lif.step(x_t, &surrogate); // state is updated internally
//! }
//! lif.reset_hidden(&device, batch, units); // analogue of init_hidden() between episodes
//! ```
//!
//! # Why `RefCell` and not `Cell`/`Mutex`
//!
//! The tensors `Tensor<D>` are not `Copy`, so `Cell` is not suitable.
//! `Mutex` would be appropriate for multi-threaded access to the SAME
//! layer, but `Stateful<L>` is not a `Sync` primitive: it models one
//! a logical "live" layer instance with state, like an object in
//! single-threaded PyTorch code. `RefCell` gives the same borrowing discipline
//! at the same runtime as `Mutex`, without the locking overhead.
//!
//! # Truncated BPTT
//!
//! [`Stateful::detach_hidden`] unlinks the internal state from the graph
//! autodifferentiation - the same case as manual call
//! `state.detach()` in explicit-state mode (see [`crate::state::NeuronState::detach`]).
//! Call it between sequence chunks if you chunk
//! time-based training; otherwise the autodiff graph will grow throughout history
//! calls `step`.

use burn::tensor::Device;
use std::cell::RefCell;

use burn::tensor::Tensor;

use crate::layers::{ConvLstmState, DelayState, SConv2dLSTM, SpikingConv2d, SynapticDelay};
use crate::neuron::{Alpha, Lapicque, Leaky, RLeaky, RSynaptic, Synaptic};
use crate::state::NeuronState;
use crate::surrogate::Surrogate;

/// Stateful wrapper ("init_hidden" mode) over a layer whose explicit-state API
/// has shape `step(&self, input, &S, ...) -> (out, S)`.
///
/// `L` is the layer (for example, [`Leaky`]), `S` is the type of its state (for example,
/// [`NeuronState`]). (Deprecated: the layer-state relationship is now expressed
/// trait [`SpikingLayer`](crate::spiking_layer::SpikingLayer), and
/// universal wrapper - [`Session`](crate::spiking_layer::Session);
/// below is the original rationale.) The connection between `L` and `S` is not intentionally expressed by the trait:
/// `step` signatures differ in number and order for different layers
/// arguments (surrogate, `spike_prev` for recurrent neurons, additional.
/// parameters, etc.), and a single trait with one signature would force either
/// impoverish some layers, or dilute generics by arity. Instead of
/// this `Stateful<L, S>` is only a state container; specific methods
/// `step`/`forward` are defined in the `impl` blocks below, one for each
/// layer, each with its own natural signature.
pub struct Stateful<L, S> {
    /// Wrapped layer. Public: access to parameters, configuration and
    /// direct explicit-state API (`layer.step(...)`) is not lost -
    /// `Stateful` doesn't hide anything, it just adds on top.
    pub layer: L,
    state: RefCell<S>,
}

impl<L, S> Stateful<L, S> {
    /// Wrap an already created layer with a given initial state.
    pub fn from_parts(layer: L, initial_state: S) -> Self {
        Self {
            layer,
            state: RefCell::new(initial_state),
        }
    }

    /// Replace the internal state explicitly - for example, restore it
    /// after saving between runs, or substitute the state of another
    /// batch episode.
    pub fn set_hidden(&self, new_state: S) {
        *self.state.borrow_mut() = new_state;
    }

    /// Snapshot of the current state (cloning).
    pub fn hidden(&self) -> S
    where
        S: Clone,
    {
        self.state.borrow().clone()
    }

    /// Unpack the wrapper, returning the layer and the last state.
    pub fn into_parts(self) -> (L, S) {
        (self.layer, self.state.into_inner())
    }
}

impl<L, S> Stateful<L, S>
where
    S: Detach + Clone,
{
    /// Detach internal state from autodifferentiation graph
    /// (truncated BPTT). See the "Truncated BPTT" section in the module documentation.
    ///
    /// Clone the state before calling `S::detach` rather than picking it up
    /// from `RefCell` by value: `RefCell::replace_with` gives the closure
    /// only `&mut S`, and `detach(self)` for all states in this crate
    /// takes `self` by value (see [`NeuronState::detach`] and
    /// analogs) - its value can only be obtained by cloning
    /// or temporary replacement with a plug. Cloning is the same here
    /// what [`StatefulRecurrent::detach_hidden`] already does: `Tensor::clone`
    /// clones the Arc wrapper over the buffer, not the data itself, so
    /// This does not create any extra tensor allocation.
    pub fn detach_hidden(&self) {
        let detached = self.state.borrow().clone().detach();
        *self.state.borrow_mut() = detached;
    }
}

/// A state that can detach itself from the autodiff graph.
///
/// Implemented for all state types in this crate
/// ([`NeuronState`], [`ConvLstmState`], [`DelayState`]), which already carry
/// its own method `detach(self) -> Self` - the trait only gives
/// [`Stateful::detach_hidden`] a single way to call it regardless of `S`.
pub trait Detach {
    /// Detach from the auto-differentiation graph.
    fn detach(self) -> Self;
}

impl Detach for NeuronState {
    fn detach(self) -> Self {
        NeuronState::detach(self)
    }
}

impl Detach for ConvLstmState {
    fn detach(self) -> Self {
        ConvLstmState::detach(self)
    }
}

impl Detach for DelayState {
    fn detach(self) -> Self {
        DelayState::detach(self)
    }
}

// ---------------------------------------------------------------------
// Leaky
// ---------------------------------------------------------------------

impl Stateful<Leaky, NeuronState> {
    /// Wrap [`Leaky`] with zero initial state (`syn = None`,
    /// as in explicit-state `NeuronState::zeros_mem_only`).
    pub fn new_neuron(layer: Leaky, device: &Device, batch_size: usize, units: usize) -> Self {
        let state = NeuronState::zeros_mem_only(device, batch_size, units);
        Self::from_parts(layer, state)
    }

    /// Analogue of `init_hidden()` in snnTorch: recreate the state with zeros,
    /// without touching the layer parameters. Call between episodes/sequences.
    pub fn reset_hidden(&self, device: &Device, batch_size: usize, units: usize) {
        self.set_hidden(NeuronState::zeros_mem_only(device, batch_size, units));
    }

    /// One step in time. The state is taken and updated internally -
    /// the calling code does not need to send or receive anything back,
    /// as in explicit-state [`Leaky::step`](crate::neuron::Leaky::step).
    pub fn step<Su: Surrogate + ?Sized>(&self, input: Tensor<2>, surrogate: &Su) -> Tensor<2> {
        let state = self.state.borrow();
        let (spike, new_state) = self.layer.step(input, &state, surrogate);
        drop(state);
        *self.state.borrow_mut() = new_state;
        spike
    }
}

// ---------------------------------------------------------------------
// Synaptic/Alpha (same state projectile: NeuronState with syn = Some)
// ---------------------------------------------------------------------

impl Stateful<Synaptic, NeuronState> {
    /// Wrap [`Synaptic`] with zero initial state (membrane and
    /// synaptic current, as in `NeuronState::zeros`).
    pub fn new_neuron(layer: Synaptic, device: &Device, batch_size: usize, units: usize) -> Self {
        let state = NeuronState::zeros(device, batch_size, units);
        Self::from_parts(layer, state)
    }

    /// Analogous to `init_hidden()`: recreate the state with zeros.
    pub fn reset_hidden(&self, device: &Device, batch_size: usize, units: usize) {
        self.set_hidden(NeuronState::zeros(device, batch_size, units));
    }

    /// One step in time, state inside.
    pub fn step<Su: Surrogate + ?Sized>(&self, input: Tensor<2>, surrogate: &Su) -> Tensor<2> {
        let state = self.state.borrow();
        let (spike, new_state) = self.layer.step(input, &state, surrogate);
        drop(state);
        *self.state.borrow_mut() = new_state;
        spike
    }
}

impl Stateful<Alpha, NeuronState> {
    /// Wrap [`Alpha`] with a zero initial state.
    pub fn new_neuron(layer: Alpha, device: &Device, batch_size: usize, units: usize) -> Self {
        let state = NeuronState::zeros(device, batch_size, units);
        Self::from_parts(layer, state)
    }

    /// Analogous to `init_hidden()`: recreate the state with zeros.
    pub fn reset_hidden(&self, device: &Device, batch_size: usize, units: usize) {
        self.set_hidden(NeuronState::zeros(device, batch_size, units));
    }

    /// One step in time, state inside.
    pub fn step<Su: Surrogate + ?Sized>(&self, input: Tensor<2>, surrogate: &Su) -> Tensor<2> {
        let state = self.state.borrow();
        let (spike, new_state) = self.layer.step(input, &state, surrogate);
        drop(state);
        *self.state.borrow_mut() = new_state;
        spike
    }
}

// ---------------------------------------------------------------------
// Lapicque (NeuronState, syn is always None - as in explicit-state mode)
// ---------------------------------------------------------------------

impl Stateful<Lapicque, NeuronState> {
    /// Wrap [`Lapicque`] with a zero initial state.
    pub fn new_neuron(layer: Lapicque, device: &Device, batch_size: usize, units: usize) -> Self {
        let state = NeuronState::zeros_mem_only(device, batch_size, units);
        Self::from_parts(layer, state)
    }

    /// Analogous to `init_hidden()`: recreate the state with zeros.
    pub fn reset_hidden(&self, device: &Device, batch_size: usize, units: usize) {
        self.set_hidden(NeuronState::zeros_mem_only(device, batch_size, units));
    }

    /// One step in time, state inside.
    pub fn step<Su: Surrogate + ?Sized>(&self, input: Tensor<2>, surrogate: &Su) -> Tensor<2> {
        let state = self.state.borrow();
        let (spike, new_state) = self.layer.step(input, &state, surrogate);
        drop(state);
        *self.state.borrow_mut() = new_state;
        spike
    }
}

// ---------------------------------------------------------------------
// RLeaky / RSynaptic - recurrent: you also need spike_prev between steps.
// It is stored as a separate field, and not squeezed into NeuronState (which has
// there is no place for it and which is also used by non-recurrent neurons).
// ---------------------------------------------------------------------

/// Stateful wrapper for recurrent neurons ([`RLeaky`], [`RSynaptic`]):
/// on top of the membrane/synaptic current it also stores `spike_prev`,
/// supplied to the recurrent input of the next step.
pub struct StatefulRecurrent<L> {
    /// Wrapped layer.
    pub layer: L,
    state: RefCell<NeuronState>,
    spike_prev: RefCell<Tensor<2>>,
}

impl<L> StatefulRecurrent<L> {
    fn from_parts(layer: L, state: NeuronState, spike_prev: Tensor<2>) -> Self {
        Self {
            layer,
            state: RefCell::new(state),
            spike_prev: RefCell::new(spike_prev),
        }
    }

    /// Analogue of `init_hidden()`: recreate membrane/current and `spike_prev`
    /// zeros. `out_features` — layer output size (coincides with the number
    /// units of the membrane and with dimension `spike_prev`).
    pub fn reset_hidden(&self, device: &Device, batch_size: usize, out_features: usize) {
        *self.state.borrow_mut() = NeuronState::zeros_mem_only(device, batch_size, out_features);
        *self.spike_prev.borrow_mut() = Tensor::zeros([batch_size, out_features], device);
    }

    /// Detach the state (membrane and `spike_prev`) from the autodiff graph
    /// (truncated BPTT) - the same case as [`Stateful::detach_hidden`],
    /// duplicated here because `StatefulRecurrent` stores state
    /// two fields, not one `S: Detach`.
    ///
    /// `Tensor::clone` is a clone of an Arc wrapper over a buffer
    /// (see `burn::tensor::Tensor` documentation) rather than copying
    /// data, so cloning here before `detach()` does not create
    /// unnecessary tensor allocation.
    pub fn detach_hidden(&self) {
        let detached_state = self.state.borrow().clone().detach();
        *self.state.borrow_mut() = detached_state;

        let detached_spike = self.spike_prev.borrow().clone().detach();
        *self.spike_prev.borrow_mut() = detached_spike;
    }
}

impl StatefulRecurrent<RLeaky> {
    /// Wrap [`RLeaky`] with zero initial state and `spike_prev`.
    pub fn new(layer: RLeaky, device: &Device, batch_size: usize, out_features: usize) -> Self {
        let state = NeuronState::zeros_mem_only(device, batch_size, out_features);
        let spike_prev = Tensor::zeros([batch_size, out_features], device);
        Self::from_parts(layer, state, spike_prev)
    }

    /// One time step: recurrent input (`spike_prev`) and state
    /// are taken and updated internally, the calling code just needs to pass
    /// only `input` and surrogate - like `spk, mem = rlif(x)` with
    /// `init_hidden=True` in snnTorch.
    pub fn step<Su: Surrogate + ?Sized>(&self, input: Tensor<2>, surrogate: &Su) -> Tensor<2> {
        let state = self.state.borrow();
        let spike_prev = self.spike_prev.borrow();
        let (spike, new_state) = self
            .layer
            .step(input, spike_prev.clone(), &state, surrogate);
        drop(state);
        drop(spike_prev);
        *self.state.borrow_mut() = new_state;
        *self.spike_prev.borrow_mut() = spike.clone();
        spike
    }
}

impl StatefulRecurrent<RSynaptic> {
    /// Wrap [`RSynaptic`] with zero initial state and `spike_prev`.
    ///
    /// Unlike [`RLeaky`], synaptic current is needed: state
    /// is built through `NeuronState::zeros`, not `zeros_mem_only`.
    pub fn new(layer: RSynaptic, device: &Device, batch_size: usize, out_features: usize) -> Self {
        let state = NeuronState::zeros(device, batch_size, out_features);
        let spike_prev = Tensor::zeros([batch_size, out_features], device);
        Self::from_parts(layer, state, spike_prev)
    }

    /// One time step, recurrent input and state - inside.
    pub fn step<Su: Surrogate + ?Sized>(&self, input: Tensor<2>, surrogate: &Su) -> Tensor<2> {
        let state = self.state.borrow();
        let spike_prev = self.spike_prev.borrow();
        let (spike, new_state) = self
            .layer
            .step(input, spike_prev.clone(), &state, surrogate);
        drop(state);
        drop(spike_prev);
        *self.state.borrow_mut() = new_state;
        *self.spike_prev.borrow_mut() = spike.clone();
        spike
    }
}

// ---------------------------------------------------------------------
// SConv2dLSTM
// ---------------------------------------------------------------------

impl Stateful<SConv2dLSTM, ConvLstmState> {
    /// Wrap [`SConv2dLSTM`](SConv2dLSTM) with null
    /// initial state for a given shape of the spatial map.
    pub fn new_conv_lstm(
        layer: SConv2dLSTM,
        device: &Device,
        batch: usize,
        hidden_channels: usize,
        height: usize,
        width: usize,
    ) -> Self {
        let state = ConvLstmState::zeros(device, batch, hidden_channels, height, width);
        Self::from_parts(layer, state)
    }

    /// Analogue of `init_hidden()`: recreate `h`/`c` with zeros for the same or
    /// new map shape.
    pub fn reset_hidden(
        &self,
        device: &Device,
        batch: usize,
        hidden_channels: usize,
        height: usize,
        width: usize,
    ) {
        self.set_hidden(ConvLstmState::zeros(
            device,
            batch,
            hidden_channels,
            height,
            width,
        ));
    }

    /// One time step, state `(h, c)` inside.
    pub fn step<Su: Surrogate + ?Sized>(&self, input: Tensor<4>, surrogate: &Su) -> Tensor<4> {
        let state = self.state.borrow();
        let (spike, new_state) = self.layer.step(input, &state, surrogate);
        drop(state);
        *self.state.borrow_mut() = new_state;
        spike
    }
}

// ---------------------------------------------------------------------
// SpikingConv2d - NeuronState, but with the shape [batch*h*w, c_out]
// (see SpikingConv2d::state_zeros), therefore it is recreated through the same
// layer function rather than directly NeuronState::zeros_mem_only.
// ---------------------------------------------------------------------

impl Stateful<SpikingConv2d, NeuronState> {
    /// Wrap [`SpikingConv2d`](SpikingConv2d) with null
    /// initial state. `(h, w)` - spatial dimensions of the output
    /// AFTER convolution (see `SpikingConv2d::state_zeros`).
    pub fn new_spiking_conv2d(
        layer: SpikingConv2d,
        device: &Device,
        batch: usize,
        c_out: usize,
        h: usize,
        w: usize,
    ) -> Self {
        let state = SpikingConv2d::state_zeros(device, batch, c_out, h, w);
        Self::from_parts(layer, state)
    }

    /// Analogue of `init_hidden()`: recreate the state with zeros for the same or
    /// new output shape.
    pub fn reset_hidden(&self, device: &Device, batch: usize, c_out: usize, h: usize, w: usize) {
        self.set_hidden(SpikingConv2d::state_zeros(device, batch, c_out, h, w));
    }

    /// One step in time, state inside.
    pub fn step<Su: Surrogate + ?Sized>(&self, input: Tensor<4>, surrogate: &Su) -> Tensor<4> {
        let state = self.state.borrow();
        let (spike, new_state) = self.layer.step(input, &state, surrogate);
        drop(state);
        *self.state.borrow_mut() = new_state;
        spike
    }
}

// ---------------------------------------------------------------------
// SynapticDelay - DelayState (ring buffer), without surrogate:
// The layer itself does not spike.
// ---------------------------------------------------------------------

impl Stateful<SynapticDelay, DelayState> {
    /// Wrap [`SynapticDelay`](SynapticDelay) with
    /// zero (cold start) ring buffer.
    pub fn new_delay(layer: SynapticDelay, device: &Device, batch: usize) -> Self {
        let channels = layer.channels();
        let max_delay = layer.max_delay();
        let state = DelayState::zeros(device, batch, channels, max_delay);
        Self::from_parts(layer, state)
    }

    /// Analog `init_hidden()`: reset the ring buffer (cold start
    /// new sequence).
    pub fn reset_hidden(&self, device: &Device, batch: usize) {
        let channels = self.layer.channels();
        let max_delay = self.layer.max_delay();
        self.set_hidden(DelayState::zeros(device, batch, channels, max_delay));
    }

    /// One time step, ring buffer inside.
    pub fn step(&self, input: Tensor<2>) -> Tensor<2> {
        let state = self.state.borrow();
        let (out, new_state) = self.layer.step(input, &state);
        drop(state);
        *self.state.borrow_mut() = new_state;
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::Device;

    fn dev() -> Device {
        Device::flex()
    }

    fn t2(data: Vec<f32>, shape: [usize; 2]) -> Tensor<2> {
        Tensor::<1>::from_floats(data.as_slice(), &dev()).reshape(shape)
    }

    fn vec2(t: Tensor<2>) -> Vec<f32> {
        t.into_data().try_to_vec::<f32>().unwrap()
    }

    /// `Stateful<Leaky, _>::step` without explicit state transfer is obliged
    /// give EXACTLY the same result as explicit-state `Leaky::step`
    /// in a manual loop - stateful mode should not be separate
    /// implementation of dynamics, just a convenient wrapper over the same.
    #[test]
    fn stateful_leaky_matches_explicit_state_loop() {
        use crate::neuron::LeakyConfig;
        use crate::surrogate::FastSigmoid;

        let device = dev();
        let surrogate = FastSigmoid::default();
        let config = LeakyConfig::new(0.85).with_threshold(0.5);

        let inputs = [
            t2(vec![0.3, 0.1], [1, 2]),
            t2(vec![0.6, 0.0], [1, 2]),
            t2(vec![0.1, 0.9], [1, 2]),
            t2(vec![0.0, 0.0], [1, 2]),
        ];

        // Explicit-state: manual cycle.
        let explicit_layer = config.init(&device);
        let mut explicit_state = NeuronState::zeros_mem_only(&device, 1, 2);
        let mut explicit_out = Vec::new();
        for x in inputs.clone() {
            let (spike, new_state) = explicit_layer.step(x, &explicit_state, &surrogate);
            explicit_state = new_state;
            explicit_out.push(vec2(spike));
        }

        // Stateful: init_hidden-mode.
        let stateful_layer = config.init(&device);
        let stateful = Stateful::<Leaky, NeuronState>::new_neuron(stateful_layer, &device, 1, 2);
        let mut stateful_out = Vec::new();
        for x in inputs {
            let spike = stateful.step(x, &surrogate);
            stateful_out.push(vec2(spike));
        }

        for (t, (e, s)) in explicit_out.iter().zip(stateful_out.iter()).enumerate() {
            for (i, (&a, &b)) in e.iter().zip(s.iter()).enumerate() {
                assert!(
                    (a - b).abs() < 1e-6,
                    "step {t}, element {i}: explicit={a} stateful={b}"
                );
            }
        }
    }

    /// `reset_hidden` is obliged to return the layer to the zero state - after
    /// reset behavior must coincide with the first call on the fresh layer,
    /// regardless of what happened before the reset (analogous to `init_hidden()`
    /// between episodes in snnTorch).
    #[test]
    fn reset_hidden_returns_to_zero_state() {
        use crate::neuron::LeakyConfig;
        use crate::surrogate::FastSigmoid;

        let device = dev();
        let surrogate = FastSigmoid::default();
        let config = LeakyConfig::new(0.85).with_threshold(0.5);

        let layer = config.init(&device);
        let stateful = Stateful::<Leaky, NeuronState>::new_neuron(layer, &device, 1, 2);

        // Let's run a few steps until the state becomes non-zero.
        let _ = stateful.step(t2(vec![0.9, 0.9], [1, 2]), &surrogate);
        let _ = stateful.step(t2(vec![0.9, 0.9], [1, 2]), &surrogate);
        assert!(
            stateful
                .hidden()
                .mem
                .into_data()
                .try_to_vec::<f32>()
                .unwrap()[0]
                .abs()
                > 1e-6
        );

        stateful.reset_hidden(&device, 1, 2);
        let after_reset = stateful.hidden();
        let mem = after_reset.mem.into_data().try_to_vec::<f32>().unwrap();
        assert!(
            mem.iter().all(|&v| v.abs() < 1e-6),
            "state after reset_hidden should be zero: {mem:?}"
        );
    }

    /// `StatefulRecurrent<RLeaky, _>` must give the same result as
    /// explicit-state `RLeaky::step` in a manual loop with yourself
    /// dragged by `spike_prev`.
    ///
    /// IMPORTANT: both layers are built manually from the same
    /// weights/bias/beta tensors rather than via `layer.clone()`, because
    /// `Module::clone` for `Linear` does not guarantee a bitwise match of the
    /// `bias` parameters in the burn version used here (`weight` and `beta`
    /// clone correctly). The test then checks exactly what its name states -
    /// that the stateful wrapper does not change the dynamics of the layer -
    /// without relying on module cloning.
    #[test]
    fn stateful_recurrent_rleaky_matches_explicit_state_loop() {
        use crate::state::ResetMode;
        use crate::surrogate::FastSigmoid;
        use burn::module::Param;
        use burn::nn::LinearConfig;

        let device = dev();
        let surrogate = FastSigmoid::default();

        // We build one layer in the usual way to get random,
        // but specific weights...
        let seed_layer = RLeaky::new(&device, 2, 2, 0.85, 0.5, ResetMode::Subtract);

        // ...and explicitly copy its tensors (not the Module itself) into two new ones,
        // independently designed layer. Tensor::clone() here -
        // this is cloning an Arc wrapper over a buffer (see documentation
        // burn::tensor::Tensor), that is, the same data is guaranteed,
        // unlike Module::clone(), whose correctness is under question here
        // question.
        let w_in = seed_layer.linear_in.weight.val();
        let b_in = seed_layer
            .linear_in
            .bias
            .as_ref()
            .expect("linear_in must be with bias")
            .val();
        let w_rec = seed_layer.linear_rec.weight.val();
        let beta_raw = seed_layer.beta.val();

        let build_layer = || -> RLeaky {
            let mut linear_in = LinearConfig::new(2, 2).init(&device);
            linear_in.weight = Param::from_tensor(w_in.clone());
            linear_in.bias = Some(Param::from_tensor(b_in.clone()));

            let mut linear_rec = LinearConfig::new(2, 2).with_bias(false).init(&device);
            linear_rec.weight = Param::from_tensor(w_rec.clone());

            RLeaky {
                linear_in,
                linear_rec,
                beta: Param::from_tensor(beta_raw.clone()),
                threshold: 0.5,
                reset_mode: ResetMode::Subtract,
                reset_order: crate::state::ResetOrder::BeforeDecay,
            }
        };

        let explicit_layer: RLeaky = build_layer();
        let stateful_layer: RLeaky = build_layer();

        // We confirm at the start of the run itself that both layers are identical -
        // if this ever stops being the case, the test will fail here,
        // and not drown in the difference in numbers somewhere in the middle of the time series.
        assert_eq!(
            explicit_layer
                .linear_in
                .weight
                .val()
                .into_data()
                .try_to_vec::<f32>()
                .unwrap(),
            stateful_layer
                .linear_in
                .weight
                .val()
                .into_data()
                .try_to_vec::<f32>()
                .unwrap(),
            "linear_in.weight must match between explicit and stateful layers"
        );
        assert_eq!(
            explicit_layer
                .linear_in
                .bias
                .as_ref()
                .unwrap()
                .val()
                .into_data()
                .try_to_vec::<f32>()
                .unwrap(),
            stateful_layer
                .linear_in
                .bias
                .as_ref()
                .unwrap()
                .val()
                .into_data()
                .try_to_vec::<f32>()
                .unwrap(),
            "linear_in.bias must match between explicit and stateful layers"
        );

        let inputs = [
            t2(vec![0.3, 0.1], [1, 2]),
            t2(vec![0.6, 0.0], [1, 2]),
            t2(vec![0.1, 0.9], [1, 2]),
        ];

        // Explicit-state: manual loop, spike_prev is dragged manually.
        let mut explicit_state = NeuronState::zeros_mem_only(&device, 1, 2);
        let mut spike_prev = Tensor::<2>::zeros([1, 2], &device);
        let mut explicit_out = Vec::new();
        for x in inputs.clone() {
            let (spike, new_state) =
                explicit_layer.step(x, spike_prev.clone(), &explicit_state, &surrogate);
            explicit_state = new_state;
            spike_prev = spike.clone();
            explicit_out.push(vec2(spike));
        }

        // Stateful: init_hidden mode, spike_prev is dragged in by itself.
        let stateful = StatefulRecurrent::<RLeaky>::new(stateful_layer, &device, 1, 2);
        let mut stateful_out = Vec::new();
        for x in inputs {
            let spike = stateful.step(x, &surrogate);
            stateful_out.push(vec2(spike));
        }

        for (t, (e, s)) in explicit_out.iter().zip(stateful_out.iter()).enumerate() {
            for (i, (&a, &b)) in e.iter().zip(s.iter()).enumerate() {
                assert!(
                    (a - b).abs() < 1e-6,
                    "step {t}, element {i}: explicit={a} stateful={b}"
                );
            }
        }
    }

    /// `Stateful<SynapticDelay, _>` must give the same result as
    /// explicit-state `SynapticDelay::step` in manual loop with ring
    /// buffer.
    #[test]
    fn stateful_synaptic_delay_matches_explicit_state_loop() {
        use crate::layers::SynapticDelayConfig;

        let device = dev();
        let config = SynapticDelayConfig::new(1, 4)
            .with_delay_init(2.0)
            .with_learn_delay(false);

        let inputs = [1.0f32, 0.0, 0.0, 0.0, 0.0];

        let explicit_layer = config.init(&device);
        let mut explicit_state = DelayState::zeros(&device, 1, 1, explicit_layer.max_delay());
        let mut explicit_out = Vec::new();
        for &x in &inputs {
            let (out, new_state) = explicit_layer.step(t2(vec![x], [1, 1]), &explicit_state);
            explicit_state = new_state;
            explicit_out.push(vec2(out)[0]);
        }

        let stateful_layer = config.init(&device);
        let stateful = Stateful::new_delay(stateful_layer, &device, 1);
        let mut stateful_out = Vec::new();
        for &x in &inputs {
            let out = stateful.step(t2(vec![x], [1, 1]));
            stateful_out.push(vec2(out)[0]);
        }

        for (i, (&a, &b)) in explicit_out.iter().zip(stateful_out.iter()).enumerate() {
            assert!((a - b).abs() < 1e-6, "step {i}: explicit={a} stateful={b}");
        }
    }
}
