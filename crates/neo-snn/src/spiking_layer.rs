//! Single trace contract for layers with state: [`SpikingLayer`].
//!
//! # Why this module
//!
//! Before it, the state of the layers lived in three incompatible ways:
//!
//! 1. **obviously** - `layer.step(x, &state, surrogate) -> (y, state)`, for everyone
//!    the layer has its own signature and its own type of state;
//! 2. **inside the wrapper** — [`Stateful`](crate::stateful::Stateful) /
//!    [`StatefulRecurrent`](crate::stateful::StatefulRecurrent) with `RefCell`,
//!    a separate `impl` for each layer;
//! 3. **inside the state itself** — [`StdpState`](crate::layers::StdpState)
//!    and [`StdpConv1dState`](crate::layers::StdpConv1dState) carry WEIGHTS, and
//!    `step` changes them on the fly.
//!
//! There was no common denominator, so it was impossible to write code generalized by
//! layer (layer stack, training loop, TBPTT wrapper), without type enumeration
//! hands. Now all three modes are reduced to one trait:
//!
//! ```text
//! init_state(batch, device) -> State
//! step(x, State)            -> (y, State)
//! forward_seq(x_seq)        -> Vec<y>
//! ```
//!
//! * mode 1 is the `step` trait itself (the state is passed by value);
//! * mode 2 is [`Session`], ONE wrapper instead of a dozen `impl`s;
//! * mode 3 – `StdpLinear::State = StdpState`, weights remain part
//!   states and experience `detach`/`step` in exactly the same way as the membrane.
//!
//! The old API (`Leaky::step(x, &state, s)`, `Stateful`, ...) is NOT touched and
//! continues to work: the trace is added on top.
//!
//! # Differences from the “bare” `step(x, state)`
//!
//! * **Surrogate.** All crate spike layers accept a surrogate
//!   gradient, so `step`/`forward_seq` get an extra argument
//!   `surrogate`. Layers that don't need it (`SynapticDelay`), its
//!   ignored.
//! * **State shape - lazy.** Element-wise neurons (`Leaky`, `Synaptic`,
//!   `Alpha`, `Lapicque`, `AdaptiveLeaky`, `Izhikevich`) do not know their own
//!   width, and convolutional layers have spatial dimensions, so
//!   `init_state(batch, device)` for these returns [`Deferred::Pending`], and
//!   real tensors are created on the first `step` according to the input shape.
//!   Layers whose shape is known (`RLeaky`, `StdpLinear`,
//!   `SynapticDelay`), deferral is not needed.
//! * **Input and output are associated types.** Layers work with tensors
//!   of different ranks (`[B,N]`, `[B,C,L]`, `[B,C,H,W]`), one `Tensor<2>`
//!   you can't cover them.
//! * **`forward_seq` takes `Vec` steps**, not a tensor with a time axis:
//!   “rank + 1” cannot be expressed in stable Rust. For the most frequent
//!   case `[batch, time, features]` is [`SpikingLayerExt`].
//!
//! # Example
//!
//! ```ignore
//! use neo_snn::prelude::*;
//!
//! // One function - any layer.
//! fn run<L: SpikingLayer>(
//!     layer: &L, xs: Vec<L::Input>, s: &impl Surrogate,
//! ) -> Vec<L::Output> {
//!     layer.forward_seq(xs, s)
//! }
//!
//! // init_hidden mode for ANY layer:
//! let lif = Session::new(LeakyConfig::new(0.9).init(&device), batch, &device);
//! let spikes = lif.step(x_t, &surrogate);
//! lif.detach_hidden();      // truncated BPTT
//! lif.reset(batch, &device); // new sequence
//! ```

use burn::tensor::Device;
use std::cell::RefCell;

use burn::tensor::Tensor;

use crate::layers::{
    ConvLstmState, DelayState, SConv2dLSTM, SpikingConv2d, StdpConv1d, StdpConv1dState, StdpLinear,
    StdpState, SynapticDelay,
};
use crate::neuron::{
    AdEx, AdExState, AdaptiveLeaky, Alpha, GradedLeaky, HodgkinHuxley, HodgkinHuxleyState,
    Izhikevich, IzhikevichState, Lapicque, Leaky, RLeaky, RSynaptic, Synaptic,
};
use crate::state::NeuronState;
use crate::stateful::Detach;
use crate::surrogate::Surrogate;

// ---------------------------------------------------------------------
// Step input
// ---------------------------------------------------------------------

/// Here's what you need to know about a step's input to create state for it:
/// batch size and device. Implemented for any `Tensor<D>`.
pub trait StepInput {
    /// Batch size (zero axis).
    fn batch_size(&self) -> usize;
    /// The device on which the input lies.
    fn input_device(&self) -> Device;
}

impl<const D: usize> StepInput for Tensor<D> {
    fn batch_size(&self) -> usize {
        self.dims()[0]
    }

    fn input_device(&self) -> Device {
        self.device()
    }
}

// ---------------------------------------------------------------------
// Trait
// ---------------------------------------------------------------------

/// Single layer contract with state. See module documentation.
pub trait SpikingLayer {
    /// What the layer receives at each step (`Tensor<2>`, `Tensor<4>`, ...).
    type Input: StepInput;
    /// What the layer reveals at each step.
    type Output;
    /// Everything that the layer transfers between steps: membrane, currents, traces,
    /// ring buffer - and STDP layers also have weights.
    type State: Clone + Detach;

    /// Initial state for batch `batch`.
    ///
    /// If the shape of the layer does not determine the size of the entire state (width
    /// element-wise neuron, spatial convolution size), returns
    /// [`Deferred::Pending`] - tensors will be created on the first `step`.
    ///
    /// For STDP layers, the weights are initialized randomly (according to `init_range` from
    /// configurations): they are part of the state. To start with given weights,
    /// collect the state via `StdpState::with_weights` and transfer it
    /// in `step`/`forward_seq_from`.
    fn init_state(&self, batch: usize, device: &Device) -> Self::State;

    /// One time step: `(x, state) -> (y, state')`.
    ///
    /// State is passed by value - a chain of steps cannot do this
    /// accidentally use outdated state.
    fn step<S: Surrogate + ?Sized>(
        &self,
        x: Self::Input,
        state: Self::State,
        surrogate: &S,
    ) -> (Self::Output, Self::State);

    /// Running the entire sequence from the zero state.
    ///
    /// The batch and device are taken from the first element `x_seq`. Returns
    /// outputs of all steps; if you need the final state, use
    /// [`forward_seq_from`](Self::forward_seq_from).
    ///
    /// # Panics
    /// If `x_seq` is empty.
    fn forward_seq<S: Surrogate + ?Sized>(
        &self,
        x_seq: Vec<Self::Input>,
        surrogate: &S,
    ) -> Vec<Self::Output> {
        assert!(
            !x_seq.is_empty(),
            "SpikingLayer::forward_seq: sequence must not be empty"
        );
        let state = self.init_state(x_seq[0].batch_size(), &x_seq[0].input_device());
        self.forward_seq_from(x_seq, state, surrogate).0
    }

    /// Running a sequence from a given state; returns outputs
    /// and final state. This is what is needed for TBPTT: drive away the chunk,
    /// `detach()` state, pass to the next chunk.
    fn forward_seq_from<S: Surrogate + ?Sized>(
        &self,
        x_seq: Vec<Self::Input>,
        state: Self::State,
        surrogate: &S,
    ) -> (Vec<Self::Output>, Self::State) {
        let mut state = state;
        let mut out = Vec::with_capacity(x_seq.len());
        for x in x_seq {
            let (y, next) = self.step(x, state, surrogate);
            out.push(y);
            state = next;
        }
        (out, state)
    }
}

// ---------------------------------------------------------------------
// [batch, time, features] - the most common case
// ---------------------------------------------------------------------

/// Convenience for layers that have both an input and an output - `Tensor<2>`:
/// running the tensor `[batch, time, features]` (same layout as
/// `MultiStepLeaky::forward` and `SynapticDelay::forward_sequence`).
pub trait SpikingLayerExt: SpikingLayer<Input = Tensor<2>, Output = Tensor<2>> {
    /// Like [`SpikingLayer::forward_seq`], but input and output are tensors
    /// `[batch, time, features]`.
    fn forward_seq_tensor<S: Surrogate + ?Sized>(&self, x: Tensor<3>, surrogate: &S) -> Tensor<3> {
        let [batch, _, _] = x.dims();
        let state = self.init_state(batch, &x.device());
        self.forward_seq_tensor_from(x, state, surrogate).0
    }

    /// Like [`SpikingLayer::forward_seq_from`] for tensor
    /// `[batch, time, features]`.
    fn forward_seq_tensor_from<S: Surrogate + ?Sized>(
        &self,
        x: Tensor<3>,
        state: Self::State,
        surrogate: &S,
    ) -> (Tensor<3>, Self::State) {
        let [_batch, steps, _features] = x.dims();
        assert!(
            steps > 0,
            "SpikingLayerExt::forward_seq_tensor: num_steps must be > 0"
        );
        // `split_time`: backward O(T·√T) instead of O(T²) for naive slices.
        let xs: Vec<Tensor<2>> = crate::neuron::split_time(x);
        let (ys, state) = self.forward_seq_from(xs, state, surrogate);
        (Tensor::stack::<3>(ys, 1), state)
    }
}

impl<L> SpikingLayerExt for L where L: SpikingLayer<Input = Tensor<2>, Output = Tensor<2>> {}

// ---------------------------------------------------------------------
// Deferred is a state whose shape is recognized at the first step
// ---------------------------------------------------------------------

/// A state whose tensors have not yet been created.
///
/// `Pending` stores only the batch size; on the first `step` layer is completed
/// `Ready(S)` by input shape. After the first step the state is always `Ready`.
#[derive(Debug, Clone)]
pub enum Deferred<S> {
    /// Tensors have not yet been created.
    Pending {
        /// The size of the batch for which the state was created.
        batch: usize,
    },
    /// Ready condition.
    Ready(S),
}

impl<S> Deferred<S> {
    /// Deferred state for batch `batch`.
    pub fn pending(batch: usize) -> Self {
        Deferred::Pending { batch }
    }

    /// Already ready state (for example, restored from a checkpoint or
    /// assembled by hand with the necessary weights).
    pub fn ready(state: S) -> Self {
        Deferred::Ready(state)
    }

    /// `true`, if tensors have not yet been created.
    pub fn is_pending(&self) -> bool {
        matches!(self, Deferred::Pending { .. })
    }

    /// Access to the finished state.
    pub fn get(&self) -> Option<&S> {
        match self {
            Deferred::Ready(s) => Some(s),
            Deferred::Pending { .. } => None,
        }
    }

    /// Pick up the finished condition.
    pub fn into_ready(self) -> Option<S> {
        match self {
            Deferred::Ready(s) => Some(s),
            Deferred::Pending { .. } => None,
        }
    }

    /// Complete the state: for `Pending` calls `build(batch)`,
    /// for `Ready` returns it as is.
    pub fn resolve(self, build: impl FnOnce(usize) -> S) -> S {
        match self {
            Deferred::Ready(s) => s,
            Deferred::Pending { batch } => build(batch),
        }
    }
}

impl<S> From<S> for Deferred<S> {
    fn from(state: S) -> Self {
        Deferred::Ready(state)
    }
}

impl<S: Detach> Detach for Deferred<S> {
    fn detach(self) -> Self {
        match self {
            Deferred::Ready(s) => Deferred::Ready(s.detach()),
            pending => pending,
        }
    }
}

// ---------------------------------------------------------------------
// State of recurrent neurons
// ---------------------------------------------------------------------

/// State of the recurrent neuron: membrane/current and spikes of the previous step,
/// which go to the recurrent input. Keeping `spike_prev` here means it does not
/// need to be passed as a separate argument (or stored in a separate field of
/// `StatefulRecurrent`).
#[derive(Debug, Clone)]
pub struct RecurrentState {
    /// Membrane (and synaptic current for `RSynaptic`).
    pub neuron: NeuronState,
    /// Spikes of the previous step `[batch, out_features]`.
    pub spike_prev: Tensor<2>,
}

impl RecurrentState {
    /// Detach from the auto-differentiation graph.
    pub fn detach(self) -> Self {
        Self {
            neuron: self.neuron.detach(),
            spike_prev: self.spike_prev.detach(),
        }
    }
}

// ---------------------------------------------------------------------
// Detach for state types that didn't have one
// ---------------------------------------------------------------------

impl Detach for RecurrentState {
    fn detach(self) -> Self {
        RecurrentState::detach(self)
    }
}

impl Detach for IzhikevichState {
    fn detach(self) -> Self {
        IzhikevichState::detach(self)
    }
}

impl Detach for AdExState {
    fn detach(self) -> Self {
        AdExState::detach(self)
    }
}

impl Detach for HodgkinHuxleyState {
    fn detach(self) -> Self {
        HodgkinHuxleyState::detach(self)
    }
}

impl Detach for StdpState {
    fn detach(self) -> Self {
        StdpState::detach(self)
    }
}

impl Detach for StdpConv1dState {
    fn detach(self) -> Self {
        StdpConv1dState::detach(self)
    }
}

// ---------------------------------------------------------------------
// Element-wise neurons: state shape = input shape
// ---------------------------------------------------------------------

macro_rules! elementwise_neuron {
    ($ty:ident, $zeros:ident) => {
        impl SpikingLayer for $ty {
            type Input = Tensor<2>;
            type Output = Tensor<2>;
            type State = Deferred<NeuronState>;

            fn init_state(&self, batch: usize, _device: &Device) -> Self::State {
                Deferred::pending(batch)
            }

            fn step<S: Surrogate + ?Sized>(
                &self,
                x: Tensor<2>,
                state: Self::State,
                surrogate: &S,
            ) -> (Tensor<2>, Self::State) {
                let [batch, units] = x.dims();
                let device = x.device();
                let state = state.resolve(|b| {
                    assert_eq!(
                        b,
                        batch,
                        "{}: the input batch ({batch}) did not match the init_state batch ({b})",
                        stringify!($ty)
                    );
                    NeuronState::$zeros(&device, batch, units)
                });
                let (y, next) = $ty::step(self, x, &state, surrogate);
                (y, Deferred::Ready(next))
            }
        }
    };
}

elementwise_neuron!(Leaky, zeros_mem_only);
elementwise_neuron!(Lapicque, zeros_mem_only);
// `syn` for AdaptiveLeaky - adaptation current; `None` is already interpreted as 0,
// but `zeros` gives a state of the same shape as after the first step.
elementwise_neuron!(AdaptiveLeaky, zeros);
elementwise_neuron!(Synaptic, zeros);
elementwise_neuron!(Alpha, zeros);
// Multi-level/ternary LIF: state - membrane only.
elementwise_neuron!(GradedLeaky, zeros_mem_only);

impl SpikingLayer for AdEx {
    type Input = Tensor<2>;
    type Output = Tensor<2>;
    type State = Deferred<AdExState>;

    fn init_state(&self, batch: usize, _device: &Device) -> Self::State {
        Deferred::pending(batch)
    }

    fn step<S: Surrogate + ?Sized>(
        &self,
        x: Tensor<2>,
        state: Self::State,
        surrogate: &S,
    ) -> (Tensor<2>, Self::State) {
        let [batch, units] = x.dims();
        let device = x.device();
        let state = state.resolve(|b| {
            assert_eq!(
                b, batch,
                "AdEx: the input batch ({batch}) did not match the init_state batch ({b})"
            );
            AdExState::resting(&device, batch, units, self.e_l)
        });
        let (y, next) = AdEx::step(self, x, &state, surrogate);
        (y, Deferred::Ready(next))
    }
}

impl SpikingLayer for HodgkinHuxley {
    type Input = Tensor<2>;
    type Output = Tensor<2>;
    type State = Deferred<HodgkinHuxleyState>;

    fn init_state(&self, batch: usize, _device: &Device) -> Self::State {
        Deferred::pending(batch)
    }

    fn step<S: Surrogate + ?Sized>(
        &self,
        x: Tensor<2>,
        state: Self::State,
        surrogate: &S,
    ) -> (Tensor<2>, Self::State) {
        let [batch, units] = x.dims();
        let device = x.device();
        let state = state.resolve(|b| {
            assert_eq!(
                b, batch,
                "HodgkinHuxley: the input batch ({batch}) did not match the init_state batch ({b})"
            );
            // Rest of the classical model: −65 mV, gate in stationary values.
            HodgkinHuxleyState::resting(&device, batch, units, -65.0)
        });
        let (y, next) = HodgkinHuxley::step(self, x, &state, surrogate);
        (y, Deferred::Ready(next))
    }
}

impl SpikingLayer for Izhikevich {
    type Input = Tensor<2>;
    type Output = Tensor<2>;
    type State = Deferred<IzhikevichState>;

    fn init_state(&self, batch: usize, _device: &Device) -> Self::State {
        Deferred::pending(batch)
    }

    fn step<S: Surrogate + ?Sized>(
        &self,
        x: Tensor<2>,
        state: Self::State,
        surrogate: &S,
    ) -> (Tensor<2>, Self::State) {
        let [batch, units] = x.dims();
        let device = x.device();
        let state = state.resolve(|b| {
            assert_eq!(
                b, batch,
                "Izhikevich: the input batch ({batch}) did not match the init_state batch ({b})"
            );
            // Rest state `v = c`, `u = 0` - the same contract as
            // `IzhikevichState::resting`, but without reading `c` to the host.
            let v = self.c.val().detach().reshape([1, 1]).expand([batch, units]);
            IzhikevichState::new(v, Tensor::zeros([batch, units], &device))
        });
        let (y, next) = Izhikevich::step(self, x, &state, surrogate);
        (y, Deferred::Ready(next))
    }
}

// ---------------------------------------------------------------------
// Recurrent neurons: width known from linear_rec
// ---------------------------------------------------------------------

macro_rules! recurrent_neuron {
    ($ty:ident, $zeros:ident) => {
        impl SpikingLayer for $ty {
            type Input = Tensor<2>;
            type Output = Tensor<2>;
            type State = RecurrentState;

            fn init_state(&self, batch: usize, device: &Device) -> Self::State {
                // Linear stores the weight as [d_input, d_output].
                let out_features = self.linear_rec.weight.val().dims()[1];
                RecurrentState {
                    neuron: NeuronState::$zeros(device, batch, out_features),
                    spike_prev: Tensor::zeros([batch, out_features], device),
                }
            }

            fn step<S: Surrogate + ?Sized>(
                &self,
                x: Tensor<2>,
                state: Self::State,
                surrogate: &S,
            ) -> (Tensor<2>, Self::State) {
                let (y, neuron) = $ty::step(self, x, state.spike_prev, &state.neuron, surrogate);
                (
                    y.clone(),
                    RecurrentState {
                        neuron,
                        spike_prev: y,
                    },
                )
            }
        }
    };
}

recurrent_neuron!(RLeaky, zeros_mem_only);
recurrent_neuron!(RSynaptic, zeros);

// ---------------------------------------------------------------------
// Compound Layers
// ---------------------------------------------------------------------

impl SpikingLayer for SpikingConv2d {
    type Input = Tensor<4>;
    type Output = Tensor<4>;
    type State = Deferred<NeuronState>;

    fn init_state(&self, batch: usize, _device: &Device) -> Self::State {
        Deferred::pending(batch)
    }

    fn step<S: Surrogate + ?Sized>(
        &self,
        x: Tensor<4>,
        state: Self::State,
        surrogate: &S,
    ) -> (Tensor<4>, Self::State) {
        // The spatial size of the output depends on the kernel/stride/padding, and
        // You can determine it only by running a batch - we do that once.
        let out = self.conv.forward(x);
        let [batch, c_out, h, w] = out.dims();
        let device = out.device();
        let state = state.resolve(|b| {
            assert_eq!(
                b, batch,
                "SpikingConv2d: input batch ({batch}) did not match init_state batch ({b})"
            );
            SpikingConv2d::state_zeros(&device, batch, c_out, h, w)
        });
        let (y, next) = self.step_from_conv_out(out, &state, surrogate);
        (y, Deferred::Ready(next))
    }
}

impl SpikingLayer for SConv2dLSTM {
    type Input = Tensor<4>;
    type Output = Tensor<4>;
    type State = Deferred<ConvLstmState>;

    fn init_state(&self, batch: usize, _device: &Device) -> Self::State {
        Deferred::pending(batch)
    }

    fn step<S: Surrogate + ?Sized>(
        &self,
        x: Tensor<4>,
        state: Self::State,
        surrogate: &S,
    ) -> (Tensor<4>, Self::State) {
        let [batch, _, h, w] = x.dims();
        let device = x.device();
        let hidden = self.hidden_channels;
        // padding `Same` preserves h×w, so the size of the state map is
        // the size of the input map.
        let state = state.resolve(|b| {
            assert_eq!(
                b, batch,
                "SConv2dLSTM: input batch ({batch}) did not match init_state batch ({b})"
            );
            ConvLstmState::zeros(&device, batch, hidden, h, w)
        });
        let (y, next) = SConv2dLSTM::step(self, x, &state, surrogate);
        (y, Deferred::Ready(next))
    }
}

impl SpikingLayer for SynapticDelay {
    type Input = Tensor<2>;
    type Output = Tensor<2>;
    type State = DelayState;

    fn init_state(&self, batch: usize, device: &Device) -> Self::State {
        DelayState::zeros(device, batch, self.channels(), self.max_delay())
    }

    /// The layer is not spiked - `surrogate` is not used.
    fn step<S: Surrogate + ?Sized>(
        &self,
        x: Tensor<2>,
        state: Self::State,
        _surrogate: &S,
    ) -> (Tensor<2>, Self::State) {
        SynapticDelay::step(self, x, &state)
    }
}

// ---------------------------------------------------------------------
// STDP: weights are part of the state
// ---------------------------------------------------------------------

impl SpikingLayer for StdpLinear {
    type Input = Tensor<2>;
    type Output = Tensor<2>;
    /// Weights, traces and membrane in one state: `step` updates the weights
    /// by STDP and returns them inside the new state.
    type State = StdpState;

    fn init_state(&self, batch: usize, device: &Device) -> Self::State {
        StdpState::zeros(self, device, batch)
    }

    fn step<S: Surrogate + ?Sized>(
        &self,
        x: Tensor<2>,
        state: Self::State,
        surrogate: &S,
    ) -> (Tensor<2>, Self::State) {
        StdpLinear::step::<S>(self, x, &state, surrogate)
    }
}

impl SpikingLayer for StdpConv1d {
    type Input = Tensor<3>;
    type Output = Tensor<3>;
    type State = Deferred<StdpConv1dState>;

    fn init_state(&self, batch: usize, _device: &Device) -> Self::State {
        Deferred::pending(batch)
    }

    fn step<S: Surrogate + ?Sized>(
        &self,
        x: Tensor<3>,
        state: Self::State,
        surrogate: &S,
    ) -> (Tensor<3>, Self::State) {
        let [batch, _, length] = x.dims();
        let device = x.device();
        // The shape of the state depends on the length of the input (the number of window positions).
        let state = state.resolve(|b| {
            assert_eq!(
                b, batch,
                "StdpConv1d: input batch ({batch}) did not match init_state batch ({b})"
            );
            self.state_zeros(&device, batch, length)
        });
        let (y, next) = StdpConv1d::step::<S>(self, x, &state, surrogate);
        (y, Deferred::Ready(next))
    }
}

// ---------------------------------------------------------------------
// Session - init_hidden mode for any layer
// ---------------------------------------------------------------------

/// “State inside” wrapper for ANY [`SpikingLayer`] - replacement
/// ten `impl`-blocks [`Stateful`](crate::stateful::Stateful) in one.
///
/// Like `Stateful`, not `Sync`: models one "live" layer instance
/// (`RefCell`).
pub struct Session<L: SpikingLayer> {
    /// Wrapped layer; parameters and explicit API are directly accessible.
    pub layer: L,
    state: RefCell<L::State>,
}

impl<L: SpikingLayer> Session<L> {
    /// Wrap the layer with `layer.init_state(batch, device)`.
    pub fn new(layer: L, batch: usize, device: &Device) -> Self {
        let state = layer.init_state(batch, device);
        Self::from_parts(layer, state)
    }

    /// Wrap a layer with a given state.
    pub fn from_parts(layer: L, state: L::State) -> Self {
        Self {
            layer,
            state: RefCell::new(state),
        }
    }

    /// One step; the state is taken and updated internally.
    pub fn step<S: Surrogate + ?Sized>(&self, x: L::Input, surrogate: &S) -> L::Output {
        let current = self.state.borrow().clone();
        let (y, next) = self.layer.step(x, current, surrogate);
        *self.state.borrow_mut() = next;
        y
    }

    /// Run the sequence, continuing from the internal state
    /// (it is being updated). To run from scratch, call
    /// [`reset`](Self::reset) before it.
    pub fn forward_seq<S: Surrogate + ?Sized>(
        &self,
        x_seq: Vec<L::Input>,
        surrogate: &S,
    ) -> Vec<L::Output> {
        let current = self.state.borrow().clone();
        let (ys, next) = self.layer.forward_seq_from(x_seq, current, surrogate);
        *self.state.borrow_mut() = next;
        ys
    }

    /// Analogue of `init_hidden()` in snnTorch: recreate the state.
    pub fn reset(&self, batch: usize, device: &Device) {
        *self.state.borrow_mut() = self.layer.init_state(batch, device);
    }

    /// Detach state from autodiff graph (truncated BPTT).
    pub fn detach_hidden(&self) {
        let detached = self.state.borrow().clone().detach();
        *self.state.borrow_mut() = detached;
    }

    /// Snapshot of the current state.
    pub fn state(&self) -> L::State {
        self.state.borrow().clone()
    }

    /// Change the state (for example, restore from a checkpoint).
    pub fn set_state(&self, state: L::State) {
        *self.state.borrow_mut() = state;
    }

    /// Unpack the wrapper.
    pub fn into_parts(self) -> (L, L::State) {
        (self.layer, self.state.into_inner())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::layers::{StdpLinearConfig, StdpRule, SynapticDelayConfig};
    use crate::neuron::{AdaptiveLeakyConfig, IzhikevichConfig, LeakyConfig};
    use crate::state::ResetMode;
    use crate::stateful::Stateful;
    use crate::surrogate::FastSigmoid;
    use burn::nn::conv::Conv2dConfig;
    use burn::tensor::Device;

    fn dev() -> Device {
        Device::flex()
    }

    fn t2(data: Vec<f32>, shape: [usize; 2]) -> Tensor<2> {
        Tensor::<1>::from_floats(data.as_slice(), &dev()).reshape(shape)
    }

    fn v<const D: usize>(t: Tensor<D>) -> Vec<f32> {
        t.into_data().try_to_vec::<f32>().unwrap()
    }

    fn assert_close(a: &[f32], b: &[f32], what: &str) {
        assert_eq!(a.len(), b.len(), "{what}: lengths vary");
        for (i, (x, y)) in a.iter().zip(b).enumerate() {
            assert!((x - y).abs() < 1e-6, "{what}[{i}]: {x} vs {y}");
        }
    }

    fn inputs() -> Vec<Tensor<2>> {
        vec![
            t2(vec![0.3, 0.1, 0.9], [1, 3]),
            t2(vec![0.6, 0.0, 0.2], [1, 3]),
            t2(vec![0.1, 0.9, 0.7], [1, 3]),
            t2(vec![0.0, 0.0, 0.5], [1, 3]),
        ]
    }

    /// Generalized code: one function works with any layer.
    fn run_generic<L: SpikingLayer>(layer: &L, xs: Vec<L::Input>) -> Vec<L::Output> {
        layer.forward_seq(xs, &FastSigmoid::default())
    }

    /// The trace `step` must coincide with the explicit `Leaky::step` (and with
    /// old `Stateful`) bitwise: this is the same dynamics code.
    #[test]
    fn leaky_trait_matches_explicit_and_stateful() {
        let device = dev();
        let s = FastSigmoid::default();
        let layer = LeakyConfig::new(0.85).with_threshold(0.5).init(&device);

        let mut st = NeuronState::zeros_mem_only(&device, 1, 3);
        let mut explicit = Vec::new();
        for x in inputs() {
            let (y, n) = layer.step(x, &st, &s);
            st = n;
            explicit.push(v(y));
        }

        let via_trait = run_generic(&layer, inputs());
        let legacy = Stateful::<Leaky, NeuronState>::new_neuron(
            LeakyConfig::new(0.85).with_threshold(0.5).init(&device),
            &device,
            1,
            3,
        );
        for (t, x) in inputs().into_iter().enumerate() {
            assert_close(&explicit[t], &v(via_trait[t].clone()), "trait vs explicit");
            assert_close(&explicit[t], &v(legacy.step(x, &s)), "stateful vs explicit");
        }
    }

    /// `init_state` is delayed in an element-wise neuron; after the first step
    /// the state is ready and has an input shape.
    #[test]
    fn elementwise_state_is_deferred_until_first_step() {
        let device = dev();
        let s = FastSigmoid::default();
        let layer = LeakyConfig::new(0.9).init(&device);

        let st0 = layer.init_state(2, &device);
        assert!(st0.is_pending());
        // Detach in a deferred state is a no-op, not a panic.
        assert!(st0.clone().detach().is_pending());

        let (_, st1) = SpikingLayer::step(&layer, Tensor::zeros([2, 5], &device), st0, &s);
        assert_eq!(st1.get().expect("after the Ready step").mem.dims(), [2, 5]);
    }

    #[test]
    #[should_panic(expected = "did not match the init_state batch")]
    fn deferred_state_rejects_wrong_batch() {
        let device = dev();
        let layer = LeakyConfig::new(0.9).init(&device);
        let st = layer.init_state(3, &device);
        let _ = SpikingLayer::step(
            &layer,
            Tensor::zeros([2, 4], &device),
            st,
            &FastSigmoid::default(),
        );
    }

    /// The remaining element-by-element neurons follow the same path.
    #[test]
    fn all_elementwise_neurons_run_through_trait() {
        let device = dev();
        let xs = || {
            vec![
                Tensor::<2>::ones([2, 4], &device).mul_scalar(0.8),
                Tensor::<2>::ones([2, 4], &device).mul_scalar(0.8),
            ]
        };
        let ys = run_generic(
            &crate::neuron::Synaptic::new(&device, 0.9, 0.8, 1.0, ResetMode::Subtract),
            xs(),
        );
        assert_eq!(ys[0].dims(), [2, 4]);
        let ys = run_generic(
            &crate::neuron::Alpha::new(&device, 0.9, 0.8, 1.0, ResetMode::Subtract),
            xs(),
        );
        assert_eq!(ys[1].dims(), [2, 4]);
        let ys = run_generic(
            &crate::neuron::Lapicque::new(&device, 5.0, 1.0, ResetMode::Subtract),
            xs(),
        );
        assert_eq!(ys[0].dims(), [2, 4]);
        let ys = run_generic(
            &AdaptiveLeaky::new(&AdaptiveLeakyConfig::new(0.9, 0.95), &device),
            xs(),
        );
        assert_eq!(ys[1].dims(), [2, 4]);
        let ys = run_generic(
            &Izhikevich::new(&IzhikevichConfig::default(), &device),
            xs(),
        );
        assert_eq!(ys[0].dims(), [2, 4]);
    }

    /// AdEx, Hodgkin–Huxley and graded/ternary LIF work through
    /// common trait, and their state is uncoupled (`Session::detach_hidden`-path).
    #[test]
    fn bio_and_graded_neurons_run_through_trait() {
        use crate::neuron::{AdExConfig, GradedLeakyConfig, HodgkinHuxleyConfig};
        let device = dev();
        let xs = |amp: f32| {
            vec![
                Tensor::<2>::ones([2, 3], &device).mul_scalar(amp),
                Tensor::<2>::ones([2, 3], &device).mul_scalar(amp),
            ]
        };
        let ys = run_generic(&AdExConfig::default().init(&device), xs(900.0));
        assert_eq!(ys[1].dims(), [2, 3]);
        let ys = run_generic(
            &HodgkinHuxleyConfig::default()
                .with_substeps(8)
                .init(&device),
            xs(10.0),
        );
        assert_eq!(ys[0].dims(), [2, 3]);
        let ys = run_generic(&GradedLeakyConfig::ternary(0.9, 1.0).init(&device), xs(1.6));
        assert_eq!(ys[0].dims(), [2, 3]);
        // Ternary: input 1.6 > θ → +1·α (α = 1 by default).
        let v = ys[0].clone().into_data().try_to_vec::<f32>().unwrap();
        assert!(v.iter().all(|x| *x == 1.0), "{v:?}");

        // detach state does not change the values.
        let layer = AdExConfig::default().init(&device);
        let st = layer.init_state(2, &device);
        let (_, st) = SpikingLayer::step(
            &layer,
            Tensor::<2>::ones([2, 3], &device),
            st,
            &FastSigmoid::default(),
        );
        let _ = st.detach();
    }

    /// `forward_seq` == manual loop; the tensor version gives `[b, t, f]`.
    #[test]
    fn forward_seq_and_tensor_variant_agree_with_loop() {
        let device = dev();
        let s = FastSigmoid::default();
        let layer = LeakyConfig::new(0.8).with_threshold(0.4).init(&device);

        let seq = run_generic(&layer, inputs());

        let stacked: Tensor<3> = Tensor::stack::<3>(inputs(), 1); // [1, 4, 3]
        let tensor_out = layer.forward_seq_tensor(stacked, &s);
        assert_eq!(tensor_out.dims(), [1, 4, 3]);
        let flat = v(tensor_out);
        for t in 0..4 {
            assert_close(
                &v(seq[t].clone()),
                &flat[t * 3..(t + 1) * 3],
                "tensor vs vec",
            );
        }
    }

    /// Recurrent layer: `spike_prev` is now inside the state, and
    /// the result coincides with the manual cycle, where it was dragged by hand.
    #[test]
    fn rleaky_state_carries_spike_prev() {
        let device = dev();
        let s = FastSigmoid::default();
        let layer = RLeaky::new(&device, 3, 2, 0.85, 0.5, ResetMode::Subtract);

        let mut st = NeuronState::zeros_mem_only(&device, 1, 2);
        let mut prev = Tensor::<2>::zeros([1, 2], &device);
        let mut explicit = Vec::new();
        for x in inputs() {
            let (y, n) = layer.step(x, prev.clone(), &st, &s);
            st = n;
            prev = y.clone();
            explicit.push(v(y));
        }

        let via_trait = run_generic(&layer, inputs());
        for t in 0..explicit.len() {
            assert_close(&explicit[t], &v(via_trait[t].clone()), "rleaky");
        }
    }

    /// STDP: weights are part of the state, `Session` does not lose them, but
    /// `detach_hidden` does not change the values.
    #[test]
    fn stdp_weights_live_in_state_and_session_matches_explicit() {
        let device = dev();
        let s = FastSigmoid::default();
        let cfg = StdpLinearConfig::new(
            3,
            2,
            StdpRule::Pair {
                a_plus: 0.05,
                a_minus: 0.03,
                tau_pre: 20.0,
                tau_post: 20.0,
            },
        )
        .with_threshold(0.3);
        let layer = cfg.init();

        let init = <StdpLinear as SpikingLayer>::init_state(&layer, 1, &device);
        let w0 = v(init.w.clone());

        // Explicit path from the SAME initial state.
        let mut st = init.clone();
        let mut explicit = Vec::new();
        for x in inputs() {
            let (y, n) = layer.step(x, &st, &s);
            st = n;
            explicit.push(v(y));
        }

        let sess = Session::<StdpLinear>::from_parts(layer, init);
        for (t, x) in inputs().into_iter().enumerate() {
            assert_close(
                &explicit[t],
                &v(sess.step(x, &s)),
                "stdp session vs explicit",
            );
        }
        let w_after = v(sess.state().w);
        assert_close(&w_after, &v(st.w.clone()), "final weights");
        assert!(
            w0.iter().zip(&w_after).any(|(a, b)| (a - b).abs() > 1e-9),
            "STDP should have changed the weights inside the state"
        );

        let before = v(sess.state().w);
        sess.detach_hidden();
        assert_close(&before, &v(sess.state().w), "detach does not change values");
    }

    #[test]
    fn synaptic_delay_trait_matches_explicit() {
        let device = dev();
        let s = FastSigmoid::default();
        let layer = SynapticDelayConfig::new(1, 4)
            .with_delay_init(2.0)
            .with_learn_delay(false)
            .init(&device);

        let xs: Vec<f32> = vec![1.0, 0.0, 0.0, 0.0, 0.0];
        let mut st = DelayState::zeros(&device, 1, 1, layer.max_delay());
        let mut explicit = Vec::new();
        for &x in &xs {
            let (y, n) = layer.step(t2(vec![x], [1, 1]), &st);
            st = n;
            explicit.push(v(y)[0]);
        }

        let seq: Vec<Tensor<2>> = xs.iter().map(|&x| t2(vec![x], [1, 1])).collect();
        let via_trait = layer.forward_seq(seq, &s);
        for (i, y) in via_trait.into_iter().enumerate() {
            assert!((v(y)[0] - explicit[i]).abs() < 1e-6, "step {i}");
        }
    }

    #[test]
    fn spiking_conv2d_state_is_inferred_from_conv_output() {
        let device = dev();
        let s = FastSigmoid::default();
        let layer = SpikingConv2d::new(
            Conv2dConfig::new([1, 2], [3, 3]),
            &LeakyConfig::new(0.9).with_threshold(0.5),
            &device,
        );
        let x = || Tensor::<4>::ones([2, 1, 5, 5], &device);

        // Manual way: 3x3 size after folding without padding.
        let mut st = SpikingConv2d::state_zeros(&device, 2, 2, 3, 3);
        let mut explicit = Vec::new();
        for _ in 0..3 {
            let (y, n) = layer.step(x(), &st, &s);
            st = n;
            explicit.push(v(y));
        }

        let via_trait = layer.forward_seq(vec![x(), x(), x()], &s);
        assert_eq!(via_trait[0].dims(), [2, 2, 3, 3]);
        for t in 0..3 {
            assert_close(&explicit[t], &v(via_trait[t].clone()), "conv2d");
        }
    }

    /// `Session` == manual loop; `reset` returns to zero, `set_state`
    /// restores the photo.
    #[test]
    fn session_reset_and_snapshot() {
        let device = dev();
        let s = FastSigmoid::default();
        let sess = Session::<_>::new(
            LeakyConfig::new(0.85).with_threshold(0.5).init(&device),
            1,
            &device,
        );

        let first: Vec<Vec<f32>> = inputs().into_iter().map(|x| v(sess.step(x, &s))).collect();
        let snapshot = sess.state();
        assert!(!snapshot.is_pending());

        sess.reset(1, &device);
        assert!(sess.state().is_pending());
        let again: Vec<Vec<f32>> = inputs().into_iter().map(|x| v(sess.step(x, &s))).collect();
        for t in 0..first.len() {
            assert_close(&first[t], &again[t], "after reset");
        }

        sess.set_state(snapshot.clone());
        let a = v(sess.step(inputs()[0].clone(), &s));
        let (layer, _) = sess.into_parts();
        let (b, _) = SpikingLayer::step(&layer, inputs()[0].clone(), snapshot, &s);
        assert_close(&a, &v(b), "set_state");
    }

    /// TBPTT: running through chunks with `detach` gives the same values as
    /// single run (detach changes the graph, but not the numbers).
    #[test]
    fn chunked_forward_matches_single_pass() {
        let device = dev();
        let s = FastSigmoid::default();
        let layer = LeakyConfig::new(0.85).with_threshold(0.5).init(&device);

        let whole = layer.forward_seq(inputs(), &s);

        let mut st = layer.init_state(1, &device);
        let mut chunked = Vec::new();
        for chunk in inputs().chunks(2) {
            let (ys, next) = layer.forward_seq_from(chunk.to_vec(), st, &s);
            chunked.extend(ys);
            st = next.detach();
        }
        for t in 0..whole.len() {
            assert_close(&v(whole[t].clone()), &v(chunked[t].clone()), "chunked");
        }
    }

    /// ConvLSTM: the state map is expanded according to the size of the input.
    #[test]
    fn conv_lstm_state_is_inferred_from_input() {
        let device = dev();
        let s = FastSigmoid::default();
        let layer = SConv2dLSTM::new(&device, 1, 2, 3, 0.5, ResetMode::Subtract);
        let x = || Tensor::<4>::ones([2, 1, 4, 4], &device).mul_scalar(0.5);

        let mut st = ConvLstmState::zeros(&device, 2, 2, 4, 4);
        let mut explicit = Vec::new();
        for _ in 0..3 {
            let (y, n) = layer.step(x(), &st, &s);
            st = n;
            explicit.push(v(y));
        }
        let via_trait = layer.forward_seq(vec![x(), x(), x()], &s);
        for t in 0..3 {
            assert_close(&explicit[t], &v(via_trait[t].clone()), "conv_lstm");
        }
    }

    /// STDP-convolution: the kernel lives in the state; from the same
    /// the initial state of the trace and the explicit path coincide.
    #[test]
    fn stdp_conv1d_kernel_lives_in_state() {
        use crate::layers::{StdpConv1dConfig, StdpConv1dRule};
        let device = dev();
        let s = FastSigmoid::default();
        let layer = StdpConv1dConfig::new(
            2,
            3,
            3,
            StdpConv1dRule::Pair {
                a_plus: 0.05,
                a_minus: 0.03,
                tau_pre: 20.0,
                tau_post: 20.0,
            },
        )
        .with_threshold(0.3)
        .init();
        let x = || Tensor::<3>::ones([1, 2, 6], &device).mul_scalar(0.8);

        let init = layer.state_zeros(&device, 1, 6);
        let k0 = v(init.kernel.clone());

        let mut st = init.clone();
        let mut explicit = Vec::new();
        for _ in 0..3 {
            let (y, n) = layer.step(x(), &st, &s);
            st = n;
            explicit.push(v(y));
        }

        let (ys, last) = layer.forward_seq_from(vec![x(), x(), x()], Deferred::ready(init), &s);
        for t in 0..3 {
            assert_close(&explicit[t], &v(ys[t].clone()), "stdp_conv1d");
        }
        let k_after = v(last.get().unwrap().kernel.clone());
        assert_close(&k_after, &v(st.kernel.clone()), "final kernel");
        assert!(k0.iter().zip(&k_after).any(|(a, b)| (a - b).abs() > 1e-9));
    }
}
