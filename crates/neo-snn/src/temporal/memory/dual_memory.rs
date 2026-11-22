//! Dual memory layer (paper's DMP-SNN): the fast LIF pathway plus the slow
//! compact memory pathway.
//!
//! One step of [`DualMemoryLayer::step`] — paper Methods:
//!
//! ```text
//! x[k]  = ReLU(W_x·s + b_x)         // scalar drive into the slow path
//! m[k]  = Ā·m[k−1] + B̄·x[k]        // slow update (compact context)
//! Im[k] = W_m·m[k]                 // memory feedback into the membrane
//! u[k]  = β·u[k−1] + W_f·s + Im[k]  // fast LIF update
//! s[k]  = Θ(u[k] − θ)              // spikes (LIF reset applied)
//! ```
//!
//! The fast path is the ordinary LIF machinery ([`Leaky`]); the slow path is
//! [`SlowMemory`]. Both live in [`DualMemoryState`].

use burn::module::{Module, Param};
use burn::tensor::{Device, Distribution, Tensor};

use crate::neuron::{Leaky, LeakyConfig};
use crate::spiking_layer::SpikingLayer;
use crate::state::NeuronState;
use crate::stateful::Detach;
use crate::surrogate::Surrogate;

use super::fast_state::FastState;
use super::memory_feedback::MemoryFeedback;
use super::slow_state::{SlowMemory, SlowMemoryConfig, SlowMemoryState};

// ---------------------------------------------------------------------
// Config / state
// ---------------------------------------------------------------------

/// Config of [`DualMemoryLayer`].
#[derive(Debug, Clone, Copy)]
pub struct DualMemoryConfig {
    /// Fast pathway: LIF neuron configuration (paper: the "fast spiking").
    pub fast: LeakyConfig,
    /// Memory dimension `d` of the slow pathway.
    pub d: usize,
    /// State buffer length `θ` of the slow pathway.
    pub theta: f32,
    /// Per-step time resolution `Δt` of the slow pathway (paper uses 1.0).
    pub dt: f32,
}

impl DualMemoryConfig {
    /// New config with default LIF fast pathway and `dt = 1.0`.
    pub fn new(d: usize, theta: f32) -> Self {
        Self {
            fast: LeakyConfig::default(),
            d,
            theta,
            dt: 1.0,
        }
    }

    /// Per-step time resolution of the slow pathway.
    pub fn with_dt(mut self, dt: f32) -> Self {
        self.dt = dt;
        self
    }

    /// LIF configuration of the fast pathway.
    pub fn with_fast(mut self, fast: LeakyConfig) -> Self {
        self.fast = fast;
        self
    }
}

/// State of [`DualMemoryLayer`]: fast (LIF membrane) + slow (compact memory).
#[derive(Debug, Clone)]
pub struct DualMemoryState {
    /// Fast pathway state: LIF membrane (see [`FastState`]).
    pub fast: FastState,
    /// Slow pathway state: `m`, `[batch, d]`.
    pub slow: SlowMemoryState,
}

impl DualMemoryState {
    /// Zero state: membrane and memory are zeros.
    pub fn zeros(batch: usize, n_neurons: usize, d: usize, device: &Device) -> Self {
        Self {
            fast: NeuronState::zeros_mem_only(device, batch, n_neurons),
            slow: SlowMemoryState::zeros(batch, d, device),
        }
    }
}

impl Detach for DualMemoryState {
    fn detach(self) -> Self {
        Self {
            fast: NeuronState::detach(self.fast),
            slow: self.slow.detach(),
        }
    }
}

// ---------------------------------------------------------------------
// Layer
// ---------------------------------------------------------------------

/// Dual memory layer (paper's DMP-SNN).
#[derive(Module, Debug)]
pub struct DualMemoryLayer {
    /// Neurons in the layer (`N`).
    pub n_neurons: usize,
    /// Fast pathway: LIF neuron.
    pub fast: Leaky,
    /// Fast-path input weights `W_f` `[N, input_dim]` (paper: `I[k] = W_f·s[k]`).
    pub w_f: Param<Tensor<2>>,
    /// Slow pathway: LMU-style memory.
    pub slow: SlowMemory,
    /// Memory feedback: `Im[k] = W_m·m[k]`.
    pub memory_feedback: MemoryFeedback,
}

impl DualMemoryLayer {
    /// Create the layer.
    ///
    /// `n_neurons` — number of LIF neurons (`N`), `input_dim` — number of
    /// presynaptic features (the layer input width; the drive `W_x` of the
    /// slow pathway compresses these, like `W_f` the fast input).
    pub fn new(
        config: DualMemoryConfig,
        n_neurons: usize,
        input_dim: usize,
        device: &Device,
    ) -> Self {
        assert!(
            n_neurons > 0,
            "DualMemoryLayer: n_neurons must be at least 1"
        );
        assert!(
            input_dim > 0,
            "DualMemoryLayer: input_dim must be at least 1"
        );
        assert!(config.d > 0, "DualMemoryLayer: d must be at least 1");
        let bound = (1.0 / input_dim as f64).sqrt();
        Self {
            n_neurons,
            fast: Leaky::new(&config.fast, device),
            w_f: Param::from_tensor(Tensor::random(
                [n_neurons, input_dim],
                Distribution::Uniform(-bound, bound),
                device,
            )),
            slow: SlowMemory::new(
                SlowMemoryConfig::new(config.d, config.theta).with_dt(config.dt),
                input_dim,
                device,
            ),
            memory_feedback: MemoryFeedback::new(n_neurons, config.d, device),
        }
    }
}

impl SpikingLayer for DualMemoryLayer {
    type Input = Tensor<2>;
    type Output = Tensor<2>;
    type State = DualMemoryState;

    fn init_state(&self, batch: usize, device: &Device) -> Self::State {
        DualMemoryState::zeros(batch, self.n_neurons, self.slow.d, device)
    }

    fn step<S: Surrogate + ?Sized>(
        &self,
        s_in: Tensor<2>,
        state: Self::State,
        surrogate: &S,
    ) -> (Tensor<2>, Self::State) {
        // Slow path: scalar drive x[k], then m[k] = Ā·m[k−1] + B̄·x[k].
        let drive = self.slow.drive(&s_in);
        let slow = self.slow.step(&drive, &state.slow);
        // Memory feedback into the fast membrane: Im[k] = W_m·m[k].
        let mem_current = self.memory_feedback.feedback(&slow.m);
        // Fast path: u[k] = β·u[k−1] + W_f·s[k] + Im[k]; spikes.
        let fast_input = s_in.matmul(self.w_f.val().transpose()) + mem_current;
        let (spike, fast) = self.fast.step(fast_input, &state.fast, surrogate);
        (spike, DualMemoryState { fast, slow })
    }
}
