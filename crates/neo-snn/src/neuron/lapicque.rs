//! Lapik model (RC chain): `mem' = mem + dt·(I - mem)/τ`.

use burn::module::{Module, Param};
use burn::tensor::Device;
use burn::tensor::Tensor;
use burn::tensor::activation::softplus;

use super::spike_with_scalar_threshold;
use crate::state::{NeuronState, ResetMode, apply_reset};
use crate::surrogate::Surrogate;

/// Lapic neuron.
///
/// τ is parameterized via softplus and is strictly positive. A raw τ parameter
/// would let `(I - mem)/τ` diverge as soon as the optimizer drove τ through
/// zero, which is reachable because the gradient of τ near zero is unbounded.
#[derive(Module, Debug)]
pub struct Lapicque {
    /// Raw τ: `softplus(tau) + eps` is the time constant.
    pub tau: Param<Tensor<1>>,
    /// Spike threshold.
    pub threshold: f32,
    /// Integration step dt.
    pub dt: f32,
    /// Reset mode.
    #[module(skip)]
    pub reset_mode: ResetMode,
}

impl Lapicque {
    /// Create a layer (dt = 1.0).
    pub fn new(device: &Device, tau_init: f32, threshold: f32, reset_mode: ResetMode) -> Self {
        Self::with_dt(device, tau_init, threshold, 1.0, reset_mode)
    }

    /// Create a layer with explicit dt.
    pub fn with_dt(
        device: &Device,
        tau_init: f32,
        threshold: f32,
        dt: f32,
        reset_mode: ResetMode,
    ) -> Self {
        let t = tau_init.max(1e-3);
        let raw = if t > 20.0 {
            t
        } else {
            (t.exp() - 1.0).max(1e-6).ln()
        };
        Self {
            tau: Param::from_tensor(Tensor::from_floats([raw], device)),
            threshold,
            dt,
            reset_mode,
        }
    }

    /// One step in time.
    pub fn step<S: Surrogate + ?Sized>(
        &self,
        input: Tensor<2>,
        state: &NeuronState,
        surrogate: &S,
    ) -> (Tensor<2>, NeuronState) {
        let dims = state.mem.dims();

        let tau = softplus(self.tau.val(), 1.0)
            .add_scalar(1e-3)
            .reshape([1, 1])
            .expand(dims);

        let mem_new = state.mem.clone() + (input - state.mem.clone()).div(tau).mul_scalar(self.dt);

        let (spike, spike_hard) =
            spike_with_scalar_threshold(mem_new.clone(), self.threshold, surrogate);
        let mem_reset = apply_reset(mem_new, &spike_hard, self.threshold, self.reset_mode);

        (
            spike,
            NeuronState {
                mem: mem_reset,
                syn: state.syn.clone(),
            },
        )
    }
}
