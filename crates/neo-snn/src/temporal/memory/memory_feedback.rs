//! Memory feedback: maps the slow state `m` into a per-neuron membrane
//! current (`Im = W_m·m`).

use burn::module::{Module, Param};
use burn::tensor::{Device, Distribution, Tensor};

/// `Im = W_m·m` — the slow state as an additional membrane current of the
/// fast path (paper: `Im^l[k] = W_m·m^l[k]`).
#[derive(Module, Debug)]
pub struct MemoryFeedback {
    /// `W_m` `[n_neurons, d]`: weights of the slow-path input current.
    pub w_m: Param<Tensor<2>>,
}

impl MemoryFeedback {
    /// Create with uniform init `±1/√d` (fan-in `d`).
    pub fn new(n_neurons: usize, d: usize, device: &Device) -> Self {
        assert!(d > 0, "MemoryFeedback: d must be at least 1");
        assert!(
            n_neurons > 0,
            "MemoryFeedback: n_neurons must be at least 1"
        );
        let bound = (1.0 / d as f64).sqrt();
        Self {
            w_m: Param::from_tensor(Tensor::random(
                [n_neurons, d],
                Distribution::Uniform(-bound, bound),
                device,
            )),
        }
    }

    /// Memory current: `m [batch, d] → Im [batch, n_neurons]`, `Im = m·W_mᵀ`.
    pub fn feedback(&self, m: &Tensor<2>) -> Tensor<2> {
        m.clone().matmul(self.w_m.val().transpose())
    }
}
