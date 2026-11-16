//! Sparse fast sigmoid: the gradient is zeroed away from the threshold.
//!
//! N. Perez-Nieves, D. F. M. Goodman (2021), «Sparse Spiking Gradient
//! Descent», NeurIPS (arXiv:2105.08810).
//!
//! ## What's the idea
//!
//! Any smooth surrogate has a gradient for EVERY neuron on every
//! step, including those that are far from the threshold and obviously would not change their
//! solutions. Their contribution is small in size, but it is still accounted for and stored
//! and adds up - and in total over millions of neurons it also introduces noise.
//!
//! Here the gradient is explicitly set to zero outside the `|x| < B` window around the threshold.
//! The authors show that the return pass becomes sparse by
//! orders (they have up to 99%) without loss of accuracy.
//!
//! ## About the window sign
//!
//! In `snntorch` the condition is written as `U > B`, that is, the window is one-sided.
//! Here we take the symmetrical `|x| < B` - it corresponds to the formulation from
//! articles (“propagate the gradient only in neurons CLOSE to the threshold”)
//! and does not turn off learning in subthreshold neurons, which are precisely
//! you need a signal to start spiking.

use burn::tensor::Tensor;

use super::Surrogate;

/// Fast sigmoid, in which the gradient is cut outside the `|x| < B` window.
#[derive(Debug, Clone, Copy)]
pub struct SparseFastSigmoid {
    /// Slope k of the base fast sigmoid.
    pub slope: f64,
    /// The half-width of window B, outside of which the gradient is zero.
    pub bound: f64,
}

impl Default for SparseFastSigmoid {
    fn default() -> Self {
        Self {
            slope: 25.0,
            bound: 1.0,
        }
    }
}

impl SparseFastSigmoid {
    /// New sparse fast sigmoid.
    pub fn new(slope: f64, bound: f64) -> Self {
        Self { slope, bound }
    }

    /// Set the half-width of window B.
    pub fn with_bound(mut self, bound: f64) -> Self {
        self.bound = bound;
        self
    }
}

impl Surrogate for SparseFastSigmoid {
    fn surrogate_forward(&self, x: Tensor<2>) -> Tensor<2> {
        let k = self.slope as f32;
        let b = self.bound as f32;

        // The mask is detached from the graph: it specifies WHERE to calculate the gradient, and itself
        // is not a teachable quantity.
        let mask = x.clone().abs().lower_elem(b).float().detach();

        let denom = x.clone().abs().mul_scalar(k).add_scalar(1.0);
        let soft = x.div(denom).mul_scalar(0.5).add_scalar(0.5);

        soft.mul(mask)
    }
}
