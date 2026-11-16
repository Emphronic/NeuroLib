//! Surrogate gradients for backpropagating through the Heaviside step.
//!
//! The forward pass is the same for all of them and is numerically equal to Heaviside; they differ
//! only in the derivative, which is substituted for the non-existent
//! derivative of the step.
//!
//! | Surrogate | Gradient | When to take |
//! |---|---|---|
//! | [`ATan`] | `∝ 1/(1+(αx)²)` | smart default choice |
//! | [`FastSigmoid`] | `0.5/(1+k\|x\|)²` | polynomial decay, fewer “dead” neurons |
//! | [`SigmoidSurrogate`] | `α·s(1−s)` | exponential decay, sharp |
//! | [`Triangular`] | `max(0, 1−k\|x\|)` | cheapest, finite support |
//! | [`SpikeRateEscape`] | `k·exp(−β\|x\|)` | biophysical motivation |
//! | [`SparseFastSigmoid`] | fast sigmoid in window `\|x\|<B` | sparse backward on large networks |
//! | [`StraightThrough`] | `1` | debugging, “don’t touch the gradient” standard |
//! | [`Heaviside`] | `1` is above the threshold, `0` is below | standard for comparison |
//! | [`LeakySpikeOperator`] | `1` / `k` | subthreshold neurons also need a signal |
//! | [`StochasticSpikeOperator`] | `1` / noise | pushes away “dead” neurons |
//!
//! Its surrogate is declared by a macro
//! [`custom_surrogate!`](crate::custom_surrogate).

mod atan;
mod custom;
mod fast_sigmoid;
mod heaviside;
mod sigmoid;
mod sparse_fast_sigmoid;
mod spike_operator;
mod spike_rate_escape;
mod straight_through;
mod trait_impl;
mod triangular;

pub use atan::ATan;
pub use fast_sigmoid::FastSigmoid;
pub use heaviside::Heaviside;
pub use sigmoid::SigmoidSurrogate;
pub use sparse_fast_sigmoid::SparseFastSigmoid;
pub use spike_operator::{LeakySpikeOperator, SpikeOperator, StochasticSpikeOperator};
pub use spike_rate_escape::SpikeRateEscape;
pub use straight_through::StraightThrough;
pub use trait_impl::Surrogate;
pub use triangular::Triangular;
