//! Probabilistic machinery — the second part of "Device Stochasticity Directly Implements Bayesian
//! Sampling in Spiking Neurons":
//!
//! - [`bayesian_inference`] — the threshold distribution IS the posterior over the neuron's
//!   excitability (`ThresholdPosterior`) — Bayesian sampling, without running the stochastic
//!   neuron step by step;
//! - [`rate_estimation`] — Rate Estimation: instead of explicitly modeling each spike, the layer's
//!   output probability is propagated (paper's RE: `μ = P·W`, `σ² = (P⊙(1−P))·(W⊙W)`,
//!   `P_spike ≈ Φ((μ − θ)/(σ + ε))`);
//! - [`probability_state`] — the probabilistic forward of a stochastic-threshold layer
//!   (no spikes generated);
//! - [`rng`] — the controllable RNG source — the software analogue of the paper's hardware
//!   randomness (reproducible with seeds).

pub mod bayesian_inference;
pub mod metrics;
pub mod probability_state;
pub mod rate_estimation;
pub mod rng;

pub use bayesian_inference::ThresholdPosterior;
pub use metrics::{expected_calibration_error, spike_nll};
pub use probability_state::{ProbabilityState, forward_seq_prob};
pub use rate_estimation::{RateEstimate, rate_estimate};
pub use rng::RngSource;
