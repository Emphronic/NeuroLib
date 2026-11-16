//! Reservoir Computing: Liquid State Machine and Echo State Network.
//!
//! Idea: random recurrent network (“reservoir”) with fixed weights
//! turns the input signal into a temporal, high-dimensional and non-linear
//! condition; Only the linear output layer is learned. Therefore, training is one
//! linear regression problem (ridge) in closed shape, without gradients and without
//! BPTT: fast, deterministic and without gradient explosion.
//!
//! | Type | Reservoir | State for readout |
//! |---|---|---|
//! | [`LiquidStateMachine`] | spike LIF neurons, Dale's principle (80% excitatory / 20% inhibitory) | exponentially filtered spikes, spikes or membrane ([`LiquidFeature`]) |
//! | [`EchoStateNetwork`] | analog `tanh` neurons with leakage and a given spectral radius | state of neurons |
//!
//! Training the output layer is [`RidgeReadout`] (classification and regression), and
//! feature preparation - [`pool_time`] / [`flatten_time`].
//!
//! ```ignore
//! use neo_snn::reservoir::*;
//!
//! let lsm = LiquidStateMachine::new(&LiquidStateMachineConfig::new(1, 200), &device)?;
//! let states = lsm.run(x, LiquidFeature::Trace)?;          // [batch, time, neurons]
//! let feats = pool_time(states, Pooling::Segments(3));     // [batch, 3·neurons]
//! let readout = RidgeReadout::fit_tensor(feats.clone(), one_hot(&labels, 2, &device)?, 1e-2)?;
//! let predicted = readout.classify(feats)?;
//! ```
//!
//! Reservoir doesn't use auto-differentiation: it works on devices too
//! without autodiff, and on the GPU - with the same matrix multiplications Burn.

mod esn;
mod lsm;
mod readout;

pub use esn::{EchoStateNetwork, EchoStateNetworkConfig, estimate_spectral_radius};
pub use lsm::{LiquidFeature, LiquidState, LiquidStateMachine, LiquidStateMachineConfig};
pub use readout::{Pooling, RidgeReadout, accuracy, flatten_time, nmse, one_hot, pool_time};
