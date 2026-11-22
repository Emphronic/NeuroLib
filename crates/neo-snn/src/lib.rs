//! # neo-snn
//!
//! Spike neural networks (SNN) for NeuroLib on top of [Burn].
//!
//! The feature set includes:
//! surrogates (triangular, escape-rate, sparse, spike-operator),
//! membrane and time losses, activity regularization, metrics,
//! monitors, `SConv2dLSTM`, state quantization, `SynapticDelay`
//! (per-channel trainable signal transmission delay), optional
//! stateful mode (`init_hidden`, see [`stateful`]), STDP
//! (pair/triplet/reward-modulated - unsupervised and on-chip training without
//! backprop, see [`functional::plasticity`], [`layers::StdpLinear`],
//! [`layers::StdpConv1d`], [`layers::StdpConv2d`]), spike-based
//! self-attention and transformer block (SSA/Spikformer and
//! SDSA/Spike-Driven Transformer, see [`attention`]), conversion of ready-made
//! regular networks into spike networks (ANN → SNN, see [`convert`]).
//!
//! Errors - typed: [`SnnError`] (see [`error`]); operations,
//! which have a `try_*` variant, return [`Result`] instead of panic.
//!
//! All stateful layers are reduced to one contract - [`spiking_layer::SpikingLayer`]
//! (`init_state` / `step` / `forward_seq`); the state lies either with the caller
//! code, or in [`spiking_layer::Session`], and for STDP layers the weights are part
//! of the state. The old explicit API and [`stateful`] continue to work.
//!
//! ```ignore
//! use neo_snn::prelude::*;
//! ```
//!
//! [Burn]: https://github.com/tracel-ai/burn

/// Spike-based self-attention and transformer block.
pub mod attention;
/// BatchNorm through time.
pub mod bntt;
/// Converting ordinary neural networks into spike networks (ANN → SNN).
pub mod convert;
/// Demo training on synthetics, writing in [`telemetry`] (feature `demo`).
#[cfg(feature = "demo")]
pub mod demo;
/// Dense (matrix) forward pass of a structural network on Burn tensors.
pub mod dense_burn;
/// Encoding of spike input.
pub mod encoding;
/// Typed crate errors.
pub mod error;
/// Event-driven SKAN: delta-gated KAN + LIF.
pub mod event_driven;
/// Structural plasticity on top of Burn layers of fixed shape (masks and reserve neurons).
pub mod fixed_shape;
/// Loss functions and utilities.
pub mod functional;
/// Kolmogorov-Arnold Network layer.
pub mod kan;
/// Composite layers (Conv+LIF, ConvLSTM, synaptic delay, etc.).
pub mod layers;
/// Models of spiking neurons.
pub mod neuron;
/// The most commonly used types.
pub mod prelude;
/// Probabilistic machinery — the stochastic-threshold neuron as a Bayesian model of the
/// neuron excitability (Rate Estimation, `ThresholdPosterior`), NLL/ECE metrics, and the
/// controllable RNG source (software model of the paper's hardware randomness).
pub mod probabilistic;
/// Quantization of the state of a neuron (QAT for membrane).
pub mod quant;
/// Reservoir computing: Liquid State Machine, Echo State Network and ridge-readout.
pub mod reservoir;
/// Scalar type for CPU computations (`f32`/`f64`/custom).
pub mod scalar;
/// Spike LSTM.
pub mod slstm;
/// Single trait `SpikingLayer` (`init_state`/`step`/`forward_seq`) and
/// universal wrapper `Session`.
pub mod spiking_layer;
/// States of neurons and layers.
pub mod state;
/// Optional stateful mode (`init_hidden` from snnTorch) on top of the
/// explicit-state API: convenient for implementing ready-made snnTorch models
/// one-to-one, when the state should be carried by the layer itself rather than by
/// the calling code. The explicit-state API does not go away —
/// both modes coexist equally; see the module documentation.
pub mod stateful;
/// Structural plasticity: growth/removal of neurons and connections on a dynamic graph (CPU).
pub mod structural;
/// Surrogate gradients.
pub mod surrogate;
/// Training telemetry: event channel, spike snapshots, metrics history, network state verdict.
/// All of this is rendered by the GUI (`neo-gui`, Training tab); there is no terminal or graphics here.
pub mod telemetry;
/// Temporal memory pathways: dual fast/slow memory (paper-based).
pub mod temporal;

pub use error::{Result, SnnError};

// The `neo-synk` kernels are available straight from `neo-snn` (and therefore as `neurolib::snn::...`):
// users do not need to know which crate holds the implementation, and the `Device` itself picks between
// the CubeCL kernel (wgpu / CUDA / ROCm) and the reference built from ordinary tensor ops.
//
// ```ignore
// use burn::tensor::{Device, Tensor};
// use neurolib::snn::leaky_scan_forward;
//
// let device = Device::wgpu(Default::default());
// let u = leaky_scan_forward(input, mem0, beta); // [batch, steps, features]
// ```
pub use neo_synk::snn::{leaky_scan_forward, leaky_scan_reference};

/// The other `neo-synk::snn` kernels (for example the fused spike `spike_surrogate` on Flex):
/// `neo_snn::kernels::...`.
pub use neo_synk::snn as kernels;

/// Crate version.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");
