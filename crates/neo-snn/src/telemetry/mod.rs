//! Telemetry of learning: everything that the learning cycle gives out, and nothing about HOW it is drawn.
//!
//! The division of responsibilities:
//!
//! ```text
//!  training flow GUI (neo-gui, Training tab)
//!  ───────────────                                  ───────────────────────────────
//!  TrainTx::step / spikes / activity ──channel──▶ TrainRx → curves, raster, health
//!  TrainTx::checkpoint() ◀── TrainControl ─── Pause / Stop buttons
//! ```
//!
//! * [`channel`] creates a pair [`TrainTx`] (into the training flow) and [`TrainRx`] (reads GUI or any
//!   other consumer: console, file, test). The channel is `async_channel`, so the GUI is waiting for the event
//!   in its async executor and does not poll.
//! * Training NEVER waits for the consumer: metric events are discarded when the queue is full
//!   (and count, see [`TrainControl::dropped`]), and an expensive spike shot is taken from
//!   devices only if [`TrainTx::wants_frame`] allows.
//! * Data taken from [`crate::functional::probe`]: gradient norm ([`grad_norm`](crate::functional::grad_norm))
//!   goes to [`StepMetrics`], tensor summary ([`ActivityStats`](crate::functional::ActivityStats)) —
//!   event [`TrainEvent::Activity`].
//! * [`SpikeFrame`] — bit map of “neuron × step” bits; [`History`] — metric ring buffer;
//!   [`assess`] - verbal verdict about the state of the network.
//!   [`SpikeFrame`] uses `OnceLock` instead of `OnceCell` so a snapshot can be shared
//!   between threads via `Arc`.
//!
//! Structural plasticity ([`crate::structural`]) goes through the same channel: [`TrainTx::structural`]
//! sends [`StructureReport`] (counters, new events with reasons, loss before/after, graph snapshot),
//! [`assess_structure`] gives a verdict on them (“growth without improvement”, “quality has deteriorated”).
//!
//! Saving the learned weights of STDP layers (`layers::record`) has nothing to do with this channel:
//! These are checkpoints, not metrics.
//!
//! ```ignore
//! use neo_snn::telemetry::{channel, ChannelConfig, StepMetrics, Progress};
//!
//! let (tx, rx) = channel(ChannelConfig::new("demo network"));
//! std::thread::spawn(move || {
//!     for epoch in 0..EPOCHS {
//!         for batch in 0..BATCHES {
//!             //...learning step...
//!             let progress = Progress { epoch, epochs: EPOCHS, batch, batches: BATCHES };
//!             tx.step(StepMetrics { progress, loss, accuracy: Some(acc), grad_norm: Some(g) });
//!             if tx.wants_frame() {
//!                 tx.spikes("hidden", progress, out.hidden_spikes, 0);
//!             }
//!             if tx.checkpoint() { return; } // Stop from GUI - along the step boundary
//!         }
//!     }
//!     tx.finish();
//! });
//! ```
//!
//! The backend must have `f32` as the element type.

mod channel;
mod evolution;
pub mod evolution_demo;
mod frame;
mod health;
mod history;
pub mod structural_demo;
mod structure;

pub use channel::{
    ChannelConfig, Progress, StepMetrics, TrainControl, TrainEvent, TrainRx, TrainTx, channel,
};
pub use evolution::EvolutionReport;
pub use frame::SpikeFrame;
pub use health::{Finding, Severity, assess, overall};
pub use history::History;
pub use structure::{
    EdgeView, NodeView, SNAPSHOT_NEURONS, SNAPSHOT_SYNAPSES, StructureLimits, StructureReport,
    TopologySnapshot, assess_structure,
};
