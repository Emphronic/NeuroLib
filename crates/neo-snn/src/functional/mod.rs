//! Loss functions, regularization, metrics, and utilities for SNN training.
//!
//! The submodule layout mirrors `snntorch.functional`:
//!
//! * [`loss`] — spike- and membrane-based losses;
//! * [`reg`] — activity penalties (energy use and degeneration);
//! * [`acc`] — accuracy metrics;
//! * [`probe`] — activity and gradient monitors;
//! * [`raster`] — a text spike raster;
//! * [`temporal`] — reductions of a spike train along the time axis;
//! * [`lateral`] — lateral inhibition and winner-take-all;
//! * [`population`] — N-neurons-per-class readout (population code);
//! * [`plasticity`] — STDP primitives (pair/triplet/reward-modulated) for
//!   unsupervised and on-chip learning without backprop.

/// Accuracy metrics over the spike output.
pub mod acc;
/// Lateral inhibition and winner-take-all.
pub mod lateral;
/// Loss functions.
pub mod loss;
/// Synaptic plasticity primitives (STDP).
pub mod plasticity;
/// N-neurons-per-class readout (population code).
pub mod population;
/// Activity and gradient monitors.
pub mod probe;
/// A text spike raster.
pub mod raster;
/// Regularization over the firing rate.
pub mod reg;
/// Reductions of a spike train along the time axis.
pub mod temporal;

pub use acc::{
    accuracy_rate, accuracy_temporal, first_spike_times, population_accuracy_rate,
    population_accuracy_temporal, population_rates, population_spike_counts, spike_counts,
};
pub use lateral::{lateral_inhibition, lateral_inhibition_soft, winner_take_all};
pub use loss::{
    ce_count_loss, ce_max_membrane_loss, ce_rate_loss, ce_temporal_loss, mse_count_loss,
    mse_membrane_loss, mse_temporal_loss, population_ce_count_loss,
    population_ce_max_membrane_loss, population_ce_rate_loss,
};
pub use plasticity::{
    FnTrace, Identity, NoWeightDependence, TraceNonlinearity, WeightBound, WeightDependence,
    apply_reward, clip_weights, eligibility_trace_update, pair_update, pair_update_with_f,
    pair_update_with_f_and_dep, triplet_update, triplet_update_with_f,
    triplet_update_with_f_and_dep, update_trace,
};
pub use population::{PopulationReduction, population_reduce, population_reduce_seq};
pub use probe::{ActivityStats, Monitor, grad_norm};
pub use raster::{RasterStyle, activity_histogram, spike_raster, spike_raster_labeled};
pub use reg::{l1_spike_rate, l2_spike_rate, rate_bounds_penalty};
pub use temporal::{max_over_time, mean_over_time, sum_over_time, time_slice};
