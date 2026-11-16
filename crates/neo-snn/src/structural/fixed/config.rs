//! Configuration of structural plasticity for fixed-shape layers.
//!
//! The schedule and rollback use the same types as in the dynamic engine
//! ([`ScheduleConfig`], [`RollbackConfig`]); the rest is specific to masks.

use crate::structural::error::{StructuralError, StructuralResult};
use crate::structural::policy::{RollbackConfig, ScheduleConfig};

/// Connection growth (enabling disabled mask elements).
#[derive(Debug, Clone, PartialEq)]
pub struct FixedGrowthConfig {
    /// Whether connection growth is enabled.
    pub enabled: bool,
    /// Maximum number of new connections across the whole network per interval.
    pub max_new_per_interval: usize,
    /// Minimum co-activity of a pair (EMA `pre·post`).
    pub min_score: f32,
    /// Initial weight of a new connection.
    pub init_weight: f32,
    /// Absolute spread of the initial weight (±).
    pub weight_jitter: f32,
    /// Maximum fraction of enabled connections among the possible ones in a layer (0..=1].
    pub max_density: f32,
}

impl Default for FixedGrowthConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            max_new_per_interval: 4,
            min_score: 1e-3,
            init_weight: 0.1,
            weight_jitter: 0.02,
            max_density: 1.0,
        }
    }
}

/// Pruning of insignificant connections.
#[derive(Debug, Clone, PartialEq)]
pub struct FixedPruningConfig {
    /// Whether connection pruning is enabled.
    pub enabled: bool,
    /// Minimum connection age (in steps) before it can be disabled.
    pub min_age_steps: u64,
    /// How many statistics observations a layer needs before making decisions.
    pub min_observed_samples: u64,
    /// A connection is a candidate if its importance ≤ `rel_threshold` × mean layer importance.
    pub rel_threshold: f32,
    /// Maximum fraction of a layer's active connections disabled per interval.
    pub max_fraction_per_interval: f32,
    /// Do not drop a layer below this number of active connections.
    pub min_synapses_per_layer: usize,
    /// Minimum in-degree of an active neuron.
    pub min_in_degree: usize,
    /// Importance = EMA `|w·grad|` (otherwise `|w|`).
    pub use_saliency: bool,
}

impl Default for FixedPruningConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            min_age_steps: 200,
            min_observed_samples: 20,
            rel_threshold: 0.05,
            max_fraction_per_interval: 0.05,
            min_synapses_per_layer: 4,
            min_in_degree: 1,
            use_saliency: true,
        }
    }
}

/// Activation of reserve neurons and deactivation of silent ones.
#[derive(Debug, Clone, PartialEq)]
pub struct FixedNeuronConfig {
    /// Whether neuron control is enabled.
    pub enabled: bool,
    /// Window (in `adapt` calls with loss) for plateau detection.
    pub plateau_patience: usize,
    /// Loss improvement between the halves of the window below this value — plateau.
    pub plateau_min_delta: f32,
    /// Do not grow capacity if loss is already below this.
    pub min_loss: f32,
    /// Maximum number of neurons activated per interval.
    pub max_activate: usize,
    /// How many inputs to connect to a new neuron.
    pub new_fan_in: usize,
    /// How many output connections to each following layer.
    pub new_fan_out: usize,
    /// Initial weight of a new neuron's connections.
    pub init_weight: f32,
    /// Deactivate silent neurons.
    pub deactivate_silent: bool,
    /// Firing rate below this — the neuron is considered silent.
    pub neuron_min_rate: f32,
    /// Minimum layer observations before a decision.
    pub min_observed_samples: u64,
    /// Minimum neuron age (steps).
    pub min_age_steps: u64,
    /// Minimum active neurons per layer.
    pub min_active_per_layer: usize,
    /// Maximum number of neurons deactivated per interval.
    pub max_deactivate: usize,
}

impl Default for FixedNeuronConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            plateau_patience: 6,
            plateau_min_delta: 1e-3,
            min_loss: 0.0,
            max_activate: 1,
            new_fan_in: 4,
            new_fan_out: 3,
            init_weight: 0.1,
            deactivate_silent: true,
            neuron_min_rate: 1e-3,
            min_observed_samples: 20,
            min_age_steps: 400,
            min_active_per_layer: 1,
            max_deactivate: 1,
        }
    }
}

/// Full configuration.
#[derive(Debug, Clone, PartialEq)]
pub struct FixedShapeConfig {
    /// Schedule (shared type).
    pub schedule: ScheduleConfig,
    /// Connection growth.
    pub growth: FixedGrowthConfig,
    /// Connection pruning.
    pub pruning: FixedPruningConfig,
    /// Neurons.
    pub neurons: FixedNeuronConfig,
    /// Rollback (shared type; `use_structure_cost` is not used here).
    pub rollback: RollbackConfig,
    /// EMA coefficient of the statistics (0..1]; the larger, the faster the forgetting.
    pub ema: f32,
    /// A neuron with a rate below this is "silent" (for reporting).
    pub silent_rate: f32,
    /// A neuron with a rate above this is "overactive" (for reporting).
    pub overactive_rate: f32,
    /// RNG seed.
    pub seed: u64,
    /// Capacity of the event log.
    pub log_capacity: usize,
    /// Length of the loss history.
    pub loss_window: usize,
}

impl Default for FixedShapeConfig {
    fn default() -> Self {
        Self {
            schedule: ScheduleConfig {
                analyze_every: 1,
                min_interval_steps: 100,
                max_ops_per_interval: 8,
            },
            growth: FixedGrowthConfig::default(),
            pruning: FixedPruningConfig::default(),
            neurons: FixedNeuronConfig::default(),
            rollback: RollbackConfig {
                cooldown_steps: 200,
                max_cooldown_steps: 6_400,
                ..RollbackConfig::default()
            },
            ema: 0.05,
            silent_rate: 0.005,
            overactive_rate: 0.5,
            seed: 0x5eed_f1e1d,
            log_capacity: 512,
            loss_window: 64,
        }
    }
}

impl FixedShapeConfig {
    /// Validate the values; on error returns [`StructuralError::InvalidConfig`].
    pub fn validate(&self) -> StructuralResult<()> {
        let bad = |m: &str| Err(StructuralError::InvalidConfig(m.to_string()));
        if self.schedule.analyze_every == 0 {
            return bad("schedule.analyze_every must be ≥ 1");
        }
        if !(self.ema > 0.0 && self.ema <= 1.0) {
            return bad("ema must be in (0, 1]");
        }
        if !(self.growth.max_density > 0.0 && self.growth.max_density <= 1.0) {
            return bad("growth.max_density must be in (0, 1]");
        }
        if !(0.0..=1.0).contains(&self.pruning.max_fraction_per_interval) {
            return bad("pruning.max_fraction_per_interval must be in [0, 1]");
        }
        if self.pruning.rel_threshold < 0.0 {
            return bad("pruning.rel_threshold cannot be negative");
        }
        if self.neurons.plateau_patience < 2 {
            return bad("neurons.plateau_patience must be ≥ 2");
        }
        if self.loss_window < self.neurons.plateau_patience {
            return bad("loss_window must be ≥ neurons.plateau_patience");
        }
        if self.rollback.rel_tolerance < 0.0 || self.rollback.abs_tolerance < 0.0 {
            return bad("rollback tolerances cannot be negative");
        }
        if !self.growth.init_weight.is_finite() || !self.neurons.init_weight.is_finite() {
            return bad("init_weight must be finite");
        }
        Ok(())
    }
}
