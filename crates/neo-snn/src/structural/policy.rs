//! Parameters, limits, and rules of structural changes.

use super::error::{StructuralError, StructuralResult};
use super::model::{LifParams, NeuronModel, NeuronPolarity};
use super::model_selection::ModelSelectionConfig;
use super::neuron_ops::SplitInit;
use super::param_learning::ParamLearningConfig;
use super::stats::LinearCost;
use super::topology::{ParamLimits, TopologyLimits};
use crate::scalar::Scalar;

/// Schedule of structural decisions.
///
/// The structure changes only at boundaries chosen by the calling code
/// (end of a sample, minibatch, or epoch), and no more often than this schedule allows.
#[derive(Debug, Clone, PartialEq)]
pub struct ScheduleConfig {
    /// Analyze on every N-th `adapt` call (≥ 1).
    pub analyze_every: u64,
    /// Minimum number of simulation steps between structural changes.
    pub min_interval_steps: u64,
    /// Maximum number of operations per interval.
    pub max_ops_per_interval: usize,
}

impl Default for ScheduleConfig {
    fn default() -> Self {
        Self {
            analyze_every: 1,
            min_interval_steps: 500,
            max_ops_per_interval: 8,
        }
    }
}

/// Activity homeostasis: keeping the spike rate within an acceptable range
/// by shifting the threshold.
#[derive(Debug, Clone, PartialEq)]
pub struct HomeostasisConfig<T = f32> {
    /// Whether homeostasis is enabled.
    pub enabled: bool,
    /// Lower bound of the target rate (spikes per step).
    pub rate_min: T,
    /// Upper bound of the target rate.
    pub rate_max: T,
    /// Minimum number of observed neuron steps before intervention.
    pub min_observed_steps: u64,
    /// Relative threshold shift per intervention (for example, 0.1).
    pub threshold_step: T,
    /// Maximum number of neurons corrected per interval.
    pub max_neurons_per_interval: usize,
}

impl<T: Scalar> Default for HomeostasisConfig<T> {
    fn default() -> Self {
        Self {
            enabled: true,
            rate_min: T::c(0.0),
            rate_max: T::c(0.5),
            min_observed_steps: 200,
            threshold_step: T::c(0.1),
            max_neurons_per_interval: 2,
        }
    }
}

/// Synapse growth.
#[derive(Debug, Clone, PartialEq)]
pub struct SynapseGrowthConfig<T = f32> {
    /// Maximum number of new connections per interval.
    pub max_new_per_interval: usize,
    /// Minimum coactivity score for a pair.
    pub min_score: T,
    /// Initial weight of a new connection (positive; `w·(1 + jitter·u)`).
    pub init_weight: T,
    /// Relative spread of the initial weight.
    pub weight_jitter: T,
    /// Delay of a new connection.
    pub delay: u16,
}

impl<T: Scalar> Default for SynapseGrowthConfig<T> {
    fn default() -> Self {
        Self {
            max_new_per_interval: 2,
            min_score: T::c(3.0),
            init_weight: T::c(0.3),
            weight_jitter: T::c(0.2),
            delay: 0,
        }
    }
}

/// Neuron growth strategy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GrowthStrategy {
    /// Split the most loaded synapse with a relay neuron.
    SplitSynapse,
    /// Clone the most active hidden neuron.
    CloneNeuron,
    /// Create a new neuron with randomly weak connections to active sources.
    NewNeuron,
}

/// Neuron growth.
#[derive(Debug, Clone, PartialEq)]
pub struct NeuronGrowthConfig<T = f32> {
    /// Whether neuron growth is enabled.
    pub enabled: bool,
    /// How many consecutive evaluations without improvement count as a plateau.
    pub plateau_patience: usize,
    /// Minimum loss improvement that counts as progress.
    pub plateau_min_delta: T,
    /// Do not grow if the loss is already below this value.
    pub min_loss: T,
    /// Maximum number of new neurons per interval.
    pub max_new_per_interval: usize,
    /// Strategies in order of preference.
    pub strategies: Vec<GrowthStrategy>,
    /// Parameters of the new neuron (`NewNeuron`) for the LIF model.
    ///
    /// Used when [`new_neuron_model`](Self::new_neuron_model) is not
    /// set. Kept for configuration compatibility.
    pub new_neuron_params: LifParams<T>,
    /// Explicit model of the new neuron (for example, `AdaptiveLif` for slow
    /// adaptation). `None` — LIF from `new_neuron_params`.
    pub new_neuron_model: Option<NeuronModel<T>>,
    /// Polarity of the new neuron.
    pub new_neuron_polarity: NeuronPolarity,
    /// Number of incoming and outgoing connections of the new neuron.
    pub new_neuron_fan: usize,
    /// Initial weight of the new neuron's connections.
    pub new_neuron_weight: T,
    /// Relative spread of the clone's incoming weights.
    pub clone_jitter: T,
    /// Relay initialization for splits.
    pub split: SplitInit<T>,
}

impl<T: Scalar> Default for NeuronGrowthConfig<T> {
    fn default() -> Self {
        Self {
            enabled: true,
            plateau_patience: 3,
            plateau_min_delta: T::c(1e-3),
            min_loss: T::c(0.0),
            max_new_per_interval: 1,
            strategies: vec![
                GrowthStrategy::SplitSynapse,
                GrowthStrategy::CloneNeuron,
                GrowthStrategy::NewNeuron,
            ],
            new_neuron_params: LifParams::new(T::c(0.9), T::ONE),
            new_neuron_model: None,
            new_neuron_polarity: NeuronPolarity::Excitatory,
            new_neuron_fan: 2,
            new_neuron_weight: T::c(0.5),
            clone_jitter: T::c(0.1),
            split: SplitInit::default(),
        }
    }
}

/// Structural growth.
#[derive(Debug, Clone, PartialEq)]
pub struct GrowthConfig<T = f32> {
    /// Whether growth is enabled.
    pub enabled: bool,
    /// Whether to track coactivity (needed for synapse growth).
    pub track_coactivity: bool,
    /// Coactivity window in steps.
    pub coactivity_window: u64,
    /// Maximum number of stored candidate pairs.
    pub coactivity_capacity: usize,
    /// Maximum number of recent spikes matched against new ones.
    pub coactivity_max_recent: usize,
    /// Forgetting of pair scores on each analysis (multiplier, `(0, 1]`).
    pub coactivity_decay: T,
    /// Grow only while the loss is above this value: a network that already solves the task
    /// is not made more complex. `None` — no limit.
    pub target_loss: Option<T>,
    /// Whether to allow new connections between two output neurons (lateral
    /// readout connections). Disallowed by default: coactivity of
    /// outputs does not imply a useful connection between them.
    pub allow_output_to_output: bool,
    /// Synapse growth rules.
    pub synapses: SynapseGrowthConfig<T>,
    /// Neuron growth rules.
    pub neurons: NeuronGrowthConfig<T>,
}

impl<T: Scalar> Default for GrowthConfig<T> {
    fn default() -> Self {
        Self {
            enabled: true,
            track_coactivity: true,
            coactivity_window: 4,
            coactivity_capacity: 4096,
            coactivity_max_recent: 64,
            coactivity_decay: T::c(0.9),
            target_loss: None,
            allow_output_to_output: false,
            synapses: SynapseGrowthConfig::default(),
            neurons: NeuronGrowthConfig::default(),
        }
    }
}

/// Safe pruning.
///
/// Nothing is removed merely for a small weight or low activity: a
/// sufficient observation period is required (in steps **and** across samples), and the contribution
/// is estimated from the usage rate and the reward/error signal.
#[derive(Debug, Clone, PartialEq)]
pub struct PruningConfig<T = f32> {
    /// Whether pruning is enabled.
    pub enabled: bool,
    /// Minimum element age in steps before the first evaluation.
    pub min_age_steps: u64,
    /// Minimum number of observation samples.
    pub min_observed_samples: u64,
    /// Minimum number of steps before the same element is re-evaluated.
    pub reeval_interval_steps: u64,
    /// Threshold of relative usage rate (to the median): below it — a disable candidate.
    pub drive_rel_threshold: T,
    /// A synapse with `credit_ema` above the threshold is protected.
    pub protect_credit: T,
    /// Sample fraction below which a synapse is considered "rare"; a rare synapse with
    /// a positive contribution is protected.
    pub rare_coverage: T,
    /// How many steps a disabled synapse waits before physical removal.
    pub grace_steps: u64,
    /// Maximum fraction of active synapses disabled per interval.
    pub max_fraction_per_interval: T,
    /// Minimum number of active synapses in the network.
    pub min_synapses: usize,
    /// Minimum number of active incoming connections of an output neuron.
    pub min_output_in_degree: usize,
    /// Minimum number of hidden neurons.
    pub min_hidden: usize,
    /// Whether to remove "dead" neurons (without active inputs or outputs).
    pub prune_dead_neurons: bool,
    /// Whether to remove "silent" neurons.
    pub prune_silent_neurons: bool,
    /// Rate below which a neuron is considered silent.
    pub neuron_min_rate: T,
    /// Sample fraction below which a silent neuron is a removal candidate.
    pub neuron_min_coverage: T,
    /// Maximum number of neurons removed per interval.
    pub max_neuron_prunes: usize,
}

impl<T: Scalar> Default for PruningConfig<T> {
    fn default() -> Self {
        Self {
            enabled: true,
            min_age_steps: 1000,
            min_observed_samples: 20,
            reeval_interval_steps: 1000,
            drive_rel_threshold: T::c(0.05),
            protect_credit: T::c(0.0),
            rare_coverage: T::c(0.1),
            grace_steps: 1000,
            max_fraction_per_interval: T::c(0.1),
            min_synapses: 1,
            min_output_in_degree: 1,
            min_hidden: 0,
            prune_dead_neurons: true,
            prune_silent_neurons: true,
            neuron_min_rate: T::c(1e-3),
            neuron_min_coverage: T::c(0.02),
            max_neuron_prunes: 1,
        }
    }
}

/// Rollback of unsuccessful changes.
#[derive(Debug, Clone, PartialEq)]
pub struct RollbackConfig<T = f32> {
    /// Whether rollback is enabled.
    pub enabled: bool,
    /// Allowed relative loss degradation.
    pub rel_tolerance: T,
    /// Allowed absolute loss degradation.
    pub abs_tolerance: T,
    /// Compare the full cost `J` (loss + structure) instead of the loss alone.
    pub use_structure_cost: bool,
    /// After how many `adapt` calls to verify a change without an external evaluation.
    pub verify_after_calls: u64,
    /// Base cooldown (in steps) for an operation type after a rollback; doubled
    /// on repeated rollbacks.
    pub cooldown_steps: u64,
    /// Upper bound of the cooldown.
    pub max_cooldown_steps: u64,
}

impl<T: Scalar> Default for RollbackConfig<T> {
    fn default() -> Self {
        Self {
            enabled: true,
            rel_tolerance: T::c(0.10),
            abs_tolerance: T::c(1e-3),
            use_structure_cost: false,
            verify_after_calls: 3,
            cooldown_steps: 1000,
            max_cooldown_steps: 64_000,
        }
    }
}

/// Full structural plasticity configuration.
#[derive(Debug, Clone, PartialEq)]
pub struct StructuralConfig<T = f32> {
    /// Graph limits.
    pub topology: TopologyLimits,
    /// Parameter limits.
    pub params: ParamLimits<T>,
    /// Schedule.
    pub schedule: ScheduleConfig,
    /// Growth.
    pub growth: GrowthConfig<T>,
    /// Pruning.
    pub pruning: PruningConfig<T>,
    /// Homeostasis.
    pub homeostasis: HomeostasisConfig<T>,
    /// Automatic neuron model selection.
    pub model_selection: ModelSelectionConfig<T>,
    /// Learning of physical neuron parameters from loss.
    pub param_learning: ParamLearningConfig<T>,
    /// Rollback.
    pub rollback: RollbackConfig<T>,
    /// Default structural cost.
    pub cost: LinearCost<T>,
    /// Structural memory budget; growth is blocked when it is reached.
    pub max_state_bytes: Option<usize>,
    /// Seed of the controller's deterministic RNG.
    pub seed: u64,
    /// Event log size.
    pub log_capacity: usize,
    /// Loss history window.
    pub loss_window: usize,
}

impl<T: Scalar> Default for StructuralConfig<T> {
    fn default() -> Self {
        Self {
            topology: TopologyLimits::default(),
            params: ParamLimits::default(),
            schedule: ScheduleConfig::default(),
            growth: GrowthConfig::default(),
            pruning: PruningConfig::default(),
            homeostasis: HomeostasisConfig::default(),
            model_selection: ModelSelectionConfig::default(),
            param_learning: ParamLearningConfig::default(),
            rollback: RollbackConfig::default(),
            cost: LinearCost::default(),
            max_state_bytes: None,
            seed: 0,
            log_capacity: 1024,
            loss_window: 16,
        }
    }
}

impl<T: Scalar> StructuralConfig<T> {
    /// Validate configuration consistency.
    pub fn validate(&self) -> StructuralResult<()> {
        let bad = |m: &str| Err(StructuralError::InvalidConfig(m.to_string()));
        let fin = |v: T| v.is_finite();
        let p = &self.params;
        if !(fin(p.beta_min)
            && fin(p.beta_max)
            && T::ZERO <= p.beta_min
            && p.beta_min <= p.beta_max
            && p.beta_max < T::ONE)
        {
            return bad("params: 0 <= beta_min <= beta_max < 1 required");
        }
        if !(fin(p.threshold_min)
            && fin(p.threshold_max)
            && p.threshold_min > T::ZERO
            && p.threshold_min <= p.threshold_max)
        {
            return bad("params: 0 < threshold_min <= threshold_max required");
        }
        if !(fin(p.weight_abs_max) && p.weight_abs_max > T::ZERO) {
            return bad("params: weight_abs_max must be > 0");
        }
        if self.topology.max_neurons == 0 || self.topology.max_synapses == 0 {
            return bad("topology: limits must be > 0");
        }
        if self.topology.max_in_degree == 0 || self.topology.max_out_degree == 0 {
            return bad("topology: degree limits must be > 0");
        }
        if self.schedule.analyze_every == 0 || self.schedule.max_ops_per_interval == 0 {
            return bad("schedule: analyze_every and max_ops_per_interval must be > 0");
        }
        let h = &self.homeostasis;
        if !(fin(h.rate_min)
            && fin(h.rate_max)
            && T::ZERO <= h.rate_min
            && h.rate_min <= h.rate_max)
        {
            return bad("homeostasis: 0 <= rate_min <= rate_max required");
        }
        if !(fin(h.threshold_step) && h.threshold_step > T::ZERO && h.threshold_step < T::ONE) {
            return bad("homeostasis: threshold_step must be in (0, 1)");
        }
        let g = &self.growth;
        if !(fin(g.coactivity_decay)
            && g.coactivity_decay > T::ZERO
            && g.coactivity_decay <= T::ONE)
        {
            return bad("growth: coactivity_decay must be in (0, 1]");
        }
        if !(fin(g.synapses.init_weight)
            && g.synapses.init_weight > T::ZERO
            && g.synapses.init_weight <= p.weight_abs_max)
        {
            return bad("growth: init_weight must be in (0, weight_abs_max]");
        }
        if !(fin(g.synapses.weight_jitter) && g.synapses.weight_jitter >= T::ZERO) {
            return bad("growth: weight_jitter must be >= 0");
        }
        if g.synapses.delay > p.delay_max {
            return bad("growth: delay is greater than delay_max");
        }
        if g.neurons.strategies.is_empty() && g.neurons.enabled {
            return bad("growth: the strategy list is empty");
        }
        match &g.neurons.new_neuron_model {
            Some(model) => model
                .validate(p)
                .map_err(|e| StructuralError::InvalidConfig(format!("growth: {e}")))?,
            None => g
                .neurons
                .new_neuron_params
                .validate(p)
                .map_err(|e| StructuralError::InvalidConfig(format!("growth: {e}")))?,
        }
        let pr = &self.pruning;
        if !(fin(pr.max_fraction_per_interval)
            && (pr.max_fraction_per_interval >= T::ZERO && pr.max_fraction_per_interval <= T::ONE))
        {
            return bad("pruning: max_fraction_per_interval must be in [0, 1]");
        }
        if !(fin(pr.drive_rel_threshold) && pr.drive_rel_threshold >= T::ZERO) {
            return bad("pruning: drive_rel_threshold must be >= 0");
        }
        let r = &self.rollback;
        if !(fin(r.rel_tolerance)
            && r.rel_tolerance >= T::ZERO
            && fin(r.abs_tolerance)
            && r.abs_tolerance >= T::ZERO)
        {
            return bad("rollback: tolerances must be >= 0");
        }
        if self.loss_window == 0 || self.log_capacity == 0 {
            return bad("loss_window and log_capacity must be > 0");
        }
        self.model_selection.validate(p)?;
        self.param_learning.validate()?;
        Ok(())
    }
}
