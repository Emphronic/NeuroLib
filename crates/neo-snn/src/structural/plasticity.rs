//! Unified per-synapse plasticity pipeline ([`PlasticityPipeline`]).
//!
//! The engine (`DynamicNetwork` + [`StdpConfig`](super::network::StdpConfig)) supports one
//! global rule per network (pair-STDP, R-STDP, Vogels inhibitory STDP). The pipeline generalizes
//! this: rules are *assigned per synapse* (specific synapse > class/mask > network default),
//! several rules can compose on one synapse, and rules other than STDP-style ones are available.
//!
//! # Rules
//!
//! | [`RuleId`] | Semantics | State | Notes |
//! |---|---|---|---|
//! | `None` | plasticity off | – | |
//! | `PairStdp` | Engine pair-STDP (`pot` on arrival, `dep` on dst spike; `dw = pot − dep`), mirroring [`functional::plasticity::pair_update`](crate::functional::plasticity::pair_update) | `PairState` | |
//! | `TripletStdp` | Pfister–Gerstner triplet: fast/slow traces of both sides, `pot2+pot3 − (dep2+dep3)`, mirroring [`triplet_update_with_f`](crate::functional::plasticity::triplet_update_with_f) | `TripletState` | |
//! | `RewardStdp` | R-STDP: eligibility accumulates `pot − dep`; weight changes when reward arrives (`w += reward_lr·reward·elig`) — engine `apply_reward` semantics, mirrored per synapse | `RewardState` | |
//! | `InhibitoryStdp` | Vogels inhibitory STDP — reuses the engine formula (`w = −clamp(−w + dm, 0, wmax)`) | `InhibitoryState` | only with Dale's principle, as in the engine |
//! | `Clopath` | Voltage-based LTP/depression | `ClopathState` | documented form, see below |
//! | `Bcm` | BCM sliding-threshold rule, canonical form per the BCM theory (Wikipedia formulation): `dw = η·c·(c − θ̄_M)`, `θ̄_M = EMA_τ{c²}` with `c = w` (weight-as-activation proxy) | `BcmState` | |
//! | `Eprop` | Delegated to [`EpropTrainer`](super::eprop::EpropTrainer) | trainer's own state | typed gate `UnsupportedEpropDynamics` for unsupported models |
//! | `Custom(id)` | User rule ([`PlasticityRule`]) | user-defined | registered in [`PlasticityPipelineConfig::custom_rules`] |
//!
//! # Clopath (documented form)
//!
//! There is no single canonical constant set reachable offline (the original paper is not
//! accessible from this environment); this is the commonly implemented Clopath-family form.
//! State: filtered pre spikes `Ṽ_pre` and post membrane `Ṽ_post` (both with `τ_v`).
//!
//! ```text
//! LTP:  arrival of spike at the synapse AND Ṽ_post > θ_LTP  →  dw += A_LTP·(Ṽ_post − θ_LTP)
//! DEP:  spike of the source AND Ṽ_post > θ_DEP             →  dw −= A_DEP·Ṽ_post
//! ```
//!
//! # BCM (documented form, matches the BCM theory formulation)
//!
//! ```text
//! c(t) = w_j(t)                          (postsynaptic activation proxy = weight)
//! θ̄_M(t) += (c² − θ̄_M)/τ_mean           (EMA of c² — sliding modification threshold)
//! dw = η·c·(c − θ̄_M)                     (LTP when c > θ̄_M, LTD when c < θ̄_M)
//! ```
//!
//! # Composition
//!
//! For each assigned synapse with nonempty rules:
//! 1. rules run in list order; each returns `dw` (mirrored formulas above);
//! 2. `dw_i *= (1 + m(s))` — modulator signal `m` of the synapse (sum of active modulator
//!    events targeting the synapse or its dst neuron);
//! 3. `RuleEntry.weight` multiplies the rule's delta (default 1);
//! 4. total `dw` is clamped to `config.max_abs_delta` if set;
//! 5. written through [`Topology::set_synapse_weight`](super::topology::Topology::set_synapse_weight)
//!    — the single projection point (ParamLimits + Dale invariant) — **once per synapse per step**;
//! 6. traces update after `dw` is computed, so the next step sees the new trace.
//!
//! The engine is strictly Off when a pipeline is attached (enforced by
//! [`DynamicNetwork::set_plasticity_pipeline`](super::network::DynamicNetwork::set_plasticity_pipeline));
//! no accidental double updates.
//!
//! # Checkpoint
//!
//! [`PlasticityPipeline::to_bytes`]/[`from_bytes`] (magic `"PLP1"`) serialize the config
//! (assignments, per-rule params, modulators, composition) and per-synapse rule states.
//! Custom rules (the `Arc<dyn PlasticityRule>` objects) and the e-prop trainer config are
//! NOT serialized — re-supply them via [`PlasticityPipeline::from_bytes`] (the restored
//! pipeline carries `eprop_state` for the next [`DynamicNetwork::set_plasticity_pipeline`]).

use std::collections::{BTreeMap, BTreeSet, VecDeque};
use std::fmt::Debug;
use std::sync::Arc;

use crate::functional::plasticity::WeightBound;
use crate::scalar::Scalar;

use super::eprop::{EpropConfig, EpropState, EpropTrainer};
use super::error::{StructuralError, StructuralResult};
use super::model::{NeuronPolarity, NeuronRole};
use super::network::DynamicNetwork;
use super::topology::{NeuronId, Synapse, SynapseId};

// ---------------------------------------------------------------- RuleId

/// Identifier of a rule assignable to synapses by [`PlasticityPipeline`].
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash, Default)]
pub enum RuleId {
    /// No plasticity (the synapse is inert).
    #[default]
    None,
    /// Engine pair-STDP (mirrors [`functional::plasticity::pair_update`](crate::functional::plasticity::pair_update)).
    PairStdp,
    /// Triplet STDP (Pfister & Gerstner; mirrors [`functional::plasticity::triplet_update_with_f`](crate::functional::plasticity::triplet_update_with_f)).
    TripletStdp,
    /// Reward-modulated STDP: eligibility accumulates `pot − dep`; weight changes
    /// when a reward is pending (`w += reward_lr·reward·elig`) — engine R-STDP semantics.
    RewardStdp,
    /// Vogels inhibitory STDP (reuses the engine formula).
    InhibitoryStdp,
    /// Clopath voltage-based LTP/depression.
    Clopath,
    /// BCM sliding-threshold rule.
    Bcm,
    /// e-prop, delegated to the [`EpropTrainer`] owned by the pipeline.
    Eprop,
    /// User-defined rule (`Custom(id)` — id registered in [`PlasticityPipelineConfig::custom_rules`]).
    Custom(String),
}

impl RuleId {
    /// Stable identifier string (used in serialization and diagnostics).
    pub fn as_str(&self) -> String {
        match self {
            RuleId::None => "none".into(),
            RuleId::PairStdp => "pair_stdp".into(),
            RuleId::TripletStdp => "triplet_stdp".into(),
            RuleId::RewardStdp => "reward_stdp".into(),
            RuleId::InhibitoryStdp => "inhibitory_stdp".into(),
            RuleId::Clopath => "clopath".into(),
            RuleId::Bcm => "bcm".into(),
            RuleId::Eprop => "eprop".into(),
            RuleId::Custom(s) => format!("custom:{s}"),
        }
    }
}

// ---------------------------------------------------------------- RuleEntry

/// One rule assignment: rule identifier + delta multiplier.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct RuleEntry<T: Scalar = f32> {
    /// Rule to assign.
    pub id: RuleId,
    /// Delta multiplier of this rule's contribution (default 1).
    pub weight: T,
}

impl<T: Scalar> RuleEntry<T> {
    /// Entry with weight 1.
    pub fn new(id: RuleId) -> Self {
        Self { id, weight: T::ONE }
    }
}

// ---------------------------------------------------------------- RuleClass

/// Predicate: "a class of synapses/mask" target ([`PlasticityPipelineConfig::class_rules`]).
#[derive(Debug, Clone, PartialEq)]
pub enum RuleClass<T: Scalar = f32> {
    /// Match on source/destination polarity (`None` = wildcard).
    ByPolarity {
        /// Source polarity (excitatory/inhibitory/modulatory) or wildcard.
        src: Option<NeuronPolarity>,
        /// Destination polarity or wildcard.
        dst: Option<NeuronPolarity>,
    },
    /// Source neuron role.
    BySrcRole(NeuronRole),
    /// Destination neuron role.
    ByDstRole(NeuronRole),
    /// Source neuron id.
    BySrcNeuron(NeuronId),
    /// Destination neuron id.
    ByDstNeuron(NeuronId),
    /// Weight magnitude in `[min, max]`.
    ByWeightRange { min: T, max: T },
    /// Synapse age ≥ `min_steps` (steps since birth).
    ByAge { min_steps: u64 },
}

impl<T: Scalar> RuleClass<T> {
    /// Does this class match synapse `s` of `net`?
    pub fn matches(&self, net: &DynamicNetwork<T>, s: &Synapse<T>) -> bool {
        let polarity = |id: NeuronId| net.topology().neuron(id).map(|n| n.polarity);
        match self {
            RuleClass::ByPolarity { src, dst } => {
                if let Some(r) = src {
                    if polarity(s.src) != Some(*r) {
                        return false;
                    }
                }
                if let Some(r) = dst {
                    if polarity(s.dst) != Some(*r) {
                        return false;
                    }
                }
                true
            }
            RuleClass::BySrcRole(r) => net.topology().neuron(s.src).is_some_and(|n| n.role == *r),
            RuleClass::ByDstRole(r) => net.topology().neuron(s.dst).is_some_and(|n| n.role == *r),
            RuleClass::BySrcNeuron(id) => s.src == *id,
            RuleClass::ByDstNeuron(id) => s.dst == *id,
            RuleClass::ByWeightRange { min, max } => s.weight >= *min && s.weight <= *max,
            RuleClass::ByAge { min_steps } => {
                net.step_count().saturating_sub(s.meta.born_step) >= *min_steps
            }
        }
    }
}

// ---------------------------------------------------------------- RuleConfig

/// Per-rule hyperparameters (mirrors the engine constants where they exist).
#[derive(Debug, Clone, PartialEq)]
pub enum RuleConfig<T = f32> {
    /// Engine pair-STDP (mirror of [`functional::plasticity::pair_update`](crate::functional::plasticity::pair_update)).
    PairStdp {
        /// Potentiation amplitude `A+`.
        a_plus: T,
        /// Depression amplitude `A−`.
        a_minus: T,
        /// Trace decay per step.
        trace_decay: T,
        /// Weight bound for the composed delta (engine `StdpConfig.bound`).
        bound: WeightBound<T>,
    },
    /// Triplet STDP (Pfister & Gerstner; mirrors [`functional::plasticity::triplet_update_with_f`](crate::functional::plasticity::triplet_update_with_f)).
    TripletStdp {
        /// Pair potentiation `a2+`.
        a2_plus: T,
        /// Pair depression `a2−`.
        a2_minus: T,
        /// Triplet potentiation `a3+` (slow PRE trace).
        a3_plus: T,
        /// Triplet depression `a3−` (slow POST trace).
        a3_minus: T,
        /// Fast trace decay per step.
        trace_decay: T,
        /// Weight bound for the composed delta.
        bound: WeightBound<T>,
    },
    /// R-STDP: eligibility accumulates `pot − dep`; weight changes when a reward is pending.
    RewardStdp {
        /// Potentiation amplitude `A+` (per arrival).
        a_plus: T,
        /// Depression amplitude `A−` (per dst spike).
        a_minus: T,
        /// Trace decay per step.
        trace_decay: T,
        /// Eligibility decay per step (engine `StdpConfig.elig_decay`).
        elig_decay: T,
        /// Reward learning rate (engine `StdpConfig.reward_lr`).
        reward_lr: T,
        /// Weight bound for the reward step (engine `StdpConfig.bound`).
        bound: WeightBound<T>,
    },
    /// Vogels inhibitory STDP (reuses the engine formula; requires Dale).
    InhibitoryStdp {
        /// Learning rate `η`.
        eta: T,
        /// Target receiver rate (spikes per step).
        target_rate: T,
        /// Trace decay per step (engine `StdpConfig.trace_decay`).
        trace_decay: T,
    },
    /// Clopath voltage-based LTP/depression.
    Clopath {
        /// LTP amplitude per post-voltage unit above threshold (on arrival).
        a_ltp: T,
        /// Depression amplitude on source spike (with filtered post voltage).
        a_dep: T,
        /// LTP threshold on the filtered post voltage.
        theta_ltp: T,
        /// DEP threshold on the filtered post voltage.
        theta_dep: T,
        /// Filter time constant (steps) of the voltage traces.
        tau_v: T,
    },
    /// BCM sliding-threshold rule (canonical form, see module docs).
    Bcm {
        /// Learning rate `η`.
        eta: T,
        /// Time constant (steps) of the sliding threshold EMA of `c²`.
        tau_mean: T,
    },
}

impl<T: Scalar> Default for RuleConfig<T> {
    fn default() -> Self {
        // Engine `StdpConfig::default()` values (pair rule).
        RuleConfig::PairStdp {
            a_plus: T::c(0.01),
            a_minus: T::c(0.012),
            trace_decay: (-T::ONE / T::c(20.0)).exp(),
            bound: WeightBound::Hard {
                w_min: -T::ONE,
                w_max: T::ONE,
            },
        }
    }
}

impl<T: Scalar> RuleConfig<T> {
    fn bound_of(&self) -> Option<WeightBound<T>> {
        match self {
            RuleConfig::PairStdp { bound, .. }
            | RuleConfig::TripletStdp { bound, .. }
            | RuleConfig::RewardStdp { bound, .. } => Some(*bound),
            RuleConfig::InhibitoryStdp { .. }
            | RuleConfig::Clopath { .. }
            | RuleConfig::Bcm { .. } => None,
        }
    }

    fn validate(&self) -> StructuralResult<()> {
        let finite = |v: &T| v.is_finite();
        let unit = |v: &T| v.is_finite() && *v > T::ZERO && *v <= T::ONE;
        let err = |msg: &str| StructuralError::param("plasticity rule config", msg);
        match self {
            RuleConfig::PairStdp {
                a_plus,
                a_minus,
                trace_decay,
                ..
            } => {
                if !finite(a_plus) || !finite(a_minus) || !unit(trace_decay) {
                    return Err(err("PairStdp: a_plus/a_minus finite, trace_decay ∈ (0, 1]"));
                }
            }
            RuleConfig::TripletStdp {
                a2_plus,
                a2_minus,
                a3_plus,
                a3_minus,
                trace_decay,
                ..
            } => {
                if !finite(a2_plus)
                    || !finite(a2_minus)
                    || !finite(a3_plus)
                    || !finite(a3_minus)
                    || !unit(trace_decay)
                {
                    return Err(err(
                        "TripletStdp: a2/a3 coefficients finite, trace_decay ∈ (0, 1]",
                    ));
                }
            }
            RuleConfig::RewardStdp {
                a_plus,
                a_minus,
                trace_decay,
                elig_decay,
                reward_lr,
                ..
            } => {
                if !finite(a_plus)
                    || !finite(a_minus)
                    || !unit(trace_decay)
                    || !unit(elig_decay)
                    || !finite(reward_lr)
                {
                    return Err(err(
                        "RewardStdp: a_plus/a_minus/reward_lr finite, decays ∈ (0, 1]",
                    ));
                }
            }
            RuleConfig::InhibitoryStdp {
                eta,
                target_rate,
                trace_decay,
            } => {
                if !finite(eta)
                    || !finite(target_rate)
                    || *target_rate < T::ZERO
                    || !unit(trace_decay)
                {
                    return Err(err(
                        "InhibitoryStdp: eta/target_rate finite, target_rate ≥ 0, trace_decay ∈ (0, 1]",
                    ));
                }
            }
            RuleConfig::Clopath {
                a_ltp,
                a_dep,
                theta_ltp,
                theta_dep,
                tau_v,
            } => {
                if !finite(a_ltp)
                    || !finite(a_dep)
                    || !finite(theta_ltp)
                    || !finite(theta_dep)
                    || !unit(tau_v)
                {
                    return Err(err("Clopath: a_ltp/a_dep/theta_* finite, tau_v ∈ (0, 1]"));
                }
            }
            RuleConfig::Bcm { eta, tau_mean } => {
                if !finite(eta) || !finite(tau_mean) || *tau_mean <= T::ZERO {
                    return Err(err("Bcm: eta finite, tau_mean > 0"));
                }
            }
        }
        Ok(())
    }

    fn default_for(id: &RuleId) -> Self {
        match id {
            RuleId::PairStdp => RuleConfig::default(),
            RuleId::TripletStdp => RuleConfig::TripletStdp {
                a2_plus: T::c(0.02),
                a2_minus: T::c(0.02),
                a3_plus: T::c(0.005),
                a3_minus: T::c(0.005),
                trace_decay: (-T::ONE / T::c(20.0)).exp(),
                bound: WeightBound::Hard {
                    w_min: -T::ONE,
                    w_max: T::ONE,
                },
            },
            RuleId::RewardStdp => RuleConfig::RewardStdp {
                a_plus: T::c(0.01),
                a_minus: T::c(0.012),
                trace_decay: (-T::ONE / T::c(20.0)).exp(),
                elig_decay: (-T::ONE / T::c(50.0)).exp(),
                reward_lr: T::ONE,
                bound: WeightBound::Hard {
                    w_min: -T::ONE,
                    w_max: T::ONE,
                },
            },
            RuleId::InhibitoryStdp => RuleConfig::InhibitoryStdp {
                eta: T::c(0.001),
                target_rate: T::c(0.02),
                trace_decay: (-T::ONE / T::c(20.0)).exp(),
            },
            RuleId::Clopath => RuleConfig::Clopath {
                a_ltp: T::c(0.05),
                a_dep: T::c(0.01),
                theta_ltp: T::c(0.5),
                theta_dep: T::c(0.5),
                tau_v: (-T::ONE / T::c(20.0)).exp(),
            },
            RuleId::Bcm => RuleConfig::Bcm {
                eta: T::c(0.01),
                tau_mean: T::c(1000.0),
            },
            _ => RuleConfig::default(),
        }
    }
}

// ---------------------------------------------------------------- per-rule states

/// Per-synapse state of the `PairStdp` rule (pre/post traces).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct PairState<T = f32> {
    /// Trace of source spikes.
    pub pre_trace: T,
    /// Trace of destination spikes.
    pub post_trace: T,
}

/// Per-synapse state of the `TripletStdp` rule (fast/slow traces of both sides).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct TripletState<T = f32> {
    /// Fast pre trace.
    pub pre_fast: T,
    /// Slow pre trace.
    pub pre_slow: T,
    /// Fast post trace.
    pub post_fast: T,
    /// Slow post trace.
    pub post_slow: T,
}

/// Per-synapse state of the `RewardStdp` rule (eligibility + pair traces).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct RewardState<T = f32> {
    /// Eligibility trace (pipeline-owned analog of `Synapse.elig`).
    pub elig: T,
    /// Trace of source spikes.
    pub pre_trace: T,
    /// Trace of destination spikes.
    pub post_trace: T,
}

/// Per-synapse state of the `InhibitoryStdp` rule (Vogels, pre/post traces).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct InhibitoryState<T = f32> {
    /// Trace of source spikes.
    pub pre_trace: T,
    /// Trace of destination spikes.
    pub post_trace: T,
}

/// Per-synapse state of the `Clopath` rule (filtered pre/post voltages).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct ClopathState<T = f32> {
    /// Filtered pre-spike trace `Ṽ_pre`.
    pub v_pre: T,
    /// Filtered post-membrane trace `Ṽ_post`.
    pub v_post: T,
}

/// Per-synapse state of the `Bcm` rule (sliding threshold of `EMA(w²)`).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct BcmState<T = f32> {
    /// Sliding modification threshold `θ̄_M`.
    pub y_th: T,
}

/// Per-synapse state of built-in rules.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct RuleState<T: Scalar = f32> {
    /// State of [`RuleId::PairStdp`].
    pub pair: Option<PairState<T>>,
    /// State of [`RuleId::TripletStdp`].
    pub triplet: Option<TripletState<T>>,
    /// State of [`RuleId::RewardStdp`].
    pub reward: Option<RewardState<T>>,
    /// State of [`RuleId::InhibitoryStdp`].
    pub inhibitory: Option<InhibitoryState<T>>,
    /// State of [`RuleId::Clopath`].
    pub clopath: Option<ClopathState<T>>,
    /// State of [`RuleId::Bcm`].
    pub bcm: Option<BcmState<T>>,
}

// ---------------------------------------------------------------- context & custom rule

/// Per-synapse update context passed to built-in rules and to [`PlasticityRule::update`].
#[derive(Debug, Clone, Copy)]
pub struct PipelineCtx<'a, T: Scalar = f32> {
    /// Step counter (`net.step_count()` after the completed step).
    pub step: u64,
    /// The synapse being updated.
    pub synapse: &'a Synapse<T>,
    /// The synapse delivered current to its receiver this step (arrival).
    pub arrival: bool,
    /// Source neuron spiked this step.
    pub src_spike: bool,
    /// Destination neuron spiked this step.
    pub dst_spike: bool,
    /// Postsynaptic membrane voltage of the destination (`NeuronState::mem()`).
    pub post_voltage: T,
    /// Source neuron polarity (for Vogels-style rules).
    pub polarity: NeuronPolarity,
    /// Pending reward (from [`DynamicNetwork::apply_reward`]), consumed by `RewardStdp`.
    pub reward: T,
    /// Modulator signal on this synapse (delta is multiplied by `1 + m`).
    pub modulator: T,
    /// `ParamLimits::weight_abs_max` of the topology (for Vogels-style clamps).
    pub wmax: T,
    /// Current learnable delay of the synapse (`None` if learnable delays are off);
    /// rules may adjust it via `DynamicNetwork::set_learnable_delay`.
    pub learnable_delay: Option<T>,
}

/// User-defined pipeline rule. State is the rule's business: store per-synapse state
/// in an internal `Mutex<BTreeMap<SynapseId, _>>` if needed.
pub trait PlasticityRule<T: Scalar>: Send + Sync + Debug {
    /// Stable rule id (must match `RuleId::Custom(id)`).
    fn id(&self) -> String;
    /// Per-synapse update; returns the weight delta (without the `RuleEntry.weight` multiplier).
    fn update(&self, synapse: SynapseId, ctx: &PipelineCtx<T>) -> StructuralResult<T>;
}

// ---------------------------------------------------------------- modulators

/// Neuromodulator channel (semantic label; not a biochemical claim).
#[derive(Debug, Clone, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ModulatorChannel {
    /// Dopamine (reward/motivation signal).
    Dopamine,
    /// Serotonin.
    Serotonin,
    /// Acetylcholine.
    Acetylcholine,
    /// Noradrenaline.
    Noradrenaline,
    /// Novelty (deviation from expectation).
    Novelty,
    /// Prediction error (temporal difference).
    PredictionError,
    /// User-defined label.
    Custom(String),
}

/// Modulator target (addressable entity).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum ModulatorTarget {
    /// A synapse.
    Synapse(SynapseId),
    /// A neuron (signal applies to its incoming synapses, like the engine's modulatory path).
    Neuron(NeuronId),
}

/// A modulator event: an instantaneous signal to an entity.
#[derive(Debug, Clone, PartialEq)]
pub struct ModulatorEvent<T = f32> {
    /// Channel (semantic label).
    pub channel: ModulatorChannel,
    /// Signal value.
    pub value: T,
    /// Step at which the signal was sent.
    pub at_step: u64,
    /// Where the signal lands.
    pub target: ModulatorTarget,
    /// Lifetime in pipeline steps: the event is active (counted in
    /// [`PlasticityPipeline::modulator_signal`]) while `pipeline_step < at_step + lifetime`;
    /// `None` = indefinite.
    pub lifetime: Option<u64>,
}

/// Modulator plumbing policy (events are queued via [`PlasticityPipeline::apply_modulator`]).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ModulatorConfig<T = f32> {
    /// Per-step decay of active modulator values (default 0.95).
    pub decay: T,
}

impl<T: Scalar> Default for ModulatorConfig<T> {
    fn default() -> Self {
        Self { decay: T::c(0.95) }
    }
}

#[derive(Debug, Clone)]
struct ModulatedEvent<T: Scalar> {
    event: ModulatorEvent<T>,
    until_step: Option<u64>,
}

// ---------------------------------------------------------------- pipeline config

/// Pipeline configuration (assignments + rule hyperparameters + composition + modulators).
#[derive(Debug, Clone, Default)]
pub struct PlasticityPipelineConfig<T: Scalar = f32> {
    /// Default rules for synapses without a specific assignment (execution order).
    pub default_rules: Vec<RuleEntry<T>>,
    /// Per-synapse overrides: full replacement list (empty = plasticity off for the synapse).
    pub synapse_rules: BTreeMap<SynapseId, Vec<RuleEntry<T>>>,
    /// Class rules (first match wins, evaluated before `default_rules`).
    pub class_rules: Vec<(RuleClass<T>, Vec<RuleEntry<T>>)>,
    /// Hyperparameters per built-in rule (missing = [`RuleConfig::default`]).
    pub rule_configs: BTreeMap<RuleId, RuleConfig<T>>,
    /// User rules by id (required for `RuleId::Custom`).
    pub custom_rules: BTreeMap<String, Arc<dyn PlasticityRule<T>>>,
    /// E-prop trainer config — required when `RuleId::Eprop` is assigned.
    pub eprop_config: Option<EpropConfig<T>>,
    /// Modulator plumbing.
    pub modulators: ModulatorConfig<T>,
    /// Optional cap on the total per-step delta of a synapse (applied after summation).
    pub max_abs_delta: Option<T>,
}

// ---------------------------------------------------------------- report

/// Result of one [`PlasticityPipeline::apply`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct PlasticityReport {
    /// Pipeline step counter (incremented once per apply).
    pub steps: u64,
    /// Number of synapses whose weights were written this step.
    pub applied_synapses: usize,
}

// ---------------------------------------------------------------- pipeline

/// Unified per-synapse plasticity pipeline: rule assignments, per-rule state, composition,
/// modulators, e-prop delegation.
///
/// # Example
///
/// ```
/// use neo_snn::structural::*;
///
/// // Network: Input → Output (excitatory LIF).
/// let mut topo = Topology::new(StructuralConfig::<f32>::default().topology, StructuralConfig::<f32>::default().params);
/// let b = Birth::default();
/// let i = topo.add_neuron(NeuronKind::Input, LifParams::default(), Origin::Initial, b).unwrap();
/// let o = topo.add_neuron(NeuronKind::Output, LifParams::new(0.9, 0.4), Origin::Initial, b).unwrap();
/// topo.add_synapse(i, o, 0.5, 0, Origin::Initial, b).unwrap();
///
/// let mut pl_cfg = PlasticityPipelineConfig::default();
/// pl_cfg.default_rules = vec![RuleEntry::new(RuleId::PairStdp)];
/// let mut net = DynamicNetwork::new(topo, EngineConfig::default()).unwrap();
/// net.set_plasticity_pipeline(Some(PlasticityPipeline::new(pl_cfg))).unwrap();
/// let before = net.topology().synapses().next().unwrap().weight;
/// net.step(&[1.0]).unwrap();
/// net.step(&[1.0]).unwrap();
/// let after = net.topology().synapses().next().unwrap().weight;
/// assert_ne!(before, after);
/// ```
#[derive(Debug, Clone)]
pub struct PlasticityPipeline<T: Scalar = f32> {
    config: PlasticityPipelineConfig<T>,
    states: BTreeMap<SynapseId, RuleState<T>>,
    eprop: Option<EpropTrainer<T>>,
    /// E-prop state for serialization roundtrip (restored on next `attach`).
    eprop_state: Option<EpropState<T>>,
    steps: u64,
    reward: Option<T>,
    modulator_events: VecDeque<ModulatedEvent<T>>,
}

impl<T: Scalar> PlasticityPipeline<T> {
    /// Create an unattached pipeline (call [`DynamicNetwork::set_plasticity_pipeline`] to attach).
    pub fn new(config: PlasticityPipelineConfig<T>) -> Self {
        Self {
            config,
            states: BTreeMap::new(),
            eprop: None,
            eprop_state: None,
            steps: 0,
            reward: None,
            modulator_events: VecDeque::new(),
        }
    }

    /// Pipeline configuration.
    pub fn config(&self) -> &PlasticityPipelineConfig<T> {
        &self.config
    }

    /// Per-synapse rule states.
    pub fn states(&self) -> &BTreeMap<SynapseId, RuleState<T>> {
        &self.states
    }

    /// Pipeline step counter.
    pub fn steps(&self) -> u64 {
        self.steps
    }

    /// E-prop trainer (if a `RuleId::Eprop` was assigned).
    pub fn eprop_trainer(&self) -> Option<&EpropTrainer<T>> {
        self.eprop.as_ref()
    }

    // -- validation / attach ------------------------------------------------

    /// Validate the config against the network (assignment targets, rule configs, e-prop presence).
    pub fn validate(&self, net: &DynamicNetwork<T>) -> StructuralResult<()> {
        let mut assigned: Vec<RuleId> = self
            .config
            .default_rules
            .iter()
            .map(|e| e.id.clone())
            .chain(
                self.config
                    .synapse_rules
                    .values()
                    .flatten()
                    .map(|e| e.id.clone()),
            )
            .chain(
                self.config
                    .class_rules
                    .iter()
                    .flat_map(|(_, v)| v.iter().map(|e| e.id.clone())),
            )
            .collect();
        assigned.sort_by_key(|r| r.as_str());
        assigned.dedup();
        for rid in assigned {
            match rid {
                RuleId::Custom(ref id) => {
                    if !self.config.custom_rules.contains_key(id) {
                        return Err(StructuralError::InvalidParameter {
                            what: "plasticity pipeline",
                            reason: format!(
                                "custom rule \"{id}\" is not registered in `PlasticityPipelineConfig::custom_rules`"
                            ),
                        });
                    }
                }
                RuleId::Eprop => {
                    if self.config.eprop_config.is_none() {
                        return Err(StructuralError::InvalidParameter {
                            what: "plasticity pipeline",
                            reason: "`RuleId::Eprop` assigned but `eprop_config` is not set".into(),
                        });
                    }
                }
                _ => {}
            }
        }
        for id in self.config.synapse_rules.keys() {
            net.topology()
                .synapse(*id)
                .ok_or(StructuralError::NoSuchSynapse(*id))?;
        }
        for (c, _) in &self.config.class_rules {
            match c {
                RuleClass::BySrcNeuron(n) | RuleClass::ByDstNeuron(n) => {
                    net.topology()
                        .neuron(*n)
                        .ok_or(StructuralError::NoSuchNeuron(*n))?;
                }
                _ => {}
            }
        }
        for (rid, cfg) in &self.config.rule_configs {
            cfg.validate()
                .map_err(|e| StructuralError::InvalidParameter {
                    what: "plasticity rule config",
                    reason: format!("{}: {e}", rid.as_str()),
                })?;
        }
        if let Some(m) = self.config.max_abs_delta {
            if !m.is_finite() || m < T::ZERO {
                return Err(StructuralError::param(
                    "plasticity pipeline",
                    "max_abs_delta must be finite and >= 0",
                ));
            }
        }
        let d = self.config.modulators.decay;
        if !(d > T::ZERO && d <= T::ONE) {
            return Err(StructuralError::param(
                "modulators.decay",
                "must be in (0, 1]",
            ));
        }
        Ok(())
    }

    /// Attach: validate + create the e-prop trainer if an e-prop rule is assigned.
    pub fn attach(&mut self, net: &DynamicNetwork<T>) -> StructuralResult<()> {
        self.validate(net)?;
        if self.eprop_assigned() {
            let cfg = self.config.eprop_config.clone().ok_or_else(|| {
                StructuralError::InvalidParameter {
                    what: "plasticity pipeline",
                    reason: "`RuleId::Eprop` assigned but `eprop_config` is not set".into(),
                }
            })?;
            let mut t = EpropTrainer::new(net, cfg)?;
            if let Some(st) = self.eprop_state.take() {
                t.load_state(&st.to_bytes()?)?;
            }
            self.eprop = Some(t);
        }
        Ok(())
    }

    fn eprop_assigned(&self) -> bool {
        self.config
            .default_rules
            .iter()
            .any(|e| e.id == RuleId::Eprop)
            || self
                .config
                .synapse_rules
                .values()
                .flatten()
                .any(|e| e.id == RuleId::Eprop)
            || self
                .config
                .class_rules
                .iter()
                .flat_map(|(_, v)| v.iter())
                .any(|e| e.id == RuleId::Eprop)
    }

    fn rule_config(&self, id: &RuleId) -> RuleConfig<T> {
        self.config
            .rule_configs
            .get(id)
            .cloned()
            .unwrap_or_else(|| RuleConfig::default_for(id))
    }

    // -- assignments ----------------------------------------------------------

    /// Resolve the rule list for a synapse: synapse override > first class match > network default.
    pub fn resolve(&self, net: &DynamicNetwork<T>, s: &Synapse<T>) -> Vec<RuleEntry<T>> {
        if let Some(v) = self.config.synapse_rules.get(&s.id) {
            return v.clone();
        }
        if let Some((_, v)) = self
            .config
            .class_rules
            .iter()
            .find(|(c, _)| c.matches(net, s))
        {
            return v.clone();
        }
        self.config.default_rules.clone()
    }

    // -- modulators -------------------------------------------------------------

    /// Set a pending reward consumed by `RewardStdp` at the next [`apply`](Self::apply).
    pub fn set_reward(&mut self, reward: T) {
        if reward.is_finite() {
            self.reward = Some(reward);
        }
    }

    /// Active modulator signal on a synapse: sum of active events targeting the
    /// synapse or its dst neuron (active while `now < at_step + lifetime`).
    pub fn modulator_signal(&self, net: &DynamicNetwork<T>, s: &Synapse<T>) -> T {
        let now = net.step_count();
        let mut sum = T::ZERO;
        for ev in &self.modulator_events {
            if let Some(u) = ev.until_step {
                if u <= now {
                    continue;
                }
            }
            let targeted = match ev.event.target {
                ModulatorTarget::Synapse(id) => id == s.id,
                ModulatorTarget::Neuron(id) => id == s.dst,
            };
            if targeted {
                sum += ev.event.value;
            }
        }
        sum
    }

    /// Queue a modulator event (sent at `event.at_step`; active while
    /// `pipeline_step < at_step + lifetime`; decays while active).
    pub fn apply_modulator(&mut self, event: ModulatorEvent<T>) {
        let until_step = event.lifetime.map(|l| event.at_step.saturating_add(l));
        self.modulator_events
            .push_back(ModulatedEvent { event, until_step });
    }

    // -- main step ---------------------------------------------------------------

    /// One pipeline step: decay + prune modulators, run e-prop (writes weights),
    /// compute per-synapse rule deltas, write them through
    /// [`Topology::set_synapse_weight`](super::topology::Topology::set_synapse_weight).
    pub fn apply(&mut self, net: &mut DynamicNetwork<T>) -> StructuralResult<PlasticityReport> {
        // 1. Decay + prune modulator events.
        let now = net.step_count();
        let decay = self.config.modulators.decay;
        for ev in &mut self.modulator_events {
            ev.event.value = ev.event.value * decay;
        }
        self.modulator_events
            .retain(|ev| ev.until_step.map_or(true, |u| u > now));

        // 2. e-prop delegation (own path, writes via `set_synapse_weight`).
        if let Some(t) = &mut self.eprop {
            t.step(net)?;
        }

        // 3. Per-synapse pipeline deltas.
        let step = net.step_count();
        let reward = self.reward.take().unwrap_or(T::ZERO);
        let arrivals: BTreeSet<SynapseId> = net.last_arrivals().iter().copied().collect();
        let mut spikes: BTreeMap<NeuronId, bool> = BTreeMap::new();
        for (n, s) in net
            .topology()
            .neurons()
            .zip(net.last_spikes().iter().copied())
        {
            spikes.insert(n.id, s);
        }
        let mut plan: Vec<(SynapseId, T)> = Vec::new();
        let synapses: Vec<SynapseId> = net.topology().synapses().map(|s| s.id).collect();
        for id in synapses {
            let Some(s) = net.topology().synapse(id) else {
                continue;
            };
            if !s.is_active() {
                continue;
            }
            let rules = self.resolve(net, s);
            if rules.is_empty() {
                continue;
            }
            // Synapses managed by the delegated e-prop trainer: pipeline deltas skip.
            let eprop_managed = self
                .eprop
                .as_ref()
                .is_some_and(|t| t.is_enabled_synapse(net, id))
                && rules.iter().any(|r| r.id == RuleId::Eprop);
            if eprop_managed {
                continue;
            }
            // Precompute configs/modulator so `st` (a `&mut` borrow of `self.states`)
            // doesn't overlap other `self` reads.
            let rule_confs: Vec<(RuleId, RuleConfig<T>)> = rules
                .iter()
                .map(|r| (r.id.clone(), self.rule_config(&r.id)))
                .collect();
            let modulator = self.modulator_signal(net, s);
            let st = self.states.entry(id).or_default();
            let ctx = PipelineCtx {
                step,
                synapse: s,
                arrival: arrivals.contains(&id),
                src_spike: spikes.get(&s.src).copied().unwrap_or(false),
                dst_spike: spikes.get(&s.dst).copied().unwrap_or(false),
                post_voltage: net
                    .topology()
                    .neuron(s.dst)
                    .map_or(T::ZERO, |n| n.state.mem()),
                polarity: net
                    .topology()
                    .neuron(s.src)
                    .map_or(NeuronPolarity::Excitatory, |n| n.polarity),
                reward,
                modulator,
                wmax: net.topology().param_limits().weight_abs_max,
                learnable_delay: net.learnable_delay(id),
            };
            let mut dw = T::ZERO;
            for (r, (rid, cfg)) in rules.iter().zip(&rule_confs) {
                let d = match rid {
                    RuleId::None => T::ZERO,
                    RuleId::PairStdp => BuiltIn::PairStdp.update(st, cfg, &ctx)?,
                    RuleId::TripletStdp => BuiltIn::TripletStdp.update(st, cfg, &ctx)?,
                    RuleId::RewardStdp => BuiltIn::RewardStdp.update(st, cfg, &ctx)?,
                    RuleId::InhibitoryStdp => BuiltIn::InhibitoryStdp.update(st, cfg, &ctx)?,
                    RuleId::Clopath => BuiltIn::Clopath.update(st, cfg, &ctx)?,
                    RuleId::Bcm => BuiltIn::Bcm.update(st, cfg, &ctx)?,
                    RuleId::Eprop => T::ZERO,
                    RuleId::Custom(id) => match self.config.custom_rules.get(id.as_str()) {
                        Some(rule) => rule.update(s.id, &ctx)?,
                        None => T::ZERO,
                    },
                };
                dw += d * r.weight * (T::ONE + ctx.modulator);
            }
            if let Some(cap) = self.config.max_abs_delta {
                dw = dw.clamp(-cap, cap);
            }
            if dw != T::ZERO {
                let bound = rules
                    .iter()
                    .find_map(|r| self.rule_config(&r.id).bound_of())
                    .unwrap_or(WeightBound::None);
                let w_new = w_with_bound(s.weight, dw, &bound);
                plan.push((id, w_new));
            }
        }
        let applied = plan.len();
        for (id, w) in plan {
            net.topology_mut().set_synapse_weight(id, w)?;
        }
        self.states
            .retain(|id, _| net.topology().synapse(*id).is_some());
        self.steps += 1;
        Ok(PlasticityReport {
            steps: self.steps,
            applied_synapses: applied,
        })
    }
}

fn w_with_bound<T: Scalar>(w: T, dw: T, bound: &WeightBound<T>) -> T {
    let w = w + dw;
    match *bound {
        WeightBound::Hard { w_min, w_max } => w.clamp(w_min, w_max),
        // The pipeline approximates the Soft headroom by the clamp (documented deviation;
        // see module docs).
        WeightBound::Soft { w_min, w_max } => w.clamp(w_min, w_max),
        WeightBound::None => w,
    }
}

// ---------------------------------------------------------------- built-in rules

#[derive(Debug, Clone, Copy)]
enum BuiltIn {
    PairStdp,
    TripletStdp,
    RewardStdp,
    InhibitoryStdp,
    Clopath,
    Bcm,
}

impl BuiltIn {
    fn update<T: Scalar>(
        &self,
        st: &mut RuleState<T>,
        cfg: &RuleConfig<T>,
        ctx: &PipelineCtx<T>,
    ) -> StructuralResult<T> {
        match self {
            BuiltIn::PairStdp => {
                let RuleConfig::PairStdp {
                    a_plus,
                    a_minus,
                    trace_decay,
                    ..
                } = cfg
                else {
                    return Err(mismatch("PairStdp"));
                };
                // Engine pair-STDP (event shape, mirrors `functional::plasticity::pair_update`):
                //   pot on arrival (dst trace), dep when dst spikes (src trace), `dw = pot − dep`.
                let st = st.pair.get_or_insert_with(Default::default);
                let pot = if ctx.arrival {
                    *a_plus * st.post_trace
                } else {
                    T::ZERO
                };
                let dep = if ctx.dst_spike {
                    *a_minus * st.pre_trace
                } else {
                    T::ZERO
                };
                st.pre_trace =
                    st.pre_trace * *trace_decay + if ctx.src_spike { T::ONE } else { T::ZERO };
                st.post_trace =
                    st.post_trace * *trace_decay + if ctx.dst_spike { T::ONE } else { T::ZERO };
                Ok(pot - dep)
            }
            BuiltIn::TripletStdp => {
                let RuleConfig::TripletStdp {
                    a2_plus,
                    a2_minus,
                    a3_plus,
                    a3_minus,
                    trace_decay,
                    ..
                } = cfg
                else {
                    return Err(mismatch("TripletStdp"));
                };
                // Mirrors `functional::plasticity::triplet_update_with_f` (identity hooks):
                //   pot2 = a2+·post_fast·pre_spike (arrival), dep2 = a2−·post_spike·pre_fast (dst spike),
                //   pot3 = a3+·post_fast·(pre_spike·pre_slow) (arrival), dep3 = a3−·(post_spike·post_slow)·pre_fast (dst spike).
                let st = st.triplet.get_or_insert_with(Default::default);
                let mut dw = T::ZERO;
                if ctx.arrival {
                    dw += *a2_plus * st.post_fast + *a3_plus * st.post_fast * st.pre_slow;
                }
                if ctx.dst_spike {
                    dw -= *a2_minus * st.pre_fast + *a3_minus * st.post_slow * st.pre_fast;
                }
                st.pre_fast =
                    st.pre_fast * *trace_decay + if ctx.src_spike { T::ONE } else { T::ZERO };
                st.pre_slow =
                    st.pre_slow * *trace_decay + if ctx.src_spike { T::ONE } else { T::ZERO };
                st.post_fast =
                    st.post_fast * *trace_decay + if ctx.dst_spike { T::ONE } else { T::ZERO };
                st.post_slow =
                    st.post_slow * *trace_decay + if ctx.dst_spike { T::ONE } else { T::ZERO };
                Ok(dw)
            }
            BuiltIn::RewardStdp => {
                let RuleConfig::RewardStdp {
                    a_plus,
                    a_minus,
                    trace_decay,
                    elig_decay,
                    reward_lr,
                    ..
                } = cfg
                else {
                    return Err(mismatch("RewardStdp"));
                };
                // Engine R-STDP (RewardModulated): elig accumulates `pot − dep`;
                // weight changes when a reward is pending (`reward_lr·reward·elig`).
                let st = st.reward.get_or_insert_with(Default::default);
                let pot = if ctx.arrival {
                    *a_plus * st.post_trace
                } else {
                    T::ZERO
                };
                let dep = if ctx.dst_spike {
                    *a_minus * st.pre_trace
                } else {
                    T::ZERO
                };
                st.elig = st.elig * *elig_decay + (pot - dep);
                st.pre_trace =
                    st.pre_trace * *trace_decay + if ctx.src_spike { T::ONE } else { T::ZERO };
                st.post_trace =
                    st.post_trace * *trace_decay + if ctx.dst_spike { T::ONE } else { T::ZERO };
                let mut dw = T::ZERO;
                if ctx.reward != T::ZERO && st.elig != T::ZERO {
                    dw += *reward_lr * ctx.reward * st.elig;
                }
                Ok(dw)
            }
            BuiltIn::InhibitoryStdp => {
                let RuleConfig::InhibitoryStdp {
                    eta,
                    target_rate,
                    trace_decay,
                } = cfg
                else {
                    return Err(mismatch("InhibitoryStdp"));
                };
                // Vogels (2011), engine formula: `dm += eta·(post_trace − α)` on arrival,
                // `dm += eta·pre_trace` on dst spike, `w = −clamp(−w + dm, 0, wmax)`.
                if ctx.polarity != NeuronPolarity::Inhibitory {
                    return Ok(T::ZERO);
                }
                let st = st.inhibitory.get_or_insert_with(Default::default);
                let alpha = T::c(2.0) * *target_rate / (T::ONE - *trace_decay).max(T::c(1e-6));
                let mut dm = T::ZERO;
                if ctx.arrival {
                    dm += *eta * (st.post_trace - alpha);
                }
                if ctx.dst_spike {
                    dm += *eta * st.pre_trace;
                }
                st.pre_trace =
                    st.pre_trace * *trace_decay + if ctx.src_spike { T::ONE } else { T::ZERO };
                st.post_trace =
                    st.post_trace * *trace_decay + if ctx.dst_spike { T::ONE } else { T::ZERO };
                let w = ctx.synapse.weight;
                let w_new = -((-w + dm).clamp(T::ZERO, ctx.wmax));
                Ok(w_new - w)
            }
            BuiltIn::Clopath => {
                let RuleConfig::Clopath {
                    a_ltp,
                    a_dep,
                    theta_ltp,
                    theta_dep,
                    tau_v,
                } = cfg
                else {
                    return Err(mismatch("Clopath"));
                };
                // Documented Clopath-family form (module docs).
                let st = st.clopath.get_or_insert_with(Default::default);
                let v = ctx.post_voltage;
                let v_post_new = st.v_post * *tau_v + v;
                let mut dw = T::ZERO;
                if ctx.arrival && v_post_new > *theta_ltp {
                    dw += *a_ltp * (v_post_new - *theta_ltp);
                }
                if ctx.src_spike && v_post_new > *theta_dep {
                    dw -= *a_dep * v_post_new;
                }
                st.v_pre = st.v_pre * *tau_v + if ctx.src_spike { T::ONE } else { T::ZERO };
                st.v_post = v_post_new;
                Ok(dw)
            }
            BuiltIn::Bcm => {
                let RuleConfig::Bcm { eta, tau_mean } = cfg else {
                    return Err(mismatch("Bcm"));
                };
                // Canonical BCM form: `θ̄_M = EMA_τ{c²}`, `dw = η·c·(c − θ̄_M)`.
                let st = st.bcm.get_or_insert_with(Default::default);
                let c = ctx.synapse.weight;
                st.y_th = st.y_th + (c * c - st.y_th) / *tau_mean;
                Ok(*eta * c * (c - st.y_th))
            }
        }
    }
}

fn mismatch(rule: &str) -> StructuralError {
    StructuralError::InvalidParameter {
        what: "plasticity pipeline",
        reason: format!("{rule} rule requires the matching RuleConfig variant"),
    }
}

// ---------------------------------------------------------------- serialization

const MAGIC: &[u8] = b"PLP1";
const VERSION: u32 = 1;

#[derive(Default)]
struct Enc {
    buf: Vec<u8>,
}
impl Enc {
    fn u8(&mut self, v: u8) {
        self.buf.push(v);
    }
    fn u32(&mut self, v: u32) {
        self.buf.extend_from_slice(&v.to_le_bytes());
    }
    fn u64(&mut self, v: u64) {
        self.buf.extend_from_slice(&v.to_le_bytes());
    }
    fn f64(&mut self, v: f64) {
        self.buf.extend_from_slice(&v.to_le_bytes());
    }
    fn opt_f64<T: Scalar>(&mut self, v: Option<T>) {
        self.u8(v.is_some() as u8);
        if let Some(v) = v {
            self.f64(v.to_f64());
        }
    }
    fn str(&mut self, s: &str) {
        self.u64(s.len() as u64);
        self.buf.extend_from_slice(s.as_bytes());
    }
    fn opt_u64(&mut self, v: Option<u64>) {
        self.u8(v.is_some() as u8);
        if let Some(v) = v {
            self.u64(v);
        }
    }
}

struct Dec<'a> {
    b: &'a [u8],
    p: usize,
}
impl Dec<'_> {
    fn take(&mut self, n: usize) -> StructuralResult<&[u8]> {
        if self.p + n > self.b.len() {
            return Err(StructuralError::param(
                "plasticity pipeline",
                "state bytes truncated",
            ));
        }
        let s = &self.b[self.p..self.p + n];
        self.p += n;
        Ok(s)
    }
    fn u8(&mut self) -> StructuralResult<u8> {
        Ok(self.take(1)?[0])
    }
    fn u32(&mut self) -> StructuralResult<u32> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
    fn u64(&mut self) -> StructuralResult<u64> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }
    fn f64(&mut self) -> StructuralResult<f64> {
        Ok(f64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }
    fn opt_f64<T: Scalar>(&mut self) -> StructuralResult<Option<T>> {
        if self.u8()? == 0 {
            return Ok(None);
        }
        Ok(Some(T::from_f64(self.f64()?)))
    }
    fn str(&mut self) -> StructuralResult<String> {
        let n = self.u64()? as usize;
        let b = self.take(n)?;
        Ok(String::from_utf8_lossy(b).into_owned())
    }
    fn opt_u64(&mut self) -> StructuralResult<Option<u64>> {
        if self.u8()? == 0 {
            return Ok(None);
        }
        Ok(Some(self.u64()?))
    }
}

fn enc_rule_id<T: Scalar>(e: &mut Enc, id: &RuleId) {
    match id {
        RuleId::None => e.u8(0),
        RuleId::PairStdp => e.u8(1),
        RuleId::TripletStdp => e.u8(2),
        RuleId::RewardStdp => e.u8(3),
        RuleId::InhibitoryStdp => e.u8(4),
        RuleId::Clopath => e.u8(5),
        RuleId::Bcm => e.u8(6),
        RuleId::Eprop => e.u8(7),
        RuleId::Custom(s) => {
            e.u8(8);
            e.str(s);
        }
    }
}

fn dec_rule_id<T: Scalar>(d: &mut Dec) -> StructuralResult<RuleId> {
    Ok(match d.u8()? {
        0 => RuleId::None,
        1 => RuleId::PairStdp,
        2 => RuleId::TripletStdp,
        3 => RuleId::RewardStdp,
        4 => RuleId::InhibitoryStdp,
        5 => RuleId::Clopath,
        6 => RuleId::Bcm,
        7 => RuleId::Eprop,
        8 => RuleId::Custom(d.str()?),
        v => {
            return Err(StructuralError::param(
                "plasticity pipeline",
                format!("unknown rule id {v}"),
            ));
        }
    })
}

fn enc_rule_entry<T: Scalar>(e: &mut Enc, entry: &RuleEntry<T>) {
    enc_rule_id::<T>(e, &entry.id);
    e.f64(entry.weight.to_f64());
}

fn dec_rule_entry<T: Scalar>(d: &mut Dec) -> StructuralResult<RuleEntry<T>> {
    Ok(RuleEntry {
        id: dec_rule_id::<T>(d)?,
        weight: T::from_f64(d.f64()?),
    })
}

fn enc_rule_config<T: Scalar>(e: &mut Enc, cfg: &RuleConfig<T>) {
    match cfg {
        RuleConfig::PairStdp {
            a_plus,
            a_minus,
            trace_decay,
            bound,
        } => {
            e.u8(1);
            e.f64(a_plus.to_f64());
            e.f64(a_minus.to_f64());
            e.f64(trace_decay.to_f64());
            enc_bound(e, bound);
        }
        RuleConfig::TripletStdp {
            a2_plus,
            a2_minus,
            a3_plus,
            a3_minus,
            trace_decay,
            bound,
        } => {
            e.u8(2);
            e.f64(a2_plus.to_f64());
            e.f64(a2_minus.to_f64());
            e.f64(a3_plus.to_f64());
            e.f64(a3_minus.to_f64());
            e.f64(trace_decay.to_f64());
            enc_bound(e, bound);
        }
        RuleConfig::RewardStdp {
            a_plus,
            a_minus,
            trace_decay,
            elig_decay,
            reward_lr,
            bound,
        } => {
            e.u8(3);
            e.f64(a_plus.to_f64());
            e.f64(a_minus.to_f64());
            e.f64(trace_decay.to_f64());
            e.f64(elig_decay.to_f64());
            e.f64(reward_lr.to_f64());
            enc_bound(e, bound);
        }
        RuleConfig::InhibitoryStdp {
            eta,
            target_rate,
            trace_decay,
        } => {
            e.u8(4);
            e.f64(eta.to_f64());
            e.f64(target_rate.to_f64());
            e.f64(trace_decay.to_f64());
        }
        RuleConfig::Clopath {
            a_ltp,
            a_dep,
            theta_ltp,
            theta_dep,
            tau_v,
        } => {
            e.u8(5);
            e.f64(a_ltp.to_f64());
            e.f64(a_dep.to_f64());
            e.f64(theta_ltp.to_f64());
            e.f64(theta_dep.to_f64());
            e.f64(tau_v.to_f64());
        }
        RuleConfig::Bcm { eta, tau_mean } => {
            e.u8(6);
            e.f64(eta.to_f64());
            e.f64(tau_mean.to_f64());
        }
    }
}

fn dec_rule_config<T: Scalar>(d: &mut Dec) -> StructuralResult<RuleConfig<T>> {
    Ok(match d.u8()? {
        1 => {
            let (a_plus, a_minus, trace_decay) = (d.f64()?, d.f64()?, d.f64()?);
            RuleConfig::PairStdp {
                a_plus: T::from_f64(a_plus),
                a_minus: T::from_f64(a_minus),
                trace_decay: T::from_f64(trace_decay),
                bound: dec_bound(d)?,
            }
        }
        2 => {
            let (a2_plus, a2_minus, a3_plus, a3_minus, trace_decay) =
                (d.f64()?, d.f64()?, d.f64()?, d.f64()?, d.f64()?);
            RuleConfig::TripletStdp {
                a2_plus: T::from_f64(a2_plus),
                a2_minus: T::from_f64(a2_minus),
                a3_plus: T::from_f64(a3_plus),
                a3_minus: T::from_f64(a3_minus),
                trace_decay: T::from_f64(trace_decay),
                bound: dec_bound(d)?,
            }
        }
        3 => {
            let (a_plus, a_minus, trace_decay, elig_decay, reward_lr) =
                (d.f64()?, d.f64()?, d.f64()?, d.f64()?, d.f64()?);
            RuleConfig::RewardStdp {
                a_plus: T::from_f64(a_plus),
                a_minus: T::from_f64(a_minus),
                trace_decay: T::from_f64(trace_decay),
                elig_decay: T::from_f64(elig_decay),
                reward_lr: T::from_f64(reward_lr),
                bound: dec_bound(d)?,
            }
        }
        4 => {
            let (eta, target_rate, trace_decay) = (d.f64()?, d.f64()?, d.f64()?);
            RuleConfig::InhibitoryStdp {
                eta: T::from_f64(eta),
                target_rate: T::from_f64(target_rate),
                trace_decay: T::from_f64(trace_decay),
            }
        }
        5 => {
            let (a_ltp, a_dep, theta_ltp, theta_dep, tau_v) =
                (d.f64()?, d.f64()?, d.f64()?, d.f64()?, d.f64()?);
            RuleConfig::Clopath {
                a_ltp: T::from_f64(a_ltp),
                a_dep: T::from_f64(a_dep),
                theta_ltp: T::from_f64(theta_ltp),
                theta_dep: T::from_f64(theta_dep),
                tau_v: T::from_f64(tau_v),
            }
        }
        6 => {
            let (eta, tau_mean) = (d.f64()?, d.f64()?);
            RuleConfig::Bcm {
                eta: T::from_f64(eta),
                tau_mean: T::from_f64(tau_mean),
            }
        }
        v => {
            return Err(StructuralError::param(
                "plasticity pipeline",
                format!("unknown rule config tag {v}"),
            ));
        }
    })
}

fn enc_bound<T: Scalar>(e: &mut Enc, bound: &WeightBound<T>) {
    match *bound {
        WeightBound::Hard { w_min, w_max } => {
            e.u8(0);
            e.f64(w_min.to_f64());
            e.f64(w_max.to_f64());
        }
        WeightBound::Soft { w_min, w_max } => {
            e.u8(1);
            e.f64(w_min.to_f64());
            e.f64(w_max.to_f64());
        }
        WeightBound::None => e.u8(2),
    }
}

fn dec_bound<T: Scalar>(d: &mut Dec) -> StructuralResult<WeightBound<T>> {
    Ok(match d.u8()? {
        0 => WeightBound::Hard {
            w_min: T::from_f64(d.f64()?),
            w_max: T::from_f64(d.f64()?),
        },
        1 => WeightBound::Soft {
            w_min: T::from_f64(d.f64()?),
            w_max: T::from_f64(d.f64()?),
        },
        _ => WeightBound::None,
    })
}

fn enc_rule_class<T: Scalar>(e: &mut Enc, c: &RuleClass<T>) {
    match c {
        RuleClass::ByPolarity { src, dst } => {
            e.u8(0);
            e.u8(src.is_some() as u8);
            if let Some(s) = src {
                e.u8(polarity_code(*s));
            }
            e.u8(dst.is_some() as u8);
            if let Some(s) = dst {
                e.u8(polarity_code(*s));
            }
        }
        RuleClass::BySrcRole(_) => e.u8(1),
        RuleClass::ByDstRole(_) => e.u8(2),
        RuleClass::BySrcNeuron(id) => {
            e.u8(3);
            e.u64(id.0);
        }
        RuleClass::ByDstNeuron(id) => {
            e.u8(4);
            e.u64(id.0);
        }
        RuleClass::ByWeightRange { min, max } => {
            e.u8(5);
            e.f64(min.to_f64());
            e.f64(max.to_f64());
        }
        RuleClass::ByAge { min_steps } => {
            e.u8(6);
            e.u64(*min_steps);
        }
    }
}

fn dec_rule_class<T: Scalar>(d: &mut Dec) -> StructuralResult<RuleClass<T>> {
    Ok(match d.u8()? {
        0 => {
            let src = if d.u8()? != 0 {
                Some(polarity_from_code(d.u8()?))
            } else {
                None
            };
            let dst = if d.u8()? != 0 {
                Some(polarity_from_code(d.u8()?))
            } else {
                None
            };
            RuleClass::ByPolarity { src, dst }
        }
        1 => RuleClass::BySrcRole(role_from_code(d.u8()?)),
        2 => RuleClass::ByDstRole(role_from_code(d.u8()?)),
        3 => RuleClass::BySrcNeuron(NeuronId(d.u64()?)),
        4 => RuleClass::ByDstNeuron(NeuronId(d.u64()?)),
        5 => RuleClass::ByWeightRange {
            min: T::from_f64(d.f64()?),
            max: T::from_f64(d.f64()?),
        },
        6 => RuleClass::ByAge {
            min_steps: d.u64()?,
        },
        v => {
            return Err(StructuralError::param(
                "plasticity pipeline",
                format!("unknown rule class tag {v}"),
            ));
        }
    })
}

fn polarity_code(p: NeuronPolarity) -> u8 {
    match p {
        NeuronPolarity::Excitatory => 0,
        NeuronPolarity::Inhibitory => 1,
        NeuronPolarity::Modulatory => 2,
    }
}

fn polarity_from_code(v: u8) -> NeuronPolarity {
    match v {
        1 => NeuronPolarity::Inhibitory,
        2 => NeuronPolarity::Modulatory,
        _ => NeuronPolarity::Excitatory,
    }
}

fn role_from_code(v: u8) -> NeuronRole {
    match v {
        1 => NeuronRole::Hidden,
        2 => NeuronRole::Output,
        _ => NeuronRole::Input,
    }
}

fn enc_rule_state<T: Scalar>(e: &mut Enc, st: &RuleState<T>) {
    if let Some(v) = &st.pair {
        e.u8(1);
        e.f64(v.pre_trace.to_f64());
        e.f64(v.post_trace.to_f64());
    } else {
        e.u8(0);
    }
    if let Some(v) = &st.triplet {
        e.u8(1);
        e.f64(v.pre_fast.to_f64());
        e.f64(v.pre_slow.to_f64());
        e.f64(v.post_fast.to_f64());
        e.f64(v.post_slow.to_f64());
    } else {
        e.u8(0);
    }
    if let Some(v) = &st.reward {
        e.u8(1);
        e.f64(v.elig.to_f64());
        e.f64(v.pre_trace.to_f64());
        e.f64(v.post_trace.to_f64());
    } else {
        e.u8(0);
    }
    if let Some(v) = &st.inhibitory {
        e.u8(1);
        e.f64(v.pre_trace.to_f64());
        e.f64(v.post_trace.to_f64());
    } else {
        e.u8(0);
    }
    if let Some(v) = &st.clopath {
        e.u8(1);
        e.f64(v.v_pre.to_f64());
        e.f64(v.v_post.to_f64());
    } else {
        e.u8(0);
    }
    if let Some(v) = &st.bcm {
        e.u8(1);
        e.f64(v.y_th.to_f64());
    } else {
        e.u8(0);
    }
}

fn dec_rule_state<T: Scalar>(d: &mut Dec) -> StructuralResult<RuleState<T>> {
    let mut st = RuleState::default();
    if d.u8()? != 0 {
        st.pair = Some(PairState {
            pre_trace: T::from_f64(d.f64()?),
            post_trace: T::from_f64(d.f64()?),
        });
    }
    if d.u8()? != 0 {
        st.triplet = Some(TripletState {
            pre_fast: T::from_f64(d.f64()?),
            pre_slow: T::from_f64(d.f64()?),
            post_fast: T::from_f64(d.f64()?),
            post_slow: T::from_f64(d.f64()?),
        });
    }
    if d.u8()? != 0 {
        st.reward = Some(RewardState {
            elig: T::from_f64(d.f64()?),
            pre_trace: T::from_f64(d.f64()?),
            post_trace: T::from_f64(d.f64()?),
        });
    }
    if d.u8()? != 0 {
        st.inhibitory = Some(InhibitoryState {
            pre_trace: T::from_f64(d.f64()?),
            post_trace: T::from_f64(d.f64()?),
        });
    }
    if d.u8()? != 0 {
        st.clopath = Some(ClopathState {
            v_pre: T::from_f64(d.f64()?),
            v_post: T::from_f64(d.f64()?),
        });
    }
    if d.u8()? != 0 {
        st.bcm = Some(BcmState {
            y_th: T::from_f64(d.f64()?),
        });
    }
    Ok(st)
}

impl<T: Scalar> PlasticityPipeline<T> {
    /// Serialize the config, per-synapse rule states, and counters (magic `"PLP1"`).
    pub fn to_bytes(&self) -> StructuralResult<Vec<u8>> {
        let mut e = Enc::default();
        e.buf.extend_from_slice(MAGIC);
        e.u32(VERSION);
        // default rules
        e.u64(self.config.default_rules.len() as u64);
        for r in &self.config.default_rules {
            enc_rule_entry::<T>(&mut e, r);
        }
        // synapse rules
        e.u64(self.config.synapse_rules.len() as u64);
        for (id, v) in &self.config.synapse_rules {
            e.u64(id.0);
            e.u64(v.len() as u64);
            for r in v {
                enc_rule_entry::<T>(&mut e, r);
            }
        }
        // class rules
        e.u64(self.config.class_rules.len() as u64);
        for (c, v) in &self.config.class_rules {
            enc_rule_class::<T>(&mut e, c);
            e.u64(v.len() as u64);
            for r in v {
                enc_rule_entry::<T>(&mut e, r);
            }
        }
        // rule configs (built-ins only; custom rules re-supplied at load)
        let cfgs: Vec<(&RuleId, &RuleConfig<T>)> = self
            .config
            .rule_configs
            .iter()
            .filter(|(id, _)| !matches!(id, RuleId::Custom(_)))
            .collect();
        e.u64(cfgs.len() as u64);
        for (id, cfg) in cfgs {
            enc_rule_id::<T>(&mut e, id);
            enc_rule_config::<T>(&mut e, cfg);
        }
        e.f64(self.config.modulators.decay.to_f64());
        e.opt_f64(self.config.max_abs_delta);
        // per-synapse states
        e.u64(self.states.len() as u64);
        for (id, st) in &self.states {
            e.u64(id.0);
            enc_rule_state::<T>(&mut e, st);
        }
        // e-prop state (if any)
        let eprop_state: Option<EpropState<T>> = match (&self.eprop, self.eprop_state.clone()) {
            (Some(t), _) => Some(t.state().clone()),
            (None, s) => s,
        };
        e.u8(if eprop_state.is_some() { 1 } else { 0 });
        if let Some(st) = eprop_state {
            let b = st.to_bytes()?;
            e.u64(b.len() as u64);
            e.buf.extend_from_slice(&b);
        }
        // counters
        e.u64(self.steps);
        e.opt_f64(self.reward);
        // modulator events
        e.u64(self.modulator_events.len() as u64);
        for ev in &self.modulator_events {
            enc_channel(&mut e, &ev.event.channel);
            e.f64(ev.event.value.to_f64());
            e.u64(ev.event.at_step);
            match ev.event.target {
                ModulatorTarget::Synapse(id) => {
                    e.u8(0);
                    e.u64(id.0);
                }
                ModulatorTarget::Neuron(id) => {
                    e.u8(1);
                    e.u64(id.0);
                }
            }
            e.opt_u64(ev.until_step);
        }
        Ok(e.buf)
    }

    /// Restore a pipeline serialized with [`to_bytes`](Self::to_bytes).
    ///
    /// Custom rules (`PlasticityPipelineConfig::custom_rules`) and the e-prop trainer config
    /// are not serialized — re-supply them in the config (the restored pipeline carries
    /// `eprop_state` for the next [`DynamicNetwork::set_plasticity_pipeline`]).
    pub fn from_bytes(bytes: &[u8]) -> StructuralResult<Self> {
        let mut d = Dec { b: bytes, p: 0 };
        let magic: [u8; 4] = d.take(4)?.try_into().map_err(|_| {
            StructuralError::param("plasticity pipeline", "bad magic (not a PLP1 blob)")
        })?;
        if &magic != MAGIC {
            return Err(StructuralError::param(
                "plasticity pipeline",
                "not a PLP1 blob",
            ));
        }
        let version = d.u32()?;
        if version != VERSION {
            return Err(StructuralError::param(
                "plasticity pipeline",
                format!("unsupported version {version} (expected {VERSION})"),
            ));
        }
        let mut pl = Self::new(PlasticityPipelineConfig::default());
        let n = d.u64()?;
        pl.config.default_rules = (0..n)
            .map(|_| dec_rule_entry::<T>(&mut d))
            .collect::<StructuralResult<_>>()?;
        let n = d.u64()?;
        for _ in 0..n {
            let id = SynapseId(d.u64()?);
            let m = d.u64()?;
            let mut v = Vec::with_capacity(m as usize);
            for _ in 0..m {
                v.push(dec_rule_entry::<T>(&mut d)?);
            }
            pl.config.synapse_rules.insert(id, v);
        }
        let n = d.u64()?;
        for _ in 0..n {
            let c = dec_rule_class::<T>(&mut d)?;
            let m = d.u64()?;
            let mut v = Vec::with_capacity(m as usize);
            for _ in 0..m {
                v.push(dec_rule_entry::<T>(&mut d)?);
            }
            pl.config.class_rules.push((c, v));
        }
        let n = d.u64()?;
        for _ in 0..n {
            let id = dec_rule_id::<T>(&mut d)?;
            let cfg = dec_rule_config::<T>(&mut d)?;
            pl.config.rule_configs.insert(id, cfg);
        }
        pl.config.modulators.decay = T::from_f64(d.f64()?);
        pl.config.max_abs_delta = d.opt_f64::<T>()?;
        let n = d.u64()?;
        for _ in 0..n {
            let id = SynapseId(d.u64()?);
            let st = dec_rule_state::<T>(&mut d)?;
            pl.states.insert(id, st);
        }
        if d.u8()? != 0 {
            let n = d.u64()?;
            let b = d.take(n as usize)?;
            pl.eprop_state = Some(EpropState::from_bytes(b)?);
        }
        pl.steps = d.u64()?;
        pl.reward = d.opt_f64::<T>()?;
        let n = d.u64()?;
        for _ in 0..n {
            let channel = dec_channel(&mut d)?;
            let value = T::from_f64(d.f64()?);
            let at_step = d.u64()?;
            let target = match d.u8()? {
                0 => ModulatorTarget::Synapse(SynapseId(d.u64()?)),
                _ => ModulatorTarget::Neuron(NeuronId(d.u64()?)),
            };
            let until_step = d.opt_u64()?;
            pl.modulator_events.push_back(ModulatedEvent {
                event: ModulatorEvent {
                    channel,
                    value,
                    at_step,
                    target,
                    lifetime: None,
                },
                until_step,
            });
        }
        Ok(pl)
    }
}

fn enc_channel(e: &mut Enc, c: &ModulatorChannel) {
    match c {
        ModulatorChannel::Dopamine => e.u8(0),
        ModulatorChannel::Serotonin => e.u8(1),
        ModulatorChannel::Acetylcholine => e.u8(2),
        ModulatorChannel::Noradrenaline => e.u8(3),
        ModulatorChannel::Novelty => e.u8(4),
        ModulatorChannel::PredictionError => e.u8(5),
        ModulatorChannel::Custom(s) => {
            e.u8(6);
            e.str(s);
        }
    }
}

fn dec_channel(d: &mut Dec) -> StructuralResult<ModulatorChannel> {
    Ok(match d.u8()? {
        0 => ModulatorChannel::Dopamine,
        1 => ModulatorChannel::Serotonin,
        2 => ModulatorChannel::Acetylcholine,
        3 => ModulatorChannel::Noradrenaline,
        4 => ModulatorChannel::Novelty,
        5 => ModulatorChannel::PredictionError,
        _ => ModulatorChannel::Custom(d.str()?),
    })
}

// ---------------------------------------------------------------- tests

#[cfg(test)]
mod tests {
    use super::*;
    use crate::structural::{
        Birth, EngineConfig, LifParams, NeuronKind, Origin, StructuralConfig, Topology,
    };

    fn mk_net() -> DynamicNetwork {
        let mut topo = Topology::new(
            StructuralConfig::<f32>::default().topology,
            StructuralConfig::<f32>::default().params,
        );
        let b = Birth::default();
        let i = topo
            .add_neuron(NeuronKind::Input, LifParams::default(), Origin::Initial, b)
            .unwrap();
        let o = topo
            .add_neuron(
                NeuronKind::Output,
                LifParams::new(0.9, 0.4),
                Origin::Initial,
                b,
            )
            .unwrap();
        let s = topo.add_synapse(i, o, 0.5, 0, Origin::Initial, b).unwrap();
        assert_eq!(i, NeuronId(0));
        assert_eq!(o, NeuronId(1));
        assert_eq!(s, SynapseId(0));
        DynamicNetwork::new(topo, EngineConfig::default()).unwrap()
    }

    fn pair_pipeline() -> PlasticityPipeline {
        let mut pl = PlasticityPipeline::new(PlasticityPipelineConfig::default());
        pl.config.default_rules = vec![RuleEntry::new(RuleId::PairStdp)];
        pl
    }

    #[test]
    fn engine_immediate_matches_pipeline_pair() {
        // Engine: pair-STDP Immediate with the same constants as `RuleConfig::default()`.
        let (topo_e, topo_p) = {
            let mk = || {
                let mut topo = Topology::new(
                    StructuralConfig::<f32>::default().topology,
                    StructuralConfig::<f32>::default().params,
                );
                let b = Birth::default();
                let i = topo
                    .add_neuron(NeuronKind::Input, LifParams::default(), Origin::Initial, b)
                    .unwrap();
                let o = topo
                    .add_neuron(
                        NeuronKind::Output,
                        LifParams::new(0.9, 0.4),
                        Origin::Initial,
                        b,
                    )
                    .unwrap();
                topo.add_synapse(i, o, 0.5, 0, Origin::Initial, b).unwrap();
                topo
            };
            (mk(), mk())
        };
        let mut eng = {
            let mut c = EngineConfig::default();
            c.stdp.mode = super::super::network::PlasticityMode::Immediate;
            DynamicNetwork::new(topo_e, c).unwrap()
        };
        let mut pip = DynamicNetwork::new(topo_p, EngineConfig::default()).unwrap();
        pip.set_plasticity_pipeline(Some(pair_pipeline())).unwrap();
        for _ in 0..20 {
            eng.step(&[1.0]).unwrap();
            pip.step(&[1.0]).unwrap();
        }
        let we = eng.topology().synapses().next().unwrap().weight;
        let wp = pip.topology().synapses().next().unwrap().weight;
        assert!((we - wp).abs() < 1e-6, "engine {we} vs pipeline {wp}");
    }

    #[test]
    fn mixed_assignment_hybrid() {
        let mut topo = Topology::new(
            StructuralConfig::<f32>::default().topology,
            StructuralConfig::<f32>::default().params,
        );
        let b = Birth::default();
        let i = topo
            .add_neuron(NeuronKind::Input, LifParams::default(), Origin::Initial, b)
            .unwrap();
        let h = topo
            .add_neuron(
                NeuronKind::Hidden,
                LifParams::new(0.9, 0.4),
                Origin::Initial,
                b,
            )
            .unwrap();
        let o = topo
            .add_neuron(
                NeuronKind::Output,
                LifParams::new(0.9, 1.0),
                Origin::Initial,
                b,
            )
            .unwrap();
        let s_ih = topo.add_synapse(i, h, 0.5, 0, Origin::Initial, b).unwrap();
        let s_ho = topo.add_synapse(h, o, 0.5, 0, Origin::Initial, b).unwrap();
        let s_io = topo.add_synapse(i, o, 0.5, 0, Origin::Initial, b).unwrap();
        let mut net = DynamicNetwork::new(topo, EngineConfig::default()).unwrap();
        let mut pl = pair_pipeline();
        // Class rule: BCM on synapses whose source is the input neuron.
        pl.config
            .class_rules
            .push((RuleClass::BySrcNeuron(i), vec![RuleEntry::new(RuleId::Bcm)]));
        pl.config.rule_configs.insert(
            RuleId::Bcm,
            RuleConfig::Bcm {
                eta: 0.05,
                tau_mean: 50.0,
            },
        );
        // Explicit off for the input→hidden synapse (specific > class).
        pl.config.synapse_rules.insert(s_ih, Vec::new());
        net.set_plasticity_pipeline(Some(pl)).unwrap();
        for _ in 0..5 {
            net.step(&[1.0]).unwrap();
        }
        let w_ih = net.topology().synapse(s_ih).unwrap().weight;
        assert_eq!(w_ih, 0.5, "explicit off assignment must stay");
        // BCM on input→output (class matches, default pair is shadowed by class).
        let w_io = net.topology().synapse(s_io).unwrap().weight;
        assert!(w_io > 0.5, "BCM should grow w_io: {w_io}");
        // Pair STDP on h→o (default): h spikes → o arrives later → weight changes.
        let w_ho = net.topology().synapse(s_ho).unwrap().weight;
        assert_ne!(w_ho, 0.5, "pair STDP should change w_ho: {w_ho}");
        net.topology().validate().unwrap();
    }

    #[test]
    fn multiple_rules_predictable_order_no_double() {
        let mut net_a = mk_net();
        let mut pl_a = pair_pipeline();
        pl_a.config.rule_configs.insert(
            RuleId::PairStdp,
            RuleConfig::PairStdp {
                a_plus: 0.01,
                a_minus: 0.0,
                trace_decay: 0.5,
                bound: WeightBound::None,
            },
        );
        net_a.set_plasticity_pipeline(Some(pl_a)).unwrap();
        let mut net_b = mk_net();
        let mut pl = pair_pipeline();
        pl.config.rule_configs.insert(
            RuleId::PairStdp,
            RuleConfig::PairStdp {
                a_plus: 0.01,
                a_minus: 0.0,
                trace_decay: 0.5,
                bound: WeightBound::None,
            },
        );
        pl.config.default_rules = vec![
            RuleEntry::new(RuleId::PairStdp),
            RuleEntry::new(RuleId::PairStdp),
        ];
        net_b.set_plasticity_pipeline(Some(pl)).unwrap();
        for _ in 0..3 {
            net_a.step(&[1.0]).unwrap();
            net_b.step(&[1.0]).unwrap();
        }
        let wa = net_a.topology().synapses().next().unwrap().weight;
        let wb = net_b.topology().synapses().next().unwrap().weight;
        assert!(wb > wa, "two rules must compose (sum), got {wa} vs {wb}");
        // And with a None rule in the list, the weight must equal the single-rule case.
        let mut net_c = mk_net();
        let mut pl = pair_pipeline();
        pl.config.rule_configs.insert(
            RuleId::PairStdp,
            RuleConfig::PairStdp {
                a_plus: 0.01,
                a_minus: 0.0,
                trace_decay: 0.5,
                bound: WeightBound::None,
            },
        );
        pl.config.default_rules = vec![
            RuleEntry::new(RuleId::PairStdp),
            RuleEntry::new(RuleId::None),
        ];
        net_c.set_plasticity_pipeline(Some(pl)).unwrap();
        for _ in 0..3 {
            net_c.step(&[1.0]).unwrap();
        }
        let wc = net_c.topology().synapses().next().unwrap().weight;
        assert!(
            (wa - wc).abs() < 1e-7,
            "None rule adds nothing: {wa} vs {wc}"
        );
    }

    #[test]
    fn clopath_ltp_when_post_above_threshold() {
        let mut net = mk_net();
        let mut pl = PlasticityPipeline::new(PlasticityPipelineConfig::default());
        pl.config.default_rules = vec![RuleEntry::new(RuleId::Clopath)];
        pl.config.rule_configs.insert(
            RuleId::Clopath,
            RuleConfig::Clopath {
                a_ltp: 1.0,
                a_dep: 0.01,
                theta_ltp: 0.05,
                theta_dep: 0.5,
                tau_v: 0.95,
            },
        );
        net.set_plasticity_pipeline(Some(pl)).unwrap();
        let w0 = net.topology().synapses().next().unwrap().weight;
        // Output spikes with input; dst voltage rises above θ_LTP → LTP on arrival.
        for _ in 0..3 {
            net.step(&[1.0]).unwrap();
        }
        let w1 = net.topology().synapses().next().unwrap().weight;
        assert!(w1 > w0, "Clopath LTP expected: {w0} -> {w1}");
    }

    #[test]
    fn clopath_dep_on_pre_spike() {
        let mut net = mk_net();
        let mut pl = PlasticityPipeline::new(PlasticityPipelineConfig::default());
        pl.config.default_rules = vec![RuleEntry::new(RuleId::Clopath)];
        pl.config.rule_configs.insert(
            RuleId::Clopath,
            RuleConfig::Clopath {
                a_ltp: 0.001,
                a_dep: 0.01,
                theta_ltp: 10.0, // above any realistic voltage
                theta_dep: 0.0,  // dst voltage above 0.0 → DEP when src spikes
                tau_v: 0.95,
            },
        );
        net.set_plasticity_pipeline(Some(pl)).unwrap();
        let w0 = net.topology().synapses().next().unwrap().weight;
        for _ in 0..3 {
            net.step(&[1.0]).unwrap();
        }
        let w1 = net.topology().synapses().next().unwrap().weight;
        assert!(
            w1 < w0,
            "Clopath depression on src spike expected: {w0} -> {w1}"
        );
    }

    #[test]
    fn bcm_direction_change() {
        let mut topo = Topology::new(
            StructuralConfig::<f32>::default().topology,
            StructuralConfig::<f32>::default().params,
        );
        let b = Birth::default();
        let i = topo
            .add_neuron(NeuronKind::Input, LifParams::default(), Origin::Initial, b)
            .unwrap();
        let o1 = topo
            .add_neuron(
                NeuronKind::Output,
                LifParams::new(0.9, 1.0),
                Origin::Initial,
                b,
            )
            .unwrap();
        let o2 = topo
            .add_neuron(
                NeuronKind::Output,
                LifParams::new(0.9, 1.0),
                Origin::Initial,
                b,
            )
            .unwrap();
        let s1 = topo.add_synapse(i, o1, 0.2, 0, Origin::Initial, b).unwrap();
        let s2 = topo.add_synapse(i, o2, 1.2, 0, Origin::Initial, b).unwrap();
        let mut net = DynamicNetwork::new(topo, EngineConfig::default()).unwrap();
        let mut pl = PlasticityPipeline::new(PlasticityPipelineConfig::default());
        pl.config.default_rules = vec![RuleEntry::new(RuleId::Bcm)];
        pl.config.rule_configs.insert(
            RuleId::Bcm,
            RuleConfig::Bcm {
                eta: 0.02,
                tau_mean: 1.0,
            },
        );
        net.set_plasticity_pipeline(Some(pl)).unwrap();
        for _ in 0..10 {
            net.step(&[1.0]).unwrap();
        }
        let w1 = net.topology().synapse(s1).unwrap().weight;
        let w2 = net.topology().synapse(s2).unwrap().weight;
        assert!(w1 > 0.2, "BCM LTP region: {w1}");
        assert!(w2 < 1.2, "BCM LTD region: {w2}");
    }

    #[test]
    fn modulator_scales_rules_locally() {
        let mut net_a = mk_net();
        let mut pl_a = pair_pipeline();
        pl_a.config.rule_configs.insert(
            RuleId::PairStdp,
            RuleConfig::PairStdp {
                a_plus: 0.1,
                a_minus: 0.0,
                trace_decay: 0.5,
                bound: WeightBound::None,
            },
        );
        net_a.set_plasticity_pipeline(Some(pl_a)).unwrap();
        let mut net_b = mk_net();
        let mut pl_b = pair_pipeline();
        pl_b.config.rule_configs.insert(
            RuleId::PairStdp,
            RuleConfig::PairStdp {
                a_plus: 0.1,
                a_minus: 0.0,
                trace_decay: 0.5,
                bound: WeightBound::None,
            },
        );
        net_b.set_plasticity_pipeline(Some(pl_b)).unwrap();
        let at = net_b.step_count();
        net_b
            .plasticity_pipeline_mut()
            .unwrap()
            .apply_modulator(ModulatorEvent {
                channel: ModulatorChannel::Dopamine,
                value: 2.0,
                at_step: at,
                target: ModulatorTarget::Synapse(SynapseId(0)),
                lifetime: Some(100),
            });
        for _ in 0..3 {
            net_a.step(&[1.0]).unwrap();
            net_b.step(&[1.0]).unwrap();
        }
        let wa = net_a.topology().synapses().next().unwrap().weight;
        let wb = net_b.topology().synapses().next().unwrap().weight;
        assert!(wb > wa, "modulated synapse must change more: {wa} vs {wb}");
    }

    #[test]
    fn pipeline_checkpoint_roundtrip() {
        let mut net = mk_net();
        net.set_plasticity_pipeline(Some(pair_pipeline())).unwrap();
        for _ in 0..2 {
            net.step(&[1.0]).unwrap();
        }
        let bytes = net.plasticity_pipeline().unwrap().to_bytes().unwrap();
        let pl2 = PlasticityPipeline::<f32>::from_bytes(&bytes).unwrap();
        assert_eq!(pl2.steps(), 2);
        // Restore into a fresh network and continue (arrival happens on step 2+).
        let mut net2 = mk_net();
        net2.set_plasticity_pipeline(Some(pl2)).unwrap();
        for _ in 0..3 {
            net2.step(&[1.0]).unwrap();
        }
        let w = net2.topology().synapses().next().unwrap().weight;
        assert_ne!(w, 0.5);
    }

    #[test]
    fn custom_rule_registration_and_errors() {
        #[derive(Debug, Clone)]
        struct MyRule;
        impl PlasticityRule<f32> for MyRule {
            fn id(&self) -> String {
                "my_rule".into()
            }
            fn update(&self, synapse: SynapseId, ctx: &PipelineCtx<f32>) -> StructuralResult<f32> {
                Ok(if ctx.arrival && ctx.synapse.id == synapse {
                    0.001
                } else {
                    0.0
                })
            }
        }
        let mut pl = PlasticityPipeline::new(PlasticityPipelineConfig::default());
        pl.config
            .custom_rules
            .insert("my_rule".into(), Arc::new(MyRule));
        pl.config.default_rules = vec![RuleEntry::new(RuleId::Custom("my_rule".into()))];
        let mut net = mk_net();
        net.set_plasticity_pipeline(Some(pl)).unwrap();
        let w0 = net.topology().synapses().next().unwrap().weight;
        net.step(&[1.0]).unwrap();
        net.step(&[1.0]).unwrap(); // first arrival on the 2nd step
        let w1 = net.topology().synapses().next().unwrap().weight;
        assert!(
            (w1 - (w0 + 0.001)).abs() < 1e-6,
            "custom rule must apply: {w0} -> {w1}"
        );
        // Unknown custom rule id is rejected.
        let mut pl2 = PlasticityPipeline::new(PlasticityPipelineConfig::default());
        pl2.config.default_rules = vec![RuleEntry::new(RuleId::Custom("nope".into()))];
        let mut net2 = mk_net();
        assert!(net2.set_plasticity_pipeline(Some(pl2)).is_err());
    }

    #[test]
    fn hybrid_network_example() {
        let mut topo = Topology::new(
            StructuralConfig::<f32>::default().topology,
            StructuralConfig::<f32>::default().params,
        );
        let b = Birth::default();
        // Hybrid network: Input → H(inhibitory) → Output.
        let i = topo
            .add_neuron(
                NeuronKind::Input,
                LifParams::new(0.9, 0.4),
                Origin::Initial,
                b,
            )
            .unwrap();
        let h = topo
            .add_neuron_with(
                NeuronKind::Hidden,
                NeuronPolarity::Inhibitory,
                LifParams::new(0.9, 1.0),
                Origin::Initial,
                b,
            )
            .unwrap();
        let o = topo
            .add_neuron(
                NeuronKind::Output,
                LifParams::new(0.9, 1.0),
                Origin::Initial,
                b,
            )
            .unwrap();
        topo.add_synapse(i, h, 0.5, 0, Origin::Initial, b).unwrap();
        topo.add_synapse(h, o, -0.5, 0, Origin::Initial, b).unwrap();
        let mut net = DynamicNetwork::new(topo, EngineConfig::default()).unwrap();
        let mut pl = pair_pipeline();
        pl.config.class_rules.push((
            RuleClass::ByPolarity {
                src: Some(NeuronPolarity::Inhibitory),
                dst: None,
            },
            vec![RuleEntry::new(RuleId::InhibitoryStdp)],
        ));
        net.set_plasticity_pipeline(Some(pl)).unwrap();
        for _ in 0..5 {
            net.step(&[1.0]).unwrap();
        }
        net.topology().validate().unwrap();
        // Dale invariant: inhibitory source has w ≤ 0.
        for s in net.topology().synapses() {
            let p = net.topology().neuron(s.src).unwrap().polarity;
            if p == NeuronPolarity::Inhibitory {
                assert!(s.weight <= 0.0, "Dale: {s:?}");
            }
        }
    }

    #[test]
    fn e_prop_typed_gate() {
        use crate::structural::{IzhikevichParams, NeuronModel};
        let mut topo = Topology::new(
            StructuralConfig::<f32>::default().topology,
            StructuralConfig::<f32>::default().params,
        );
        let b = Birth::default();
        let i = topo
            .add_neuron(NeuronKind::Input, LifParams::default(), Origin::Initial, b)
            .unwrap();
        let o = topo
            .add_neuron(
                NeuronKind::Output,
                NeuronModel::Izhikevich(IzhikevichParams::regular_spiking()),
                Origin::Initial,
                b,
            )
            .unwrap();
        topo.add_synapse(i, o, 0.5, 0, Origin::Initial, b).unwrap();
        let mut net = DynamicNetwork::new(topo, EngineConfig::default()).unwrap();
        let mut pl = PlasticityPipeline::new(PlasticityPipelineConfig::default());
        pl.config.eprop_config = Some(EpropConfig::default());
        pl.config.default_rules = vec![RuleEntry::new(RuleId::Eprop)];
        let err = net.set_plasticity_pipeline(Some(pl)).unwrap_err();
        assert!(matches!(err, StructuralError::UnsupportedEpropDynamics(_)));
    }

    #[test]
    fn e_prop_path_works() {
        let mut net = mk_net();
        let mut pl = PlasticityPipeline::new(PlasticityPipelineConfig::default());
        pl.config.eprop_config = Some(EpropConfig::default());
        pl.config.default_rules = vec![RuleEntry::new(RuleId::Eprop)];
        net.set_plasticity_pipeline(Some(pl)).unwrap();
        for _ in 0..3 {
            net.step(&[1.0]).unwrap();
        }
        let tr = net.plasticity_pipeline().unwrap().eprop_trainer().unwrap();
        assert!(tr.state().steps() > 0);
        assert!(tr.state().synapse_trace_count() > 0);
    }

    #[test]
    fn pipeline_respects_dale_and_bounds() {
        let mut net = mk_net();
        let mut pl = pair_pipeline();
        pl.config.max_abs_delta = Some(10.0);
        net.set_plasticity_pipeline(Some(pl)).unwrap();
        for _ in 0..10 {
            net.step(&[1.0]).unwrap();
        }
        net.topology().validate().unwrap();
        for s in net.topology().synapses() {
            assert!(s.weight.abs() <= net.topology().param_limits().weight_abs_max);
        }
    }

    #[test]
    fn reward_rule_accumulates_elig_and_applies_on_reward() {
        let mut net = mk_net();
        let mut pl = PlasticityPipeline::new(PlasticityPipelineConfig::default());
        pl.config.rule_configs.insert(
            RuleId::RewardStdp,
            RuleConfig::RewardStdp {
                a_plus: 0.05,
                a_minus: 0.0,
                trace_decay: 0.95,
                elig_decay: 0.98,
                reward_lr: 1.0,
                bound: WeightBound::None,
            },
        );
        pl.config.default_rules = vec![RuleEntry::new(RuleId::RewardStdp)];
        net.set_plasticity_pipeline(Some(pl)).unwrap();
        let sid = SynapseId(0);
        let w0 = net.topology().synapse(sid).unwrap().weight;
        for _ in 0..3 {
            net.step(&[1.0]).unwrap();
        }
        let elig = net
            .plasticity_pipeline()
            .unwrap()
            .states()
            .get(&sid)
            .unwrap()
            .reward
            .as_ref()
            .unwrap()
            .elig;
        assert!(elig > 0.0, "eligibility should accumulate on arrival");
        net.apply_reward(1.0);
        net.step(&[1.0]).unwrap();
        let w1 = net.topology().synapse(sid).unwrap().weight;
        assert!(
            w1 > w0,
            "rewarded R-STDP must increase weight: {w0} -> {w1}"
        );
    }
}
