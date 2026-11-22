//! Dynamic CPU-executor of a network with a variable topology.
//!
//! This is the "correct minimal engine" from Stage 3: Event Delivery
//! spikes at active synapses, LIF-neuron dynamics, local rules
//! plasticity and statistics for structural solutions. It doesn't replace
//! Burn-layers of fixed shape (`layers::*`, `neuron::*`): structural
//! plasticity cannot be applied to fixed tensors without reassembly
//! size, so the graph changes here, but the layers remain as they are.
//!
//! # Simulation step
//!
//! 1. **Delivery.** For each spike from history (age `a` = how many steps
//!    back) current is added along its outgoing active synapses with `delay == a`
//!    to the receiver. The cost is proportional to the number of spikes, not the size of the network.
//! 2. **Dynamics.** In [`InputMode::Spike`] input
//!    spikes based on an external signal (`≥ 0.5`); in
//!    [`InputMode::Current`] external value —
//!    this is the current, and the spike determines the model of the input neuron. The rest:
//!    `U ← β·U + I`, spike at `U ≥ θ`, reset at `ResetMode`.
//! 3. **Plasticity** (if enabled): pair-STDP using the same formulas as
//!    `functional::plasticity::pair_update`, but for events (see below).
//! 4. Spike history, statistics, joint activity accumulator.
//!
//! # Connect with `functional::plasticity`
//!
//! The formula and semantics are the same: `Δw = a₊·x_post·pre − a₋·x_pre·post`, traces
//! **before** updates are taken at this step, weight limit is the same
//! [`WeightBound`] (`Hard`/`Soft`/`None`), reward-modulated option - the same
//! three-factor: eligibility accumulates STDP, weight changes when
//! [`DynamicNetwork::apply_reward`]. There is only one difference: the calculations take place there
//! dense tensors `[out, in]`, here - by events for sparse
//! the graph has a variable shape, so tensor functions cannot be called; code
//! the rules do not duplicate, but repeat the scalar shape of the same formula.
//! A `pre` spike is considered to be a spike that **arrived** at the synapse at this step (with
//! taking into account the delay), `pre`-trace - trace of the source neuron.
//!
//! # SafeBorders
//!
//! Topology (`topology_mut`, `replace_topology`, controller actions)
//! change between samples or minibatch, not in the middle of an episode. External
//! The engine does not use autograd graphs.

use crate::scalar::Scalar;
use std::collections::{BTreeMap, BTreeSet, VecDeque};

use crate::functional::plasticity::WeightBound;

use super::custom::NeuronStepContext;
use super::error::{StructuralError, StructuralResult};
use super::model::{ModelSwitchOutcome, NeuronModel, NeuronPolarity, StateMigration};
use super::plasticity::PlasticityPipeline;
use super::stats::{CoactivityTracker, StructuralStats};
use super::topology::{NeuronId, NeuronRole, SynapseId, Topology};

/// Synaptic plasticity mode of the engine.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PlasticityMode {
    /// The weights do not change.
    Off,
    /// pair-STDP is applied to the weights immediately.
    Immediate,
    /// Reward-modulated: STDP is accumulated in the eligibility trace, the weight changes when
    /// `apply_reward`.
    RewardModulated,
}

/// Parameters pair-STDP (names and meaning are the same as `functional::plasticity`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StdpConfig<T = f32> {
    /// Mode.
    pub mode: PlasticityMode,
    /// Potentiation amplitude `A+`.
    pub a_plus: T,
    /// Depression amplitude `A−`.
    pub a_minus: T,
    /// Decay of trace `exp(-1/τ)` per step.
    pub trace_decay: T,
    /// Limiting weights.
    pub bound: WeightBound<T>,
    /// Decay of eligibility trace per step (for `RewardModulated`).
    pub elig_decay: T,
    /// Reward learning rate.
    pub reward_lr: T,
}

impl<T: Scalar> Default for StdpConfig<T> {
    fn default() -> Self {
        Self {
            mode: PlasticityMode::Off,
            a_plus: T::c(0.01),
            a_minus: T::c(0.012),
            trace_decay: (-T::ONE / T::c(20.0)).exp(),
            bound: WeightBound::Hard {
                w_min: -T::ONE,
                w_max: T::ONE,
            },
            elig_decay: (-T::ONE / T::c(50.0)).exp(),
            reward_lr: T::ONE,
        }
    }
}

/// Short-term plasticity mode (Tsodyks–Markram semantics) — see [`StpConfig`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum StpMode {
    /// No short-term plasticity (release multiplier ≡ 1 — legacy behavior).
    Static,
    /// Facilitation: utilization reversion is slow (`τ_fac` preset is large),
    /// so bursts keep `u` elevated and close events release more.
    Facilitation,
    /// Depression: resource recovery is slow (`τ_rec` preset is large),
    /// so bursts deplete `x` and close events release less.
    Depression,
    /// Mixed: moderate `τ_fac` and `τ_rec` presets.
    Mixed,
}

impl StpMode {
    /// Numeric code for checkpointing.
    pub fn code(self) -> u8 {
        match self {
            StpMode::Static => 0,
            StpMode::Facilitation => 1,
            StpMode::Depression => 2,
            StpMode::Mixed => 3,
        }
    }

    /// From numeric code.
    pub fn from_code(v: u8) -> Option<Self> {
        Some(match v {
            0 => StpMode::Static,
            1 => StpMode::Facilitation,
            2 => StpMode::Depression,
            3 => StpMode::Mixed,
            _ => return None,
        })
    }

    /// Parameter presets `(U, τ_fac, τ_rec)` for the mode
    /// (explicit `StpConfig` fields override presets).
    pub fn preset<T: Scalar>(self) -> (T, T, T) {
        match self {
            StpMode::Static => (T::c(0.1), T::c(30.0), T::c(5.0)),
            StpMode::Facilitation => (T::c(0.1), T::c(50.0), T::c(5.0)),
            StpMode::Depression => (T::c(0.5), T::c(5.0), T::c(50.0)),
            StpMode::Mixed => (T::c(0.2), T::c(10.0), T::c(10.0)),
        }
    }
}

/// Short-term synaptic plasticity (STP) config — Tsodyks–Markram 2-variable model.
///
/// Documented version (see module docs of `structural::network` for the algorithm):
/// per-synapse state `x ∈ [0,1]` (available resources, starts 1) and
/// `u ∈ [0,1]` (utilization, starts `u`). On arrival of a spike at the synapse
/// at step `t`:
///
/// ```text
/// 1. Recovery between events:  x ← 1 + (x − 1)·e^(−Δt/τ_rec),  u ← U + (u − U)·e^(−Δt/τ_fac)
/// 2. Release:                  y = u·x
/// 3. Depletion:                x ← x − y
/// 4. Facilitation on event:   u ← min(u + (1 − U)·(1 − e^(−1/τ_fac)), 1)
/// 5. Delivered amplitude:      current += e·y  (e = weight × sign)
/// ```
///
/// The delivery multiplies the transmitted signal by the release fraction `y`;
/// the long-term `weight` is untouched (short-term vs long-term are separate
/// data — `StpState` vs `Synapse.weight`). STP is disabled by default
/// (`EngineConfig.stp = None`) and must not be enabled together with the
/// engine's own long-term STDP paths — `DynamicNetwork::stp_config`.
///
/// [`StpMode`] presets: `Facilitation` — bursts raise `u` (slow `τ_fac` revert);
/// `Depression` — bursts deplete `x` (slow `τ_rec` recovery); `Mixed` — both
/// moderate; `Static` — off.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StpConfig<T = f32> {
    /// Semantic mode (semantic label; explicit params override the preset).
    pub mode: StpMode,
    /// Base utilization `U` ∈ (0, 1].
    pub u: T,
    /// Utilization time constant (facilitation revert time).
    pub tau_fac: T,
    /// Resource recovery time constant.
    pub tau_rec: T,
}

impl<T: Scalar> Default for StpConfig<T> {
    fn default() -> Self {
        StpConfig::preset(StpMode::Static)
    }
}

impl<T: Scalar> StpConfig<T> {
    /// Build from mode + explicit params (override the preset of `mode`).
    pub fn new(mode: StpMode, u: T, tau_fac: T, tau_rec: T) -> Self {
        Self {
            mode,
            u,
            tau_fac,
            tau_rec,
        }
    }

    /// Mode preset constructor: `StpConfig::preset(StpMode::Mixed)`.
    pub fn preset(mode: StpMode) -> Self {
        let (u, tau_fac, tau_rec) = mode.preset::<T>();
        Self {
            mode,
            u,
            tau_fac,
            tau_rec,
        }
    }

    /// Validate the parameters.
    fn validate(&self) -> StructuralResult<()> {
        if !self.u.is_finite() || self.u <= T::ZERO || self.u > T::ONE {
            return Err(StructuralError::InvalidConfig(
                "StpConfig: u must be finite and in (0, 1]".into(),
            ));
        }
        if !self.tau_fac.is_finite()
            || self.tau_fac <= T::ZERO
            || !self.tau_rec.is_finite()
            || self.tau_rec <= T::ZERO
        {
            return Err(StructuralError::InvalidConfig(
                "StpConfig: tau_fac/tau_rec must be finite and > 0".into(),
            ));
        }
        Ok(())
    }
}

/// Per-synapse STP state (Tsodyks–Markram `x`/`u` + telemetry of the last release).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StpState<T = f32> {
    /// Available resources `x ∈ [0, 1]` (start 1).
    pub x: T,
    /// Utilization `u ∈ [0, 1]` (start `u`).
    pub u: T,
    /// Step of the last event (`Δt` is measured from here).
    pub last_event_step: u64,
    /// Last released fraction `y` (telemetry).
    pub released: T,
}

impl<T: Scalar> StpState<T> {
    /// New state at idle (`x = 1`, `u = U`, `released = 1`).
    pub fn new(cfg: StpConfig<T>) -> Self {
        Self {
            x: T::ONE,
            u: cfg.u,
            last_event_step: 0,
            released: T::ONE,
        }
    }

    /// Recovery/revert between events (steps `Δt`).
    pub fn update_between(&mut self, step: u64, cfg: StpConfig<T>) {
        let dt = step.saturating_sub(self.last_event_step);
        if dt == 0 {
            return;
        }
        let dt = T::from_usize(dt as usize);
        self.x = T::ONE + (self.x - T::ONE) * (-dt / cfg.tau_rec).exp();
        self.u = cfg.u + (self.u - cfg.u) * (-dt / cfg.tau_fac).exp();
    }

    /// Release on arrival: `y = u·x` + depletion; returns the release fraction.
    pub fn release(&mut self, cfg: StpConfig<T>) -> T {
        let y = self.u * self.x;
        self.x = (self.x - y).clamp(T::ZERO, T::ONE);
        let bump = (T::ONE - cfg.u) * (T::ONE - (-T::ONE / cfg.tau_fac).exp());
        self.u = (self.u + bump).clamp(T::ZERO, T::ONE);
        y
    }

    /// Record the event: updates `last_event_step` and the telemetry `released`.
    pub fn mark_event(&mut self, step: u64, y: T) {
        self.last_event_step = step;
        self.released = y;
    }
}

/// Learnable delay config — the trainable-delay subsystem (PROMPT 5).
///
/// Per-synapse delays on `DynamicNetwork` driven by a three-factor rule:
/// a teaching signal (see [`DynamicNetwork::set_teaching_signal`]) shifts the
/// delay through a decaying eligibility trace. Independent of STP and
/// long-term plasticity.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LearnableDelayConfig<T = f32> {
    /// Enable learnable delays (all synapses).
    pub enabled: bool,
    /// Minimum delay (≥ 0).
    pub min_delay: T,
    /// Maximum delay.
    pub max_delay: T,
    /// Initial delay (clamped to `[min_delay, max_delay]`).
    pub initial: T,
    /// Learning rate of the delay per teaching-signal unit (per dispatch).
    pub learning_rate: T,
    /// Decay of the teaching-signal eligibility per step (`[0, 1)`).
    pub elig_decay: T,
}

impl<T: Scalar> Default for LearnableDelayConfig<T> {
    fn default() -> Self {
        Self {
            enabled: false,
            min_delay: T::ZERO,
            max_delay: T::c(2.0),
            initial: T::ZERO,
            learning_rate: T::c(0.05),
            elig_decay: T::c(0.95),
        }
    }
}

impl<T: Scalar> LearnableDelayConfig<T> {
    /// Validate the config.
    pub fn validate(&self) -> StructuralResult<()> {
        if !self.min_delay.is_finite() || self.min_delay < T::ZERO {
            return Err(StructuralError::InvalidConfig(
                "LearnableDelayConfig: min_delay must be finite and >= 0".into(),
            ));
        }
        if !self.max_delay.is_finite() || self.max_delay < self.min_delay {
            return Err(StructuralError::InvalidConfig(
                "LearnableDelayConfig: max_delay must be finite and >= min_delay".into(),
            ));
        }
        if !self.initial.is_finite()
            || self.initial.clamp(self.min_delay, self.max_delay) != self.initial
        {
            return Err(StructuralError::InvalidConfig(
                "LearnableDelayConfig: initial must be in [min_delay, max_delay]".into(),
            ));
        }
        if !self.learning_rate.is_finite() {
            return Err(StructuralError::InvalidConfig(
                "LearnableDelayConfig: learning_rate must be finite".into(),
            ));
        }
        if !self.elig_decay.is_finite() || self.elig_decay < T::ZERO || self.elig_decay >= T::ONE {
            return Err(StructuralError::InvalidConfig(
                "LearnableDelayConfig: elig_decay must be in [0, 1)".into(),
            ));
        }
        Ok(())
    }

    /// Clamped initial delay (used when a state is created).
    pub fn init_delay(&self) -> T {
        self.initial.clamp(self.min_delay, self.max_delay)
    }
}

/// Trainable delay state of a synapse (learnable-delay subsystem, PROMPT 5).
///
/// Storage is separate from the legacy integer `Synapse.delay` scheduling
/// field: with learnable delays enabled, the engine uses this state
/// (`Synapse.delay` is ignored for delivery).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TrainableDelayState<T = f32> {
    /// Current delay value (float, engine units — bins to the delivery step).
    pub delay: T,
    /// Accumulated teaching-signal eligibility (three-factor rule; decays per step).
    pub elig: T,
}

impl<T: Scalar> TrainableDelayState<T> {
    /// New state at idle (`delay = initial`).
    pub fn new(initial: T) -> Self {
        Self {
            delay: initial,
            elig: T::ZERO,
        }
    }
}

/// Transmission model of a synapse ([`Synapse::model`](super::topology::Synapse::model)).
///
/// Legacy `CurrentBased` (default): `I = w` — the weight is the current amplitude.
/// Conductance models: `I = g·(E_rev − V)` — the delivered current is computed
/// per step in the engine from the per-synapse conductance `g` and the
/// postsynaptic voltage `V` (pre-step voltage, `NeuronState::mem()`).
///
/// Voltage-dependent models:
/// * `Nmda` — NMDA conductance with Mg²⁺ block (documented form):
///   `I_NMDA = g·B(V)·(E_rev − V)`, `B(V) = 1/(1 + exp((v_half − V)/ΔV))`
///   — the block opens (B→1) at depolarized `V > v_half`;
/// * `Gaba` — GABA conductance with reversal potential (`E_rev < 0`);
///   unlike a sign flip, GABA uses the reversal potential (Dale-compatible:
///   inhibitory sources must have `E_rev ≤ 0`, see `Topology::validate`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SynapseModel<T = f32> {
    /// Legacy current-based (`I = w`) — default.
    CurrentBased,
    /// Conductance-based: `I = g·(E_rev − V)` — `ConductanceParams`.
    ConductanceBased(ConductanceParams<T>),
    /// AMPA — fast excitatory conductance (defaults in [`ConductanceParams::amp`]).
    Ampa,
    /// NMDA — slow excitatory conductance, Mg²⁺ block (defaults in [`ConductanceParams::nmda`]).
    Nmda,
    /// GABA — inhibitory conductance with reversal potential (defaults in [`ConductanceParams::gaba`]).
    Gaba,
}

impl<T: Scalar> SynapseModel<T> {
    /// Is the model conductance-based (dynamic)?
    pub fn is_cond(&self) -> bool {
        matches!(
            self,
            SynapseModel::ConductanceBased(_)
                | SynapseModel::Ampa
                | SynapseModel::Nmda
                | SynapseModel::Gaba
        )
    }

    /// Validate the model parameters (reversal, time constants, g_peak).
    pub fn validate(&self) -> StructuralResult<()> {
        if let SynapseModel::ConductanceBased(p) = self {
            p.validate()?;
        }
        Ok(())
    }

    /// Conductance parameters of this model (`None` for `CurrentBased`).
    pub fn params(&self) -> Option<ConductanceParams<T>> {
        match self {
            SynapseModel::ConductanceBased(p) => Some(*p),
            SynapseModel::Ampa => Some(ConductanceParams::amp()),
            SynapseModel::Nmda => Some(ConductanceParams::nmda()),
            SynapseModel::Gaba => Some(ConductanceParams::gaba()),
            SynapseModel::CurrentBased => None,
        }
    }

    /// The delivered current at voltage `v` given conductance `g`
    /// (for the `ConductanceBased` form: `g·(E_rev − V)`).
    pub fn current_at(&self, g: T, v: T) -> T {
        let Some(p) = self.params() else {
            return T::ZERO;
        };
        if self == &SynapseModel::Ampa {
            p.current(g, v)
        } else {
            p.current(g, v)
        }
    }
}

/// Parameters of the conductance model — `I_syn = g·(E_rev − V)`
/// (engine units — same scale as `LIF` voltages/currents).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ConductanceParams<T = f32> {
    /// Peak conductance `g_peak` on arrival (state jumps to `g` + `g_peak`; decays with `tau`).
    pub g_peak: T,
    /// Decay time constant of the conductance (engine steps).
    pub tau: T,
    /// Reversal potential `E_rev` (GABA: negative; AMPA/NMDA: ≈ 0).
    pub e_rev: T,
    /// (NMDA only) half-voltage of the Mg²⁺ block: `B(V) = 1/(1 + exp((v_half − V)/ΔV))`.
    pub v_half: T,
    /// (NMDA only) slope of the Mg²⁺ block `ΔV`.
    pub delta_v: T,
}

impl<T: Scalar> ConductanceParams<T> {
    /// AMPA (fast excitatory conductance): `τ = 2`, `g_peak = 0.5`, `E_rev = 0`.
    pub fn amp() -> Self {
        Self {
            g_peak: T::c(0.5),
            tau: T::c(2.0),
            e_rev: T::ZERO,
            v_half: T::ZERO,
            delta_v: T::c(0.05),
        }
    }

    /// NMDA (slow conductance, Mg²⁺ block): `τ = 8`, `g_peak = 0.5`, `E_rev = 0`,
    /// block half-voltage `v_half = −0.4`, slope `ΔV = 0.05`.
    pub fn nmda() -> Self {
        Self {
            g_peak: T::c(0.5),
            tau: T::c(8.0),
            e_rev: T::ZERO,
            v_half: T::c(-0.4),
            delta_v: T::c(0.05),
        }
    }

    /// GABA (inhibitory, reversal potential): `τ = 6`, `g_peak = 0.3`, `E_rev = −0.7`.
    pub fn gaba() -> Self {
        Self {
            g_peak: T::c(0.3),
            tau: T::c(6.0),
            e_rev: T::c(-0.7),
            v_half: T::ZERO,
            delta_v: T::c(0.05),
        }
    }

    /// Current contribution `I = g·(E_rev − V)` at voltage `v`.
    pub fn current(&self, g: T, v: T) -> T {
        g * (self.e_rev - v)
    }

    /// NMDA Mg²⁺ block: `B(V) = 1/(1 + exp((v_half − V)/ΔV))` (Open at V > v_half).
    pub fn nmda_block(&self, v: T) -> T {
        T::ONE / (T::ONE + (-(v - self.v_half) / self.delta_v).exp())
    }

    /// Validate parameters.
    pub fn validate(&self) -> StructuralResult<()> {
        if !self.g_peak.is_finite() || self.g_peak < T::ZERO {
            return Err(StructuralError::param(
                "conductance",
                "g_peak must be finite and >= 0",
            ));
        }
        if !self.tau.is_finite() || self.tau <= T::ZERO {
            return Err(StructuralError::param(
                "conductance",
                "tau must be finite and > 0",
            ));
        }
        if !self.e_rev.is_finite() {
            return Err(StructuralError::param(
                "conductance",
                "e_rev must be finite",
            ));
        }
        if !self.delta_v.is_finite() || self.delta_v <= T::ZERO {
            return Err(StructuralError::param(
                "conductance",
                "delta_v must be finite and > 0",
            ));
        }
        Ok(())
    }
}

/// Per-synapse state of the conductance model (`g` — total conductance on the edge, engine units).
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct CondState<T = f32> {
    /// Current conductance `g` (total conductance of the synapse).
    pub g: T,
}

/// Inhibitory plasticity (Vogels et al., 2011): STDP inhibitory synapses,
/// which simultaneously homeostatically maintains the receiver frequency at
/// `target_rate`.
///
/// Works only with Dale principle ([`Topology::dale`]) and enabled
/// plasticity of the engine, regardless of the `StdpConfig::mode` mode. For
/// inhibitory synapse `m = |w|`: at source spike
/// `m += η·(trace_post − α)`, when the receiver spikes `m += η·trace_pre`, where
/// `α = 2·target_rate / (1 − trace_decay)`. A receiver that spikes more often
/// targets, receives stronger inhibition; quieter than the target - weaker.
/// Excitatory synapses are not changed by this rule, but ordinary STDP are inhibitory
/// does not affect synapses.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct InhibitoryPlasticityConfig<T = f32> {
    /// Whether the rule is enabled.
    pub enabled: bool,
    /// Learning rate `η`.
    pub eta: T,
    /// Receiver target frequency (spikes per step).
    pub target_rate: T,
}

impl<T: Scalar> Default for InhibitoryPlasticityConfig<T> {
    fn default() -> Self {
        Self {
            enabled: false,
            eta: T::c(0.001),
            target_rate: T::c(0.02),
        }
    }
}

/// Local adjustment of inhibitory strength based on the balance of excitation and inhibition.
///
/// For each neuron, `inh_drive / exc_drive` (sliding
/// average input current, [`NeuronStats::exc_drive_ema`](super::NeuronStats))
/// with `target_ratio`, and all incoming inhibitory weights of the neuron are multiplied by
/// general coefficient: little inhibition - increases, a lot - weakens.
/// [`DynamicNetwork::balance_inhibition`] is called (the controller calls it
/// on each adaptation cycle, if `enabled`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EiBalanceConfig<T = f32> {
    /// Is adjustment enabled in the controller?
    pub enabled: bool,
    /// Target ratio of inhibition to excitation.
    pub target_ratio: T,
    /// Fraction of error processed per call.
    pub rate: T,
    /// Maximum relative weight change per challenge.
    pub max_step: T,
    /// The minimum excitatory current at which a neuron is regulated.
    pub min_drive: T,
    /// Minimum steps for monitoring a neuron.
    pub min_observed_steps: u64,
}

impl<T: Scalar> Default for EiBalanceConfig<T> {
    fn default() -> Self {
        Self {
            enabled: false,
            target_ratio: T::ONE,
            rate: T::c(0.1),
            max_step: T::c(0.1),
            min_drive: T::c(1e-3),
            min_observed_steps: 200,
        }
    }
}

/// Result [`DynamicNetwork::balance_inhibition`].
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct EiBalanceReport<T = f32> {
    /// How many neurons received a new inhibition coefficient.
    pub adjusted: usize,
    /// Average `inh/exc` for the considered neurons before adjustment.
    pub mean_ratio: T,
}

/// How an external signal affects input neurons.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum InputMode {
    /// Previous mode: value `≥ 0.5` - ready spike, model and state
    /// the input neuron does not participate in the dynamics.
    #[default]
    Spike,
    /// Input neuron - normal dynamic neuron: external value
    /// is added to the incoming current, and the spike is determined by its model.
    Current,
}

impl InputMode {
    /// Short name.
    pub fn as_str(self) -> &'static str {
        match self {
            InputMode::Spike => "spike",
            InputMode::Current => "current",
        }
    }

    pub(crate) fn code(self) -> u8 {
        match self {
            InputMode::Spike => 0,
            InputMode::Current => 1,
        }
    }

    pub(crate) fn from_code(c: u8) -> Option<Self> {
        match c {
            0 => Some(InputMode::Spike),
            1 => Some(InputMode::Current),
            _ => None,
        }
    }
}

/// Engine parameters.
#[derive(Debug, Clone, PartialEq)]
pub struct EngineConfig<T = f32> {
    /// Rule of plasticity.
    pub stdp: StdpConfig<T>,
    /// Coefficient of the moving average spike frequency of a neuron.
    pub rate_alpha: T,
    /// Moving average synapse contribution coefficient.
    pub credit_alpha: T,
    /// Decay of the spike transmission trace by a synapse per step.
    pub activity_trace_decay: T,
    /// Decay of accumulated modulation per step (for modulating sources).
    pub modulation_decay: T,
    /// Input neuron mode.
    pub input_mode: InputMode,
    /// Inhibitory plasticity.
    pub inhibitory: InhibitoryPlasticityConfig<T>,
    /// Adjusting the balance of excitation and inhibition.
    pub balance: EiBalanceConfig<T>,
    /// Short-term synaptic plasticity (Tsodyks–Markram; `None` = off).
    ///
    /// Independent of the long-term plasticity (engine STDP/pipeline).
    pub stp: Option<StpConfig<T>>,
    /// Learnable per-synapse delays (three-factor rule on teaching signals;
    /// `None` = off). Independent of STP and long-term plasticity.
    pub learnable_delay: Option<LearnableDelayConfig<T>>,
}

impl<T: Scalar> Default for EngineConfig<T> {
    fn default() -> Self {
        Self {
            stdp: StdpConfig::default(),
            rate_alpha: T::c(0.01),
            credit_alpha: T::c(0.1),
            activity_trace_decay: T::c(0.9),
            modulation_decay: T::c(0.9),
            input_mode: InputMode::default(),
            inhibitory: InhibitoryPlasticityConfig::default(),
            balance: EiBalanceConfig::default(),
            stp: None,
            learnable_delay: None,
        }
    }
}

impl<T: Scalar> EngineConfig<T> {
    fn validate(&self) -> StructuralResult<()> {
        let unit = |v: T| v.is_finite() && v > T::ZERO && v <= T::ONE;
        let s = &self.stdp;
        let ok = unit(self.rate_alpha)
            && unit(self.credit_alpha)
            && unit(self.activity_trace_decay)
            && unit(self.modulation_decay)
            && unit(s.trace_decay)
            && unit(s.elig_decay)
            && s.a_plus.is_finite()
            && s.a_minus.is_finite()
            && s.reward_lr.is_finite()
            && self.inhibitory.eta.is_finite()
            && self.inhibitory.eta >= T::ZERO
            && self.inhibitory.target_rate.is_finite()
            && self.inhibitory.target_rate >= T::ZERO
            && self.balance.target_ratio.is_finite()
            && self.balance.target_ratio >= T::ZERO
            && self.balance.rate.is_finite()
            && self.balance.rate >= T::ZERO
            && self.balance.max_step.is_finite()
            && self.balance.max_step >= T::ZERO
            && self.balance.max_step < T::ONE
            && self.balance.min_drive.is_finite()
            && self.balance.min_drive >= T::ZERO;
        let stp_ok = match &self.stp {
            Some(cfg) => cfg.validate().is_ok(),
            None => true,
        };
        let ld_ok = match &self.learnable_delay {
            Some(c) => c.validate().is_ok(),
            None => true,
        };
        if ok && stp_ok && ld_ok {
            Ok(())
        } else {
            Err(StructuralError::InvalidConfig(
                "EngineConfig: coefficients must be finite, attenuations must be in (0, 1]".into(),
            ))
        }
    }
}

/// The result of one step.
#[derive(Debug, Clone, PartialEq)]
pub struct StepOutput {
    /// Spikes of output neurons (in ascending order of id).
    pub output_spikes: Vec<bool>,
    /// Total spikes per step (including inputs).
    pub total_spikes: usize,
}

#[derive(Debug, Clone, Default)]
struct Layout {
    version: u64,
    order: Vec<NeuronId>,
    index: BTreeMap<NeuronId, usize>,
    input_slot: Vec<usize>,
    inputs: usize,
    outputs: Vec<usize>,
}

/// Lazy fade: value `v` calculated at step `last`, at step `now`.
pub(crate) fn decayed<T: Scalar>(v: T, last: u64, now: u64, decay: T) -> T {
    let dt = now.saturating_sub(last);
    if dt == 0 || v == T::ZERO {
        v
    } else {
        v * decay.powi(dt.min(100_000) as i32)
    }
}

/// Dynamic network: topology + execution state.
#[derive(Debug, Clone)]
pub struct DynamicNetwork<T: Scalar = f32> {
    pub(super) topo: Topology<T>,
    pub(super) cfg: EngineConfig<T>,
    pub(super) step: u64,
    pub(super) sample: u64,
    pub(super) history: VecDeque<Vec<NeuronId>>,
    /// Spikes of all neurons of the last step — set in `step_inner`
    /// (see [`DynamicNetwork::last_spikes`]).
    last_spikes: Vec<bool>,
    /// Synapses that delivered current to receivers in the last step —
    /// set in `step_inner` (see [`DynamicNetwork::last_arrivals`]).
    last_arrivals: Vec<SynapseId>,
    pub(super) tracker: Option<CoactivityTracker<T>>,
    pub(super) plasticity_enabled: bool,
    /// Unified per-synapse rule pipeline ([`PlasticityPipeline`]) — if attached, the
    /// engine STDP/Vogels paths are off and the pipeline owns weight updates.
    pipeline: Option<PlasticityPipeline<T>>,
    /// Per-synapse Tsodyks–Markram STP states (`stp` engine feature).
    pub(super) stp_states: BTreeMap<SynapseId, StpState<T>>,
    /// Per-synapse conductance state (`SynapseModel::*` models).
    pub(super) cond_states: BTreeMap<SynapseId, CondState<T>>,
    /// Per-synapse trainable delay states (learnable-delay subsystem) — mirrors
    /// [`EngineConfig::learnable_delay`].
    pub(super) delays: BTreeMap<SynapseId, TrainableDelayState<T>>,
    /// External teaching signals per neuron (input to the learnable-delay rule).
    teaching_signals: BTreeMap<NeuronId, T>,
    /// Fractional deliveries scheduled by delivery step
    /// (`frac_deliveries[step] = [(synapse, amplitude), ...]`; integral preservation).
    frac_deliveries: BTreeMap<u64, Vec<(SynapseId, T)>>,
    /// Active-input events — `(step, neuron)` of recent spikes, time-ordered
    /// (mirrors `history` as the canonical sparse input list of the engine —
    /// paper-style: sparse lists of active inputs).
    active_inputs: VecDeque<(u64, NeuronId)>,
    layout: Layout,
}

impl<T: Scalar> DynamicNetwork<T> {
    /// Create a network; topology and parameters are checked.
    pub fn new(topo: Topology<T>, cfg: EngineConfig<T>) -> StructuralResult<Self> {
        topo.validate()?;
        cfg.validate()?;
        Ok(Self {
            topo,
            cfg,
            step: 0,
            sample: 0,
            history: VecDeque::new(),
            last_spikes: Vec::new(),
            last_arrivals: Vec::new(),
            tracker: None,
            plasticity_enabled: true,
            pipeline: None,
            stp_states: BTreeMap::new(),
            cond_states: BTreeMap::new(),
            delays: BTreeMap::new(),
            teaching_signals: BTreeMap::new(),
            frac_deliveries: BTreeMap::new(),
            active_inputs: VecDeque::new(),
            layout: Layout::default(),
        })
    }

    /// Topology (read only).
    pub fn topology(&self) -> &Topology<T> {
        &self.topo
    }

    /// Topology for structural operations. Change it at a safe border.
    pub fn topology_mut(&mut self) -> &mut Topology<T> {
        &mut self.topo
    }

    /// Spikes of ALL neurons of the last step — in the order of
    /// [`Topology::neurons`](crate::structural::Topology::neurons) — for activity recording
    /// (see [`analysis::ActivityHistory::record_step`](crate::structural::analysis::ActivityHistory::record_step)).
    pub fn last_spikes(&self) -> &[bool] {
        &self.last_spikes
    }

    /// Active-input events: `(step, neuron)` of recent spikes, time-ordered —
    /// the canonical sparse input list of the engine (mirrors `history`;
    /// paper-style: sparse lists of active inputs — see `docs/` for the
    /// related work: arXiv:2610.16022).
    pub fn active_inputs(&self) -> &VecDeque<(u64, NeuronId)> {
        &self.active_inputs
    }

    /// Active inputs of the last processed step (sparse input list — the
    /// neurons that spiked, sorted by id).
    pub fn last_active_inputs(&self) -> Vec<NeuronId> {
        let Some(last) = self.active_inputs.iter().map(|&(s, _)| s).max() else {
            return Vec::new();
        };
        let mut v: Vec<NeuronId> = self
            .active_inputs
            .iter()
            .filter_map(|&(s, nid)| (s == last).then_some(nid))
            .collect();
        v.sort_unstable();
        v
    }

    /// Sparse scoring per paper (arXiv:2610.16022, `c_j = Σ_{i∈Q} M_ij`):
    /// per neuron `j`, the count of active-input synapses with non-zero weight
    /// (`M_ij = 1[W_ij ≠ 0]`) — an **integer count** (`u32`) of binary-mask
    /// synapses matching the query's active inputs `Q`.
    ///
    /// `active_inputs` — the set `Q` of active input neurons (e.g. from
    /// [`DynamicNetwork::last_active_inputs`]).
    pub fn count_scores(&self, active_inputs: &[NeuronId]) -> BTreeMap<NeuronId, u32> {
        let active: BTreeSet<NeuronId> = active_inputs.iter().copied().collect();
        let mut out: BTreeMap<NeuronId, u32> = BTreeMap::new();
        for s in self.topo.synapses.values() {
            if !s.is_active() || s.weight == T::ZERO {
                continue;
            }
            if active.contains(&s.src) {
                *out.entry(s.dst).or_insert(0) += 1;
            }
        }
        out
    }

    /// Synapses of the last step that delivered current to receivers
    /// (a delivery, not just a transmission trace: a synapse with delay
    /// `a` delivers when the source spike is in history at age `a`).
    /// Used by the online plasticity system ([`structural::eprop`](super::eprop)).
    pub fn last_arrivals(&self) -> &[SynapseId] {
        &self.last_arrivals
    }

    /// Change the neuron model with an explicit state migration policy.
    ///
    /// A wrapper over [`Topology::change_neuron_model`] for the calling code,
    /// which works with the entire network: resets the layout cache and returns
    /// [`ModelSwitchOutcome`]. Change the model at a safe boundary (between
    /// samples/minibatch), as well as other structural operations.
    pub fn change_neuron_model(
        &mut self,
        id: NeuronId,
        model: impl Into<NeuronModel<T>>,
        migration: StateMigration,
    ) -> StructuralResult<ModelSwitchOutcome<T>> {
        let outcome = self.topo.change_neuron_model(id, model, migration)?;
        self.layout = Layout::default();
        Ok(outcome)
    }

    /// Replace the topology (for example, rollback to a snapshot).
    pub fn replace_topology(&mut self, topo: Topology<T>) {
        self.topo = topo;
        self.layout = Layout::default();
    }

    /// Engine parameters.
    pub fn config(&self) -> &EngineConfig<T> {
        &self.cfg
    }

    /// Global step.
    pub fn step_count(&self) -> u64 {
        self.step
    }

    /// Sample number.
    pub fn sample_count(&self) -> u64 {
        self.sample
    }

    /// Accumulated joint activity, if enabled.
    pub fn tracker(&self) -> Option<&CoactivityTracker<T>> {
        self.tracker.as_ref()
    }

    pub(crate) fn tracker_mut(&mut self) -> Option<&mut CoactivityTracker<T>> {
        self.tracker.as_mut()
    }

    /// Enable the shared activity drive (if not already enabled).
    pub fn enable_coactivity(&mut self, window: u64, capacity: usize, max_recent: usize) {
        if self.tracker.is_none() {
            self.tracker = Some(CoactivityTracker::new(window, capacity, max_recent));
        }
    }

    /// Enable/disable weight changes (for example, during assessment).
    pub fn set_plasticity_enabled(&mut self, on: bool) {
        self.plasticity_enabled = on;
    }

    /// Is plasticity included?
    pub fn plasticity_enabled(&self) -> bool {
        self.plasticity_enabled
    }

    /// Attached [`PlasticityPipeline`] (if any).
    pub fn plasticity_pipeline(&self) -> Option<&PlasticityPipeline<T>> {
        self.pipeline.as_ref()
    }

    /// Mutable access to the attached [`PlasticityPipeline`].
    pub fn plasticity_pipeline_mut(&mut self) -> Option<&mut PlasticityPipeline<T>> {
        self.pipeline.as_mut()
    }

    /// Attach or detach the unified per-synapse rule pipeline ([`PlasticityPipeline`]).
    ///
    /// Strict composition: the engine plasticity must be off when a pipeline is attached
    /// (`StdpConfig::mode == Off` and `InhibitoryPlasticityConfig::enabled == false`), so
    /// the pipeline is the single writer of weight updates and accidental double updates
    /// are impossible. Returns a typed error on conflict or invalid configuration.
    pub fn set_plasticity_pipeline(
        &mut self,
        pl: Option<PlasticityPipeline<T>>,
    ) -> StructuralResult<()> {
        if let Some(p) = &pl {
            if self.cfg.stdp.mode != PlasticityMode::Off || self.cfg.inhibitory.enabled {
                return Err(StructuralError::InvariantViolation(
                    "plasticity pipeline requires the engine plasticity off: \
                     set StdpConfig::mode = Off and InhibitoryPlasticityConfig::enabled = false; \
                     the pipeline owns weight updates (strict composition order: engine OFF, pipeline ON)"
                        .into(),
                ));
            }
            p.validate(self)?;
        }
        let mut pl = pl;
        if let Some(p) = pl.as_mut() {
            p.attach(self)?;
        }
        self.pipeline = pl;
        Ok(())
    }

    /// Short-term plasticity config (if STP is enabled).
    pub fn stp_config(&self) -> Option<StpConfig<T>> {
        self.cfg.stp
    }

    /// Set the STP config; `None` disables STP (clears states).
    pub fn set_stp(&mut self, cfg: Option<StpConfig<T>>) {
        if cfg.is_none() {
            self.stp_states.clear();
        }
        self.cfg.stp = cfg;
    }

    /// Per-synapse STP state of a synapse (telemetry: `x`, `u`, `released`).
    pub fn stp_state(&self, id: SynapseId) -> Option<&StpState<T>> {
        self.stp_states.get(&id)
    }

    /// All per-synapse STP states (telemetry).
    pub fn stp_states(&self) -> &BTreeMap<SynapseId, StpState<T>> {
        &self.stp_states
    }

    /// Per-synapse conductance states (telemetry).
    pub fn cond_states(&self) -> &BTreeMap<SynapseId, CondState<T>> {
        &self.cond_states
    }

    /// Current learnable delay of a synapse (telemetry); `None` if learnable is off.
    pub fn learnable_delay(&self, id: SynapseId) -> Option<T> {
        self.delays.get(&id).map(|st| st.delay)
    }

    /// Eligibility of the learnable-delay rule of a synapse (telemetry).
    pub fn learnable_delay_elig(&self, id: SynapseId) -> Option<T> {
        self.delays.get(&id).map(|st| st.elig)
    }

    /// All learnable-delay states (telemetry).
    pub fn learnable_delays(&self) -> &BTreeMap<SynapseId, TrainableDelayState<T>> {
        &self.delays
    }

    /// Set the learnable delay of a synapse (state is created if needed;
    /// ignored when learnable delays are disabled in the engine config).
    pub fn set_learnable_delay(&mut self, id: SynapseId, value: T) {
        if let Some(c) = self.cfg.learnable_delay {
            if c.enabled {
                let v = value.clamp(c.min_delay, c.max_delay);
                self.delays.insert(id, TrainableDelayState::new(v));
            }
        }
    }

    /// Latest teaching signal of a neuron (input to the learnable-delay rule).
    pub fn teaching_signal(&self, id: NeuronId) -> T {
        self.teaching_signals.get(&id).copied().unwrap_or(T::ZERO)
    }

    /// Set the teaching signal of a neuron (read by the learnable-delay rule at dispatch).
    pub fn set_teaching_signal(&mut self, id: NeuronId, value: T) {
        self.teaching_signals.insert(id, value);
    }

    /// Set teaching signals of the network (replaces the map).
    pub fn set_teaching_signals(&mut self, map: BTreeMap<NeuronId, T>) {
        self.teaching_signals = map;
    }

    /// Fractional deliveries scheduled by delivery step (telemetry).
    pub fn frac_deliveries(&self) -> &BTreeMap<u64, Vec<(SynapseId, T)>> {
        &self.frac_deliveries
    }

    /// Reset dynamic state (membranes, traces, spike history),
    /// without touching the weights and accumulated statistics.
    pub fn reset_state(&mut self) {
        for n in self.topo.neurons.values_mut() {
            n.state = n.model.init_state();
            n.modulation = T::ZERO;
            n.stats.trace = T::ZERO;
        }
        for s in self.topo.synapses.values_mut() {
            s.stats.activity_trace = T::ZERO;
            s.stats.trace_step = self.step;
            s.elig = T::ZERO;
            s.elig_step = self.step;
        }
        self.history.clear();
        if let Some(t) = &mut self.tracker {
            t.clear_recent();
        }
    }

    /// Start a new sample (episode): counter +1 and reset state.
    pub fn begin_sample(&mut self) {
        self.sample += 1;
        self.reset_state();
    }

    /// Quality Assessment Copy: Plasticity Off, Condition
    /// reset, the drive is not copied. The original network is not affected.
    pub fn probe_copy(&self) -> DynamicNetwork<T> {
        let mut c = self.clone();
        c.plasticity_enabled = false;
        c.tracker = None;
        c.reset_state();
        c
    }

    /// Structural indicators.
    pub fn stats(&self, silent_rate: T, overactive_rate: T) -> StructuralStats<T> {
        StructuralStats::compute(
            &self.topo,
            self.step,
            self.sample,
            silent_rate,
            overactive_rate,
        )
    }

    fn ensure_layout(&mut self) {
        if self.layout.version == self.topo.version {
            return;
        }
        let order: Vec<NeuronId> = self.topo.neurons.keys().copied().collect();
        let index: BTreeMap<NeuronId, usize> =
            order.iter().enumerate().map(|(i, id)| (*id, i)).collect();
        let mut input_slot = vec![usize::MAX; order.len()];
        let mut inputs = 0;
        let mut outputs = Vec::new();
        for (i, id) in order.iter().enumerate() {
            match self.topo.neurons[id].role {
                NeuronRole::Input => {
                    input_slot[i] = inputs;
                    inputs += 1;
                }
                NeuronRole::Output => outputs.push(i),
                NeuronRole::Hidden => {}
            }
        }
        self.layout = Layout {
            version: self.topo.version,
            order,
            index,
            input_slot,
            inputs,
            outputs,
        };
    }

    /// One step; `inputs` - value for each input neuron (in order
    /// increasing id). In [`InputMode::Spike`] the value `≥ 0.5` means
    /// spike; in [`InputMode::Current`] the value is added to the incoming current
    /// input neuron (finite number).
    pub fn step(&mut self, inputs: &[T]) -> StructuralResult<StepOutput> {
        self.step_with_current(inputs, &[])
    }

    /// Step with additional external current into selected neurons (e.g.
    /// “teacher”, forcibly driving the correct output).
    pub fn step_with_current(
        &mut self,
        inputs: &[T],
        extra: &[(NeuronId, T)],
    ) -> StructuralResult<StepOutput> {
        self.ensure_layout();
        let lay = std::mem::take(&mut self.layout);
        let res = self.step_inner(&lay, inputs, extra);
        self.layout = lay;
        res
    }

    /// Deliver scheduled fractional events of step `step` (learnable delays).
    ///
    /// Replicates the legacy delivery path (polarity/model/STP/E-I tracking) with
    /// a fractional amplitude — used only in learnable-delay mode.
    fn deliver_frac_deliveries(
        &mut self,
        step: u64,
        lay: &Layout,
        current: &mut [T],
        exc_in: &mut [T],
        inh_in: &mut [T],
        arrivals: &mut Vec<SynapseId>,
        dale: bool,
    ) {
        let Some(v) = self.frac_deliveries.remove(&step) else {
            return;
        };
        for (sid, amount) in v {
            let Some(s) = self.topo.synapses.get(&sid) else {
                continue;
            };
            if !s.is_active() {
                continue;
            }
            let Some(&di) = lay.index.get(&s.dst) else {
                continue;
            };
            // STP multiplier (same as legacy).
            let mut stp_y = T::ONE;
            if let Some(cfg) = self.cfg.stp {
                if cfg.mode != StpMode::Static {
                    let st = self
                        .stp_states
                        .entry(sid)
                        .or_insert_with(|| StpState::new(cfg));
                    st.update_between(step, cfg);
                    stp_y = st.release(cfg);
                    st.mark_event(step, stp_y);
                }
            }
            if let Some(p) = s.model.params() {
                let st = self.cond_states.entry(sid).or_default();
                st.g = st.g + p.g_peak;
                let v_pre = self
                    .topo
                    .neurons
                    .get(&s.dst)
                    .map_or(T::ZERO, |n| n.state.mem());
                let e = p.current(st.g, v_pre) * stp_y * amount;
                if e != T::ZERO {
                    current[di] += e;
                    if e >= T::ZERO {
                        exc_in[di] += e;
                    } else {
                        inh_in[di] -= e;
                    }
                }
            } else {
                let polarity = self
                    .topo
                    .neurons
                    .get(&s.src)
                    .map_or(NeuronPolarity::Excitatory, |n| n.polarity);
                let effective = match polarity {
                    NeuronPolarity::Excitatory => Some(s.weight),
                    NeuronPolarity::Inhibitory if dale => Some(s.weight),
                    NeuronPolarity::Inhibitory => Some(-s.weight),
                    NeuronPolarity::Modulatory => None,
                };
                if let Some(e0) = effective {
                    let e = e0 * stp_y * amount;
                    current[di] += e;
                    if e >= T::ZERO {
                        exc_in[di] += e;
                    } else {
                        inh_in[di] -= e;
                    }
                }
            }
            arrivals.push(sid);
        }
    }

    fn step_inner(
        &mut self,
        lay: &Layout,
        inputs: &[T],
        extra: &[(NeuronId, T)],
    ) -> StructuralResult<StepOutput> {
        if inputs.len() != lay.inputs {
            return Err(StructuralError::InputSize {
                op: "DynamicNetwork::step",
                expected: lay.inputs,
                actual: inputs.len(),
            });
        }
        if self.cfg.input_mode == InputMode::Current && inputs.iter().any(|v| !v.is_finite()) {
            return Err(StructuralError::param(
                "input",
                "in Current mode the external current must be finite",
            ));
        }
        let n = lay.order.len();
        let step = self.step;
        let sample = self.sample;
        let mut current = vec![T::ZERO; n];
        for (id, c) in extra {
            let i = *lay
                .index
                .get(id)
                .ok_or(StructuralError::NoSuchNeuron(*id))?;
            if !c.is_finite() {
                return Err(StructuralError::param("current", "non-numeric current"));
            }
            current[i] += *c;
        }

        // 1. Delivery of spikes to active synapses. Source polarity
        //    determines the contribution: excitatory - current, inhibitory - current with reverse
        //    sign, modulating - damped gain of the receiver.
        let dale = self.topo.limits.enforce_dale;
        // Conductance models: decay all conductance states before delivery.
        for (sid, st) in self.cond_states.iter_mut() {
            let Some(s) = self.topo.synapses.get(sid) else {
                continue;
            };
            let Some(p) = s.model.params() else {
                continue;
            };
            if s.is_active() {
                st.g = st.g * (-T::ONE / p.tau).exp();
            }
        }
        self.cond_states
            .retain(|sid, _| self.topo.synapses.get(sid).is_some_and(|s| s.is_active()));
        self.stp_states
            .retain(|sid, _| self.topo.synapses.get(sid).is_some_and(|s| s.is_active()));
        self.delays
            .retain(|sid, _| self.topo.synapses.get(sid).is_some_and(|s| s.is_active()));
        let mut arrivals: Vec<SynapseId> = Vec::new();
        let mut mod_in = vec![T::ZERO; n];
        let mut exc_in = vec![T::ZERO; n];
        let mut inh_in = vec![T::ZERO; n];
        // Learnable delays: deliver events due this step (scheduled by dispatches).
        self.deliver_frac_deliveries(
            step,
            &lay,
            &mut current,
            &mut exc_in,
            &mut inh_in,
            &mut arrivals,
            dale,
        );
        let learnable = self.cfg.learnable_delay.filter(|c| c.enabled);
        for (age, frame) in self.history.iter().enumerate() {
            for nid in frame {
                for sid in self.topo.out_synapses(*nid) {
                    let Some(s) = self.topo.synapses.get(sid) else {
                        continue;
                    };
                    if !s.is_active() {
                        continue;
                    }
                    // Learnable delay: update the state (three-factor rule) and
                    // schedule fractional deliveries (dispatch step = `step − 1 − age`
                    // — the frame of this spike). Legacy delivery of these synapses
                    // happens via `frac_deliveries`.
                    if let Some(cfg) = learnable {
                        let d_sig = self
                            .teaching_signals
                            .get(&s.dst)
                            .copied()
                            .unwrap_or(T::ZERO);
                        let st = self
                            .delays
                            .entry(*sid)
                            .or_insert_with(|| TrainableDelayState::new(cfg.initial));
                        st.elig = st.elig * cfg.elig_decay + d_sig;
                        st.delay = (st.delay + cfg.learning_rate * st.elig)
                            .clamp(cfg.min_delay, cfg.max_delay);
                        let dv = st.delay.to_f64();
                        let base = dv.floor();
                        let frac = dv - base;
                        let base_u = base as u64;
                        let dispatch = step.saturating_sub(1 + age as u64);
                        // Fractional semantics (documented): the pulse splits between
                        // `dispatch + 1 + floor(d)` and the next bin — adjacent bins,
                        // integral (charge) preserved.
                        let due = (dispatch + 1 + base_u).max(step);
                        self.frac_deliveries
                            .entry(due)
                            .or_default()
                            .push((*sid, T::from_f64(1.0 - frac)));
                        if frac > 0.0 {
                            self.frac_deliveries
                                .entry(due + 1)
                                .or_default()
                                .push((*sid, T::from_f64(frac)));
                        }
                        continue;
                    }
                    if s.delay as usize != age {
                        continue;
                    }
                    // STP: per-arrival Tsodyks–Markram release multiplier `y`
                    // (`Static` mode = no STP, y ≡ 1).
                    let mut stp_y = T::ONE;
                    if let Some(cfg) = self.cfg.stp {
                        if cfg.mode != StpMode::Static {
                            let st = self
                                .stp_states
                                .entry(*sid)
                                .or_insert_with(|| StpState::new(cfg));
                            st.update_between(step, cfg);
                            stp_y = st.release(cfg);
                            st.mark_event(step, stp_y);
                        }
                    }
                    if let Some(&di) = lay.index.get(&s.dst) {
                        if let Some(p) = s.model.params() {
                            // Conductance-based delivery: on arrival, `g += g_peak`;
                            // delivered current = `g·(E_rev − V_pre)·stp_y`
                            // (`V_pre` = postsynaptic voltage at the start of the step).
                            let st = self.cond_states.entry(*sid).or_default();
                            st.g = st.g + p.g_peak;
                            let v_pre = self
                                .topo
                                .neurons
                                .get(&s.dst)
                                .map_or(T::ZERO, |n| n.state.mem());
                            let e = p.current(st.g, v_pre) * stp_y;
                            if e != T::ZERO {
                                current[di] += e;
                                if e >= T::ZERO {
                                    exc_in[di] += e;
                                } else {
                                    inh_in[di] -= e;
                                }
                            }
                        } else {
                            // Legacy current-based (weight) delivery.
                            let polarity = self
                                .topo
                                .neurons
                                .get(&s.src)
                                .map_or(NeuronPolarity::Excitatory, |n| n.polarity);
                            // With Dale's principle, the sign is already in weight (E ≥ 0, I ≤ 0);
                            // in legacy mode, the inhibitory source reverses the sign.
                            let effective = match polarity {
                                NeuronPolarity::Excitatory => Some(s.weight),
                                NeuronPolarity::Inhibitory if dale => Some(s.weight),
                                NeuronPolarity::Inhibitory => Some(-s.weight),
                                NeuronPolarity::Modulatory => {
                                    mod_in[di] += s.weight.abs() * stp_y;
                                    None
                                }
                            };
                            if let Some(e) = effective {
                                // STP multiplies the delivered amplitude, sign preserved (E/I).
                                current[di] += e * stp_y;
                                if e >= T::ZERO {
                                    exc_in[di] += e * stp_y;
                                } else {
                                    inh_in[di] -= e * stp_y;
                                }
                            }
                        }
                        arrivals.push(*sid);
                    }
                }
            }
        }
        // Learnable delays: deliver events due this step scheduled during the
        // legacy-loop dispatch pass (same-step deliveries).
        self.deliver_frac_deliveries(
            step,
            &lay,
            &mut current,
            &mut exc_in,
            &mut inh_in,
            &mut arrivals,
            dale,
        );

        // 2. Dynamics of neurons.
        let st = self.cfg.stdp;
        let alpha = self.cfg.rate_alpha;
        let mod_decay = self.cfg.modulation_decay;
        let mut old_trace = vec![T::ZERO; n];
        let mut spiked = vec![false; n];
        let mut spikers: Vec<NeuronId> = Vec::new();
        for (i, nid) in lay.order.iter().enumerate() {
            let Some(neuron) = self.topo.neurons.get_mut(nid) else {
                continue;
            };
            old_trace[i] = neuron.stats.trace;
            let ctx = NeuronStepContext {
                neuron_id: *nid,
                step,
                sample,
                dt: T::ONE,
            };
            let spike = if neuron.role == NeuronRole::Input {
                let external = inputs[lay.input_slot[i]];
                match self.cfg.input_mode {
                    InputMode::Spike => external >= T::c(0.5),
                    InputMode::Current => {
                        let gain = neuron.modulation + mod_in[i];
                        let effective = (current[i] + external) * (T::ONE + gain);
                        neuron.modulation = gain * mod_decay;
                        neuron.model.step(&mut neuron.state, effective, &ctx)?
                    }
                }
            } else {
                let gain = neuron.modulation + mod_in[i];
                let effective = current[i] * (T::ONE + gain);
                neuron.modulation = gain * mod_decay;
                neuron.model.step(&mut neuron.state, effective, &ctx)?
            };
            let stt = &mut neuron.stats;
            stt.exc_drive_ema += alpha * (exc_in[i] - stt.exc_drive_ema);
            stt.inh_drive_ema += alpha * (inh_in[i] - stt.inh_drive_ema);
            stt.steps += 1;
            if spike {
                stt.spikes += 1;
                stt.last_spike_step = Some(step);
                if stt.last_marked_sample != sample + 1 {
                    stt.active_samples += 1;
                    stt.last_marked_sample = sample + 1;
                }
                spikers.push(*nid);
            }
            let s = if spike { T::ONE } else { T::ZERO };
            stt.rate_ema += alpha * (s - stt.rate_ema);
            stt.trace = stt.trace * st.trace_decay + s;
            spiked[i] = spike;
        }

        // Synapse usage statistics.
        let act_decay = self.cfg.activity_trace_decay;
        for sid in &arrivals {
            if let Some(s) = self.topo.synapses.get_mut(sid) {
                s.stats.uses += 1;
                s.stats.drive_sum += s.weight.abs();
                if s.stats.last_marked_sample != sample + 1 {
                    s.stats.active_samples += 1;
                    s.stats.last_marked_sample = sample + 1;
                }
                s.stats.activity_trace =
                    decayed(s.stats.activity_trace, s.stats.trace_step, step, act_decay) + T::ONE;
                s.stats.trace_step = step;
            }
        }

        // 3. Plasticity (event shape pair-STDP).
        if self.plasticity_enabled && st.mode != PlasticityMode::Off {
            let mut upd: BTreeMap<SynapseId, (T, T)> = BTreeMap::new();
            for sid in &arrivals {
                let Some(s) = self.topo.synapses.get(sid) else {
                    continue;
                };
                let Some(&di) = lay.index.get(&s.dst) else {
                    continue;
                };
                if dale && self.topo.polarity_of(s.src) == NeuronPolarity::Inhibitory {
                    continue; // inhibitory synapses learn their own rules
                }
                let pot = st.a_plus * old_trace[di];
                if pot != T::ZERO {
                    upd.entry(*sid).or_default().0 += pot;
                }
            }
            for dst in &spikers {
                for sid in self.topo.in_synapses(*dst) {
                    let Some(s) = self.topo.synapses.get(sid) else {
                        continue;
                    };
                    if !s.is_active() {
                        continue;
                    }
                    if dale && self.topo.polarity_of(s.src) == NeuronPolarity::Inhibitory {
                        continue;
                    }
                    let Some(&si) = lay.index.get(&s.src) else {
                        continue;
                    };
                    let dep = st.a_minus * old_trace[si];
                    if dep != T::ZERO {
                        upd.entry(*sid).or_default().1 += dep;
                    }
                }
            }
            let wmax = self.topo.param_limits.weight_abs_max;
            let topo = &mut self.topo;
            for (sid, (pot, dep)) in upd {
                let Some(s) = topo.synapses.get_mut(&sid) else {
                    continue;
                };
                let src_pol = topo
                    .neurons
                    .get(&s.src)
                    .map_or(NeuronPolarity::Excitatory, |n| n.polarity);
                let dw = match st.bound {
                    WeightBound::Soft { w_min, w_max } => {
                        pot * (w_max - s.weight) - dep * (s.weight - w_min)
                    }
                    _ => pot - dep,
                };
                match st.mode {
                    PlasticityMode::Immediate => {
                        s.weight = bound_weight(s.weight + dw, st.bound, wmax);
                        if dale {
                            s.weight = src_pol.clamp_sign(s.weight);
                        }
                    }
                    PlasticityMode::RewardModulated => {
                        s.elig = decayed(s.elig, s.elig_step, step, st.elig_decay) + dw;
                        s.elig_step = step;
                    }
                    PlasticityMode::Off => {}
                }
            }
        }

        // 3b. Inhibitory plasticity (Vogels): only with Dale's principle.
        let ic = self.cfg.inhibitory;
        if dale && ic.enabled && self.plasticity_enabled && ic.eta > T::ZERO {
            let alpha_i = T::c(2.0) * ic.target_rate / (T::ONE - st.trace_decay).max(T::c(1e-6));
            let mut dm: BTreeMap<SynapseId, T> = BTreeMap::new();
            for sid in &arrivals {
                let Some(s) = self.topo.synapses.get(sid) else {
                    continue;
                };
                if self.topo.polarity_of(s.src) != NeuronPolarity::Inhibitory {
                    continue;
                }
                if let Some(&di) = lay.index.get(&s.dst) {
                    *dm.entry(*sid).or_default() += ic.eta * (old_trace[di] - alpha_i);
                }
            }
            for dst in &spikers {
                for sid in self.topo.in_synapses(*dst) {
                    let Some(s) = self.topo.synapses.get(sid) else {
                        continue;
                    };
                    if !s.is_active() || self.topo.polarity_of(s.src) != NeuronPolarity::Inhibitory
                    {
                        continue;
                    }
                    if let Some(&si) = lay.index.get(&s.src) {
                        *dm.entry(*sid).or_default() += ic.eta * old_trace[si];
                    }
                }
            }
            let wmax = self.topo.param_limits.weight_abs_max;
            for (sid, delta) in dm {
                if let Some(s) = self.topo.synapses.get_mut(&sid) {
                    s.weight = -((-s.weight + delta).clamp(T::ZERO, wmax));
                }
            }
        }

        self.last_arrivals = arrivals;

        // 4. History and accumulator of joint activity.
        let total_spikes = spikers.len();
        if let Some(t) = &mut self.tracker {
            t.observe(&self.topo, step, &spikers);
        }
        // Active-input list — `(step, neuron)` events, time-ordered (push_back with
        // increasing steps). Same window bound as `history`.
        for &nid in &spikers {
            self.active_inputs.push_back((step, nid));
        }
        self.active_inputs
            .truncate(self.topo.param_limits.delay_max as usize + 1);
        self.history.push_front(spikers);
        self.history
            .truncate(self.topo.param_limits.delay_max as usize + 1);
        self.step += 1;

        let out = StepOutput {
            output_spikes: lay.outputs.iter().map(|&i| spiked[i]).collect(),
            total_spikes,
        };
        self.last_spikes = spiked;

        // Pipeline: unified per-synapse rule updates after the step
        // (engine STDP is Off when a pipeline is attached — see
        // `DynamicNetwork::set_plasticity_pipeline`).
        if let Some(mut pl) = self.pipeline.take() {
            let r = pl.apply(self);
            self.pipeline = Some(pl);
            r?;
        }
        Ok(out)
    }

    /// Apply Reward: In `RewardModulated` mode, the weight changes to
    /// `lr · r · eligibility`; In any mode, the contribution rating is updated
    /// synapses ([`assign_credit`](Self::assign_credit)).
    pub fn apply_reward(&mut self, reward: T) {
        if !reward.is_finite() {
            return;
        }
        // Adapter: with an attached `PlasticityPipeline`, the reward is routed to the
        // pipeline's `RewardStdp` path (engine STDP is Off, see `set_plasticity_pipeline`).
        if let Some(pl) = &mut self.pipeline {
            pl.set_reward(reward);
        }
        let st = self.cfg.stdp;
        let step = self.step;
        if self.plasticity_enabled && st.mode == PlasticityMode::RewardModulated {
            let wmax = self.topo.param_limits.weight_abs_max;
            let dale = self.topo.limits.enforce_dale;
            let topo = &mut self.topo;
            for s in topo.synapses.values_mut() {
                if !s.is_active() {
                    continue;
                }
                let e = decayed(s.elig, s.elig_step, step, st.elig_decay);
                if e != T::ZERO {
                    let mut w = bound_weight(s.weight + st.reward_lr * reward * e, st.bound, wmax);
                    if dale {
                        let pol = topo
                            .neurons
                            .get(&s.src)
                            .map_or(NeuronPolarity::Excitatory, |n| n.polarity);
                        w = pol.clamp_sign(w);
                    }
                    s.weight = w;
                }
            }
        }
        self.assign_credit(reward);
    }

    /// Local adjustment of inhibitory strength (see [`EiBalanceConfig`]).
    ///
    /// Works on Dale's principle; otherwise it does nothing. Only changes
    /// the magnitudes of inhibitory weights (the sign is preserved), so the structure and
    /// the Dale invariant are not affected.
    pub fn balance_inhibition(&mut self, cfg: &EiBalanceConfig<T>) -> EiBalanceReport<T> {
        let mut report = EiBalanceReport::default();
        if !self.topo.limits.enforce_dale || cfg.target_ratio <= T::ZERO {
            return report;
        }
        let wmax = self.topo.param_limits.weight_abs_max;
        let mut factors: BTreeMap<NeuronId, T> = BTreeMap::new();
        let mut ratio_sum = T::ZERO;
        for n in self.topo.neurons.values() {
            if n.role == NeuronRole::Input
                || n.stats.steps < cfg.min_observed_steps
                || n.stats.exc_drive_ema < cfg.min_drive
            {
                continue;
            }
            let ratio = n.stats.inh_drive_ema / n.stats.exc_drive_ema;
            ratio_sum += ratio;
            let error = (cfg.target_ratio - ratio) / cfg.target_ratio;
            let factor =
                (T::ONE + cfg.rate * error).clamp(T::ONE - cfg.max_step, T::ONE + cfg.max_step);
            factors.insert(n.id, factor);
        }
        if factors.is_empty() {
            return report;
        }
        let considered = factors.len();
        let topo = &mut self.topo;
        let mut touched: BTreeMap<NeuronId, bool> = BTreeMap::new();
        for s in topo.synapses.values_mut() {
            let Some(&f) = factors.get(&s.dst) else {
                continue;
            };
            if !s.is_active()
                || topo.neurons.get(&s.src).map(|n| n.polarity) != Some(NeuronPolarity::Inhibitory)
            {
                continue;
            }
            s.weight = (s.weight * f).clamp(-wmax, T::ZERO);
            touched.insert(s.dst, true);
        }
        report.adjusted = touched.len();
        report.mean_ratio = ratio_sum / T::from_usize(considered);
        report
    }

    /// Take into account the quality signal (more is better: reward or
    /// improvement in loss) in assessing the contribution of active synapses: `credit_ema` is growing
    /// at synapses that transmitted spikes before the positive signal.
    /// Synapses without recent transmission do not change their score, so they rarely
    /// a used but useful synapse does not “forget” its contribution.
    pub fn assign_credit(&mut self, signal: T) {
        if !signal.is_finite() {
            return;
        }
        let (step, beta, decay) = (
            self.step,
            self.cfg.credit_alpha,
            self.cfg.activity_trace_decay,
        );
        for s in self.topo.synapses.values_mut() {
            if !s.is_active() {
                continue;
            }
            let tr = decayed(s.stats.activity_trace, s.stats.trace_step, step, decay);
            if tr > T::c(1e-6) {
                s.stats.credit_ema += beta * (signal * tr - s.stats.credit_ema);
            }
        }
    }
}

fn bound_weight<T: Scalar>(w: T, bound: WeightBound<T>, wmax: T) -> T {
    let w = match bound {
        WeightBound::Hard { w_min, w_max } | WeightBound::Soft { w_min, w_max } => {
            w.clamp(w_min, w_max)
        }
        WeightBound::None => w,
    };
    w.clamp(-wmax, wmax)
}
