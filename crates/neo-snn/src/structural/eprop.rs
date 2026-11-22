//! Online plasticity system of the dynamic network: a local-eligibility
//! weight update without backprop through time ([`DynamicNetwork`](super::DynamicNetwork)).
//!
//! This is *not* R-STDP and *not* backpropagation: the rule couples a
//! **neuron learning signal** `d_j(t)` (a local, instantaneous quantity) with
//! a **synapse eligibility trace** `e` (decaying memory of presynaptic
//! activity). The weight update is applied online after every step
//! (`W ← W − η·d·e`), so the system needs `O(N + S)` memory:
//! `O(S)` traces, `O(N)` signals, `O(1)` config. No BPTT, no global loss,
//! no graph replay: the trace *is* the history.
//!
//! # Concepts (separated)
//!
//! | Concept | What it is | Storage |
//! |---|---|---|
//! | Synapse eligibility trace `e_{j→i}(t)` | Decaying trace of the eligibility factor, updated when the synapse delivers and/or the postsynaptic neuron spikes | `EpropState::synapse_traces`, `O(S)` |
//! | Neuron learning signal `d_j(t)` | Instantaneous local gradient-like value: readout error (outputs), feedback cascade (hidden), or external value (RL/hybrid) | `EpropState::neuron_signals`, `O(N)` |
//! | Weight update rule | `W_{j→i} ← W_{j→i} − η·d_j(t)·e_{j→i}(t)` then projection (weight bounds + Dale) | writes through [`Topology::set_synapse_weight`](super::topology::Topology::set_synapse_weight) |
//! | Online trainer state | Counters (`steps`, `samples`, `updates`), targets, external signals, feedback weights | `EpropState` |
//! | Weight bounds | Projection onto `ParamLimits::weight_abs_max` and the Dale half-plane of the source | `set_synapse_weight` |
//!
//! # Learning signals (requirement: local error / feedback / external)
//!
//! * [`ReadoutMode::LocalError`] — output neurons get `d_j(t) = ŷ_j(t) − y_j^*(t)`
//!   (target `y^*` set with [`EpropTrainer::set_target`], default `0`);
//!   hidden neurons get the feedback cascade (see below).
//! * [`FeedbackScheme`] — how hidden neurons receive signals:
//!   [`FeedbackScheme::RandomFixed`] (fixed random weights, seeded),
//!   [`FeedbackScheme::Symmetric`] (mirrored learned weights
//!   `F[j][k] = W_{j→k}`), [`FeedbackScheme::None`].
//!   For hidden `j` and output `k`: `d_j(t) = Σ_k F[j][k]·d_k(t)`.
//! * [`ReadoutMode::External`] — all learning signals come from
//!   [`EpropTrainer::set_external_signal`] (RL/hybrid systems).
//!
//! # Mathematical derivation for LIF and Adaptive LIF
//!
//! **LIF** (as implemented by `lif_step`):
//!
//! ```text
//! U_j(t) = β·U_j(t−1) + I_j(t)          I_j(t) = Σ_{i ∈ in(j)} W_{j→i}·x_i(t)
//! ŷ_j(t) = [U_j(t) ≥ θ_j]               (hard spike; Subtract reset: U ← U − θ)
//! ```
//!
//! Eligibility of synapse `i→j` (the trace is nonzero on the spike steps of
//! the postsynaptic neuron):
//!
//! ```text
//! e_{j→i}(t) = γ·e_{j→i}(t−1) + ŷ_j(t)·x_i(t)
//! ```
//!
//! with the convention `x_i(t) = 1` if the synapse `i→j` delivered current at
//! step `t` (`DynamicNetwork::last_arrivals`), else `0`.
//!
//! Weight update (one-step online form; the trace accumulates the history,
//! so `∂L/∂W ≈ Σ_{t'≤t} d_j(t')·e(t')` is approximated by `d_j(t)·e(t)`):
//!
//! ```text
//! W_{j→i}(t) ← W_{j→i}(t−1) − η·d_j(t)·e_{j→i}(t)
//! ```
//!
//! where `d_j(t)` is the learning signal of neuron `j` (§"Learning signals").
//!
//! **Adaptive LIF** (`adaptive_lif_step`):
//!
//! ```text
//! U_j(t) = β·U_j(t−1) + I_j(t) − A_j(t−1)
//! A_j(t) = ρ·A_j(t−1) + a·spike_spk(t)   (on spike; A *= ρ otherwise)
//! ŷ_j(t) = [U_j(t) ≥ θ_j]
//! ```
//!
//! Introduce the potential `V_j(t) = U_j(t) + A_j(t−1) = β·U_j(t−1) + I_j(t)`.
//! Since the spike condition is `U_j(t) ≥ θ_j` ⟺ `V_j(t) ≥ θ_j + A_j(t−1)`,
//! the ALIF neuron is exactly an LIF neuron with the membrane `V_j` and the
//! **adaptive threshold** `Θ_j(t) = θ_j + A_j(t−1)`.
//! Its eligibility follows the LIF rule:
//!
//! ```text
//! e_{j→i}(t) = γ·e_{j→i}(t−1) + ŷ_j(t)·x_i(t)
//! ```
//!
//! (a larger `A_j` raises `Θ_j`, so repeated firing slows down — the standard
//! adaptation effect falls out of the derivation).
//!
//! # Hybrid use and composition order
//!
//! The trainer is the only writer of weights for its enabled synapses
//! (writes go through [`Topology::set_synapse_weight`](super::topology::Topology::set_synapse_weight)).
//! R-STDP/pair-STDP are separate engine rules: they do not call the trainer
//! and vice versa — no accidental double updates. When combining with STDP:
//! the trainer updates at `trainer.step(net)`; explicit calls to
//! [`DynamicNetwork::apply_reward`](super::network::DynamicNetwork::apply_reward)
//! are the caller's choice.
//!
//! # Limitations
//!
//! * Online system (`structural::eprop`) is a **dynamic** (`Scalar`) system:
//!   CPU-only, LIF and AdaptiveLIF only. Other models
//!   (Izhikevich, AdEx, Hodgkin–Huxley, custom) are rejected with
//!   [`StructuralError::UnsupportedEpropDynamics`](super::error::StructuralError::UnsupportedEpropDynamics).
//! * Traces of disabled synapses are not updated; `e = 0` on non-spike
//!   steps (the trace decays by `γ`).

use std::collections::{BTreeMap, BTreeSet};

use crate::scalar::Scalar;

use super::error::{StructuralError, StructuralResult};
use super::model::NeuronRole;
use super::network::DynamicNetwork;
use super::rng::Rng64;
use super::topology::{NeuronId, SynapseId};

fn flag<T: Scalar>(b: bool) -> T {
    if b { T::ONE } else { T::ZERO }
}

// ---------------------------------------------------------------------
// Scope
// ---------------------------------------------------------------------

/// Independent enabling of the plasticity on a set of connections:
/// by source/target roles ("layers"), or by explicit synapse sets
/// ("connections"). Default: sources = {Input, Hidden, Output},
/// targets = {Hidden, Output}.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct EpropScope {
    /// Enabled source roles (synapse source).
    pub sources: BTreeSet<NeuronRole>,
    /// Enabled target roles (synapse receiver).
    pub targets: BTreeSet<NeuronRole>,
    /// If `Some`: only these synapses are enabled (the allow-list).
    pub only_synapses: Option<BTreeSet<SynapseId>>,
    /// Explicitly disabled synapses (the deny-list).
    pub disabled_synapses: BTreeSet<SynapseId>,
}

impl Default for EpropScope {
    fn default() -> Self {
        Self {
            sources: [NeuronRole::Input, NeuronRole::Hidden, NeuronRole::Output].into(),
            targets: [NeuronRole::Hidden, NeuronRole::Output].into(),
            only_synapses: None,
            disabled_synapses: BTreeSet::new(),
        }
    }
}

impl EpropScope {
    /// All connections (inputs, hidden, outputs) are enabled.
    pub fn all() -> Self {
        Self {
            sources: [NeuronRole::Input, NeuronRole::Hidden, NeuronRole::Output].into(),
            targets: [NeuronRole::Input, NeuronRole::Hidden, NeuronRole::Output].into(),
            only_synapses: None,
            disabled_synapses: BTreeSet::new(),
        }
    }

    /// No connection is enabled (plasticity fully off, for debugging).
    pub fn none() -> Self {
        Self {
            sources: BTreeSet::new(),
            targets: BTreeSet::new(),
            only_synapses: None,
            disabled_synapses: BTreeSet::new(),
        }
    }
}

// ---------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------

/// Where the learning signals of output neurons come from.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ReadoutMode {
    /// `d_j(t) = ŷ_j(t) − y_j^*(t)` for output neurons
    /// (targets via [`EpropTrainer::set_target`]); hidden neurons get
    /// the feedback cascade (`FeedbackScheme`).
    #[default]
    LocalError,
    /// All learning signals come from [`EpropTrainer::set_external_signal`]
    /// (RL/hybrid systems).
    External,
}

/// How hidden neurons receive learning signals.
#[derive(Debug, Clone, PartialEq)]
pub enum FeedbackScheme<T = f32> {
    /// Fixed random weights `F[j][k] ~ U(−m, +m)` (seeded by `seed`, drawn in
    /// deterministic order of hidden neurons then output neurons).
    RandomFixed {
        /// Seed.
        seed: u64,
        /// Magnitude of the random weights.
        magnitude: T,
    },
    /// Symmetric (mirrored learned weights): `F[j][k]` = the weight of the
    /// synapse `j → k` (the hidden neuron's outgoing weight), `0` if absent.
    Symmetric,
    /// No feedback (hidden neurons have zero signals unless external
    /// signals are set).
    None,
}

/// Configuration of the online plasticity system.
#[derive(Debug, Clone, PartialEq)]
pub struct EpropConfig<T = f32> {
    /// Learning rate `η` of the weight update `W ← W − η·d·e`.
    pub learning_rate: T,
    /// Trace decay `γ` ∈ `[0, 1)`: `e(t) = γ·e(t−1) + κ·x(t)`.
    pub trace_decay: T,
    /// Source of learning signals for output neurons.
    pub readout_mode: ReadoutMode,
    /// Feedback scheme for hidden neurons.
    pub feedback: FeedbackScheme<T>,
    /// Independent enabling of the system (roles and/or synapse sets).
    pub scope: EpropScope,
    /// Clear synapse traces on [`EpropTrainer::begin_sample`]
    /// (sample/sequence boundaries).
    pub detach_on_sample: bool,
}

impl<T: Scalar> Default for EpropConfig<T> {
    fn default() -> Self {
        Self {
            learning_rate: T::c(0.1),
            trace_decay: T::c(0.95),
            readout_mode: ReadoutMode::LocalError,
            feedback: FeedbackScheme::RandomFixed {
                seed: 0x5EED_0C0FFEE_u64,
                magnitude: T::c(0.05),
            },
            scope: EpropScope::default(),
            detach_on_sample: true,
        }
    }
}

impl<T: Scalar> EpropConfig<T> {
    /// Validate the configuration values.
    pub fn validate(&self) -> StructuralResult<()> {
        if !self.learning_rate.is_finite() || self.learning_rate < T::ZERO {
            return Err(StructuralError::param(
                "learning_rate",
                "must be finite and >= 0",
            ));
        }
        if !self.trace_decay.is_finite() || self.trace_decay < T::ZERO || self.trace_decay >= T::ONE
        {
            return Err(StructuralError::param(
                "trace_decay",
                "must be finite and in [0, 1)",
            ));
        }
        if let FeedbackScheme::RandomFixed { magnitude, .. } = self.feedback
            && (!magnitude.is_finite() || magnitude < T::ZERO)
        {
            return Err(StructuralError::param(
                "feedback.magnitude",
                "must be finite and >= 0",
            ));
        }
        Ok(())
    }
}

// ---------------------------------------------------------------------
// State
// ---------------------------------------------------------------------

/// Online state of the plasticity system: synapse eligibility traces
/// (`O(S)`), learning signals (`O(N)`), feedback weights, targets,
/// external signals, counters. Checkpointable via [`to_bytes`](Self::to_bytes).
#[derive(Debug, Clone, Default, PartialEq)]
pub struct EpropState<T = f32> {
    /// Global trainer step counter.
    steps: u64,
    /// Number of started samples.
    samples: u64,
    /// Number of applied weight updates.
    updates: u64,
    /// Synapse eligibility traces: `e_{j→i}(t)`, `O(S)`.
    synapse_traces: BTreeMap<SynapseId, T>,
    /// Latest learning signals `d_j(t)` per neuron, `O(N)`.
    neuron_signals: BTreeMap<NeuronId, T>,
    /// Targets `y_j^*` for output neurons (readout errors).
    targets: BTreeMap<NeuronId, T>,
    /// External learning signals ([`ReadoutMode::External`]).
    external: BTreeMap<NeuronId, T>,
    /// Feedback weights: `feedback[hidden j][output k] = F[j][k]`.
    feedback: BTreeMap<NeuronId, BTreeMap<NeuronId, T>>,
}

impl<T: Scalar> EpropState<T> {
    pub(crate) fn new() -> Self {
        Self::default()
    }

    /// Number of stored synapse traces (`O(S)`-bounded memory).
    pub fn synapse_trace_count(&self) -> usize {
        self.synapse_traces.len()
    }

    /// Number of stored learning signals (`O(N)`).
    pub fn neuron_signal_count(&self) -> usize {
        self.neuron_signals.len()
    }

    /// Trainer steps (`net.step()` + `trainer.step()` calls).
    pub fn steps(&self) -> u64 {
        self.steps
    }

    /// Number of started samples.
    pub fn samples(&self) -> u64 {
        self.samples
    }

    /// Number of applied weight updates.
    pub fn updates(&self) -> u64 {
        self.updates
    }

    /// Synapse eligibility trace `e_{j→i}(t)`, if stored (nonzero).
    pub fn synapse_trace(&self, id: SynapseId) -> Option<T> {
        self.synapse_traces.get(&id).copied()
    }

    /// Latest learning signal `d_j(t)` of a neuron (if any).
    pub fn neuron_signal(&self, id: NeuronId) -> Option<T> {
        self.neuron_signals.get(&id).copied()
    }

    /// Targets of output neurons.
    pub fn targets(&self) -> &BTreeMap<NeuronId, T> {
        &self.targets
    }

    /// External learning signals ([`ReadoutMode::External`]).
    pub fn external_signals(&self) -> &BTreeMap<NeuronId, T> {
        &self.external
    }

    /// Feedback weights: `feedback[hidden j][output k] = F[j][k]`.
    pub fn feedback(&self) -> &BTreeMap<NeuronId, BTreeMap<NeuronId, T>> {
        &self.feedback
    }

    /// Encode the state for checkpointing (`save`/`load` around `DynamicNetwork::save_checkpoint`).
    pub fn to_bytes(&self) -> StructuralResult<Vec<u8>> {
        let mut w = Enc { buf: Vec::new() };
        w.buf.extend_from_slice(MAGIC);
        w.u32(VERSION);
        w.u64(self.steps);
        w.u64(self.samples);
        w.u64(self.updates);
        w.u64(self.synapse_traces.len() as u64);
        for (id, v) in &self.synapse_traces {
            w.syn(id.0);
            w.f64(v.to_f64());
        }
        w.u64(self.neuron_signals.len() as u64);
        for (id, v) in &self.neuron_signals {
            w.nrn(id.0);
            w.f64(v.to_f64());
        }
        w.u64(self.feedback.len() as u64);
        for (h, inner) in &self.feedback {
            w.nrn(h.0);
            w.u64(inner.len() as u64);
            for (k, v) in inner {
                w.nrn(k.0);
                w.f64(v.to_f64());
            }
        }
        w.u64(self.targets.len() as u64);
        for (id, v) in &self.targets {
            w.nrn(id.0);
            w.f64(v.to_f64());
        }
        w.u64(self.external.len() as u64);
        for (id, v) in &self.external {
            w.nrn(id.0);
            w.f64(v.to_f64());
        }
        Ok(w.buf)
    }

    #[allow(clippy::field_reassign_with_default)]
    /// Decode a state written by [`to_bytes`](Self::to_bytes).
    pub fn from_bytes(bytes: &[u8]) -> StructuralResult<Self> {
        let mut r = Dec { b: bytes, p: 0 };
        let magic: [u8; 4] = r.take(4)?.try_into().unwrap();
        if magic != MAGIC {
            return Err(StructuralError::param(
                "eprop",
                "not an eprop state blob (bad magic)",
            ));
        }
        let version = r.u32()?;
        if version != VERSION {
            return Err(StructuralError::param(
                "eprop",
                format!("state version {version} is not supported"),
            ));
        }
        let mut s = Self::default();
        s.steps = r.u64()?;
        s.samples = r.u64()?;
        s.updates = r.u64()?;
        let n = r.u64()?;
        for _ in 0..n {
            let id = SynapseId(r.u64()?);
            let v = T::from_f64(r.f64()?);
            if v.is_finite() {
                s.synapse_traces.insert(id, v);
            }
        }
        let n = r.u64()?;
        for _ in 0..n {
            let id = NeuronId(r.u64()?);
            let v = T::from_f64(r.f64()?);
            if v.is_finite() {
                s.neuron_signals.insert(id, v);
            }
        }
        let n = r.u64()?;
        for _ in 0..n {
            let h = NeuronId(r.u64()?);
            let m = r.u64()?;
            let mut inner = BTreeMap::new();
            for _ in 0..m {
                let k = NeuronId(r.u64()?);
                let v = T::from_f64(r.f64()?);
                if v.is_finite() {
                    inner.insert(k, v);
                }
            }
            s.feedback.insert(h, inner);
        }
        let n = r.u64()?;
        for _ in 0..n {
            let id = NeuronId(r.u64()?);
            let v = T::from_f64(r.f64()?);
            if v.is_finite() {
                s.targets.insert(id, v);
            }
        }
        let n = r.u64()?;
        for _ in 0..n {
            let id = NeuronId(r.u64()?);
            let v = T::from_f64(r.f64()?);
            if v.is_finite() {
                s.external.insert(id, v);
            }
        }
        Ok(s)
    }
}

const MAGIC: &[u8] = b"ELST";
const VERSION: u32 = 1;

struct Enc {
    buf: Vec<u8>,
}
impl Enc {
    fn u32(&mut self, v: u32) {
        self.buf.extend_from_slice(&v.to_le_bytes());
    }
    fn u64(&mut self, v: u64) {
        self.buf.extend_from_slice(&v.to_le_bytes());
    }
    fn f64(&mut self, v: f64) {
        self.buf.extend_from_slice(&v.to_le_bytes());
    }
    fn nrn(&mut self, id: u64) {
        self.u64(id);
    }
    fn syn(&mut self, id: u64) {
        self.u64(id);
    }
}

struct Dec<'a> {
    b: &'a [u8],
    p: usize,
}
impl Dec<'_> {
    fn take(&mut self, n: usize) -> StructuralResult<&[u8]> {
        if self.p + n > self.b.len() {
            return Err(StructuralError::param("eprop", "state bytes truncated"));
        }
        let s = &self.b[self.p..self.p + n];
        self.p += n;
        Ok(s)
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
}

// ---------------------------------------------------------------------
// Feedback matrix (built once at `EpropTrainer::new`)
// ---------------------------------------------------------------------

fn build_feedback<T: Scalar>(
    scheme: &FeedbackScheme<T>,
    net: &DynamicNetwork<T>,
) -> BTreeMap<NeuronId, BTreeMap<NeuronId, T>> {
    let mut out: BTreeMap<NeuronId, BTreeMap<NeuronId, T>> = BTreeMap::new();
    match scheme {
        FeedbackScheme::None => {}
        FeedbackScheme::Symmetric => {
            for h in net.topology().hidden() {
                for k in net.topology().outputs() {
                    if let Some(sid) = net.topology().find_synapse(h, k)
                        && let Some(s) = net.topology().synapse(sid)
                    {
                        out.entry(h).or_default().insert(k, s.weight);
                    }
                }
            }
        }
        FeedbackScheme::RandomFixed { seed, magnitude } => {
            let mut rng = Rng64::new(*seed);
            for h in net.topology().hidden() {
                for k in net.topology().outputs() {
                    let f = *magnitude * rng.symmetric_as::<T>();
                    out.entry(h).or_default().insert(k, f);
                }
            }
        }
    }
    out
}

// ---------------------------------------------------------------------
// Trainer
// ---------------------------------------------------------------------

/// Result of [`EpropTrainer::step`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct EpropReport {
    /// Number of neurons with learning signals (readout errors / feedback).
    pub learning_signals: usize,
    /// Number of synapse traces updated this step.
    pub traces_updated: usize,
    /// Number of synapses whose weight was changed this step.
    pub weights_updated: usize,
}

/// Online plasticity trainer: holds [`EpropConfig`], [`EpropState`]
/// (eligibility traces, signals, feedback, targets, external signals,
/// counters) and drives weight updates after each `net.step()` call.
///
/// Usage:
///
/// ```ignore
/// let mut trainer = EpropTrainer::new(&net, EpropConfig::default())?;
/// for sample in samples {
///     net.step(&inputs)?;
///     trainer.set_target(out, target);
///     trainer.step(&mut net)?;
/// }
/// ```
///
/// Capability interface: `EpropTrainer::new` rejects networks whose models
/// do not declare [`OnlineEpropFamily`] with
/// [`StructuralError::UnsupportedEpropDynamics`](super::error::StructuralError::UnsupportedEpropDynamics).
#[derive(Debug, Clone, PartialEq)]
pub struct EpropTrainer<T = f32> {
    config: EpropConfig<T>,
    state: EpropState<T>,
    feedback_weights: BTreeMap<NeuronId, BTreeMap<NeuronId, T>>,
}

impl<T: Scalar> EpropTrainer<T> {
    /// Create a trainer for the network: validates the config and the
    /// dynamics capability of every neuron in the network.
    pub fn new(net: &DynamicNetwork<T>, config: EpropConfig<T>) -> StructuralResult<Self> {
        config.validate()?;
        let mut t = Self {
            config,
            state: EpropState::new(),
            feedback_weights: BTreeMap::new(),
        };
        t.feedback_weights = build_feedback(&t.config.feedback, net);
        t.check_dynamics(net)?;
        Ok(t)
    }

    /// Configuration.
    pub fn config(&self) -> &EpropConfig<T> {
        &self.config
    }

    /// Online state (eligibility traces, signals, feedback, counters).
    pub fn state(&self) -> &EpropState<T> {
        &self.state
    }

    /// Set the target `y_j^*` of an output neuron (readout error mode).
    /// Default target: 0.
    pub fn set_target(&mut self, id: NeuronId, value: T) {
        self.state.targets.insert(id, value);
    }

    /// Set an external learning signal of a neuron (`ReadoutMode::External`).
    pub fn set_external_signal(&mut self, id: NeuronId, value: T) {
        self.state.external.insert(id, value);
    }

    /// Targets of output neurons.
    pub fn targets(&self) -> &BTreeMap<NeuronId, T> {
        &self.state.targets
    }

    /// External learning signals.
    pub fn external_signals(&self) -> &BTreeMap<NeuronId, T> {
        &self.state.external
    }

    /// Feedback weights (hidden `j` -> output `k`).
    pub fn feedback_weights(&self) -> &BTreeMap<NeuronId, BTreeMap<NeuronId, T>> {
        &self.feedback_weights
    }

    /// Is the synapse enabled by the scope ([`EpropScope`])?
    pub fn is_enabled_synapse(&self, net: &DynamicNetwork<T>, id: SynapseId) -> bool {
        let Some(s) = net.topology().synapse(id) else {
            return false;
        };
        let scope = &self.config.scope;
        if scope.disabled_synapses.contains(&id) {
            return false;
        }
        if let Some(only) = &scope.only_synapses {
            return only.contains(&id);
        }
        let src_role = net
            .topology()
            .neuron(s.src)
            .map_or(NeuronRole::Input, |n| n.role);
        let dst_role = net
            .topology()
            .neuron(s.dst)
            .map_or(NeuronRole::Input, |n| n.role);
        scope.sources.contains(&src_role) && scope.targets.contains(&dst_role)
    }

    /// The online step: reads the spikes of the **just completed** `net.step()`,
    /// computes learning signals, updates eligibility traces and applies weight
    /// updates (projected through [`Topology::set_synapse_weight`](super::topology::Topology::set_synapse_weight)).
    ///
    /// Call as `net.step(&inputs)?; trainer.step(&mut net)?;`.
    pub fn step(&mut self, net: &mut DynamicNetwork<T>) -> StructuralResult<EpropReport> {
        self.check_dynamics(net)?;
        let order: Vec<NeuronId> = net.topology().neurons().map(|n| n.id).collect();
        let spikes = net.last_spikes();
        if spikes.len() != order.len() {
            return Err(StructuralError::param(
                "eprop",
                "the network state is stale: call net.step() (after structural changes)",
            ));
        }
        let signals = self.compute_signals(net, &order, spikes);
        let traces_updated = self.update_traces(net, &order, spikes);
        let weights_updated = self.update_weights(net, &signals)?;
        self.state.steps += 1;
        self.state.updates += weights_updated as u64;
        self.state.neuron_signals = signals;
        Ok(EpropReport {
            learning_signals: self.state.neuron_signals.len(),
            traces_updated,
            weights_updated,
        })
    }

    /// Clear synapse eligibility traces ("detach" between sequences).
    pub fn reset_sequence(&mut self) {
        self.state.synapse_traces.clear();
    }

    /// Start a new sample (episode): `net.begin_sample()` + counter +1 +
    /// `reset_sequence()` when [`EpropConfig::detach_on_sample`].
    pub fn begin_sample(&mut self, net: &mut DynamicNetwork<T>) {
        net.begin_sample();
        self.state.samples += 1;
        if self.config.detach_on_sample {
            self.reset_sequence();
        }
    }

    /// Save the trainer state (eligibility traces, signals, feedback,
    /// targets, external signals, counters) for checkpoint-resume.
    pub fn save_state(&self) -> StructuralResult<Vec<u8>> {
        self.state.to_bytes()
    }

    /// Restore the trainer state.
    pub fn load_state(&mut self, bytes: &[u8]) -> StructuralResult<()> {
        self.state = EpropState::from_bytes(bytes)?;
        Ok(())
    }

    /// Checkpoint-resume helper: serialize and restore the state.
    pub fn clone_state(&self) -> EpropState<T> {
        self.state.clone()
    }

    // -- internals -----------------------------------------------------

    fn check_dynamics(&self, net: &DynamicNetwork<T>) -> StructuralResult<()> {
        for n in net.topology().neurons() {
            if n.model.online_eprop_family().is_none() {
                return Err(StructuralError::UnsupportedEpropDynamics(
                    n.model.as_str().to_string(),
                ));
            }
        }
        Ok(())
    }

    /// Learning signals `d_j(t)` for all neurons (empty for input neurons).
    fn compute_signals(
        &self,
        net: &DynamicNetwork<T>,
        order: &[NeuronId],
        spikes: &[bool],
    ) -> BTreeMap<NeuronId, T> {
        let mut signals: BTreeMap<NeuronId, T> = BTreeMap::new();
        let pos = |id: NeuronId| order.iter().position(|x| *x == id).unwrap_or(usize::MAX);
        match self.config.readout_mode {
            ReadoutMode::LocalError => {
                for o in net.topology().outputs() {
                    let idx = pos(o);
                    if idx == usize::MAX {
                        continue;
                    }
                    let d = flag::<T>(spikes[idx])
                        - self.state.targets.get(&o).copied().unwrap_or(T::ZERO);
                    if d != T::ZERO {
                        signals.insert(o, d);
                    }
                }
                for h in net.topology().hidden() {
                    let mut d = T::ZERO;
                    if let Some(fb) = self.state.feedback.get(&h) {
                        for (k, f) in fb {
                            if let Some(&dk) = signals.get(k) {
                                d += *f * dk;
                            }
                        }
                    }
                    if d != T::ZERO {
                        signals.insert(h, d);
                    }
                }
            }
            ReadoutMode::External => {
                for n in net.topology().neurons() {
                    if let Some(&v) = self.state.external.get(&n.id) {
                        signals.insert(n.id, v);
                    }
                }
            }
        }
        signals
    }

    /// Update synapse eligibility traces:
    /// `e_{j→i}(t) = γ·e_{j→i}(t−1) + ŷ_j(t)·x_i(t)`.
    fn update_traces(
        &mut self,
        net: &DynamicNetwork<T>,
        order: &[NeuronId],
        spikes: &[bool],
    ) -> usize {
        let gamma = self.config.trace_decay;
        let arrivals: BTreeSet<SynapseId> = net.last_arrivals().iter().copied().collect();
        let mut updated = 0;
        for s in net.topology().synapses() {
            if !self.is_enabled_synapse(net, s.id) {
                continue;
            }
            let Some(dst_idx) = order.iter().position(|x| *x == s.dst) else {
                continue;
            };
            // κ_j(t): eligibility factor of the postsynaptic neuron (spike).
            let kappa = flag::<T>(spikes[dst_idx]);
            // x_i(t): delivery of the synapse this step.
            let x = flag(arrivals.contains(&s.id));
            let prev = self
                .state
                .synapse_traces
                .get(&s.id)
                .copied()
                .unwrap_or(T::ZERO);
            let e = gamma * prev + kappa * x;
            if e != T::ZERO {
                self.state.synapse_traces.insert(s.id, e);
                updated += 1;
            } else {
                self.state.synapse_traces.remove(&s.id);
            }
        }
        updated
    }

    /// Apply the weight update: `W_{j→i} ← W_{j→i} − η·d_j(t)·e_{j→i}(t)`,
    /// then project through `Topology::set_synapse_weight`
    /// (`ParamLimits::weight_abs_max` clamp + Dale sign clamp).
    fn update_weights(
        &mut self,
        net: &mut DynamicNetwork<T>,
        signals: &BTreeMap<NeuronId, T>,
    ) -> StructuralResult<usize> {
        let lr = self.config.learning_rate;
        let mut plan: Vec<(SynapseId, T)> = Vec::new();
        for s in net.topology().synapses() {
            if !self.is_enabled_synapse(net, s.id) {
                continue;
            }
            let Some(&e) = self.state.synapse_traces.get(&s.id) else {
                continue;
            };
            if e == T::ZERO {
                continue;
            }
            let Some(&d) = signals.get(&s.dst) else {
                continue;
            };
            if d == T::ZERO {
                continue;
            }
            let new_w = s.weight - lr * d * e;
            if new_w != s.weight {
                plan.push((s.id, new_w));
            }
        }
        let mut applied = 0;
        for (id, w) in plan {
            net.topology_mut().set_synapse_weight(id, w)?;
            applied += 1;
        }
        Ok(applied)
    }
}
