//! Dynamic network graph: stable identifiers, neurons, synapses and
//! checking invariants.
//!
//! # Performance
//!
//! Neurons and synapses lie in `BTreeMap` along stable `NeuronId`/`SynapseId`
//! (identifiers are issued monotonously and **never reused**), plus
//! “outgoing/incoming” adjacency lists for a neuron. The choice is justified as follows:
//! adding and deleting - `O(log n)` without rebuilding the rest of the network;
//! iteration is deterministic (important for reproducibility); spike delivery
//! goes according to `out`-lists, that is, it is proportional to the number of spikes, and not to the size
//! networks. Cost - cache misses compared to a dense array; for dense
//! For quick paths, see the optimization step in the module documentation.
//!
//! # Transactions
//!
//! [`Topology::transaction`] performs a compound change on the copy and
//! replaces the topology only if the closure returned `Ok` **and** the resulting graph
//! passed [`Topology::validate`]. Otherwise, the original topology does not change.

use std::collections::BTreeMap;
use std::fmt;
use std::sync::atomic::{AtomicU64, Ordering};

use crate::scalar::Scalar;

use super::error::{StructuralError, StructuralResult};
pub use super::model::{NeuronModel, NeuronPolarity, NeuronRole, NeuronState};
use super::network::SynapseModel;

static VERSION_COUNTER: AtomicU64 = AtomicU64::new(1);

/// Globally unique "version" of structure: changes with each structural
/// change and is never repeated, so the executor cache built by
/// version cannot be mistakenly matched after a rollback to the snapshot.
pub(crate) fn fresh_version() -> u64 {
    VERSION_COUNTER.fetch_add(1, Ordering::Relaxed)
}

/// Stable neuron identifier.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct NeuronId(pub u64);

/// Stable synapse ID.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub struct SynapseId(pub u64);

impl fmt::Display for NeuronId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "N{}", self.0)
    }
}

impl fmt::Display for SynapseId {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "S{}", self.0)
    }
}

/// Where does the structure element come from?
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Origin {
    /// Part of the original network.
    Initial,
    /// Created by automatic growth.
    Grown,
    /// Neuron clone.
    Clone(NeuronId),
    /// Created by the division of a synapse.
    Split(SynapseId),
    /// Created manually via API.
    Manual,
}

impl Origin {
    pub(crate) fn encode(self) -> (u8, u64) {
        match self {
            Origin::Initial => (0, 0),
            Origin::Grown => (1, 0),
            Origin::Clone(n) => (2, n.0),
            Origin::Split(s) => (3, s.0),
            Origin::Manual => (4, 0),
        }
    }

    pub(crate) fn decode(tag: u8, v: u64) -> Option<Self> {
        match tag {
            0 => Some(Origin::Initial),
            1 => Some(Origin::Grown),
            2 => Some(Origin::Clone(NeuronId(v))),
            3 => Some(Origin::Split(SynapseId(v))),
            4 => Some(Origin::Manual),
            _ => None,
        }
    }
}

/// Moment of creation (step and number of the training sample) - needed to estimate age.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct Birth {
    /// Global simulation step.
    pub step: u64,
    /// Training sample (episode) number.
    pub sample: u64,
}

/// Physical and numerical limitations of parameters.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ParamLimits<T = f32> {
    /// Minimum β.
    pub beta_min: T,
    /// Maximum β (strictly less than 1, otherwise the membrane does not attenuate).
    pub beta_max: T,
    /// Minimum threshold (> 0).
    pub threshold_min: T,
    /// Maximum threshold.
    pub threshold_max: T,
    /// Maximum `|w|` synapse.
    pub weight_abs_max: T,
    /// Maximum delay in steps.
    pub delay_max: u16,
}

impl<T: Scalar> Default for ParamLimits<T> {
    fn default() -> Self {
        Self {
            beta_min: T::ZERO,
            beta_max: T::c(0.999),
            threshold_min: T::c(0.05),
            threshold_max: T::c(10.0),
            weight_abs_max: T::c(5.0),
            delay_max: 8,
        }
    }
}

/// Graph restrictions.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TopologyLimits {
    /// Maximum neurons.
    pub max_neurons: usize,
    /// Maximum synapses.
    pub max_synapses: usize,
    /// Maximum incoming synapses per neuron.
    pub max_in_degree: usize,
    /// Maximum outgoing synapses in a neuron.
    pub max_out_degree: usize,
    /// Are `n → n` loops allowed.
    pub allow_self_loops: bool,
    /// Are loops allowed (recurrence). For `false` the graph must be a DAG.
    pub allow_recurrent: bool,
    /// Are parallel synapses allowed between one pair of neurons (they
    /// differ in delay).
    pub allow_duplicates: bool,
    /// Dale's principle as a mandatory invariant: the sign of the synapse weight is determined
    /// by the polarity of the source (excitatory and modulating - `w ≥ 0`, inhibitory -
    /// `w ≤ 0`). New and redirected links receive the source sign,
    /// weight changes cannot disturb it, a change in the polarity of the neuron
    /// reverses the signs of its outgoing weights, and `validate` rejects
    /// violations. At `false` the signs are free, and the inhibitory neuron reverses
    /// the positive weight sign.
    pub enforce_dale: bool,
}

impl Default for TopologyLimits {
    fn default() -> Self {
        Self {
            max_neurons: 1024,
            max_synapses: 16_384,
            max_in_degree: 256,
            max_out_degree: 256,
            allow_self_loops: false,
            allow_recurrent: true,
            allow_duplicates: false,
            enforce_dale: true,
        }
    }
}

/// Neuron activity statistics.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct NeuronStats<T = f32> {
    /// Number of spikes.
    pub spikes: u64,
    /// Number of observation steps.
    pub steps: u64,
    /// Moving average of spike frequency (spikes per step).
    pub rate_ema: T,
    /// Last spike step.
    pub last_spike_step: Option<u64>,
    /// Exponential spike trace (like `update_trace` to STDP).
    pub trace: T,
    /// In how many different patterns did the neuron splice?
    pub active_samples: u64,
    /// The number of the sample in which the activity has already been counted (+1, 0 = no).
    pub last_marked_sample: u64,
    /// Moving average of the driving input current per step (`≥ 0`).
    pub exc_drive_ema: T,
    /// Running average of the inhibitory input current per step (magnitude, `≥ 0`).
    pub inh_drive_ema: T,
}

/// Structural plasticity metadata.
#[derive(Debug, Clone, PartialEq)]
pub struct StructMeta {
    /// Creation step.
    pub born_step: u64,
    /// Sample creation.
    pub born_sample: u64,
    /// Origin.
    pub origin: Origin,
    /// Is it protected from automatic deletion?
    pub protected: bool,
    /// Last utility evaluation step.
    pub last_evaluated_step: u64,
}

impl StructMeta {
    pub(crate) fn new(birth: Birth, origin: Origin) -> Self {
        Self {
            born_step: birth.step,
            born_sample: birth.sample,
            origin,
            protected: false,
            last_evaluated_step: birth.step,
        }
    }
}

/// Neuron.
///
/// Role, dynamic model and polarity are three independent concepts:
/// the role is responsible for the place in the network, the model is responsible for the law of state changes,
/// polarity - as a sign of the influence of spikes on the receivers. State always
/// corresponds to the model variant (checked [`Topology::validate`]).
#[derive(Debug, Clone, PartialEq)]
pub struct Neuron<T: Scalar = f32> {
    /// ID.
    pub id: NeuronId,
    /// Role in the network.
    pub role: NeuronRole,
    /// Physiological polarity.
    pub polarity: NeuronPolarity,
    /// Dynamics model (also used by input neurons in
    /// [`InputMode::Current`](super::network::InputMode::Current)).
    pub model: NeuronModel<T>,
    /// Model condition.
    pub state: NeuronState<T>,
    /// Damped modulation coefficient accumulated from modulating
    /// sources. The input current is multiplied by `1 + modulation`.
    pub modulation: T,
    /// Activity statistics.
    pub stats: NeuronStats<T>,
    /// Structural plasticity metadata.
    pub meta: StructMeta,
}

impl<T: Scalar> Neuron<T> {
    /// The role of the neuron (formerly `kind`).
    pub fn kind(&self) -> NeuronRole {
        self.role
    }

    /// Membrane potential (for all models this is `mem` or `v`).
    pub fn mem(&self) -> T {
        self.state.mem()
    }

    /// Record the membrane potential.
    pub fn set_mem(&mut self, value: T) {
        self.state.set_mem(value);
    }

    /// Current model threshold.
    pub fn threshold(&self) -> T {
        self.model.threshold()
    }
}

/// State of the synapse.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SynapseState {
    /// Transmits spikes.
    Active,
    /// Disabled (first stage of pruning); physically still exists.
    Disabled {
        /// Shutdown step.
        since_step: u64,
    },
}

/// Synapse usage and usefulness statistics.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct SynapseStats<T = f32> {
    /// How many spikes were delivered.
    pub uses: u64,
    /// Total absolute transmission `Σ|w|` across delivered spikes.
    pub drive_sum: T,
    /// In how many different patterns did the synapse transmit spikes?
    pub active_samples: u64,
    /// A pattern in which the activity is counted (+1, 0 = not).
    pub last_marked_sample: u64,
    /// Rolling contribution estimate: average `signal × transmission trace`
    /// (see `DynamicNetwork::assign_credit`).
    pub credit_ema: T,
    /// Spike transmission trace (fade out lazily).
    pub activity_trace: T,
    /// The step for which `activity_trace` is calculated.
    pub trace_step: u64,
}

/// Synapse.
#[derive(Debug, Clone, PartialEq)]
pub struct Synapse<T = f32> {
    /// ID.
    pub id: SynapseId,
    /// Source.
    pub src: NeuronId,
    /// Receiver.
    pub dst: NeuronId,
    /// Weight.
    pub weight: T,
    /// Delivery delay in steps (0 - spike arrives at the next step).
    pub delay: u16,
    /// Transmission model ([`SynapseModel`]) — legacy `CurrentBased` = weight-as-current.
    pub model: SynapseModel<T>,
    /// Binary weight flag — the weight is `0` or `±W_max`
    /// (paper-style binary weights, see arXiv:2610.16022).
    /// Storage/semantics only: delivery is unchanged — the weight itself is the value.
    pub binary: bool,
    /// Condition.
    pub state: SynapseState,
    /// Statistics.
    pub stats: SynapseStats<T>,
    /// Eligibility trace for reward-modulated plasticity.
    pub elig: T,
    /// The step for which `elig` is calculated.
    pub elig_step: u64,
    /// Metadata.
    pub meta: StructMeta,
}

impl<T: Scalar> Synapse<T> {
    /// Does the synapse transmit spikes?
    pub fn is_active(&self) -> bool {
        matches!(self.state, SynapseState::Active)
    }
}

/// Dynamic network graph.
#[derive(Debug, Clone)]
pub struct Topology<T: Scalar = f32> {
    pub(super) neurons: BTreeMap<NeuronId, Neuron<T>>,
    pub(super) synapses: BTreeMap<SynapseId, Synapse<T>>,
    pub(super) out_adj: BTreeMap<NeuronId, Vec<SynapseId>>,
    pub(super) in_adj: BTreeMap<NeuronId, Vec<SynapseId>>,
    pub(super) next_neuron: u64,
    pub(super) next_synapse: u64,
    pub(super) limits: TopologyLimits,
    pub(super) param_limits: ParamLimits<T>,
    pub(super) version: u64,
}

impl<T: Scalar> PartialEq for Topology<T> {
    fn eq(&self, o: &Self) -> bool {
        self.neurons == o.neurons
            && self.synapses == o.synapses
            && self.next_neuron == o.next_neuron
            && self.next_synapse == o.next_synapse
            && self.limits == o.limits
            && self.param_limits == o.param_limits
    }
}

const EMPTY: &[SynapseId] = &[];

impl<T: Scalar> Topology<T> {
    /// Empty topology with specified restrictions.
    pub fn new(limits: TopologyLimits, param_limits: ParamLimits<T>) -> Self {
        Self {
            neurons: BTreeMap::new(),
            synapses: BTreeMap::new(),
            out_adj: BTreeMap::new(),
            in_adj: BTreeMap::new(),
            next_neuron: 0,
            next_synapse: 0,
            limits,
            param_limits,
            version: fresh_version(),
        }
    }

    /// Graph restrictions.
    pub fn limits(&self) -> &TopologyLimits {
        &self.limits
    }

    /// Parameter restrictions.
    pub fn param_limits(&self) -> &ParamLimits<T> {
        &self.param_limits
    }

    /// Does Dale's principle apply (see [`TopologyLimits::enforce_dale`]).
    pub fn dale(&self) -> bool {
        self.limits.enforce_dale
    }

    /// Turn Dale's principle on or off. When turned on, the signs of all
    /// existing weights are reduced to the polarity of the source (the magnitude
    /// is preserved); the number of changed synapses is returned. Thus a legacy network
    /// with free signs is transferred to the E/I network.
    pub fn set_enforce_dale(&mut self, enforce: bool) -> usize {
        self.limits.enforce_dale = enforce;
        if !enforce {
            return 0;
        }
        let mut changed = 0;
        let pols: BTreeMap<NeuronId, NeuronPolarity> =
            self.neurons.iter().map(|(k, n)| (*k, n.polarity)).collect();
        for s in self.synapses.values_mut() {
            let w = pols[&s.src].resign(s.weight);
            if w != s.weight {
                s.weight = w;
                changed += 1;
            }
        }
        if changed > 0 {
            self.touch();
        }
        changed
    }

    /// Is it possible to separate a synapse with a relay neuron without changing the meaning of the connection?
    ///
    /// Relay (`src → relay → dst`) transmits a spike only from an excitatory
    /// source. With Dale's principle, the inhibitory source gives a negative
    /// current on the relay, which does not activate it, and the modulatory one
    /// creates no current, so such synapses do not divide (neither
    /// [`split_synapse`](Self::split_synapse), nor growth, nor NEAT `add_node`).
    /// Without Dale's principle there is no limitation.
    pub fn can_split_synapse(&self, id: SynapseId) -> bool {
        let Some(s) = self.synapses.get(&id) else {
            return false;
        };
        s.is_active()
            && s.src != s.dst
            && (!self.limits.enforce_dale || self.polarity_of(s.src) == NeuronPolarity::Excitatory)
    }

    /// Polarity of the source neuron (excitatory if there is no neuron).
    pub(super) fn polarity_of(&self, id: NeuronId) -> NeuronPolarity {
        self.neurons
            .get(&id)
            .map_or(NeuronPolarity::Excitatory, |n| n.polarity)
    }

    /// The weight of the new connection: with the Dale principle receives the sign of the polarity of the source.
    pub(super) fn dale_new_weight(&self, src: NeuronId, w: T) -> T {
        if self.limits.enforce_dale {
            self.polarity_of(src).resign(w)
        } else {
            w
        }
    }

    /// Weight after change: with Dale's principle it cannot pass through zero.
    pub(super) fn dale_clamp_weight(&self, src: NeuronId, w: T) -> T {
        if self.limits.enforce_dale {
            self.polarity_of(src).clamp_sign(w)
        } else {
            w
        }
    }

    /// Rearrange the signs of the neuron's outgoing weights to match its polarity.
    pub(super) fn resign_outgoing(&mut self, src: NeuronId) {
        if !self.limits.enforce_dale {
            return;
        }
        let pol = self.polarity_of(src);
        let ids: Vec<SynapseId> = self.out_adj.get(&src).cloned().unwrap_or_default();
        for id in ids {
            if let Some(s) = self.synapses.get_mut(&id) {
                s.weight = pol.resign(s.weight);
            }
        }
    }

    /// Unique version of the structure (changes with each structural change).
    pub fn version(&self) -> u64 {
        self.version
    }

    pub(super) fn touch(&mut self) {
        self.version = fresh_version();
    }

    /// Neuron by id.
    pub fn neuron(&self, id: NeuronId) -> Option<&Neuron<T>> {
        self.neurons.get(&id)
    }

    /// Synapse by id.
    pub fn synapse(&self, id: SynapseId) -> Option<&Synapse<T>> {
        self.synapses.get(&id)
    }

    pub(crate) fn synapse_mut(&mut self, id: SynapseId) -> Option<&mut Synapse<T>> {
        self.synapses.get_mut(&id)
    }

    /// All neurons are in order id.
    pub fn neurons(&self) -> impl Iterator<Item = &Neuron<T>> {
        self.neurons.values()
    }

    /// All synapses are ok id.
    pub fn synapses(&self) -> impl Iterator<Item = &Synapse<T>> {
        self.synapses.values()
    }

    /// Identifiers of neurons of a given role, in ascending order.
    pub fn ids_of_kind(&self, kind: NeuronRole) -> Vec<NeuronId> {
        self.neurons
            .values()
            .filter(|n| n.role == kind)
            .map(|n| n.id)
            .collect()
    }

    /// Identifiers of neurons of a given role (synonym [`ids_of_kind`](Self::ids_of_kind)).
    pub fn ids_of_role(&self, role: NeuronRole) -> Vec<NeuronId> {
        self.ids_of_kind(role)
    }

    /// Input neurons.
    pub fn inputs(&self) -> Vec<NeuronId> {
        self.ids_of_kind(NeuronRole::Input)
    }

    /// Output neurons.
    pub fn outputs(&self) -> Vec<NeuronId> {
        self.ids_of_kind(NeuronRole::Output)
    }

    /// Hidden neurons.
    pub fn hidden(&self) -> Vec<NeuronId> {
        self.ids_of_kind(NeuronRole::Hidden)
    }

    /// Number of neurons.
    pub fn neuron_count(&self) -> usize {
        self.neurons.len()
    }

    /// Number of synapses (including disabled ones).
    pub fn synapse_count(&self) -> usize {
        self.synapses.len()
    }

    /// Number of active synapses.
    pub fn active_synapse_count(&self) -> usize {
        self.synapses.values().filter(|s| s.is_active()).count()
    }

    /// Outgoing synapses of a neuron.
    pub fn out_synapses(&self, id: NeuronId) -> &[SynapseId] {
        self.out_adj.get(&id).map_or(EMPTY, |v| v.as_slice())
    }

    /// Incoming synapses of a neuron.
    pub fn in_synapses(&self, id: NeuronId) -> &[SynapseId] {
        self.in_adj.get(&id).map_or(EMPTY, |v| v.as_slice())
    }

    /// Is there at least one synapse `src → dst`.
    pub fn connected(&self, src: NeuronId, dst: NeuronId) -> bool {
        self.find_synapse(src, dst).is_some()
    }

    /// The first synapse is `src → dst`.
    pub fn find_synapse(&self, src: NeuronId, dst: NeuronId) -> Option<SynapseId> {
        self.out_synapses(src)
            .iter()
            .copied()
            .find(|sid| self.synapses.get(sid).is_some_and(|s| s.dst == dst))
    }

    /// Is `to` reachable from `from` via synapses (ignoring `skip`).
    pub(super) fn reaches(&self, from: NeuronId, to: NeuronId, skip: Option<SynapseId>) -> bool {
        let mut stack = vec![from];
        let mut seen = std::collections::BTreeSet::new();
        while let Some(n) = stack.pop() {
            if n == to {
                return true;
            }
            if !seen.insert(n) {
                continue;
            }
            for sid in self.out_synapses(n) {
                if Some(*sid) == skip {
                    continue;
                }
                if let Some(s) = self.synapses.get(sid) {
                    stack.push(s.dst);
                }
            }
        }
        false
    }

    // ---- low-level inserts/deletions (no constraint checks) ----

    pub(super) fn raw_insert_neuron(
        &mut self,
        role: NeuronRole,
        polarity: NeuronPolarity,
        model: NeuronModel<T>,
        birth: Birth,
        origin: Origin,
    ) -> NeuronId {
        let id = NeuronId(self.next_neuron);
        self.next_neuron += 1;
        self.neurons.insert(
            id,
            Neuron {
                id,
                role,
                polarity,
                state: model.init_state(),
                model,
                modulation: T::ZERO,
                stats: NeuronStats::default(),
                meta: StructMeta::new(birth, origin),
            },
        );
        self.out_adj.insert(id, Vec::new());
        self.in_adj.insert(id, Vec::new());
        self.touch();
        id
    }

    pub(super) fn raw_insert_synapse(
        &mut self,
        src: NeuronId,
        dst: NeuronId,
        weight: T,
        delay: u16,
        model: SynapseModel<T>,
        birth: Birth,
        origin: Origin,
    ) -> SynapseId {
        let id = SynapseId(self.next_synapse);
        self.next_synapse += 1;
        self.synapses.insert(
            id,
            Synapse {
                id,
                src,
                dst,
                weight,
                delay,
                model,
                binary: false,
                state: SynapseState::Active,
                stats: SynapseStats {
                    trace_step: birth.step,
                    ..Default::default()
                },
                elig: T::ZERO,
                elig_step: birth.step,
                meta: StructMeta::new(birth, origin),
            },
        );
        self.out_adj.entry(src).or_default().push(id);
        self.in_adj.entry(dst).or_default().push(id);
        self.touch();
        id
    }

    pub(super) fn raw_remove_synapse(&mut self, id: SynapseId) -> Option<Synapse<T>> {
        let s = self.synapses.remove(&id)?;
        if let Some(v) = self.out_adj.get_mut(&s.src) {
            v.retain(|x| *x != id);
        }
        if let Some(v) = self.in_adj.get_mut(&s.dst) {
            v.retain(|x| *x != id);
        }
        self.touch();
        Some(s)
    }

    /// Reassemble adjacency lists by synapses (after loading).
    pub(crate) fn rebuild_adjacency(&mut self) {
        self.out_adj = self.neurons.keys().map(|k| (*k, Vec::new())).collect();
        self.in_adj = self.neurons.keys().map(|k| (*k, Vec::new())).collect();
        for s in self.synapses.values() {
            if let Some(v) = self.out_adj.get_mut(&s.src) {
                v.push(s.id);
            }
            if let Some(v) = self.in_adj.get_mut(&s.dst) {
                v.push(s.id);
            }
        }
        self.touch();
    }

    /// Perform a composite change atomically: on closure error or
    /// If invariants are violated, the topology remains the same.
    pub fn transaction<R>(
        &mut self,
        f: impl FnOnce(&mut Topology<T>) -> StructuralResult<R>,
    ) -> StructuralResult<R> {
        let backup = self.clone();
        match f(self) {
            Ok(v) => match self.validate() {
                Ok(()) => Ok(v),
                Err(e) => {
                    *self = backup;
                    Err(e)
                }
            },
            Err(e) => {
                *self = backup;
                Err(e)
            }
        }
    }

    /// Check all invariants: uniqueness and correctness of id, absence
    /// dangling links, adjacency list consistency, admissibility
    /// parameters and compliance with network restrictions.
    pub fn validate(&self) -> StructuralResult<()> {
        let bad = |m: String| Err(StructuralError::InvariantViolation(m));
        if self.neurons.len() > self.limits.max_neurons {
            return bad(format!(
                "neurons {} > max_neurons {}",
                self.neurons.len(),
                self.limits.max_neurons
            ));
        }
        if self.synapses.len() > self.limits.max_synapses {
            return bad(format!(
                "synapses {} > max_synapses {}",
                self.synapses.len(),
                self.limits.max_synapses
            ));
        }
        for (k, n) in &self.neurons {
            if *k != n.id {
                return bad(format!("key {k} does not match neuron id {}", n.id));
            }
            if n.id.0 >= self.next_neuron {
                return bad(format!(
                    "{} is not less than counter {}",
                    n.id, self.next_neuron
                ));
            }
            n.model
                .validate(&self.param_limits)
                .map_err(|e| StructuralError::InvariantViolation(format!("{}: {e}", n.id)))?;
            if !n.state.matches(&n.model) {
                return bad(format!("{}: state does not match the model", n.id));
            }
            if !n.state.is_finite()
                || !n.modulation.is_finite()
                || !n.stats.rate_ema.is_finite()
                || !n.stats.trace.is_finite()
            {
                return bad(format!("{}: non-numeric state", n.id));
            }
            if !self.out_adj.contains_key(&n.id) || !self.in_adj.contains_key(&n.id) {
                return bad(format!("{}: no entry in adjacency lists", n.id));
            }
        }
        for (k, s) in &self.synapses {
            if *k != s.id {
                return bad(format!("key {k} does not match synapse id {}", s.id));
            }
            if s.id.0 >= self.next_synapse {
                return bad(format!(
                    "{} is not less than counter {}",
                    s.id, self.next_synapse
                ));
            }
            let (Some(_), Some(dst)) = (self.neurons.get(&s.src), self.neurons.get(&s.dst)) else {
                return bad(format!("{}: dangling link {} → {}", s.id, s.src, s.dst));
            };
            if dst.role == NeuronRole::Input {
                return bad(format!("{}: receiver {} - input", s.id, s.dst));
            }
            if !s.weight.is_finite()
                || s.weight.abs() > self.param_limits.weight_abs_max * (T::ONE + T::c(1e-6))
            {
                return bad(format!("{}: invalid weight {}", s.id, s.weight));
            }
            if self.limits.enforce_dale {
                let src = &self.neurons[&s.src];
                let ok = match src.polarity {
                    NeuronPolarity::Inhibitory => s.weight <= T::ZERO,
                    _ => s.weight >= T::ZERO,
                };
                if !ok {
                    return bad(format!(
                        "{}: Dale's principle is violated: {} neuron {} has weight {}",
                        s.id,
                        src.polarity.as_str(),
                        s.src,
                        s.weight
                    ));
                }
                // Conductance-based synapses: GABA (inhibitory source) must have E_rev ≤ 0
                // (reversal potential — never sign flip).
                if let Some(p) = s.model.params()
                    && src.polarity == NeuronPolarity::Inhibitory
                    && p.e_rev > T::ZERO
                {
                    return bad(format!(
                        "{}: Dale's principle is violated: {} neuron {} has E_rev {}",
                        s.id,
                        src.polarity.as_str(),
                        s.src,
                        p.e_rev
                    ));
                }
            }
            if s.delay > self.param_limits.delay_max {
                return bad(format!(
                    "{}: delay {} > {}",
                    s.id, s.delay, self.param_limits.delay_max
                ));
            }
            if s.src == s.dst && !self.limits.allow_self_loops {
                return bad(format!("{}: loop prohibited", s.id));
            }
            if !s.elig.is_finite() || !s.stats.credit_ema.is_finite() {
                return bad(format!("{}: non-numeric statistics", s.id));
            }
        }
        // Adjacency consistency: Each connection occurs exactly once.
        let mut out_total = 0;
        let mut in_total = 0;
        for (n, list) in &self.out_adj {
            if !self.neurons.contains_key(n) {
                return bad(format!("out_adj refers to the missing {n}"));
            }
            if list.len() > self.limits.max_out_degree {
                return bad(format!(
                    "{n}: outdegree {} > {}",
                    list.len(),
                    self.limits.max_out_degree
                ));
            }
            for sid in list {
                match self.synapses.get(sid) {
                    Some(s) if s.src == *n => {}
                    _ => {
                        return bad(format!("out_adj[{n}] contains foreign or missing {sid}"));
                    }
                }
            }
            out_total += list.len();
        }
        for (n, list) in &self.in_adj {
            if !self.neurons.contains_key(n) {
                return bad(format!("in_adj refers to the missing {n}"));
            }
            if list.len() > self.limits.max_in_degree {
                return bad(format!(
                    "{n}: incoming power {} > {}",
                    list.len(),
                    self.limits.max_in_degree
                ));
            }
            for sid in list {
                match self.synapses.get(sid) {
                    Some(s) if s.dst == *n => {}
                    _ => {
                        return bad(format!("in_adj[{n}] contains foreign or missing {sid}"));
                    }
                }
            }
            in_total += list.len();
        }
        if out_total != self.synapses.len() || in_total != self.synapses.len() {
            return bad(format!(
                "adjacency contains {out_total}/{in_total} entries at {} synapses",
                self.synapses.len()
            ));
        }
        // Duplicates.
        if !self.limits.allow_duplicates {
            let mut seen = std::collections::BTreeSet::new();
            for s in self.synapses.values() {
                // Conductance-based synapses are allowed as multiedges (each has its
                // own independent receptor dynamics).
                if s.model.is_cond() {
                    continue;
                }
                if !seen.insert((s.src, s.dst)) {
                    return bad(format!("duplicate connection {} → {}", s.src, s.dst));
                }
            }
        } else {
            let mut seen = std::collections::BTreeSet::new();
            for s in self.synapses.values() {
                if !seen.insert((s.src, s.dst, s.delay)) {
                    return bad(format!(
                        "duplicate connection {} → {} with delay {}",
                        s.src, s.dst, s.delay
                    ));
                }
            }
        }
        // Acyclicity if recurrence is turned off (Cahn algorithm).
        if !self.limits.allow_recurrent {
            let mut indeg: BTreeMap<NeuronId, usize> =
                self.neurons.keys().map(|k| (*k, 0)).collect();
            for s in self.synapses.values() {
                if let Some(d) = indeg.get_mut(&s.dst) {
                    *d += 1;
                }
            }
            let mut ready: Vec<NeuronId> = indeg
                .iter()
                .filter(|(_, d)| **d == 0)
                .map(|(k, _)| *k)
                .collect();
            let mut visited = 0;
            while let Some(n) = ready.pop() {
                visited += 1;
                for sid in self.out_synapses(n) {
                    if let Some(s) = self.synapses.get(sid)
                        && let Some(d) = indeg.get_mut(&s.dst)
                    {
                        *d -= 1;
                        if *d == 0 {
                            ready.push(s.dst);
                        }
                    }
                }
            }
            if visited != self.neurons.len() {
                return bad("cycle detected when allow_recurrent = false".into());
            }
        }
        Ok(())
    }
}
