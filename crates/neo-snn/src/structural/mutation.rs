//! Structural actions, transactions and event log.
//!
//! Each topology change is a [`StructuralAction`] value with a reason
//! [`Reason`]. The action is applied atomically ([`apply_action`]); a batch of
//! actions is applied all-or-nothing ([`apply_batch`]). The result is recorded in
//! [`EventLog`], which can be sent to telemetry/GUI and saved in checkpoint.

use std::collections::{BTreeMap, VecDeque};
use std::fmt;

use super::error::{StructuralError, StructuralResult};
use super::model::{NeuronModel, NeuronPolarity, NeuronRole, StateMigration};
use super::network::DynamicNetwork;
use super::neuron_ops::{NeuronMutation, SplitInit};
use super::rng::Rng64;
use super::synapse_ops::SynapseMutation;
use super::topology::{Birth, NeuronId, Origin, SynapseId};
use crate::scalar::Scalar;

/// Type of operation (for statistics, pauses after rollbacks and log).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum ActionKind {
    /// Creation of a synapse.
    AddSynapse,
    /// Disabling the synapse.
    DisableSynapse,
    /// Synapse removal.
    RemoveSynapse,
    /// Synapse redirection.
    RewireSynapse,
    /// Changing synapse parameters.
    MutateSynapse,
    /// Creation of a neuron.
    AddNeuron,
    /// Cloning a neuron.
    CloneNeuron,
    /// Synapse separation.
    SplitSynapse,
    /// Removing a neuron.
    RemoveNeuron,
    /// Changing neuron parameters.
    MutateNeuron,
    /// Changing the neuron model with state migration.
    ChangeNeuronModel,
}

impl ActionKind {
    /// All types of operations.
    pub const ALL: [ActionKind; 11] = [
        ActionKind::AddSynapse,
        ActionKind::DisableSynapse,
        ActionKind::RemoveSynapse,
        ActionKind::RewireSynapse,
        ActionKind::MutateSynapse,
        ActionKind::AddNeuron,
        ActionKind::CloneNeuron,
        ActionKind::SplitSynapse,
        ActionKind::RemoveNeuron,
        ActionKind::MutateNeuron,
        ActionKind::ChangeNeuronModel,
    ];

    /// Short name.
    pub fn as_str(self) -> &'static str {
        match self {
            ActionKind::AddSynapse => "add_synapse",
            ActionKind::DisableSynapse => "disable_synapse",
            ActionKind::RemoveSynapse => "remove_synapse",
            ActionKind::RewireSynapse => "rewire_synapse",
            ActionKind::MutateSynapse => "mutate_synapse",
            ActionKind::AddNeuron => "add_neuron",
            ActionKind::CloneNeuron => "clone_neuron",
            ActionKind::SplitSynapse => "split_synapse",
            ActionKind::RemoveNeuron => "remove_neuron",
            ActionKind::MutateNeuron => "mutate_neuron",
            ActionKind::ChangeNeuronModel => "change_neuron_model",
        }
    }

    pub(crate) fn code(self) -> u8 {
        Self::ALL.iter().position(|k| *k == self).unwrap_or(0) as u8
    }

    pub(crate) fn from_code(c: u8) -> Option<Self> {
        Self::ALL.get(c as usize).copied()
    }

    /// Whether the operation increases the size of the network.
    pub fn is_growth(self) -> bool {
        matches!(
            self,
            ActionKind::AddSynapse
                | ActionKind::AddNeuron
                | ActionKind::CloneNeuron
                | ActionKind::SplitSynapse
        )
    }
}

/// Structural action.
#[derive(Debug, Clone, PartialEq)]
pub enum StructuralAction<T: Scalar> {
    /// Create a synapse.
    AddSynapse {
        /// Source.
        src: NeuronId,
        /// Receiver.
        dst: NeuronId,
        /// Starting weight.
        weight: T,
        /// Delay.
        delay: u16,
    },
    /// Disable synapse.
    DisableSynapse {
        /// Synapse.
        id: SynapseId,
    },
    /// Physically remove the synapse.
    RemoveSynapse {
        /// Synapse.
        id: SynapseId,
    },
    /// Redirect the synapse.
    RewireSynapse {
        /// Synapse.
        id: SynapseId,
        /// New source.
        new_src: Option<NeuronId>,
        /// New receiver.
        new_dst: Option<NeuronId>,
    },
    /// Change synapse parameters.
    MutateSynapse {
        /// Synapse.
        id: SynapseId,
        /// Change.
        mutation: SynapseMutation<T>,
    },
    /// Create a hidden neuron with the given connections `src → new → dst`.
    AddNeuron {
        /// Dynamics model.
        model: NeuronModel<T>,
        /// Polarity.
        polarity: NeuronPolarity,
        /// Incoming connections `(source, weight)`.
        srcs: Vec<(NeuronId, T)>,
        /// Outgoing connections `(receiver, weight)`.
        dsts: Vec<(NeuronId, T)>,
    },
    /// Cloning a hidden neuron.
    CloneNeuron {
        /// Source neuron.
        id: NeuronId,
        /// Relative spread of incoming weights.
        jitter: T,
    },
    /// Divide the synapse with a relay neuron.
    SplitSynapse {
        /// Synapse.
        id: SynapseId,
        /// Relay initialization.
        init: SplitInit<T>,
    },
    /// Remove the hidden neuron with all synapses.
    RemoveNeuron {
        /// Neuron.
        id: NeuronId,
    },
    /// Change neuron parameters.
    MutateNeuron {
        /// Neuron.
        id: NeuronId,
        /// Change.
        mutation: NeuronMutation<T>,
    },
    /// Change the neuron model with explicit state migration.
    ChangeNeuronModel {
        /// Neuron.
        id: NeuronId,
        /// New model.
        model: NeuronModel<T>,
        /// How to transfer the old state.
        migration: StateMigration,
    },
}

impl<T: Scalar> StructuralAction<T> {
    /// Operation type.
    pub fn kind(&self) -> ActionKind {
        match self {
            StructuralAction::AddSynapse { .. } => ActionKind::AddSynapse,
            StructuralAction::DisableSynapse { .. } => ActionKind::DisableSynapse,
            StructuralAction::RemoveSynapse { .. } => ActionKind::RemoveSynapse,
            StructuralAction::RewireSynapse { .. } => ActionKind::RewireSynapse,
            StructuralAction::MutateSynapse { .. } => ActionKind::MutateSynapse,
            StructuralAction::AddNeuron { .. } => ActionKind::AddNeuron,
            StructuralAction::CloneNeuron { .. } => ActionKind::CloneNeuron,
            StructuralAction::SplitSynapse { .. } => ActionKind::SplitSynapse,
            StructuralAction::RemoveNeuron { .. } => ActionKind::RemoveNeuron,
            StructuralAction::MutateNeuron { .. } => ActionKind::MutateNeuron,
            StructuralAction::ChangeNeuronModel { .. } => ActionKind::ChangeNeuronModel,
        }
    }
}

impl<T: Scalar> fmt::Display for StructuralAction<T> {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            StructuralAction::AddSynapse {
                src,
                dst,
                weight,
                delay,
            } => {
                write!(f, "add_synapse {src}→{dst} w={weight:.3} d={delay}")
            }
            StructuralAction::DisableSynapse { id } => write!(f, "disable_synapse {id}"),
            StructuralAction::RemoveSynapse { id } => write!(f, "remove_synapse {id}"),
            StructuralAction::RewireSynapse {
                id,
                new_src,
                new_dst,
            } => {
                write!(f, "rewire_synapse {id} src={new_src:?} dst={new_dst:?}")
            }
            StructuralAction::MutateSynapse { id, mutation } => {
                write!(f, "mutate_synapse {id} {mutation:?}")
            }
            StructuralAction::AddNeuron { srcs, dsts, .. } => {
                write!(f, "add_neuron in={} out={}", srcs.len(), dsts.len())
            }
            StructuralAction::CloneNeuron { id, .. } => write!(f, "clone_neuron {id}"),
            StructuralAction::SplitSynapse { id, .. } => write!(f, "split_synapse {id}"),
            StructuralAction::RemoveNeuron { id } => write!(f, "remove_neuron {id}"),
            StructuralAction::MutateNeuron { id, mutation } => {
                write!(f, "mutate_neuron {id} {mutation:?}")
            }
            StructuralAction::ChangeNeuronModel {
                id,
                model,
                migration,
            } => write!(
                f,
                "change_neuron_model {id} → {} [{}]",
                model.display_name(),
                migration.as_str()
            ),
        }
    }
}

/// Reason for structural change.
#[derive(Debug, Clone, PartialEq)]
pub enum Reason {
    /// High joint activity of the couple.
    CoActivity {
        /// Pair assessment.
        score: f32,
    },
    /// The quality is not improving.
    Plateau {
        /// How many grades in a row without improvement.
        evaluations: usize,
    },
    /// Low synapse utility.
    LowUtility {
        /// Relative transmission (to median).
        utility: f32,
    },
    /// The disabled synapse has completed its waiting period.
    GraceElapsed,
    /// A neuron with no active inputs or outputs.
    DeadNeuron,
    /// The neuron remains silent throughout the observation.
    SilentNeuron {
        /// Spike frequency.
        rate: f32,
    },
    /// The frequency is higher than permissible.
    Overactive {
        /// Spike frequency.
        rate: f32,
    },
    /// The frequency is below acceptable.
    Underactive {
        /// Spike frequency.
        rate: f32,
    },
    /// Degree/density limitation.
    Connectivity,
    /// Automatic model selection based on efficiency or specialization.
    ModelSelection {
        /// Gain in valuation relative to the current model.
        score: f32,
    },
    /// Learning the physical parameter of a neuron based on the measured loss.
    ParameterLearning {
        /// Parameter name.
        parameter: &'static str,
        /// Improvement of loss relative to the original value.
        score: f32,
    },
    /// Manual action.
    Manual,
}

impl fmt::Display for Reason {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Reason::CoActivity { score } => write!(f, "joint activity {score:.2}"),
            Reason::Plateau { evaluations } => write!(f, "quality plateau ({evaluations} ratings)"),
            Reason::LowUtility { utility } => write!(f, "low utility {utility:.3}"),
            Reason::GraceElapsed => write!(f, "the waiting period after disconnection has expired"),
            Reason::DeadNeuron => write!(f, "neuron with no active inputs or outputs"),
            Reason::SilentNeuron { rate } => write!(f, "neuron is silent (rate {rate:.4})"),
            Reason::Overactive { rate } => write!(f, "overload (rate {rate:.3})"),
            Reason::Underactive { rate } => write!(f, "insufficient activity (rate {rate:.4})"),
            Reason::Connectivity => write!(f, "connectivity constraint"),
            Reason::ModelSelection { score } => {
                write!(f, "model selection (winning {score:.3})")
            }
            Reason::ParameterLearning { parameter, score } => {
                write!(f, "training parameter {parameter} (gain {score:.3})")
            }
            Reason::Manual => write!(f, "manually"),
        }
    }
}

/// Result of the operation.
#[derive(Debug, Clone, PartialEq)]
pub enum Outcome {
    /// Applied and saved.
    Applied,
    /// Applied, but rolled back due to deterioration in quality.
    RolledBack,
    /// Not applied (validation error).
    Failed(String),
}

impl Outcome {
    pub(crate) fn code(&self) -> u8 {
        match self {
            Outcome::Applied => 0,
            Outcome::RolledBack => 1,
            Outcome::Failed(_) => 2,
        }
    }
}

/// Structural event log entry.
#[derive(Debug, Clone, PartialEq)]
pub struct StructuralEvent {
    /// Serial number.
    pub seq: u64,
    /// Simulation step.
    pub step: u64,
    /// Sample number.
    pub sample: u64,
    /// Operation type.
    pub kind: ActionKind,
    /// Description of the action.
    pub action: String,
    /// Description of the reason.
    pub reason: String,
    /// Bottom line.
    pub outcome: Outcome,
    /// Loss before change (if measured).
    pub loss_before: Option<f32>,
    /// Loss after change (if measured).
    pub loss_after: Option<f32>,
    /// Neurons up.
    pub neurons_before: usize,
    /// Neurons after.
    pub neurons_after: usize,
    /// Synapses to.
    pub synapses_before: usize,
    /// Synapses after.
    pub synapses_after: usize,
}

impl fmt::Display for StructuralEvent {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(
            f,
            "#{} step={} {} [{}] → {:?}; N {}→{}, S {}→{}",
            self.seq,
            self.step,
            self.action,
            self.reason,
            self.outcome,
            self.neurons_before,
            self.neurons_after,
            self.synapses_before,
            self.synapses_after
        )
    }
}

/// Outcome counters by operation type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub struct OutcomeCounts {
    /// Applied and saved.
    pub applied: u64,
    /// Rolled back.
    pub rolled_back: u64,
    /// Not applied.
    pub failed: u64,
}

/// Structural event log (ring buffer) and summary counters.
#[derive(Debug, Clone, PartialEq)]
pub struct EventLog {
    pub(crate) events: VecDeque<StructuralEvent>,
    pub(crate) capacity: usize,
    pub(crate) next_seq: u64,
    pub(crate) totals: BTreeMap<ActionKind, OutcomeCounts>,
}

impl EventLog {
    /// Empty log.
    pub fn new(capacity: usize) -> Self {
        Self {
            events: VecDeque::new(),
            capacity: capacity.max(1),
            next_seq: 0,
            totals: BTreeMap::new(),
        }
    }

    /// Add an event; `seq` is assigned by the log.
    pub fn push(&mut self, mut e: StructuralEvent) -> u64 {
        e.seq = self.next_seq;
        self.next_seq += 1;
        let c = self.totals.entry(e.kind).or_default();
        match e.outcome {
            Outcome::Applied => c.applied += 1,
            Outcome::RolledBack => c.rolled_back += 1,
            Outcome::Failed(_) => c.failed += 1,
        }
        if self.events.len() == self.capacity {
            self.events.pop_front();
        }
        let seq = e.seq;
        self.events.push_back(e);
        seq
    }

    /// Mark an event recorded earlier as rolled back (delayed check).
    /// Counters by operation type are adjusted, even if the event has already
    /// evicted from the ring buffer.
    pub fn reclassify_rolled_back(&mut self, seq: u64, kind: ActionKind, loss_after: f32) {
        if let Some(e) = self.events.iter_mut().find(|e| e.seq == seq) {
            e.outcome = Outcome::RolledBack;
            e.loss_after = Some(loss_after);
        }
        let c = self.totals.entry(kind).or_default();
        c.applied = c.applied.saturating_sub(1);
        c.rolled_back += 1;
    }

    /// Events from old to new.
    pub fn iter(&self) -> impl Iterator<Item = &StructuralEvent> {
        self.events.iter()
    }

    /// Number of events stored.
    pub fn len(&self) -> usize {
        self.events.len()
    }

    /// Is the log empty?
    pub fn is_empty(&self) -> bool {
        self.events.is_empty()
    }

    /// Summary counters by type of operation (for all time, not only by
    /// stored events).
    pub fn totals(&self) -> &BTreeMap<ActionKind, OutcomeCounts> {
        &self.totals
    }
}

/// What the operation created.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct ActionEffect {
    /// Created neurons.
    pub neurons: Vec<NeuronId>,
    /// Created synapses.
    pub synapses: Vec<SynapseId>,
}

fn birth<T: Scalar>(net: &DynamicNetwork<T>) -> Birth {
    Birth {
        step: net.step_count(),
        sample: net.sample_count(),
    }
}

/// Apply the action atomically. If there is an error, the topology does not change.
///
/// Condition: neurons removed by surgery disappear along with their
/// synapses; accumulated joint activity is forgotten.
/// New elements are created with the source `Origin::Grown`/`Clone`/`Split`.
pub fn apply_action<T: Scalar>(
    net: &mut DynamicNetwork<T>,
    action: &StructuralAction<T>,
    rng: &mut Rng64,
) -> StructuralResult<ActionEffect> {
    let b = birth(net);
    let mut eff = ActionEffect::default();
    match action {
        StructuralAction::AddSynapse {
            src,
            dst,
            weight,
            delay,
        } => {
            let id =
                net.topology_mut()
                    .add_synapse(*src, *dst, *weight, *delay, Origin::Grown, b)?;
            eff.synapses.push(id);
            if let Some(t) = net.tracker_mut() {
                t.forget_pair(*src, *dst);
            }
        }
        StructuralAction::DisableSynapse { id } => {
            net.topology_mut().disable_synapse(*id, b.step)?
        }
        StructuralAction::RemoveSynapse { id } => {
            net.topology_mut().remove_synapse(*id)?;
        }
        StructuralAction::RewireSynapse {
            id,
            new_src,
            new_dst,
        } => {
            net.topology_mut().rewire_synapse(*id, *new_src, *new_dst)?;
        }
        StructuralAction::MutateSynapse { id, mutation } => {
            net.topology_mut().mutate_synapse(*id, *mutation)?
        }
        StructuralAction::AddNeuron {
            model,
            polarity,
            srcs,
            dsts,
        } => {
            let (n, ss) = net.topology_mut().transaction(|t| {
                let n = t.add_neuron_with(
                    NeuronRole::Hidden,
                    *polarity,
                    model.clone(),
                    Origin::Grown,
                    b,
                )?;
                let mut ids = Vec::new();
                for (s, w) in srcs {
                    ids.push(t.add_synapse(*s, n, *w, 0, Origin::Grown, b)?);
                }
                for (d, w) in dsts {
                    ids.push(t.add_synapse(n, *d, *w, 0, Origin::Grown, b)?);
                }
                Ok((n, ids))
            })?;
            eff.neurons.push(n);
            eff.synapses = ss;
        }
        StructuralAction::CloneNeuron { id, jitter } => {
            let n = net.topology_mut().clone_neuron(*id, *jitter, rng, b)?;
            eff.neurons.push(n);
        }
        StructuralAction::SplitSynapse { id, init } => {
            let r = net.topology_mut().split_synapse(*id, *init, b)?;
            eff.neurons.push(r.neuron);
            eff.synapses.extend([r.incoming, r.outgoing]);
        }
        StructuralAction::RemoveNeuron { id } => {
            net.topology_mut().remove_neuron(*id)?;
            if let Some(t) = net.tracker_mut() {
                t.forget_neuron(*id);
            }
        }
        StructuralAction::MutateNeuron { id, mutation } => {
            net.topology_mut().mutate_neuron(*id, mutation.clone())?;
        }
        StructuralAction::ChangeNeuronModel {
            id,
            model,
            migration,
        } => {
            net.change_neuron_model(*id, model.clone(), *migration)?;
        }
    }
    Ok(eff)
}

/// Apply a batch of actions as one transaction: if any action
/// impossible, the entire network returns to the state before the packet.
pub fn apply_batch<T: Scalar>(
    net: &mut DynamicNetwork<T>,
    actions: &[StructuralAction<T>],
    rng: &mut Rng64,
) -> StructuralResult<Vec<ActionEffect>> {
    let backup = net.topology().clone();
    let rng_backup = rng.clone();
    let mut out = Vec::with_capacity(actions.len());
    for a in actions {
        match apply_action(net, a, rng) {
            Ok(e) => out.push(e),
            Err(e) => {
                net.replace_topology(backup);
                *rng = rng_backup;
                return Err(e);
            }
        }
    }
    if let Err(e) = net.topology().validate() {
        net.replace_topology(backup);
        *rng = rng_backup;
        return Err(e);
    }
    Ok(out)
}

impl From<StructuralError> for crate::error::SnnError {
    fn from(e: StructuralError) -> Self {
        crate::error::SnnError::InvalidConfig {
            what: "structural",
            reason: e.to_string(),
        }
    }
}
