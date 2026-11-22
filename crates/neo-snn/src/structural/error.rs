//! Typed structural plasticity errors.

use thiserror::Error;

use super::topology::{NeuronId, SynapseId};

/// Result of structural operations.
pub type StructuralResult<T> = core::result::Result<T, StructuralError>;

/// Structural operation error. No operation in the module panics:
/// an invalid change returns an error and the topology stays as it was.
#[derive(Debug, Clone, PartialEq, Error)]
#[non_exhaustive]
pub enum StructuralError {
    /// There is no neuron with this identifier.
    #[error("neuron {0} not found")]
    NoSuchNeuron(NeuronId),
    /// There is no synapse with this identifier.
    #[error("synapse {0} not found")]
    NoSuchSynapse(SynapseId),
    /// This connection already exists, and duplicates are forbidden.
    #[error("connection {src} → {dst} already exists")]
    DuplicateSynapse {
        /// Source.
        src: NeuronId,
        /// Target.
        dst: NeuronId,
    },
    /// The self-loop `n → n` is forbidden by the configuration.
    #[error("self-loop on neuron {0} is forbidden")]
    SelfLoopForbidden(NeuronId),
    /// The connection would create a cycle, and recurrence is disabled.
    #[error("connection {src} → {dst} forms a cycle, recurrent connections are disabled")]
    RecurrenceForbidden {
        /// Source.
        src: NeuronId,
        /// Target.
        dst: NeuronId,
    },
    /// An input neuron does not accept synapses.
    #[error("input neuron {0} cannot be the target of a synapse")]
    InputCannotReceive(NeuronId),
    /// The network limit was exceeded.
    #[error("limit exceeded: {what} (maximum {limit})")]
    LimitExceeded {
        /// What is limited.
        what: &'static str,
        /// Limit value.
        limit: usize,
    },
    /// The operation would violate the minimum allowed size.
    #[error("minimum violated: {what} (minimum {min})")]
    BelowMinimum {
        /// What is limited.
        what: &'static str,
        /// Minimum.
        min: usize,
    },
    /// Invalid parameter value.
    #[error("{what}: {reason}")]
    InvalidParameter {
        /// Parameter.
        what: &'static str,
        /// Reason.
        reason: String,
    },
    /// The parameter is not supported by this neuron type.
    #[error("neuron type \"{kind}\" does not support parameter \"{parameter}\"")]
    UnsupportedParameter {
        /// Neuron type.
        kind: &'static str,
        /// Parameter.
        parameter: &'static str,
    },
    /// The operation does not apply to a neuron of this type.
    #[error("{op}: neuron {id} of type \"{kind}\" is not suitable")]
    WrongKind {
        /// Operation.
        op: &'static str,
        /// Neuron.
        id: NeuronId,
        /// Its type.
        kind: &'static str,
    },
    /// A topology invariant was violated (detected by `Topology::validate`).
    #[error("topology invariant violated: {0}")]
    InvariantViolation(String),
    /// Invalid configuration.
    #[error("invalid configuration: {0}")]
    InvalidConfig(String),
    /// Checkpoint read/write error.
    #[error("checkpoint: {0}")]
    Checkpoint(String),
    /// The input of the dynamic network did not match in size.
    #[error("{op}: expected {expected} values, got {actual}")]
    InputSize {
        /// Operation.
        op: &'static str,
        /// Expected.
        expected: usize,
        /// Got.
        actual: usize,
    },
    /// A user-defined dynamic neuron model is not registered in the registry.
    #[error("custom model \"{0}\" is not registered in the registry")]
    UnknownCustomModel(String),
    /// A custom model and the state are incompatible (model id or state type mismatch).
    #[error("custom model \"{0}\": incompatible state")]
    IncompatibleCustomState(String),
    /// Invalid configuration of a custom model.
    #[error("custom model \"{0}\": invalid configuration: {1}")]
    InvalidCustomConfig(String, String),
    /// Invalid state of a custom model.
    #[error("custom model \"{0}\": invalid state: {1}")]
    InvalidCustomState(String, String),
    /// Serialization of a custom model failed.
    #[error("custom model \"{0}\": serialization error: {1}")]
    CustomSerialization(String, String),
    /// Schema version of the custom model is not supported by the registry entry.
    #[error("custom model \"{0}\": schema version {1} is not supported (expected {2})")]
    UnsupportedSchemaVersion(String, u32, u32),
    /// Migration of the custom model state is not possible under the given policy.
    #[error("custom model \"{0}\": migration to {1} is not supported")]
    MigrationNotSupported(String, &'static str),
    /// The custom model does not provide the dynamic network execution handler
    /// (for example, a non-`Default`-constructible model is stored in a network
    /// built without a registry).
    #[error("custom model \"{0}\" has no registered handler for the dynamic network")]
    MissingCustomHandler(String),
    /// The neuron model does not support the online local eligibility dynamics
    /// of the plasticity system (`structural::eprop`). Typed error, not a
    /// fallback approximation: models that do not declare support
    /// (`online_eprop_family() == None`) are rejected instead of being trained
    /// with an incorrect rule.
    #[error("neuron model \"{0}\" does not support online local eligibility dynamics")]
    UnsupportedEpropDynamics(String),
}

impl StructuralError {
    pub(crate) fn param(what: &'static str, reason: impl Into<String>) -> Self {
        Self::InvalidParameter {
            what,
            reason: reason.into(),
        }
    }
}
