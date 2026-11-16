//! Errors `neo-nn`.
//!
//! Crate convention: configurations have `validate() -> Result<(), String>`
//! (error text for people), `try_init(..) -> Result<_, NnError>` is the path without
//! panic, and `init(..)` is a thin Burn-style wrapper that panics with
//! with the message [`NnError`] (in tests and examples this is shorter). Direct calls
//! `.unwrap()`/`.expect()` are not used in the combat code.

use burn::tensor::{Element, TensorData};
use thiserror::Error;

/// Layer `neo-nn` error.
#[derive(Debug, Clone, PartialEq, Eq, Error)]
#[non_exhaustive]
pub enum NnError {
    /// Configuration failed `validate()`.
    #[error("{layer}: incorrect configuration: {reason}")]
    InvalidConfig {
        /// Configuration/layer name.
        layer: &'static str,
        /// What exactly is wrong.
        reason: String,
    },
    /// Failed to read tensor data to host (eg indexes
    /// routing).
    #[error("{op}: Failed to read tensor data: {reason}")]
    TensorRead {
        /// The operation that read the tensor.
        op: &'static str,
        /// Backend error text.
        reason: String,
    },
    /// An internal invariant or state is violated (empty cache, invalid
    /// FFN option, etc.).
    #[error("{op}: {reason}")]
    InvalidState {
        /// Operation.
        op: &'static str,
        /// Description.
        reason: String,
    },
    /// Invalid input data (for example, id is outside the dictionary).
    #[error("{op}: {reason}")]
    InvalidInput {
        /// Operation.
        op: &'static str,
        /// Description.
        reason: String,
    },
}

impl NnError {
    /// `InvalidConfig` from the text `validate()`.
    pub fn config(layer: &'static str, reason: impl Into<String>) -> Self {
        Self::InvalidConfig {
            layer,
            reason: reason.into(),
        }
    }

    /// `InvalidState`.
    pub fn state(op: &'static str, reason: impl Into<String>) -> Self {
        Self::InvalidState {
            op,
            reason: reason.into(),
        }
    }

    /// `InvalidInput`.
    pub fn input(op: &'static str, reason: impl Into<String>) -> Self {
        Self::InvalidInput {
            op,
            reason: reason.into(),
        }
    }
}

/// Read `TensorData` into `Vec<E>` with type cast.
pub(crate) fn data_to_vec<E: Element>(
    data: TensorData,
    op: &'static str,
) -> Result<Vec<E>, NnError> {
    data.convert::<E>()
        .try_to_vec::<E>()
        .map_err(|e| NnError::TensorRead {
            op,
            reason: format!("{e:?}"),
        })
}

/// Panic with error message: the only place where the `init`/`forward` wrappers
/// turn [`NnError`] into panic.
#[cold]
#[track_caller]
pub(crate) fn fail(e: NnError) -> ! {
    panic!("{e}")
}
