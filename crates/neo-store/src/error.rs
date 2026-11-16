//! Weight storage errors.

use std::fmt;
use std::io;

/// Result of operations neo-store.
pub type StoreResult<T> = Result<T, StoreError>;

/// Error reading, writing, or checking storage.
#[derive(Debug)]
#[non_exhaustive]
pub enum StoreError {
    /// I/O error.
    Io(io::Error),
    /// The file is damaged or has an invalid structure.
    Format(String),
    /// The CRC-32C checksum did not match.
    Checksum {
        /// What was checked (header, index, tensor name).
        what: String,
        /// Recorded value.
        expected: u32,
        /// Calculated value.
        actual: u32,
    },
    /// The file is written in a newer, incompatible format.
    UnsupportedVersion {
        /// Older version of the file.
        major: u16,
        /// Minor version of the file.
        minor: u16,
    },
    /// The file requires a capability that this reader does not know.
    UnsupportedFeature(String),
    /// There is no tensor with this name.
    TensorNotFound(String),
    /// The name is already taken.
    Duplicate(String),
    /// Invalid name (tensor, key, node).
    InvalidName(String),
    /// The shape was not as expected.
    ShapeMismatch {
        /// Tensor name.
        name: String,
        /// expected shape.
        expected: Vec<u64>,
        /// actual shape.
        actual: Vec<u64>,
    },
    /// The element type did not match the expected one.
    DTypeMismatch {
        /// Tensor name.
        name: String,
        /// Expected type.
        expected: String,
        /// Actual type.
        actual: String,
    },
    /// NaN or ±Inf found in float tensor (policy `Reject`).
    NonFinite {
        /// Tensor name.
        name: String,
        /// Index of the first non-finite element.
        index: u64,
    },
    /// The network graph is incorrect.
    Graph(String),
    /// Invalid argument or condition.
    Invalid(String),
}

impl fmt::Display for StoreError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            StoreError::Io(e) => write!(f, "I/O error: {e}"),
            StoreError::Format(m) => write!(f, "corrupted or invalid file: {m}"),
            StoreError::Checksum {
                what,
                expected,
                actual,
            } => write!(
                f,
                "checksum did not match ({what}): {expected:#010x} written, {actual:#010x} calculated"
            ),
            StoreError::UnsupportedVersion { major, minor } => {
                write!(f, "unsupported version of the {major}.{minor} format")
            }
            StoreError::UnsupportedFeature(m) => write!(f, "unsupported feature: {m}"),
            StoreError::TensorNotFound(n) => write!(f, "tensor '{n}' not found"),
            StoreError::Duplicate(n) => write!(f, "the name '{n}' is already taken"),
            StoreError::InvalidName(n) => write!(f, "invalid name: {n}"),
            StoreError::ShapeMismatch {
                name,
                expected,
                actual,
            } => write!(
                f,
                "tensor “{name}”: expected shape {expected:?}, received {actual:?}"
            ),
            StoreError::DTypeMismatch {
                name,
                expected,
                actual,
            } => write!(
                f,
                "tensor '{name}': expected type {expected}, actual {actual}"
            ),
            StoreError::NonFinite { name, index } => write!(
                f,
                "tensor '{name}': element {index} - NaN or infinity (weights not written)"
            ),
            StoreError::Graph(m) => write!(f, "graph: {m}"),
            StoreError::Invalid(m) => write!(f, "{m}"),
        }
    }
}

impl std::error::Error for StoreError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            StoreError::Io(e) => Some(e),
            _ => None,
        }
    }
}

impl From<io::Error> for StoreError {
    fn from(e: io::Error) -> Self {
        StoreError::Io(e)
    }
}
