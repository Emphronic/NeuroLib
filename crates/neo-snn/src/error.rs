//! Typed crate errors.
//!
//! Shape checks and tensor-data extraction return [`SnnError`] with the
//! operation name, expected and actual shape, rather than panicking with
//! impersonal text somewhere inside Burn.
//!
//! API convention: for operations that can fail, there is a pair
//! `try_xxx` (returns [`Result`]) and `xxx` (calls `try_xxx` and
//! panics with the same error text). Existing code keeps working; new code
//! can handle the error.

use burn::tensor::{Tensor, TensorData};

/// Result of crate operations.
pub type Result<T> = core::result::Result<T, SnnError>;

/// Error `neo-snn`.
#[derive(Debug, Clone, PartialEq, thiserror::Error)]
pub enum SnnError {
    /// The shape of the tensor did not coincide with the expected one.
    #[error("{op}: expected shape {expected}, received {actual}")]
    Shape {
        /// The operation in which a discrepancy was found.
        op: &'static str,
        /// What was expected (human readable, for example `[batch, 3, h, w]`).
        expected: String,
        /// What's received.
        actual: String,
    },
    /// The number of input channels/features did not match the layer configuration.
    #[error("{op}: input has {actual} channels, layer is configured to {expected}")]
    Channels {
        /// Operation.
        op: &'static str,
        /// Expected number of channels.
        expected: usize,
        /// Actual number of channels.
        actual: usize,
    },
    /// Invalid configuration parameter.
    #[error("{what}: {reason}")]
    InvalidConfig {
        /// What parameter or layer.
        what: &'static str,
        /// What exactly is wrong.
        reason: String,
    },
    /// Failed to read tensor data as `f32`
    /// (backend with a different element type or tensor on the device, from where
    /// the data is not readable).
    #[error("{op}: tensor data is not readable as f32 ({detail})")]
    Data {
        /// Operation.
        op: &'static str,
        /// Details from Burn.
        detail: String,
    },
    /// An empty sequence where at least one step is needed.
    #[error("{op}: the sequence does not contain any steps")]
    EmptySequence {
        /// Operation.
        op: &'static str,
    },
    /// Error (de)serializing state.
    #[error("{what}: {reason}")]
    Record {
        /// What was saved/loaded.
        what: &'static str,
        /// Reason.
        reason: String,
    },
}

impl SnnError {
    /// shape error; `expected`/`actual` are formatted as `[a, b, c]`.
    pub fn shape(
        op: &'static str,
        expected: impl core::fmt::Debug,
        actual: impl core::fmt::Debug,
    ) -> Self {
        SnnError::Shape {
            op,
            expected: format!("{expected:?}"),
            actual: format!("{actual:?}"),
        }
    }

    /// shape error with free text description of what is expected.
    pub fn shape_msg(
        op: &'static str,
        expected: impl Into<String>,
        actual: impl core::fmt::Debug,
    ) -> Self {
        SnnError::Shape {
            op,
            expected: expected.into(),
            actual: format!("{actual:?}"),
        }
    }
}

/// Tensor data as `Vec<f32>` without panic.
pub(crate) fn tensor_to_vec<const D: usize>(
    tensor: Tensor<D>,
    op: &'static str,
) -> Result<Vec<f32>> {
    let data: TensorData = tensor.into_data();
    data.try_to_vec::<f32>().map_err(|e| SnnError::Data {
        op,
        detail: format!("{e:?}"),
    })
}

/// Input shape check: panic with description of operation expected and
/// actual shape (instead of panicking inside Burn).
#[track_caller]
pub(crate) fn assert_dims<const D: usize>(op: &str, what: &str, got: [usize; D], want: [usize; D]) {
    assert!(
        got == want,
        "{op}: {what} has shape {got:?}, expected {want:?}"
    );
}

/// Checking that the goals `[batch, classes]` are consistent with the prediction
/// (`pred_batch`, `pred_classes`).
#[track_caller]
pub(crate) fn assert_targets(
    op: &str,
    pred_batch: usize,
    pred_classes: usize,
    targets: [usize; 2],
) {
    assert!(
        targets == [pred_batch, pred_classes],
        "{op}: targets have the shape {targets:?}, expected [batch, classes] = [{pred_batch}, {pred_classes}]"
    );
}

/// Expands the result of `try_*` into a value or panics with the text
/// errors (behavior of the previous `expect`, but with a typed message).
#[track_caller]
pub(crate) fn or_panic<T>(result: Result<T>) -> T {
    match result {
        Ok(v) => v,
        Err(e) => panic!("{e}"),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn messages_name_operation_and_shapes() {
        let e = SnnError::shape("StdpConv2d::step", [4, 2], [3, 2]);
        let msg = e.to_string();
        assert!(msg.contains("StdpConv2d::step"));
        assert!(msg.contains("[4, 2]") && msg.contains("[3, 2]"));
    }

    #[test]
    fn channels_error_mentions_both_counts() {
        let msg = SnnError::Channels {
            op: "x",
            expected: 3,
            actual: 5,
        }
        .to_string();
        assert!(msg.contains('3') && msg.contains('5') && msg.contains("channels"));
    }
}
