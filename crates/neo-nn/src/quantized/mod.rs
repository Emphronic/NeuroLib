//! Layers on quantized weights.
//!
//! * [`QuantizedLinear`] — `y = x·Wᵀ + b`, where `W` lies in
//!   [`neo_quantization::PackedBuffer`] and **does not unfold** entirely:
//!   matmul comes with the fused core `neo-synk` directly from the packaging.

pub mod quantized_linear;

pub use quantized_linear::{ExecutionPolicy, QuantizedLinear, QuantizedLinearError};

#[cfg(test)]
mod tests;
