//! # neo-nn
//!
//! Universal layers of neural networks for Neo.

pub mod attention;
pub mod embedding;
pub mod error;
pub mod ffn;
pub mod norm;
pub mod peft;
pub mod quantized;
pub mod transformer;

pub use error::NnError;

#[cfg(test)]
mod tests_errors;
