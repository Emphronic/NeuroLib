//! Custom surrogate: your own formula without manual `impl Surrogate`.
//!
//! ## Why a macro and not a closure
//!
//! [`Surrogate::surrogate_forward`](super::Surrogate::surrogate_forward)
//! generalized across the `B` backend, that is, one implementation must work for
//! ALL backends at once. A closure in Rust cannot have such generality:
//! `Fn(Tensor<NdArray, 2>) -> Tensor<NdArray, 2>` and its analogue for CUDA -
//! these are different types, and there is no common `Fn` for them. Therefore, where
//! in Python it is enough to pass a function, here it is the type that is needed - and the macro
//! eliminates the need to write it by hand.
//!
//! ## How to use
//!
//! ```ignore
//! use neo_snn::custom_surrogate;
//!
//! // Hyperbolic surrogate with slope 3: one line instead
//! // structures, Default, constructor and impl.
//! custom_surrogate!(
//!     /// Example surrogate: 0.5 + 0.5·tanh(3x).
//!     MyTanh,
//!     |x| x.mul_scalar(3.0).tanh().mul_scalar(0.5).add_scalar(0.5)
//! );
//!
//! let spike_grad = MyTanh;
//! ```
//!
//! The body receives `Tensor<2>` (usually `membrane - threshold`) and must
//! return the SMOOTH value σ(x), the derivative of which will become
//! surrogate gradient. The value itself does not affect forward: it
//! abbreviated in straight-through expression.

/// Declare a type that implements [`Surrogate`](crate::surrogate::Surrogate),
/// from one expression.
///
/// Syntax: `custom_surrogate!(Name, |x| expression);` - see examples in
/// [module](self).
#[macro_export]
macro_rules! custom_surrogate {
    ($(#[$meta:meta])* $name:ident, |$x:ident| $body:expr) => {
        $(#[$meta])*
        #[derive(Debug, Clone, Copy, Default)]
        pub struct $name;

        impl $crate::surrogate::Surrogate for $name {
            fn surrogate_forward<B: ::burn::tensor::backend::Backend>(
                &self,
                $x: ::burn::tensor::Tensor<2>,
            ) -> ::burn::tensor::Tensor<2> {
                $body
            }
        }
    };
}
