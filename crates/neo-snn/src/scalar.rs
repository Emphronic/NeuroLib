//! Scalar calculation type of the CPU part of the library.
//!
//! [`Scalar`] - the minimum set of operations that is enough for dynamic
//! neuron models, topology and engine `structural`: arithmetic, comparison,
//! `exp`/`ln`/`sqrt`/`tanh`, conversion from/to `f64`. Implemented for `f32` and
//! `f64`; other types (eg `f16`, fixed point) are connected
//! implementation of the trait.
//!
//! All generic library types are declared as `Type<T = f32>`: so
//! `LifParams` still means `LifParams<f32>`, so the existing code
//! does not change, and `LifParams<f64>` gives double precision.
//!
//! Burn layers (`neuron::Leaky`, ...) do not use this trait: the type of their tensors
//! selected by the backend at runtime.

use std::fmt::{Debug, Display};
use std::ops::{Add, AddAssign, Div, DivAssign, Mul, MulAssign, Neg, Sub, SubAssign};

use crate::structural::StatsScalar;

/// Real scalar for neuron dynamics and synapse weights.
pub trait Scalar:
    StatsScalar
    + Debug
    + Display
    + Default
    + PartialEq
    + PartialOrd
    + Send
    + Sync
    + 'static
    + Add<Output = Self>
    + Sub<Output = Self>
    + Mul<Output = Self>
    + Div<Output = Self>
    + Neg<Output = Self>
    + AddAssign
    + SubAssign
    + MulAssign
    + DivAssign
{
    /// Zero.
    const ZERO: Self;
    /// Unit.
    const ONE: Self;

    /// Constant from `f64`: this is how literals are written in generalized code
    /// (`T::c(0.04)`).
    fn c(v: f64) -> Self {
        Self::from_f64(v)
    }
    /// From `f32` (expansion or rounding).
    fn from_f32(v: f32) -> Self {
        Self::from_f64(f64::from(v))
    }
    /// In `f32` (rounded).
    fn to_f32(self) -> f32 {
        self.to_f64() as f32
    }
    /// From the integer `usize` (number of sub-steps, counters).
    fn from_usize(v: usize) -> Self {
        Self::from_f64(v as f64)
    }
    /// `e^x`.
    fn exp(self) -> Self;
    /// Natural logarithm.
    fn ln(self) -> Self;
    /// Square root.
    fn sqrt(self) -> Self;
    /// Hyperbolic tangent.
    fn tanh(self) -> Self;
    /// Module.
    fn abs(self) -> Self;
    /// A whole degree.
    fn powi(self, n: i32) -> Self;
    /// Is the number finite (not `NaN` and not `±∞`).
    fn is_finite(self) -> bool;

    /// The smaller of the two (`NaN` is ignored, as with `f32::min`).
    fn min(self, other: Self) -> Self {
        if other < self || self != self {
            other
        } else {
            self
        }
    }
    /// The greater of the two (`NaN` is ignored, as with `f32::max`).
    fn max(self, other: Self) -> Self {
        if other > self || self != self {
            other
        } else {
            self
        }
    }
    /// Limit to range `[lo, hi]`.
    fn clamp(self, lo: Self, hi: Self) -> Self {
        self.max(lo).min(hi)
    }
}

macro_rules! impl_scalar {
    ($t:ty) => {
        impl Scalar for $t {
            const ZERO: Self = 0.0;
            const ONE: Self = 1.0;
            fn exp(self) -> Self {
                <$t>::exp(self)
            }
            fn ln(self) -> Self {
                <$t>::ln(self)
            }
            fn sqrt(self) -> Self {
                <$t>::sqrt(self)
            }
            fn tanh(self) -> Self {
                <$t>::tanh(self)
            }
            fn abs(self) -> Self {
                <$t>::abs(self)
            }
            fn powi(self, n: i32) -> Self {
                <$t>::powi(self, n)
            }
            fn is_finite(self) -> bool {
                <$t>::is_finite(self)
            }
            fn min(self, other: Self) -> Self {
                <$t>::min(self, other)
            }
            fn max(self, other: Self) -> Self {
                <$t>::max(self, other)
            }
            fn clamp(self, lo: Self, hi: Self) -> Self {
                <$t>::clamp(self, lo, hi)
            }
        }
    };
}

impl_scalar!(f32);
impl_scalar!(f64);

#[cfg(test)]
mod tests {
    use super::*;

    fn norm<T: Scalar>(x: T, y: T) -> T {
        (x * x + y * y).sqrt()
    }

    #[test]
    fn scalar_works_for_f32_and_f64() {
        assert_eq!(norm(3.0f32, 4.0), 5.0);
        assert_eq!(norm(3.0f64, 4.0), 5.0);
        assert_eq!(f32::c(0.5), 0.5);
        assert_eq!(<f64 as Scalar>::from_f32(0.25), 0.25);
        assert_eq!(Scalar::clamp(5.0f64, 0.0, 1.0), 1.0);
        assert!(!Scalar::is_finite(f32::NAN));
        // f64 stores the constant more accurately than f32.
        assert!((f64::c(0.1) - 0.1).abs() < 1e-17);
    }
}
