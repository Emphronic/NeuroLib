//! Normalizations and scaling: four small independent primitives.
//!
//! ```text
//! ScaleNorm y = g x/‖x‖ g — one scalar
//! LayerScale y = γ ⊙ x γ ∈ [D]; not normalization
//! ConditionalRmsNorm y = (1+scale) ⊙ RMS(x) + shift scale/shift come from outside
//! AdaptiveRmsNorm     c → Linear → scale, shift → ConditionalRmsNorm
//! ```
//!
//! All layers accept a tensor of any rank and work along the last axis
//! (`[B, D]`, `[B, T, D]`, ...), entirely on the device, without contacting the host.
//! Mismatch of the last axis with `d_model` - panic with a clear message,
//! there are no silent `reshape`. Configs are checked through `validate()`.
//!
//! `LayerScale` does not know about residual: `h = x + γ ⊙ F(x)` collects the calling code.

mod adaptive;
mod conditional;
mod layerscale;
mod scale;

#[cfg(test)]
mod tests;

pub use adaptive::{AdaptiveRmsNorm, AdaptiveRmsNormConfig};
pub use conditional::{ConditionalRmsNorm, ConditionalRmsNormConfig};
pub use layerscale::{LayerScale, LayerScaleConfig};
pub use scale::{ScaleNorm, ScaleNormConfig};

use burn::module::Param;
use burn::prelude::*;

pub(crate) fn check_eps(eps: f32) -> Result<(), String> {
    if eps.is_finite() && eps > 0.0 {
        Ok(())
    } else {
        Err(format!("eps must be finite and > 0, obtained {eps}"))
    }
}

pub(crate) fn check_d_model(d: usize) -> Result<(), String> {
    if d > 0 {
        Ok(())
    } else {
        Err("d_model must be > 0".into())
    }
}

pub(crate) fn assert_last_dim<const D: usize>(t: &Tensor<D>, expected: usize, what: &str) {
    let got = t.dims()[D - 1];
    assert_eq!(
        got, expected,
        "{what}: last axis {got}, expected {expected}"
    );
}

pub(crate) fn numel<const D: usize>(p: &Param<Tensor<D>>) -> usize {
    p.dims().iter().product()
}
