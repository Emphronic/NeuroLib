//! Surrogate gradient trait: forward - Heaviside, backward - smooth function.

use burn::tensor::Tensor;

/// Surrogate for spike nonlinearity.
///
/// The implementation returns a smooth value σ(x), which is substituted into
/// straight-through expression:
/// `spike = heaviside(x).detach() + (σ(x) - σ(x).detach())`.
///
/// Forward is numerically equal to Heaviside, backward goes through σ.
pub trait Surrogate: Send + Sync {
    /// A smooth surrogate value for the input (usually `membrane - threshold`).
    fn surrogate_forward(&self, x: Tensor<2>) -> Tensor<2>;

    /// Type of surrogate with a closed derivative formula, if it is in `neo-synk`.
    ///
    /// On supported devices (Flex), the straight-through bundle is replaced by ONE
    /// operation (`neo_synk::snn::spike_surrogate`): forward — Heaviside, backward —
    /// `grad·σ'(x)`. `None` (default) - the surrogate is computed from normal tensor
    /// operations. The implementation must match [`surrogate_forward`](Self::surrogate_forward):
    /// `σ'` from `SurrogateKind` is the derivative of exactly the `σ` that it returns.
    fn fused_kind(&self) -> Option<neo_synk::snn::SurrogateKind> {
        None
    }
}
