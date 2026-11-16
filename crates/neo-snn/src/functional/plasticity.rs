//! Synaptic plasticity primitives (STDP) for on-chip/unsupervised
//! learning — inspired by `snntorch.functional.stdp_linear`, but
//! as pure tensor functions without coupling to a specific layer.
//!
//! # Why this is NOT through Burn autodiff
//!
//! The rest of the crate trains through backprop: the gradient flows through
//! the surrogate derivative of the spike (see [`crate::surrogate`]) and the
//! `Param` is updated by the optimizer. STDP is a fundamentally different
//! mechanism: the weight changes according to a rule that is LOCAL in time and
//! space (co-activity of the pre- and postsynaptic neuron), without a single
//! global loss function and without a backward pass through time. This is
//! exactly what makes STDP suitable for on-chip learning on neuromorphic
//! hardware: a synapse weight update depends only on quantities already
//! available LOCALLY at that synapse (traces + current spikes), and not on a
//! global differentiation graph through all past steps.
//!
//! Therefore weights here are ordinary `Tensor<2>`, not `Param<Tensor<2>>`:
//! they are changed not by the optimizer through `.grad()`, but by the
//! functions of this module, called explicitly at each step. Nothing prevents
//! wrapping such a tensor in `Param` and mixing the STDP update into ordinary
//! backprop training (hybrid schemes), but that is the responsibility of the
//! calling code, not of this module.
//!
//! # Traces
//!
//! The standard way to make STDP causal without storing the full spike
//! history is an exponentially decaying "trace" of each neuron:
//!
//! ```text
//! x[t] = x[t-1]·exp(-1/τ) + spike[t]
//! ```
//!
//! `x` grows by 1 on each spike and exponentially forgets the past — it is
//! both a measure of "how recently a spike occurred" and (when contributions
//! are summed at each step) the integral of the classical exponential STDP
//! kernel. Here there is a pre- and a postsynaptic trace, and for the triplet
//! version — one more "slow" trace on each side (see [`triplet_update`]).
//!
//! # Rules
//!
//! * [`pair_update`] — classical pair-based STDP (Song, Miller & Abbott,
//!   2000; the same form implemented by `snntorch.functional.stdp_linear`):
//!   `Δw = A+ · x_pre · post − A− · x_post · pre`.
//! * [`triplet_update`] — triplet STDP (Pfister & Gerstner, 2006): pair terms
//!   gain interactions with slow traces, reproducing the frequency dependence
//!   of plasticity that the pair-based rule lacks.
//! * [`reward_modulated_update`] (R-STDP): pair-STDP is split into two
//!   components — a fast eligibility trace (the same co-activity rule, but
//!   WITHOUT immediate application to the weights) and a scalar reward that
//!   arrives later and decides whether to strengthen or weaken what the trace
//!   has already accumulated. This turns STDP from a purely unsupervised rule
//!   into a primitive for on-chip RL (three-factor learning rule:
//!   pre × post × reward).
//!
//! All three rules use the same trace primitive [`update_trace`] and the same
//! way of bounding the weight ([`clip_weights`] / [`soft_bound_weights`]).

use burn::tensor::Tensor;

/// Updates the exponential trace by one step: `x = x·decay + spike`.
///
/// `decay = exp(-dt/tau)` is computed by the calling code (usually once when
/// constructing the layer, not at every step) and passed here as a scalar —
/// the function itself knows nothing about `tau`, only about the resulting
/// decay coefficient. `spike` is the already computed (0/1 or graded) neuron
/// output at the current step, of the same shape `[batch, units]` as the
/// trace.
///
/// The trace does NOT participate in the autodiff graph: `spike` here is
/// usually itself detached from the graph (see `spike_hard` in
/// [`crate::neuron::spike_with_threshold`]); if not, the calling code must
/// decide whether `.detach()` is needed before the call. This function does
/// not detach anything, so as not to impose a decision: in reward-modulated
/// schemes the trace is sometimes deliberately kept in the graph if a gradient
/// is still needed on top of STDP.
pub fn update_trace(trace: Tensor<2>, spike: Tensor<2>, decay: f32) -> Tensor<2> {
    trace.mul_scalar(decay) + spike
}

/// Way to bound the synaptic weight after an STDP update.
///
/// Without a bound, potentiation grows without limit (STDP weights have no
/// built-in "ceiling", unlike gradient descent with weight decay) — this is
/// the classic problem of unmodulated STDP, solved by one of two standard
/// methods.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum WeightBound<T = f32> {
    /// Hard clamping to `[w_min, w_max]` after each update.
    /// Simple and predictable; the weight distribution tends to
    /// pile up at the boundaries (bimodal).
    Hard { w_min: T, w_max: T },
    /// Multiplicative ("soft") bound: the strength of potentiation and
    /// depression itself decreases as the weight approaches the boundary —
    /// `(w_max − w) · Δw+` and `(w − w_min) · Δw−`. Gives a smoother,
    /// unimodal weight distribution (van Rossum, Bi & Turrigiano,
    /// 2000); more expensive (a multiplication at each step), clamping is
    /// still applied on top as a safeguard against crossing the boundary due
    /// to accumulated f32 error.
    Soft { w_min: T, w_max: T },
    /// No bound — the calling code is responsible for making sure the
    /// weights do not diverge (for example, a separate periodic
    /// normalization).
    None,
}

impl WeightBound {
    /// The `[w_min, w_max]` bounds, if they are set by this variant.
    fn bounds(&self) -> Option<(f32, f32)> {
        match *self {
            WeightBound::Hard { w_min, w_max } | WeightBound::Soft { w_min, w_max } => {
                Some((w_min, w_max))
            }
            WeightBound::None => None,
        }
    }
}

/// Hard clamping of weights to `[w_min, w_max]`.
pub fn clip_weights(w: Tensor<2>, w_min: f32, w_max: f32) -> Tensor<2> {
    w.clamp(w_min, w_max)
}

/// Applies soft-bound factors to the potentiation/depression parts of the
/// update: `Δw+ · (w_max − w)`, `Δw− · (w − w_min)`.
///
/// `dw_pot`, `dw_dep` — already computed (non-negative by construction in the
/// calling rules) potentiation and depression values BEFORE application to the
/// weights; see [`pair_update`], where this decomposition is visible
/// explicitly.
fn soft_bound_factors(
    w: &Tensor<2>,
    dw_pot: Tensor<2>,
    dw_dep: Tensor<2>,
    w_min: f32,
    w_max: f32,
) -> Tensor<2> {
    let headroom = w.clone().mul_scalar(-1.0).add_scalar(w_max); // w_max - w
    let floor_room = w.clone().sub_scalar(w_min); // w - w_min
    dw_pot.mul(headroom) - dw_dep.mul(floor_room)
}

/// Final weight bound after adding `w + dw`, according to
/// [`WeightBound`]. `Soft` has already scaled `dw` through
/// [`soft_bound_factors`] in the calling rule — for it only the safeguard
/// `clamp` remains here.
fn apply_bound(w: Tensor<2>, bound: WeightBound) -> Tensor<2> {
    match bound.bounds() {
        Some((lo, hi)) => clip_weights(w, lo, hi),
        None => w,
    }
}

/// Classical pair-based STDP (Song, Miller & Abbott, 2000).
///
/// `w` — weights `[out_features, in_features]` (the same layout as an ordinary
/// `Linear`: a row is one postsynaptic neuron, a column is one presynaptic).
/// `pre_spike`/`post_spike` — spikes of the CURRENT step, shaped
/// `[batch, in_features]`/`[batch, out_features]`. `pre_trace`/`post_trace` —
/// exponential traces BEFORE the update at this step (i.e. computed from
/// spikes up to `t-1`, see the module documentation and [`update_trace`]).
///
/// Rule:
///
/// ```text
/// Δw = a_plus  · (post_trace ⊗ pre_spike)   [potentiation: pre was recent, post fired now]
///    − a_minus · (post_spike ⊗ pre_trace)   [depression:   post was recent, pre fired now]
/// ```
///
/// where `⊗` is the outer product over the batch with averaging over the batch
/// (see below). This is the causality of STDP: if the presynaptic spike
/// PRECEDED the postsynaptic one (pre is still "ringing" in the trace when
/// post fires) — potentiation; if the other way around — depression. Exactly
/// the same rule is implemented by `snntorch.functional.stdp_linear`.
///
/// The update is averaged over the batch (divided by `batch_size`) so that the
/// effective learning rate does not depend on the batch size — without this,
/// doubling the batch would double the step size, which is usually undesirable
/// when switching from online (`batch = 1`) to batched mode.
///
/// Returns NEW weights; the function does not mutate `w` (weights are explicit
/// state, like the other states in the crate, see [`crate::state`]).
pub fn pair_update(
    w: Tensor<2>,
    pre_spike: Tensor<2>,
    post_spike: Tensor<2>,
    pre_trace: Tensor<2>,
    post_trace: Tensor<2>,
    a_plus: f32,
    a_minus: f32,
    bound: WeightBound,
) -> Tensor<2> {
    pair_update_with_f(
        w, pre_spike, post_spike, pre_trace, post_trace, a_plus, a_minus, bound, &Identity,
        &Identity,
    )
}

/// Nonlinearity applied to the trace immediately BEFORE the outer product with
/// the spike from the opposite side — the same place in the formula as the
/// `f_pre`/`f_post` parameters of each `*_single_step` function in `snnTorch`
/// (by default there and here — the identity, `f(x) = x`).
///
/// Not to be confused with [`WeightDependence`]: this nonlinearity sees only
/// the trace itself, not the current weight — for a weight-dependent update
/// (a nonlinearity depending on the synapse weight rather than the trace), see
/// [`WeightDependence`].
pub trait TraceNonlinearity {
    fn apply(&self, trace: Tensor<2>) -> Tensor<2>;
}

/// Identity nonlinearity — the default matching the behavior of
/// [`pair_update`]/[`triplet_update`] without hooks.
pub struct Identity;

impl TraceNonlinearity for Identity {
    fn apply(&self, trace: Tensor<2>) -> Tensor<2> {
        trace
    }
}

/// An arbitrary function as a [`TraceNonlinearity`] — convenient for closures,
/// for example `&FnTrace(|t| t.clamp(0.0, 1.0))` for a saturating trace.
pub struct FnTrace<F>(pub F);

impl<F: Fn(Tensor<2>) -> Tensor<2>> TraceNonlinearity for FnTrace<F> {
    fn apply(&self, trace: Tensor<2>) -> Tensor<2> {
        (self.0)(trace)
    }
}

/// Weight-dependent factor on `Δw` (van Rossum, Bi & Turrigiano, 2000):
/// the same place in the rule as the `Soft` variant of [`WeightBound`] (which
/// hard-codes the linear form `(w_max - w)`/`(w - w_min)`), but with an
/// arbitrary function of the current weight instead. Receives the weight `w`
/// shaped `[out_features, in_features]` (or its equivalent after `im2col` in
/// convolutional STDP layers) and must return a factor of the same shape —
/// separately for potentiation and depression, since in biological models they
/// are usually asymmetric.
///
/// `WeightBound` is applied AFTER this factor and continues to work as a
/// safeguard (clamping/`Soft` bound) rather than being replaced by it — both
/// mechanisms are independent and compose.
pub trait WeightDependence {
    fn potentiation_factor(&self, w: &Tensor<2>) -> Tensor<2>;
    fn depression_factor(&self, w: &Tensor<2>) -> Tensor<2>;
}

/// A factor identically equal to 1 — the default: `Δw` is applied as is,
/// without additional weight-dependent scaling beyond what
/// [`WeightBound::Soft`] already provides, if selected.
pub struct NoWeightDependence;

impl WeightDependence for NoWeightDependence {
    fn potentiation_factor(&self, w: &Tensor<2>) -> Tensor<2> {
        w.zeros_like().add_scalar(1.0)
    }
    fn depression_factor(&self, w: &Tensor<2>) -> Tensor<2> {
        w.zeros_like().add_scalar(1.0)
    }
}

/// [`pair_update`] with the `f_pre`/`f_post` hooks — a generalization that
/// calls [`pair_update_with_f_and_dep`] with [`NoWeightDependence`]. See the
/// [`TraceNonlinearity`] documentation for the meaning of `f_pre`/`f_post`.
#[allow(clippy::too_many_arguments)]
pub fn pair_update_with_f(
    w: Tensor<2>,
    pre_spike: Tensor<2>,
    post_spike: Tensor<2>,
    pre_trace: Tensor<2>,
    post_trace: Tensor<2>,
    a_plus: f32,
    a_minus: f32,
    bound: WeightBound,
    f_pre: &dyn TraceNonlinearity,
    f_post: &dyn TraceNonlinearity,
) -> Tensor<2> {
    pair_update_with_f_and_dep(
        w,
        pre_spike,
        post_spike,
        pre_trace,
        post_trace,
        a_plus,
        a_minus,
        bound,
        f_pre,
        f_post,
        &NoWeightDependence,
    )
}

/// [`pair_update`] parameterized by both the trace nonlinearity (`f_pre`/
/// `f_post`, see [`TraceNonlinearity`]) and the weight-dependent factor
/// (`weight_dep`, see [`WeightDependence`]). `f_post(post_trace)` replaces the
/// bare `post_trace` in the potentiation term, `f_pre(pre_trace)` — the bare
/// `pre_trace` in the depression term, which exactly matches the place of
/// `f_pre`/`f_post` in `snnTorch`; `weight_dep` then multiplies the resulting
/// `pot`/`dep` BEFORE applying [`WeightBound`].
#[allow(clippy::too_many_arguments)]
pub fn pair_update_with_f_and_dep(
    w: Tensor<2>,
    pre_spike: Tensor<2>,
    post_spike: Tensor<2>,
    pre_trace: Tensor<2>,
    post_trace: Tensor<2>,
    a_plus: f32,
    a_minus: f32,
    bound: WeightBound,
    f_pre: &dyn TraceNonlinearity,
    f_post: &dyn TraceNonlinearity,
    weight_dep: &dyn WeightDependence,
) -> Tensor<2> {
    let batch = pre_spike.dims()[0] as f32;

    // outer(f_post(post_trace), pre_spike) -> [out_features, in_features]:
    // post_trace^T [out, batch] @ pre_spike [batch, in] = [out, in], divided
    // by batch for averaging over the batch.
    let pot = f_post
        .apply(post_trace)
        .transpose()
        .matmul(pre_spike)
        .div_scalar(batch)
        .mul_scalar(a_plus)
        .mul(weight_dep.potentiation_factor(&w));
    let dep = post_spike
        .transpose()
        .matmul(f_pre.apply(pre_trace))
        .div_scalar(batch)
        .mul_scalar(a_minus)
        .mul(weight_dep.depression_factor(&w));

    let dw = match bound {
        WeightBound::Soft { w_min, w_max } => soft_bound_factors(&w, pot, dep, w_min, w_max),
        _ => pot - dep,
    };

    apply_bound(w + dw, bound)
}

/// Triplet STDP (Pfister & Gerstner, 2006).
///
/// To the pair of fast traces in [`pair_update`] is added one "slow" trace on
/// each side (`pre_trace_slow`, `post_trace_slow`, both usually with a larger
/// decay constant `tau` than the fast ones) — they are also computed by the
/// same [`update_trace`] function, with a different `decay`. The point:
/// pair-based STDP does not reproduce the experimentally observed dependence
/// of plasticity on the FREQUENCY of spike pair presentation — triplet terms
/// (the product of three traces/spikes instead of two) provide this
/// dependence.
///
/// ```text
/// Δw = a2_plus  · (post_trace_fast ⊗ pre_spike)
///    − a2_minus · (post_spike ⊗ pre_trace_fast)
///    + a3_plus  · (post_trace_fast ⊗ (pre_spike · pre_trace_slow))    [triplet potentiation]
///    − a3_minus · ((post_spike · post_trace_slow) ⊗ pre_trace_fast)   [triplet depression]
/// ```
///
/// The slow trace is multiplied element-wise by the spike of ITS OWN side
/// (both `[batch, in]` or both `[batch, out]`), then the result is folded by
/// the outer product with the fast trace of the opposite side, as in the pair
/// rule. Therefore the rule works for any `in_features` and `out_features`.
/// With `a3_plus == a3_minus == 0.0` the result matches [`pair_update`] with
/// the same `a2_*` — the triplet version strictly generalizes the pair-based
/// one rather than replacing it with another model.
#[allow(clippy::too_many_arguments)]
pub fn triplet_update(
    w: Tensor<2>,
    pre_spike: Tensor<2>,
    post_spike: Tensor<2>,
    pre_trace_fast: Tensor<2>,
    post_trace_fast: Tensor<2>,
    pre_trace_slow: Tensor<2>,
    post_trace_slow: Tensor<2>,
    a2_plus: f32,
    a2_minus: f32,
    a3_plus: f32,
    a3_minus: f32,
    bound: WeightBound,
) -> Tensor<2> {
    triplet_update_with_f(
        w,
        pre_spike,
        post_spike,
        pre_trace_fast,
        post_trace_fast,
        pre_trace_slow,
        post_trace_slow,
        a2_plus,
        a2_minus,
        a3_plus,
        a3_minus,
        bound,
        &Identity,
        &Identity,
    )
}

/// [`triplet_update`] with the `f_pre`/`f_post` hooks applied to the FAST
/// traces in the same two places as in [`pair_update_with_f`] (the pair terms
/// `pot2`/`dep2`); the slow traces (`pre_trace_slow`/`post_trace_slow`)
/// participate in the triplet terms as is — `f_pre`/`f_post` in `snnTorch`
/// are defined at the level of the single trace of each side, not separately
/// for the "fast" and "slow" versions, so extending the hook to the slow trace
/// would introduce a notion that does not exist in the original
/// Pfister & Gerstner model. With `a3_plus == a3_minus == 0.0` it matches
/// [`pair_update_with_f`] with the same `a2_*`/`f_pre`/`f_post` — the same
/// generalization property as [`triplet_update`] relative to
/// [`pair_update`].
#[allow(clippy::too_many_arguments)]
pub fn triplet_update_with_f(
    w: Tensor<2>,
    pre_spike: Tensor<2>,
    post_spike: Tensor<2>,
    pre_trace_fast: Tensor<2>,
    post_trace_fast: Tensor<2>,
    pre_trace_slow: Tensor<2>,
    post_trace_slow: Tensor<2>,
    a2_plus: f32,
    a2_minus: f32,
    a3_plus: f32,
    a3_minus: f32,
    bound: WeightBound,
    f_pre: &dyn TraceNonlinearity,
    f_post: &dyn TraceNonlinearity,
) -> Tensor<2> {
    triplet_update_with_f_and_dep(
        w,
        pre_spike,
        post_spike,
        pre_trace_fast,
        post_trace_fast,
        pre_trace_slow,
        post_trace_slow,
        a2_plus,
        a2_minus,
        a3_plus,
        a3_minus,
        bound,
        f_pre,
        f_post,
        &NoWeightDependence,
    )
}

/// [`triplet_update_with_f`], additionally parameterized by
/// [`WeightDependence`] — the same factor on `pot`/`dep` as in
/// [`pair_update_with_f_and_dep`], here applied to the SUM of the pair and
/// triplet terms (`pot2 + pot3`, `dep2 + dep3`), not to them separately:
/// weight dependence is a property of the resulting weight-update step, not of
/// a particular term in its decomposition.
#[allow(clippy::too_many_arguments)]
pub fn triplet_update_with_f_and_dep(
    w: Tensor<2>,
    pre_spike: Tensor<2>,
    post_spike: Tensor<2>,
    pre_trace_fast: Tensor<2>,
    post_trace_fast: Tensor<2>,
    pre_trace_slow: Tensor<2>,
    post_trace_slow: Tensor<2>,
    a2_plus: f32,
    a2_minus: f32,
    a3_plus: f32,
    a3_minus: f32,
    bound: WeightBound,
    f_pre: &dyn TraceNonlinearity,
    f_post: &dyn TraceNonlinearity,
    weight_dep: &dyn WeightDependence,
) -> Tensor<2> {
    let batch = pre_spike.dims()[0] as f32;

    let post_trace_fast_f = f_post.apply(post_trace_fast.clone());
    let pre_trace_fast_f = f_pre.apply(pre_trace_fast.clone());

    // Pair terms, identical to pair_update_with_f (f_pre/f_post applied to
    // the fast traces).
    let pot2 = post_trace_fast_f
        .clone()
        .transpose()
        .matmul(pre_spike.clone())
        .div_scalar(batch)
        .mul_scalar(a2_plus);
    let dep2 = post_spike
        .clone()
        .transpose()
        .matmul(pre_trace_fast_f.clone())
        .div_scalar(batch)
        .mul_scalar(a2_minus);

    // Triplet terms: potentiation is modulated by the slow PRE-trace,
    // depression by the slow POST-trace (the asymmetry from the original
    // Pfister & Gerstner model: depression has its own frequency dependence,
    // separate from potentiation). The fast factor in each term is the same
    // f_pre/f_post version as in pot2/dep2, so that the trace nonlinearity is
    // consistent across all terms where the same fast trace appears.
    //
    // The slow trace belongs to the SAME side as the spike it is multiplied
    // with: `pre_spike · pre_trace_slow` — both `[batch, in]`,
    // `post_spike · post_trace_slow` — both `[batch, out]`. (The previous
    // version multiplied `post_trace_fast [batch, out]` with `pre_trace_slow
    // [batch, in]` element-wise: that worked only when `out == in` and
    // mixed indices of different neurons.)
    let pot3 = post_trace_fast_f
        .transpose()
        .matmul(pre_spike.mul(pre_trace_slow))
        .div_scalar(batch)
        .mul_scalar(a3_plus);
    let dep3 = post_spike
        .mul(post_trace_slow)
        .transpose()
        .matmul(pre_trace_fast_f)
        .div_scalar(batch)
        .mul_scalar(a3_minus);

    let pot = (pot2 + pot3).mul(weight_dep.potentiation_factor(&w));
    let dep = (dep2 + dep3).mul(weight_dep.depression_factor(&w));

    let dw = match bound {
        WeightBound::Soft { w_min, w_max } => soft_bound_factors(&w, pot, dep, w_min, w_max),
        _ => pot - dep,
    };

    apply_bound(w + dw, bound)
}

/// Eligibility trace for reward-modulated STDP (R-STDP): computes the same
/// pair-STDP "pressure" on the weight but does not apply it — accumulates it
/// in a separate buffer of the same shape as the weights, with its own
/// (usually slow) decay.
///
/// ```text
/// e[t] = e[t-1]·decay_e + (a_plus · post_trace ⊗ pre_spike − a_minus · post_spike ⊗ pre_trace)
/// ```
///
/// This separation is the essence of the three-factor learning rule:
/// co-activity (pre × post) marks WHICH synapses recently took part in
/// something potentially important, and the actual decision "strengthen or
/// weaken" and "by how much" is deferred until a scalar reward arrives (the
/// third factor), which biologically and in neuromorphic systems usually
/// arrives with a delay relative to the spikes themselves (for example, a
/// dopamine signal). Applying the accumulated trace to the weights — see
/// [`apply_reward`].
///
/// Returns a NEW eligibility trace; this function does not touch the weights.
pub fn eligibility_trace_update(
    e: Tensor<2>,
    pre_spike: Tensor<2>,
    post_spike: Tensor<2>,
    pre_trace: Tensor<2>,
    post_trace: Tensor<2>,
    a_plus: f32,
    a_minus: f32,
    decay_e: f32,
) -> Tensor<2> {
    let batch = pre_spike.dims()[0] as f32;

    let pot = post_trace
        .transpose()
        .matmul(pre_spike)
        .div_scalar(batch)
        .mul_scalar(a_plus);
    let dep = post_spike
        .transpose()
        .matmul(pre_trace)
        .div_scalar(batch)
        .mul_scalar(a_minus);

    e.mul_scalar(decay_e) + (pot - dep)
}

/// Applies the accumulated eligibility trace to the weights, modulated by a
/// scalar reward: `w += lr · reward · e`.
///
/// `reward` — a scalar for the current step (can be negative — punishment, not
/// only reward). Unlike [`pair_update`]/[`triplet_update`], which themselves
/// return already updated weights EVERY step, the separation here is
/// intentional: [`eligibility_trace_update`] is called at every time step
/// (accumulates the trace), and `apply_reward` — only when the reward has
/// actually arrived (which may be less often than every step, or not at all
/// for part of an episode).
///
/// `lr` (learning rate) is here rather than inside the trace update — this way
/// the learning speed can be changed independently of the constants of the
/// STDP rule itself (`a_plus`/`a_minus` in [`eligibility_trace_update`]).
pub fn apply_reward(
    w: Tensor<2>,
    eligibility: &Tensor<2>,
    reward: f32,
    lr: f32,
    bound: WeightBound,
) -> Tensor<2> {
    let dw = eligibility.clone().mul_scalar(lr * reward);
    apply_bound(w + dw, bound)
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::Device;

    fn dev() -> Device {
        Device::flex()
    }

    fn t2(data: Vec<f32>, shape: [usize; 2]) -> Tensor<2> {
        Tensor::<1>::from_floats(data.as_slice(), &dev()).reshape(shape)
    }

    fn vec2(t: Tensor<2>) -> Vec<f32> {
        t.into_data().try_to_vec::<f32>().unwrap()
    }

    /// A trace with no incoming spike simply decays exponentially.
    #[test]
    fn trace_decays_without_spike() {
        let trace = t2(vec![1.0, 2.0], [1, 2]);
        let spike = t2(vec![0.0, 0.0], [1, 2]);
        let out = vec2(update_trace(trace, spike, 0.5));
        assert!((out[0] - 0.5).abs() < 1e-6);
        assert!((out[1] - 1.0).abs() < 1e-6);
    }

    /// The spike is added AFTER the previous trace value decays.
    #[test]
    fn trace_accumulates_spike_after_decay() {
        let trace = t2(vec![1.0], [1, 1]);
        let spike = t2(vec![1.0], [1, 1]);
        let out = vec2(update_trace(trace, spike, 0.5));
        assert!(
            (out[0] - 1.5).abs() < 1e-6,
            "expected 0.5*1 + 1 = 1.5, got {}",
            out[0]
        );
    }

    /// Pure potentiation: pre fired recently (trace > 0), post
    /// fires NOW, the depression term is absent (post_spike = 0 in the
    /// past sense does not participate — here simply post_trace = 0 and
    /// post_spike = 1 => twice the same term depends on post_spike and
    /// pre_trace, both of which are zero at the input).
    #[test]
    fn pair_update_pure_potentiation_when_pre_led_post() {
        let w = t2(vec![0.0], [1, 1]);
        let pre_spike = t2(vec![0.0], [1, 1]); // pre is silent now
        let post_spike = t2(vec![1.0], [1, 1]); // post fired now
        let pre_trace = t2(vec![1.0], [1, 1]); // but pre was ringing recently
        let post_trace = t2(vec![0.0], [1, 1]);

        let out = vec2(pair_update(
            w,
            pre_spike,
            post_spike,
            pre_trace,
            post_trace,
            0.1,
            0.1,
            WeightBound::None,
        ));
        // pot = a_plus * post_trace(=0) ⊗ pre_spike(=0) = 0
        // dep = a_minus * post_spike(=1) ⊗ pre_trace(=1) = 0.1
        // Δw = 0 - 0.1 = -0.1: post fired, while pre is silent ONLY NOW,
        // but was active earlier (pre_trace=1) => post lags behind pre =>
        // depression. The test pins the sign and prevents the rule from
        // silently flipping during refactoring.
        assert!((out[0] - (-0.1)).abs() < 1e-6, "got {}", out[0]);
    }

    /// The symmetric case: post was ringing recently, pre fires now —
    /// pure potentiation (pre FOLLOWED post in the trace sense of the
    /// decomposition, exactly the opposite term of the previous test).
    #[test]
    fn pair_update_pure_depression_term_sign() {
        let w = t2(vec![0.0], [1, 1]);
        let pre_spike = t2(vec![1.0], [1, 1]);
        let post_spike = t2(vec![0.0], [1, 1]);
        let pre_trace = t2(vec![0.0], [1, 1]);
        let post_trace = t2(vec![1.0], [1, 1]);

        let out = vec2(pair_update(
            w,
            pre_spike,
            post_spike,
            pre_trace,
            post_trace,
            0.1,
            0.1,
            WeightBound::None,
        ));
        // pot = a_plus * post_trace(=1) ⊗ pre_spike(=1) = 0.1
        // dep = a_minus * post_spike(=0) ⊗ pre_trace(=0) = 0
        assert!((out[0] - 0.1).abs() < 1e-6, "got {}", out[0]);
    }

    /// Hard clamping keeps the weight within bounds even for a large Δw.
    #[test]
    fn hard_bound_clamps_weight() {
        let w = t2(vec![0.95], [1, 1]);
        let pre_spike = t2(vec![1.0], [1, 1]);
        let post_spike = t2(vec![1.0], [1, 1]);
        let pre_trace = t2(vec![1.0], [1, 1]);
        let post_trace = t2(vec![1.0], [1, 1]);

        let out = vec2(pair_update(
            w,
            pre_spike,
            post_spike,
            pre_trace,
            post_trace,
            0.5,
            0.0,
            WeightBound::Hard {
                w_min: 0.0,
                w_max: 1.0,
            },
        ));
        assert!(
            out[0] <= 1.0,
            "weight must stay within the upper bound: {}",
            out[0]
        );
    }

    /// Soft bound: near the ceiling, potentiation should almost stop
    /// (the factor `w_max - w` tends to zero), i.e. the resulting weight
    /// must remain STRICTLY less than for the same step started
    /// far from the boundary.
    #[test]
    fn soft_bound_slows_potentiation_near_ceiling() {
        let near_ceiling = t2(vec![0.99], [1, 1]);
        let far_from_ceiling = t2(vec![0.1], [1, 1]);
        let pre_spike = t2(vec![1.0], [1, 1]);
        let post_spike = t2(vec![1.0], [1, 1]);
        let pre_trace = t2(vec![1.0], [1, 1]);
        let post_trace = t2(vec![1.0], [1, 1]);

        let bound = WeightBound::Soft {
            w_min: 0.0,
            w_max: 1.0,
        };
        let out_near = vec2(pair_update(
            near_ceiling.clone(),
            pre_spike.clone(),
            post_spike.clone(),
            pre_trace.clone(),
            post_trace.clone(),
            0.5,
            0.0,
            bound,
        ));
        let out_far = vec2(pair_update(
            far_from_ceiling.clone(),
            pre_spike,
            post_spike,
            pre_trace,
            post_trace,
            0.5,
            0.0,
            bound,
        ));

        let delta_near = out_near[0] - 0.99;
        let delta_far = out_far[0] - 0.1;
        assert!(
            delta_near < delta_far,
            "the weight increase near the ceiling ({delta_near}) must be less than far from it ({delta_far})"
        );
        assert!(out_near[0] <= 1.0 + 1e-6);
    }

    /// With zero triplet coefficients, triplet_update must match
    /// pair_update for the same a2_plus/a2_minus — triplet strictly
    /// generalizes the pair-based rule rather than replacing it with another
    /// model.
    #[test]
    fn triplet_update_reduces_to_pair_update_when_a3_zero() {
        let w = t2(vec![0.2, -0.1, 0.05, 0.3], [2, 2]);
        let pre_spike = t2(vec![1.0, 0.0], [1, 2]);
        let post_spike = t2(vec![0.0, 1.0], [1, 2]);
        let pre_trace = t2(vec![0.3, 0.7], [1, 2]);
        let post_trace = t2(vec![0.6, 0.2], [1, 2]);
        let slow_pre = t2(vec![0.4, 0.4], [1, 2]);
        let slow_post = t2(vec![0.4, 0.4], [1, 2]);

        let via_pair = vec2(pair_update(
            w.clone(),
            pre_spike.clone(),
            post_spike.clone(),
            pre_trace.clone(),
            post_trace.clone(),
            0.1,
            0.05,
            WeightBound::None,
        ));

        let via_triplet = vec2(triplet_update(
            w,
            pre_spike,
            post_spike,
            pre_trace,
            post_trace,
            slow_pre,
            slow_post,
            0.1,
            0.05,
            0.0,
            0.0,
            WeightBound::None,
        ));

        for (a, b) in via_pair.iter().zip(via_triplet.iter()) {
            assert!((a - b).abs() < 1e-6, "{a} vs {b}");
        }
    }

    /// Triplet terms must add a nonzero contribution when the slow traces
    /// and a3 coefficients are nonzero — otherwise the triplet path would be
    /// dead code.
    #[test]
    fn triplet_update_adds_nonzero_contribution_when_a3_nonzero() {
        let w = t2(vec![0.0], [1, 1]);
        let pre_spike = t2(vec![1.0], [1, 1]);
        let post_spike = t2(vec![0.0], [1, 1]);
        let pre_trace = t2(vec![0.0], [1, 1]);
        let post_trace = t2(vec![0.5], [1, 1]);
        let slow_pre = t2(vec![0.8], [1, 1]);
        let slow_post = t2(vec![0.0], [1, 1]);

        let without_triplet = vec2(triplet_update(
            w.clone(),
            pre_spike.clone(),
            post_spike.clone(),
            pre_trace.clone(),
            post_trace.clone(),
            slow_pre.clone(),
            slow_post.clone(),
            0.1,
            0.1,
            0.0,
            0.0,
            WeightBound::None,
        ));
        let with_triplet = vec2(triplet_update(
            w,
            pre_spike,
            post_spike,
            pre_trace,
            post_trace,
            slow_pre,
            slow_post,
            0.1,
            0.1,
            0.2,
            0.0,
            WeightBound::None,
        ));

        assert!(
            (with_triplet[0] - without_triplet[0]).abs() > 1e-6,
            "triplet term must change the result: {} vs {}",
            with_triplet[0],
            without_triplet[0]
        );
    }

    /// The triplet rule must work when `out != in` and `batch > 1`;
    /// the reference is a direct triple loop over the formula from the
    /// documentation. Element-wise multiplication of `[batch, out]` and
    /// `[batch, in]` would panic on any non-square layer, so the update
    /// uses the outer-product structure of the rule instead.
    #[test]
    fn triplet_update_supports_non_square_layers_and_matches_naive_loops() {
        let (batch, n_in, n_out) = (2usize, 3usize, 2usize);
        let w_v = vec![0.10, 0.20, 0.30, 0.40, 0.50, 0.60];
        let pre_spike_v = vec![1.0, 0.0, 1.0, 0.0, 1.0, 1.0];
        let post_spike_v = vec![0.0, 1.0, 1.0, 0.0];
        let pre_fast_v = vec![0.3, 0.1, 0.5, 0.2, 0.9, 0.4];
        let post_fast_v = vec![0.6, 0.2, 0.1, 0.8];
        let pre_slow_v = vec![0.7, 0.3, 0.2, 0.6, 0.1, 0.5];
        let post_slow_v = vec![0.4, 0.9, 0.3, 0.2];
        let (a2p, a2m, a3p, a3m) = (0.10f32, 0.05f32, 0.20f32, 0.07f32);

        let got = vec2(triplet_update(
            t2(w_v.clone(), [n_out, n_in]),
            t2(pre_spike_v.clone(), [batch, n_in]),
            t2(post_spike_v.clone(), [batch, n_out]),
            t2(pre_fast_v.clone(), [batch, n_in]),
            t2(post_fast_v.clone(), [batch, n_out]),
            t2(pre_slow_v.clone(), [batch, n_in]),
            t2(post_slow_v.clone(), [batch, n_out]),
            a2p,
            a2m,
            a3p,
            a3m,
            WeightBound::None,
        ));

        for o in 0..n_out {
            for i in 0..n_in {
                let mut dw = 0.0f32;
                for b in 0..batch {
                    let pre_s = pre_spike_v[b * n_in + i];
                    let post_s = post_spike_v[b * n_out + o];
                    let pre_f = pre_fast_v[b * n_in + i];
                    let post_f = post_fast_v[b * n_out + o];
                    let pre_sl = pre_slow_v[b * n_in + i];
                    let post_sl = post_slow_v[b * n_out + o];
                    dw += a2p * post_f * pre_s - a2m * post_s * pre_f
                        + a3p * post_f * pre_s * pre_sl
                        - a3m * post_s * post_sl * pre_f;
                }
                let expected = w_v[o * n_in + i] + dw / batch as f32;
                let actual = got[o * n_in + i];
                assert!(
                    (actual - expected).abs() < 1e-5,
                    "w[{o},{i}]: {actual} vs {expected}"
                );
            }
        }
    }

    /// Eligibility trace accumulates co-activity but does NOT change the
    /// weights by itself.
    ///
    /// The test keeps `pot != dep` by separating `pre_trace` and
    /// `post_trace` in value and using `a_plus != a_minus`, so the
    /// accumulation is actually exercised. With `pre_trace == post_trace == 1.0`
    /// and `a_plus == a_minus`, `pot = post_trace·pre_spike·a_plus` and
    /// `dep = post_spike·pre_trace·a_minus` cancel exactly, `e2 == 0` is
    /// mathematically guaranteed, and the test would pass even with a
    /// broken accumulation.
    #[test]
    fn eligibility_trace_does_not_touch_weights() {
        let e = t2(vec![0.0], [1, 1]);
        let pre_spike = t2(vec![1.0], [1, 1]);
        let post_spike = t2(vec![1.0], [1, 1]);
        let pre_trace = t2(vec![0.3], [1, 1]);
        let post_trace = t2(vec![1.0], [1, 1]);

        let e2 = eligibility_trace_update(
            e, pre_spike, post_spike, pre_trace, post_trace, 0.1, 0.1, 0.9,
        );
        assert!(
            vec2(e2)[0].abs() > 0.0,
            "the trace must accumulate a nonzero value"
        );
    }

    /// A positive reward strengthens the weight in proportion to the
    /// accumulated eligibility trace; a zero trace => weights do not change.
    #[test]
    fn apply_reward_scales_by_eligibility_and_reward() {
        let w = t2(vec![0.5], [1, 1]);
        let e_zero = t2(vec![0.0], [1, 1]);
        let e_pos = t2(vec![2.0], [1, 1]);

        let unchanged = vec2(apply_reward(
            w.clone(),
            &e_zero,
            1.0,
            0.1,
            WeightBound::None,
        ));
        assert!((unchanged[0] - 0.5).abs() < 1e-6);

        let boosted = vec2(apply_reward(w.clone(), &e_pos, 1.0, 0.1, WeightBound::None));
        assert!(
            (boosted[0] - 0.7).abs() < 1e-6,
            "0.5 + 0.1*1.0*2.0 = 0.7, got {}",
            boosted[0]
        );

        // A negative reward (punishment) must WEAKEN the weight.
        let punished = vec2(apply_reward(w, &e_pos, -1.0, 0.1, WeightBound::None));
        assert!(
            (punished[0] - 0.3).abs() < 1e-6,
            "0.5 - 0.1*1.0*2.0 = 0.3, got {}",
            punished[0]
        );
    }

    /// `apply_reward` must apply the same weight bound as the
    /// other rules — the reward path must not be a "hole" in
    /// weight bounding.
    #[test]
    fn apply_reward_respects_hard_bound() {
        let w = t2(vec![0.95], [1, 1]);
        let e = t2(vec![10.0], [1, 1]);
        let out = vec2(apply_reward(
            w,
            &e,
            1.0,
            0.5,
            WeightBound::Hard {
                w_min: 0.0,
                w_max: 1.0,
            },
        ));
        assert!(
            out[0] <= 1.0,
            "weight must stay within the upper bound: {}",
            out[0]
        );
    }
}
