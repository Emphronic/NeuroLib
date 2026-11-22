//! Heterogeneous models of dynamic graph neurons.
//!
//! There are three independent concepts here:
//!
//! * [`NeuronRole`] — location of the neuron in the network (`Input`/`Hidden`/`Output`);
//! * [`NeuronModel<T>`] – law of dynamics (LIF, ALIF, Izhikevich, AdEx, HH);
//! * [`NeuronPolarity`] - physiological sign of influence on receivers
//!   (excitatory, inhibitory, modulating).
//!
//! The model is a regular enumeration, not `Box<dyn Trait>`: the state of each
//! neuron lies nearby in [`NeuronState<T>`], so there is no virtual call to
//! every event, and the size of the state is predictable and serializable without
//! dynamic dispatch.
//!
//! # Units and numerical schemes
//!
//! * LIF/ALIF - dimensionless discrete shape, as in the rest of the frame:
//!   `mem' = β·mem + I`; ALIF also has `w' = ρ·w + a·spike`.
//! * Izhikevich - mV and ms, quadratic membrane, `substeps` sub-steps by `v`
//!   per step (default 2, as in the reference implementation).
//! * AdEx — mV, ms, pF/nS/pA; `dt` ms is divided into `substeps` sub-steps,
//!   reset is checked at each sub-step.
//! * Hodgkin–Huxley — mV, ms, μA/cm²; gate variables are updated
//!   exponential Euler, the membrane - direct Euler according to `substeps`.
//!
//! Scalar formulas repeat Burn-layers `neuron::*`, but do not depend on
//! tensors and work on one neuron.

use crate::state::ResetMode;

use super::custom::{CustomNeuronInstance, CustomStateEntryBox, NeuronStepContext};
use super::error::{StructuralError, StructuralResult};
use crate::scalar::Scalar;

use super::topology::ParamLimits;

/// The role of a neuron in a network.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum NeuronRole {
    /// Input: the spike is set externally, there are no dynamics or incoming synapses.
    Input,
    /// Hidden neuron; it can be created, cloned and deleted
    /// structural plasticity.
    Hidden,
    /// Output neuron; protected from deletion.
    Output,
}

/// Previous role name (API compatible before role/model separation).
pub type NeuronKind = NeuronRole;

impl NeuronRole {
    /// Short name for error messages.
    pub fn as_str(self) -> &'static str {
        match self {
            NeuronRole::Input => "input",
            NeuronRole::Hidden => "hidden",
            NeuronRole::Output => "output",
        }
    }

    pub(crate) fn code(self) -> u8 {
        match self {
            NeuronRole::Input => 0,
            NeuronRole::Hidden => 1,
            NeuronRole::Output => 2,
        }
    }

    pub(crate) fn from_code(c: u8) -> Option<Self> {
        match c {
            0 => Some(NeuronRole::Input),
            1 => Some(NeuronRole::Hidden),
            2 => Some(NeuronRole::Output),
            _ => None,
        }
    }
}

/// Physiological polarity of a neuron.
///
/// The polarity belongs to the **source** of the spike and determines how much its weight
/// affects the receiver:
///
/// * [`Excitatory`](Self::Excitatory) - the weight is added to the current as is;
/// * [`Inhibitory`](Self::Inhibitory) - the weight sign is reversed;
/// * [`Modulatory`](Self::Modulatory) - the weight does not go directly into the current, but
///   increases the damping gain of the receiver (`modulation`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Default)]
pub enum NeuronPolarity {
    /// Excitatory neuron.
    #[default]
    Excitatory,
    /// Inhibitory neuron.
    Inhibitory,
    /// Modulating neuron (strengthens the receiver current, rather than creating its own).
    Modulatory,
}

impl NeuronPolarity {
    /// Short name.
    pub fn as_str(self) -> &'static str {
        match self {
            NeuronPolarity::Excitatory => "excitatory",
            NeuronPolarity::Inhibitory => "inhibitory",
            NeuronPolarity::Modulatory => "modulatory",
        }
    }

    /// The sign of the outgoing weights when the Dale principle is turned on: `+1` for
    /// excitatory and modulating sources, `−1` for inhibitory ones.
    pub fn sign<T: Scalar>(self) -> T {
        match self {
            NeuronPolarity::Inhibitory => -T::ONE,
            _ => T::ONE,
        }
    }

    /// Bring the weight to the polarity sign, preserving the modulus (this is how the weight is obtained
    /// new or redirected communication).
    pub fn resign<T: Scalar>(self, weight: T) -> T {
        self.sign::<T>() * weight.abs()
    }

    /// Limit the weight to the permissible polarity of the semi-axis without changing the sign:
    /// the weight that would “pass” through zero becomes zero.
    pub fn clamp_sign<T: Scalar>(self, weight: T) -> T {
        if self == NeuronPolarity::Inhibitory {
            weight.min(T::ZERO)
        } else {
            weight.max(T::ZERO)
        }
    }

    pub(crate) fn code(self) -> u8 {
        match self {
            NeuronPolarity::Excitatory => 0,
            NeuronPolarity::Inhibitory => 1,
            NeuronPolarity::Modulatory => 2,
        }
    }

    pub(crate) fn from_code(c: u8) -> Option<Self> {
        match c {
            0 => Some(NeuronPolarity::Excitatory),
            1 => Some(NeuronPolarity::Inhibitory),
            2 => Some(NeuronPolarity::Modulatory),
            _ => None,
        }
    }
}

/// Parameters of the LIF neuron: `U' = β·U + I`, spike at `U ≥ θ`.
///
/// Same meaning as `neuron::LeakyConfig` (`beta_init`, `threshold`,
/// `reset_mode`), but here these are scalars of a specific neuron.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LifParams<T = f32> {
    /// Membrane damping β ∈ [0, 1).
    pub beta: T,
    /// Threshold θ > 0.
    pub threshold: T,
    /// Reset method.
    pub reset: ResetMode,
}

impl<T: Scalar> LifParams<T> {
    /// Parameters with `ResetMode::Subtract`.
    pub fn new(beta: T, threshold: T) -> Self {
        Self {
            beta,
            threshold,
            reset: ResetMode::Subtract,
        }
    }

    pub(crate) fn validate(&self, lim: &ParamLimits<T>) -> StructuralResult<()> {
        if !self.beta.is_finite() || self.beta < lim.beta_min || self.beta > lim.beta_max {
            return Err(StructuralError::param(
                "beta",
                format!("{} out [{}, {}]", self.beta, lim.beta_min, lim.beta_max),
            ));
        }
        if !self.threshold.is_finite()
            || self.threshold < lim.threshold_min
            || self.threshold > lim.threshold_max
        {
            return Err(StructuralError::param(
                "threshold",
                format!(
                    "{} out [{}, {}]",
                    self.threshold, lim.threshold_min, lim.threshold_max
                ),
            ));
        }
        Ok(())
    }
}

impl<T: Scalar> Default for LifParams<T> {
    fn default() -> Self {
        Self::new(T::c(0.9), T::ONE)
    }
}

/// Adaptive LIF (ALIF) parameters: LIF with slow adaptive current.
///
/// ```text
/// mem' = β·mem + I − w
/// w'   = ρ·w + a·spike
/// ```
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AdaptiveLifParams<T = f32> {
    /// Membrane damping β ∈ [0, 1).
    pub beta: T,
    /// Attenuation of the adaptation current ρ ∈ [0, 1).
    pub rho: T,
    /// Increase in adaptation to spike `a ≥ 0`.
    pub adapt_increment: T,
    /// Threshold θ > 0.
    pub threshold: T,
    /// Membrane reset method.
    pub reset: ResetMode,
}

impl<T: Scalar> AdaptiveLifParams<T> {
    /// Parameters with `ResetMode::Subtract`.
    pub fn new(beta: T, rho: T, threshold: T) -> Self {
        Self {
            beta,
            rho,
            adapt_increment: T::c(0.1),
            threshold,
            reset: ResetMode::Subtract,
        }
    }

    /// Slow adaptation: `ρ` is close to 1, the increase is moderate.
    ///
    /// A convenient preset for structural growth that needs
    /// a “specialized” slowly adapting neuron.
    pub fn slow_adaptation() -> Self {
        Self {
            beta: T::c(0.9),
            rho: T::c(0.99),
            adapt_increment: T::c(0.1),
            threshold: T::ONE,
            reset: ResetMode::Subtract,
        }
    }

    pub(crate) fn validate(&self, lim: &ParamLimits<T>) -> StructuralResult<()> {
        LifParams {
            beta: self.beta,
            threshold: self.threshold,
            reset: self.reset,
        }
        .validate(lim)?;
        if !self.rho.is_finite() || !(self.rho >= T::ZERO && self.rho < T::ONE) {
            return Err(StructuralError::param(
                "rho",
                format!("{} outside [0, 1)", self.rho),
            ));
        }
        if !self.adapt_increment.is_finite() || self.adapt_increment < T::ZERO {
            return Err(StructuralError::param(
                "adapt_increment",
                "must be finite and >= 0",
            ));
        }
        Ok(())
    }
}

impl<T: Scalar> Default for AdaptiveLifParams<T> {
    fn default() -> Self {
        Self {
            beta: T::c(0.9),
            rho: T::c(0.95),
            adapt_increment: T::c(0.1),
            threshold: T::ONE,
            reset: ResetMode::Subtract,
        }
    }
}

/// Parameters of the Izikevich neuron.
///
/// `v' = 0.04·v² + 5·v + 140 − u + I`, `u' = a·(b·v − u)`,
/// reset `v ← c`, `u ← u + d` at `v ≥ threshold`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct IzhikevichParams<T = f32> {
    /// Recovery rate `u`.
    pub a: T,
    /// Sensitivity of `u` to `v`.
    pub b: T,
    /// Reset potential `v` after a spike, mV.
    pub c: T,
    /// Increase in `u` after the spike.
    pub d: T,
    /// Spike detection threshold, mV.
    pub threshold: T,
    /// The number of sub-steps by `v` per network step.
    pub substeps: usize,
}

impl<T: Scalar> IzhikevichParams<T> {
    /// Regular Spiking is a typical cortical pyramidal neuron.
    pub fn regular_spiking() -> Self {
        Self {
            a: T::c(0.02),
            b: T::c(0.2),
            c: -T::c(65.0),
            d: T::c(8.0),
            threshold: T::c(30.0),
            substeps: 2,
        }
    }

    /// Fast Spiking - fast spike with almost no adaptation.
    pub fn fast_spiking() -> Self {
        Self {
            a: T::c(0.1),
            b: T::c(0.2),
            c: -T::c(65.0),
            d: T::c(2.0),
            threshold: T::c(30.0),
            substeps: 2,
        }
    }

    pub(crate) fn validate(&self) -> StructuralResult<()> {
        let finite = |name: &'static str, v: T| -> StructuralResult<()> {
            if v.is_finite() {
                Ok(())
            } else {
                Err(StructuralError::param(name, "non-numeric value"))
            }
        };
        finite("a", self.a)?;
        finite("b", self.b)?;
        finite("c", self.c)?;
        finite("d", self.d)?;
        finite("threshold", self.threshold)?;
        if self.a <= T::ZERO || self.a > T::ONE {
            return Err(StructuralError::param("a", "must be in (0, 1]"));
        }
        if self.b < T::ZERO {
            return Err(StructuralError::param("b", "must be >= 0"));
        }
        if self.d < T::ZERO {
            return Err(StructuralError::param("d", "must be >= 0"));
        }
        if self.threshold <= self.c {
            return Err(StructuralError::param(
                "threshold",
                "must be greater than the reset potential c",
            ));
        }
        if self.substeps == 0 || self.substeps > 1024 {
            return Err(StructuralError::param("substeps", "should be in [1, 1024]"));
        }
        Ok(())
    }
}

impl<T: Scalar> Default for IzhikevichParams<T> {
    fn default() -> Self {
        Self::regular_spiking()
    }
}

/// Parameters AdEx (Brette & Gerstner, 2005), units - see module `neuron::adex`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AdExParams<T = f32> {
    /// Membrane capacity `C`, pF.
    pub c: T,
    /// Leakage conductivity `g_L`, nS.
    pub g_l: T,
    /// Resting potential `E_L`, mV.
    pub e_l: T,
    /// Exponent threshold `V_T`, mV.
    pub v_t: T,
    /// Sharpness of the exponential `Δ_T`, mV.
    pub delta_t: T,
    /// Subthreshold adaptation `a`, nSm.
    pub a: T,
    /// Adaptation time constant `τ_w`, ms.
    pub tau_w: T,
    /// Increase in adaptation current during spike `b`, pA.
    pub b: T,
    /// Reset potential `V_reset`, mV.
    pub v_reset: T,
    /// Spike detection threshold `V_peak`, mV.
    pub v_peak: T,
    /// Step duration, ms.
    pub dt: T,
    /// Number of Euler sub-steps per step.
    pub substeps: usize,
    /// The upper limit of the exponent.
    pub exp_clip: T,
}

impl<T: Scalar> Default for AdExParams<T> {
    fn default() -> Self {
        Self {
            c: T::c(281.0),
            g_l: T::c(30.0),
            e_l: -T::c(70.6),
            v_t: -T::c(50.4),
            delta_t: T::c(2.0),
            a: T::c(4.0),
            tau_w: T::c(144.0),
            b: T::c(80.5),
            v_reset: -T::c(70.6),
            v_peak: T::ZERO,
            dt: T::ONE,
            substeps: 4,
            exp_clip: T::c(20.0),
        }
    }
}

impl<T: Scalar> AdExParams<T> {
    pub(crate) fn validate(&self) -> StructuralResult<()> {
        let finite = |name: &'static str, v: T| -> StructuralResult<()> {
            if v.is_finite() {
                Ok(())
            } else {
                Err(StructuralError::param(name, "non-numeric value"))
            }
        };
        for (name, v) in [
            ("c", self.c),
            ("g_l", self.g_l),
            ("e_l", self.e_l),
            ("v_t", self.v_t),
            ("delta_t", self.delta_t),
            ("a", self.a),
            ("tau_w", self.tau_w),
            ("b", self.b),
            ("v_reset", self.v_reset),
            ("v_peak", self.v_peak),
            ("dt", self.dt),
            ("exp_clip", self.exp_clip),
        ] {
            finite(name, v)?;
        }
        if self.c <= T::ZERO
            || self.g_l <= T::ZERO
            || self.delta_t <= T::ZERO
            || self.tau_w <= T::ZERO
        {
            return Err(StructuralError::param("c/g_l/delta_t/tau_w", "must be > 0"));
        }
        if self.dt <= T::ZERO {
            return Err(StructuralError::param("dt", "must be > 0"));
        }
        if self.substeps == 0 || self.substeps > 1024 {
            return Err(StructuralError::param("substeps", "should be in [1, 1024]"));
        }
        if self.exp_clip <= T::ZERO {
            return Err(StructuralError::param("exp_clip", "must be > 0"));
        }
        if self.a < T::ZERO || self.b < T::ZERO {
            return Err(StructuralError::param("a/b", "must be >= 0"));
        }
        if self.v_peak <= self.v_reset {
            return Err(StructuralError::param(
                "v_peak",
                "should be greater than v_reset",
            ));
        }
        Ok(())
    }
}

/// Hodgkin–Huxley parameters, units - see module `neuron::hodgkin_huxley`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HodgkinHuxleyParams<T = f32> {
    /// Membrane capacity `C`, µF/cm².
    pub c: T,
    /// Maximum sodium conductivity `g_Na`, mS/cm².
    pub g_na: T,
    /// Maximum potassium conductivity `g_K`, mS/cm².
    pub g_k: T,
    /// Leakage conductivity `g_L`, mS/cm².
    pub g_l: T,
    /// Equilibrium potential of sodium `E_Na`, mV.
    pub e_na: T,
    /// Equilibrium potassium potential `E_K`, mV.
    pub e_k: T,
    /// Leakage potential `E_L`, mV.
    pub e_l: T,
    /// The resting potential at which the gate is initialized, mV.
    pub v_rest: T,
    /// Spike detection threshold (ascending crossing), mV.
    pub spike_threshold: T,
    /// Step duration, ms.
    pub dt: T,
    /// Number of sub-steps per step.
    pub substeps: usize,
}

impl<T: Scalar> Default for HodgkinHuxleyParams<T> {
    fn default() -> Self {
        Self {
            c: T::ONE,
            g_na: T::c(120.0),
            g_k: T::c(36.0),
            g_l: T::c(0.3),
            e_na: T::c(50.0),
            e_k: -T::c(77.0),
            e_l: -T::c(54.387),
            v_rest: -T::c(65.0),
            spike_threshold: T::ZERO,
            dt: T::ONE,
            substeps: 40,
        }
    }
}

impl<T: Scalar> HodgkinHuxleyParams<T> {
    pub(crate) fn validate(&self) -> StructuralResult<()> {
        let finite = |name: &'static str, v: T| -> StructuralResult<()> {
            if v.is_finite() {
                Ok(())
            } else {
                Err(StructuralError::param(name, "non-numeric value"))
            }
        };
        for (name, v) in [
            ("c", self.c),
            ("g_na", self.g_na),
            ("g_k", self.g_k),
            ("g_l", self.g_l),
            ("e_na", self.e_na),
            ("e_k", self.e_k),
            ("e_l", self.e_l),
            ("v_rest", self.v_rest),
            ("spike_threshold", self.spike_threshold),
            ("dt", self.dt),
        ] {
            finite(name, v)?;
        }
        if self.c <= T::ZERO || self.g_na <= T::ZERO || self.g_k <= T::ZERO || self.g_l <= T::ZERO {
            return Err(StructuralError::param("c/g_na/g_k/g_l", "must be > 0"));
        }
        if self.dt <= T::ZERO {
            return Err(StructuralError::param("dt", "must be > 0"));
        }
        if self.substeps == 0 || self.substeps > 10_000 {
            return Err(StructuralError::param(
                "substeps",
                "should be in [1, 10000]",
            ));
        }
        Ok(())
    }
}

/// State LIF: membrane.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct LifState<T = f32> {
    /// Membrane potential.
    pub mem: T,
}

/// ALIF state: membrane and adaptation current.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct AdaptiveLifState<T = f32> {
    /// Membrane potential.
    pub mem: T,
    /// Slow adaptation current.
    pub adaptation: T,
}

/// Izikevich state: membrane `v` and restoration of `u`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct IzhikevichState<T = f32> {
    /// Membrane potential, mV.
    pub v: T,
    /// Recovery variable.
    pub u: T,
}

/// State AdEx: membrane `v` and adaptation current `w`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AdExState<T = f32> {
    /// Membrane potential, mV.
    pub v: T,
    /// Adaptation current, pA.
    pub w: T,
}

/// Hodgkin–Huxley state: membrane, hilum, and “was above threshold” sign.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct HodgkinHuxleyState<T = f32> {
    /// Membrane potential, mV.
    pub v: T,
    /// Activation of sodium `m`.
    pub m: T,
    /// Inactivation of sodium `h`.
    pub h: T,
    /// Potassium activation `n`.
    pub n: T,
    /// The membrane was above threshold in the previous sub-step.
    pub above: bool,
}

/// Family of builtin models that supports the online local eligibility
/// dynamics of the plasticity system (see [`structural::eprop`](super::eprop)).
///
/// The capability interface of a model:
/// [`NeuronModel::online_eprop_family`] returns `Some(family)` when the
/// model's law of dynamics admits a well-defined local eligibility rule,
/// `None` otherwise (then the trainer returns
/// [`StructuralError::UnsupportedEpropDynamics`](super::error::StructuralError::UnsupportedEpropDynamics)).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OnlineEpropFamily {
    /// LIF (`U ← βU + I`, spike at `U ≥ θ`): eligibility factor is the
    /// hard spike of the postsynaptic neuron.
    Lif,
    /// Adaptive LIF (`U ← βU + I − A`, `A ← ρA + a·spike`, spike at `U ≥ θ`):
    /// see the derivation in [`structural::eprop`](super::eprop) (threshold
    /// formulation `U + A ≥ θ + A`).
    AdaptiveLif,
}

/// State transfer strategy when changing a neuron model.
///
/// Changing a model is not only about replacing parameters: old variables can
/// have a different physical meaning (dimensionless membrane LIF and potential in mV
/// y Izhikevich/AdEx/HH). Therefore, the calling code explicitly selects the policy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum StateMigration {
    /// Reset the state to the initial state for the new model.
    Reset,
    /// Transfer only the membrane potential “as is”, without converting units.
    /// It is useful when the user knows what they are doing.
    Membrane,
    /// Transfer compatible variables: if the model variant matches -
    /// the entire state, if the family matches (LIF/ALIF or
    /// Izhikevich/AdEx/HH) - membrane only, otherwise - reset.
    #[default]
    Compatible,
    /// Transfer the membrane with conversion of units: position between rest and
    /// threshold (`share = (v − v_rest)/(v_threshold − v_rest)`) is preserved, and
    /// the absolute value is recalculated into the scale of the new model (dimensionless
    /// LIF/ALIF ↔ mV at Izhikevich/AdEx/HH). Quick Variables
    /// (`u`, `w`, gate HH) are put into equilibrium at the new potential,
    /// The adaptation current ALIF is reset. If the model option matches
    /// the state is preserved entirely.
    Scaled,
}

impl StateMigration {
    /// Short name for messages and serialization.
    pub fn as_str(self) -> &'static str {
        match self {
            StateMigration::Reset => "reset",
            StateMigration::Membrane => "membrane",
            StateMigration::Compatible => "compatible",
            StateMigration::Scaled => "scaled",
        }
    }

    pub(crate) fn code(self) -> u8 {
        match self {
            StateMigration::Reset => 0,
            StateMigration::Membrane => 1,
            StateMigration::Compatible => 2,
            StateMigration::Scaled => 3,
        }
    }

    pub(crate) fn from_code(c: u8) -> Option<Self> {
        match c {
            0 => Some(StateMigration::Reset),
            1 => Some(StateMigration::Membrane),
            2 => Some(StateMigration::Compatible),
            3 => Some(StateMigration::Scaled),
            _ => None,
        }
    }
}

/// The result of changing the neuron model.
///
/// Returns [`Topology::change_neuron_model`](super::Topology::change_neuron_model)
/// and [`DynamicNetwork::change_neuron_model`](super::DynamicNetwork::change_neuron_model),
/// so that the calling code can see what the state was before and what happened after
/// chosen policy.
#[derive(Debug, Clone, PartialEq)]
pub struct ModelSwitchOutcome<T: Scalar = f32> {
    /// Model before modification.
    pub old_model: NeuronModel<T>,
    /// Model after change.
    pub new_model: NeuronModel<T>,
    /// State before change.
    pub old_state: NeuronState<T>,
    /// State after migration.
    pub new_state: NeuronState<T>,
    /// Applied policy.
    pub migration: StateMigration,
    /// Has the model really changed (`false` - the model was already like this).
    pub changed: bool,
}

/// The state of the neuron corresponding to the variant [`NeuronModel<T>`].
///
/// `Custom` holds the erased user state ([`CustomStateEntryBox`]).
#[derive(Debug, Clone, PartialEq)]
pub enum NeuronState<T: Scalar = f32> {
    /// LIF.
    Lif(LifState<T>),
    /// ALIF.
    AdaptiveLif(AdaptiveLifState<T>),
    /// Izikevich.
    Izhikevich(IzhikevichState<T>),
    /// AdEx.
    AdEx(AdExState<T>),
    /// Hodgkin-Huxley.
    HodgkinHuxley(HodgkinHuxleyState<T>),
    /// User-defined model state.
    Custom(CustomStateEntryBox<T>),
}

impl<T: Scalar> Default for NeuronState<T> {
    fn default() -> Self {
        Self::Lif(LifState::default())
    }
}

impl<T: Scalar> NeuronState<T> {
    /// Is the state of a user-defined model ([`NeuronModel::Custom`])?
    pub fn is_custom(&self) -> bool {
        matches!(self, NeuronState::Custom(_))
    }

    /// Same dynamics as `other`: for custom - equal model ids (and schema
    /// versions of the owning models); for builtin - equal variant tags.
    pub fn is_same_dynamics(&self, other: &NeuronState<T>) -> bool {
        match (self, other) {
            (NeuronState::Custom(a), NeuronState::Custom(b)) => {
                a.inner().model_id() == b.inner().model_id()
                    && a.inner().schema_version() == b.inner().schema_version()
            }
            _ => self.tag() == other.tag(),
        }
    }

    /// Membrane potential (for all models this is `mem` or `v`).
    ///
    /// For a custom model without a membrane, returns `T::ZERO`.
    pub fn mem(&self) -> T {
        match self {
            NeuronState::Lif(s) => s.mem,
            NeuronState::AdaptiveLif(s) => s.mem,
            NeuronState::Izhikevich(s) => s.v,
            NeuronState::AdEx(s) => s.v,
            NeuronState::HodgkinHuxley(s) => s.v,
            NeuronState::Custom(s) => s.inner().membrane().unwrap_or(T::ZERO),
        }
    }

    /// Record the membrane potential.
    pub fn set_mem(&mut self, value: T) {
        match self {
            NeuronState::Lif(s) => s.mem = value,
            NeuronState::AdaptiveLif(s) => s.mem = value,
            NeuronState::Izhikevich(s) => s.v = value,
            NeuronState::AdEx(s) => s.v = value,
            NeuronState::HodgkinHuxley(s) => s.v = value,
            NeuronState::Custom(s) => s.inner_mut().set_membrane(value),
        }
    }

    /// All state components are finite.
    pub fn is_finite(&self) -> bool {
        match self {
            NeuronState::Lif(s) => s.mem.is_finite(),
            NeuronState::AdaptiveLif(s) => s.mem.is_finite() && s.adaptation.is_finite(),
            NeuronState::Izhikevich(s) => s.v.is_finite() && s.u.is_finite(),
            NeuronState::AdEx(s) => s.v.is_finite() && s.w.is_finite(),
            NeuronState::HodgkinHuxley(s) => {
                s.v.is_finite() && s.m.is_finite() && s.h.is_finite() && s.n.is_finite()
            }
            NeuronState::Custom(s) => s.inner().is_finite(),
        }
    }

    /// Transfer the state to the `target` model using an explicit policy.
    ///
    /// * [`StateMigration::Reset`] - initial state of the new model;
    /// * [`StateMigration::Membrane`] - the initial state to which the
    ///   membrane potential only (without unit conversion);
    /// * [`StateMigration::Compatible`] - if the option matches, everything is saved
    ///   state, if the family matches (LIF/ALIF or
    ///   Izhikevich/AdEx/HH) - only membrane, otherwise - reset.
    ///
    /// Policy [`StateMigration::Scaled`] requires knowing the original model, so
    /// here behaves like `Compatible`; to convert units use
    /// [`NeuronState::migrate_between`].
    pub fn migrate_to(self, target: &NeuronModel<T>, policy: StateMigration) -> NeuronState<T> {
        match (self, target, policy) {
            (_, _, StateMigration::Reset) => target.init_state(),
            // Custom target, Membrane: membrane-only transfer.
            (NeuronState::Custom(entry), NeuronModel::Custom(cm), StateMigration::Membrane) => {
                let mut s = cm.adapter.init_state();
                if let Some(m) = entry.inner().membrane() {
                    cm.adapter.set_mem(&mut *s, m);
                }
                NeuronState::Custom(CustomStateEntryBox::from_boxed(s))
            }
            // Custom target, Compatible/Scaled: user hook.
            (
                NeuronState::Custom(entry),
                NeuronModel::Custom(cm),
                StateMigration::Compatible | StateMigration::Scaled,
            ) => match cm.adapter.migrate_state(entry.inner(), policy) {
                Ok(s) => NeuronState::Custom(CustomStateEntryBox::from_boxed(s)),
                Err(_) => target.init_state(),
            },
            // Builtin target, custom source: membrane-only transfer
            // (the builtin membrane is the natural common quantity).
            (
                NeuronState::Custom(entry),
                target @ (NeuronModel::Lif(_) | NeuronModel::AdaptiveLif(_)),
                StateMigration::Membrane | StateMigration::Compatible | StateMigration::Scaled,
            ) => {
                let mut s = target.init_state();
                s.set_mem(entry.inner().membrane().unwrap_or(T::ZERO));
                s
            }
            (
                NeuronState::Custom(entry),
                target @ (NeuronModel::Izhikevich(_) | NeuronModel::HodgkinHuxley(_)),
                StateMigration::Membrane | StateMigration::Compatible | StateMigration::Scaled,
            ) => {
                let mut s = target.init_state();
                s.set_mem(entry.inner().membrane().unwrap_or(T::ZERO));
                s
            }
            // Builtin -> custom target: membrane-only transfer.
            (
                NeuronState::Lif(s),
                NeuronModel::Custom(cm),
                StateMigration::Membrane | StateMigration::Compatible | StateMigration::Scaled,
            ) => {
                let mut st = cm.adapter.init_state();
                cm.adapter.set_mem(&mut *st, s.mem);
                NeuronState::Custom(CustomStateEntryBox::from_boxed(st))
            }
            // General: Membrane transfer as-is.
            (s, target, StateMigration::Membrane) => {
                let mut next = target.init_state();
                next.set_mem(s.mem());
                next
            }
            // General: Compatible/Scaled for builtin→builtin (original semantics).
            (s, target, _) => {
                if s.matches(target) {
                    s
                } else if state_family(s.tag()) == state_family(target.tag()) {
                    let mut next = target.init_state();
                    next.set_mem(s.mem());
                    next
                } else {
                    target.init_state()
                }
            }
        }
    }

    /// Transfer the state from the `source` model to the `target` model.
    ///
    /// For [`StateMigration::Scaled`] a physical recalculation is performed
    /// membranes (see [`NeuronModel::membrane_range`]); other politicians
    /// work like [`NeuronState::migrate_to`].
    pub fn migrate_between(
        self,
        source: &NeuronModel<T>,
        target: &NeuronModel<T>,
        policy: StateMigration,
    ) -> NeuronState<T> {
        if policy != StateMigration::Scaled || self.matches(target) {
            return self.migrate_to(target, policy);
        }
        let (Some((src_rest, src_fire)), Some((dst_rest, dst_fire))) =
            (source.membrane_range(), target.membrane_range())
        else {
            return self.migrate_to(target, StateMigration::Compatible);
        };
        let fraction =
            ((self.mem() - src_rest) / (src_fire - src_rest)).clamp(-T::c(0.5), T::c(0.999));
        if !fraction.is_finite() {
            return target.init_state();
        }
        let v = dst_rest + fraction * (dst_fire - dst_rest);
        let mut next = target.init_state();
        match (&mut next, target) {
            (NeuronState::Lif(s), _) => s.mem = v,
            (NeuronState::AdaptiveLif(s), _) => s.mem = v,
            (NeuronState::Izhikevich(s), NeuronModel::Izhikevich(p)) => {
                s.v = v;
                s.u = p.b * v;
            }
            (NeuronState::AdEx(s), NeuronModel::AdEx(p)) => {
                s.v = v;
                s.w = p.a * (v - p.e_l);
            }
            (NeuronState::HodgkinHuxley(s), _) => {
                let (m, h, n) = crate::neuron::hh_steady_state_as(v);
                s.v = v;
                s.m = m;
                s.h = h;
                s.n = n;
                s.above = false;
            }
            (NeuronState::Custom(s), NeuronModel::Custom(cm)) => {
                cm.adapter.set_mem(s.inner_mut(), v)
            }
            _ => next.set_mem(v),
        }
        next
    }

    /// Does the state variant match the model?
    pub fn matches(&self, model: &NeuronModel<T>) -> bool {
        match (self, model) {
            (NeuronState::Lif(_), NeuronModel::Lif(_))
            | (NeuronState::AdaptiveLif(_), NeuronModel::AdaptiveLif(_))
            | (NeuronState::Izhikevich(_), NeuronModel::Izhikevich(_))
            | (NeuronState::AdEx(_), NeuronModel::AdEx(_))
            | (NeuronState::HodgkinHuxley(_), NeuronModel::HodgkinHuxley(_)) => true,
            (NeuronState::Custom(entry), NeuronModel::Custom(cm)) => {
                entry.inner().model_id() == cm.model_id()
            }
            _ => false,
        }
    }

    /// Variant tag (same as model tag).
    pub(crate) fn tag(&self) -> u8 {
        match self {
            NeuronState::Lif(_) => 0,
            NeuronState::AdaptiveLif(_) => 1,
            NeuronState::Izhikevich(_) => 2,
            NeuronState::AdEx(_) => 3,
            NeuronState::HodgkinHuxley(_) => 4,
            NeuronState::Custom(_) => 5,
        }
    }
}

/// State family for `StateMigration::Compatible`:
/// LIF/ALIF are dimensionless, Izhikevich/AdEx/HH are mV.
fn state_family(tag: u8) -> u8 {
    if tag <= 1 { 0 } else { 1 }
}

/// Law of neuron dynamics.
///
/// Builtin variants keep the historic semantics; `Custom` carries a user-defined
/// model (see [`structural::custom`](super::custom)).
#[derive(Debug, Clone, PartialEq)]
pub enum NeuronModel<T: Scalar = f32> {
    /// Leaky integrate-and-fire.
    Lif(LifParams<T>),
    /// Adaptive LIF.
    AdaptiveLif(AdaptiveLifParams<T>),
    /// Izhikevich.
    Izhikevich(IzhikevichParams<T>),
    /// AdEx.
    AdEx(AdExParams<T>),
    /// Hodgkin–Huxley.
    HodgkinHuxley(HodgkinHuxleyParams<T>),
    /// User-defined dynamic model.
    Custom(CustomNeuronInstance<T>),
}

/// Continuous physical parameter of the neuron model.
///
/// The enumeration gives unified access to the parameters of all five models: its
/// use [`NeuronModel::param`], [`NeuronModel::set_param`],
/// [`NeuronMutation`](super::NeuronMutation) and learning/evolution mechanisms.
/// Numerical parameters (`substeps`, `dt`, `exp_clip`) and discrete mode
/// resets are not included here: they change the calculation scheme, not the physiology
/// neuron. The spike threshold for AdEx is [`NeuronParam::AdExVPeak`], for HH -
/// [`NeuronParam::HhSpikeThreshold`]; general [`NeuronParam::Threshold`]
/// only applies to LIF/ALIF/Izhikevich.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum NeuronParam {
    /// Membrane β attenuation (LIF/ALIF).
    Beta,
    /// Attenuation of adaptation current ρ (ALIF).
    Rho,
    /// Adaptation gain per spike (ALIF).
    AdaptIncrement,
    /// Membrane threshold (LIF/ALIF/Izhikevich).
    Threshold,
    /// Recovery rate `a` (Izhikevich).
    IzhA,
    /// Recovery sensitivity `b` (Izhikevich).
    IzhB,
    /// Reset potential `c` (Izhikevich), mV.
    IzhC,
    /// Recovery increment `d` (Izhikevich).
    IzhD,
    /// Membrane capacity `C` (AdEx), pF.
    AdExC,
    /// Leakage conductivity `g_L` (AdEx), nS.
    AdExGl,
    /// Resting potential `E_L` (AdEx), mV.
    AdExEl,
    /// Exponent threshold `V_T` (AdEx), mV.
    AdExVt,
    /// Sharpness of the exponential `Δ_T` (AdEx), mV.
    AdExDeltaT,
    /// Subthreshold adaptation `a` (AdEx), nSm.
    AdExA,
    /// Adaptation time constant `τ_w` (AdEx), ms.
    AdExTauW,
    /// Increase in adaptation current `b` (AdEx), pA.
    AdExB,
    /// Reset potential `V_reset` (AdEx), mV.
    AdExVReset,
    /// Spike detection threshold `V_peak` (AdEx), mV.
    AdExVPeak,
    /// Membrane capacity `C` (HH), µF/cm².
    HhC,
    /// Maximum sodium conductivity `g_Na` (HH), mS/cm².
    HhGNa,
    /// Maximum conductivity of potassium `g_K` (HH), mS/cm².
    HhGK,
    /// Leakage conductivity `g_L` (HH), mS/cm².
    HhGl,
    /// Equilibrium potential of sodium `E_Na` (HH), mV.
    HhENa,
    /// Equilibrium potential of potassium `E_K` (HH), mV.
    HhEK,
    /// Leakage potential `E_L` (HH), mV.
    HhEL,
    /// Resting potential for gate initialization (HH), mV.
    HhVRest,
    /// Spike detection threshold (HH), mV.
    HhSpikeThreshold,
}

impl NeuronParam {
    /// All continuous physical parameters.
    pub const ALL: [NeuronParam; 27] = [
        NeuronParam::Beta,
        NeuronParam::Rho,
        NeuronParam::AdaptIncrement,
        NeuronParam::Threshold,
        NeuronParam::IzhA,
        NeuronParam::IzhB,
        NeuronParam::IzhC,
        NeuronParam::IzhD,
        NeuronParam::AdExC,
        NeuronParam::AdExGl,
        NeuronParam::AdExEl,
        NeuronParam::AdExVt,
        NeuronParam::AdExDeltaT,
        NeuronParam::AdExA,
        NeuronParam::AdExTauW,
        NeuronParam::AdExB,
        NeuronParam::AdExVReset,
        NeuronParam::AdExVPeak,
        NeuronParam::HhC,
        NeuronParam::HhGNa,
        NeuronParam::HhGK,
        NeuronParam::HhGl,
        NeuronParam::HhENa,
        NeuronParam::HhEK,
        NeuronParam::HhEL,
        NeuronParam::HhVRest,
        NeuronParam::HhSpikeThreshold,
    ];

    /// Short name for errors, log and serialization.
    pub fn as_str(self) -> &'static str {
        match self {
            NeuronParam::Beta => "beta",
            NeuronParam::Rho => "rho",
            NeuronParam::AdaptIncrement => "adapt_increment",
            NeuronParam::Threshold => "threshold",
            NeuronParam::IzhA => "izh.a",
            NeuronParam::IzhB => "izh.b",
            NeuronParam::IzhC => "izh.c",
            NeuronParam::IzhD => "izh.d",
            NeuronParam::AdExC => "adex.c",
            NeuronParam::AdExGl => "adex.g_l",
            NeuronParam::AdExEl => "adex.e_l",
            NeuronParam::AdExVt => "adex.v_t",
            NeuronParam::AdExDeltaT => "adex.delta_t",
            NeuronParam::AdExA => "adex.a",
            NeuronParam::AdExTauW => "adex.tau_w",
            NeuronParam::AdExB => "adex.b",
            NeuronParam::AdExVReset => "adex.v_reset",
            NeuronParam::AdExVPeak => "adex.v_peak",
            NeuronParam::HhC => "hh.c",
            NeuronParam::HhGNa => "hh.g_na",
            NeuronParam::HhGK => "hh.g_k",
            NeuronParam::HhGl => "hh.g_l",
            NeuronParam::HhENa => "hh.e_na",
            NeuronParam::HhEK => "hh.e_k",
            NeuronParam::HhEL => "hh.e_l",
            NeuronParam::HhVRest => "hh.v_rest",
            NeuronParam::HhSpikeThreshold => "hh.spike_threshold",
        }
    }

    /// Does the model support this parameter?
    pub fn supported_by<T: Scalar>(self, model: &NeuronModel<T>) -> bool {
        match self {
            NeuronParam::Beta => matches!(model, NeuronModel::Lif(_) | NeuronModel::AdaptiveLif(_)),
            NeuronParam::Rho | NeuronParam::AdaptIncrement => {
                matches!(model, NeuronModel::AdaptiveLif(_))
            }
            NeuronParam::Threshold => matches!(
                model,
                NeuronModel::Lif(_) | NeuronModel::AdaptiveLif(_) | NeuronModel::Izhikevich(_)
            ),
            NeuronParam::IzhA | NeuronParam::IzhB | NeuronParam::IzhC | NeuronParam::IzhD => {
                matches!(model, NeuronModel::Izhikevich(_))
            }
            NeuronParam::AdExC
            | NeuronParam::AdExGl
            | NeuronParam::AdExEl
            | NeuronParam::AdExVt
            | NeuronParam::AdExDeltaT
            | NeuronParam::AdExA
            | NeuronParam::AdExTauW
            | NeuronParam::AdExB
            | NeuronParam::AdExVReset
            | NeuronParam::AdExVPeak => matches!(model, NeuronModel::AdEx(_)),
            NeuronParam::HhC
            | NeuronParam::HhGNa
            | NeuronParam::HhGK
            | NeuronParam::HhGl
            | NeuronParam::HhENa
            | NeuronParam::HhEK
            | NeuronParam::HhEL
            | NeuronParam::HhVRest
            | NeuronParam::HhSpikeThreshold => matches!(model, NeuronModel::HodgkinHuxley(_)),
        }
    }
}

impl<T: Scalar> Default for NeuronModel<T> {
    fn default() -> Self {
        Self::Lif(LifParams::default())
    }
}

impl<T: Scalar> From<LifParams<T>> for NeuronModel<T> {
    fn from(p: LifParams<T>) -> Self {
        Self::Lif(p)
    }
}

impl<T: Scalar> From<AdaptiveLifParams<T>> for NeuronModel<T> {
    fn from(p: AdaptiveLifParams<T>) -> Self {
        Self::AdaptiveLif(p)
    }
}

impl<T: Scalar> From<IzhikevichParams<T>> for NeuronModel<T> {
    fn from(p: IzhikevichParams<T>) -> Self {
        Self::Izhikevich(p)
    }
}

impl<T: Scalar> From<AdExParams<T>> for NeuronModel<T> {
    fn from(p: AdExParams<T>) -> Self {
        Self::AdEx(p)
    }
}

impl<T: Scalar> From<HodgkinHuxleyParams<T>> for NeuronModel<T> {
    fn from(p: HodgkinHuxleyParams<T>) -> Self {
        Self::HodgkinHuxley(p)
    }
}

impl<T: Scalar> From<CustomNeuronInstance<T>> for NeuronModel<T> {
    fn from(cm: CustomNeuronInstance<T>) -> Self {
        Self::Custom(cm)
    }
}

impl<T: Scalar> NeuronModel<T> {
    /// Capability interface for online local eligibility dynamics:
    /// `Some(family)` for models with a supported law of dynamics
    /// ([`OnlineEpropFamily::Lif`] for LIF, [`OnlineEpropFamily::AdaptiveLif`]
    /// for Adaptive LIF), `None` otherwise.
    pub fn online_eprop_family(&self) -> Option<OnlineEpropFamily> {
        match self {
            NeuronModel::Lif(_) => Some(OnlineEpropFamily::Lif),
            NeuronModel::AdaptiveLif(_) => Some(OnlineEpropFamily::AdaptiveLif),
            _ => None,
        }
    }

    /// Short model name (`"custom"` for user models).
    pub fn as_str(&self) -> &'static str {
        match self {
            NeuronModel::Lif(_) => "lif",
            NeuronModel::AdaptiveLif(_) => "adaptive_lif",
            NeuronModel::Izhikevich(_) => "izhikevich",
            NeuronModel::AdEx(_) => "adex",
            NeuronModel::HodgkinHuxley(_) => "hodgkin_huxley",
            NeuronModel::Custom(_) => "custom",
        }
    }

    /// Display name: builtin name or the stable user model id (`"custom/my_neuron"`).
    pub fn display_name(&self) -> String {
        match self {
            NeuronModel::Custom(cm) => cm.model_id().to_string(),
            other => other.as_str().to_string(),
        }
    }

    /// Stable user model id (`None` for builtin models).
    pub fn custom_model_id(&self) -> Option<&'static str> {
        match self {
            NeuronModel::Custom(cm) => Some(cm.model_id()),
            _ => None,
        }
    }

    /// Are the dynamics equal: same model id + schema version for custom,
    /// variant equality for builtin (states may differ by policy).
    pub fn is_same_dynamics(&self, other: &NeuronModel<T>) -> bool {
        match (self, other) {
            (NeuronModel::Custom(a), NeuronModel::Custom(b)) => {
                a.model_id() == b.model_id() && a.schema_version() == b.schema_version()
            }
            (a, b) => a == b,
        }
    }

    /// Tag for checkpoint and genome.
    pub(crate) fn tag(&self) -> u8 {
        match self {
            NeuronModel::Lif(_) => 0,
            NeuronModel::AdaptiveLif(_) => 1,
            NeuronModel::Izhikevich(_) => 2,
            NeuronModel::AdEx(_) => 3,
            NeuronModel::HodgkinHuxley(_) => 4,
            NeuronModel::Custom(_) => 5,
        }
    }

    /// LIF-model?
    pub fn is_lif(&self) -> bool {
        matches!(self, NeuronModel::Lif(_))
    }

    /// Membrane attenuation β, if the model has one.
    pub fn beta(&self) -> Option<T> {
        match self {
            NeuronModel::Lif(p) => Some(p.beta),
            NeuronModel::AdaptiveLif(p) => Some(p.beta),
            _ => None,
        }
    }

    /// Change β at LIF/ALIF; for other models - [`StructuralError`].
    pub fn set_beta(&mut self, value: T, lim: &ParamLimits<T>) -> StructuralResult<T> {
        if !value.is_finite() {
            return Err(StructuralError::param("beta", "non-numeric value"));
        }
        let beta = value.clamp(lim.beta_min, lim.beta_max);
        match self {
            NeuronModel::Lif(p) => p.beta = beta,
            NeuronModel::AdaptiveLif(p) => p.beta = beta,
            NeuronModel::Custom(cm) => {
                return Err(StructuralError::UnsupportedParameter {
                    kind: cm.model_id(),
                    parameter: "beta",
                });
            }
            other => {
                return Err(StructuralError::UnsupportedParameter {
                    kind: other.as_str(),
                    parameter: "beta",
                });
            }
        }
        Ok(beta)
    }

    /// Decay of ρ adaptation in ALIF.
    pub fn set_rho(&mut self, value: T) -> StructuralResult<T> {
        if !value.is_finite() {
            return Err(StructuralError::param("rho", "non-numeric value"));
        }
        match self {
            NeuronModel::AdaptiveLif(p) => {
                p.rho = value.clamp(T::ZERO, T::c(0.999_999));
                Ok(p.rho)
            }
            other => Err(StructuralError::UnsupportedParameter {
                kind: other.as_str(),
                parameter: "rho",
            }),
        }
    }

    /// Increase in adaptation per spike for ALIF.
    pub fn set_adapt_increment(&mut self, value: T) -> StructuralResult<T> {
        if !value.is_finite() || value < T::ZERO {
            return Err(StructuralError::param(
                "adapt_increment",
                "must be finite and >= 0",
            ));
        }
        match self {
            NeuronModel::AdaptiveLif(p) => {
                p.adapt_increment = value;
                Ok(p.adapt_increment)
            }
            other => Err(StructuralError::UnsupportedParameter {
                kind: other.as_str(),
                parameter: "adapt_increment",
            }),
        }
    }

    /// Membrane reset mode for LIF/ALIF.
    pub fn reset_mode(&self) -> Option<ResetMode> {
        match self {
            NeuronModel::Lif(p) => Some(p.reset),
            NeuronModel::AdaptiveLif(p) => Some(p.reset),
            _ => None,
        }
    }

    /// Change the reset mode of LIF/ALIF.
    pub fn set_reset(&mut self, reset: ResetMode) -> StructuralResult<()> {
        match self {
            NeuronModel::Lif(p) => p.reset = reset,
            NeuronModel::AdaptiveLif(p) => p.reset = reset,
            NeuronModel::Custom(cm) => {
                return Err(StructuralError::UnsupportedParameter {
                    kind: cm.model_id(),
                    parameter: "reset",
                });
            }
            other => {
                return Err(StructuralError::UnsupportedParameter {
                    kind: other.as_str(),
                    parameter: "reset",
                });
            }
        }
        Ok(())
    }

    /// Check parameters; `lim` applies to LIF-like models.
    pub fn validate(&self, lim: &ParamLimits<T>) -> StructuralResult<()> {
        match self {
            NeuronModel::Lif(p) => p.validate(lim),
            NeuronModel::AdaptiveLif(p) => p.validate(lim),
            NeuronModel::Izhikevich(p) => p.validate(),
            NeuronModel::AdEx(p) => p.validate(),
            NeuronModel::HodgkinHuxley(p) => p.validate(),
            NeuronModel::Custom(cm) => cm.validate(lim),
        }
    }

    /// Bring parameters within acceptable limits (for mutations and homeostasis).
    pub fn clamp(&mut self, lim: &ParamLimits<T>) {
        match self {
            NeuronModel::Lif(p) => {
                p.beta = p.beta.clamp(lim.beta_min, lim.beta_max);
                p.threshold = p.threshold.clamp(lim.threshold_min, lim.threshold_max);
            }
            NeuronModel::AdaptiveLif(p) => {
                p.beta = p.beta.clamp(lim.beta_min, lim.beta_max);
                p.threshold = p.threshold.clamp(lim.threshold_min, lim.threshold_max);
                p.rho = p.rho.clamp(T::ZERO, T::c(0.999_999));
                p.adapt_increment = p.adapt_increment.max(T::ZERO);
            }
            NeuronModel::Izhikevich(p) => {
                p.a = p.a.clamp(T::c(1e-4), T::ONE);
                p.b = p.b.max(T::ZERO);
                p.d = p.d.max(T::ZERO);
                p.threshold = p.threshold.max(p.c + T::c(1e-3));
                p.substeps = p.substeps.clamp(1, 1024);
            }
            NeuronModel::AdEx(p) => {
                p.c = p.c.max(T::c(1e-6));
                p.g_l = p.g_l.max(T::c(1e-6));
                p.delta_t = p.delta_t.max(T::c(1e-6));
                p.tau_w = p.tau_w.max(T::c(1e-6));
                p.dt = p.dt.max(T::c(1e-6));
                p.exp_clip = p.exp_clip.max(T::c(1e-6));
                p.a = p.a.max(T::ZERO);
                p.b = p.b.max(T::ZERO);
                p.v_peak = p.v_peak.max(p.v_reset + T::c(1e-3));
                p.substeps = p.substeps.clamp(1, 1024);
            }
            NeuronModel::HodgkinHuxley(p) => {
                p.c = p.c.max(T::c(1e-6));
                p.g_na = p.g_na.max(T::c(1e-6));
                p.g_k = p.g_k.max(T::c(1e-6));
                p.g_l = p.g_l.max(T::c(1e-6));
                p.dt = p.dt.max(T::c(1e-6));
                p.substeps = p.substeps.clamp(1, 10_000);
            }
            // Custom models clamp themselves in `validate`/`set_param`.
            NeuronModel::Custom(_) => {}
        }
    }

    /// Initial (calm) state.
    pub fn init_state(&self) -> NeuronState<T> {
        match self {
            NeuronModel::Lif(_) => NeuronState::Lif(LifState::default()),
            NeuronModel::AdaptiveLif(_) => NeuronState::AdaptiveLif(AdaptiveLifState::default()),
            NeuronModel::Custom(cm) => {
                NeuronState::Custom(CustomStateEntryBox::from_boxed(cm.adapter.init_state()))
            }
            NeuronModel::Izhikevich(p) => {
                NeuronState::Izhikevich(IzhikevichState { v: p.c, u: T::ZERO })
            }
            NeuronModel::AdEx(p) => NeuronState::AdEx(AdExState {
                v: p.e_l,
                w: T::ZERO,
            }),
            NeuronModel::HodgkinHuxley(p) => {
                let (m, h, n) = crate::neuron::hh_steady_state_as(p.v_rest);
                NeuronState::HodgkinHuxley(HodgkinHuxleyState {
                    v: p.v_rest,
                    m,
                    h,
                    n,
                    above: false,
                })
            }
        }
    }

    /// Convert state received from another model by policy
    /// [`StateMigration`]. Semantics - like
    /// [`NeuronState::migrate_to`](NeuronState::migrate_to).
    pub fn migrate_state(&self, state: NeuronState<T>, policy: StateMigration) -> NeuronState<T> {
        state.migrate_to(self, policy)
    }

    /// The resting-threshold pair of membrane potential in the model scale; on it
    /// [`StateMigration::Scaled`] recalculates units. `None`, if a pair
    /// degenerate (the threshold is not higher than rest).
    ///
    /// LIF/ALIF: `(0, threshold)`; Izhikevich: `(c, threshold)`; AdEx:
    /// `(E_L, V_T)` — exponential acceleration begins with `V_T`;
    /// Hodgkin–Huxley: `(v_rest, spike_threshold)`.
    pub fn membrane_range(&self) -> Option<(T, T)> {
        let (rest, fire) = match self {
            NeuronModel::Lif(p) => (T::ZERO, p.threshold),
            NeuronModel::AdaptiveLif(p) => (T::ZERO, p.threshold),
            NeuronModel::Izhikevich(p) => (p.c, p.threshold),
            NeuronModel::AdEx(p) => (p.e_l, p.v_t),
            NeuronModel::HodgkinHuxley(p) => (p.v_rest, p.spike_threshold),
            NeuronModel::Custom(cm) => return cm.adapter.membrane_range(),
        };
        (rest.is_finite() && fire.is_finite() && fire > rest).then_some((rest, fire))
    }

    /// Current threshold (for homeostasis); `T::ZERO` for custom models without one.
    pub fn threshold(&self) -> T {
        self.threshold_opt().unwrap_or(T::ZERO)
    }

    /// Current threshold if the model has one.
    pub fn threshold_opt(&self) -> Option<T> {
        match self {
            NeuronModel::Lif(p) => Some(p.threshold),
            NeuronModel::AdaptiveLif(p) => Some(p.threshold),
            NeuronModel::Izhikevich(p) => Some(p.threshold),
            NeuronModel::AdEx(p) => Some(p.v_peak),
            NeuronModel::HodgkinHuxley(p) => Some(p.spike_threshold),
            NeuronModel::Custom(cm) => cm.adapter.threshold(),
        }
    }

    /// Change the threshold with a clamp within acceptable limits; returns a new value.
    pub fn set_threshold(&mut self, value: T, lim: &ParamLimits<T>) -> T {
        if !value.is_finite() {
            return self.threshold();
        }
        match self {
            NeuronModel::Lif(p) => {
                p.threshold = value.clamp(lim.threshold_min, lim.threshold_max);
                p.threshold
            }
            NeuronModel::AdaptiveLif(p) => {
                p.threshold = value.clamp(lim.threshold_min, lim.threshold_max);
                p.threshold
            }
            NeuronModel::Izhikevich(p) => {
                p.threshold = value.max(p.c + T::c(1e-3));
                p.threshold
            }
            NeuronModel::AdEx(p) => {
                p.v_peak = value.max(p.v_reset + T::c(1e-3));
                p.v_peak
            }
            NeuronModel::HodgkinHuxley(p) => {
                p.spike_threshold = value;
                p.spike_threshold
            }
            NeuronModel::Custom(_) => self.threshold(),
        }
    }

    /// The value of a continuous physical parameter, if its model
    /// supports.
    pub fn param(&self, param: NeuronParam) -> Option<T> {
        Some(match (self, param) {
            (NeuronModel::Lif(p), NeuronParam::Beta) => p.beta,
            (NeuronModel::Lif(p), NeuronParam::Threshold) => p.threshold,
            (NeuronModel::AdaptiveLif(p), NeuronParam::Beta) => p.beta,
            (NeuronModel::AdaptiveLif(p), NeuronParam::Rho) => p.rho,
            (NeuronModel::AdaptiveLif(p), NeuronParam::AdaptIncrement) => p.adapt_increment,
            (NeuronModel::AdaptiveLif(p), NeuronParam::Threshold) => p.threshold,
            (NeuronModel::Izhikevich(p), NeuronParam::IzhA) => p.a,
            (NeuronModel::Izhikevich(p), NeuronParam::IzhB) => p.b,
            (NeuronModel::Izhikevich(p), NeuronParam::IzhC) => p.c,
            (NeuronModel::Izhikevich(p), NeuronParam::IzhD) => p.d,
            (NeuronModel::Izhikevich(p), NeuronParam::Threshold) => p.threshold,
            (NeuronModel::AdEx(p), NeuronParam::AdExC) => p.c,
            (NeuronModel::AdEx(p), NeuronParam::AdExGl) => p.g_l,
            (NeuronModel::AdEx(p), NeuronParam::AdExEl) => p.e_l,
            (NeuronModel::AdEx(p), NeuronParam::AdExVt) => p.v_t,
            (NeuronModel::AdEx(p), NeuronParam::AdExDeltaT) => p.delta_t,
            (NeuronModel::AdEx(p), NeuronParam::AdExA) => p.a,
            (NeuronModel::AdEx(p), NeuronParam::AdExTauW) => p.tau_w,
            (NeuronModel::AdEx(p), NeuronParam::AdExB) => p.b,
            (NeuronModel::AdEx(p), NeuronParam::AdExVReset) => p.v_reset,
            (NeuronModel::AdEx(p), NeuronParam::AdExVPeak) => p.v_peak,
            (NeuronModel::HodgkinHuxley(p), NeuronParam::HhC) => p.c,
            (NeuronModel::HodgkinHuxley(p), NeuronParam::HhGNa) => p.g_na,
            (NeuronModel::HodgkinHuxley(p), NeuronParam::HhGK) => p.g_k,
            (NeuronModel::HodgkinHuxley(p), NeuronParam::HhGl) => p.g_l,
            (NeuronModel::HodgkinHuxley(p), NeuronParam::HhENa) => p.e_na,
            (NeuronModel::HodgkinHuxley(p), NeuronParam::HhEK) => p.e_k,
            (NeuronModel::HodgkinHuxley(p), NeuronParam::HhEL) => p.e_l,
            (NeuronModel::HodgkinHuxley(p), NeuronParam::HhVRest) => p.v_rest,
            (NeuronModel::HodgkinHuxley(p), NeuronParam::HhSpikeThreshold) => p.spike_threshold,
            _ => return None,
        })
    }

    /// All continuous physical parameters supported by the model.
    ///
    /// For custom models - the parameters they expose via `DynamicNeuronModel`
    /// (adapter `params`); named string-id parameters are not included
    /// (see [`NeuronModel::custom_params`]).
    pub fn params(&self) -> Vec<(NeuronParam, T)> {
        match self {
            NeuronModel::Custom(cm) => cm.params(),
            _ => NeuronParam::ALL
                .iter()
                .filter_map(|p| self.param(*p).map(|v| (*p, v)))
                .collect(),
        }
    }

    /// Named parameters of a custom model (string ids), if any.
    pub fn custom_params(&self) -> Vec<super::custom::CustomParameter<T>> {
        match self {
            NeuronModel::Custom(cm) => cm.adapter.custom_params(),
            _ => Vec::new(),
        }
    }

    /// Current value of a named custom parameter of a custom model (None - not supported).
    pub fn get_custom_param(&self, id: &str) -> Option<T> {
        match self {
            NeuronModel::Custom(cm) => cm.adapter.get_custom_param(id),
            _ => None,
        }
    }

    /// Write a named custom parameter of a custom model; returns the actual value
    /// (None - unsupported or not a custom model).
    pub fn set_custom_param(&mut self, id: &str, value: T) -> Option<T> {
        match self {
            NeuronModel::Custom(cm) => cm.adapter.set_custom_param(id, value),
            _ => None,
        }
    }

    /// Write a continuous physical parameter with a clamp into the allowable
    /// boundaries; returns the actual value.
    ///
    /// For parameters that the model does not have, [`StructuralError::UnsupportedParameter`].
    /// The clamp can also adjust the associated parameter (for example, `V_peak`
    /// when `V_reset` changes in AdEx or threshold when `c` changes in
    /// Izhikevich) - this keeps the model valid.
    pub fn set_param(
        &mut self,
        param: NeuronParam,
        value: T,
        lim: &ParamLimits<T>,
    ) -> StructuralResult<T> {
        if let NeuronModel::Custom(cm) = self {
            return match cm.adapter.set_param(param, value) {
                Some(actual) => Ok(actual),
                None => Err(StructuralError::UnsupportedParameter {
                    kind: "custom",
                    parameter: param.as_str(),
                }),
            };
        }
        if !value.is_finite() {
            return Err(StructuralError::param(param.as_str(), "non-numeric value"));
        }
        let kind = self.as_str();
        if !param.supported_by(self) {
            return Err(StructuralError::UnsupportedParameter {
                kind,
                parameter: param.as_str(),
            });
        }
        match (&mut *self, param) {
            (NeuronModel::Lif(p), NeuronParam::Beta) => p.beta = value,
            (NeuronModel::Lif(p), NeuronParam::Threshold) => p.threshold = value,
            (NeuronModel::AdaptiveLif(p), NeuronParam::Beta) => p.beta = value,
            (NeuronModel::AdaptiveLif(p), NeuronParam::Rho) => p.rho = value,
            (NeuronModel::AdaptiveLif(p), NeuronParam::AdaptIncrement) => {
                p.adapt_increment = value;
            }
            (NeuronModel::AdaptiveLif(p), NeuronParam::Threshold) => p.threshold = value,
            (NeuronModel::Izhikevich(p), NeuronParam::IzhA) => p.a = value,
            (NeuronModel::Izhikevich(p), NeuronParam::IzhB) => p.b = value,
            (NeuronModel::Izhikevich(p), NeuronParam::IzhC) => p.c = value,
            (NeuronModel::Izhikevich(p), NeuronParam::IzhD) => p.d = value,
            (NeuronModel::Izhikevich(p), NeuronParam::Threshold) => p.threshold = value,
            (NeuronModel::AdEx(p), NeuronParam::AdExC) => p.c = value,
            (NeuronModel::AdEx(p), NeuronParam::AdExGl) => p.g_l = value,
            (NeuronModel::AdEx(p), NeuronParam::AdExEl) => p.e_l = value,
            (NeuronModel::AdEx(p), NeuronParam::AdExVt) => p.v_t = value,
            (NeuronModel::AdEx(p), NeuronParam::AdExDeltaT) => p.delta_t = value,
            (NeuronModel::AdEx(p), NeuronParam::AdExA) => p.a = value,
            (NeuronModel::AdEx(p), NeuronParam::AdExTauW) => p.tau_w = value,
            (NeuronModel::AdEx(p), NeuronParam::AdExB) => p.b = value,
            (NeuronModel::AdEx(p), NeuronParam::AdExVReset) => p.v_reset = value,
            (NeuronModel::AdEx(p), NeuronParam::AdExVPeak) => p.v_peak = value,
            (NeuronModel::HodgkinHuxley(p), NeuronParam::HhC) => p.c = value,
            (NeuronModel::HodgkinHuxley(p), NeuronParam::HhGNa) => p.g_na = value,
            (NeuronModel::HodgkinHuxley(p), NeuronParam::HhGK) => p.g_k = value,
            (NeuronModel::HodgkinHuxley(p), NeuronParam::HhGl) => p.g_l = value,
            (NeuronModel::HodgkinHuxley(p), NeuronParam::HhENa) => p.e_na = value,
            (NeuronModel::HodgkinHuxley(p), NeuronParam::HhEK) => p.e_k = value,
            (NeuronModel::HodgkinHuxley(p), NeuronParam::HhEL) => p.e_l = value,
            (NeuronModel::HodgkinHuxley(p), NeuronParam::HhVRest) => p.v_rest = value,
            (NeuronModel::HodgkinHuxley(p), NeuronParam::HhSpikeThreshold) => {
                p.spike_threshold = value;
            }
            _ => {
                return Err(StructuralError::UnsupportedParameter {
                    kind,
                    parameter: param.as_str(),
                });
            }
        }
        self.clamp(lim);
        Ok(self.param(param).expect("parameter supported by the model"))
    }

    /// One step of dynamics. Returns whether there was a spike.
    ///
    /// Errors of user-defined models are surfaced here (the network step
    /// returns `StructuralResult`).
    ///
    /// If the state variant does not match the model (should not happen
    /// after `Topology::validate`), the state is reinitialized, and the step
    /// is considered without a spike - the network does not panic.
    pub fn step(
        &self,
        state: &mut NeuronState<T>,
        current: T,
        ctx: &NeuronStepContext<T>,
    ) -> StructuralResult<bool> {
        match (self, &mut *state) {
            (NeuronModel::Lif(p), NeuronState::Lif(s)) => Ok(lif_step(p, s, current)),
            (NeuronModel::AdaptiveLif(p), NeuronState::AdaptiveLif(s)) => {
                Ok(adaptive_lif_step(p, s, current))
            }
            (NeuronModel::Izhikevich(p), NeuronState::Izhikevich(s)) => {
                Ok(izhikevich_step(p, s, current))
            }
            (NeuronModel::AdEx(p), NeuronState::AdEx(s)) => Ok(adex_step(p, s, current)),
            (NeuronModel::HodgkinHuxley(p), NeuronState::HodgkinHuxley(s)) => {
                Ok(hodgkin_huxley_step(p, s, current))
            }
            (NeuronModel::Custom(cm), NeuronState::Custom(entry)) => {
                cm.adapter.step(current, entry.inner_mut(), ctx)
            }
            _ => {
                debug_assert!(false, "the state does not correspond to the neuron model");
                *state = self.init_state();
                Ok(false)
            }
        }
    }
}

fn lif_step<T: Scalar>(p: &LifParams<T>, s: &mut LifState<T>, current: T) -> bool {
    s.mem = p.beta * s.mem + current;
    let fired = s.mem >= p.threshold;
    if fired {
        match p.reset {
            ResetMode::Subtract => s.mem -= p.threshold,
            ResetMode::Zero => s.mem = T::ZERO,
            ResetMode::None => {}
        }
    }
    fired
}

fn adaptive_lif_step<T: Scalar>(
    p: &AdaptiveLifParams<T>,
    s: &mut AdaptiveLifState<T>,
    current: T,
) -> bool {
    s.mem = p.beta * s.mem + current - s.adaptation;
    let fired = s.mem >= p.threshold;
    if fired {
        match p.reset {
            ResetMode::Subtract => s.mem -= p.threshold,
            ResetMode::Zero => s.mem = T::ZERO,
            ResetMode::None => {}
        }
        s.adaptation = p.rho * s.adaptation + p.adapt_increment;
    } else {
        s.adaptation *= p.rho;
    }
    fired
}

fn izhikevich_step<T: Scalar>(
    p: &IzhikevichParams<T>,
    s: &mut IzhikevichState<T>,
    current: T,
) -> bool {
    let dt = T::ONE / T::from_usize(p.substeps.max(1));
    for _ in 0..p.substeps {
        let dv = T::c(0.04) * s.v * s.v + T::c(5.0) * s.v + T::c(140.0) - s.u + current;
        s.v += dv * dt;
    }
    s.u += p.a * (p.b * s.v - s.u);
    let fired = s.v >= p.threshold;
    if fired {
        s.v = p.c;
        s.u += p.d;
    }
    if !s.v.is_finite() || !s.u.is_finite() {
        s.v = p.c;
        s.u = T::ZERO;
        return false;
    }
    fired
}

fn adex_step<T: Scalar>(p: &AdExParams<T>, s: &mut AdExState<T>, current: T) -> bool {
    let dt = p.dt / T::from_usize(p.substeps.max(1));
    let inv_c = T::ONE / p.c;
    let inv_delta = T::ONE / p.delta_t;
    let mut fired = false;
    for _ in 0..p.substeps {
        let exp_arg = ((s.v - p.v_t) * inv_delta).min(p.exp_clip);
        let i_exp = p.g_l * p.delta_t * exp_arg.exp();
        let i_leak = -p.g_l * (s.v - p.e_l);
        let dv = (i_leak + i_exp - s.w + current) * inv_c;
        let dw = (p.a * (s.v - p.e_l) - s.w) / p.tau_w;
        s.v += dv * dt;
        s.w += dw * dt;
        if s.v >= p.v_peak {
            s.v = p.v_reset;
            s.w += p.b;
            fired = true;
        }
    }
    if !s.v.is_finite() || !s.w.is_finite() {
        s.v = p.e_l;
        s.w = T::ZERO;
        return false;
    }
    fired
}

fn hodgkin_huxley_step<T: Scalar>(
    p: &HodgkinHuxleyParams<T>,
    s: &mut HodgkinHuxleyState<T>,
    current: T,
) -> bool {
    let dt = p.dt / T::from_usize(p.substeps.max(1));
    let inv_c = T::ONE / p.c;
    let mut fired = false;
    for _ in 0..p.substeps {
        let [am, bm, ah, bh, an, bn] = crate::neuron::hh_rates(s.v);
        s.m = gate_update(s.m, am, bm, dt);
        s.h = gate_update(s.h, ah, bh, dt);
        s.n = gate_update(s.n, an, bn, dt);
        let i_na = p.g_na * s.m * s.m * s.m * s.h * (s.v - p.e_na);
        let i_k = p.g_k * s.n * s.n * s.n * s.n * (s.v - p.e_k);
        let i_l = p.g_l * (s.v - p.e_l);
        let dv = (current - i_na - i_k - i_l) * inv_c;
        s.v += dv * dt;
        let above = s.v >= p.spike_threshold;
        if above && !s.above {
            fired = true;
        }
        s.above = above;
    }
    if !s.v.is_finite() || !s.m.is_finite() || !s.h.is_finite() || !s.n.is_finite() {
        let init = NeuronModel::HodgkinHuxley(*p).init_state();
        if let NeuronState::HodgkinHuxley(reset) = init {
            *s = reset;
        }
        return false;
    }
    fired
}

fn gate_update<T: Scalar>(x: T, alpha: T, beta: T, dt: T) -> T {
    let total = alpha + beta;
    if total <= T::ZERO || !total.is_finite() {
        return x;
    }
    let x_inf = alpha / total;
    x_inf + (x - x_inf) * (-dt * total).exp()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::structural::topology::NeuronId;

    fn ctx() -> NeuronStepContext<f32> {
        NeuronStepContext::manual(NeuronId(0), 0, 0)
    }

    #[test]
    fn model_state_variants_match() {
        let models = [
            NeuronModel::Lif(LifParams::<f32>::default()),
            NeuronModel::AdaptiveLif(AdaptiveLifParams::default()),
            NeuronModel::Izhikevich(IzhikevichParams::default()),
            NeuronModel::AdEx(AdExParams::default()),
            NeuronModel::HodgkinHuxley(HodgkinHuxleyParams::default()),
        ];
        for m in models {
            let s = m.init_state();
            assert!(s.matches(&m));
            assert!(s.is_finite());
            assert_eq!(s.tag(), m.tag());
        }
    }

    #[test]
    fn lif_fires_and_resets() {
        let p = LifParams::new(0.5, 1.0);
        let model = NeuronModel::Lif(p);
        let mut state = NeuronState::Lif(LifState::default());
        assert!(!model.step(&mut state, 0.4, &ctx()).unwrap());
        // step with current 1.0: mem = 0.5*0.4 + 1.0 = 1.2 → spike, reset Subtract
        assert!(model.step(&mut state, 1.0, &ctx()).unwrap());
        let NeuronState::Lif(s) = state else {
            unreachable!()
        };
        assert!((s.mem - 0.2).abs() < 1e-6);
    }

    #[test]
    fn adaptive_lif_adapts_after_spike() {
        let p = AdaptiveLifParams {
            beta: 0.0,
            rho: 0.9,
            adapt_increment: 0.5,
            threshold: 1.0,
            reset: ResetMode::Zero,
        };
        let mut s = AdaptiveLifState::default();
        let mut state = NeuronState::AdaptiveLif(s);
        assert!(
            NeuronModel::AdaptiveLif(p)
                .step(&mut state, 1.0, &ctx())
                .unwrap()
        );
        if let NeuronState::AdaptiveLif(st) = state {
            assert!((st.adaptation - 0.5).abs() < 1e-6);
            s = st;
        }
        // without input, adaptation inhibits the membrane and gradually fades
        assert!(
            !NeuronModel::AdaptiveLif(p)
                .step(&mut NeuronState::AdaptiveLif(s), 0.0, &ctx())
                .unwrap()
        );
    }

    #[test]
    fn izhikevich_regular_spiking_stays_finite_and_fires() {
        let p = IzhikevichParams::regular_spiking();
        let mut state = NeuronModel::Izhikevich(p).init_state();
        let model = NeuronModel::Izhikevich(p);
        let mut fired = 0;
        for _ in 0..200 {
            if model.step(&mut state, 10.0, &ctx()).unwrap() {
                fired += 1;
            }
            assert!(state.is_finite());
        }
        assert!(fired > 0);
    }

    #[test]
    fn adex_fires_under_strong_current() {
        let p = AdExParams::default();
        let model = NeuronModel::AdEx(p);
        let mut state = model.init_state();
        let mut fired = 0;
        for _ in 0..400 {
            if model.step(&mut state, 1000.0, &ctx()).unwrap() {
                fired += 1;
            }
            assert!(state.is_finite());
        }
        assert!(fired > 0);
    }

    #[test]
    fn hodgkin_huxley_fires_under_current() {
        let p = HodgkinHuxleyParams::default();
        let model = NeuronModel::HodgkinHuxley(p);
        let mut state = model.init_state();
        let mut fired = 0;
        for _ in 0..200 {
            if model.step(&mut state, 10.0, &ctx()).unwrap() {
                fired += 1;
            }
            assert!(state.is_finite());
        }
        assert!(fired > 0);
    }

    #[test]
    fn clamp_keeps_model_valid() {
        let lim = ParamLimits::default();
        let mut models = [
            NeuronModel::Lif(LifParams {
                beta: 2.0,
                threshold: -1.0,
                reset: ResetMode::Zero,
            }),
            NeuronModel::AdaptiveLif(AdaptiveLifParams {
                beta: 2.0,
                rho: 2.0,
                adapt_increment: -1.0,
                threshold: -1.0,
                reset: ResetMode::Zero,
            }),
            NeuronModel::Izhikevich(IzhikevichParams {
                a: 5.0,
                b: -1.0,
                c: -65.0,
                d: -1.0,
                threshold: -100.0,
                substeps: 0,
            }),
            NeuronModel::AdEx(AdExParams {
                c: 0.0,
                g_l: 0.0,
                delta_t: 0.0,
                tau_w: 0.0,
                dt: 0.0,
                exp_clip: 0.0,
                substeps: 0,
                v_peak: -100.0,
                v_reset: 0.0,
                ..AdExParams::default()
            }),
            NeuronModel::HodgkinHuxley(HodgkinHuxleyParams {
                c: 0.0,
                g_na: 0.0,
                g_k: 0.0,
                g_l: 0.0,
                dt: 0.0,
                substeps: 0,
                ..HodgkinHuxleyParams::default()
            }),
        ];
        for m in &mut models {
            m.clamp(&lim);
            m.validate(&lim)
                .unwrap_or_else(|e| panic!("{:?}: {e}", m.as_str()));
        }
    }
}
