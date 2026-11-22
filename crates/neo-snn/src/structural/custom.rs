//! User-defined dynamic models for the event-driven [`DynamicNetwork`](crate::structural::DynamicNetwork).
//!
//! This module is the extension point for the scalar world. A *builtin* model is one of
//! [`NeuronModel`](crate::structural::NeuronModel) (`Lif`, `AdaptiveLif`, `Izhikevich`, `AdEx`,
//! `HodgkinHuxley`) and a *custom* model is anything the user implements through the
//! [`DynamicNeuronModel`] trait: arbitrary dynamics, arbitrary state, arbitrary spike rule,
//! reset, parameters, serialization and evolution hooks.
//!
//! # Two kinds of state
//!
//! * **Builtin**: [`NeuronState`](crate::structural::NeuronState) keeps `Lif/AdaptiveLif/...`
//!   variants and the network executes builtin dynamics through the static executor.
//! * **Custom**: [`NeuronState::Custom`](crate::structural::NeuronState::Custom) holds an
//!   erased [`CustomStateEntry`]; the user state struct can have any fields and types.
//!
//! # Contract (read [`DynamicNeuronModel::step`])
//!
//! The network delivers already-modulated currents and handles spikes, delays, Dale
//! principle, statistics and plasticity — the user model only integrates its own
//! variables and reports whether it spiked this step.
//!
//! # Example
//!
//! See [docs/custom_neuron_models.md](../../../docs/custom_neuron_models.md)
//! for a full FitzHugh–Nagumo walkthrough, and
//! `examples/custom_dynamic_neuron.rs` / `examples/mixed_custom_network.rs`
//! for runnable examples.
//!
//! ```ignore
//! # let model = /* any `DynamicNeuronModel<T>` */;
//! # let mut net = net;
//! net.change_neuron_model(id, CustomNeuronInstance::new(model), StateMigration::Reset)?;
//! ```

use std::any::Any;
use std::collections::BTreeMap;

use crate::scalar::Scalar;

use super::error::{StructuralError, StructuralResult};
use super::rng::Rng64;
use super::topology::{NeuronId, ParamLimits};

use super::model::{NeuronParam, StateMigration};

// ---------------------------------------------------------------------
// Step context
// ---------------------------------------------------------------------

/// Context of one dynamic step, passed to every custom model.
///
/// Keeps the engine free of globals: the user model gets reproducible time and
/// counters without reading clocks or thread-locals.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NeuronStepContext<T = f32> {
    /// Neuron identifier in the topology (`NeuronId`).
    pub neuron_id: NeuronId,
    /// Global step counter of the network (`DynamicNetwork::step_count`).
    pub step: u64,
    /// Global sample (episode) counter.
    pub sample: u64,
    /// Step duration of the network. May be `T::ONE` if the engine does not define one.
    pub dt: T,
}

impl<T: Scalar> NeuronStepContext<T> {
    /// Context for a manual step.
    pub fn manual(neuron_id: NeuronId, step: u64, sample: u64) -> Self {
        Self {
            neuron_id,
            step,
            sample,
            dt: T::ONE,
        }
    }
}

// ---------------------------------------------------------------------
// Parameters
// ---------------------------------------------------------------------

/// Mutation strategy for a user parameter.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CustomParamMutation<T = f32> {
    /// No automatic mutation (default).
    None,
    /// Random walk: `value += scale * u`, `u ∈ (-1, 1)`.
    RandomWalk {
        /// Absolute step scale.
        scale: T,
    },
    /// Multiplicative: `value *= factor`.
    Factor {
        /// Relative factor.
        factor: T,
    },
}

impl<T: Scalar> CustomParamMutation<T> {
    /// Apply the strategy mutation step to `current`.
    pub fn step(&self, rng: &mut Rng64, current: T) -> T {
        match *self {
            CustomParamMutation::None => current,
            CustomParamMutation::RandomWalk { scale } => current + scale * rng.symmetric_as::<T>(),
            CustomParamMutation::Factor { factor } => current * factor,
        }
    }
}

/// A named parameter of a custom model (string id - no enum variants needed).
#[derive(Debug, Clone, PartialEq)]
pub struct CustomParameter<T = f32> {
    /// Stable identifier (see [`CustomNeuronModel::set_custom_param`](DynamicNeuronModel::set_custom_param)).
    pub id: String,
    /// Current value.
    pub value: T,
    /// Minimum (optional).
    pub min: Option<T>,
    /// Maximum (optional).
    pub max: Option<T>,
    /// May it be mutated automatically by evolution?
    pub mutable: bool,
    /// Mutation strategy.
    pub mutation: CustomParamMutation<T>,
}

impl<T: Scalar> CustomParameter<T> {
    /// Builder: parameter with name, value and strategy.
    pub fn new(id: impl Into<String>, value: T) -> Self {
        Self {
            id: id.into(),
            value,
            min: None,
            max: None,
            mutable: false,
            mutation: CustomParamMutation::None,
        }
    }

    /// With bounds.
    pub fn bounds(mut self, min: Option<T>, max: Option<T>) -> Self {
        self.min = min;
        self.max = max;
        self
    }

    /// With the automatic mutation strategy (requires `mutable`).
    pub fn mutable_random_walk(mut self, scale: T) -> Self {
        self.mutable = true;
        self.mutation = CustomParamMutation::RandomWalk { scale };
        self
    }

    /// Clamp `value` into `[min, max]` if the bounds are set.
    pub fn clamp_value(&self, value: T) -> T {
        value.clamp(self.min.unwrap_or(value), self.max.unwrap_or(value))
    }
}

// ---------------------------------------------------------------------
// Observation
// ---------------------------------------------------------------------

/// Unified description of observable model characteristics.
///
/// A model returns `None` where the quantity is not physical for it
/// (for example, membrane of a Resonate-and-Fire model). The engine and GUI
/// handle `None` gracefully: nothing is invented, nothing is displayed.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct NeuronObservation<T = f32> {
    /// Membrane/potential value, if the model has one.
    pub membrane: Option<T>,
    /// Current spike threshold, if the model has one.
    pub threshold: Option<T>,
    /// Resting membrane, if the model has one.
    pub resting_potential: Option<T>,
    /// Firing membrane value, if the model has one.
    pub firing_potential: Option<T>,
}

impl<T: Scalar> NeuronObservation<T> {
    /// Membrane: [`None`] if not supported.
    pub fn membrane(&self) -> Option<T> {
        self.membrane
    }
    /// Threshold: [`None`] if not supported.
    pub fn threshold(&self) -> Option<T> {
        self.threshold
    }
}

// ---------------------------------------------------------------------
// Custom state entry (type-erased)
// ---------------------------------------------------------------------

/// Erased view of a user state struct. Implemented per user model
/// (the trait is object-safe: clone/equality go through the box methods).
pub trait CustomStateEntry<T: Scalar>: std::fmt::Debug + Send + Sync + 'static {
    /// Downcast to the concrete state type.
    fn as_any(&self) -> &dyn Any;
    /// Downcast to the concrete state type (mutable).
    fn as_any_mut(&mut self) -> &mut dyn Any;
    /// Stable model id (the same as [`DynamicNeuronModel::model_id`]).
    fn model_id(&self) -> &'static str;
    /// Schema version of the model that created this state.
    fn schema_version(&self) -> u32;
    /// All state components are finite (not NaN, not ±Inf).
    fn is_finite(&self) -> bool;
    /// Membrane observation, if the model provides one.
    fn membrane(&self) -> Option<T>;
    /// Set the membrane value (no-op if unsupported).
    fn set_membrane(&mut self, value: T);
    /// Membrane range `(rest, fire)` of the state, if physical (default: none).
    fn membrane_range(&self) -> Option<(T, T)> {
        None
    }
    /// Clone into an erased box (implemented per model as `Box::new(self.clone())`
    /// with `M::State: Clone`).
    fn cloned_box(&self) -> Box<dyn CustomStateEntry<T>>;
    /// Structural equality against another entry of the same model type
    /// (downcast-based, by default compares the concrete values).
    fn state_eq_box(&self, other: &dyn CustomStateEntry<T>) -> bool;
}

/// Public wrapper of `Box<dyn CustomStateEntry<T>>`: provides Clone/PartialEq
/// via the entry box methods.
#[derive(Debug)]
pub struct CustomStateEntryBox<T: Scalar>(pub(crate) Box<dyn CustomStateEntry<T>>);

impl<T: Scalar> Clone for CustomStateEntryBox<T> {
    fn clone(&self) -> Self {
        Self(self.0.cloned_box())
    }
}

impl<T: Scalar> PartialEq for CustomStateEntryBox<T> {
    fn eq(&self, other: &Self) -> bool {
        self.0.state_eq_box(&*other.0)
    }
}

impl<T: Scalar> CustomStateEntryBox<T> {
    /// Wrap a concrete state entry.
    pub fn new(entry: impl CustomStateEntry<T> + 'static) -> Self {
        Self(Box::new(entry))
    }
    /// Wrap an already boxed entry.
    pub fn from_boxed(entry: Box<dyn CustomStateEntry<T>>) -> Self {
        Self(entry)
    }
    /// Access the erased entry.
    pub fn inner(&self) -> &dyn CustomStateEntry<T> {
        &*self.0
    }
    /// Access the erased entry (mutable).
    pub fn inner_mut(&mut self) -> &mut dyn CustomStateEntry<T> {
        &mut *self.0
    }
    /// Downcast the erased entry to the concrete state type of the model `M`.
    pub fn downcast_ref<M: 'static>(&self) -> Option<&M> {
        self.0.as_any().downcast_ref::<M>()
    }
    /// Downcast the erased entry to the concrete state type of the model `M` (mutable).
    pub fn downcast_ref_mut<M: 'static>(&mut self) -> Option<&mut M> {
        self.0.as_any_mut().downcast_mut::<M>()
    }
}

// ---------------------------------------------------------------------
// The user model trait
// ---------------------------------------------------------------------

/// User-defined dynamic model for the event-driven network.
///
/// # Dynamics contract
///
/// `step` receives the already delivered current and updates the state. The model
/// returns `true` when it fired during this step. The network handles spikes,
/// delays, statistics, STDP, etc. — the model does not.
///
/// # State
///
/// The model defines `State` freely (any struct). The state must be `Clone + Debug + PartialEq`
/// and must reflect all quantities used in dynamics.
///
/// # Optional capabilities
///
/// * [`membrane`](Self::membrane)/[`threshold`](Self::threshold) for monitoring;
/// * [`migrate_from_custom`](Self::migrate_from_custom) for state migration policies;
/// * [`crossover`](Self::crossover)/[`mutate`](Self::mutate) for NEAT/evolution;
/// * [`codec`](Self::codec) for checkpoint save/load.
pub trait DynamicNeuronModel<T: Scalar>: Clone + Send + Sync + 'static {
    /// State of the model: any struct that implements [`CustomStateEntry`]
    /// (reflects all quantities used in dynamics).
    type State: CustomStateEntry<T> + Clone;

    /// Stable model identifier (`"company/fhn"`, `"org_name/morris_lecar"`, ...).
    /// Must be unique in a [`CustomNeuronRegistry`](super::custom::CustomNeuronRegistry).
    fn model_id(&self) -> &'static str;

    /// Schema version of the model state/config format (for migration checks).
    fn schema_version(&self) -> u32 {
        1
    }

    /// Initial state (calm).
    fn init_state(&self) -> Self::State;

    /// One dynamics step. Returns `true` when the neuron fired this step.
    ///
    /// Errors are surfaced to the network: the network step returns
    /// [`StructuralError`](super::error::StructuralError).
    fn step(
        &self,
        input: T,
        state: &mut Self::State,
        context: &NeuronStepContext<T>,
    ) -> StructuralResult<bool>;

    /// Validate the model parameters (called by `Topology::validate` on add/change).
    fn validate(&self, limits: &ParamLimits<T>) -> StructuralResult<()> {
        let _ = limits;
        Ok(())
    }

    /// Membrane observation of the state, if physical for the model.
    fn membrane(&self, state: &Self::State) -> Option<T> {
        let _ = state;
        None
    }

    /// Spike threshold, if the model has one (used by homeostasis; `None` = not applicable).
    fn threshold(&self) -> Option<T> {
        None
    }

    /// Membrane range `(rest, fire)` of the model, if physical
    /// (used by [`StateMigration::Scaled`]).
    fn membrane_range(&self) -> Option<(T, T)> {
        None
    }

    /// Set the membrane value (delegates to the entry
    /// [`CustomStateEntry::set_membrane`] by default).
    fn state_set_mem(&self, state: &mut Self::State, value: T) {
        state.set_membrane(value);
    }

    /// Are all state components finite?
    fn state_is_finite(state: &Self::State) -> bool {
        let _ = state;
        true
    }

    /// All state variables (used by structural monitoring).
    fn observe(&self, state: &Self::State) -> NeuronObservation<T> {
        let _ = state;
        NeuronObservation::default()
    }

    /// Continuous physical parameters supported by the model (engine-level;
    /// for custom models usually empty - see
    /// [`NeuronModel::custom_params`](crate::structural::NeuronModel::custom_params)).
    fn params(&self) -> Vec<(NeuronParam, T)> {
        Vec::new()
    }

    /// Write a builtin physical parameter (see [`NeuronParam`]); `None` - unsupported.
    fn set_param(&mut self, param: NeuronParam, value: T) -> Option<T> {
        let _ = (param, value);
        None
    }

    // -----------------------------------------------------------------
    // State migration (Custom target, policies Membrane/Compatible/Scaled)
    // -----------------------------------------------------------------

    /// Migrate a state from another custom model (different id) under the given policy.
    ///
    /// Default implementation:
    /// * `Compatible`/`Scaled`: same model id - full state is kept
    ///   (`source.as_any().downcast_ref::<Self::State>()`);
    /// * otherwise the state is re-initialized and the membrane is copied
    ///   (scaled between rest/fire ranges if both models provide them).
    ///
    /// Implement to transfer additional variables on model switch.
    fn migrate_from_custom(
        &self,
        source: &dyn CustomStateEntry<T>,
        policy: StateMigration,
    ) -> StructuralResult<Self::State> {
        if source.model_id() == self.model_id()
            && let Some(state) = source.as_any().downcast_ref::<Self::State>()
        {
            return Ok((*state).clone());
        }
        match policy {
            StateMigration::Reset => Ok(self.init_state()),
            StateMigration::Membrane | StateMigration::Compatible | StateMigration::Scaled => {
                let mut state = self.init_state();
                if let Some(mem) = source.membrane() {
                    self.state_set_mem(&mut state, mem);
                }
                if policy == StateMigration::Scaled
                    && let (Some(mem), (Some((src_rest, src_fire)), Some((dst_rest, dst_fire)))) = (
                        source.membrane(),
                        (source.membrane_range(), self.membrane_range()),
                    )
                    && src_fire > src_rest
                    && dst_fire > dst_rest
                {
                    let frac =
                        ((mem - src_rest) / (src_fire - src_rest)).clamp(-T::c(0.5), T::c(0.999));
                    if frac.is_finite() {
                        self.state_set_mem(&mut state, dst_rest + frac * (dst_fire - dst_rest));
                    }
                }
                Ok(state)
            }
        }
    }

    /// Migrate the state from a builtin state to this custom model
    /// (Builtin → Custom).
    ///
    /// Default: reset the state, then copy the builtin membrane if supported
    /// (`Membrane`/`Compatible`/`Scaled`).
    fn migrate_from_builtin(
        &self,
        source_mem: Option<T>,
        policy: StateMigration,
    ) -> StructuralResult<Self::State> {
        let mut state = self.init_state();
        if let Some(mem) = source_mem {
            self.state_set_mem(&mut state, mem);
        }
        let _ = policy;
        Ok(state)
    }

    // -----------------------------------------------------------------
    // Evolution (NEAT) - may be omitted.
    // -----------------------------------------------------------------

    /// Crossover of two genes. Returns `Ok(Some(child))` - the child (any
    /// model wins the crossover), `Ok(None)` - keep `self` (no strategy /
    /// incompatible models).
    ///
    /// `same_id == true`: both models have the same `model_id`.
    fn crossover(&self, _other: &Self, same_id: bool) -> StructuralResult<Option<Self>>
    where
        Self: Sized,
    {
        // Default: keep parent for different ids, standard compatible
        // strategy for the same id (the state itself is later reset by the
        // engine through `StateMigration::Compatible`).
        if same_id {
            Ok(Some(self.clone()))
        } else {
            Ok(None)
        }
    }

    /// Random mutation of the model in place. Returns whether anything changed.
    fn mutate(&mut self, _rng: &mut Rng64) -> bool {
        // Mutate the declared custom parameters.
        let mut changed = false;
        for p in self.custom_params() {
            if !p.mutable {
                continue;
            }
            let new = p.mutation.step(_rng, p.value);
            let actual = p.clamp_value(new);
            if self.set_custom_param(&p.id, actual) != Some(actual) || actual != p.value {
                changed = true;
            }
        }
        changed
    }

    /// Named (string-id) parameters of the model, for evolution/telemetry.
    ///
    /// Return `Vec::new()` when the model has no named parameters
    /// (builtin-style parameters are exposed through
    /// [`NeuronModel::params`](crate::structural::NeuronModel::params)).
    fn custom_params(&self) -> Vec<CustomParameter<T>> {
        Vec::new()
    }

    /// Current value of a named parameter (None = unsupported/unknown).
    fn get_custom_param(&self, id: &str) -> Option<T> {
        let _ = id;
        None
    }

    /// Set a named parameter; returns the actual value after clamping
    /// (None = unsupported).
    fn set_custom_param(&mut self, id: &str, value: T) -> Option<T> {
        let _ = (id, value);
        None
    }

    // -----------------------------------------------------------------
    // Serialization (checkpoint, genome). Optional.
    // -----------------------------------------------------------------

    /// Serialize the model configuration. `None` when unsupported:
    /// `save_checkpoint` then fails with a clear error instead of losing the neuron.
    fn codec(&self) -> Option<Box<dyn CustomCodec<T>>> {
        None
    }
}

// ---------------------------------------------------------------------
// Codec (checkpoint & genome codec)
// ---------------------------------------------------------------------

/// Serialization of a custom model: model config, state, and decode back.
///
/// Only `model_id`+`schema_version`+bytes are stored; no code is loaded back
/// (registry provides the model implementation).
pub trait CustomCodec<T: Scalar>: Send + Sync {
    /// Encode the model configuration (without state).
    fn encode_model(&self, model: &CustomNeuronInstance<T>) -> StructuralResult<Vec<u8>>;
    /// Encode the state of `state` (the model id/version must match `model`).
    fn encode_state(
        &self,
        model: &CustomNeuronInstance<T>,
        state: &dyn CustomStateEntry<T>,
    ) -> StructuralResult<Vec<u8>>;
    /// Decode a model + optionally state bytes into a new instance.
    fn decode_model(
        &self,
        model_bytes: &[u8],
        state_bytes: &[u8],
    ) -> StructuralResult<CustomNeuronInstance<T>>;
    /// Decode just the state bytes into an erased entry (for `r_state`).
    fn decode_state(&self, state_bytes: &[u8]) -> StructuralResult<Box<dyn CustomStateEntry<T>>>;
    /// Stable model id.
    fn model_id(&self) -> &str;
    /// Schema version.
    fn schema_version(&self) -> u32;
}

/// Helper: encode a list of `f32` values as a little-endian byte vector.
pub fn encode_f32s(values: &[f32]) -> Vec<u8> {
    let mut out = Vec::with_capacity(values.len() * 4);
    for v in values {
        out.extend_from_slice(&v.to_bits().to_le_bytes());
    }
    out
}

/// Helper: decode a list of `f32` values written by [`encode_f32s`].
pub fn read_f32s(data: &[u8]) -> StructuralResult<Vec<f32>> {
    if !data.len().is_multiple_of(4) {
        return Err(StructuralError::CustomSerialization(
            "f32-block".into(),
            "odd length".into(),
        ));
    }
    Ok(data
        .as_chunks::<4>()
        .0
        .iter()
        .map(|c| f32::from_bits(u32::from_le_bytes(*c)))
        .collect())
}

/// Helper: encode scalar values of `T: Scalar` (through `f64` bits).
pub fn encode_scalars<T: Scalar>(values: &[T]) -> Vec<u8> {
    let mut out = Vec::with_capacity(values.len() * 8);
    for v in values {
        out.extend_from_slice(&v.to_f64().to_bits().to_le_bytes());
    }
    out
}

/// Helper: decode scalar values of `T: Scalar` written by [`encode_scalars`].
pub fn read_scalars<T: Scalar>(data: &[u8]) -> StructuralResult<Vec<T>> {
    if !data.len().is_multiple_of(8) {
        return Err(StructuralError::CustomSerialization(
            "scalar-block".into(),
            "odd length".into(),
        ));
    }
    Ok(data
        .as_chunks::<8>()
        .0
        .iter()
        .map(|c| T::from_f64(f64::from_bits(u64::from_le_bytes(*c))))
        .collect())
}

// ---------------------------------------------------------------------
// Erased adapter (private)
// ---------------------------------------------------------------------

pub(crate) trait CustomAdapter<T: Scalar>: Send + Sync {
    fn as_any(&self) -> &dyn Any;
    fn clone_adapter(&self) -> Box<dyn CustomAdapter<T>>;
    fn validate(&self, limits: &ParamLimits<T>) -> StructuralResult<()>;
    fn model_id(&self) -> &'static str;
    fn schema_version(&self) -> u32;
    fn config_fingerprint(&self) -> Option<Vec<u8>>;
    fn codec(&self) -> Option<Box<dyn CustomCodec<T>>>;
    fn step(
        &self,
        input: T,
        state: &mut dyn CustomStateEntry<T>,
        ctx: &NeuronStepContext<T>,
    ) -> StructuralResult<bool>;
    fn init_state(&self) -> Box<dyn CustomStateEntry<T>>;
    fn state_is_finite(&self, state: &dyn CustomStateEntry<T>) -> bool;
    fn observe(&self, state: &dyn CustomStateEntry<T>) -> NeuronObservation<T>;
    fn set_mem(&self, state: &mut dyn CustomStateEntry<T>, value: T);
    fn membrane(&self, state: &dyn CustomStateEntry<T>) -> Option<T>;
    fn threshold(&self) -> Option<T>;
    fn membrane_range(&self) -> Option<(T, T)>;
    fn params(&self) -> Vec<(super::model::NeuronParam, T)>;
    fn set_param(&mut self, param: super::model::NeuronParam, value: T) -> Option<T>;
    fn custom_params(&self) -> Vec<CustomParameter<T>>;
    fn get_custom_param(&self, id: &str) -> Option<T>;
    fn set_custom_param(&mut self, id: &str, value: T) -> Option<T>;
    fn encode_config(&self) -> StructuralResult<Vec<u8>>;
    fn encode_state(&self, state: &dyn CustomStateEntry<T>) -> StructuralResult<Vec<u8>>;
    fn decode_state(&self, bytes: &[u8]) -> StructuralResult<Box<dyn CustomStateEntry<T>>>;
    fn migrate_state(
        &self,
        source: &dyn CustomStateEntry<T>,
        policy: StateMigration,
    ) -> StructuralResult<Box<dyn CustomStateEntry<T>>>;
    fn crossover(
        &self,
        other: &CustomNeuronInstance<T>,
        same_id: bool,
    ) -> StructuralResult<Option<CustomNeuronInstance<T>>>;
    fn mutate(&mut self, rng: &mut Rng64) -> bool;
}

struct CustomAdapterImpl<M> {
    model: M,
}

impl<M: 'static> CustomAdapterImpl<M> {
    fn codec_or_err<T: Scalar>(&self) -> StructuralResult<Box<dyn CustomCodec<T>>>
    where
        M: DynamicNeuronModel<T>,
    {
        self.model.codec().ok_or_else(|| {
            StructuralError::CustomSerialization(
                self.model.model_id().to_string(),
                "model does not implement a codec (`DynamicNeuronModel::codec`)".into(),
            )
        })
    }
}

impl<T: Scalar, M: DynamicNeuronModel<T>> CustomAdapter<T> for CustomAdapterImpl<M> {
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn validate(&self, limits: &ParamLimits<T>) -> StructuralResult<()> {
        self.model.validate(limits)
    }
    fn clone_adapter(&self) -> Box<dyn CustomAdapter<T>> {
        Box::new(Self {
            model: self.model.clone(),
        })
    }
    fn model_id(&self) -> &'static str {
        self.model.model_id()
    }
    fn schema_version(&self) -> u32 {
        self.model.schema_version()
    }
    fn config_fingerprint(&self) -> Option<Vec<u8>> {
        self.encode_config().ok()
    }
    fn codec(&self) -> Option<Box<dyn CustomCodec<T>>> {
        self.model.codec()
    }
    fn step(
        &self,
        input: T,
        state: &mut dyn CustomStateEntry<T>,
        ctx: &NeuronStepContext<T>,
    ) -> StructuralResult<bool> {
        let Some(state) = state.as_any_mut().downcast_mut::<M::State>() else {
            return Err(StructuralError::IncompatibleCustomState(
                self.model.model_id().to_string(),
            ));
        };
        self.model.step(input, state, ctx)
    }
    fn init_state(&self) -> Box<dyn CustomStateEntry<T>> {
        Box::new(self.model.init_state())
    }
    fn state_is_finite(&self, state: &dyn CustomStateEntry<T>) -> bool {
        state
            .as_any()
            .downcast_ref::<M::State>()
            .is_some_and(M::state_is_finite)
    }
    fn observe(&self, state: &dyn CustomStateEntry<T>) -> NeuronObservation<T> {
        match state.as_any().downcast_ref::<M::State>() {
            Some(s) => self.model.observe(s),
            None => NeuronObservation::default(),
        }
    }
    fn set_mem(&self, state: &mut dyn CustomStateEntry<T>, value: T) {
        if let Some(s) = state.as_any_mut().downcast_mut::<M::State>() {
            self.model.state_set_mem(s, value);
        }
    }
    fn membrane(&self, state: &dyn CustomStateEntry<T>) -> Option<T> {
        state
            .as_any()
            .downcast_ref::<M::State>()
            .and_then(|s| self.model.membrane(s))
    }
    fn threshold(&self) -> Option<T> {
        self.model.threshold()
    }
    fn membrane_range(&self) -> Option<(T, T)> {
        self.model.membrane_range()
    }
    fn params(&self) -> Vec<(super::model::NeuronParam, T)> {
        self.model.params()
    }
    fn set_param(&mut self, param: super::model::NeuronParam, value: T) -> Option<T> {
        self.model.set_param(param, value)
    }
    fn custom_params(&self) -> Vec<CustomParameter<T>> {
        self.model.custom_params()
    }
    fn get_custom_param(&self, id: &str) -> Option<T> {
        self.model.get_custom_param(id)
    }
    fn set_custom_param(&mut self, id: &str, value: T) -> Option<T> {
        self.model.set_custom_param(id, value)
    }
    fn encode_config(&self) -> StructuralResult<Vec<u8>> {
        let codec = self.codec_or_err()?;
        let instance = CustomNeuronInstance {
            adapter: Box::new(Self {
                model: self.model.clone(),
            }),
        };
        codec.encode_model(&instance)
    }
    fn encode_state(&self, state: &dyn CustomStateEntry<T>) -> StructuralResult<Vec<u8>> {
        let codec = self.codec_or_err()?;
        let instance = CustomNeuronInstance {
            adapter: Box::new(Self {
                model: self.model.clone(),
            }),
        };
        codec.encode_state(&instance, state)
    }
    fn decode_state(&self, bytes: &[u8]) -> StructuralResult<Box<dyn CustomStateEntry<T>>> {
        let codec = self.codec_or_err()?;
        codec.decode_state(bytes)
    }
    fn migrate_state(
        &self,
        source: &dyn CustomStateEntry<T>,
        policy: StateMigration,
    ) -> StructuralResult<Box<dyn CustomStateEntry<T>>> {
        Ok(Box::new(self.model.migrate_from_custom(source, policy)?))
    }
    fn crossover(
        &self,
        other: &CustomNeuronInstance<T>,
        same_id: bool,
    ) -> StructuralResult<Option<CustomNeuronInstance<T>>> {
        let Some(other_m) = other.adapter.as_any().downcast_ref::<Self>() else {
            return Err(StructuralError::IncompatibleCustomState(
                self.model.model_id().to_string(),
            ));
        };
        self.model
            .crossover(&other_m.model, same_id)
            .map(|m| m.map(CustomNeuronInstance::new))
    }
    fn mutate(&mut self, rng: &mut Rng64) -> bool {
        self.model.mutate(rng)
    }
}

// ---------------------------------------------------------------------
// CustomNeuronInstance
// ---------------------------------------------------------------------

/// Type-erased handle of a user model + its state type.
///
/// Created via [`CustomNeuronInstance::new`] from any concrete [`DynamicNeuronModel`].
/// Stored inside [`NeuronModel::Custom`](crate::structural::NeuronModel::Custom).
pub struct CustomNeuronInstance<T: Scalar = f32> {
    pub(crate) adapter: Box<dyn CustomAdapter<T>>,
}

impl<T: Scalar> std::fmt::Debug for CustomNeuronInstance<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("CustomNeuronInstance")
            .field("model_id", &self.model_id())
            .field("schema_version", &self.schema_version())
            .finish()
    }
}

impl<T: Scalar> CustomNeuronInstance<T> {
    /// Wrap a concrete model.
    pub fn new<M: DynamicNeuronModel<T>>(model: M) -> Self {
        Self {
            adapter: Box::new(CustomAdapterImpl { model }),
        }
    }

    /// Stable model id.
    pub fn model_id(&self) -> &'static str {
        self.adapter.model_id()
    }
    /// Schema version.
    pub fn schema_version(&self) -> u32 {
        self.adapter.schema_version()
    }
    /// Initial state entry.
    pub fn init_state(&self) -> Box<dyn CustomStateEntry<T>> {
        self.adapter.init_state()
    }
    /// One dynamics step on an erased state entry.
    pub fn step(
        &self,
        input: T,
        state: &mut dyn CustomStateEntry<T>,
        ctx: &NeuronStepContext<T>,
    ) -> StructuralResult<bool> {
        self.adapter.step(input, state, ctx)
    }
    /// Membrane observation.
    pub fn membrane(&self, state: &dyn CustomStateEntry<T>) -> Option<T> {
        self.adapter.membrane(state)
    }
    /// Threshold (None = not applicable).
    pub fn threshold(&self) -> Option<T> {
        self.adapter.threshold()
    }
    /// Membrane range.
    pub fn membrane_range(&self) -> Option<(T, T)> {
        self.adapter.membrane_range()
    }
    /// Full observation.
    pub fn observe(&self, state: &dyn CustomStateEntry<T>) -> NeuronObservation<T> {
        self.adapter.observe(state)
    }
    /// All builtin physical parameters (usually empty for custom).
    pub fn params(&self) -> Vec<(super::model::NeuronParam, T)> {
        self.adapter.params()
    }
    /// Set a builtin physical parameter (None = unsupported).
    pub fn set_param(&mut self, param: super::model::NeuronParam, value: T) -> Option<T> {
        self.adapter.set_param(param, value)
    }
    /// Named custom parameters.
    pub fn custom_params(&self) -> Vec<CustomParameter<T>> {
        self.adapter.custom_params()
    }
    /// Current value of a named custom parameter (None = not supported).
    pub fn get_custom_param(&self, id: &str) -> Option<T> {
        self.adapter.get_custom_param(id)
    }
    /// Set a named custom parameter; returns the actual value (None = unsupported).
    pub fn set_custom_param(&mut self, id: &str, value: T) -> Option<T> {
        self.adapter.set_custom_param(id, value)
    }
    /// Validate the model parameters.
    pub fn validate(&self, limits: &ParamLimits<T>) -> StructuralResult<()> {
        self.adapter.validate(limits)
    }
    /// State is finite?
    pub fn state_is_finite(&self, state: &dyn CustomStateEntry<T>) -> bool {
        self.adapter.state_is_finite(state)
    }
    /// Set membrane on the erased state.
    pub fn set_state_mem(&self, state: &mut dyn CustomStateEntry<T>, value: T) {
        self.adapter.set_mem(state, value);
    }
    /// Encode model configuration bytes.
    pub fn encode_config(&self) -> StructuralResult<Vec<u8>> {
        self.adapter.encode_config()
    }
    /// Encode state bytes (requires the codec).
    pub fn encode_state(&self, state: &dyn CustomStateEntry<T>) -> StructuralResult<Vec<u8>> {
        self.adapter.encode_state(state)
    }
    /// Decode state bytes into an entry.
    pub fn decode_state(&self, bytes: &[u8]) -> StructuralResult<Box<dyn CustomStateEntry<T>>> {
        self.adapter.decode_state(bytes)
    }
    /// Migrate a state from a custom source entry (user hook).
    pub fn migrate_state(
        &self,
        source: &dyn CustomStateEntry<T>,
        policy: StateMigration,
    ) -> StructuralResult<Box<dyn CustomStateEntry<T>>> {
        self.adapter.migrate_state(source, policy)
    }
    /// Crossover two instances.
    pub fn crossover(
        &self,
        other: &CustomNeuronInstance<T>,
        same_id: bool,
    ) -> StructuralResult<Option<Self>> {
        self.adapter.crossover(other, same_id)
    }
    /// Mutate the model in place.
    pub fn mutate(&mut self, rng: &mut Rng64) -> bool {
        self.adapter.mutate(rng)
    }
    /// Config fingerprint for equality checks (None when unsupported).
    pub fn config_fingerprint(&self) -> Option<Vec<u8>> {
        self.adapter.config_fingerprint()
    }
}

impl<T: Scalar> Clone for CustomNeuronInstance<T> {
    fn clone(&self) -> Self {
        Self {
            adapter: self.adapter.clone_adapter(),
        }
    }
}

impl<T: Scalar> PartialEq for CustomNeuronInstance<T> {
    fn eq(&self, other: &Self) -> bool {
        self.adapter.model_id() == other.adapter.model_id()
            && self.adapter.schema_version() == other.adapter.schema_version()
            && match (
                self.adapter.config_fingerprint(),
                other.adapter.config_fingerprint(),
            ) {
                (Some(a), Some(b)) => a == b,
                (None, None) => true,
                _ => false,
            }
    }
}

impl<T: Scalar> CustomNeuronInstance<T> {
    /// Downcast the stored model to the concrete type `M` (clone).
    /// Used by [`CustomCodec`] implementations.
    pub fn downcast_model<M: 'static + Clone>(&self) -> Option<M>
where {
        self.adapter
            .as_any()
            .downcast_ref::<CustomAdapterImpl<M>>()
            .map(|a| a.model.clone())
    }
}

// ---------------------------------------------------------------------
// Registry
// ---------------------------------------------------------------------

/// Entry of the registry: codec + id + version.
pub struct RegisteredCustomModel<T: Scalar = f32> {
    /// Stable model id.
    pub model_id: String,
    /// Schema version.
    pub schema_version: u32,
    codec: Box<dyn CustomCodec<T>>,
    prototype: CustomNeuronInstance<T>,
}

impl<T: Scalar> RegisteredCustomModel<T> {
    /// Codec of the registered model.
    pub fn codec(&self) -> &dyn CustomCodec<T> {
        &*self.codec
    }
    /// Fresh instance of the registered model.
    pub fn prototype(&self) -> &CustomNeuronInstance<T> {
        &self.prototype
    }
}

/// Registry for decoding custom models on load (no global mutable state).
#[derive(Default)]
pub struct CustomNeuronRegistry<T: Scalar = f32> {
    models: BTreeMap<String, RegisteredCustomModel<T>>,
}

impl<T: Scalar> CustomNeuronRegistry<T> {
    /// Empty registry.
    pub fn new() -> Self {
        Self {
            models: BTreeMap::new(),
        }
    }

    /// Register a concrete model `M: DynamicNeuronModel<T>` (requires `M: Default`).
    ///
    /// Returns [`StructuralError::param`] on duplicate ids or unsupported codecs.
    pub fn register<M: DynamicNeuronModel<T> + Default>(&mut self) -> StructuralResult<()> {
        let instance = CustomNeuronInstance::new(M::default());
        self.register_instance(instance)
    }

    /// Register an already constructed model instance (`model_id` + codec from the model).
    pub fn register_instance(&mut self, instance: CustomNeuronInstance<T>) -> StructuralResult<()> {
        let id = instance.model_id().to_string();
        if self.models.contains_key(&id) {
            return Err(StructuralError::param(
                "registry",
                format!("duplicate custom model id \"{id}\""),
            ));
        }
        let codec = instance.adapter.codec().ok_or_else(|| {
            StructuralError::CustomSerialization(
                id.clone(),
                "registration requires a codec (`DynamicNeuronModel::codec`)".into(),
            )
        })?;
        let version = instance.schema_version();
        self.models.insert(
            id.clone(),
            RegisteredCustomModel {
                model_id: id,
                schema_version: version,
                codec,
                prototype: instance,
            },
        );
        Ok(())
    }

    /// Find a registered model entry.
    pub fn get(&self, id: &str) -> Option<&RegisteredCustomModel<T>> {
        self.models.get(id)
    }

    #[allow(clippy::type_complexity)]
    /// Decode a custom model + optionally a state entry from bytes,
    /// checking the schema version. Returns `(instance, state_entry)`.
    pub fn create(
        &self,
        id: &str,
        version: u32,
        model_bytes: &[u8],
        state_bytes: &[u8],
    ) -> StructuralResult<(
        CustomNeuronInstance<T>,
        Option<Box<dyn CustomStateEntry<T>>>,
    )> {
        let entry = self
            .models
            .get(id)
            .ok_or_else(|| StructuralError::UnknownCustomModel(id.to_string()))?;
        if entry.schema_version != version {
            return Err(StructuralError::UnsupportedSchemaVersion(
                id.to_string(),
                version,
                entry.schema_version,
            ));
        }
        let instance = entry.codec.decode_model(model_bytes, &[])?;
        let state = if state_bytes.is_empty() {
            None
        } else {
            Some(instance.adapter.decode_state(state_bytes)?)
        };
        Ok((instance, state))
    }
}

// ---------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    /// FitzHugh–Nagumo custom model (v, w) for tests and examples.
    #[derive(Debug, Clone, PartialEq)]
    pub struct FitzHughNagumo {
        pub v: f64,
        pub w: f64,
    }
    impl DynamicNeuronModel<f64> for FitzHughNagumo {
        type State = FitzHughNagumoState;
        fn model_id(&self) -> &'static str {
            "test/fhn"
        }
        fn init_state(&self) -> Self::State {
            FitzHughNagumoState { v: -0.8, w: -0.5 }
        }
        fn step(
            &self,
            input: f64,
            state: &mut Self::State,
            _ctx: &NeuronStepContext<f64>,
        ) -> StructuralResult<bool> {
            state.v += 0.01 * (state.v - state.v.powi(3) / 3.0 - state.w + input);
            state.w += 0.08 * (state.v + 0.7 - 0.2 * state.w);
            Ok(state.v >= 1.0 && {
                state.v = -1.2;
                state.w += 0.2;
                true
            })
        }
        fn membrane(&self, state: &Self::State) -> Option<f64> {
            Some(state.v)
        }
        fn threshold(&self) -> Option<f64> {
            Some(1.0)
        }
    }
    #[derive(Debug, Clone, PartialEq)]
    pub struct FitzHughNagumoState {
        pub v: f64,
        pub w: f64,
    }
    impl CustomStateEntry<f64> for FitzHughNagumoState {
        fn as_any(&self) -> &dyn Any {
            self
        }
        fn as_any_mut(&mut self) -> &mut dyn Any {
            self
        }
        fn model_id(&self) -> &'static str {
            "test/fhn"
        }
        fn schema_version(&self) -> u32 {
            1
        }
        fn is_finite(&self) -> bool {
            self.v.is_finite() && self.w.is_finite()
        }
        fn membrane(&self) -> Option<f64> {
            Some(self.v)
        }
        fn set_membrane(&mut self, value: f64) {
            self.v = value;
        }
        fn cloned_box(&self) -> Box<dyn CustomStateEntry<f64>> {
            Box::new(self.clone())
        }
        fn state_eq_box(&self, other: &dyn CustomStateEntry<f64>) -> bool {
            other.as_any().downcast_ref::<FitzHughNagumoState>() == Some(self)
        }
    }

    #[test]
    fn fhn_instance_step_fires_and_migrates() {
        let m = FitzHughNagumo { v: -0.8, w: -0.5 };
        let inst = CustomNeuronInstance::new(m.clone());
        let mut entry = inst.init_state();
        let ctx = NeuronStepContext::manual(NeuronId(0), 0, 0);
        for _ in 0..200 {
            inst.step(1.0, &mut *entry, &ctx).unwrap();
        }
        assert!(inst.membrane(entry.as_ref()).unwrap() < 1.0);
        assert!(inst.threshold() == Some(1.0));
    }

    #[test]
    fn custom_state_entry_clone_and_eq() {
        let entry = FitzHughNagumoState { v: 0.1, w: 0.2 };
        let boxed = CustomStateEntryBox::new(entry.clone());
        assert_eq!(boxed.downcast_ref::<FitzHughNagumoState>(), Some(&entry));
    }

    #[test]
    fn registry_requires_codec_and_duplicate_ids() {
        let reg = CustomNeuronRegistry::<f64>::new();
        assert!(reg.get("test/fhn").is_none());
    }
}
