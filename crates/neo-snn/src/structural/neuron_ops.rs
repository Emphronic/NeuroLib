//! Neuron operations: creation, removal, cloning, synapse splitting, and
//! parameter changes.
//!
//! # What happens to state
//!
//! | operation | new state | on removal |
//! |----------|-----------------|--------------|
//! | `add_neuron` | membrane 0, empty statistics | — |
//! | `remove_neuron` | — | the neuron and **all** its synapses are removed together with their eligibility/statistics; returned to the caller for undo |
//! | `clone_neuron` | membrane copied, statistics empty | — |
//! | `split_synapse` | membrane 0 | the original synapse is removed |
//!
//! Structural operations do not touch the optimizer or autograd: the topology
//! changes only at a safe boundary (between samples/minibatches), see the
//! `structural` module documentation.

use crate::scalar::Scalar;
use std::collections::BTreeSet;

use crate::state::ResetMode;

use super::custom::CustomNeuronInstance;
use super::error::{StructuralError, StructuralResult};
use super::model::{
    LifParams, ModelSwitchOutcome, NeuronModel, NeuronParam, NeuronPolarity, NeuronRole,
    StateMigration,
};
use super::rng::Rng64;
use super::topology::{Birth, Neuron, NeuronId, Origin, Synapse, SynapseId, Topology};

/// Change to a neuron's parameters. `None` means "do not change".
///
/// The special fields (`beta`, `threshold`, `reset`, `rho`, `adapt_increment`)
/// are kept for compatibility; `params` is the general path for all physical
/// parameters of Izhikevich, AdEx, and Hodgkin–Huxley (see
/// [`NeuronParam`](super::NeuronParam)). The special fields are applied first,
/// then `params` in order, so a general parameter can override a
/// special one.
///
/// Fields that belong to another model (for example, `rho` on LIF) return
/// [`StructuralError::UnsupportedParameter`].
#[derive(Debug, Clone, Default, PartialEq)]
pub struct NeuronMutation<T = f32> {
    /// New β (LIF/ALIF).
    pub beta: Option<T>,
    /// New threshold.
    pub threshold: Option<T>,
    /// New reset mode (LIF/ALIF).
    pub reset: Option<ResetMode>,
    /// New adaptation decay ρ (ALIF).
    pub rho: Option<T>,
    /// New adaptation increment per spike (ALIF).
    pub adapt_increment: Option<T>,
    /// Arbitrary physical parameters in application order.
    pub params: Vec<(NeuronParam, T)>,
    /// New physiological polarity.
    pub polarity: Option<NeuronPolarity>,
}

impl<T: Scalar> NeuronMutation<T> {
    fn has_model_params(&self) -> bool {
        self.beta.is_some()
            || self.threshold.is_some()
            || self.reset.is_some()
            || self.rho.is_some()
            || self.adapt_increment.is_some()
            || !self.params.is_empty()
    }

    fn is_empty(&self) -> bool {
        !self.has_model_params() && self.polarity.is_none()
    }
}

/// Result of `mutate_neuron`.
#[derive(Debug, Clone, PartialEq)]
pub struct MutationOutcome<T: Scalar = f32> {
    /// Model before the change.
    pub old: NeuronModel<T>,
    /// Model after the change (taking limits into account).
    pub new: NeuronModel<T>,
    /// Whether the value was clamped by `ParamLimits`.
    pub clamped: bool,
}

/// Initialization of the relay neuron when splitting a synapse.
///
/// A relay with `beta = 0`, threshold `θ`, and input weight `gain ≥ θ` spikes on
/// every incoming spike and resets to zero, i.e. it transmits spikes
/// as is, with an extra delay of one step.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SplitInit<T = f32> {
    /// β of the relay.
    pub relay_beta: T,
    /// Threshold of the relay.
    pub relay_threshold: T,
    /// Weight `src → relay` (at least the threshold and within the weight bounds of the
    /// plasticity rule, otherwise STDP will clip it and the relay will stop firing).
    pub relay_gain: T,
}

impl<T: Scalar> Default for SplitInit<T> {
    fn default() -> Self {
        Self {
            relay_beta: T::ZERO,
            relay_threshold: T::c(0.5),
            relay_gain: T::ONE,
        }
    }
}

/// Result of `split_synapse`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SplitResult {
    /// New neuron.
    pub neuron: NeuronId,
    /// Synapse `src → neuron`.
    pub incoming: SynapseId,
    /// Synapse `neuron → dst`.
    pub outgoing: SynapseId,
}

/// A removed neuron with all its synapses.
#[derive(Debug, Clone, PartialEq)]
pub struct RemovedNeuron<T: Scalar = f32> {
    /// Neuron.
    pub neuron: Neuron<T>,
    /// Removed synapses.
    pub synapses: Vec<Synapse<T>>,
}

impl<T: Scalar> Topology<T> {
    /// Add a neuron. The parameters are validated.
    ///
    /// `params` can be any [`NeuronModel`] or parameters of a concrete
    /// model (`LifParams`, `AdaptiveLifParams`, …) thanks to `Into`.
    /// The default polarity is excitatory; the explicit form is
    /// [`add_neuron_with`](Self::add_neuron_with).
    pub fn add_neuron<M: Into<NeuronModel<T>>>(
        &mut self,
        role: NeuronRole,
        params: M,
        origin: Origin,
        birth: Birth,
    ) -> StructuralResult<NeuronId> {
        self.add_neuron_with(role, NeuronPolarity::Excitatory, params, origin, birth)
    }

    /// Add a neuron with an explicit polarity and dynamics model.
    pub fn add_neuron_with<M: Into<NeuronModel<T>>>(
        &mut self,
        role: NeuronRole,
        polarity: NeuronPolarity,
        model: M,
        origin: Origin,
        birth: Birth,
    ) -> StructuralResult<NeuronId> {
        if self.neurons.len() >= self.limits.max_neurons {
            return Err(StructuralError::LimitExceeded {
                what: "max_neurons",
                limit: self.limits.max_neurons,
            });
        }
        let model = model.into();
        model.validate(&self.param_limits)?;
        Ok(self.raw_insert_neuron(role, polarity, model, birth, origin))
    }

    /// Add a custom (user-defined) neuron.
    ///
    /// The model is type-erased internally; `Topology` and `DynamicNetwork`
    /// accept heterogeneous networks (see [`structural::custom`](super::custom)).
    pub fn add_custom_neuron<M: super::custom::DynamicNeuronModel<T>>(
        &mut self,
        role: NeuronRole,
        polarity: NeuronPolarity,
        model: M,
        origin: Origin,
        birth: Birth,
    ) -> StructuralResult<NeuronId> {
        self.add_neuron_with(
            role,
            polarity,
            CustomNeuronInstance::new(model),
            origin,
            birth,
        )
    }

    /// Remove a hidden neuron together with all its incoming and outgoing
    /// synapses. Inputs and outputs cannot be removed.
    pub fn remove_neuron(&mut self, id: NeuronId) -> StructuralResult<RemovedNeuron<T>> {
        let kind = self
            .neurons
            .get(&id)
            .ok_or(StructuralError::NoSuchNeuron(id))?
            .role;
        if kind != NeuronRole::Hidden {
            return Err(StructuralError::WrongKind {
                op: "remove_neuron",
                id,
                kind: kind.as_str(),
            });
        }
        let ids: BTreeSet<SynapseId> = self
            .out_synapses(id)
            .iter()
            .chain(self.in_synapses(id).iter())
            .copied()
            .collect();
        let mut synapses = Vec::with_capacity(ids.len());
        for sid in ids {
            if let Some(s) = self.raw_remove_synapse(sid) {
                synapses.push(s);
            }
        }
        self.out_adj.remove(&id);
        self.in_adj.remove(&id);
        let neuron = self
            .neurons
            .remove(&id)
            .ok_or(StructuralError::NoSuchNeuron(id))?;
        self.touch();
        Ok(RemovedNeuron { neuron, synapses })
    }

    /// Clone a hidden neuron "preserving behavior" (in the spirit of Net2Net).
    ///
    /// * incoming synapses are copied with weight `w·(1 + jitter·u)`, `u ∈ (-1, 1)`;
    /// * every outgoing weight is split in half between the original and the clone, so
    ///   that the total effect on the receiver does not change when the two
    ///   neurons behave identically;
    /// * the loop `n → n` is copied as the clone's loop;
    /// * the membrane is copied; disabled synapses are not copied.
    ///
    /// With `jitter > 0` the neurons diverge and can specialize.
    pub fn clone_neuron(
        &mut self,
        id: NeuronId,
        jitter: T,
        rng: &mut Rng64,
        birth: Birth,
    ) -> StructuralResult<NeuronId> {
        let n = self
            .neurons
            .get(&id)
            .ok_or(StructuralError::NoSuchNeuron(id))?;
        if n.role != NeuronRole::Hidden {
            return Err(StructuralError::WrongKind {
                op: "clone_neuron",
                id,
                kind: n.role.as_str(),
            });
        }
        if !jitter.is_finite() || jitter < T::ZERO {
            return Err(StructuralError::param("jitter", "must be finite and >= 0"));
        }
        let (model, polarity, state, modulation) =
            (n.model.clone(), n.polarity, n.state.clone(), n.modulation);
        let active = |t: &Topology<T>, list: &[SynapseId]| -> Vec<Synapse<T>> {
            list.iter()
                .filter_map(|s| t.synapses.get(s))
                .filter(|s| s.is_active())
                .cloned()
                .collect()
        };
        let ins = active(self, self.in_synapses(id));
        let outs = active(self, self.out_synapses(id));
        self.transaction(|t| {
            let new = t.add_neuron_with(
                NeuronRole::Hidden,
                polarity,
                model,
                Origin::Clone(id),
                birth,
            )?;
            if let Some(nn) = t.neurons.get_mut(&new) {
                nn.state = state;
                nn.modulation = modulation;
            }
            for s in ins.iter().filter(|s| s.src != id) {
                let w = s.weight * (T::ONE + jitter * rng.symmetric_as::<T>());
                t.add_synapse(s.src, new, w, s.delay, Origin::Clone(id), birth)?;
            }
            for s in outs.iter().filter(|s| s.dst != id) {
                let half = s.weight * T::c(0.5);
                t.set_synapse_weight(s.id, half)?;
                t.add_synapse(new, s.dst, half, s.delay, Origin::Clone(id), birth)?;
            }
            for s in ins.iter().filter(|s| s.src == id) {
                t.add_synapse(new, new, s.weight, s.delay, Origin::Clone(id), birth)?;
            }
            Ok(new)
        })
    }

    /// Split an active synapse `src → dst` through a new relay neuron:
    /// `src → relay → dst`. The original synapse is removed, the weight is kept on
    /// the `relay → dst` segment (and the sign too), the original synapse's delay
    /// moves to the `src → relay` segment. Behavior is preserved exactly, not counting
    /// the extra delay of one step at the relay neuron.
    pub fn split_synapse(
        &mut self,
        id: SynapseId,
        init: SplitInit<T>,
        birth: Birth,
    ) -> StructuralResult<SplitResult> {
        let orig = self
            .synapses
            .get(&id)
            .ok_or(StructuralError::NoSuchSynapse(id))?
            .clone();
        if !orig.is_active() {
            return Err(StructuralError::param(
                "split_synapse",
                "a disabled synapse cannot be split",
            ));
        }
        if self.limits.enforce_dale && self.polarity_of(orig.src) != NeuronPolarity::Excitatory {
            return Err(StructuralError::param(
                "split_synapse",
                format!(
                    "source {} is {}: a relay neuron would not preserve the meaning of the connection under Dale's principle \
                     (a negative current would not activate the relay, while the positive segment \
                     relay → receiver would become excitatory)",
                    orig.src,
                    self.polarity_of(orig.src).as_str()
                ),
            ));
        }
        if !(init.relay_gain.is_finite() && init.relay_gain >= init.relay_threshold) {
            return Err(StructuralError::param(
                "relay_gain",
                "must be finite and not less than relay_threshold",
            ));
        }
        let params = LifParams {
            beta: init.relay_beta,
            threshold: init.relay_threshold,
            reset: ResetMode::Zero,
        };
        self.transaction(|t| {
            t.remove_synapse(id)?;
            let n = t.add_neuron(NeuronRole::Hidden, params, Origin::Split(id), birth)?;
            let incoming = t.add_synapse(
                orig.src,
                n,
                init.relay_gain,
                orig.delay,
                Origin::Split(id),
                birth,
            )?;
            let outgoing = t.add_synapse(n, orig.dst, orig.weight, 0, Origin::Split(id), birth)?;
            Ok(SplitResult {
                neuron: n,
                incoming,
                outgoing,
            })
        })
    }

    /// Change a neuron's parameters. Values are clamped by `ParamLimits`.
    /// Input neurons are ordinary dynamic neurons, so their parameters
    /// can be changed too (this matters for `InputMode::Current`).
    pub fn mutate_neuron(
        &mut self,
        id: NeuronId,
        m: NeuronMutation<T>,
    ) -> StructuralResult<MutationOutcome<T>> {
        let lim = self.param_limits;
        let n = self
            .neurons
            .get_mut(&id)
            .ok_or(StructuralError::NoSuchNeuron(id))?;
        let old = n.model.clone();
        if m.is_empty() {
            return Ok(MutationOutcome {
                old: old.clone(),
                new: old,
                clamped: false,
            });
        }
        let mut new = old.clone();
        let mut clamped = false;
        if let Some(b) = m.beta {
            let c = new.set_beta(b, &lim)?;
            clamped |= c != b;
        }
        if let Some(th) = m.threshold {
            if !th.is_finite() {
                return Err(StructuralError::param("threshold", "non-finite value"));
            }
            let c = new.set_threshold(th, &lim);
            clamped |= c != th;
        }
        if let Some(r) = m.reset {
            new.set_reset(r)?;
        }
        if let Some(rho) = m.rho {
            let c = new.set_rho(rho)?;
            clamped |= c != rho;
        }
        if let Some(a) = m.adapt_increment {
            let c = new.set_adapt_increment(a)?;
            clamped |= c != a;
        }
        for (param, value) in &m.params {
            let c = new.set_param(*param, *value, &lim)?;
            clamped |= c != *value;
        }
        let mut polarity_changed = false;
        if let Some(p) = m.polarity {
            polarity_changed = n.polarity != p;
            n.polarity = p;
        }
        n.model = new.clone();
        if polarity_changed {
            // Dale's principle: outgoing weights change sign together with the polarity.
            self.resign_outgoing(id);
            self.touch();
        }
        Ok(MutationOutcome { old, new, clamped })
    }

    /// Change a neuron's model during training with an explicit state migration.
    ///
    /// The model is validated against `ParamLimits` for any role, including input
    /// neurons. If the model did not change, the state is left untouched and
    /// `changed == false`. Otherwise the state is transformed by the chosen
    /// [`StateMigration`] policy; a non-finite result is replaced with the initial state
    /// of the new model.
    pub fn change_neuron_model(
        &mut self,
        id: NeuronId,
        model: impl Into<NeuronModel<T>>,
        migration: StateMigration,
    ) -> StructuralResult<ModelSwitchOutcome<T>> {
        let model = model.into();
        model.validate(&self.param_limits)?;
        let outcome = {
            let n = self
                .neurons
                .get_mut(&id)
                .ok_or(StructuralError::NoSuchNeuron(id))?;
            let old_model = n.model.clone();
            let old_state = n.state.clone();
            if old_model.is_same_dynamics(&model) {
                return Ok(ModelSwitchOutcome {
                    old_model,
                    new_model: model.clone(),
                    old_state: old_state.clone(),
                    new_state: old_state,
                    migration,
                    changed: false,
                });
            }
            let source = if old_state.matches(&old_model) {
                old_state.clone()
            } else {
                old_model.init_state()
            };
            let mut new_state = source.migrate_between(&old_model, &model, migration);
            if !new_state.is_finite() || !new_state.matches(&model) {
                new_state = model.init_state();
            }
            n.model = model.clone();
            n.state = new_state.clone();
            ModelSwitchOutcome {
                old_model,
                new_model: model,
                old_state,
                new_state,
                migration,
                changed: true,
            }
        };
        self.touch();
        Ok(outcome)
    }
}
