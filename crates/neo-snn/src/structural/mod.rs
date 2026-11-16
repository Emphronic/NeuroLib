//! # Structural plasticity and self-modifying SNNs
//!
//! Synaptic plasticity (`functional::plasticity`, `layers::Stdp*`) changes the
//! **weights** of existing connections. This module changes the **graph**: it
//! creates and disables connections, adds and removes neurons, splits paths, and
//! adapts neuron parameters — automatically, by measurable criteria, with
//! verification, rollback, and a log of reasons.
//!
//! ```text
//!  Topology ── stable NeuronId/SynapseId, invariants, transactions
//!     │
//!  DynamicNetwork ── CPU executor: spike events, LIF/ALIF/Izhikevich/
//!                     AdEx/Hodgkin–Huxley, pair/R-STDP
//!     │
//!  StructuralPlasticityController ── growth · pruning · homeostasis ·
//!     │                              model selection · rollback
//!     │                                         │
//!  EventLog (reasons and results)          checkpoint (topology + state)
//! ```
//!
//! # Why a separate engine instead of Burn layers
//!
//! `neo-snn` layers are fixed-shape tensors; their size cannot be changed at
//! runtime without rebuilding the module, parameters, and optimizer state.
//! Structural plasticity therefore runs on a dynamic graph (CPU, event-driven
//! spike delivery). Burn layers remain unchanged; the dynamic GPU path is
//! **not implemented** (see "Limitations").
//!
//! # Safety boundaries and training state
//!
//! * The topology changes only inside `StructuralPlasticityController::adapt`
//!   (or explicit API calls) **between samples/minibatches** — never in the
//!   middle of an episode or an autograd graph (the engine does not use autograd).
//! * Creation operations: a new neuron/synapse gets empty statistics and
//!   zero eligibility; a clone copies the membrane; a split produces a relay
//!   that reproduces spikes exactly with an extra delay of one step.
//! * Removal: a neuron is removed together with all its synapses, their
//!   eligibility, and statistics; its data in the coactivity accumulator is forgotten.
//! * Optimizer: STDP/R-STDP has none (the rule changes the weight). If an
//!   external optimizer is used on top, its state for removed elements must be
//!   discarded by the calling code; for saving, see
//!   `checkpoint::save_checkpoint(.., extra)`.
//! * Cancellation: every operation is atomic; the controller's plan is rolled
//!   back entirely if quality degrades.
//!
//! # Example
//!
//! ```
//! use neo_snn::structural::*;
//!
//! // Network: 2 inputs → 1 output.
//! let cfg = StructuralConfig::default();
//! let mut topo = Topology::new(cfg.topology, cfg.params);
//! let b = Birth::default();
//! let a = topo.add_neuron(NeuronKind::Input, LifParams::default(), Origin::Initial, b).unwrap();
//! let o = topo.add_neuron(NeuronKind::Output, LifParams::new(0.9, 1.0), Origin::Initial, b).unwrap();
//! topo.add_synapse(a, o, 1.2, 0, Origin::Initial, b).unwrap();
//!
//! let mut net = DynamicNetwork::new(topo, EngineConfig::default()).unwrap();
//! net.step(&[1.0]).unwrap();          // input spike
//! let out = net.step(&[0.0]).unwrap(); // reaches the output
//! assert_eq!(out.output_spikes, vec![true]);
//!
//! let mut ctl = StructuralPlasticityController::new(cfg).unwrap();
//! let report = ctl.adapt(&mut net, Metrics::loss(0.5), None).unwrap();
//! assert!(report.skipped.is_some() || report.applied > 0);
//! ```
//!
//! # Heterogeneous neurons
//!
//! Role, dynamics model, and polarity are independent neuron properties:
//!
//! * [`NeuronRole`](crate::structural::NeuronRole) (`Input`/`Hidden`/`Output`) — position in the network;
//! * [`NeuronModel`](crate::structural::NeuronModel) (`Lif`, `AdaptiveLif`, `Izhikevich`, `AdEx`,
//!   `HodgkinHuxley`) — the dynamics law; each model keeps its own state in
//!   [`NeuronState`](crate::structural::NeuronState), so a mixed network needs neither `dyn Trait` nor
//!   virtual calls per event;
//! * [`NeuronPolarity`](crate::structural::NeuronPolarity) (`Excitatory`/`Inhibitory`/`Modulatory`) — the sign of
//!   the source spike's effect on the receiver. Excitatory polarity adds the
//!   weight as is, inhibitory — with the opposite sign, modulatory — creates no
//!   current but increases the receiver's decaying gain (`modulation`,
//!   `EngineConfig::modulation_decay`).
//!
//! ```ignore
//! let h = topo.add_neuron_with(
//!     NeuronRole::Hidden,
//!     NeuronPolarity::Inhibitory,
//!     AdaptiveLifParams::slow_adaptation(),
//!     Origin::Initial,
//!     Birth::default(),
//! )?;
//! ```
//!
//! # Dale's principle and the E/I network
//!
//! `TopologyLimits::enforce_dale` (default `true`) makes the weight sign an
//! invariant: an excitatory or modulatory source has `w ≥ 0`, an inhibitory
//! one has `w ≤ 0`; the inhibitory neuron's current equals the weight as is. A
//! new or rewired connection takes the source's sign, `set_synapse_weight`,
//! `mutate_synapse`, STDP, and reward learning never move a weight through zero,
//! a polarity change flips the signs of outgoing weights, a clone keeps the
//! polarity, evolution normalizes the genome after every mutation
//! (`Genome::normalize_dale`), and `Topology::validate` rejects violations. A
//! legacy network with free signs is converted to E/I via
//! `Topology::set_enforce_dale(true)`.
//!
//! The invariant is built upon by:
//!
//! * [`InhibitoryPlasticityConfig`] — inhibitory STDP following Vogels et al. (2011),
//!   which also homeostatically regulates the receiver's rate;
//! * [`EiBalanceConfig`] / `DynamicNetwork::balance_inhibition` — local
//!   adjustment of inhibition strength by the input current's `inh/exc` ratio
//!   (the controller calls it on every adaptation cycle);
//! * [`EiStats`] / `Topology::ei_stats` — separate statistics for E/I populations,
//!   weights, rates, and balance.
//!
//! Splitting a synapse with a relay neuron (`split_synapse`, `SplitSynapse` growth, NEAT
//! `add_node`) is allowed under Dale's principle only for excitatory sources:
//! with an inhibitory source the relay would receive a negative current and would not
//! transmit the spike, and the excitatory segment `relay → receiver` would change the
//! sign of the effect (`Topology::can_split_synapse`).
//!
//! # Switching models during training
//!
//! A neuron's model can be changed at a safe boundary (between samples or
//! minibatches) via [`Topology::change_neuron_model`](crate::structural::Topology::change_neuron_model) or
//! [`DynamicNetwork::change_neuron_model`](crate::structural::DynamicNetwork::change_neuron_model). The old state is not lost
//! silently: the [`StateMigration`](crate::structural::StateMigration) policy explicitly defines the transformation.
//!
//! | policy | behavior |
//! |---|---|
//! | `Reset` | initial state of the new model |
//! | `Membrane` | only `mem()` is copied, without unit conversion |
//! | `Compatible` (default) | same model — the whole state; same family (LIF/ALIF or Izhikevich/AdEx/HH) — the membrane; otherwise reset |
//!
//! [`ModelSwitchOutcome`](crate::structural::ModelSwitchOutcome) returns the old/new model, the old/new
//! state, the policy, and the `changed` flag. The operation enters the log and
//! rollback as [`StructuralAction::ChangeNeuronModel`](crate::structural::StructuralAction::ChangeNeuronModel); the numeric `ActionKind`
//! code was appended at the end so that saved codes of earlier actions do not shift.
//!
//! # Automatic model selection
//!
//! [`ModelSelectionConfig`](crate::structural::ModelSelectionConfig) — an optional (disabled by default) mechanism:
//! the controller scores how well the neuron's current rate matches a
//! candidate's specialization and subtracts the model cost.
//! [`propose_model_switches`](crate::structural::propose_model_switches)
//! only proposes [`StructuralAction::ChangeNeuronModel`](crate::structural::StructuralAction::ChangeNeuronModel); the
//! common mechanism with verification and rollback applies them, with reason `Reason::ModelSelection`.
//! On equal score the cheaper model is chosen. Input and output
//! neurons are included via the `include_inputs`/`include_outputs` flags.
//!
//! # Learning neuron types
//!
//! A neuron's type (model and polarity) is inherited and changed
//! evolutionarily: `MutationConfig::model_mutate_prob` + `model_candidates` select a
//! model from the pool, `polarity_mutate_prob` — the polarity, `split_model` sets the
//! model of the neuron created by splitting a synapse (for example, a slow ALIF).
//! `MutationSummary` reports `models_changed`/`polarities_changed`. This is
//! discrete architecture evolution, not differentiable type learning.
//!
//! # Dynamic input neurons
//!
//! [`InputMode::Spike`](crate::structural::InputMode::Spike) (default): `≥ 0.5` means
//! a ready spike, the input model is not involved. [`InputMode::Current`](crate::structural::InputMode::Current) makes
//! the input an ordinary dynamic neuron: the external value is added to
//! the input current, the input's model, state, and parameters determine the
//! spike and can be changed, including model switches. A non-numeric current in
//! this mode is an error. [`DenseNet`](crate::structural::dense::DenseNet) still treats inputs as spike sources.
//!
//! # Checkpoint compatibility
//!
//! [`FORMAT_VERSION`](crate::structural::FORMAT_VERSION) = 5. Version 5 saves Dale's principle,
//! per-neuron E/I statistics, [`InhibitoryPlasticityConfig`], and [`EiBalanceConfig`];
//! version 4 adds the model-selection loss estimate and
//! [`ParamLearningConfig`](crate::structural::ParamLearningConfig); version 3 — [`InputMode`](crate::structural::InputMode)
//! and [`ModelSelectionConfig`](crate::structural::ModelSelectionConfig); version 2 is read with `InputMode::Spike` and
//! model selection disabled; version 1 migrates to LIF/excitatory. Versions 1–4
//! load in legacy mode (free weight signs, `enforce_dale = false`);
//! converting such a network to E/I — [`Checkpoint::enable_dale`](crate::structural::Checkpoint::enable_dale).
//!
//! # Limitations
//!
//! * CPU only; the GPU path for the dynamic graph is not implemented. The dense
//!   executor [`dense`](crate::structural::dense) supports only LIF and excitatory neurons:
//!   it rejects a heterogeneous network; use
//!   [`DynamicNetwork`](crate::structural::DynamicNetwork) for that.
//! * Automatic model selection is a heuristic based on rate and cost. It does not
//!   simulate candidates and does not guarantee an improvement in the task metric, so it is
//!   disabled by default; the decision is verified by the controller's common rollback.
//! * A synapse's contribution is estimated from transmission and the quality signal; exact
//!   single-synapse ablation is not performed (there is a before/after check at the level of
//!   the whole plan and a deferred check against training metrics).
//! * Composite operations (`clone_neuron`, `split_synapse`, `apply_batch`,
//!   `Topology::transaction`) roll back by copying the topology, i.e. they
//!   cost `O(N + S)`: a reference point is about 0.2 s on a network of ~170k synapses
//!   (`examples/bench_structural.rs`). With structural changes every
//!   few hundred steps this is acceptable; for large graphs an undo log is needed
//!   instead of a snapshot (an optimization stage).
//! * Step cost is proportional to the number of **delivered spikes** (on the order of
//!   a microsecond per event: `BTreeMap` access); dense storage for the
//!   fast path is not implemented yet.
//! * `SplitInit::relay_gain` must fit within the weight bounds of the plasticity
//!   rule (`WeightBound`): otherwise STDP will clip the relay's input, it will stop
//!   firing, and the controller will roll back such a split.
//! * If `GrowthConfig::target_loss` is set, growth (of synapses and neurons)
//!   is allowed only while the loss is above that value.

pub mod checkpoint;
pub mod controller;
pub mod dense;
pub mod error;
pub mod evolution;
pub mod fixed;
pub mod growth;
pub mod model;
pub mod model_selection;
pub mod mutation;
pub mod network;
pub mod neuron_ops;
pub mod param_learning;
pub mod policy;
pub mod pruning;
pub mod rng;
pub mod stats;
pub mod synapse_ops;
pub mod topology;

#[cfg(test)]
mod tests;

pub use checkpoint::{
    Checkpoint, FORMAT_VERSION, FORMAT_VERSION_F64, load_checkpoint, load_checkpoint_as,
    load_from_file, load_from_file_as, save_checkpoint, save_to_file,
};
pub use controller::{
    AdaptReport, BestSnapshot, Metrics, SkipReason, StructuralPlasticityController,
};
pub use error::{StructuralError, StructuralResult};
pub use growth::{grow_neurons, grow_synapses, propose_grow_neuron, propose_grow_synapses};
pub use model::{
    AdExParams, AdExState, AdaptiveLifParams, AdaptiveLifState, HodgkinHuxleyParams,
    HodgkinHuxleyState, IzhikevichParams, IzhikevichState, LifParams, LifState, ModelSwitchOutcome,
    NeuronKind, NeuronModel, NeuronParam, NeuronPolarity, NeuronRole, NeuronState, StateMigration,
};
pub use model_selection::{
    ModelCandidate, ModelSelectionConfig, propose_model_switches, propose_model_switches_evaluated,
};
pub use mutation::{
    ActionEffect, ActionKind, EventLog, Outcome, OutcomeCounts, Reason, StructuralAction,
    StructuralEvent, apply_action, apply_batch,
};
pub use network::{
    DynamicNetwork, EiBalanceConfig, EiBalanceReport, EngineConfig, InhibitoryPlasticityConfig,
    InputMode, PlasticityMode, StdpConfig, StepOutput,
};
pub use neuron_ops::{MutationOutcome, NeuronMutation, RemovedNeuron, SplitInit, SplitResult};
pub use param_learning::{ParamLearningConfig, propose_param_updates};
pub use policy::{
    GrowthConfig, GrowthStrategy, HomeostasisConfig, NeuronGrowthConfig, PruningConfig,
    RollbackConfig, ScheduleConfig, StructuralConfig, SynapseGrowthConfig,
};
pub use pruning::{
    SynapseVerdict, mark_evaluated, propose_prune_neurons, propose_prune_synapses, prune_neurons,
    prune_synapses, synapse_verdicts,
};
pub use rng::Rng64;
pub use stats::{
    CoactivityTracker, EiStats, LinearCost, StatsScalar, StructuralStats, StructureCost,
};
pub use synapse_ops::{DegreePolicy, SynapseMutation};
pub use topology::{
    Birth, Neuron, NeuronId, NeuronStats, Origin, ParamLimits, StructMeta, Synapse, SynapseId,
    SynapseState, SynapseStats, Topology, TopologyLimits,
};
