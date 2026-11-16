//! Dynamic CPU-executor of a network with a variable topology.
//!
//! This is the "correct minimal engine" from Stage 3: Event Delivery
//! spikes at active synapses, LIF-neuron dynamics, local rules
//! plasticity and statistics for structural solutions. It doesn't replace
//! Burn-layers of fixed shape (`layers::*`, `neuron::*`): structural
//! plasticity cannot be applied to fixed tensors without reassembly
//! size, so the graph changes here, but the layers remain as they are.
//!
//! # Simulation step
//!
//! 1. **Delivery.** For each spike from history (age `a` = how many steps
//!    back) current is added along its outgoing active synapses with `delay == a`
//!    to the receiver. The cost is proportional to the number of spikes, not the size of the network.
//! 2. **Dynamics.** In [`InputMode::Spike`] input
//!    spikes based on an external signal (`≥ 0.5`); in
//!    [`InputMode::Current`] external value —
//!    this is the current, and the spike determines the model of the input neuron. The rest:
//!    `U ← β·U + I`, spike at `U ≥ θ`, reset at `ResetMode`.
//! 3. **Plasticity** (if enabled): pair-STDP using the same formulas as
//!    `functional::plasticity::pair_update`, but for events (see below).
//! 4. Spike history, statistics, joint activity accumulator.
//!
//! # Connect with `functional::plasticity`
//!
//! The formula and semantics are the same: `Δw = a₊·x_post·pre − a₋·x_pre·post`, traces
//! **before** updates are taken at this step, weight limit is the same
//! [`WeightBound`] (`Hard`/`Soft`/`None`), reward-modulated option - the same
//! three-factor: eligibility accumulates STDP, weight changes when
//! [`DynamicNetwork::apply_reward`]. There is only one difference: the calculations take place there
//! dense tensors `[out, in]`, here - by events for sparse
//! the graph has a variable shape, so tensor functions cannot be called; code
//! the rules do not duplicate, but repeat the scalar shape of the same formula.
//! A `pre` spike is considered to be a spike that **arrived** at the synapse at this step (with
//! taking into account the delay), `pre`-trace - trace of the source neuron.
//!
//! # SafeBorders
//!
//! Topology (`topology_mut`, `replace_topology`, controller actions)
//! change between samples or minibatch, not in the middle of an episode. External
//! The engine does not use autograd graphs.

use crate::scalar::Scalar;
use std::collections::{BTreeMap, VecDeque};

use crate::functional::plasticity::WeightBound;

use super::error::{StructuralError, StructuralResult};
use super::model::{ModelSwitchOutcome, NeuronModel, NeuronPolarity, StateMigration};
use super::stats::{CoactivityTracker, StructuralStats};
use super::topology::{NeuronId, NeuronRole, SynapseId, Topology};

/// Synaptic plasticity mode of the engine.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PlasticityMode {
    /// The weights do not change.
    Off,
    /// pair-STDP is applied to the weights immediately.
    Immediate,
    /// Reward-modulated: STDP is accumulated in the eligibility trace, the weight changes when
    /// `apply_reward`.
    RewardModulated,
}

/// Parameters pair-STDP (names and meaning are the same as `functional::plasticity`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StdpConfig<T = f32> {
    /// Mode.
    pub mode: PlasticityMode,
    /// Potentiation amplitude `A+`.
    pub a_plus: T,
    /// Depression amplitude `A−`.
    pub a_minus: T,
    /// Decay of trace `exp(-1/τ)` per step.
    pub trace_decay: T,
    /// Limiting weights.
    pub bound: WeightBound<T>,
    /// Decay of eligibility trace per step (for `RewardModulated`).
    pub elig_decay: T,
    /// Reward learning rate.
    pub reward_lr: T,
}

impl<T: Scalar> Default for StdpConfig<T> {
    fn default() -> Self {
        Self {
            mode: PlasticityMode::Off,
            a_plus: T::c(0.01),
            a_minus: T::c(0.012),
            trace_decay: (-T::ONE / T::c(20.0)).exp(),
            bound: WeightBound::Hard {
                w_min: -T::ONE,
                w_max: T::ONE,
            },
            elig_decay: (-T::ONE / T::c(50.0)).exp(),
            reward_lr: T::ONE,
        }
    }
}

/// Inhibitory plasticity (Vogels et al., 2011): STDP inhibitory synapses,
/// which simultaneously homeostatically maintains the receiver frequency at
/// `target_rate`.
///
/// Works only with Dale principle ([`Topology::dale`]) and enabled
/// plasticity of the engine, regardless of the `StdpConfig::mode` mode. For
/// inhibitory synapse `m = |w|`: at source spike
/// `m += η·(trace_post − α)`, when the receiver spikes `m += η·trace_pre`, where
/// `α = 2·target_rate / (1 − trace_decay)`. A receiver that spikes more often
/// targets, receives stronger inhibition; quieter than the target - weaker.
/// Excitatory synapses are not changed by this rule, but ordinary STDP are inhibitory
/// does not affect synapses.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct InhibitoryPlasticityConfig<T = f32> {
    /// Whether the rule is enabled.
    pub enabled: bool,
    /// Learning rate `η`.
    pub eta: T,
    /// Receiver target frequency (spikes per step).
    pub target_rate: T,
}

impl<T: Scalar> Default for InhibitoryPlasticityConfig<T> {
    fn default() -> Self {
        Self {
            enabled: false,
            eta: T::c(0.001),
            target_rate: T::c(0.02),
        }
    }
}

/// Local adjustment of inhibitory strength based on the balance of excitation and inhibition.
///
/// For each neuron, `inh_drive / exc_drive` (sliding
/// average input current, [`NeuronStats::exc_drive_ema`](super::NeuronStats))
/// with `target_ratio`, and all incoming inhibitory weights of the neuron are multiplied by
/// general coefficient: little inhibition - increases, a lot - weakens.
/// [`DynamicNetwork::balance_inhibition`] is called (the controller calls it
/// on each adaptation cycle, if `enabled`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct EiBalanceConfig<T = f32> {
    /// Is adjustment enabled in the controller?
    pub enabled: bool,
    /// Target ratio of inhibition to excitation.
    pub target_ratio: T,
    /// Fraction of error processed per call.
    pub rate: T,
    /// Maximum relative weight change per challenge.
    pub max_step: T,
    /// The minimum excitatory current at which a neuron is regulated.
    pub min_drive: T,
    /// Minimum steps for monitoring a neuron.
    pub min_observed_steps: u64,
}

impl<T: Scalar> Default for EiBalanceConfig<T> {
    fn default() -> Self {
        Self {
            enabled: false,
            target_ratio: T::ONE,
            rate: T::c(0.1),
            max_step: T::c(0.1),
            min_drive: T::c(1e-3),
            min_observed_steps: 200,
        }
    }
}

/// Result [`DynamicNetwork::balance_inhibition`].
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct EiBalanceReport<T = f32> {
    /// How many neurons received a new inhibition coefficient.
    pub adjusted: usize,
    /// Average `inh/exc` for the considered neurons before adjustment.
    pub mean_ratio: T,
}

/// How an external signal affects input neurons.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum InputMode {
    /// Previous mode: value `≥ 0.5` - ready spike, model and state
    /// the input neuron does not participate in the dynamics.
    #[default]
    Spike,
    /// Input neuron - normal dynamic neuron: external value
    /// is added to the incoming current, and the spike is determined by its model.
    Current,
}

impl InputMode {
    /// Short name.
    pub fn as_str(self) -> &'static str {
        match self {
            InputMode::Spike => "spike",
            InputMode::Current => "current",
        }
    }

    pub(crate) fn code(self) -> u8 {
        match self {
            InputMode::Spike => 0,
            InputMode::Current => 1,
        }
    }

    pub(crate) fn from_code(c: u8) -> Option<Self> {
        match c {
            0 => Some(InputMode::Spike),
            1 => Some(InputMode::Current),
            _ => None,
        }
    }
}

/// Engine parameters.
#[derive(Debug, Clone, PartialEq)]
pub struct EngineConfig<T = f32> {
    /// Rule of plasticity.
    pub stdp: StdpConfig<T>,
    /// Coefficient of the moving average spike frequency of a neuron.
    pub rate_alpha: T,
    /// Moving average synapse contribution coefficient.
    pub credit_alpha: T,
    /// Decay of the spike transmission trace by a synapse per step.
    pub activity_trace_decay: T,
    /// Decay of accumulated modulation per step (for modulating sources).
    pub modulation_decay: T,
    /// Input neuron mode.
    pub input_mode: InputMode,
    /// Inhibitory plasticity.
    pub inhibitory: InhibitoryPlasticityConfig<T>,
    /// Adjusting the balance of excitation and inhibition.
    pub balance: EiBalanceConfig<T>,
}

impl<T: Scalar> Default for EngineConfig<T> {
    fn default() -> Self {
        Self {
            stdp: StdpConfig::default(),
            rate_alpha: T::c(0.01),
            credit_alpha: T::c(0.1),
            activity_trace_decay: T::c(0.9),
            modulation_decay: T::c(0.9),
            input_mode: InputMode::default(),
            inhibitory: InhibitoryPlasticityConfig::default(),
            balance: EiBalanceConfig::default(),
        }
    }
}

impl<T: Scalar> EngineConfig<T> {
    fn validate(&self) -> StructuralResult<()> {
        let unit = |v: T| v.is_finite() && v > T::ZERO && v <= T::ONE;
        let s = &self.stdp;
        let ok = unit(self.rate_alpha)
            && unit(self.credit_alpha)
            && unit(self.activity_trace_decay)
            && unit(self.modulation_decay)
            && unit(s.trace_decay)
            && unit(s.elig_decay)
            && s.a_plus.is_finite()
            && s.a_minus.is_finite()
            && s.reward_lr.is_finite()
            && self.inhibitory.eta.is_finite()
            && self.inhibitory.eta >= T::ZERO
            && self.inhibitory.target_rate.is_finite()
            && self.inhibitory.target_rate >= T::ZERO
            && self.balance.target_ratio.is_finite()
            && self.balance.target_ratio >= T::ZERO
            && self.balance.rate.is_finite()
            && self.balance.rate >= T::ZERO
            && self.balance.max_step.is_finite()
            && self.balance.max_step >= T::ZERO
            && self.balance.max_step < T::ONE
            && self.balance.min_drive.is_finite()
            && self.balance.min_drive >= T::ZERO;
        if ok {
            Ok(())
        } else {
            Err(StructuralError::InvalidConfig(
                "EngineConfig: coefficients must be finite, attenuations must be in (0, 1]".into(),
            ))
        }
    }
}

/// The result of one step.
#[derive(Debug, Clone, PartialEq)]
pub struct StepOutput {
    /// Spikes of output neurons (in ascending order of id).
    pub output_spikes: Vec<bool>,
    /// Total spikes per step (including inputs).
    pub total_spikes: usize,
}

#[derive(Debug, Clone, Default)]
struct Layout {
    version: u64,
    order: Vec<NeuronId>,
    index: BTreeMap<NeuronId, usize>,
    input_slot: Vec<usize>,
    inputs: usize,
    outputs: Vec<usize>,
}

/// Lazy fade: value `v` calculated at step `last`, at step `now`.
pub(crate) fn decayed<T: Scalar>(v: T, last: u64, now: u64, decay: T) -> T {
    let dt = now.saturating_sub(last);
    if dt == 0 || v == T::ZERO {
        v
    } else {
        v * decay.powi(dt.min(100_000) as i32)
    }
}

/// Dynamic network: topology + execution state.
#[derive(Debug, Clone)]
pub struct DynamicNetwork<T = f32> {
    pub(super) topo: Topology<T>,
    pub(super) cfg: EngineConfig<T>,
    pub(super) step: u64,
    pub(super) sample: u64,
    pub(super) history: VecDeque<Vec<NeuronId>>,
    pub(super) tracker: Option<CoactivityTracker<T>>,
    pub(super) plasticity_enabled: bool,
    layout: Layout,
}

impl<T: Scalar> DynamicNetwork<T> {
    /// Create a network; topology and parameters are checked.
    pub fn new(topo: Topology<T>, cfg: EngineConfig<T>) -> StructuralResult<Self> {
        topo.validate()?;
        cfg.validate()?;
        Ok(Self {
            topo,
            cfg,
            step: 0,
            sample: 0,
            history: VecDeque::new(),
            tracker: None,
            plasticity_enabled: true,
            layout: Layout::default(),
        })
    }

    /// Topology (read only).
    pub fn topology(&self) -> &Topology<T> {
        &self.topo
    }

    /// Topology for structural operations. Change it at a safe border.
    pub fn topology_mut(&mut self) -> &mut Topology<T> {
        &mut self.topo
    }

    /// Change the neuron model with an explicit state migration policy.
    ///
    /// A wrapper over [`Topology::change_neuron_model`] for the calling code,
    /// which works with the entire network: resets the layout cache and returns
    /// [`ModelSwitchOutcome`]. Change the model at a safe boundary (between
    /// samples/minibatch), as well as other structural operations.
    pub fn change_neuron_model(
        &mut self,
        id: NeuronId,
        model: impl Into<NeuronModel<T>>,
        migration: StateMigration,
    ) -> StructuralResult<ModelSwitchOutcome<T>> {
        let outcome = self.topo.change_neuron_model(id, model, migration)?;
        self.layout = Layout::default();
        Ok(outcome)
    }

    /// Replace the topology (for example, rollback to a snapshot).
    pub fn replace_topology(&mut self, topo: Topology<T>) {
        self.topo = topo;
        self.layout = Layout::default();
    }

    /// Engine parameters.
    pub fn config(&self) -> &EngineConfig<T> {
        &self.cfg
    }

    /// Global step.
    pub fn step_count(&self) -> u64 {
        self.step
    }

    /// Sample number.
    pub fn sample_count(&self) -> u64 {
        self.sample
    }

    /// Accumulated joint activity, if enabled.
    pub fn tracker(&self) -> Option<&CoactivityTracker<T>> {
        self.tracker.as_ref()
    }

    pub(crate) fn tracker_mut(&mut self) -> Option<&mut CoactivityTracker<T>> {
        self.tracker.as_mut()
    }

    /// Enable the shared activity drive (if not already enabled).
    pub fn enable_coactivity(&mut self, window: u64, capacity: usize, max_recent: usize) {
        if self.tracker.is_none() {
            self.tracker = Some(CoactivityTracker::new(window, capacity, max_recent));
        }
    }

    /// Enable/disable weight changes (for example, during assessment).
    pub fn set_plasticity_enabled(&mut self, on: bool) {
        self.plasticity_enabled = on;
    }

    /// Is plasticity included?
    pub fn plasticity_enabled(&self) -> bool {
        self.plasticity_enabled
    }

    /// Reset dynamic state (membranes, traces, spike history),
    /// without touching the weights and accumulated statistics.
    pub fn reset_state(&mut self) {
        for n in self.topo.neurons.values_mut() {
            n.state = n.model.init_state();
            n.modulation = T::ZERO;
            n.stats.trace = T::ZERO;
        }
        for s in self.topo.synapses.values_mut() {
            s.stats.activity_trace = T::ZERO;
            s.stats.trace_step = self.step;
            s.elig = T::ZERO;
            s.elig_step = self.step;
        }
        self.history.clear();
        if let Some(t) = &mut self.tracker {
            t.clear_recent();
        }
    }

    /// Start a new sample (episode): counter +1 and reset state.
    pub fn begin_sample(&mut self) {
        self.sample += 1;
        self.reset_state();
    }

    /// Quality Assessment Copy: Plasticity Off, Condition
    /// reset, the drive is not copied. The original network is not affected.
    pub fn probe_copy(&self) -> DynamicNetwork<T> {
        let mut c = self.clone();
        c.plasticity_enabled = false;
        c.tracker = None;
        c.reset_state();
        c
    }

    /// Structural indicators.
    pub fn stats(&self, silent_rate: T, overactive_rate: T) -> StructuralStats<T> {
        StructuralStats::compute(
            &self.topo,
            self.step,
            self.sample,
            silent_rate,
            overactive_rate,
        )
    }

    fn ensure_layout(&mut self) {
        if self.layout.version == self.topo.version {
            return;
        }
        let order: Vec<NeuronId> = self.topo.neurons.keys().copied().collect();
        let index: BTreeMap<NeuronId, usize> =
            order.iter().enumerate().map(|(i, id)| (*id, i)).collect();
        let mut input_slot = vec![usize::MAX; order.len()];
        let mut inputs = 0;
        let mut outputs = Vec::new();
        for (i, id) in order.iter().enumerate() {
            match self.topo.neurons[id].role {
                NeuronRole::Input => {
                    input_slot[i] = inputs;
                    inputs += 1;
                }
                NeuronRole::Output => outputs.push(i),
                NeuronRole::Hidden => {}
            }
        }
        self.layout = Layout {
            version: self.topo.version,
            order,
            index,
            input_slot,
            inputs,
            outputs,
        };
    }

    /// One step; `inputs` - value for each input neuron (in order
    /// increasing id). In [`InputMode::Spike`] the value `≥ 0.5` means
    /// spike; in [`InputMode::Current`] the value is added to the incoming current
    /// input neuron (finite number).
    pub fn step(&mut self, inputs: &[T]) -> StructuralResult<StepOutput> {
        self.step_with_current(inputs, &[])
    }

    /// Step with additional external current into selected neurons (e.g.
    /// “teacher”, forcibly driving the correct output).
    pub fn step_with_current(
        &mut self,
        inputs: &[T],
        extra: &[(NeuronId, T)],
    ) -> StructuralResult<StepOutput> {
        self.ensure_layout();
        let lay = std::mem::take(&mut self.layout);
        let res = self.step_inner(&lay, inputs, extra);
        self.layout = lay;
        res
    }

    fn step_inner(
        &mut self,
        lay: &Layout,
        inputs: &[T],
        extra: &[(NeuronId, T)],
    ) -> StructuralResult<StepOutput> {
        if inputs.len() != lay.inputs {
            return Err(StructuralError::InputSize {
                op: "DynamicNetwork::step",
                expected: lay.inputs,
                actual: inputs.len(),
            });
        }
        if self.cfg.input_mode == InputMode::Current && inputs.iter().any(|v| !v.is_finite()) {
            return Err(StructuralError::param(
                "input",
                "in Current mode the external current must be finite",
            ));
        }
        let n = lay.order.len();
        let step = self.step;
        let sample = self.sample;
        let mut current = vec![T::ZERO; n];
        for (id, c) in extra {
            let i = *lay
                .index
                .get(id)
                .ok_or(StructuralError::NoSuchNeuron(*id))?;
            if !c.is_finite() {
                return Err(StructuralError::param("current", "non-numeric current"));
            }
            current[i] += *c;
        }

        // 1. Delivery of spikes to active synapses. Source polarity
        //    determines the contribution: excitatory - current, inhibitory - current with reverse
        //    sign, modulating - damped gain of the receiver.
        let dale = self.topo.limits.enforce_dale;
        let mut arrivals: Vec<SynapseId> = Vec::new();
        let mut mod_in = vec![T::ZERO; n];
        let mut exc_in = vec![T::ZERO; n];
        let mut inh_in = vec![T::ZERO; n];
        for (age, frame) in self.history.iter().enumerate() {
            for nid in frame {
                for sid in self.topo.out_synapses(*nid) {
                    let Some(s) = self.topo.synapses.get(sid) else {
                        continue;
                    };
                    if s.delay as usize != age || !s.is_active() {
                        continue;
                    }
                    if let Some(&di) = lay.index.get(&s.dst) {
                        let polarity = self
                            .topo
                            .neurons
                            .get(&s.src)
                            .map_or(NeuronPolarity::Excitatory, |n| n.polarity);
                        // With Dale's principle, the sign is already in weight (E ≥ 0, I ≤ 0);
                        // in legacy mode, the inhibitory source reverses the sign.
                        let effective = match polarity {
                            NeuronPolarity::Excitatory => Some(s.weight),
                            NeuronPolarity::Inhibitory if dale => Some(s.weight),
                            NeuronPolarity::Inhibitory => Some(-s.weight),
                            NeuronPolarity::Modulatory => {
                                mod_in[di] += s.weight.abs();
                                None
                            }
                        };
                        if let Some(e) = effective {
                            current[di] += e;
                            if e >= T::ZERO {
                                exc_in[di] += e;
                            } else {
                                inh_in[di] -= e;
                            }
                        }
                        arrivals.push(*sid);
                    }
                }
            }
        }

        // 2. Dynamics of neurons.
        let st = self.cfg.stdp;
        let alpha = self.cfg.rate_alpha;
        let mod_decay = self.cfg.modulation_decay;
        let mut old_trace = vec![T::ZERO; n];
        let mut spiked = vec![false; n];
        let mut spikers: Vec<NeuronId> = Vec::new();
        for (i, nid) in lay.order.iter().enumerate() {
            let Some(neuron) = self.topo.neurons.get_mut(nid) else {
                continue;
            };
            old_trace[i] = neuron.stats.trace;
            let spike = if neuron.role == NeuronRole::Input {
                let external = inputs[lay.input_slot[i]];
                match self.cfg.input_mode {
                    InputMode::Spike => external >= T::c(0.5),
                    InputMode::Current => {
                        let gain = neuron.modulation + mod_in[i];
                        let effective = (current[i] + external) * (T::ONE + gain);
                        neuron.modulation = gain * mod_decay;
                        neuron.model.step(&mut neuron.state, effective)
                    }
                }
            } else {
                let gain = neuron.modulation + mod_in[i];
                let effective = current[i] * (T::ONE + gain);
                neuron.modulation = gain * mod_decay;
                neuron.model.step(&mut neuron.state, effective)
            };
            let stt = &mut neuron.stats;
            stt.exc_drive_ema += alpha * (exc_in[i] - stt.exc_drive_ema);
            stt.inh_drive_ema += alpha * (inh_in[i] - stt.inh_drive_ema);
            stt.steps += 1;
            if spike {
                stt.spikes += 1;
                stt.last_spike_step = Some(step);
                if stt.last_marked_sample != sample + 1 {
                    stt.active_samples += 1;
                    stt.last_marked_sample = sample + 1;
                }
                spikers.push(*nid);
            }
            let s = if spike { T::ONE } else { T::ZERO };
            stt.rate_ema += alpha * (s - stt.rate_ema);
            stt.trace = stt.trace * st.trace_decay + s;
            spiked[i] = spike;
        }

        // Synapse usage statistics.
        let act_decay = self.cfg.activity_trace_decay;
        for sid in &arrivals {
            if let Some(s) = self.topo.synapses.get_mut(sid) {
                s.stats.uses += 1;
                s.stats.drive_sum += s.weight.abs();
                if s.stats.last_marked_sample != sample + 1 {
                    s.stats.active_samples += 1;
                    s.stats.last_marked_sample = sample + 1;
                }
                s.stats.activity_trace =
                    decayed(s.stats.activity_trace, s.stats.trace_step, step, act_decay) + T::ONE;
                s.stats.trace_step = step;
            }
        }

        // 3. Plasticity (event shape pair-STDP).
        if self.plasticity_enabled && st.mode != PlasticityMode::Off {
            let mut upd: BTreeMap<SynapseId, (T, T)> = BTreeMap::new();
            for sid in &arrivals {
                let Some(s) = self.topo.synapses.get(sid) else {
                    continue;
                };
                let Some(&di) = lay.index.get(&s.dst) else {
                    continue;
                };
                if dale && self.topo.polarity_of(s.src) == NeuronPolarity::Inhibitory {
                    continue; // inhibitory synapses learn their own rules
                }
                let pot = st.a_plus * old_trace[di];
                if pot != T::ZERO {
                    upd.entry(*sid).or_default().0 += pot;
                }
            }
            for dst in &spikers {
                for sid in self.topo.in_synapses(*dst) {
                    let Some(s) = self.topo.synapses.get(sid) else {
                        continue;
                    };
                    if !s.is_active() {
                        continue;
                    }
                    if dale && self.topo.polarity_of(s.src) == NeuronPolarity::Inhibitory {
                        continue;
                    }
                    let Some(&si) = lay.index.get(&s.src) else {
                        continue;
                    };
                    let dep = st.a_minus * old_trace[si];
                    if dep != T::ZERO {
                        upd.entry(*sid).or_default().1 += dep;
                    }
                }
            }
            let wmax = self.topo.param_limits.weight_abs_max;
            let topo = &mut self.topo;
            for (sid, (pot, dep)) in upd {
                let Some(s) = topo.synapses.get_mut(&sid) else {
                    continue;
                };
                let src_pol = topo
                    .neurons
                    .get(&s.src)
                    .map_or(NeuronPolarity::Excitatory, |n| n.polarity);
                let dw = match st.bound {
                    WeightBound::Soft { w_min, w_max } => {
                        pot * (w_max - s.weight) - dep * (s.weight - w_min)
                    }
                    _ => pot - dep,
                };
                match st.mode {
                    PlasticityMode::Immediate => {
                        s.weight = bound_weight(s.weight + dw, st.bound, wmax);
                        if dale {
                            s.weight = src_pol.clamp_sign(s.weight);
                        }
                    }
                    PlasticityMode::RewardModulated => {
                        s.elig = decayed(s.elig, s.elig_step, step, st.elig_decay) + dw;
                        s.elig_step = step;
                    }
                    PlasticityMode::Off => {}
                }
            }
        }

        // 3b. Inhibitory plasticity (Vogels): only with Dale's principle.
        let ic = self.cfg.inhibitory;
        if dale && ic.enabled && self.plasticity_enabled && ic.eta > T::ZERO {
            let alpha_i = T::c(2.0) * ic.target_rate / (T::ONE - st.trace_decay).max(T::c(1e-6));
            let mut dm: BTreeMap<SynapseId, T> = BTreeMap::new();
            for sid in &arrivals {
                let Some(s) = self.topo.synapses.get(sid) else {
                    continue;
                };
                if self.topo.polarity_of(s.src) != NeuronPolarity::Inhibitory {
                    continue;
                }
                if let Some(&di) = lay.index.get(&s.dst) {
                    *dm.entry(*sid).or_default() += ic.eta * (old_trace[di] - alpha_i);
                }
            }
            for dst in &spikers {
                for sid in self.topo.in_synapses(*dst) {
                    let Some(s) = self.topo.synapses.get(sid) else {
                        continue;
                    };
                    if !s.is_active() || self.topo.polarity_of(s.src) != NeuronPolarity::Inhibitory
                    {
                        continue;
                    }
                    if let Some(&si) = lay.index.get(&s.src) {
                        *dm.entry(*sid).or_default() += ic.eta * old_trace[si];
                    }
                }
            }
            let wmax = self.topo.param_limits.weight_abs_max;
            for (sid, delta) in dm {
                if let Some(s) = self.topo.synapses.get_mut(&sid) {
                    s.weight = -((-s.weight + delta).clamp(T::ZERO, wmax));
                }
            }
        }

        // 4. History and accumulator of joint activity.
        let total_spikes = spikers.len();
        if let Some(t) = &mut self.tracker {
            t.observe(&self.topo, step, &spikers);
        }
        self.history.push_front(spikers);
        self.history
            .truncate(self.topo.param_limits.delay_max as usize + 1);
        self.step += 1;

        Ok(StepOutput {
            output_spikes: lay.outputs.iter().map(|&i| spiked[i]).collect(),
            total_spikes,
        })
    }

    /// Apply Reward: In `RewardModulated` mode, the weight changes to
    /// `lr · r · eligibility`; In any mode, the contribution rating is updated
    /// synapses ([`assign_credit`](Self::assign_credit)).
    pub fn apply_reward(&mut self, reward: T) {
        if !reward.is_finite() {
            return;
        }
        let st = self.cfg.stdp;
        let step = self.step;
        if self.plasticity_enabled && st.mode == PlasticityMode::RewardModulated {
            let wmax = self.topo.param_limits.weight_abs_max;
            let dale = self.topo.limits.enforce_dale;
            let topo = &mut self.topo;
            for s in topo.synapses.values_mut() {
                if !s.is_active() {
                    continue;
                }
                let e = decayed(s.elig, s.elig_step, step, st.elig_decay);
                if e != T::ZERO {
                    let mut w = bound_weight(s.weight + st.reward_lr * reward * e, st.bound, wmax);
                    if dale {
                        let pol = topo
                            .neurons
                            .get(&s.src)
                            .map_or(NeuronPolarity::Excitatory, |n| n.polarity);
                        w = pol.clamp_sign(w);
                    }
                    s.weight = w;
                }
            }
        }
        self.assign_credit(reward);
    }

    /// Local adjustment of inhibitory strength (see [`EiBalanceConfig`]).
    ///
    /// Works on Dale's principle; otherwise it does nothing. Only changes
    /// the magnitudes of inhibitory weights (the sign is preserved), so the structure and
    /// the Dale invariant are not affected.
    pub fn balance_inhibition(&mut self, cfg: &EiBalanceConfig<T>) -> EiBalanceReport<T> {
        let mut report = EiBalanceReport::default();
        if !self.topo.limits.enforce_dale || cfg.target_ratio <= T::ZERO {
            return report;
        }
        let wmax = self.topo.param_limits.weight_abs_max;
        let mut factors: BTreeMap<NeuronId, T> = BTreeMap::new();
        let mut ratio_sum = T::ZERO;
        for n in self.topo.neurons.values() {
            if n.role == NeuronRole::Input
                || n.stats.steps < cfg.min_observed_steps
                || n.stats.exc_drive_ema < cfg.min_drive
            {
                continue;
            }
            let ratio = n.stats.inh_drive_ema / n.stats.exc_drive_ema;
            ratio_sum += ratio;
            let error = (cfg.target_ratio - ratio) / cfg.target_ratio;
            let factor =
                (T::ONE + cfg.rate * error).clamp(T::ONE - cfg.max_step, T::ONE + cfg.max_step);
            factors.insert(n.id, factor);
        }
        if factors.is_empty() {
            return report;
        }
        let considered = factors.len();
        let topo = &mut self.topo;
        let mut touched: BTreeMap<NeuronId, bool> = BTreeMap::new();
        for s in topo.synapses.values_mut() {
            let Some(&f) = factors.get(&s.dst) else {
                continue;
            };
            if !s.is_active()
                || topo.neurons.get(&s.src).map(|n| n.polarity) != Some(NeuronPolarity::Inhibitory)
            {
                continue;
            }
            s.weight = (s.weight * f).clamp(-wmax, T::ZERO);
            touched.insert(s.dst, true);
        }
        report.adjusted = touched.len();
        report.mean_ratio = ratio_sum / T::from_usize(considered);
        report
    }

    /// Take into account the quality signal (more is better: reward or
    /// improvement in loss) in assessing the contribution of active synapses: `credit_ema` is growing
    /// at synapses that transmitted spikes before the positive signal.
    /// Synapses without recent transmission do not change their score, so they rarely
    /// a used but useful synapse does not “forget” its contribution.
    pub fn assign_credit(&mut self, signal: T) {
        if !signal.is_finite() {
            return;
        }
        let (step, beta, decay) = (
            self.step,
            self.cfg.credit_alpha,
            self.cfg.activity_trace_decay,
        );
        for s in self.topo.synapses.values_mut() {
            if !s.is_active() {
                continue;
            }
            let tr = decayed(s.stats.activity_trace, s.stats.trace_step, step, decay);
            if tr > T::c(1e-6) {
                s.stats.credit_ema += beta * (signal * tr - s.stats.credit_ema);
            }
        }
    }
}

fn bound_weight<T: Scalar>(w: T, bound: WeightBound<T>, wmax: T) -> T {
    let w = match bound {
        WeightBound::Hard { w_min, w_max } | WeightBound::Soft { w_min, w_max } => {
            w.clamp(w_min, w_max)
        }
        WeightBound::None => w,
    };
    w.clamp(-wmax, wmax)
}
