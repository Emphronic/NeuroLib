//! Versioned checkpoint of structural plasticity.
//!
//! Unlike the “weights only” checkpoint, everything you need is saved here
//! to continue self-adaptation from the same place:
//!
//! * topology: neurons (parameters, membranes, statistics, metadata) and
//!   synapses (weights, delays, status, statistics, eligibility);
//! * engine state: step, sample number, spike history, accumulated
//!   joint activity, engine parameters, and plasticity;
//! * controller state: configuration, RNG, log and counters, loss window,
//!   plateau, pauses after rollbacks, delayed check with snapshot, best
//!   snapshot;
//! * `extra` - opaque block of calling code (for example, state
//!   external optimizer and training mode): the engine on the STDP optimizer does not
//!   has, and for hybrid schemes with their own optimizer, its state is put
//!   here and returns unchanged.
//!
//! # Format
//!
//! `"NSTR"`, version of `u32` (currently [`FORMAT_VERSION`] = 5), payload in
//! little-endian, FNV-1a-64 checksum of all preceding bytes.
//! Version 1 (only LIF, role + `LifParams` + membrane) read and
//! migrates: the role is preserved, the polarity becomes excitatory, the model -
//! LIF, state – `LifState`. Version 2 adds model/polarity and
//! `modulation_decay`; when reading version 2, the input mode is the same
//! [`InputMode::Spike`], automatic selection
//! models is turned off. Version 3 saves `InputMode` and settings
//! `ModelSelectionConfig`. Version 4 adds model selection loss evaluation fields
//! (`loss_evaluation`, `loss_min_improvement`, `max_probes_per_interval`) and
//! [`ParamLearningConfig`]; version 3 reads with learning settings
//! default settings (off). Version 5 adds Dale's principle
//! (`TopologyLimits::enforce_dale`), moving averages of excitatory and
//! inhibitory input current of neurons, `InhibitoryPlasticityConfig` and
//! `EiBalanceConfig`; versions 1–4 are read in legacy mode with free
//! weight signs (`enforce_dale = false`) and zero E/I statistics -
//! You can enable the Dale principle for such a network via
//! `Topology::set_enforce_dale(true)`.
//! Version 6 ([`FORMAT_VERSION_F64`]) is written for networks on `f64`: after version
//! there is a value width byte (8), all real values occupy 8 bytes.
//! For `f32` version 5 is still written, so `f32` files are readable by the old one
//! code. You can read into any scalar: `load_checkpoint` (in `f32`) and
//! `load_checkpoint_as::<T>`; `f32 → f64` without loss, `f64 → f32` with rounding.
//! The newer version file is rejected with an understandable error; data corruption
//! is caught by the checksum, and the restored topology passes
//! `Topology::validate`.
//!
//! Custom `StructureCost` (`with_cost`) not serializing: after
//! loading is used `LinearCost` from the configuration, you need your own criterion
//! ask again.

use std::collections::{BTreeMap, VecDeque};
use std::path::Path;

use crate::functional::plasticity::WeightBound;
use crate::scalar::Scalar;
use crate::state::ResetMode;

use super::controller::{BestSnapshot, Pending, StructuralPlasticityController};
use super::error::{StructuralError, StructuralResult};
use super::model::{
    AdExParams, AdExState, AdaptiveLifParams, AdaptiveLifState, HodgkinHuxleyParams,
    HodgkinHuxleyState, IzhikevichParams, IzhikevichState, LifParams, LifState, NeuronModel,
    NeuronPolarity, NeuronRole, NeuronState, StateMigration,
};
use super::model_selection::{ModelCandidate, ModelSelectionConfig};
use super::mutation::{ActionKind, EventLog, Outcome, OutcomeCounts, StructuralEvent};
use super::network::{
    DynamicNetwork, EiBalanceConfig, EngineConfig, InhibitoryPlasticityConfig, InputMode,
    PlasticityMode, StdpConfig,
};
use super::neuron_ops::SplitInit;
use super::param_learning::ParamLearningConfig;
use super::policy::{GrowthStrategy, StructuralConfig};
use super::rng::Rng64;
use super::stats::{CoactivityTracker, LinearCost};
use super::topology::{
    Neuron, NeuronId, NeuronStats, Origin, ParamLimits, StructMeta, Synapse, SynapseId,
    SynapseState, SynapseStats, Topology, TopologyLimits,
};

/// Current format version.
pub const FORMAT_VERSION: u32 = 5;
/// Real value version of the format in `f64` (written for `T = f64`).
/// After the version there is a byte of the width of the value (now always 8).
pub const FORMAT_VERSION_F64: u32 = 6;
const MAGIC: &[u8; 4] = b"NSTR";

/// Restored condition.
#[derive(Debug)]
pub struct Checkpoint<T: Scalar = f32> {
    /// Network.
    pub network: DynamicNetwork<T>,
    /// Controller.
    pub controller: StructuralPlasticityController<T>,
    /// Calling code block, saved as is.
    pub extra: Vec<u8>,
}

impl<T: Scalar> Checkpoint<T> {
    /// Convert the loaded legacy network (versions 1–4 are read with free
    /// weight signs) into E/I mode: the signs of all weights are adjusted to the polarity of the
    /// source (the magnitude is preserved) on the network, in controller rollback snapshots
    /// and in its configuration. Returns the number of synapses changed; repeated
    /// the call returns `0`.
    ///
    /// Attention: the legacy synapse has an “excitatory source with a negative
    /// weight" the meaning changes to excitatory - neurons that should
    /// slow down, you need to do `NeuronPolarity::Inhibitory` in advance.
    pub fn enable_dale(&mut self) -> usize {
        let mut changed = self.network.topology_mut().set_enforce_dale(true);
        changed += self.controller.enable_dale();
        changed
    }
}

fn fnv1a(data: &[u8]) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for b in data {
        h ^= *b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

/// Writer; `wide` — width of real values (`false`: f32, versions 1–5;
/// `true`: f64, version 6).
struct W(Vec<u8>, bool);

impl W {
    fn u8(&mut self, v: u8) {
        self.0.push(v);
    }
    fn bool(&mut self, v: bool) {
        self.0.push(v as u8);
    }
    fn u16(&mut self, v: u16) {
        self.0.extend_from_slice(&v.to_le_bytes());
    }
    fn u32(&mut self, v: u32) {
        self.0.extend_from_slice(&v.to_le_bytes());
    }
    fn u64(&mut self, v: u64) {
        self.0.extend_from_slice(&v.to_le_bytes());
    }
    fn usize(&mut self, v: usize) {
        self.u64(v as u64);
    }
    /// Real value in format width (historical name).
    fn f32<T: Scalar>(&mut self, v: T) {
        if self.1 {
            self.0.extend_from_slice(&v.to_f64().to_le_bytes());
        } else {
            self.0.extend_from_slice(&(v.to_f64() as f32).to_le_bytes());
        }
    }
    fn str(&mut self, v: &str) {
        self.usize(v.len());
        self.0.extend_from_slice(v.as_bytes());
    }
    fn bytes(&mut self, v: &[u8]) {
        self.usize(v.len());
        self.0.extend_from_slice(v);
    }
    fn opt_u64(&mut self, v: Option<u64>) {
        self.bool(v.is_some());
        self.u64(v.unwrap_or(0));
    }
    fn opt_f32<T: Scalar>(&mut self, v: Option<T>) {
        self.bool(v.is_some());
        self.f32(v.unwrap_or(T::ZERO));
    }
}

struct R<'a> {
    b: &'a [u8],
    p: usize,
    /// Real values width: `true` - f64.
    wide: bool,
}

type Res<T> = StructuralResult<T>;

fn err(m: impl Into<String>) -> StructuralError {
    StructuralError::Checkpoint(m.into())
}

impl<'a> R<'a> {
    fn take(&mut self, n: usize) -> Res<&'a [u8]> {
        let end = self
            .p
            .checked_add(n)
            .filter(|e| *e <= self.b.len())
            .ok_or_else(|| err("unexpected end of data"))?;
        let s = &self.b[self.p..end];
        self.p = end;
        Ok(s)
    }
    fn arr<const N: usize>(&mut self) -> Res<[u8; N]> {
        let mut a = [0u8; N];
        a.copy_from_slice(self.take(N)?);
        Ok(a)
    }
    fn u8(&mut self) -> Res<u8> {
        Ok(self.take(1)?[0])
    }
    fn bool(&mut self) -> Res<bool> {
        match self.u8()? {
            0 => Ok(false),
            1 => Ok(true),
            v => Err(err(format!("invalid bool {v}"))),
        }
    }
    fn u16(&mut self) -> Res<u16> {
        Ok(u16::from_le_bytes(self.arr()?))
    }
    fn u32(&mut self) -> Res<u32> {
        Ok(u32::from_le_bytes(self.arr()?))
    }
    fn u64(&mut self) -> Res<u64> {
        Ok(u64::from_le_bytes(self.arr()?))
    }
    fn usize(&mut self) -> Res<usize> {
        usize::try_from(self.u64()?).map_err(|_| err("the number does not fit into usize"))
    }
    /// Real value in format width (historical name).
    fn f32<T: Scalar>(&mut self) -> Res<T> {
        if self.wide {
            Ok(T::from_f64(f64::from_le_bytes(self.arr()?)))
        } else {
            Ok(T::from_f64(f32::from_le_bytes(self.arr()?) as f64))
        }
    }
    /// Number of elements; cannot exceed the number of bytes remaining.
    fn count(&mut self) -> Res<usize> {
        let n = self.usize()?;
        if n > self.b.len() - self.p {
            return Err(err("collection size exceeds data size"));
        }
        Ok(n)
    }
    fn str(&mut self) -> Res<String> {
        let n = self.count()?;
        String::from_utf8(self.take(n)?.to_vec()).map_err(|_| err("string is not UTF-8"))
    }
    fn bytes(&mut self) -> Res<Vec<u8>> {
        let n = self.count()?;
        Ok(self.take(n)?.to_vec())
    }
    fn opt_u64(&mut self) -> Res<Option<u64>> {
        let some = self.bool()?;
        let v = self.u64()?;
        Ok(some.then_some(v))
    }
    fn opt_f32<T: Scalar>(&mut self) -> Res<Option<T>> {
        let some = self.bool()?;
        let v = self.f32()?;
        Ok(some.then_some(v))
    }
}

fn kind_code(k: ActionKind) -> u8 {
    k.code()
}

fn kind_from(c: u8) -> Res<ActionKind> {
    ActionKind::from_code(c).ok_or_else(|| err(format!("unknown operation type {c}")))
}

// ---------------- simple types ----------------

fn w_reset(w: &mut W, r: ResetMode) {
    w.u8(match r {
        ResetMode::Subtract => 0,
        ResetMode::Zero => 1,
        ResetMode::None => 2,
    });
}

fn r_reset(r: &mut R) -> Res<ResetMode> {
    match r.u8()? {
        0 => Ok(ResetMode::Subtract),
        1 => Ok(ResetMode::Zero),
        2 => Ok(ResetMode::None),
        v => Err(err(format!("unknown ResetMode {v}"))),
    }
}

fn w_lif<T: Scalar>(w: &mut W, p: &LifParams<T>) {
    w.f32(p.beta);
    w.f32(p.threshold);
    w_reset(w, p.reset);
}

fn r_lif<T: Scalar>(r: &mut R) -> Res<LifParams<T>> {
    Ok(LifParams {
        beta: r.f32()?,
        threshold: r.f32()?,
        reset: r_reset(r)?,
    })
}

fn w_model<T: Scalar>(w: &mut W, m: &NeuronModel<T>) {
    w.u8(m.tag());
    match m {
        NeuronModel::Lif(p) => w_lif(w, p),
        NeuronModel::AdaptiveLif(p) => {
            w.f32(p.beta);
            w.f32(p.rho);
            w.f32(p.adapt_increment);
            w.f32(p.threshold);
            w_reset(w, p.reset);
        }
        NeuronModel::Izhikevich(p) => {
            w.f32(p.a);
            w.f32(p.b);
            w.f32(p.c);
            w.f32(p.d);
            w.f32(p.threshold);
            w.usize(p.substeps);
        }
        NeuronModel::AdEx(p) => {
            for v in [
                p.c, p.g_l, p.e_l, p.v_t, p.delta_t, p.a, p.tau_w, p.b, p.v_reset, p.v_peak, p.dt,
                p.exp_clip,
            ] {
                w.f32(v);
            }
            w.usize(p.substeps);
        }
        NeuronModel::HodgkinHuxley(p) => {
            for v in [
                p.c,
                p.g_na,
                p.g_k,
                p.g_l,
                p.e_na,
                p.e_k,
                p.e_l,
                p.v_rest,
                p.spike_threshold,
                p.dt,
            ] {
                w.f32(v);
            }
            w.usize(p.substeps);
        }
    }
}

fn r_model<T: Scalar>(r: &mut R) -> Res<NeuronModel<T>> {
    Ok(match r.u8()? {
        0 => NeuronModel::Lif(r_lif(r)?),
        1 => NeuronModel::AdaptiveLif(AdaptiveLifParams {
            beta: r.f32()?,
            rho: r.f32()?,
            adapt_increment: r.f32()?,
            threshold: r.f32()?,
            reset: r_reset(r)?,
        }),
        2 => NeuronModel::Izhikevich(IzhikevichParams {
            a: r.f32()?,
            b: r.f32()?,
            c: r.f32()?,
            d: r.f32()?,
            threshold: r.f32()?,
            substeps: r.usize()?,
        }),
        3 => NeuronModel::AdEx(AdExParams {
            c: r.f32()?,
            g_l: r.f32()?,
            e_l: r.f32()?,
            v_t: r.f32()?,
            delta_t: r.f32()?,
            a: r.f32()?,
            tau_w: r.f32()?,
            b: r.f32()?,
            v_reset: r.f32()?,
            v_peak: r.f32()?,
            dt: r.f32()?,
            exp_clip: r.f32()?,
            substeps: r.usize()?,
        }),
        4 => NeuronModel::HodgkinHuxley(HodgkinHuxleyParams {
            c: r.f32()?,
            g_na: r.f32()?,
            g_k: r.f32()?,
            g_l: r.f32()?,
            e_na: r.f32()?,
            e_k: r.f32()?,
            e_l: r.f32()?,
            v_rest: r.f32()?,
            spike_threshold: r.f32()?,
            dt: r.f32()?,
            substeps: r.usize()?,
        }),
        v => return Err(err(format!("unknown neuron model {v}"))),
    })
}

fn w_state<T: Scalar>(w: &mut W, s: &NeuronState<T>) {
    w.u8(s.tag());
    match s {
        NeuronState::Lif(s) => w.f32(s.mem),
        NeuronState::AdaptiveLif(s) => {
            w.f32(s.mem);
            w.f32(s.adaptation);
        }
        NeuronState::Izhikevich(s) => {
            w.f32(s.v);
            w.f32(s.u);
        }
        NeuronState::AdEx(s) => {
            w.f32(s.v);
            w.f32(s.w);
        }
        NeuronState::HodgkinHuxley(s) => {
            w.f32(s.v);
            w.f32(s.m);
            w.f32(s.h);
            w.f32(s.n);
            w.bool(s.above);
        }
    }
}

fn r_state<T: Scalar>(r: &mut R) -> Res<NeuronState<T>> {
    Ok(match r.u8()? {
        0 => NeuronState::Lif(LifState { mem: r.f32()? }),
        1 => NeuronState::AdaptiveLif(AdaptiveLifState {
            mem: r.f32()?,
            adaptation: r.f32()?,
        }),
        2 => NeuronState::Izhikevich(IzhikevichState {
            v: r.f32()?,
            u: r.f32()?,
        }),
        3 => NeuronState::AdEx(AdExState {
            v: r.f32()?,
            w: r.f32()?,
        }),
        4 => NeuronState::HodgkinHuxley(HodgkinHuxleyState {
            v: r.f32()?,
            m: r.f32()?,
            h: r.f32()?,
            n: r.f32()?,
            above: r.bool()?,
        }),
        v => return Err(err(format!("unknown state of neuron {v}"))),
    })
}

fn r_role(r: &mut R) -> Res<NeuronRole> {
    NeuronRole::from_code(r.u8()?).ok_or_else(|| err("unknown role of the neuron"))
}

fn r_polarity(r: &mut R) -> Res<NeuronPolarity> {
    NeuronPolarity::from_code(r.u8()?).ok_or_else(|| err("unknown neuron polarity"))
}

fn w_meta(w: &mut W, m: &StructMeta) {
    w.u64(m.born_step);
    w.u64(m.born_sample);
    let (t, v) = m.origin.encode();
    w.u8(t);
    w.u64(v);
    w.bool(m.protected);
    w.u64(m.last_evaluated_step);
}

fn r_meta(r: &mut R) -> Res<StructMeta> {
    let born_step = r.u64()?;
    let born_sample = r.u64()?;
    let (t, v) = (r.u8()?, r.u64()?);
    let origin = Origin::decode(t, v).ok_or_else(|| err("unknown Origin"))?;
    Ok(StructMeta {
        born_step,
        born_sample,
        origin,
        protected: r.bool()?,
        last_evaluated_step: r.u64()?,
    })
}

// ---------------- configuration ----------------

fn w_config<T: Scalar>(w: &mut W, c: &StructuralConfig<T>, version: u32) {
    let t = &c.topology;
    w.usize(t.max_neurons);
    w.usize(t.max_synapses);
    w.usize(t.max_in_degree);
    w.usize(t.max_out_degree);
    w.bool(t.allow_self_loops);
    w.bool(t.allow_recurrent);
    w.bool(t.allow_duplicates);
    if version >= 5 {
        w.bool(t.enforce_dale);
    }
    w_param_limits(w, &c.params);
    w.u64(c.schedule.analyze_every);
    w.u64(c.schedule.min_interval_steps);
    w.usize(c.schedule.max_ops_per_interval);
    let g = &c.growth;
    w.bool(g.enabled);
    w.bool(g.track_coactivity);
    w.u64(g.coactivity_window);
    w.usize(g.coactivity_capacity);
    w.usize(g.coactivity_max_recent);
    w.f32(g.coactivity_decay);
    w.opt_f32(g.target_loss);
    w.bool(g.allow_output_to_output);
    w.usize(g.synapses.max_new_per_interval);
    w.f32(g.synapses.min_score);
    w.f32(g.synapses.init_weight);
    w.f32(g.synapses.weight_jitter);
    w.u16(g.synapses.delay);
    let n = &g.neurons;
    w.bool(n.enabled);
    w.usize(n.plateau_patience);
    w.f32(n.plateau_min_delta);
    w.f32(n.min_loss);
    w.usize(n.max_new_per_interval);
    w.usize(n.strategies.len());
    for s in &n.strategies {
        w.u8(match s {
            GrowthStrategy::SplitSynapse => 0,
            GrowthStrategy::CloneNeuron => 1,
            GrowthStrategy::NewNeuron => 2,
        });
    }
    w_lif(w, &n.new_neuron_params);
    w.usize(n.new_neuron_fan);
    w.f32(n.new_neuron_weight);
    w.f32(n.clone_jitter);
    w.f32(n.split.relay_beta);
    w.f32(n.split.relay_threshold);
    w.f32(n.split.relay_gain);
    let p = &c.pruning;
    w.bool(p.enabled);
    w.u64(p.min_age_steps);
    w.u64(p.min_observed_samples);
    w.u64(p.reeval_interval_steps);
    w.f32(p.drive_rel_threshold);
    w.f32(p.protect_credit);
    w.f32(p.rare_coverage);
    w.u64(p.grace_steps);
    w.f32(p.max_fraction_per_interval);
    w.usize(p.min_synapses);
    w.usize(p.min_output_in_degree);
    w.usize(p.min_hidden);
    w.bool(p.prune_dead_neurons);
    w.bool(p.prune_silent_neurons);
    w.f32(p.neuron_min_rate);
    w.f32(p.neuron_min_coverage);
    w.usize(p.max_neuron_prunes);
    let h = &c.homeostasis;
    w.bool(h.enabled);
    w.f32(h.rate_min);
    w.f32(h.rate_max);
    w.u64(h.min_observed_steps);
    w.f32(h.threshold_step);
    w.usize(h.max_neurons_per_interval);
    let r = &c.rollback;
    w.bool(r.enabled);
    w.f32(r.rel_tolerance);
    w.f32(r.abs_tolerance);
    w.bool(r.use_structure_cost);
    w.u64(r.verify_after_calls);
    w.u64(r.cooldown_steps);
    w.u64(r.max_cooldown_steps);
    w.f32(c.cost.lambda_neurons);
    w.f32(c.cost.lambda_synapses);
    w.f32(c.cost.lambda_activity);
    w.f32(c.cost.lambda_state_bytes);
    w.bool(c.max_state_bytes.is_some());
    w.usize(c.max_state_bytes.unwrap_or(0));
    w.u64(c.seed);
    w.usize(c.log_capacity);
    w.usize(c.loss_window);
    // Version 2: model and polarity of new neurons.
    if version >= 2 {
        w.bool(n.new_neuron_model.is_some());
        if let Some(m) = &n.new_neuron_model {
            w_model(w, m);
        }
        w.u8(n.new_neuron_polarity.code());
    }
    // Version 3: automatic model selection.
    if version >= 3 {
        w_model_selection(w, &c.model_selection, version);
    }
    // Version 4: training the physical parameters of neurons.
    if version >= 4 {
        w_param_learning(w, &c.param_learning);
    }
}

fn w_model_selection<T: Scalar>(w: &mut W, m: &ModelSelectionConfig<T>, version: u32) {
    w.bool(m.enabled);
    w.u64(m.min_observed_steps);
    w.usize(m.max_switches_per_interval);
    w.f32(m.min_improvement);
    w.f32(m.efficiency_weight);
    w.f32(m.specialization_weight);
    if version >= 4 {
        w.bool(m.loss_evaluation);
        w.f32(m.loss_min_improvement);
        w.usize(m.max_probes_per_interval);
    }
    w.u8(m.migration.code());
    w.bool(m.include_inputs);
    w.bool(m.include_outputs);
    w.usize(m.candidates.len());
    for c in &m.candidates {
        w_model(w, &c.model);
        w.f32(c.cost);
        w.opt_f32(c.target_rate);
        w.f32(c.rate_tolerance);
    }
}

fn w_param_learning<T: Scalar>(w: &mut W, p: &ParamLearningConfig<T>) {
    w.bool(p.enabled);
    w.u64(p.min_observed_steps);
    w.usize(p.max_neurons_per_interval);
    w.usize(p.max_params_per_neuron);
    w.usize(p.max_probes_per_interval);
    w.f32(p.step);
    w.f32(p.min_step);
    w.f32(p.min_improvement);
    w.bool(p.include_inputs);
    w.bool(p.include_outputs);
}

fn w_param_limits<T: Scalar>(w: &mut W, p: &ParamLimits<T>) {
    w.f32(p.beta_min);
    w.f32(p.beta_max);
    w.f32(p.threshold_min);
    w.f32(p.threshold_max);
    w.f32(p.weight_abs_max);
    w.u16(p.delay_max);
}

fn r_param_limits<T: Scalar>(r: &mut R) -> Res<ParamLimits<T>> {
    Ok(ParamLimits {
        beta_min: r.f32()?,
        beta_max: r.f32()?,
        threshold_min: r.f32()?,
        threshold_max: r.f32()?,
        weight_abs_max: r.f32()?,
        delay_max: r.u16()?,
    })
}

#[allow(clippy::field_reassign_with_default)]
fn r_config<T: Scalar>(r: &mut R, version: u32) -> Res<StructuralConfig<T>> {
    let mut c = StructuralConfig::default();
    c.topology = TopologyLimits {
        max_neurons: r.usize()?,
        max_synapses: r.usize()?,
        max_in_degree: r.usize()?,
        max_out_degree: r.usize()?,
        allow_self_loops: r.bool()?,
        allow_recurrent: r.bool()?,
        allow_duplicates: r.bool()?,
        enforce_dale: if version >= 5 { r.bool()? } else { false },
    };
    c.params = r_param_limits(r)?;
    c.schedule.analyze_every = r.u64()?;
    c.schedule.min_interval_steps = r.u64()?;
    c.schedule.max_ops_per_interval = r.usize()?;
    let g = &mut c.growth;
    g.enabled = r.bool()?;
    g.track_coactivity = r.bool()?;
    g.coactivity_window = r.u64()?;
    g.coactivity_capacity = r.usize()?;
    g.coactivity_max_recent = r.usize()?;
    g.coactivity_decay = r.f32()?;
    g.target_loss = r.opt_f32()?;
    g.allow_output_to_output = r.bool()?;
    g.synapses.max_new_per_interval = r.usize()?;
    g.synapses.min_score = r.f32()?;
    g.synapses.init_weight = r.f32()?;
    g.synapses.weight_jitter = r.f32()?;
    g.synapses.delay = r.u16()?;
    let n = &mut g.neurons;
    n.enabled = r.bool()?;
    n.plateau_patience = r.usize()?;
    n.plateau_min_delta = r.f32()?;
    n.min_loss = r.f32()?;
    n.max_new_per_interval = r.usize()?;
    let cnt = r.count()?;
    n.strategies = Vec::with_capacity(cnt);
    for _ in 0..cnt {
        n.strategies.push(match r.u8()? {
            0 => GrowthStrategy::SplitSynapse,
            1 => GrowthStrategy::CloneNeuron,
            2 => GrowthStrategy::NewNeuron,
            v => return Err(err(format!("unknown growth strategy {v}"))),
        });
    }
    n.new_neuron_params = r_lif(r)?;
    n.new_neuron_fan = r.usize()?;
    n.new_neuron_weight = r.f32()?;
    n.clone_jitter = r.f32()?;
    n.split = SplitInit {
        relay_beta: r.f32()?,
        relay_threshold: r.f32()?,
        relay_gain: r.f32()?,
    };
    let p = &mut c.pruning;
    p.enabled = r.bool()?;
    p.min_age_steps = r.u64()?;
    p.min_observed_samples = r.u64()?;
    p.reeval_interval_steps = r.u64()?;
    p.drive_rel_threshold = r.f32()?;
    p.protect_credit = r.f32()?;
    p.rare_coverage = r.f32()?;
    p.grace_steps = r.u64()?;
    p.max_fraction_per_interval = r.f32()?;
    p.min_synapses = r.usize()?;
    p.min_output_in_degree = r.usize()?;
    p.min_hidden = r.usize()?;
    p.prune_dead_neurons = r.bool()?;
    p.prune_silent_neurons = r.bool()?;
    p.neuron_min_rate = r.f32()?;
    p.neuron_min_coverage = r.f32()?;
    p.max_neuron_prunes = r.usize()?;
    let h = &mut c.homeostasis;
    h.enabled = r.bool()?;
    h.rate_min = r.f32()?;
    h.rate_max = r.f32()?;
    h.min_observed_steps = r.u64()?;
    h.threshold_step = r.f32()?;
    h.max_neurons_per_interval = r.usize()?;
    let b = &mut c.rollback;
    b.enabled = r.bool()?;
    b.rel_tolerance = r.f32()?;
    b.abs_tolerance = r.f32()?;
    b.use_structure_cost = r.bool()?;
    b.verify_after_calls = r.u64()?;
    b.cooldown_steps = r.u64()?;
    b.max_cooldown_steps = r.u64()?;
    c.cost = LinearCost {
        lambda_neurons: r.f32()?,
        lambda_synapses: r.f32()?,
        lambda_activity: r.f32()?,
        lambda_state_bytes: r.f32()?,
    };
    let has = r.bool()?;
    let bytes = r.usize()?;
    c.max_state_bytes = has.then_some(bytes);
    c.seed = r.u64()?;
    c.log_capacity = r.usize()?;
    c.loss_window = r.usize()?;
    if version >= 2 {
        let n = &mut c.growth.neurons;
        n.new_neuron_model = if r.bool()? { Some(r_model(r)?) } else { None };
        n.new_neuron_polarity = r_polarity(r)?;
    }
    if version >= 3 {
        c.model_selection = r_model_selection(r, version)?;
    }
    if version >= 4 {
        c.param_learning = r_param_learning(r)?;
    }
    Ok(c)
}

fn r_model_selection<T: Scalar>(r: &mut R, version: u32) -> Res<ModelSelectionConfig<T>> {
    let mut m = ModelSelectionConfig::default();
    m.enabled = r.bool()?;
    m.min_observed_steps = r.u64()?;
    m.max_switches_per_interval = r.usize()?;
    m.min_improvement = r.f32()?;
    m.efficiency_weight = r.f32()?;
    m.specialization_weight = r.f32()?;
    if version >= 4 {
        m.loss_evaluation = r.bool()?;
        m.loss_min_improvement = r.f32()?;
        m.max_probes_per_interval = r.usize()?;
    }
    m.migration =
        StateMigration::from_code(r.u8()?).ok_or_else(|| err("unknown state migration policy"))?;
    m.include_inputs = r.bool()?;
    m.include_outputs = r.bool()?;
    let count = r.count()?;
    m.candidates = Vec::with_capacity(count);
    for _ in 0..count {
        m.candidates.push(ModelCandidate {
            model: r_model(r)?,
            cost: r.f32()?,
            target_rate: r.opt_f32()?,
            rate_tolerance: r.f32()?,
        });
    }
    Ok(m)
}

fn r_param_learning<T: Scalar>(r: &mut R) -> Res<ParamLearningConfig<T>> {
    Ok(ParamLearningConfig {
        enabled: r.bool()?,
        min_observed_steps: r.u64()?,
        max_neurons_per_interval: r.usize()?,
        max_params_per_neuron: r.usize()?,
        max_probes_per_interval: r.usize()?,
        step: r.f32()?,
        min_step: r.f32()?,
        min_improvement: r.f32()?,
        include_inputs: r.bool()?,
        include_outputs: r.bool()?,
    })
}

// ---------------- topology ----------------

fn w_topology<T: Scalar>(w: &mut W, t: &Topology<T>, version: u32) {
    w.usize(t.limits.max_neurons);
    w.usize(t.limits.max_synapses);
    w.usize(t.limits.max_in_degree);
    w.usize(t.limits.max_out_degree);
    w.bool(t.limits.allow_self_loops);
    w.bool(t.limits.allow_recurrent);
    w.bool(t.limits.allow_duplicates);
    if version >= 5 {
        w.bool(t.limits.enforce_dale);
    }
    w_param_limits(w, &t.param_limits);
    w.u64(t.next_neuron);
    w.u64(t.next_synapse);
    w.usize(t.neurons.len());
    for n in t.neurons.values() {
        w.u64(n.id.0);
        w.u8(n.role.code());
        if version >= 2 {
            w.u8(n.polarity.code());
            w_model(w, &n.model);
            w_state(w, &n.state);
            w.f32(n.modulation);
        } else {
            // Version 1 only knows LIF and the membrane.
            let lif = match &n.model {
                NeuronModel::Lif(p) => *p,
                _ => LifParams::default(),
            };
            w_lif(w, &lif);
            w.f32(n.state.mem());
        }
        w.u64(n.stats.spikes);
        w.u64(n.stats.steps);
        w.f32(n.stats.rate_ema);
        w.opt_u64(n.stats.last_spike_step);
        w.f32(n.stats.trace);
        w.u64(n.stats.active_samples);
        w.u64(n.stats.last_marked_sample);
        if version >= 5 {
            w.f32(n.stats.exc_drive_ema);
            w.f32(n.stats.inh_drive_ema);
        }
        w_meta(w, &n.meta);
    }
    w.usize(t.synapses.len());
    for s in t.synapses.values() {
        w.u64(s.id.0);
        w.u64(s.src.0);
        w.u64(s.dst.0);
        w.f32(s.weight);
        w.u16(s.delay);
        match s.state {
            SynapseState::Active => {
                w.u8(0);
                w.u64(0);
            }
            SynapseState::Disabled { since_step } => {
                w.u8(1);
                w.u64(since_step);
            }
        }
        w.u64(s.stats.uses);
        w.f32(s.stats.drive_sum);
        w.u64(s.stats.active_samples);
        w.u64(s.stats.last_marked_sample);
        w.f32(s.stats.credit_ema);
        w.f32(s.stats.activity_trace);
        w.u64(s.stats.trace_step);
        w.f32(s.elig);
        w.u64(s.elig_step);
        w_meta(w, &s.meta);
    }
}

fn r_topology<T: Scalar>(r: &mut R, version: u32) -> Res<Topology<T>> {
    let limits = TopologyLimits {
        max_neurons: r.usize()?,
        max_synapses: r.usize()?,
        max_in_degree: r.usize()?,
        max_out_degree: r.usize()?,
        allow_self_loops: r.bool()?,
        allow_recurrent: r.bool()?,
        allow_duplicates: r.bool()?,
        enforce_dale: if version >= 5 { r.bool()? } else { false },
    };
    let pl = r_param_limits(r)?;
    let mut t = Topology::<T>::new(limits, pl);
    t.next_neuron = r.u64()?;
    t.next_synapse = r.u64()?;
    for _ in 0..r.count()? {
        let id = NeuronId(r.u64()?);
        let (role, polarity, model, state, modulation) = if version >= 2 {
            let role = r_role(r)?;
            let polarity = r_polarity(r)?;
            let model = r_model(r)?;
            let state = r_state(r)?;
            let modulation = r.f32()?;
            (role, polarity, model, state, modulation)
        } else {
            let role = r_role(r)?;
            let params = r_lif(r)?;
            let mem = r.f32()?;
            (
                role,
                NeuronPolarity::Excitatory,
                NeuronModel::Lif(params),
                NeuronState::Lif(LifState { mem }),
                T::ZERO,
            )
        };
        let stats = NeuronStats::<T> {
            spikes: r.u64()?,
            steps: r.u64()?,
            rate_ema: r.f32()?,
            last_spike_step: r.opt_u64()?,
            trace: r.f32()?,
            active_samples: r.u64()?,
            last_marked_sample: r.u64()?,
            exc_drive_ema: if version >= 5 { r.f32()? } else { T::ZERO },
            inh_drive_ema: if version >= 5 { r.f32()? } else { T::ZERO },
        };
        let meta = r_meta(r)?;
        t.neurons.insert(
            id,
            Neuron {
                id,
                role,
                polarity,
                model,
                state,
                modulation,
                stats,
                meta,
            },
        );
    }
    for _ in 0..r.count()? {
        let id = SynapseId(r.u64()?);
        let (src, dst) = (NeuronId(r.u64()?), NeuronId(r.u64()?));
        let weight = r.f32()?;
        let delay = r.u16()?;
        let (tag, since) = (r.u8()?, r.u64()?);
        let state = match tag {
            0 => SynapseState::Active,
            1 => SynapseState::Disabled { since_step: since },
            v => return Err(err(format!("unknown state of synapse {v}"))),
        };
        let stats = SynapseStats {
            uses: r.u64()?,
            drive_sum: r.f32()?,
            active_samples: r.u64()?,
            last_marked_sample: r.u64()?,
            credit_ema: r.f32()?,
            activity_trace: r.f32()?,
            trace_step: r.u64()?,
        };
        let (elig, elig_step) = (r.f32()?, r.u64()?);
        let meta = r_meta(r)?;
        t.synapses.insert(
            id,
            Synapse {
                id,
                src,
                dst,
                weight,
                delay,
                state,
                stats,
                elig,
                elig_step,
                meta,
            },
        );
    }
    t.rebuild_adjacency();
    t.validate()
        .map_err(|e| err(format!("topology damaged: {e}")))?;
    Ok(t)
}

// ---------------- network ----------------

fn w_network<T: Scalar>(w: &mut W, n: &DynamicNetwork<T>, version: u32) {
    let s = &n.cfg.stdp;
    w.u8(match s.mode {
        PlasticityMode::Off => 0,
        PlasticityMode::Immediate => 1,
        PlasticityMode::RewardModulated => 2,
    });
    w.f32(s.a_plus);
    w.f32(s.a_minus);
    w.f32(s.trace_decay);
    match s.bound {
        WeightBound::Hard { w_min, w_max } => {
            w.u8(0);
            w.f32(w_min);
            w.f32(w_max);
        }
        WeightBound::Soft { w_min, w_max } => {
            w.u8(1);
            w.f32(w_min);
            w.f32(w_max);
        }
        WeightBound::None => {
            w.u8(2);
            w.f32(0.0);
            w.f32(0.0);
        }
    }
    w.f32(s.elig_decay);
    w.f32(s.reward_lr);
    w.f32(n.cfg.rate_alpha);
    w.f32(n.cfg.credit_alpha);
    w.f32(n.cfg.activity_trace_decay);
    // Version 2: modulating gain attenuation.
    if version >= 2 {
        w.f32(n.cfg.modulation_decay);
    }
    // Version 3: input neuron mode.
    if version >= 3 {
        w.u8(n.cfg.input_mode.code());
    }
    // Version 5: Inhibitory plasticity and E/I balance.
    if version >= 5 {
        let (i, b) = (&n.cfg.inhibitory, &n.cfg.balance);
        w.bool(i.enabled);
        w.f32(i.eta);
        w.f32(i.target_rate);
        w.bool(b.enabled);
        w.f32(b.target_ratio);
        w.f32(b.rate);
        w.f32(b.max_step);
        w.f32(b.min_drive);
        w.u64(b.min_observed_steps);
    }
    w.u64(n.step);
    w.u64(n.sample);
    w.bool(n.plasticity_enabled);
    w.usize(n.history.len());
    for f in &n.history {
        w.usize(f.len());
        for id in f {
            w.u64(id.0);
        }
    }
    w.bool(n.tracker.is_some());
    if let Some(t) = &n.tracker {
        w.u64(t.window);
        w.usize(t.capacity);
        w.usize(t.max_recent);
        w.usize(t.pairs.len());
        for ((a, b), v) in &t.pairs {
            w.u64(a.0);
            w.u64(b.0);
            w.f32(*v);
        }
        w.usize(t.recent.len());
        for (id, st) in &t.recent {
            w.u64(id.0);
            w.u64(*st);
        }
    }
}

fn r_network<T: Scalar>(r: &mut R, topo: Topology<T>, version: u32) -> Res<DynamicNetwork<T>> {
    let mode = match r.u8()? {
        0 => PlasticityMode::Off,
        1 => PlasticityMode::Immediate,
        2 => PlasticityMode::RewardModulated,
        v => return Err(err(format!("unknown plasticity mode {v}"))),
    };
    let (a_plus, a_minus, trace_decay) = (r.f32()?, r.f32()?, r.f32()?);
    let (tag, lo, hi) = (r.u8()?, r.f32()?, r.f32()?);
    let bound = match tag {
        0 => WeightBound::Hard {
            w_min: lo,
            w_max: hi,
        },
        1 => WeightBound::Soft {
            w_min: lo,
            w_max: hi,
        },
        2 => WeightBound::None,
        v => return Err(err(format!("unknown weight limit {v}"))),
    };
    let stdp = StdpConfig {
        mode,
        a_plus,
        a_minus,
        trace_decay,
        bound,
        elig_decay: r.f32()?,
        reward_lr: r.f32()?,
    };
    let cfg = EngineConfig::<T> {
        stdp,
        rate_alpha: r.f32()?,
        credit_alpha: r.f32()?,
        activity_trace_decay: r.f32()?,
        modulation_decay: if version >= 2 { r.f32()? } else { T::c(0.9) },
        input_mode: if version >= 3 {
            InputMode::from_code(r.u8()?).ok_or_else(|| err("unknown mode of input neurons"))?
        } else {
            InputMode::Spike
        },
        ..EngineConfig::<T>::default()
    };
    let mut cfg = cfg;
    if version >= 5 {
        cfg.inhibitory = InhibitoryPlasticityConfig {
            enabled: r.bool()?,
            eta: r.f32()?,
            target_rate: r.f32()?,
        };
        cfg.balance = EiBalanceConfig {
            enabled: r.bool()?,
            target_ratio: r.f32()?,
            rate: r.f32()?,
            max_step: r.f32()?,
            min_drive: r.f32()?,
            min_observed_steps: r.u64()?,
        };
    }
    let mut net = DynamicNetwork::new(topo, cfg)?;
    net.step = r.u64()?;
    net.sample = r.u64()?;
    net.plasticity_enabled = r.bool()?;
    for _ in 0..r.count()? {
        let mut frame = Vec::new();
        for _ in 0..r.count()? {
            frame.push(NeuronId(r.u64()?));
        }
        net.history.push_back(frame);
    }
    if r.bool()? {
        let (window, capacity, max_recent) = (r.u64()?, r.usize()?, r.usize()?);
        let mut t = CoactivityTracker::new(window, capacity, max_recent);
        for _ in 0..r.count()? {
            let (a, b, v) = (NeuronId(r.u64()?), NeuronId(r.u64()?), r.f32()?);
            t.pairs.insert((a, b), v);
        }
        for _ in 0..r.count()? {
            t.recent.push((NeuronId(r.u64()?), r.u64()?));
        }
        net.tracker = Some(t);
    }
    Ok(net)
}

// ---------------- controller ----------------

fn w_event(w: &mut W, e: &StructuralEvent) {
    w.u64(e.seq);
    w.u64(e.step);
    w.u64(e.sample);
    w.u8(kind_code(e.kind));
    w.str(&e.action);
    w.str(&e.reason);
    w.u8(e.outcome.code());
    if let Outcome::Failed(m) = &e.outcome {
        w.str(m);
    }
    w.opt_f32(e.loss_before);
    w.opt_f32(e.loss_after);
    w.usize(e.neurons_before);
    w.usize(e.neurons_after);
    w.usize(e.synapses_before);
    w.usize(e.synapses_after);
}

fn r_event(r: &mut R) -> Res<StructuralEvent> {
    let (seq, step, sample) = (r.u64()?, r.u64()?, r.u64()?);
    let kind = kind_from(r.u8()?)?;
    let (action, reason) = (r.str()?, r.str()?);
    let outcome = match r.u8()? {
        0 => Outcome::Applied,
        1 => Outcome::RolledBack,
        2 => Outcome::Failed(r.str()?),
        v => return Err(err(format!("unknown outcome {v}"))),
    };
    Ok(StructuralEvent {
        seq,
        step,
        sample,
        kind,
        action,
        reason,
        outcome,
        loss_before: r.opt_f32()?,
        loss_after: r.opt_f32()?,
        neurons_before: r.usize()?,
        neurons_after: r.usize()?,
        synapses_before: r.usize()?,
        synapses_after: r.usize()?,
    })
}

fn w_controller<T: Scalar>(w: &mut W, c: &StructuralPlasticityController<T>, version: u32) {
    w.u64(c.rng.state());
    w.usize(c.log.capacity);
    w.u64(c.log.next_seq);
    w.usize(c.log.totals.len());
    for (k, v) in &c.log.totals {
        w.u8(kind_code(*k));
        w.u64(v.applied);
        w.u64(v.rolled_back);
        w.u64(v.failed);
    }
    w.usize(c.log.events.len());
    for e in &c.log.events {
        w_event(w, e);
    }
    w.usize(c.losses.len());
    for l in &c.losses {
        w.f32(*l);
    }
    w.f32(c.best_loss);
    w.usize(c.since_improvement);
    w.u64(c.calls);
    w.opt_u64(c.last_change_step);
    w.usize(c.cooldown.len());
    for (k, v) in &c.cooldown {
        w.u8(kind_code(*k));
        w.u64(*v);
    }
    w.usize(c.streak.len());
    for (k, v) in &c.streak {
        w.u8(kind_code(*k));
        w.u32(*v);
    }
    w.bool(c.pending.is_some());
    if let Some(p) = &c.pending {
        w_topology(w, &p.backup, version);
        w.f32(p.loss_before);
        w.f32(p.cost_before);
        w.u64(p.calls_left);
        w.usize(p.seqs.len());
        for (s, k) in &p.seqs {
            w.u64(*s);
            w.u8(kind_code(*k));
        }
    }
    w.bool(c.best.is_some());
    if let Some(b) = &c.best {
        w_topology(w, &b.topology, version);
        w.f32(b.cost);
        w.f32(b.loss);
        w.u64(b.step);
    }
}

fn r_controller<T: Scalar>(
    r: &mut R,
    cfg: StructuralConfig<T>,
    version: u32,
) -> Res<StructuralPlasticityController<T>> {
    let mut c = StructuralPlasticityController::new(cfg)?;
    c.rng = Rng64::from_state(r.u64()?);
    let capacity = r.usize()?;
    let next_seq = r.u64()?;
    let mut totals = BTreeMap::new();
    for _ in 0..r.count()? {
        let k = kind_from(r.u8()?)?;
        totals.insert(
            k,
            OutcomeCounts {
                applied: r.u64()?,
                rolled_back: r.u64()?,
                failed: r.u64()?,
            },
        );
    }
    let mut events = VecDeque::new();
    for _ in 0..r.count()? {
        events.push_back(r_event(r)?);
    }
    c.log = EventLog {
        events,
        capacity: capacity.max(1),
        next_seq,
        totals,
    };
    for _ in 0..r.count()? {
        c.losses.push_back(r.f32()?);
    }
    c.best_loss = r.f32()?;
    c.since_improvement = r.usize()?;
    c.calls = r.u64()?;
    c.last_change_step = r.opt_u64()?;
    for _ in 0..r.count()? {
        let k = kind_from(r.u8()?)?;
        c.cooldown.insert(k, r.u64()?);
    }
    for _ in 0..r.count()? {
        let k = kind_from(r.u8()?)?;
        c.streak.insert(k, r.u32()?);
    }
    if r.bool()? {
        let backup = r_topology(r, version)?;
        let (loss_before, cost_before, calls_left) = (r.f32()?, r.f32()?, r.u64()?);
        let mut seqs = Vec::new();
        for _ in 0..r.count()? {
            seqs.push((r.u64()?, kind_from(r.u8()?)?));
        }
        c.pending = Some(Pending {
            backup,
            loss_before,
            cost_before,
            calls_left,
            seqs,
        });
    }
    if r.bool()? {
        let topology = r_topology(r, version)?;
        c.best = Some(BestSnapshot {
            topology,
            cost: r.f32()?,
            loss: r.f32()?,
            step: r.u64()?,
        });
    }
    Ok(c)
}

// ---------------- public API ----------------

/// Serialize the network and controller (and the optional `extra` block).
///
/// For `f32`, the version [`FORMAT_VERSION`] is written (the format is the same, read by the old code);
/// for `f64` - [`FORMAT_VERSION_F64`] with values of 8 bytes, without loss of precision.
pub fn save_checkpoint<T: Scalar>(
    net: &DynamicNetwork<T>,
    ctl: &StructuralPlasticityController<T>,
    extra: &[u8],
) -> Vec<u8> {
    let wide = std::mem::size_of::<T>() > 4;
    let version = if wide {
        FORMAT_VERSION_F64
    } else {
        FORMAT_VERSION
    };
    let mut w = W(Vec::new(), wide);
    w.0.extend_from_slice(MAGIC);
    w.u32(version);
    if wide {
        w.u8(8);
    }
    w_config(&mut w, ctl.config(), version);
    w_topology(&mut w, net.topology(), version);
    w_network(&mut w, net, version);
    w_controller(&mut w, ctl, version);
    w.bytes(extra);
    let sum = fnv1a(&w.0);
    w.u64(sum);
    w.0
}

/// Restore state from bytes. Error with invalid signature, more
/// new version of the format, data corruption or violation of topology invariants.
///
/// Reads in `f32`; values from file `f64` are given with loss of precision.
/// For other scalars - [`load_checkpoint_as`].
pub fn load_checkpoint(bytes: &[u8]) -> StructuralResult<Checkpoint<f32>> {
    load_checkpoint_as::<f32>(bytes)
}

/// Same as [`load_checkpoint`], but into the selected scalar `T`. File `f32` (versions 1–5)
/// reads in `f64` without loss; the `f64` file (version 6) is read in `f32` with rounding.
pub fn load_checkpoint_as<T: Scalar>(bytes: &[u8]) -> StructuralResult<Checkpoint<T>> {
    if bytes.len() < 4 + 4 + 8 || &bytes[..4] != MAGIC {
        return Err(err("invalid file signature"));
    }
    let (body, tail) = bytes.split_at(bytes.len() - 8);
    let mut sum = [0u8; 8];
    sum.copy_from_slice(tail);
    if fnv1a(body) != u64::from_le_bytes(sum) {
        return Err(err("checksum does not match: data is corrupted"));
    }
    let mut r = R {
        b: body,
        p: 4,
        wide: false,
    };
    let version = r.u32()?;
    if version == 0 || version > FORMAT_VERSION_F64 {
        return Err(err(format!(
            "format version {version} is not supported (maximum {FORMAT_VERSION_F64})"
        )));
    }
    if version >= FORMAT_VERSION_F64 {
        r.wide = match r.u8()? {
            4 => false,
            8 => true,
            x => return Err(err(format!("value width {x} is not supported"))),
        };
    }
    let cfg = r_config(&mut r, version)?;
    let topo = r_topology(&mut r, version)?;
    let network = r_network(&mut r, topo, version)?;
    let controller = r_controller(&mut r, cfg, version)?;
    let extra = r.bytes()?;
    if r.p != body.len() {
        return Err(err("extra data at the end of the file"));
    }
    Ok(Checkpoint {
        network,
        controller,
        extra,
    })
}

/// Write the checkpoint to a file (via a temporary file and renaming).
pub fn save_to_file<T: Scalar>(
    path: impl AsRef<Path>,
    net: &DynamicNetwork<T>,
    ctl: &StructuralPlasticityController<T>,
    extra: &[u8],
) -> StructuralResult<()> {
    let path = path.as_ref();
    let tmp = path.with_extension("tmp");
    std::fs::write(&tmp, save_checkpoint(net, ctl, extra))
        .map_err(|e| err(format!("entry {}: {e}", tmp.display())))?;
    std::fs::rename(&tmp, path).map_err(|e| err(format!("renaming to {}: {e}", path.display())))
}

/// Read checkpoint from file.
pub fn load_from_file(path: impl AsRef<Path>) -> StructuralResult<Checkpoint<f32>> {
    load_from_file_as::<f32>(path)
}

/// Read checkpoint from file into selected scalar `T`.
pub fn load_from_file_as<T: Scalar>(path: impl AsRef<Path>) -> StructuralResult<Checkpoint<T>> {
    let path = path.as_ref();
    let bytes = std::fs::read(path).map_err(|e| err(format!("reading {}: {e}", path.display())))?;
    load_checkpoint_as(&bytes)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::state::ResetMode;
    use crate::structural::controller::Metrics;
    use crate::structural::topology::Birth;

    fn sample_topology(cfg: &StructuralConfig) -> Topology {
        let mut t = Topology::new(cfg.topology, cfg.params);
        let b = Birth::default();
        let i = t
            .add_neuron(NeuronRole::Input, LifParams::default(), Origin::Initial, b)
            .unwrap();
        let h = t
            .add_neuron_with(
                NeuronRole::Hidden,
                NeuronPolarity::Inhibitory,
                AdaptiveLifParams::slow_adaptation(),
                Origin::Initial,
                b,
            )
            .unwrap();
        let o = t
            .add_neuron_with(
                NeuronRole::Output,
                NeuronPolarity::Modulatory,
                IzhikevichParams::fast_spiking(),
                Origin::Initial,
                b,
            )
            .unwrap();
        t.add_synapse(i, h, 0.7, 1, Origin::Initial, b).unwrap();
        t.add_synapse(h, o, -0.4, 0, Origin::Initial, b).unwrap();
        // Non-empty model states and accumulated modulation.
        t.neurons.get_mut(&h).unwrap().state = NeuronState::AdaptiveLif(AdaptiveLifState {
            mem: 0.35,
            adaptation: 0.2,
        });
        t.neurons.get_mut(&o).unwrap().state =
            NeuronState::Izhikevich(IzhikevichState { v: -60.0, u: 1.5 });
        t.neurons.get_mut(&o).unwrap().modulation = 0.25;
        t
    }

    fn encode(
        version: u32,
        cfg: &StructuralConfig,
        net: &DynamicNetwork,
        ctl: &StructuralPlasticityController,
    ) -> Vec<u8> {
        let mut w = W(Vec::new(), false);
        w.0.extend_from_slice(MAGIC);
        w.u32(version);
        w_config(&mut w, cfg, version);
        w_topology(&mut w, net.topology(), version);
        w_network(&mut w, net, version);
        w_controller(&mut w, ctl, version);
        w.bytes(b"extra");
        let sum = fnv1a(&w.0);
        w.u64(sum);
        w.0
    }

    #[test]
    fn heterogeneous_roundtrip_keeps_model_polarity_and_state() {
        let cfg = StructuralConfig::default();
        let t = sample_topology(&cfg);
        let mut net = DynamicNetwork::new(t, EngineConfig::default()).unwrap();
        net.step(&[1.0]).unwrap();
        // The modulation fades at each step, so we check the actual
        // value in the network at the time of saving, and not with the original 0.25.
        let expected_mod = net
            .topology()
            .neurons()
            .find(|n| n.role == NeuronRole::Output)
            .unwrap()
            .modulation;
        assert!(expected_mod > 0.0);
        let ctl = StructuralPlasticityController::new(cfg.clone()).unwrap();
        let c = load_checkpoint(&save_checkpoint(&net, &ctl, b"x")).unwrap();
        assert_eq!(*c.network.topology(), *net.topology());
        let h = c
            .network
            .topology()
            .neurons()
            .find(|n| n.role == NeuronRole::Hidden)
            .unwrap();
        assert_eq!(h.polarity, NeuronPolarity::Inhibitory);
        assert!(matches!(h.model, NeuronModel::AdaptiveLif(_)));
        assert!(matches!(h.state, NeuronState::AdaptiveLif(_)));
        let o = c
            .network
            .topology()
            .neurons()
            .find(|n| n.role == NeuronRole::Output)
            .unwrap();
        assert_eq!(o.polarity, NeuronPolarity::Modulatory);
        assert!(matches!(o.model, NeuronModel::Izhikevich(_)));
        assert!(matches!(o.state, NeuronState::Izhikevich(_)));
        assert_eq!(o.modulation, expected_mod);
    }

    #[test]
    fn current_version_roundtrip_keeps_input_mode_model_selection_and_param_learning() {
        let mut cfg = StructuralConfig::default();
        cfg.model_selection.enabled = true;
        cfg.model_selection.min_observed_steps = 17;
        cfg.model_selection.migration = StateMigration::Membrane;
        cfg.model_selection.loss_evaluation = false;
        cfg.model_selection.loss_min_improvement = 0.003;
        cfg.model_selection.max_probes_per_interval = 7;
        cfg.model_selection.candidates = vec![
            ModelCandidate::new(
                NeuronModel::AdaptiveLif(AdaptiveLifParams::slow_adaptation()),
                0.2,
            )
            .with_target_rate(0.02, 0.01),
        ];
        cfg.param_learning.enabled = true;
        cfg.param_learning.min_observed_steps = 33;
        cfg.param_learning.max_neurons_per_interval = 2;
        cfg.param_learning.max_params_per_neuron = 3;
        cfg.param_learning.max_probes_per_interval = 9;
        cfg.param_learning.step = 0.07;
        cfg.param_learning.min_step = 0.002;
        cfg.param_learning.min_improvement = 0.004;
        cfg.param_learning.include_inputs = true;
        cfg.param_learning.include_outputs = false;
        let t = sample_topology(&cfg);
        let mut engine = EngineConfig::default();
        engine.input_mode = InputMode::Current;
        let net = DynamicNetwork::new(t, engine).unwrap();
        let ctl = StructuralPlasticityController::new(cfg.clone()).unwrap();
        let bytes = save_checkpoint(&net, &ctl, b"v4");
        assert_eq!(&bytes[4..8], &FORMAT_VERSION.to_le_bytes());
        let c = load_checkpoint(&bytes).unwrap();
        assert_eq!(c.network.config().input_mode, InputMode::Current);
        assert_eq!(c.controller.config().model_selection, cfg.model_selection);
        assert_eq!(c.controller.config().param_learning, cfg.param_learning);
        assert_eq!(c.extra, b"v4");
    }

    #[test]
    fn version_four_checkpoint_loads_in_legacy_dale_mode_and_can_be_migrated() {
        let mut cfg = StructuralConfig::default();
        cfg.topology.enforce_dale = false;
        cfg.param_learning.enabled = true;
        cfg.param_learning.min_observed_steps = 33;
        let mut t = sample_topology(&cfg);
        // Legacy: an excitatory source with a negative weight.
        let (src, dst) = {
            let s = t.synapses().next().unwrap();
            (s.id, s.src)
        };
        t.set_synapse_weight(src, -0.4).unwrap();
        let _ = dst;
        let net = DynamicNetwork::new(t, EngineConfig::default()).unwrap();
        let ctl = StructuralPlasticityController::new(cfg.clone()).unwrap();
        let bytes = encode(4, &cfg, &net, &ctl);
        assert_eq!(&bytes[4..8], &4u32.to_le_bytes());
        let mut c = load_checkpoint(&bytes).unwrap();
        // Fields v4 are read, v5 are default; Dale's principle is turned off.
        assert_eq!(c.controller.config().param_learning.min_observed_steps, 33);
        assert!(!c.network.topology().dale());
        assert!(!c.controller.config().topology.enforce_dale);
        assert_eq!(
            c.network.config().inhibitory,
            InhibitoryPlasticityConfig::default()
        );
        assert_eq!(c.network.topology().synapse(src).unwrap().weight, -0.4);
        c.network.topology().validate().unwrap();
        // Explicit migration procedure to E/I mode.
        let changed = c.enable_dale();
        assert!(changed >= 1);
        assert!(c.network.topology().dale());
        assert!(c.controller.config().topology.enforce_dale);
        c.network.topology().validate().unwrap();
        assert_eq!(c.network.topology().synapse(src).unwrap().weight, 0.4);
        assert_eq!(c.enable_dale(), 0, "re-migration doesn't change anything");
        // After migration, the checkpoint is saved in strict mode.
        let again = load_checkpoint(&save_checkpoint(&c.network, &c.controller, b"")).unwrap();
        assert!(again.network.topology().dale());
        assert_eq!(
            &save_checkpoint(&again.network, &again.controller, b"")[4..8],
            &FORMAT_VERSION.to_le_bytes()
        );
    }

    #[test]
    fn version_three_checkpoint_defaults_to_loss_evaluation_and_no_param_learning() {
        let mut cfg = StructuralConfig::default();
        cfg.model_selection.enabled = true;
        cfg.model_selection.min_observed_steps = 21;
        let t = sample_topology(&cfg);
        let net = DynamicNetwork::new(t, EngineConfig::default()).unwrap();
        let ctl = StructuralPlasticityController::new(cfg.clone()).unwrap();
        let bytes = encode(3, &cfg, &net, &ctl);
        let c = load_checkpoint(&bytes).unwrap();
        let m = &c.controller.config().model_selection;
        assert!(m.enabled);
        assert_eq!(m.min_observed_steps, 21);
        // The new v4 fields are not present in v3 - default values are taken.
        let d = ModelSelectionConfig::<f32>::default();
        assert_eq!(m.loss_evaluation, d.loss_evaluation);
        assert_eq!(m.loss_min_improvement, d.loss_min_improvement);
        assert_eq!(m.max_probes_per_interval, d.max_probes_per_interval);
        assert_eq!(
            c.controller.config().param_learning,
            ParamLearningConfig::default()
        );
        assert!(!c.controller.config().param_learning.enabled);
    }

    #[test]
    fn version_two_checkpoint_defaults_to_spike_inputs_and_no_selection() {
        let cfg = StructuralConfig::default();
        let t = sample_topology(&cfg);
        let net = DynamicNetwork::new(t, EngineConfig::default()).unwrap();
        let ctl = StructuralPlasticityController::new(cfg.clone()).unwrap();
        let bytes = encode(2, &cfg, &net, &ctl);
        let c = load_checkpoint(&bytes).unwrap();
        assert_eq!(c.network.config().input_mode, InputMode::Spike);
        assert_eq!(
            c.controller.config().model_selection,
            ModelSelectionConfig::default()
        );
        assert!(!c.controller.config().model_selection.enabled);
    }

    #[test]
    fn version_one_checkpoint_migrates_to_lif_excitatory() {
        let cfg = StructuralConfig::default();
        let mut t = Topology::new(cfg.topology, cfg.params);
        let b = Birth::default();
        let i = t
            .add_neuron(NeuronRole::Input, LifParams::default(), Origin::Initial, b)
            .unwrap();
        let h = t
            .add_neuron(
                NeuronRole::Hidden,
                LifParams {
                    beta: 0.8,
                    threshold: 1.2,
                    reset: ResetMode::Subtract,
                },
                Origin::Initial,
                b,
            )
            .unwrap();
        let o = t
            .add_neuron(NeuronRole::Output, LifParams::default(), Origin::Initial, b)
            .unwrap();
        t.add_synapse(i, h, 0.5, 0, Origin::Initial, b).unwrap();
        t.add_synapse(h, o, 0.9, 0, Origin::Initial, b).unwrap();
        t.neurons.get_mut(&h).unwrap().state = NeuronState::Lif(LifState { mem: 0.45 });
        let net = DynamicNetwork::new(t, EngineConfig::default()).unwrap();
        let ctl = StructuralPlasticityController::new(cfg.clone()).unwrap();
        let bytes = encode(1, &cfg, &net, &ctl);
        let c = load_checkpoint(&bytes).unwrap();
        let h2 = c
            .network
            .topology()
            .neurons()
            .find(|n| n.role == NeuronRole::Hidden)
            .unwrap();
        assert_eq!(h2.polarity, NeuronPolarity::Excitatory);
        assert!(matches!(h2.model, NeuronModel::Lif(p) if p.beta == 0.8));
        assert!(matches!(h2.state, NeuronState::Lif(s) if s.mem == 0.45));
        assert_eq!(h2.modulation, 0.0);
        assert_eq!(
            c.network.config().modulation_decay,
            EngineConfig::<f32>::default().modulation_decay
        );
        assert_eq!(c.extra, b"extra");
    }

    #[test]
    fn migrated_network_continues_with_model_dynamics() {
        let cfg = StructuralConfig::default();
        let t = sample_topology(&cfg);
        let net = DynamicNetwork::new(t, EngineConfig::default()).unwrap();
        let mut ctl = StructuralPlasticityController::new(cfg.clone()).unwrap();
        let mut c = load_checkpoint(&save_checkpoint(&net, &ctl, b"")).unwrap();
        // One step should not panic and should leave the states finite.
        c.network.step(&[1.0]).unwrap();
        c.network.topology().validate().unwrap();
        let _ = ctl.adapt(&mut c.network, Metrics::loss(1.0), None);
    }
}
