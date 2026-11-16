//! Structure statistics, cost estimation and joint tracking
//! neuron activity (a source of candidates for the growth of connections).

use std::collections::BTreeMap;

use crate::scalar::Scalar;

use super::topology::{Neuron, NeuronId, NeuronRole, Synapse, Topology};

/// Numeric type in which statistical indicators are displayed
/// ([`EiStats`]).
///
/// Calculations always go to `f64` (accumulation of sums and averages), and the target type
/// gets a ready-made result via [`from_f64`](Self::from_f64), so
/// narrow types do not suffer from loss of precision during summation. Implemented for
/// `f32` and `f64`; for `f16`/`bf16` you can implement the trait yourself
/// (`from_f64` - rounding of the result, `to_f64` - expansion), separate
/// The library does not require any dependency on `half`.
pub trait StatsScalar: Copy {
    /// Round the calculated value to this type.
    fn from_f64(v: f64) -> Self;
    /// Expand the value to `f64`.
    fn to_f64(self) -> f64;
}

impl StatsScalar for f32 {
    fn from_f64(v: f64) -> Self {
        v as f32
    }
    fn to_f64(self) -> f64 {
        f64::from(self)
    }
}

impl StatsScalar for f64 {
    fn from_f64(v: f64) -> Self {
        v
    }
    fn to_f64(self) -> f64 {
        self
    }
}

/// E/I activity and connectivity statistics (see [`Topology::ei_stats`]).
///
/// The type `T` is only a representation of the result (by default `f32`, so
/// the previous code `EiStats` does not change); statistics are calculated in `f64`
/// ([`Topology::ei_stats_as`]).
///
/// Frequencies and input currents are averaged across non-input neurons (no inputs).
/// "Excitatory" synapses - emanating from excitatory neurons,
/// “Inhibitory” synapses - those from inhibitory neurons;
/// modulating ones are counted separately.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct EiStats<T = f32> {
    /// Excitatory neurons.
    pub excitatory: usize,
    /// Inhibitory neurons.
    pub inhibitory: usize,
    /// Modulating neurons.
    pub modulatory: usize,
    /// The share of inhibitory ones among excitatory and inhibitory ones (`0` in their absence).
    pub inhibitory_fraction: T,
    /// Active synapses from excitatory neurons.
    pub excitatory_synapses: usize,
    /// Active synapses from inhibitory neurons.
    pub inhibitory_synapses: usize,
    /// Average modulus of the weight of excitatory synapses.
    pub mean_excitatory_weight: T,
    /// Average weight modulus of inhibitory synapses.
    pub mean_inhibitory_weight: T,
    /// Average spike frequency of excitatory neurons.
    pub excitatory_rate: T,
    /// Average spike frequency of inhibitory neurons.
    pub inhibitory_rate: T,
    /// Average excitatory input current of neurons.
    pub excitatory_drive: T,
    /// Average inhibitory input current of neurons (modulus).
    pub inhibitory_drive: T,
    /// Balance: `inhibitory_drive / excitatory_drive` (`0` if there is no current).
    pub ei_ratio: T,
    /// Violations of the Dale principle (synapses with a “foreign” sign); when turned on
    /// in principle always `0`.
    pub dale_violations: usize,
}

impl<T: StatsScalar> EiStats<T> {
    /// Convert indicators to another numeric type (via `f64`).
    pub fn cast<U: StatsScalar>(&self) -> EiStats<U> {
        let c = |v: T| U::from_f64(v.to_f64());
        EiStats {
            excitatory: self.excitatory,
            inhibitory: self.inhibitory,
            modulatory: self.modulatory,
            inhibitory_fraction: c(self.inhibitory_fraction),
            excitatory_synapses: self.excitatory_synapses,
            inhibitory_synapses: self.inhibitory_synapses,
            mean_excitatory_weight: c(self.mean_excitatory_weight),
            mean_inhibitory_weight: c(self.mean_inhibitory_weight),
            excitatory_rate: c(self.excitatory_rate),
            inhibitory_rate: c(self.inhibitory_rate),
            excitatory_drive: c(self.excitatory_drive),
            inhibitory_drive: c(self.inhibitory_drive),
            ei_ratio: c(self.ei_ratio),
            dale_violations: self.dale_violations,
        }
    }
}

impl<T: Scalar> Topology<T> {
    /// E/I activity statistics in `f32` (the `ei_stats_as::<f32>` specialization).
    pub fn ei_stats(&self) -> EiStats<f32> {
        self.ei_stats_as::<f32>()
    }

    /// Statistics of E/I activity in the selected numeric type. Amounts and averages
    /// are accumulated in `f64`, in `T` only the total is rounded.
    pub fn ei_stats_as<S: StatsScalar>(&self) -> EiStats<S> {
        use super::topology::NeuronPolarity as P;
        let mut s = EiStats::<f64>::default();
        let (mut e_rate, mut i_rate) = (0.0f64, 0.0f64);
        let (mut e_cnt, mut i_cnt) = (0usize, 0usize);
        let (mut e_drive, mut i_drive, mut drive_cnt) = (0.0f64, 0.0f64, 0usize);
        for n in self.neurons() {
            match n.polarity {
                P::Excitatory => s.excitatory += 1,
                P::Inhibitory => s.inhibitory += 1,
                P::Modulatory => s.modulatory += 1,
            }
            if n.role == NeuronRole::Input {
                continue;
            }
            match n.polarity {
                P::Excitatory => {
                    e_rate += n.stats.rate_ema.to_f64();
                    e_cnt += 1;
                }
                P::Inhibitory => {
                    i_rate += n.stats.rate_ema.to_f64();
                    i_cnt += 1;
                }
                P::Modulatory => {}
            }
            e_drive += n.stats.exc_drive_ema.to_f64();
            i_drive += n.stats.inh_drive_ema.to_f64();
            drive_cnt += 1;
        }
        let (mut e_w, mut i_w) = (0.0f64, 0.0f64);
        for syn in self.synapses().filter(|x| x.is_active()) {
            let pol = self.polarity_of(syn.src);
            match pol {
                P::Excitatory => {
                    s.excitatory_synapses += 1;
                    e_w += syn.weight.abs().to_f64();
                }
                P::Inhibitory => {
                    s.inhibitory_synapses += 1;
                    i_w += syn.weight.abs().to_f64();
                }
                P::Modulatory => {}
            }
            let ok = if pol == P::Inhibitory {
                syn.weight <= T::ZERO
            } else {
                syn.weight >= T::ZERO
            };
            if !ok {
                s.dale_violations += 1;
            }
        }
        let mean = |sum: f64, n: usize| if n == 0 { 0.0 } else { sum / n as f64 };
        s.inhibitory_fraction = mean(s.inhibitory as f64, s.excitatory + s.inhibitory);
        s.mean_excitatory_weight = mean(e_w, s.excitatory_synapses);
        s.mean_inhibitory_weight = mean(i_w, s.inhibitory_synapses);
        s.excitatory_rate = mean(e_rate, e_cnt);
        s.inhibitory_rate = mean(i_rate, i_cnt);
        s.excitatory_drive = mean(e_drive, drive_cnt);
        s.inhibitory_drive = mean(i_drive, drive_cnt);
        s.ei_ratio = if s.excitatory_drive > 0.0 {
            s.inhibitory_drive / s.excitatory_drive
        } else {
            0.0
        };
        s.cast::<S>()
    }
}

/// Snapshot of network structural indicators.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct StructuralStats<T = f32> {
    /// Global step.
    pub step: u64,
    /// Sample number.
    pub sample: u64,
    /// Total neurons.
    pub neurons: usize,
    /// Input.
    pub inputs: usize,
    /// Hidden.
    pub hidden: usize,
    /// Weekends.
    pub outputs: usize,
    /// Total synapses.
    pub synapses: usize,
    /// Active synapses.
    pub active_synapses: usize,
    /// Disabled synapses.
    pub disabled_synapses: usize,
    /// "Working" neurons: inputs and neurons that have at least one
    /// active incident synapse.
    pub active_neurons: usize,
    /// Average spike frequency across non-input neurons.
    pub mean_rate: T,
    /// Activity costs: the sum of the spike frequencies of all non-input neurons.
    pub activity_cost: T,
    /// Non-input neurons with a frequency below the “silence” threshold.
    pub silent_neurons: usize,
    /// Non-input neurons with a frequency above the “overload” threshold.
    pub overactive_neurons: usize,
    /// Average `|w|` of active synapses.
    pub mean_abs_weight: T,
    /// Density `S / N²`.
    pub density: T,
    /// Maximum incoming degree.
    pub max_in_degree: usize,
    /// Maximum outgoing degree.
    pub max_out_degree: usize,
    /// Estimated memory occupied by structure and state, in bytes.
    pub state_bytes: usize,
}

impl<T: Scalar> StructuralStats<T> {
    /// Calculate the indicators.
    pub fn compute(
        topo: &Topology<T>,
        step: u64,
        sample: u64,
        silent_rate: T,
        overactive_rate: T,
    ) -> Self {
        let mut s = StructuralStats {
            step,
            sample,
            ..Default::default()
        };
        s.neurons = topo.neuron_count();
        s.synapses = topo.synapse_count();
        let mut rate_sum = T::ZERO;
        let mut non_input = 0usize;
        for n in topo.neurons() {
            match n.role {
                NeuronRole::Input => s.inputs += 1,
                NeuronRole::Hidden => s.hidden += 1,
                NeuronRole::Output => s.outputs += 1,
            }
            let has_active = topo
                .in_synapses(n.id)
                .iter()
                .chain(topo.out_synapses(n.id))
                .any(|sid| topo.synapse(*sid).is_some_and(|x| x.is_active()));
            if n.role == NeuronRole::Input || has_active {
                s.active_neurons += 1;
            }
            s.max_in_degree = s.max_in_degree.max(topo.in_synapses(n.id).len());
            s.max_out_degree = s.max_out_degree.max(topo.out_synapses(n.id).len());
            if n.role != NeuronRole::Input {
                non_input += 1;
                let rate = n.stats.rate_ema;
                rate_sum += rate;
                if n.stats.steps > 0 && rate < silent_rate {
                    s.silent_neurons += 1;
                }
                if rate > overactive_rate {
                    s.overactive_neurons += 1;
                }
            }
        }
        let mut wsum = T::ZERO;
        for y in topo.synapses() {
            if y.is_active() {
                s.active_synapses += 1;
                wsum += y.weight.abs();
            } else {
                s.disabled_synapses += 1;
            }
        }
        s.activity_cost = rate_sum;
        s.mean_rate = if non_input > 0 {
            rate_sum / T::from_usize(non_input)
        } else {
            T::ZERO
        };
        s.mean_abs_weight = if s.active_synapses > 0 {
            wsum / T::from_usize(s.active_synapses)
        } else {
            T::ZERO
        };
        let n2 = T::from_usize((s.neurons * s.neurons).max(1));
        s.density = T::from_usize(s.synapses) / n2;
        s.state_bytes = s.neurons * (std::mem::size_of::<Neuron<T>>() + 64)
            + s.synapses * (std::mem::size_of::<Synapse<T>>() + 64);
        s
    }
}

/// Structure cost. Implement a trait to replace or combine
/// criteria; the controller uses it to select the "best" topology.
pub trait StructureCost<T = f32> {
    /// Total cost `J` for quality of task (less is better) and structure.
    fn cost(&self, task_loss: T, stats: &StructuralStats<T>) -> T;
}

/// Linear cost:
/// `J = loss + λ_n·active_neurons + λ_s·active_synapses + λ_a·activity_cost + λ_m·state_bytes`.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct LinearCost<T = f32> {
    /// Penalty per neuron.
    pub lambda_neurons: T,
    /// Synapse penalty.
    pub lambda_synapses: T,
    /// Penalty for activity.
    pub lambda_activity: T,
    /// Memory penalty (per byte).
    pub lambda_state_bytes: T,
}

impl<T: Scalar> Default for LinearCost<T> {
    fn default() -> Self {
        Self {
            lambda_neurons: T::c(1e-3),
            lambda_synapses: T::c(1e-4),
            lambda_activity: T::c(1e-3),
            lambda_state_bytes: T::ZERO,
        }
    }
}

impl<T: Scalar> StructureCost<T> for LinearCost<T> {
    fn cost(&self, task_loss: T, s: &StructuralStats<T>) -> T {
        task_loss
            + self.lambda_neurons * T::from_usize(s.active_neurons)
            + self.lambda_synapses * T::from_usize(s.active_synapses)
            + self.lambda_activity * s.activity_cost
            + self.lambda_state_bytes * T::from_usize(s.state_bytes)
    }
}

/// An accumulator of joint activity of yet **unconnected** pairs of neurons.
///
/// When the receiver neuron `dst` spikes, and the neuron `src` spikes within
/// windows earlier, the pair `(src, dst)` gets a contribution of `1/(steps between them)` -
/// causal joint activity in the spirit of STDP. Number of stored pairs
/// limited to `capacity` (the weakest half is discarded).
#[derive(Debug, Clone, PartialEq)]
pub struct CoactivityTracker<T = f32> {
    pub(crate) window: u64,
    pub(crate) capacity: usize,
    pub(crate) max_recent: usize,
    pub(crate) pairs: BTreeMap<(NeuronId, NeuronId), T>,
    pub(crate) recent: Vec<(NeuronId, u64)>,
}

impl<T: Scalar> CoactivityTracker<T> {
    /// Create a tracker.
    pub fn new(window: u64, capacity: usize, max_recent: usize) -> Self {
        Self {
            window: window.max(1),
            capacity: capacity.max(1),
            max_recent: max_recent.max(1),
            pairs: BTreeMap::new(),
            recent: Vec::new(),
        }
    }

    /// Take into account the spikes of step `step`.
    pub fn observe(&mut self, topo: &Topology<T>, step: u64, spikers: &[NeuronId]) {
        for &dst in spikers {
            match topo.neuron(dst) {
                Some(n) if n.role != NeuronRole::Input => {}
                _ => continue,
            }
            for &(src, t) in &self.recent {
                let gap = step.saturating_sub(t);
                if src == dst || gap == 0 || gap > self.window || topo.connected(src, dst) {
                    continue;
                }
                *self.pairs.entry((src, dst)).or_insert(T::ZERO) +=
                    T::ONE / T::from_usize(gap as usize);
            }
        }
        let window = self.window;
        self.recent
            .retain(|&(_, t)| step.saturating_sub(t) < window);
        self.recent.extend(spikers.iter().map(|s| (*s, step)));
        if self.recent.len() > self.max_recent {
            let drop = self.recent.len() - self.max_recent;
            self.recent.drain(..drop);
        }
        if self.pairs.len() > self.capacity {
            self.shrink();
        }
    }

    fn shrink(&mut self) {
        let mut v: Vec<T> = self.pairs.values().copied().collect();
        v.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let cut = v[v.len() / 2];
        self.pairs.retain(|_, s| *s > cut);
    }

    /// Top candidates `((src, dst), score)`, in descending order of score; couples,
    /// those that have become invalid or are already bound are skipped.
    pub fn top(
        &self,
        topo: &Topology<T>,
        k: usize,
        min_score: T,
    ) -> Vec<((NeuronId, NeuronId), T)> {
        let mut v: Vec<_> = self
            .pairs
            .iter()
            .filter(|&(&(s, d), &sc)| {
                sc >= min_score
                    && topo.neuron(s).is_some()
                    && topo.neuron(d).is_some()
                    && !topo.connected(s, d)
            })
            .map(|(k, v)| (*k, *v))
            .collect();
        v.sort_by(|a, b| {
            b.1.partial_cmp(&a.1)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.0.cmp(&b.0))
        });
        v.truncate(k);
        v
    }

    /// Multiply accumulated scores by `factor ∈ (0, 1]` (forgetting).
    pub fn decay(&mut self, factor: T) {
        for s in self.pairs.values_mut() {
            *s *= factor;
        }
        self.pairs.retain(|_, s| *s > T::c(1e-6));
    }

    /// Forget a pair (for example, when a synapse is created on it).
    pub fn forget_pair(&mut self, src: NeuronId, dst: NeuronId) {
        self.pairs.remove(&(src, dst));
    }

    /// Forget everything related to the neuron.
    pub fn forget_neuron(&mut self, id: NeuronId) {
        self.pairs.retain(|&(s, d), _| s != id && d != id);
        self.recent.retain(|&(n, _)| n != id);
    }

    /// Clear recent spikes (sample boundary).
    pub fn clear_recent(&mut self) {
        self.recent.clear();
    }

    /// Number of accumulated pairs.
    pub fn len(&self) -> usize {
        self.pairs.len()
    }

    /// Is the drive empty?
    pub fn is_empty(&self) -> bool {
        self.pairs.is_empty()
    }
}
