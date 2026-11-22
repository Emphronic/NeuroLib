//! Execution of the compiled topology on the Burn device (CPU/GPU) - in batches.
//!
//! [`DenseNet`] is the Burn-independent reference; it matches `DynamicNetwork`
//! spike-by-spike and across membranes (12 random networks with delays 0–3 and
//! all reset modes, tests `structural::dense`). This module mirrors that
//! reference row by row on tensors, and `tests/dense_burn.rs` compares both
//! implementations.
//!
//! ```text
//! cur = Σ_d hist[d] · W_d // matmul [B,n] · [n,n] for each delay
//! m    = mem ⊙ β + cur − adapt        // ALIF: adapt is the previous value
//! fire = (m − θ ≥ 0) // input neurons: θ = 1e30, do not fire
//! mem' = m ⊙ (1 − fire ⊙ zero) − fire ⊙ sub ⊙ θ // Subtract: sub=1; Zero: zero=1; None: both 0
//! adapt' = adapt ⊙ ρ + fire ⊙ a       // ALIF; LIF: a = 0, ρ = 1
//! spk = fire + (inputs · Scatter ≥ 0.5) // Scatter [n_in, n] puts inputs into columns of input neurons
//! out  = spk · Gather                           // Gather [n, n_out]
//! ```

use std::collections::VecDeque;

use burn::tensor::{Device, Tensor, TensorData};

use crate::state::ResetMode;
use crate::structural::dense::DenseNet;

/// "Infinite" threshold limit for input neurons (inf in matrix operations is dangerous).
const NEVER: f32 = 1.0e30;

/// Inhibitory (Vogels) plasticity config for the GPU executor.
/// Mirrors `structural::network::InhibitoryPlasticityConfig`, with the engine's
/// `StdpConfig::trace_decay` folded in (needed for the homeostatic `alpha`).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct InhibitoryPlasticityConfig {
    /// Whether the rule is enabled.
    pub enabled: bool,
    /// Learning rate `η`.
    pub eta: f32,
    /// Receiver target frequency (spikes per step).
    pub target_rate: f32,
    /// Trace decay (engine: `StdpConfig::trace_decay`) — used for `α = 2·target_rate/(1 − decay)`.
    pub trace_decay: f32,
}

impl Default for InhibitoryPlasticityConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            eta: 0.001,
            target_rate: 0.02,
            trace_decay: 0.95,
        }
    }
}

fn vec_of(t: &Tensor<2>) -> Vec<f32> {
    t.clone()
        .into_data()
        .try_to_vec::<f32>()
        .unwrap_or_default()
}

/// Network on the device.
pub struct TensorNet {
    n: usize,
    n_in: usize,
    n_out: usize,
    weights: Vec<Tensor<2>>,
    beta: Tensor<2>,
    thr: Tensor<2>,
    sub: Tensor<2>,
    zero: Tensor<2>,
    scatter: Tensor<2>,
    gather: Tensor<2>,
    mem0: Tensor<2>,
    adapt0: Tensor<2>, // [1, n] initial adaptation (ALIF; zeros for LIF)
    rho: Tensor<2>,    // [1, n] ALIF decay (1.0 for LIF)
    a: Tensor<2>,      // [1, n] ALIF increment (0.0 for LIF)
    vogels: Option<InhibitoryPlasticityConfig>, // Vogels inhibitory STDP (None = off)
    trace_decay: f32,  // StdpConfig::trace_decay (default 0.95)
    mod_decay: f32,    // EngineConfig::modulation_decay (default 0.9)
    weight_abs_max: f32, // ParamLimits::weight_abs_max (default 5)
    device: Device,
}

impl TensorNet {
    /// Number of neurons.
    pub fn n(&self) -> usize {
        self.n
    }

    /// Device of the network.
    pub fn device(&self) -> &Device {
        &self.device
    }

    /// Weight matrices `[d][i*n + j]` (read-only view).
    pub fn weights(&self) -> &[Tensor<2>] {
        &self.weights
    }

    /// Set weight matrices (for trainers that write weights online).
    pub fn set_weight_matrices(&mut self, weights: Vec<Tensor<2>>) {
        self.weights = weights;
    }

    /// Vogels inhibitory STDP (None = off).
    pub fn set_vogels(&mut self, ic: Option<InhibitoryPlasticityConfig>) {
        self.vogels = ic;
    }

    /// `StdpConfig::trace_decay` (default 0.95).
    pub fn set_trace_decay(&mut self, d: f32) {
        self.trace_decay = d;
    }

    /// `EngineConfig::modulation_decay` (default 0.9).
    pub fn set_mod_decay(&mut self, d: f32) {
        self.mod_decay = d;
    }

    /// `ParamLimits::weight_abs_max` (default 5).
    pub fn set_weight_abs_max(&mut self, w: f32) {
        self.weight_abs_max = w;
    }
}

/// The state of a batch on the device.
pub struct TensorState {
    batch: usize,
    mem: Tensor<2>,
    adapt: Tensor<2>,
    hist: VecDeque<Tensor<2>>,
    steps: u64,
    uses: Tensor<2>,   // [1, n] — spike counts of all neurons (input spikes included)
    traces: Tensor<2>, // [1, n] — per-neuron exponential traces (engine: `stt.trace`)
    modulation: Tensor<2>, // [1, n] — damped modulation gain (engine: `neuron.modulation`)
}

impl TensorState {
    /// Per-neuron exponential traces (`stt.trace`), updated per step:
    /// `trace = trace·decay + spike`.
    pub fn traces(&self) -> &Tensor<2> {
        &self.traces
    }

    /// Modulation gain of each neuron (`neuron.modulation`, `EngineConfig::modulation_decay`).
    pub fn modulation(&self) -> &Tensor<2> {
        &self.modulation
    }
}

impl TensorState {
    /// Batch size.
    pub fn batch(&self) -> usize {
        self.batch
    }

    /// Spike history `[batch, n]` per delay (`hist[d]` — spikes of the last `d+1` steps).
    pub fn hist(&self) -> &VecDeque<Tensor<2>> {
        &self.hist
    }

    /// Steps counted so far.
    pub fn steps(&self) -> u64 {
        self.steps
    }

    /// Spike counts per neuron (`[n]`) — inputs included.
    pub fn uses(&self) -> Vec<f32> {
        self.uses
            .clone()
            .into_data()
            .try_to_vec::<f32>()
            .unwrap_or_default()
    }

    /// Rates per neuron (`[n]`): `uses / steps` (0 when `steps == 0`).
    pub fn rates(&self) -> Vec<f32> {
        let uses = self.uses();
        let s = self.steps as f32;
        if s <= 0.0 {
            return uses;
        }
        uses.into_iter().map(|u| u / s).collect()
    }

    /// Steps, spike counts and rates in one struct (GPU stats).
    pub fn stats(&self) -> TensorNetStats {
        TensorNetStats {
            steps: self.steps,
            uses: self.uses(),
            rates: self.rates(),
        }
    }
}

/// Stats accumulated by [`TensorState`]: steps, per-neuron spike counts and rates.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct TensorNetStats {
    /// Steps counted so far.
    pub steps: u64,
    /// Spike counts per neuron (`[n]`).
    pub uses: Vec<f32>,
    /// Rates per neuron (`[n]`): `uses / steps`.
    pub rates: Vec<f32>,
}

fn t2(data: Vec<f32>, shape: [usize; 2], device: &Device) -> Tensor<2> {
    Tensor::from_data(TensorData::new(data, shape), device)
}

impl TensorNet {
    /// Transfer the compiled network to the device.
    pub fn from_dense(d: &DenseNet, device: &Device) -> Self {
        let n = d.n;
        let weights = d
            .weights
            .iter()
            .map(|w| t2(w.clone(), [n, n], device))
            .collect();
        let mut scatter = vec![0.0f32; d.n_inputs() * n];
        for (k, &j) in d.input_idx.iter().enumerate() {
            scatter[k * n + j] = 1.0;
        }
        let mut gather = vec![0.0f32; n * d.n_outputs()];
        for (k, &j) in d.output_idx.iter().enumerate() {
            gather[j * d.n_outputs() + k] = 1.0;
        }
        let sub = d
            .reset
            .iter()
            .map(|r| if *r == ResetMode::Subtract { 1.0 } else { 0.0 })
            .collect();
        let zero = d
            .reset
            .iter()
            .map(|r| if *r == ResetMode::Zero { 1.0 } else { 0.0 })
            .collect();
        Self {
            n,
            n_in: d.n_inputs(),
            n_out: d.n_outputs(),
            weights,
            beta: t2(d.beta.clone(), [1, n], device),
            thr: t2(
                d.threshold.iter().map(|&t| t.min(NEVER)).collect(),
                [1, n],
                device,
            ),
            sub: t2(sub, [1, n], device),
            zero: t2(zero, [1, n], device),
            scatter: t2(scatter, [d.n_inputs(), n], device),
            gather: t2(gather, [n, d.n_outputs()], device),
            mem0: t2(d.mem0.clone(), [1, n], device),
            adapt0: t2(d.adapt.clone(), [1, n], device),
            rho: t2(d.rho.clone(), [1, n], device),
            a: t2(d.a.clone(), [1, n], device),
            vogels: None,
            trace_decay: 0.95,
            mod_decay: 0.9,
            weight_abs_max: 5.0,
            device: device.clone(),
        }
    }

    /// Number of inputs.
    pub fn n_inputs(&self) -> usize {
        self.n_in
    }

    /// Number of outputs.
    pub fn n_outputs(&self) -> usize {
        self.n_out
    }

    /// Fresh state for a batch.
    pub fn new_state(&self, batch: usize) -> TensorState {
        TensorState {
            batch,
            mem: self.mem0.clone().expand([batch, self.n]),
            adapt: self.adapt0.clone().expand([batch, self.n]),
            hist: VecDeque::new(),
            steps: 0,
            uses: Tensor::zeros([1, self.n], &self.device),
            traces: Tensor::zeros([1, self.n], &self.device),
            modulation: Tensor::zeros([1, self.n], &self.device),
        }
    }

    /// One step; `inputs` - `[batch, n_inputs]`, returns spikes of `[batch, n_outputs]` outputs.
    ///
    /// Dynamics: `m = mem·β + cur − adapt` (ALIF; `adapt` is the previous value);
    /// `fired = m − θ ≥ 0`; `mem' = m·(1 − fired·zero) − fired·sub·θ`;
    /// `adapt' = adapt·ρ + fired·a` (LIF: a = 0, ρ = 1).
    pub fn step(&mut self, st: &mut TensorState, inputs: Tensor<2>) -> Tensor<2> {
        self.step_and_spikes(st, inputs).0
    }

    /// Like [`step`](Self::step) but also returns the spikes of **all** neurons
    /// (`[batch, n]`) — needed by online plasticity trainers.
    pub fn step_and_spikes(
        &mut self,
        st: &mut TensorState,
        inputs: Tensor<2>,
    ) -> (Tensor<2>, Tensor<2>) {
        let (b, n) = (st.batch, self.n);
        let mut cur: Tensor<2> = Tensor::zeros([b, n], &self.device);
        for (d, frame) in st.hist.iter().enumerate().take(self.weights.len()) {
            cur = cur + frame.clone().matmul(self.weights[d].clone());
        }
        let beta = self.beta.clone().expand([b, n]);
        let thr = self.thr.clone().expand([b, n]);
        // Modulation: `gain = st.modulation` (pre-decay); `st.modulation = gain·mod_decay`;
        // `m = mem·β + cur·(1 + gain)` (engine: `effective = current·(1+gain)`).
        let gain = st.modulation.clone().expand([b, n]);
        let m = st.mem.clone() * beta + cur * gain.add_scalar(1.0) - st.adapt.clone();
        let fired = (m.clone() - thr.clone()).greater_equal_elem(0.0).float();
        st.modulation = st.modulation.clone() * self.mod_decay;
        let keep = fired
            .clone()
            .mul(self.zero.clone().expand([b, n]))
            .neg()
            .add_scalar(1.0);
        st.mem = m * keep - fired.clone() * self.sub.clone().expand([b, n]) * thr;
        st.adapt = st.adapt.clone() * self.rho.clone().expand([b, n])
            + fired.clone() * self.a.clone().expand([b, n]);
        let in_spk = inputs
            .matmul(self.scatter.clone())
            .greater_equal_elem(0.5)
            .float();
        let spikes = fired + in_spk;
        let out = spikes.clone().matmul(self.gather.clone());
        st.hist.push_front(spikes.clone());
        st.hist.truncate(self.weights.len());
        // Stats: spike counts (all neurons), steps, traces (`stt.trace = trace·decay + spike`).
        st.uses = st.uses.clone() + spikes.clone().sum_dims(&[0]);
        st.steps += 1;
        st.traces = st.traces.clone() * self.trace_decay + spikes.clone().sum_dims(&[0]);
        // Vogels inhibitory STDP (if enabled): magnitudes of inhibitory
        // synapses move toward `η·(trace_post − α)` on arrival and
        // `η·trace_pre` on receiver spikes; applied as
        // `w = −((−w + dm).clamp(0, w_max))`.
        if let Some(ic) = self.vogels
            && ic.enabled
            && ic.eta > 0.0
        {
            self.vogels_update(st, ic);
        }
        (out, spikes)
    }

    /// Vogels inhibitory STDP step (host math on the weight matrices).
    /// Inhibitory synapses are those with `W_d[i,j] < 0` (Dale: inhibitory sources
    /// keep their weights negative — exact when Dale is enforced).
    fn vogels_update(&mut self, st: &mut TensorState, ic: InhibitoryPlasticityConfig) {
        let n = self.n;
        // Inhibitory mask (pairs).
        let mut inh = vec![vec![false; n]; n];
        for w in self.weights() {
            let wv = vec_of(w);
            for i in 0..n {
                for j in 0..n {
                    inh[i][j] = inh[i][j] || (wv[i * n + j] < 0.0);
                }
            }
        }
        if !inh.iter().any(|r| r.iter().any(|x| *x)) {
            return;
        }
        let alpha = 2.0 * ic.target_rate / (1.0 - ic.trace_decay).max(1e-6);
        let old_tr = st
            .traces
            .clone()
            .into_data()
            .try_to_vec::<f32>()
            .unwrap_or_default();
        let fired = st
            .hist
            .iter()
            .next()
            .map(vec_of)
            .unwrap_or_else(|| vec![0.0; n]); // spikes of the current step
        // Arrival: the inhibitory synapse delivered this step
        // (x[i,j] = Σ_d hist_d[i]·W_d[i,j] != 0).
        let mut arrival = vec![vec![0.0f32; n]; n];
        for (wd, w) in self.weights().iter().enumerate() {
            let wv = vec_of(w);
            let hv = st
                .hist()
                .get(wd)
                .map(vec_of)
                .unwrap_or_else(|| vec![0.0; n]);
            for i in 0..n {
                if hv[i] == 0.0 {
                    continue;
                }
                for j in 0..n {
                    arrival[i][j] += hv[i] * wv[i * n + j];
                }
            }
        }
        // dm[i,j]: magnitude delta.
        // (1) arrival of inhibitory synapse i→j: dm += η·(old_trace[j] − α).
        // (2) receiver j spikes: for inhibitory i→j: dm += η·old_trace[i].
        let eta = ic.eta;
        let mut new_ws: Vec<Tensor<2>> = self.weights().to_vec();
        for (wd, w) in new_ws.iter_mut().enumerate() {
            let mut wv = vec_of(w);
            let wd_mat = wv.clone();
            for i in 0..n {
                for j in 0..n {
                    if !inh[i][j] || wd_mat[i * n + j] >= 0.0 {
                        continue;
                    }
                    let mut dm = 0.0f32;
                    if arrival[i][j] != 0.0 {
                        // inhibitory arrival
                        dm += eta * ((old_tr.get(j).copied().unwrap_or(0.0)) - alpha);
                    }
                    if fired[j] > 0.0 {
                        dm += eta * old_tr.get(i).copied().unwrap_or(0.0);
                    }
                    let m = (-wv[i * n + j] + dm).max(0.0).min(self.weight_abs_max);
                    wv[i * n + j] = -m;
                }
            }
            let _ = wd;
            *w = Tensor::from_data(TensorData::new(wv, [n, n]), self.device());
        }
        self.set_weight_matrices(new_ws);
    }

    /// Run `[steps, batch, n_inputs]` → number of output spikes `[batch, n_outputs]`.
    pub fn run(&mut self, inputs: Tensor<3>) -> Tensor<2> {
        let [steps, b, k] = inputs.dims();
        let mut st = self.new_state(b);
        let mut counts: Tensor<2> = Tensor::zeros([b, self.n_out], &self.device);
        for t in 0..steps {
            let x = inputs.clone().slice([t..t + 1, 0..b, 0..k]).reshape([b, k]);
            counts = counts + self.step(&mut st, x);
        }
        counts
    }
}
