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
//! m    = mem ⊙ β + cur
//! fire = (m − θ ≥ 0) // input neurons: θ = 1e30, do not fire
//! mem' = m ⊙ (1 − fire ⊙ zero) − fire ⊙ sub ⊙ θ // Subtract: sub=1; Zero: zero=1; None: both 0
//! spk = fire + (inputs · Scatter ≥ 0.5) // Scatter [n_in, n] puts inputs into columns of input neurons
//! out  = spk · Gather                           // Gather [n, n_out]
//! ```

use std::collections::VecDeque;

use burn::tensor::{Device, Tensor, TensorData};

use crate::state::ResetMode;
use crate::structural::dense::DenseNet;

/// "Infinite" threshold limit for input neurons (inf in matrix operations is dangerous).
const NEVER: f32 = 1.0e30;

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
    device: Device,
}

/// The state of a batch on the device.
pub struct TensorState {
    batch: usize,
    mem: Tensor<2>,
    hist: VecDeque<Tensor<2>>,
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
            hist: VecDeque::new(),
        }
    }

    /// One step; `inputs` - `[batch, n_inputs]`, returns spikes of `[batch, n_outputs]` outputs.
    pub fn step(&self, st: &mut TensorState, inputs: Tensor<2>) -> Tensor<2> {
        let (b, n) = (st.batch, self.n);
        let mut cur: Tensor<2> = Tensor::zeros([b, n], &self.device);
        for (d, frame) in st.hist.iter().enumerate().take(self.weights.len()) {
            cur = cur + frame.clone().matmul(self.weights[d].clone());
        }
        let beta = self.beta.clone().expand([b, n]);
        let thr = self.thr.clone().expand([b, n]);
        let m = st.mem.clone() * beta + cur;
        let fired = (m.clone() - thr.clone()).greater_equal_elem(0.0).float();
        let keep = fired
            .clone()
            .mul(self.zero.clone().expand([b, n]))
            .neg()
            .add_scalar(1.0);
        st.mem = m * keep - fired.clone() * self.sub.clone().expand([b, n]) * thr;
        let in_spk = inputs
            .matmul(self.scatter.clone())
            .greater_equal_elem(0.5)
            .float();
        let spikes = fired + in_spk;
        let out = spikes.clone().matmul(self.gather.clone());
        st.hist.push_front(spikes);
        st.hist.truncate(self.weights.len());
        out
    }

    /// Run `[steps, batch, n_inputs]` → number of output spikes `[batch, n_outputs]`.
    pub fn run(&self, inputs: Tensor<3>) -> Tensor<2> {
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
