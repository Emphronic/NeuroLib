//! Dense batched topology executor — the reference for the GPU path.
//!
//! [`DenseNet::compile`] turns [`Topology`] into a set of dense matrices (one per delay) and
//! parameter vectors; [`DenseNet::step`] performs one step for a **batch** of independent samples
//! with the same formulas as `DynamicNetwork::step`, but using matrix operations:
//!
//! ```text
//! cur[b, j]  = Σ_d Σ_i spikes_{t-1-d}[b, i] · W_d[i, j]       // delivery with delay d
//! mem[b, j]  = β_j · mem[b, j] + cur[b, j]                     // non-input neurons
//! fire[b, j] = mem[b, j] ≥ θ_j                                 // inputs: input[b, k] ≥ 0.5
//! mem        = mem − fire·θ  (Subtract) | 0 (Zero) | mem (None)
//! ```
//!
//! Why this exists even though the GPU code (`neo_snn::dense_burn`) is written in Burn: the formulas here can be
//! **checked against the CPU engine without Burn** (see the tests: matching spikes and membranes), and
//! the Burn version is a row-by-row mirror of this file. The executor is forward-only: plasticity and
//! statistics are not computed; recompilation is needed after any structural change
//! (the topology changes between generations/intervals, not within an episode).

use std::collections::{BTreeMap, VecDeque};

use super::error::{StructuralError, StructuralResult};
use super::model::{NeuronModel, NeuronPolarity, NeuronRole};
use super::topology::Topology;
use crate::state::ResetMode;

/// Compiled network.
#[derive(Debug, Clone, PartialEq)]
pub struct DenseNet {
    /// Number of neurons.
    pub n: usize,
    /// Indices of input neurons (in ascending id order) — columns of the input vector.
    pub input_idx: Vec<usize>,
    /// Indices of output neurons.
    pub output_idx: Vec<usize>,
    /// β per neuron (0 for input neurons).
    pub beta: Vec<f32>,
    /// Threshold per neuron (+∞ for input neurons — they do not fire from the membrane).
    pub threshold: Vec<f32>,
    /// Reset mode per neuron.
    pub reset: Vec<ResetMode>,
    /// Initial membrane.
    pub mem0: Vec<f32>,
    /// `weights[d][i * n + j]` — total weight of active synapses `i → j` with delay `d`.
    pub weights: Vec<Vec<f32>>,
}

/// Batch state.
#[derive(Debug, Clone, PartialEq)]
pub struct DenseState {
    /// Batch size.
    pub batch: usize,
    /// Membranes `[batch, n]`.
    pub mem: Vec<f32>,
    /// Spike history `[batch, n]` (0.0/1.0); `hist[0]` is the previous step.
    pub hist: VecDeque<Vec<f32>>,
}

impl DenseNet {
    /// Compile a topology. Disabled synapses are excluded; parallel connections are summed.
    pub fn compile(topo: &Topology) -> StructuralResult<Self> {
        let order: Vec<_> = topo.neurons().map(|n| n.id).collect();
        let n = order.len();
        let index: BTreeMap<_, _> = order.iter().enumerate().map(|(i, id)| (*id, i)).collect();
        let (mut input_idx, mut output_idx) = (vec![], vec![]);
        let mut beta = vec![0.0; n];
        let mut threshold = vec![f32::INFINITY; n];
        let mut reset = vec![ResetMode::None; n];
        let mut mem0 = vec![0.0; n];
        for (i, nr) in topo.neurons().enumerate() {
            if nr.polarity != NeuronPolarity::Excitatory {
                return Err(StructuralError::InvalidConfig(format!(
                    "DenseNet supports only excitatory neurons: {} — {}",
                    nr.id,
                    nr.polarity.as_str()
                )));
            }
            match nr.role {
                NeuronRole::Input => input_idx.push(i),
                NeuronRole::Output => output_idx.push(i),
                NeuronRole::Hidden => {}
            }
            if nr.role != NeuronRole::Input {
                let NeuronModel::Lif(p) = nr.model else {
                    return Err(StructuralError::InvalidConfig(format!(
                        "DenseNet supports only LIF: {} — {}",
                        nr.id,
                        nr.model.as_str()
                    )));
                };
                beta[i] = p.beta;
                threshold[i] = p.threshold;
                reset[i] = p.reset;
                mem0[i] = nr.state.mem();
            }
        }
        let max_delay = topo
            .synapses()
            .filter(|s| s.is_active())
            .map(|s| s.delay as usize)
            .max()
            .unwrap_or(0);
        let mut weights = vec![vec![0.0f32; n * n]; max_delay + 1];
        for s in topo.synapses().filter(|s| s.is_active()) {
            let (Some(&i), Some(&j)) = (index.get(&s.src), index.get(&s.dst)) else {
                return Err(StructuralError::InvariantViolation(format!(
                    "synapse {} references a nonexistent neuron",
                    s.id
                )));
            };
            weights[s.delay as usize][i * n + j] += s.weight;
        }
        Ok(Self {
            n,
            input_idx,
            output_idx,
            beta,
            threshold,
            reset,
            mem0,
            weights,
        })
    }

    /// Number of inputs.
    pub fn n_inputs(&self) -> usize {
        self.input_idx.len()
    }

    /// Number of outputs.
    pub fn n_outputs(&self) -> usize {
        self.output_idx.len()
    }

    /// Maximum delay (number of matrices − 1).
    pub fn max_delay(&self) -> usize {
        self.weights.len() - 1
    }

    /// Fresh state for a batch.
    pub fn new_state(&self, batch: usize) -> DenseState {
        let mut mem = Vec::with_capacity(batch * self.n);
        for _ in 0..batch {
            mem.extend_from_slice(&self.mem0);
        }
        DenseState {
            batch,
            mem,
            hist: VecDeque::new(),
        }
    }

    /// One batch step. `inputs` is `[batch, n_inputs]`; returns output spikes `[batch, n_outputs]`.
    pub fn step(&self, st: &mut DenseState, inputs: &[f32]) -> StructuralResult<Vec<f32>> {
        let (n, b) = (self.n, st.batch);
        let k = self.n_inputs();
        if inputs.len() != b * k {
            return Err(StructuralError::InputSize {
                op: "DenseNet::step",
                expected: b * k,
                actual: inputs.len(),
            });
        }
        let mut cur = vec![0.0f32; b * n];
        for (d, frame) in st.hist.iter().enumerate().take(self.weights.len()) {
            let w = &self.weights[d];
            for bi in 0..b {
                let row = &mut cur[bi * n..(bi + 1) * n];
                for (i, &s) in frame[bi * n..(bi + 1) * n].iter().enumerate() {
                    if s != 0.0 {
                        for (c, &wij) in row.iter_mut().zip(&w[i * n..(i + 1) * n]) {
                            *c += s * wij;
                        }
                    }
                }
            }
        }
        let mut spikes = vec![0.0f32; b * n];
        for bi in 0..b {
            for j in 0..n {
                let idx = bi * n + j;
                let m = self.beta[j] * st.mem[idx] + cur[idx];
                let fired = m >= self.threshold[j];
                st.mem[idx] = if fired {
                    match self.reset[j] {
                        ResetMode::Subtract => m - self.threshold[j],
                        ResetMode::Zero => 0.0,
                        ResetMode::None => m,
                    }
                } else {
                    m
                };
                spikes[idx] = if fired { 1.0 } else { 0.0 };
            }
            for (slot, &j) in self.input_idx.iter().enumerate() {
                spikes[bi * n + j] = if inputs[bi * k + slot] >= 0.5 {
                    1.0
                } else {
                    0.0
                };
            }
        }
        let out = (0..b)
            .flat_map(|bi| self.output_idx.iter().map(move |&j| (bi, j)))
            .map(|(bi, j)| spikes[bi * n + j])
            .collect();
        st.hist.push_front(spikes);
        st.hist.truncate(self.weights.len());
        Ok(out)
    }

    /// Run `steps` steps. `inputs` is `[steps, batch, n_inputs]`. Returns the spike count
    /// of each output over the run `[batch, n_outputs]`.
    pub fn run(&self, inputs: &[f32], steps: usize, batch: usize) -> StructuralResult<Vec<f32>> {
        let k = self.n_inputs();
        if inputs.len() != steps * batch * k {
            return Err(StructuralError::InputSize {
                op: "DenseNet::run",
                expected: steps * batch * k,
                actual: inputs.len(),
            });
        }
        let mut st = self.new_state(batch);
        let mut counts = vec![0.0f32; batch * self.n_outputs()];
        for t in 0..steps {
            let out = self.step(&mut st, &inputs[t * batch * k..(t + 1) * batch * k])?;
            counts.iter_mut().zip(out).for_each(|(c, o)| *c += o);
        }
        Ok(counts)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::structural::{
        Birth, DynamicNetwork, EngineConfig, LifParams, NeuronKind, Origin, Rng64,
        StructuralConfig, Topology,
    };

    fn random_net(seed: u64, n_in: usize, n_hid: usize, n_out: usize, syn: usize) -> Topology {
        let cfg = StructuralConfig::default();
        let mut t = Topology::new(cfg.topology, cfg.params);
        let mut r = Rng64::new(seed);
        let b = Birth::default();
        let mut ins = vec![];
        let mut dst = vec![];
        for _ in 0..n_in {
            ins.push(
                t.add_neuron(NeuronKind::Input, LifParams::default(), Origin::Initial, b)
                    .unwrap(),
            );
        }
        for k in 0..n_hid + n_out {
            let reset = [ResetMode::Subtract, ResetMode::Zero, ResetMode::None][k % 3];
            let p = LifParams {
                beta: 0.5 + 0.4 * r.next_f32(),
                threshold: 0.6 + r.next_f32(),
                reset,
            };
            let kind = if k < n_hid {
                NeuronKind::Hidden
            } else {
                NeuronKind::Output
            };
            dst.push(t.add_neuron(kind, p, Origin::Initial, b).unwrap());
        }
        let srcs: Vec<_> = ins.iter().chain(dst.iter()).copied().collect();
        let (mut made, mut guard) = (0, 0);
        while made < syn && guard < 10_000 {
            guard += 1;
            let s = srcs[r.below(srcs.len())];
            let d = dst[r.below(dst.len())];
            let delay = r.below(4) as u16;
            let w = 1.4 * r.symmetric() + 0.3;
            if t.add_synapse(s, d, w, delay, Origin::Initial, b).is_ok() {
                made += 1;
            }
        }
        t
    }

    fn input_pattern(r: &mut Rng64, steps: usize, n_in: usize) -> Vec<Vec<f32>> {
        (0..steps)
            .map(|_| {
                (0..n_in)
                    .map(|_| if r.chance(0.35) { 1.0 } else { 0.0 })
                    .collect()
            })
            .collect()
    }

    #[test]
    fn matches_event_engine_spike_for_spike_and_membrane() {
        for seed in 1..=12u64 {
            let (n_in, n_hid, n_out) = (
                3 + (seed % 3) as usize,
                4 + (seed % 5) as usize,
                2 + (seed % 2) as usize,
            );
            let topo = random_net(seed, n_in, n_hid, n_out, 40);
            let dense = DenseNet::compile(&topo).unwrap();
            let mut net = DynamicNetwork::new(topo, EngineConfig::default()).unwrap();
            net.set_plasticity_enabled(false);
            let mut st = dense.new_state(1);
            let mut r = Rng64::new(seed * 77);
            let mut total_out = 0.0;
            for (t, x) in input_pattern(&mut r, 150, n_in).iter().enumerate() {
                let a = net.step(x).unwrap();
                let b = dense.step(&mut st, x).unwrap();
                let a: Vec<f32> = a
                    .output_spikes
                    .iter()
                    .map(|&s| if s { 1.0 } else { 0.0 })
                    .collect();
                assert_eq!(a, b, "seed {seed} step {t}: output spikes");
                total_out += b.iter().sum::<f32>();
                for (i, nr) in net.topology().neurons().enumerate() {
                    if nr.kind() != NeuronKind::Input {
                        assert!(
                            (nr.mem() - st.mem[i]).abs() < 1e-4,
                            "seed {seed} step {t} neuron {i}: {} vs {}",
                            nr.mem(),
                            st.mem[i]
                        );
                    }
                }
            }
            assert!(
                total_out > 0.0,
                "seed {seed}: the test is meaningless without output spikes"
            );
        }
    }

    #[test]
    fn batch_rows_are_independent() {
        let topo = random_net(5, 4, 6, 3, 50);
        let dense = DenseNet::compile(&topo).unwrap();
        let mut r = Rng64::new(9);
        let (steps, batch, k) = (60usize, 4usize, dense.n_inputs());
        let per_row: Vec<Vec<Vec<f32>>> = (0..batch)
            .map(|_| input_pattern(&mut r, steps, k))
            .collect();
        let mut flat = vec![0.0f32; steps * batch * k];
        for t in 0..steps {
            for b in 0..batch {
                flat[(t * batch + b) * k..(t * batch + b + 1) * k].copy_from_slice(&per_row[b][t]);
            }
        }
        let together = dense.run(&flat, steps, batch).unwrap();
        for b in 0..batch {
            let single: Vec<f32> = per_row[b].iter().flatten().copied().collect();
            let alone = dense.run(&single, steps, 1).unwrap();
            assert_eq!(
                &together[b * dense.n_outputs()..(b + 1) * dense.n_outputs()],
                &alone[..],
                "row {b}"
            );
        }
    }

    #[test]
    fn shapes_are_checked() {
        let topo = random_net(3, 3, 4, 2, 20);
        let dense = DenseNet::compile(&topo).unwrap();
        assert_eq!(dense.n, 9);
        assert_eq!((dense.n_inputs(), dense.n_outputs()), (3, 2));
        let mut st = dense.new_state(2);
        assert!(matches!(
            dense.step(&mut st, &[0.0; 5]),
            Err(StructuralError::InputSize { .. })
        ));
        assert!(matches!(
            dense.run(&[0.0; 5], 2, 1),
            Err(StructuralError::InputSize { .. })
        ));
    }
}
