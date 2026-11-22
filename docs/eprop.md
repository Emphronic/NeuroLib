# Online plasticity system (`structural::eprop`)

The online plasticity system is a **local-eligibility** rule: each weight is
updated online, after every step of [`DynamicNetwork`](../crates/neo-snn/src/structural/network.rs),
from an eligibility trace (a decaying memory of presynaptic activity) and an
instantaneous **learning signal** of the postsynaptic neuron. It is not BPTT,
not a surrogate gradient, and not R-STDP — see the composition rules below.

```text
e_{j→i}(t) = γ·e_{j→i}(t−1) + κ_j(t)·x_i(t)
W_{j→i}(t)  ← W_{j→i}(t−1) − η·d_j(t)·e_{j→i}(t)   (projected: bounds + Dale)
```

## 1. Concepts (separated)

| Concept | What it is | Where |
|---|---|---|
| Synapse eligibility trace | Decaying trace `e_{j→i}(t)` of the eligibility factor; nonzero on spike steps of the postsynaptic neuron | `EpropState::synapse_traces` — `O(S)` |
| Neuron learning signal | Instantaneous `d_j(t)`: readout error (outputs), feedback cascade (hidden), or external value (RL/hybrid) | `EpropState::neuron_signals` — `O(N)` |
| Weight update rule | `W_{j→i} ← W_{j→i} − η·d_j(t)·e_{j→i}(t)` then projection | trainer writes via `Topology::set_synapse_weight` |
| Online trainer state | Counters (`steps`, `samples`, `updates`), targets, external signals, feedback weights | `EpropState` — serializable |
| Weight bounds | Projection onto `ParamLimits::weight_abs_max` and onto the Dale half-plane of the source | `set_synapse_weight` |

The trainer is the only writer of weights for enabled synapses. There is no
random double update: if you also enable STDP (`StdpConfig::mode`), the STDP
rule and the trainer never touch the same weight through one call — see the
composition section.

## 2. Mathematical derivation for LIF and Adaptive LIF

### LIF

Engine (`lif_step`):

```text
U_j(t) = β·U_j(t−1) + I_j(t),      I_j(t) = Σ_{i ∈ in(j)} W_{j→i}·x_i(t)
ŷ_j(t) = [U_j(t) ≥ θ_j]            (hard spike; Subtract reset: U ← U − θ)
```

Eligibility of synapse `i→j` (nonzero on spike steps of the postsynaptic
neuron):

```text
e_{j→i}(t) = γ·e_{j→i}(t−1) + ŷ_j(t)·x_i(t)
```

with the convention `x_i(t) = 1` if the synapse `i→j` delivered current at
step `t` (`DynamicNetwork::last_arrivals`).

Online weight update (one-step online form of the accumulated gradient; the
trace carries the history, so `∂L/∂W ≈ Σ_{t'≤t} d_j(t')·e(t')` is approximated
by `d_j(t)·e(t)`):

```text
W_{j→i}(t) ← W_{j→i}(t−1) − η·d_j(t)·e_{j→i}(t)
```

### Adaptive LIF

Engine (`adaptive_lif_step`):

```text
U_j(t) = β·U_j(t−1) + I_j(t) − A_j(t−1)
A_j(t) = ρ·A_j(t−1) + a·ŷ_spike(t)   (on spike; A *= ρ otherwise)
ŷ_j(t) = [U_j(t) ≥ θ_j]
```

Define the potential `V_j(t) = U_j(t) + A_j(t−1) = β·U_j(t−1) + I_j(t)`.
Since the spike condition is `U_j(t) ≥ θ_j` ⟺ `V_j(t) ≥ θ_j + A_j(t−1)`,
ALIF is exactly an LIF neuron with the membrane `V_j` and the **adaptive
threshold** `Θ_j(t) = θ_j + A_j(t−1)`. Its eligibility follows the LIF rule:

```text
e_{j→i}(t) = γ·e_{j→i}(t−1) + ŷ_j(t)·x_i(t)
```

A larger `A_j` raises `Θ_j`, so repeated firing slows down — the classic
adaptation effect falls out of the derivation.

## 3. Learning signals

| Mode | What `d_j(t)` is | Config |
|---|---|---|
| `ReadoutMode::LocalError` (default) | Output neurons: `d_j(t) = ŷ_j(t) − y_j^*(t)` (target, default `0`); hidden: `d_j(t) = Σ_k F[j][k]·d_k(t)` | `EpropTrainer::set_target` |
| `FeedbackScheme::RandomFixed { seed, magnitude }` | `F[j][k] ~ U(−m, +m)` fixed random weights (seeded, deterministic) | `EpropConfig::feedback` |
| `FeedbackScheme::Symmetric` | `F[j][k]` = the learned weight of the synapse `j → k` (mirrored sign); `0` if absent | `EpropConfig::feedback` |
| `FeedbackScheme::None` | Hidden neurons: `d = 0` | `EpropConfig::feedback` |
| `ReadoutMode::External` | All signals from `EpropTrainer::set_external_signal` (RL/hybrid) | `EpropConfig::readout_mode` |

The trainer applies learning signals only to synapses that are enabled by
`EpropScope` (see §5); readout errors are applied to output synapses only
when the target differs from the spike.

## 4. Weight update and projection

`W_{j→i}(t) ← W_{j→i}(t−1) − η·d_j(t)·e_{j→i}(t)`:

- updated weights are always projected into `[−w_max, +w_max]`
  (`ParamLimits::weight_abs_max`) and onto the Dale half-plane of the source
  (`TopologyLimits::enforce_dale`) — the write goes through
  `Topology::set_synapse_weight`, the same projection used by all other
  weight writers;
- `d_j(t) == 0` or `e == 0` steps skip the update (no spurious movement).

## 5. Hybrid use: independent enabling and no double updates

```rust
EpropScope {
    sources: [NeuronRole::Input, NeuronRole::Hidden].into(),
    targets: [NeuronRole::Hidden, NeuronRole::Output].into(),
    only_synapses: None,
    disabled_synapses: [SynapseId(7)].into(),   // per-connection opt-out
}
```

- By roles ("layers"): `sources`/`targets` sets of `NeuronRole`.
- Per synapse ("connections"): `only_synapses` (allow-list) and
  `disabled_synapses` (deny-list) — `EpropTrainer::is_enabled_synapse(net, id)`.

Composition order (explicit, no random double updates):

1. `net.step(&inputs)?;` — engine delivers spikes/currents (STDP/R-STDP rules
   run here if configured).
2. `trainer.step(&mut net)?;` — the online system updates eligibility traces
   and writes weights via `set_synapse_weight`.
3. Explicit `net.apply_reward(...)` (R-STDP) is a separate call — it is never
   called by the trainer and the trainer never applies rewards.

## 6. Online memory

- Traces: `BTreeMap<SynapseId, T>`, at most one per enabled synapse — `O(S)`.
  Zero traces are removed; decay applies on non-spike steps.
- Signals: `BTreeMap<NeuronId, T>` — `O(N)`.
- Feedback weights: `BTreeMap<NeuronId, BTreeMap<NeuronId, T>>` — built once
  at `EpropTrainer::new` (for `RandomFixed`, O(#hidden × #output); the map is
  constant w.r.t. sequence length, so the memory is `O(N + S + F)`).
- No BPTT: nothing per step is retained except the traces.

## 7. Batch/sample boundaries, reset, checkpoint-resume

- `trainer.begin_sample(&mut net)` — `net.begin_sample()` + `samples += 1`
  + `reset_sequence()` when `EpropConfig::detach_on_sample` (default `true`).
- `trainer.reset_sequence()` — clears synapse traces (manual detach).
- `trainer.save_state()` / `trainer.load_state(&bytes)` —
  `EpropState::to_bytes`/`from_bytes`: traces, signals, feedback, targets,
  external signals, counters. Continue training afterwards.
- Full network checkpoint: `save_checkpoint(&net, &ctl, extra)`
  (trainer state is managed by the caller, the network state by the engine).

## 8. Limitations

- **Models**: LIF and AdaptiveLIF only (`OnlineEpropFamily::Lif` /
  `OnlineEpropFamily::AdaptiveLif`). Izhikevich, AdEx, Hodgkin–Huxley and
  custom models are rejected at `EpropTrainer::new` and
  `EpropTrainer::step` with
  `StructuralError::UnsupportedEpropDynamics` — a typed error, not an
  approximate rule.
- **Space**: CPU only (the dynamic engine is CPU-only).
- Traces of disabled synapses are not updated; `e = 0` on non-spike steps
  (the trace decays by `γ`).

## 9. GPU e-prop trainer (`structural::eprop_gpu`)

The same rule on the Burn dynamic executor (`dense_burn::TensorNet`), on
tensors (`GpuEpropTrainer`):

```rust
use neo_snn::dense_burn::{TensorNet, TensorState};
use neo_snn::structural::dense::DenseNet;
use neo_snn::structural::eprop_gpu::{GpuEpropConfig, GpuEpropTrainer};
use neo_snn::structural::{Topology, ...};

let dense = DenseNet::compile(&topo)?;
let mut trainer = GpuEpropTrainer::new(TensorNet::from_dense(&dense, &Device::flex()), GpuEpropConfig::default())?;
let mut st = trainer.net().new_state(1);
trainer.step(&mut st, inputs)?;       // dynamics + eligibility + weight update
let traces: &Tensor<2> = trainer.traces();          // e[i,j], [n, n]
let ws = trainer.net().weights();                   // updated weights
```

Math (see §2 — same as the scalar system):

```text
e[i,j](t) = γ·e[i,j](t−1) + κ_j(t)·x[i,j](t)   // x[i,j] = Σ_d hist_d[i]·W_d[i,j]
W_d[i,j](t) ← W_d[i,j](t−1) − η·d_j(t)·e[i,j](t)
```

- Signals: readout error for outputs (`targets` per neuron index),
  feedback cascade for hidden (`feedback: Tensor<2> [n_h, n_o]`), or external
  (`use_external` + `set_external_signals`).
- LIF + ALIF, all `ResetMode`s, inhibition — same as `DenseNet`.
- Batch size 1; traces `Tensor<2> [n, n]`; weight update via `set_weight_matrices`.
- Tests: `tests/eprop_gpu.rs` (`gpu_eprop_traces_match_reference`,
  `gpu_eprop_traces_nonzero_on_spikes_and_external`), plus unit tests in the module.

## 10. Examples

- `examples/eprop_sequence.rs` — long sequence training without BPTT (scalar).

## 11. API reference (scalar)

```rust
use neo_snn::structural::eprop::{EpropConfig, EpropScope, EpropTrainer, ReadoutMode, FeedbackScheme};

let mut trainer = EpropTrainer::new(&net, EpropConfig::default())?;
// typical loop:
//   net.step(&inputs)?;
//   trainer.set_target(output_id, target);
//   trainer.step(&mut net)?;
```

Key types: `EpropConfig`, `EpropState`, `EpropTrainer`, `EpropReport`,
`EpropScope`, `ReadoutMode`, `FeedbackScheme`, `OnlineEpropFamily`
(`NeuronModel::online_eprop_family`), `StructuralError::UnsupportedEpropDynamics`.
