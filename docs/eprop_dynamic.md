# E-prop on `DynamicNetwork` (`neo-snn::structural::eprop`)

Online (local-eligibility) plasticity for the dynamic network
(`DynamicNetwork`): `W ← W − η·d_j(t)·e_{j→i}(t)` after every `net.step()` —
no backprop through time, no BPTT, O(N + S) memory.

See module docs of `crates/neo-snn/src/structural/eprop.rs` for the full
reference (formulas, state lifecycles, config).

## Quick start

```rust
use neo_snn::structural::*;

let mut net = DynamicNetwork::new(topo, EngineConfig::default()).unwrap();
let mut trainer = EpropTrainer::new(&net, EpropConfig::default()).unwrap();
for _ in 0..10 {
    net.step(&[1.0])?;
    trainer.step(&mut net)?;
}
```

- Call `trainer.step(&mut net)?` **after** `net.step(&inputs)?`.
- `EpropTrainer::new` validates the config and the dynamics capability of
  every neuron (typed `StructuralError::UnsupportedEpropDynamics` on LIF/ALIF
  incompatibility).

## API

- `EpropTrainer::new(&net, EpropConfig)` — validates; `state()`, `config()`.
- `trainer.step(&mut net) -> StructuralResult<EpropReport>` — `{learning_signals, traces_updated, weights_updated}`.
- `set_target(neuron, y*)` — readout error (`ReadoutMode::LocalError`, default);
  `set_external_signal(neuron, d)` — `ReadoutMode::External` (RL/hybrid).
- `state()` — `synapse_trace_count()`, `neuron_signal_count()`, `steps()`,
  `targets()`, `feedback_weights()`; `reset_sequence()`, `begin_sample()`.
- `EpropScope` — enable/disable synapses, allow-lists (per-role default:
  sources Input/Hidden/Output, targets Hidden/Output).
- `EpropState` — encode for checkpointing via `save`/`load` around
  `DynamicNetwork::save_checkpoint` (state_bytes_roundtrip).

## Semantics

- Eligibility trace (per synapse): `e(t) = γ·e(t−1) + κ·x(t)` — nonzero when
  the synapse delivered and/or the postsynaptic neuron spiked
  (`x(t) = last_arrivals ∋ synapse`).
- Learning signals: output `d_j(t) = spike_j(t) − y_j^*` (LocalError);
  hidden `d_j(t) = Σ_k F[j][k]·d_k(t)` (feedback cascade: `FeedbackScheme::RandomFixed`
  or `Symmetric`); external: `set_external_signal`.
- Weight update (online form): `W ← W − η·d_j(t)·e_{j→i}(t)`, then projection via
  `Topology::set_synapse_weight` (weight bounds + Dale).

## Examples

- `cargo run --example eprop_dynamic -p neo-snn`.

## Verification status

- `cargo test -p neo-snn --test eprop` — 14 pass (LIF/ALIF eligibility references,
  toy training, scope, feedback schemes, checkpoint resume, ...).
- Lib test: `structural::tests::eprop_on_dynamic_network_trains_readout`.
- Full suite: `cargo test -p neo-snn` — green.
- Independent of the engine STDP/pipeline (composes with both).
