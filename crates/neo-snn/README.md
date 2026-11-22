# neo-snn

Spiking neural network library built on [Burn](https://github.com/burn-rs/burn): LIF,
Izhikevich, AdEx, graded spikes, STDP plasticity, spike-driven attention, structural
plasticity, telemetry, and more.

## What it provides

| Module | Contents |
|---|---|
| [`neuron`](./src/neuron) | Neuron models: `Leaky` (LIF), `Izhikevich`, `AdEx`, `GradedLeaky`, `RLeaky`, `RSynaptic`, `AdaptiveLeaky`, `HodgkinHuxley`, `Lapicque`, `Alpha`, `MultiStepLeaky`, ... |
| [`layers`](./src/layers) | Composite layers: `StdpLinear`, `StdpConv1d`, `StdpConv2d`, `SpikingConv2d`, `SConv2dLSTM`, `SynapticDelay`, `StdpRule` (Pair/Triplet/RewardModulated) |
| [`functional`](./src/functional) | Losses (`ce_rate_loss`, `ce_temporal_loss`, `mse_count_loss`, ...), metrics (`accuracy_rate`, ...), STDP plumbing (`pair_update`, `triplet_update`, `apply_reward`), monitors |
| [`attention`](./src/attention) | Spike-based self-attention, Spikformer blocks, SpikingMlp, spike-driven transformers |
| [`encoding`](./src/encoding) | Spike encodings: `rate_encode`, `latency_encode`, `delta_encode`, `periodic_encode`, `burst_encode` |
| [`surrogate`](./src/surrogate) | Surrogates: `Triangular`, `FastSigmoid`, `ATan`, `StraightThrough`, `SpikeRateEscape`, ... |
| [`spiking_layer`](./src/spiking_layer.rs) | The `SpikingLayer` contract: `init_state` / `step` / `forward_seq` — one API for every layer; `Session` wrapper |
| [`state`](./src/state.rs) / [`stateful`](./src/stateful.rs) | Neuron/layer states, reset semantics, snnTorch-compatible `init_hidden` mode |
| [`structural`](./src/structural) | Structural plasticity: growth/removal of neurons and connections on a dynamic graph; evolution |
| [`analysis`](./src/structural/analysis) | Connectivity from activity — the estimation of the network connectivity from the observed activity ([arXiv:2611.21592](https://arxiv.org/abs/2611.21592)) — [book](../../neo-book/src/advanced/connectivity-from-activity.md) |
| [`convert`](./src/convert) | ANN → SNN conversion (`convert` / `ann_to_snn`) |
| [`reservoir`](./src/reservoir) | Liquid State Machine / Echo State Network, RidgeReadout |
| [`quant`](./src/quant.rs) | State quantization (QAT for the membrane) |
| [`telemetry`](./src/telemetry) | Training telemetry: events, structure/evolution snapshots, weight distributions (rendered by the GUI Training tab) |
| [`event_driven`](./src/event_driven) | SKAN: delta-gated KAN + LIF |
| [`bntt`](./src/bntt.rs) | BatchNorm through time |
| [`dense_burn`](./src/dense_burn.rs) | Dense (matrix) forward pass of a structural network on Burn tensors |
| [`prelude`](./src/prelude.rs) | Everyday names of the crate |
| [`probabilistic`](./src/probabilistic) | Stochastic thresholds (Bayesian) — `StochasticThresholdLeaky`, `ThresholdDistribution`; Rate Estimation (`rate_estimate`), `ProbabilityState`, `ThresholdPosterior`; NLL/ECE metrics — [book](../../neo-book/src/building-blocks/stochastic-bayesian.md) |

## Features

| Feature | Description |
|---|---|
| `demo` | Enable `neo_snn::demo` training loop (requires `burn/optim`) |

## Usage

```rust
use neurolib::snn::Session;
use neurolib::snn::neuron::LeakyConfig;
use neurolib::snn::surrogate::Triangular;

let lif = LeakyConfig::new(128).init(&device);
let session = Session::new(lif, batch, &device);
let spikes = session.step(x_t, &Triangular::new(1.0));
```

## Cargo

```toml
[dependencies]
neo-snn = { path = "crates/neo-snn", features = ["demo"] }
```

## Book

[Basic workflow](../../neo-book/src/basic-workflow/model.md),
[STDP](../../neo-book/src/building-blocks/stdp.md),
[Telemetry](../../neo-book/src/performance/profiling.md)



