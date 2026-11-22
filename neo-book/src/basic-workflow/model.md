# Model

Models in NeuroLib are **spiking layers** — everything follows one contract, the `SpikingLayer` trait:

```text
init_state(batch, device) -> State           // membrane, traces, ring buffer, ...
step(x, state, surrogate) -> (y, state')      // one time step
forward_seq(vec![x_0, ..., x_T], surrogate) -> vec![y_0, ...]
```

Everything in the workspace implements this contract — the trait lives in `neo-snn::spiking_layer`. Layers with a
known shape implement it directly; element-wise layers (LIF, Izhikevich, ...) return `Deferred::Pending` from
`init_state` and create the real state on the first `step` according to the input shape.

## Quick example

```rust
use neurolib::snn::{neuron::LeakyConfig, spiking_layer::SpikingLayer, surrogate::Triangular};
use burn::tensor::Tensor;

let device = Default::default();

// LIF layer: U' = β·U + I, spike when U ≥ θ
let lif = LeakyConfig::new(128).init(&device);

// Drive the whole sequence with a surrogate gradient
let spikes = lif.forward_seq(vec![x; 20], &Triangular::new(1.0));
```

## Layers

- **Neuron models** (`neo-snn::neuron`): `Leaky` (LIF), `Izhikevich`, `AdEx`, `HodgkinHuxley`, `AdaptiveLeaky`
  (RLeaky), `RSynaptic`, `GradedLeaky`, `Lapicque`, `Alpha`, `MultiStepLeaky`, `LeakyParallel` — each with a config
  type (`LeakyConfig`, `IzhikevichConfig`, ...). See [Neuron models](../building-blocks/neurons.md).
- **Composite layers** (`neo-snn::layers`): `StdpLinear`, `StdpConv1d`, `StdpConv2d`, `SpikingConv2d`,
  `SConv2dLSTM`, `SynapticDelay` — networks of neurons trained by STDP or delay, not backprop. See
  [STDP](../building-blocks/stdp.md).
- **Attention & transformer blocks** (`neo-snn::attention`): `SpikingSelfAttention`, `SpikeDrivenSelfAttention`
  (SSA/SDSA), `SpikingMlp`, `SpikingTransformerBlock`, `AttentionKind`.
- **Reservoir computing** (`neo-snn::reservoir`): `LiquidStateMachine`, `EchoStateNetwork`, `RidgeReadout`.
- **Encoder/decoder helpers**: `rate_encode`, `latency_encode`, `delta_encode`, `periodic_encode`, `burst_encode`
  (see [Spikes & encoding](../building-blocks/spikes.md)).

## Guardrails from the workspace design

- **Errors are typed**: `SnnError` (`neo-snn::SnnError`); operations with a `try_*` variant return `Result` instead of
  panicking.
- **State is explicit**: pass the state through `step`/`forward_seq_from` by value — the state chain cannot
  accidentally reuse a stale one. `StdpLinear`-style layers keep weights *inside* the state.
