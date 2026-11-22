# Training

Two kinds of training in NeuroLib: **surrogate-gradient training** through Burn autodiff, and **local STDP** updates.

## Surrogate-gradient loop

```rust
use neurolib::snn::{neuron::LeakyConfig, spiking_layer::Session, surrogate::Triangular};

let lif = LeakyConfig::new(128).init(&device);
let session = Session::new(lif, batch, &device);

for x_t in inputs {
    let spikes = session.step(x_t, &Triangular::new(1.0));
    // losses & metrics: ce_rate_loss(&out, &labels), winner_take_all, ...
}
```

- **Surrogates** (`neo-snn::surrogate`): `Triangular`, `FastSigmoid`, `ATan`, `StraightThrough`, `SpikeRateEscape`,
  `SpikeOperator`, `Heaviside`, `SigmoidSurrogate`, ...
- **Functional API** (`neo-snn::functional`): losses — `ce_rate_loss`, `ce_temporal_loss`, `ce_count_loss`,
  `ce_max_membrane_loss`, `mse_count_loss`, `mse_membrane_loss`, `mse_temporal_loss`, `l2_spike_rate`, `l1_spike_rate`;
  reading — `mean_over_time`, `max_over_time`, `sum_over_time`, `time_slice`, `spike_counts`, `first_spike_times`,
  `population_rates`, `population_reduce`, `accuracy_rate`, `accuracy_temporal`, `population_accuracy_rate`;
  tools — `winner_take_all`, `apply_reward`, `grad_norm`, `spike_raster`, `lateral_inhibition`.
- **Telemetry** (`neo-snn::telemetry`): `channel(ChannelConfig::new("name"))` → `TrainTx` / `TrainRx`; training
  events via `tx.step(StepMetrics { progress, loss, accuracy, grad_norm })`, spike snapshots via `tx.spikes(...)`
  (on demand), `tx.checkpoint()` for Stop, `tx.finish()` at the end. Everything is rendered by the GUI training
  dashboard (Training tab).

## STDP training (local, no backprop)

`StdpLinear` / `StdpConv1d` / `StdpConv2d` learn through STDP — see [STDP](../building-blocks/stdp.md). Weights are
part of the state, and updates happen per step by local co-activity rules.

See [Custom training loop](../advanced/custom-training-loop.md) for a chunk-wise/TBPTT loop and
[Advanced](../advanced/README.md) for the rest of the machinery.
