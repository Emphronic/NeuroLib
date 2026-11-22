# Neuron models

Neuron models in `neo-snn::neuron` — config + layer; all are `SpikingLayer`s.

| Model | Note |
|---|---|
| `Leaky` (LIF) | `U' = β·U + I`, Heaviside spike at `U ≥ θ`; see [Model](../basic-workflow/model.md) |
| `Izhikevich` | two-variable spiking model, `IzhikevichPreset`s |
| `AdEx` | adaptive exponential leaky integrate-and-fire |
| `HodgkinHuxley` | conductance-based (`HodgkinHuxleyState`, `hh_steady_state`, ...) |
| `AdaptiveLeaky` (`RLeaky`), `RSynaptic`, `GradedLeaky` | adaptive / graded variants |
| `MultiStepLeaky`, `MultiStepRLeaky`, `Lapicque`, `Alpha`, `LeakyParallel` | other standard variants |

## `LeakyConfig` (LIF) — knobs

```rust
use neurolib::snn::neuron::LeakyConfig;

let mut cfg = LeakyConfig::new(0.9);      // β_init
cfg.threshold = 1.0;                     // spike threshold (if not learned)
cfg.learn_threshold = true;              // trainable θ (positivity via softplus)
cfg.reset_mode = ResetMode::Subtract;    // Subtract (default) | Zero | None
cfg.reset_order = ResetOrder::BeforeDecay; // snnTorch-compatible reset/decay order
cfg.inhibition = false;                  // winner-take-all inside the layer
cfg.graded_spikes_factor = 1.0;          // trainable amplitude of spikes
```

Reset semantics: `ResetMode::{Subtract (default), Zero, None}`; `ResetOrder::{BeforeDecay (default), AfterDecay}`
matches snnTorch (`U[t+1] = β·U[t] − θ·S[t] + I[t+1]`); `apply_reset` / `apply_reset_tensor` are the two implementations
(scalar vs. tensor threshold — the tensor variant avoids a GPU sync per step).

## States

`neo-snn::state` — `NeuronState`, `LayerState`, `apply_reset`, `apply_reset_tensor`, `hard_spike`, ... The state is
transferred through `step`/`forward_seq_from` by value; a state's `detach()` truncates BPTT between chunks
(`Detach` trait).

## Trainable threshold / graded spikes

Both `LeakyConfig.threshold` and `LeakyConfig.graded_spikes_factor` can be made trainable
(`learn_threshold`, `learn_graded_spikes_factor`) — implemented with softplus, so positivity is ensured
(`spike_with_threshold`, `multi_level_spike`, `apply_graded_spikes`).
