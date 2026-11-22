# Spikes & encoding

Spike trains in NeuroLib are `Tensor<3>`: `[batch, num_steps, features]` — the layout `rate_encode` produces
("`data` in [0, 1] → spike train in `num_steps` steps").

```rust
use neurolib::snn::encoding::rate_encode;
use burn::tensor::{Distribution, Tensor};

let device = Default::default();
let data = Tensor::<2>::random([batch, features], Distribution::Uniform(0.0, 1.0), &device);
let spikes = rate_encode(data, 100, &device); // [batch, 100, features]
```

Encodings (`neo-snn::encoding`):

- `rate_encode(data, num_steps, device)` — Poisson rate: spikes sampled as `r ≤ data` per step;
- `latency_encode` — early spikes for high intensities;
- `delta_encode` — first-spike-delta schemes for streaming;
- `periodic_encode` — periodic spike train;
- `burst_encode` — bursts (e.g. for bursting models).

Reading spikes back: `sum_over_time`, `mean_over_time`, `spike_counts`, `population_reduce` — see
[Model](../basic-workflow/model.md) and [Inference](../basic-workflow/inference.md).
