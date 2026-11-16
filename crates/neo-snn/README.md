# neo-snn

Spiking neural network library built on [Burn](https://github.com/burn-rs/burn). LIF,
Izhikevich, AdEx, graded spikes, STDP plasticity, spike-driven attention, structural
plasticity, telemetry, and more.

## What it provides

- **Neuron models** — LIF, Izhikevich, AdEx, graded-spike, multi-step, parallel scan
- **Layers** — `StdpLinear`, `StdpConv1d`, `StdpConv2d`, `SynapticDelay`, `SpikingConv2d`
- **Plasticity** — STDP (pair/triplet/reward-modulated), structural growth/pruning
- **Attention** — spike-driven self-attention, Spikformer blocks, MLP
- **Surrogates** — triangular, fast sigmoid, ATan, Heaviside, stochastic spike operator
- **Structural plasticity** — neuron/synapse growth, Dale's principle, evolution, fixed-shape networks
- **Telemetry** — training events, structure/evolution snapshots, weight distributions
- **Encoding** — rate, latency, delta, periodic, burst spike encoding

## Features

| Feature | Description |
|---------|-------------|
| `demo` | Enable `neo_snn::demo` training loop (requires `burn/optim`) |

## Usage

```rust
use neo_snn::{LeakyConfig, Leaky, SpikingLayer};

let lif = LeakyConfig::new(128).init(&device);
let (spikes, state) = lif.step(input, &init_state, &surrogate)?;
```

## Cargo

```toml
[dependencies]
neo-snn = { path = "crates/neo-snn", features = ["demo"] }
```

## License

MIT. See `LICENSE` for details.
