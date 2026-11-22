# Why NeuroLib?

NeuroLib was built for the niche where SNN is used: a framework that is *not* just a copy of the backprop ML stack.
What is different:

- **SNN first.** Neuron models (LIF, Izhikevich, AdEx, Hodgkin–Huxley, adaptive variants), spike encodings (rate,
  latency, delta, burst, periodic), surrogate gradients, STDP and reward-modulated STDP as first-class training rules.
- **Explicit state.** Layers carry state between steps: one trait — [`SpikingLayer`](./building-blocks/spikes.md)
  (`init_state` / `step` / `forward_seq`). Stateful layers can also be driven through a wrapper (`Session`) — one API
  for every layer.
- **Quantization as a first-class feature.** GGUF, K-quant, IQ, NF4, TQ formats, runtime scheme switching via
  `WeightRegistry`, and fused quantized matmul directly from packed weights.
- **Storage as a format.** `neo-store`'s `.syn` container: a graph of nodes + weights in one mmap-able binary file with
  CRC-32C integrity, sharding and NIR JSON exchange.
- **Signal-first reading.** `neo-core` knows about samples, sample rates, time ranges, channels and memory layouts;
  `neo-io` reads recordings; `neo-view` prepares min/max envelopes for drawing.

The pieces fit together as a workspace — see the [crate map](./crate-map.md).
