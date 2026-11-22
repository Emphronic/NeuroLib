# Structural plasticity

Two kinds in NeuroLib:

- `neo-snn::structural` — **structural plasticity on a dynamic graph (CPU)**: growth and removal of neurons and
  connections, driven through telemetry (`TrainTx::structural`, `StructureReport` with counters, new events with
  reasons, loss before/after, graph snapshot; `assess_structure` — a verbal verdict like "growth without
  improvement"). Analogy: evolution of a spiking network's topology.
- `neo-snn::fixed_shape` — **structural plasticity on Burn layers of fixed shape** (masks and reserve neurons):
  sparsity masks + extra reserve neurons that can be activated.

State quantization of the membrane is a separate topic — [State quantization](./state-quantization.md).
