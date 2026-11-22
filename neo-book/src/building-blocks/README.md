# Building Blocks

The pieces of NeuroLib, one page each:

| Chapter | Content |
|---|---|
| [Backend](./backend.md) | `ComputeTarget` — the one runtime name (wgpu/cpu/cuda/rocm) |
| [Spikes & encoding](./spikes.md) | Spike trains, `rate_encode`, `latency_encode`, `delta_encode`, `periodic_encode`, `burst_encode` |
| [Neuron models](./neurons.md) | LIF, Izhikevich, AdEx, HH, adaptive variants, state/reset semantics |
| [STDP](./stdp.md) | `StdpLinear`/`StdpConv*`, `StdpRule::{Pair, Triplet, RewardModulated}` |
| [Quantization](./quantization.md) | GGUF/K-quant/IQ/NF4/TQ formats, `QuantizeConfig`, runtime switching |
| [Weight storage](./storage.md) | The `.syn` container (graph + CRC-32C + mmap + shards) |
| [Viewing](./viewing.md) | min/max envelopes & pyramids for drawing |
| [Spike metrics](./spike-metrics.md) | Correlograms, ISI, firing rate, snippets, PCA for curation |
| [Neo-nn layers](./nn-layers.md) | Conventional NN layers: quantized linear, MoE, transformer blocks, attention, embeddings, norm, PEFT |
| [Stochastic Bayesian](./stochastic-bayesian.md) | Stochastic-threshold neurons — Bayesian threshold sampling, hardware randomness analogue, RE & NLL/ECE metrics |

The layers follow the `SpikingLayer` contract — see [Model](../basic-workflow/model.md).
