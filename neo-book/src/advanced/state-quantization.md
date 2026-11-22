# State quantization

`neo-snn::quant` — quantization of the state of a neuron (QAT — quantization-aware training — for the membrane).
Types: `StateQuant`, `StateQuantScheme` — see the crate docs for the schemes.

This is the state-level counterpart of [Quantization](../building-blocks/quantization.md) (weight formats);
for the activation state (spikes, membrane), use `StateQuant`.
