# Neo-nn layers (conventional NN)

The conventional neural network layers of NeuroLib — built on Burn — see also
[neo-nn README](../../../crates/neo-nn/README.md). Spiking NN lives in `neo-snn`.

## What it provides

- **QuantizedLinear** — `y = x·Wᵀ + b` on packed weights — see [Quantization](./quantization.md);
- **Mixture of Experts** — sparse and dense MoE with auxiliary loss and load balancing;
- **Transformer blocks** — LLM-style blocks with GQA attention, pre-norm, FFN (MoE or dense);
- **Embeddings** — adaptive, factorized, Fourier positional embeddings;
- **Normalization** — RMSNorm, LayerNorm, conditional norm;
- **PEFT** — LoRA, DoRA, IA³ adapters with frozen base weights;

## Modules

| Module | Contents |
|---|---|
| [`neo-nn::attention`](../../../crates/neo-nn/src/attention) | Multi-head attention blocks (spiking attention is `neo-snn::attention`) |
| [`neo-nn::ffn`](../../../crates/neo-nn/src/ffn) | Feed-forward blocks, gated FFN, MoE |
| [`neo-nn::norm`](../../../crates/neo-nn/src/norm) | RMSNorm, LayerNorm, conditional norm |
| [`neo-nn::embedding`](../../../crates/neo-nn/src/embedding) | Embeddings (adaptive, factorized, Fourier positional) |
| [`neo-nn::peft`](../../../crates/neo-nn/src/peft) | LoRA, DoRA, IA³ adapters |
| [`neo-nn::transformer`](../../../crates/neo-nn/src/transformer) | Transformer blocks (Llama/Qwen-style) |
| [`neo-nn::quantized`](../../../crates/neo-nn/src/quantized) | Quantized layers — see [Quantization](./quantization.md) |

## Usage

```rust
use burn::nn::LinearConfig;
use neurolib::nn::quantized::QuantizedLinear;
use neurolib::quant::QuantizeConfig;

// Quantized linear layer on packed weights (no matrix expansion in f32)
let linear = LinearConfig::new(4096, 4096).init(&device);
let q = QuantizedLinear::from_linear(&linear, &QuantizeConfig::int4_per_group_32());
let y = q.forward(x);
```
