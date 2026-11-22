# neo-nn

Generic neural network layers for NeuroLib — quantized linear layers, mixture-of-experts,
transformer blocks, embeddings, normalization, and PEFT adapters. Built on [Burn](https://github.com/burn-rs/burn).

## What it provides

- **QuantizedLinear** — `y = x·Wᵀ + b` on packed weights (`neo-quantization` + `neo-synk` fused GEMM; see [quantization](../neo-quantization/README.md))
- **Mixture of Experts** — sparse and dense MoE with auxiliary loss and load balancing
- **Transformer blocks** — LLM-style blocks with GQA attention, pre-norm, FFN (MoE or dense)
- **Embeddings** — adaptive, factorized, Fourier positional embeddings
- **Normalization** — RMSNorm, LayerNorm, conditional norm
- **PEFT** — LoRA, DoRA, IA³ adapters with frozen base weights

## Modules

| Module | Contents |
|---|---|
| [`attention`](./src/attention) | Multi-head attention blocks |
| [`ffn`](./src/ffn) | Feed-forward blocks, gated FFN, MoE |
| [`norm`](./src/norm) | RMSNorm, LayerNorm, conditional norm |
| [`embedding`](./src/embedding) | Embeddings (adaptive, factorized, Fourier positional) |
| [`peft`](./src/peft) | LoRA, DoRA, IA³ adapters |
| [`transformer`](./src/transformer) | Transformer blocks (Llama/Qwen-style) |
| [`quantized`](./src/quantized) | Quantized layers |

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

## Book

[Neo-nn layers](../../neo-book/src/building-blocks/nn-layers.md)



