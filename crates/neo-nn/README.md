# neo-nn

Generic neural network layers for NeuroLib: quantized linear layers, mixture-of-experts,
transformer blocks, embeddings, normalization, and PEFT adapters. Built on [Burn](https://github.com/burn-rs/burn).

## What it provides

- **QuantizedLinear** — linear layer on packed weights (via `neo-quantization` + `neo-synk` fused GEMM)
- **Mixture of Experts** — sparse and dense MoE with auxiliary loss and load balancing
- **Transformer blocks** — LLM-style blocks with GQA attention, pre-norm, FFN
- **Embeddings** — adaptive, factorized, Fourier positional embeddings
- **Normalization** — RMSNorm, LayerNorm, conditional norm
- **PEFT** — LoRA, DoRA, IA³ adapters with frozen base weights

## Usage

```rust
use neo_nn::{QuantizedLinear, QuantizedLinearConfig};

// Quantized linear layer on packed weights
let qlinear = QuantizedLinearConfig::new(784, 128).init(&device);
let output = qlinear.forward(&input)?;
```

## Cargo

```toml
[dependencies]
neo-nn = { path = "crates/neo-nn" }
```

## License

MIT. See `LICENSE` for details.
