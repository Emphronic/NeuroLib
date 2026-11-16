# neo-quantization

Quantization formats and algorithms for neural network weights: GGUF block formats,
K-quants (super-block), IQ codebook series, NF4, ternary, and TQ packaging. Designed for
bitwise compatibility with [ggml/llama.cpp](https://github.com/ggerganov/llama.cpp).

## What it provides

- **GGUF simple blocks** — Q4_0, Q4_1, Q5_0, Q5_1, Q8_0
- **GGUF K-quants** — Q2_K, Q3_K, Q4_K, Q5_K, Q6_K, Q8_K (bitwise compatible with ggml-quants.c)
- **IQ series** — IQ2_XXS, IQ2_XS, IQ3_XXS, IQ3_S, IQ1_S, IQ1_M (codebook/lattice quantization)
- **NF4** — NormalFloat 4-bit (QLoRA-style)
- **Ternary** — {-1, 0, +1} at 2 bits/weight with base-3 packaging (TQ1_0/TQ2_0)
- **Calibration** — GPTQ and AWQ algorithms with Hessian-based weight selection
- **PackedBuffer** — runtime representation for quantized weights, readable by `neo-synk` GEMM kernels

## Third-party code

Portions of this crate are derived from [ggml/llama.cpp]
(https://github.com/ggerganov/llama.cpp), Copyright (c) the ggml/llama.cpp authors,
used under the MIT License. See `NOTICE` for the full list of derived files.

## Features

This crate has no optional features — all formats are available by default.

## Usage

```rust
use neo_quantization::{quantize, QuantizeConfig, PackedBuffer};

// Quantize f32 weights to Q4_0
let packed = quantize(weights, &QuantizeConfig::new(DType::Q4_0))?;

// Dequantize back to f32
let f32_weights = packed.dequantize()?;
```

## Cargo

```toml
[dependencies]
neo-quantization = { path = "crates/neo-quantization" }
```

## License

MIT. See `LICENSE` and `NOTICE` for details, including ggml/llama.cpp attribution.
