# neo-quantization

Quantization formats and algorithms for NeuroLib's neural network weights: GGUF block formats,
K-quants, the IQ codebook series, NF4, ternary/TQ packaging and INT1..INT8. Designed for bitwise
compatibility with [ggml/llama.cpp](https://github.com/ggerganov/llama.cpp) — parts are derived
from it (see `NOTICE`).

## What it provides

- **GGUF** — simple blocks `Q4_0`, `Q4_1`, `Q5_0`, `Q5_1`, `Q8_0` (+`Q8_1`) — simple block formats;
  **K-quants** — `Q2_K`..`Q8_K` — super-block formats (bitwise compatible with ggml-quants.c);
- **IQ series** — `IQ2_XXS`/`IQ2_XS`, `IQ3_XXS`/`IQ3_XS`, `IQ4_NL`, `IQ4_XS`, ... — codebook/lattice quantization;
- **NF4** — normal-float 4-bit (QLoRA-style);
- **Ternary** — `{-1, 0, +1}` at 2 bits/weight with base-3 packaging (`TQ1_0`/`TQ2_0`);
- **INT1..INT8** — 1-bit (ternary) to int8;
- **Calibration** — GPTQ and AWQ algorithms with Hessian-based weight selection;
- **`PackedBuffer`** — runtime representation of packed weights, readable by `neo-synk` GEMM kernels;
- **`CustomScheme`** — the main extension point: implement it and yours works wherever the built-ins do,
  including `WeightRegistry` and `convert_checkpoint`.

## Modules

| Module | Contents |
|---|---|
| [`dtype`](./src/dtype.rs) | `DType` — the GGUF block table (`block_bytes`, gguf type ids) |
| [`pack`](./src/pack) | `pack/gguf` (Q4_0..Q8_1), `pack/kquant` (Q2_K..Q8_K), `pack/iq`, `pack/tq` |
| [`quantize`](./src/quantize.rs) / [`dequantize`](./src/dequantize.rs) | single-entry quantize/dequantize |
| [`config`](./src/config.rs) | `QuantizeConfig` — rounding/clip config (`ternary_per_group`, `ternary_per_tensor`, `int4_per_group_32`, `bitnet_style`, ...) |
| [`scheme`](./src/scheme.rs) | `PackedScheme`, `ScaleGranularity`, ... |
| [`convert`](./src/convert.rs) | whole-checkpoint conversion: `convert_checkpoint`, `ConversionPlan`, `compute_stats` |
| [`switch`](./src/switch.rs) | `WeightRegistry` — runtime switching of schemes |
| [`custom`](./src/custom.rs) | `CustomScheme`, `QuantContext` |
| [`testkit`](./src/testkit.rs) | test helpers |

## Usage

```rust
use neurolib::quant::{QuantizeConfig, quantize, dequantize};

let weights: Vec<f32> = (0..1024).map(|i| (i as f32 - 512.0) * 0.01).collect();

// BitNet b1.58-style: 2 bits per weight + one scale per group of 128
let cfg = QuantizeConfig::ternary_per_group(128);
let packed = quantize(&weights, &cfg, 1024);
assert!(packed.effective_bits_per_weight() < 3.0);

// Dequantize back for verification
let restored = dequantize(&packed);
assert_eq!(restored.len(), weights.len());
```

Runtime scheme switching:

```rust
use neurolib::quant::{SwitchMode, WeightRegistry};

let registry = WeightRegistry::new(SwitchMode::Cached);
registry.register("layer0.weight", weights, 4096);
// later, without recalculation (already cached):
registry.activate("layer0.weight", &QuantizeConfig::ternary_per_tensor());
```

## Book

[Quantization](../../neo-book/src/building-blocks/quantization.md)

