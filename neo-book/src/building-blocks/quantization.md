# Quantization

`neo-quantization` — conversion of float weights to quantized formats and back, with fast scheme switching while
the application is running.

```rust
use neurolib::quant::{QuantizeConfig, quantize, dequantize};

let weights: Vec<f32> = (0..1024).map(|i| (i as f32 - 512.0) * 0.01).collect();

// BitNet b1.58-style: 2 bits per weight + one scale per group of 128.
let cfg = QuantizeConfig::ternary_per_group(128);
let packed = quantize(&weights, &cfg, 1024);
assert!(packed.effective_bits_per_weight() < 3.0);

// Dequantization — for verification and for the fallback path of the backend.
let restored = dequantize(&packed);
assert_eq!(restored.len(), weights.len());
```

Formats (22 block formats total):

- **GGUF**: `Q4_0`, `Q4_1`, `Q5_0`, `Q5_1`, `Q8_0`, `Q8_1`, ... (table `neo-quantization::dtype::DType`);
- **K-quants**: `Q2_K`..`Q8_K` — super-block formats;
- **IQ series**: `IQ4_NL`, `IQ4_XS`, `IQ2_XXS`/`IQ2_XS`, `IQ3_XXS`/`IQ3_XS`, codebook-based;
- **NF4** (nf4, 4-bit normal);
- **TQ** (ternary, `TQ1_0`/`TQ2_0`);
- **INT1..INT8** — ternary/1-bit to int8.

`PackedBuffer` — the packed representation consumed by `QuantizedLinear` and the fused GEMM cores.

`QuantizeConfig` — rounding/clip config: `ternary_per_group`, `ternary_per_tensor`, `int4_per_group_32`,
`int8_per_group_32`, `bitnet_style`... (see `QuantizeConfig` in the crate).

Runtime switching: `WeightRegistry` — `registry.register("layer0.weight", weights, 4096)`,
`registry.activate("layer0.weight", &cfg)` (`SwitchMode::Cached` = no recalculation, `SwitchMode::Eager`), plus
`ConversionPlan::bitnet_style(128)` and `convert_checkpoint(&tensors, &plan)` for whole checkpoints, and
`compute_stats(&converted)` → `compression_ratio()`, `avg_bits_per_weight()`.

The main extension point of the crate is `CustomScheme` — implement it and yours works wherever the built-ins do,
including `WeightRegistry` and `convert_checkpoint`.

Layered usage — `QuantizedLinear` (`neo-nn::quantized`): `y = x·Wᵀ + b` where `W` lives in `PackedBuffer` and does
not unfold to f32:

```rust
use burn::nn::LinearConfig;
use neurolib::nn::quantized::QuantizedLinear;
use neurolib::quant::QuantizeConfig;

let linear = LinearConfig::new(4096, 4096).init(&device);
let q = QuantizedLinear::from_linear(&linear, &QuantizeConfig::int4_per_group_32());
println!("{:.2} bits/weight", q.bits_per_weight());
let y = q.forward(x);
```

`ExecutionPolicy`: `Auto` (default — `Fused` if a fused core exists, else `Dequantize`), `Fused` (packed GEMM,
host-side, **no autodiff** — leaf tensor; for QLoRA-style training use `Dequantize` so gradients flow to `x`),
`Dequantize` (expand W to a tensor, normal matmul).

State quantization of the membrane: [State quantization](../advanced/state-quantization.md) (`neo-snn::quant`).
