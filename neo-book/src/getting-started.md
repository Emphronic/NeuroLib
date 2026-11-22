# Getting started

## Install

NeuroLib is a Cargo workspace (Rust ≥ 1.95, edition 2024). In a project:

```toml
[dependencies]
neurolib = { version = "0.1", features = ["full"] }
burn = "0.22"            # declared separately — see [crate map](./crate-map.md)
```

## Features

| Feature | What it enables |
|---|---|
| `ml` (default) | `snn`, `nn`, `quant`, `kernels`, `store` |
| `signal` | `core`, `io`, `view`, `synapse`, `zarr`, `neuro` |
| `full` | `ml` + `signal` + `demo` |
| `snn` / `nn` / `quant` / `kernels` / `store` / `core` / `io` / `view` / `synapse` | individual library crates |
| `wgpu` / `cuda` / `rocm` | GPU backends (`fusion` included) for `neo-synk`, `neo-core`, `neo-view` |
| `flex` | Burn CPU backend (convenient for tests) |
| `demo` | `neo_snn::demo` — demo training on synthetics |

## Examples

A minimal model is in [Model](./basic-workflow/model.md). A typical end-to-end flow is in
[Basic Workflow](./basic-workflow/README.md).
