# neo-store — `syn` format (`.syn`)

NeuroLib weight storage: a binary container for weights **and the network graph** of neural
networks (primarily SNN). Analogue of the **NIR** format (nodes + edges + parameters), without
HDF5, with a number of improvements.

## Quick start

```rust
use neurolib::store::*;

write_store_file("model.syn", WriteOptions::default(), |w| {
    w.set_meta("model.name", "demo")?;
    w.add_f32("fc1.weight", &[128, 784], TensorRole::Weight, &w1)?;
    w.add_f32("fc1.bias", &[128], TensorRole::Bias, &b1)?;

    let mut g = Graph::new();
    g.add_node(Node::input("in", &[784]))
     .add_node(Node::affine("fc1", "fc1.weight", "fc1.bias"))
     .add_node(Node::output("out", &[128]));
    g.chain(&["in", "fc1", "out"]);
    w.set_graph(g)?;           // validates shapes when writing completes
    Ok(())
})?;

let store = Store::open("model.syn")?;        // mmap, structure and index check
let w1 = store.read_f32("fc1.weight")?;      // verifies the tensor CRC on read
let report = store.verify()?;                // full check of all data
```

CLI: `synctl info|ls|verify model.syn` (binary `synctl`, `crates/neo-store/src/bin/synctl.rs`).

## Features

| Feature | Description |
|---|---|
| `mmap` (default) | Reading via file-to-memory mapping (zero copy) — the only `unsafe` of the crate |
| `quantization` | Unpacking GGUF block formats via `neo-quantization` (`Store::read_dequantized_f32`) |

## What is improved compared to NIR

| | NIR (HDF5) | neo-store |
|---|---|---|
| dependencies | HDF5 (C-library) | only `std` (+ optional `memmap2`) |
| integrity | no | CRC-32C of header, index and **each tensor**, file length, fingerprint |
| writing | free-form | streaming, atomic (tmp → fsync → rename), byte-for-byte reproducible |
| reading | via the library | mmap without copying, partial row reads, on-demand verification |
| types | f32/f64 | f64/f32/f16/bf16, integer, bool, N-bit (ternary/binary), 22 block formats GGUF |
| sparsity | no | CSR (thinned and structurally plastic networks) |
| compression | gzip inside HDF5 | optional, per chunk: block LZ (LZ4 format) with byte swapping, no dependencies; rows are read without unpacking the entire tensor |
| quantized tensors | no | bytes as is; `read_bits` for N-bit, `read_dequantized_f32` (feature `quantization`) via `neo-quantization` for GGUF blocks |
| exchange with NIR | HDF5 | `Store::to_nir_json` and `StoreWriter::add_nir_json`: JSON in the spirit of NIR (graph and weights), without HDF5 |
| graph | nodes and edges without validation | shape inference, validation against weights, cycles only through `recurrent`, edge delays, subgraphs |
| NaN/Inf | are recorded silently | default write failure with index indication |
| large models | one file | shards with manifest and substitution check |
| versions | — | major/minor, required and optional feature flags |
| inspection | need code | min/max/zeros/norm of each tensor in the index without reading data, CLI `synctl` |

## NIR compatibility

NIR nodes are supported under the same names and parameters:
`Input`, `Output`, `Linear`, `Affine`, `Conv1d`, `Conv2d`, `Flatten`, `SumPool2d`,
`AvgPool2d`, `Scale`, `Threshold`, `Delay`, `I`, `IF`, `LI`, `LIF`, `CubaLI`, `CubaLIF`
(`NodeOp::name()` / `NodeOp::from_name()`), plus `Alpha`, `Synaptic`, `Izhikevich`,
`AdaptiveLIF`, `Subgraph`, `Custom`. Shapes are one step, without a batch, as in NIR.

## Book

[Saving & loading](../../neo-book/src/saving-and-loading.md),
[Weight storage](../../neo-book/src/building-blocks/storage.md)

