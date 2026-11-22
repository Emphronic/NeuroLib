# Weight storage (.syn)

`neo-store` — the `.syn` container: a binary container for weights **and the topology graph** of neural networks
(primarily SNN). Conceived as an analogue of the **NIR** format, without HDF5, with improvements — the
comparison table is in `neo_store`'s crate docs.

```rust
use neurolib::store::*;

let w1 = vec![0.1f32; 8 * 4];
write_store_file("model.syn", WriteOptions::default(), |w| {
    w.set_meta("model.name", "demo")?;
    w.add_f32("fc1.weight", &[8, 4], TensorRole::Weight, &w1)?;
    w.add_f32("fc1.bias", &[8], TensorRole::Bias, &[0.0; 8])?;

    let mut g = Graph::new();
    g.add_node(Node::input("in", &[4]))
        .add_node(Node::affine("fc1", "fc1.weight", "fc1.bias"))
        .add_node(Node::output("out", &[8]));
    g.chain(&["in", "fc1", "out"]);
    w.set_graph(g)?; // validates shapes when writing finishes
    Ok(())
})?;

let store = Store::open("model.syn")?;
let weights = store.read_f32("fc1.weight")?;
```

Features:

- **Integrity**: CRC-32C of header, index and each tensor, file length, fingerprint — `Store::verify` /
  `VerifyReport` for on-demand verification.
- **Writing**: streaming, atomic (tmp → fsync → rename), byte-for-byte reproducible.
- **Reading**: mmap without copying, partial row reads, on-demand verification.
- **Types**: f64/f32/f16/bf16, integer, bool, N-bit (ternary/binary), 22 block formats (GGUF).
- **Sparsity**: CSR (`SparseCsr`) for thinned/structurally plastic networks.
- **Compression**: optional, per chunk — block LZ (LZ4 format) with byte swapping, no dependencies; CRC over the
  unpacked data (`Compression::Auto` compresses when the first chunk wins at least 10%).
- **Quantized tensors**: stored as bytes, `read_bits` for N-bit; `read_dequantized_f32` (feature `quantization`) via
  `neo-quantization` for GGUF blocks.
- **Graph**: `Node::{input, output, affine, ...}` with shape inference and cycle checks, edge delays, subgraphs.
- **Shards**: `ShardedStore` / `ShardedWriter` / `ShardedSummary` for large models, with manifest and substitution
  check.
- **Exchange with NIR**: `Store::to_nir_json`, `nir_json_file_to_syn`, `NirJson`, `NirImportOptions`.
- **STDP layers**: `StdpLinearRecord` etc. — "Saving/loading STDP layers via the standard Burn `Record`" — see
  [STDP](./stdp.md).
- **CLI**: `synctl` — inspect min/max/zeros/norm of each tensor in the index.

```rust
use neurolib::store::*;

// Compression on write
let opts = WriteOptions {
    compression: Compression::Auto,
    ..WriteOptions::default()
};
```
