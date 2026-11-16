# neo-store — `.syn` format

NeuroLib weight storage: a binary container for weights and network graphs (primarily SNN).
An analogue of the **NIR** format (nodes + edges + parameters), but without HDF5 and with
a number of improvements.

## Quick start

```rust
use neo_store::*;

write_store_file("model.syn", WriteOptions::default(), |w| {
    w.set_meta("model.name", "mnist-snn")?;
    w.add_f32("fc1.weight", &[128, 784], TensorRole::Weight, &w1)?;
    w.add_f32_as("fc1.bias", &[128], TensorRole::Bias, DType::F16, &b1)?;   // compression to f16
    w.add_csr("fc2.weight", &csr, TensorRole::Weight, DType::F32)?;         // sparse weight

    let mut g = Graph::new();
    g.add_node(Node::input("in", &[784]))
     .add_node(Node::affine("fc1", "fc1.weight", "fc1.bias"))
     .add_node(Node::neuron("lif1", NodeOp::Lif, "lif1"))   // lif1.tau, lif1.r, lif1.v_leak, ...
     .add_node(Node::output("out", &[10]));
    g.chain(&["in", "fc1", "lif1", "out"]);
    w.set_graph(g)?;           // validated against the weights when writing completes
    Ok(())
})?;

let store = Store::open("model.syn")?;       // mmap, structure and index check
let w1 = store.read_f32("fc1.weight")?;      // verifies the tensor CRC on read
store.verify()?.into_result()?;              // full check of all data
```

CLI: `synctl info|ls|verify model.syn`.

## What is improved compared to NIR

| | NIR (HDF5) | neo-store |
|---|---|---|
| dependencies | HDF5 (C) | only `std`, `memmap2` (optional) |
| integrity | no | CRC-32C of header, index, and each tensor; file length; fingerprint |
| writing | — | streaming; atomic (tmp → fsync → rename); byte-for-byte reproducible |
| reading | via the library | mmap without copying; partial row reads; on-demand verification |
| types | f32/f64 | f64/f32/f16/bf16, i8..i64, u8..u64, bool, N-bit (1..7), 22 block formats GGUF |
| sparsity | no | CSR |
| graph | without validation | shape inference, validation against weights, cycles only through `recurrent`, edge delays, subgraphs, custom ops |
| NaN/Inf | written silently | fails by default with element index; also catches f16 overflow |
| large models | one file | shards + manifest, shard replacement/loss detected |
| versions | — | major/minor + required/optional feature flags |
| inspection | requires code | min/max/zeros/norm of each tensor in the index |

NIR nodes are supported under the same names and parameters:
`Input`, `Output`, `Linear`, `Affine`, `Conv1d`, `Conv2d`, `Flatten`, `SumPool2d`,
`AvgPool2d`, `Scale`, `Threshold`, `Delay`, `I`, `IF`, `LI`, `LIF`, `CubaLI`, `CubaLIF`
(`NodeOp::name()` / `NodeOp::from_name()`), plus `Alpha`, `Synaptic`, `Izhikevich`,
`AdaptiveLIF`, `Subgraph`, `Custom`. Shapes are one step, without a batch, as in NIR.

## File format (version 1.0, little-endian)

```
0 Header, 64 B: "\x89SYN\r\n\x1a\n"(8) | major u16 | minor u16 | required_flags u32 |
                optional_flags u32 | alignment u32 | created_unix_ms u64 | 0… | crc32c(0..60) u32
64 tensor data, each from an address multiple of alignment (default 64)
... Index (in the footer):
   meta: u32 n, n × (str key, value)
   tensors: u32 n, n × tensor record
   graph: bool, [graph]
… Footer, 64 B: "SYNEND\0"(8) | index_offset u64 | index_len u64 | index_crc u32 |
                tensor_count u32 | data_end u64 | file_len u64 | fingerprint u64 | 0(4) | crc32c(0..60) u32
```

Tensor record: `str name | u16 dtype | u8 role | u8 encoding(=0) | u8 ndim | ndim × u64 dim |
u64 offset | u64 nbytes | u32 crc32c | bool has_stats [min f64, max f64, zeros u64,
non_finite u64, l2 f64] | meta attrs`. Strings are `u32` length + UTF-8. The PNG-style magic
catches text transcoding. The reader rejects the file: with a different major version, with an
unknown required flag, with non-zero data encoding, with a length that does not match the
footer, or with overlapping or unaligned tensors. Unknown optional flags, roles, and attributes
are ignored.

Element type (`u16`): `1 f64, 2 f32, 3 f16, 4 bf16, 5 i64, 6 i32, 7 i16, 8 i8, 9 u64,
10 u32, 11 u16, 12 u8, 13 bool`, `0x0100\|n` — `n`-bit integers (bit stream, LSB first),
`0x0200\|id` — block format of GGUF with the same id as in GGUF and `neo-quantization`.

## Workspace integration

Add a crate to `members` of the root `Cargo.toml`:

```toml
members = [ …, "crates/neo-store" ]
```

and connect as `neo-store = { path = "../neo-store" }`. The crate does not depend on Burn
and neo-snn; adapters (“Burn layer ↔ storage tensors”) are written on top of `add_f32` /
`read_f32_into`.

## Known limitations

* No data compression: the encoding byte in the tensor record is reserved (`0` = raw).
* Quantized types are stored and returned as bytes; unpacking happens in `neo-quantization`.

## Cargo

```toml
[dependencies]
neo-store = { path = "crates/neo-store", features = ["mmap", "quantization"] }
```

## License

Dual-licensed under [MIT](LICENSE-MIT) or [Apache-2.0](LICENSE-APACHE).
