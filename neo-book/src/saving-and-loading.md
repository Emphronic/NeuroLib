# Saving & Loading

Weight saving/loading lives in two places:

- `neo-store` — the `.syn` container: weights + graph in one mmap-able binary file
  ([Weight storage](./building-blocks/storage.md)). `write_store_file` / `Store::open` / `Store::read_f32`,
  `StoreWriter`, `Compression`, shards, NIR JSON exchange.
- STDP layer checkpoints: `StdpLinearRecord` — STDP layers are saved via the standard Burn `Record` mechanism
  (see [STDP](./building-blocks/stdp.md)).

```rust
use neurolib::store::*;

// Write
write_store_file("model.syn", WriteOptions::default(), |w| {
    w.add_f32("fc1.weight", &[8, 4], TensorRole::Weight, &w)?;
    w.set_graph(g)?;
    Ok(())
})?;

// Read back
let store = Store::open("model.syn")?;
let weights = store.read_f32("fc1.weight")?;
```

See [Weight storage](./building-blocks/storage.md) for the full format description (CRC-32C integrity, mmap,
sharding, sparse, compression, NIR exchange).
