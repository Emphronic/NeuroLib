# Custom training loop

The lower-level contract is `SpikingLayer` — [Model](../basic-workflow/model.md). `Session` is one wrapper that
gives every layer the same loop:

```rust
use neurolib::snn::{neuron::LeakyConfig, spiking_layer::Session, surrogate::Triangular};

let lif = LeakyConfig::new(0.9).init(&device);
let session = Session::new(lif, batch, &device);

for x_t in inputs {
    let spikes = session.step(x_t, &Triangular::new(1.0));
}
```

`Session` also has `Session::forward_seq(...)`. Between sequences: `session.detach_hidden()` (truncated BPTT),
`session.reset(batch, &device)` (new sequence).

Chunk-wise loop with truncated BPTT — the recommended pattern for long sequences:

```rust
use neurolib::snn::{neuron::LeakyConfig, spiking_layer::SpikingLayer, surrogate::Triangular};

// Any SpikingLayer
let lif = LeakyConfig::new(0.9).init(&device);
let mut state = lif.init_state(batch, &device);

for chunk in chunks {
    let (ys, next) = lif.forward_seq_from(chunk, state, &Triangular::new(1.0));
    // ... losses / optimizer step on this chunk ...
    state = next;
    state.detach(); // drive away the chunk — truncated BPTT (see the `Detach` trait)
}
```

For STDP layers: `StdpLinear` updates happen inside `step` — the state carries the weights. See
[STDP](../building-blocks/stdp.md).
