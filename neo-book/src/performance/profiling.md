# Profiling & telemetry

Training telemetry is the live channel: `neo-snn::telemetry` — `channel(ChannelConfig::new("network name"))`
returns a `TrainTx` (sender into the training flow) and a `TrainRx` (reader: GUI or any other consumer).

- **Non-blocking**: the channel is `async_channel`; training never waits for the consumer — metric events are
  discarded when the queue is full (see `TrainControl::dropped`); an expensive spike snapshot is taken from the
  device only when `TrainTx::wants_frame()` allows.
- **Data source**: `neo-snn::functional::probe` — gradient norm (`grad_norm`) in `StepMetrics`; tensor summary
  (`ActivityStats`) in `TrainEvent::Activity`.
- **GUI**: the dashboard in the Training tab of `neo-gui` renders curves, raster, health. Structural plasticity
  events (`StructureReport`, `assess_structure`) flow through the same channel.

```rust
use neurolib::snn::telemetry::{channel, ChannelConfig, StepMetrics, Progress};

let (tx, rx) = channel(ChannelConfig::new("demo network"));
std::thread::spawn(move || {
    for epoch in 0..EPOCHS {
        for batch in 0..BATCHES {
            // ...learning step...
            let progress = Progress { epoch, epochs: EPOCHS, batch, batches: BATCHES };
            tx.step(StepMetrics { progress, loss, accuracy: Some(acc), grad_norm: Some(g) });
            if tx.wants_frame() {
                tx.spikes("hidden", progress, out.hidden_spikes, 0);
            }
            if tx.checkpoint() { return; } // Stop from GUI — along the step boundary
        }
    }
    tx.finish();
});
```
