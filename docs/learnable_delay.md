# Learnable delays (`neo-snn::structural::network`)

Per-synapse trainable delays on `DynamicNetwork` — separate from STP (release
multiplier) and long-term weight plasticity.

See module docs of `crates/neo-snn/src/structural/network.rs` for the full
reference (types, formulas, state lifecycles).

## Design
- Legacy integer `Synapse.delay` remains the scheduling field; with learnable
  off, delivery is unchanged (legacy behavior).
- With `EngineConfig::learnable_delay = Some(cfg)` (`enabled: true`), the engine
  uses per-synapse `TrainableDelayState { delay, elig }` (telemetry:
  `learnable_delay(sid)`, `learnable_delay_elig(sid)`, `learnable_delays()`).
- `set_learnable_delay(sid, value)` — set the initial delay (clamped to
  `[min_delay, max_delay]`).
- Teaching signals: `set_teaching_signal(neuron, value)` /
  `set_teaching_signals(map)` — read by the engine at dispatch (three-factor rule).

## Delivery semantics (documented)
- Legacy: integer delay `d` — delivery timing parity with learnable integer
  delays (see `learnable_delay_integer_parity_with_legacy`).
- Learnable: float delay `d = floor(d) + frac` — the pulse splits into
  `(1 − frac)` and `frac` across two adjacent bins (see
  `learnable_delay_fractional_split`): integral (charge) preserved, causal,
  no lost events.

## Three-factor rule (documented)
At dispatch of a learnable spike of synapse `s` at step `t` with teaching signal
`d_s = teaching_signals[dst]`:

```text
elig[s]  = elig_decay · elig[s] + d_s
delay[s] = clamp(delay[s] + learning_rate · elig[s], min_delay, max_delay)
```

Zero teaching signal leaves the state unchanged
(see `learnable_delay_training_moves_delay`).

## Checkpoint
Version 7 (`FORMAT_VERSION`); engine config (`LearnableDelayConfig`) +
per-synapse `TrainableDelayState` in the network block. Old checkpoints →
`learnable_delay = None`.

## Examples
- `cargo run --example learnable_delay -p neo-snn`.

## Verification status
- `cargo test -p neo-snn --lib structural::tests::learnable_delay_*` — 6 pass
  (integer parity with legacy, fractional split, training, checkpoint roundtrip,
  config validation).
- Full suite: `cargo test -p neo-snn` — green.
- Pipeline integration: `PipelineCtx.learnable_delay` exposes the current delay
  to rules.
