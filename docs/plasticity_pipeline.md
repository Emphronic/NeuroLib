# PlasticityPipeline (`neo-snn::structural::plasticity`)

Unified, per-synapse rule assignment + execution for the dynamic network
(`DynamicNetwork`). Rules are *assigned to synapses* and the engine applies
them after each step — one writer of weight updates, no double updates.

See module docs on `crates/neo-snn/src/structural/plasticity.rs` for the full
reference (formulas, state lifecycles, composition rules).

## Quick start

```rust
use neo_snn::structural::*;

let mut pl_cfg = PlasticityPipelineConfig::default();
pl_cfg.default_rules = vec![RuleEntry::new(RuleId::PairStdp)];
let mut net = DynamicNetwork::new(topo, EngineConfig::default()).unwrap();
net.set_plasticity_pipeline(Some(PlasticityPipeline::new(pl_cfg))).unwrap();
for step in 0..100 { net.step(&[1.0]).unwrap(); }
```

- Engine must be Off when a pipeline is attached
  (`StdpConfig::mode == Off`, `InhibitoryPlasticityConfig::enabled == false`);
  enforced with a typed error — no accidental double updates.
- `DynamicNetwork::apply_reward(r)` routes the reward into the pipeline's
  `RewardStdp` path when one is assigned.

## Assignment precedence (per synapse, in `resolve`)

1. `synapse_rules[id]` — full replacement list (empty = off);
2. first matching `class_rules` entry — `RuleClass` predicates
   (`ByPolarity`, `BySrcRole`, `ByDstRole`, `BySrcNeuron`, `ByDstNeuron`,
   `ByWeightRange`, `ByAge`);
3. `default_rules` — network-wide default.

## Rules

| Rule | Semantics (documented forms, see module docs) |
|---|---|
| `PairStdp` | Engine pair-STDP: `pot = a+·post_trace` on arrival, `dep = a−·pre_trace` on dst spike, `dw = pot − dep`; mirrors `functional::plasticity::pair_update`. |
| `TripletStdp` | Pfister–Gerstner: `pot2/dep2/pot3/dep3` from fast+slow pre/post traces; mirrors `triplet_update_with_f`. |
| `RewardStdp` | R-STDP: `elig += pot − dep` per step; `w += reward_lr·reward·elig` on pending reward (mirrors engine `apply_reward`). |
| `InhibitoryStdp` | Vogels (2011): `w = −clamp(−w + dm, 0, wmax)`, requires Dale; engine formula. |
| `Clopath` | Voltage-dependent STDP: LTP on arrival when filtered post voltage `> θ_LTP`; depression on src spike; documented family form. |
| `Bcm` | Canonical BCM: `dw = η·c·(c − θ̄_M)`, `θ̄_M = EMA_τ{c²}` (Wikipedia form). |
| `Eprop` | Delegated to the `EpropTrainer` owned by the pipeline; typed gate `UnsupportedEpropDynamics`. |
| `Custom(id)` | User rule, implement `PlasticityRule<T>`; registry in `custom_rules`. |

## Composition

- Rules run in list order; deltas summed with `RuleEntry.weight` multipliers;
  each delta scaled by `(1 + modulator_signal)`;
- total `dw` clamped to `PlasticityPipelineConfig::max_abs_delta` if set;
- written through `Topology::set_synapse_weight` once per synapse per step —
  single projection (ParamLimits + Dale).
- Per-rule state lives in `PlasticityPipeline::states()` (typed per rule).

## Modulators

`ModulatorEvent { channel, value, at_step, target, lifetime }` queued via
`PlasticityPipeline::apply_modulator`; decay per step; addressed per synapse or
per neuron (dst). Semantic channels: `Dopamine, Serotonin, Acetylcholine,
Noradrenaline, Novelty, PredictionError, Custom`.

## Checkpoint / resume

`PlasticityPipeline::to_bytes`/`from_bytes` (magic `PLP1`): config + per-synapse
rule states + modulator events. Custom rules/e-prop configs re-supplied at load.

## Examples

- `crates/neo-snn/examples/plasticity_pipeline.rs` — mixed hybrid example
  (default pair STDP + Vogels class rule + per-synapse Clopath).

## Verification status

- `cargo test -p neo-snn` (lib + integration + doctests): pass.
- Parity test: pipeline `PairStdp` == engine `Immediate` STDP (same formulas,
  `engine_immediate_matches_pipeline_pair`).
- Regression: no-pipeline legacy behavior unchanged (full existing suite green).
