# Custom dynamic neuron models (neo-snn)

NeuroLib has two independent extension mechanisms. Choose the one that fits the level
you want to touch:

| | Tensor API | Dynamic API (scalar, `DynamicNetwork`) |
|---|---|---|
| Model base | implement `SpikingLayer` | implement `DynamicNeuronModel` |
| State | any `Clone + Detach` tensors `[batch, units]` | any `Clone + PartialEq + Send + Sync` struct |
| Execution | Burn autodiff, GPU/CPU backends | `Scalar` (`f32`/`f64`), CPU event-driven |
| Spike | surrogate gradients through `spike_with_scalar_threshold` | hard spike returned as bool |
| Network | static dispatch via `SpikingLayer` | type-erased dispatch (`NeuronModel::Custom`) |
| Delays/STDP/stats/plasticity | n/a (feed-forward) | full support by the engine |

The two are not mixed: tensor models use `Tensor<2>` everywhere; scalar dynamic
models use `Scalar` (`f32`/`f64`) and the dynamic engine.

See:
- `crates/neo-snn/examples/custom_tensor_neuron.rs` — FitzHugh–Nagumo on Burn.
- `crates/neo-snn/examples/custom_dynamic_neuron.rs` — FitzHugh–Nagumo in `DynamicNetwork`.
- `crates/neo-snn/examples/mixed_custom_network.rs` — builtin + custom in one network.

---

## 1. Differences between the two APIs

`SpikingLayer` (tensor):

```rust
pub trait SpikingLayer {
    type Input: StepInput;
    type Output;
    type State: Clone + Detach;
    fn init_state(&self, batch: usize, device: &Device) -> Self::State;
    fn step<S: Surrogate + ?Sized>(&self, x: Self::Input, state: Self::State, surrogate: &S)
        -> (Self::Output, Self::State);
    // + forward_seq / forward_seq_from (default)
}
```

Implementations must provide `init_state`, `step`, and `forward_seq`/`forward_seq_from`
(the last two have defaults and run the loop of `step`). Implement `Detach` for the state
so truncated BPTT works. For lazy shapes (unknown width/spatial size), return
`Deferred::Pending` from `init_state` and finish it in the first `step` with
`Deferred::ready`.

`DynamicNeuronModel` (scalar, dynamic engine):

```rust
pub trait DynamicNeuronModel<T: Scalar>: Clone + Send + Sync + 'static {
    type State: CustomStateEntry<T> + Clone;
    fn model_id(&self) -> &'static str;
    fn schema_version(&self) -> u32 { 1 }
    fn init_state(&self) -> Self::State;
    fn step(&self, input: T, state: &mut Self::State, ctx: &NeuronStepContext<T>)
        -> StructuralResult<bool>;
    fn validate(&self, limits: &ParamLimits<T>) -> StructuralResult<()> { Ok(()) }
    // capabilities (optional):
    fn membrane(&self, state: &Self::State) -> Option<T> { None }
    fn threshold(&self) -> Option<T> { None }
    fn membrane_range(&self) -> Option<(T, T)> { None }
    fn state_is_finite(state: &Self::State) -> bool { true }
    fn migrate_from_custom(&self, source: &dyn CustomStateEntry<T>, policy) -> ...;
    fn crossover(&self, other: &Self, same_id: bool) -> ...;
    fn mutate(&mut self, rng: &mut Rng64) -> bool { false }
    fn codec(&self) -> Option<Box<dyn CustomCodec<T>>> { None }
    fn custom_params(&self) -> Vec<CustomParameter<T>> { vec![] }
}
```

The scalar model does NOT handle network mechanics: the engine delivers modulated
currents (`DynamicNetwork::step`), applies the Dale principle, delivers spikes
through synapses, applies synaptic delays, tracks statistics, runs STDP, and pushes
spikes into the history buffer.

## 2. Creating a scalar dynamic model

A model = (config struct with `DynamicNeuronModel::model_id() + `Default`, state
struct implementing `CustomStateEntry`, implementation of `DynamicNeuronModel`).

Example (from `examples/custom_dynamic_neuron.rs` — FitzHugh–Nagumo, two state variables):

```rust
#[derive(Debug, Clone, PartialEq)]
pub struct FhnState { pub v: f32, pub w: f32 }

impl CustomStateEntry<f32> for FhnState {
    fn as_any(&self) -> &dyn Any { self }
    fn as_any_mut(&mut self) -> &mut dyn Any { self }
    fn model_id(&self) -> &'static str { "docs/fhn" }
    fn schema_version(&self) -> u32 { 1 }
    fn is_finite(&self) -> bool { self.v.is_finite() && self.w.is_finite() }
    fn membrane(&self) -> Option<f32> { Some(self.v) }
    fn set_membrane(&mut self, value: f32) { self.v = value; }
    fn membrane_range(&self) -> Option<(f32, f32)> { Some((-1.2, 0.5)) }
    fn cloned_box(&self) -> Box<dyn CustomStateEntry<f32>> { Box::new(self.clone()) }
    fn state_eq_box(&self, other: &dyn CustomStateEntry<f32>) -> bool {
        other.as_any().downcast_ref::<FhnState>() == Some(self)
    }
}

#[derive(Debug, Clone)]
pub struct Fhn { pub cfg: FhnConfig }

impl DynamicNeuronModel<f32> for Fhn {
    type State = FhnState;
    fn model_id(&self) -> &'static str { "docs/fhn" }
    fn init_state(&self) -> Self::State { FhnState { v: 0.0, w: 0.0 } }
    fn step(&self, input: f32, state: &mut Self::State, _ctx) -> StructuralResult<bool> {
        // v' = v + dt·(v − a·v³/3 − w + I)
        let v_new = state.v + self.cfg.dt * (state.v - self.cfg.a * state.v.powi(3) / 3.0 - state.w + input);
        // w' = w + dt·b·(v + 0.7 − 0.2·w)
        state.w += self.cfg.dt * self.cfg.b * (v_new + 0.7 - 0.2 * state.w);
        // custom reset + spike rule
        if v_new >= self.cfg.threshold {
            state.v = self.cfg.v_reset;
            state.w += self.cfg.w_incr;
            Ok(true)
        } else {
            state.v = v_new;
            Ok(false)
        }
    }
    fn membrane(&self, state: &Self::State) -> Option<f32> { Some(state.v) }
    fn threshold(&self) -> Option<f32> { Some(self.cfg.threshold) }
    fn membrane_range(&self) -> Option<(f32, f32)> { Some((self.cfg.v_reset, self.cfg.threshold)) }
    fn validate(&self, _limits) -> StructuralResult<()> { /* check cfg */ Ok(()) }
    fn codec(&self) -> Option<Box<dyn CustomCodec<f32>>> { Some(Box::new(FhnCodec)) }
}
```

`CustomStateEntry` is the contract for the state: `Debug + Send + Sync + 'static`
supertraits, plus `as_any/as_any_mut` (downcasts), `model_id`, `schema_version`,
`is_finite`, `membrane`, `set_membrane`, and the box helpers `cloned_box` /
`state_eq_box`. A state without a membrane does NOT have to pretend to have one:
`membrane()` returns `Option`; GUI and monitoring handle `None`.

Note the separation: the model's `state_set_mem` (on `DynamicNeuronModel`) delegates
to `state.set_membrane` by default — keep `set_membrane` on the entry struct as the
physical operation.

## 3. Creating a tensor model

Implement `SpikingLayer` for your struct — see §1 for the trait shape. Use existing
helpers instead of re-implementing: `spike_with_scalar_threshold` /
`spike_with_threshold` (from `neo_snn::neuron`), `apply_reset`/`apply_reset_tensor`
(`neo_snn::state`), `Detach`, `SpikingLayerExt` (`forward_seq_tensor`), `Session`.
Do not copy these without a reason.

State: arbitrary tensors (`[batch, units]`) — see `FhnState` in
`examples/custom_tensor_neuron.rs`. For lazy shapes return `Deferred::Pending`.

## 4. Updating state in `step`

The user model computes dynamics from `input` (already modulated) and updates
`state` in place. For a two-variable FHN-like model:

```
v[t] = v[t-1] + dt * f(v, w, I)
w[t] = w[t-1] + dt * g(v, w)
```

## 5. Custom spike + reset rules

Detect the spike in `DynamicNeuronModel::step` and return `true`. The engine handles
everything else: spike delivery to receivers, STDP, statistics. You do NOT
mechanically apply `apply_reset` — a custom model may reset several variables
(FHN: `v ← v_reset`, `w += w_incr`).

## 6. Surrogates

Tensor models: any `S: Surrogate` (`ATan`, `FastSigmoid`, `Triangular`, ...).
Use `spike_with_scalar_threshold(mem, threshold, surrogate)` (or
`spike_with_threshold` for a trainable tensor threshold).

Scalar dynamic models: no surrogates — spikes are hard by nature
(`DynamicNeuronModel::step -> bool`).

## 7. Trainable parameters (tensor API)

Any `Param<Tensor<1>>` inside your `Module`. Optimize with Burn:
`loss.backward()` → `GradientsParams::from_grads(grads, &model)` →
`optim.step(lr, model, grads)` (see `examples/custom_tensor_neuron.rs`).
For scalar models: name-typed parameters via
`custom_params`/`get_custom_param`/`set_custom_param` and
`DynamicNeuronModel::mutate` (used by evolution), or use the builtin
`NeuronParam` path through the adapter.

## 8. Adding a model to `Topology`

```rust
let mut topo = Topology::new(TopologyLimits::default(), ParamLimits::default());
let b = Birth::default();

let i0 = topo.add_neuron(NeuronRole::Input, LifParams::new(0.9, 1.0), Origin::Initial, b)?;
let h = topo.add_custom_neuron(
    NeuronRole::Hidden,
    NeuronPolarity::Excitatory,
    Fhn::new(FhnConfig::default()),
    Origin::Manual,
    b,
)?;
let out = topo.add_neuron(NeuronRole::Output, LifParams::new(0.9, 1.0), Origin::Initial, b)?;
topo.add_synapse(i0, h, 0.6, 0, Origin::Initial, b)?;
topo.add_synapse(h, out, 0.8, 1, Origin::Initial, b)?;
```

Also usable through `add_neuron_with` with `NeuronModel::Custom(CustomNeuronInstance::new(m))`.

## 9. Mixed networks

Different models coexist: `DynamicNetwork::new(topo, EngineConfig::default())` —
see `examples/mixed_custom_network.rs` (LIF + AdaptiveLIF + Izhikevich + 2 customs).

## 10. Interaction with STDP

Custom models receive already-delivered currents (`DynamicNetwork::step_inner`).
Spikes, delays, excitatory/inhibitory currents, modulation — all handled by the
engine (`StdpConfig::{Off, Immediate, RewardModulated}`).

## 11. Structural plasticity

Custom models are first-class: `apply_action` (AddNeuron/CloneNeuron/SplitSynapse/
ChangeNeuronModel/RemoveNeuron/MutateNeuron), `apply_batch`, rollback via
`Topology::transaction`, `StructuralPlasticityController` — all work without
knowing the state type. `clone_neuron` copies the custom state (documented
behaviour); `change_neuron_model` uses `StateMigration` hooks.

## 12. State migration (`StateMigration`)

* `Reset` — re-initialize the state of the target model.
* `Membrane` — transfer the membrane value only, if available.
* `Compatible` — same model id: keep the full state; custom target: user hook
  (`DynamicNeuronModel::migrate_from_custom`); builtin↔custom: membrane transfer.
* `Scaled` — between-model rescale of the membrane by rest/fire ranges, if
  available; user hook otherwise.

Transition matrix (all tested): custom A → custom A (full state), custom A → custom B
(re-init + membrane), builtin → custom (membrane), custom → builtin (membrane),
reset/compatible/scaled.

## 13. Checkpoint & registry

```rust
struct FhnCodec;
impl CustomCodec<f32> for FhnCodec {
    fn encode_model(&self, model: &CustomNeuronInstance<f32>) -> StructuralResult<Vec<u8>> { ... }
    fn encode_state(&self, model: &CustomNeuronInstance<f32>, state: &dyn CustomStateEntry<f32>) -> StructuralResult<Vec<u8>> { ... }
    fn decode_model(&self, model_bytes: &[u8], _state_bytes: &[u8]) -> StructuralResult<CustomNeuronInstance<f32>> { ... }
    fn decode_state(&self, state_bytes: &[u8]) -> StructuralResult<Box<dyn CustomStateEntry<f32>>> { ... }
    fn model_id(&self) -> &str { "docs/fhn" }
    fn schema_version(&self) -> u32 { 1 }
}

impl DynamicNeuronModel<f32> for Fhn {
    fn codec(&self) -> Option<Box<dyn CustomCodec<f32>>> { Some(Box::new(FhnCodec)) }
    // ...
}

let mut registry = CustomNeuronRegistry::new();
registry.register::<Fhn>()?;

let bytes = save_checkpoint(&net, &ctl, b"")?;
let mut c = load_checkpoint_with_registry(&bytes, &registry)?.network;
```

Facts:
- Custom models without a codec: `save_checkpoint` returns a clear
  `StructuralError::CustomSerialization`, the neuron is NOT lost silently.
- Unknown model id: `StructuralError::UnknownCustomModel` — no silent reset.
- No global registry; `register` duplicates are rejected.
- `Population::save/load` support custom genomes: `save()`, `load_with_registry()`.

## 14. NEAT / genome support

`NodeGene.model` may hold `NeuronModel::Custom`. Genome serialization stores
`model_id + schema_version + config bytes`. Crossover:
- same id: user hook `DynamicNeuronModel::crossover`, standard-compatible by default;
- different ids: keep parent, or user strategy; no strategy → explicit skip (no crash);
- custom param mutation: `DynamicNeuronModel::mutate` + `custom_params`.

## 15. Telemetry & GUI

`telemetry::NodeView.model: NeuronModel` — custom models display
`NeuronModel::display_name()` ("docs/fhn"). Custom neurons show through the same
views; no fake LIF fallbacks. Structural tabs show model switches.

## 16. Limitations

- Scalar dynamic models run on CPU (`f32`/`f64` only). No GPU for `DynamicNetwork`.
- `NeuronModel`/`NeuronState` are not `Copy` anymore (custom state needs cloning).
- Custom models may be slower: virtual calls per step. Builtin models keep the
  fast static path.

## Izhikevich-style two-variable example

```rust
#[derive(Debug, Clone, PartialEq)]
pub struct IzhState { pub v: f32, pub u: f32 }
impl CustomStateEntry<f32> for IzhState { /* as_any, model_id = "my/izh" ... */ }

#[derive(Debug, Clone)]
pub struct Izh { pub a: f32, pub b: f32, pub c: f32, pub d: f32, pub threshold: f32 }

impl DynamicNeuronModel<f32> for Izh {
    type State = IzhState;
    fn model_id(&self) -> &'static str { "my/izh" }
    fn init_state(&self) -> Self::State { IzhState { v: self.c, u: 0.0 } }
    fn step(&self, input: f32, state: &mut Self::State, _ctx) -> StructuralResult<bool> {
        for _ in 0..2 { // substeps
            state.v += (0.04 * state.v * state.v + 5.0 * state.v + 140.0 - state.u + input) / 2.0;
        }
        state.u += self.a * (self.b * state.v - state.u);
        if state.v >= self.threshold {
            state.v = self.c;
            state.u += self.d;
            Ok(true)
        } else { Ok(false) }
    }
}
```

The library does not force the variable set — your model chooses it.
