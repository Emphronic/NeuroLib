//! Custom scalar dynamic neuron (FitzHugh–Nagumo) in a dynamic network,
//! without modifying the crate sources.
//!
//! The model implements [`DynamicNeuronModel`](neo_snn::structural::custom::DynamicNeuronModel):
//! arbitrary equations, arbitrary state, custom spike rule, user-defined
//! reset, named parameters, validation and an optional codec for checkpoints.
//!
//! Run: `cargo run --example custom_dynamic_neuron -p neo-snn`
//!
//! Demonstrated:
//! 1. custom model structure + state + equations;
//! 2. topology: builtin + custom neurons mixed;
//! 3. DynamicNetwork execution with spike delivery through synapses;
//! 4. statistics of the custom neuron;
//! 5. named parameters (get/set) + validation;
//! 6. model switch Custom ↔ LIF (state migration);
//! 7. checkpoint save/load through `CustomNeuronRegistry`;
//! 8. continuing execution after restore.

use std::any::Any;

use neo_snn::structural::custom::{
    CustomCodec, CustomNeuronInstance, CustomNeuronRegistry, CustomParameter, CustomStateEntry,
    DynamicNeuronModel, NeuronObservation, NeuronStepContext, encode_scalars, read_scalars,
};
use neo_snn::structural::{
    Birth, DynamicNetwork, EngineConfig, LifParams, NeuronModel, NeuronPolarity, NeuronRole,
    Origin, ParamLimits, StateMigration, StructuralConfig, StructuralError,
    StructuralPlasticityController, StructuralResult, Topology, TopologyLimits,
    load_checkpoint_with_registry, save_checkpoint,
};

// ---------------------------------------------------------------------
// FitzHugh–Nagumo: 2 state variables (v, w), user equations
// ---------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq)]
pub struct FhnConfig {
    pub a: f32,
    pub b: f32,
    pub v_reset: f32,
    pub threshold: f32,
    pub w_incr: f32,
    pub dt: f32,
}

impl Default for FhnConfig {
    fn default() -> Self {
        Self {
            a: 0.05,
            b: 0.25,
            v_reset: -1.2,
            threshold: 0.5,
            w_incr: 0.2,
            dt: 1.0,
        }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub struct FhnState {
    pub v: f32,
    pub w: f32,
}

impl CustomStateEntry<f32> for FhnState {
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
    fn model_id(&self) -> &'static str {
        "docs/fhn"
    }
    fn schema_version(&self) -> u32 {
        1
    }
    fn is_finite(&self) -> bool {
        self.v.is_finite() && self.w.is_finite()
    }
    fn membrane(&self) -> Option<f32> {
        Some(self.v)
    }
    fn set_membrane(&mut self, value: f32) {
        self.v = value;
    }
    fn membrane_range(&self) -> Option<(f32, f32)> {
        Some((-1.2, 0.5))
    }
    fn cloned_box(&self) -> Box<dyn CustomStateEntry<f32>> {
        Box::new(self.clone())
    }
    fn state_eq_box(&self, other: &dyn CustomStateEntry<f32>) -> bool {
        other.as_any().downcast_ref::<FhnState>() == Some(self)
    }
}

#[derive(Debug, Clone)]
pub struct Fhn {
    pub cfg: FhnConfig,
}

impl Fhn {
    pub fn new(cfg: FhnConfig) -> Self {
        Self { cfg }
    }
}

impl Default for Fhn {
    fn default() -> Self {
        Self::new(FhnConfig::default())
    }
}

impl DynamicNeuronModel<f32> for Fhn {
    type State = FhnState;

    fn model_id(&self) -> &'static str {
        "docs/fhn"
    }
    fn schema_version(&self) -> u32 {
        1
    }

    fn init_state(&self) -> Self::State {
        FhnState { v: 0.0, w: 0.0 }
    }

    fn step(
        &self,
        input: f32,
        state: &mut Self::State,
        _ctx: &NeuronStepContext<f32>,
    ) -> StructuralResult<bool> {
        // v' = v + dt·(v − a·v³/3 − w + I)
        let v_new = state.v
            + self.cfg.dt * (state.v - self.cfg.a * state.v.powi(3) / 3.0 - state.w + input);
        // w' = w + dt·b·(v + 0.7 − 0.2·w) — driven by the new v.
        state.w += self.cfg.dt * self.cfg.b * (v_new + 0.7 - 0.2 * state.w);
        // User spike rule: v ≥ threshold → spike, custom reset.
        if v_new >= self.cfg.threshold {
            state.v = self.cfg.v_reset;
            state.w += self.cfg.w_incr;
            Ok(true)
        } else {
            state.v = v_new;
            Ok(false)
        }
    }

    fn validate(&self, limits: &ParamLimits<f32>) -> StructuralResult<()> {
        let _ = limits;
        for v in [
            self.cfg.a,
            self.cfg.b,
            self.cfg.v_reset,
            self.cfg.threshold,
            self.cfg.w_incr,
            self.cfg.dt,
        ] {
            if !v.is_finite() || v.abs() > 1e6 {
                return Err(StructuralError::InvalidCustomConfig(
                    "docs/fhn".into(),
                    format!("non-finite or huge parameter {v}"),
                ));
            }
        }
        if self.cfg.dt.is_nan() || self.cfg.dt <= 0.0 {
            return Err(StructuralError::InvalidCustomConfig(
                "docs/fhn".into(),
                "dt must be > 0".into(),
            ));
        }
        if self.cfg.threshold.is_nan()
            || self.cfg.v_reset.is_nan()
            || self.cfg.threshold <= self.cfg.v_reset
        {
            return Err(StructuralError::InvalidCustomConfig(
                "docs/fhn".into(),
                "threshold must be greater than v_reset".into(),
            ));
        }
        Ok(())
    }

    fn membrane(&self, state: &Self::State) -> Option<f32> {
        Some(state.v)
    }
    fn threshold(&self) -> Option<f32> {
        Some(self.cfg.threshold)
    }
    fn membrane_range(&self) -> Option<(f32, f32)> {
        Some((self.cfg.v_reset, self.cfg.threshold))
    }
    fn state_is_finite(state: &Self::State) -> bool {
        state.is_finite()
    }
    fn observe(&self, state: &Self::State) -> NeuronObservation<f32> {
        NeuronObservation {
            membrane: Some(state.v),
            threshold: Some(self.cfg.threshold),
            ..Default::default()
        }
    }

    // ---- named parameters for evolution/telemetry (demo) ----
    fn custom_params(&self) -> Vec<CustomParameter<f32>> {
        vec![
            CustomParameter::new("a", self.cfg.a).mutable_random_walk(0.01),
            CustomParameter::new("threshold", self.cfg.threshold).mutable_random_walk(0.05),
            CustomParameter::new("w_incr", self.cfg.w_incr),
        ]
    }
    fn get_custom_param(&self, id: &str) -> Option<f32> {
        Some(match id {
            "a" => self.cfg.a,
            "threshold" => self.cfg.threshold,
            "w_incr" => self.cfg.w_incr,
            _ => return None,
        })
    }
    fn set_custom_param(&mut self, id: &str, value: f32) -> Option<f32> {
        match id {
            "a" => {
                self.cfg.a = value;
                Some(value)
            }
            "threshold" => {
                self.cfg.threshold = value;
                Some(value)
            }
            "w_incr" => {
                self.cfg.w_incr = value;
                Some(value)
            }
            _ => None,
        }
    }

    // ---- codec for checkpoints ----
    fn codec(&self) -> Option<Box<dyn CustomCodec<f32>>> {
        Some(Box::new(FhnCodec))
    }
}

// ---------------------------------------------------------------------
// Codec for the example model
// ---------------------------------------------------------------------

struct FhnCodec;

impl CustomCodec<f32> for FhnCodec {
    fn encode_model(&self, model: &CustomNeuronInstance<f32>) -> StructuralResult<Vec<u8>> {
        let Fhn { cfg } = model.downcast_model::<Fhn>().expect("docs/fhn");
        Ok(encode_scalars(&[
            cfg.a,
            cfg.b,
            cfg.v_reset,
            cfg.threshold,
            cfg.w_incr,
            cfg.dt,
        ]))
    }
    fn encode_state(
        &self,
        _model: &CustomNeuronInstance<f32>,
        state: &dyn CustomStateEntry<f32>,
    ) -> StructuralResult<Vec<u8>> {
        let st = state.as_any().downcast_ref::<FhnState>().unwrap();
        Ok(encode_scalars(&[st.v, st.w]))
    }
    fn decode_model(
        &self,
        model_bytes: &[u8],
        _state_bytes: &[u8],
    ) -> StructuralResult<CustomNeuronInstance<f32>> {
        let v = read_scalars::<f32>(model_bytes)?;
        if v.len() != 6 {
            return Err(StructuralError::CustomSerialization(
                "docs/fhn".into(),
                "bad config bytes".into(),
            ));
        }
        Ok(CustomNeuronInstance::new(Fhn::new(FhnConfig {
            a: v[0],
            b: v[1],
            v_reset: v[2],
            threshold: v[3],
            w_incr: v[4],
            dt: v[5],
        })))
    }
    fn decode_state(&self, state_bytes: &[u8]) -> StructuralResult<Box<dyn CustomStateEntry<f32>>> {
        let v = read_scalars::<f32>(state_bytes)?;
        if v.len() != 2 {
            return Err(StructuralError::CustomSerialization(
                "docs/fhn".into(),
                "bad state bytes".into(),
            ));
        }
        Ok(Box::new(FhnState { v: v[0], w: v[1] }))
    }
    fn model_id(&self) -> &str {
        "docs/fhn"
    }
    fn schema_version(&self) -> u32 {
        1
    }
}

// ---------------------------------------------------------------------
// main
// ---------------------------------------------------------------------

fn main() -> StructuralResult<()> {
    let mut topo = Topology::new(TopologyLimits::default(), ParamLimits::default());
    let b = Birth::default();

    let i0 = topo.add_neuron(
        NeuronRole::Input,
        LifParams::new(0.9, 1.0),
        Origin::Initial,
        b,
    )?;
    let i1 = topo.add_neuron(
        NeuronRole::Input,
        LifParams::new(0.9, 1.0),
        Origin::Initial,
        b,
    )?;
    let custom = topo.add_custom_neuron(
        NeuronRole::Hidden,
        NeuronPolarity::Excitatory,
        Fhn::new(FhnConfig::default()),
        Origin::Manual,
        b,
    )?;
    let out = topo.add_neuron(
        NeuronRole::Output,
        LifParams::new(0.9, 1.0),
        Origin::Initial,
        b,
    )?;
    topo.add_synapse(i0, custom, 0.6, 0, Origin::Initial, b)?;
    topo.add_synapse(custom, out, 0.8, 1, Origin::Initial, b)?;
    topo.add_synapse(i1, custom, 0.0, 2, Origin::Initial, b)?;

    let mut net = DynamicNetwork::new(topo, EngineConfig::default())?;
    println!(
        "network: builtin LIF inputs/outputs + custom model \"{}\" at {custom}",
        net.topology().neuron(custom).unwrap().model.display_name()
    );

    // ---- run the network: input spikes arrive, custom neuron integrates ----
    println!("--- run: 2 inputs spikes into the custom neuron ---");
    let mut output_spikes = 0;
    for t in 0..30 {
        let s0 = t % 5 < 2;
        let s1 = t % 7 < 2;
        let o = net.step(&[s0 as u8 as f32, s1 as u8 as f32])?;
        output_spikes += o.output_spikes.iter().filter(|x| **x).count();
    }
    println!("output spikes after 30 steps: {output_spikes}");
    let st = net.topology().neuron(custom).unwrap();
    println!(
        "custom neuron stats: spikes={} rate_ema={:.4}",
        st.stats.spikes, st.stats.rate_ema
    );
    println!("custom state: v={:.3}", st.state.mem());
    assert!(st.stats.spikes > 0);

    // ---- named parameters + validation ----
    let model = net.topology().neuron(custom).unwrap().model.clone();
    let NeuronModel::Custom(cm) = &model else {
        unreachable!()
    };
    println!(
        "named params: {:?}",
        cm.custom_params()
            .iter()
            .map(|p| (&p.id, p.value))
            .collect::<Vec<_>>()
    );
    assert_eq!(cm.get_custom_param("a"), Some(0.05));
    let mut cm = cm.clone();
    let set = cm.set_custom_param("a", 0.08);
    println!("set custom param 'a' -> {set:?} (expected Some(0.08))");
    assert_eq!(set, Some(0.08));

    // ---- model switch: Custom → LIF (state migration) ----
    println!("--- switch: Custom → LIF (StateMigration::Reset) ---");
    let sw = net.change_neuron_model(custom, LifParams::new(0.9, 1.0), StateMigration::Reset)?;
    println!(
        "changed={} old={} new={}",
        sw.changed,
        sw.old_model.display_name(),
        sw.new_model.display_name()
    );
    assert!(sw.changed);
    assert!(matches!(sw.new_model, NeuronModel::Lif(_)));

    // ---- model switch back: LIF → Custom with Compatible ----
    println!("--- switch: LIF → Custom (StateMigration::Compatible) ---");
    let sw2 = net.change_neuron_model(
        custom,
        CustomNeuronInstance::new(Fhn::new(FhnConfig::default())),
        StateMigration::Compatible,
    )?;
    println!(
        "changed={} new={}",
        sw2.changed,
        sw2.new_model.display_name()
    );

    // ---- checkpoint roundtrip through the registry ----
    println!("--- checkpoint: save → load (with registry) ---");
    let ctl = StructuralPlasticityController::new(StructuralConfig::default())?;
    let bytes = save_checkpoint(&net, &ctl, b"")?;
    let mut registry = CustomNeuronRegistry::new();
    registry.register::<Fhn>()?;
    let mut restored = load_checkpoint_with_registry(&bytes, &registry)?.network;
    let name = restored
        .topology()
        .neuron(custom)
        .unwrap()
        .model
        .display_name();
    println!("restored custom model: {name}");
    assert_eq!(name, "docs/fhn");
    // continue execution
    let o = restored.step(&[0.0, 1.0])?;
    println!("restored net step ok: output_spikes={:?}", o.output_spikes);

    println!("OK: custom dynamic neuron works in the mixed topology.");
    Ok(())
}
