//! Mixed dynamic network: builtin LIF, AdaptiveLIF, Izhikevich + two custom
//! models (FitzHugh–Nagumo + Adaptive Resonate) — without modifying the crate.
//!
//! Run: `cargo run --example mixed_custom_network -p neo-snn`

use std::any::Any;

use neo_snn::structural::custom::{
    CustomNeuronInstance, CustomStateEntry, DynamicNeuronModel, NeuronStepContext,
};
use neo_snn::structural::{
    AdaptiveLifParams, Birth, DynamicNetwork, EngineConfig, IzhikevichParams, LifParams, NeuronId,
    NeuronModel, NeuronPolarity, NeuronRole, Origin, StructuralError, StructuralResult, Topology,
    TopologyLimits,
};

// ---------------------------------------------------------------------
// FitzHugh–Nagumo custom model (same shape as in custom_dynamic_neuron.rs)
// ---------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq)]
struct FhnConfig {
    a: f32,
    b: f32,
    v_reset: f32,
    threshold: f32,
    w_incr: f32,
    dt: f32,
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
struct FhnState {
    v: f32,
    w: f32,
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
    fn cloned_box(&self) -> Box<dyn CustomStateEntry<f32>> {
        Box::new(self.clone())
    }
    fn state_eq_box(&self, other: &dyn CustomStateEntry<f32>) -> bool {
        other.as_any().downcast_ref::<FhnState>() == Some(self)
    }
}

#[derive(Debug, Clone)]
struct Fhn {
    cfg: FhnConfig,
}

impl DynamicNeuronModel<f32> for Fhn {
    type State = FhnState;
    fn model_id(&self) -> &'static str {
        "docs/fhn"
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
        let v_new = state.v
            + self.cfg.dt * (state.v - self.cfg.a * state.v.powi(3) / 3.0 - state.w + input);
        state.w += self.cfg.dt * self.cfg.b * (v_new + 0.7 - 0.2 * state.w);
        if v_new >= self.cfg.threshold {
            state.v = self.cfg.v_reset;
            state.w += self.cfg.w_incr;
            Ok(true)
        } else {
            state.v = v_new;
            Ok(false)
        }
    }
    fn membrane(&self, state: &Self::State) -> Option<f32> {
        Some(state.v)
    }
    fn threshold(&self) -> Option<f32> {
        Some(self.cfg.threshold)
    }
    fn validate(&self, _limits: &neo_snn::structural::ParamLimits<f32>) -> StructuralResult<()> {
        Ok(())
    }
}

// ---------------------------------------------------------------------
// Adaptive resonate-and-fire (1 state variable)
// ---------------------------------------------------------------------

#[derive(Debug, Clone, PartialEq)]
struct ResonateState {
    phase: f32,
}

impl CustomStateEntry<f32> for ResonateState {
    fn as_any(&self) -> &dyn Any {
        self
    }
    fn as_any_mut(&mut self) -> &mut dyn Any {
        self
    }
    fn model_id(&self) -> &'static str {
        "docs/adaptive_resonate"
    }
    fn schema_version(&self) -> u32 {
        1
    }
    fn is_finite(&self) -> bool {
        self.phase.is_finite()
    }
    fn membrane(&self) -> Option<f32> {
        Some(self.phase)
    }
    fn set_membrane(&mut self, value: f32) {
        self.phase = value;
    }
    fn cloned_box(&self) -> Box<dyn CustomStateEntry<f32>> {
        Box::new(self.clone())
    }
    fn state_eq_box(&self, other: &dyn CustomStateEntry<f32>) -> bool {
        other.as_any().downcast_ref::<ResonateState>() == Some(self)
    }
}

/// `phase' = phase + input + drive`; when `phase >= period` → spike, `phase -= period`.
#[derive(Debug, Clone)]
struct AdaptiveResonate {
    period: f32,
    drive: f32,
}

impl DynamicNeuronModel<f32> for AdaptiveResonate {
    type State = ResonateState;
    fn model_id(&self) -> &'static str {
        "docs/adaptive_resonate"
    }
    fn init_state(&self) -> Self::State {
        ResonateState { phase: 0.0 }
    }
    fn step(
        &self,
        input: f32,
        state: &mut Self::State,
        _ctx: &NeuronStepContext<f32>,
    ) -> StructuralResult<bool> {
        state.phase += input + self.drive;
        if state.phase >= self.period {
            state.phase -= self.period;
            Ok(true)
        } else {
            Ok(false)
        }
    }
    fn membrane(&self, state: &Self::State) -> Option<f32> {
        Some(state.phase)
    }
    fn threshold(&self) -> Option<f32> {
        Some(self.period)
    }
}

// ---------------------------------------------------------------------
// main: one mixed network, two custom models of different types
// ---------------------------------------------------------------------

fn main() -> StructuralResult<()> {
    let mut topo = Topology::new(TopologyLimits::default(), Default::default());
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
    let h_lif = topo.add_neuron(
        NeuronRole::Hidden,
        LifParams::new(0.9, 1.0),
        Origin::Initial,
        b,
    )?;
    let h_alif = topo.add_neuron(
        NeuronRole::Hidden,
        AdaptiveLifParams::new(0.9, 0.95, 1.0),
        Origin::Initial,
        b,
    )?;
    let h_izh = topo.add_neuron(
        NeuronRole::Hidden,
        IzhikevichParams::regular_spiking(),
        Origin::Initial,
        b,
    )?;
    let h_fhn = topo.add_custom_neuron(
        NeuronRole::Hidden,
        NeuronPolarity::Excitatory,
        Fhn {
            cfg: FhnConfig {
                a: 0.05,
                b: 0.25,
                v_reset: -1.2,
                threshold: 0.5,
                w_incr: 0.2,
                dt: 1.0,
            },
        },
        Origin::Manual,
        b,
    )?;
    let h_ar = topo.add_custom_neuron(
        NeuronRole::Hidden,
        NeuronPolarity::Excitatory,
        AdaptiveResonate {
            period: 3.0,
            drive: 0.6,
        },
        Origin::Manual,
        b,
    )?;
    let out = topo.add_neuron(
        NeuronRole::Output,
        LifParams::new(0.9, 1.0),
        Origin::Initial,
        b,
    )?;

    // Synapses: builtin → custom → custom → builtin, without knowing the types.
    for (src, dst) in [
        (i0, h_lif),
        (i0, h_alif),
        (i1, h_izh),
        (h_lif, h_fhn),
        (h_alif, h_fhn),
        (h_fhn, h_ar),
        (h_izh, h_ar),
        (h_ar, out),
    ] {
        topo.add_synapse(src, dst, 0.5, 0, Origin::Initial, b)?;
    }

    let mut net = DynamicNetwork::new(topo, EngineConfig::default())?;

    println!("--- model names (heterogeneous network) ---");
    for (label, id) in [
        ("i0 (LIF input)", i0),
        ("h_lif (LIF)", h_lif),
        ("h_alif (AdaptiveLIF)", h_alif),
        ("h_izh (Izhikevich)", h_izh),
        ("h_fhn (custom #1)", h_fhn),
        ("h_ar (custom #2)", h_ar),
        ("out (LIF)", out),
    ] {
        println!(
            "  {label}: {}",
            net.topology().neuron(id).unwrap().model.display_name()
        );
    }
    let _ = NeuronId(0);

    println!("--- run: periodic spikes into the network ---");
    let mut out_spikes = 0;
    for t in 0..60 {
        let s0 = (t % 5 < 2) as u8 as f32;
        let s1 = (t % 7 < 2) as u8 as f32;
        let o = net.step(&[s0, s1])?;
        out_spikes += o.output_spikes.iter().filter(|x| **x).count();
    }
    println!("output spikes over 60 steps: {out_spikes}");
    for (label, id) in [("h_fhn", h_fhn), ("h_ar", h_ar), ("h_izh", h_izh)] {
        let st = net.topology().neuron(id).unwrap();
        println!(
            "  {label}: spikes={} rate_ema={:.4}",
            st.stats.spikes, st.stats.rate_ema
        );
    }
    assert!(net.topology().neuron(h_fhn).unwrap().stats.spikes > 0);
    assert!(net.topology().neuron(h_ar).unwrap().stats.spikes > 0);

    // Mixed network without any structurally-relevant assumptions:
    // changing models through `Topology` works the same for builtin and custom.
    let sw = net.change_neuron_model(
        h_fhn,
        LifParams::new(0.9, 1.0),
        neo_snn::structural::StateMigration::Reset,
    )?;
    println!(
        "change h_fhn -> {}; changed={}",
        sw.new_model.display_name(),
        sw.changed
    );
    assert!(sw.changed);
    let restored = net.change_neuron_model(
        h_fhn,
        CustomNeuronInstance::new(Fhn {
            cfg: FhnConfig::default(),
        }),
        neo_snn::structural::StateMigration::Compatible,
    )?;
    println!(
        "change h_fhn back -> {}; changed={}",
        restored.new_model.display_name(),
        restored.changed
    );
    assert_eq!(
        restored.new_model,
        NeuronModel::Custom(CustomNeuronInstance::new(Fhn {
            cfg: FhnConfig::default()
        }))
    );

    let _ = StructuralError::UnknownCustomModel(String::new());
    println!("OK: mixed network with builtin + custom models runs and interoperates.");
    Ok(())
}
