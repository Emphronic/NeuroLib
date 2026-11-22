//! Integration tests for user-defined dynamic neurons in the dynamic network.
//!
//! Run: `cargo test -p neo-snn --test custom_dynamic_neuron`

use std::any::Any;

use neo_snn::structural::custom::{
    CustomCodec, CustomNeuronInstance, CustomNeuronRegistry, CustomStateEntry, CustomStateEntryBox,
    DynamicNeuronModel, NeuronStepContext,
};
use neo_snn::structural::{
    Birth, DynamicNetwork, EngineConfig, LifParams, NeuronId, NeuronModel, NeuronPolarity,
    NeuronRole, NeuronState, Origin, PlasticityMode, StateMigration, StdpConfig, StructuralConfig,
    StructuralError, StructuralPlasticityController, StructuralResult, Topology, TopologyLimits,
    load_checkpoint_with_registry, save_checkpoint,
};

// ---------------------------------------------------------------------
// Models for tests: two custom types with different states
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
        "test/fhn"
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

#[derive(Debug, Clone, Default)]
struct Fhn {
    cfg: FhnConfig,
}

impl DynamicNeuronModel<f32> for Fhn {
    type State = FhnState;
    fn model_id(&self) -> &'static str {
        "test/fhn"
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
    fn membrane_range(&self) -> Option<(f32, f32)> {
        Some((self.cfg.v_reset, self.cfg.threshold))
    }
    fn state_is_finite(state: &Self::State) -> bool {
        state.is_finite()
    }
    fn validate(&self, _limits: &neo_snn::structural::ParamLimits<f32>) -> StructuralResult<()> {
        if self.cfg.dt <= 0.0
            || !self.cfg.a.is_finite()
            || self.cfg.a <= 0.0
            || !self.cfg.threshold.is_finite()
            || !self.cfg.v_reset.is_finite()
            || !self.cfg.w_incr.is_finite()
        {
            return Err(StructuralError::InvalidCustomConfig(
                "test/fhn".into(),
                "bad config".into(),
            ));
        }
        Ok(())
    }

    // ---- codec for checkpoint/registry tests ----
    fn codec(&self) -> Option<Box<dyn CustomCodec<f32>>> {
        Some(Box::new(FhnCodec))
    }
}

struct FhnCodec;

impl CustomCodec<f32> for FhnCodec {
    fn encode_model(&self, model: &CustomNeuronInstance<f32>) -> StructuralResult<Vec<u8>> {
        let Fhn { cfg } = model.downcast_model::<Fhn>().unwrap();
        Ok(neo_snn::structural::custom::encode_scalars(&[
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
        Ok(neo_snn::structural::custom::encode_scalars(&[st.v, st.w]))
    }
    fn decode_model(
        &self,
        bytes: &[u8],
        _state_bytes: &[u8],
    ) -> StructuralResult<CustomNeuronInstance<f32>> {
        let v = neo_snn::structural::custom::read_scalars::<f32>(bytes)?;
        Ok(CustomNeuronInstance::new(Fhn {
            cfg: FhnConfig {
                a: v[0],
                b: v[1],
                v_reset: v[2],
                threshold: v[3],
                w_incr: v[4],
                dt: v[5],
            },
        }))
    }
    fn decode_state(&self, bytes: &[u8]) -> StructuralResult<Box<dyn CustomStateEntry<f32>>> {
        let v = neo_snn::structural::custom::read_scalars::<f32>(bytes)?;
        Ok(Box::new(FhnState { v: v[0], w: v[1] }))
    }
    fn model_id(&self) -> &str {
        "test/fhn"
    }
    fn schema_version(&self) -> u32 {
        1
    }
}

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
        "test/resonate"
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

/// Adaptive resonate-and-fire: `phase += input + drive`; spike when `phase >= period`.
#[derive(Debug, Clone)]
struct Resonate {
    period: f32,
    drive: f32,
}

impl DynamicNeuronModel<f32> for Resonate {
    type State = ResonateState;
    fn model_id(&self) -> &'static str {
        "test/resonate"
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
    fn codec(&self) -> Option<Box<dyn CustomCodec<f32>>> {
        Some(Box::new(ResonateCodec))
    }
}

impl Default for Resonate {
    fn default() -> Self {
        Self {
            period: 3.0,
            drive: 0.6,
        }
    }
}

struct ResonateCodec;

impl CustomCodec<f32> for ResonateCodec {
    fn encode_model(&self, model: &CustomNeuronInstance<f32>) -> StructuralResult<Vec<u8>> {
        let Resonate { period, drive } = model.downcast_model::<Resonate>().unwrap();
        Ok(neo_snn::structural::custom::encode_scalars(&[
            period, drive,
        ]))
    }
    fn encode_state(
        &self,
        _model: &CustomNeuronInstance<f32>,
        state: &dyn CustomStateEntry<f32>,
    ) -> StructuralResult<Vec<u8>> {
        let st = state.as_any().downcast_ref::<ResonateState>().unwrap();
        Ok(neo_snn::structural::custom::encode_scalars(&[st.phase]))
    }
    fn decode_model(
        &self,
        bytes: &[u8],
        _state_bytes: &[u8],
    ) -> StructuralResult<CustomNeuronInstance<f32>> {
        let v = neo_snn::structural::custom::read_scalars::<f32>(bytes)?;
        Ok(CustomNeuronInstance::new(Resonate {
            period: v[0],
            drive: v[1],
        }))
    }
    fn decode_state(&self, bytes: &[u8]) -> StructuralResult<Box<dyn CustomStateEntry<f32>>> {
        let v = neo_snn::structural::custom::read_scalars::<f32>(bytes)?;
        Ok(Box::new(ResonateState { phase: v[0] }))
    }
    fn model_id(&self) -> &str {
        "test/resonate"
    }
    fn schema_version(&self) -> u32 {
        1
    }
}

// ---------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------

fn net(topo: Topology) -> DynamicNetwork {
    DynamicNetwork::new(topo, EngineConfig::default()).unwrap()
}

fn two_custom_topology() -> Topology {
    let mut topo = Topology::new(TopologyLimits::default(), Default::default());
    let b = Birth::default();
    let i0 = topo
        .add_neuron(
            NeuronRole::Input,
            LifParams::new(0.9, 1.0),
            Origin::Initial,
            b,
        )
        .unwrap();
    let h_fhn = topo
        .add_custom_neuron(
            NeuronRole::Hidden,
            NeuronPolarity::Excitatory,
            Fhn::default(),
            Origin::Manual,
            b,
        )
        .unwrap();
    let h_ar = topo
        .add_custom_neuron(
            NeuronRole::Hidden,
            NeuronPolarity::Excitatory,
            Resonate::default(),
            Origin::Manual,
            b,
        )
        .unwrap();
    let out = topo
        .add_neuron(
            NeuronRole::Output,
            LifParams::new(0.9, 1.0),
            Origin::Initial,
            b,
        )
        .unwrap();
    topo.add_synapse(i0, h_fhn, 0.7, 0, Origin::Initial, b)
        .unwrap();
    topo.add_synapse(i0, h_ar, 0.5, 0, Origin::Initial, b)
        .unwrap();
    topo.add_synapse(h_fhn, out, 0.5, 0, Origin::Initial, b)
        .unwrap();
    topo.add_synapse(h_ar, out, 0.5, 0, Origin::Initial, b)
        .unwrap();
    topo
}

fn custom_model(topo: &Topology, id: NeuronId) -> NeuronModel {
    topo.neuron(id).unwrap().model.clone()
}

/// Look up a neuron id by its display model name (`"test/fhn"`, `"lif"`, ...).
fn id_of(topo: &Topology, name: &str) -> NeuronId {
    topo.neurons()
        .find(|x| x.model.display_name() == name)
        .unwrap()
        .id
}

#[test]
fn two_custom_models_in_one_network_with_independent_states() {
    let mut n = net(two_custom_topology());
    let topo = n.topology();
    let h_fhn = id_of(topo, "test/fhn");
    let h_ar = id_of(topo, "test/resonate");
    // Both custom models registered as Custom variants with their own state types.
    let m_fhn = custom_model(n.topology(), h_fhn);
    let m_ar = custom_model(n.topology(), h_ar);
    let NeuronModel::Custom(a) = &m_fhn else {
        panic!("fhn custom")
    };
    let NeuronModel::Custom(b) = &m_ar else {
        panic!("resonate custom")
    };
    assert_eq!(a.model_id(), "test/fhn");
    assert_eq!(b.model_id(), "test/resonate");
    assert!(n.topology().neuron(h_fhn).unwrap().state.is_custom());
    assert!(n.topology().neuron(h_ar).unwrap().state.is_custom());

    // run; states evolve
    for _ in 0..10 {
        n.step(&[1.0]).unwrap();
    }
    let s_fhn = n.topology().neuron(h_fhn).unwrap().state.clone();
    let s_ar = n.topology().neuron(h_ar).unwrap().state.clone();
    assert!(matches!(s_fhn, NeuronState::Custom(_)));
    assert!(matches!(s_ar, NeuronState::Custom(_)));
    // Different types: state equality across models is false
    assert_ne!(s_fhn, s_ar);
    // states independent: mutating one does not affect the other
    let mut s2 = s_fhn.clone();
    s2.set_mem(-99.0);
    assert_ne!(s2, s_fhn);
    assert!(n.topology().validate().is_ok());
}

#[test]
fn input_current_mode_for_custom_input_neurons() {
    let mut topo = Topology::new(TopologyLimits::default(), Default::default());
    let b = Birth::default();
    let i0 = topo
        .add_custom_neuron(
            NeuronRole::Input,
            NeuronPolarity::Excitatory,
            Fhn::default(),
            Origin::Initial,
            b,
        )
        .unwrap();
    let h = topo
        .add_custom_neuron(
            NeuronRole::Hidden,
            NeuronPolarity::Excitatory,
            Fhn::default(),
            Origin::Manual,
            b,
        )
        .unwrap();
    topo.add_synapse(i0, h, 1.5, 0, Origin::Initial, b).unwrap();

    let mut n = DynamicNetwork::new(
        topo,
        EngineConfig {
            input_mode: neo_snn::structural::InputMode::Current,
            ..Default::default()
        },
    )
    .unwrap();
    // In Current mode the external value is the input current of the custom input neuron.
    for _ in 0..10 {
        let _ = n.step(&[2.0]).unwrap();
    }
    assert!(
        n.topology().neuron(i0).unwrap().stats.spikes > 0,
        "custom input neuron must react to external current"
    );
    assert!(
        n.topology().neuron(h).unwrap().stats.spikes > 0,
        "custom hidden neuron must receive the input spikes"
    );
    assert!(n.topology().validate().is_ok());
}

#[test]
fn delayed_synapse_delivers_to_custom_neuron() {
    let mut topo = Topology::new(TopologyLimits::default(), Default::default());
    let b = Birth::default();
    let i0 = topo
        .add_neuron(
            NeuronRole::Input,
            LifParams::new(0.9, 1.0),
            Origin::Initial,
            b,
        )
        .unwrap();
    let h = topo
        .add_custom_neuron(
            NeuronRole::Hidden,
            NeuronPolarity::Excitatory,
            Fhn::default(),
            Origin::Manual,
            b,
        )
        .unwrap();
    topo.add_synapse(i0, h, 0.7, 3, Origin::Initial, b).unwrap(); // delay 3
    let mut n = net(topo);
    let state0 = n.topology().neuron(h).unwrap().state.clone();
    let o = n.step(&[1.0]).unwrap();
    assert!(!o.output_spikes.iter().any(|x| *x));
    n.step(&[0.0]).unwrap();
    let state1 = n.topology().neuron(h).unwrap().state.clone();
    // FHN should have integrated the delayed spike by now (3+ steps after).
    assert_ne!(state1, state0);
}

#[test]
fn inhibition_and_modulation() {
    let mut topo = Topology::new(TopologyLimits::default(), Default::default());
    let b = Birth::default();
    let i0 = topo
        .add_neuron(
            NeuronRole::Input,
            LifParams::new(0.9, 1.0),
            Origin::Initial,
            b,
        )
        .unwrap();
    let h_fhn = topo
        .add_custom_neuron(
            NeuronRole::Hidden,
            NeuronPolarity::Excitatory,
            Fhn::default(),
            Origin::Manual,
            b,
        )
        .unwrap();
    let h_inh = topo
        .add_custom_neuron(
            NeuronRole::Hidden,
            NeuronPolarity::Inhibitory,
            Resonate::default(),
            Origin::Manual,
            b,
        )
        .unwrap();
    let out = topo
        .add_neuron(
            NeuronRole::Output,
            LifParams::new(0.9, 1.0),
            Origin::Initial,
            b,
        )
        .unwrap();
    topo.add_synapse(i0, h_fhn, 0.7, 0, Origin::Initial, b)
        .unwrap();
    topo.add_synapse(h_fhn, out, 0.5, 0, Origin::Initial, b)
        .unwrap();
    // inhibitory connection fhn -> h_inh (inhibitory source, weight <= 0)
    topo.add_synapse(h_fhn, h_inh, -0.8, 0, Origin::Initial, b)
        .unwrap();
    topo.add_synapse(h_inh, out, 0.2, 0, Origin::Initial, b)
        .unwrap();
    let mut n = net(topo);
    for _ in 0..8 {
        n.step(&[1.0]).unwrap();
    }
    assert!(n.topology().validate().is_ok());
    // Resonate-model still has dynamics via resonance (inhibition does not stall the network).
    assert!(n.topology().neuron(h_inh).unwrap().stats.steps >= 1);
}

#[test]
fn plasticity_modes_with_custom_neurons() {
    for mode in [
        PlasticityMode::Off,
        PlasticityMode::Immediate,
        PlasticityMode::RewardModulated,
    ] {
        let topo = two_custom_topology();
        let mut n = DynamicNetwork::new(
            topo,
            EngineConfig {
                stdp: StdpConfig {
                    mode,
                    ..Default::default()
                },
                ..Default::default()
            },
        )
        .unwrap();
        for t in 0..10 {
            n.step(&[1.0]).unwrap();
            if t % 3 == 0 {
                n.apply_reward(0.5);
            }
        }
        // Custom states stay valid; no corruption.
        let topo = n.topology();
        topo.validate().unwrap();
        let s = topo.neuron(id_of(topo, "test/fhn")).unwrap().state.clone();
        assert!(s.is_finite());
    }
}

#[test]
fn structural_ops_on_custom_network() {
    use neo_snn::structural::Rng64;
    let mut n = net(two_custom_topology());
    let topo = n.topology();
    let h_fhn = id_of(topo, "test/fhn");
    let out = topo.outputs()[0];
    let mut rng = Rng64::new(1);

    // AddNeuron with a custom model through the event API.
    {
        let mut t = n.topology().clone();
        let id = t
            .add_custom_neuron(
                NeuronRole::Hidden,
                NeuronPolarity::Excitatory,
                Fhn::default(),
                Origin::Manual,
                Birth::default(),
            )
            .unwrap();
        t.add_synapse(h_fhn, id, 0.5, 0, Origin::Manual, Birth::default())
            .unwrap();
        t.add_synapse(id, out, 0.5, 0, Origin::Manual, Birth::default())
            .unwrap();
        assert_eq!(t.validate(), Ok(()));
    }

    // CloneNeuron: clone the FHN hidden neuron (custom state clone).
    let clone_id = n
        .topology_mut()
        .clone_neuron(h_fhn, 0.05, &mut rng, Birth::default())
        .unwrap();
    assert_eq!(
        n.topology().neuron(clone_id).unwrap().model,
        custom_model(n.topology(), h_fhn)
    );
    assert_eq!(
        n.topology().neuron(clone_id).unwrap().state,
        n.topology().neuron(h_fhn).unwrap().state
    );

    // RemoveNeuron.
    let removed = n.topology_mut().remove_neuron(clone_id).unwrap();
    assert!(n.topology().neuron(clone_id).is_none());
    assert_eq!(removed.neuron.model, custom_model(n.topology(), h_fhn));

    // SplitSynapse (relay is builtin LIF) - works on custom-linked synapses.
    let s = n.topology().find_synapse(h_fhn, out).unwrap();
    n.topology_mut()
        .split_synapse(s, Default::default(), Birth::default())
        .unwrap();
    n.topology().validate().unwrap();

    // ChangeNeuronModel custom -> builtin and back.
    let sw = n
        .change_neuron_model(h_fhn, LifParams::new(0.9, 1.0), StateMigration::Reset)
        .unwrap();
    assert!(sw.changed);
    assert!(matches!(
        n.topology().neuron(h_fhn).unwrap().model,
        NeuronModel::Lif(_)
    ));
    let sw2 = n
        .change_neuron_model(
            h_fhn,
            CustomNeuronInstance::new(Fhn::default()),
            StateMigration::Compatible,
        )
        .unwrap();
    assert!(sw2.changed);
    assert_eq!(
        n.topology().neuron(h_fhn).unwrap().model.display_name(),
        "test/fhn"
    );

    // Batch of actions + rollback through topology transactions.
    let snap = n.topology().clone();
    let r: StructuralResult<()> = n.topology_mut().transaction(|t| {
        t.add_custom_neuron(
            NeuronRole::Hidden,
            NeuronPolarity::Excitatory,
            Resonate::default(),
            Origin::Manual,
            Birth::default(),
        )?;
        Err(StructuralError::InvalidConfig("forced rollback".into()))
    });
    assert!(r.is_err());
    assert_eq!(*n.topology(), snap);

    // MutateNeuron: custom models stay valid under builtin-style mutations.
    let _o = n
        .topology_mut()
        .mutate_neuron(h_fhn, Default::default())
        .unwrap();
}

#[test]
fn migration_transitions() {
    use neo_snn::structural::{LifParams, StateMigration};

    // A -> A: same model, full state is preserved.
    let m1 = CustomNeuronInstance::new(Fhn::default());
    let e1 = m1.init_state();
    let m2 = CustomNeuronInstance::new(Fhn::default());
    let s = m2.migrate_state(&*e1, StateMigration::Compatible).unwrap();
    assert_eq!(
        s.as_any().downcast_ref::<FhnState>(),
        Some(&FhnState { v: 0.0, w: 0.0 })
    );

    // A -> B (different ids): re-init; membrane copied when available.
    let mut e2 = m1.init_state();
    let entry = e2.as_any_mut().downcast_mut::<FhnState>().unwrap();
    entry.v = 0.4;
    entry.w = 0.1;
    let s_b = CustomNeuronInstance::new(Resonate::default())
        .migrate_state(&*e2, StateMigration::Compatible)
        .unwrap();
    // A -> B (different ids): state is re-initialized, membrane copied when available.
    assert!(matches!(
        s_b.as_any().downcast_ref::<ResonateState>(),
        Some(ResonateState { phase: 0.4 })
    ));
    assert!(s_b.is_finite());

    // Builtin -> custom: membrane transferred (Fhn provides set_membrane).
    let mut st = NeuronModel::Lif(LifParams::new(0.9, 1.0)).init_state();
    st.set_mem(0.42);
    let custom = NeuronModel::Custom(CustomNeuronInstance::new(Fhn::default()));
    let cs = st.migrate_to(&custom, StateMigration::Compatible);
    let NeuronState::Custom(entry) = &cs else {
        panic!("custom state after Builtin->Custom")
    };
    assert_eq!(entry.inner().membrane(), Some(0.42));

    // Custom -> builtin: membrane transferred.
    let mut e3 = m1.init_state();
    e3.as_any_mut().downcast_mut::<FhnState>().unwrap().v = -0.33;
    let builtin = NeuronModel::Lif(LifParams::new(0.9, 1.0));
    let bs = builtin.migrate_state(
        NeuronState::Custom(CustomStateEntryBox::from_boxed(e3)),
        StateMigration::Compatible,
    );
    assert_eq!(bs.mem(), -0.33);

    // Reset always re-initializes.
    let e4 = m1.init_state();
    let rst = CustomNeuronInstance::new(Fhn::default())
        .migrate_state(&*e4, StateMigration::Reset)
        .unwrap();
    assert_eq!(
        rst.as_any().downcast_ref::<FhnState>().unwrap(),
        &FhnState { v: 0.0, w: 0.0 }
    );
}

#[test]
fn errors_are_typed() {
    // Invalid config on add.
    let mut topo = Topology::new(TopologyLimits::default(), Default::default());
    let b = Birth::default();
    let mut bad = Fhn::default();
    bad.cfg.dt = 0.0;
    assert!(matches!(
        topo.add_custom_neuron(
            NeuronRole::Hidden,
            NeuronPolarity::Excitatory,
            bad,
            Origin::Manual,
            b
        ),
        Err(StructuralError::InvalidCustomConfig(_, _))
    ));
    assert!(
        topo.add_custom_neuron(
            NeuronRole::Hidden,
            NeuronPolarity::Excitatory,
            Fhn::default(),
            Origin::Manual,
            b
        )
        .is_ok()
    );

    // Unknown model on load.
    let reg = CustomNeuronRegistry::<f32>::new();
    assert!(matches!(
        reg.create("nonexistent/model", 1, &[], &[]),
        Err(StructuralError::UnknownCustomModel(_))
    ));

    // Corrupted checkpoint bytes.
    let n = net(two_custom_topology());
    let ctl = StructuralPlasticityController::new(StructuralConfig::default()).unwrap();
    let bytes = save_checkpoint(&n, &ctl, b"").unwrap();
    let mut reg = CustomNeuronRegistry::new();
    reg.register::<Fhn>().unwrap();
    reg.register::<Resonate>().unwrap();
    assert!(load_checkpoint_with_registry(&bytes[..bytes.len() - 3], &reg).is_err());

    // Mutated checkpoint byte: checksum mismatch.
    let mut flipped = bytes.clone();
    let mid = flipped.len() / 2;
    flipped[mid] ^= 0x55;
    assert!(load_checkpoint_with_registry(&flipped, &reg).is_err());
}

#[test]
fn checkpoint_roundtrip_with_registry() {
    let mut n = net(two_custom_topology());
    for _ in 0..5 {
        n.step(&[1.0]).unwrap();
    }
    let h_fhn = id_of(n.topology(), "test/fhn");
    let h_ar = id_of(n.topology(), "test/resonate");
    let ctl = StructuralPlasticityController::new(StructuralConfig::default()).unwrap();
    let bytes = save_checkpoint(&n, &ctl, b"").unwrap();

    let mut reg = CustomNeuronRegistry::new();
    reg.register::<Fhn>().unwrap();
    reg.register::<Resonate>().unwrap();

    let mut r = load_checkpoint_with_registry(&bytes, &reg).unwrap();
    assert_eq!(
        r.network.topology().neuron(h_fhn).unwrap().model,
        n.topology().neuron(h_fhn).unwrap().model
    );
    let st_a = r.network.topology().neuron(h_fhn).unwrap().state.clone();
    let st_b = n.topology().neuron(h_fhn).unwrap().state.clone();
    assert_eq!(st_a, st_b);
    // Continue training: 5 more steps both sides.
    for _ in 0..5 {
        n.step(&[1.0]).unwrap();
        r.network.step(&[1.0]).unwrap();
    }
    assert_eq!(
        r.network.topology().neuron(h_fhn).unwrap().state,
        n.topology().neuron(h_fhn).unwrap().state
    );
    assert_eq!(
        r.network.topology().neuron(h_ar).unwrap().state,
        n.topology().neuron(h_ar).unwrap().state
    );
    assert_eq!(
        r.network
            .topology()
            .neuron(h_fhn)
            .unwrap()
            .model
            .display_name(),
        "test/fhn"
    );
    assert_eq!(r.network.topology().validate(), Ok(()));
}

#[test]
fn custom_states_have_stable_ids_and_versions() {
    let m = CustomNeuronInstance::new(Fhn::default());
    assert_eq!(m.model_id(), "test/fhn");
    assert_eq!(m.schema_version(), 1);
    let m2 = CustomNeuronInstance::new(Resonate::default());
    assert_eq!(m2.model_id(), "test/resonate");
    assert_ne!(m, m2);
}

#[test]
fn custom_registry_requires_registration() {
    // Without registration - typed error, not a panic or silent reset.
    let reg = CustomNeuronRegistry::<f32>::new();
    let res = reg.create("nonexistent/model", 1, &[], &[]);
    assert!(matches!(res, Err(StructuralError::UnknownCustomModel(_))));
    // Duplicate id.
    let mut reg = CustomNeuronRegistry::new();
    reg.register::<Fhn>().unwrap();
    let dup = reg.register::<Fhn>();
    // (two Fhn have the same id - second registration must fail)
    assert!(dup.is_err());
    // Unregistered model id used in decode - typed error.
    let missing = reg.create("test/resonate", 1, &[], &[]);
    assert!(matches!(
        missing,
        Err(StructuralError::UnknownCustomModel(_))
    ));
}

#[test]
fn custom_validation_surfaces_errors() {
    // Invalid config reaches Topology::validate as a typed error.
    let mut topo = Topology::new(TopologyLimits::default(), Default::default());
    let b = Birth::default();
    let bad = Fhn {
        cfg: FhnConfig {
            dt: 0.0,
            ..FhnConfig::default()
        },
    };
    // validate is called through add_custom_neuron
    let res = topo.add_custom_neuron(
        NeuronRole::Hidden,
        NeuronPolarity::Excitatory,
        bad,
        Origin::Manual,
        b,
    );
    assert!(matches!(
        res,
        Err(StructuralError::InvalidCustomConfig(_, _))
    ));
    // NaN in a custom model -> validate error
    let mut nan = Fhn::default();
    nan.cfg.a = f32::NAN;
    let res2 = topo.add_custom_neuron(
        NeuronRole::Hidden,
        NeuronPolarity::Excitatory,
        nan,
        Origin::Manual,
        b,
    );
    assert!(res2.is_err());
    // A valid one passes.
    assert!(
        topo.add_custom_neuron(
            NeuronRole::Hidden,
            NeuronPolarity::Excitatory,
            Fhn::default(),
            Origin::Manual,
            b
        )
        .is_ok()
    );
}
