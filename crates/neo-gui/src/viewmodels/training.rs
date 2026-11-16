//! The Training workspace's view model: the run (a [`TrainingState`] fed from a training thread
//! over `neo_snn::telemetry`), the tab shown, and the images drawn from it (one slot each on the
//! work pool).
//!
//! The training runs on its own thread and writes to a channel; a task here reads the channel
//! (`async_channel`, awaited: nothing polls) and folds each batch of events into the state. An
//! image is asked for on change, at most every [`MIN_RENDER_GAP`] (a thousand events a second
//! cost thirty renders), only for the tab shown, and on the work pool, like the trace views'.

use std::sync::Arc;
use std::time::{Duration, Instant};

use burn::tensor::Device;
use gpui_kit::{App, Context, Entity, RenderImage, Subscription, Task};
use neo_snn::demo::{self, DemoConfig};
use neo_snn::telemetry::evolution_demo::{self, DemoConfig as EvolutionDemoConfig};
use neo_snn::telemetry::structural_demo::{self, DemoConfig as StructuralDemoConfig};
use neo_snn::telemetry::{ChannelConfig, TrainControl, TrainEvent, TrainTx, channel};

use crate::engine::palette::Palette;
use crate::engine::training::{
    BrainCamera, BrainInfo, BrainLayout, CountsRequest, CurvePanel, CurveRequest, DashboardInfo,
    EvolutionRequest, NetCamera, NetLayout, RasterLayout, RasterRequest, RasterStats, RunStatus,
    TopologyLayout, TopologyRequest, TrainingState, render_brain, render_counts, render_curves,
    render_dashboard, render_evolution, render_network, render_raster, render_topology,
};
use crate::engine::work_pool::JobKey;
use crate::store::{AppEvent, Store};

use super::services::Services;

/// Work-pool owner of this view model's renders (trace views use small view ids).
pub const TRAINING_OWNER: u64 = 1 << 40;
/// Renders at most this often.
pub const MIN_RENDER_GAP: Duration = Duration::from_millis(33);
/// Events applied per update of the view model (the channel is drained in batches).
const BATCH: usize = 256;
const TITLE: &str = "two-layer SNN synthetic";
const STRUCTURAL_TITLE: &str = "dynamic SNN structural plasticity";
const EVOLUTION_TITLE: &str = "evolution of architectures XOR on a spike network";
/// The fitness at which the evolution demo stops (drawn as the target line).
const EVOLUTION_TARGET: f32 = 1.05;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum TrainingTab {
    #[default]
    Network,
    Brain,
    Dashboard,
    Curves,
    Raster,
    Structure,
    Evolution,
    Health,
}

impl TrainingTab {
    pub const ALL: [TrainingTab; 8] = [
        TrainingTab::Network,
        TrainingTab::Brain,
        TrainingTab::Dashboard,
        TrainingTab::Curves,
        TrainingTab::Raster,
        TrainingTab::Structure,
        TrainingTab::Evolution,
        TrainingTab::Health,
    ];

    pub fn title(self) -> &'static str {
        match self {
            TrainingTab::Network => "Network",
            TrainingTab::Brain => "Brain",
            TrainingTab::Dashboard => "Dashboard",
            TrainingTab::Curves => "Curves",
            TrainingTab::Raster => "Spike raster",
            TrainingTab::Structure => "Structure",
            TrainingTab::Evolution => "Evolution",
            TrainingTab::Health => "Health",
        }
    }

    pub fn hint(self) -> &'static str {
        match self {
            TrainingTab::Brain => {
                "Which neurons are in use: regions, usage of every neuron, the synapses that fired in order, activity over time"
            }
            TrainingTab::Network => {
                "The network as it learns: weights as glowing curves, spikes as beads. Wheel zooms, drag pans, click a neuron"
            }
            TrainingTab::Dashboard => {
                "Everything at once: network, spikes, accuracy, weights, firing rates, state space"
            }
            TrainingTab::Curves => "Loss, accuracy and gradient norm over the run",
            TrainingTab::Raster => {
                "Which neurons fired when, per layer; scrub back through earlier frames"
            }
            TrainingTab::Structure => {
                "Neurons and synapses over the run, every structural change with its reason, and the graph as it is now"
            }
            TrainingTab::Evolution => {
                "Architecture search: fitness of the best, mean and champion per generation, the size of the best network, and the species"
            }
            TrainingTab::Health => "What the activity says about the network, and how to fix it",
        }
    }

    /// The images this tab shows (a tab without any is plain elements).
    fn slots(self) -> &'static [PlotSlot] {
        match self {
            TrainingTab::Network => &[PlotSlot::Network],
            TrainingTab::Brain => &[PlotSlot::Brain],
            TrainingTab::Dashboard => &[PlotSlot::Dashboard],
            TrainingTab::Curves => &[PlotSlot::Curves],
            TrainingTab::Raster => &[PlotSlot::Raster],
            TrainingTab::Structure => &[PlotSlot::Counts, PlotSlot::Topology],
            TrainingTab::Evolution => &[PlotSlot::Evolution],
            TrainingTab::Health => &[],
        }
    }
}

/// One image drawn on the work pool. A tab shows one or more of them.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PlotSlot {
    Curves,
    Raster,
    /// The counters over the run (Structure tab, left).
    Counts,
    /// The graph of the network (Structure tab, right).
    Topology,
    /// Fitness, size of the best network and species per generation (Evolution tab).
    Evolution,
    /// The all-in-one overview (Dashboard tab).
    Dashboard,
    /// The glowing network (Network tab).
    Network,
    /// The map of which neurons are in use (Brain tab).
    Brain,
}

impl PlotSlot {
    const COUNT: usize = 8;

    fn index(self) -> usize {
        self as usize
    }
}

pub struct CurvesShown {
    pub image: Arc<RenderImage>,
    pub panels: Vec<CurvePanel>,
}

pub struct BrainShown {
    pub image: Arc<RenderImage>,
    pub layout: BrainLayout,
    pub info: BrainInfo,
}

pub struct NetworkShown {
    pub image: Arc<RenderImage>,
    /// Where the layers and neurons are in the image: the labels and the clicks use it.
    pub layout: NetLayout,
}

pub struct DashboardShown {
    pub image: Arc<RenderImage>,
    pub info: DashboardInfo,
}

pub struct RasterShown {
    pub image: Arc<RenderImage>,
    pub layout: RasterLayout,
    pub stats: RasterStats,
}

pub struct CountsShown {
    pub image: Arc<RenderImage>,
    pub panels: Vec<CurvePanel>,
}

pub struct EvolutionShown {
    pub image: Arc<RenderImage>,
    pub panels: Vec<CurvePanel>,
}

pub struct TopologyShown {
    pub image: Arc<RenderImage>,
    pub layout: TopologyLayout,
}

enum Delivered {
    Brain(BrainShown),
    Network(NetworkShown),
    Dashboard(DashboardShown),
    Curves(CurvesShown),
    Raster(RasterShown),
    Counts(CountsShown),
    Topology(TopologyShown),
    Evolution(EvolutionShown),
}

/// What a render was last asked for, so an unchanged view is not drawn again.
#[derive(Clone, PartialEq)]
struct RenderKey {
    revision: u64,
    width: u32,
    height: u32,
    scale_bits: u32,
    /// The network view's camera and selected neuron; `None` on the other tabs.
    net: Option<([u32; 5], Option<(usize, usize)>)>,
    palette: Palette,
}

pub struct TrainingVm {
    store: Entity<Store>,
    pub state: TrainingState,
    pub tab: TrainingTab,
    control: Option<TrainControl>,
    pub curves: Option<CurvesShown>,
    pub raster: Option<RasterShown>,
    pub counts: Option<CountsShown>,
    pub topology: Option<TopologyShown>,
    pub evolution: Option<EvolutionShown>,
    pub dashboard: Option<DashboardShown>,
    pub network: Option<NetworkShown>,
    pub brain: Option<BrainShown>,
    /// Where the brain view looks, and the neuron picked in it.
    pub brain_camera: BrainCamera,
    pub brain_selected: Option<(usize, usize)>,
    /// Where the network view looks, and the neuron picked in it.
    pub net_camera: NetCamera,
    pub net_selected: Option<(usize, usize)>,
    /// Replaced images, released one frame later (they may still be on screen).
    retired: Vec<Arc<RenderImage>>,
    /// Plot size in physical px, by [`PlotSlot`].
    plot: [(u32, u32); PlotSlot::COUNT],
    /// Device pixels per logical pixel, as the view last reported it.
    scale: f32,
    last_key: [Option<RenderKey>; PlotSlot::COUNT],
    last_submit: Option<Instant>,
    scheduled: bool,
    pending: Option<Task<()>>,
    run_task: Option<Task<()>>,
    delivered: async_channel::Sender<Delivered>,
    _delivery: Task<()>,
    _store: Subscription,
}

impl TrainingVm {
    pub fn new(store: Entity<Store>, cx: &mut Context<Self>) -> Self {
        let (delivered, delivered_rx) = async_channel::unbounded::<Delivered>();
        let delivery = cx.spawn(async move |this, cx| {
            while let Ok(d) = delivered_rx.recv().await {
                if this.update(cx, |vm, cx| vm.on_delivered(d, cx)).is_err() {
                    break;
                }
            }
        });
        let sub = cx.subscribe(&store, |vm, _, event, cx| {
            if matches!(event, AppEvent::PaletteChanged) {
                vm.request_render(cx);
            }
        });
        Self {
            store,
            state: TrainingState::new(),
            tab: TrainingTab::default(),
            control: None,
            curves: None,
            raster: None,
            counts: None,
            topology: None,
            evolution: None,
            dashboard: None,
            network: None,
            brain: None,
            brain_camera: BrainCamera::default(),
            brain_selected: None,
            net_camera: NetCamera::default(),
            net_selected: None,
            retired: Vec::new(),
            plot: [(0, 0); PlotSlot::COUNT],
            scale: 1.0,
            last_key: std::array::from_fn(|_| None),
            last_submit: None,
            scheduled: false,
            pending: None,
            run_task: None,
            delivered,
            _delivery: delivery,
            _store: sub,
        }
    }

    pub fn palette(&self, cx: &App) -> Palette {
        self.store.read(cx).palette()
    }

    // ─── the run ────────────────────────────────────────────────────────

    /// Starts the demo training on its own thread (a run in progress is left alone).
    pub fn start(&mut self, cx: &mut Context<Self>) {
        self.start_run(TITLE, cx, |tx| {
            // The device is made on this thread: the training owns it
            let device = Device::flex().autodiff();
            demo::run(&device, &DemoConfig::default(), &tx);
        });
    }

    /// Starts the structural-plasticity demo: a dynamic spiking network that grows and prunes its
    /// own synapses and neurons while it trains, on its own thread. The Structure tab shows what
    /// the mechanism really does (it is the real controller, not an animation).
    pub fn start_structural(&mut self, cx: &mut Context<Self>) {
        if self.state.status.active() {
            return;
        }
        self.tab = TrainingTab::Structure;
        self.start_run(STRUCTURAL_TITLE, cx, |tx| {
            structural_demo::run(&tx, &StructuralDemoConfig::default());
        });
    }

    /// Starts the architecture-search demo (NEAT-like evolution of spiking networks on XOR) on
    /// its own thread. The Evolution tab shows the real search, generation by generation.
    pub fn start_evolution(&mut self, cx: &mut Context<Self>) {
        if self.state.status.active() {
            return;
        }
        self.tab = TrainingTab::Evolution;
        self.start_run(EVOLUTION_TITLE, cx, |tx| {
            evolution_demo::run(
                &tx,
                &EvolutionDemoConfig {
                    target: EVOLUTION_TARGET,
                    ..EvolutionDemoConfig::default()
                },
            );
        });
    }

    /// Any run that reports through a [`TrainTx`]: the body is `job`, started on its own thread.
    /// An arbitrary model plugs in here (it needs only to send `TrainEvent`s).
    pub fn start_run(
        &mut self,
        title: &'static str,
        cx: &mut Context<Self>,
        job: impl FnOnce(TrainTx) + Send + 'static,
    ) {
        if self.state.status.active() {
            return;
        }
        let (tx, rx) = channel(ChannelConfig::new(title));
        let spawned = std::thread::Builder::new()
            .name("neo-training".into())
            .spawn(move || job(tx));
        if let Err(e) = spawned {
            tracing::error!("could not start the training thread: {e}");
            self.store.update(cx, |s, cx| {
                s.set_status(format!("Could not start the training: {e}"), cx)
            });
            return;
        }
        self.control = Some(rx.control());
        self.state.reset(title);
        self.state.status = RunStatus::Running;
        self.curves = None;
        self.raster = None;
        self.counts = None;
        self.topology = None;
        self.dashboard = None;
        self.network = None;
        self.net_selected = None;
        self.brain = None;
        self.brain_selected = None;
        self.last_key = std::array::from_fn(|_| None);
        // Replacing the task drops the previous run's reader; its sender sees the channel close
        // and that run stops.
        self.run_task = Some(cx.spawn(async move |this, cx| {
            while let Ok(first) = rx.events.recv().await {
                let mut batch = vec![first];
                while batch.len() < BATCH {
                    match rx.events.try_recv() {
                        Ok(e) => batch.push(e),
                        Err(_) => break,
                    }
                }
                if this.update(cx, |vm, cx| vm.on_events(batch, cx)).is_err() {
                    break;
                }
            }
        }));
        cx.notify();
    }

    pub fn toggle_pause(&mut self, cx: &mut Context<Self>) {
        let Some(control) = &self.control else { return };
        let pause = self.state.status == RunStatus::Running;
        if !pause && self.state.status != RunStatus::Paused {
            return;
        }
        control.set_paused(pause);
        self.state.set_paused(pause);
        cx.notify();
    }

    /// Asks the training to stop at the end of the current step.
    pub fn stop(&mut self, cx: &mut Context<Self>) {
        if let Some(control) = &self.control {
            control.stop();
        }
        cx.notify();
    }

    fn on_events(&mut self, events: Vec<TrainEvent>, cx: &mut Context<Self>) {
        let before = self.state.status;
        for e in events {
            self.state.apply(e);
        }
        if let Some(c) = &self.control {
            self.state.set_dropped(c.dropped());
        }
        // The status bar says how a run ended
        let after = self.state.status;
        if after != before && matches!(after, RunStatus::Finished | RunStatus::Stopped) {
            let acc = self
                .state
                .accuracy
                .last()
                .map_or(String::new(), |a| format!(", accuracy {:.0}%", a * 100.0));
            let text = format!(
                "Training {} after {} steps: loss {:.4}{acc}",
                after.label().to_lowercase(),
                self.state.steps_seen,
                self.state.loss.last().unwrap_or(f32::NAN)
            );
            self.store.update(cx, |s, cx| s.set_status(text, cx));
        }
        cx.notify();
        self.request_render(cx);
    }

    // ─── view settings ──────────────────────────────────────────────────

    pub fn set_tab(&mut self, tab: TrainingTab, cx: &mut Context<Self>) {
        if self.tab != tab {
            self.tab = tab;
            cx.notify();
            self.request_render(cx);
        }
    }

    pub fn select_layer(&mut self, index: usize, cx: &mut Context<Self>) {
        self.state.select_layer(index);
        cx.notify();
        self.request_render(cx);
    }

    pub fn step_back(&mut self, cx: &mut Context<Self>) {
        self.state.step_back();
        cx.notify();
        self.request_render(cx);
    }

    pub fn step_forward(&mut self, cx: &mut Context<Self>) {
        self.state.step_forward();
        cx.notify();
        self.request_render(cx);
    }

    pub fn go_live(&mut self, cx: &mut Context<Self>) {
        self.state.go_live();
        cx.notify();
        self.request_render(cx);
    }

    // ─── rendering ──────────────────────────────────────────────────────

    /// Last plot size reported for `slot`'s image (physical px).
    pub fn plot_size(&self, slot: PlotSlot) -> (u32, u32) {
        self.plot[slot.index()]
    }

    /// Plot area of `slot`'s image in physical pixels, and the scale factor of the window (reported
    /// by the view as it is laid out; the dashboard needs the scale to leave room for its titles).
    pub fn set_plot_size(
        &mut self,
        slot: PlotSlot,
        width: u32,
        height: u32,
        scale: f32,
        cx: &mut Context<Self>,
    ) {
        if self.plot[slot.index()] != (width, height) || self.scale != scale {
            self.plot[slot.index()] = (width, height);
            self.scale = scale;
            self.request_render(cx);
        }
    }

    /// Drags the network view by `dfx`, `dfy` (fractions of its size).
    pub fn net_pan(&mut self, dfx: f32, dfy: f32, cx: &mut Context<Self>) {
        self.net_camera.pan(dfx, dfy);
        self.request_render(cx);
    }

    /// Zooms the network view by `factor` around the point (`fx`, `fy`) (fractions of its size).
    pub fn net_zoom(&mut self, factor: f32, fx: f32, fy: f32, cx: &mut Context<Self>) {
        self.net_camera.zoom_at(factor, fx, fy);
        self.request_render(cx);
    }

    /// A click in the network view: picks the neuron under it, or clears the pick on empty space or on
    /// the neuron that is already picked.
    pub fn net_click(&mut self, fx: f32, fy: f32, cx: &mut Context<Self>) {
        let hit = self
            .network
            .as_ref()
            .and_then(|n| n.layout.hit(fx, fy, 14.0));
        self.net_selected = if hit == self.net_selected { None } else { hit };
        self.request_render(cx);
    }

    /// Turns the brain by `dfx`, `dfy` (fractions of the view).
    pub fn brain_rotate(&mut self, dfx: f32, dfy: f32, cx: &mut Context<Self>) {
        self.brain_camera.rotate(dfx, dfy);
        self.request_render(cx);
    }

    /// Zooms the brain by `factor` around the point (`fx`, `fy`) (fractions of the view), which stays under the pointer.
    pub fn brain_zoom(&mut self, factor: f32, fx: f32, fy: f32, cx: &mut Context<Self>) {
        let (w, h) = self.plot[PlotSlot::Brain.index()];
        let middle = crate::engine::training::brain::middle_of_map(w, h, self.scale);
        self.brain_camera.zoom_at(factor, fx, fy, middle);
        self.request_render(cx);
    }

    /// Moves the brain by `dfx`, `dfy` (fractions of the view).
    pub fn brain_slide(&mut self, dfx: f32, dfy: f32, cx: &mut Context<Self>) {
        self.brain_camera.slide(dfx, dfy);
        self.request_render(cx);
    }

    /// A click in the brain view: picks the neuron under it (in the usage grid or on the map), or
    /// clears the pick on empty space or on the neuron that is already picked.
    pub fn brain_click(&mut self, fx: f32, fy: f32, cx: &mut Context<Self>) {
        let hit = self.brain.as_ref().and_then(|b| {
            b.layout
                .hit_usage(fx, fy)
                .and_then(|(_, unit)| unit)
                .or_else(|| b.layout.hit_node(fx, fy, 14.0))
        });
        self.brain_selected = if hit == self.brain_selected {
            None
        } else {
            hit
        };
        self.request_render(cx);
    }

    pub fn brain_reset(&mut self, cx: &mut Context<Self>) {
        self.brain_camera = BrainCamera::default();
        self.brain_selected = None;
        self.request_render(cx);
    }

    pub fn net_reset(&mut self, cx: &mut Context<Self>) {
        self.net_camera = NetCamera::default();
        self.net_selected = None;
        self.request_render(cx);
    }

    /// Asks for the shown tab's image once the current update ends, and not sooner than
    /// [`MIN_RENDER_GAP`] after the last one.
    pub fn request_render(&mut self, cx: &mut Context<Self>) {
        if self.scheduled {
            return;
        }
        self.scheduled = true;
        let wait = self.last_submit.map_or(Duration::ZERO, |t| {
            MIN_RENDER_GAP.saturating_sub(t.elapsed())
        });
        self.pending = Some(cx.spawn(async move |this, cx| {
            if !wait.is_zero() {
                cx.background_executor().timer(wait).await;
            }
            let _ = this.update(cx, |vm, cx| {
                vm.scheduled = false;
                vm.submit(cx);
            });
        }));
    }

    fn submit(&mut self, cx: &mut Context<Self>) {
        let mut any = false;
        for &slot in self.tab.slots() {
            any |= self.submit_slot(slot, cx);
        }
        if any {
            self.last_submit = Some(Instant::now());
        }
    }

    /// Asks the work pool for `slot`'s image when it is stale; `true` when it did.
    fn submit_slot(&mut self, slot: PlotSlot, cx: &mut Context<Self>) -> bool {
        let (width, height) = self.plot[slot.index()];
        if width == 0 || height == 0 {
            return false;
        }
        let palette = self.palette(cx);
        let net = match slot {
            PlotSlot::Network => Some((self.net_camera.bits(), self.net_selected)),
            PlotSlot::Brain => Some((self.brain_camera.bits(), self.brain_selected)),
            _ => None,
        };
        let key = RenderKey {
            revision: self.state.revision(),
            width,
            height,
            scale_bits: self.scale.to_bits(),
            net,
            palette,
        };
        if self.last_key[slot.index()].as_ref() == Some(&key) {
            return false;
        }
        let tx = self.delivered.clone();
        let work: crate::engine::work_pool::Work = match slot {
            PlotSlot::Brain => {
                let req = self.state.brain_request(
                    width,
                    height,
                    self.scale,
                    self.brain_camera,
                    self.brain_selected,
                );
                Box::new(move |_: &std::sync::atomic::AtomicBool| {
                    let (frame, info, layout) = render_brain(&req);
                    if let Some(image) = to_image(frame) {
                        let _ = tx.try_send(Delivered::Brain(BrainShown {
                            image,
                            layout,
                            info,
                        }));
                    }
                })
            }
            PlotSlot::Network => {
                let req = self.state.network_request(
                    width,
                    height,
                    self.scale,
                    self.net_camera,
                    self.net_selected,
                );
                Box::new(move |_: &std::sync::atomic::AtomicBool| {
                    let (frame, layout) = render_network(&req);
                    if let Some(image) = to_image(frame) {
                        let _ = tx.try_send(Delivered::Network(NetworkShown { image, layout }));
                    }
                })
            }
            PlotSlot::Dashboard => {
                let req = self
                    .state
                    .dashboard_request(width, height, self.scale, palette);
                Box::new(move |_: &std::sync::atomic::AtomicBool| {
                    let (frame, info) = render_dashboard(&req);
                    if let Some(image) = to_image(frame) {
                        let _ = tx.try_send(Delivered::Dashboard(DashboardShown { image, info }));
                    }
                })
            }
            PlotSlot::Curves => {
                let req = CurveRequest {
                    loss: self.state.loss.values().to_vec(),
                    accuracy: self.state.accuracy.values().to_vec(),
                    grad: self.state.grad.values().to_vec(),
                    width,
                    height,
                    palette,
                };
                Box::new(move |_: &std::sync::atomic::AtomicBool| {
                    let (frame, panels) = render_curves(&req);
                    if let Some(image) = to_image(frame) {
                        let _ = tx.try_send(Delivered::Curves(CurvesShown { image, panels }));
                    }
                })
            }
            PlotSlot::Raster => {
                let Some(shown) = self.state.shown_frame() else {
                    return false;
                };
                let req = RasterRequest {
                    frame: shown.frame.clone(),
                    width,
                    height,
                    palette,
                };
                Box::new(move |_: &std::sync::atomic::AtomicBool| {
                    let stats = RasterStats::of(&req.frame);
                    let (frame, layout) = render_raster(&req);
                    if let Some(image) = to_image(frame) {
                        let _ = tx.try_send(Delivered::Raster(RasterShown {
                            image,
                            layout,
                            stats,
                        }));
                    }
                })
            }
            PlotSlot::Counts => {
                let st = &self.state.structure;
                if st.is_empty() {
                    return false;
                }
                let req = CountsRequest {
                    neurons: st.neurons.values().to_vec(),
                    synapses: st.synapses.values().to_vec(),
                    active_synapses: st.active_synapses.values().to_vec(),
                    mean_rate: st.mean_rate.values().to_vec(),
                    max_neurons: st.limits.max_neurons,
                    max_synapses: st.limits.max_synapses,
                    markers: st.markers_in_window(),
                    width,
                    height,
                    palette,
                };
                Box::new(move |_: &std::sync::atomic::AtomicBool| {
                    let (frame, panels) = render_counts(&req);
                    if let Some(image) = to_image(frame) {
                        let _ = tx.try_send(Delivered::Counts(CountsShown { image, panels }));
                    }
                })
            }
            PlotSlot::Evolution => {
                let st = &self.state.evolution;
                if st.is_empty() {
                    return false;
                }
                let req = EvolutionRequest {
                    best: st.best.values().to_vec(),
                    mean: st.mean.values().to_vec(),
                    champion: st.champion.values().to_vec(),
                    conns: st.best_conns.values().to_vec(),
                    hidden: st.best_hidden.values().to_vec(),
                    species: st.species.values().to_vec(),
                    // Only the demo's target is known; another search has none to draw
                    target: (self.state.title == EVOLUTION_TITLE).then_some(EVOLUTION_TARGET),
                    width,
                    height,
                    palette,
                };
                Box::new(move |_: &std::sync::atomic::AtomicBool| {
                    let (frame, panels) = render_evolution(&req);
                    if let Some(image) = to_image(frame) {
                        let _ = tx.try_send(Delivered::Evolution(EvolutionShown { image, panels }));
                    }
                })
            }
            PlotSlot::Topology => {
                let Some(snapshot) = self.state.structure.topology.clone() else {
                    return false;
                };
                let req = TopologyRequest {
                    snapshot,
                    width,
                    height,
                    palette,
                };
                Box::new(move |_: &std::sync::atomic::AtomicBool| {
                    let (frame, layout) = render_topology(&req);
                    if let Some(image) = to_image(frame) {
                        let _ = tx.try_send(Delivered::Topology(TopologyShown { image, layout }));
                    }
                })
            }
        };
        self.last_key[slot.index()] = Some(key);
        cx.global::<Services>()
            .pool
            .request(JobKey::new(TRAINING_OWNER, slot.index() as u32), work);
        true
    }

    fn on_delivered(&mut self, d: Delivered, cx: &mut Context<Self>) {
        for old in std::mem::take(&mut self.retired) {
            cx.drop_image(old, None);
        }
        match d {
            Delivered::Brain(b) => self.retired.extend(self.brain.replace(b).map(|s| s.image)),
            Delivered::Network(n) => self
                .retired
                .extend(self.network.replace(n).map(|s| s.image)),
            Delivered::Dashboard(d) => self
                .retired
                .extend(self.dashboard.replace(d).map(|s| s.image)),
            Delivered::Curves(c) => self.retired.extend(self.curves.replace(c).map(|s| s.image)),
            Delivered::Raster(r) => self.retired.extend(self.raster.replace(r).map(|s| s.image)),
            Delivered::Counts(c) => self.retired.extend(self.counts.replace(c).map(|s| s.image)),
            Delivered::Topology(t) => self
                .retired
                .extend(self.topology.replace(t).map(|s| s.image)),
            Delivered::Evolution(e) => self
                .retired
                .extend(self.evolution.replace(e).map(|s| s.image)),
        }
        cx.notify();
    }

    /// The workspace was left: stop asking for images and release them.
    pub fn close(&mut self, cx: &mut Context<Self>) {
        cx.global::<Services>().pool.forget(TRAINING_OWNER);
        let shown = [
            self.curves.take().map(|s| s.image),
            self.raster.take().map(|s| s.image),
            self.counts.take().map(|s| s.image),
            self.topology.take().map(|s| s.image),
            self.evolution.take().map(|s| s.image),
            self.dashboard.take().map(|s| s.image),
            self.network.take().map(|s| s.image),
            self.brain.take().map(|s| s.image),
        ];
        for image in shown
            .into_iter()
            .flatten()
            .chain(std::mem::take(&mut self.retired))
        {
            cx.drop_image(image, None);
        }
        self.last_key = std::array::from_fn(|_| None);
        // Forget the sizes too: the view reports them again when the workspace is shown, which
        // draws the images anew
        self.plot = [(0, 0); PlotSlot::COUNT];
    }
}

/// BGRA frame → the image GPUI draws, converted on the work thread (as the trace views do).
fn to_image(frame: crate::engine::canvas::Frame) -> Option<Arc<RenderImage>> {
    let (w, h) = (frame.width, frame.height);
    let buffer = image::RgbaImage::from_raw(w, h, frame.into_bgra())?;
    Some(Arc::new(RenderImage::new(vec![image::Frame::new(buffer)])))
}
