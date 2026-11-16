//! The Training workspace: a toolbar (start / pause / stop, the tabs, progress) over the tab
//! shown. Network, Brain, Dashboard, Curves, Spike raster, Structure and Evolution are images drawn by
//! the engine on a work thread, with the text (names, values, axes) laid over them as elements;
//! Health is plain elements.
//!
//! The tabs replace the terminal dashboard of `neo-snn` (`tui`): the same curves, raster and
//! verdict, with the training running on its own thread.

use std::cell::Cell;
use std::rc::Rc;
use std::sync::Arc;

use gpui_kit::TestSupportExt as _;
use gpui_kit::component::button::{Button, ButtonVariants as _};
use gpui_kit::component::{
    ActiveTheme as _, IconName, Selectable as _, Sizable as _, h_flex, v_flex,
};
use gpui_kit::prelude::FluentBuilder as _;
use gpui_kit::{
    AnyElement, AppContext as _, Bounds, Context, Corners, Entity, FontWeight,
    InteractiveElement as _, IntoElement, MouseButton, MouseDownEvent, MouseMoveEvent,
    MouseUpEvent, ParentElement as _, Pixels, Render, RenderImage, ScrollWheelEvent, SharedString,
    StatefulInteractiveElement as _, Styled as _, Subscription, Window, canvas, div, px, relative,
};
use neo_snn::demo::{
    LAYER_HIDDEN, LAYER_INPUT, LAYER_OUTPUT, WEIGHTS_HIDDEN_OUTPUT, WEIGHTS_INPUT_HIDDEN,
};
use neo_snn::functional::ActivityStats;
use neo_snn::telemetry::{Finding, Severity};

use crate::engine::canvas::Pixel;
use crate::engine::palette::Palette;
use crate::engine::training::brain::{self, LANE_GUTTER, MAP, RASTER, STATS, USAGE};
use crate::engine::training::dashboard::{GAP, PAD, TITLE_H, layer_color};
use crate::engine::training::{
    BrainInfo, CurvePanel, DashboardInfo, EventRow, MarkerKind, PanelKind, PanelRect, RunStatus,
    TrainingState, marker_color, panel_rects,
};
use crate::store::{Store, duration};
use crate::viewmodels::{PlotSlot, TrainingTab, TrainingVm};
use crate::views::trace::color;
use crate::widgets::{EmptyState, MenuSelect};

/// A press in the network view: where the pointer was last, and whether it has moved enough to be a drag.
#[derive(Clone, Copy)]
struct NetDrag {
    x: f32,
    y: f32,
    moved: bool,
}

pub struct TrainingView {
    pub vm: Entity<TrainingVm>,
    /// Where the network view's image is on screen (set as it is laid out), for turning pointer
    /// positions into fractions of it.
    net_bounds: Rc<Cell<Bounds<Pixels>>>,
    net_drag: Option<NetDrag>,
    /// The neuron under the pointer and where the pointer is (fractions of the view): `(layer, unit, fx, fy)`.
    net_hover: Option<(usize, usize, f32, f32)>,
    /// The same three for the brain view.
    brain_bounds: Rc<Cell<Bounds<Pixels>>>,
    brain_drag: Option<NetDrag>,
    brain_hover: Option<(usize, usize, f32, f32)>,
    _vm: Subscription,
}

impl TrainingView {
    pub fn new(store: Entity<Store>, cx: &mut Context<Self>) -> Self {
        let vm = cx.new(|cx| TrainingVm::new(store, cx));
        let sub = cx.observe(&vm, |_, _, cx| cx.notify());
        Self {
            vm,
            net_bounds: Rc::new(Cell::new(Bounds::default())),
            net_drag: None,
            net_hover: None,
            brain_bounds: Rc::new(Cell::new(Bounds::default())),
            brain_drag: None,
            brain_hover: None,
            _vm: sub,
        }
    }

    /// Starts a run (the `--train` flag and the Start button).
    pub fn start(&mut self, cx: &mut Context<Self>) {
        self.vm.update(cx, |vm, cx| vm.start(cx));
    }

    // ─── toolbar ────────────────────────────────────────────────────────

    fn toolbar(&self, cx: &mut Context<Self>) -> impl IntoElement + use<> {
        let (tab, status, fraction, progress, elapsed, palette, title, dropped) = {
            let vm = self.vm.read(cx);
            (
                vm.tab,
                vm.state.status,
                vm.state.fraction_done(),
                vm.state.progress,
                vm.state.elapsed(),
                vm.palette(cx),
                vm.state.title.clone(),
                vm.state.dropped,
            )
        };
        let t = cx.theme();
        let (muted, border) = (t.muted_foreground, t.border);
        let vm = self.vm.clone();

        let controls = h_flex()
            .gap_1()
            .when(!status.active(), |d| {
                let vm_main = vm.clone();
                d.child(
                    Button::new("train-start")
                        .small()
                        .primary()
                        .icon(IconName::Play)
                        .label(if status == RunStatus::Idle { "Start" } else { "Restart" })
                        .tooltip("Train the demo spiking network on a synthetic task")
                        .on_click(move |_, _, cx| vm_main.update(cx, |vm, cx| vm.start(cx))),
                )
                .child({
                    let vm = vm.clone();
                    Button::new("train-start-structural")
                        .small()
                        .outline()
                        .icon(IconName::Play)
                        .label("Structural demo")
                        .tooltip("Train a dynamic spiking network that grows and prunes its own neurons and synapses (structural plasticity)")
                        .on_click(move |_, _, cx| vm.update(cx, |vm, cx| vm.start_structural(cx)))
                })
                .child({
                    let vm = vm.clone();
                    Button::new("train-start-evolution")
                        .small()
                        .outline()
                        .icon(IconName::Play)
                        .label("Evolution demo")
                        .tooltip("Search for a spiking network's architecture by evolution (NEAT-like): the population grows neurons and connections generation by generation")
                        .on_click(move |_, _, cx| vm.update(cx, |vm, cx| vm.start_evolution(cx)))
                })
            })
            .when(status.active(), |d| {
                let paused = status == RunStatus::Paused;
                let (vm_pause, vm_stop) = (vm.clone(), vm.clone());
                d.child(
                    Button::new("train-pause")
                        .small()
                        .outline()
                        .icon(if paused { IconName::Play } else { IconName::Pause })
                        .label(if paused { "Resume" } else { "Pause" })
                        .tooltip("Pause or resume at the end of the current step")
                        .on_click(move |_, _, cx| vm_pause.update(cx, |vm, cx| vm.toggle_pause(cx))),
                )
                .child(
                    Button::new("train-stop")
                        .small()
                        .outline()
                        .icon(gpui_kit::assets::IconName::Square)
                        .label("Stop")
                        .tooltip("Stop at the end of the current step")
                        .on_click(move |_, _, cx| vm_stop.update(cx, |vm, cx| vm.stop(cx))),
                )
            });

        let tabs = h_flex()
            .gap_0p5()
            .children(TrainingTab::ALL.into_iter().map(|tb| {
                let vm = vm.clone();
                Button::new(SharedString::from(format!(
                    "training-tab-{}",
                    tb.title().to_lowercase().replace(' ', "-")
                )))
                .small()
                .ghost()
                .label(tb.title())
                .tooltip(tb.hint())
                .selected(tb == tab)
                .on_click(move |_, _, cx| vm.update(cx, |vm, cx| vm.set_tab(tb, cx)))
            }));

        let readout = (status != RunStatus::Idle).then(|| {
            let step = (progress.step() + 1).min(progress.total_steps().max(1));
            let mut text = format!(
                "{} · step {step}/{} · epoch {}/{}",
                status.label(),
                progress.total_steps(),
                progress.epoch + 1,
                progress.epochs.max(1)
            );
            if let Some(e) = elapsed {
                text.push_str(&format!(" · {}", duration(e.as_secs_f64())));
            }
            if dropped > 0 {
                text.push_str(&format!(" · {dropped} events dropped"));
            }
            h_flex()
                .gap_2()
                .child(div().text_xs().text_color(muted).child(title))
                .child(div().text_xs().text_color(muted).child(text))
        });
        let bar = (status != RunStatus::Idle).then(|| {
            div()
                .h(px(2.))
                .w(relative(fraction))
                .bg(color(palette.channel(0)))
        });

        v_flex()
            .id("training-toolbar")
            .w_full()
            .border_b_1()
            .border_color(border)
            .child(
                h_flex()
                    .w_full()
                    .gap_3()
                    .px_2()
                    .py_1()
                    .child(controls)
                    .child(div().h(px(18.)).w(px(1.)).bg(border))
                    .child(tabs)
                    .child(div().flex_1())
                    .children(readout),
            )
            .child(div().h(px(2.)).w_full().children(bar))
    }

    // ─── bodies ─────────────────────────────────────────────────────────

    fn empty(&self) -> AnyElement {
        let vm = self.vm.clone();
        div()
            .id("training-empty")
            .test_support()
            .size_full()
            .child(
                EmptyState::new(
                    gpui_kit::assets::IconName::Activity,
                    "No training run yet",
                    "Start a demo run: a two-layer spiking network learns a synthetic task on its own thread, and its loss, spikes and health show up here as it goes.",
                )
                .child(Button::new("train-start-empty").primary().icon(IconName::Play).label("Start training").on_click(move |_, _, cx| vm.update(cx, |vm, cx| vm.start(cx)))),
            )
            .into_any_element()
    }

    /// A canvas that reports its size (physical px) to the view model and paints `image` over it.
    fn plot_canvas(
        &self,
        slot: PlotSlot,
        known: (u32, u32),
        image: Option<Arc<RenderImage>>,
    ) -> impl IntoElement + use<> {
        let weak = self.vm.downgrade();
        canvas(
            move |bounds, window, cx| {
                let scale = window.scale_factor();
                let (w, h) = (
                    (bounds.size.width.as_f32() * scale).round() as u32,
                    (bounds.size.height.as_f32() * scale).round() as u32,
                );
                if (w, h) != known && w > 0 && h > 0 {
                    let vm = weak.clone();
                    cx.defer(move |cx| {
                        let _ = vm.update(cx, |vm, cx| vm.set_plot_size(slot, w, h, scale, cx));
                    });
                }
            },
            move |bounds, _, window, _| {
                if let Some(image) = image {
                    let _ = window.paint_image(bounds, bounds, Corners::default(), image, 0, false);
                }
            },
        )
        .absolute()
        .size_full()
    }

    // ─── network view ───────────────────────────────────────────────────

    /// Position of `p` in the network view as fractions (clamped).
    fn net_fractions(&self, p: gpui_kit::Point<Pixels>) -> (f32, f32) {
        let b = self.net_bounds.get();
        let (w, h) = (
            b.size.width.as_f32().max(1.0),
            b.size.height.as_f32().max(1.0),
        );
        (
            ((p.x - b.origin.x).as_f32() / w).clamp(0.0, 1.0),
            ((p.y - b.origin.y).as_f32() / h).clamp(0.0, 1.0),
        )
    }

    fn on_net_down(&mut self, e: &MouseDownEvent, _: &mut Window, _: &mut Context<Self>) {
        self.net_drag = Some(NetDrag {
            x: e.position.x.as_f32(),
            y: e.position.y.as_f32(),
            moved: false,
        });
    }

    fn on_net_move(&mut self, e: &MouseMoveEvent, _: &mut Window, cx: &mut Context<Self>) {
        if e.pressed_button != Some(MouseButton::Left) {
            self.net_drag = None;
            // Nothing pressed: look for a neuron under the pointer, for the tooltip
            let (fx, fy) = self.net_fractions(e.position);
            let hit = self
                .vm
                .read(cx)
                .network
                .as_ref()
                .and_then(|n| n.layout.hit(fx, fy, 12.0))
                .map(|(l, u)| (l, u, fx, fy));
            if hit != self.net_hover {
                self.net_hover = hit;
                cx.notify();
            }
            return;
        }
        self.net_hover = None;
        let Some(mut drag) = self.net_drag else {
            return;
        };
        let b = self.net_bounds.get();
        let (w, h) = (
            b.size.width.as_f32().max(1.0),
            b.size.height.as_f32().max(1.0),
        );
        let (x, y) = (e.position.x.as_f32(), e.position.y.as_f32());
        let (dx, dy) = (x - drag.x, y - drag.y);
        // A few pixels of wobble are still a click
        if !drag.moved && dx.abs() + dy.abs() < 4.0 {
            return;
        }
        drag = NetDrag { x, y, moved: true };
        self.net_drag = Some(drag);
        self.vm.update(cx, |vm, cx| vm.net_pan(dx / w, dy / h, cx));
    }

    fn on_net_up(&mut self, e: &MouseUpEvent, _: &mut Window, cx: &mut Context<Self>) {
        let Some(drag) = self.net_drag.take() else {
            return;
        };
        if !drag.moved {
            let (fx, fy) = self.net_fractions(e.position);
            self.vm.update(cx, |vm, cx| vm.net_click(fx, fy, cx));
        }
    }

    fn on_net_wheel(&mut self, e: &ScrollWheelEvent, _: &mut Window, cx: &mut Context<Self>) {
        let d = e.delta.pixel_delta(px(20.));
        let (fx, fy) = self.net_fractions(e.position);
        // Wheel up = closer
        let factor = (d.y.as_f32() * 0.004).exp();
        self.vm.update(cx, |vm, cx| vm.net_zoom(factor, fx, fy, cx));
    }

    /// Like [`Self::plot_canvas`], and it also records where it is on screen (in `at`) for the pointer.
    fn image_canvas(
        &self,
        slot: PlotSlot,
        at: Rc<Cell<Bounds<Pixels>>>,
        known: (u32, u32),
        image: Option<Arc<RenderImage>>,
    ) -> impl IntoElement + use<> {
        let weak = self.vm.downgrade();
        canvas(
            move |bounds, window, cx| {
                at.set(bounds);
                let scale = window.scale_factor();
                let (w, h) = (
                    (bounds.size.width.as_f32() * scale).round() as u32,
                    (bounds.size.height.as_f32() * scale).round() as u32,
                );
                if (w, h) != known && w > 0 && h > 0 {
                    let vm = weak.clone();
                    cx.defer(move |cx| {
                        let _ = vm.update(cx, |vm, cx| vm.set_plot_size(slot, w, h, scale, cx));
                    });
                }
            },
            move |bounds, _, window, _| {
                if let Some(image) = image {
                    let _ = window.paint_image(bounds, bounds, Corners::default(), image, 0, false);
                }
            },
        )
        .absolute()
        .size_full()
    }

    // ─── brain view ─────────────────────────────────────────────────────

    fn brain_fractions(&self, p: gpui_kit::Point<Pixels>) -> (f32, f32) {
        let b = self.brain_bounds.get();
        let (w, h) = (
            b.size.width.as_f32().max(1.0),
            b.size.height.as_f32().max(1.0),
        );
        (
            ((p.x - b.origin.x).as_f32() / w).clamp(0.0, 1.0),
            ((p.y - b.origin.y).as_f32() / h).clamp(0.0, 1.0),
        )
    }

    fn on_brain_down(&mut self, e: &MouseDownEvent, _: &mut Window, _: &mut Context<Self>) {
        self.brain_drag = Some(NetDrag {
            x: e.position.x.as_f32(),
            y: e.position.y.as_f32(),
            moved: false,
        });
    }

    fn on_brain_move(&mut self, e: &MouseMoveEvent, _: &mut Window, cx: &mut Context<Self>) {
        if e.pressed_button != Some(MouseButton::Left) {
            self.brain_drag = None;
            // Nothing pressed: the neuron under the pointer, for the tooltip
            let (fx, fy) = self.brain_fractions(e.position);
            let hit = self
                .vm
                .read(cx)
                .brain
                .as_ref()
                .and_then(|b| {
                    b.layout
                        .hit_usage(fx, fy)
                        .and_then(|(_, unit)| unit)
                        .or_else(|| b.layout.hit_node(fx, fy, 12.0))
                })
                .map(|(l, u)| (l, u, fx, fy));
            if hit != self.brain_hover {
                self.brain_hover = hit;
                cx.notify();
            }
            return;
        }
        self.brain_hover = None;
        let Some(drag) = self.brain_drag else { return };
        let b = self.brain_bounds.get();
        let (w, h) = (
            b.size.width.as_f32().max(1.0),
            b.size.height.as_f32().max(1.0),
        );
        let (x, y) = (e.position.x.as_f32(), e.position.y.as_f32());
        let (dx, dy) = (x - drag.x, y - drag.y);
        if !drag.moved && dx.abs() + dy.abs() < 4.0 {
            return;
        }
        self.brain_drag = Some(NetDrag { x, y, moved: true });
        // Shift moves the brain, a plain drag turns it
        if e.modifiers.shift {
            self.vm
                .update(cx, |vm, cx| vm.brain_slide(dx / w, dy / h, cx));
        } else {
            self.vm
                .update(cx, |vm, cx| vm.brain_rotate(dx / w, dy / h, cx));
        }
    }

    fn on_brain_up(&mut self, e: &MouseUpEvent, _: &mut Window, cx: &mut Context<Self>) {
        let Some(drag) = self.brain_drag.take() else {
            return;
        };
        if !drag.moved {
            let (fx, fy) = self.brain_fractions(e.position);
            self.vm.update(cx, |vm, cx| vm.brain_click(fx, fy, cx));
        }
    }

    fn on_brain_wheel(&mut self, e: &ScrollWheelEvent, _: &mut Window, cx: &mut Context<Self>) {
        let d = e.delta.pixel_delta(px(20.));
        let (fx, fy) = self.brain_fractions(e.position);
        // Wheel up = closer, around the pointer
        let factor = (d.y.as_f32() * 0.004).exp();
        self.vm
            .update(cx, |vm, cx| vm.brain_zoom(factor, fx, fy, cx));
    }

    /// Which neurons are in use: the brain map, a usage cell for every neuron, and a raster per region,
    /// with the headline of what just fired, the numbers and the names laid over the image.
    fn brain_body(&self, cx: &mut Context<Self>) -> AnyElement {
        let (known, image, info, spikes_seen, readout, tip) = {
            let vm = self.vm.read(cx);
            let st = &vm.state;
            let readout = vm
                .brain_selected
                .and_then(|(l, u)| neuron_info(st, l, u, true));
            let tip = self
                .brain_hover
                .and_then(|(l, u, fx, fy)| neuron_info(st, l, u, false).map(|i| (i, fx, fy)));
            (
                vm.plot_size(PlotSlot::Brain),
                vm.brain.as_ref().map(|b| b.image.clone()),
                vm.brain.as_ref().map(|b| b.info.clone()),
                st.spikes_seen,
                readout,
                tip,
            )
        };
        let t = cx.theme();
        let muted = t.muted_foreground;
        let (title_c, body_c, box_bg, box_edge, tag_c, motor_c, trace_c) = (
            color(Pixel::rgb(158, 197, 255)),
            color(Pixel::rgb(214, 224, 245)),
            color(Pixel::rgb(10, 14, 24)),
            color(Pixel::rgb(60, 72, 100)),
            color(Pixel::rgb(143, 176, 232)),
            color(Pixel::rgb(255, 90, 90)),
            color(Pixel::rgb(120, 235, 160)),
        );
        let (pos_c, neg_c) = (
            color(Pixel::rgb(60, 235, 140)),
            color(Pixel::rgb(255, 90, 60)),
        );
        let rgb = |c: brain::Rgb| color(Pixel::rgb(c[0], c[1], c[2]));
        let rect = |f: [f32; 4]| PanelRect {
            kind: PanelKind::Metrics,
            x0: f[0],
            y0: f[1],
            x1: f[2],
            y1: f[3],
        };

        let mut overlays: Vec<AnyElement> = Vec::new();
        let vm_reset = self.vm.clone();
        let reset = Button::new("brain-reset")
            .small()
            .label("Reset view")
            .on_click(move |_, _, cx| vm_reset.update(cx, |vm, cx| vm.brain_reset(cx)));
        overlays.push(panel_header(
            &rect(MAP),
            "Brain map",
            Some(reset.into_any_element()),
            body_c,
        ));
        overlays.push(panel_header(
            &rect(USAGE),
            "Neuron usage",
            Some(aside_text(
                "red outline = never fired · white = firing now".to_string(),
                muted,
            )),
            body_c,
        ));

        if let Some(info) = &info {
            // The headline: what fired, how many synapses took part, where the trace began
            let (headline, lines, origin) = brain_headline(info);
            overlays.push(
                div()
                    .absolute()
                    .left(relative(MAP[0]))
                    .top(relative(MAP[1]))
                    .ml(px(PAD + GAP))
                    .mt(px(TITLE_H + GAP + 6.0))
                    // Narrow enough that the text wraps before the picked-neuron panel on the right
                    .w(relative(0.36))
                    .child(
                        div()
                            .text_lg()
                            .font_weight(FontWeight::BOLD)
                            .text_color(if info.output_fired.is_empty() {
                                muted
                            } else {
                                motor_c
                            })
                            .child(headline),
                    )
                    .children(
                        lines
                            .into_iter()
                            .map(|l| div().text_xs().text_color(body_c).child(l)),
                    )
                    .children(origin.map(|o| div().text_xs().text_color(trace_c).child(o)))
                    .into_any_element(),
            );
            // Legend: the regions and the trace
            let mut legend = v_flex()
                .absolute()
                .left(relative(MAP[0]))
                .bottom(relative(1.0 - MAP[3]))
                .ml(px(PAD + GAP))
                .mb(px(PAD + GAP))
                .gap_0p5();
            for g in &info.groups {
                legend = legend.child(
                    h_flex()
                        .gap_2()
                        .items_center()
                        .child(div().w(px(8.)).h(px(8.)).rounded_full().bg(rgb(g.color)))
                        .child(div().text_xs().text_color(body_c).child(g.name.clone())),
                );
            }
            legend = legend.child(
                h_flex()
                    .gap_2()
                    .items_center()
                    .child(
                        h_flex()
                            .gap_0p5()
                            .child(
                                div()
                                    .w(px(8.))
                                    .h(px(3.))
                                    .rounded_full()
                                    .bg(color(Pixel::rgb(255, 255, 255))),
                            )
                            .child(
                                div()
                                    .w(px(8.))
                                    .h(px(3.))
                                    .rounded_full()
                                    .bg(color(Pixel::rgb(255, 140, 64))),
                            ),
                    )
                    .child(
                        div()
                            .text_xs()
                            .text_color(body_c)
                            .child("synapses that fired in order (white, then orange)"),
                    ),
            );
            legend = legend.child(
                div()
                    .text_xs()
                    .text_color(muted)
                    .child("drag: turn · shift+drag: move · wheel: zoom at the pointer"),
            );
            if info.lod {
                legend = legend.child(div().text_xs().text_color(muted).child(format!(
                    "cloud view: dim dots are a sample, bright ones fired on the last step ({})",
                    thousands(info.spikers as u64)
                )));
            }
            overlays.push(legend.into_any_element());

            // Above each usage block: whose it is and how it is doing
            for g in &info.groups {
                overlays.push(
                    h_flex()
                        .absolute()
                        .left(relative(g.label_at.0))
                        .top(relative(g.label_at.1))
                        .gap_2()
                        .child(
                            div()
                                .text_xs()
                                .font_weight(FontWeight::SEMIBOLD)
                                .text_color(rgb(g.color))
                                .child(shorten(&g.name, 26)),
                        )
                        .child(div().text_xs().text_color(muted).child(format!(
                            "{} neurons · {} firing · {} silent · {:.0}%",
                            thousands(g.units as u64),
                            thousands(g.firing_now as u64),
                            thousands(g.silent as u64),
                            g.usage_mean * 100.0
                        )))
                        .into_any_element(),
                );
            }
            // Lane names and the time axis of the raster
            for lane in &info.lanes {
                overlays.push(
                    div()
                        .absolute()
                        .left(relative(RASTER[0]))
                        .top(relative(lane.y0))
                        .h(relative(lane.y1 - lane.y0))
                        .ml(px(PAD + GAP / 2.0))
                        .w(px(LANE_GUTTER - 6.0))
                        .flex()
                        .items_center()
                        .text_xs()
                        .text_color(rgb(lane.color))
                        .child(shorten(&lane.name, 18))
                        .into_any_element(),
                );
            }
            if info.raster_frames > 0 {
                // `left` or `right` is the inset the label hangs from
                let axis = |left: Option<f32>, right: Option<f32>, text: String| {
                    let el = div()
                        .absolute()
                        .bottom(relative(1.0 - RASTER[3]))
                        .mb(px(GAP / 2.0 + 4.0))
                        .text_xs()
                        .text_color(muted)
                        .child(text);
                    match (left, right) {
                        (Some(l), _) => el.left(relative(l)).into_any_element(),
                        (_, Some(r)) => el.right(relative(r)).into_any_element(),
                        _ => el.into_any_element(),
                    }
                };
                overlays.push(axis(
                    Some(info.raster_x.0),
                    None,
                    format!(
                        "−{} frames ({} steps)",
                        info.raster_frames,
                        info.raster_frames * info.steps_per_frame
                    ),
                ));
                overlays.push(axis(None, Some(1.0 - info.raster_x.1), "now".to_string()));
                overlays.push(axis(Some(info.raster_x.0 + 0.22 * (info.raster_x.1 - info.raster_x.0)), None, "rows: slices of the group's neurons · brightness: share firing, scaled to each lane".to_string()));
            }
            // The numbers
            let stat = |at: f32, value: String, label: &'static str| {
                v_flex()
                    .absolute()
                    .left(relative(at))
                    .top(relative(STATS[1]))
                    .mt(px(GAP + 6.0))
                    .child(
                        div()
                            .text_2xl()
                            .font_weight(FontWeight::BOLD)
                            .text_color(body_c)
                            .child(value),
                    )
                    .child(div().text_xs().text_color(muted).child(label))
                    .into_any_element()
            };
            overlays.push(stat(
                0.04,
                format!("{:.2}%", info.firing_now * 100.0),
                "FIRING NOW",
            ));
            overlays.push(stat(
                0.36,
                format!(
                    "{} / {}",
                    thousands(info.neurons_fired as u64),
                    thousands(info.neurons as u64)
                ),
                "NEURONS FIRED",
            ));
            overlays.push(stat(0.68, thousands(spikes_seen), "SPIKES SEEN"));
        }

        // The neuron under the pointer
        if let Some((i, fx, fy)) = tip {
            overlays.push(
                div()
                    .absolute()
                    .left(relative((fx + 0.012).min(0.78)))
                    .top(relative((fy + 0.02).min(0.85)))
                    .px_2()
                    .py_1()
                    .bg(box_bg)
                    .border_1()
                    .border_color(title_c)
                    .rounded_sm()
                    .child(
                        div()
                            .text_xs()
                            .font_weight(FontWeight::SEMIBOLD)
                            .text_color(title_c)
                            .child(i.title),
                    )
                    .children(
                        i.lines
                            .into_iter()
                            .map(|l| div().text_xs().text_color(body_c).child(l)),
                    )
                    .into_any_element(),
            );
        }
        // The picked neuron, in full
        if let Some(info) = readout {
            overlays.push(
                v_flex()
                    .absolute()
                    .right(relative(1.0 - MAP[2]))
                    .top(relative(MAP[1]))
                    .mr(px(PAD + GAP))
                    .mt(px(TITLE_H + GAP + 6.0))
                    .w(px(240.))
                    .gap_1()
                    .px_2()
                    .py_2()
                    .bg(box_bg.opacity(0.9))
                    .border_1()
                    .border_color(box_edge)
                    .rounded_sm()
                    .child(
                        div()
                            .text_sm()
                            .font_weight(FontWeight::SEMIBOLD)
                            .text_color(title_c)
                            .child(info.title),
                    )
                    .children(
                        info.lines
                            .into_iter()
                            .map(|l| div().text_xs().text_color(body_c).child(l)),
                    )
                    .children(info.links.into_iter().map(|(heading, rows)| {
                        v_flex()
                            .gap_0p5()
                            .child(
                                div()
                                    .text_xs()
                                    .font_weight(FontWeight::SEMIBOLD)
                                    .text_color(tag_c)
                                    .child(heading),
                            )
                            .children(rows.into_iter().map(|(name, w)| {
                                h_flex()
                                    .gap_2()
                                    .child(
                                        div().w(px(60.)).text_xs().text_color(body_c).child(name),
                                    )
                                    .child(
                                        div()
                                            .text_xs()
                                            .text_color(if w >= 0.0 { pos_c } else { neg_c })
                                            .child(format!("{w:+.3}")),
                                    )
                            }))
                    }))
                    .child(
                        div()
                            .text_xs()
                            .text_color(muted)
                            .child("Click it again, or empty space, to let go"),
                    )
                    .into_any_element(),
            );
        }

        let wait = image.is_none().then(|| {
            div()
                .absolute()
                .size_full()
                .flex()
                .items_center()
                .justify_center()
                .text_sm()
                .text_color(tag_c)
                .child("Waiting for the first frames…")
        });
        div()
            .id("training-brain")
            .test_support()
            .size_full()
            .relative()
            .overflow_hidden()
            .bg(color(Pixel::rgb(4, 6, 10)))
            .on_mouse_down(MouseButton::Left, cx.listener(Self::on_brain_down))
            .on_mouse_move(cx.listener(Self::on_brain_move))
            .on_mouse_up(MouseButton::Left, cx.listener(Self::on_brain_up))
            .on_scroll_wheel(cx.listener(Self::on_brain_wheel))
            .on_hover(cx.listener(|this, hovered: &bool, _, cx| {
                if !*hovered && this.brain_hover.take().is_some() {
                    cx.notify();
                }
            }))
            .child(self.image_canvas(PlotSlot::Brain, self.brain_bounds.clone(), known, image))
            .children(overlays)
            .children(wait)
            .into_any_element()
    }

    /// The network as glowing curves, with everything that says what it shows: a box on every layer,
    /// the name of every neuron, a tooltip under the pointer, a readout of the picked neuron (what
    /// it fires on, its strongest links), a summary of the run, a legend. All of it elements over
    /// the image, placed from the layout the renderer returned with it.
    fn network_body(&self, cx: &mut Context<Self>) -> AnyElement {
        let (known, image, layout, picked, zoom, texts, summary, status, readout, tip) = {
            let vm = self.vm.read(cx);
            let layout = vm.network.as_ref().map(|n| n.layout.clone());
            let st = &vm.state;
            let facts = DashFacts::of(st, None);
            let status = format!(
                "{} · loss {} · accuracy {} · {}",
                facts.progress,
                st.loss
                    .last()
                    .map_or("–".to_string(), |v| format!("{v:.4}")),
                facts.headline,
                facts.speed
            );
            let readout = vm
                .net_selected
                .and_then(|(l, u)| neuron_info(st, l, u, true));
            let tip = self
                .net_hover
                .and_then(|(l, u, fx, fy)| neuron_info(st, l, u, false).map(|i| (i, fx, fy)));
            (
                vm.plot_size(PlotSlot::Network),
                vm.network.as_ref().map(|n| n.image.clone()),
                layout,
                vm.net_selected,
                vm.net_camera.zoom,
                NetTexts::of(st),
                net_summary(st),
                status,
                readout,
                tip,
            )
        };
        let t = cx.theme();
        let muted = t.muted_foreground;
        let (title_c, body_c, accent_c, box_bg, box_edge, tag_c) = (
            color(Pixel::rgb(158, 197, 255)),
            color(Pixel::rgb(214, 224, 245)),
            color(Pixel::rgb(255, 107, 90)),
            color(Pixel::rgb(14, 24, 58)),
            color(Pixel::rgb(60, 92, 168)),
            color(Pixel::rgb(143, 176, 232)),
        );
        let (pos_c, neg_c) = (
            color(Pixel::rgb(60, 235, 140)),
            color(Pixel::rgb(255, 90, 60)),
        );
        let hovered = self.net_hover.map(|(l, u, _, _)| (l, u));

        let mut overlays: Vec<AnyElement> = Vec::new();
        if let Some(layout) = &layout {
            let (sw, sh) = (layout.size.0.max(1.0), layout.size.1.max(1.0));
            let last = layout.layers.len().saturating_sub(1);
            for (l, g) in layout.layers.iter().enumerate() {
                if let Some(text) = texts.get(l) {
                    // The box sits on top of its column, kept inside the view when the column is taller than it
                    let top = ((g.top * sh - 58.0) / sh).clamp(0.005, 0.90);
                    let left = ((g.x * sw - 10.0) / sw).clamp(0.0, 0.92);
                    overlays.push(
                        div()
                            .absolute()
                            .left(relative(left))
                            .top(relative(top))
                            .px_2()
                            .py_1()
                            .bg(box_bg)
                            .border_1()
                            .border_color(if picked.is_some_and(|(pl, _)| pl == l) {
                                title_c
                            } else {
                                box_edge
                            })
                            .rounded_sm()
                            .child(
                                div()
                                    .text_xs()
                                    .font_weight(FontWeight::SEMIBOLD)
                                    .text_color(title_c)
                                    .child(text.title.clone()),
                            )
                            .child(
                                div()
                                    .text_xs()
                                    .text_color(body_c)
                                    .child(text.neurons.clone()),
                            )
                            .child(
                                div()
                                    .text_xs()
                                    .text_color(accent_c)
                                    .child(text.activation.clone()),
                            )
                            .into_any_element(),
                    );
                }
                // The name of every neuron beside it (x0…, h0…, class 0…), when they are far enough apart
                let gap_px = (g.bottom - g.top) * sh / g.units.max(1) as f32;
                if gap_px >= 12.0 && zoom >= 0.6 {
                    for (u, &(nx, ny)) in g.nodes.iter().enumerate() {
                        let on = picked == Some((l, u)) || hovered == Some((l, u));
                        let y = ((ny * sh - 7.0) / sh).clamp(0.0, 0.97);
                        let label = div()
                            .absolute()
                            .top(relative(y))
                            .px_1()
                            .rounded_sm()
                            .text_xs()
                            .bg(box_bg.opacity(if on { 0.95 } else { 0.55 }))
                            .text_color(if on { title_c } else { tag_c })
                            .child(unit_name(l, u));
                        // Outputs read to the right of their column, everything else to the left
                        overlays.push(if l == last {
                            label
                                .left(relative(((nx * sw + 16.0) / sw).clamp(0.0, 0.98)))
                                .into_any_element()
                        } else {
                            label
                                .right(relative((1.0 - (nx * sw - 16.0) / sw).clamp(0.0, 1.0)))
                                .into_any_element()
                        });
                    }
                }
            }
        }

        // The pointer's tooltip
        if let Some(((title, lines), fx, fy)) = tip.map(|(i, fx, fy)| ((i.title, i.lines), fx, fy))
        {
            overlays.push(
                div()
                    .absolute()
                    .left(relative((fx + 0.012).min(0.8)))
                    .top(relative((fy + 0.02).min(0.85)))
                    .px_2()
                    .py_1()
                    .bg(box_bg)
                    .border_1()
                    .border_color(title_c)
                    .rounded_sm()
                    .child(
                        div()
                            .text_xs()
                            .font_weight(FontWeight::SEMIBOLD)
                            .text_color(title_c)
                            .child(title),
                    )
                    .children(
                        lines
                            .into_iter()
                            .map(|l| div().text_xs().text_color(body_c).child(l)),
                    )
                    .into_any_element(),
            );
        }

        // The side panel: the picked neuron, or the summary of the network
        let panel = match readout {
            Some(info) => v_flex()
                .gap_1()
                .child(
                    div()
                        .text_sm()
                        .font_weight(FontWeight::SEMIBOLD)
                        .text_color(title_c)
                        .child(info.title),
                )
                .children(
                    info.lines
                        .into_iter()
                        .map(|l| div().text_xs().text_color(body_c).child(l)),
                )
                .children(info.links.into_iter().map(|(heading, rows)| {
                    v_flex()
                        .gap_0p5()
                        .child(
                            div()
                                .text_xs()
                                .font_weight(FontWeight::SEMIBOLD)
                                .text_color(tag_c)
                                .child(heading),
                        )
                        .children(rows.into_iter().map(|(name, w)| {
                            h_flex()
                                .gap_2()
                                .child(div().w(px(60.)).text_xs().text_color(body_c).child(name))
                                .child(
                                    div()
                                        .text_xs()
                                        .text_color(if w >= 0.0 { pos_c } else { neg_c })
                                        .child(format!("{w:+.3}")),
                                )
                        }))
                }))
                .child(
                    div()
                        .text_xs()
                        .text_color(muted)
                        .child("Click the neuron again, or empty space, to let go"),
                )
                .into_any_element(),
            None => v_flex()
                .gap_1()
                .child(
                    div()
                        .text_sm()
                        .font_weight(FontWeight::SEMIBOLD)
                        .text_color(title_c)
                        .child("Network"),
                )
                .children(
                    summary
                        .into_iter()
                        .map(|l| div().text_xs().text_color(body_c).child(l)),
                )
                .child(
                    div()
                        .text_xs()
                        .text_color(muted)
                        .child("Click a neuron to follow its connections"),
                )
                .into_any_element(),
        };
        overlays.push(
            div()
                .absolute()
                .right(px(12.))
                .top(px(46.))
                .w(px(250.))
                .px_2()
                .py_2()
                .bg(box_bg.opacity(0.88))
                .border_1()
                .border_color(box_edge)
                .rounded_sm()
                .child(panel)
                .into_any_element(),
        );

        // Where the run stands
        overlays.push(
            div()
                .absolute()
                .left(px(12.))
                .top(px(12.))
                .px_2()
                .py_1()
                .bg(box_bg.opacity(0.80))
                .rounded_sm()
                .text_xs()
                .text_color(body_c)
                .child(status)
                .into_any_element(),
        );

        // What the colours mean
        let key = |swatch: AnyElement, text: &'static str| {
            h_flex()
                .gap_2()
                .items_center()
                .child(swatch)
                .child(div().text_xs().text_color(body_c).child(text))
        };
        let line = |c: gpui_kit::Hsla| {
            div()
                .w(px(18.))
                .h(px(3.))
                .rounded_full()
                .bg(c)
                .into_any_element()
        };
        let dot = |c: gpui_kit::Hsla| {
            div()
                .w(px(8.))
                .h(px(8.))
                .rounded_full()
                .bg(c)
                .into_any_element()
        };
        // Too big for a curve per connection: the ribbons are sums, and the legend says so
        let aggregated = layout.as_ref().is_some_and(|l| l.lod);
        let legend = v_flex()
            .absolute()
            .right(px(12.))
            .bottom(px(10.))
            .gap_1()
            .px_2()
            .py_1()
            .bg(box_bg.opacity(0.80))
            .rounded_sm();
        let legend = if aggregated {
            legend
                .child(key(line(pos_c), "ribbon: the connections between two blocks of neurons mostly excite"))
                .child(key(line(neg_c), "ribbon: they mostly inhibit"))
                .child(key(dot(color(Pixel::rgb(200, 255, 220))), "bead: spikes leaving a block that fires"))
                .child(key(dot(color(Pixel::rgb(255, 255, 255))), "thin curve: one of the strongest single connections in view"))
                .child(div().text_xs().text_color(muted).child("Bar brightness = how much a block fires · zoom in and the blocks open into neurons"))
        } else {
            legend
                .child(key(line(pos_c), "positive weight (excites)"))
                .child(key(line(neg_c), "negative weight (inhibits)"))
                .child(key(
                    dot(color(Pixel::rgb(200, 255, 220))),
                    "spike travelling along a connection",
                ))
                .child(key(
                    dot(color(Pixel::rgb(255, 215, 120))),
                    "neuron that fired on the last step",
                ))
        };
        overlays.push(legend.into_any_element());

        let vm = self.vm.clone();
        let hint = div()
            .absolute()
            .left(px(12.))
            .bottom(px(10.))
            .text_xs()
            .text_color(muted)
            .child("Wheel: zoom · drag: pan · hover or click a neuron");
        let reset = div().absolute().right(px(12.)).top(px(10.)).child(
            Button::new("net-reset")
                .small()
                .label("Reset view")
                .on_click(move |_, _, cx| vm.update(cx, |vm, cx| vm.net_reset(cx))),
        );
        let wait = image.is_none().then(|| {
            div()
                .absolute()
                .size_full()
                .flex()
                .items_center()
                .justify_center()
                .text_sm()
                .text_color(tag_c)
                .child("Waiting for the first frames…")
        });
        div()
            .id("training-network")
            .test_support()
            .size_full()
            .relative()
            .overflow_hidden()
            .bg(color(Pixel::rgb(7, 10, 28)))
            .on_mouse_down(MouseButton::Left, cx.listener(Self::on_net_down))
            .on_mouse_move(cx.listener(Self::on_net_move))
            .on_mouse_up(MouseButton::Left, cx.listener(Self::on_net_up))
            .on_scroll_wheel(cx.listener(Self::on_net_wheel))
            .on_hover(cx.listener(|this, hovered: &bool, _, cx| {
                if !*hovered && this.net_hover.take().is_some() {
                    cx.notify();
                }
            }))
            .child(self.image_canvas(PlotSlot::Network, self.net_bounds.clone(), known, image))
            .children(overlays)
            .child(hint)
            .child(reset)
            .children(wait)
            .into_any_element()
    }

    /// The overview: one image holding the graphics of ten panels, with their titles and numbers
    /// laid over it from the same layout the renderer used ([`panel_rects`]).
    fn dashboard_body(&self, cx: &mut Context<Self>) -> AnyElement {
        let (palette, known, image, info, facts) = {
            let vm = self.vm.read(cx);
            let info = vm.dashboard.as_ref().map(|d| d.info);
            (
                vm.palette(cx),
                vm.plot_size(PlotSlot::Dashboard),
                vm.dashboard.as_ref().map(|d| d.image.clone()),
                info,
                DashFacts::of(&vm.state, info),
            )
        };
        let t = cx.theme();
        let (fg, muted) = (t.foreground, t.muted_foreground);
        let text = color(palette.text);
        let wait = image.is_none();

        let mut overlays: Vec<AnyElement> = Vec::new();
        for r in panel_rects() {
            let (title, aside): (&'static str, Option<AnyElement>) = match r.kind {
                PanelKind::Network => ("Neural Network", None),
                PanelKind::Raster => (
                    "Spike Raster Plot",
                    Some(legend(
                        &palette,
                        &[("input", 0), ("hidden", 1), ("output", 2)],
                        text,
                    )),
                ),
                PanelKind::Accuracy => (
                    "Training Accuracy",
                    facts.accuracy.clone().map(|a| aside_text(a, muted)),
                ),
                PanelKind::HeatInputHidden => (
                    "Input : Hidden",
                    facts.weight_scale.clone().map(|a| aside_text(a, muted)),
                ),
                PanelKind::HeatHiddenOutput => (
                    "Hidden : Output",
                    facts.weight_scale.clone().map(|a| aside_text(a, muted)),
                ),
                PanelKind::Rates => (
                    "Spike Rates (Hidden)",
                    facts.rates.clone().map(|a| aside_text(a, muted)),
                ),
                PanelKind::WeightDist => (
                    "Weight Dist.",
                    facts.weight_range.clone().map(|a| aside_text(a, muted)),
                ),
                PanelKind::Pca => (
                    "State Space (PCA)",
                    info.and_then(|i| i.pca_variance).map(|(a, b)| {
                        aside_text(
                            format!("PC1 {:.0}% · PC2 {:.0}%", a * 100.0, b * 100.0),
                            muted,
                        )
                    }),
                ),
                PanelKind::Metrics => ("Learning Metrics", None),
                PanelKind::Run => ("Run", None),
            };
            overlays.push(panel_header(&r, title, aside, text));
            match r.kind {
                PanelKind::Network => overlays.push(network_labels(&r, text)),
                PanelKind::Metrics => {
                    overlays.push(panel_body(&r, metrics_rows(&facts, fg, muted)))
                }
                PanelKind::Run => overlays.push(panel_body(
                    &r,
                    run_summary(&facts, &palette, fg, muted, color(palette.channel(1))),
                )),
                _ => {}
            }
        }
        let hint = wait.then(|| {
            div()
                .absolute()
                .size_full()
                .flex()
                .items_center()
                .justify_center()
                .text_sm()
                .text_color(text)
                .child("Waiting for the first frames…")
        });
        div()
            .id("training-dashboard")
            .test_support()
            .size_full()
            .relative()
            .bg(color(palette.background))
            .child(self.plot_canvas(PlotSlot::Dashboard, known, image))
            .children(overlays)
            .children(hint)
            .into_any_element()
    }

    fn curves_body(&self, cx: &mut Context<Self>) -> AnyElement {
        let (palette, known, image, panels, last_steps) = {
            let vm = self.vm.read(cx);
            (
                vm.palette(cx),
                vm.plot_size(PlotSlot::Curves),
                vm.curves.as_ref().map(|c| c.image.clone()),
                vm.curves
                    .as_ref()
                    .map(|c| c.panels.clone())
                    .unwrap_or_default(),
                vm.state.loss.total(),
            )
        };
        let (text, strong) = (color(palette.text), color(palette.channel(0)));
        let _ = strong;
        let labels = panels.into_iter().enumerate().map(|(i, p)| {
            let tint = color(palette.channel(i));
            let range = if p.points == 0 {
                String::new()
            } else {
                format!(
                    "{}  –  {}",
                    axis_value(p.lo, p.log, p.name),
                    axis_value(p.hi, p.log, p.name)
                )
            };
            let last = p.last.map_or("–".to_string(), |v| last_value(v, p.name));
            div()
                .absolute()
                .left(px(10.))
                .right(px(10.))
                .top(relative(p.y0))
                .mt(px(4.))
                .flex()
                .justify_between()
                .child(
                    h_flex()
                        .gap_2()
                        .child(
                            div()
                                .text_xs()
                                .font_weight(FontWeight::SEMIBOLD)
                                .text_color(tint)
                                .child(p.name),
                        )
                        .child(div().text_xs().text_color(tint).child(last)),
                )
                .child(div().text_xs().text_color(text).child(range))
        });
        let hint = (last_steps == 0).then(|| {
            div()
                .absolute()
                .size_full()
                .flex()
                .items_center()
                .justify_center()
                .text_sm()
                .text_color(text)
                .child("Waiting for the first step…")
        });
        div()
            .id("training-curves")
            .test_support()
            .size_full()
            .relative()
            .bg(color(palette.background))
            .child(self.plot_canvas(PlotSlot::Curves, known, image))
            .children(labels)
            .children(hint)
            .into_any_element()
    }

    fn raster_body(&self, window: &Window, cx: &mut Context<Self>) -> AnyElement {
        let scale = window.scale_factor();
        let (
            palette,
            known,
            image,
            layout,
            stats,
            layer_names,
            selected,
            position,
            live,
            pinned_label,
        ) = {
            let vm = self.vm.read(cx);
            let layer = vm.state.selected_layer();
            (
                vm.palette(cx),
                vm.plot_size(PlotSlot::Raster),
                vm.raster.as_ref().map(|r| r.image.clone()),
                vm.raster.as_ref().map(|r| r.layout),
                vm.raster.as_ref().map(|r| r.stats),
                vm.state
                    .layers()
                    .iter()
                    .map(|l| SharedString::from(l.name.clone()))
                    .collect::<Vec<_>>(),
                vm.state.selected_index(),
                layer.and_then(|l| l.position()),
                layer.is_none_or(|l| l.is_live()),
                vm.state.shown_frame().map(|f| {
                    format!(
                        "epoch {} · batch {} · step {}",
                        f.progress.epoch + 1,
                        f.progress.batch + 1,
                        f.progress.step() + 1
                    )
                }),
            )
        };
        let t = cx.theme();
        let (muted, border) = (t.muted_foreground, t.border);
        let vm = self.vm.clone();

        if layer_names.is_empty() {
            return div()
                .id("training-raster")
                .test_support()
                .size_full()
                .flex()
                .items_center()
                .justify_center()
                .text_sm()
                .text_color(muted)
                .child("No spike frames yet: they arrive a few times a second while a run is going")
                .into_any_element();
        }

        let layer_select = {
            let vm = vm.clone();
            MenuSelect::new("training-layer", layer_names, selected, move |i, _, cx| {
                vm.update(cx, |vm, cx| vm.select_layer(i, cx))
            })
        };
        let nav = {
            let (back, fwd, live_vm) = (vm.clone(), vm.clone(), vm.clone());
            h_flex()
                .gap_0p5()
                .child(
                    Button::new("frame-back")
                        .small()
                        .ghost()
                        .label("◀")
                        .tooltip("Earlier frame")
                        .on_click(move |_, _, cx| back.update(cx, |vm, cx| vm.step_back(cx))),
                )
                .child(
                    Button::new("frame-live")
                        .small()
                        .ghost()
                        .label("Live")
                        .tooltip("Follow the newest frame")
                        .selected(live)
                        .on_click(move |_, _, cx| live_vm.update(cx, |vm, cx| vm.go_live(cx))),
                )
                .child(
                    Button::new("frame-forward")
                        .small()
                        .ghost()
                        .label("▶")
                        .tooltip("Later frame")
                        .on_click(move |_, _, cx| fwd.update(cx, |vm, cx| vm.step_forward(cx))),
                )
        };
        let place = {
            let mut text = pinned_label.unwrap_or_default();
            if let Some((i, n)) = position {
                text.push_str(&format!(" · frame {i}/{n}"));
            }
            div().text_xs().text_color(muted).child(text)
        };
        let caption = stats.map(|s| {
            div().text_xs().text_color(muted).child(format!(
                "{} neurons × {} steps · {:.0}% fired · {:.0}% silent · {} always on · burst ×{:.1}",
                s.units,
                s.steps,
                s.mean_rate * 100.0,
                s.silent_fraction * 100.0,
                s.saturated,
                s.burstiness
            ))
        });

        // Labels on the image, placed from the layout the renderer returned (physical px)
        let text = color(palette.text);
        let overlays = layout.filter(|l| l.rows > 0).map(|l| {
            let (rate_x, strip_y) = (l.rate_x as f32 / scale, l.strip_y as f32 / scale);
            div()
                .absolute()
                .size_full()
                .child(
                    div()
                        .absolute()
                        .left(px(rate_x))
                        .top(px(2.))
                        .text_xs()
                        .text_color(text)
                        .child("rate"),
                )
                .when(l.strip_h > 0, |d| {
                    d.child(
                        div()
                            .absolute()
                            .left(px(6.))
                            .top(px(strip_y + 2.))
                            .text_xs()
                            .text_color(text)
                            .child("population firing"),
                    )
                })
        });
        let axis = layout.filter(|l| l.cols > 0).map(|l| {
            let w = l.plot_w as f32 / scale;
            let steps = stats.map_or(0, |s| s.steps);
            h_flex()
                .w(px(w))
                .justify_between()
                .text_xs()
                .text_color(muted)
                .child("step 1")
                .child(format!("step {steps}  →  time"))
        });

        v_flex()
            .id("training-raster")
            .test_support()
            .size_full()
            .child(
                h_flex()
                    .w_full()
                    .gap_3()
                    .px_2()
                    .py_1()
                    .border_b_1()
                    .border_color(border)
                    .child(layer_select)
                    .child(nav)
                    .child(place)
                    .child(div().flex_1())
                    .children(caption),
            )
            .child(
                div()
                    .flex_1()
                    .min_h_0()
                    .relative()
                    .bg(color(palette.background))
                    .child(self.plot_canvas(PlotSlot::Raster, known, image))
                    .children(overlays),
            )
            .child(h_flex().px_2().py_0p5().children(axis))
            .into_any_element()
    }

    fn health_body(&self, cx: &mut Context<Self>) -> AnyElement {
        let (palette, findings, rows) = {
            let vm = self.vm.read(cx);
            let st = &vm.state;
            let mut rows: Vec<(String, String)> = vec![
                ("Status".into(), st.status.label().into()),
                (
                    "Steps".into(),
                    format!("{} of {}", st.steps_seen, st.progress.total_steps()),
                ),
                (
                    "Loss".into(),
                    metric_line(st.loss.last(), st.loss.best(), "best", |v| {
                        format!("{v:.4}")
                    }),
                ),
                (
                    "Accuracy".into(),
                    metric_line(
                        st.accuracy.last(),
                        st.accuracy.values().iter().copied().reduce(f32::max),
                        "best",
                        |v| format!("{:.1}%", v * 100.0),
                    ),
                ),
                (
                    "Gradient norm".into(),
                    st.grad.last().map_or("–".into(), |v| format!("{v:.3e}")),
                ),
                ("Events dropped".into(), format!("{}", st.dropped)),
            ];
            for (name, a) in &st.activity {
                rows.push((format!("Activity · {name}"), activity_line(a)));
            }
            (vm.palette(cx), st.findings(), rows)
        };
        let t = cx.theme();
        let (muted, border, fg) = (t.muted_foreground, t.border, t.foreground);

        let verdict = findings
            .into_iter()
            .enumerate()
            .map(|(i, f)| finding_row(i, f, &palette, fg, muted));
        let facts = rows.into_iter().map(|(k, v)| {
            h_flex()
                .gap_3()
                .child(div().w(px(150.)).text_sm().text_color(muted).child(k))
                .child(div().text_sm().text_color(fg).child(v))
        });
        div()
            .id("training-health")
            .test_support()
            .size_full()
            .overflow_y_scroll()
            .child(
                v_flex()
                    .p_4()
                    .gap_4()
                    .max_w(px(820.))
                    .child(
                        v_flex()
                            .gap_2()
                            .child(
                                div()
                                    .text_xs()
                                    .font_weight(FontWeight::SEMIBOLD)
                                    .text_color(muted)
                                    .child("State of the network"),
                            )
                            .children(verdict),
                    )
                    .child(div().h(px(1.)).w_full().bg(border))
                    .child(
                        v_flex()
                            .gap_1()
                            .child(
                                div()
                                    .text_xs()
                                    .font_weight(FontWeight::SEMIBOLD)
                                    .text_color(muted)
                                    .child("The run"),
                            )
                            .children(facts),
                    ),
            )
            .into_any_element()
    }
    // ─── structure ──────────────────────────────────────────────────────

    fn evolution_body(&self, cx: &mut Context<Self>) -> AnyElement {
        let (palette, known, image, panels, summary, findings) = {
            let vm = self.vm.read(cx);
            let st = &vm.state.evolution;
            (
                vm.palette(cx),
                vm.plot_size(PlotSlot::Evolution),
                vm.evolution.as_ref().map(|e| e.image.clone()),
                vm.evolution
                    .as_ref()
                    .map(|e| e.panels.clone())
                    .unwrap_or_default(),
                st.summary(),
                st.findings(),
            )
        };
        let t = cx.theme();
        let (muted, border, fg) = (t.muted_foreground, t.border, t.foreground);

        if summary.is_empty() {
            return div()
                .id("training-evolution")
                .test_support()
                .size_full()
                .flex()
                .items_center()
                .justify_center()
                .text_sm()
                .text_color(muted)
                .child("No generation reports: this run is not an architecture search. Start the \"Evolution demo\" to see one.")
                .into_any_element();
        }

        let text = color(palette.text);
        let tiles = h_flex()
            .w_full()
            .flex_wrap()
            .gap_x_6()
            .gap_y_1()
            .px_2()
            .py_1()
            .border_b_1()
            .border_color(border)
            .children(summary.into_iter().map(|(label, value)| {
                v_flex()
                    .gap_0p5()
                    .child(div().text_xs().text_color(muted).child(label))
                    .child(
                        div()
                            .text_sm()
                            .font_weight(FontWeight::MEDIUM)
                            .text_color(fg)
                            .child(value),
                    )
            }));
        // Names, last values and ranges over the image, like the other charts; the first panel
        // has three lines, so its legend is spelled out below
        let labels = panels.into_iter().enumerate().map(|(i, p)| {
            let tint = color(palette.channel([0, 0, 3][i % 3]));
            let last = p.last.map_or("–".to_string(), |v| {
                if i == 0 {
                    format!("{v:.3}")
                } else {
                    format!("{v:.0}")
                }
            });
            let range = if p.points == 0 {
                String::new()
            } else if i == 0 {
                format!("{:.2}  –  {:.2}", p.lo, p.hi)
            } else {
                format!("0  –  {:.0}", p.hi)
            };
            div()
                .absolute()
                .left(px(10.))
                .right(px(10.))
                .top(relative(p.y0))
                .mt(px(4.))
                .flex()
                .justify_between()
                .child(
                    h_flex()
                        .gap_2()
                        .child(
                            div()
                                .text_xs()
                                .font_weight(FontWeight::SEMIBOLD)
                                .text_color(tint)
                                .child(p.name),
                        )
                        .child(div().text_xs().text_color(tint).child(last)),
                )
                .child(div().text_xs().text_color(text).child(range))
        });
        let legend = h_flex()
            .gap_3()
            .px_2()
            .py_0p5()
            .text_xs()
            .text_color(muted)
            .children(
                [(0usize, "best"), (1, "mean"), (3, "champion of all time")].map(|(ch, name)| {
                    h_flex()
                        .gap_1()
                        .items_center()
                        .child(
                            div()
                                .size(px(8.))
                                .rounded_full()
                                .bg(color(palette.channel(ch))),
                        )
                        .child(name)
                }),
            );
        let chart = v_flex()
            .flex_1()
            .min_h_0()
            .child(
                div()
                    .flex_1()
                    .min_h_0()
                    .relative()
                    .bg(color(palette.background))
                    .child(self.plot_canvas(PlotSlot::Evolution, known, image))
                    .children(labels),
            )
            .child(legend);
        let verdict = findings
            .into_iter()
            .enumerate()
            .map(|(i, f)| finding_row(i, f, &palette, fg, muted));
        v_flex()
            .id("training-evolution")
            .test_support()
            .size_full()
            .child(tiles)
            .child(chart)
            .child(
                div()
                    .id("evolution-verdict")
                    .max_h(px(140.))
                    .w_full()
                    .border_t_1()
                    .border_color(border)
                    .overflow_y_scroll()
                    .child(v_flex().p_2().gap_1().children(verdict)),
            )
            .into_any_element()
    }

    fn structure_body(&self, window: &Window, cx: &mut Context<Self>) -> AnyElement {
        let scale = window.scale_factor();
        let (
            palette,
            counts_known,
            topo_known,
            counts_image,
            panels,
            topo_image,
            topo_layout,
            summary,
            findings,
            rows,
            snapshot_note,
        ) = {
            let vm = self.vm.read(cx);
            let st = &vm.state.structure;
            let note = st.topology.as_ref().map(|t| {
                let mut text = format!(
                    "{} neurons · {} synapses",
                    t.total_neurons, t.total_synapses
                );
                if t.truncated {
                    text.push_str(&format!(
                        " · drawing the first {} neurons and {} synapses",
                        t.nodes.len(),
                        t.edges.len()
                    ));
                }
                text
            });
            (
                vm.palette(cx),
                vm.plot_size(PlotSlot::Counts),
                vm.plot_size(PlotSlot::Topology),
                vm.counts.as_ref().map(|c| c.image.clone()),
                vm.counts
                    .as_ref()
                    .map(|c| c.panels.clone())
                    .unwrap_or_default(),
                vm.topology.as_ref().map(|t| t.image.clone()),
                vm.topology.as_ref().map(|t| t.layout.clone()),
                st.summary(),
                st.findings(&vm.state.loss),
                st.events_newest_first()
                    .take(60)
                    .map(EventRow::of)
                    .collect::<Vec<_>>(),
                note,
            )
        };
        let t = cx.theme();
        let (muted, border, fg) = (t.muted_foreground, t.border, t.foreground);
        let _ = scale;

        if summary.is_empty() {
            return div()
                .id("training-structure")
                .test_support()
                .size_full()
                .flex()
                .items_center()
                .justify_center()
                .text_sm()
                .text_color(muted)
                .child("No structural reports: this run does not use structural plasticity. Start the \"Structural demo\" to see it work.")
                .into_any_element();
        }

        let text = color(palette.text);
        let tiles = h_flex()
            .w_full()
            .flex_wrap()
            .gap_x_6()
            .gap_y_1()
            .px_2()
            .py_1()
            .border_b_1()
            .border_color(border)
            .children(summary.into_iter().map(|(label, value)| {
                v_flex()
                    .gap_0p5()
                    .child(div().text_xs().text_color(muted).child(label))
                    .child(
                        div()
                            .text_sm()
                            .font_weight(FontWeight::MEDIUM)
                            .text_color(fg)
                            .child(value),
                    )
            }));

        // Chart: names, last values and ranges laid over the image, like the curves tab
        let labels = panels.into_iter().enumerate().map(|(i, p)| {
            let tint = color(palette.channel([2, 0, 3][i % 3]));
            let last = p.last.map_or("–".to_string(), |v| {
                if p.name == "Mean spike rate" {
                    format!("{v:.3}")
                } else {
                    format!("{v:.0}")
                }
            });
            let range = if p.points == 0 {
                String::new()
            } else if p.name == "Mean spike rate" {
                format!("0  –  {:.3}", p.hi)
            } else {
                format!("0  –  {:.0}", p.hi)
            };
            div()
                .absolute()
                .left(px(10.))
                .right(px(10.))
                .top(relative(p.y0))
                .mt(px(4.))
                .flex()
                .justify_between()
                .child(
                    h_flex()
                        .gap_2()
                        .child(
                            div()
                                .text_xs()
                                .font_weight(FontWeight::SEMIBOLD)
                                .text_color(tint)
                                .child(p.name),
                        )
                        .child(div().text_xs().text_color(tint).child(last)),
                )
                .child(div().text_xs().text_color(text).child(range))
        });
        let legend = h_flex()
            .gap_3()
            .px_2()
            .py_0p5()
            .children(MarkerKind::ALL.into_iter().map(|k| {
                h_flex()
                    .gap_1()
                    .items_center()
                    .child(
                        div()
                            .size(px(8.))
                            .rounded_full()
                            .bg(color(marker_color(&palette, k))),
                    )
                    .child(div().text_xs().text_color(muted).child(k.label()))
            }));
        let chart = v_flex()
            .flex_1()
            .min_w_0()
            .min_h_0()
            .h_full()
            .child(
                div()
                    .flex_1()
                    .min_h_0()
                    .relative()
                    .bg(color(palette.background))
                    .child(self.plot_canvas(PlotSlot::Counts, counts_known, counts_image))
                    .children(labels),
            )
            .child(legend);

        // Graph: a caption over the image; what it draws is the snapshot the controller sent
        let unreachable = topo_layout.as_ref().map_or(0, |l| l.unreachable);
        let graph = v_flex().flex_1().min_w_0().min_h_0().h_full().border_l_1().border_color(border).child(
            div()
                .flex_1()
                .min_h_0()
                .relative()
                .bg(color(palette.background))
                .child(self.plot_canvas(PlotSlot::Topology, topo_known, topo_image))
                .child(div().absolute().left(px(10.)).top(px(4.)).text_xs().text_color(text).child(snapshot_note.unwrap_or_else(|| "waiting for the first graph…".into())))
                .child(
                    h_flex().absolute().left(px(10.)).bottom(px(4.)).gap_3().text_xs().text_color(text).children([(0usize, "inputs"), (2, "hidden"), (3, "outputs")].map(|(ch, name)| {
                        h_flex().gap_1().items_center().child(div().size(px(8.)).rounded_full().bg(color(palette.channel(ch)))).child(name)
                    })).child("synapses: green excitatory, red inhibitory · yellow ring = neuron grown during the run"),
                )
                .when(unreachable > 0, |d| {
                    d.child(div().absolute().right(px(10.)).bottom(px(18.)).text_xs().text_color(color(palette.playhead)).child(format!("{unreachable} hidden neuron(s) no input reaches (bottom row)")))
                }),
        );

        let verdict = findings
            .into_iter()
            .enumerate()
            .map(|(i, f)| finding_row(i, f, &palette, fg, muted));
        let list = rows.into_iter().enumerate().map(|(i, r)| {
            let dot = r.marker.map_or(palette.text, |k| marker_color(&palette, k));
            v_flex()
                .id(("structure-event", i))
                .gap_0p5()
                .pb_1()
                .child(
                    h_flex()
                        .gap_2()
                        .items_center()
                        .child(div().size(px(8.)).rounded_full().bg(color(dot)).flex_none())
                        .child(
                            div()
                                .text_sm()
                                .font_weight(FontWeight::MEDIUM)
                                .text_color(fg)
                                .child(r.head),
                        )
                        .child(div().text_xs().text_color(muted).child(r.outcome)),
                )
                .child(
                    div()
                        .pl(px(16.))
                        .text_xs()
                        .text_color(muted)
                        .child(format!("why: {}", r.reason)),
                )
                .child(
                    div()
                        .pl(px(16.))
                        .text_xs()
                        .text_color(muted)
                        .child(match r.loss {
                            Some(loss) => format!("{loss} · {}", r.size),
                            None => r.size,
                        }),
                )
        });

        v_flex()
            .id("training-structure")
            .test_support()
            .size_full()
            .child(tiles)
            // `h_flex` centres its children: without `h_full` on them the columns shrink to their content,
            // the canvases get a zero height, never report a size and no image is ever drawn
            .child(
                h_flex()
                    .w_full()
                    .flex_1()
                    .min_h_0()
                    .child(chart)
                    .child(graph),
            )
            .child(
                div()
                    .id("structure-log")
                    .h(px(220.))
                    .w_full()
                    .border_t_1()
                    .border_color(border)
                    .overflow_y_scroll()
                    .child(
                        v_flex()
                            .p_2()
                            .gap_2()
                            .child(v_flex().gap_1().children(verdict))
                            .child(
                                div()
                                    .text_xs()
                                    .font_weight(FontWeight::SEMIBOLD)
                                    .text_color(muted)
                                    .child("Structural changes, newest first"),
                            )
                            .child(v_flex().gap_1().children(list)),
                    ),
            )
            .into_any_element()
    }
}

impl Render for TrainingView {
    fn render(&mut self, window: &mut Window, cx: &mut Context<Self>) -> impl IntoElement {
        let (tab, status) = {
            let vm = self.vm.read(cx);
            (vm.tab, vm.state.status)
        };
        let body = if status == RunStatus::Idle {
            self.empty()
        } else {
            match tab {
                TrainingTab::Network => self.network_body(cx),
                TrainingTab::Brain => self.brain_body(cx),
                TrainingTab::Dashboard => self.dashboard_body(cx),
                TrainingTab::Curves => self.curves_body(cx),
                TrainingTab::Raster => self.raster_body(window, cx),
                TrainingTab::Structure => self.structure_body(window, cx),
                TrainingTab::Evolution => self.evolution_body(cx),
                TrainingTab::Health => self.health_body(cx),
            }
        };
        v_flex()
            .id("training")
            .test_support()
            .size_full()
            .child(self.toolbar(cx))
            .child(div().flex_1().min_h_0().child(body))
    }
}

// ─── brain text ─────────────────────────────────────────────────────────

/// `text` cut to at most `max` characters, with an ellipsis when it was longer.
fn shorten(text: &str, max: usize) -> String {
    if text.chars().count() <= max {
        text.to_string()
    } else {
        text.chars().take(max.saturating_sub(1)).collect::<String>() + "…"
    }
}

/// `1.234.567`: digits in groups of three, as the counters read.
fn thousands(n: u64) -> String {
    let digits = n.to_string();
    let mut out = String::new();
    for (i, c) in digits.chars().enumerate() {
        if i > 0 && (digits.len() - i) % 3 == 0 {
            out.push('.');
        }
        out.push(c);
    }
    out
}

/// The headline of the brain view: what fired, how many synapses took part, and where the trace began.
fn brain_headline(info: &BrainInfo) -> (String, Vec<String>, Option<String>) {
    // A few names, then a count: a big output layer can fire dozens of neurons at once
    let mut fired = info
        .output_fired
        .iter()
        .take(3)
        .map(|&u| unit_name(2, u).to_uppercase())
        .collect::<Vec<_>>()
        .join(" + ");
    if info.output_fired.len() > 3 {
        fired.push_str(&format!(" + {} MORE", info.output_fired.len() - 3));
    }
    let headline = if fired.is_empty() {
        "NO OUTPUT SPIKE IN THIS FRAME".to_string()
    } else {
        format!("{fired} FIRED")
    };
    let lines = vec![if info.trace.is_empty() {
        "No excitatory synapse fired together in this frame".to_string()
    } else {
        format!(
            "{} synapses that fired in order, across {} steps of the latest frame",
            info.trace.len(),
            info.steps_per_frame
        )
    }];
    let origin = info.trace.first().map(|first| {
        let from = unit_name(first.from.0, first.from.1);
        match info.output_step {
            Some(t) if t >= first.step => format!(
                "The trace starts at {from} on step {}, {} steps before {fired} fired",
                first.step,
                t - first.step
            ),
            _ => format!("The trace starts at {from} on step {}", first.step),
        }
    });
    (headline, lines, origin)
}

// ─── network text ───────────────────────────────────────────────────────

/// The three lines of a layer's label.
struct NetTexts {
    title: String,
    neurons: String,
    activation: String,
}

impl NetTexts {
    /// One label per layer (input, hidden, output), with the live firing share where frames exist.
    fn of(st: &TrainingState) -> Vec<NetTexts> {
        let rate = |name: &str| {
            st.layer_named(name)
                .and_then(|l| l.shown())
                .map(|s| s.frame.clone())
                .filter(|f| !f.is_empty())
                .map(|f| (f.units, f.mean_rate()))
        };
        [
            ("INPUT", LAYER_INPUT, "Poisson rate code"),
            ("ML 1 · HIDDEN", LAYER_HIDDEN, "RLeaky (LIF)"),
            ("OUTPUT", LAYER_OUTPUT, "Leaky readout"),
        ]
        .into_iter()
        .map(|(title, name, activation)| {
            let live = rate(name);
            NetTexts {
                title: title.to_string(),
                neurons: match live {
                    Some((units, r)) => format!("Neurons: {units} · {:.0}% firing", r * 100.0),
                    None => {
                        if st.weights.is_empty() && st.sparse.is_empty() {
                            "Neurons: –".to_string()
                        } else {
                            "Neurons: …".to_string()
                        }
                    }
                },
                activation: format!("Activation: {activation}"),
            }
        })
        .collect()
    }
}

/// What the readout says about one neuron.
struct NeuronInfo {
    title: String,
    lines: Vec<String>,
    /// `(heading, [(neuron, weight)])`: the strongest connections in and out. Empty for the tooltip.
    links: Vec<(String, Vec<(String, f32)>)>,
}

/// `x3`, `h10`, `class 1`: how the network view names a neuron.
fn unit_name(layer: usize, unit: usize) -> String {
    match layer {
        0 => format!("x{unit}"),
        1 => format!("h{unit}"),
        _ => format!("class {unit}"),
    }
}

const LAYER_NAMES: [(&str, &str); 3] = [
    (LAYER_INPUT, "input"),
    (LAYER_HIDDEN, "hidden"),
    (LAYER_OUTPUT, "output"),
];

/// The links between layer `pair` and `pair + 1`: `rows()` neurons of the first, `cols()` of the second.
fn pair_links(st: &TrainingState, pair: usize) -> Option<crate::engine::training::links::Links> {
    st.connections(
        [WEIGHTS_INPUT_HIDDEN, WEIGHTS_HIDDEN_OUTPUT]
            .get(pair)
            .copied()?,
    )
}

/// Links read per frame at most when listing a neuron's strongest connections: a list this long is read
/// on every redraw while the neuron is picked, so a bigger one is not listed (the counts still show).
const MAX_LISTED_LINKS: usize = 2_000_000;

/// The strongest three connections into (`incoming`) or out of neuron `u` of layer `l`.
fn top_links(st: &TrainingState, l: usize, u: usize, incoming: bool) -> Vec<(String, f32)> {
    let (pair, other) = if incoming {
        (l.checked_sub(1), l.wrapping_sub(1))
    } else {
        (Some(l), l + 1)
    };
    let Some(m) = pair.and_then(|p| pair_links(st, p)) else {
        return Vec::new();
    };
    if m.len() > MAX_LISTED_LINKS {
        return Vec::new();
    }
    let strongest = if incoming {
        m.strongest_into(u, 3)
    } else {
        m.strongest_from(u, 3)
    };
    strongest
        .into_iter()
        .map(|(o, w)| (unit_name(other, o), w))
        .collect()
}

/// What is known about one neuron: how often it fires, how it is connected and, with `detailed`, its
/// strongest connections.
fn neuron_info(st: &TrainingState, l: usize, u: usize, detailed: bool) -> Option<NeuronInfo> {
    let (name, layer) = LAYER_NAMES.get(l).copied()?;
    let frame = st
        .layer_named(name)
        .and_then(|s| s.shown())
        .map(|s| s.frame.clone())
        .filter(|f| !f.is_empty() && u < f.units);
    let mut lines = Vec::new();
    match &frame {
        Some(f) => {
            let rate = f.unit_rate(u);
            lines.push(format!(
                "fires on {:.0}% of the steps ({} of {})",
                rate * 100.0,
                (rate * f.steps as f32).round() as usize,
                f.steps
            ));
        }
        None => lines.push("no frame yet".to_string()),
    }
    let fan_in = if l > 0 {
        pair_links(st, l - 1).map(|m| m.rows())
    } else {
        None
    };
    let fan_out = if l < 2 {
        pair_links(st, l).map(|m| m.cols())
    } else {
        None
    };
    lines.push(match (fan_in, fan_out) {
        (Some(i), Some(o)) => format!("receives from {i} neurons · sends to {o}"),
        (Some(i), None) => format!("receives from {i} neurons"),
        (None, Some(o)) => format!("sends to {o} neurons"),
        (None, None) => "connections not seen yet".to_string(),
    });
    let mut links = Vec::new();
    if detailed {
        let into = top_links(st, l, u, true);
        if !into.is_empty() {
            links.push(("Strongest inputs".to_string(), into));
        }
        let out = top_links(st, l, u, false);
        if !out.is_empty() {
            links.push(("Strongest outputs".to_string(), out));
        }
    }
    Some(NeuronInfo {
        title: format!("{layer} neuron {} ({})", u, unit_name(l, u)),
        lines,
        links,
    })
}

/// The lines of the side panel when no neuron is picked.
fn net_summary(st: &TrainingState) -> Vec<String> {
    let sizes: Vec<String> = LAYER_NAMES
        .iter()
        .map(|&(name, _)| {
            st.layer_named(name)
                .and_then(|l| l.shown())
                .map_or("?".to_string(), |s| s.frame.units.to_string())
        })
        .collect();
    let mut lines = vec![format!("Neurons: {}", sizes.join(" → "))];
    let pairs: Vec<_> = (0..2).filter_map(|p| pair_links(st, p)).collect();
    let count: usize = pairs.iter().map(|m| m.len()).sum();
    if count > 0 {
        let wmax = pairs.iter().map(|m| m.abs_max()).fold(0.0f32, f32::max);
        lines.push(format!(
            "Weights: {} · largest {wmax:.2}",
            thousands(count as u64)
        ));
    }
    if let Some(f) = st
        .layer_named(LAYER_HIDDEN)
        .and_then(|l| l.shown())
        .map(|s| s.frame.clone())
        .filter(|f| !f.is_empty())
    {
        let rates: Vec<f32> = (0..f.units).map(|u| f.unit_rate(u)).collect();
        if let Some((best, &top)) = rates.iter().enumerate().max_by(|a, b| a.1.total_cmp(b.1)) {
            lines.push(format!(
                "Busiest hidden: h{best} ({:.0}% of steps)",
                top * 100.0
            ));
        }
        lines.push(format!(
            "Silent hidden neurons: {} of {}",
            rates.iter().filter(|&&r| r == 0.0).count(),
            f.units
        ));
    }
    lines
}

// ─── dashboard text ─────────────────────────────────────────────────────

/// The numbers the dashboard's text panels and titles show, worked out from the state.
struct DashFacts {
    /// Latest accuracy, as the title's note.
    accuracy: Option<String>,
    weight_scale: Option<String>,
    weight_range: Option<String>,
    rates: Option<String>,
    rows: Vec<(&'static str, String)>,
    headline: String,
    progress: String,
    speed: String,
}

impl DashFacts {
    fn of(st: &TrainingState, info: Option<DashboardInfo>) -> Self {
        let pct = |v: f32| format!("{:.1}%", v * 100.0);
        let acc = st.accuracy.values();
        let recent = &acc[acc.len().saturating_sub(50)..];
        let recent_mean =
            (!recent.is_empty()).then(|| recent.iter().sum::<f32>() / recent.len() as f32);

        let weights: Vec<f32> = st
            .weights
            .values()
            .flat_map(|m| m.values.iter().copied())
            .filter(|v| v.is_finite())
            .collect();
        let (w_mean, w_std) = if weights.is_empty() {
            (0.0, 0.0)
        } else {
            let n = weights.len() as f32;
            let mean = weights.iter().sum::<f32>() / n;
            (
                mean,
                (weights.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / n).sqrt(),
            )
        };
        let wmax = info.map_or(0.0, |i| i.weight_abs_max);

        let hidden = st
            .layer_named(LAYER_HIDDEN)
            .and_then(|l| l.shown())
            .map(|s| s.frame.clone())
            .filter(|f| !f.is_empty());
        let rate = hidden.as_ref().map(|f| f.mean_rate());
        let active = hidden.as_ref().map(|f| {
            f.units - ((f.silent_fraction() * f.units as f32).round() as usize).min(f.units)
        });
        let units = hidden.as_ref().map_or(0, |f| f.units);

        let speed = st
            .elapsed()
            .filter(|e| e.as_secs_f64() > 0.2)
            .map_or("–".to_string(), |e| {
                format!("{:.1} steps/s", st.steps_seen as f64 / e.as_secs_f64())
            });
        let rows = vec![
            (
                "Current accuracy",
                st.accuracy.last().map_or("–".into(), pct),
            ),
            ("Recent (50)", recent_mean.map_or("–".into(), pct)),
            (
                "Loss",
                metric_line(st.loss.last(), st.loss.best(), "best", |v| {
                    format!("{v:.4}")
                }),
            ),
            (
                "Gradient norm",
                st.grad.last().map_or("–".into(), |v| format!("{v:.2e}")),
            ),
            (
                "Weights",
                if weights.is_empty() {
                    "–".into()
                } else {
                    format!("mean {w_mean:.3} · std {w_std:.3}")
                },
            ),
            (
                "Avg hidden rate",
                rate.map_or("–".into(), |r| format!("{:.1}% of steps", r * 100.0)),
            ),
            (
                "Active neurons",
                active.map_or("–".into(), |a| format!("{a}/{units}")),
            ),
            (
                "Steps",
                format!("{} of {}", st.steps_seen, st.progress.total_steps()),
            ),
        ];
        Self {
            accuracy: st.accuracy.last().map(pct),
            weight_scale: (wmax > 0.0).then(|| format!("±{wmax:.2}")),
            weight_range: (wmax > 0.0).then(|| format!("{:.2} … {:.2}", -wmax, wmax)),
            rates: rate
                .map(|r| format!("mean {r:.2} · max {:.2}", info.map_or(0.0, |i| i.rate_max))),
            rows,
            headline: st.accuracy.last().map_or("–".into(), pct),
            progress: format!(
                "epoch {}/{} · step {}/{}",
                st.progress.epoch + 1,
                st.progress.epochs.max(1),
                (st.progress.step() + 1).min(st.progress.total_steps().max(1)),
                st.progress.total_steps()
            ),
            speed,
        }
    }
}

/// A panel's title in its top left corner, with a note right-aligned in the same line.
fn panel_header(
    r: &PanelRect,
    title: &'static str,
    aside: Option<AnyElement>,
    tint: gpui_kit::Hsla,
) -> AnyElement {
    div()
        .absolute()
        .left(relative(r.x0))
        .right(relative(1.0 - r.x1))
        .top(relative(r.y0))
        .mt(px(GAP / 2.0 + 6.0))
        .ml(px(PAD + GAP / 2.0))
        .mr(px(PAD + GAP / 2.0))
        .flex()
        .justify_between()
        .child(
            div()
                .text_xs()
                .font_weight(FontWeight::SEMIBOLD)
                .text_color(tint)
                .child(title),
        )
        .children(aside)
        .into_any_element()
}

fn aside_text(text: String, tint: gpui_kit::Hsla) -> AnyElement {
    div()
        .text_xs()
        .text_color(tint)
        .child(text)
        .into_any_element()
}

/// Coloured dots with names; `layers` pairs a name with its layer colour index.
fn legend(palette: &Palette, layers: &[(&'static str, usize)], text: gpui_kit::Hsla) -> AnyElement {
    h_flex()
        .gap_2()
        .children(layers.iter().map(|&(name, l)| {
            h_flex()
                .gap_1()
                .items_center()
                .child(
                    div()
                        .size(px(7.))
                        .rounded_full()
                        .bg(color(layer_color(palette, l)))
                        .flex_none(),
                )
                .child(div().text_xs().text_color(text).child(name))
        }))
        .into_any_element()
}

/// The names under the three columns of the network graph, centred on them (the renderer puts
/// the columns at 8%, 50% and 92% of the panel's width).
fn network_labels(r: &PanelRect, text: gpui_kit::Hsla) -> AnyElement {
    let (w, col_w) = (r.x1 - r.x0, 0.30);
    let labels = [("Input", 0.08), ("Hidden", 0.50), ("Output", 0.92)]
        .into_iter()
        .map(|(name, at)| {
            div()
                .absolute()
                .left(relative(r.x0 + w * (at - col_w / 2.0)))
                .right(relative(1.0 - (r.x0 + w * (at + col_w / 2.0))))
                .bottom(relative(1.0 - r.y1))
                .mb(px(GAP / 2.0 + 5.0))
                .flex()
                .justify_center()
                .text_xs()
                .text_color(text)
                .child(name)
        });
    div()
        .absolute()
        .size_full()
        .children(labels)
        .into_any_element()
}

/// Room for a text panel's content: below its title, inside its padding.
fn panel_body(r: &PanelRect, body: impl IntoElement) -> AnyElement {
    div()
        .absolute()
        .left(relative(r.x0))
        .right(relative(1.0 - r.x1))
        .top(relative(r.y0))
        .bottom(relative(1.0 - r.y1))
        .ml(px(PAD + GAP / 2.0))
        .mr(px(PAD + GAP / 2.0))
        .mt(px(GAP / 2.0 + TITLE_H))
        .mb(px(GAP / 2.0 + PAD))
        .child(body)
        .into_any_element()
}

fn metrics_rows(
    facts: &DashFacts,
    fg: gpui_kit::Hsla,
    muted: gpui_kit::Hsla,
) -> impl IntoElement + use<> {
    v_flex()
        .gap_1()
        .children(facts.rows.clone().into_iter().map(|(k, v)| {
            h_flex()
                .gap_2()
                .child(div().w(px(130.)).text_xs().text_color(muted).child(k))
                .child(div().text_xs().text_color(fg).child(v))
        }))
}

/// The big accuracy, where the run stands, and the colour key.
fn run_summary(
    facts: &DashFacts,
    palette: &Palette,
    fg: gpui_kit::Hsla,
    muted: gpui_kit::Hsla,
    accent: gpui_kit::Hsla,
) -> impl IntoElement + use<> {
    let key = h_flex()
        .gap_3()
        .child(legend(
            palette,
            &[("input", 0), ("hidden", 1), ("output", 2)],
            muted,
        ))
        .child(
            h_flex()
                .gap_1()
                .items_center()
                .child(
                    div()
                        .size(px(7.))
                        .rounded_full()
                        .bg(color(palette.marker))
                        .flex_none(),
                )
                .child(div().text_xs().text_color(muted).child("spike")),
        );
    v_flex()
        .gap_1()
        .child(h_flex().gap_2().items_end().child(div().text_2xl().font_weight(FontWeight::BOLD).text_color(accent).child(facts.headline.clone())).child(div().text_xs().text_color(muted).child("accuracy")))
        .child(div().text_xs().text_color(fg).child(facts.progress.clone()))
        .child(div().text_xs().text_color(muted).child(facts.speed.clone()))
        .child(div().h(px(6.)))
        .child(div().text_xs().text_color(muted).child("Task: tell two noisy 8-feature patterns apart (class 0 near 0.2, class 1 near 0.8), rate-coded over 24 time steps."))
        .child(div().h(px(4.)))
        .child(key)
}

// ─── text helpers ───────────────────────────────────────────────────────

fn finding_row(
    i: usize,
    f: Finding,
    palette: &Palette,
    fg: gpui_kit::Hsla,
    muted: gpui_kit::Hsla,
) -> impl IntoElement {
    let dot: Pixel = match f.severity {
        Severity::Ok => palette.channel(1),
        Severity::Notice => palette.marker,
        Severity::Problem => palette.playhead,
    };
    h_flex()
        .id(("finding", i))
        .gap_2()
        .items_start()
        .child(
            div()
                .mt(px(5.))
                .size(px(8.))
                .rounded_full()
                .bg(color(dot))
                .flex_none(),
        )
        .child(
            v_flex()
                .gap_0p5()
                .child(
                    div()
                        .text_sm()
                        .font_weight(FontWeight::MEDIUM)
                        .text_color(fg)
                        .child(f.what),
                )
                .when(!f.advice.is_empty(), |d| {
                    d.child(div().text_xs().text_color(muted).child(f.advice))
                }),
        )
}

fn axis_value(v: f32, log: bool, name: &str) -> String {
    if log {
        format!("1e{}", v as i32)
    } else if name == "Accuracy" {
        format!("{:.0}%", v * 100.0)
    } else {
        format!("{v:.3}")
    }
}

fn last_value(v: f32, name: &str) -> String {
    match name {
        "Accuracy" => format!("{:.1}%", v * 100.0),
        "Gradient norm" => format!("{v:.2e}"),
        _ => format!("{v:.4}"),
    }
}

fn metric_line(
    last: Option<f32>,
    best: Option<f32>,
    best_label: &str,
    f: impl Fn(f32) -> String,
) -> String {
    match (last, best) {
        (Some(l), Some(b)) => format!("{} ({best_label} {})", f(l), f(b)),
        (Some(l), None) => f(l),
        _ => "–".into(),
    }
}

fn activity_line(a: &ActivityStats) -> String {
    format!(
        "mean {:.4} · |max| {:.3} · {:.0}% zeros · {} values",
        a.mean,
        a.abs_max,
        a.zero_fraction * 100.0,
        a.count
    )
}

#[allow(dead_code)]
fn _panel_is_used(_: &CurvePanel) {}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn long_names_are_cut_with_an_ellipsis_and_short_ones_are_left_alone() {
        assert_eq!(shorten("INPUT", 18), "INPUT");
        assert_eq!(
            shorten("SENSORY AND VISUAL PROJECTION", 18),
            "SENSORY AND VISUA…"
        );
        assert_eq!(
            shorten("SENSORY AND VISUAL PROJECTION", 18).chars().count(),
            18
        );
        assert_eq!(shorten("abcde", 3), "ab…");
    }

    #[test]
    fn counters_group_digits_by_thousands() {
        assert_eq!(thousands(0), "0");
        assert_eq!(thousands(999), "999");
        assert_eq!(thousands(1_000), "1.000");
        assert_eq!(thousands(1_002_040), "1.002.040");
    }

    #[test]
    fn the_headline_names_a_few_outputs_and_counts_the_rest() {
        let mut info = BrainInfo::default();
        assert_eq!(brain_headline(&info).0, "NO OUTPUT SPIKE IN THIS FRAME");
        info.output_fired = vec![1];
        assert_eq!(brain_headline(&info).0, "CLASS 1 FIRED");
        info.output_fired = (0..40).collect();
        assert_eq!(
            brain_headline(&info).0,
            "CLASS 0 + CLASS 1 + CLASS 2 + 37 MORE FIRED"
        );
    }
}
