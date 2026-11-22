//! Rasterizers of the Structure tab, drawn into a [`Frame`] on a work thread like
//! [`super::plot`]'s: the counters over the run (neurons, synapses, mean spike rate) with a mark
//! at every structural change, and the graph of the network as it is now. Text is laid over the
//! images by the view, from the layouts returned here.

use std::sync::Arc;

use neo_snn::structural::{NeuronKind, NeuronPolarity};
use neo_snn::telemetry::TopologySnapshot;

use crate::engine::canvas::{Canvas, Frame, Pixel};
use crate::engine::palette::Palette;

use super::plot::{CurvePanel, GAP, draw_panel_frame, fill, line, map_y, polyline};
use super::structure::MarkerKind;

/// The colour of a change on the charts and in the list.
pub fn marker_color(p: &Palette, kind: MarkerKind) -> Pixel {
    match kind {
        MarkerKind::Growth => p.channel(1),
        MarkerKind::Pruning => p.marker,
        MarkerKind::Rollback => p.playhead,
        MarkerKind::Failed => p.text,
    }
}

// ─── counters over the run ──────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct CountsRequest {
    pub neurons: Vec<f32>,
    pub synapses: Vec<f32>,
    pub active_synapses: Vec<f32>,
    pub mean_rate: Vec<f32>,
    /// Limits drawn as a dashed line (0 = none).
    pub max_neurons: usize,
    pub max_synapses: usize,
    /// `(index into the series, kind)`.
    pub markers: Vec<(usize, MarkerKind)>,
    pub width: u32,
    pub height: u32,
    pub palette: Palette,
}

/// Draws neurons, synapses (all and active) and the mean spike rate as stacked plots sharing the
/// time axis (one point per structural report), with a tick at every change.
pub fn render_counts(req: &CountsRequest) -> (Frame, Vec<CurvePanel>) {
    let (w, h) = (req.width.max(1), req.height.max(1));
    let mut frame = Frame::new(w, h);
    let p = &req.palette;
    let mut canvas = frame.canvas();
    fill(&mut canvas, p.background);

    let n = req.neurons.len();
    let gap = GAP / h as f32;
    let each = (1.0 - gap * 2.0) / 3.0;
    #[allow(clippy::type_complexity)]
    let spec: [(&'static str, &[f32], Option<&[f32]>, usize, Pixel, bool); 3] = [
        (
            "Neurons",
            &req.neurons,
            None,
            req.max_neurons,
            p.channel(2),
            true,
        ),
        (
            "Synapses",
            &req.synapses,
            Some(&req.active_synapses),
            req.max_synapses,
            p.channel(0),
            true,
        ),
        (
            "Mean spike rate",
            &req.mean_rate,
            None,
            0,
            p.channel(3),
            false,
        ),
    ];
    let mut panels = Vec::with_capacity(3);
    for (i, (name, values, second, limit, color, stepped)) in spec.into_iter().enumerate() {
        let y0 = i as f32 * (each + gap);
        let y1 = y0 + each;
        let (top, bottom) = (y0 * h as f32, y1 * h as f32);
        let peak = values
            .iter()
            .chain(second.unwrap_or(&[]))
            .copied()
            .filter(|v| v.is_finite())
            .fold(0.0f32, f32::max);
        // The limit joins the scale only once the count gets near it (half way): far away it would
        // only flatten the curve, near it is what the eye needs to see
        let near = limit > 0 && peak >= limit as f32 * 0.5;
        let hi = if near {
            peak.max(limit as f32) * 1.08
        } else {
            peak.max(if stepped { 1.0 } else { 1e-3 }) * 1.25 + if stepped { 1.0 } else { 0.0 }
        };
        draw_panel_frame(&mut canvas, p, top, bottom);
        if near {
            dashed_hline(
                &mut canvas,
                map_y(limit as f32, 0.0, hi, false, top, bottom),
                p.text,
            );
        }
        for &(idx, kind) in &req.markers {
            if n > 0 && idx < n {
                let x = x_of(idx, n, w) as usize;
                canvas.vline_alpha(
                    x,
                    top.round() as usize,
                    bottom.round() as usize,
                    marker_color(p, kind),
                    0.55,
                );
            }
        }
        if let Some(second) = second {
            draw_line(
                &mut canvas,
                second,
                top,
                bottom,
                hi,
                w,
                color,
                0.45,
                stepped,
            );
        }
        draw_line(&mut canvas, values, top, bottom, hi, w, color, 1.0, stepped);
        panels.push(CurvePanel {
            name,
            y0,
            y1,
            lo: 0.0,
            hi,
            log: false,
            last: values.last().copied(),
            points: values.len(),
        });
    }
    (frame, panels)
}

fn x_of(i: usize, n: usize, w: u32) -> f32 {
    if n <= 1 {
        w as f32 - 1.0
    } else {
        i as f32 * (w as f32 - 1.0) / (n as f32 - 1.0)
    }
}

#[allow(clippy::too_many_arguments)]
fn draw_line(
    c: &mut Canvas,
    values: &[f32],
    top: f32,
    bottom: f32,
    hi: f32,
    w: u32,
    color: Pixel,
    alpha: f32,
    stepped: bool,
) {
    let n = values.len();
    let mut pts: Vec<(f32, f32)> = Vec::with_capacity(n * 2);
    for (i, &v) in values.iter().enumerate() {
        if !v.is_finite() {
            continue;
        }
        let (x, y) = (x_of(i, n, w), map_y(v, 0.0, hi, false, top, bottom));
        // A count changes at a report, it does not slide: hold the old value up to the new one
        if let (true, Some(&(_, py))) = (stepped, pts.last()) {
            pts.push((x, py));
        }
        pts.push((x, y));
    }
    polyline(c, &pts, color, alpha);
    if let Some(&(x, y)) = pts.last() {
        for dy in -2i32..=2 {
            for dx in -2i32..=2 {
                if dx * dx + dy * dy <= 5 {
                    c.blend(
                        (x.round() as i32 + dx).max(0) as usize,
                        (y.round() as i32 + dy).max(0) as usize,
                        color,
                        alpha,
                    );
                }
            }
        }
    }
}

fn dashed_hline(c: &mut Canvas, y: f32, color: Pixel) {
    let y = y.round().max(0.0) as usize;
    for x in 0..c.width {
        if (x / 6) % 2 == 0 {
            c.blend(x, y, color, 0.7);
        }
    }
}

// ─── the graph ──────────────────────────────────────────────────────────

/// The positions of [`place`] (tests read them without the count).
#[cfg(test)]
pub fn layout(s: &TopologySnapshot) -> Vec<(f32, f32)> {
    place(s).0
}

/// Row (fraction of the image height) of the hidden neurons no input reaches. Kept clear of the
/// legend the view lays over the bottom edge of the image.
const DEAD_END_Y: f32 = 0.9;
/// Lowest row of the other neurons, with and without a row of dead ends under them.
const LOWEST_ROW_WITH_DEAD_ENDS: f32 = 0.78;
const LOWEST_ROW: f32 = 0.86;

/// Where each node of a snapshot sits, in fractions of the image (`0..=1`), in the order of
/// `TopologySnapshot::nodes`.
///
/// Inputs are the left column and outputs the right one; a hidden neuron sits in the column of
/// its distance (along active synapses) from the nearest input, so a feed-forward network reads
/// left to right. Hidden neurons no input reaches (dead ends) go on the bottom row: the layout
/// makes them visible instead of hiding them in the middle.
///
/// Also returns how many hidden neurons no input reaches (the ones on the bottom row). The count
/// comes from the placement itself, not from the coordinates: the rows of reachable neurons may
/// share them.
fn place(s: &TopologySnapshot) -> (Vec<(f32, f32)>, usize) {
    use std::collections::{BTreeMap, VecDeque};

    const MAX_DEPTH: usize = 6;
    let index: BTreeMap<u64, usize> = s.nodes.iter().enumerate().map(|(i, n)| (n.id, i)).collect();
    let mut out_adj: Vec<Vec<usize>> = vec![Vec::new(); s.nodes.len()];
    for e in s.edges.iter().filter(|e| e.active) {
        if let (Some(&a), Some(&b)) = (index.get(&e.src), index.get(&e.dst)) {
            out_adj[a].push(b);
        }
    }
    let mut depth: Vec<Option<usize>> = vec![None; s.nodes.len()];
    let mut queue = VecDeque::new();
    for (i, n) in s.nodes.iter().enumerate() {
        if n.kind == NeuronKind::Input {
            depth[i] = Some(0);
            queue.push_back(i);
        }
    }
    while let Some(i) = queue.pop_front() {
        let d = depth[i].unwrap_or(0);
        for &j in &out_adj[i] {
            if depth[j].is_none() && s.nodes[j].kind == NeuronKind::Hidden {
                depth[j] = Some(d + 1);
                queue.push_back(j);
            }
        }
    }
    let max_hidden = (0..s.nodes.len())
        .filter(|&i| s.nodes[i].kind == NeuronKind::Hidden)
        .filter_map(|i| depth[i])
        .max()
        .unwrap_or(0)
        .min(MAX_DEPTH);
    let columns = max_hidden + 1;
    let col_x = |d: usize| 0.08 + 0.84 * d as f32 / columns as f32;

    // (column key, node): column keys 0 = inputs, 1..=max = hidden by depth, max+1 = outputs,
    // usize::MAX = unreachable hidden
    let mut cols: BTreeMap<usize, Vec<usize>> = BTreeMap::new();
    for (i, n) in s.nodes.iter().enumerate() {
        let key = match n.kind {
            NeuronKind::Input => 0,
            NeuronKind::Output => columns,
            NeuronKind::Hidden => depth[i].map_or(usize::MAX, |d| d.min(MAX_DEPTH)),
        };
        cols.entry(key).or_default().push(i);
    }
    let mut pos = vec![(0.5, 0.5); s.nodes.len()];
    let bottom_band = cols.contains_key(&usize::MAX);
    let (y_lo, y_hi) = (
        0.07,
        if bottom_band {
            LOWEST_ROW_WITH_DEAD_ENDS
        } else {
            LOWEST_ROW
        },
    );
    for (key, members) in &cols {
        if *key == usize::MAX {
            let k = members.len();
            for (r, &i) in members.iter().enumerate() {
                pos[i] = (0.1 + 0.8 * (r as f32 + 0.5) / k as f32, DEAD_END_Y);
            }
            continue;
        }
        let k = members.len();
        for (r, &i) in members.iter().enumerate() {
            let y = if k == 1 {
                (y_lo + y_hi) / 2.0
            } else {
                y_lo + (y_hi - y_lo) * r as f32 / (k - 1) as f32
            };
            pos[i] = (col_x(*key), y);
        }
    }
    let unreachable = cols.get(&usize::MAX).map_or(0, Vec::len);
    (pos, unreachable)
}

#[derive(Debug, Clone)]
pub struct TopologyRequest {
    pub snapshot: Arc<TopologySnapshot>,
    pub width: u32,
    pub height: u32,
    pub palette: Palette,
}

/// What the view needs to label the graph.
#[derive(Debug, Clone, PartialEq)]
pub struct TopologyLayout {
    pub points: Vec<(f32, f32)>,
    /// Hidden neurons on the bottom row (no input reaches them).
    pub unreachable: usize,
}

/// Draws the graph: synapses as lines (green excitatory, red inhibitory, opacity by `|w|`;
/// a disabled one is grey and faint), neurons as discs (radius by firing rate; a ring marks a
/// neuron grown during the run).
pub fn render_topology(req: &TopologyRequest) -> (Frame, TopologyLayout) {
    let (w, h) = (req.width.max(1), req.height.max(1));
    let mut frame = Frame::new(w, h);
    let p = &req.palette;
    let mut canvas = frame.canvas();
    fill(&mut canvas, p.background);

    let s = &req.snapshot;
    let (points, unreachable) = place(s);
    let px = |i: usize| {
        (
            points[i].0 * (w as f32 - 1.0),
            points[i].1 * (h as f32 - 1.0),
        )
    };
    let index = |id: u64| s.nodes.binary_search_by_key(&id, |n| n.id).ok();
    let max_w = s
        .edges
        .iter()
        .map(|e| e.weight.abs())
        .fold(0.0f32, f32::max)
        .max(f32::EPSILON);
    let scale = (w.min(h) as f32 / 400.0).clamp(0.6, 2.0);

    for e in &s.edges {
        let (Some(a), Some(b)) = (index(e.src), index(e.dst)) else {
            continue;
        };
        let (color, alpha) = if !e.active {
            (p.text, 0.2)
        } else {
            (
                if e.weight >= 0.0 {
                    p.channel(1)
                } else {
                    p.playhead
                },
                0.15 + 0.75 * (e.weight.abs() / max_w),
            )
        };
        if a == b {
            let (x, y) = px(a);
            ring(
                &mut canvas,
                x,
                y - 9.0 * scale,
                6.0 * scale,
                1.2,
                color,
                alpha,
            );
        } else {
            line(&mut canvas, px(a), px(b), color, alpha);
        }
    }
    let rate_max = s
        .nodes
        .iter()
        .map(|n| n.rate)
        .fold(0.0f32, f32::max)
        .max(1e-3);
    for (i, n) in s.nodes.iter().enumerate() {
        let (x, y) = px(i);
        let color = match n.kind {
            NeuronKind::Input => p.channel(0),
            NeuronKind::Hidden => p.channel(2),
            NeuronKind::Output => p.channel(3),
        };
        let r = (4.0 + 5.0 * (n.rate / rate_max).clamp(0.0, 1.0)) * scale;
        disc(&mut canvas, x, y, r, color, 1.0);
        // The polarity is visible as a ring: inhibitory - red, modulating - orange.
        match n.polarity {
            NeuronPolarity::Excitatory => {}
            NeuronPolarity::Inhibitory => {
                ring(&mut canvas, x, y, r + 2.0 * scale, 1.5, p.channel(4), 0.9)
            }
            NeuronPolarity::Modulatory => {
                ring(&mut canvas, x, y, r + 2.0 * scale, 1.5, p.channel(6), 0.9)
            }
        }
        if n.kind == NeuronKind::Hidden && n.born_step > 0 {
            ring(&mut canvas, x, y, r + 3.0 * scale, 1.5, p.marker, 0.9);
        }
    }
    (
        frame,
        TopologyLayout {
            points,
            unreachable,
        },
    )
}

/// Anti-aliased filled circle.
fn disc(c: &mut Canvas, cx: f32, cy: f32, r: f32, color: Pixel, alpha: f32) {
    let (x0, x1) = (
        (cx - r - 1.0).floor().max(0.0) as usize,
        (cx + r + 1.0).ceil().max(0.0) as usize,
    );
    let (y0, y1) = (
        (cy - r - 1.0).floor().max(0.0) as usize,
        (cy + r + 1.0).ceil().max(0.0) as usize,
    );
    for y in y0..=y1 {
        for x in x0..=x1 {
            let d = ((x as f32 + 0.5 - cx).powi(2) + (y as f32 + 0.5 - cy).powi(2)).sqrt();
            c.blend(x, y, color, ((r + 0.5 - d).clamp(0.0, 1.0)) * alpha);
        }
    }
}

/// Anti-aliased ring of radius `r` and thickness `t`.
fn ring(c: &mut Canvas, cx: f32, cy: f32, r: f32, t: f32, color: Pixel, alpha: f32) {
    let m = r + t + 1.0;
    let (x0, x1) = (
        (cx - m).floor().max(0.0) as usize,
        (cx + m).ceil().max(0.0) as usize,
    );
    let (y0, y1) = (
        (cy - m).floor().max(0.0) as usize,
        (cy + m).ceil().max(0.0) as usize,
    );
    for y in y0..=y1 {
        for x in x0..=x1 {
            let d = ((x as f32 + 0.5 - cx).powi(2) + (y as f32 + 0.5 - cy).powi(2)).sqrt();
            let cover = (t / 2.0 + 0.5 - (d - r).abs()).clamp(0.0, 1.0);
            c.blend(x, y, color, cover * alpha);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use neo_snn::structural::NeuronModel;
    use neo_snn::telemetry::{EdgeView, NodeView};

    fn node(id: u64, kind: NeuronKind, rate: f32, born: u64) -> NodeView {
        NodeView {
            id,
            kind,
            polarity: NeuronPolarity::Excitatory,
            model: NeuronModel::default(),
            rate,
            born_step: born,
        }
    }

    fn edge(id: u64, src: u64, dst: u64, weight: f32, active: bool) -> EdgeView {
        EdgeView {
            id,
            src,
            dst,
            weight,
            active,
        }
    }

    fn snap() -> TopologySnapshot {
        // inputs 0,1 → hidden 2 → hidden 3 → outputs 4,5; hidden 6 is a dead end
        TopologySnapshot {
            version: 1,
            nodes: vec![
                node(0, NeuronKind::Input, 0.0, 0),
                node(1, NeuronKind::Input, 0.0, 0),
                node(2, NeuronKind::Hidden, 0.1, 0),
                node(3, NeuronKind::Hidden, 0.2, 50),
                node(4, NeuronKind::Output, 0.1, 0),
                node(5, NeuronKind::Output, 0.0, 0),
                node(6, NeuronKind::Hidden, 0.0, 0),
            ],
            edges: vec![
                edge(0, 0, 2, 0.1, true),
                edge(1, 1, 2, -0.1, true),
                edge(2, 2, 3, 0.1, true),
                edge(3, 3, 4, 0.1, true),
                edge(4, 3, 5, 0.1, false),
                edge(5, 3, 3, 0.05, true),
                edge(6, 6, 4, 0.1, true),
            ],
            total_neurons: 7,
            total_synapses: 7,
            truncated: false,
        }
    }

    fn differs(f: &Frame, bg: Pixel) -> usize {
        f.pixels().iter().filter(|p| **p != bg).count()
    }

    #[test]
    fn layout_reads_left_to_right_and_puts_dead_ends_at_the_bottom() {
        let s = snap();
        let pos = layout(&s);
        assert_eq!(pos.len(), s.nodes.len());
        assert!(
            pos.iter()
                .all(|&(x, y)| (0.0..=1.0).contains(&x) && (0.0..=1.0).contains(&y))
        );
        let x = |id: usize| pos[id].0;
        assert!(x(0) == x(1) && x(0) < x(2) && x(2) < x(3) && x(3) < x(4) && x(4) == x(5));
        assert!(
            (pos[6].1 - DEAD_END_Y).abs() < 1e-6,
            "a hidden neuron no input reaches is on the bottom row"
        );
        assert!(pos[..6].iter().all(|p| p.1 < DEAD_END_Y - 0.05));
        assert_eq!(layout(&s), pos, "deterministic");
    }

    #[test]
    fn a_full_column_is_not_mistaken_for_dead_ends() {
        // inputs 0,1 → hidden 2,3 (both one step from an input) → output 4: every neuron is reached
        let s = TopologySnapshot {
            version: 1,
            nodes: vec![
                node(0, NeuronKind::Input, 0.0, 0),
                node(1, NeuronKind::Input, 0.0, 0),
                node(2, NeuronKind::Hidden, 0.1, 0),
                node(3, NeuronKind::Hidden, 0.1, 0),
                node(4, NeuronKind::Output, 0.1, 0),
            ],
            edges: vec![
                edge(0, 0, 2, 0.1, true),
                edge(1, 1, 3, 0.1, true),
                edge(2, 2, 4, 0.1, true),
                edge(3, 3, 4, 0.1, true),
            ],
            total_neurons: 5,
            total_synapses: 4,
            truncated: false,
        };
        assert_eq!(place(&s).1, 0);
        let (_, lay) = render_topology(&TopologyRequest {
            snapshot: Arc::new(s),
            width: 400,
            height: 300,
            palette: Palette::DARK,
        });
        assert_eq!(
            lay.unreachable, 0,
            "the lowest hidden neuron of a column is not a dead end"
        );
    }

    #[test]
    fn layout_of_an_empty_or_inputless_graph_does_not_panic() {
        assert!(layout(&TopologySnapshot::default()).is_empty());
        let s = TopologySnapshot {
            nodes: vec![node(0, NeuronKind::Hidden, 0.0, 0)],
            ..Default::default()
        };
        assert_eq!(layout(&s).len(), 1);
    }

    #[test]
    fn the_graph_is_drawn_and_both_palettes_work() {
        for palette in [Palette::DARK, Palette::of(false)] {
            let req = TopologyRequest {
                snapshot: Arc::new(snap()),
                width: 400,
                height: 300,
                palette,
            };
            let (frame, lay) = render_topology(&req);
            assert_eq!((frame.width, frame.height), (400, 300));
            assert!(differs(&frame, palette.background) > 500);
            assert_eq!(lay.points.len(), 7);
            assert_eq!(lay.unreachable, 1);
        }
    }

    #[test]
    fn a_grown_neuron_has_a_ring_and_a_disabled_synapse_is_faint() {
        let p = Palette::DARK;
        let (with, _) = render_topology(&TopologyRequest {
            snapshot: Arc::new(snap()),
            width: 400,
            height: 300,
            palette: p,
        });
        let mut s = snap();
        s.nodes[3].born_step = 0;
        let (without, _) = render_topology(&TopologyRequest {
            snapshot: Arc::new(s),
            width: 400,
            height: 300,
            palette: p,
        });
        assert!(
            differs(&with, p.background) > differs(&without, p.background),
            "the ring adds pixels"
        );
    }

    #[test]
    fn degenerate_sizes_and_empty_graphs_do_not_panic() {
        for (w, h) in [(0, 0), (1, 1), (3, 2)] {
            render_topology(&TopologyRequest {
                snapshot: Arc::new(snap()),
                width: w,
                height: h,
                palette: Palette::DARK,
            });
            render_topology(&TopologyRequest {
                snapshot: Arc::new(TopologySnapshot::default()),
                width: w,
                height: h,
                palette: Palette::DARK,
            });
            render_counts(&CountsRequest {
                neurons: vec![],
                synapses: vec![],
                active_synapses: vec![],
                mean_rate: vec![],
                max_neurons: 0,
                max_synapses: 0,
                markers: vec![],
                width: w,
                height: h,
                palette: Palette::DARK,
            });
        }
    }

    fn counts(markers: Vec<(usize, MarkerKind)>) -> CountsRequest {
        CountsRequest {
            neurons: vec![4.0, 4.0, 5.0, 5.0, 3.0],
            synapses: vec![5.0, 6.0, 8.0, 9.0, 9.0],
            active_synapses: vec![5.0, 5.0, 8.0, 8.0, 9.0],
            mean_rate: vec![0.0, 0.05, 0.1, 0.1, 0.12],
            max_neurons: 16,
            max_synapses: 40,
            markers,
            width: 300,
            height: 300,
            palette: Palette::DARK,
        }
    }

    #[test]
    fn counts_panels_tile_the_image_and_report_the_last_values() {
        let (frame, panels) = render_counts(&counts(vec![]));
        assert_eq!(
            panels.iter().map(|p| p.name).collect::<Vec<_>>(),
            ["Neurons", "Synapses", "Mean spike rate"]
        );
        assert!(
            panels
                .windows(2)
                .all(|w| w[0].y1 <= w[1].y0 && w[0].y0 < w[0].y1)
        );
        assert_eq!(panels[0].last, Some(3.0));
        assert!(
            panels[0].hi >= 5.0 && panels[0].hi < 16.0,
            "a far limit does not flatten the curve: {}",
            panels[0].hi
        );
        // Near the limit it is drawn and inside the scale
        let mut near = counts(vec![]);
        near.neurons = vec![4.0, 9.0, 15.0];
        let (_, panels) = render_counts(&near);
        assert!(panels[0].hi >= 16.0);
        assert!(differs(&frame, Palette::DARK.background) > 300);
    }

    #[test]
    fn a_marker_is_drawn_at_the_report_it_belongs_to() {
        let p = Palette::DARK;
        let (plain, _) = render_counts(&counts(vec![]));
        let (marked, _) = render_counts(&counts(vec![(2, MarkerKind::Rollback)]));
        let x = 2 * 299 / 4; // the third of five points
        let changed: Vec<usize> = (0..300usize)
            .filter(|&i| plain.pixels()[i * 300 + x] != marked.pixels()[i * 300 + x])
            .collect();
        assert!(
            changed.len() > 100,
            "a vertical tick through the panels: {}",
            changed.len()
        );
        assert!(changed.len() < 300);
        // Nothing far from that column changed
        let far =
            (0..300usize).any(|i| plain.pixels()[i * 300 + 10] != marked.pixels()[i * 300 + 10]);
        assert!(!far);
        // Kinds have distinct colours
        let colors: Vec<_> = MarkerKind::ALL
            .iter()
            .map(|k| marker_color(&p, *k))
            .collect();
        for i in 0..colors.len() {
            for j in (i + 1)..colors.len() {
                assert_ne!(colors[i], colors[j]);
            }
        }
        // A marker beyond the series is ignored
        render_counts(&counts(vec![(99, MarkerKind::Growth)]));
    }
}
