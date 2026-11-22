//! The brain view: which parts of the network are in use. Every neuron is a dot in a translucent
//! rotating brain, tinted by the region it belongs to (sensory input, the groups of the hidden
//! layer, the motor outputs), lit by how much it has fired lately and flashing white when it fires.
//! Threads between dots are the synapses that fired in order in the latest frame, coloured from
//! white (early) to orange (late). Beside the brain a usage grid shows every neuron as a cell, so the
//! silent ones are easy to find, and under both runs a raster with one lane per region over the
//! recent frames.
//!
//! The hidden layer is split into groups by what its neurons vote for (the output they drive most
//! strongly), so the "parts" are meaningful for this network and not just geometry. The renderer draws
//! no text: names, numbers and the headline are elements the view lays over the image, placed from the
//! [`BrainInfo`] and [`BrainLayout`] that come back with it. Big layers are sampled (the map) or pooled
//! (the usage grid and raster), so the cost of a frame stays bounded.

use std::f32::consts::TAU;
use std::sync::Arc;

use neo_snn::telemetry::SpikeFrame;

use crate::engine::canvas::{Frame, Pixel};

use super::dashboard::{GAP, PAD, TITLE_H};
use super::links::{Links, Regions};
use super::network::{Hdr, bloom, hash01};

pub type Rgb = [u8; 3];

const SENSORY: Rgb = [70, 140, 255];
const CENTRAL: [Rgb; 4] = [
    [255, 196, 60],
    [80, 230, 160],
    [190, 120, 255],
    [255, 140, 60],
];
const MIXED: Rgb = [200, 205, 215];
const MOTOR: Rgb = [255, 85, 85];

/// Panels as `[x0, y0, x1, y1]`, fractions of the image.
pub const MAP: [f32; 4] = [0.0, 0.0, 0.60, 0.62];
pub const USAGE: [f32; 4] = [0.60, 0.0, 1.0, 0.62];
pub const RASTER: [f32; 4] = [0.0, 0.62, 1.0, 0.90];
pub const STATS: [f32; 4] = [0.0, 0.90, 1.0, 1.0];
/// Width of the lane names on the left of the raster, logical px.
pub const LANE_GUTTER: f32 = 132.0;
/// Frames the raster looks back over.
pub const RASTER_FRAMES: usize = 16;
/// Neurons of one layer drawn on the map; more are sampled.
const MAX_DRAWN: usize = 20_000;
/// Synapses of the trace kept (the strongest ones).
const MAX_TRACE: usize = 160;

// ─── camera ─────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct BrainCamera {
    pub yaw: f32,
    pub pitch: f32,
    pub zoom: f32,
    /// How far the brain is moved from the middle of the map, as fractions of the image's width and height.
    pub pan: [f32; 2],
}

impl Default for BrainCamera {
    fn default() -> Self {
        Self {
            yaw: 0.0,
            pitch: 0.12,
            zoom: 1.0,
            pan: [0.0, 0.0],
        }
    }
}

impl BrainCamera {
    pub const MIN_ZOOM: f32 = 0.5;
    pub const MAX_ZOOM: f32 = 6.0;

    /// Drag by `dfx`, `dfy` (fractions of the map): the brain turns with the pointer.
    pub fn rotate(&mut self, dfx: f32, dfy: f32) {
        self.yaw += dfx * 3.2;
        self.pitch = (self.pitch + dfy * 2.0).clamp(-0.7, 0.7);
    }

    /// Zooms by `factor` so that the point of the brain under the pointer (`fx`, `fy`: fractions of the image)
    /// stays under it. `middle` is where the brain sits at zoom 1 with no pan ([`middle_of_map`]): the
    /// picture grows from it, so the pan has to make up for the rest.
    pub fn zoom_at(&mut self, factor: f32, fx: f32, fy: f32, middle: (f32, f32)) {
        let z2 = (self.zoom * factor).clamp(Self::MIN_ZOOM, Self::MAX_ZOOM);
        let k = z2 / self.zoom;
        self.pan[0] = fx - middle.0 - (fx - middle.0 - self.pan[0]) * k;
        self.pan[1] = fy - middle.1 - (fy - middle.1 - self.pan[1]) * k;
        self.zoom = z2;
        self.clamp_pan();
    }

    /// Moves the brain with the pointer: `dfx`, `dfy` are fractions of the image.
    pub fn slide(&mut self, dfx: f32, dfy: f32) {
        self.pan[0] += dfx;
        self.pan[1] += dfy;
        self.clamp_pan();
    }

    fn clamp_pan(&mut self) {
        for p in &mut self.pan {
            *p = p.clamp(-3.0, 3.0);
        }
    }

    /// Bits of the numbers, for keys that must tell two cameras apart.
    pub fn bits(&self) -> [u32; 5] {
        [
            self.yaw.to_bits(),
            self.pitch.to_bits(),
            self.zoom.to_bits(),
            self.pan[0].to_bits(),
            self.pan[1].to_bits(),
        ]
    }
}

/// Where the middle of the brain is on the map at zoom 1, as fractions of an image of `width` by `height`
/// physical pixels at `scale` device pixels per logical one.
pub fn middle_of_map(width: u32, height: u32, scale: f32) -> (f32, f32) {
    let (wf, hf, s) = (width.max(1) as f32, height.max(1) as f32, scale.max(0.5));
    let inner = Area::of(MAP, wf, hf).inset(PAD * s, TITLE_H * s, PAD * s, PAD * s);
    (
        (inner.x0 + inner.w() / 2.0) / wf,
        (inner.y0 + inner.h() / 2.0) / hf,
    )
}

// ─── request and result ─────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct BrainLayer {
    pub units: usize,
    /// The frame being shown.
    pub frame: Option<Arc<SpikeFrame>>,
    /// Recent frames, oldest first: usage and the raster come from these.
    pub history: Vec<Arc<SpikeFrame>>,
    /// Regions the model gave this layer (cell types, parts of the brain). Without them the hidden layer is
    /// grouped by what its neurons vote for.
    pub regions: Option<Regions>,
}

#[derive(Debug, Clone)]
pub struct BrainRequest {
    /// Input, hidden, output.
    pub layers: Vec<BrainLayer>,
    /// `weights[l]` joins layer `l` to `l + 1` (`rows` = units of `l`).
    pub weights: Vec<Option<Links>>,
    pub camera: BrainCamera,
    pub selected: Option<(usize, usize)>,
    /// Advances as the run does: it makes the brain sway slowly.
    pub phase: f32,
    pub width: u32,
    pub height: u32,
    pub scale: f32,
}

/// A region: a set of neurons of one layer that share a colour and a lane.
#[derive(Debug, Clone, PartialEq)]
pub struct Group {
    pub name: String,
    pub layer: usize,
    pub units: Vec<usize>,
    pub color: Rgb,
}

#[derive(Debug, Clone, PartialEq)]
pub struct GroupInfo {
    pub name: String,
    pub layer: usize,
    pub color: Rgb,
    pub units: usize,
    /// Neurons of the group that fired on the last step of the frame.
    pub firing_now: usize,
    /// Neurons that have not fired at all in the recent frames.
    pub silent: usize,
    /// Mean share of steps its neurons fired on.
    pub usage_mean: f32,
    /// Where its usage block starts (fractions): the view puts the name above it.
    pub label_at: (f32, f32),
}

#[derive(Debug, Clone, PartialEq)]
pub struct Lane {
    pub name: String,
    pub color: Rgb,
    pub y0: f32,
    pub y1: f32,
    /// Rows of the lane's heat map (slices of the group's neurons).
    pub bands: usize,
}

/// One synapse that fired: `from` neuron fired and so did `to` on the same step through a positive weight.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Synapse {
    pub step: usize,
    pub from: (usize, usize),
    pub to: (usize, usize),
    pub weight: f32,
}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct BrainInfo {
    pub groups: Vec<GroupInfo>,
    pub lanes: Vec<Lane>,
    /// Horizontal extent of the raster's plot (fractions).
    pub raster_x: (f32, f32),
    pub raster_frames: usize,
    pub steps_per_frame: usize,
    /// The synapses of the trace, in the order they fired.
    pub trace: Vec<Synapse>,
    /// Output neurons that fired on the latest step that had any output spike, and that step.
    pub output_fired: Vec<usize>,
    pub output_step: Option<usize>,
    /// Share of all neurons that fired on the last step.
    pub firing_now: f32,
    /// Neurons that fired at least once in the recent frames, and all neurons.
    pub neurons_fired: usize,
    pub neurons: usize,
    /// True when a layer is too big to draw neuron by neuron: the map then shows a dim sample as
    /// background and lights up only the neurons that fired.
    pub lod: bool,
    /// Neurons of the big layers that fired on the last step.
    pub spikers: usize,
}

#[derive(Debug, Clone, PartialEq)]
pub struct UsageBlock {
    pub group: usize,
    pub layer: usize,
    /// The grid of cells (fractions of the image).
    pub x0: f32,
    pub y0: f32,
    pub cell_w: f32,
    pub cell_h: f32,
    pub cols: usize,
    pub rows: usize,
    /// Neurons pooled in one cell.
    pub per_cell: usize,
    pub units: Vec<usize>,
}

#[derive(Debug, Clone, PartialEq, Default)]
pub struct BrainLayout {
    /// Size of the image in logical pixels.
    pub size: (f32, f32),
    /// Drawn neurons on the map: `(layer, unit, x, y)`, fractions of the image.
    pub nodes: Vec<(usize, usize, f32, f32)>,
    pub blocks: Vec<UsageBlock>,
}

impl BrainLayout {
    /// The map neuron nearest to the point if one is within `radius` logical pixels. Of two overlapping
    /// ones the later drawn (nearer the viewer) wins.
    pub fn hit_node(&self, fx: f32, fy: f32, radius: f32) -> Option<(usize, usize)> {
        let (w, h) = self.size;
        let mut best: Option<((usize, usize), f32)> = None;
        for &(l, u, x, y) in &self.nodes {
            let d = (((x - fx) * w).powi(2) + ((y - fy) * h).powi(2)).sqrt();
            if d <= radius && best.is_none_or(|(_, b)| d <= b) {
                best = Some(((l, u), d));
            }
        }
        best.map(|(n, _)| n)
    }

    /// The group whose usage grid holds the point, and the neuron under it (when cells are not pooled).
    pub fn hit_usage(&self, fx: f32, fy: f32) -> Option<(usize, Option<(usize, usize)>)> {
        for b in &self.blocks {
            let (gx, gy) = (
                (fx - b.x0) / b.cell_w.max(1e-9),
                (fy - b.y0) / b.cell_h.max(1e-9),
            );
            if gx < 0.0 || gy < 0.0 || gx >= b.cols as f32 || gy >= b.rows as f32 {
                continue;
            }
            let idx = (gy as usize * b.cols + gx as usize) * b.per_cell;
            return Some((
                b.group,
                b.units
                    .get(idx)
                    .filter(|_| b.per_cell == 1)
                    .copied()
                    .map(|u| (b.layer, u)),
            ));
        }
        None
    }
}

// ─── groups, usage and the trace ────────────────────────────────────────

/// Colours of regions the model named itself: blue, amber, green, red first (sensory, central, projection, motor).
const REGION_COLORS: [Rgb; 8] = [
    SENSORY,
    [255, 196, 60],
    [80, 230, 160],
    MOTOR,
    [190, 120, 255],
    [255, 140, 60],
    [60, 220, 220],
    [255, 120, 190],
];
/// Regions drawn as separate groups of one layer; the rest are left out of the legend.
const MAX_REGIONS: usize = 16;

fn region_groups(layer: usize, units: usize, regions: &Regions) -> Vec<Group> {
    regions
        .members(units)
        .into_iter()
        .enumerate()
        .filter(|(_, m)| !m.is_empty())
        .take(MAX_REGIONS)
        .map(|(r, m)| Group {
            name: regions.names[r].to_uppercase(),
            layer,
            units: m,
            color: REGION_COLORS[r % REGION_COLORS.len()],
        })
        .collect()
}

/// The regions of the network. A layer the model gave regions to is split by them. Otherwise: input;
/// the hidden layer split by the output each neuron drives most (and a group of the undecided); each
/// output neuron.
pub fn build_groups(req: &BrainRequest) -> Vec<Group> {
    let mut groups = Vec::new();
    for (l, layer) in req.layers.iter().enumerate() {
        let units = layer.units;
        if units == 0 {
            continue;
        }
        if let Some(r) = &layer.regions {
            let g = region_groups(l, units, r);
            if !g.is_empty() {
                groups.extend(g);
                continue;
            }
        }
        match l {
            0 => groups.push(Group {
                name: "INPUT".into(),
                layer: 0,
                units: (0..units).collect(),
                color: SENSORY,
            }),
            1 => {
                let votes = req
                    .weights
                    .get(1)
                    .and_then(|m| m.as_ref())
                    .filter(|m| m.rows() == units && (2..=4).contains(&m.cols()))
                    .and_then(|m| m.small_dense(4, 8_000_000).map(|d| (d, m.cols())));
                match votes {
                    Some((values, cols)) => {
                        let wmax = values.iter().fold(0.0f32, |a, v| a.max(v.abs())).max(1e-6);
                        let mut by_class: Vec<Vec<usize>> = vec![Vec::new(); cols];
                        let mut mixed = Vec::new();
                        for j in 0..units {
                            let row = &values[j * cols..(j + 1) * cols];
                            let (best, top) =
                                row.iter()
                                    .enumerate()
                                    .fold(
                                        (0, f32::MIN),
                                        |(bi, bv), (i, &v)| if v > bv { (i, v) } else { (bi, bv) },
                                    );
                            let second = row
                                .iter()
                                .enumerate()
                                .filter(|&(i, _)| i != best)
                                .map(|(_, &v)| v)
                                .fold(f32::MIN, f32::max);
                            if top - second < 0.12 * wmax {
                                mixed.push(j);
                            } else {
                                by_class[best].push(j);
                            }
                        }
                        for (k, us) in by_class.into_iter().enumerate() {
                            if !us.is_empty() {
                                groups.push(Group {
                                    name: format!("HIDDEN → class {k}"),
                                    layer: 1,
                                    units: us,
                                    color: CENTRAL[k % CENTRAL.len()],
                                });
                            }
                        }
                        if !mixed.is_empty() {
                            groups.push(Group {
                                name: "HIDDEN · mixed".into(),
                                layer: 1,
                                units: mixed,
                                color: MIXED,
                            });
                        }
                    }
                    None => groups.push(Group {
                        name: "HIDDEN".into(),
                        layer: 1,
                        units: (0..units).collect(),
                        color: CENTRAL[0],
                    }),
                }
            }
            _ => {
                if units <= 8 {
                    for k in 0..units {
                        groups.push(Group {
                            name: format!("OUTPUT class {k}"),
                            layer: l,
                            units: vec![k],
                            color: MOTOR,
                        });
                    }
                } else {
                    groups.push(Group {
                        name: "OUTPUT".into(),
                        layer: l,
                        units: (0..units).collect(),
                        color: MOTOR,
                    });
                }
            }
        }
    }
    groups
}

/// Share of steps each neuron fired on over the recent frames (the shown frame when there is no history).
fn usage_of(layer: &BrainLayer) -> Vec<f32> {
    let frames: Vec<&Arc<SpikeFrame>> = if layer.history.is_empty() {
        layer.frame.iter().collect()
    } else {
        layer.history.iter().collect()
    };
    let mut counts = vec![0u64; layer.units];
    let mut steps = 0u64;
    for f in frames
        .iter()
        .filter(|f| !f.is_empty() && f.units == layer.units)
    {
        steps += f.steps as u64;
        for (c, &n) in counts.iter_mut().zip(f.unit_counts()) {
            *c += u64::from(n);
        }
    }
    counts
        .into_iter()
        .map(|c| {
            if steps == 0 {
                0.0
            } else {
                c as f32 / steps as f32
            }
        })
        .collect()
}

/// Which steps each neuron fired on, one bit per step (frames of up to 64 steps).
fn spike_masks(f: &SpikeFrame) -> Option<Vec<u64>> {
    if f.steps > 64 {
        return None;
    }
    let mut masks = vec![0u64; f.units];
    for t in 0..f.steps {
        for (u, m) in masks.iter_mut().enumerate() {
            if f.spiked(t, u) {
                *m |= 1 << t;
            }
        }
    }
    Some(masks)
}

fn keep_strongest(all: &mut Vec<Synapse>) {
    all.sort_by(|x, y| y.weight.total_cmp(&x.weight));
    all.truncate(MAX_TRACE);
}

/// The synapses that fired in the shown frames: on a step where a neuron fired and one it excites
/// fired too. The strongest [`MAX_TRACE`], in firing order. Every stored link is looked at once
/// (an AND of two bit masks), so a layer of hundreds of thousands of neurons costs milliseconds.
pub fn trace_synapses(req: &BrainRequest) -> Vec<Synapse> {
    let mut all: Vec<Synapse> = Vec::new();
    for l in 0..req.layers.len().saturating_sub(1) {
        let (Some(a), Some(b)) = (
            req.layers[l].frame.as_deref(),
            req.layers[l + 1].frame.as_deref(),
        ) else {
            continue;
        };
        let Some(links) = req.weights.get(l).and_then(|m| m.as_ref()) else {
            continue;
        };
        if a.is_empty()
            || b.is_empty()
            || a.steps != b.steps
            || links.rows() != a.units
            || links.cols() != b.units
        {
            continue;
        }
        let (Some(ma), Some(mb)) = (spike_masks(a), spike_masks(b)) else {
            continue;
        };
        links.for_each(|i, j, w| {
            if !(w.is_finite() && w > 0.0) {
                return;
            }
            let mut bits = ma[i] & mb[j];
            while bits != 0 {
                let t = bits.trailing_zeros() as usize;
                bits &= bits - 1;
                all.push(Synapse {
                    step: t,
                    from: (l, i),
                    to: (l + 1, j),
                    weight: w,
                });
            }
            // Keep the pile small however many links fire together
            if all.len() > 8 * MAX_TRACE {
                keep_strongest(&mut all);
            }
        });
    }
    keep_strongest(&mut all);
    all.sort_by_key(|s| s.step);
    all
}

// ─── geometry ───────────────────────────────────────────────────────────

/// Centre and radii of the region of group `gi` of `ng` in `layer`, in brain units.
fn region(layer: usize, gi: usize, ng: usize, custom: bool) -> ([f32; 3], [f32; 3]) {
    if custom && ng > 1 {
        // Regions the model named sit around the middle of the brain
        let a = gi as f32 / ng as f32 * TAU;
        return (
            [0.52 * a.cos(), -0.05 + 0.38 * a.sin(), 0.0],
            [0.26, 0.22, 0.30],
        );
    }
    match layer {
        0 => ([-0.62, 0.05, 0.0], [0.22, 0.38, 0.38]),
        1 if ng <= 1 => ([0.02, -0.08, 0.0], [0.55, 0.44, 0.46]),
        1 => {
            let a = gi as f32 / ng as f32 * TAU + 0.6;
            (
                [0.02 + 0.28 * a.cos(), -0.08 + 0.22 * a.sin(), 0.0],
                [0.33, 0.28, 0.38],
            )
        }
        _ => {
            let off = (gi as f32 - (ng as f32 - 1.0) / 2.0) * 0.2;
            (
                [0.58, 0.34 + off, 0.0],
                [0.12, if ng > 1 { 0.09 } else { 0.14 }, 0.16],
            )
        }
    }
}

fn neuron_point(layer: usize, unit: usize, (c, r): ([f32; 3], [f32; 3])) -> [f32; 3] {
    let (a, b, d) = (
        hash01(unit, layer + 1, 31),
        hash01(unit, layer + 1, 32),
        hash01(unit, layer + 1, 33),
    );
    let z = 2.0 * a - 1.0;
    let (phi, rad, s) = (TAU * b, d.cbrt() * 0.92, (1.0 - z * z).sqrt());
    [
        c[0] + r[0] * rad * s * phi.cos(),
        c[1] + r[1] * rad * s * phi.sin(),
        c[2] + r[2] * rad * z,
    ]
}

/// The translucent outline: a dome, a cerebellum and a stem, as dots.
fn shell_points() -> Vec<[f32; 3]> {
    let mut pts = Vec::new();
    let golden = std::f32::consts::PI * (3.0 - 5.0f32.sqrt());
    let sphere = |n: usize, c: [f32; 3], r: [f32; 3], pts: &mut Vec<[f32; 3]>| {
        for i in 0..n {
            let z = 1.0 - 2.0 * (i as f32 + 0.5) / n as f32;
            let (s, phi) = ((1.0 - z * z).sqrt(), golden * i as f32);
            pts.push([
                c[0] + r[0] * s * phi.cos(),
                c[1] + r[1] * s * phi.sin(),
                c[2] + r[2] * z,
            ]);
        }
    };
    sphere(900, [0.0, -0.05, 0.0], [0.95, 0.62, 0.55], &mut pts);
    sphere(260, [-0.45, 0.52, 0.0], [0.32, 0.20, 0.30], &mut pts);
    for i in 0..120 {
        let t = i as f32 / 119.0;
        let phi = hash01(i, 5, 71) * TAU;
        pts.push([
            0.25 + 0.05 * t + 0.07 * phi.cos(),
            0.45 + 0.5 * t,
            0.07 * phi.sin(),
        ]);
    }
    pts
}

/// Rotates and projects a point of the brain: `(x, y, nearness)`, nearness about 1 at the middle,
/// above 1 for what is toward the viewer.
fn project(
    p: [f32; 3],
    cam: &BrainCamera,
    yaw: f32,
    centre: (f32, f32),
    unit: f32,
) -> (f32, f32, f32) {
    let (sy, cy) = yaw.sin_cos();
    let (x1, z1) = (p[0] * cy + p[2] * sy, -p[0] * sy + p[2] * cy);
    let (sp, cp) = cam.pitch.sin_cos();
    let (y2, z2) = (p[1] * cp - z1 * sp, p[1] * sp + z1 * cp);
    let near = 1.0 / (1.0 + 0.35 * z2);
    (
        centre.0 + x1 * unit * near,
        centre.1 + y2 * unit * near,
        near,
    )
}

// ─── drawing helpers ────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy)]
struct Area {
    x0: f32,
    y0: f32,
    x1: f32,
    y1: f32,
}

impl Area {
    fn of(f: [f32; 4], w: f32, h: f32) -> Area {
        Area {
            x0: f[0] * w,
            y0: f[1] * h,
            x1: f[2] * w,
            y1: f[3] * h,
        }
    }

    fn w(&self) -> f32 {
        (self.x1 - self.x0).max(0.0)
    }

    fn h(&self) -> f32 {
        (self.y1 - self.y0).max(0.0)
    }

    fn inset(&self, l: f32, t: f32, r: f32, b: f32) -> Area {
        Area {
            x0: self.x0 + l,
            y0: self.y0 + t,
            x1: self.x1 - r,
            y1: self.y1 - b,
        }
    }
}

fn rgbf(c: Rgb) -> [f32; 3] {
    [
        c[0] as f32 / 255.0,
        c[1] as f32 / 255.0,
        c[2] as f32 / 255.0,
    ]
}

fn mixc(a: Rgb, b: Rgb, t: f32) -> Pixel {
    let m = |x: u8, y: u8| (x as f32 + (y as f32 - x as f32) * t.clamp(0.0, 1.0)).round() as u8;
    Pixel::rgb(m(a[0], b[0]), m(a[1], b[1]), m(a[2], b[2]))
}

const BLACK: Rgb = [4, 6, 10];
const PANEL: Rgb = [8, 11, 17];
const LANE_BG: Rgb = [11, 14, 21];
const BORDER: Rgb = [30, 38, 52];

fn outline(c: &mut crate::engine::canvas::Canvas, a: &Area, color: Pixel) {
    c.rect(a.x0, a.y0, a.x1, a.y0 + 1.0, color, 1.0);
    c.rect(a.x0, a.y1 - 1.0, a.x1, a.y1, color, 1.0);
    c.rect(a.x0, a.y0, a.x0 + 1.0, a.y1, color, 1.0);
    c.rect(a.x1 - 1.0, a.y0, a.x1, a.y1, color, 1.0);
}

/// Heat for `t` in `0..1`: dark, then the region's colour, then toward white for the hottest.
fn heat(base: Rgb, t: f32) -> Pixel {
    let t = t.clamp(0.0, 1.0);
    if t < 0.75 {
        mixc(LANE_BG, base, 0.12 + 0.88 * t / 0.75)
    } else {
        mixc(base, [255, 255, 255], 0.5 * (t - 0.75) / 0.25)
    }
}

// ─── the picture ────────────────────────────────────────────────────────

pub fn render_brain(req: &BrainRequest) -> (Frame, BrainInfo, BrainLayout) {
    let (w, h) = (req.width.max(1), req.height.max(1));
    let (wf, hf) = (w as f32, h as f32);
    let s = req.scale.max(0.5);
    let groups = build_groups(req);
    let usage: Vec<Vec<f32>> = req.layers.iter().map(usage_of).collect();
    let trace = trace_synapses(req);

    // ── what the text needs
    let mut info = BrainInfo {
        trace: trace.clone(),
        ..BrainInfo::default()
    };
    let last_step =
        |f: &SpikeFrame, u: usize| !f.is_empty() && u < f.units && f.spiked(f.steps - 1, u);
    let frame_of = |l: usize| {
        req.layers
            .get(l)
            .and_then(|l| l.frame.as_deref())
            .filter(|f| !f.is_empty())
    };
    for g in &groups {
        let f = frame_of(g.layer);
        let u = &usage[g.layer];
        info.groups.push(GroupInfo {
            name: g.name.clone(),
            layer: g.layer,
            color: g.color,
            units: g.units.len(),
            firing_now: f.map_or(0, |f| g.units.iter().filter(|&&n| last_step(f, n)).count()),
            silent: g
                .units
                .iter()
                .filter(|&&n| u.get(n).copied().unwrap_or(0.0) == 0.0)
                .count(),
            usage_mean: g
                .units
                .iter()
                .map(|&n| u.get(n).copied().unwrap_or(0.0))
                .sum::<f32>()
                / g.units.len().max(1) as f32,
            label_at: (0.0, 0.0),
        });
    }
    info.neurons = req.layers.iter().map(|l| l.units).sum();
    let firing: usize = (0..req.layers.len())
        .map(|l| frame_of(l).map_or(0, |f| (0..f.units).filter(|&u| last_step(f, u)).count()))
        .sum();
    info.firing_now = firing as f32 / info.neurons.max(1) as f32;
    info.neurons_fired = usage
        .iter()
        .map(|u| u.iter().filter(|&&v| v > 0.0).count())
        .sum();
    if let Some(out) = frame_of(2)
        && let Some(t) = (0..out.steps)
            .rev()
            .find(|&t| (0..out.units).any(|u| out.spiked(t, u)))
    {
        info.output_step = Some(t);
        info.output_fired = (0..out.units).filter(|&u| out.spiked(t, u)).collect();
    }

    // ── the map, in linear light
    let map = Area::of(MAP, wf, hf);
    let inner = map.inset(PAD * s, TITLE_H * s, PAD * s, PAD * s);
    let mut hdr = Hdr::new(w as usize, h as usize);
    let mut layout = BrainLayout {
        size: (wf / s, hf / s),
        ..BrainLayout::default()
    };
    let unit = (inner.w() / 2.3).min(inner.h() / 1.9).max(1.0) * req.camera.zoom;
    let centre = (
        inner.x0 + inner.w() / 2.0 + req.camera.pan[0] * wf,
        inner.y0 + inner.h() / 2.0 + 0.05 * unit + req.camera.pan[1] * hf,
    );
    let yaw = req.camera.yaw + 0.45 * (req.phase * 0.12).sin();
    let clip = |x: f32, y: f32| x >= map.x0 && x <= map.x1 && y >= map.y0 && y <= map.y1;
    let sel = req
        .selected
        .filter(|&(l, u)| req.layers.get(l).is_some_and(|l| u < l.units));

    for p in shell_points() {
        let (x, y, near) = project(p, &req.camera, yaw, centre, unit);
        if clip(x, y) {
            hdr.glow(x, y, 1.1 * s, [0.35, 0.60, 1.0], 0.34 * near * near);
        }
    }

    // Where a neuron is on screen, worked out when asked: nothing is stored per neuron, so a layer of
    // a million costs only the neurons actually drawn
    const NONE: u16 = u16::MAX;
    let layer_groups: Vec<Vec<&Group>> = (0..req.layers.len())
        .map(|l| groups.iter().filter(|g| g.layer == l).collect())
        .collect();
    let regions_of: Vec<Vec<([f32; 3], [f32; 3])>> = layer_groups
        .iter()
        .enumerate()
        .map(|(l, gs)| {
            (0..gs.len())
                .map(|gi| region(l, gi, gs.len(), req.layers[l].regions.is_some()))
                .collect()
        })
        .collect();
    let group_of: Vec<Vec<u16>> = req
        .layers
        .iter()
        .enumerate()
        .map(|(l, layer)| {
            let mut of = vec![NONE; layer.units];
            for (gi, g) in layer_groups[l].iter().enumerate() {
                for &u in &g.units {
                    if let Some(slot) = of.get_mut(u) {
                        *slot = gi as u16;
                    }
                }
            }
            of
        })
        .collect();
    let place = |l: usize, u: usize| -> Option<(f32, f32, f32)> {
        let gi = *group_of.get(l)?.get(u)?;
        if gi == NONE {
            return None;
        }
        Some(project(
            neuron_point(l, u, regions_of[l][gi as usize]),
            &req.camera,
            yaw,
            centre,
            unit,
        ))
    };
    // Neurons drawn one by one per layer: more when zoomed in, since less of the brain is in view
    let budget =
        ((MAX_DRAWN as f32) * req.camera.zoom.powi(2)).clamp(MAX_DRAWN as f32, 150_000.0) as usize;

    // The trace: threads from white (early) to orange (late), with a bead at the end that fired last
    let steps = frame_of(0).or(frame_of(1)).map_or(1, |f| f.steps).max(2);
    let wmax = trace.iter().map(|t| t.weight).fold(1e-6f32, f32::max);
    for t in &trace {
        let (Some(a), Some(b)) = (place(t.from.0, t.from.1), place(t.to.0, t.to.1)) else {
            continue;
        };
        let f = t.step as f32 / (steps - 1) as f32;
        let rgb = [1.0, 1.0 - 0.45 * f, 1.0 - 0.75 * f];
        let dim = if sel.is_some() { 0.25 } else { 1.0 };
        draw_arc(
            &mut hdr,
            (a.0, a.1),
            (b.0, b.1),
            rgb,
            (0.05 + 0.09 * t.weight / wmax) * dim,
            &clip,
        );
        if clip(b.0, b.1) {
            hdr.glow(b.0, b.1, 2.0 * s, rgb, 0.22 * dim);
        }
    }
    // The picked neuron's own synapses, all of them
    if let Some((sl, su)) = sel
        && let Some(me) = place(sl, su)
    {
        for (pair, from_me) in [(sl.checked_sub(1), false), (Some(sl), true)] {
            let Some(p) = pair else { continue };
            let Some(m) = req.weights.get(p).and_then(|m| m.as_ref()) else {
                continue;
            };
            let others = if from_me {
                m.strongest_from(su, 400)
            } else {
                m.strongest_into(su, 400)
            };
            let other_layer = if from_me { p + 1 } else { p };
            for (o, v) in others {
                if let Some(pt) = place(other_layer, o) {
                    draw_arc(
                        &mut hdr,
                        (me.0, me.1),
                        (pt.0, pt.1),
                        if v >= 0.0 {
                            [0.6, 1.0, 0.8]
                        } else {
                            [1.0, 0.55, 0.45]
                        },
                        0.35,
                        &clip,
                    );
                }
            }
        }
    }

    // Layers within the budget are drawn neuron by neuron; many neurons add up to a white blob, so each gets
    // dimmer and smaller as their count grows. Bigger layers are level of detail, below.
    let planned: usize = req
        .layers
        .iter()
        .filter(|l| l.units <= budget)
        .map(|l| l.units)
        .sum();
    let thin = (800.0 / planned.max(800) as f32).clamp(0.005, 1.0);
    let map_area = (map.w() * map.h()).max(1.0);

    for (l, layer) in req.layers.iter().enumerate() {
        let frame = layer
            .frame
            .as_deref()
            .filter(|f| !f.is_empty() && f.units == layer.units);
        let lod = layer.units > budget;
        info.lod |= lod;
        let stride = layer.units.div_ceil(budget).max(1);
        let dim_of = |u: usize| {
            if sel.is_some_and(|me| me != (l, u) && me.0.abs_diff(l) > 1) {
                0.35
            } else {
                1.0
            }
        };

        if !lod {
            for g in &layer_groups[l] {
                let rgb = rgbf(g.color);
                for &u in &g.units {
                    let Some((x, y, near)) = place(l, u) else {
                        continue;
                    };
                    if !clip(x, y) {
                        continue;
                    }
                    layout.nodes.push((l, u, x / wf, y / hf));
                    let used = (usage[l][u] / 0.5).min(1.0);
                    let dim = dim_of(u);
                    let r = ((2.4 + 2.6 * used)
                        * s
                        * near.max(0.6)
                        * req.camera.zoom.sqrt()
                        * thin.sqrt().max(0.45))
                    .clamp(0.9, 9.0 * s);
                    if usage[l][u] == 0.0 {
                        hdr.glow(x, y, r * 0.8, [0.55, 0.30, 0.30], 0.30 * near * dim * thin);
                    } else {
                        hdr.glow(x, y, r, rgb, (0.9 + 2.2 * used) * near * dim * thin);
                    }
                    if frame.is_some_and(|f| last_step(f, u)) {
                        hdr.glow(x, y, r * 3.0, rgb, 0.8 * dim * thin.sqrt());
                        hdr.glow(x, y, r * 0.7, [1.0, 1.0, 1.0], 1.6 * dim * thin.sqrt());
                    }
                    if sel == Some((l, u)) {
                        hdr.glow(x, y, r * 5.0, [1.0, 1.0, 1.0], 1.2);
                    }
                }
            }
            continue;
        }

        // 1. An evenly spaced sample as a dim translucent background. Only what is in view is kept, and it is
        //    thinned until the dots hardly overlap: zooming in shows separate neurons, not a white blob
        let mut visible: Vec<(usize, f32, f32, f32, usize)> = Vec::new();
        for (gi, g) in layer_groups[l].iter().enumerate() {
            for &u in g.units.iter().step_by(stride) {
                let Some((x, y, near)) = place(l, u) else {
                    continue;
                };
                if clip(x, y) {
                    visible.push((u, x, y, near, gi));
                }
            }
        }
        let r_dot = (3.6 * s * req.camera.zoom.sqrt()).clamp(1.0, 9.0 * s);
        let keep = ((visible.len() as f32 * std::f32::consts::PI * r_dot * r_dot)
            / (map_area * 0.5))
            .ceil()
            .max(1.0) as usize;
        for (k, &(u, x, y, near, gi)) in visible.iter().enumerate() {
            let rgb = rgbf(layer_groups[l][gi].color);
            // A faint grain from every sampled neuron keeps the cloud readable from afar
            hdr.splat(x, y, rgb, 0.05 * near);
            if k % keep != 0 {
                continue;
            }
            layout.nodes.push((l, u, x / wf, y / hf));
            let used = (usage[l][u] / 0.5).min(1.0);
            let dim = dim_of(u);
            let r = ((2.4 + 2.6 * used) * s * near.max(0.6) * req.camera.zoom.sqrt())
                .clamp(0.9, 9.0 * s);
            if usage[l][u] == 0.0 {
                hdr.glow(x, y, r * 0.8, [0.55, 0.30, 0.30], 0.30 * near * dim);
            } else {
                hdr.glow(x, y, r, rgb, (0.35 + 0.8 * used) * near * dim);
            }
        }
        if let Some((sx, sy, _)) = sel
            .filter(|&(sl, _)| sl == l)
            .and_then(|(_, su)| place(l, su))
            && clip(sx, sy)
        {
            hdr.glow(sx, sy, 5.0 * s, [1.0, 1.0, 1.0], 1.2);
        }

        // 2. Every neuron that fired lit up: a glow of density from all of them, and dots over it from as many as
        //    fit in the view without covering each other
        let Some(f) = frame else { continue };
        let spikers: Vec<usize> = (0..f.units).filter(|&u| last_step(f, u)).collect();
        info.spikers += spikers.len();
        let lit: Vec<(f32, f32, f32, usize, usize)> = spikers
            .iter()
            .filter_map(|&u| {
                let (x, y, near) = place(l, u)?;
                clip(x, y).then(|| (x, y, near, u, group_of[l][u] as usize))
            })
            .collect();
        let e_density = (0.35 * map_area / lit.len().max(1) as f32).min(0.4);
        let r = (2.0 * s * req.camera.zoom.sqrt()).clamp(1.0, 6.0 * s);
        let big_cap = ((map_area * 0.4 / (std::f32::consts::PI * (2.5 * r) * (2.5 * r))) as usize)
            .clamp(50, 16_500);
        let big_every = (lit.len() / big_cap).max(1);
        for (k, &(x, y, near, u, gi)) in lit.iter().enumerate() {
            let rgb = rgbf(layer_groups[l][gi].color);
            hdr.splat(x, y, rgb, e_density * near);
            if k % big_every == 0 {
                hdr.glow(x, y, r * 2.5 * near.max(0.6), rgb, 0.7 * near);
                hdr.glow(x, y, r * near.max(0.6), [1.0, 1.0, 1.0], 1.1 * near);
                layout.nodes.push((l, u, x / wf, y / hf));
            }
        }
    }

    // ── onto black, with bloom
    let tight = bloom(&hdr, 0.30, 6.0 * s);
    let wide = bloom(&hdr, 0.7, 24.0 * s);
    let mut frame = Frame::new(w, h);
    {
        let pixels = frame.pixels_mut();
        for (i, px) in pixels.iter_mut().enumerate() {
            let mut out = [0u8; 3];
            for k in 0..3 {
                let e = hdr.px[i * 3 + k] + 0.9 * tight[i * 3 + k] + 0.7 * wide[i * 3 + k];
                let tm = 1.0 - (-1.6 * e).exp();
                let bg = BLACK[k] as f32 / 255.0;
                out[k] = ((1.0 - (1.0 - bg) * (1.0 - tm)) * 255.0).round() as u8;
            }
            *px = Pixel::rgb(out[0], out[1], out[2]);
        }
    }

    // ── the other panels, drawn over whatever the map's light spilled into them
    let mut c = frame.canvas();
    let fill = |c: &mut crate::engine::canvas::Canvas, a: &Area| {
        c.rect(
            a.x0,
            a.y0,
            a.x1,
            a.y1,
            Pixel::rgb(PANEL[0], PANEL[1], PANEL[2]),
            1.0,
        )
    };
    let g2 = GAP * s / 2.0;
    let boxed = |f: [f32; 4]| Area::of(f, wf, hf).inset(g2, g2, g2, g2);
    let border = Pixel::rgb(BORDER[0], BORDER[1], BORDER[2]);
    for f in [USAGE, RASTER, STATS] {
        let a = boxed(f);
        fill(&mut c, &a);
        outline(&mut c, &a, border);
    }
    outline(&mut c, &boxed(MAP), border);

    // Usage grid: every neuron a cell, the silent ones outlined in red
    let ua = boxed(USAGE).inset(PAD * s, TITLE_H * s, PAD * s, PAD * s);
    if ua.w() > 8.0 && ua.h() > 8.0 && !groups.is_empty() {
        let label_px = 18.0 * s;
        let gap = 6.0 * s;
        let weights: Vec<f32> = groups
            .iter()
            .map(|g| (g.units.len() as f32).sqrt().max(2.5))
            .collect();
        let total: f32 = weights.iter().sum();
        let usable = (ua.h() - (label_px + gap) * groups.len() as f32).max(groups.len() as f32);
        let mut y = ua.y0;
        for (gi, g) in groups.iter().enumerate() {
            let bh = (usable * weights[gi] / total).max(4.0);
            info.groups[gi].label_at = (ua.x0 / wf, y / hf);
            y += label_px;
            let n = g.units.len();
            let cell = ((ua.w() * bh / n.max(1) as f32).sqrt()).clamp(3.0 * s, 26.0 * s);
            let cols = ((ua.w() / cell).floor() as usize).max(1);
            let rows_fit = ((bh / cell).floor() as usize).max(1);
            let per_cell = n.div_ceil(cols * rows_fit).max(1);
            let cells = n.div_ceil(per_cell);
            let rows = cells.div_ceil(cols);
            let (cw, ch) = (
                ua.w() / cols as f32,
                (bh / rows as f32).min(ua.w() / cols as f32),
            );
            let color = g.color;
            let fr = frame_of(g.layer);
            let group_share = info.groups[gi].firing_now as f32 / n.max(1) as f32;
            for ci in 0..cells {
                let (cx, cy) = ((ci % cols) as f32, (ci / cols) as f32);
                let range = &g.units[ci * per_cell..((ci + 1) * per_cell).min(n)];
                let u = range.iter().map(|&n| usage[g.layer][n]).sum::<f32>() / range.len() as f32;
                let (x0, y0) = (ua.x0 + cx * cw, y + cy * ch);
                let pad = if cw >= 5.0 && ch >= 5.0 { 1.0 } else { 0.0 };
                let silent = range.iter().all(|&n| usage[g.layer][n] == 0.0);
                if silent {
                    c.rect(
                        x0 + pad,
                        y0 + pad,
                        x0 + cw - pad,
                        y0 + ch - pad,
                        Pixel::rgb(LANE_BG[0], LANE_BG[1], LANE_BG[2]),
                        1.0,
                    );
                    outline(
                        &mut c,
                        &Area {
                            x0: x0 + pad,
                            y0: y0 + pad,
                            x1: x0 + cw - pad,
                            y1: y0 + ch - pad,
                        },
                        Pixel::rgb(170, 60, 60),
                    );
                } else {
                    c.rect(
                        x0 + pad,
                        y0 + pad,
                        x0 + cw - pad,
                        y0 + ch - pad,
                        mixc(LANE_BG, color, 0.18 + 0.82 * (u / 0.5).min(1.0).powf(0.8)),
                        1.0,
                    );
                }
                // A pooled cell holds many neurons, so "one of them fired" would light every cell: it lights when
                // a clearly bigger share of a sample of its neurons fired than in the group as a whole
                let firing = fr.is_some_and(|f| {
                    if per_cell == 1 {
                        return last_step(f, range[0]);
                    }
                    let sample: Vec<usize> = range
                        .iter()
                        .copied()
                        .step_by(range.len().div_ceil(32).max(1))
                        .collect();
                    let share = sample.iter().filter(|&&n| last_step(f, n)).count() as f32
                        / sample.len().max(1) as f32;
                    share > 1.5 * group_share + 0.02
                });
                if firing {
                    let (mx, my) = (x0 + cw / 2.0, y0 + ch / 2.0);
                    let r = (cw.min(ch) * 0.22).max(1.0);
                    c.rect(
                        mx - r,
                        my - r,
                        mx + r,
                        my + r,
                        Pixel::rgb(255, 255, 255),
                        1.0,
                    );
                }
                if sel.is_some_and(|(sl, su)| sl == g.layer && range.contains(&su)) {
                    outline(
                        &mut c,
                        &Area {
                            x0,
                            y0,
                            x1: x0 + cw,
                            y1: y0 + ch,
                        },
                        Pixel::rgb(255, 255, 255),
                    );
                }
            }
            layout.blocks.push(UsageBlock {
                group: gi,
                layer: g.layer,
                x0: ua.x0 / wf,
                y0: y / hf,
                cell_w: cw / wf,
                cell_h: ch / hf,
                cols,
                rows,
                per_cell,
                units: if per_cell == 1 {
                    g.units.clone()
                } else {
                    Vec::new()
                },
            });
            y += bh + gap;
        }
    }

    // Raster: a lane per group over the recent frames, and the whole network in blue below
    let ra = boxed(RASTER).inset(PAD * s + LANE_GUTTER * s, 8.0 * s, PAD * s, 24.0 * s);
    let history = |l: usize| {
        req.layers
            .get(l)
            .map(|l| {
                if l.history.is_empty() {
                    l.frame.iter().cloned().collect::<Vec<_>>()
                } else {
                    l.history.clone()
                }
            })
            .unwrap_or_default()
    };
    let n_frames = (0..req.layers.len())
        .map(|l| history(l).len())
        .filter(|&n| n > 0)
        .min()
        .unwrap_or(0)
        .min(RASTER_FRAMES);
    let steps_pf = frame_of(1).or(frame_of(0)).map_or(0, |f| f.steps);
    info.raster_frames = n_frames;
    info.steps_per_frame = steps_pf;
    info.raster_x = (ra.x0 / wf, ra.x1 / wf);
    if ra.w() > 8.0 && ra.h() > 8.0 && n_frames > 0 && steps_pf > 0 {
        let total_t = n_frames * steps_pf;
        let hist: Vec<Vec<Arc<SpikeFrame>>> = (0..req.layers.len())
            .map(|l| {
                let hs = history(l);
                hs[hs.len().saturating_sub(n_frames)..].to_vec()
            })
            .collect();
        // One firing share per time step for a sample of at most SAMPLE neurons (evenly spaced, so a big
        // group costs no more than a small one)
        const SAMPLE: usize = 256;
        let series_of = |members: &[(usize, usize)]| -> Vec<f32> {
            let stride = members.len().div_ceil(SAMPLE).max(1);
            let sampled: Vec<(usize, usize)> = members.iter().copied().step_by(stride).collect();
            let mut series = vec![0.0f32; total_t];
            for fi in 0..n_frames {
                for t in 0..steps_pf {
                    let hits = sampled
                        .iter()
                        .filter(|&&(l, u)| {
                            hist[l].get(fi).is_some_and(|f| {
                                f.steps == steps_pf && u < f.units && f.spiked(t, u)
                            })
                        })
                        .count();
                    series[fi * steps_pf + t] = hits as f32 / sampled.len().max(1) as f32;
                }
            }
            series
        };
        // A lane is a heat map: rows are consecutive slices of the group's neurons (one row per neuron while
        // there is room), columns are time, brightness is the share of the slice that fired. However many
        // neurons a group has, it costs at most 24 rows of 256 sampled neurons.
        let lane_gap = 3.0 * s;
        let n_lanes = groups.len() + 1;
        let lane_h = ((ra.h() - lane_gap * (n_lanes as f32 - 1.0)) / n_lanes as f32).max(2.0);
        let max_bands = ((lane_h / (3.0 * s)).floor() as usize).clamp(1, 24);
        let mut lanes: Vec<(String, Rgb, Vec<Vec<f32>>)> = groups
            .iter()
            .map(|g| {
                let n = g.units.len();
                let bands = n.clamp(1, max_bands);
                let rows = (0..bands)
                    .map(|bi| {
                        series_of(
                            &g.units[bi * n / bands..(bi + 1) * n / bands]
                                .iter()
                                .map(|&u| (g.layer, u))
                                .collect::<Vec<_>>(),
                        )
                    })
                    .collect();
                (g.name.clone(), g.color, rows)
            })
            .collect();
        let whole = (0..req.layers.len())
            .take(max_bands)
            .map(|l| series_of(&(0..req.layers[l].units).map(|u| (l, u)).collect::<Vec<_>>()))
            .collect();
        lanes.push(("WHOLE NETWORK".into(), SENSORY, whole));
        let plot_w = ra.w() as usize;
        for (li, (name, color, rows)) in lanes.iter().enumerate() {
            let y0 = ra.y0 + li as f32 * (lane_h + lane_gap);
            c.rect(
                ra.x0,
                y0,
                ra.x1,
                y0 + lane_h,
                Pixel::rgb(LANE_BG[0], LANE_BG[1], LANE_BG[2]),
                1.0,
            );
            // The lane's own range, so that a layer whose share hardly moves still shows its structure
            let (lo, hi) = rows
                .iter()
                .flatten()
                .fold((f32::MAX, f32::MIN), |(lo, hi), &v| (lo.min(v), hi.max(v)));
            let span = hi - lo;
            let band_h = lane_h / rows.len().max(1) as f32;
            for (bi, series) in rows.iter().enumerate() {
                let by = y0 + bi as f32 * band_h;
                for px in 0..plot_w {
                    let (a, z) = (
                        px * total_t / plot_w,
                        ((px + 1) * total_t / plot_w)
                            .max(px * total_t / plot_w + 1)
                            .min(total_t),
                    );
                    let run = &series[a..z];
                    let v = if total_t <= plot_w {
                        run.iter().copied().fold(0.0, f32::max)
                    } else {
                        run.iter().sum::<f32>() / run.len().max(1) as f32
                    };
                    let t = if span > 1e-6 {
                        (v - lo) / span
                    } else if hi > 0.0 {
                        0.5
                    } else {
                        0.0
                    };
                    if t > 0.0 {
                        c.rect(
                            ra.x0 + px as f32,
                            by,
                            ra.x0 + px as f32 + 1.0,
                            by + band_h,
                            heat(*color, t.powf(0.7)),
                            1.0,
                        );
                    }
                }
            }
            info.lanes.push(Lane {
                name: name.clone(),
                color: *color,
                y0: y0 / hf,
                y1: (y0 + lane_h) / hf,
                bands: rows.len(),
            });
        }
        // Frame boundaries
        for fi in 1..n_frames {
            let x = ra.x0 + ra.w() * fi as f32 / n_frames as f32;
            c.rect(x, ra.y0, x + 1.0, ra.y1, Pixel::rgb(255, 255, 255), 0.06);
        }
    }
    (frame, info, layout)
}

/// A thread between two projected neurons: an arc, bowed to one side.
fn draw_arc(
    h: &mut Hdr,
    a: (f32, f32),
    b: (f32, f32),
    rgb: [f32; 3],
    per_px: f32,
    clip: &dyn Fn(f32, f32) -> bool,
) {
    let (dx, dy) = (b.0 - a.0, b.1 - a.1);
    let len = (dx * dx + dy * dy).sqrt();
    if len < 1.0 {
        return;
    }
    let n = ((len / 1.4).ceil() as usize).clamp(6, 900);
    let (nx, ny) = (-dy / len, dx / len);
    let bow = 0.16 * len;
    for i in 0..=n {
        let t = i as f32 / n as f32;
        let (x, y) = (a.0 + dx * t, a.1 + dy * t);
        let lift = 4.0 * t * (1.0 - t) * bow;
        let (px, py) = (x + nx * lift, y + ny * lift);
        if clip(px, py) {
            h.splat(px, py, rgb, per_px * len / n as f32);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::super::dashboard::WeightMatrix;
    use super::super::links::SparseLinks;
    use super::*;

    fn frame(steps: usize, units: usize, f: impl Fn(usize, usize) -> bool) -> Arc<SpikeFrame> {
        Arc::new(SpikeFrame::from_fn(steps, units, f))
    }

    fn matrix(rows: usize, cols: usize, f: impl Fn(usize, usize) -> f32) -> Links {
        Links::Dense(WeightMatrix {
            rows,
            cols,
            values: Arc::new((0..rows * cols).map(|i| f(i / cols, i % cols)).collect()),
        })
    }

    /// The same links as a sparse list: only the cells that are not zero.
    fn sparse_of(rows: usize, cols: usize, f: impl Fn(usize, usize) -> f32) -> Links {
        let (mut from, mut to, mut weight) = (Vec::new(), Vec::new(), Vec::new());
        for i in 0..rows {
            for j in 0..cols {
                if f(i, j) != 0.0 {
                    from.push(i as u32);
                    to.push(j as u32);
                    weight.push(f(i, j));
                }
            }
        }
        Links::Sparse(SparseLinks {
            rows,
            cols,
            from: Arc::new(from),
            to: Arc::new(to),
            weight: Arc::new(weight),
        })
    }

    fn layer(units: usize, f: impl Fn(usize, usize, usize) -> bool + Copy) -> BrainLayer {
        let hist: Vec<_> = (0..6)
            .map(|k| frame(24, units, move |t, u| f(k, t, u)))
            .collect();
        BrainLayer {
            units,
            frame: hist.last().cloned(),
            history: hist,
            regions: None,
        }
    }

    fn request(w: u32, h: u32) -> BrainRequest {
        BrainRequest {
            layers: vec![
                layer(8, |k, t, u| (t + u + k) % 3 == 0),
                layer(24, |k, t, u| (t * 5 + u * 3 + k) % 7 < 2),
                layer(2, |k, t, u| (t + u + k) % 4 == 0),
            ],
            weights: vec![
                Some(matrix(8, 24, |r, c| {
                    ((r * 31 + c * 17) % 23) as f32 / 11.0 - 1.0
                })),
                Some(matrix(24, 2, |r, c| {
                    if (r % 2 == 0) == (c == 0) { 1.0 } else { -1.0 }
                })),
            ],
            camera: BrainCamera::default(),
            selected: None,
            phase: 0.0,
            width: w,
            height: h,
            scale: 1.0,
        }
    }

    #[test]
    fn hidden_neurons_are_grouped_by_the_output_they_drive() {
        let groups = build_groups(&request(100, 100));
        let names: Vec<&str> = groups.iter().map(|g| g.name.as_str()).collect();
        assert_eq!(
            names,
            [
                "INPUT",
                "HIDDEN → class 0",
                "HIDDEN → class 1",
                "OUTPUT class 0",
                "OUTPUT class 1"
            ]
        );
        let (c0, c1) = (&groups[1], &groups[2]);
        assert!(c0.units.iter().all(|u| u % 2 == 0) && c1.units.iter().all(|u| u % 2 == 1));
        assert_eq!(
            c0.units.len() + c1.units.len(),
            24,
            "every hidden neuron is in some group"
        );
    }

    #[test]
    fn undecided_neurons_form_their_own_group_and_missing_weights_a_single_one() {
        let mut r = request(100, 100);
        r.weights[1] = Some(matrix(24, 2, |_, _| 0.3)); // all equal: nobody decides
        let names: Vec<String> = build_groups(&r).into_iter().map(|g| g.name).collect();
        assert!(
            names.contains(&"HIDDEN · mixed".to_string())
                && !names.iter().any(|n| n.starts_with("HIDDEN →"))
        );
        r.weights[1] = None;
        assert!(
            build_groups(&r)
                .iter()
                .any(|g| g.name == "HIDDEN" && g.units.len() == 24)
        );
    }

    #[test]
    fn usage_is_the_share_of_steps_a_neuron_fired_on() {
        let l = BrainLayer {
            units: 3,
            regions: None,
            frame: None,
            history: vec![
                frame(10, 3, |t, u| u == 0 || (u == 1 && t < 5)),
                frame(10, 3, |t, u| u == 0 || (u == 1 && t < 5)),
            ],
        };
        let u = usage_of(&l);
        assert_eq!(u, vec![1.0, 0.5, 0.0]);
        assert_eq!(
            usage_of(&BrainLayer {
                units: 2,
                frame: None,
                history: Vec::new(),
                regions: None
            }),
            vec![0.0, 0.0]
        );
    }

    #[test]
    fn the_trace_holds_only_excitatory_synapses_that_fired_together_in_order() {
        let mut r = request(100, 100);
        // input 1 and hidden 2 fire together at step 3 only; input 5 and hidden 2 at step 7 (negative weight)
        r.layers[0].frame = Some(frame(24, 8, |t, u| {
            (t == 3 && u == 1) || (t == 7 && u == 5)
        }));
        r.layers[1].frame = Some(frame(24, 24, |t, u| u == 2 && (t == 3 || t == 7)));
        r.layers[2].frame = None;
        r.weights[0] = Some(matrix(8, 24, |i, j| {
            if (i, j) == (1, 2) {
                0.8
            } else if (i, j) == (5, 2) {
                -0.9
            } else {
                0.0
            }
        }));
        let t = trace_synapses(&r);
        assert_eq!(
            t,
            vec![Synapse {
                step: 3,
                from: (0, 1),
                to: (1, 2),
                weight: 0.8
            }]
        );
    }

    #[test]
    fn the_trace_is_capped_and_ordered_by_step() {
        let mut r = request(100, 100);
        r.layers[0].frame = Some(frame(24, 8, |_, _| true));
        r.layers[1].frame = Some(frame(24, 24, |_, _| true));
        r.weights[0] = Some(matrix(8, 24, |_, _| 1.0));
        let t = trace_synapses(&r);
        assert_eq!(t.len(), MAX_TRACE);
        assert!(t.windows(2).all(|p| p[0].step <= p[1].step));
    }

    #[test]
    fn info_reports_what_fired_now_and_what_is_silent() {
        let mut r = request(900, 600);
        r.layers[2].frame = Some(frame(24, 2, |t, u| t == 20 && u == 1));
        r.layers[2].history = vec![r.layers[2].frame.clone().unwrap()];
        let (_, info, _) = render_brain(&r);
        assert_eq!(info.output_step, Some(20));
        assert_eq!(info.output_fired, vec![1]);
        assert_eq!(info.neurons, 34);
        assert!(info.firing_now >= 0.0 && info.firing_now <= 1.0);
        let out0 = info
            .groups
            .iter()
            .find(|g| g.name == "OUTPUT class 0")
            .unwrap();
        assert_eq!(
            (out0.units, out0.silent),
            (1, 1),
            "class 0 never fired in the history"
        );
        assert!(info.neurons_fired < info.neurons);
        assert_eq!(info.lanes.len(), info.groups.len() + 1);
        assert_eq!(info.lanes.last().unwrap().name, "WHOLE NETWORK");
        assert!(info.raster_frames > 0 && info.steps_per_frame == 24);
    }

    #[test]
    fn the_picture_is_the_right_size_and_lights_up_with_data() {
        let (f, _, layout) = render_brain(&request(1000, 600));
        assert_eq!((f.width, f.height), (1000, 600));
        assert!(!layout.nodes.is_empty() && layout.blocks.len() == 5);
        let empty = BrainRequest {
            layers: Vec::new(),
            weights: Vec::new(),
            ..request(1000, 600)
        };
        let (e, info, l) = render_brain(&empty);
        let lit = |f: &Frame| {
            f.pixels()
                .iter()
                .filter(|p| p.r as u32 + p.g as u32 + p.b as u32 > 150)
                .count()
        };
        assert!(lit(&f) > lit(&e) + 500);
        assert!(info.groups.is_empty() && l.nodes.is_empty());
    }

    #[test]
    fn regions_are_tinted_in_their_own_colours() {
        let (f, _, _) = render_brain(&request(1000, 600));
        let count = |test: &dyn Fn(&Pixel) -> bool| f.pixels().iter().filter(|p| test(p)).count();
        assert!(
            count(&|p| p.b as i32 > p.r as i32 + 60 && p.b as i32 > p.g as i32 + 30) > 30,
            "sensory blue"
        );
        assert!(
            count(&|p| p.r > 180 && p.g > 130 && p.b < 100) > 30,
            "central amber"
        );
        assert!(
            count(&|p| p.g as i32 > p.r as i32 + 50 && p.g as i32 > p.b as i32 + 20) > 30,
            "central green"
        );
        assert!(
            count(&|p| p.r > 150
                && (p.g as i32) < p.r as i32 - 70
                && (p.b as i32) < p.r as i32 - 70)
                > 30,
            "motor red"
        );
    }

    #[test]
    fn hits_find_map_neurons_and_usage_cells() {
        let (_, _, layout) = render_brain(&request(1000, 600));
        let &(l, u, x, y) = layout.nodes.first().unwrap();
        assert_eq!(layout.hit_node(x, y, 4.0).map(|n| n.0), Some(l), "{u}");
        assert_eq!(layout.hit_node(0.01, 0.99, 4.0), None);
        let b = &layout.blocks[1];
        let (gx, gy) = (b.x0 + b.cell_w * 0.5, b.y0 + b.cell_h * 0.5);
        let (g, unit) = layout.hit_usage(gx, gy).unwrap();
        assert_eq!(g, 1);
        assert!(unit.is_some(), "cells are not pooled in a small network");
        assert_eq!(
            layout.hit_usage(0.2, 0.2),
            None,
            "the map is not a usage cell"
        );
    }

    #[test]
    fn selecting_a_neuron_changes_the_picture_and_a_bad_selection_does_not() {
        let plain = render_brain(&request(800, 500)).0;
        let mut r = request(800, 500);
        r.selected = Some((1, 4));
        assert_ne!(plain, render_brain(&r).0);
        r.selected = Some((7, 7));
        assert!(plain == render_brain(&r).0);
    }

    #[test]
    fn dragging_turns_the_brain_and_zoom_is_bounded() {
        let mut cam = BrainCamera::default();
        cam.rotate(0.1, 100.0);
        assert!(cam.yaw > 0.3 && cam.pitch <= 0.7);
        let middle = middle_of_map(600, 400, 1.0);
        cam.zoom_at(1e6, 0.3, 0.3, middle);
        assert_eq!(cam.zoom, BrainCamera::MAX_ZOOM);
        cam.zoom_at(1e-9, 0.3, 0.3, middle);
        assert_eq!(cam.zoom, BrainCamera::MIN_ZOOM);
        let a = render_brain(&request(600, 400)).0;
        let mut r = request(600, 400);
        r.camera.yaw = 1.0;
        assert_ne!(a, render_brain(&r).0);
    }

    #[test]
    fn odd_sizes_and_missing_frames_do_not_panic() {
        for (w, h, s) in [(1, 1, 1.0), (40, 30, 1.0), (300, 200, 2.0)] {
            let mut r = request(w, h);
            r.scale = s;
            r.layers[1].frame = None;
            r.layers[0].history.clear();
            r.weights = vec![None, Some(matrix(3, 3, |_, _| 1.0))];
            let (f, _, _) = render_brain(&r);
            assert_eq!((f.width, f.height), (w, h));
        }
    }

    #[test]
    fn a_huge_layer_is_sampled_and_pooled() {
        let mut r = request(900, 600);
        let big = 60_000;
        r.layers[1] = BrainLayer {
            units: big,
            frame: Some(frame(24, big, |t, u| (t + u) % 9 == 0)),
            history: vec![frame(24, big, |t, u| (t + u) % 9 == 0)],
            regions: None,
        };
        r.weights[0] = None;
        r.weights[1] = None;
        let start = std::time::Instant::now();
        let (_, info, layout) = render_brain(&r);
        assert!(layout.nodes.len() <= 3 * MAX_DRAWN + 100);
        let block = layout
            .blocks
            .iter()
            .find(|b| b.units.is_empty())
            .expect("the big group is pooled");
        assert!(block.per_cell > 1);
        assert_eq!(info.neurons, 8 + big + 2);
        assert!(start.elapsed().as_secs() < 20, "{:?}", start.elapsed());
    }

    #[test]
    fn a_sparse_list_gives_the_same_trace_and_groups_as_the_dense_matrix() {
        let wi = |i: usize, j: usize| {
            if (i * 5 + j).is_multiple_of(4) {
                ((i + j) % 7) as f32 / 3.0 - 0.8
            } else {
                0.0
            }
        };
        let wo = |i: usize, c: usize| {
            if i.is_multiple_of(2) == (c == 0) {
                1.0
            } else {
                -1.0
            }
        };
        let mut dense_req = request(100, 100);
        dense_req.weights = vec![Some(matrix(8, 24, wi)), Some(matrix(24, 2, wo))];
        let mut sparse_req = request(100, 100);
        sparse_req.weights = vec![Some(sparse_of(8, 24, wi)), Some(sparse_of(24, 2, wo))];
        assert_eq!(trace_synapses(&dense_req), trace_synapses(&sparse_req));
        assert_eq!(build_groups(&dense_req), build_groups(&sparse_req));
        assert!(!trace_synapses(&sparse_req).is_empty());
    }

    #[test]
    fn regions_the_model_names_replace_the_vote_groups() {
        let mut r = request(100, 100);
        r.layers[1].regions = Some(Regions {
            names: Arc::new(vec![
                "optic lobe".into(),
                "central brain".into(),
                "unused".into(),
            ]),
            labels: Arc::new((0..24).map(|u| (u % 2) as u16).collect()),
        });
        let groups = build_groups(&r);
        let names: Vec<&str> = groups.iter().map(|g| g.name.as_str()).collect();
        assert_eq!(
            names,
            [
                "INPUT",
                "OPTIC LOBE",
                "CENTRAL BRAIN",
                "OUTPUT class 0",
                "OUTPUT class 1"
            ],
            "an empty region has no group"
        );
        assert_eq!((groups[1].units.len(), groups[2].units.len()), (12, 12));
        assert_ne!(groups[1].color, groups[2].color);
        // and it draws, with the regions around the middle
        let (_, info, layout) = render_brain(&r);
        assert_eq!(info.groups.len(), 5);
        assert!(layout.nodes.len() >= 30);
    }

    #[test]
    fn the_picked_neurons_links_come_from_a_sparse_list_too() {
        let mut dense_req = request(700, 450);
        let mut sparse_req = request(700, 450);
        let wi = |i: usize, j: usize| if (i + j).is_multiple_of(3) { 0.7 } else { 0.0 };
        dense_req.weights[0] = Some(matrix(8, 24, wi));
        sparse_req.weights[0] = Some(sparse_of(8, 24, wi));
        dense_req.selected = Some((1, 6));
        sparse_req.selected = Some((1, 6));
        assert!(
            render_brain(&dense_req).0 == render_brain(&sparse_req).0,
            "dense and sparse links draw the same picked neuron"
        );
    }

    #[test]
    fn a_big_sparse_network_traces_and_groups_in_a_blink() {
        let big = 200_000;
        let mut r = request(900, 600);
        let rate = |t: usize, u: usize| (t * 7 + u).is_multiple_of(31);
        r.layers[1] = BrainLayer {
            units: big,
            frame: Some(frame(24, big, rate)),
            history: vec![frame(24, big, rate)],
            regions: None,
        };
        // 8 inputs reach 12 random hidden neurons each; every hidden neuron votes for one of two outputs
        let (mut from, mut to, mut weight) = (Vec::new(), Vec::new(), Vec::new());
        for i in 0..8u32 {
            for k in 0..12u32 {
                from.push(i);
                to.push((i * 7919 + k * 15_485) % big as u32);
                weight.push(0.5 + k as f32 * 0.01);
            }
        }
        r.weights[0] = Some(Links::Sparse(SparseLinks {
            rows: 8,
            cols: big,
            from: Arc::new(from),
            to: Arc::new(to),
            weight: Arc::new(weight),
        }));
        r.weights[1] = Some(Links::Sparse(SparseLinks {
            rows: big,
            cols: 2,
            from: Arc::new((0..big as u32).collect()),
            to: Arc::new((0..big as u32).map(|u| u % 2).collect()),
            weight: Arc::new(vec![1.0; big]),
        }));
        let start = std::time::Instant::now();
        let groups = build_groups(&r);
        let trace = trace_synapses(&r);
        assert!(
            groups
                .iter()
                .any(|g| g.name == "HIDDEN → class 0" && g.units.len() == big / 2)
        );
        assert!(trace.len() <= MAX_TRACE);
        assert!(start.elapsed().as_secs() < 10, "{:?}", start.elapsed());
    }

    #[test]
    fn a_layer_over_the_budget_is_a_dim_sample_with_only_the_firing_neurons_lit() {
        let mut r = request(900, 600);
        let big = 60_000;
        let fire = |t: usize, u: usize| (t * 3 + u).is_multiple_of(40);
        r.layers[1] = BrainLayer {
            units: big,
            frame: Some(frame(24, big, fire)),
            history: vec![frame(24, big, fire)],
            regions: None,
        };
        r.weights = vec![None, None];
        let (_, info, layout) = render_brain(&r);
        assert!(info.lod);
        let expected = (0..big).filter(|&u| fire(23, u)).count();
        assert_eq!(
            info.spikers, expected,
            "every neuron that fired on the last step is counted"
        );
        // At most the sample and a capped number of lit dots are individually pickable
        assert!(
            layout.nodes.len() < MAX_DRAWN + 1500 + 8 + 2 + 100,
            "{}",
            layout.nodes.len()
        );
        let small = request(900, 600);
        assert!(!render_brain(&small).1.lod && render_brain(&small).1.spikers == 0);
    }

    #[test]
    fn zooming_in_draws_more_neurons_of_a_big_layer() {
        let mut r = request(900, 600);
        let big = 150_000;
        r.layers[1] = BrainLayer {
            units: big,
            frame: Some(frame(24, big, |_, _| false)),
            history: vec![frame(24, big, |_, _| false)],
            regions: None,
        };
        r.weights = vec![None, None];
        let far = render_brain(&r).2.nodes.len();
        r.camera.zoom = 3.0;
        let near = render_brain(&r).2.nodes.len();
        assert!(near > far, "{near} vs {far}");
    }

    #[test]
    fn lanes_are_heat_maps_with_a_row_per_neuron_while_there_is_room() {
        let (_, info, _) = render_brain(&request(1400, 800));
        let by_name = |n: &str| info.lanes.iter().find(|l| l.name == n).unwrap().bands;
        assert_eq!(by_name("OUTPUT class 0"), 1, "one neuron, one row");
        assert!(
            by_name("INPUT") > 1 && by_name("INPUT") <= 8,
            "eight neurons, up to eight rows"
        );
        assert!(by_name("HIDDEN → class 0") > 1);
        assert!(by_name("WHOLE NETWORK") >= 1);
        let small = render_brain(&request(300, 120)).1;
        assert!(small.lanes.iter().all(|l| l.bands >= 1));
    }

    #[test]
    fn a_pooled_usage_cell_lights_only_where_more_fire_than_average() {
        // Neurons 0..20000 of a 40000-neuron layer fire, the rest do not: only the cells over the firing half light up
        let mut r = request(1400, 800);
        let big = 40_000;
        let fire = |_: usize, u: usize| u < 20_000;
        r.layers[1] = BrainLayer {
            units: big,
            frame: Some(frame(24, big, fire)),
            history: vec![frame(24, big, fire)],
            regions: None,
        };
        r.weights = vec![None, None];
        let (f, _, layout) = render_brain(&r);
        let b = layout
            .blocks
            .iter()
            .find(|b| b.per_cell > 1)
            .expect("pooled block");
        let white = |gx: usize, gy: usize| {
            let (cx, cy) = (
                (b.x0 + (gx as f32 + 0.5) * b.cell_w) * 1400.0,
                (b.y0 + (gy as f32 + 0.5) * b.cell_h) * 800.0,
            );
            let p = f.pixels()[cy as usize * 1400 + cx as usize];
            p.r > 240 && p.g > 240 && p.b > 240
        };
        let first_half = (0..b.cols / 2).filter(|&x| white(x, 0)).count();
        let second_half = (b.cols / 2 + 1..b.cols)
            .filter(|&x| white(x, b.rows - 1))
            .count();
        assert!(first_half > second_half, "{first_half} vs {second_half}");
    }

    #[test]
    fn zooming_keeps_the_neuron_under_the_pointer_where_it_is() {
        let (w, h) = (1000u32, 600u32);
        let middle = middle_of_map(w, h, 1.0);
        let mut r = request(w, h);
        let (_, _, before) = render_brain(&r);
        // Point at a hidden neuron, away from the middle of the map
        let &(l, u, nx, ny) = before
            .nodes
            .iter()
            .filter(|n| n.0 == 1)
            .max_by(|a, b| (a.2 - middle.0).abs().total_cmp(&(b.2 - middle.0).abs()))
            .unwrap();
        for factor in [2.0f32, 1.5, 0.6] {
            r.camera.zoom_at(factor, nx, ny, middle);
            let (_, _, after) = render_brain(&r);
            let &(_, _, x, y) = after
                .nodes
                .iter()
                .find(|n| n.0 == l && n.1 == u)
                .expect("still in view");
            assert!(
                ((x - nx) * w as f32).abs() < 2.0 && ((y - ny) * h as f32).abs() < 2.0,
                "factor {factor}: ({x}, {y}) vs ({nx}, {ny})"
            );
        }
        assert!((r.camera.zoom - 1.8).abs() < 1e-4);
    }

    #[test]
    fn sliding_moves_the_whole_brain_with_the_pointer() {
        let mut r = request(1000, 600);
        let at = |r: &BrainRequest| {
            render_brain(r)
                .2
                .nodes
                .iter()
                .find(|n| n.0 == 1 && n.1 == 3)
                .map(|n| (n.2, n.3))
                .unwrap()
        };
        let (x0, y0) = at(&r);
        r.camera.slide(0.05, -0.04);
        let (x1, y1) = at(&r);
        assert!(
            ((x1 - x0) - 0.05).abs() < 0.002 && ((y1 - y0) + 0.04).abs() < 0.002,
            "({x0}, {y0}) -> ({x1}, {y1})"
        );
        r.camera.slide(100.0, 100.0);
        assert!(r.camera.pan[0] <= 3.0 && r.camera.pan[1] <= 3.0);
    }
}
