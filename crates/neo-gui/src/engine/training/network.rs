//! The network view: the whole spiking network as columns of neurons joined by glowing curves, on
//! a dark starfield. Green curves are positive weights, red ones negative, and where many cross
//! the light adds up to yellow and white. Spikes travel along the curves as small beads, and
//! clicking a neuron (see [`NetLayout::hit`]) dims everything but its own connections.
//!
//! The picture is built in linear light: every curve, neuron and particle adds energy into an `f32`
//! buffer, a blurred copy of the bright parts is added back (bloom), and the sum is tone-mapped
//! onto the background. The renderer draws no text; the labels of the layers are elements the view
//! lays over the image from the [`NetLayout`] it returns.

use std::sync::Arc;

use neo_snn::telemetry::SpikeFrame;

use crate::engine::canvas::{Frame, Pixel};

use super::links::Links;

const POSITIVE: [f32; 3] = [0.18, 0.95, 0.52];
const NEGATIVE: [f32; 3] = [1.0, 0.30, 0.16];
const NEUTRAL: [f32; 3] = [0.45, 0.55, 0.95];
const WHITE: [f32; 3] = [0.95, 0.97, 1.0];

// ─── camera and layout ──────────────────────────────────────────────────

/// Where the view looks: the centre in world units (the network spans `0..1` both ways) and the zoom.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct NetCamera {
    pub cx: f32,
    pub cy: f32,
    pub zoom: f32,
}

impl Default for NetCamera {
    fn default() -> Self {
        Self {
            cx: 0.5,
            cy: 0.5,
            zoom: 1.0,
        }
    }
}

impl NetCamera {
    pub const MIN_ZOOM: f32 = 0.5;
    pub const MAX_ZOOM: f32 = 8.0;

    /// Zoom by `factor` keeping the world point under the pointer (`fx`, `fy`: fractions of the view) fixed.
    pub fn zoom_at(&mut self, factor: f32, fx: f32, fy: f32) {
        let (wx, wy) = (
            self.cx + (fx - 0.5) / self.zoom,
            self.cy + (fy - 0.5) / self.zoom,
        );
        self.zoom = (self.zoom * factor).clamp(Self::MIN_ZOOM, Self::MAX_ZOOM);
        self.cx = wx - (fx - 0.5) / self.zoom;
        self.cy = wy - (fy - 0.5) / self.zoom;
        self.clamp();
    }

    /// Drag the picture by `dfx`, `dfy` (fractions of the view): the content follows the pointer.
    pub fn pan(&mut self, dfx: f32, dfy: f32) {
        self.cx -= dfx / self.zoom;
        self.cy -= dfy / self.zoom;
        self.clamp();
    }

    fn clamp(&mut self) {
        self.cx = self.cx.clamp(0.0, 1.0);
        self.cy = self.cy.clamp(0.0, 1.0);
    }

    /// Bits of the three numbers, for keys that must tell two cameras apart.
    pub fn bits(&self) -> [u32; 5] {
        [
            self.cx.to_bits(),
            self.cy.to_bits(),
            self.zoom.to_bits(),
            0,
            0,
        ]
    }
}

/// One layer on screen. Positions are fractions of the image.
#[derive(Debug, Clone, PartialEq)]
pub struct LayerGeom {
    pub x: f32,
    pub top: f32,
    pub bottom: f32,
    pub units: usize,
    /// `(x, y)` of every neuron; empty for a layer with more than [`MAX_NODES`] (its neurons are found by height).
    pub nodes: Vec<(f32, f32)>,
}

/// Neurons per layer listed in the layout. A bigger layer is drawn as blocks (see [`NetLayout::lod`]).
pub const MAX_NODES: usize = 4_000;
/// Connections up to which every one is drawn as its own curve; more are drawn as ribbons between blocks.
pub const CLASSIC_MAX_LINKS: usize = 20_000;
/// Up to this many connections each is a bundle of three strands; above it one strand (three times cheaper).
const BUNDLE_MAX_LINKS: usize = 4_000;

/// Where everything is in the image, for the labels laid over it and for picking a neuron.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct NetLayout {
    pub layers: Vec<LayerGeom>,
    /// Size of the image in logical pixels.
    pub size: (f32, f32),
    /// True when the network is too big to draw connection by connection: neurons are blocks and the
    /// connections between two blocks are one ribbon (set by [`render_network`]).
    pub lod: bool,
}

impl NetLayout {
    /// The neuron `(layer, unit)` nearest to the point (`fx`, `fy`: fractions of the image), if one is
    /// within `radius` logical pixels.
    pub fn hit(&self, fx: f32, fy: f32, radius: f32) -> Option<(usize, usize)> {
        let (w, h) = self.size;
        let mut best: Option<((usize, usize), f32)> = None;
        for (l, layer) in self.layers.iter().enumerate() {
            // A layer too big to list its neurons: the column is a ruler, the neuron is where the point falls on it
            if layer.nodes.is_empty() && layer.units > 0 {
                let (dx, span) = (((layer.x - fx) * w).abs(), (layer.bottom - layer.top) * h);
                if dx <= radius + 4.0 && fy >= layer.top && fy <= layer.bottom && span > 0.0 {
                    let u = (((fy - layer.top) * h / span * layer.units as f32) as usize)
                        .min(layer.units - 1);
                    if best.is_none_or(|(_, b)| dx < b) {
                        best = Some(((l, u), dx));
                    }
                }
                continue;
            }
            for (u, &(x, y)) in layer.nodes.iter().enumerate() {
                let d = (((x - fx) * w).powi(2) + ((y - fy) * h).powi(2)).sqrt();
                if d <= radius && best.is_none_or(|(_, b)| d < b) {
                    best = Some(((l, u), d));
                }
            }
        }
        best.map(|(n, _)| n)
    }
}

fn world_x(layer: usize, layers: usize) -> f32 {
    if layers <= 1 {
        0.5
    } else {
        0.17 + 0.52 * layer as f32 / (layers - 1) as f32
    }
}

/// A layer's column is taller the more neurons it has, but not in proportion: a two-neuron output
/// stays a visible column.
fn world_height(units: usize, most: usize) -> f32 {
    0.40 + 0.44 * (units as f32 / most.max(1) as f32).powf(0.6)
}

/// Layer columns for the given neuron counts as `camera` sees them in an image of `size_px` physical pixels
/// (`scale` device pixels per logical one).
pub fn network_layout(
    units: &[usize],
    camera: &NetCamera,
    size_px: (u32, u32),
    scale: f32,
) -> NetLayout {
    let most = units.iter().copied().max().unwrap_or(1);
    let layers = units.len();
    let layers = (0..layers)
        .map(|l| {
            let n = units[l].max(1);
            let hh = world_height(n, most);
            let x = 0.5 + (world_x(l, layers) - camera.cx) * camera.zoom;
            let y_of = |wy: f32| 0.5 + (wy - camera.cy) * camera.zoom;
            let nodes = if units[l] <= MAX_NODES {
                (0..units[l])
                    .map(|j| (x, y_of(0.5 + ((j as f32 + 0.5) / n as f32 - 0.5) * hh)))
                    .collect()
            } else {
                Vec::new()
            };
            LayerGeom {
                x,
                top: y_of(0.5 - hh / 2.0),
                bottom: y_of(0.5 + hh / 2.0),
                units: units[l],
                nodes,
            }
        })
        .collect();
    let scale = scale.max(0.5);
    NetLayout {
        layers,
        size: (size_px.0 as f32 / scale, size_px.1 as f32 / scale),
        lod: false,
    }
}

// ─── the request ────────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct NetLayer {
    pub units: usize,
    pub frame: Option<Arc<SpikeFrame>>,
}

#[derive(Debug, Clone)]
pub struct NetworkRequest {
    pub layers: Vec<NetLayer>,
    /// `weights[l]` joins layer `l` to `l + 1` (`rows` = units of `l`, `cols` = units of `l + 1`).
    pub weights: Vec<Option<Links>>,
    pub camera: NetCamera,
    pub selected: Option<(usize, usize)>,
    /// Advances as the run does: it moves the beads and makes the particles twinkle.
    pub phase: f32,
    pub width: u32,
    pub height: u32,
    pub scale: f32,
}

// ─── linear-light buffer ────────────────────────────────────────────────

pub(super) struct Hdr {
    pub(super) w: usize,
    pub(super) h: usize,
    pub(super) px: Vec<f32>,
}

impl Hdr {
    pub(super) fn new(w: usize, h: usize) -> Self {
        Self {
            w,
            h,
            px: vec![0.0; w * h * 3],
        }
    }

    /// Adds `rgb * e` at a fractional position, spread over the four pixels around it.
    pub(super) fn splat(&mut self, x: f32, y: f32, rgb: [f32; 3], e: f32) {
        let (x0, y0) = (x.floor(), y.floor());
        let (fx, fy) = (x - x0, y - y0);
        let (xi, yi) = (x0 as i64, y0 as i64);
        for (dx, dy, wt) in [
            (0, 0, (1.0 - fx) * (1.0 - fy)),
            (1, 0, fx * (1.0 - fy)),
            (0, 1, (1.0 - fx) * fy),
            (1, 1, fx * fy),
        ] {
            let (px, py) = (xi + dx, yi + dy);
            if px >= 0 && py >= 0 && (px as usize) < self.w && (py as usize) < self.h {
                let i = (py as usize * self.w + px as usize) * 3;
                for k in 0..3 {
                    self.px[i + k] += rgb[k] * e * wt;
                }
            }
        }
    }

    /// A soft round spot of radius `r` pixels.
    pub(super) fn glow(&mut self, cx: f32, cy: f32, r: f32, rgb: [f32; 3], e: f32) {
        let reach = r * 2.4 + 1.0;
        let (x0, x1) = (
            (cx - reach).floor().max(0.0) as usize,
            ((cx + reach).ceil().max(0.0) as usize).min(self.w.saturating_sub(1)),
        );
        let (y0, y1) = (
            (cy - reach).floor().max(0.0) as usize,
            ((cy + reach).ceil().max(0.0) as usize).min(self.h.saturating_sub(1)),
        );
        if x0 > x1 || y0 > y1 {
            return;
        }
        for y in y0..=y1 {
            for x in x0..=x1 {
                let d2 = ((x as f32 + 0.5 - cx).powi(2) + (y as f32 + 0.5 - cy).powi(2))
                    / (r * r).max(1e-4);
                let a = e * (-d2 * 1.4).exp();
                if a > 1e-4 {
                    let i = (y * self.w + x) * 3;
                    for k in 0..3 {
                        self.px[i + k] += rgb[k] * a;
                    }
                }
            }
        }
    }
}

/// `p(t)` of the cubic through `p0` and `p3` with horizontal tangents at both ends (an S), its two
/// inner control points lifted by `bend` and `-bend` to make a bundle of curves look less combed.
pub(super) fn curve_point(p0: (f32, f32), p3: (f32, f32), bend: f32, t: f32) -> (f32, f32) {
    let dx = p3.0 - p0.0;
    let c1 = (p0.0 + dx * 0.5, p0.1 + bend);
    let c2 = (p3.0 - dx * 0.5, p3.1 - bend);
    let u = 1.0 - t;
    let (b0, b1, b2, b3) = (u * u * u, 3.0 * u * u * t, 3.0 * u * t * t, t * t * t);
    (
        b0 * p0.0 + b1 * c1.0 + b2 * c2.0 + b3 * p3.0,
        b0 * p0.1 + b1 * c1.1 + b2 * c2.1 + b3 * p3.1,
    )
}

fn draw_curve(h: &mut Hdr, p0: (f32, f32), p3: (f32, f32), bend: f32, rgb: [f32; 3], per_px: f32) {
    // The curve stays inside the box of its control points: skip what is entirely off screen
    let ys = [p0.1, p3.1, p0.1 + bend, p3.1 - bend];
    let (ymin, ymax) = (
        ys.iter().copied().fold(f32::MAX, f32::min),
        ys.iter().copied().fold(f32::MIN, f32::max),
    );
    if p0.0.max(p3.0) < 0.0 || p0.0.min(p3.0) > h.w as f32 || ymax < 0.0 || ymin > h.h as f32 {
        return;
    }
    let len = (p3.0 - p0.0).abs() + (p3.1 - p0.1).abs() + bend.abs();
    let n = ((len / 1.2).ceil() as usize).clamp(8, 2600);
    let step = len / n as f32;
    for i in 0..=n {
        let (x, y) = curve_point(p0, p3, bend, i as f32 / n as f32);
        h.splat(x, y, rgb, per_px * step);
    }
}

// ─── bloom ──────────────────────────────────────────────────────────────

fn box_blur(data: &mut [f32], w: usize, h: usize, r: usize) {
    let mut tmp = vec![0.0f32; data.len()];
    let norm = 1.0 / (2 * r + 1) as f32;
    for y in 0..h {
        for x in 0..w {
            for k in 0..3 {
                let mut s = 0.0;
                for d in 0..=2 * r {
                    let xx = (x + d).saturating_sub(r).min(w - 1);
                    s += data[(y * w + xx) * 3 + k];
                }
                tmp[(y * w + x) * 3 + k] = s * norm;
            }
        }
    }
    for y in 0..h {
        for x in 0..w {
            for k in 0..3 {
                let mut s = 0.0;
                for d in 0..=2 * r {
                    let yy = (y + d).saturating_sub(r).min(h - 1);
                    s += tmp[(yy * w + x) * 3 + k];
                }
                data[(y * w + x) * 3 + k] = s * norm;
            }
        }
    }
}

/// The bright part of `hdr` (above `threshold`) blurred by about `radius` pixels, at full size.
pub(super) fn bloom(hdr: &Hdr, threshold: f32, radius: f32) -> Vec<f32> {
    const F: usize = 4;
    let (bw, bh) = (hdr.w.div_ceil(F), hdr.h.div_ceil(F));
    let mut small = vec![0.0f32; bw * bh * 3];
    for y in 0..hdr.h {
        for x in 0..hdr.w {
            let (i, j) = ((y * hdr.w + x) * 3, ((y / F) * bw + x / F) * 3);
            for k in 0..3 {
                small[j + k] += (hdr.px[i + k] - threshold).max(0.0) / (F * F) as f32;
            }
        }
    }
    let r = ((radius / F as f32).round() as usize).max(1);
    for _ in 0..2 {
        box_blur(&mut small, bw, bh, r);
    }
    let mut out = vec![0.0f32; hdr.w * hdr.h * 3];
    for y in 0..hdr.h {
        let sy = ((y as f32 + 0.5) / F as f32 - 0.5).clamp(0.0, (bh - 1) as f32);
        let (y0, fy) = (sy.floor() as usize, sy.fract());
        let y1 = (y0 + 1).min(bh - 1);
        for x in 0..hdr.w {
            let sx = ((x as f32 + 0.5) / F as f32 - 0.5).clamp(0.0, (bw - 1) as f32);
            let (x0, fx) = (sx.floor() as usize, sx.fract());
            let x1 = (x0 + 1).min(bw - 1);
            for k in 0..3 {
                let at = |xx: usize, yy: usize| small[(yy * bw + xx) * 3 + k];
                let top = at(x0, y0) * (1.0 - fx) + at(x1, y0) * fx;
                let bottom = at(x0, y1) * (1.0 - fx) + at(x1, y1) * fx;
                out[(y * hdr.w + x) * 3 + k] = top * (1.0 - fy) + bottom * fy;
            }
        }
    }
    out
}

// ─── the picture ────────────────────────────────────────────────────────

pub(super) fn hash01(a: usize, b: usize, c: usize) -> f32 {
    let mut h = (a as u64).wrapping_mul(0x9E37_79B9_7F4A_7C15)
        ^ (b as u64).wrapping_mul(0xC2B2_AE3D_27D4_EB4F)
        ^ (c as u64).wrapping_mul(0x1656_67B1_9E37_79F9);
    h ^= h >> 29;
    h = h.wrapping_mul(0xBF58_476D_1CE4_E5B9);
    h ^= h >> 32;
    (h % 10_000) as f32 / 10_000.0
}

/// Dust and stars: fixed spots that drift a little with the camera (the far ones less) and twinkle.
fn draw_particles(h: &mut Hdr, req: &NetworkRequest, s: f32) {
    let (w, hh) = (h.w as f32, h.h as f32);
    for i in 0..170 {
        let depth = 0.2 + 0.8 * hash01(i, 1, 0);
        let (u, v) = (hash01(i, 2, 0), hash01(i, 3, 0));
        let x = (u - (req.camera.cx - 0.5) * depth * 0.30 * req.camera.zoom).rem_euclid(1.0) * w;
        let y = (v - (req.camera.cy - 0.5) * depth * 0.30 * req.camera.zoom).rem_euclid(1.0) * hh;
        let twinkle =
            0.65 + 0.35 * (req.phase * 1.7 + hash01(i, 4, 0) * std::f32::consts::TAU).sin();
        let e = (0.10 + 0.55 * hash01(i, 5, 0)) * twinkle * depth;
        let tint = [0.75 + 0.2 * hash01(i, 6, 0), 0.82, 1.0];
        h.glow(x, y, (0.5 + 1.1 * hash01(i, 7, 0)) * s, tint, e);
    }
    // Two bright ones, like moons
    for (u, v, r) in [(0.57, 0.08, 4.2), (0.21, 0.17, 2.6)] {
        h.glow(u * w, v * hh, r * s, [0.95, 0.97, 1.0], 1.3);
    }
}

// ─── how much to draw ───────────────────────────────────────────────────

/// Connections stored between layer `l` and `l + 1`: the links, or every pair of neurons while the weights have not arrived.
fn pair_size(req: &NetworkRequest, l: usize) -> usize {
    match req.weights.get(l).and_then(|m| m.as_ref()) {
        Some(m) => m.len(),
        None => req.layers[l].units.saturating_mul(req.layers[l + 1].units),
    }
}

/// True when the network is too big to draw connection by connection.
fn is_big(req: &NetworkRequest) -> bool {
    req.layers.iter().any(|l| l.units > MAX_NODES)
        || (0..req.layers.len().saturating_sub(1))
            .map(|l| pair_size(req, l))
            .fold(0usize, usize::saturating_add)
            > CLASSIC_MAX_LINKS
}

fn lerp3(a: [f32; 3], b: [f32; 3], t: f32) -> [f32; 3] {
    let t = t.clamp(0.0, 1.0);
    [
        a[0] + (b[0] - a[0]) * t,
        a[1] + (b[1] - a[1]) * t,
        a[2] + (b[2] - a[2]) * t,
    ]
}

// ─── every connection a curve ───────────────────────────────────────────

/// Every neuron a dot, every connection a bundle of curves, a bead on the curves of a firing neuron.
fn draw_classic(
    h: &mut Hdr,
    req: &NetworkRequest,
    layout: &NetLayout,
    s: f32,
    sel: Option<(usize, usize)>,
) {
    let (wf, hf) = (h.w as f32, h.h as f32);
    let units: Vec<usize> = req.layers.iter().map(|l| l.units).collect();
    let px = |l: usize, u: usize| {
        (
            layout.layers[l].nodes[u].0 * wf,
            layout.layers[l].nodes[u].1 * hf,
        )
    };
    let incident = |l: usize, i: usize, j: usize| {
        sel.is_some_and(|(sl, su)| (sl == l && su == i) || (sl == l + 1 && su == j))
    };
    let dim_other = if sel.is_some() { 0.10 } else { 1.0 };
    let links_of = |l: usize| {
        req.weights
            .get(l)
            .and_then(|m| m.as_ref())
            .filter(|m| m.rows() == units[l] && m.cols() == units[l + 1])
    };
    let total: usize = (0..units.len().saturating_sub(1))
        .map(|l| pair_size(req, l))
        .fold(0, usize::saturating_add);
    // A dense network of a hundred neurons has 10 000 connections: one strand each keeps the frame quick
    let strands: &[(usize, f32)] = if total > BUNDLE_MAX_LINKS {
        &[(0, 0.9)]
    } else {
        &[(0, 0.55), (1, 0.30), (2, 0.30)]
    };
    // The more connections, the more of them pile up in the same place: each gets dimmer, or the picture is a white slab
    let gain = (400.0 / total.max(400) as f32).powf(0.8);
    let bead_floor = if total > 2_000 { 0.35 } else { 0.0 };
    let bead_r = if total > 400 { 2.8 * s } else { 2.0 * s };

    // Connections
    for l in 0..units.len().saturating_sub(1) {
        let links = links_of(l);
        let wmax = links.map_or(0.0, |m| m.abs_max()).max(1e-6);
        let mut draw = |i: usize, j: usize, v: Option<f32>| {
            let (p0, p3) = (px(l, i), px(l + 1, j));
            let bend = (hash01(i + 31 * l, j, 11) - 0.5) * 0.34 * hf * req.camera.zoom;
            let hot = incident(l, i, j);
            let (rgb, per_px) = match v {
                Some(v) if v.is_finite() => {
                    let strength = (v.abs() / wmax).powf(0.9);
                    (
                        if v >= 0.0 { POSITIVE } else { NEGATIVE },
                        0.13 + 0.55 * strength,
                    )
                }
                _ => (NEUTRAL, 0.09),
            };
            if hot {
                draw_curve(
                    h,
                    p0,
                    p3,
                    bend * 0.5,
                    WHITE,
                    per_px.max(0.2) * 2.2 * gain.sqrt(),
                );
            } else {
                // A bundle of three strands, each bowed a little differently: a dense, hairy look
                for &(k, share) in strands {
                    let bow = bend * (1.0 + 0.9 * k as f32) * if k == 2 { -0.7 } else { 1.0 };
                    draw_curve(h, p0, p3, bow, rgb, per_px * share * dim_other * gain);
                }
            }
        };
        match links {
            Some(m) => m.for_each(|i, j, w| draw(i, j, Some(w))),
            None => {
                for i in 0..units[l] {
                    for j in 0..units[l + 1] {
                        draw(i, j, None);
                    }
                }
            }
        }
    }

    // Spikes in flight: a bead per active source neuron on each of its connections
    for l in 0..units.len().saturating_sub(1) {
        let Some(f) = req.layers[l]
            .frame
            .as_deref()
            .filter(|f| !f.is_empty() && f.units == units[l])
        else {
            continue;
        };
        let links = links_of(l);
        let wmax = links.map_or(0.0, |m| m.abs_max()).max(1e-6);
        let mut bead = |i: usize, j: usize, v: f32| {
            let rate = f.unit_rate(i);
            let hot = incident(l, i, j);
            // With thousands of connections only the stronger ones carry a bead, or the beads are lost among the curves
            if rate < 0.04 || (!hot && v.abs() < bead_floor * wmax) {
                return;
            }
            let bend = (hash01(i + 31 * l, j, 11) - 0.5) * 0.34 * hf * req.camera.zoom;
            let t = (req.phase * 0.30 + hash01(i, j + 97 * l, 5)).fract();
            let (x, y) = curve_point(
                px(l, i),
                px(l + 1, j),
                if hot { bend * 0.5 } else { bend },
                t,
            );
            let rgb = if hot {
                WHITE
            } else if v >= 0.0 {
                [0.55, 1.0, 0.75]
            } else {
                [1.0, 0.75, 0.50]
            };
            let e = 3.0
                * rate
                * (0.35 + 0.65 * (v.abs() / wmax))
                * if sel.is_some() && !hot { 0.15 } else { 1.0 };
            h.glow(x, y, bead_r, rgb, e);
        };
        match links {
            Some(m) => m.for_each(&mut bead),
            None => {
                for i in 0..units[l] {
                    for j in 0..units[l + 1] {
                        bead(i, j, 0.0);
                    }
                }
            }
        }
    }

    // Neurons
    let dot = (1.5 * s * req.camera.zoom.sqrt()).clamp(1.0, 4.0 * s);
    for (l, layer) in req.layers.iter().enumerate() {
        let frame = layer
            .frame
            .as_deref()
            .filter(|f| !f.is_empty() && f.units == layer.units);
        for u in 0..layer.units {
            let (x, y) = px(l, u);
            let rate = frame.map_or(0.0, |f| f.unit_rate(u).clamp(0.0, 1.0));
            let fired = frame.is_some_and(|f| f.spiked(f.steps - 1, u));
            let near = sel.is_none_or(|(sl, su)| (sl, su) == (l, u) || sl.abs_diff(l) == 1);
            let k = if near { 1.0 } else { 0.30 };
            h.glow(x, y, dot, [0.85, 0.95, 1.0], (1.3 + 2.6 * rate) * k);
            if fired {
                h.glow(x, y, dot * 3.0, [1.0, 0.85, 0.45], 0.55 * k);
            }
            if sel == Some((l, u)) {
                h.glow(x, y, dot * 4.0, WHITE, 1.4);
            }
        }
    }
}

// ─── level of detail: blocks and ribbons ────────────────────────────────

/// A band of the given half-width that follows an S-curve between two points, its width changing from one end to the
/// other. `e` is the energy per pixel of area, so a wide ribbon is not brighter per pixel than a narrow one.
fn draw_ribbon(
    h: &mut Hdr,
    p0: (f32, f32),
    p3: (f32, f32),
    hw0: f32,
    hw3: f32,
    rgb: [f32; 3],
    e: f32,
) {
    let dx = p3.0 - p0.0;
    if dx.abs() < 1.0 {
        return;
    }
    // Steps along the curve, not along x: a steep ribbon would otherwise be dashes of vertical hatching
    let n = (((dx.abs() + (p3.1 - p0.1).abs()) / 3.0).ceil() as usize).clamp(4, 3000);
    let sx = dx.abs() / n as f32;
    let hf = h.h as f32;
    for i in 0..=n {
        let t = i as f32 / n as f32;
        let (x, y) = curve_point(p0, p3, 0.0, t);
        let hw = (hw0 + (hw3 - hw0) * t).max(0.5);
        // Only the rows on screen
        let (lo, hi) = (
            ((y - hw).floor() as i32).max(-2),
            ((y + hw).ceil() as i32).min(hf as i32 + 2),
        );
        for yy in lo..=hi {
            let d = (yy as f32 + 0.5 - y) / hw;
            let k = 1.0 - d * d;
            if k > 0.0 {
                h.splat(x, yy as f32, rgb, e * sx * k);
            }
        }
    }
}

/// One column of the picture in the aggregated view: the layer's neurons in `blocks` equal slices.
struct Column {
    x: f32,
    top: f32,
    height: f32,
    units: usize,
    blocks: usize,
    /// Mean firing rate of each block (from a sample of its neurons).
    rate: Vec<f32>,
}

/// Aggregated drawing for networks too big to show connection by connection. Neurons are blocks (a bar
/// whose brightness is how much it fires), the connections between two blocks are one ribbon (as thick as
/// the weights between them add up to, green when they excite and red when they inhibit), and over the
/// ribbons go the strongest single connections in view, so that zooming in brings detail back.
fn draw_lod(
    h: &mut Hdr,
    req: &NetworkRequest,
    layout: &NetLayout,
    s: f32,
    sel: Option<(usize, usize)>,
) {
    const RIBBONS: usize = 400;
    const TOP_LINKS: usize = 300;
    let (wf, hf) = (h.w as f32, h.h as f32);
    let n = req.layers.len();
    let margin = 40.0 * s;
    let columns: Vec<Column> = (0..n)
        .map(|l| {
            let g = &layout.layers[l];
            let (top, height) = (g.top * hf, ((g.bottom - g.top) * hf).max(1.0));
            let units = req.layers[l].units.max(1);
            let blocks = ((height / (6.0 * s)).floor() as usize)
                .clamp(4, 400)
                .min(units);
            let frame = req.layers[l]
                .frame
                .as_deref()
                .filter(|f| !f.is_empty() && f.units == req.layers[l].units);
            let rate = (0..blocks)
                .map(|b| {
                    let (a, z) = (
                        b * units / blocks,
                        ((b + 1) * units / blocks)
                            .max(b * units / blocks + 1)
                            .min(units),
                    );
                    let Some(f) = frame else { return 0.0 };
                    let step = (z - a).div_ceil(48).max(1);
                    let (mut sum, mut count) = (0.0, 0);
                    let mut u = a;
                    while u < z {
                        sum += f.unit_rate(u);
                        count += 1;
                        u += step;
                    }
                    sum / count.max(1) as f32
                })
                .collect();
            Column {
                x: g.x * wf,
                top,
                height,
                units,
                blocks,
                rate,
            }
        })
        .collect();
    let y_of = |l: usize, u: usize| {
        columns[l].top + (u as f32 + 0.5) / columns[l].units as f32 * columns[l].height
    };
    let in_view = |y: f32| y >= -margin && y <= hf + margin;
    let dim = if sel.is_some() { 0.25 } else { 1.0 };

    for l in 0..n.saturating_sub(1) {
        let (a, b) = (&columns[l], &columns[l + 1]);
        let Some(links) = req
            .weights
            .get(l)
            .and_then(|m| m.as_ref())
            .filter(|m| m.rows() == req.layers[l].units && m.cols() == req.layers[l + 1].units)
        else {
            continue;
        };
        let (rows, cols) = (links.rows().max(1), links.cols().max(1));
        let wmax = links.abs_max().max(1e-6);

        // One pass over the links: add each into its pair of blocks, and keep the strongest that are in view
        let mut sums = vec![(0.0f32, 0.0f32); a.blocks * b.blocks];
        let mut strongest: Vec<(f32, u32, u32)> = Vec::new();
        let mut floor = 0.0f32;
        links.for_each(|i, j, w| {
            if !w.is_finite() {
                return;
            }
            let bi = ((i as u64 * a.blocks as u64 / rows as u64) as usize).min(a.blocks - 1);
            let bj = ((j as u64 * b.blocks as u64 / cols as u64) as usize).min(b.blocks - 1);
            let cell = &mut sums[bi * b.blocks + bj];
            if w >= 0.0 {
                cell.0 += w;
            } else {
                cell.1 -= w;
            }
            if w.abs() > floor && in_view(y_of(l, i)) && in_view(y_of(l + 1, j)) {
                strongest.push((w, i as u32, j as u32));
                if strongest.len() > 8 * TOP_LINKS {
                    strongest.sort_by(|x, y| y.0.abs().total_cmp(&x.0.abs()));
                    strongest.truncate(TOP_LINKS);
                    floor = strongest.last().map_or(0.0, |c| c.0.abs());
                }
            }
        });
        strongest.sort_by(|x, y| y.0.abs().total_cmp(&x.0.abs()));
        strongest.truncate(TOP_LINKS);

        // Ribbons: the strongest block pairs
        let mut ribbons: Vec<(f32, usize, usize)> = sums
            .iter()
            .enumerate()
            .filter(|(_, c)| c.0 + c.1 > 0.0)
            .map(|(k, c)| (c.0 + c.1, k / b.blocks, k % b.blocks))
            .collect();
        if ribbons.len() > RIBBONS {
            ribbons.select_nth_unstable_by(RIBBONS, |x, y| y.0.total_cmp(&x.0));
            ribbons.truncate(RIBBONS);
        }
        ribbons.sort_by(|x, y| y.0.total_cmp(&x.0));
        let peak = ribbons.first().map_or(1e-9, |r| r.0).max(1e-9);
        let (bh_a, bh_b) = (a.height / a.blocks as f32, b.height / b.blocks as f32);
        for &(total, bi, bj) in &ribbons {
            let (ya, yb) = (
                a.top + (bi as f32 + 0.5) * bh_a,
                b.top + (bj as f32 + 0.5) * bh_b,
            );
            let (hw_a, hw_b) = ((bh_a * 0.45).max(0.6), (bh_b * 0.45).max(0.6));
            if ya.max(yb) + hw_a.max(hw_b) < 0.0 || ya.min(yb) - hw_a.max(hw_b) > hf {
                continue;
            }
            let c = sums[bi * b.blocks + bj];
            // The colour follows the balance of the links between the two blocks, stretched so that a
            // mostly-excitatory pair is clearly green and a mostly-inhibitory one clearly red
            let balance = ((c.0 / (c.0 + c.1).max(1e-9) - 0.5) * 2.4 + 0.5).clamp(0.0, 1.0);
            let rgb = lerp3(NEGATIVE, POSITIVE, balance);
            let strength = (total / peak).powf(1.2);
            draw_ribbon(
                h,
                (a.x, ya),
                (b.x, yb),
                hw_a,
                hw_b,
                rgb,
                (0.003 + 0.030 * strength) * dim,
            );
        }

        // Spikes in flight: beads on the ribbons that leave a firing block
        for &(total, bi, bj) in ribbons.iter().take(160) {
            let rate = a.rate[bi];
            if rate < 0.02 {
                continue;
            }
            let (ya, yb) = (
                a.top + (bi as f32 + 0.5) * bh_a,
                b.top + (bj as f32 + 0.5) * bh_b,
            );
            let t = (req.phase * 0.30 + hash01(bi, bj + 97 * l, 5)).fract();
            let (x, y) = curve_point((a.x, ya), (b.x, yb), 0.0, t);
            let off = (hash01(bi, bj, 9) - 0.5) * bh_a.min(bh_b) * 0.6;
            h.glow(
                x,
                y + off,
                2.0 * s,
                [0.7, 1.0, 0.8],
                1.6 * rate * (0.3 + 0.7 * total / peak) * if sel.is_some() { 0.2 } else { 1.0 },
            );
        }

        // The strongest single connections in view, over the ribbons
        for &(w, i, j) in &strongest {
            let (p0, p3) = ((a.x, y_of(l, i as usize)), (b.x, y_of(l + 1, j as usize)));
            let bend = (hash01(i as usize, j as usize, 11) - 0.5) * 0.2 * hf.min(b.height);
            let strength = (w.abs() / wmax).powf(0.9);
            draw_curve(
                h,
                p0,
                p3,
                bend,
                if w >= 0.0 { POSITIVE } else { NEGATIVE },
                (0.12 + 0.50 * strength) * dim,
            );
        }
    }

    // The picked neuron's own connections, the strongest of them
    if let Some((sl, su)) = sel {
        for (pair, from_me) in [(sl.checked_sub(1), false), (Some(sl), true)] {
            let Some(p) = pair else { continue };
            let Some(m) = req.weights.get(p).and_then(|m| m.as_ref()) else {
                continue;
            };
            let others = if from_me {
                m.strongest_from(su, 200)
            } else {
                m.strongest_into(su, 200)
            };
            for (o, w) in others {
                let (pa, pb) = if from_me {
                    ((sl, su), (sl + 1, o))
                } else {
                    ((sl - 1, o), (sl, su))
                };
                let (p0, p3) = (
                    (columns[pa.0].x, y_of(pa.0, pa.1)),
                    (columns[pb.0].x, y_of(pb.0, pb.1)),
                );
                draw_curve(
                    h,
                    p0,
                    p3,
                    0.0,
                    WHITE,
                    (0.25 + 0.4 * (w.abs() / m.abs_max().max(1e-6))).min(0.6),
                );
            }
        }
        h.glow(columns[sl].x, y_of(sl, su), 5.0 * s, WHITE, 1.6);
    }

    // The columns: a bar per block, brighter the more it fires; real neurons once they are far enough apart
    for (l, c) in columns.iter().enumerate() {
        let bh = c.height / c.blocks as f32;
        let b0 = ((-margin - c.top) / bh).floor().max(0.0) as usize;
        let b1 = (((hf + margin - c.top) / bh).ceil().max(0.0) as usize).min(c.blocks);
        for b in b0..b1 {
            let (y0, y1) = (c.top + b as f32 * bh, c.top + (b + 1) as f32 * bh - 1.0);
            let rate = c.rate[b].clamp(0.0, 1.0).sqrt();
            let rgb = lerp3([0.55, 0.70, 1.0], [1.0, 0.90, 0.50], rate);
            let e = (0.35 + 2.4 * rate) * 0.35;
            let mut y = y0.max(-margin);
            while y < y1.min(hf + margin) {
                for dx in [-1.0f32, 0.0, 1.0] {
                    h.splat(c.x + dx, y, rgb, e);
                }
                y += 1.0;
            }
        }
        let per = c.height / c.units as f32;
        if per >= 6.0 * s {
            let frame = req.layers[l]
                .frame
                .as_deref()
                .filter(|f| !f.is_empty() && f.units == req.layers[l].units);
            let u0 = ((-margin - c.top) / per).floor().max(0.0) as usize;
            let u1 = (((hf + margin - c.top) / per).ceil().max(0.0) as usize).min(c.units);
            for u in u0..u1 {
                let rate = frame.map_or(0.0, |f| f.unit_rate(u).clamp(0.0, 1.0));
                h.glow(
                    c.x,
                    y_of(l, u),
                    1.6 * s,
                    [0.85, 0.95, 1.0],
                    1.2 + 2.0 * rate,
                );
            }
        }
    }
}

pub fn render_network(req: &NetworkRequest) -> (Frame, NetLayout) {
    let (w, ht) = (req.width.max(1), req.height.max(1));
    let s = req.scale.max(0.5);
    let units: Vec<usize> = req.layers.iter().map(|l| l.units).collect();
    let big = is_big(req);
    let mut layout = network_layout(&units, &req.camera, (w, ht), s);
    layout.lod = big;
    let (wf, hf) = (w as f32, ht as f32);
    let mut hdr = Hdr::new(w as usize, ht as usize);

    draw_particles(&mut hdr, req, s);

    let sel = req
        .selected
        .filter(|&(l, u)| units.get(l).is_some_and(|&n| u < n));
    if big {
        draw_lod(&mut hdr, req, &layout, s, sel);
    } else {
        draw_classic(&mut hdr, req, &layout, s, sel);
    }

    // Bloom, then onto the background
    let tight = bloom(&hdr, 0.25, 7.0 * s);
    let wide = bloom(&hdr, 0.6, 28.0 * s);
    let mut frame = Frame::new(w, ht);
    {
        let (eu_l, eu_r): (Vec<f32>, Vec<f32>) = (0..w as usize)
            .map(|x| {
                let u = (x as f32 + 0.5) / wf;
                ((-3.0 * u * u).exp(), (-3.0 * (1.0 - u) * (1.0 - u)).exp())
            })
            .unzip();
        let pixels = frame.pixels_mut();
        for y in 0..ht as usize {
            let v = (y as f32 + 0.5) / hf;
            let ev = (-9.0 * v).exp();
            let base = [
                15.0 + (6.0 - 15.0) * v.powf(0.8),
                18.0 + (9.0 - 18.0) * v.powf(0.8),
                48.0 + (28.0 - 48.0) * v.powf(0.8),
            ];
            for x in 0..w as usize {
                let glow = ev * (0.95 * eu_l[x] + 0.75 * eu_r[x] + 0.06);
                let i = (y * w as usize + x) * 3;
                let mut out = [0u8; 3];
                for k in 0..3 {
                    let tint = [92.0, 48.0, 88.0][k];
                    let bg = ((base[k] + tint * glow) / 255.0).clamp(0.0, 1.0);
                    let e = hdr.px[i + k] + 1.1 * tight[i + k] + 0.9 * wide[i + k];
                    let tm = 1.0 - (-1.7 * e).exp();
                    out[k] = ((1.0 - (1.0 - bg) * (1.0 - tm)) * 255.0).round() as u8;
                }
                pixels[y * w as usize + x] = Pixel::rgb(out[0], out[1], out[2]);
            }
        }
    }

    // Frames round the columns, drawn over the light
    {
        let mut c = frame.canvas();
        for (l, g) in layout.layers.iter().enumerate() {
            let pad = 10.0 * s * req.camera.zoom.sqrt();
            let (x0, x1) = (g.x * wf - pad, g.x * wf + pad);
            let (y0, y1) = (g.top * hf - pad, g.bottom * hf + pad);
            let alpha = if sel.is_some_and(|(sl, _)| sl == l) {
                0.75
            } else {
                0.30
            };
            let edge = Pixel::rgb(120, 160, 255);
            c.rect(x0, y0, x1, y0 + 1.0, edge, alpha);
            c.rect(x0, y1 - 1.0, x1, y1, edge, alpha);
            c.rect(x0, y0, x0 + 1.0, y1, edge, alpha);
            c.rect(x1 - 1.0, y0, x1, y1, edge, alpha);
        }
    }
    (frame, layout)
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

    fn sparse(rows: usize, cols: usize, from: Vec<u32>, to: Vec<u32>, weight: Vec<f32>) -> Links {
        Links::Sparse(SparseLinks {
            rows,
            cols,
            from: Arc::new(from),
            to: Arc::new(to),
            weight: Arc::new(weight),
        })
    }

    /// A big network: `n` hidden neurons between 8 inputs and 2 outputs, `per` sparse links from each input and into each output.
    fn big_request(n: usize, per: usize, w: u32, h: u32) -> NetworkRequest {
        let fire = |t: usize, u: usize| (t * 7 + u).is_multiple_of(23);
        let (mut f1, mut t1, mut w1) = (Vec::new(), Vec::new(), Vec::new());
        for i in 0..8u32 {
            for k in 0..per as u32 {
                f1.push(i);
                t1.push((i * 7919 + k * 15_485) % n as u32);
                w1.push(0.3 + (k % 7) as f32 * 0.1);
            }
        }
        let f2: Vec<u32> = (0..n as u32).collect();
        let t2: Vec<u32> = (0..n as u32).map(|u| u % 2).collect();
        NetworkRequest {
            layers: vec![
                NetLayer {
                    units: 8,
                    frame: Some(frame(24, 8, |t, u| (t + u) % 3 == 0)),
                },
                NetLayer {
                    units: n,
                    frame: Some(frame(24, n, fire)),
                },
                NetLayer {
                    units: 2,
                    frame: Some(frame(24, 2, |t, u| (t + u) % 4 == 0)),
                },
            ],
            weights: vec![
                Some(sparse(8, n, f1, t1, w1)),
                Some(sparse(n, 2, f2, t2, vec![-0.5; n])),
            ],
            camera: NetCamera::default(),
            selected: None,
            phase: 0.0,
            width: w,
            height: h,
            scale: 1.0,
        }
    }

    fn request(w: u32, h: u32) -> NetworkRequest {
        NetworkRequest {
            layers: vec![
                NetLayer {
                    units: 8,
                    frame: Some(frame(24, 8, |t, u| (t + u) % 3 == 0)),
                },
                NetLayer {
                    units: 24,
                    frame: Some(frame(24, 24, |t, u| (t * 5 + u * 3) % 7 < 2)),
                },
                NetLayer {
                    units: 2,
                    frame: Some(frame(24, 2, |t, u| (t + u) % 4 == 0)),
                },
            ],
            weights: vec![
                Some(matrix(8, 24, |r, c| {
                    ((r * 31 + c * 17) % 23) as f32 / 11.0 - 1.0
                })),
                Some(matrix(24, 2, |r, c| {
                    ((r * 13 + c * 29) % 19) as f32 / 9.0 - 1.0
                })),
            ],
            camera: NetCamera::default(),
            selected: None,
            phase: 0.0,
            width: w,
            height: h,
            scale: 1.0,
        }
    }

    fn brightness(f: &Frame) -> u64 {
        f.pixels()
            .iter()
            .map(|p| p.r as u64 + p.g as u64 + p.b as u64)
            .sum()
    }

    #[test]
    fn layout_puts_layers_left_to_right_and_the_tallest_widest_column_in_the_middle() {
        let l = network_layout(&[8, 24, 2], &NetCamera::default(), (1000, 600), 1.0);
        assert_eq!(l.layers.len(), 3);
        assert!(l.layers[0].x < l.layers[1].x && l.layers[1].x < l.layers[2].x);
        let heights: Vec<f32> = l.layers.iter().map(|g| g.bottom - g.top).collect();
        assert!(
            heights[1] > heights[0] && heights[0] > heights[2],
            "{heights:?}"
        );
        assert_eq!(l.layers[1].nodes.len(), 24);
        assert!(
            l.layers[1].nodes.windows(2).all(|n| n[1].1 > n[0].1),
            "neurons go down the column"
        );
        assert_eq!(l.size, (1000.0, 600.0));
    }

    #[test]
    fn zoom_keeps_the_point_under_the_pointer_and_clamps() {
        let mut cam = NetCamera::default();
        let before = |c: &NetCamera| 0.5 + (0.30 - c.cx) * c.zoom;
        let at = before(&cam);
        cam.zoom_at(2.0, at, 0.5);
        assert!(
            (before(&cam) - at).abs() < 1e-5,
            "the world point stays where the pointer is"
        );
        assert!((cam.zoom - 2.0).abs() < 1e-6);
        cam.zoom_at(1000.0, 0.5, 0.5);
        assert_eq!(cam.zoom, NetCamera::MAX_ZOOM);
        cam.zoom_at(1e-6, 0.5, 0.5);
        assert_eq!(cam.zoom, NetCamera::MIN_ZOOM);
        cam.pan(10.0, 10.0);
        assert!((0.0..=1.0).contains(&cam.cx) && (0.0..=1.0).contains(&cam.cy));
    }

    #[test]
    fn dragging_moves_the_content_with_the_pointer() {
        let mut cam = NetCamera::default();
        let x = |c: &NetCamera| network_layout(&[8, 24, 2], c, (1000, 600), 1.0).layers[1].x;
        let before = x(&cam);
        cam.pan(0.1, 0.0);
        assert!((x(&cam) - (before + 0.1)).abs() < 1e-5);
    }

    #[test]
    fn hit_finds_the_nearest_neuron_within_the_radius() {
        let l = network_layout(&[8, 24, 2], &NetCamera::default(), (1000, 600), 1.0);
        let (x, y) = l.layers[1].nodes[10];
        assert_eq!(l.hit(x, y, 6.0), Some((1, 10)));
        assert_eq!(l.hit(x + 3.0 / 1000.0, y, 6.0), Some((1, 10)));
        assert_eq!(l.hit(0.5, 0.01, 6.0), None);
        assert_eq!(NetLayout::default().hit(0.5, 0.5, 100.0), None);
    }

    #[test]
    fn the_picture_has_light_in_it_and_sizes_match() {
        let (f, layout) = render_network(&request(900, 500));
        assert_eq!((f.width, f.height), (900, 500));
        assert_eq!(layout.layers.len(), 3);
        let empty = {
            let mut r = request(900, 500);
            r.weights = vec![None, None];
            r.layers.iter_mut().for_each(|l| l.frame = None);
            render_network(&r).0
        };
        let green = |f: &Frame| f.pixels().iter().filter(|p| p.g > 110).count();
        assert!(
            brightness(&f) > brightness(&empty) && green(&f) > green(&empty) + 500,
            "the weighted curves add light to the bare network"
        );
    }

    #[test]
    fn positive_weights_come_out_green_and_negative_ones_red() {
        let mut r = request(900, 500);
        r.weights = vec![
            Some(matrix(8, 24, |_, _| 1.0)),
            Some(matrix(24, 2, |_, _| 1.0)),
        ];
        let (green, _) = render_network(&r);
        r.weights = vec![
            Some(matrix(8, 24, |_, _| -1.0)),
            Some(matrix(24, 2, |_, _| -1.0)),
        ];
        let (red, _) = render_network(&r);
        let mean = |f: &Frame| {
            let n = f.pixels().len() as f64;
            (
                f.pixels().iter().map(|p| p.r as f64).sum::<f64>() / n,
                f.pixels().iter().map(|p| p.g as f64).sum::<f64>() / n,
            )
        };
        let (gr, gg) = mean(&green);
        let (rr, rg) = mean(&red);
        assert!(
            gg > gr,
            "green weights: more green than red ({gr:.1} {gg:.1})"
        );
        assert!(
            rr > rg,
            "red weights: more red than green ({rr:.1} {rg:.1})"
        );
    }

    #[test]
    fn selecting_a_neuron_dims_the_rest_and_lights_its_own_connections() {
        let plain = render_network(&request(900, 500)).0;
        let mut r = request(900, 500);
        r.selected = Some((1, 10));
        let (selected, layout) = render_network(&r);
        assert!(
            brightness(&selected) < brightness(&plain),
            "the unrelated curves fade"
        );
        // Along the selected neuron's own curves the picture is whiter than anywhere else it is dimmed
        let (x, y) = layout.layers[1].nodes[10];
        let (cx, cy) = ((x * 900.0) as usize, (y * 500.0) as usize);
        let around: Vec<&Pixel> = (0..9)
            .flat_map(|d| (0..9).map(move |e| (d, e)))
            .map(|(d, e)| &selected.pixels()[(cy + d - 4) * 900 + cx + e - 4])
            .collect();
        assert!(
            around.iter().any(|p| p.r > 200 && p.g > 200 && p.b > 200),
            "the selected neuron glows white"
        );
    }

    #[test]
    fn a_selection_outside_the_network_is_ignored() {
        let mut r = request(300, 200);
        r.selected = Some((9, 9));
        let a = render_network(&r).0;
        r.selected = None;
        assert_eq!(a, render_network(&r).0);
    }

    #[test]
    fn missing_pieces_and_odd_sizes_do_not_panic() {
        let mut r = request(1, 1);
        let _ = render_network(&r);
        for (w, h, s) in [(30, 20, 1.0), (200, 90, 2.0), (640, 360, 1.5)] {
            r = request(w, h);
            r.scale = s;
            r.weights = vec![None, Some(matrix(3, 3, |_, _| 1.0))]; // wrong shape
            r.layers[1].frame = Some(frame(5, 7, |_, _| true)); // wrong unit count
            r.camera = NetCamera {
                cx: 0.9,
                cy: 0.1,
                zoom: 8.0,
            };
            let (f, _) = render_network(&r);
            assert_eq!((f.width, f.height), (w, h));
        }
        let empty = NetworkRequest {
            layers: Vec::new(),
            weights: Vec::new(),
            ..request(100, 60)
        };
        let (f, layout) = render_network(&empty);
        assert_eq!(f.width, 100);
        assert!(layout.layers.is_empty());
        let one = NetworkRequest {
            layers: vec![NetLayer {
                units: 4,
                frame: None,
            }],
            weights: Vec::new(),
            ..request(100, 60)
        };
        let _ = render_network(&one);
    }

    #[test]
    fn the_phase_moves_the_beads() {
        let mut r = request(600, 360);
        let a = render_network(&r).0;
        r.phase = 1.37;
        let b = render_network(&r).0;
        assert_ne!(a, b);
    }

    #[test]
    fn a_small_network_is_drawn_connection_by_connection_and_a_big_one_as_blocks() {
        let (_, small) = render_network(&request(600, 360));
        assert!(!small.lod && small.layers[1].nodes.len() == 24);
        let (_, big) = render_network(&big_request(50_000, 40, 600, 360));
        assert!(big.lod, "50 000 neurons do not fit as curves");
        assert!(
            big.layers[1].nodes.is_empty() && big.layers[1].units == 50_000,
            "no node list for a big layer"
        );
        // Many links among few neurons: also too many curves
        let mut dense = request(600, 360);
        dense.layers[1].units = 2_500;
        dense.layers[1].frame = None;
        dense.weights = vec![
            Some(matrix(8, 2_500, |_, _| 0.5)),
            Some(matrix(2_500, 2, |_, _| 0.5)),
        ];
        assert!(
            render_network(&dense).1.lod,
            "20 000 + 5 000 links are over the limit of curves"
        );
        dense.layers[1].units = 100;
        dense.weights = vec![
            Some(matrix(8, 100, |_, _| 0.5)),
            Some(matrix(100, 2, |_, _| 0.5)),
        ];
        assert!(
            !render_network(&dense).1.lod,
            "800 + 200 links are drawn one by one"
        );
        // A hundred neurons in every layer: ten thousand connections, still each its own curve
        let mut hundred = request(600, 360);
        hundred.layers = vec![
            NetLayer {
                units: 100,
                frame: None,
            },
            NetLayer {
                units: 100,
                frame: None,
            },
            NetLayer {
                units: 10,
                frame: None,
            },
        ];
        hundred.weights = vec![
            Some(matrix(100, 100, |i, j| ((i * 7 + j * 3) % 11) as f32 - 5.0)),
            Some(matrix(100, 10, |i, j| ((i + j) % 5) as f32 - 2.0)),
        ];
        assert!(!render_network(&hundred).1.lod, "11 000 links fit");
    }

    #[test]
    fn in_a_big_layer_the_neuron_under_the_pointer_is_found_by_its_height() {
        let (_, l) = render_network(&big_request(50_000, 40, 600, 360));
        let g = &l.layers[1];
        let (fx, top, bottom) = (g.x, g.top, g.bottom);
        assert_eq!(
            l.hit(fx, top + (bottom - top) * 0.5, 6.0),
            Some((1, 25_000))
        );
        assert_eq!(l.hit(fx, top + 1e-7, 6.0), Some((1, 0)));
        assert_eq!(l.hit(fx, top - 0.05, 6.0), None, "above the column");
        assert_eq!(
            l.hit(fx + 0.2, (top + bottom) / 2.0, 6.0),
            None,
            "beside the column"
        );
    }

    #[test]
    fn ribbons_are_green_where_blocks_excite_and_red_where_they_inhibit() {
        let mean = |f: &Frame| {
            let n = f.pixels().len() as f64;
            (
                f.pixels().iter().map(|p| p.r as f64).sum::<f64>() / n,
                f.pixels().iter().map(|p| p.g as f64).sum::<f64>() / n,
            )
        };
        let mut r = big_request(20_000, 40, 700, 400);
        let n = 20_000usize;
        r.weights[1] = Some(sparse(
            n,
            2,
            (0..n as u32).collect(),
            (0..n as u32).map(|u| u % 2).collect(),
            vec![0.8; n],
        ));
        r.weights[0] = Some(sparse(
            8,
            n,
            (0..8u32 * 40).map(|k| k / 40).collect(),
            (0..8u32 * 40).map(|k| (k * 499) % n as u32).collect(),
            vec![0.8; 320],
        ));
        let (green, _) = mean(&render_network(&r).0);
        let (g_r, g_g) = mean(&render_network(&r).0);
        r.weights[1] = Some(sparse(
            n,
            2,
            (0..n as u32).collect(),
            (0..n as u32).map(|u| u % 2).collect(),
            vec![-0.8; n],
        ));
        r.weights[0] = Some(sparse(
            8,
            n,
            (0..8u32 * 40).map(|k| k / 40).collect(),
            (0..8u32 * 40).map(|k| (k * 499) % n as u32).collect(),
            vec![-0.8; 320],
        ));
        let (r_r, r_g) = mean(&render_network(&r).0);
        let _ = green;
        assert!(
            g_g > g_r,
            "excitatory: more green than red ({g_r:.1} {g_g:.1})"
        );
        assert!(
            r_r > r_g,
            "inhibitory: more red than green ({r_r:.1} {r_g:.1})"
        );
    }

    #[test]
    fn picking_a_neuron_in_a_big_layer_dims_the_ribbons_and_lights_its_links() {
        let plain = render_network(&big_request(30_000, 40, 700, 400)).0;
        let mut r = big_request(30_000, 40, 700, 400);
        r.selected = Some((1, 7_919));
        let (picked, layout) = render_network(&r);
        assert!(picked != plain);
        let g = &layout.layers[1];
        let y = g.top + (g.bottom - g.top) * (7_919.5 / 30_000.0);
        let (cx, cy) = ((g.x * 700.0) as usize, (y * 400.0) as usize);
        let bright = (0..9)
            .flat_map(|d| (0..9).map(move |e| (d, e)))
            .filter(|&(d, e)| cy + d >= 4 && cx + e >= 4)
            .any(|(d, e)| {
                let p = picked.pixels()[(cy + d - 4).min(399) * 700 + (cx + e - 4).min(699)];
                p.r > 200 && p.g > 200 && p.b > 200
            });
        assert!(bright, "the picked neuron glows white");
    }

    #[test]
    fn a_million_neurons_with_two_million_links_render_in_seconds() {
        let start = std::time::Instant::now();
        let (f, layout) = render_network(&big_request(1_000_000, 200, 900, 500));
        assert_eq!((f.width, f.height), (900, 500));
        assert!(layout.lod);
        assert!(start.elapsed().as_secs() < 20, "{:?}", start.elapsed());
    }

    #[test]
    fn zooming_in_reveals_individual_neurons_of_a_mid_sized_layer() {
        // 1 200 neurons: blocks when far, real neurons once they are 6 px apart
        let mut r = big_request(1_200, 30, 700, 400);
        let far = render_network(&r).0;
        r.camera = NetCamera {
            cx: 0.5,
            cy: 0.5,
            zoom: 8.0,
        };
        let near = render_network(&r).0;
        assert!(far != near);
    }
}
