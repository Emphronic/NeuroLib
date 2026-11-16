//! The training dashboard: ten panels drawn into one image on a work thread, in the layout of a
//! classic spiking-network monitor. Network graph, spike raster, accuracy, two weight maps, firing
//! rates, weight histogram, a PCA of the hidden state, and two text panels the view fills in.
//!
//! Like the other rasterizers it draws no text: titles and numbers are elements the view lays over
//! the image, placed from [`panel_rects`] (fractions of the image) and the insets below (logical px).
//! What the renderer had to compute that the text needs comes back in [`DashboardInfo`].

use std::sync::Arc;

use neo_snn::telemetry::SpikeFrame;

use crate::engine::canvas::{Canvas, Frame, Pixel};
use crate::engine::palette::Palette;

use super::plot::{ema, fill, line, polyline};

/// Gap between two panels, logical px (half on each side).
pub const GAP: f32 = 3.0;
/// Room for the panel title above the plot, logical px.
pub const TITLE_H: f32 = 26.0;
/// Padding inside a panel, logical px.
pub const PAD: f32 = 8.0;

/// Spikes kept for the PCA plot (the newest frames fill it first).
const MAX_PCA_POINTS: usize = 800;
const HISTOGRAM_BINS: usize = 24;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PanelKind {
    Network,
    Raster,
    Accuracy,
    HeatInputHidden,
    HeatHiddenOutput,
    Rates,
    WeightDist,
    Pca,
    Metrics,
    Run,
}

/// Where a panel is, in fractions of the image.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct PanelRect {
    pub kind: PanelKind,
    pub x0: f32,
    pub y0: f32,
    pub x1: f32,
    pub y1: f32,
}

/// Three rows: network, raster and accuracy; two weight maps, rates and histogram; PCA and the
/// two text panels.
pub fn panel_rects() -> [PanelRect; 10] {
    use PanelKind::*;
    const ROW: [f32; 4] = [0.0, 0.38, 0.70, 1.0];
    let r = |kind, x0, x1, row: usize| PanelRect {
        kind,
        x0,
        y0: ROW[row],
        x1,
        y1: ROW[row + 1],
    };
    [
        r(Network, 0.0, 0.34, 0),
        r(Raster, 0.34, 0.66, 0),
        r(Accuracy, 0.66, 1.0, 0),
        r(HeatInputHidden, 0.0, 0.25, 1),
        r(HeatHiddenOutput, 0.25, 0.5, 1),
        r(Rates, 0.5, 0.75, 1),
        r(WeightDist, 0.75, 1.0, 1),
        r(Pca, 0.0, 0.25, 2),
        r(Metrics, 0.25, 0.62, 2),
        r(Run, 0.62, 1.0, 2),
    ]
}

/// A weight matrix as the training sent it: `rows` inputs by `cols` outputs, row by row.
#[derive(Debug, Clone)]
pub struct WeightMatrix {
    pub rows: usize,
    pub cols: usize,
    pub values: Arc<Vec<f32>>,
}

impl WeightMatrix {
    fn at(&self, row: usize, col: usize) -> f32 {
        self.values
            .get(row * self.cols + col)
            .copied()
            .unwrap_or(0.0)
    }

    fn abs_max(&self) -> f32 {
        self.values
            .iter()
            .filter(|v| v.is_finite())
            .fold(0.0, |m, v| m.max(v.abs()))
    }
}

/// What to draw; everything is shared or copied, so the request can go to another thread.
#[derive(Debug, Clone)]
pub struct DashboardRequest {
    pub input: Option<Arc<SpikeFrame>>,
    pub hidden: Option<Arc<SpikeFrame>>,
    pub output: Option<Arc<SpikeFrame>>,
    pub w_input_hidden: Option<WeightMatrix>,
    pub w_hidden_output: Option<WeightMatrix>,
    /// Hidden-layer frames for the state-space plot, oldest first.
    pub pca_frames: Vec<Arc<SpikeFrame>>,
    pub accuracy: Vec<f32>,
    pub width: u32,
    pub height: u32,
    /// Device pixels per logical pixel: the insets above are logical.
    pub scale: f32,
    pub palette: Palette,
}

/// Numbers the renderer computed that the view's text needs.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct DashboardInfo {
    /// Share of the hidden state's variance on the two axes of the PCA plot.
    pub pca_variance: Option<(f32, f32)>,
    /// Largest |weight| over both matrices: the scale of the weight maps and the histogram.
    pub weight_abs_max: f32,
    /// Highest firing rate of a hidden neuron in the frame (share of steps it fired).
    pub rate_max: f32,
}

// ─── geometry ───────────────────────────────────────────────────────────

#[derive(Debug, Clone, Copy)]
struct Area {
    x0: f32,
    y0: f32,
    x1: f32,
    y1: f32,
}

impl Area {
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

fn mix(a: Pixel, b: Pixel, t: f32) -> Pixel {
    let m = |x: u8, y: u8| (x as f32 + (y as f32 - x as f32) * t.clamp(0.0, 1.0)).round() as u8;
    Pixel::rgb(m(a.r, b.r), m(a.g, b.g), m(a.b, b.b))
}

fn fill_area(c: &mut Canvas, a: &Area, color: Pixel) {
    c.rect(a.x0, a.y0, a.x1, a.y1, color, 1.0);
}

fn outline(c: &mut Canvas, a: &Area, color: Pixel) {
    c.rect(a.x0, a.y0, a.x1, a.y0 + 1.0, color, 1.0);
    c.rect(a.x0, a.y1 - 1.0, a.x1, a.y1, color, 1.0);
    c.rect(a.x0, a.y0, a.x0 + 1.0, a.y1, color, 1.0);
    c.rect(a.x1 - 1.0, a.y0, a.x1, a.y1, color, 1.0);
}

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
            let cover = (r + 0.5 - d).clamp(0.0, 1.0);
            if cover > 0.0 {
                c.blend(x, y, color, cover * alpha);
            }
        }
    }
}

/// Layer colours: input, hidden, output. The view's legend uses the same.
pub fn layer_color(p: &Palette, layer: usize) -> Pixel {
    match layer {
        0 => p.channel(0),
        1 => p.channel(1),
        _ => p.channel(4),
    }
}

// ─── the image ──────────────────────────────────────────────────────────

pub fn render_dashboard(req: &DashboardRequest) -> (Frame, DashboardInfo) {
    let (w, h) = (req.width.max(1), req.height.max(1));
    let mut frame = Frame::new(w, h);
    let p = &req.palette;
    let s = req.scale.max(0.5);
    let mut c = frame.canvas();
    fill(&mut c, p.background);

    let mut info = DashboardInfo::default();
    info.weight_abs_max = [&req.w_input_hidden, &req.w_hidden_output]
        .into_iter()
        .flatten()
        .map(WeightMatrix::abs_max)
        .fold(0.0, f32::max);
    let panel_fill = mix(p.background, p.baseline, 0.25);

    for r in panel_rects() {
        let outer = Area {
            x0: r.x0 * w as f32,
            y0: r.y0 * h as f32,
            x1: r.x1 * w as f32,
            y1: r.y1 * h as f32,
        }
        .inset(GAP * s / 2.0, GAP * s / 2.0, GAP * s / 2.0, GAP * s / 2.0);
        fill_area(&mut c, &outer, panel_fill);
        outline(&mut c, &outer, p.baseline);
        let area = outer.inset(PAD * s, TITLE_H * s, PAD * s, PAD * s);
        if area.w() < 8.0 || area.h() < 8.0 {
            continue;
        }
        match r.kind {
            PanelKind::Network => draw_network(&mut c, p, &area, s, req),
            PanelKind::Raster => draw_raster(&mut c, p, &area, s, req),
            PanelKind::Accuracy => draw_accuracy(&mut c, p, &area, &req.accuracy),
            PanelKind::HeatInputHidden => draw_heat(
                &mut c,
                p,
                &area,
                req.w_input_hidden.as_ref(),
                false,
                info.weight_abs_max,
            ),
            PanelKind::HeatHiddenOutput => draw_heat(
                &mut c,
                p,
                &area,
                req.w_hidden_output.as_ref(),
                true,
                info.weight_abs_max,
            ),
            PanelKind::Rates => {
                info.rate_max = draw_rates(&mut c, p, &area, s, req.hidden.as_deref())
            }
            PanelKind::WeightDist => draw_weight_dist(&mut c, p, &area, req, info.weight_abs_max),
            PanelKind::Pca => info.pca_variance = draw_pca(&mut c, p, &area, s, &req.pca_frames),
            PanelKind::Metrics | PanelKind::Run => {}
        }
    }
    (frame, info)
}

// ─── network graph ──────────────────────────────────────────────────────

fn draw_network(c: &mut Canvas, p: &Palette, area: &Area, s: f32, req: &DashboardRequest) {
    // The names under the columns are the view's; keep that strip free
    let area = area.inset(0.0, 2.0 * s, 0.0, 20.0 * s);
    let layers = [
        req.input.as_deref(),
        req.hidden.as_deref(),
        req.output.as_deref(),
    ];
    let xs = [
        area.x0 + area.w() * 0.08,
        area.x0 + area.w() * 0.50,
        area.x0 + area.w() * 0.92,
    ];
    let node_y = |i: usize, n: usize| area.y0 + (i as f32 + 0.5) * area.h() / n.max(1) as f32;

    // Connections first, so the neurons sit on top: blue for a positive weight, rose for a negative one
    let edges = [
        (0usize, 1usize, &req.w_input_hidden),
        (1, 2, &req.w_hidden_output),
    ];
    for (from, to, matrix) in edges {
        let (Some(a), Some(b), Some(m)) = (layers[from], layers[to], matrix.as_ref()) else {
            continue;
        };
        if m.rows != a.units || m.cols != b.units {
            continue;
        }
        let wmax = m.abs_max().max(f32::EPSILON);
        for i in 0..m.rows {
            for j in 0..m.cols {
                let v = m.at(i, j);
                let alpha = (v.abs() / wmax).powf(0.9) * 0.45;
                if alpha < 0.04 || !v.is_finite() {
                    continue;
                }
                let color = if v >= 0.0 { p.channel(0) } else { p.channel(4) };
                line(
                    c,
                    (xs[from], node_y(i, m.rows)),
                    (xs[to], node_y(j, m.cols)),
                    color,
                    alpha,
                );
            }
        }
    }

    // Neurons: a quiet one is its layer's colour, a busy one warms to the spike colour and grows;
    // one that fired on the last step gets a halo
    for (l, frame) in layers.iter().enumerate() {
        let Some(f) = frame else { continue };
        if f.is_empty() {
            continue;
        }
        let base = mix(layer_color(p, l), p.background, 0.35);
        let r0 = (area.h() / (f.units as f32 * 2.4)).clamp(2.0 * s, 6.0 * s);
        for u in 0..f.units {
            let rate = f.unit_rate(u).clamp(0.0, 1.0);
            let (x, y) = (xs[l], node_y(u, f.units));
            if f.spiked(f.steps - 1, u) {
                disc(c, x, y, r0 + 3.0 * s, p.marker, 0.28);
            }
            disc(
                c,
                x,
                y,
                r0 * (0.8 + 0.5 * rate),
                mix(base, p.marker, rate.sqrt()),
                1.0,
            );
        }
    }
}

// ─── spike raster ───────────────────────────────────────────────────────

fn draw_raster(c: &mut Canvas, p: &Palette, area: &Area, s: f32, req: &DashboardRequest) {
    let layers: Vec<(usize, &SpikeFrame)> = [
        req.input.as_deref(),
        req.hidden.as_deref(),
        req.output.as_deref(),
    ]
    .into_iter()
    .enumerate()
    .filter_map(|(l, f)| f.filter(|f| !f.is_empty()).map(|f| (l, f)))
    .collect();
    if layers.is_empty() {
        return;
    }
    // A layer's share of the height follows its neurons, but a two-neuron layer still gets a visible band
    let gap = 4.0 * s;
    let weights: Vec<f32> = layers.iter().map(|(_, f)| f.units.max(6) as f32).collect();
    let total: f32 = weights.iter().sum();
    let usable = (area.h() - gap * (layers.len() as f32 - 1.0)).max(1.0);
    let mut y = area.y0;
    for ((l, f), wt) in layers.iter().zip(&weights) {
        let h = usable * wt / total;
        let region = Area {
            x0: area.x0,
            y0: y,
            x1: area.x1,
            y1: y + h,
        };
        y += h + gap;
        fill_area(c, &region, p.background);

        let (rw, rh) = (region.w() as usize, region.h() as usize);
        if rw == 0 || rh == 0 {
            continue;
        }
        let rows = f.units.min(rh).max(1);
        let cols = f.steps.min(rw).max(1);
        let grid = f.bin_density(rows, cols);
        if grid.len() < rows * cols {
            continue;
        }
        let gaps = rw / cols >= 4 && rh / rows >= 4;
        let color = layer_color(p, *l);
        for py in 0..rh {
            let r = py * rows / rh;
            let row_gap = gaps && (py + 1) * rows / rh != r;
            for px in 0..rw {
                let col = px * cols / rw;
                let col_gap = gaps && (px + 1) * cols / rw != col;
                let density = grid[r * cols + col];
                if density > 0.0 && !row_gap && !col_gap {
                    c.set(
                        region.x0 as usize + px,
                        region.y0 as usize + py,
                        mix(p.background, color, density.sqrt().max(0.45)),
                    );
                }
            }
        }
    }
}

// ─── accuracy ───────────────────────────────────────────────────────────

fn draw_accuracy(c: &mut Canvas, p: &Palette, area: &Area, values: &[f32]) {
    fill_area(c, area, p.background);
    // 0, 25, 50, 75 and 100 %: the half and the full lines a little stronger
    for q in 0..=4 {
        let y = area.y1 - 1.0 - (area.h() - 2.0) * q as f32 / 4.0;
        let color = if q == 2 || q == 4 { p.baseline } else { p.grid };
        c.rect(area.x0, y, area.x1, y + 1.0, color, 1.0);
    }
    plot_series(c, area, values, 0.0, 1.0, p.channel(1));
}

/// A series over the whole width of `area`, raw values faint under an exponential average.
fn plot_series(c: &mut Canvas, area: &Area, values: &[f32], lo: f32, hi: f32, color: Pixel) {
    if values.is_empty() || area.w() < 2.0 {
        return;
    }
    let width = area.w() as usize;
    let y_of = |v: f32| {
        let f = ((v - lo) / (hi - lo).max(f32::EPSILON)).clamp(0.0, 1.0);
        area.y1 - 2.0 - f * (area.h() - 3.0).max(0.0)
    };
    let to_points = |series: &[f32]| -> Vec<(f32, f32)> {
        let n = series.len();
        series
            .iter()
            .enumerate()
            .filter(|(_, v)| v.is_finite())
            .map(|(i, &v)| {
                (
                    area.x0 + i as f32 * (area.w() - 1.0) / (n - 1).max(1) as f32,
                    y_of(v),
                )
            })
            .collect()
    };
    polyline(c, &to_points(&thin(values, width)), color, 0.30);
    let smooth = to_points(&thin(&ema(values), width));
    polyline(c, &smooth, color, 1.0);
    if let Some(&(x, y)) = smooth.last() {
        disc(c, x, y, 3.0, color, 1.0);
    }
}

/// At most `max` values: runs of neighbours are averaged, so nothing is dropped.
fn thin(values: &[f32], max: usize) -> Vec<f32> {
    let n = values.len();
    if n <= max.max(1) {
        return values.to_vec();
    }
    let max = max.max(1);
    (0..max)
        .map(|i| {
            let (a, b) = (i * n / max, ((i + 1) * n / max).max(i * n / max + 1));
            let run = &values[a..b.min(n)];
            run.iter().sum::<f32>() / run.len().max(1) as f32
        })
        .collect()
}

// ─── weight maps ────────────────────────────────────────────────────────

/// A weight matrix as a grid: blue for positive, rose for negative, the strength as intensity.
/// `transpose` puts the matrix's columns down, so that both maps share the hidden layer's axis.
fn draw_heat(
    c: &mut Canvas,
    p: &Palette,
    area: &Area,
    matrix: Option<&WeightMatrix>,
    transpose: bool,
    wmax: f32,
) {
    fill_area(c, area, p.background);
    let Some(m) = matrix.filter(|m| m.rows > 0 && m.cols > 0) else {
        return;
    };
    let (rows, cols) = if transpose {
        (m.cols, m.rows)
    } else {
        (m.rows, m.cols)
    };
    let (aw, ah) = (area.w() as usize, area.h() as usize);
    if aw == 0 || ah == 0 {
        return;
    }
    let wmax = wmax.max(f32::EPSILON);
    let gaps = aw / cols >= 4 && ah / rows >= 4;
    for py in 0..ah {
        let r = py * rows / ah;
        let row_gap = gaps && (py + 1) * rows / ah != r;
        for px in 0..aw {
            let col = px * cols / aw;
            let col_gap = gaps && (px + 1) * cols / aw != col;
            if row_gap || col_gap {
                continue;
            }
            let v = if transpose {
                m.at(col, r)
            } else {
                m.at(r, col)
            };
            if !v.is_finite() {
                continue;
            }
            let base = if v >= 0.0 { p.channel(0) } else { p.channel(4) };
            c.set(
                area.x0 as usize + px,
                area.y0 as usize + py,
                mix(p.background, base, (v.abs() / wmax).powf(0.7)),
            );
        }
    }
}

// ─── firing rates ───────────────────────────────────────────────────────

/// One bar per hidden neuron (share of steps it fired) and a dashed line at the mean. Returns the highest rate.
fn draw_rates(c: &mut Canvas, p: &Palette, area: &Area, s: f32, frame: Option<&SpikeFrame>) -> f32 {
    fill_area(c, area, p.background);
    c.rect(area.x0, area.y1 - 1.0, area.x1, area.y1, p.baseline, 1.0);
    let Some(f) = frame.filter(|f| !f.is_empty()) else {
        return 0.0;
    };
    let rates: Vec<f32> = (0..f.units).map(|u| f.unit_rate(u)).collect();
    let max = rates.iter().copied().fold(0.0, f32::max);
    // Scaled to the busiest neuron (never to less than 10 %), so that a quiet layer still reads
    let top = (max * 1.1).clamp(0.1, 1.0);
    let n = rates.len();
    let bar_w = area.w() / n as f32;
    let gap = if bar_w >= 4.0 { 1.0 } else { 0.0 };
    for (u, &rate) in rates.iter().enumerate() {
        let x0 = area.x0 + u as f32 * bar_w;
        let y0 = area.y1 - 1.0 - (area.h() - 1.0) * (rate / top).clamp(0.0, 1.0);
        c.rect(x0, y0, x0 + bar_w - gap, area.y1 - 1.0, p.channel(4), 0.9);
    }
    let mean = rates.iter().sum::<f32>() / n as f32;
    let ym = area.y1 - 1.0 - (area.h() - 1.0) * (mean / top).clamp(0.0, 1.0);
    let dash = 5.0 * s;
    let mut x = area.x0;
    while x < area.x1 {
        c.rect(x, ym, (x + dash).min(area.x1), ym + 1.0, p.marker, 0.85);
        x += dash * 2.0;
    }
    max
}

// ─── weight histogram ───────────────────────────────────────────────────

fn draw_weight_dist(c: &mut Canvas, p: &Palette, area: &Area, req: &DashboardRequest, wmax: f32) {
    fill_area(c, area, p.background);
    c.rect(area.x0, area.y1 - 1.0, area.x1, area.y1, p.baseline, 1.0);
    if wmax <= 0.0 {
        return;
    }
    let mut counts = [0u32; HISTOGRAM_BINS];
    for m in [&req.w_input_hidden, &req.w_hidden_output]
        .into_iter()
        .flatten()
    {
        for &v in m.values.iter().filter(|v| v.is_finite()) {
            let f = (v / wmax + 1.0) * 0.5;
            counts[((f * HISTOGRAM_BINS as f32) as usize).min(HISTOGRAM_BINS - 1)] += 1;
        }
    }
    let peak = counts.iter().copied().max().unwrap_or(0).max(1) as f32;
    let bar_w = area.w() / HISTOGRAM_BINS as f32;
    let gap = if bar_w >= 4.0 { 1.0 } else { 0.0 };
    for (i, &n) in counts.iter().enumerate() {
        let x0 = area.x0 + i as f32 * bar_w;
        let y0 = area.y1 - 1.0 - (area.h() - 1.0) * n as f32 / peak;
        c.rect(x0, y0, x0 + bar_w - gap, area.y1 - 1.0, p.channel(5), 0.9);
    }
    // Zero in the middle
    let xz = area.x0 + area.w() * 0.5;
    c.rect(xz, area.y0, xz + 1.0, area.y1, p.marker, 0.35);
}

// ─── state space (PCA) ──────────────────────────────────────────────────

/// The hidden layer's state over time: every time step of a frame is a point (which neurons fired),
/// projected on the two directions of greatest variance. Older frames are dimmer. Returns the share of
/// variance on the two axes, or `None` when there is nothing to project yet.
fn draw_pca(
    c: &mut Canvas,
    p: &Palette,
    area: &Area,
    s: f32,
    frames: &[Arc<SpikeFrame>],
) -> Option<(f32, f32)> {
    fill_area(c, area, p.background);
    let (cx, cy) = (area.x0 + area.w() / 2.0, area.y0 + area.h() / 2.0);
    c.rect(area.x0, cy, area.x1, cy + 1.0, p.grid, 1.0);
    c.rect(cx, area.y0, cx + 1.0, area.y1, p.grid, 1.0);
    let pca = pca2(frames)?;
    let (mx, my) = pca
        .points
        .iter()
        .fold((f32::EPSILON, f32::EPSILON), |(a, b), &(x, y, _)| {
            (a.max(x.abs()), b.max(y.abs()))
        });
    let (kx, ky) = (area.w() * 0.46 / mx, area.h() * 0.46 / my);
    for &(x, y, age) in &pca.points {
        let color = mix(p.channel(2), p.channel(4), age);
        disc(
            c,
            cx + x * kx,
            cy - y * ky,
            2.2 * s,
            color,
            0.30 + 0.65 * age,
        );
    }
    Some(pca.variance)
}

struct Pca2 {
    /// `(pc1, pc2, age)`, age 0 for the oldest frame and 1 for the newest.
    points: Vec<(f32, f32, f32)>,
    variance: (f32, f32),
}

fn pca2(frames: &[Arc<SpikeFrame>]) -> Option<Pca2> {
    let units = frames.iter().find(|f| !f.is_empty())?.units;
    let dims = units.min(64);
    if dims < 2 {
        return None;
    }
    let pick: Vec<usize> = (0..dims).map(|k| k * units / dims).collect();
    let usable: Vec<&Arc<SpikeFrame>> = frames
        .iter()
        .filter(|f| f.units == units && !f.is_empty())
        .collect();

    // Newest frames first, until the budget is spent
    let mut rows: Vec<(Vec<f64>, f32)> = Vec::new();
    for (i, f) in usable.iter().enumerate().rev() {
        if !rows.is_empty() && rows.len() + f.steps > MAX_PCA_POINTS {
            break;
        }
        let age = if usable.len() > 1 {
            i as f32 / (usable.len() - 1) as f32
        } else {
            1.0
        };
        for t in 0..f.steps {
            rows.push((
                pick.iter()
                    .map(|&u| f64::from(f.spiked(t, u) as u8))
                    .collect(),
                age,
            ));
        }
    }
    let n = rows.len();
    if n < 3 {
        return None;
    }

    let mut mean = vec![0.0f64; dims];
    for (v, _) in &rows {
        for (m, x) in mean.iter_mut().zip(v) {
            *m += x / n as f64;
        }
    }
    let mut cov = vec![0.0f64; dims * dims];
    for (v, _) in &rows {
        let x: Vec<f64> = v.iter().zip(&mean).map(|(a, m)| a - m).collect();
        for i in 0..dims {
            for j in i..dims {
                cov[i * dims + j] += x[i] * x[j] / (n as f64 - 1.0);
            }
        }
    }
    for i in 0..dims {
        for j in 0..i {
            cov[i * dims + j] = cov[j * dims + i];
        }
    }
    let trace: f64 = (0..dims).map(|i| cov[i * dims + i]).sum();
    if trace < 1e-12 {
        return None;
    }

    let (l1, v1) = top_eigen(&cov, dims, 0)?;
    for i in 0..dims {
        for j in 0..dims {
            cov[i * dims + j] -= l1 * v1[i] * v1[j];
        }
    }
    // A second axis that does not exist (the state moves along a line) is a flat axis
    let (l2, v2) = top_eigen(&cov, dims, 1).unwrap_or((0.0, vec![0.0; dims]));

    let points = rows
        .iter()
        .map(|(v, age)| {
            let (mut a, mut b) = (0.0, 0.0);
            for k in 0..dims {
                let x = v[k] - mean[k];
                a += x * v1[k];
                b += x * v2[k];
            }
            (a as f32, b as f32, *age)
        })
        .collect();
    Some(Pca2 {
        points,
        variance: ((l1 / trace) as f32, (l2.max(0.0) / trace) as f32),
    })
}

/// The largest eigenvalue of a symmetric positive semi-definite `d × d` matrix and its unit vector,
/// by power iteration. `None` for a null matrix.
fn top_eigen(m: &[f64], d: usize, seed: usize) -> Option<(f64, Vec<f64>)> {
    let mut v: Vec<f64> = (0..d)
        .map(|k| 1.0 + ((k * 7 + seed * 13) % 11) as f64 * 0.1)
        .collect();
    let norm = v.iter().map(|x| x * x).sum::<f64>().sqrt();
    v.iter_mut().for_each(|x| *x /= norm);
    let mut lambda = 0.0;
    for _ in 0..80 {
        let w: Vec<f64> = (0..d)
            .map(|i| (0..d).map(|j| m[i * d + j] * v[j]).sum())
            .collect();
        let norm = w.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm < 1e-12 {
            return None;
        }
        lambda = norm;
        v = w.into_iter().map(|x| x / norm).collect();
    }
    Some((lambda, v))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn pal() -> Palette {
        Palette::DARK
    }

    fn frame(steps: usize, units: usize, f: impl Fn(usize, usize) -> bool) -> Arc<SpikeFrame> {
        Arc::new(SpikeFrame::from_fn(steps, units, f))
    }

    fn matrix(rows: usize, cols: usize, f: impl Fn(usize, usize) -> f32) -> WeightMatrix {
        WeightMatrix {
            rows,
            cols,
            values: Arc::new((0..rows * cols).map(|i| f(i / cols, i % cols)).collect()),
        }
    }

    fn request(width: u32, height: u32) -> DashboardRequest {
        DashboardRequest {
            input: None,
            hidden: None,
            output: None,
            w_input_hidden: None,
            w_hidden_output: None,
            pca_frames: Vec::new(),
            accuracy: Vec::new(),
            width,
            height,
            scale: 1.0,
            palette: pal(),
        }
    }

    fn full_request(width: u32, height: u32) -> DashboardRequest {
        let hidden = frame(24, 24, |t, u| (t * 5 + u * 3) % 7 < 2);
        let mut req = request(width, height);
        req.input = Some(frame(24, 8, |t, u| (t + u) % 3 == 0));
        req.hidden = Some(hidden.clone());
        req.output = Some(frame(24, 2, |t, u| (t + u) % 4 == 0));
        req.w_input_hidden = Some(matrix(8, 24, |r, c| {
            ((r * 31 + c * 17) % 23) as f32 / 11.0 - 1.0
        }));
        req.w_hidden_output = Some(matrix(24, 2, |r, c| {
            ((r * 13 + c * 29) % 19) as f32 / 9.0 - 1.0
        }));
        req.pca_frames = (0..6)
            .map(|k| frame(24, 24, move |t, u| (t * 5 + u * 3 + k) % 7 < 2))
            .collect();
        req.accuracy = (0..200)
            .map(|i| 0.5 + 0.45 * (1.0 - (-(i as f32) / 60.0).exp()))
            .collect();
        req
    }

    fn count(f: &Frame, color: Pixel) -> usize {
        f.pixels().iter().filter(|&&p| p == color).count()
    }

    #[test]
    fn panels_tile_the_image() {
        let rects = panel_rects();
        // The three rows each cover the full width, and the rows cover the full height
        for (row, widths) in [
            (0usize, [0.34, 0.32, 0.34].as_slice()),
            (1, &[0.25; 4]),
            (2, &[0.25, 0.37, 0.38]),
        ] {
            let in_row: Vec<&PanelRect> = rects
                .iter()
                .filter(|r| (r.y0 - [0.0, 0.38, 0.70][row]).abs() < 1e-6)
                .collect();
            assert_eq!(in_row.len(), widths.len());
            let total: f32 = in_row.iter().map(|r| r.x1 - r.x0).sum();
            assert!((total - 1.0).abs() < 1e-5, "row {row}: {total}");
        }
        assert!(
            rects
                .iter()
                .all(|r| r.x1 > r.x0 && r.y1 > r.y0 && r.y1 <= 1.0)
        );
    }

    #[test]
    fn nothing_yet_draws_only_the_panels() {
        let (f, info) = render_dashboard(&request(900, 600));
        assert_eq!((f.width, f.height), (900, 600));
        assert_eq!(info, DashboardInfo::default());
        let p = pal();
        for ch in [p.channel(0), p.channel(1), p.channel(4), p.marker] {
            assert_eq!(count(&f, ch), 0, "no data, no coloured pixels");
        }
    }

    #[test]
    fn a_full_request_draws_every_layer_the_accuracy_and_the_weights() {
        let (f, info) = render_dashboard(&full_request(1200, 700));
        let p = pal();
        assert!(
            count(&f, p.channel(1)) > 50,
            "hidden raster and accuracy are green"
        );
        assert!(
            count(&f, mix(p.background, p.channel(0), 1.0)) > 20,
            "input raster is blue"
        );
        assert!(
            count(&f, p.channel(5)) > 0 || f.pixels().iter().any(|&q| q.b > 200 && q.g > 100),
            "histogram bars"
        );
        assert!(info.weight_abs_max > 0.5);
        assert!(info.rate_max > 0.0 && info.rate_max <= 1.0);
        let (v1, v2) = info.pca_variance.expect("PCA over six frames");
        assert!(
            v1 > 0.0 && v1 <= 1.0 && v2 >= 0.0 && v1 + v2 <= 1.0001,
            "{v1} {v2}"
        );
        assert!(v1 >= v2, "the first axis carries the most variance");
    }

    #[test]
    fn tiny_and_odd_sizes_do_not_panic() {
        for (w, h, s) in [
            (1, 1, 1.0),
            (30, 20, 1.0),
            (200, 90, 2.0),
            (4000, 2000, 1.5),
        ] {
            let mut req = full_request(w, h);
            req.scale = s;
            let (f, _) = render_dashboard(&req);
            assert_eq!((f.width, f.height), (w, h));
        }
    }

    #[test]
    fn a_missing_layer_leaves_the_rest_drawn() {
        let mut req = full_request(900, 600);
        req.input = None;
        req.w_input_hidden = None;
        let (f, _) = render_dashboard(&req);
        assert!(count(&f, pal().channel(1)) > 50);
    }

    #[test]
    fn weights_of_the_wrong_shape_are_ignored_by_the_graph() {
        let mut req = full_request(900, 600);
        req.w_input_hidden = Some(matrix(3, 3, |_, _| 1.0));
        let _ = render_dashboard(&req); // must not index out of range
    }

    #[test]
    fn thin_averages_runs_and_keeps_short_series() {
        assert_eq!(thin(&[1.0, 2.0, 3.0], 10), vec![1.0, 2.0, 3.0]);
        let t = thin(&[1.0, 3.0, 5.0, 7.0], 2);
        assert_eq!(t, vec![2.0, 6.0]);
        assert_eq!(
            thin(&(0..1000).map(|i| i as f32).collect::<Vec<_>>(), 7).len(),
            7
        );
    }

    #[test]
    fn pca_finds_the_direction_the_state_moves_along() {
        // Two groups of neurons that fire alternately: one axis explains nearly everything
        let f = frame(40, 10, |t, u| (t % 2 == 0) == (u < 5));
        let pca = pca2(&[f]).expect("pca");
        assert!(pca.variance.0 > 0.95, "{:?}", pca.variance);
        assert!(pca.variance.1 < 0.05);
        // The points land on two clusters on the first axis
        let xs: Vec<f32> = pca.points.iter().map(|p| p.0).collect();
        assert!(xs.iter().any(|&x| x > 1.0) && xs.iter().any(|&x| x < -1.0));
    }

    #[test]
    fn pca_of_a_silent_or_empty_layer_is_none() {
        assert!(pca2(&[]).is_none());
        assert!(pca2(&[frame(10, 8, |_, _| false)]).is_none());
        assert!(
            pca2(&[frame(10, 1, |t, _| t % 2 == 0)]).is_none(),
            "one neuron has no plane"
        );
        assert!(
            pca2(&[frame(2, 8, |t, u| t == u)]).is_none(),
            "two points are not enough"
        );
    }

    #[test]
    fn heat_map_colours_follow_the_sign() {
        let p = pal();
        let mut req = request(1000, 700);
        req.w_input_hidden = Some(matrix(
            2,
            2,
            |r, c| if (r + c) % 2 == 0 { 1.0 } else { -1.0 },
        ));
        let (f, _) = render_dashboard(&req);
        assert!(count(&f, p.channel(0)) > 100, "positive weights are blue");
        assert!(count(&f, p.channel(4)) > 100, "negative weights are rose");
    }
}
