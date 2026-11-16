//! Rasterizers of the dashboard: the three metric curves and the spike raster, drawn into a
//! [`Frame`] on a work thread. Text (axis values, labels) is drawn by the view as elements over
//! the image; the layouts returned here say where each thing is, in fractions of the image.

use neo_snn::telemetry::SpikeFrame;

use crate::engine::canvas::{Canvas, Frame, Pixel};
use crate::engine::palette::Palette;

// ─── curves ─────────────────────────────────────────────────────────────

/// What to draw: the series are copied, so the request can go to another thread.
#[derive(Debug, Clone)]
pub struct CurveRequest {
    pub loss: Vec<f32>,
    pub accuracy: Vec<f32>,
    pub grad: Vec<f32>,
    pub width: u32,
    pub height: u32,
    pub palette: Palette,
}

/// One stacked plot of the curves image, in fractions of its height.
#[derive(Debug, Clone, PartialEq)]
pub struct CurvePanel {
    pub name: &'static str,
    pub y0: f32,
    pub y1: f32,
    /// Value range drawn (in log10 when `log`).
    pub lo: f32,
    pub hi: f32,
    pub log: bool,
    pub last: Option<f32>,
    pub points: usize,
}

pub(super) const GAP: f32 = 6.0;

/// Draws loss, accuracy and gradient norm as three stacked plots sharing the time axis.
///
/// Each curve is its raw values, faint, under a smoothed line (exponential average), so the
/// trend reads through the batch-to-batch noise. Accuracy is on a fixed `0..1` scale; the others
/// autoscale, in log10 when their values span more than two decades (a gradient norm does).
pub fn render_curves(req: &CurveRequest) -> (Frame, Vec<CurvePanel>) {
    let (w, h) = (req.width.max(1), req.height.max(1));
    let mut frame = Frame::new(w, h);
    let p = &req.palette;
    let mut canvas = frame.canvas();
    fill(&mut canvas, p.background);

    let series = [
        ("Loss", &req.loss, p.channel(0), Scale::Auto),
        ("Accuracy", &req.accuracy, p.channel(1), Scale::Unit),
        ("Gradient norm", &req.grad, p.channel(2), Scale::Auto),
    ];
    let n = series.len();
    let gap = GAP / h as f32;
    let each = (1.0 - gap * (n as f32 - 1.0)) / n as f32;
    let mut panels = Vec::with_capacity(n);
    for (i, (name, values, color, scale)) in series.into_iter().enumerate() {
        let y0 = i as f32 * (each + gap);
        let y1 = y0 + each;
        let (top, bottom) = (y0 * h as f32, y1 * h as f32);
        let (lo, hi, log) = scale.resolve(values);
        draw_panel_frame(&mut canvas, p, top, bottom);
        draw_series(&mut canvas, values, top, bottom, lo, hi, log, color);
        panels.push(CurvePanel {
            name,
            y0,
            y1,
            lo,
            hi,
            log,
            last: values.last().copied(),
            points: values.len(),
        });
    }
    (frame, panels)
}

#[derive(Clone, Copy)]
enum Scale {
    Auto,
    Unit,
}

impl Scale {
    /// `(lo, hi, log)` for these values.
    fn resolve(self, values: &[f32]) -> (f32, f32, bool) {
        match self {
            Scale::Unit => (0.0, 1.0, false),
            Scale::Auto => {
                let finite = || values.iter().copied().filter(|v| v.is_finite());
                let (mut lo, mut hi) = (f32::INFINITY, f32::NEG_INFINITY);
                for v in finite() {
                    lo = lo.min(v);
                    hi = hi.max(v);
                }
                if !lo.is_finite() {
                    return (0.0, 1.0, false);
                }
                if lo > 0.0 && hi / lo > 100.0 {
                    return (lo.log10().floor(), hi.log10().ceil(), true);
                }
                if (hi - lo).abs() < f32::EPSILON {
                    return (lo - 0.5, hi + 0.5, false);
                }
                let pad = (hi - lo) * 0.05;
                (lo - pad, hi + pad, false)
            }
        }
    }
}

pub(super) fn fill(c: &mut Canvas, color: Pixel) {
    for p in c.pixels.iter_mut() {
        *p = color;
    }
}

pub(super) fn draw_panel_frame(c: &mut Canvas, p: &Palette, top: f32, bottom: f32) {
    let (t, b) = (
        top.round() as usize,
        (bottom.round() as usize).min(c.height),
    );
    // Horizontal grid at quarters, a baseline at the bottom
    for q in 1..4 {
        let y = t + (b.saturating_sub(t)) * q / 4;
        for x in 0..c.width {
            c.blend(x, y, p.grid, 1.0);
        }
    }
    for x in 0..c.width {
        c.set(x, b.saturating_sub(1), p.baseline);
    }
}

pub(super) fn map_y(v: f32, lo: f32, hi: f32, log: bool, top: f32, bottom: f32) -> f32 {
    let v = if log {
        v.max(f32::MIN_POSITIVE).log10()
    } else {
        v
    };
    let f = ((v - lo) / (hi - lo).max(f32::EPSILON)).clamp(0.0, 1.0);
    // One pixel of margin so a line on the border is not cut
    let (t, b) = (top + 1.0, bottom - 2.0);
    b - f * (b - t).max(0.0)
}

#[allow(clippy::too_many_arguments)]
fn draw_series(
    c: &mut Canvas,
    values: &[f32],
    top: f32,
    bottom: f32,
    lo: f32,
    hi: f32,
    log: bool,
    color: Pixel,
) {
    if values.is_empty() || c.width < 2 {
        return;
    }
    let n = values.len();
    let w = c.width as f32;
    let x_of = |i: usize| {
        if n == 1 {
            w - 1.0
        } else {
            i as f32 * (w - 1.0) / (n as f32 - 1.0)
        }
    };
    let y_of = |v: f32| map_y(v, lo, hi, log, top, bottom);

    let raw: Vec<(f32, f32)> = values
        .iter()
        .enumerate()
        .filter(|(_, v)| v.is_finite())
        .map(|(i, &v)| (x_of(i), y_of(v)))
        .collect();
    // Raw values: more points than columns collapse into a min/max span per column (no peak lost)
    if n > c.width {
        let mut cols: Vec<Option<(f32, f32)>> = vec![None; c.width];
        for &(x, y) in &raw {
            let col = (x as usize).min(c.width - 1);
            cols[col] = Some(cols[col].map_or((y, y), |(a, b)| (a.min(y), b.max(y))));
        }
        for (x, span) in cols.into_iter().enumerate() {
            if let Some((y0, y1)) = span {
                c.span_aa_alpha(x, y0 - 0.5, y1 + 1.5, color, 0.28);
            }
        }
    } else {
        polyline(c, &raw, color, 0.35);
    }
    // Smoothed line on top
    let smooth = ema(values);
    let pts: Vec<(f32, f32)> = smooth
        .iter()
        .enumerate()
        .map(|(i, &v)| (x_of(i), y_of(v)))
        .collect();
    if n > c.width {
        let step = n as f32 / c.width as f32;
        let thinned: Vec<(f32, f32)> = (0..c.width)
            .map(|x| pts[((x as f32 * step) as usize).min(n - 1)])
            .collect();
        polyline(c, &thinned, color, 1.0);
    } else {
        polyline(c, &pts, color, 1.0);
    }
    // A marker on the last value
    if let Some(&(x, y)) = pts.last() {
        for dy in -2i32..=2 {
            for dx in -2i32..=2 {
                if dx * dx + dy * dy <= 5 {
                    let (px, py) = (
                        (x.round() as i32 + dx).max(0) as usize,
                        (y.round() as i32 + dy).max(0) as usize,
                    );
                    c.set(px, py, color);
                }
            }
        }
    }
}

/// Exponential moving average; the smoothing factor falls as the series grows, so a long run is
/// smoothed over more steps than a short one (~1/20 of the length, between 0.02 and 0.5).
pub(super) fn ema(values: &[f32]) -> Vec<f32> {
    let alpha = (20.0 / values.len().max(1) as f32).clamp(0.02, 0.5);
    let mut out = Vec::with_capacity(values.len());
    let mut acc: Option<f32> = None;
    for &v in values {
        if !v.is_finite() {
            out.push(acc.unwrap_or(0.0));
            continue;
        }
        let next = acc.map_or(v, |a| a + alpha * (v - a));
        acc = Some(next);
        out.push(next);
    }
    out
}

/// Anti-aliased polyline (Xiaolin Wu), blended with `alpha`.
pub(super) fn polyline(c: &mut Canvas, pts: &[(f32, f32)], color: Pixel, alpha: f32) {
    for w in pts.windows(2) {
        line(c, w[0], w[1], color, alpha);
    }
    if pts.len() == 1 {
        c.blend(
            pts[0].0.round().max(0.0) as usize,
            pts[0].1.round().max(0.0) as usize,
            color,
            alpha,
        );
    }
}

pub(super) fn line(
    c: &mut Canvas,
    (x0, y0): (f32, f32),
    (x1, y1): (f32, f32),
    color: Pixel,
    alpha: f32,
) {
    // The loop below ends on `x > xb`, which a NaN never satisfies: a non-finite end would spin a
    // work thread forever
    if !(x0.is_finite() && y0.is_finite() && x1.is_finite() && y1.is_finite()) {
        return;
    }
    let steep = (y1 - y0).abs() > (x1 - x0).abs();
    let ((x0, y0), (x1, y1)) = if steep {
        ((y0, x0), (y1, x1))
    } else {
        ((x0, y0), (x1, y1))
    };
    let ((x0, y0), (x1, y1)) = if x0 > x1 {
        ((x1, y1), (x0, y0))
    } else {
        ((x0, y0), (x1, y1))
    };
    let dx = x1 - x0;
    let gradient = if dx.abs() < f32::EPSILON {
        0.0
    } else {
        (y1 - y0) / dx
    };
    let mut put = |x: f32, y: f32, cover: f32| {
        if x < 0.0 || y < 0.0 {
            return;
        }
        let (px, py) = if steep {
            (y as usize, x as usize)
        } else {
            (x as usize, y as usize)
        };
        c.blend(px, py, color, cover * alpha);
    };
    let xa = x0.round();
    let xb = x1.round();
    let mut y = y0 + gradient * (xa - x0);
    let mut x = xa;
    loop {
        put(x, y.floor(), 1.0 - y.fract());
        put(x, y.floor() + 1.0, y.fract());
        y += gradient;
        x += 1.0;
        if x > xb {
            break;
        }
    }
}

// ─── spike raster ───────────────────────────────────────────────────────

#[derive(Debug, Clone)]
pub struct RasterRequest {
    pub frame: std::sync::Arc<SpikeFrame>,
    pub width: u32,
    pub height: u32,
    pub palette: Palette,
}

/// Where things are in the raster image (physical px), for the view's labels.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RasterLayout {
    /// Rows and columns of the density grid drawn (≤ units × steps: a big layer is binned).
    pub rows: usize,
    pub cols: usize,
    /// Raster area: `x` is where the grid starts, `plot_w × plot_h` its size.
    pub plot_w: u32,
    pub plot_h: u32,
    /// Population strip below the raster (a bar per column: share of neurons firing).
    pub strip_y: u32,
    pub strip_h: u32,
    /// Per-row rate bars at the right (`rate_x` .. width).
    pub rate_x: u32,
}

/// Numbers about the frame for the view's caption.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct RasterStats {
    pub units: usize,
    pub steps: usize,
    pub total_spikes: u64,
    pub mean_rate: f32,
    pub silent_fraction: f32,
    pub saturated: usize,
    pub burstiness: f32,
}

impl RasterStats {
    pub fn of(frame: &SpikeFrame) -> Self {
        Self {
            units: frame.units,
            steps: frame.steps,
            total_spikes: frame.total_spikes(),
            mean_rate: frame.mean_rate(),
            silent_fraction: frame.silent_fraction(),
            saturated: frame.saturated_units(),
            burstiness: frame.burstiness(),
        }
    }
}

const RATE_W: u32 = 44;
const STRIP_H: u32 = 44;
const SEP: u32 = 8;

/// Draws a spike frame: neurons down, time across.
///
/// One cell per (neuron, step) when it fits; a layer with more neurons than rows is binned into
/// groups of consecutive neurons (a cell is the share of its neuron-steps that fired, so sparse
/// activity is not lost — the same reduction the terminal raster used). Beside it a bar per row
/// shows its mean rate, red for a row that never fired; below it a strip shows, per time column,
/// how much of the population fired, which is where synchronous bursts show.
pub fn render_raster(req: &RasterRequest) -> (Frame, RasterLayout) {
    let (w, h) = (req.width.max(1), req.height.max(1));
    let mut frame = Frame::new(w, h);
    let p = &req.palette;
    let f = &*req.frame;
    let mut c = frame.canvas();
    fill(&mut c, p.background);

    let plot_w = w.saturating_sub(RATE_W + SEP).max(1);
    let strip_h = if h > STRIP_H * 3 { STRIP_H } else { 0 };
    let plot_h = h
        .saturating_sub(strip_h + if strip_h > 0 { SEP } else { 0 })
        .max(1);
    let mut layout = RasterLayout {
        rows: 0,
        cols: 0,
        plot_w,
        plot_h,
        strip_y: plot_h + SEP,
        strip_h,
        rate_x: plot_w + SEP,
    };
    if f.is_empty() {
        return (frame, layout);
    }

    let rows = f.units.min(plot_h as usize);
    let cols = f.steps.min(plot_w as usize);
    layout.rows = rows;
    layout.cols = cols;
    let grid = f.bin_density(rows, cols);

    // Cells large enough get a one-pixel gap so single spikes read as cells, not a smear
    let gaps = plot_w as usize / cols >= 4 && plot_h as usize / rows >= 4;
    for y in 0..plot_h as usize {
        let r = y * rows / plot_h as usize;
        let row_gap = gaps && (y + 1) * rows / plot_h as usize != r;
        for x in 0..plot_w as usize {
            let col = x * cols / plot_w as usize;
            let col_gap = gaps && (x + 1) * cols / plot_w as usize != col;
            let color = if row_gap || col_gap {
                p.grid
            } else {
                cell_color(p, grid[r * cols + col])
            };
            c.set(x, y, color);
        }
    }

    // Per-row mean rate
    let bar_max = RATE_W as usize - 4;
    for r in 0..rows {
        let mean = grid[r * cols..(r + 1) * cols].iter().sum::<f32>() / cols as f32;
        let (y0, y1) = (
            r * plot_h as usize / rows,
            ((r + 1) * plot_h as usize / rows).max(r * plot_h as usize / rows + 1),
        );
        let len =
            ((mean.clamp(0.0, 1.0) * bar_max as f32).round() as usize).max((mean > 0.0) as usize);
        let color = if mean == 0.0 {
            p.playhead
        } else {
            p.channel(0)
        };
        let thick = (y1 - y0).saturating_sub((y1 - y0 >= 4) as usize).max(1);
        for y in y0..y0 + thick {
            if mean == 0.0 {
                // A silent row: a short red tick, so it stands out from a row that is merely quiet
                for x in 0..3 {
                    c.set(layout.rate_x as usize + x, y, color);
                }
            }
            for x in 0..len {
                c.set(layout.rate_x as usize + x, y, color);
            }
        }
    }

    // Population strip
    if strip_h > 0 {
        let pop = f.population_density(cols);
        let peak = pop.iter().copied().fold(0.0f32, f32::max).max(f32::EPSILON);
        for y in layout.strip_y as usize..(layout.strip_y + strip_h) as usize {
            c.set(0, y.min(h as usize - 1), p.grid);
        }
        for x in 0..plot_w as usize {
            let v = pop[x * cols / plot_w as usize] / peak;
            let bar = (v * strip_h as f32).round() as usize;
            for dy in 0..bar {
                c.set(
                    x,
                    (layout.strip_y + strip_h) as usize - 1 - dy,
                    p.channel(1),
                );
            }
            c.set(x, (layout.strip_y + strip_h) as usize - 1, p.baseline);
        }
    }
    (frame, layout)
}

/// Density → colour. Square root: a cell with 1% of its neuron-steps firing must not read as empty.
fn cell_color(p: &Palette, density: f32) -> Pixel {
    if density <= 0.0 {
        p.background
    } else {
        p.heat(density.sqrt().max(0.12))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    fn pal() -> Palette {
        Palette::DARK
    }

    fn at(frame: &Frame, x: u32, y: u32) -> Pixel {
        frame.pixels()[(y * frame.width + x) as usize]
    }

    fn curves(
        loss: Vec<f32>,
        acc: Vec<f32>,
        grad: Vec<f32>,
        w: u32,
        h: u32,
    ) -> (Frame, Vec<CurvePanel>) {
        render_curves(&CurveRequest {
            loss,
            accuracy: acc,
            grad,
            width: w,
            height: h,
            palette: pal(),
        })
    }

    fn differs_from_background(f: &Frame, x0: u32, x1: u32, y0: u32, y1: u32) -> usize {
        let bg = pal().background;
        (y0..y1)
            .flat_map(|y| (x0..x1).map(move |x| (x, y)))
            .filter(|&(x, y)| at(f, x, y) != bg)
            .count()
    }

    // ─── curves ─────────────────────────────────────────────────────────

    #[test]
    fn empty_series_draw_only_the_frame() {
        let (f, panels) = curves(vec![], vec![], vec![], 200, 300);
        assert_eq!((f.width, f.height), (200, 300));
        assert_eq!(panels.len(), 3);
        assert!(panels.iter().all(|p| p.last.is_none() && p.points == 0));
        // Grid lines only: nothing in the colour of a series
        let c0 = pal().channel(0);
        assert!(f.pixels().iter().all(|&p| p != c0));
    }

    #[test]
    fn panels_tile_the_image_without_overlap() {
        let (_, panels) = curves(vec![1.0], vec![1.0], vec![1.0], 100, 300);
        assert!(panels[0].y0 == 0.0 && (panels[2].y1 - 1.0).abs() < 1e-5);
        for w in panels.windows(2) {
            assert!(w[0].y1 < w[1].y0, "there is a gap between the panels");
        }
    }

    #[test]
    fn accuracy_has_a_fixed_unit_scale_and_loss_autoscales() {
        let (_, panels) = curves(
            vec![5.0, 3.0, 1.0],
            vec![0.2, 0.9],
            vec![1.0, 2.0],
            100,
            300,
        );
        assert_eq!(
            (panels[1].lo, panels[1].hi, panels[1].log),
            (0.0, 1.0, false)
        );
        assert!(panels[0].lo < 1.0 && panels[0].hi > 5.0);
        assert!(!panels[0].log);
        assert_eq!(panels[0].last, Some(1.0));
    }

    #[test]
    fn a_gradient_over_many_decades_goes_logarithmic() {
        let (_, panels) = curves(vec![], vec![], vec![1e-6, 1e-3, 1.0, 10.0], 100, 300);
        assert!(panels[2].log);
        assert_eq!((panels[2].lo, panels[2].hi), (-6.0, 1.0));
    }

    #[test]
    fn a_constant_series_does_not_divide_by_zero() {
        let (f, panels) = curves(vec![2.0; 50], vec![], vec![], 120, 240);
        assert!(panels[0].hi > panels[0].lo);
        assert!(
            differs_from_background(&f, 0, 120, 0, 76) > 100,
            "flat line drawn"
        );
    }

    #[test]
    fn a_falling_loss_is_drawn_from_top_left_to_bottom_right() {
        let loss: Vec<f32> = (0..100).map(|i| 1.0 / (1.0 + i as f32)).collect();
        let (f, panels) = curves(loss, vec![], vec![], 200, 300);
        let h = (panels[0].y1 * 300.0) as u32;
        let c0 = pal().channel(0);
        let col_y = |x: u32| {
            (0..h).find(|&y| {
                at(&f, x, y) != pal().background
                    && at(&f, x, y) != pal().grid
                    && at(&f, x, y) != pal().baseline
            })
        };
        let (left, right) = (col_y(3), col_y(196));
        assert!(left.is_some() && right.is_some());
        assert!(
            left.unwrap() < right.unwrap(),
            "the curve falls: higher on the left than on the right"
        );
        let _ = c0;
    }

    #[test]
    fn more_points_than_columns_keep_a_spike() {
        // 2000 values, one outlier: with 100 columns it should not disappear
        let mut loss = vec![1.0f32; 2000];
        loss[1000] = 100.0;
        let (f, panels) = curves(loss, vec![], vec![], 100, 300);
        let h = (panels[0].y1 * 300.0) as u32;
        // The ejection column goes to the top of the panel
        let top_rows = differs_from_background(&f, 48, 53, 0, h / 6);
        assert!(top_rows > 0, "ejection lost during compression");
    }

    #[test]
    fn non_finite_values_do_not_panic_or_draw() {
        let (f, _) = curves(
            vec![1.0, f32::NAN, f32::INFINITY, 0.5],
            vec![f32::NAN],
            vec![f32::NEG_INFINITY],
            80,
            120,
        );
        assert_eq!((f.width, f.height), (80, 120));
    }

    #[test]
    fn degenerate_sizes_do_not_panic() {
        for (w, h) in [(1, 1), (2, 3), (5, 400), (400, 5)] {
            let (f, _) = curves(vec![1.0, 2.0, 3.0], vec![0.5], vec![1.0], w, h);
            assert_eq!((f.width, f.height), (w, h));
        }
        let (f, _) = curves(vec![], vec![], vec![], 0, 0);
        assert_eq!((f.width, f.height), (1, 1), "size zero goes up to 1x1");
    }

    #[test]
    fn ema_follows_but_smooths() {
        let noisy: Vec<f32> = (0..200)
            .map(|i| if i % 2 == 0 { 0.0 } else { 1.0 })
            .collect();
        let s = ema(&noisy);
        assert_eq!(s.len(), 200);
        assert!(
            s[150..].iter().all(|v| (0.2..0.8).contains(v)),
            "smoothing suppresses interleaving"
        );
        assert_eq!(ema(&[3.0]), vec![3.0]);
        assert!(ema(&[]).is_empty());
    }

    // ─── raster ─────────────────────────────────────────────────────────

    fn raster(f: SpikeFrame, w: u32, h: u32) -> (Frame, RasterLayout) {
        render_raster(&RasterRequest {
            frame: Arc::new(f),
            width: w,
            height: h,
            palette: pal(),
        })
    }

    #[test]
    fn a_spike_lands_in_its_cell() {
        // 8 steps × 4 neurons, spike (t=3, u=2); area 80×40 → cell 10×10 px
        let f = SpikeFrame::from_fn(8, 4, |t, u| t == 3 && u == 2);
        let (img, l) = raster(f, 80 + RATE_W + SEP, 40);
        assert_eq!((l.rows, l.cols, l.plot_w, l.plot_h), (4, 8, 80, 40));
        let bg = pal().background;
        // Inside the cell (x 30..39, y 20..29) spike color; next to it - background
        assert_ne!(at(&img, 34, 24), bg);
        assert_eq!(at(&img, 14, 24), bg);
        assert_eq!(at(&img, 34, 4), bg);
        assert_eq!(at(&img, 54, 24), bg);
    }

    #[test]
    fn every_spike_is_visible() {
        let f = SpikeFrame::from_fn(24, 24, |t, u| (t * 7 + u * 3) % 5 == 0);
        let spikes = f.total_spikes();
        let (img, l) = raster(f, 24 * 5 + RATE_W + SEP, 24 * 5);
        let lit = (0..l.plot_h)
            .flat_map(|y| (0..l.plot_w).map(move |x| (x, y)))
            .filter(|&(x, y)| at(&img, x, y) != pal().background && at(&img, x, y) != pal().grid)
            .count();
        // 5x5 grid with gap - 4x4 pixels per spike
        assert_eq!(lit as u64, spikes * 16);
    }

    #[test]
    fn a_silent_row_is_marked_red_and_an_active_one_is_not() {
        // neuron 0 is silent, neuron 1 is always spiked
        let f = SpikeFrame::from_fn(10, 2, |_, u| u == 1);
        let (img, l) = raster(f, 100 + RATE_W + SEP, 40);
        assert_eq!(l.rows, 2);
        let (red, active) = (pal().playhead, pal().channel(0));
        assert_eq!(at(&img, l.rate_x, 5), red, "silent line");
        assert_eq!(at(&img, l.rate_x, 25), active, "active line");
        assert!((0..40).all(|y| at(&img, l.rate_x + 30, y) != red));
    }

    #[test]
    fn a_big_layer_is_binned_to_the_screen_and_keeps_sparse_activity() {
        let f = SpikeFrame::from_fn(24, 100_000, |t, u| t == 5 && u == 77_777);
        let (img, l) = raster(f, 240 + RATE_W + SEP, 100);
        assert!(l.rows <= 100 && l.cols == 24, "{l:?}");
        assert!(
            differs_from_background(&img, 0, l.plot_w, 0, l.plot_h) > 0,
            "one spike out of 2.4 million cells is not lost"
        );
    }

    #[test]
    fn the_population_strip_shows_a_synchronous_burst() {
        let f = SpikeFrame::from_fn(20, 8, |t, _| t == 10);
        let (img, l) = raster(f, 200 + RATE_W + SEP, 300);
        assert!(l.strip_h > 0);
        let strip_top = l.strip_y;
        let bar_x = 10 * l.plot_w / 20 + 2;
        assert_ne!(
            at(&img, bar_x, strip_top + 1),
            pal().background,
            "burst - full column"
        );
        assert_eq!(
            at(&img, 2, strip_top + 1),
            pal().background,
            "in quiet steps there is no column"
        );
    }

    #[test]
    fn a_short_image_drops_the_strip_instead_of_squeezing_the_raster() {
        let (_, l) = raster(SpikeFrame::from_fn(8, 8, |_, _| true), 200, 100);
        assert_eq!(l.strip_h, 0);
        assert_eq!(l.plot_h, 100);
    }

    #[test]
    fn an_empty_frame_gives_a_blank_image() {
        let (img, l) = raster(SpikeFrame::default(), 150, 100);
        assert_eq!((l.rows, l.cols), (0, 0));
        assert!(img.pixels().iter().all(|&p| p == pal().background));
    }

    #[test]
    fn raster_degenerate_sizes_do_not_panic() {
        let f = SpikeFrame::from_fn(24, 24, |t, u| (t + u) % 2 == 0);
        for (w, h) in [(1, 1), (3, 3), (RATE_W + SEP, 50), (500, 1)] {
            let (img, _) = raster(f.clone(), w, h);
            assert_eq!((img.width, img.height), (w, h));
        }
    }

    #[test]
    fn both_palettes_draw() {
        for p in [Palette::DARK, Palette::LIGHT] {
            let f = SpikeFrame::from_fn(8, 8, |t, u| t == u);
            let (img, _) = render_raster(&RasterRequest {
                frame: Arc::new(f),
                width: 160,
                height: 80,
                palette: p,
            });
            assert!(img.pixels().iter().any(|&q| q != p.background));
        }
    }

    #[test]
    fn stats_describe_the_frame() {
        let f = SpikeFrame::from_fn(10, 4, |t, u| u == 1 || (u == 2 && t % 2 == 0));
        let s = RasterStats::of(&f);
        assert_eq!((s.units, s.steps), (4, 10));
        assert_eq!(s.total_spikes, 15);
        assert_eq!(s.saturated, 1);
        assert!((s.silent_fraction - 0.5).abs() < 1e-6);
    }

    #[test]
    fn a_line_with_a_non_finite_end_is_skipped_instead_of_looping() {
        let mut frame = Frame::new(8, 8);
        let mut canvas = frame.canvas();
        let white = Pixel::rgb(255, 255, 255);
        line(&mut canvas, (0.0, 0.0), (f32::NAN, 3.0), white, 1.0);
        line(&mut canvas, (0.0, 0.0), (f32::INFINITY, 3.0), white, 1.0);
        line(
            &mut canvas,
            (f32::NEG_INFINITY, 0.0),
            (3.0, 3.0),
            white,
            1.0,
        );
        assert!(frame.pixels().iter().all(|p| *p == Pixel::default()));
    }
}
