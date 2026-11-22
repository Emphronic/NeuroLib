//! Rasterizer of the Evolution tab, drawn into a [`Frame`] on a work thread like
//! [`super::structure_plot`]'s. Three stacked plots sharing the generation axis: fitness (the
//! best, the mean, the champion of all time), the size of the best network (connections and
//! hidden neurons) and the number of species. Text is laid over the image by the view from the
//! panels returned here.

use crate::engine::canvas::{Canvas, Frame, Pixel};
use crate::engine::palette::Palette;

use super::plot::{CurvePanel, GAP, draw_panel_frame, fill, map_y, polyline};

#[derive(Debug, Clone)]
pub struct EvolutionRequest {
    pub best: Vec<f32>,
    pub mean: Vec<f32>,
    pub champion: Vec<f32>,
    pub conns: Vec<f32>,
    pub hidden: Vec<f32>,
    pub species: Vec<f32>,
    /// The fitness that ends the search, drawn as a dashed line when the curves come near it.
    pub target: Option<f32>,
    pub width: u32,
    pub height: u32,
    pub palette: Palette,
}

fn x_of(i: usize, n: usize, w: u32) -> f32 {
    if n <= 1 {
        w as f32 - 1.0
    } else {
        i as f32 * (w as f32 - 1.0) / (n as f32 - 1.0)
    }
}

fn series_points(
    values: &[f32],
    lo: f32,
    hi: f32,
    top: f32,
    bottom: f32,
    w: u32,
    stepped: bool,
) -> Vec<(f32, f32)> {
    let n = values.len();
    let mut pts = Vec::with_capacity(n * 2);
    for (i, &v) in values.iter().enumerate() {
        if !v.is_finite() {
            continue;
        }
        let (x, y) = (x_of(i, n, w), map_y(v, lo, hi, false, top, bottom));
        if let (true, Some(&(_, py))) = (stepped, pts.last()) {
            pts.push((x, py));
        }
        pts.push((x, y));
    }
    pts
}

fn dot(c: &mut Canvas, (x, y): (f32, f32), color: Pixel) {
    for dy in -2i32..=2 {
        for dx in -2i32..=2 {
            if dx * dx + dy * dy <= 5 {
                c.blend(
                    (x.round() as i32 + dx).max(0) as usize,
                    (y.round() as i32 + dy).max(0) as usize,
                    color,
                    1.0,
                );
            }
        }
    }
}

/// Draws the three plots. Fitness is autoscaled with a floor of `0..1`; the two counts start at 0.
pub fn render_evolution(req: &EvolutionRequest) -> (Frame, Vec<CurvePanel>) {
    let (w, h) = (req.width.max(1), req.height.max(1));
    let mut frame = Frame::new(w, h);
    let p = &req.palette;
    let mut canvas = frame.canvas();
    fill(&mut canvas, p.background);

    let finite_max = |sets: &[&[f32]]| {
        sets.iter()
            .flat_map(|s| s.iter())
            .copied()
            .filter(|v| v.is_finite())
            .fold(f32::NEG_INFINITY, f32::max)
    };
    let finite_min = |sets: &[&[f32]]| {
        sets.iter()
            .flat_map(|s| s.iter())
            .copied()
            .filter(|v| v.is_finite())
            .fold(f32::INFINITY, f32::min)
    };

    let gap = GAP / h as f32;
    let each = (1.0 - gap * 2.0) / 3.0;
    let mut panels = Vec::with_capacity(3);

    // (name, [(series, colour, alpha, stepped)], lo, hi)
    let fit_sets = [
        req.best.as_slice(),
        req.mean.as_slice(),
        req.champion.as_slice(),
    ];
    let fmax = finite_max(&fit_sets);
    let fmin = finite_min(&fit_sets);
    let (flo, mut fhi) = if fmax.is_finite() {
        (fmin.min(0.0), fmax.max(1.0))
    } else {
        (0.0, 1.0)
    };
    let near_target = req
        .target
        .filter(|t| t.is_finite() && fmax.is_finite() && fmax >= *t * 0.5);
    if let Some(t) = near_target {
        fhi = fhi.max(t);
    }
    fhi = flo + (fhi - flo) * 1.08;
    let cnt_sets = [req.conns.as_slice(), req.hidden.as_slice()];
    let cmax = finite_max(&cnt_sets);
    let chi = if cmax.is_finite() {
        cmax.max(1.0) * 1.25 + 1.0
    } else {
        2.0
    };
    let smax = finite_max(&[req.species.as_slice()]);
    let shi = if smax.is_finite() {
        smax.max(1.0) * 1.25 + 1.0
    } else {
        2.0
    };

    #[allow(clippy::type_complexity)]
    let specs: [(
        &'static str,
        Vec<(&[f32], Pixel, f32, bool)>,
        f32,
        f32,
        &[f32],
    ); 3] = [
        (
            "Fitness",
            vec![
                (&req.mean, p.channel(1), 0.45, false),
                (&req.champion, p.channel(3), 0.8, false),
                (&req.best, p.channel(0), 1.0, false),
            ],
            flo,
            fhi,
            &req.best,
        ),
        (
            "Best network (connections, hidden)",
            vec![
                (&req.hidden, p.channel(2), 0.6, true),
                (&req.conns, p.channel(0), 1.0, true),
            ],
            0.0,
            chi,
            &req.conns,
        ),
        (
            "Species",
            vec![(&req.species, p.channel(3), 1.0, true)],
            0.0,
            shi,
            &req.species,
        ),
    ];
    for (i, (name, lines, lo, hi, last_of)) in specs.into_iter().enumerate() {
        let y0 = i as f32 * (each + gap);
        let y1 = y0 + each;
        let (top, bottom) = (y0 * h as f32, y1 * h as f32);
        draw_panel_frame(&mut canvas, p, top, bottom);
        if i == 0
            && let Some(t) = near_target
        {
            let y = map_y(t, lo, hi, false, top, bottom).round().max(0.0) as usize;
            for x in 0..canvas.width {
                if (x / 6) % 2 == 0 {
                    canvas.blend(x, y, p.text, 0.7);
                }
            }
        }
        for (values, color, alpha, stepped) in lines {
            let pts = series_points(values, lo, hi, top, bottom, w, stepped);
            polyline(&mut canvas, &pts, color, alpha);
            if let Some(&last) = pts.last() {
                dot(&mut canvas, last, color);
            }
        }
        panels.push(CurvePanel {
            name,
            y0,
            y1,
            lo,
            hi,
            log: false,
            last: last_of.last().copied(),
            points: last_of.len(),
        });
    }
    (frame, panels)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn req(n: usize) -> EvolutionRequest {
        let f = |k: f32| {
            (0..n)
                .map(|i| 0.3 + k * i as f32 / n.max(1) as f32)
                .collect::<Vec<_>>()
        };
        EvolutionRequest {
            best: f(0.6),
            mean: f(0.3),
            champion: f(0.6),
            conns: (0..n).map(|i| 3.0 + (i / 5) as f32).collect(),
            hidden: (0..n).map(|i| (i / 10) as f32).collect(),
            species: (0..n).map(|i| 1.0 + (i % 4) as f32).collect(),
            target: Some(1.05),
            width: 400,
            height: 300,
            palette: Palette::of(true),
        }
    }

    fn painted(f: &Frame) -> usize {
        let bg = f.pixels().first().copied();
        f.pixels().iter().filter(|p| Some(**p) != bg).count()
    }

    #[test]
    fn empty_request_draws_frames_without_panicking() {
        let (f, panels) = render_evolution(&EvolutionRequest {
            best: vec![],
            mean: vec![],
            champion: vec![],
            conns: vec![],
            hidden: vec![],
            species: vec![],
            target: None,
            ..req(0)
        });
        assert_eq!((f.width, f.height), (400, 300));
        assert_eq!(panels.len(), 3);
        assert!(panels.iter().all(|p| p.points == 0 && p.last.is_none()));
    }

    #[test]
    fn panels_stack_without_overlap_and_report_last_values() {
        let (_, panels) = render_evolution(&req(40));
        assert_eq!(panels.len(), 3);
        for pair in panels.windows(2) {
            assert!(pair[0].y1 < pair[1].y0, "{pair:?}");
        }
        assert!((panels[2].y1 - 1.0).abs() < 1e-4);
        assert_eq!(panels[0].points, 40);
        assert!(panels[0].last.unwrap() > 0.8);
        assert!(panels[0].hi >= 1.0);
    }

    #[test]
    fn curves_leave_pixels_and_more_data_means_more_pixels() {
        let (a, _) = render_evolution(&req(2));
        let (b, _) = render_evolution(&req(60));
        assert!(painted(&a) > 0);
        assert!(painted(&b) > painted(&a));
    }

    #[test]
    fn nonfinite_values_are_skipped() {
        let mut r = req(10);
        r.best[3] = f32::NAN;
        r.mean[4] = f32::INFINITY;
        let (f, panels) = render_evolution(&r);
        assert!(painted(&f) > 0);
        assert!(panels[0].hi.is_finite() && panels[0].lo.is_finite());
    }

    #[test]
    fn target_line_joins_scale_only_when_near() {
        let mut far = req(10);
        far.best = vec![0.01; 10];
        far.mean = vec![0.01; 10];
        far.champion = vec![0.01; 10];
        far.target = Some(100.0);
        let (_, p_far) = render_evolution(&far);
        assert!(
            p_far[0].hi < 5.0,
            "a far target must not flatten the curves: {}",
            p_far[0].hi
        );
        let mut near = req(10);
        near.target = Some(1.5);
        let (_, p_near) = render_evolution(&near);
        assert!(p_near[0].hi >= 1.5);
    }
}
