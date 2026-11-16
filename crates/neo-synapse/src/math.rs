//! Small numeric helpers shared by the metrics and the curation views.

/// Counts of `values` in `bins` equal bins over `[lo, hi)` (`hi` itself goes in the last bin);
/// values outside are not counted.
pub fn histogram(values: impl IntoIterator<Item = f64>, lo: f64, hi: f64, bins: usize) -> Vec<u64> {
    let mut counts = vec![0u64; bins];
    if bins == 0 || hi.is_nan() || lo.is_nan() || hi <= lo {
        return counts;
    }
    let scale = bins as f64 / (hi - lo);
    for v in values {
        if v < lo || v > hi || v.is_nan() {
            continue;
        }
        counts[(((v - lo) * scale) as usize).min(bins - 1)] += 1;
    }
    counts
}

/// Centres of `bins` equal bins over `[lo, hi)`.
pub fn bin_centers(lo: f64, hi: f64, bins: usize) -> Vec<f64> {
    let w = (hi - lo) / bins.max(1) as f64;
    (0..bins).map(|b| lo + (b as f64 + 0.5) * w).collect()
}

/// `max − min` of `signal` (`numpy.ptp`); 0 when empty.
pub fn peak_to_peak(signal: &[f32]) -> f32 {
    if signal.is_empty() {
        return 0.0;
    }
    let (lo, hi) = signal
        .iter()
        .fold((f32::INFINITY, f32::NEG_INFINITY), |(lo, hi), &v| {
            (lo.min(v), hi.max(v))
        });
    hi - lo
}

/// Whether `(x, y)` lies inside the closed polygon `vertices` (even-odd rule; fewer than three
/// vertices contain nothing).
pub fn point_in_polygon(x: f64, y: f64, vertices: &[(f64, f64)]) -> bool {
    if vertices.len() < 3 {
        return false;
    }
    let mut inside = false;
    let mut j = vertices.len() - 1;
    for i in 0..vertices.len() {
        let ((xi, yi), (xj, yj)) = (vertices[i], vertices[j]);
        if (yi > y) != (yj > y) && x < (xj - xi) * (y - yi) / (yj - yi) + xi {
            inside = !inside;
        }
        j = i;
    }
    inside
}

/// Zero-phase Gaussian smoothing of `signal` (`sigma` in samples, kernel cut at 4σ, scipy's
/// `reflect` edges).
pub fn gaussian_smooth_1d(signal: &[f32], sigma: f32) -> Vec<f32> {
    if signal.is_empty() || sigma <= 1e-4 {
        return signal.to_vec();
    }
    let radius = (4.0 * sigma + 0.5) as isize;
    let mut taps: Vec<f32> = (-radius..=radius)
        .map(|k| (-(k as f32).powi(2) / (2.0 * sigma * sigma)).exp())
        .collect();
    let sum: f32 = taps.iter().sum();
    taps.iter_mut().for_each(|t| *t /= sum);
    let n = signal.len() as isize;
    let reflect = |p: isize| -> usize {
        let q = if p < 0 {
            -p - 1
        } else if p >= n {
            2 * n - 1 - p
        } else {
            p
        };
        q.clamp(0, n - 1) as usize
    };
    (0..n)
        .map(|i| {
            taps.iter()
                .enumerate()
                .map(|(k, &w)| signal[reflect(i + k as isize - radius)] * w)
                .sum()
        })
        .collect()
}

/// `Σ x[t]·y[t + lag]` over the overlap.
pub fn lagged_dot(x: &[f32], y: &[f32], lag: isize) -> f32 {
    let start = (-lag).max(0) as usize;
    let end = (y.len() as isize - lag).clamp(0, x.len() as isize) as usize;
    (start..end.max(start))
        .map(|t| x[t] * y[(t as isize + lag) as usize])
        .sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn histogram_puts_the_upper_edge_in_the_last_bin() {
        assert_eq!(histogram([0.0, 0.5, 1.0, 2.0], 0.0, 1.0, 2), vec![1, 2]);
        assert_eq!(bin_centers(0.0, 1.0, 2), vec![0.25, 0.75]);
    }

    #[test]
    fn polygon_even_odd() {
        let square = [(0.0, 0.0), (2.0, 0.0), (2.0, 2.0), (0.0, 2.0)];
        assert!(point_in_polygon(1.0, 1.0, &square));
        assert!(!point_in_polygon(3.0, 1.0, &square));
        let l = [
            (0.0, 0.0),
            (2.0, 0.0),
            (2.0, 1.0),
            (1.0, 1.0),
            (1.0, 2.0),
            (0.0, 2.0),
        ];
        assert!(!point_in_polygon(1.5, 1.5, &l));
        assert!(!point_in_polygon(0.0, 0.0, &square[..2]));
    }

    #[test]
    fn smoothing_keeps_a_constant() {
        let s = gaussian_smooth_1d(&[2.0; 40], 3.0);
        assert!(s.iter().all(|v| (v - 2.0).abs() < 1e-4));
    }
}
