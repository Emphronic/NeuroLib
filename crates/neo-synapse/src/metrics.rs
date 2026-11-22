//! Spike-train metrics: correlograms, inter-spike intervals, instantaneous firing rate, and the
//! quality criteria curation compares them with. Spike trains are sample indices.

use serde::{Deserialize, Serialize};

/// Symmetric spike train lag histogram over `[-window_ms, +window_ms]`.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Correlogram {
    /// Bin center lags in milliseconds (`len == 2 * half_bins + 1`).
    pub bin_centers_ms: Vec<f32>,
    /// Spike pair counts per lag bin.
    pub counts: Vec<u64>,
    pub bin_size_ms: f32,
    pub window_ms: f32,
}

/// Narrowest correlogram bin (ms).
const MIN_BIN_MS: f32 = 0.05;

/// Computes the symmetric Auto-Correlogram (ACG) of a sorted spike train (in sample indices),
/// excluding the trivial zero-lag self-coincidence ($i = j$).
///
/// Counts are over **ordered** pairs $(i, j)$, $i \ne j$, at lag $t_j - t_i$ (as SpikeInterface's
/// correlograms): every unordered pair adds one count at $+\Delta t$ and one at $-\Delta t$, so two
/// distinct spikes in the same sample add 2 to the centre bin, and the ACG equals the CCG of the
/// train with itself minus the $i = j$ terms.
pub fn compute_autocorrelogram(
    sorted_samples: &[u64],
    sample_rate_hz: f64,
    bin_size_ms: f32,
    window_ms: f32,
) -> Correlogram {
    let bin_ms = bin_size_ms.max(MIN_BIN_MS);
    let half_bins = ((window_ms.max(bin_ms) / bin_ms).round() as usize).max(1);
    let num_bins = 2 * half_bins + 1;
    let max_lag_ms = (half_bins as f32 + 0.5) * bin_ms;

    let mut bin_centers_ms = Vec::with_capacity(num_bins);
    for b in 0..num_bins {
        bin_centers_ms.push(((b as isize - half_bins as isize) as f32) * bin_ms);
    }

    let mut counts = vec![0u64; num_bins];
    let n = sorted_samples.len();
    if n < 2 || sample_rate_hz <= 0.0 {
        return Correlogram {
            bin_centers_ms,
            counts,
            bin_size_ms: bin_ms,
            window_ms: (half_bins as f32) * bin_ms,
        };
    }

    let ms_per_sample = 1000.0 / sample_rate_hz;

    for i in 0..n {
        let t_i = sorted_samples[i];
        #[allow(clippy::needless_range_loop)]
        for j in (i + 1)..n {
            let dt_ms = ((sorted_samples[j].saturating_sub(t_i)) as f64 * ms_per_sample) as f32;
            if dt_ms >= max_lag_ms {
                break;
            }
            let offset_bins = ((dt_ms / bin_ms).round() as usize).min(half_bins);
            counts[half_bins + offset_bins] += 1;
            counts[half_bins - offset_bins] += 1;
        }
    }

    Correlogram {
        bin_centers_ms,
        counts,
        bin_size_ms: bin_ms,
        window_ms: (half_bins as f32) * bin_ms,
    }
}

/// Computes the Cross-Correlogram (CCG) between reference train `samples_a` and target train `samples_b`
/// (lags $t_b - t_a \in [-W, +W]$).
pub fn compute_crosscorrelogram(
    sorted_samples_a: &[u64],
    sorted_samples_b: &[u64],
    sample_rate_hz: f64,
    bin_size_ms: f32,
    window_ms: f32,
) -> Correlogram {
    let bin_ms = bin_size_ms.max(MIN_BIN_MS);
    let half_bins = ((window_ms.max(bin_ms) / bin_ms).round() as usize).max(1);
    let num_bins = 2 * half_bins + 1;
    let max_lag_ms = (half_bins as f32 + 0.5) * bin_ms;

    let mut bin_centers_ms = Vec::with_capacity(num_bins);
    for b in 0..num_bins {
        bin_centers_ms.push(((b as isize - half_bins as isize) as f32) * bin_ms);
    }

    let mut counts = vec![0u64; num_bins];
    if sorted_samples_a.is_empty() || sorted_samples_b.is_empty() || sample_rate_hz <= 0.0 {
        return Correlogram {
            bin_centers_ms,
            counts,
            bin_size_ms: bin_ms,
            window_ms: (half_bins as f32) * bin_ms,
        };
    }

    let ms_per_sample = 1000.0 / sample_rate_hz;
    let mut left_b = 0usize;
    let n_b = sorted_samples_b.len();

    for &t_a in sorted_samples_a {
        while left_b < n_b {
            let dt_ms = ((sorted_samples_b[left_b] as f64 - t_a as f64) * ms_per_sample) as f32;
            if dt_ms < -max_lag_ms {
                left_b += 1;
            } else {
                break;
            }
        }

        #[allow(clippy::needless_range_loop)]
        for j in left_b..n_b {
            let dt_ms = ((sorted_samples_b[j] as f64 - t_a as f64) * ms_per_sample) as f32;
            if dt_ms >= max_lag_ms {
                break;
            }
            let signed_bin = (dt_ms / bin_ms).round() as isize;
            let bin_idx =
                (half_bins as isize + signed_bin).clamp(0, (num_bins - 1) as isize) as usize;
            counts[bin_idx] += 1;
        }
    }

    Correlogram {
        bin_centers_ms,
        counts,
        bin_size_ms: bin_ms,
        window_ms: (half_bins as f32) * bin_ms,
    }
}

/// ISI-violation metrics of one unit.
#[derive(Debug, Clone, PartialEq)]
pub struct IsiMetrics {
    pub total_spikes: usize,
    /// Consecutive intervals shorter than the threshold.
    pub violation_count: usize,
    /// `violation_count` as a percentage of the intervals.
    pub violation_rate_pct: f32,
    /// Hill et al. (2011) ratio of the violation rate to the unit's rate (`isi_violations_ratio`); NaN without spikes.
    pub isi_violations_ratio: f64,
    /// Violations per second of recording (`isi_violations_rate`).
    pub violations_per_sec: f64,
    /// Spikes per second of recording.
    pub firing_rate_hz: f64,
}

/// ISI violations of a spike train (sample indices, any order) over a recording of `total_duration_sec`.
pub fn compute_isi_violations(
    spike_samples: &[u64],
    sample_rate_hz: f64,
    total_duration_sec: f64,
    isi_threshold_ms: f64,
    min_isi_ms: f64,
) -> IsiMetrics {
    let mut sorted = spike_samples.to_vec();
    sorted.sort_unstable();
    let n = sorted.len();
    let threshold_s = isi_threshold_ms * 1e-3;
    let violations = sorted
        .windows(2)
        .filter(|w| w[1] as f64 / sample_rate_hz - w[0] as f64 / sample_rate_hz < threshold_s)
        .count();

    let (ratio, per_sec) = if n > 0 {
        let violation_time = 2.0 * n as f64 * (threshold_s - min_isi_ms * 1e-3);
        let total_rate = n as f64 / total_duration_sec;
        let violation_rate = violations as f64 / violation_time;
        (
            violation_rate / total_rate,
            violations as f64 / total_duration_sec,
        )
    } else {
        (f64::NAN, f64::NAN)
    };

    IsiMetrics {
        total_spikes: n,
        violation_count: violations,
        violation_rate_pct: if n > 1 {
            violations as f32 / (n - 1) as f32 * 100.0
        } else {
            0.0
        },
        isi_violations_ratio: ratio,
        violations_per_sec: per_sec,
        firing_rate_hz: if total_duration_sec > 0.0 {
            n as f64 / total_duration_sec
        } else {
            0.0
        },
    }
}

/// Inter-spike-interval histogram of a spike train (sample indices, any order): counts of the
/// intervals in `bin_ms` bins over `[0, max_ms)`, and the bin centres in ms.
pub fn isi_histogram(
    spike_samples: &[u64],
    sample_rate_hz: f64,
    bin_ms: f64,
    max_ms: f64,
) -> (Vec<f64>, Vec<u64>) {
    let bin_ms = bin_ms.max(1e-3);
    let bins = ((max_ms.max(bin_ms) / bin_ms).round() as usize).max(1);
    let hi = bins as f64 * bin_ms;
    let mut sorted = spike_samples.to_vec();
    sorted.sort_unstable();
    let ms = 1e3 / sample_rate_hz.max(f64::MIN_POSITIVE);
    // Intervals of exactly `hi` belong to the next bin, not the last one
    let intervals = sorted
        .windows(2)
        .map(|w| (w[1] - w[0]) as f64 * ms)
        .filter(|&d| d < hi);
    (
        crate::math::bin_centers(0.0, hi, bins),
        crate::math::histogram(intervals, 0.0, hi, bins),
    )
}

/// Continuous instantaneous firing rate profile $r(t)$ in Hz.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct FiringRateCurve {
    pub time_bin_centers_sec: Vec<f64>,
    pub rate_hz: Vec<f32>,
    pub bin_width_sec: f64,
    pub kernel_sigma_ms: f64,
}

/// Computes a continuous Gaussian-smoothed instantaneous firing rate $r(t)$ (in Hz) over `[0, total_duration_sec]`.
pub fn compute_instantaneous_firing_rate(
    spike_times_sec: &[f64],
    total_duration_sec: f64,
    bin_width_ms: f64,
    kernel_sigma_ms: f64,
) -> FiringRateCurve {
    let dt = (bin_width_ms * 1e-3).max(1e-4);
    let num_bins = ((total_duration_sec.max(dt) / dt).ceil() as usize).max(1);
    let mut raw_rate = vec![0.0f32; num_bins];
    let inv_dt = (1.0 / dt) as f32;

    for &t in spike_times_sec {
        if t >= 0.0 && t <= total_duration_sec {
            let b = ((t / dt).floor() as usize).min(num_bins - 1);
            raw_rate[b] += inv_dt;
        }
    }

    let sigma_samples = (kernel_sigma_ms.max(0.0) / bin_width_ms.max(1e-3)) as f32;
    let rate_hz = if sigma_samples > 0.05 {
        crate::math::gaussian_smooth_1d(&raw_rate, sigma_samples)
    } else {
        raw_rate
    };

    let time_bin_centers_sec: Vec<f64> = (0..num_bins).map(|b| (b as f64 + 0.5) * dt).collect();
    FiringRateCurve {
        time_bin_centers_sec,
        rate_hz,
        bin_width_sec: dt,
        kernel_sigma_ms,
    }
}

/// Configurable Allen / IBL / Phy quality metric parameters and automated unit classification
/// thresholds (eliminating hardcoded refractory periods and SNR / ISI cutoffs across sorters,
/// storage loaders, and curation).
#[derive(Debug, Clone, Copy, PartialEq, serde::Serialize, serde::Deserialize)]
pub struct QualityCriteria {
    /// Refractory period in milliseconds (default `1.5` ms for IBL/Allen, `2.0` ms in Phy).
    pub refractory_ms: f64,
    /// Censored / dead-time window in milliseconds (default `0.0` ms).
    pub censored_ms: f64,
    /// Minimum spikes required before a unit can be classified above `Noise` (default `3`).
    pub min_spikes: usize,
    /// SNR below which a unit is classified as `Noise` when SNR is finite (default `1.5`).
    pub noise_snr_threshold: f32,
    /// Minimum SNR required for automated `SingleUnit` (`good`) classification (default `3.0`).
    pub sua_snr_threshold: f32,
    /// Maximum Hill et al. ISI violation ratio for `SingleUnit` classification (default `0.5`).
    pub max_isi_violation_ratio: f64,
}

impl Default for QualityCriteria {
    fn default() -> Self {
        Self {
            refractory_ms: 1.5,
            censored_ms: 0.0,
            min_spikes: 3,
            noise_snr_threshold: 1.5,
            sua_snr_threshold: 3.0,
            max_isi_violation_ratio: 0.5,
        }
    }
}

impl QualityCriteria {
    /// Standard Phy curation defaults (`2.0` ms refractory period).
    pub const PHY: Self = Self {
        refractory_ms: 2.0,
        censored_ms: 0.0,
        min_spikes: 3,
        noise_snr_threshold: 1.5,
        sua_snr_threshold: 3.0,
        max_isi_violation_ratio: 0.5,
    };
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn autocorrelogram_counts_ordered_pairs() {
        // Spikes 1 ms apart at 1 kHz: one pair at ±1 ms
        let c = compute_autocorrelogram(&[0, 1], 1000.0, 1.0, 5.0);
        assert_eq!(c.counts.len(), 11);
        assert_eq!(c.counts[5 + 1], 1);
        assert_eq!(c.counts[5 - 1], 1);
        assert_eq!(c.counts.iter().sum::<u64>(), 2);
    }

    #[test]
    fn crosscorrelogram_lags_are_b_minus_a() {
        let c = compute_crosscorrelogram(&[10], &[12], 1000.0, 1.0, 5.0);
        assert_eq!(c.counts[5 + 2], 1);
        assert_eq!(c.counts.iter().sum::<u64>(), 1);
    }

    #[test]
    fn isi_histogram_and_violations() {
        let samples = [0u64, 30, 3000, 6000];
        let (centers, counts) = isi_histogram(&samples, 30_000.0, 1.0, 200.0);
        assert_eq!(centers.len(), 200);
        assert_eq!(counts.iter().sum::<u64>(), 3);
        let m = compute_isi_violations(&samples, 30_000.0, 1.0, 2.0, 0.0);
        assert_eq!(m.violation_count, 1);
        assert!((m.violation_rate_pct - 100.0 / 3.0).abs() < 1e-3);
    }

    #[test]
    fn firing_rate_conserves_spikes() {
        let times = [0.5, 1.5, 1.6, 3.2];
        let curve = compute_instantaneous_firing_rate(&times, 4.0, 100.0, 0.0);
        let total: f64 = curve
            .rate_hz
            .iter()
            .map(|&r| r as f64 * curve.bin_width_sec)
            .sum();
        assert!((total - 4.0).abs() < 1e-3);
    }
}
