//! Activity statistics: per-neuron autocorrelations of spike trains.

/// Autocorrelation of a spike train `s[0..n]` at lags `0..=max_lag`:
/// `C(τ) = #{ordered pairs (a, b) of spike indices with b − a = τ}` — the number of spike pairs
/// at distance `τ` (O(n²); fine for histories of thousands of steps).
///
/// Empty and one-spike trains give all-zero vectors.
pub fn autocorr(spikes: &[u64], max_lag: usize) -> Vec<f64> {
    let mut out = vec![0.0; max_lag + 1];
    for &x in spikes {
        for &y in spikes {
            if x > y {
                let d = (x - y) as usize;
                if d <= max_lag {
                    out[d] += 1.0;
                }
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn autocorr_counts_pairs() {
        // spikes [1, 3, 4]: pairs (3,1) → d=2, (4,3) → d=1, (4,1) → d=3
        assert_eq!(autocorr(&[1, 3, 4], 3), vec![0.0, 1.0, 1.0, 1.0]);
    }

    #[test]
    fn autocorr_zero_for_no_pairs() {
        assert_eq!(autocorr(&[], 3), vec![0.0, 0.0, 0.0, 0.0]);
        assert_eq!(autocorr(&[7], 3), vec![0.0, 0.0, 0.0, 0.0]);
    }
}
