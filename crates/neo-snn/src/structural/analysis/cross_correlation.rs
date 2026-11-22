//! Cross-correlation of two spike trains — the paper's \(C_{ij}\).

/// Cross-correlation of spike trains `a` and `b` at lags `0..=max_lag`:
/// `C_ab(τ) = #{ordered pairs (x ∈ a, y ∈ b) with x − y = τ}` — pairs of spikes of the `a`
/// neuron following the `b` neuron by τ.
pub fn cross_corr(a: &[u64], b: &[u64], max_lag: usize) -> Vec<f64> {
    let mut out = vec![0.0; max_lag + 1];
    for &x in a {
        for &y in b {
            if x >= y {
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
    fn cross_corr_counts_pairs() {
        // a = [1, 3], b = [2]: the pair (3, 2) → d = 1
        assert_eq!(cross_corr(&[1, 3], &[2], 2), vec![0.0, 1.0, 0.0]);
    }

    #[test]
    fn cross_corr_zero_when_a_precedes_b() {
        // `a` spikes before `b`: no pairs with x >= y
        assert_eq!(cross_corr(&[1, 2], &[3, 4], 2), vec![0.0, 0.0, 0.0]);
    }
}
