// neo-quantization/src/rng.rs
//
// Tiny deterministic PRNG and rounding based on it - for
// `RoundingMode::Stochastic`. Without external dependencies (`rand` in the crate -
// only dev-dependency for tests).

/// Simple xorshift64 PRNG: compact, uniform enough for
/// stochastic rounding and jitter, fully reproducible by seed.
pub(crate) struct XorShift64(u64);

impl XorShift64 {
    /// New generator. `seed == 0` (degenerate xorshift state)
    /// is replaced by a fixed non-zero constant.
    pub(crate) fn new(seed: u64) -> Self {
        XorShift64(if seed == 0 {
            0xdead_beef_cafe_babe
        } else {
            seed
        })
    }

    /// Next uniform value from `[0, 1)` (24 significant bits).
    pub(crate) fn next_f32(&mut self) -> f32 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        // Map the top 24 bits to [0, 1).
        ((x >> 40) as u32 as f32) / (1u32 << 24) as f32
    }
}

/// Round `x` to the nearest integer: to the nearest one at `rng == None`, otherwise -
/// STOCHASTICALLY: up with probability `frac(x)`, down with probability
/// `1 - frac(x)`. The mathematical expectation of the result is `x` (estimate
/// unbiased), and the integer `x` remains exactly the same: `frac == 0`, and the condition
/// `u < frac` is never executed (so not used here
/// `floor(x + u)` - on the border of f32-rounding the sum could occasionally
/// "jump over" the whole).
///
/// For `rng == None` this is exactly `x.round()` - the behavior of `Nearest` does not change.
#[inline]
pub(crate) fn round_or_stochastic(x: f32, rng: &mut Option<XorShift64>) -> f32 {
    match rng {
        Some(r) => {
            let floor = x.floor();
            if r.next_f32() < x - floor {
                floor + 1.0
            } else {
                floor
            }
        }
        None => x.round(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_next_f32_in_unit_interval_and_deterministic() {
        let mut a = XorShift64::new(42);
        let mut b = XorShift64::new(42);
        for _ in 0..10_000 {
            let (x, y) = (a.next_f32(), b.next_f32());
            assert!((0.0..1.0).contains(&x), "out of [0,1): {x}");
            assert_eq!(x, y);
        }
    }

    #[test]
    fn test_zero_seed_is_not_degenerate() {
        let mut r = XorShift64::new(0);
        let first = r.next_f32();
        let second = r.next_f32();
        assert!(first != second || first != 0.0);
    }

    #[test]
    fn test_nearest_when_no_rng() {
        let mut none: Option<XorShift64> = None;
        assert_eq!(round_or_stochastic(2.4, &mut none), 2.0);
        assert_eq!(round_or_stochastic(2.6, &mut none), 3.0);
        assert_eq!(round_or_stochastic(-2.6, &mut none), -3.0);
    }

    #[test]
    fn test_stochastic_round_is_unbiased_and_picks_neighbours() {
        for &x in &[2.3f32, -2.3, 0.75, -0.1] {
            let mut rng = Some(XorShift64::new(9));
            let n = 40_000;
            let mut sum = 0.0f64;
            for _ in 0..n {
                let r = round_or_stochastic(x, &mut rng);
                assert!(r == x.floor() || r == x.floor() + 1.0, "{x} -> {r}");
                sum += r as f64;
            }
            let mean = sum / n as f64;
            assert!((mean - x as f64).abs() < 0.02, "x={x} mean={mean}");
        }
    }

    #[test]
    fn test_stochastic_round_keeps_integers_exact() {
        let mut rng = Some(XorShift64::new(3));
        for &x in &[-4.0f32, -1.0, 0.0, 1.0, 3.0, 127.0, 255.0] {
            for _ in 0..50_000 {
                assert_eq!(round_or_stochastic(x, &mut rng), x);
            }
        }
    }
}
