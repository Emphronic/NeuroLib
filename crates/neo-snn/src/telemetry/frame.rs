//! A snapshot of spike activity, untethered from the backend.
//!
//! ## Why bitmap
//!
//! A `bool` per cell stores a whole byte per “neuron × step”. With
//! 24 steps and a million neurons that means 24 MB per snapshot, and history
//! keeps a series of frames for later viewing. Here one cell is one bit:
//! the same 24 million cells occupy 3 MB, and the search skips completely
//! empty machine words (spikes are sparse - that is why SNN
//! benefit from event-based storage).
//!
//! Neuron statistics (frequencies, silent, saturated) are calculated lazily and
//! exactly once per image - on the first call. Pictures from history,
//! which no one has opened, they don’t pay it.

use std::sync::OnceLock;

use burn::tensor::Tensor;

/// Aggregates that are needed for each rendering frame. Counts as one pass
/// according to the set bits.
#[derive(Debug, Clone, Default)]
struct Stats {
    /// How many times each neuron spiked.
    unit_counts: Vec<u32>,
    /// How many neurons were spiked at each step.
    step_counts: Vec<u32>,
    /// Total ranks.
    total: u64,
    /// Neurons that have never spiked.
    silent: usize,
    /// Neurons with frequency ≥ 0.999.
    saturated: usize,
}

/// Spike activity of one sample taken from the device.
///
/// The tensor is read to the host ONCE when creating a snapshot, then the dashboard
/// works with regular memory. This is important: redrawing occurs dozens of times per
/// second, and jerking the device for each frame would mean synchronizing
/// it with the terminal.
///
/// The backend must have `f32` as the element type.
#[derive(Debug, Clone, Default)]
pub struct SpikeFrame {
    /// Number of time steps.
    pub steps: usize,
    /// Number of neurons.
    pub units: usize,
    /// Bits, bit number `step * units + unit`.
    words: Vec<u64>,
    stats: OnceLock<Stats>,
}

/// The beginning of the `i`-th of the `bins` intervals into which the `n` elements are divided
/// “equally”: `ceil(i·n / bins)`. The element `k` falls within the interval
/// `k·bins / n`, and this is exactly the inverse function.
pub(crate) fn bin_start(i: usize, n: usize, bins: usize) -> usize {
    ((i as u128 * n as u128).div_ceil(bins.max(1) as u128)) as usize
}

/// Which of the `bins` intervals does the element `k` from `n` fall into?
#[inline]
pub(crate) fn bin_of(k: usize, n: usize, bins: usize) -> usize {
    ((k as u128 * bins as u128) / n.max(1) as u128) as usize
}

impl SpikeFrame {
    fn from_words(steps: usize, units: usize, words: Vec<u64>) -> Self {
        Self {
            steps,
            units,
            words,
            stats: OnceLock::new(),
        }
    }

    /// Remove sample `sample` from tensor `[batch, steps, units]`.
    ///
    /// A value greater than 0.5 is considered a digit.
    ///
    /// Only the selected sample is transferred to the host, and not the entire batch: when
    /// in large layers the difference is `batch` times tens of megabytes per
    /// every frame.
    pub fn from_tensor(spikes: Tensor<3>, sample: usize) -> Self {
        let [batch, steps, units] = spikes.dims();
        assert!(
            sample < batch,
            "SpikeFrame: sample {sample} outside batch size {batch}"
        );

        let one = spikes.slice([sample..sample + 1, 0..steps, 0..units]);
        let flat = crate::error::or_panic(crate::error::tensor_to_vec(one, "SpikeFrame"));

        let n = steps * units;
        let mut words = vec![0u64; n.div_ceil(64)];
        for (w, chunk) in flat[..n].chunks(64).enumerate() {
            let mut word = 0u64;
            for (b, &v) in chunk.iter().enumerate() {
                word |= ((v > 0.5) as u64) << b;
            }
            words[w] = word;
        }
        Self::from_words(steps, units, words)
    }

    /// Assemble a snapshot directly from bits, without a tensor or backend.
    ///
    /// Needed where activity already exists in the shape of ordinary `bool` - in
    /// primarily in the `health.rs` tests, which the `burn` backend does not
    /// necessary and should not be an addiction.
    ///
    /// `bits` is in order `[step * units + unit]`.
    pub fn from_bits(steps: usize, units: usize, bits: Vec<bool>) -> Self {
        debug_assert_eq!(
            bits.len(),
            steps * units,
            "SpikeFrame::from_bits: {} bits, expected {steps}×{units} = {}",
            bits.len(),
            steps * units,
        );
        let mut words = vec![0u64; bits.len().div_ceil(64)];
        for (w, chunk) in bits.chunks(64).enumerate() {
            let mut word = 0u64;
            for (b, &v) in chunk.iter().enumerate() {
                word |= (v as u64) << b;
            }
            words[w] = word;
        }
        Self::from_words(steps, units, words)
    }

    /// Collect a snapshot from the function `(step, unit) -> whether spiked`.
    ///
    /// Does not highlight the intermediate `Vec<bool>`: for a million neurons this is
    /// difference between 3 MB and 24 MB per frame. Order of calls - step by step, inside
    /// steps across neurons so that the stateful generator (RNG) runs.
    pub fn from_fn(steps: usize, units: usize, mut f: impl FnMut(usize, usize) -> bool) -> Self {
        let n = steps * units;
        let mut words = vec![0u64; n.div_ceil(64)];
        let mut i = 0usize;
        for t in 0..steps {
            for u in 0..units {
                if f(t, u) {
                    words[i >> 6] |= 1u64 << (i & 63);
                }
                i += 1;
            }
        }
        Self::from_words(steps, units, words)
    }

    /// How many bytes does a snapshot take (bitmap plus statistics, if any)
    /// already calculated). You need frame history to stay within your memory budget.
    pub fn size_bytes(&self) -> usize {
        let stats = self
            .stats
            .get()
            .map(|s| (s.unit_counts.len() + s.step_counts.len()) * 4)
            .unwrap_or(0);
        self.words.len() * 8 + stats
    }

    /// Call `f(step, unit)` for each digit.
    ///
    /// Follows machine words and skips zeros, so the cost is -
    /// `O(words + bits)`, not `O(steps × neurons)`.
    pub fn for_each_spike(&self, mut f: impl FnMut(usize, usize)) {
        if self.units == 0 {
            return;
        }
        for (w, &word) in self.words.iter().enumerate() {
            let mut bits = word;
            while bits != 0 {
                let b = bits.trailing_zeros() as usize;
                bits &= bits - 1;
                let i = w * 64 + b;
                f(i / self.units, i % self.units);
            }
        }
    }

    fn stats(&self) -> &Stats {
        self.stats.get_or_init(|| {
            let mut unit_counts = vec![0u32; self.units];
            let mut step_counts = vec![0u32; self.steps];
            let mut total = 0u64;
            self.for_each_spike(|t, u| {
                unit_counts[u] += 1;
                step_counts[t] += 1;
                total += 1;
            });
            let silent = unit_counts.iter().filter(|&&c| c == 0).count();
            let saturated = if self.steps == 0 {
                0
            } else {
                unit_counts
                    .iter()
                    .filter(|&&c| c as f32 / self.steps as f32 >= 0.999)
                    .count()
            };
            Stats {
                unit_counts,
                step_counts,
                total,
                silent,
                saturated,
            }
        })
    }

    /// Whether neuron `unit` spiked at step `step`.
    pub fn spiked(&self, step: usize, unit: usize) -> bool {
        if step >= self.steps || unit >= self.units {
            return false;
        }
        let i = step * self.units + unit;
        (self.words[i >> 6] >> (i & 63)) & 1 == 1
    }

    /// Is the picture empty?
    pub fn is_empty(&self) -> bool {
        self.steps == 0 || self.units == 0
    }

    /// Total digits in the image.
    pub fn total_spikes(&self) -> u64 {
        self.stats().total
    }

    /// The number of spikes of each neuron.
    pub fn unit_counts(&self) -> &[u32] {
        &self.stats().unit_counts
    }

    /// The firing rate of a neuron is the proportion of steps at which it spikes.
    pub fn unit_rate(&self, unit: usize) -> f32 {
        if self.steps == 0 {
            return 0.0;
        }
        self.stats()
            .unit_counts
            .get(unit)
            .map_or(0.0, |&c| c as f32 / self.steps as f32)
    }

    /// How many neurons were spiked at step `step`.
    pub fn step_count(&self, step: usize) -> usize {
        self.stats()
            .step_counts
            .get(step)
            .map_or(0, |&c| c as usize)
    }

    /// Average frequency over all neurons.
    pub fn mean_rate(&self) -> f32 {
        if self.is_empty() {
            return 0.0;
        }
        (self.stats().total as f64 / (self.steps as f64 * self.units as f64)) as f32
    }

    /// The percentage of neurons that have not spiked ONCE.
    ///
    /// The main sign of degeneration: if it creeps towards unity, the layer
    /// goes silent and there will be no further gradient.
    pub fn silent_fraction(&self) -> f32 {
        if self.units == 0 {
            return 0.0;
        }
        self.stats().silent as f32 / self.units as f32
    }

    /// Number of neurons spiking at EACH step (frequency 1.0).
    ///
    /// Such a neuron physically cannot carry information about the input: it
    /// the output is the same regardless of what is supplied to the input. It's the same
    /// degeneration is the same as complete silence, only at the other end of the scale.
    pub fn saturated_units(&self) -> usize {
        self.stats().saturated
    }

    /// The extent to which the spikes are grouped in time and not spread evenly.
    ///
    /// Calculated as the ratio of peak to average number of spikes per step
    /// populations: `max(step_count) / mean(step_count)`. The network that
    /// encodes the input uniformly in time, this ratio is close to
    /// unit. If the entire population is spiked in one gulp on one or two
    /// steps and is silent on the rest, the peak is several times larger than average - hence
    /// threshold `3.0` used in [`super::health::assess`].
    ///
    /// Returns `1.0` if there are no digits at all (nothing to group).
    pub fn burstiness(&self) -> f32 {
        if self.steps == 0 || self.units == 0 {
            return 1.0;
        }
        let counts = &self.stats().step_counts;
        let peak = counts.iter().copied().max().unwrap_or(0) as f32;
        let mean = counts.iter().map(|&c| c as usize).sum::<usize>() as f32 / self.steps as f32;
        if mean <= 0.0 {
            return 1.0;
        }
        peak / mean
    }

    /// Neuron spike density, compressed to `cols` columns.
    ///
    /// There are usually more time steps than there are columns on the screen. Each
    /// the column receives a share of digits in its interval: zero - silence,
    /// one is the digit at each step of the interval. So the compression doesn't lose
    /// rare spikes, as the selection would lose at every nth step.
    pub fn unit_density(&self, unit: usize, cols: usize) -> Vec<f32> {
        if cols == 0 || self.steps == 0 {
            return Vec::new();
        }
        (0..cols)
            .map(|c| {
                let from = c * self.steps / cols;
                let to = ((c + 1) * self.steps / cols).max(from + 1).min(self.steps);
                let width = to.saturating_sub(from);
                if width == 0 {
                    return 0.0;
                }
                let fired = (from..to).filter(|&t| self.spiked(t, unit)).count();
                fired as f32 / width as f32
            })
            .collect()
    }

    /// Population activity in steps, compressed to `cols` columns.
    ///
    /// Value is the average proportion of spiked neurons in the interval.
    pub fn population_density(&self, cols: usize) -> Vec<f32> {
        if cols == 0 || self.steps == 0 || self.units == 0 {
            return Vec::new();
        }
        let counts = &self.stats().step_counts;
        (0..cols)
            .map(|c| {
                let from = c * self.steps / cols;
                let to = ((c + 1) * self.steps / cols).max(from + 1).min(self.steps);
                let width = to.saturating_sub(from);
                if width == 0 {
                    return 0.0;
                }
                let sum: usize = counts[from..to].iter().map(|&c| c as usize).sum();
                sum as f32 / (width * self.units) as f32
            })
            .collect()
    }

    /// Spike density on the grid `rows × cols`: the entire layer, compressed to
    /// screen size.
    ///
    /// Neurons are divided into `rows` consecutive groups, steps - into `cols`
    /// intervals; cell value is the proportion of “neuron × step” cells inside it,
    /// in which there was a digit, from 0 to 1. The order is row by row, `[row * cols +
    /// col]`. This is the only way to show a million neurons in their entirety:
    /// a point per neuron will not fit into a hundred lines, and the density is
    /// fits and maintains structure (strips of silence, volleys).
    ///
    /// Cost `O(words + digits)`; `rows` and `cols` are limited
    /// the size of the image, so there are no empty cells.
    pub fn bin_density(&self, rows: usize, cols: usize) -> Vec<f32> {
        if self.is_empty() || rows == 0 || cols == 0 {
            return Vec::new();
        }
        let rows = rows.min(self.units);
        let cols = cols.min(self.steps);

        let mut counts = vec![0u32; rows * cols];
        self.for_each_spike(|t, u| {
            counts[bin_of(u, self.units, rows) * cols + bin_of(t, self.steps, cols)] += 1;
        });

        let mut out = vec![0.0f32; rows * cols];
        for r in 0..rows {
            let h = bin_start(r + 1, self.units, rows) - bin_start(r, self.units, rows);
            for c in 0..cols {
                let w = bin_start(c + 1, self.steps, cols) - bin_start(c, self.steps, cols);
                let cells = (h * w).max(1);
                out[r * cols + c] = counts[r * cols + c] as f32 / cells as f32;
            }
        }
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn frame(steps: usize, units: usize, f: impl Fn(usize, usize) -> bool) -> SpikeFrame {
        SpikeFrame::from_fn(steps, units, f)
    }

    #[test]
    fn from_bits_and_from_fn_agree() {
        let f = |t: usize, u: usize| (t * 7 + u * 3).is_multiple_of(5);
        let bits: Vec<bool> = (0..13)
            .flat_map(|t| (0..11).map(move |u| (t, u)))
            .map(|(t, u)| f(t, u))
            .collect();
        let a = SpikeFrame::from_bits(13, 11, bits);
        let b = frame(13, 11, f);
        for t in 0..13 {
            for u in 0..11 {
                assert_eq!(a.spiked(t, u), b.spiked(t, u));
                assert_eq!(a.spiked(t, u), f(t, u));
            }
        }
        assert_eq!(a.total_spikes(), b.total_spikes());
    }

    #[test]
    fn statistics_match_brute_force() {
        // Size 13x11 = 143 bits - not a multiple of 64, checks the tail of the word.
        let f = |t: usize, u: usize| (t + 2 * u).is_multiple_of(4) || u == 3;
        let fr = frame(13, 11, f);

        let mut total = 0;
        for u in 0..11 {
            let cnt = (0..13).filter(|&t| f(t, u)).count();
            total += cnt;
            assert_eq!(fr.unit_counts()[u] as usize, cnt, "neuron {u}");
            assert!((fr.unit_rate(u) - cnt as f32 / 13.0).abs() < 1e-6);
        }
        for t in 0..13 {
            let cnt = (0..11).filter(|&u| f(t, u)).count();
            assert_eq!(fr.step_count(t), cnt, "step {t}");
        }
        assert_eq!(fr.total_spikes() as usize, total);
        assert!((fr.mean_rate() - total as f32 / 143.0).abs() < 1e-6);
    }

    #[test]
    fn silent_and_saturated_are_counted() {
        // neuron 0 is silent, neuron 1 always spikes, neuron 2 every other time
        let fr = frame(10, 3, |t, u| u == 1 || (u == 2 && t % 2 == 0));
        assert!((fr.silent_fraction() - 1.0 / 3.0).abs() < 1e-6);
        assert_eq!(fr.saturated_units(), 1);
    }

    #[test]
    fn out_of_range_queries_are_safe() {
        let fr = frame(4, 4, |_, _| true);
        assert!(!fr.spiked(4, 0));
        assert!(!fr.spiked(0, 4));
        assert_eq!(fr.unit_rate(99), 0.0);
        assert_eq!(fr.step_count(99), 0);
    }

    #[test]
    fn empty_frame_is_harmless() {
        let fr = SpikeFrame::default();
        assert!(fr.is_empty());
        assert_eq!(fr.mean_rate(), 0.0);
        assert_eq!(fr.silent_fraction(), 0.0);
        assert!(fr.bin_density(4, 4).is_empty());
        assert_eq!(fr.size_bytes(), 0);
    }

    #[test]
    fn bin_helpers_are_inverse() {
        for &(n, bins) in &[(10usize, 3usize), (1_000_000, 240), (7, 7), (100, 1)] {
            let mut prev = 0;
            for k in 0..n {
                let b = bin_of(k, n, bins);
                assert!(b < bins && b >= prev, "n={n} bins={bins} k={k}");
                assert!(
                    bin_start(b, n, bins) <= k && k < bin_start(b + 1, n, bins),
                    "n={n} bins={bins} k={k} b={b}"
                );
                prev = b;
            }
        }
    }

    #[test]
    fn bin_density_conserves_spikes_and_stays_in_range() {
        let fr = frame(24, 1000, |t, u| (t * 31 + u * 17) % 11 == 0);
        let (rows, cols) = (37, 10);
        let grid = fr.bin_density(rows, cols);
        assert_eq!(grid.len(), rows * cols);
        assert!(grid.iter().all(|&v| (0.0..=1.0).contains(&v)));

        // The sum (density × cell capacity) must return the number of digits.
        let mut recovered = 0.0f64;
        for r in 0..rows {
            let h = bin_start(r + 1, 1000, rows) - bin_start(r, 1000, rows);
            for c in 0..cols {
                let w = bin_start(c + 1, 24, cols) - bin_start(c, 24, cols);
                recovered += grid[r * cols + c] as f64 * (h * w) as f64;
            }
        }
        assert!((recovered - fr.total_spikes() as f64).abs() < 0.5);
    }

    #[test]
    fn bin_density_clamps_to_frame_size() {
        let fr = frame(3, 5, |_, _| true);
        let grid = fr.bin_density(100, 100);
        assert_eq!(
            grid.len(),
            5 * 3,
            "the grid cannot be larger than the photo"
        );
        assert!(grid.iter().all(|&v| (v - 1.0).abs() < 1e-6));
    }

    #[test]
    fn unit_density_keeps_sparse_spikes() {
        let fr = frame(8, 1, |t, _| t == 5);
        let d = fr.unit_density(0, 4);
        assert!(d.iter().any(|&x| x > 0.0));
        assert!((d.iter().sum::<f32>() - 0.5).abs() < 1e-6);
    }

    #[test]
    fn population_density_uses_all_neurons() {
        // at step 0 all 4 neurons are spiked, at step 1 - none
        let fr = frame(2, 4, |t, _| t == 0);
        assert_eq!(fr.population_density(2), vec![1.0, 0.0]);
    }

    #[test]
    fn burstiness_flags_synchrony() {
        let flat = frame(10, 10, |t, u| (t + u) % 2 == 0);
        let burst = frame(10, 10, |t, _| t == 3);
        assert!(flat.burstiness() < 1.5);
        assert!(burst.burstiness() > 3.0);
        assert_eq!(frame(4, 4, |_, _| false).burstiness(), 1.0);
    }

    #[test]
    fn million_neurons_fit_in_a_few_megabytes() {
        let fr = frame(24, 1_000_000, |t, u| (t + u) % 23 == 0);
        assert!(
            fr.size_bytes() <= 24 * 1_000_000 / 8 + 64,
            "{}",
            fr.size_bytes()
        );
        assert!(fr.total_spikes() > 0);
    }
}
