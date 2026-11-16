//! Quantization of the neuron STATE (not the weights).
//!
//! ## Why separately from `neo-quantization`
//!
//! [`neo-quantization`] deals with weights: they are static, quantized once
//! and stored packed. The membrane potential is the opposite - it
//! lives exactly one time step, is recomputed at every step, and is never
//! saved. Packing it into bits is pointless; the point is different:
//! LIMIT the bit depth at which it exists so that learning
//! happens under the same conditions in which the network will later run on
//! neuromorphic hardware, where 4–8 register bits are allocated to the membrane.
//!
//! This is quantization-in-training (QAT) for the state: the network learns in advance
//! to live with a coarse membrane instead of discovering it later at
//! inference. The approach is known as SQUAT.
//!
//! S. Venkatesh et al. (2024), «Squat: Stateful Quantization-Aware Training
//! in Recurrent Spiking Neural Networks».
//!
//! ## How does this not break learning?
//!
//! Rounding is not differentiable, so straight-through is used:
//! `q = x + (round(x) − x).detach()`. Forward is numerically equal to the rounded
//! value, backward passes straight through - the same technique as
//! surrogate gradients in [`crate::surrogate`].
//!
//! ## The threshold is a fixed quantization point
//!
//! `levels()` — the number of grid INTERVALS, `2^bits − 1`, always ODD.
//! For [`StateQuantScheme::ThresholdCentered`] and
//! [`StateQuantScheme::ThresholdExponential`] the threshold `θ` is exactly
//! the middle of the range `[θ−span, θ+span]`, so it falls on the middle of
//! the grid, index `levels/2`. For odd `levels` the middle is NOT an integer
//! index (`15/2 = 7.5`); rounding the ABSOLUTE position `x` on the
//! `[min, max]` grid would round this middle in one direction
//! (`round(7.5) = 8`), giving `quantize(θ) != θ`: a value exactly at the
//! threshold would receive a systematic offset `span/levels` (for example
//! `span/15` at 4 bits) to the same side. For a membrane, where the position
//! relative to the threshold is the neuron's decision, this would mean that
//! a state exactly at the threshold could become suprathreshold after
//! quantization.
//!
//! Rounding is therefore applied to the OFFSET from the threshold rather than
//! the absolute position on the grid: `round((x − θ)/scale)` instead of
//! `round((x − min)/scale)`. Offset `0` (that is, `x == θ`) always rounds to
//! `0`, so `quantize(θ) == θ` - exactly, for any parity of `levels`. The same
//! idea applies to the rectified coordinate `w` in
//! [`ThresholdExponential`](StateQuantScheme::ThresholdExponential):
//! `w == 0` (also corresponding to the threshold) is always quantized to `0`.
//!
//! Side effect: with this offset, the grid does not need to contain
//! `min`/`max` as exact nodes (the outermost quantized level can be a
//! fraction of a grid step away from `min`/`max`) - but this does not violate
//! the requirement of coarsening the state and does not affect the neuron's
//! decision, which is determined by the position relative to `θ`, not relative
//! to the edges of the range.
//!
//! ## How many REALLY unique codes do `ThresholdCentered`/`ThresholdExponential` give?
//!
//! `levels()` returns `2^bits − 1` - the number of grid INTERVALS, always
//! odd. For `Uniform` this directly gives `levels() + 1 = 2^bits`
//! distinguishable nodes on `[min, max]` - that is, an honest `bits`-bit
//! codebook (for example 16 levels at `bits = 4`).
//!
//! For `ThresholdCentered`/`ThresholdExponential` rounding is done by
//! offset `round(offset / scale)` with `half = levels * 0.5` as the
//! multiplier, and NOT by absolute position on the `[0, levels]` grid. Offset
//! indices run through integer values in `[-half, +half]`, which for
//! odd `levels` gives `half = (2^bits − 1)/2` - NOT an integer
//! (for example `7.5` at 4 bits). This means that the reachable integer indices are:
//! `{-7, -6, ..., 0, ..., 7}` (at 4 bits), i.e. `2 floor(half) + 1
//! = 2^bits − 1` unique levels, not `2^bits`. In other words:
//! `ThresholdCentered`/`ThresholdExponential` with `bits` bits has exactly
//! `2^bits − 1` distinguishable codes (15 at 4 bits, 255 at 8 bits),
//! ONE LESS than `Uniform` with the same `bits` (16 and 256
//! respectively) - that is the price paid for the threshold always falling
//! exactly in code `0`. This scheme CANNOT be called a strict
//! hardware `bits`-bit codebook in the sense of "`2^bits` states";
//! the correct formulation is "`2^bits − 1` states symmetric
//! about the threshold."
//!
//! There are no changes for [`StateQuantScheme::Uniform`]: there is no privileged
//! point around which it would be worth centering the rounding - only
//! `min`/`max`, so the absolute position on the grid remains as before.
//!
//! ## OFFSET rounding stays within the window
//!
//! `round(offset / scale)` rounds to the CLOSEST integer index, not
//! necessarily to an index INSIDE `[-half, +half]`. For odd `levels`
//! (`2^bits − 1`, always odd) `half = levels/2` is not an integer (`7.5` with
//! 4 bits), so at the very edge of the window, where
//! `offset/scale = ±half = ±7.5`, `round` gives `±8` - one MORE in absolute
//! value than the permissible extreme index `±7`. After multiplying back by
//! `scale` this would give `quantize(θ+span) ≈ θ + 1.0667·span` instead of
//! `θ + span` - the quantized value would fall outside the documented window
//! `[θ−span, θ+span]`.
//!
//! Clamping the rounded index to `[-half, +half]` BEFORE the reverse
//! multiplication by `scale` - `round(offset/scale).clamp(-half, half)` -
//! restores both invariants simultaneously:
//! * `quantize(x) ∈ [θ−span, θ+span]` for any `x` (including `θ±span`);
//! * the number of unique codes remains exactly `2^bits − 1` (without this clamp
//!   the extreme outliers `±8` add distinguishable values beyond the
//!   documented `{-7, …, 7}`).
//!
//! [`neo-quantization`]: https://docs.rs/neo-quantization

use burn::tensor::Tensor;
use burn::tensor::activation::relu;

use crate::state::NeuronState;

/// How to place quantization levels on a number line.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum StateQuantScheme {
    /// Uniform grid on the segment `[min, max]`.
    ///
    /// Simple option: suitable when the membrane range is known and
    /// the values are distributed more or less evenly across it.
    Uniform {
        /// Bottom border of the grid.
        min: f32,
        /// The upper border of the grid.
        max: f32,
    },
    /// Uniform grid CENTERED on threshold: `[θ − span, θ + span]`.
    ///
    /// The observation for which this is done: the neuron’s decision depends
    /// not on the size of the membrane, but on its position RELATIVE to the threshold.
    /// A meaning that has gone far down still only means “silent”
    /// and there is no need to spend grid levels on it. Entire bit budget
    /// goes to the vicinity of the threshold, where the decision is made.
    ///
    /// Exponential thickening of levels to the threshold (as in the reference
    /// implementation of SQUAT) is NOT implemented here: the grid inside the window remains
    /// uniform.
    ThresholdCentered {
        /// Spike threshold θ.
        threshold: f32,
        /// Half-width of the window around the threshold.
        span: f32,
    },
    /// A grid EXPONENTIALLY CONDENSING towards the threshold.
    ///
    /// Continuation of the same idea as [`ThresholdCentered`], brought to
    /// end. There the bit budget is spent on the window around the threshold, but
    /// inside the window the levels are spaced equally - although the value is at the edge of the window
    /// and there it just means “confidently silent.”
    ///
    /// Here the levels are set more densely, the closer to the threshold: solution
    /// neuron is determined by a small neighborhood θ, and it is this neighborhood that gets
    /// resolution. Distant meanings become coarser, but their coarsening does not matter
    /// influences.
    ///
    /// Implemented by scaling: the value is converted to rectified
    /// coordinate `w = sign(v)·|v|^(1/γ)`, where `v = (x − θ)/span`,
    /// quantizes there UNIFORMLY and returns back
    /// `v̂ = sign(w)·|w|^γ`. For `γ = 1` the scheme degenerates exactly to
    /// [`ThresholdCentered`].
    ///
    /// `gamma > 1` - condensation to the threshold; the more, the sharper.
    /// Values 2...3 are usually sufficient: with `γ = 2` and 4 bits, the closest to
    /// threshold the level is separated from it by `(2/15)²·span ≈ 0.018·span`
    /// instead of `0.067·span` for a uniform grid.
    ThresholdExponential {
        /// Spike threshold θ.
        threshold: f32,
        /// Half-width of the window around the threshold.
        span: f32,
        /// Condensation index γ ≥ 1.
        gamma: f32,
    },
}

impl StateQuantScheme {
    /// Grid boundaries `(min, max)`.
    fn bounds(&self) -> (f32, f32) {
        match *self {
            StateQuantScheme::Uniform { min, max } => (min, max),
            StateQuantScheme::ThresholdCentered { threshold, span } => {
                (threshold - span, threshold + span)
            }
            StateQuantScheme::ThresholdExponential {
                threshold, span, ..
            } => (threshold - span, threshold + span),
        }
    }
}

/// State quantization configuration.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct StateQuant {
    /// Number of bits per value. 2...8 are meaningful.
    pub bits: u8,
    /// Layout of levels.
    pub scheme: StateQuantScheme,
}

impl Default for StateQuant {
    fn default() -> Self {
        Self {
            bits: 8,
            scheme: StateQuantScheme::ThresholdCentered {
                threshold: 1.0,
                span: 1.0,
            },
        }
    }
}

impl StateQuant {
    /// New configuration with uniform grid.
    pub fn uniform(bits: u8, min: f32, max: f32) -> Self {
        Self {
            bits,
            scheme: StateQuantScheme::Uniform { min, max },
        }
    }

    /// New configuration with mesh around the threshold.
    pub fn threshold_centered(bits: u8, threshold: f32, span: f32) -> Self {
        Self {
            bits,
            scheme: StateQuantScheme::ThresholdCentered { threshold, span },
        }
    }

    /// New configuration with condensation of levels towards the threshold.
    ///
    /// `gamma` is clamped from below by one: `γ < 1` would thin out the grid
    /// near the threshold, that is, it would do exactly the opposite of what was intended.
    pub fn threshold_exponential(bits: u8, threshold: f32, span: f32, gamma: f32) -> Self {
        Self {
            bits,
            scheme: StateQuantScheme::ThresholdExponential {
                threshold,
                span,
                gamma: gamma.max(1.0),
            },
        }
    }

    /// Number of grid levels, `2^bits − 1` intervals.
    pub fn levels(&self) -> f32 {
        let bits = self.bits.clamp(1, 16);
        ((1u32 << bits) - 1) as f32
    }

    /// Grid pitch.
    ///
    /// For uniform patterns, this is a step everywhere. For
    /// [`ThresholdExponential`](StateQuantScheme::ThresholdExponential)
    /// the grid is uneven, and the SMALLEST step is returned - the one
    /// the threshold itself; away from the threshold it is larger.
    pub fn step(&self) -> f32 {
        let (min, max) = self.scheme.bounds();
        match self.scheme {
            StateQuantScheme::ThresholdExponential { span, gamma, .. } => {
                // The level of the straightened grid closest to the threshold is separated from
                // zero to 2/L; reverse scaling gives (2/L)^γ.
                span * (2.0 / self.levels()).powf(gamma)
            }
            _ => (max - min) / self.levels(),
        }
    }

    /// Quantize a tensor of any rank with a straight-through gradient.
    pub fn apply<const D: usize>(&self, x: Tensor<D>) -> Tensor<D> {
        let (min, max) = self.scheme.bounds();
        assert!(
            max > min,
            "StateQuant: empty range [{min}, {max}] - the mesh is degenerate"
        );
        let levels = self.levels();

        let quantized = match self.scheme {
            StateQuantScheme::ThresholdExponential {
                threshold,
                span,
                gamma,
            } => {
                let gamma = gamma.max(1.0);

                // v = (x − θ)/span, sandwiched in [−1, 1].
                let v = x
                    .clone()
                    .sub_scalar(threshold)
                    .div_scalar(span)
                    .clamp(-1.0, 1.0);

                // Rectifying overscale w = sign(v)·|v|^(1/γ),
                // written through relu: the sign is taken by splitting into two
                // non-negative branches, so powf never gets
                // negative reason.
                let w = warp(v, 1.0 / gamma);

                // Uniformly quantize w on [−1, 1], rounding the OFFSET
                // from the center (w=0, which corresponds to the threshold x=θ), not the
                // absolute position on the grid [0, levels]: for odd
                // levels the middle of the range (levels/2) is not an integer, and
                // rounding the absolute position would systematically shift
                // w=0 to one side. round(0) == 0 always,
                // so the center of the grid exactly coincides with the threshold.
                let half = levels * 0.5;
                let wq = w.mul_scalar(half).round().div_scalar(half).clamp(-1.0, 1.0);

                // Reverse scaling and return to original units.
                warp(wq, gamma).mul_scalar(span).add_scalar(threshold)
            }
            StateQuantScheme::ThresholdCentered { threshold, span } => {
                let scale = (2.0 * span) / levels;
                // Round the OFFSET from the threshold, not the absolute position on
                // grid [min, max] - the same reason and the same technique as
                // above for ThresholdExponential: round(0) == 0 guarantees
                // quantize(threshold) == threshold regardless of parity
                // levels, whereas round((threshold - min)/scale) with
                // odd levels are always exactly in the middle of the range
                // rounded to one side.
                //
                // Rounding the offset can select the outermost
                // index OUTWARD - half = levels/2 is not an integer for odd
                // levels (7.5 at 4 bits), so offset/scale = ±7.5 at the
                // edge of the window rounds to ±8 rather than ±7, and the resulting value
                // after multiplying by scale goes beyond [threshold-span,
                // threshold+span] (1.0667·span instead of span at 4 bits).
                // Clamping the NORMALIZED index (before multiplication
                // back to scale) keeps it at [-half, +half]:
                // this keeps the value inside the window and avoids
                // extra codes at the edge beyond the documented
                // 2^bits - 1 unique levels.
                let half = levels * 0.5;
                let offset = x.clone().sub_scalar(threshold).clamp(-span, span);
                offset
                    .div_scalar(scale)
                    .round()
                    .clamp(-half, half)
                    .mul_scalar(scale)
                    .add_scalar(threshold)
            }
            StateQuantScheme::Uniform { min, max } => {
                // Scheme without preferred point - absolute position
                // on the [min, max] grid as before.
                let scale = (max - min) / levels;
                x.clone()
                    .clamp(min, max)
                    .sub_scalar(min)
                    .div_scalar(scale)
                    .round()
                    .mul_scalar(scale)
                    .add_scalar(min)
            }
        };

        // Straight-through: the quantized value is taken, the gradient is from x.
        x.clone() + (quantized - x).detach()
    }
}

impl NeuronState {
    /// Quantize the membrane (and synaptic current, if any).
    ///
    /// Called between time steps, immediately after
    /// [`step`](crate::neuron::Leaky::step): state transitioning to
    /// the next step turns out to be already coarsened - exactly as it
    /// would be stored in the hardware register.
    pub fn quantize(self, config: &StateQuant) -> Self {
        Self {
            mem: config.apply(self.mem),
            syn: self.syn.map(|s| config.apply(s)),
        }
    }
}

/// Odd power overscale `sign(v)·|v|^p`.
///
/// Splitting into `relu(v)` and `relu(−v)` eliminates the call to `sign`: both
/// the branches are non-negative, so raising to a fractional power is defined,
/// and the sign is restored by subtraction. The function is monotonic and translates
/// `[−1, 1]` into itself for any `p > 0`.
fn warp<const D: usize>(v: Tensor<D>, p: f32) -> Tensor<D> {
    let pos = relu(v.clone()).powf_scalar(p);
    let neg = relu(v.mul_scalar(-1.0)).powf_scalar(p);
    pos - neg
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::Device;

    fn dev() -> Device {
        Device::flex()
    }

    fn t1(data: Vec<f32>) -> Tensor<1> {
        Tensor::<1>::from_floats(data.as_slice(), &dev())
    }

    fn vec1(t: Tensor<1>) -> Vec<f32> {
        t.into_data().try_to_vec::<f32>().unwrap()
    }

    /// The value exactly at the threshold must quantize to itself for
    /// `ThresholdCentered`, regardless of the parity of `levels`
    /// (`2^bits - 1` is always odd).
    #[test]
    fn threshold_centered_is_a_fixed_point_at_4_bits() {
        let q = StateQuant::threshold_centered(4, 1.0, 1.0);
        let out = vec1(q.apply(t1(vec![1.0])));
        assert!(
            (out[0] - 1.0).abs() < 1e-6,
            "quantize(threshold) should be == threshold, obtained {}",
            out[0]
        );
    }

    /// The same invariant at 8 bits; only the offset value differs
    /// (`span/255` instead of `span/15`).
    #[test]
    fn threshold_centered_is_a_fixed_point_at_8_bits() {
        let q = StateQuant::threshold_centered(8, 1.0, 1.0);
        let out = vec1(q.apply(t1(vec![1.0])));
        assert!(
            (out[0] - 1.0).abs() < 1e-6,
            "quantize(threshold) should be == threshold, obtained {}",
            out[0]
        );
    }

    /// The same invariant for `ThresholdExponential`, which centers rounding
    /// on `w == 0`.
    #[test]
    fn threshold_exponential_is_a_fixed_point() {
        let q = StateQuant::threshold_exponential(4, 1.0, 1.0, 2.0);
        let out = vec1(q.apply(t1(vec![1.0])));
        assert!(
            (out[0] - 1.0).abs() < 1e-6,
            "quantize(threshold) should be == threshold, obtained {}",
            out[0]
        );
    }

    /// The invariant holds for arbitrary threshold and span, beyond the
    /// `threshold=1.0, span=1.0` example.
    #[test]
    fn threshold_centered_fixed_point_holds_for_arbitrary_threshold_and_span() {
        let q = StateQuant::threshold_centered(6, -0.3, 2.5);
        let out = vec1(q.apply(t1(vec![-0.3])));
        assert!((out[0] - (-0.3)).abs() < 1e-5, "received {}", out[0]);
    }

    /// `Uniform` quantizes the absolute position on the `[min, max]` grid,
    /// not an offset from some point: it has no privileged point like a
    /// threshold.
    ///
    /// `Tensor::round()` in this backend rounds half to even, so
    /// `round(7.5) == 7`; index 7 is the expected rounded result. If a burn
    /// backend changes the rounding mode to half-away-from-zero, this test
    /// will report it first.
    #[test]
    fn uniform_scheme_quantizes_absolute_position() {
        let q = StateQuant::uniform(4, 0.0, 2.0);
        // Grid: 15 intervals per [0,2], step = 2/15 ≈ 0.1333.
        // x=1.0 -> (1.0-0.0)/scale = 7.5 -> round(7.5) = 7 (round-to-even
        // or half-down in this backend) -> 7*scale ≈ 0.9333.
        let out = vec1(q.apply(t1(vec![1.0])));
        let expected = 7.0 * (2.0 / 15.0);
        assert!(
            (out[0] - expected).abs() < 1e-4,
            "The uniform scheme should not have changed: {}",
            out[0]
        );
    }

    /// The quantized values at the window edges (`threshold ± span`) stay
    /// within one grid step of the boundary. The tolerance of one step is a
    /// weaker condition that is robust to future changes.
    #[test]
    fn threshold_centered_upper_bound_close_to_span_edge() {
        let q = StateQuant::threshold_centered(4, 1.0, 1.0);
        let out = vec1(q.apply(t1(vec![2.0]))); // x = threshold + span
        let step = q.step();
        assert!(
            (out[0] - 2.0).abs() <= step,
            "the upper boundary must remain within one grid step: {} (step={step})",
            out[0]
        );
    }

    /// `quantize(threshold + span)` must not exceed `threshold + span`.
    /// Rounding the offset at the window edge (`offset/scale = half = 7.5` at
    /// 4 bits) could otherwise produce index `8` instead of `7`, giving
    /// `≈ 2.0667` instead of `2.0` with `threshold=1.0, span=1.0`. The check is
    /// strictly `<=`, not merely "within the grid step" as in the previous test.
    #[test]
    fn threshold_centered_does_not_overshoot_span_edge() {
        let q = StateQuant::threshold_centered(4, 1.0, 1.0);
        let out = vec1(q.apply(t1(vec![2.0]))); // x = threshold + span
        assert!(
            out[0] <= 2.0 + 1e-5,
            "quantize(threshold+span) should not go outside the window: {}",
            out[0]
        );
        let out_neg = vec1(q.apply(t1(vec![0.0]))); // x = threshold - span
        assert!(
            out_neg[0] >= 0.0 - 1e-5,
            "quantize(threshold-span) should not go outside the window: {}",
            out_neg[0]
        );
    }

    /// `ThresholdExponential` has the same edge requirement; the protective
    /// `wq.clamp(-1.0, 1.0)` in the source preserves it.
    #[test]
    fn threshold_exponential_does_not_overshoot_span_edge() {
        let q = StateQuant::threshold_exponential(4, 1.0, 1.0, 2.5);
        let out = vec1(q.apply(t1(vec![2.0]))); // x = threshold + span
        assert!(
            out[0] <= 2.0 + 1e-5,
            "quantize(threshold+span) should not go outside the window: {}",
            out[0]
        );
    }

    /// Documents and records the actual number of unique codes for
    /// `ThresholdCentered`: `2^bits − 1`, and NOT `2^bits` - for one code
    /// less than `Uniform` with the same number of bits. Scans the range
    /// dense grid and counts the number of distinguishable quantized values.
    #[test]
    fn threshold_centered_has_2_pow_bits_minus_1_unique_codes() {
        let bits = 4u8;
        let q = StateQuant::threshold_centered(bits, 0.0, 1.0);

        let n_samples = 100_001; // denser than the grid pitch, with a margin.
        let samples: Vec<f32> = (0..n_samples)
            .map(|i| -1.5 + 3.0 * (i as f32) / (n_samples - 1) as f32)
            .collect();
        let out = vec1(q.apply(t1(samples)));

        let mut unique: Vec<i64> = out
            .iter()
            .map(|&v| (v / (q.step() * 0.1)).round() as i64) // quantize to grid for deduplication
            .collect();
        unique.sort_unstable();
        unique.dedup();

        let expected = (1u32 << bits) - 1; // 2^bits - 1, NOT 2^bits.
        assert_eq!(
            unique.len() as u32,
            expected,
            "ThresholdCentered with {bits} bits must give 2^bits-1={expected} unique codes, received {}",
            unique.len()
        );
    }

    /// The same score for `Uniform`: obliged to give an honest `2^bits` codebook
    /// (one code more than `ThresholdCentered` with the same bits) -
    /// direct comparison confirming the asymmetry from the module documentation.
    #[test]
    fn uniform_has_2_pow_bits_unique_codes() {
        let bits = 4u8;
        let q = StateQuant::uniform(bits, -1.5, 1.5);

        let n_samples = 100_001;
        let samples: Vec<f32> = (0..n_samples)
            .map(|i| -1.5 + 3.0 * (i as f32) / (n_samples - 1) as f32)
            .collect();
        let out = vec1(q.apply(t1(samples)));

        let mut unique: Vec<i64> = out
            .iter()
            .map(|&v| (v / (q.step() * 0.1)).round() as i64)
            .collect();
        unique.sort_unstable();
        unique.dedup();

        let expected = 1u32 << bits; // 2^bits, honest codebook.
        assert_eq!(
            unique.len() as u32,
            expected,
            "Uniform with {bits} bits must give 2^bits={expected} unique codes, received {}",
            unique.len()
        );
    }

    /// Monotonicity: Quantization should not rearrange the order of values.
    #[test]
    fn quantization_preserves_order() {
        let q = StateQuant::threshold_centered(5, 0.0, 3.0);
        let out = vec1(q.apply(t1(vec![-2.0, -1.0, 0.0, 1.0, 2.0])));
        for pair in out.windows(2) {
            assert!(pair[0] <= pair[1], "the order is broken: {out:?}");
        }
    }
}
