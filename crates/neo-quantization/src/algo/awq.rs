// neo-quantization/src/algo/awq.rs
//
// AWQ — activation-aware weight quantization.
//
// Lin et al., «AWQ: Activation-aware Weight Quantization for LLM
// Compression and Acceleration», MLSys 2024 (arXiv:2306.00978).
//
// ## What's the idea
//
// Authors' observation: the quality of the 4-bit model is determined by a small
// shares of scales - those that are multiplied by BIG activations. The error is in them
// gets into the output with a large coefficient, the error in the rest is almost the same
// visible In this case, the importance of weight is determined not by its size, but by the size
// corresponding INPUT.
//
// Keeping some of the weights in high precision is inconvenient: mixed bit depth
// breaks the memory and kernel layout. AWQ solves differently - by scaling.
// For the linear layer `y = Wx` is true
//
//     W·x = (W·diag(s)) · (diag(s)⁻¹·x)
//
// for any positive `s`. Multiplying important input channels by `s > 1`
// BEFORE quantization, we increase their weight within the group, and the grid
// quantization gives them more resolution. The inverse multiplier goes to
// activation, where it can be added to the previous operation (normalization
// or the previous linear layer) and get it for free.
//
// The scales are searched as `s = mean|x|^α`, where `α ∈ [0,1]` is selected
// Lattice search: minimizes the layer output error. `α = 0` - all
// channels are equal (usual quantization), `α = 1` - scale strictly
// proportional to the channel activity.

use std::vec::Vec;

use super::{dequantize_group, quantize_group, GroupQuant};

/// AWQ configuration.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct AwqConfig {
    /// Number of bits per weight; canonical mode - 4.
    pub bits: u8,
    /// The size of a group of weights per scale. `0` - one group per line.
    pub group_size: usize,
    /// How many values of `α` to sort on the interval [0, 1].
    ///
    /// In the reference implementation 20. Brute force is cheap: every point costs
    /// one quantization pass plus error calculation.
    pub search_steps: usize,
    /// Should the scale be limited from above?
    ///
    /// Without limitation, a channel with an abnormally large activation receives
    /// a huge multiplier, the remaining channels of the group are compressed into several
    /// grid levels, and winnings turn into losses. Upper limit
    /// is given by [`AwqConfig::max_scale`].
    pub clamp_scales: bool,
    /// Maximum scale ratio within a layer.
    pub max_scale: f32,
}

impl Default for AwqConfig {
    fn default() -> Self {
        Self {
            bits: 4,
            group_size: 128,
            search_steps: 20,
            clamp_scales: true,
            max_scale: 16.0,
        }
    }
}

impl AwqConfig {
    /// Canonical mode: 4 bits, groups of 128, search of 20 points.
    pub fn int4_g128() -> Self {
        Self::default()
    }

    /// Set the number of bits.
    pub fn with_bits(mut self, bits: u8) -> Self {
        self.bits = bits;
        self
    }

    /// Set the group size (`0` - one group per line).
    pub fn with_group_size(mut self, group_size: usize) -> Self {
        self.group_size = group_size;
        self
    }

    /// Set the number of search points `α`.
    pub fn with_search_steps(mut self, steps: usize) -> Self {
        self.search_steps = steps;
        self
    }
}

/// Why AWQ could not work.
#[derive(Debug, Clone, PartialEq)]
pub enum AwqError {
    /// Calibration activations were not transferred.
    ///
    /// Like GPTQ, failure instead of silent rollback: the whole point of AWQ is
    /// which channels are active, and without activations it degenerates into normal
    /// rounding.
    NoCalibration,
    /// Sizes are not consistent.
    ShapeMismatch(String),
    /// The number of bits is outside the supported range.
    UnsupportedBits(u8),
}

impl core::fmt::Display for AwqError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            AwqError::NoCalibration => write!(
                f,
                "AWQ requires calibration activations: without them it is not known which channels are important"
            ),
            AwqError::ShapeMismatch(m) => write!(f, "inconsistent dimensions: {m}"),
            AwqError::UnsupportedBits(b) => {
                write!(f, "AWQ: {b} bits out of supported range 2..=8")
            }
        }
    }
}

impl std::error::Error for AwqError {}

/// AWQ result for one layer.
#[derive(Debug, Clone, PartialEq)]
pub struct AwqResult {
    /// Quantized weights `[out_features, in_features]` in ORIGINAL
    /// scale: the inverse multiplier has already been applied, so they can be
    /// use directly without changing anything in the rest of the model.
    pub weights: Vec<f32>,
    /// Found scales based on input channels, `[in_features]`.
    ///
    /// Needed if you want to get acceleration “for real”: fold
    /// `s⁻¹` with the previous operation and store the weights already scaled.
    /// They are not required for simple application of the result.
    pub channel_scales: Vec<f32>,
    /// The winning indicator is `α`.
    pub alpha: f32,
    /// The layer output error in this case is `α` - according to the calibration data.
    pub output_error: f64,
    /// Number of output features.
    pub out_features: usize,
    /// Number of input features.
    pub in_features: usize,
}

impl AwqResult {
    /// The standard deviation of the quantized weights from the original ones.
    pub fn weight_mse(&self, original: &[f32]) -> f64 {
        original
            .iter()
            .zip(&self.weights)
            .map(|(a, b)| ((a - b) as f64).powi(2))
            .sum::<f64>()
            / original.len().max(1) as f64
    }
}

/// Quantize the layer using the AWQ algorithm.
///
/// * `weights` — `[out_features, in_features]`, row-major.
/// * `activations` — `[n_samples, in_features]`, row-major.
///
/// # Example
///
/// ```
/// use neo_quantization::algo::{awq_quantize, AwqConfig};
///
/// let (out_features, in_features) = (8, 16);
/// let weights: Vec<f32> = (0..out_features * in_features)
///     .map(|i| ((i * 37 % 101) as f32 - 50.0) / 50.0)
///     .collect();
/// let activations: Vec<f32> = (0..256 * in_features)
///     .map(|i| ((i * 13 % 71) as f32 - 35.0) / 35.0)
///     .collect();
///
/// let result = awq_quantize(
///     &weights,
///     out_features,
///     in_features,
///     &activations,
///     &AwqConfig::int4_g128(),
/// ).unwrap();
///
/// assert_eq!(result.weights.len(), weights.len());
/// assert!((0.0..=1.0).contains(&result.alpha));
/// ```
pub fn awq_quantize(
    weights: &[f32],
    out_features: usize,
    in_features: usize,
    activations: &[f32],
    config: &AwqConfig,
) -> Result<AwqResult, AwqError> {
    if !(2..=8).contains(&config.bits) {
        return Err(AwqError::UnsupportedBits(config.bits));
    }
    if activations.is_empty() {
        return Err(AwqError::NoCalibration);
    }
    if weights.len() != out_features * in_features {
        return Err(AwqError::ShapeMismatch(format!(
            "weights {}, expected {out_features}×{in_features} = {}",
            weights.len(),
            out_features * in_features
        )));
    }
    if !activations.len().is_multiple_of(in_features) {
        return Err(AwqError::ShapeMismatch(format!(
            "activations {}, not a multiple of in_features {in_features}",
            activations.len()
        )));
    }

    // ── Step 1: channel importance = average activation value ────────────
    let importance = channel_importance(activations, in_features);

    // ── Step 2: search for α ──────────────────────── ────────────────────────
    let steps = config.search_steps.max(1);
    let mut best: Option<(f32, Vec<f32>, Vec<f32>, f64)> = None;

    for step in 0..=steps {
        let alpha = step as f32 / steps as f32;
        let scales = channel_scales(&importance, alpha, config);
        let quantized = quantize_with_scales(weights, out_features, in_features, &scales, config);
        let err = super::output_mse(weights, &quantized, activations, out_features, in_features);

        let better = match &best {
            None => true,
            Some((_, _, _, best_err)) => err < *best_err,
        };
        if better && err.is_finite() {
            best = Some((alpha, scales, quantized, err));
        }
    }

    // Enumeration always includes α = 0 (usual quantization), so empty
    // the result cannot be - unless the error turned out to be non-numeric
    // at all points, which would mean non-numeric input data.
    let (alpha, channel_scales, quantized, output_error) = best.ok_or_else(|| {
        AwqError::ShapeMismatch("on all α the error was non-numeric: check the input data".into())
    })?;

    Ok(AwqResult {
        weights: quantized,
        channel_scales,
        alpha,
        output_error,
        out_features,
        in_features,
    })
}

/// Average activation value for each input channel.
fn channel_importance(activations: &[f32], in_features: usize) -> Vec<f32> {
    let n_samples = activations.len() / in_features;
    let mut acc = vec![0.0f64; in_features];
    for s in 0..n_samples {
        let row = &activations[s * in_features..(s + 1) * in_features];
        for (c, &v) in row.iter().enumerate() {
            if v.is_finite() {
                acc[c] += v.abs() as f64;
            }
        }
    }
    let n = n_samples.max(1) as f64;
    acc.into_iter().map(|v| (v / n) as f32).collect()
}

/// Channel scales `s = importance^α`, normalized to mean 1.
///
/// Normalization is mandatory: without it, the common factor goes away along with `α`,
/// and it is incorrect to compare the search points with each other - they would change immediately
/// and the shape of the scale distribution and its magnitude.
fn channel_scales(importance: &[f32], alpha: f32, config: &AwqConfig) -> Vec<f32> {
    if alpha == 0.0 {
        return vec![1.0; importance.len()];
    }

    let mut s: Vec<f32> = importance
        .iter()
        .map(|&v| {
            // The channel that was never used by calibration receives
            // neutral scale: raising zero to a power is pointless,
            // and to reset the scale means to divide by zero when going back.
            if v > 0.0 && v.is_finite() {
                v.powf(alpha)
            } else {
                1.0
            }
        })
        .collect();

    let mean = s.iter().map(|&v| v as f64).sum::<f64>() / s.len().max(1) as f64;
    if mean > 0.0 && mean.is_finite() {
        for v in s.iter_mut() {
            *v /= mean as f32;
        }
    }

    if config.clamp_scales {
        let lo = 1.0 / config.max_scale;
        for v in s.iter_mut() {
            *v = v.clamp(lo, config.max_scale);
        }
    }
    s
}

/// Quantize the weights by first multiplying the input channels by `scales`,
/// and return the result at the original scale.
fn quantize_with_scales(
    weights: &[f32],
    out_features: usize,
    in_features: usize,
    scales: &[f32],
    config: &AwqConfig,
) -> Vec<f32> {
    let group = if config.group_size == 0 {
        in_features
    } else {
        config.group_size
    };
    let levels = (1i32 << config.bits) - 1;
    let mut out = vec![0.0f32; weights.len()];

    for r in 0..out_features {
        for gs in (0..in_features).step_by(group) {
            let end = (gs + group).min(in_features);

            // Scaled group weights.
            let scaled: Vec<f64> = (gs..end)
                .map(|c| weights[r * in_features + c] as f64 * scales[c] as f64)
                .collect();
            let gq: GroupQuant = quantize_group(&scaled, config.bits);

            for (k, c) in (gs..end).enumerate() {
                let code = if gq.scale == 0.0 {
                    0
                } else {
                    (((scaled[k] - gq.zero) / gq.scale).round() as i32).clamp(0, levels)
                };
                let restored = dequantize_group(code, &gq);
                // Reverse multiplier: return the weight to the original scale.
                let s = scales[c] as f64;
                out[r * in_features + c] = if s != 0.0 && s.is_finite() {
                    (restored / s) as f32
                } else {
                    restored as f32
                };
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::algo::{output_mse, round_to_nearest};

    fn lcg(n: usize, seed: u64) -> Vec<f32> {
        let mut s = seed;
        (0..n)
            .map(|_| {
                s = s
                    .wrapping_mul(6364136223846793005)
                    .wrapping_add(1442695040888963407);
                ((s >> 40) as f32) / ((1u64 << 24) as f32) - 0.5
            })
            .collect()
    }

    /// Activations with EMISSIONS across channels - a scenario for which AWQ and
    /// invented: some of the input channels are systematically much more active
    /// the rest.
    fn activations_with_channel_outliers(n_samples: usize, in_f: usize, seed: u64) -> Vec<f32> {
        let mut x = lcg(n_samples * in_f, seed);
        for s in 0..n_samples {
            for c in 0..in_f {
                // Every eighth channel is 20 times more active than others.
                if c % 8 == 0 {
                    x[s * in_f + c] *= 20.0;
                }
            }
        }
        x
    }

    #[test]
    fn test_awq_requires_calibration() {
        let w = lcg(64, 1);
        let err = awq_quantize(&w, 4, 16, &[], &AwqConfig::default()).unwrap_err();
        assert_eq!(err, AwqError::NoCalibration);
        assert!(err.to_string().contains("channels"));
    }

    #[test]
    fn test_awq_rejects_bad_shapes_and_bits() {
        let w = lcg(64, 1);
        let x = lcg(160, 2);
        assert!(matches!(
            awq_quantize(&w, 4, 17, &x, &AwqConfig::default()).unwrap_err(),
            AwqError::ShapeMismatch(_)
        ));
        assert_eq!(
            awq_quantize(&w, 4, 16, &x, &AwqConfig::default().with_bits(9)).unwrap_err(),
            AwqError::UnsupportedBits(9)
        );
    }

    #[test]
    fn test_awq_preserves_shape_and_finiteness() {
        let (out_f, in_f) = (8, 32);
        let w = lcg(out_f * in_f, 11);
        let x = lcg(256 * in_f, 12);
        let r = awq_quantize(
            &w,
            out_f,
            in_f,
            &x,
            &AwqConfig::default().with_group_size(32),
        )
        .unwrap();

        assert_eq!(r.weights.len(), w.len());
        assert_eq!(r.channel_scales.len(), in_f);
        assert!(r.weights.iter().all(|v| v.is_finite()));
        assert!(r.channel_scales.iter().all(|v| v.is_finite() && *v > 0.0));
    }

    /// Main check: on activations with emissions through channels, AWQ is obliged to
    /// beat the usual rounding of the same bit depth.
    #[test]
    fn test_awq_beats_round_to_nearest_with_channel_outliers() {
        let (out_f, in_f) = (16, 64);
        let w = lcg(out_f * in_f, 7);
        let x = activations_with_channel_outliers(512, in_f, 8);

        let cfg = AwqConfig::int4_g128().with_group_size(32);
        let awq = awq_quantize(&w, out_f, in_f, &x, &cfg).unwrap();
        let baseline = round_to_nearest(&w, out_f, in_f, cfg.bits, 32);

        let err_awq = output_mse(&w, &awq.weights, &x, out_f, in_f);
        let err_rtn = output_mse(&w, &baseline, &x, out_f, in_f);

        assert!(
            err_awq < err_rtn,
            "AWQ ({err_awq:.6e} output error) must beat round-to-nearest ({err_rtn:.6e}); α={}",
            awq.alpha
        );
    }

    /// The search must find a non-trivial α where the channels are unequal.
    /// α = 0 means “scaling did not help” - for data with
    /// outliers are a sign that the search is not working.
    #[test]
    fn test_awq_finds_nontrivial_alpha_on_skewed_channels() {
        let (out_f, in_f) = (16, 64);
        let w = lcg(out_f * in_f, 21);
        let x = activations_with_channel_outliers(512, in_f, 22);

        let r = awq_quantize(
            &w,
            out_f,
            in_f,
            &x,
            &AwqConfig::default().with_group_size(32),
        )
        .unwrap();
        assert!(
            r.alpha > 0.0,
            "non-zero α was expected, {} was obtained",
            r.alpha
        );
        // The scales must vary between channels, otherwise they are of no use
        // do not influence.
        let min = r
            .channel_scales
            .iter()
            .cloned()
            .fold(f32::INFINITY, f32::min);
        let max = r.channel_scales.iter().cloned().fold(0.0f32, f32::max);
        assert!(
            max / min > 1.5,
            "the channel scales are almost the same: {min}..{max}"
        );
    }

    /// On uniform activations, scaling cannot help, and AWQ
    /// should not do WORSE than usual rounding: the search includes α = 0.
    #[test]
    fn test_awq_never_worse_than_baseline_on_uniform_channels() {
        let (out_f, in_f) = (8, 32);
        let w = lcg(out_f * in_f, 31);
        let x = lcg(512 * in_f, 32);

        let cfg = AwqConfig::default().with_group_size(32);
        let awq = awq_quantize(&w, out_f, in_f, &x, &cfg).unwrap();
        let baseline = round_to_nearest(&w, out_f, in_f, cfg.bits, 32);

        let err_awq = output_mse(&w, &awq.weights, &x, out_f, in_f);
        let err_rtn = output_mse(&w, &baseline, &x, out_f, in_f);
        assert!(
            err_awq <= err_rtn * 1.0001,
            "AWQ worsened the result: {err_awq:.6e} vs. {err_rtn:.6e}"
        );
    }

    #[test]
    fn test_awq_more_bits_lower_error() {
        let (out_f, in_f) = (8, 32);
        let w = lcg(out_f * in_f, 41);
        let x = lcg(256 * in_f, 42);

        let mut prev = f64::INFINITY;
        for bits in [2u8, 3, 4, 6, 8] {
            let cfg = AwqConfig::default().with_bits(bits).with_group_size(32);
            let r = awq_quantize(&w, out_f, in_f, &x, &cfg).unwrap();
            assert!(
                r.output_error <= prev * 1.05,
                "{bits} bit worse than previous"
            );
            prev = r.output_error;
        }
    }

    #[test]
    fn test_awq_is_deterministic() {
        let (out_f, in_f) = (4, 16);
        let w = lcg(out_f * in_f, 51);
        let x = lcg(256 * in_f, 52);
        let cfg = AwqConfig::default().with_group_size(16);
        assert_eq!(
            awq_quantize(&w, out_f, in_f, &x, &cfg).unwrap(),
            awq_quantize(&w, out_f, in_f, &x, &cfg).unwrap()
        );
    }

    #[test]
    fn test_awq_handles_dead_channels() {
        let (out_f, in_f) = (4, 16);
        let w = lcg(out_f * in_f, 61);
        let mut x = lcg(256 * in_f, 62);
        for s in 0..256 {
            x[s * in_f + 3] = 0.0; // the channel is never active
        }
        let r = awq_quantize(
            &w,
            out_f,
            in_f,
            &x,
            &AwqConfig::default().with_group_size(16),
        )
        .unwrap();
        assert!(r.weights.iter().all(|v| v.is_finite()));
        // A dead channel is given a neutral scale rather than zero.
        assert!(r.channel_scales[3] > 0.0);
    }

    #[test]
    fn test_scale_clamping_bounds_the_ratio() {
        let importance: Vec<f32> = (0..64).map(|i| if i == 0 { 1e6 } else { 1e-6 }).collect();
        let cfg = AwqConfig::default();
        let s = channel_scales(&importance, 1.0, &cfg);
        let min = s.iter().cloned().fold(f32::INFINITY, f32::min);
        let max = s.iter().cloned().fold(0.0f32, f32::max);
        assert!(
            max / min <= cfg.max_scale * cfg.max_scale + 1e-3,
            "scale ratio {} is not limited",
            max / min
        );
    }
}
