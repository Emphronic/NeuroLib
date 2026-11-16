// neo-quantization/src/algo/gptq.rs
//
// GPTQ - layer-by-layer quantization with error compensation based on the Hessian matrix.
//
// Frantar et al., «GPTQ: Accurate Post-Training Quantization for Generative
// Pre-trained Transformers», ICLR 2023 (arXiv:2210.17323).
//
// ## What's the idea
//
// Conventional quantization rounds each weight independently: `q = round(w/s)`.
// It minimizes the error IN WEIGHTS, although the error IN THE OUTPUT of the layer is important -
// and the weights enter the output with different weights, depending on how much
// the corresponding input is active.
//
// GPTQ minimizes exactly `‖XW − XŴ‖²`. Columns are processed according to
// queues; after rounding the next column, the error that occurred does not
// is thrown away, but DISTRIBUTED among the columns that have not yet been processed so that
// to compensate for its contribution to the output. The amount of compensation is set
// inverse Hessian matrix `H⁻¹ = (2·XᵀX)⁻¹`: it describes how the inputs
// correlate with each other.
//
// Practical result: at 4 bits GPTQ loses noticeably less quality than
// round-to-nearest, with the same model size and without additional training.
//
// ## What is needed at the input
//
// Calibration activations - several hundred rows of real layer inputs
// (usually running 128 random fragments of the training corpus). Without them
// the algorithm degenerates into regular rounding and returns
// [`GptqError::NoCalibration`].

use std::vec::Vec;

use super::linalg::{hessian_from_activations, inverse_cholesky_upper, Matrix};
use super::{dequantize_group, quantize_group, GroupQuant};

/// GPTQ configuration.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GptqConfig {
    /// Number of bits per weight. 2...8 are practically meaningful; canonical mode - 4.
    pub bits: u8,
    /// The size of a group of weights sharing the same scale. `0` - one group per
    /// line. Typical value 128: trade-off between accuracy and
    /// storage overhead of scales.
    pub group_size: usize,
    /// Hessian diagonal damping, fraction of its average.
    ///
    /// Without it, the matrix will almost certainly be degenerate: real
    /// activations are highly correlated and there are fewer calibration samples,
    /// than input features. The value 0.01 is from the reference implementation.
    pub damping: f64,
    /// Process the columns in descending order of the Hessian diagonal.
    ///
    /// “Active” inputs are quantized first, while the compensation budget is not yet
    /// spent. In the literature the technique is known as `act-order` and is usually
    /// gives a noticeable gain on 3-4 bits.
    pub act_order: bool,
}

impl Default for GptqConfig {
    fn default() -> Self {
        Self {
            bits: 4,
            group_size: 128,
            damping: 0.01,
            act_order: true,
        }
    }
}

impl GptqConfig {
    /// Canonical mode: 4 bits, groups of 128, act-order enabled.
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

    /// Enable or disable `act-order`.
    pub fn with_act_order(mut self, act_order: bool) -> Self {
        self.act_order = act_order;
        self
    }
}

/// Why GPTQ couldn't work.
#[derive(Debug, Clone, PartialEq)]
pub enum GptqError {
    /// Calibration activations have not been transferred or are empty.
    ///
    /// GPTQ is impossible without them: the whole essence of the algorithm is in the Hessian matrix,
    /// which is built based on activations. An error is returned, not silent
    /// rollback to round-to-nearest - silent substitution of the algorithm would give
    /// worst quality under the guise of GPTQ.
    NoCalibration,
    /// Sizes are not consistent.
    ShapeMismatch(String),
    /// The Hessian matrix is not positive definite even after damping.
    ///
    /// Almost always means there is a shortage of calibration samples:
    /// there should be noticeably more of them than input features, or you need
    /// raise [`GptqConfig::damping`].
    SingularHessian {
        /// Damping used.
        damping: f64,
    },
    /// The number of bits is outside the supported range.
    UnsupportedBits(u8),
    /// The internal invariant of the algorithm is violated (a bug, not an input error
    /// data): the line describes which one.
    Internal(String),
}

impl core::fmt::Display for GptqError {
    fn fmt(&self, f: &mut core::fmt::Formatter<'_>) -> core::fmt::Result {
        match self {
            GptqError::NoCalibration => write!(
                f,
                "GPTQ requires calibration activations: without them, the Hessian matrix cannot be built"
            ),
            GptqError::ShapeMismatch(m) => write!(f, "inconsistent dimensions: {m}"),
            GptqError::SingularHessian { damping } => write!(
                f,
                "Hessian matrix is degenerate at damping {damping}: add calibration samples or raise damping"
            ),
            GptqError::UnsupportedBits(b) => {
                write!(f, "GPTQ: {b} bits out of supported range 2..=8")
            }
            GptqError::Internal(m) => write!(f, "GPTQ: internal invariant violated: {m}"),
        }
    }
}

impl std::error::Error for GptqError {}

/// GPTQ result for one layer.
#[derive(Debug, Clone, PartialEq)]
pub struct GptqResult {
    /// Quantized weights `[out_features, in_features]`, already unwrapped
    /// back to f32.
    ///
    /// It is they, and not the original ones, that should go into the model: error compensation
    /// already taken into account, and re-quantizing them will spoil them.
    pub weights: Vec<f32>,
    /// Scales by group: `[out_features, n_groups]`.
    pub scales: Vec<f32>,
    /// Zero points by group if quantization is asymmetric.
    pub zeros: Vec<f32>,
    /// Number of output features (rows).
    pub out_features: usize,
    /// Number of input features (columns).
    pub in_features: usize,
    /// Group size actually used.
    pub group_size: usize,
    /// The order in which columns are processed if `act_order` was enabled.
    ///
    /// Needed by the consumer: the weights are in the original order, but knowledge
    /// permutations allows you to reproduce the calculation.
    pub permutation: Option<Vec<usize>>,
}

impl GptqResult {
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

/// Quantize the layer using the GPTQ algorithm.
///
/// * `weights` — `[out_features, in_features]`, row-major.
/// * `activations` — `[n_samples, in_features]`, row-major: real inputs
///   layers collected by running the calibration set.
///
/// # Example
///
/// ```
/// use neo_quantization::algo::{gptq_quantize, GptqConfig};
///
/// let (out_features, in_features) = (8, 16);
/// let weights: Vec<f32> = (0..out_features * in_features)
///     .map(|i| ((i * 37 % 101) as f32 - 50.0) / 50.0)
///     .collect();
/// let activations: Vec<f32> = (0..256 * in_features)
///     .map(|i| ((i * 13 % 71) as f32 - 35.0) / 35.0)
///     .collect();
///
/// let result = gptq_quantize(
///     &weights,
///     out_features,
///     in_features,
///     &activations,
///     &GptqConfig::int4_g128(),
/// ).expect("calibration is sufficient");
///
/// assert_eq!(result.weights.len(), weights.len());
/// ```
pub fn gptq_quantize(
    weights: &[f32],
    out_features: usize,
    in_features: usize,
    activations: &[f32],
    config: &GptqConfig,
) -> Result<GptqResult, GptqError> {
    if !(2..=8).contains(&config.bits) {
        return Err(GptqError::UnsupportedBits(config.bits));
    }
    if activations.is_empty() {
        return Err(GptqError::NoCalibration);
    }
    if weights.len() != out_features * in_features {
        return Err(GptqError::ShapeMismatch(format!(
            "weights {}, expected {out_features}×{in_features} = {}",
            weights.len(),
            out_features * in_features
        )));
    }
    if !activations.len().is_multiple_of(in_features) {
        return Err(GptqError::ShapeMismatch(format!(
            "activations {}, not a multiple of in_features {in_features}",
            activations.len()
        )));
    }

    // ── Step 1: Hessian matrix with damping
    let mut h = hessian_from_activations(activations, in_features);

    // Inputs that have never been activated by calibration give zero
    // diagonal. They cannot be left as is - the appeal will fall apart.
    // Upstream resets the corresponding weights to zero and puts 1 on the diagonal:
    // such an input does not affect the output, and its weight is not important.
    let mut dead = Vec::new();
    for i in 0..in_features {
        if h.get(i, i) == 0.0 {
            h.set(i, i, 1.0);
            dead.push(i);
        }
    }

    let damp = config.damping * h.mean_diagonal();
    h.add_diagonal(damp);

    // ── Step 2: order of column processing
    let order: Vec<usize> = if config.act_order {
        let mut idx: Vec<usize> = (0..in_features).collect();
        // Descending Hessian diagonal - the most “new” inputs first.
        idx.sort_by(|&a, &b| {
            h.get(b, b)
                .partial_cmp(&h.get(a, a))
                .unwrap_or(core::cmp::Ordering::Equal)
        });
        idx
    } else {
        (0..in_features).collect()
    };

    // We rearrange Hessian rows and columns in processing order: next
    // the algorithm works with indexes “by order” and not “by attribute”.
    let h = permute_symmetric(&h, &order);

    // ── Step 3: reverse Cholesky ───────────────────────────────────────
    let hinv = inverse_cholesky_upper(&h).ok_or(GptqError::SingularHessian {
        damping: config.damping,
    })?;

    // ── Step 4: working copy of the scales in processing order ───────────────────
    // w[row][k] — weight of the row `row` for the k-th input in sequence.
    let mut w: Vec<Vec<f64>> = (0..out_features)
        .map(|r| {
            order
                .iter()
                .map(|&c| weights[r * in_features + c] as f64)
                .collect()
        })
        .collect();

    // Positions of dead inputs in processing order. Their weights are reset and
    // excluded from compensation: an input that has never been calibrated
    // activated, does not affect the output, and there is no need to spend
    // resolution on it, let alone spread its “error” across the neighbors.
    let mut is_dead = vec![false; in_features];
    for &d in &dead {
        if let Some(k) = order.iter().position(|&c| c == d) {
            is_dead[k] = true;
            for row in w.iter_mut() {
                row[k] = 0.0;
            }
        }
    }

    let group_size = if config.group_size == 0 {
        in_features
    } else {
        config.group_size
    };
    let n_groups = in_features.div_ceil(group_size);

    let mut scales = vec![0.0f32; out_features * n_groups];
    let mut zeros = vec![0.0f32; out_features * n_groups];
    let mut quantized: Vec<Vec<f64>> = vec![vec![0.0; in_features]; out_features];

    // ── Step 5: main loop - column by column
    //
    // The scale of a group is calculated ONCE, on its first column, and then not
    // changes: otherwise error compensation spread over the following
    // columns, would recalculate the scale to fit the already corrected values, and
    // the reverse conversion on the consumer side would not match.
    let mut group_quant: Vec<Option<GroupQuant>> = vec![None; out_features];

    for k in 0..in_features {
        let group_idx = k / group_size;
        let group_start = group_idx * group_size;

        if k == group_start {
            for r in 0..out_features {
                let end = (group_start + group_size).min(in_features);
                let slice: Vec<f64> = (group_start..end).map(|c| w[r][c]).collect();
                let gq = quantize_group(&slice, config.bits);
                scales[r * n_groups + group_idx] = gq.scale as f32;
                zeros[r * n_groups + group_idx] = gq.zero as f32;
                group_quant[r] = Some(gq);
            }
        }

        let d = hinv.get(k, k);
        // A column with zero “cost” carries no information about the output:
        // We quantize without compensation, otherwise we divide by zero.
        let compensate = d.is_finite() && d.abs() > 1e-12;

        for r in 0..out_features {
            if is_dead[k] {
                // Exact zero, not the nearest grid point: weight is not involved
                // in the output, and any non-zero value here is just noise
                // in the model file.
                quantized[r][k] = 0.0;
                continue;
            }
            let Some(gq) = group_quant[r].as_ref() else {
                return Err(GptqError::Internal(format!(
                    "group scale not calculated for row {r}, column {k} \
                     (it is computed in the first column of the group)"
                )));
            };
            let orig = w[r][k];
            let q = dequantize_group(quantize_value(orig, gq, config.bits), gq);
            quantized[r][k] = q;

            if !compensate {
                continue;
            }
            // Error normalized by the “cost” of the column. It was its
            // distribute over the remaining columns with weights Hinv[k][j].
            let err = (orig - q) / d;
            for j in (k + 1)..in_features {
                w[r][j] -= err * hinv.get(k, j);
            }
        }
    }

    // ── Step 6: return the weights to the original order of features ────────────────
    let mut out = vec![0.0f32; weights.len()];
    for r in 0..out_features {
        for (k, &c) in order.iter().enumerate() {
            out[r * in_features + c] = quantized[r][k] as f32;
        }
    }

    Ok(GptqResult {
        weights: out,
        scales,
        zeros,
        out_features,
        in_features,
        group_size,
        permutation: if config.act_order { Some(order) } else { None },
    })
}

/// Permutation of rows and columns of a symmetric matrix.
fn permute_symmetric(h: &Matrix, order: &[usize]) -> Matrix {
    let n = order.len();
    let mut out = Matrix::zeros(n);
    for (i, &oi) in order.iter().enumerate() {
        for (j, &oj) in order.iter().enumerate() {
            out.set(i, j, h.get(oi, oj));
        }
    }
    out
}

/// Round one value to the group grid.
fn quantize_value(v: f64, gq: &GroupQuant, bits: u8) -> i32 {
    let levels = (1i32 << bits) - 1;
    if gq.scale == 0.0 {
        return 0;
    }
    let q = ((v - gq.zero) / gq.scale).round();
    (q as i32).clamp(0, levels)
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Deterministic pseudo-random data.
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

    /// The layer OUTPUT error is what GPTQ minimizes.
    fn output_error(w: &[f32], wq: &[f32], x: &[f32], out_f: usize, in_f: usize) -> f64 {
        let n = x.len() / in_f;
        let mut err = 0.0;
        for s in 0..n {
            let row = &x[s * in_f..(s + 1) * in_f];
            for r in 0..out_f {
                let a: f64 = (0..in_f)
                    .map(|c| w[r * in_f + c] as f64 * row[c] as f64)
                    .sum();
                let b: f64 = (0..in_f)
                    .map(|c| wq[r * in_f + c] as f64 * row[c] as f64)
                    .sum();
                err += (a - b).powi(2);
            }
        }
        err / (n * out_f) as f64
    }

    /// Round-to-nearest of the same bit depth and grouping - the baseline.
    fn rtn(w: &[f32], out_f: usize, in_f: usize, bits: u8, group: usize) -> Vec<f32> {
        let mut out = vec![0.0f32; w.len()];
        for r in 0..out_f {
            for gs in (0..in_f).step_by(group) {
                let end = (gs + group).min(in_f);
                let slice: Vec<f64> = (gs..end).map(|c| w[r * in_f + c] as f64).collect();
                let gq = quantize_group(&slice, bits);
                for c in gs..end {
                    let v = w[r * in_f + c] as f64;
                    out[r * in_f + c] = dequantize_group(quantize_value(v, &gq, bits), &gq) as f32;
                }
            }
        }
        out
    }

    #[test]
    fn test_gptq_requires_calibration() {
        let w = lcg(64, 1);
        let err = gptq_quantize(&w, 4, 16, &[], &GptqConfig::default()).unwrap_err();
        assert_eq!(err, GptqError::NoCalibration);
        // The message should explain why, not just "error".
        assert!(err.to_string().contains("Hessian"));
    }

    #[test]
    fn test_gptq_rejects_bad_shapes() {
        let w = lcg(64, 1);
        let x = lcg(160, 2);
        let e = gptq_quantize(&w, 4, 17, &x, &GptqConfig::default()).unwrap_err();
        assert!(matches!(e, GptqError::ShapeMismatch(_)), "{e:?}");
    }

    #[test]
    fn test_gptq_rejects_unsupported_bits() {
        let w = lcg(64, 1);
        let x = lcg(160, 2);
        let cfg = GptqConfig::default().with_bits(1);
        assert_eq!(
            gptq_quantize(&w, 4, 16, &x, &cfg).unwrap_err(),
            GptqError::UnsupportedBits(1)
        );
    }

    #[test]
    fn test_gptq_preserves_shape() {
        let (out_f, in_f) = (8, 32);
        let w = lcg(out_f * in_f, 11);
        let x = lcg(512 * in_f, 12);
        let r = gptq_quantize(&w, out_f, in_f, &x, &GptqConfig::default()).unwrap();

        assert_eq!(r.weights.len(), w.len());
        assert_eq!(r.out_features, out_f);
        assert_eq!(r.in_features, in_f);
        assert!(r.weights.iter().all(|v| v.is_finite()));
    }

    /// Main test: GPTQ is required to produce a smaller OUTPUT error than
    /// normal rounding of the same bit depth. If this is not the case, the algorithm
    /// doesn't work - its entire price is in this one property.
    #[test]
    fn test_gptq_beats_round_to_nearest_on_output_error() {
        let (out_f, in_f) = (16, 64);
        let w = lcg(out_f * in_f, 7);

        // Correlated activations: this is what compensation works for.
        // With noise, the inputs are independent, there is nothing to compensate.
        let n_samples = 512;
        let base = lcg(n_samples * in_f, 8);
        let mut x = vec![0.0f32; n_samples * in_f];
        for s in 0..n_samples {
            for c in 0..in_f {
                let prev = if c == 0 { 0.0 } else { x[s * in_f + c - 1] };
                x[s * in_f + c] = 0.7 * prev + base[s * in_f + c];
            }
        }

        let cfg = GptqConfig::int4_g128().with_group_size(32);
        let gptq = gptq_quantize(&w, out_f, in_f, &x, &cfg).unwrap();
        let baseline = rtn(&w, out_f, in_f, cfg.bits, 32);

        let err_gptq = output_error(&w, &gptq.weights, &x, out_f, in_f);
        let err_rtn = output_error(&w, &baseline, &x, out_f, in_f);

        assert!(
            err_gptq < err_rtn,
            "GPTQ ({err_gptq:.6e} output error) must beat round-to-nearest ({err_rtn:.6e})"
        );
    }

    #[test]
    fn test_gptq_more_bits_lower_error() {
        let (out_f, in_f) = (8, 32);
        let w = lcg(out_f * in_f, 3);
        let x = lcg(512 * in_f, 4);

        let mut prev = f64::INFINITY;
        for bits in [2u8, 3, 4, 6, 8] {
            let cfg = GptqConfig::default().with_bits(bits).with_group_size(32);
            let r = gptq_quantize(&w, out_f, in_f, &x, &cfg).unwrap();
            let mse = r.weight_mse(&w);
            assert!(
                mse <= prev,
                "{bits} bit: MSE {mse} is worse than previous bit {prev}"
            );
            prev = mse;
        }
    }

    #[test]
    fn test_gptq_is_deterministic() {
        let (out_f, in_f) = (4, 16);
        let w = lcg(out_f * in_f, 5);
        let x = lcg(256 * in_f, 6);
        let cfg = GptqConfig::default().with_group_size(16);

        let a = gptq_quantize(&w, out_f, in_f, &x, &cfg).unwrap();
        let b = gptq_quantize(&w, out_f, in_f, &x, &cfg).unwrap();
        assert_eq!(a, b);
    }

    #[test]
    fn test_gptq_handles_dead_inputs() {
        // The third input is always zero: calibration never activates it.
        let (out_f, in_f) = (4, 8);
        let w = lcg(out_f * in_f, 9);
        let mut x = lcg(256 * in_f, 10);
        for s in 0..256 {
            x[s * in_f + 2] = 0.0;
        }

        let r = gptq_quantize(
            &w,
            out_f,
            in_f,
            &x,
            &GptqConfig::default().with_group_size(8),
        )
        .expect("a dead input shouldn't break the algorithm");
        assert!(r.weights.iter().all(|v| v.is_finite()));
        // The weight of a dead input is reset to zero - it does not affect the output.
        for row in 0..out_f {
            assert_eq!(r.weights[row * in_f + 2], 0.0);
        }
    }

    #[test]
    fn test_act_order_changes_nothing_structurally() {
        let (out_f, in_f) = (4, 16);
        let w = lcg(out_f * in_f, 13);
        let x = lcg(256 * in_f, 14);

        let with = gptq_quantize(
            &w,
            out_f,
            in_f,
            &x,
            &GptqConfig::default().with_group_size(16),
        )
        .unwrap();
        let without = gptq_quantize(
            &w,
            out_f,
            in_f,
            &x,
            &GptqConfig::default()
                .with_group_size(16)
                .with_act_order(false),
        )
        .unwrap();

        assert!(with.permutation.is_some());
        assert!(without.permutation.is_none());
        assert_eq!(with.weights.len(), without.weights.len());
        // A permutation is just that: a permutation: all indexes are exactly the same.
        let mut p = with.permutation.unwrap();
        p.sort_unstable();
        assert_eq!(p, (0..in_f).collect::<Vec<_>>());
    }
}
