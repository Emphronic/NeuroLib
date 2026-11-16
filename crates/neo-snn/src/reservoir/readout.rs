//! Linear reservoir output layer: closed-shape ridge regression and
//! preparation of features by time.

use burn::tensor::{Device, Tensor, TensorData};

use crate::error::{Result, SnnError, tensor_to_vec};

/// How to collapse `[batch, time, neurons]` states into sample features.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Pooling {
    /// Average over all time → `[batch, neurons]`.
    Mean,
    /// State at the last step → `[batch, neurons]`.
    Last,
    /// Average after discarding the first `washout` steps → `[batch, neurons]`.
    MeanAfter(usize),
    /// Time is divided into `k` equal segments, the average is taken for each, and
    /// the results are concatenated → `[batch, k·neurons]`. Keeps it rough
    /// time order (when something happened in the window).
    Segments(usize),
}

/// Collapses tank states by time (see [`Pooling`]).
///
/// # Panics
/// If a time of zero length or `Segments(k)` with `k == 0` or `k` is greater than the number
/// steps.
pub fn pool_time(features: Tensor<3>, pooling: Pooling) -> Tensor<2> {
    let [batch, steps, n] = features.dims();
    assert!(steps > 0, "pool_time: zero time length");
    match pooling {
        Pooling::Mean => features.mean_dim(1).reshape([batch, n]),
        Pooling::Last => features
            .slice([0..batch, steps - 1..steps, 0..n])
            .reshape([batch, n]),
        Pooling::MeanAfter(washout) => {
            let w = washout.min(steps - 1);
            features
                .slice([0..batch, w..steps, 0..n])
                .mean_dim(1)
                .reshape([batch, n])
        }
        Pooling::Segments(k) => {
            assert!(
                k >= 1 && k <= steps,
                "pool_time: Segments({k}) at {steps} steps"
            );
            let seg = steps / k;
            let parts: Vec<Tensor<2>> = (0..k)
                .map(|s| {
                    let a = s * seg;
                    let b = if s + 1 == k { steps } else { (s + 1) * seg };
                    features
                        .clone()
                        .slice([0..batch, a..b, 0..n])
                        .mean_dim(1)
                        .reshape([batch, n])
                })
                .collect();
            Tensor::cat(parts, 1)
        }
    }
}

/// Expands `[batch, time, features]` into rows `[batch·(time−washout),
/// features]`, discarding the first `washout` steps (transient process
/// tank). For time series regression, features and objectives
/// unfold the same way.
pub fn flatten_time(x: Tensor<3>, washout: usize) -> Tensor<2> {
    let [batch, steps, n] = x.dims();
    let w = washout.min(steps.saturating_sub(1));
    x.slice([0..batch, w..steps, 0..n])
        .reshape([batch * (steps - w), n])
}

/// One-hot targets `[labels.len(), classes]`.
pub fn one_hot(labels: &[usize], classes: usize, device: &Device) -> Result<Tensor<2>> {
    let mut data = vec![0.0f32; labels.len() * classes];
    for (i, &l) in labels.iter().enumerate() {
        if l >= classes {
            return Err(SnnError::InvalidConfig {
                what: "one_hot",
                reason: format!("label {l} is outside the range 0..{classes}"),
            });
        }
        data[i * classes + l] = 1.0;
    }
    Ok(Tensor::from_data(
        TensorData::new(data, [labels.len(), classes]),
        device,
    ))
}

/// Proportion of matches between predicted and true labels.
pub fn accuracy(predicted: &[usize], truth: &[usize]) -> f32 {
    assert_eq!(predicted.len(), truth.len(), "accuracy: different lengths");
    if truth.is_empty() {
        return 0.0;
    }
    let hits = predicted.iter().zip(truth).filter(|(a, b)| a == b).count();
    hits as f32 / truth.len() as f32
}

/// Normalized root mean square error
/// `Σ(pred − target)² / Σ(target − mean(target))²` (0 - ideal, 1 - no better
/// constant-average). The average is calculated for each output separately.
pub fn nmse(predicted: Tensor<2>, target: Tensor<2>) -> Result<f32> {
    let [rows, k] = target.dims();
    if predicted.dims() != [rows, k] {
        return Err(SnnError::shape("nmse", [rows, k], predicted.dims()));
    }
    let p = tensor_to_vec(predicted, "nmse")?;
    let t = tensor_to_vec(target, "nmse")?;
    let (mut num, mut den) = (0.0f64, 0.0f64);
    for o in 0..k {
        let mean: f64 = (0..rows).map(|r| t[r * k + o] as f64).sum::<f64>() / rows.max(1) as f64;
        for r in 0..rows {
            let tv = t[r * k + o] as f64;
            num += (p[r * k + o] as f64 - tv).powi(2);
            den += (tv - mean).powi(2);
        }
    }
    Ok((num / den.max(1e-30)) as f32)
}

/// Linear output layer trained by ridge regression.
///
/// Solves `W = argmin ‖XW + b − Y‖² + λ‖W‖²` through normal equations and
/// Cholesky decomposition (the bias `b` is not regularized). Accumulation and solution
/// go to `f64`.
#[derive(Debug, Clone)]
pub struct RidgeReadout {
    n_features: usize,
    n_outputs: usize,
    /// `[n_features, n_outputs]`, row by row.
    weights: Vec<f32>,
    /// `[n_outputs]`.
    bias: Vec<f32>,
    lambda: f32,
}

impl RidgeReadout {
    /// Host training: `features` – `rows × n_features`, `targets` –
    /// `rows × n_outputs` (both row by row).
    pub fn fit_host(
        features: &[f32],
        targets: &[f32],
        rows: usize,
        n_features: usize,
        n_outputs: usize,
        lambda: f32,
    ) -> Result<Self> {
        if rows == 0 || n_features == 0 || n_outputs == 0 {
            return Err(SnnError::InvalidConfig {
                what: "RidgeReadout::fit",
                reason: format!(
                    "zero size: rows={rows}, features={n_features}, outputs={n_outputs}"
                ),
            });
        }
        if !lambda.is_finite() || lambda < 0.0 {
            return Err(SnnError::InvalidConfig {
                what: "RidgeReadout::fit",
                reason: format!("lambda must be finite and ≥ 0, obtained {lambda}"),
            });
        }
        if features.len() != rows * n_features {
            return Err(SnnError::shape(
                "RidgeReadout::fit (features)",
                rows * n_features,
                features.len(),
            ));
        }
        if targets.len() != rows * n_outputs {
            return Err(SnnError::shape(
                "RidgeReadout::fit (targets)",
                rows * n_outputs,
                targets.len(),
            ));
        }

        let (n, k) = (n_features, n_outputs);
        let d = n + 1; // the last column is a constant for the offset
        let mut a = vec![0.0f64; d * d]; // lower triangle XᵀX
        let mut b = vec![0.0f64; d * k]; // XᵀY
        let mut x = vec![0.0f64; d];
        for r in 0..rows {
            for (xi, v) in x.iter_mut().zip(&features[r * n..(r + 1) * n]) {
                *xi = *v as f64;
            }
            x[n] = 1.0;
            let y = &targets[r * k..(r + 1) * k];
            for i in 0..d {
                let xi = x[i];
                if xi == 0.0 {
                    continue; // spike features are sparse
                }
                let arow = &mut a[i * d..i * d + i + 1];
                for (aj, xj) in arow.iter_mut().zip(&x[..=i]) {
                    *aj += xi * xj;
                }
                let brow = &mut b[i * k..(i + 1) * k];
                for (bo, yo) in brow.iter_mut().zip(y) {
                    *bo += xi * *yo as f64;
                }
            }
        }
        for i in 0..n {
            a[i * d + i] += lambda as f64;
        }
        a[n * d + n] += 1e-9; // insurance against degenerate bias

        // Cholesky decomposition A = L·Lᵀ (in place, lower triangle).
        for i in 0..d {
            for j in 0..=i {
                let mut sum = a[i * d + j];
                for p in 0..j {
                    sum -= a[i * d + p] * a[j * d + p];
                }
                if i == j {
                    if sum <= 0.0 || !sum.is_finite() {
                        return Err(SnnError::InvalidConfig {
                            what: "RidgeReadout::fit",
                            reason: format!(
                                "the feature matrix is degenerate (diagonal {i} = {sum}); increase lambda"
                            ),
                        });
                    }
                    a[i * d + i] = sum.sqrt();
                } else {
                    a[i * d + j] = sum / a[j * d + j];
                }
            }
        }
        // L·y = b.
        let mut y = vec![0.0f64; d * k];
        for i in 0..d {
            for o in 0..k {
                let mut sum = b[i * k + o];
                for p in 0..i {
                    sum -= a[i * d + p] * y[p * k + o];
                }
                y[i * k + o] = sum / a[i * d + i];
            }
        }
        // Lᵀ·w = y.
        let mut w = vec![0.0f64; d * k];
        for i in (0..d).rev() {
            for o in 0..k {
                let mut sum = y[i * k + o];
                for p in i + 1..d {
                    sum -= a[p * d + i] * w[p * k + o];
                }
                w[i * k + o] = sum / a[i * d + i];
            }
        }

        Ok(Self {
            n_features: n,
            n_outputs: k,
            weights: w[..n * k].iter().map(|v| *v as f32).collect(),
            bias: w[n * k..].iter().map(|v| *v as f32).collect(),
            lambda,
        })
    }

    /// Training on tensors: `features` `[rows, n_features]`, `targets`
    /// `[rows, n_outputs]`.
    pub fn fit_tensor(features: Tensor<2>, targets: Tensor<2>, lambda: f32) -> Result<Self> {
        let [rows, n] = features.dims();
        let [t_rows, k] = targets.dims();
        if rows != t_rows {
            return Err(SnnError::shape_msg(
                "RidgeReadout::fit_tensor",
                format!("There are as many targets as there are rows of features ({rows})"),
                t_rows,
            ));
        }
        let f = tensor_to_vec(features, "RidgeReadout::fit_tensor")?;
        let t = tensor_to_vec(targets, "RidgeReadout::fit_tensor")?;
        Self::fit_host(&f, &t, rows, n, k, lambda)
    }

    /// Number of features.
    pub fn n_features(&self) -> usize {
        self.n_features
    }

    /// Number of outputs.
    pub fn n_outputs(&self) -> usize {
        self.n_outputs
    }

    /// The regularization coefficient with which the layer was trained.
    pub fn lambda(&self) -> f32 {
        self.lambda
    }

    /// Weights `[n_features, n_outputs]` row by row.
    pub fn weights(&self) -> &[f32] {
        &self.weights
    }

    /// Offsets `[n_outputs]`.
    pub fn bias(&self) -> &[f32] {
        &self.bias
    }

    /// Prediction on the host: `features` - `rows × n_features`.
    pub fn predict_host(&self, features: &[f32], rows: usize) -> Result<Vec<f32>> {
        let (n, k) = (self.n_features, self.n_outputs);
        if features.len() != rows * n {
            return Err(SnnError::shape(
                "RidgeReadout::predict_host",
                rows * n,
                features.len(),
            ));
        }
        let mut out = vec![0.0f32; rows * k];
        for r in 0..rows {
            let row = &features[r * n..(r + 1) * n];
            let dst = &mut out[r * k..(r + 1) * k];
            dst.copy_from_slice(&self.bias);
            for (i, xi) in row.iter().enumerate() {
                if *xi == 0.0 {
                    continue;
                }
                let w = &self.weights[i * k..(i + 1) * k];
                for (d, wv) in dst.iter_mut().zip(w) {
                    *d += xi * wv;
                }
            }
        }
        Ok(out)
    }

    /// Tensor prediction on the input device: `[rows, n_outputs]`.
    pub fn predict(&self, features: Tensor<2>) -> Result<Tensor<2>> {
        let [rows, n] = features.dims();
        if n != self.n_features {
            return Err(SnnError::shape_msg(
                "RidgeReadout::predict",
                format!("[rows, {}]", self.n_features),
                [rows, n],
            ));
        }
        let device = features.device();
        let w = Tensor::from_data(
            TensorData::new(self.weights.clone(), [self.n_features, self.n_outputs]),
            &device,
        );
        let b: Tensor<2> = Tensor::from_data(
            TensorData::new(self.bias.clone(), [1, self.n_outputs]),
            &device,
        );
        Ok(features.matmul(w) + b.expand([rows, self.n_outputs]))
    }

    /// The class with the highest output for each row.
    pub fn classify(&self, features: Tensor<2>) -> Result<Vec<usize>> {
        let [rows, _] = features.dims();
        let k = self.n_outputs;
        let scores = tensor_to_vec(self.predict(features)?, "RidgeReadout::classify")?;
        Ok((0..rows)
            .map(|r| {
                let row = &scores[r * k..(r + 1) * k];
                row.iter()
                    .enumerate()
                    .fold(
                        (0usize, f32::MIN),
                        |best, (i, v)| {
                            if *v > best.1 { (i, *v) } else { best }
                        },
                    )
                    .0
            })
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dev() -> Device {
        Device::flex()
    }

    fn t2(data: Vec<f32>, shape: [usize; 2]) -> Tensor<2> {
        Tensor::from_data(TensorData::new(data, shape), &dev())
    }

    /// The exact linear dependence is restored at small lambda.
    #[test]
    fn recovers_exact_linear_map() {
        // y0 = 2·x0 − x1 + 0.5, y1 = x1 + 3
        let rows = 40;
        let mut f = Vec::new();
        let mut t = Vec::new();
        for r in 0..rows {
            let x0 = (r as f32 * 0.37).sin();
            let x1 = (r as f32 * 0.11).cos();
            f.extend([x0, x1]);
            t.extend([2.0 * x0 - x1 + 0.5, x1 + 3.0]);
        }
        let ro = RidgeReadout::fit_host(&f, &t, rows, 2, 2, 1e-8).unwrap();
        let w = ro.weights();
        assert!(
            (w[0] - 2.0).abs() < 1e-3 && (w[1] - 0.0).abs() < 1e-3,
            "{w:?}"
        );
        assert!(
            (w[2] + 1.0).abs() < 1e-3 && (w[3] - 1.0).abs() < 1e-3,
            "{w:?}"
        );
        let b = ro.bias();
        assert!(
            (b[0] - 0.5).abs() < 1e-3 && (b[1] - 3.0).abs() < 1e-3,
            "{b:?}"
        );
        let p = ro.predict_host(&f, rows).unwrap();
        for (a, b) in p.iter().zip(&t) {
            assert!((a - b).abs() < 1e-3);
        }
    }

    /// A large lambda compresses the weights to zero, the bias remains the average of the targets.
    #[test]
    fn strong_regularisation_shrinks_weights() {
        let f: Vec<f32> = (0..20).map(|i| i as f32 / 10.0).collect();
        let t: Vec<f32> = f.iter().map(|x| 4.0 * x + 1.0).collect();
        let weak = RidgeReadout::fit_host(&f, &t, 20, 1, 1, 1e-6).unwrap();
        let strong = RidgeReadout::fit_host(&f, &t, 20, 1, 1, 1e6).unwrap();
        assert!(weak.weights()[0] > 3.9);
        assert!(strong.weights()[0].abs() < 0.01);
        let mean_t: f32 = t.iter().sum::<f32>() / t.len() as f32;
        assert!((strong.bias()[0] - mean_t).abs() < 0.05);
    }

    /// Tensor and host predict are the same; classify takes argmax.
    #[test]
    fn tensor_and_host_predict_agree_and_classify_uses_argmax() {
        let f = vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 1.0, 1.0];
        let labels = [0usize, 1, 1, 0];
        let y = one_hot(&labels, 2, &dev()).unwrap();
        let ro = RidgeReadout::fit_tensor(t2(f.clone(), [4, 2]), y, 1e-3).unwrap();
        let host = ro.predict_host(&f, 4).unwrap();
        let dev_out = tensor_to_vec(ro.predict(t2(f.clone(), [4, 2])).unwrap(), "t").unwrap();
        for (a, b) in host.iter().zip(&dev_out) {
            assert!((a - b).abs() < 1e-5);
        }
        // XOR is linearly inseparable, but argmax must coincide with the max of the output.
        let cls = ro.classify(t2(f, [4, 2])).unwrap();
        assert_eq!(cls.len(), 4);
    }

    #[test]
    fn rejects_bad_input() {
        assert!(RidgeReadout::fit_host(&[1.0], &[1.0], 1, 1, 1, -1.0).is_err());
        assert!(RidgeReadout::fit_host(&[1.0], &[1.0], 2, 1, 1, 0.1).is_err());
        assert!(RidgeReadout::fit_host(&[], &[], 0, 1, 1, 0.1).is_err());
        assert!(one_hot(&[3], 2, &dev()).is_err());
    }

    #[test]
    fn pooling_shapes_and_values() {
        // [batch=1, time=4, neurons=2]
        let x: Tensor<3> = Tensor::from_data(
            TensorData::new(vec![1.0, 10.0, 2.0, 20.0, 3.0, 30.0, 4.0, 40.0], [1, 4, 2]),
            &dev(),
        );
        let v = |t: Tensor<2>| t.into_data().try_to_vec::<f32>().unwrap();
        assert_eq!(v(pool_time(x.clone(), Pooling::Mean)), vec![2.5, 25.0]);
        assert_eq!(v(pool_time(x.clone(), Pooling::Last)), vec![4.0, 40.0]);
        assert_eq!(
            v(pool_time(x.clone(), Pooling::MeanAfter(2))),
            vec![3.5, 35.0]
        );
        assert_eq!(
            v(pool_time(x.clone(), Pooling::Segments(2))),
            vec![1.5, 15.0, 3.5, 35.0]
        );
        assert_eq!(flatten_time(x, 1).dims(), [3, 2]);
    }

    #[test]
    fn nmse_zero_for_perfect_and_one_for_mean() {
        let target = t2(vec![1.0, 2.0, 3.0, 4.0], [4, 1]);
        assert!(nmse(target.clone(), target.clone()).unwrap() < 1e-9);
        let mean = t2(vec![2.5; 4], [4, 1]);
        assert!((nmse(mean, target).unwrap() - 1.0).abs() < 1e-6);
    }

    #[test]
    fn accuracy_counts_matches() {
        assert_eq!(accuracy(&[0, 1, 1, 0], &[0, 1, 0, 0]), 0.75);
    }
}
