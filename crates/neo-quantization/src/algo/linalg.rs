// neo-quantization/src/algo/linalg.rs
//
// Minimal dense linear algebra for GPTQ: Cholesky decomposition,
// inversion of a symmetric positive definite matrix.
//
// ## Why no external library
//
// We need exactly three operations on one symmetric matrix of size
// `in_features × in_features`. Pull `nalgebra` or BLAS binding for them
// would mean making it a mandatory dependency of the quantization crate
// general-purpose linear algebra - and its assembly system at the same time.
// The implementation below takes two hundred lines, is deterministic and does not require
// nothing but `std`.
//
// ## Storage
//
// The matrices are dense, row-major, square. The element `(i, j)` lies in
// `data[i * n + j]`. The format is intentionally simple: these matrices only live
// they do not go inside or outside GPTQ.

use std::vec::Vec;

/// Dense square matrix `n × n` in row-major order.
#[derive(Debug, Clone, PartialEq)]
pub struct Matrix {
    /// Elements, `n * n` pieces, row-major.
    pub data: Vec<f64>,
    /// Matrix order.
    pub n: usize,
}

impl Matrix {
    /// Zero matrix of order `n`.
    pub fn zeros(n: usize) -> Self {
        Self {
            data: vec![0.0; n * n],
            n,
        }
    }

    /// Identity matrix of order `n`.
    pub fn identity(n: usize) -> Self {
        let mut m = Self::zeros(n);
        for i in 0..n {
            m.data[i * n + i] = 1.0;
        }
        m
    }

    /// Element `(i, j)`.
    #[inline]
    pub fn get(&self, i: usize, j: usize) -> f64 {
        self.data[i * self.n + j]
    }

    /// Write element `(i, j)`.
    #[inline]
    pub fn set(&mut self, i: usize, j: usize, v: f64) {
        self.data[i * self.n + j] = v;
    }

    /// Add `(i, j)` to element.
    #[inline]
    pub fn add_to(&mut self, i: usize, j: usize, v: f64) {
        self.data[i * self.n + j] += v;
    }

    /// The average value on the diagonal is the basis for damping in GPTQ.
    pub fn mean_diagonal(&self) -> f64 {
        if self.n == 0 {
            return 0.0;
        }
        (0..self.n).map(|i| self.get(i, i)).sum::<f64>() / self.n as f64
    }

    /// Add `value` to all diagonal elements.
    pub fn add_diagonal(&mut self, value: f64) {
        for i in 0..self.n {
            self.add_to(i, i, value);
        }
    }
}

/// Hessian matrix `H = 2·XᵀX` from calibration activations.
///
/// `activations` — `[n_samples, n_features]` in row-major order.
///
/// The factor 2 is due to the fact that the objective function of GPTQ is `‖XW − XŴ‖²`, and its
/// the Hessian over `W` is equal to `2·XᵀX`. The multiplier does not affect the algorithm itself (it
/// abbreviated when divided), but retained so that the values coincide with
/// reference implementation of GPTQ and references.
pub fn hessian_from_activations(activations: &[f32], n_features: usize) -> Matrix {
    assert!(n_features > 0, "hessian: n_features must be > 0");
    assert_eq!(
        activations.len() % n_features,
        0,
        "hessian: length of activations {} is not a multiple of n_features {n_features}",
        activations.len()
    );

    let n_samples = activations.len() / n_features;
    let mut h = Matrix::zeros(n_features);

    for s in 0..n_samples {
        let row = &activations[s * n_features..(s + 1) * n_features];
        // Only the upper triangle is used, then mirrored: matrix
        // symmetrical in construction, and half the work is half
        // time on the layer with in_features in thousands.
        for i in 0..n_features {
            let xi = row[i] as f64;
            if xi == 0.0 {
                continue;
            }
            for j in i..n_features {
                h.add_to(i, j, 2.0 * xi * row[j] as f64);
            }
        }
    }
    for i in 0..n_features {
        for j in 0..i {
            let v = h.get(j, i);
            h.set(i, j, v);
        }
    }
    h
}

/// The Cholesky decomposition of `A = L·Lᵀ`, returns the lower triangular `L`.
///
/// `None` if the matrix is not positive definite - calling code
/// obliged to process this, and not continue counting through garbage: in GPTQ this is
/// means that the calibration data was not enough for the rank of the matrix, and
/// you need to either increase the damping or add samples.
pub fn cholesky(a: &Matrix) -> Option<Matrix> {
    let n = a.n;
    let mut l = Matrix::zeros(n);

    for i in 0..n {
        for j in 0..=i {
            let mut sum = a.get(i, j);
            for k in 0..j {
                sum -= l.get(i, k) * l.get(j, k);
            }
            if i == j {
                if sum <= 0.0 || !sum.is_finite() {
                    return None;
                }
                l.set(i, j, sum.sqrt());
            } else {
                let ljj = l.get(j, j);
                if ljj == 0.0 {
                    return None;
                }
                l.set(i, j, sum / ljj);
            }
        }
    }
    Some(l)
}

/// Inversion of a symmetric positive definite matrix via Cholesky.
///
/// `A⁻¹ = (L·Lᵀ)⁻¹ = L⁻ᵀ·L⁻¹`. Invert the lower triangular `L` by direct
/// substitution, then multiply.
pub fn inverse_spd(a: &Matrix) -> Option<Matrix> {
    let n = a.n;
    let l = cholesky(a)?;

    // Inversion of a lower triangular matrix by direct substitution.
    let mut inv_l = Matrix::zeros(n);
    for i in 0..n {
        let lii = l.get(i, i);
        if lii == 0.0 {
            return None;
        }
        inv_l.set(i, i, 1.0 / lii);
        for j in 0..i {
            let mut sum = 0.0;
            for k in j..i {
                sum += l.get(i, k) * inv_l.get(k, j);
            }
            inv_l.set(i, j, -sum / lii);
        }
    }

    // A⁻¹ = inv_Lᵀ · inv_L; the result is symmetrical, consider the upper one
    // triangle and mirror.
    let mut out = Matrix::zeros(n);
    for i in 0..n {
        for j in i..n {
            let mut sum = 0.0;
            for k in j.max(i)..n {
                sum += inv_l.get(k, i) * inv_l.get(k, j);
            }
            out.set(i, j, sum);
            out.set(j, i, sum);
        }
    }
    Some(out)
}

/// Upper triangular Cholesky multiplier from the matrix INVERSE to `a`.
///
/// This is the same `Hinv` along which GPTQ propagates the error: element
/// `Hinv[j][j]` specifies the “cost” of quantizing the column `j`, and the row
/// `Hinv[j][j+1..]` - how to spread an error that has not yet occurred
/// processed columns.
pub fn inverse_cholesky_upper(a: &Matrix) -> Option<Matrix> {
    let inv = inverse_spd(a)?;
    let l = cholesky(&inv)?;
    // Transpose to upper triangular.
    let n = l.n;
    let mut u = Matrix::zeros(n);
    for i in 0..n {
        for j in 0..n {
            u.set(i, j, l.get(j, i));
        }
    }
    Some(u)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn mat(n: usize, v: &[f64]) -> Matrix {
        Matrix {
            data: v.to_vec(),
            n,
        }
    }

    #[test]
    fn test_cholesky_reconstructs_original() {
        // Positive definite by construction.
        let a = mat(3, &[4.0, 2.0, 1.0, 2.0, 5.0, 3.0, 1.0, 3.0, 6.0]);
        let l = cholesky(&a).expect("matrix is positive definite");

        for i in 0..3 {
            for j in 0..3 {
                let mut s = 0.0;
                for k in 0..3 {
                    s += l.get(i, k) * l.get(j, k);
                }
                assert!(
                    (s - a.get(i, j)).abs() < 1e-10,
                    "({i},{j}): {s} vs {}",
                    a.get(i, j)
                );
            }
        }
    }

    #[test]
    fn test_cholesky_rejects_non_positive_definite() {
        // Degenerate: the second line is double the first.
        let a = mat(2, &[1.0, 2.0, 2.0, 4.0]);
        assert!(cholesky(&a).is_none(), "singular matrix must be rejected");

        // Negative definite.
        let b = mat(2, &[-1.0, 0.0, 0.0, -1.0]);
        assert!(cholesky(&b).is_none());
    }

    #[test]
    fn test_inverse_spd_gives_identity() {
        let a = mat(3, &[4.0, 2.0, 1.0, 2.0, 5.0, 3.0, 1.0, 3.0, 6.0]);
        let inv = inverse_spd(&a).unwrap();

        for i in 0..3 {
            for j in 0..3 {
                let mut s = 0.0;
                for k in 0..3 {
                    s += a.get(i, k) * inv.get(k, j);
                }
                let expected = if i == j { 1.0 } else { 0.0 };
                assert!((s - expected).abs() < 1e-9, "A·A⁻¹({i},{j}) = {s}");
            }
        }
    }

    #[test]
    fn test_hessian_is_symmetric_and_matches_manual() {
        // 2 samples, 3 features.
        let x = [1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0];
        let h = hessian_from_activations(&x, 3);

        for i in 0..3 {
            for j in 0..3 {
                assert!((h.get(i, j) - h.get(j, i)).abs() < 1e-12, "not symmetrical");
                // H = 2·XᵀX: element (i,j) = 2·Σ_s x[s][i]·x[s][j]
                let manual = 2.0 * (x[i] as f64 * x[j] as f64 + x[3 + i] as f64 * x[3 + j] as f64);
                assert!((h.get(i, j) - manual).abs() < 1e-9, "({i},{j})");
            }
        }
    }

    #[test]
    fn test_inverse_cholesky_upper_is_upper_triangular() {
        let a = mat(3, &[4.0, 1.0, 0.5, 1.0, 3.0, 0.7, 0.5, 0.7, 2.0]);
        let u = inverse_cholesky_upper(&a).unwrap();
        for i in 0..3 {
            for j in 0..i {
                assert!(
                    u.get(i, j).abs() < 1e-12,
                    "({i},{j}) not zero: {}",
                    u.get(i, j)
                );
            }
            assert!(u.get(i, i) > 0.0, "the diagonal must be positive");
        }
    }
}
