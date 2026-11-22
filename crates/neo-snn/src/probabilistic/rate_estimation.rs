//! Rate Estimation — the second part of the paper: instead of explicitly modeling each spike
//! during training, the method propagates probabilities of firing.
//!
//! For the sum of independent Bernoulli input events the authors use the mean/variance
//! approximation:
//!
//! ```text
//! μ   = P·W
//! σ²  = (P ⊙ (1−P))·(W ⊙ W)
//! P_spike ≈ Φ((μ − θ)/(σ + ε))     — normal CDF
//! ```
//!
//! where `P` — input spike probabilities `[batch, in]`, `W` — connection weights `[in, out]`,
//! `θ` — threshold `[batch, out]`, `ε` — a small guard against `σ = 0`.

use burn::tensor::Tensor;

/// Rate Estimate — the propagated probabilities of a step.
#[derive(Debug, Clone)]
pub struct RateEstimate {
    /// `μ = P·W`, `[batch, out]`.
    pub mean: Tensor<2>,
    /// `σ² = (P ⊙ (1−P))·(W ⊙ W)`, `[batch, out]`.
    pub variance: Tensor<2>,
    /// `P_spike ≈ Φ((μ − θ)/(σ + ε))`, `[batch, out]`.
    pub prob_spike: Tensor<2>,
}

/// Rate estimation of one step (the paper's RE formulas).
pub fn rate_estimate(
    inputs: &Tensor<2>,
    weights: &Tensor<2>,
    threshold: &Tensor<2>,
    eps: f32,
) -> RateEstimate {
    // μ = P·W
    let mean = inputs.clone().matmul(weights.clone());
    let threshold = threshold.clone().expand(mean.dims());

    // σ² = (P ⊙ (1−P))·(W ⊙ W)
    let p = inputs.clone();
    let one_minus_p = p.clone().mul_scalar(-1.0).add_scalar(1.0);
    let var = p
        .mul(one_minus_p)
        .matmul(weights.clone().mul(weights.clone()));

    // P_spike ≈ Φ((μ − θ)/(σ + ε)); Φ(z) = 0.5·(1 + erf(z/√2))
    let z = mean
        .clone()
        .sub(threshold)
        .div(var.clone().sqrt().add_scalar(eps));
    let prob_spike = z
        .div_scalar(std::f32::consts::SQRT_2)
        .erf()
        .add_scalar(1.0)
        .mul_scalar(0.5);

    RateEstimate {
        mean,
        variance: var,
        prob_spike,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::{Device, Tensor};

    fn t2(values: Vec<f32>, shape: [usize; 2]) -> Tensor<2> {
        Tensor::<1>::from_floats(values.as_slice(), &Device::flex()).reshape(shape)
    }

    #[test]
    fn rate_estimate_matches_paper_formulas() {
        // P = 0.5, W = [[1.0]], θ = [1.0]:
        // μ = 0.5, σ² = 0.5·(1−0.5)·1 = 0.25, z = (0.5 − 1)/0.5 ≈ −1 → Φ(−1) ≈ 0.1587.
        let inputs = t2(vec![0.5], [1, 1]);
        let weights = t2(vec![1.0], [1, 1]);
        let threshold = t2(vec![1.0], [1, 1]);
        let est = rate_estimate(&inputs, &weights, &threshold, 1e-6);
        let mu = est.mean.into_data().try_to_vec::<f32>().unwrap()[0];
        let var = est.variance.into_data().try_to_vec::<f32>().unwrap()[0];
        let p = est.prob_spike.into_data().try_to_vec::<f32>().unwrap()[0];
        assert!((mu - 0.5).abs() < 1e-6);
        assert!((var - 0.25).abs() < 1e-5);
        assert!((p - 0.15865525).abs() < 1e-3, "P_spike = {p}");
    }

    #[test]
    fn rate_estimate_shapes() {
        // [batch=2, in=3] × [in=3, out=2] → [2, 2]
        let inputs = t2(vec![0.1, 0.2, 0.3, 0.4, 0.5, 0.6], [2, 3]);
        let weights = t2(vec![0.5, 0.1, 0.0, 0.2, 0.7, 0.3], [3, 2]);
        let threshold = t2(vec![1.0, 1.5], [1, 2]);
        let est = rate_estimate(&inputs, &weights, &threshold, 1e-6);
        assert_eq!(est.mean.dims(), [2, 2]);
        assert_eq!(est.variance.dims(), [2, 2]);
        assert_eq!(est.prob_spike.dims(), [2, 2]);
    }
}
