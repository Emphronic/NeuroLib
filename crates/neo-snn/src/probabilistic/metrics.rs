//! Metrics of probabilistic predictions (paper: "Probabilistic predictions were evaluated
//! using NLL and ECE").

use burn::tensor::Tensor;

/// Binary negative log-likelihood of spike probabilities against hard spikes:
/// `NLL = −mean(S·ln(p) + (1−S)·ln(1−p))`.
pub fn spike_nll(probs: &Tensor<2>, spikes: &Tensor<2>, eps: f32) -> f32 {
    let p = probs.clone().into_data().try_to_vec::<f32>().unwrap();
    let s = spikes.clone().into_data().try_to_vec::<f32>().unwrap();
    let mut nll = 0.0f64;
    for (&p, &s) in p.iter().zip(s.iter()) {
        let p = (p as f64).clamp(eps as f64, 1.0 - eps as f64);
        nll -= s as f64 * p.ln() + (1.0 - s as f64) * (1.0 - p).ln();
    }
    (nll / p.len().max(1) as f64) as f32
}

/// Expected calibration error of spike probabilities against hard spikes.
///
/// Bin by confidence (probability of spiking), then
/// `ECE = Σ_bins (n_bin/N)·|mean_confidence − mean_accuracy|` — how much the predicted
/// probabilities are, on average, different from the observed firing rate per bin.
pub fn expected_calibration_error(probs: &Tensor<2>, spikes: &Tensor<2>, bins: usize) -> f32 {
    let p = probs.clone().into_data().try_to_vec::<f32>().unwrap();
    let s = spikes.clone().into_data().try_to_vec::<f32>().unwrap();
    let n = p.len();
    let bins = bins.max(1);
    let mut ece = 0.0f64;
    for bin in 0..bins {
        let mut count = 0usize;
        let mut conf_sum = 0.0f64;
        let mut acc_sum = 0.0f64;
        for (&pv, &sv) in p.iter().zip(s.iter()) {
            let idx = ((pv as f64 * bins as f64).floor() as usize).min(bins - 1);
            if idx == bin {
                count += 1;
                conf_sum += pv as f64;
                acc_sum += sv as f64;
            }
        }
        if count > 0 {
            ece += (count as f64 / n.max(1) as f64)
                * (conf_sum / count as f64 - acc_sum / count as f64).abs();
        }
    }
    ece as f32
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::{Device, Tensor};

    fn t2(values: Vec<f32>) -> Tensor<2> {
        Tensor::<1>::from_floats(values.as_slice(), &Device::flex()).reshape([1, values.len()])
    }

    #[test]
    fn nll_is_zero_for_confident_correct_predictions() {
        let p = t2(vec![1.0, 0.0]);
        let s = t2(vec![1.0, 0.0]);
        assert!(spike_nll(&p, &s, 1e-6).abs() < 1e-5);
    }

    #[test]
    fn nll_of_half_prediction_is_ln2() {
        let p = t2(vec![0.5]);
        let s = t2(vec![1.0]);
        assert!((spike_nll(&p, &s, 1e-6) - std::f32::consts::LN_2).abs() < 1e-5);
    }

    #[test]
    fn ece_is_zero_when_confident_matches_spikes() {
        let p = t2(vec![1.0, 0.0, 1.0, 0.0]);
        let s = t2(vec![1.0, 0.0, 1.0, 0.0]);
        assert!(expected_calibration_error(&p, &s, 4) < 1e-6);
    }

    #[test]
    fn ece_is_large_when_confident_but_wrong() {
        let p = t2(vec![1.0]);
        let s = t2(vec![0.0]);
        assert!(expected_calibration_error(&p, &s, 10) > 0.9);
    }
}
