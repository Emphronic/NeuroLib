//! Probabilistic forward — the probabilities of spikes without generating spikes
//! (the paper's RE — Rate Estimation).

use burn::tensor::{Device, Tensor};

use crate::neuron::StochasticThresholdLeaky;
use crate::probabilistic::rate_estimation::{RateEstimate, rate_estimate};

/// Probabilistic state of one step — the probability of a spike, no spikes generated.
#[derive(Debug, Clone)]
pub struct ProbabilityState {
    /// Probability of spiking, `[batch, units]`.
    pub prob: Tensor<2>,
}

impl ProbabilityState {
    /// State of ready-made probabilities.
    pub fn new(prob: Tensor<2>) -> Self {
        Self { prob }
    }

    /// Zero probabilities.
    pub fn zeros(device: &Device, batch: usize, units: usize) -> Self {
        Self {
            prob: Tensor::zeros([batch, units], device),
        }
    }

    /// From a [`RateEstimate`].
    pub fn from_rate_estimate(est: &RateEstimate) -> Self {
        Self {
            prob: est.prob_spike.clone(),
        }
    }
}

/// Probabilistic forward of a stochastic-threshold layer over a sequence — the probabilities of
/// spikes of each step, without generating any spikes.
///
/// `inputs` — the input spike probabilities `[batch, in]`, `weights` — connection weights
/// `[in, out]`, `eps` — the guard of [`rate_estimate`]. The threshold used is the distribution
/// mean of the layer (the paper's `θ`).
pub fn forward_seq_prob(
    layer: &StochasticThresholdLeaky,
    inputs: Vec<Tensor<2>>,
    weights: &Tensor<2>,
    eps: f32,
) -> Vec<ProbabilityState> {
    assert!(
        !inputs.is_empty(),
        "forward_seq_prob: inputs must not be empty"
    );
    let dims = inputs[0].dims();
    let device = inputs[0].device();
    let out_dims = [dims[0], weights.dims()[1]];
    let threshold = layer.threshold_distribution_tensor(out_dims, &device);
    inputs
        .into_iter()
        .map(|x| ProbabilityState::from_rate_estimate(&rate_estimate(&x, weights, &threshold, eps)))
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::neuron::StochasticThresholdLeakyConfig;
    use crate::probabilistic::RngSource;

    #[test]
    fn forward_seq_prob_matches_rate_estimate() {
        let device = Device::flex();
        let layer =
            StochasticThresholdLeaky::new(&StochasticThresholdLeakyConfig::new(0.9), &device);
        let inputs = vec![Tensor::<1>::from_floats(vec![0.5].as_slice(), &device).reshape([1, 1])];
        let weights = Tensor::<1>::from_floats(vec![1.0].as_slice(), &device).reshape([1, 1]);
        let threshold = layer.threshold_distribution_tensor([1, 1], &device);
        let est = rate_estimate(&inputs[0], &weights, &threshold, 1e-6);

        let out = forward_seq_prob(&layer, inputs, &weights, 1e-6);
        assert_eq!(out.len(), 1);
        let a = est.prob_spike.into_data().try_to_vec::<f32>().unwrap()[0];
        let b = out[0].prob.clone().into_data().try_to_vec::<f32>().unwrap()[0];
        assert!((a - b).abs() < 1e-6, "a={a}, b={b}");
    }

    #[test]
    fn probability_state_zero_and_new() {
        let device = Device::flex();
        let zeros = ProbabilityState::zeros(&device, 2, 3);
        let v = zeros.prob.into_data().try_to_vec::<f32>().unwrap();
        assert_eq!(v.len(), 6);
        assert!(v.iter().all(|&x| x == 0.0));
        let state = ProbabilityState::new(
            Tensor::<1>::from_floats(vec![0.25].as_slice(), &device).reshape([1, 1]),
        );
        assert_eq!(state.prob.dims(), [1, 1]);
        let _ = RngSource::new(None);
    }
}
