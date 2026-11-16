//! Loss functions for SNNs: spike-count, rate, and time based.

use burn::tensor::Tensor;
use burn::tensor::loss::cross_entropy_with_logits;

use super::population::{PopulationReduction, population_reduce, population_reduce_seq};
use super::temporal::{max_over_time, sum_over_time};
use crate::error::{assert_dims, assert_targets};

/// MSE between the total number of spikes and the target count.
///
/// `spike_counts`, `target_counts` — `[batch, num_classes]`.
pub fn mse_count_loss(spike_counts: Tensor<2>, target_counts: Tensor<2>) -> Tensor<1> {
    assert_dims(
        "mse_count_loss",
        "target_counts",
        target_counts.dims(),
        spike_counts.dims(),
    );
    let diff = spike_counts - target_counts;
    diff.clone().mul(diff).mean()
}

/// Cross-entropy over mean firing rates (rate-coded readout).
///
/// `rates` are treated as logits, `target_probs` — class probabilities
/// (for example, one-hot as floats). Both `[batch, num_classes]`.
pub fn ce_rate_loss(rates: Tensor<2>, target_probs: Tensor<2>) -> Tensor<1> {
    // cross_entropy_with_logits in burn already returns a scalar averaged
    // over the batch — the extra .mean() in the original was a no-op.
    assert_dims(
        "ce_rate_loss",
        "target_probs",
        target_probs.dims(),
        rates.dims(),
    );
    cross_entropy_with_logits(rates, target_probs)
}

/// Cross-entropy over logits against class probabilities.
///
/// `logits`, `target_probs` — `[batch, num_classes]`.
pub fn ce_temporal_loss(logits: Tensor<2>, target_probs: Tensor<2>) -> Tensor<1> {
    assert_dims(
        "ce_temporal_loss",
        "target_probs",
        target_probs.dims(),
        logits.dims(),
    );
    cross_entropy_with_logits(logits, target_probs)
}

// ═══════════════════════════════════════════════════════════════════════════
//  Losses over the whole spike train: [batch, num_steps, num_classes]
// ═══════════════════════════════════════════════════════════════════════════

/// Cross-entropy over the TOTAL number of spikes across all time.
///
/// The difference from [`ce_rate_loss`] is fundamental, even if it looks
/// minor: here spikes are first summed over time, and cross-entropy is applied
/// ONCE to the accumulated counts. In `ce_rate_loss` it is applied to already
/// averaged rates. The count variant does not require the correct class to
/// fire at EVERY step — it is enough that it fires more often than the others.
/// On tasks where the answer is not formed immediately (delayed networks, deep
/// SNNs), this is a noticeably softer target.
///
/// `spikes` — `[batch, num_steps, num_classes]`, `target_probs` —
/// `[batch, num_classes]` (usually one-hot as floats).
pub fn ce_count_loss(spikes: Tensor<3>, target_probs: Tensor<2>) -> Tensor<1> {
    let [batch, _, classes] = spikes.dims();
    assert_targets("ce_count_loss", batch, classes, target_probs.dims());
    cross_entropy_with_logits(sum_over_time(spikes), target_probs)
}

/// Cross-entropy over the MAXIMUM membrane potential across all time.
///
/// Readout without spikes: the output layer does not fire at all, and the class
/// is the one whose membrane rose highest. The threshold on the output layer is
/// usually disabled in this case (`ResetMode::None`), otherwise the reset would
/// cut off exactly the quantity the decision is based on.
///
/// `membrane` — `[batch, num_steps, num_classes]`.
pub fn ce_max_membrane_loss(membrane: Tensor<3>, target_probs: Tensor<2>) -> Tensor<1> {
    let [batch, _, classes] = membrane.dims();
    assert_targets("ce_max_membrane_loss", batch, classes, target_probs.dims());
    cross_entropy_with_logits(max_over_time(membrane), target_probs)
}

/// MSE between the membrane potential at EVERY step and the target value.
///
/// The target is held for the whole duration, not just at the end: this is the
/// strictest form of membrane-based training, but it prevents the network from
/// "second-guessing the answer" with its last step.
///
/// `membrane` — `[batch, num_steps, num_classes]`, `target` —
/// `[batch, num_classes]`, broadcast over all steps.
pub fn mse_membrane_loss(membrane: Tensor<3>, target: Tensor<2>) -> Tensor<1> {
    let [batch, steps, classes] = membrane.dims();
    assert!(steps > 0, "mse_membrane_loss: empty time axis");
    assert_targets("mse_membrane_loss", batch, classes, target.dims());

    // The mean over steps of the means over (batch, classes), with equal step sizes, is the mean
    // over the whole tensor: one operation instead of `steps` slices (the backward pass of a slice is O(T²)).
    let target = target
        .reshape([batch, 1, classes])
        .expand([batch, steps, classes]);
    let diff = membrane - target;
    diff.clone().mul(diff).mean()
}

/// MSE over spike TIME: latency-coded readout.
///
/// The counterpart of [`latency_encode`](crate::encoding::latency_encode):
/// there intensity is converted into spike time, here spike time is compared
/// with the target.
///
/// ## How it differs from the reference implementation
///
/// The index of the FIRST spike is an integer, piecewise-constant quantity with
/// no gradient through it. Therefore the firing time here is taken as the
/// expected step index over the normalized spike train:
/// `t̂ = Σ t·s_t / Σ s_t`. This estimate is smooth, coincides with the
/// first-spike index when there is exactly one spike (and in a latency code
/// there is exactly one), and gives a meaningful gradient when there are
/// several spikes.
///
/// A neuron that never fired is penalized as if it fired at the last step:
/// silence is the answer "latest of all," not the absence of an answer.
///
/// `spikes` — `[batch, num_steps, num_classes]`, `target_steps` —
/// `[batch, num_classes]` in step indices.
pub fn mse_temporal_loss(spikes: Tensor<3>, target_steps: Tensor<2>) -> Tensor<1> {
    let [batch, steps, classes] = spikes.dims();
    assert!(steps > 0, "mse_temporal_loss: empty time axis");
    assert_targets("mse_temporal_loss", batch, classes, target_steps.dims());
    let device = spikes.device();

    // Σ_t t·spikes[:, t, :] — with a single reduction along the time axis.
    let step_ids: Vec<f32> = (0..steps).map(|t| t as f32).collect();
    let step_w = Tensor::<1>::from_floats(step_ids.as_slice(), &device)
        .reshape([1, steps, 1])
        .expand([batch, steps, classes]);
    let weighted: Tensor<2> = (spikes.clone() * step_w)
        .sum_dim(1)
        .reshape([batch, classes]);

    let total = sum_over_time(spikes);
    // Silent neuron: the zero denominator is replaced by one, and the last
    // step is added to the numerator — yielding t̂ = steps − 1.
    let silent = total.clone().lower_equal_elem(0.0).float().detach();
    let last = (steps - 1) as f32;

    let numerator = weighted + silent.clone().mul_scalar(last);
    let denominator = total + silent;

    let predicted = numerator.div(denominator);
    let diff = predicted - target_steps;
    diff.clone().mul(diff).mean()
}

// ═══════════════════════════════════════════════════════════════════════════
//  Population code: N neurons per class
// ═══════════════════════════════════════════════════════════════════════════
//
// The same three losses as above for a "one neuron per class" readout layer,
// but the output layer has `num_classes * neurons_per_class` neurons. Each
// function folds the population through
// [`population_reduce`]/[`population_reduce_seq`] (see
// [`super::population`]) and delegates to the existing per-class loss — the
// same consistency guarantee as for `population_*` in [`super::acc`].

/// [`ce_rate_loss`] for population readout: `rates` — `[batch,
/// num_classes * neurons_per_class]`.
///
/// The group is averaged ([`PopulationReduction::Mean`]), not summed: `rates`
/// are treated as logits, and the sum of `neurons_per_class` logits would scale
/// them up along with the population size, shifting the effective softmax
/// "temperature" inside `cross_entropy_with_logits`. The mean keeps the logit
/// scale independent of `neurons_per_class`.
pub fn population_ce_rate_loss(
    rates: Tensor<2>,
    neurons_per_class: usize,
    target_probs: Tensor<2>,
) -> Tensor<1> {
    ce_rate_loss(
        population_reduce(rates, neurons_per_class, PopulationReduction::Mean),
        target_probs,
    )
}

/// [`ce_count_loss`] for population readout: `spikes` — `[batch,
/// num_steps, num_classes * neurons_per_class]`.
///
/// The group is summed ([`PopulationReduction::Sum`]): the total number of
/// spikes of a class is the sum of spikes over all its neurons and all time,
/// and the order (time first, then population, or vice versa) does not matter
/// for a sum.
pub fn population_ce_count_loss(
    spikes: Tensor<3>,
    neurons_per_class: usize,
    target_probs: Tensor<2>,
) -> Tensor<1> {
    ce_count_loss(
        population_reduce_seq(spikes, neurons_per_class, PopulationReduction::Sum),
        target_probs,
    )
}

/// [`ce_max_membrane_loss`] for population readout: `membrane` —
/// `[batch, num_steps, num_classes * neurons_per_class]`.
///
/// The group is averaged ([`PopulationReduction::Mean`]) for the same reason
/// as in [`population_ce_rate_loss`]: the membrane here is also a logit for
/// `cross_entropy_with_logits`, and summing would scale it along with
/// `neurons_per_class`.
pub fn population_ce_max_membrane_loss(
    membrane: Tensor<3>,
    neurons_per_class: usize,
    target_probs: Tensor<2>,
) -> Tensor<1> {
    ce_max_membrane_loss(
        population_reduce_seq(membrane, neurons_per_class, PopulationReduction::Mean),
        target_probs,
    )
}

#[cfg(test)]
mod population_tests {
    use super::*;
    use burn::tensor::Device;

    fn dev() -> Device {
        Device::flex()
    }

    fn t2(data: Vec<f32>, shape: [usize; 2]) -> Tensor<2> {
        Tensor::<1>::from_floats(data.as_slice(), &dev()).reshape(shape)
    }

    fn t3(data: Vec<f32>, shape: [usize; 3]) -> Tensor<3> {
        Tensor::<1>::from_floats(data.as_slice(), &dev()).reshape(shape)
    }

    fn scalar(t: Tensor<1>) -> f32 {
        t.into_data().as_slice::<f32>().unwrap()[0]
    }

    #[test]
    fn population_ce_rate_matches_plain_when_npc_one() {
        let rates = t2(vec![2.0, -1.0, 0.5, 1.5, 0.0, -0.5], [2, 3]);
        let targets = t2(vec![1.0, 0.0, 0.0, 0.0, 0.0, 1.0], [2, 3]);
        let plain = scalar(ce_rate_loss(rates.clone(), targets.clone()));
        let pop = scalar(population_ce_rate_loss(rates, 1, targets));
        assert!((plain - pop).abs() < 1e-6);
    }

    #[test]
    fn population_ce_count_matches_plain_when_npc_one() {
        let spikes = t3(vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0], [1, 2, 3]);
        let targets = t2(vec![0.0, 1.0, 0.0], [1, 3]);
        let plain = scalar(ce_count_loss(spikes.clone(), targets.clone()));
        let pop = scalar(population_ce_count_loss(spikes, 1, targets));
        assert!((plain - pop).abs() < 1e-6);
    }

    #[test]
    fn population_ce_count_sums_group_spikes() {
        // batch=1, steps=2, 2 classes x 2 neurons.
        // class0 total: (1+0)+(0+1) = 2; class1 total: (0+1)+(0+0) = 1.
        let spikes = t3(
            vec![
                1.0, 0.0, 0.0, 1.0, // t=0
                0.0, 1.0, 0.0, 0.0, // t=1
            ],
            [1, 2, 4],
        );
        let target_probs = t2(vec![1.0, 0.0], [1, 2]);
        let pop_loss = scalar(population_ce_count_loss(spikes, 2, target_probs.clone()));

        let manual_counts = t2(vec![2.0, 1.0], [1, 2]);
        let manual_loss = scalar(ce_count_loss(
            // ce_count_loss expects [batch, steps, classes]; here
            // one step with counts already summed over time is
            // enough — sum_over_time of a single step is the identity.
            manual_counts.reshape([1, 1, 2]),
            target_probs,
        ));
        assert!((pop_loss - manual_loss).abs() < 1e-5);
    }
}
