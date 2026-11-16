//! Accuracy metrics over the spike output.
//!
//! ## Why they compute on the host
//!
//! Accuracy is a quantity for humans and logs, not for the autodiff graph:
//! no gradient is taken through it, and at the end of an epoch it would have to
//! be read on the CPU anyway. That is why the data is fetched once through
//! `into_data()` and then computed with ordinary Rust code: no hidden
//! GPU→CPU synchronization in the middle of the training loop, everything
//! happens explicitly and exactly where it is called.
//!
//! All functions expect a backend with `f32` as its element type — like
//! `NdArray<f32>` in the examples and tests.

use burn::tensor::Tensor;

use super::population::{PopulationReduction, population_reduce_seq};
use crate::error::{Result, or_panic, tensor_to_vec};

/// Time step index of the first spike for each `(sample, class)` pair.
///
/// `spikes` — `[batch, num_steps, num_classes]`. `None` means the neuron
/// never fired.
#[track_caller]
pub fn first_spike_times(spikes: Tensor<3>) -> Vec<Vec<Option<usize>>> {
    or_panic(try_first_spike_times(spikes))
}

/// [`first_spike_times`] without panicking: tensor data that cannot be read as
/// `f32` yields [`SnnError::Data`](crate::SnnError::Data).
pub fn try_first_spike_times(spikes: Tensor<3>) -> Result<Vec<Vec<Option<usize>>>> {
    let [batch, steps, classes] = spikes.dims();
    let flat = tensor_to_vec(spikes, "first_spike_times")?;

    let mut out = vec![vec![None; classes]; batch];
    for b in 0..batch {
        for c in 0..classes {
            for t in 0..steps {
                if flat[b * steps * classes + t * classes + c] > 0.5 {
                    out[b][c] = Some(t);
                    break;
                }
            }
        }
    }
    Ok(out)
}

/// Total number of spikes of each class, computed on the host.
///
/// `spikes` — `[batch, num_steps, num_classes]` → `[batch][num_classes]`.
#[track_caller]
pub fn spike_counts(spikes: Tensor<3>) -> Vec<Vec<f32>> {
    or_panic(try_spike_counts(spikes))
}

/// [`spike_counts`] without panicking (see [`try_first_spike_times`]).
pub fn try_spike_counts(spikes: Tensor<3>) -> Result<Vec<Vec<f32>>> {
    let [batch, steps, classes] = spikes.dims();
    let flat = tensor_to_vec(spikes, "spike_counts")?;

    let mut out = vec![vec![0.0f32; classes]; batch];
    for b in 0..batch {
        for t in 0..steps {
            for c in 0..classes {
                out[b][c] += flat[b * steps * classes + t * classes + c];
            }
        }
    }
    Ok(out)
}

/// Accuracy of rate-coded readout: the class with the largest number of
/// spikes wins.
///
/// `targets` — indices of the correct classes, one per sample. Ties are
/// resolved in favor of the lower index — the same rule as `argmax`.
pub fn accuracy_rate(spikes: Tensor<3>, targets: &[usize]) -> f32 {
    let counts = spike_counts(spikes);
    assert_eq!(
        counts.len(),
        targets.len(),
        "accuracy_rate: {} samples vs {} labels",
        counts.len(),
        targets.len()
    );

    let mut correct = 0usize;
    for (row, &target) in counts.iter().zip(targets) {
        let mut best = 0usize;
        for (c, &v) in row.iter().enumerate() {
            if v > row[best] {
                best = c;
            }
        }
        // Empty output (network stayed silent) is not credited to anyone.
        if row[best] > 0.0 && best == target {
            correct += 1;
        }
    }
    correct as f32 / counts.len().max(1) as f32
}

/// Accuracy of latency-coded readout: the class that fired EARLIEST wins.
///
/// A sample where nobody fired counts as an error: there was no answer.
pub fn accuracy_temporal(spikes: Tensor<3>, targets: &[usize]) -> f32 {
    let times = first_spike_times(spikes);
    assert_eq!(
        times.len(),
        targets.len(),
        "accuracy_temporal: {} samples vs {} labels",
        times.len(),
        targets.len()
    );

    let mut correct = 0usize;
    for (row, &target) in times.iter().zip(targets) {
        let mut best: Option<(usize, usize)> = None; // (class, step)
        for (c, &t) in row.iter().enumerate() {
            if let Some(t) = t {
                match best {
                    Some((_, bt)) if bt <= t => {}
                    _ => best = Some((c, t)),
                }
            }
        }
        if let Some((c, _)) = best {
            if c == target {
                correct += 1;
            }
        }
    }
    correct as f32 / times.len().max(1) as f32
}

// ═══════════════════════════════════════════════════════════════════════════
//  Population code: N neurons per class
// ═══════════════════════════════════════════════════════════════════════════
//
// All functions below are thin wrappers: they fold the `num_classes *
// neurons_per_class` axis through `population_reduce`/`population_reduce_seq`
// (see [`super::population`]) and pass the result to the existing per-class
// function. This guarantees that population and single-neuron readout agree by
// construction — not two independent computation paths that may diverge.

/// [`spike_counts`] for population readout: `spikes` — `[batch,
/// num_steps, num_classes * neurons_per_class]`, each group of
/// `neurons_per_class` neurons is first summed over time (as usual), then over
/// the population.
///
/// Returns `[batch][num_classes]`, like [`spike_counts`].
pub fn population_spike_counts(spikes: Tensor<3>, neurons_per_class: usize) -> Vec<Vec<f32>> {
    let classes_total = spikes.dims()[2];
    assert!(
        neurons_per_class >= 1 && classes_total % neurons_per_class == 0,
        "population_spike_counts: number of neurons {classes_total} must be divisible by neurons_per_class = {neurons_per_class} (>= 1)"
    );
    let per_neuron = spike_counts(spikes);
    // spike_counts already returned counts on the host (Vec<Vec<f32>>), so
    // grouping by population here is ordinary Rust code, without a trip to
    // tensors and back. The sum is the only meaningful choice for counters
    // already accumulated over time: it is the total number of spikes of the
    // whole group.
    per_neuron
        .into_iter()
        .map(|row| {
            row.chunks(neurons_per_class)
                .map(|group| group.iter().sum())
                .collect()
        })
        .collect()
}

/// [`accuracy_rate`] for population readout: `spikes` — `[batch,
/// num_steps, num_classes * neurons_per_class]`. The class whose GROUP of
/// neurons produced the largest total number of spikes wins.
///
/// `neurons_per_class = 1` matches [`accuracy_rate`] exactly.
pub fn population_accuracy_rate(
    spikes: Tensor<3>,
    neurons_per_class: usize,
    targets: &[usize],
) -> f32 {
    let counts = population_spike_counts(spikes, neurons_per_class);
    assert_eq!(
        counts.len(),
        targets.len(),
        "population_accuracy_rate: {} samples vs {} labels",
        counts.len(),
        targets.len()
    );

    let mut correct = 0usize;
    for (row, &target) in counts.iter().zip(targets) {
        let mut best = 0usize;
        for (c, &v) in row.iter().enumerate() {
            if v > row[best] {
                best = c;
            }
        }
        if row[best] > 0.0 && best == target {
            correct += 1;
        }
    }
    correct as f32 / counts.len().max(1) as f32
}

/// [`accuracy_temporal`] for population readout: `spikes` — `[batch,
/// num_steps, num_classes * neurons_per_class]`. The class whose GROUP of
/// neurons produced the earliest spike wins (the minimum first-spike time
/// among the group's neurons) — early firing by ANY neuron of the group is
/// credited to the class, so the aggregation here is a minimum, not a
/// sum/mean, unlike [`population_spike_counts`].
///
/// `neurons_per_class = 1` matches [`accuracy_temporal`] exactly.
pub fn population_accuracy_temporal(
    spikes: Tensor<3>,
    neurons_per_class: usize,
    targets: &[usize],
) -> f32 {
    assert!(
        neurons_per_class >= 1,
        "population_accuracy_temporal: neurons_per_class must be >= 1"
    );
    let per_neuron = first_spike_times(spikes);

    let times: Vec<Vec<Option<usize>>> = per_neuron
        .into_iter()
        .map(|row| {
            assert_eq!(
                row.len() % neurons_per_class,
                0,
                "population_accuracy_temporal: {} neurons is not divisible by neurons_per_class ({})",
                row.len(),
                neurons_per_class
            );
            row.chunks(neurons_per_class)
                .map(|group| group.iter().filter_map(|&t| t).min())
                .collect()
        })
        .collect();

    assert_eq!(
        times.len(),
        targets.len(),
        "population_accuracy_temporal: {} samples vs {} labels",
        times.len(),
        targets.len()
    );

    let mut correct = 0usize;
    for (row, &target) in times.iter().zip(targets) {
        let mut best: Option<(usize, usize)> = None;
        for (c, &t) in row.iter().enumerate() {
            if let Some(t) = t {
                match best {
                    Some((_, bt)) if bt <= t => {}
                    _ => best = Some((c, t)),
                }
            }
        }
        if let Some((c, _)) = best {
            if c == target {
                correct += 1;
            }
        }
    }
    correct as f32 / times.len().max(1) as f32
}

/// Folds the neuron population into a per-class tensor reduced over time
/// (`[batch, num_steps, classes * npc]` → `[batch, classes]`).
///
/// A thin wrapper over [`population_reduce_seq`] + [`super::temporal`],
/// exported here for convenience — accuracy readout almost always starts with
/// this step. See [`PopulationReduction`] for the choice of aggregation within
/// a group.
pub fn population_rates(
    spikes: Tensor<3>,
    neurons_per_class: usize,
    reduction: PopulationReduction,
) -> Tensor<2> {
    use super::temporal::mean_over_time;
    mean_over_time(population_reduce_seq(spikes, neurons_per_class, reduction))
}

#[cfg(test)]
mod population_tests {
    use super::*;
    use burn::tensor::Device;

    fn dev() -> Device {
        Device::flex()
    }

    fn t3(data: Vec<f32>, shape: [usize; 3]) -> Tensor<3> {
        Tensor::<1>::from_floats(data.as_slice(), &dev()).reshape(shape)
    }

    #[test]
    fn population_counts_sum_within_group() {
        // batch=1, steps=1, 2 classes x 2 neurons: class0=(1,0), class1=(1,1).
        let spikes = t3(vec![1.0, 0.0, 1.0, 1.0], [1, 1, 4]);
        let counts = population_spike_counts(spikes, 2);
        assert_eq!(counts, vec![vec![1.0, 2.0]]);
    }

    #[test]
    fn population_rate_accuracy_matches_single_neuron_when_npc_one() {
        // Equivalent to accuracy_rate with neurons_per_class = 1.
        let spikes = t3(
            vec![1.0, 0.0, 0.0, 1.0, 1.0, 0.0],
            [2, 1, 3], // batch=2, steps=1, 3 classes
        );
        let targets = [0usize, 1usize];
        let plain = accuracy_rate(spikes.clone(), &targets);
        let pop = population_accuracy_rate(spikes, 1, &targets);
        assert_eq!(plain, pop);
    }

    #[test]
    fn population_rate_accuracy_rewards_group_majority() {
        // 1 sample, 1 step, 2 classes x 2 neurons. Class1 wins by
        // sum (1+1=2) even if no individual neuron is the
        // global maximum among all 4.
        let spikes = t3(vec![1.0, 0.0, 1.0, 1.0], [1, 1, 4]);
        let acc = population_accuracy_rate(spikes, 2, &[1]);
        assert_eq!(acc, 1.0);
    }

    #[test]
    fn population_temporal_accuracy_takes_earliest_in_group() {
        // batch=1, 3 steps, 2 classes x 2 neurons.
        // class0: neuron A silent, neuron B spikes at step 2.
        // class1: neuron A spikes at step 1, neuron B silent.
        // class1 wins (step 1 is earlier than step 2).
        let spikes = t3(
            vec![
                0.0, 0.0, 0.0, 0.0, // t=0
                0.0, 0.0, 1.0, 0.0, // t=1
                1.0, 0.0, 0.0, 0.0, // t=2
            ],
            [1, 3, 4],
        );
        let acc = population_accuracy_temporal(spikes, 2, &[1]);
        assert_eq!(acc, 1.0);
    }
}
