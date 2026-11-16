//! Population-code readout: N neurons per class instead of one.
//!
//! # Why
//!
//! In the simplest readout scheme the output layer has exactly `num_classes`
//! neurons, and the class is determined by a single neuron (its firing rate,
//! first-spike time, or membrane). In practice this does not scale well:
//! extending the classifier requires extending the output layer one-to-one
//! with the number of classes, and a single neuron is a noisy, easily
//! "burned-out" counting unit.
//!
//! Population code is the same trick as group convolution in CNNs or
//! multi-head in transformers: each class is allocated a GROUP of
//! `neurons_per_class` neurons, and the decision is made from the group
//! aggregate (usually the sum or the mean). This simultaneously averages out
//! the noise of individual neurons and does not tie the number of classes to
//! the number of individual learnable "detectors" one-to-one — you can grow
//! `neurons_per_class` without touching the class semantics.
//!
//! # Layout
//!
//! The output of a population readout layer is `[..., num_classes *
//! neurons_per_class]`, where the neuron group of class `c` occupies the slice
//! `[c * neurons_per_class .. (c + 1) * neurons_per_class]` of the last axis
//! (the same order as `chunk`/`view` in PyTorch implementations of population
//! coding, such as snnTorch). [`population_reduce`] /
//! [`population_reduce_seq`] fold this axis into `[..., num_classes]`, after
//! which the result is an ordinary per-class tensor accepted by all functions
//! in [`super::loss`] and [`super::acc`].
//!
//! There are two variants because the library has exactly two tensor shapes
//! with classes on the last axis: already reduced over time (`[batch,
//! classes]`, as used for counts and logits) and the raw spike/membrane train
//! (`[batch, steps, classes]`). Generalizing them with one function through a
//! const generic over the rank would needlessly complicate the code for two
//! concrete cases that already cover all uses in [`super::loss`] and
//! [`super::acc`].

use burn::tensor::Tensor;

/// Aggregation method for neurons within one group (one class).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum PopulationReduction {
    /// Sum of the group's activities. Natural for spike COUNTS: the total
    /// number of spikes of a class is the sum of the spikes of its neurons.
    #[default]
    Sum,
    /// Mean of the group's activities. Gives the same scale as one neuron per
    /// class, regardless of `neurons_per_class` — convenient when
    /// comparability with that scheme matters (for example, before
    /// [`super::loss::ce_rate_loss`], where the input is treated as logits).
    Mean,
    /// Maximum of the group's activities. A class wins if at least one of its
    /// neurons fired more strongly than all the others — gentler toward
    /// "silent" neurons of the group than the sum or mean.
    Max,
}

/// Checks that the last axis is divisible by `neurons_per_class` without a
/// remainder and returns `num_classes`.
fn classes_from_last_axis(last: usize, neurons_per_class: usize, ctx: &str) -> usize {
    assert!(
        neurons_per_class >= 1,
        "{ctx}: neurons_per_class must be >= 1"
    );
    assert_eq!(
        last % neurons_per_class,
        0,
        "{ctx}: last axis ({last}) is not divisible by neurons_per_class ({neurons_per_class})"
    );
    last / neurons_per_class
}

/// Folds `[batch, num_classes * neurons_per_class]` into `[batch,
/// num_classes]`, aggregating each group of `neurons_per_class` neurons
/// according to `reduction`.
///
/// For tensors already reduced over time: spike counts
/// ([`super::acc::spike_counts`]), rates, membrane maximum, etc.
pub fn population_reduce(
    x: Tensor<2>,
    neurons_per_class: usize,
    reduction: PopulationReduction,
) -> Tensor<2> {
    let [batch, last] = x.dims();
    let num_classes = classes_from_last_axis(last, neurons_per_class, "population_reduce");

    if neurons_per_class == 1 {
        return x;
    }

    let x3 = x.reshape([batch, num_classes, neurons_per_class]);
    match reduction {
        PopulationReduction::Sum => x3.sum_dim(2).reshape([batch, num_classes]),
        PopulationReduction::Mean => x3.mean_dim(2).reshape([batch, num_classes]),
        PopulationReduction::Max => x3.max_dim(2).reshape([batch, num_classes]),
    }
}

/// Folds `[batch, num_steps, num_classes * neurons_per_class]` into
/// `[batch, num_steps, num_classes]`, aggregating each group of
/// `neurons_per_class` neurons according to `reduction`, INDEPENDENTLY at
/// each time step.
///
/// For a raw spike or membrane train — before reduction over time by functions
/// from [`super::temporal`]. Applying it here, BEFORE the reduction over time
/// rather than after, gives the same result for
/// [`PopulationReduction::Sum`] and [`PopulationReduction::Mean`] (both are
/// linear and commute with summation/averaging over time), but not for
/// [`PopulationReduction::Max`] — max over the population and max over time do
/// not commute. A single order is chosen here so that behavior does not change
/// with the selected reduction.
pub fn population_reduce_seq(
    x: Tensor<3>,
    neurons_per_class: usize,
    reduction: PopulationReduction,
) -> Tensor<3> {
    let [batch, steps, last] = x.dims();
    let num_classes = classes_from_last_axis(last, neurons_per_class, "population_reduce_seq");

    if neurons_per_class == 1 {
        return x;
    }

    let x4 = x.reshape([batch, steps, num_classes, neurons_per_class]);
    match reduction {
        PopulationReduction::Sum => x4.sum_dim(3).reshape([batch, steps, num_classes]),
        PopulationReduction::Mean => x4.mean_dim(3).reshape([batch, steps, num_classes]),
        PopulationReduction::Max => x4.max_dim(3).reshape([batch, steps, num_classes]),
    }
}

#[cfg(test)]
mod tests {
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

    fn vec2(t: Tensor<2>) -> Vec<f32> {
        t.into_data().as_slice::<f32>().unwrap().to_vec()
    }

    fn vec3(t: Tensor<3>) -> Vec<f32> {
        t.into_data().as_slice::<f32>().unwrap().to_vec()
    }

    #[test]
    fn sum_reduces_groups_of_neurons() {
        // 2 classes, 3 neurons per class, 1 sample.
        let x = t2(vec![1.0, 2.0, 3.0, 10.0, 20.0, 30.0], [1, 6]);
        let out = population_reduce(x, 3, PopulationReduction::Sum);
        assert_eq!(vec2(out), vec![6.0, 60.0]);
    }

    #[test]
    fn mean_matches_single_neuron_scale() {
        let x = t2(vec![1.0, 3.0, 5.0, 2.0, 2.0, 2.0], [1, 6]);
        let out = population_reduce(x, 3, PopulationReduction::Mean);
        assert_eq!(vec2(out), vec![3.0, 2.0]);
    }

    #[test]
    fn max_picks_best_neuron_in_group() {
        let x = t2(vec![1.0, 9.0, 3.0, 4.0, 5.0, 6.0], [1, 6]);
        let out = population_reduce(x, 3, PopulationReduction::Max);
        assert_eq!(vec2(out), vec![9.0, 6.0]);
    }

    #[test]
    fn neurons_per_class_one_is_identity() {
        let x = t2(vec![1.0, 2.0, 3.0], [1, 3]);
        let out = population_reduce(x.clone(), 1, PopulationReduction::Sum);
        assert_eq!(vec2(out), vec2(x));
    }

    #[test]
    fn works_on_batch_of_several_samples() {
        // 2 samples, 2 classes, 2 neurons per class.
        let x = t2(vec![1.0, 1.0, 2.0, 2.0, 0.0, 4.0, 1.0, 1.0], [2, 4]);
        let out = population_reduce(x, 2, PopulationReduction::Sum);
        assert_eq!(vec2(out), vec![2.0, 4.0, 4.0, 2.0]);
    }

    #[test]
    #[should_panic(expected = "not divisible")]
    fn panics_when_last_axis_not_divisible() {
        let x = t2(vec![1.0, 2.0, 3.0, 4.0, 5.0], [1, 5]);
        let _ = population_reduce(x, 3, PopulationReduction::Sum);
    }

    #[test]
    fn seq_reduces_independently_per_step() {
        // batch=1, steps=2, 2 classes x 2 neurons.
        let x = t3(
            vec![
                // t=0: class0=(1,1)=2, class1=(0,4)=4
                1.0, 1.0, 0.0, 4.0, // t=1: class0=(2,0)=2, class1=(1,1)=2
                2.0, 0.0, 1.0, 1.0,
            ],
            [1, 2, 4],
        );
        let out = population_reduce_seq(x, 2, PopulationReduction::Sum);
        assert_eq!(vec3(out), vec![2.0, 4.0, 2.0, 2.0]);
    }

    #[test]
    fn seq_agrees_with_reduce_per_time_slice() {
        let x = t3(vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0], [1, 2, 4]);
        let seq_out = vec3(population_reduce_seq(
            x.clone(),
            2,
            PopulationReduction::Mean,
        ));

        // The same result as applying population_reduce to each time slice
        // separately.
        let t0 = x.clone().slice([0..1, 0..1, 0..4]).reshape([1, 4]);
        let t1 = x.slice([0..1, 1..2, 0..4]).reshape([1, 4]);
        let r0 = vec2(population_reduce(t0, 2, PopulationReduction::Mean));
        let r1 = vec2(population_reduce(t1, 2, PopulationReduction::Mean));

        assert_eq!(seq_out, [r0, r1].concat());
    }
}
