//! Poisson rate encoding: spikes are sampled from intensities.

use burn::tensor::Device;
use burn::tensor::{Distribution, Tensor};

/// Rate encoding: `data` in [0, 1] → spike train in `num_steps` steps.
///
/// Returns the tensor `[batch, num_steps, features]`.
///
/// # Panics
///
/// `num_steps == 0` panics with a clear message, matching
/// [`super::latency_encode`], [`super::periodic_encode`] and
/// [`super::burst_encode`]. Without the check the function would build an
/// empty `Vec` and pass it to `Tensor::stack::<3>`, producing an obscure
/// panic inside `stack` instead of a configuration error that names the
/// real cause (zero number of steps).
pub fn rate_encode(data: Tensor<2>, num_steps: usize, device: &Device) -> Tensor<3> {
    assert!(num_steps > 0, "rate_encode: num_steps must be > 0");

    let [batch, features] = data.dims();
    let mut spikes = Vec::with_capacity(num_steps);

    for _ in 0..num_steps {
        let r: Tensor<2> =
            Tensor::random([batch, features], Distribution::Uniform(0.0, 1.0), device);
        spikes.push(r.lower_equal(data.clone()).float());
    }

    Tensor::stack::<3>(spikes, 1)
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

    /// `num_steps == 0` must panic with a clear message at the function
    /// boundary, not deep inside `Tensor::stack` on an empty `Vec`.
    #[test]
    #[should_panic(expected = "num_steps must be > 0")]
    fn zero_num_steps_panics_with_clear_message() {
        let device = dev();
        let data = t2(vec![0.5], [1, 1]);
        let _ = rate_encode(data, 0, &device);
    }

    /// `data == 1.0` must give a spike at every step - probability 1.0
    /// always passes `r <= data` for `r` to `[0, 1)`.
    #[test]
    fn max_intensity_always_spikes() {
        let device = dev();
        let data = t2(vec![1.0], [1, 1]);
        let out = rate_encode(data, 20, &device)
            .into_data()
            .try_to_vec::<f32>()
            .unwrap();
        assert!(
            out.iter().all(|&v| v == 1.0),
            "data=1.0 must spike at every step: {out:?}"
        );
    }

    /// `data == 0.0` must not give a single spike - probability 0.0
    /// never passes `r <= data` for `r` strictly `> 0`.
    #[test]
    fn zero_intensity_never_spikes() {
        let device = dev();
        let data = t2(vec![0.0], [1, 1]);
        let out = rate_encode(data, 20, &device)
            .into_data()
            .try_to_vec::<f32>()
            .unwrap();
        assert!(
            out.iter().all(|&v| v == 0.0),
            "data=0.0 should not spike at all: {out:?}"
        );
    }
}
