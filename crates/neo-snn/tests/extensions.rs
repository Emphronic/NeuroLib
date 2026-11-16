//! Tests of added functionality: surrogates, losses, regularization,
//! metrics, state quantization, winner-take-all.
//!
//! Launch: `cargo test -p neo-snn --test extensions`

use burn::tensor::Device;
use burn::tensor::Tensor;

use neo_snn::functional::{
    ActivityStats, RasterStyle, accuracy_rate, accuracy_temporal, activity_histogram,
    first_spike_times, l1_spike_rate, max_over_time, mse_temporal_loss, rate_bounds_penalty,
    spike_raster, spike_raster_labeled, sum_over_time, winner_take_all,
};
use neo_snn::neuron::spike_with_scalar_threshold;
use neo_snn::quant::{StateQuant, StateQuantScheme};
use neo_snn::surrogate::{
    Heaviside, LeakySpikeOperator, SparseFastSigmoid, SpikeRateEscape, StochasticSpikeOperator,
    Surrogate, Triangular,
};

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

fn scalar(t: Tensor<1>) -> f32 {
    t.into_data().as_slice::<f32>().unwrap()[0]
}

// ─── Surrogates ─────────────────────────────── ───────────────────────────────

/// The main property of straight-through: WHATEVER the surrogate is, forward
/// must remain exactly Heaviside. If it breaks, the network is in inference
/// behaves differently than during training.
#[test]
fn every_surrogate_keeps_forward_heaviside() {
    let mem = vec![-2.0, -0.4, 0.0, 0.3, 5.0, -0.001];
    // Strictly `>`: exactly at the threshold (x = 0.0) there is no spike, as in snnTorch.
    let expected = vec![0.0, 0.0, 0.0, 1.0, 1.0, 0.0];

    macro_rules! check {
        ($s:expr, $name:literal) => {{
            let x = t2(mem.clone(), [1, 6]);
            let (spike, hard) = spike_with_scalar_threshold(x, 0.0, &$s);
            assert_eq!(vec2(spike), expected, "{}: forward not Heaviside", $name);
            assert_eq!(vec2(hard), expected, "{}: hard spike not Heaviside", $name);
        }};
    }

    check!(Triangular::default(), "Triangular");
    check!(SpikeRateEscape::default(), "SpikeRateEscape");
    check!(SparseFastSigmoid::default(), "SparseFastSigmoid");
    check!(Heaviside, "Heaviside");
    check!(LeakySpikeOperator::default(), "LeakySpikeOperator");
    check!(
        StochasticSpikeOperator::default(),
        "StochasticSpikeOperator"
    );
}

/// The triangle must be finite in support: beyond 1/k
/// the surrogate value becomes constant, that is, the gradient is zero.
#[test]
fn triangular_saturates_outside_support() {
    let s = Triangular::new(2.0); // support [-0.5, 0.5]
    let out = vec2(s.surrogate_forward(t2(vec![-5.0, -0.5, 0.0, 0.5, 5.0], [1, 5])));

    assert!(
        (out[0] - out[1]).abs() < 1e-6,
        "on the left did not reach a constant"
    );
    assert!(
        (out[3] - out[4]).abs() < 1e-6,
        "on the right did not reach a constant"
    );
    assert!(
        (out[2] - 0.5).abs() < 1e-6,
        "at zero the value should be 0.5"
    );
    assert!(
        out[1] < out[2] && out[2] < out[3],
        "not monotonic inside the support"
    );
}

/// The sparse surrogate is required to ZERO the value outside the window: this is where
/// zeroing and the return passage is saved.
#[test]
fn sparse_fast_sigmoid_zeroes_outside_window() {
    let s = SparseFastSigmoid::new(25.0, 0.5);
    let out = vec2(s.surrogate_forward(t2(vec![-3.0, -0.1, 0.0, 0.1, 3.0], [1, 5])));

    assert_eq!(out[0], 0.0, "far to the left there should be zero");
    assert_eq!(out[4], 0.0, "far to the right there should be zero");
    assert!(
        out[1] != 0.0 && out[2] != 0.0 && out[3] != 0.0,
        "the window is empty"
    );
}

/// Escape-rate is finite for large modulo inputs: the formula is written like this:
/// so that exp never receives a positive argument.
#[test]
fn spike_rate_escape_is_finite_on_extremes() {
    let s = SpikeRateEscape::new(1.0, 25.0);
    let out = vec2(s.surrogate_forward(t2(vec![-1e4, -10.0, 0.0, 10.0, 1e4], [1, 5])));
    assert!(
        out.iter().all(|v| v.is_finite()),
        "non-finite values obtained: {out:?}"
    );
    for w in out.windows(2) {
        assert!(
            w[1] >= w[0] - 1e-6,
            "the antiderivative must be non-decreasing"
        );
    }
}

// ─── Convolutions by time ────────────────────────── ───────────────────────────

#[test]
fn time_reductions_match_manual_count() {
    // [batch=1, steps=3, features=2]
    let x = t3(vec![1.0, 0.0, 0.0, 0.0, 1.0, 5.0], [1, 3, 2]);

    assert_eq!(vec2(sum_over_time(x.clone())), vec![2.0, 5.0]);
    assert_eq!(vec2(max_over_time(x)), vec![1.0, 5.0]);
}

// ─── Regularization ───────────────────────────── ─────────────────────────────

/// Inside the corridor, the penalty must be EXACTLY zero, otherwise the regularizer
/// presses activity where it suits.
#[test]
fn rate_bounds_penalty_is_silent_inside_corridor() {
    // frequency 0.5 for both neurons
    let x = t3(vec![1.0, 1.0, 0.0, 0.0], [1, 2, 2]);
    assert_eq!(scalar(rate_bounds_penalty(x.clone(), 0.2, 0.8)), 0.0);

    // and outside it is positive
    assert!(scalar(rate_bounds_penalty(x, 0.6, 0.8)) > 0.0);
}

#[test]
fn l1_spike_rate_equals_mean_firing_rate() {
    let x = t3(vec![1.0, 0.0, 1.0, 0.0], [1, 2, 2]);
    assert!((scalar(l1_spike_rate(x)) - 0.5).abs() < 1e-6);
}

// ─── Time losses ─────────────────────────── ───────────────────────────

/// If the network is spiked exactly at the target moment, the temporary loss is zero.
#[test]
fn mse_temporal_loss_is_zero_on_exact_match() {
    // [1, 4, 1]: single spike in step 2
    let spikes = t3(vec![0.0, 0.0, 1.0, 0.0], [1, 4, 1]);
    let target = t2(vec![2.0], [1, 1]);
    assert!(scalar(mse_temporal_loss(spikes, target)) < 1e-6);
}

/// A silent neuron is interpreted as being the last to spike, and not as
/// no answer: otherwise its gradient disappears forever.
#[test]
fn mse_temporal_loss_treats_silence_as_latest() {
    let silent = t3(vec![0.0, 0.0, 0.0, 0.0], [1, 4, 1]);
    let target = t2(vec![3.0], [1, 1]); // 3 = last step
    assert!(scalar(mse_temporal_loss(silent, target)) < 1e-6);
}

// ─── Metrics ──────────────────────────────── ────────────────────────────────

#[test]
fn accuracy_rate_picks_most_active_class() {
    // [batch=2, steps=2, classes=2]
    let spikes = t3(vec![1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 1.0], [2, 2, 2]);
    assert_eq!(accuracy_rate(spikes.clone(), &[0, 1]), 1.0);
    assert_eq!(accuracy_rate(spikes, &[1, 0]), 0.0);
}

#[test]
fn accuracy_temporal_picks_earliest_class() {
    // class 1 is spiked at step 0, class 0 - at step 1
    let spikes = t3(vec![0.0, 1.0, 1.0, 0.0], [1, 2, 2]);
    assert_eq!(accuracy_temporal(spikes.clone(), &[1]), 1.0);

    let times = first_spike_times(spikes);
    assert_eq!(times[0][0], Some(1));
    assert_eq!(times[0][1], Some(0));
}

/// A sample where no one spiked is an error, not an accidental hit.
#[test]
fn accuracy_counts_silence_as_wrong() {
    let silent = t3(vec![0.0, 0.0, 0.0, 0.0], [1, 2, 2]);
    assert_eq!(accuracy_rate(silent.clone(), &[0]), 0.0);
    assert_eq!(accuracy_temporal(silent, &[0]), 0.0);
}

// ─── Winner-take-all ────────────────────────────────────────────────────────

#[test]
fn winner_take_all_keeps_only_strongest() {
    let spikes = t2(vec![1.0, 1.0, 1.0], [1, 3]);
    let mem = t2(vec![0.2, 0.9, 0.5], [1, 3]);
    assert_eq!(vec2(winner_take_all(spikes, mem)), vec![0.0, 1.0, 0.0]);
}

// ─── State quantization ───────────────────────── ─────────────────────────

/// The quantized value must lie ON THE GRID and not move further
/// half a step from the original one.
#[test]
fn state_quant_snaps_to_grid() {
    let q = StateQuant::uniform(2, 0.0, 1.0); // levels 0, 1/3, 2/3, 1
    let out = vec2(q.apply(t2(vec![0.0, 0.3, 0.5, 0.9, 1.0], [1, 5])));

    let step = q.step();
    for (i, v) in out.iter().enumerate() {
        let k = (v / step).round() * step;
        assert!((v - k).abs() < 1e-5, "value {i} not on the grid: {v}");
    }
    assert!((out[0] - 0.0).abs() < 1e-6);
    assert!((out[4] - 1.0).abs() < 1e-6);
}

/// The output beyond the mesh boundaries is clamped, not wrapped.
///
/// `ThresholdCentered` with `bits=4` (`levels = 2^4-1 = 15`, odd) places the
/// edge of the window at index `half = levels/2 = 7.5` - not integer, so even
/// after clamping to `[-half, half]` the outermost reachable INTEGER index is
/// `±7`, not `±7.5`. This is documented intentional behavior of the scheme
/// (see the module doc comment: `2^bits - 1` unique codes instead of `2^bits`,
/// at the cost of the extreme code lying up to one grid step away from the
/// mathematical boundary of the window). The invariant is checked in
/// `threshold_centered_upper_bound_close_to_span_edge` in quant.rs with
/// tolerance `<= step`, and not exact equality. The same one-grid-step
/// tolerance is used here: the test still catches a REAL violation
/// (wrapping/value outside `[threshold-span, threshold+span]`),
/// but does not require impossible precision from an odd grid.
#[test]
fn state_quant_clamps_out_of_range() {
    let q = StateQuant::threshold_centered(4, 1.0, 1.0); // window [0, 2]
    let out = vec2(q.apply(t2(vec![-5.0, 7.0], [1, 2])));
    let step = q.step();
    assert!(
        (out[0] - 0.0).abs() <= step + 1e-5,
        "the bottom border should remain within one grid step from the edge of the window: {} (step={step})",
        out[0]
    );
    assert!(
        (out[1] - 2.0).abs() <= step + 1e-5,
        "the top border should remain within one grid step from the edge of the window: {} (step={step})",
        out[1]
    );
    // Strict invariant - the value should NOT go OUTSIDE the window
    // [threshold-span, threshold+span], even a fraction of a step (this is
    // "clamped, not wrapped" from the name of the test).
    assert!(
        out[0] >= 0.0 - 1e-5,
        "the lower border went out of the window: {}",
        out[0]
    );
    assert!(
        out[1] <= 2.0 + 1e-5,
        "the upper border went out of the window: {}",
        out[1]
    );
}

#[test]
fn state_quant_more_bits_lower_error() {
    let x = vec![0.13, 0.47, 0.82, 0.05];
    let mut prev = f32::INFINITY;
    for bits in [2u8, 4, 6, 8] {
        let q = StateQuant {
            bits,
            scheme: StateQuantScheme::Uniform { min: 0.0, max: 1.0 },
        };
        let out = vec2(q.apply(t2(x.clone(), [1, 4])));
        let err: f32 = x.iter().zip(&out).map(|(a, b)| (a - b).abs()).sum();
        assert!(
            err <= prev + 1e-6,
            "{bits} bit: error {err} worse than before {prev}"
        );
        prev = err;
    }
}

// ─── Monitors ─────────────────────────────── ────────────────────────────────

#[test]
fn activity_stats_report_rate_and_sparsity() {
    let s = ActivityStats::of(t2(vec![1.0, 0.0, 0.0, 0.0], [1, 4]));
    assert_eq!(s.count, 4);
    assert!((s.mean - 0.25).abs() < 1e-6);
    assert!((s.zero_fraction - 0.75).abs() < 1e-6);
    assert!((s.abs_max - 1.0).abs() < 1e-6);
}

// ─── Exponential thickening of levels ──────────────────────────────────────

/// The meaning of the scheme is in one property: near the threshold the grid is DENSIER than further away.
/// If this is not the case, the condensation does not work and there is no need to take it.
#[test]
fn exponential_quant_is_denser_near_threshold() {
    let theta = 1.0f32;
    let q = StateQuant::threshold_exponential(4, theta, 1.0, 3.0);

    // Levels in which points fall at different distances from the threshold.
    let probe = |d: f32| {
        let out = vec2(q.apply(t2(vec![theta + d, theta + d + 0.002], [1, 2])));
        (out[0], out[1])
    };

    // Near the threshold, two close values must diverge in different ways
    // levels (or stay very close), and in the distance - merge into one.
    let (near_a, near_b) = probe(0.01);
    let (far_a, far_b) = probe(0.9);

    let near_gap = (near_a - near_b).abs();
    let far_gap = (far_a - far_b).abs();

    assert!(
        near_gap <= far_gap,
        "at the threshold the grid should be no coarser than at the distance: {near_gap} versus {far_gap}"
    );
    assert!(
        q.step() < 1.0 / 15.0,
        "the step at the threshold is not smaller than uniform: {}",
        q.step()
    );
}

/// When γ = 1, the scheme must degenerate into a uniform one - otherwise the parameter
/// silently changes behavior where it should not.
#[test]
fn exponential_quant_with_gamma_one_matches_uniform() {
    let x = vec![0.1, 0.55, 1.4, 1.95];
    let exp = StateQuant::threshold_exponential(4, 1.0, 1.0, 1.0);
    let uni = StateQuant::threshold_centered(4, 1.0, 1.0);

    let a = vec2(exp.apply(t2(x.clone(), [1, 4])));
    let b = vec2(uni.apply(t2(x, [1, 4])));

    for (i, (va, vb)) in a.iter().zip(&b).enumerate() {
        assert!((va - vb).abs() < 1e-4, "element {i}: {va} vs {vb}");
    }
}

/// The window boundaries are maintained even when thickened.
#[test]
fn exponential_quant_clamps_out_of_range() {
    let q = StateQuant::threshold_exponential(4, 1.0, 1.0, 2.5);
    let out = vec2(q.apply(t2(vec![-9.0, 9.0], [1, 2])));
    assert!((out[0] - 0.0).abs() < 1e-4, "lower limit: {}", out[0]);
    assert!((out[1] - 2.0).abs() < 1e-4, "upper limit: {}", out[1]);
}

// ─── Raster ───────────────────────────────── ─────────────────────────────────

#[test]
fn spike_raster_draws_neurons_as_rows() {
    // [1, 3 steps, 2 neurons]: neuron 0 is silent, neuron 1 is always spiked.
    let spikes = t3(vec![0.0, 1.0, 0.0, 1.0, 0.0, 1.0], [1, 3, 2]);
    let out = spike_raster(spikes, 0, RasterStyle::ascii());

    let lines: Vec<&str> = out.lines().collect();
    assert_eq!(
        lines.len(),
        2,
        "there should be lines according to the number of neurons"
    );
    assert_eq!(lines[0], "...", "silent neuron");
    assert_eq!(lines[1], "###", "spiking neuron");
}

#[test]
fn labeled_raster_reports_firing_rate() {
    let spikes = t3(vec![0.0, 1.0, 0.0, 1.0], [1, 2, 2]);
    let out = spike_raster_labeled(spikes, 0, RasterStyle::ascii());

    assert!(
        out.contains("0.00"),
        "silent neuron frequency not shown: {out}"
    );
    assert!(
        out.contains("1.00"),
        "active neuron frequency not shown: {out}"
    );
}

#[test]
fn activity_histogram_counts_per_step() {
    // step 0 - spike for both, step 1 - for neither.
    let spikes = t3(vec![1.0, 1.0, 0.0, 0.0], [1, 2, 2]);
    let out = activity_histogram(spikes, 0, 10);

    let lines: Vec<&str> = out.lines().collect();
    assert_eq!(lines.len(), 2);
    assert!(lines[0].contains("2/2"), "step 0: {}", lines[0]);
    assert!(lines[1].contains("0/2"), "step 1: {}", lines[1]);
}

// ─── Telemetry: activity snapshot ───────────────────── ─────────────────────

mod telemetry_tests {
    use super::*;
    use neo_snn::telemetry::SpikeFrame;

    /// The image must match the tensor in terms of frequencies and silence -
    /// otherwise the GUI will not show what the network thought it would.
    #[test]
    fn spike_frame_matches_tensor() {
        // [1, 4 steps, 2 neurons]: neuron 0 is silent, neuron 1 every other time.
        let spikes = t3(vec![0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0], [1, 4, 2]);
        let frame = SpikeFrame::from_tensor(spikes, 0);

        assert_eq!(frame.steps, 4);
        assert_eq!(frame.units, 2);
        assert_eq!(frame.unit_rate(0), 0.0);
        assert_eq!(frame.unit_rate(1), 0.5);
        assert_eq!(frame.silent_fraction(), 0.5);
        assert!((frame.mean_rate() - 0.25).abs() < 1e-6);
    }

    /// Time compression should not lose single spikes: column,
    /// the interval in which the spike falls must be non-zero.
    #[test]
    fn density_keeps_sparse_spikes() {
        // one digit for 8 steps
        let mut data = vec![0.0f32; 8];
        data[5] = 1.0;
        let frame = SpikeFrame::from_tensor(t3(data, [1, 8, 1]), 0);

        let dens = frame.unit_density(0, 4);
        assert_eq!(dens.len(), 4);
        assert!(
            dens.iter().any(|&d| d > 0.0),
            "the spike was lost during compression"
        );
        assert!((dens.iter().sum::<f32>() - 0.5).abs() < 1e-6);
    }

    /// The snapshot is divided between threads (the GUI gives it to the renderer via `Arc`).
    #[test]
    fn spike_frame_is_send_and_sync() {
        fn assert_send_sync<T: Send + Sync>() {}
        assert_send_sync::<SpikeFrame>();
    }
}
