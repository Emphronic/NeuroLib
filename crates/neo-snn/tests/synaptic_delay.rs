// File: crates/neo-snn/tests/synaptic_delay.rs
// Updated: comment "softplus-parameterization" -> "sigmoid-parameterization"
// (deprecated after fixing delay_tensor). Added two regression tests:
// delay_tensor_never_negative_at_extreme_raw and
// delay_tensor_never_exceeds_max_delay_at_extreme_raw - both directly
// overwrite delay_raw with an extreme value and check d ∈ [0, max_delay].

//! `SynapticDelay` tests: integer and fractional delay, equivalence
//! step and sequential modes, boundaries, fixed delay.
//!
//! Launch: `cargo test -p neo-snn --test synaptic_delay`

use burn::tensor::Device;
use burn::tensor::Tensor;

use neo_snn::layers::{DelayState, SynapticDelay, SynapticDelayConfig};

fn dev() -> Device {
    Device::flex()
}

fn t2(data: Vec<f32>, shape: [usize; 2]) -> Tensor<2> {
    Tensor::<1>::from_floats(data.as_slice(), &dev()).reshape(shape)
}

fn t1(data: Vec<f32>) -> Tensor<1> {
    Tensor::<1>::from_floats(data.as_slice(), &dev())
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

/// A pulse at the input of one channel must appear at the output exactly after
/// `delay_init` steps with integer delay - basic check that
/// that interpolation in the absence of a fractional part does not blur the signal.
#[test]
fn integer_delay_shifts_impulse_exactly() {
    let device = dev();
    let config = SynapticDelayConfig::new(1, 5)
        .with_delay_init(3.0)
        .with_learn_delay(false);
    let layer = SynapticDelay::new(&config, &device);

    let mut state = DelayState::zeros(&device, 1, 1, layer.max_delay());
    let input = [1.0f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    let mut out = Vec::with_capacity(input.len());

    for &x in &input {
        let (spike, new_state) = layer.step(t2(vec![x], [1, 1]), &state);
        state = new_state;
        out.push(vec2(spike)[0]);
    }

    let expected = [0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0];
    for (i, (&got, &want)) in out.iter().zip(expected.iter()).enumerate() {
        assert!(
            (got - want).abs() < 1e-5,
            "step {i}: received {got}, expected {want} (delay=3)"
        );
    }
}

/// A fractional delay must linearly spread the impulse between two
/// adjacent integer shifts - direct verification of the interpolation formula.
#[test]
fn fractional_delay_splits_impulse_linearly() {
    let device = dev();
    let config = SynapticDelayConfig::new(1, 5)
        .with_delay_init(2.5)
        .with_learn_delay(false);
    let layer = SynapticDelay::new(&config, &device);

    let mut state = DelayState::zeros(&device, 1, 1, layer.max_delay());
    let input = [1.0f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    let mut out = Vec::with_capacity(input.len());

    for &x in &input {
        let (spike, new_state) = layer.step(t2(vec![x], [1, 1]), &state);
        state = new_state;
        out.push(vec2(spike)[0]);
    }

    assert!(
        (out[2] - 0.5).abs() < 1e-4,
        "step 2: {} (expected 0.5)",
        out[2]
    );
    assert!(
        (out[3] - 0.5).abs() < 1e-4,
        "step 3: {} (expected 0.5)",
        out[3]
    );
    for (i, &v) in out.iter().enumerate() {
        if i != 2 && i != 3 {
            assert!(
                v.abs() < 1e-5,
                "step {i}: leak outside the interpolation window: {v}"
            );
        }
    }
}

/// Zero delay - identity: the layer should not change anything in the stream.
#[test]
fn zero_delay_is_identity() {
    let device = dev();
    let config = SynapticDelayConfig::new(2, 4)
        .with_delay_init(0.0)
        .with_learn_delay(false);
    let layer = SynapticDelay::new(&config, &device);

    let mut state = DelayState::zeros(&device, 1, 2, layer.max_delay());
    let steps: [[f32; 2]; 4] = [[1.0, 2.0], [3.0, -1.0], [0.0, 0.5], [-2.0, 0.0]];

    for &x in &steps {
        let (out, new_state) = layer.step(t2(x.to_vec(), [1, 2]), &state);
        state = new_state;
        let got = vec2(out);
        assert!(
            (got[0] - x[0]).abs() < 1e-5 && (got[1] - x[1]).abs() < 1e-5,
            "d=0 should be identity"
        );
    }
}

/// The delay at the upper bound `max_delay` must behave like
/// an integer shift over the entire length of the buffer, without overflowing for it.
#[test]
fn delay_saturates_at_max_delay() {
    let device = dev();
    let max_delay = 4usize;
    let config = SynapticDelayConfig::new(1, max_delay)
        .with_delay_init(max_delay as f32)
        .with_learn_delay(false);
    let layer = SynapticDelay::new(&config, &device);

    let mut state = DelayState::zeros(&device, 1, 1, layer.max_delay());
    let input = [1.0f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    let mut out = Vec::with_capacity(input.len());

    for &x in &input {
        let (spike, new_state) = layer.step(t2(vec![x], [1, 1]), &state);
        state = new_state;
        out.push(vec2(spike)[0]);
    }

    // The tolerance is wider than in other tests: at the max_delay border itself
    // softplus parameterization delay_tensor (double saturation, see
    // SynapticDelay::delay_tensor) gives d a little LESS than max_delay even after
    // exact inversion of init (last eps clip in inverse_delay_tensor),
    // so the fraction of weight on the adjacent shift remains, but is vanishingly small.
    assert!(
        (out[max_delay] - 1.0).abs() < 1e-3,
        "the impulse should appear at step {max_delay}, obtained {:?}",
        out
    );
}

/// Stepping and sequential modes must coincide at the same
/// delay - they implement the same operation in two different rounds.
#[test]
fn step_and_sequence_modes_agree() {
    let device = dev();
    let config = SynapticDelayConfig::new(2, 6)
        .with_delay_init(2.5)
        .with_learn_delay(false);
    let layer = SynapticDelay::new(&config, &device);

    let steps: Vec<[f32; 2]> = vec![
        [1.0, 0.0],
        [0.5, -0.5],
        [0.0, 1.0],
        [2.0, 0.0],
        [0.0, 0.0],
        [-1.0, 3.0],
        [0.0, 0.0],
        [0.0, 0.0],
    ];
    let n = steps.len();

    // Sequential mode.
    let flat: Vec<f32> = steps.iter().flat_map(|s| s.iter().copied()).collect();
    let seq_out = vec3(layer.forward_sequence(t3(flat, [1, n, 2])));

    // Step mode.
    let mut state = DelayState::zeros(&device, 1, 2, layer.max_delay());
    let mut step_out = Vec::with_capacity(n * 2);
    for &x in &steps {
        let (out, new_state) = layer.step(t2(x.to_vec(), [1, 2]), &state);
        state = new_state;
        step_out.extend(vec2(out));
    }

    for (i, (&a, &b)) in seq_out.iter().zip(step_out.iter()).enumerate() {
        assert!(
            (a - b).abs() < 1e-4,
            "element {i}: sequence={a} vs step={b}"
        );
    }
}

/// `learn_delay = false` must give a layer without a trainable parameter
/// delays (no `Param` in `delay_raw`), otherwise behaving the same way.
#[test]
fn fixed_delay_has_no_learnable_param() {
    let device = dev();
    let config = SynapticDelayConfig::new(1, 3)
        .with_delay_init(1.0)
        .with_learn_delay(false);
    let layer = SynapticDelay::new(&config, &device);

    assert!(
        layer.delay_raw.is_none(),
        "learn_delay=false must leave delay_raw empty"
    );

    let d = vec2(layer.delay_tensor(&device).reshape([1, 1]));
    assert!((d[0] - 1.0).abs() < 1e-4, "fixed delay distorted: {}", d[0]);
}

/// `learn_delay = true` must create the parameter, and `delay_tensor` must
/// reproduce the specified initialization (with the accuracy of sigmoid parameterization).
#[test]
fn learnable_delay_recovers_init_value() {
    let device = dev();
    let config = SynapticDelayConfig::new(3, 8).with_delay_init(4.0);
    let layer = SynapticDelay::new(&config, &device);

    assert!(
        layer.delay_raw.is_some(),
        "learn_delay=true must create a parameter"
    );

    let d = vec2(layer.delay_tensor(&device).reshape([1, 3]));
    for (i, &v) in d.iter().enumerate() {
        assert!((v - 4.0).abs() < 1e-3, "channel {i}: {v} (expected 4.0)");
    }
}

/// `delay_tensor` must remain STRICTLY NON-NEGATIVE even with an extremely
/// negative raw parameter. The sigmoid parameterization
/// `d = max_delay · σ(raw)` keeps `d ∈ (0, max_delay)` for any finite `raw`.
/// The test directly overwrites `delay_raw` with a very negative value,
/// simulating where the trained parameter could go during training, and
/// checks that the delay still remains in `[0, max_delay]`.
#[test]
fn delay_tensor_never_negative_at_extreme_raw() {
    let device = dev();
    let max_delay = 4usize;
    let config = SynapticDelayConfig::new(2, max_delay).with_delay_init(1.0);
    let mut layer = SynapticDelay::new(&config, &device);

    // We rewrite the raw parameter with an extremely negative value -
    // the sigmoid parameterization must saturate near zero, not go negative.
    layer.delay_raw = Some(burn::module::Param::from_tensor(t1(vec![-1000.0, -1000.0])));

    let d = vec2(layer.delay_tensor(&device).reshape([1, 2]));
    for (i, &v) in d.iter().enumerate() {
        assert!(
            v >= 0.0,
            "channel {i}: delay_tensor returned negative delay {v} with extreme raw"
        );
        assert!(
            v <= max_delay as f32,
            "channel {i}: delay_tensor returned delay {v} > max_delay ({max_delay})"
        );
    }
}

/// Symmetrical case: an extremely POSITIVE raw parameter must
/// saturate from BOTTOM to `max_delay`, and not pass beyond it - the same
/// invariant on the other side of the range.
#[test]
fn delay_tensor_never_exceeds_max_delay_at_extreme_raw() {
    let device = dev();
    let max_delay = 4usize;
    let config = SynapticDelayConfig::new(1, max_delay).with_delay_init(1.0);
    let mut layer = SynapticDelay::new(&config, &device);

    layer.delay_raw = Some(burn::module::Param::from_tensor(t1(vec![1000.0])));

    let d = vec2(layer.delay_tensor(&device).reshape([1, 1]));
    assert!(
        d[0] <= max_delay as f32 + 1e-4,
        "delay_tensor returned {} > max_delay ({max_delay}) with extreme raw",
        d[0]
    );
    assert!(d[0] >= 0.0, "delay_tensor returned negative delay {}", d[0]);
}

/// `DelayState` uses `VecDeque` with `pop_front`/`push_back` (both `O(1)`),
/// so the observable behavior is unchanged: this test takes noticeably more
/// steps than `max_delay`, so the ring repeatedly overflows and rotates, and
/// checks that an integer delay still shifts each pulse by exactly
/// `delay` steps.
#[test]
fn many_steps_still_shift_correctly_after_ring_buffer_wraps_many_times() {
    let device = dev();
    let max_delay = 4usize;
    let delay = 3.0f32;
    let config = SynapticDelayConfig::new(1, max_delay)
        .with_delay_init(delay)
        .with_learn_delay(false);
    let layer = SynapticDelay::new(&config, &device);

    // Noticeably more steps than max_delay - the buffer must turn
    // (overflow and push out old elements) many times in a row.
    let num_steps = 50usize;
    let mut input = vec![0.0f32; num_steps];
    // Pulses at several isolated steps, sufficiently spaced,
    // so that their shifted copies do not overlap (delay=3, spacing by 6).
    let impulse_steps = [0usize, 6, 12, 24, 40];
    for &i in &impulse_steps {
        input[i] = 1.0;
    }

    let mut state = DelayState::zeros(&device, 1, 1, layer.max_delay());
    let mut out = Vec::with_capacity(num_steps);
    for &x in &input {
        let (spike, new_state) = layer.step(t2(vec![x], [1, 1]), &state);
        state = new_state;
        out.push(vec2(spike)[0]);
    }

    for &i in &impulse_steps {
        let expected_at = i + delay as usize;
        if expected_at < num_steps {
            assert!(
                (out[expected_at] - 1.0).abs() < 1e-4,
                "impulse from step {i} should appear at step {expected_at}, received {}",
                out[expected_at]
            );
        }
    }
}

/// Panic when the number of input channels and layer configuration do not match.
#[test]
#[should_panic(expected = "channels")]
fn step_panics_on_channel_mismatch() {
    let device = dev();
    let config = SynapticDelayConfig::new(2, 3).with_learn_delay(false);
    let layer = SynapticDelay::new(&config, &device);
    let state = DelayState::zeros(&device, 1, 2, layer.max_delay());

    // 3 channel input with layer configured to 2.
    let _ = layer.step(t2(vec![0.0, 0.0, 0.0], [1, 3]), &state);
}
