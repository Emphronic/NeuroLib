//! Regression tests for delta gating and for `latency_encode`.
//!
//! Launch: `cargo test -p neo-snn`

use burn::tensor::Device;
use burn::tensor::Tensor;

use neo_snn::encoding::latency_encode;
use neo_snn::event_driven::EventDrivenKANSynapse;

fn t2(data: Vec<f32>, shape: [usize; 2]) -> Tensor<2> {
    Tensor::<1>::from_floats(data.as_slice(), &Device::flex()).reshape(shape)
}

fn vec(t: Tensor<2>) -> Vec<f32> {
    t.into_data().as_slice::<f32>().unwrap().to_vec()
}

/// Slow drift (each step less than delta) must eventually open the
/// gate: the gate compares the current trace against the last accepted
/// reference, which is not overwritten on every step.
#[test]
fn slow_drift_eventually_opens_gate() {
    let device = Device::flex();
    let delta = 0.1;
    let syn = EventDrivenKANSynapse::new(2, 3, 5, delta, true, &device);

    // First step: there are no reference values yet, it is considered fair.
    let mut trace = t2(vec![0.0, 0.0], [1, 2]);
    let (first_out, mut ref_trace, mut ref_out) = syn.forward(trace.clone(), None, None);
    let first = vec(first_out);

    // Drift of 0.03 per step: single step less than delta = 0.1.
    let mut opened = false;
    for _ in 0..10 {
        trace = trace + t2(vec![0.03, 0.03], [1, 2]);
        let (out, new_ref_trace, new_ref_out) =
            syn.forward(trace.clone(), Some(ref_trace), Some(ref_out));
        if vec(out) != first {
            opened = true;
        }
        ref_trace = new_ref_trace;
        ref_out = new_ref_out;
    }
    assert!(
        opened,
        "the gate never opened with accumulated drift > delta"
    );
}

/// As long as the change is less than delta, the output should remain the same.
#[test]
fn small_change_keeps_output() {
    let device = Device::flex();
    let syn = EventDrivenKANSynapse::new(2, 3, 5, 0.5, true, &device);

    let trace0 = t2(vec![0.1, 0.2], [1, 2]);
    let (out0, ref_trace, ref_out) = syn.forward(trace0, None, None);

    let trace1 = t2(vec![0.12, 0.21], [1, 2]); // change << 0.5
    let (out1, _, _) = syn.forward(trace1, Some(ref_trace), Some(ref_out));

    assert_eq!(vec(out0), vec(out1));
}

/// Each neuron in the latency code produces EXACTLY one spike on any inputs,
/// including values that fall exactly on bin boundaries.
#[test]
fn latency_encode_exactly_one_spike() {
    let device = Device::flex();
    let num_steps = 5;
    // Values at the boundaries: 0, 0.25, 0.5, 0.75, 1 plus out of range.
    let data = t2(vec![0.0, 0.25, 0.5, 0.75, 1.0, -0.3, 1.7], [1, 7]);

    let spikes = latency_encode(data, num_steps, &device); // [1, steps, 7]
    let flat = spikes.into_data().as_slice::<f32>().unwrap().to_vec();

    for f in 0..7 {
        let mut sum = 0.0;
        for t in 0..num_steps {
            sum += flat[t * 7 + f];
        }
        assert_eq!(
            sum, 1.0,
            "neuron {f}: exactly one spike was expected, not {sum}"
        );
    }
}
