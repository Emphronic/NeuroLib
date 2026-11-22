//! Integration tests for the user-defined tensor neuron API
//! (`SpikingLayer` + `Session` + autodiff).
//!
//! Run: `cargo test -p neo-snn --test custom_tensor_neuron`

use burn::module::{Module, Param};
use burn::optim::{AdamConfig, GradientsParams};
use burn::tensor::{Device, Tensor};
use neo_snn::neuron::spike_with_scalar_threshold;
use neo_snn::spiking_layer::{Deferred, Session, SpikingLayer, SpikingLayerExt};
use neo_snn::stateful::Detach;
use neo_snn::surrogate::{FastSigmoid, Surrogate};

// ---------------------------------------------------------------------
// FitzHugh–Nagumo: two state variables (v, w), custom equations.
// ---------------------------------------------------------------------

#[derive(Module, Debug)]
struct FhnParams {
    a: Param<Tensor<1>>,
    b: Param<Tensor<1>>,
    v_reset: f32,
    threshold: f32,
    w_incr: f32,
}

impl FhnParams {
    fn new(a: f32, b: f32, v_reset: f32, threshold: f32, w_incr: f32, device: &Device) -> Self {
        Self {
            a: Param::from_tensor(Tensor::from_floats([a], device)),
            b: Param::from_tensor(Tensor::from_floats([b], device)),
            v_reset,
            threshold,
            w_incr,
        }
    }
}

#[derive(Debug, Clone)]
struct FhnState {
    v: Tensor<2>,
    w: Tensor<2>,
}

impl Detach for FhnState {
    fn detach(self) -> Self {
        Self {
            v: self.v.detach(),
            w: self.w.detach(),
        }
    }
}

#[derive(Module, Debug)]
struct Fhn {
    params: FhnParams,
    dt: f32,
}

impl Fhn {
    fn new(
        a: f32,
        b: f32,
        v_reset: f32,
        threshold: f32,
        w_incr: f32,
        dt: f32,
        device: &Device,
    ) -> Self {
        Self {
            params: FhnParams::new(a, b, v_reset, threshold, w_incr, device),
            dt: dt.max(0.5),
        }
    }

    fn step_impl<S: Surrogate + ?Sized>(
        &self,
        x: Tensor<2>,
        state: &FhnState,
        surrogate: &S,
    ) -> (Tensor<2>, FhnState) {
        let dims = state.v.dims();
        let a = self.params.a.val().expand(dims);
        let b = self.params.b.val().expand(dims);
        let v3 = state.v.clone().mul(state.v.clone()).mul(state.v.clone());
        let v_new = state.v.clone()
            + (state.v.clone() - a.mul(v3.div_scalar(3.0)) - state.w.clone() + x)
                .mul_scalar(self.dt);
        let (spike, spike_hard) =
            spike_with_scalar_threshold(v_new.clone(), self.params.threshold, surrogate);
        let not_spike = spike_hard.clone().mul_scalar(-1.0).add_scalar(1.0);
        let v = v_new.clone() * not_spike.clone() + spike_hard.clone() * self.params.v_reset;
        let w_new = state.w.clone()
            + (v_new * not_spike + spike_hard.clone() * self.params.v_reset + 0.7
                - b.mul(state.w.clone() * 0.2))
            .mul_scalar(self.dt);
        let w = w_new + spike_hard * self.params.w_incr;
        (spike, FhnState { v, w })
    }
}

impl SpikingLayer for Fhn {
    type Input = Tensor<2>;
    type Output = Tensor<2>;
    type State = Deferred<FhnState>;

    fn init_state(&self, batch: usize, _device: &Device) -> Self::State {
        Deferred::pending(batch)
    }

    fn step<S: Surrogate + ?Sized>(
        &self,
        x: Tensor<2>,
        state: Self::State,
        surrogate: &S,
    ) -> (Tensor<2>, Self::State) {
        let [_batch, units] = x.dims();
        let device = x.device();
        let state = state.resolve(|b| FhnState {
            v: Tensor::zeros([b, units], &device),
            w: Tensor::zeros([b, units], &device),
        });
        let (y, next) = self.step_impl(x, &state, surrogate);
        (y, Deferred::Ready(next))
    }
}

// ---------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------

fn dev() -> Device {
    Device::flex().autodiff()
}

fn vals<const D: usize>(t: &Tensor<D>) -> Vec<f32> {
    t.clone().into_data().as_slice::<f32>().unwrap().to_vec()
}

fn spike_vec(y: &Tensor<2>) -> Vec<f32> {
    vals(y)
}

fn inputs(dev: &Device) -> Vec<Tensor<2>> {
    (0..6)
        .map(|i| {
            let mut data = vec![0.0f32; 3];
            data[(i % 3) as usize] = 1.5;
            Tensor::<1>::from_floats(data.as_slice(), dev).reshape([1, 3])
        })
        .collect()
}

fn assert_close(a: &[f32], b: &[f32], what: &str) {
    assert_eq!(a.len(), b.len(), "{what}: lengths vary");
    for (i, (x, y)) in a.iter().zip(b.iter()).enumerate() {
        assert!((x - y).abs() < 1e-5, "{what}[{i}]: {x} vs {y}");
    }
}

// ---------------------------------------------------------------------
// Basic correctness
// ---------------------------------------------------------------------

#[test]
fn fhn_two_state_variables_and_custom_spike() {
    let d = dev();
    let layer = Fhn::new(0.05, 0.25, -1.2, 0.5, 0.2, 1.0, &d);
    let mut state = layer.init_state(1, &d);
    let s = FastSigmoid::default();
    let mut any_spike = false;
    for x in inputs(&d).into_iter().take(6) {
        let (y, next) = layer.step(x, state, &s);
        state = next;
        let v = state.get().expect("ready after first step").v.clone();
        let w = state.get().expect("ready after first step").w.clone();
        assert_eq!(v.dims(), [1, 3]);
        assert_eq!(w.dims(), [1, 3]);
        any_spike |= spike_vec(&y).iter().any(|s| *s > 0.5);
    }
    assert!(
        any_spike,
        "the model should fire with dt = 1.0 on the demo input"
    );
}

#[test]
fn fhn_forward_seq_matches_manual_loop() {
    let d = dev();
    let layer = Fhn::new(0.05, 0.25, -1.2, 0.5, 0.2, 1.0, &d);
    let s = FastSigmoid::default();

    let mut state = layer.init_state(1, &d);
    let mut manual = Vec::new();
    for x in inputs(&d).into_iter().take(5) {
        let (y, next) = layer.step(x, state, &s);
        state = next;
        manual.push(spike_vec(&y));
    }
    let seq = layer.forward_seq(inputs(&d).into_iter().take(5).collect(), &s);
    for (i, y) in seq.iter().enumerate() {
        assert_close(&manual[i], &spike_vec(y), "manual vs forward_seq");
    }
}

#[test]
fn fhn_forward_seq_tensor_and_batch_independence() {
    let d = dev();
    let layer = Fhn::new(0.05, 0.25, -1.2, 0.5, 0.2, 1.0, &d);
    let s = FastSigmoid::default();

    // Batch independence: the first row of the batch must behave exactly
    // like a standalone single-row network fed the same input.
    let row0 = Tensor::<2>::from_floats([[1.5, 0.0, 0.0]], &d);
    let row1 = Tensor::<2>::from_floats([[0.0, 1.5, 0.0]], &d);
    let x2 = Tensor::cat(vec![row0.clone(), row1.clone()], 0);
    let (y, st2) = layer.step(x2, layer.init_state(2, &d), &s);
    assert_eq!(y.dims(), [2, 3]);
    let (_, st1) = layer.step(row0, layer.init_state(1, &d), &s);
    assert_close(
        &vals(&st2.get().unwrap().v)[..3],
        &vals(&st1.get().unwrap().v),
        "row 0 of the batch == single-row run",
    );

    // [batch, time, features] tensor variant.
    let xs: Vec<Tensor<2>> = inputs(&d).into_iter().take(4).collect();
    let x = Tensor::stack(xs, 1);
    let out = layer.forward_seq_tensor(x, &s);
    assert_eq!(out.dims(), [1, 4, 3]);
}

// ---------------------------------------------------------------------
// Session
// ---------------------------------------------------------------------

#[test]
fn fhn_session_reset_detach() {
    let d = dev();
    let s = FastSigmoid::default();
    let sess = Session::new(Fhn::new(0.05, 0.25, -1.2, 0.5, 0.2, 1.0, &d), 1, &d);
    for x in inputs(&d).into_iter().take(4) {
        sess.step(x, &s);
    }
    sess.detach_hidden();
    sess.reset(1, &d);
    let y = sess.step(inputs(&d)[0].clone(), &s);
    assert_eq!(y.dims(), [1, 3]);
    let _ = sess.state();
}

// ---------------------------------------------------------------------
// Autodiff
// ---------------------------------------------------------------------

#[test]
fn fhn_autodiff_grads_finite_through_surrogate() {
    let d = dev();
    let model = Fhn::new(0.05, 0.25, -1.2, 0.5, 0.2, 1.0, &d);
    let s = FastSigmoid::default();
    let xs: Vec<Tensor<2>> = (0..4)
        .map(|i| {
            let mut data = vec![0.0f32; 3];
            data[(i % 3) as usize] = 1.5;
            Tensor::<1>::from_floats(data.as_slice(), &d).reshape([1, 3])
        })
        .collect();
    let x = Tensor::stack(xs, 1).clone().require_grad();
    let out = model.forward_seq_tensor(x.clone(), &s);
    let loss = out.sum();
    let grads = loss.backward();
    let gx = x.grad(&grads).expect("input grad exists");
    assert!(
        gx.clone()
            .into_data()
            .as_slice::<f32>()
            .unwrap()
            .iter()
            .all(|g| g.is_finite())
    );
    // Optimizer path (trainable params).
    let mut optim = AdamConfig::new().init();
    let grads = GradientsParams::from_grads(grads, &model);
    let _ = optim.step(0.01, model, grads);
}

// ---------------------------------------------------------------------
// TBPTT: detach between chunks
// ---------------------------------------------------------------------

#[test]
fn fhn_chunked_bptt_detach_matches_single_pass() {
    let d = dev();
    let layer = Fhn::new(0.05, 0.25, -1.2, 0.5, 0.2, 1.0, &d);
    let s = FastSigmoid::default();
    let whole = layer.forward_seq(inputs(&d), &s);

    let mut st = layer.init_state(1, &d);
    let mut chunked = Vec::new();
    for chunk in inputs(&d).chunks(2) {
        let (ys, next) = layer.forward_seq_from(chunk.to_vec(), st, &s);
        chunked.extend(ys);
        st = next.detach();
    }
    for (a, b) in whole.iter().zip(&chunked) {
        assert_close(&spike_vec(a), &spike_vec(b), "chunked detach");
    }
}
