//! FitzHugh–Nagumo — a user-defined tensor neuron on Burn.
//!
//! Shows that the library provides a universal extension point:
//! just implement `SpikingLayer`. The dynamics, state,
//! reset, spike, surrogate and training are all defined by user code.
//!
//! Run: `cargo run --example custom_tensor_neuron -p neo-snn`
//!
//! Covered:
//! 1. model/state structs with arbitrary state variables;
//! 2. `Detach` for the state;
//! 3. a `SpikingLayer` implementation with a custom reset;
//! 4. custom dynamics equations;
//! 5. spiking via `spike_with_scalar_threshold` + FastSigmoid;
//! 6. `Session` (init_hidden), `forward_seq`, `forward_seq_tensor`;
//! 7. truncated BPTT: detaching the state between chunks;
//! 8. training: autodiff + Adam + trainable model parameters.

use burn::module::{Module, Param};
use burn::optim::{AdamConfig, GradientsParams};
use burn::tensor::{Device, Tensor};
use neo_snn::neuron::spike_with_scalar_threshold;
use neo_snn::spiking_layer::{Deferred, Session, SpikingLayer, SpikingLayerExt};
use neo_snn::stateful::Detach;
use neo_snn::surrogate::{FastSigmoid, Surrogate};

// ---------------------------------------------------------------------
// Model: two state variables v, w (FitzHugh–Nagumo)
// ---------------------------------------------------------------------

/// Trainable parameters of the FHN model. `a` and `b` are Burn
/// `Module` params (`Param<Tensor<1>>`): they can be optimized with
/// Burn autodiff (see the training demo at the end of `main`).
#[derive(Module, Debug)]
pub struct FhnParams {
    /// Coefficient in `v' = v + dt·(v − v³/3 − w + I)`.
    pub a: Param<Tensor<1>>,
    /// Coefficient in `w' = w + dt·b·(v + 0.7 − 0.2·w)`.
    pub b: Param<Tensor<1>>,
    /// Reset value of `v` after a spike.
    pub v_reset: f32,
    /// Spike threshold on `v`.
    pub threshold: f32,
    /// Recovery increment on `w` after a spike.
    pub w_incr: f32,
}

impl FhnParams {
    pub fn new(a: f32, b: f32, v_reset: f32, threshold: f32, w_incr: f32, device: &Device) -> Self {
        Self {
            a: Param::from_tensor(Tensor::from_floats([a], device)),
            b: Param::from_tensor(Tensor::from_floats([b], device)),
            v_reset,
            threshold,
            w_incr,
        }
    }
}

/// State of the neuron: `v` and `w` — 2 state variables, user-defined
/// shape (`[batch, units]`).
#[derive(Debug, Clone)]
pub struct FhnState {
    pub v: Tensor<2>,
    pub w: Tensor<2>,
}

/// State transfers between steps (truncated BPTT needs `Detach`).
impl Detach for FhnState {
    fn detach(self) -> Self {
        Self {
            v: self.v.detach(),
            w: self.w.detach(),
        }
    }
}

/// FitzHugh–Nagumo model.
#[derive(Module, Debug)]
pub struct Fhn {
    /// Model parameters (trainable).
    pub params: FhnParams,
    /// Time step of the equations.
    pub dt: f32,
}

impl Fhn {
    pub fn new(
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

    /// One dynamics step:
    ///
    /// ```text
    /// v' = v + dt·(v − v³/3 − w + I)
    /// w' = w + dt·b·(v + 0.7 − 0.2·w)
    /// ```
    ///
    /// Spike: `v >= threshold` → return 1, `v ← v_reset`, `w += w_incr`.
    fn step_impl<S: Surrogate + ?Sized>(
        &self,
        x: Tensor<2>,
        state: &FhnState,
        surrogate: &S,
    ) -> (Tensor<2>, FhnState) {
        let dt = self.dt;
        let dims = state.v.dims();
        let a = self.params.a.val().expand(dims);
        let b = self.params.b.val().expand(dims);

        // v' = v + dt·(v − v³/3 − w + I)
        let v3 = state.v.clone().mul(state.v.clone()).mul(state.v.clone());
        let v_new = state.v.clone()
            + (state.v.clone() - a.mul(v3.div_scalar(3.0)) - state.w.clone() + x).mul_scalar(dt);

        // Spike detection with the user threshold + straight-through surrogate.
        let (spike, spike_hard) =
            spike_with_scalar_threshold(v_new.clone(), self.params.threshold, surrogate);

        // User reset: `v ← v_reset`, `w ← w + w_incr` (no library `apply_reset`).
        let not_spike = spike_hard.clone().mul_scalar(-1.0).add_scalar(1.0);
        let v = v_new.clone() * not_spike.clone() + spike_hard.clone() * self.params.v_reset;

        // w' = w + dt·b·(v + 0.7 − 0.2·w) — driven by the new v.
        let w_new = state.w.clone()
            + (v_new * not_spike + spike_hard.clone() * self.params.v_reset + 0.7
                - b.mul(state.w.clone() * 0.2))
            .mul_scalar(dt);
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

fn xs(dev: &Device) -> Vec<Tensor<2>> {
    (0..8)
        .map(|i| {
            let mut data = vec![0.0f32; 3];
            data[(i % 3) as usize] = 1.5;
            Tensor::<1>::from_floats(data.as_slice(), dev).reshape([1, 3])
        })
        .collect()
}

fn spikes_of(y: &Tensor<2>) -> Vec<f32> {
    y.clone().into_data().as_slice::<f32>().unwrap().to_vec()
}

fn main() {
    let dev = Device::flex().autodiff();
    let surrogate = FastSigmoid::default();

    println!("== 1. Explicit steps (two state variables: v, w) ==");
    let layer = Fhn::new(0.05, 0.25, -1.2, 0.5, 0.2, 1.0, &dev);
    let mut state = layer.init_state(1, &dev);
    for (i, x) in xs(&dev).into_iter().enumerate().take(4) {
        let (y, next) = layer.step(x, state, &surrogate);
        state = next;
        println!("  step {i}: spikes {:?}", spikes_of(&y));
    }

    println!("== 2. forward_seq (explicit state chain) ==");
    let inputs = xs(&dev);
    let ys = layer.forward_seq(inputs.clone(), &surrogate);
    println!(
        "  got {} outputs, e.g. spikes {:?}",
        ys.len(),
        spikes_of(&ys[0])
    );

    println!("== 3. forward_seq_tensor ([batch, time, units]) ==");
    let x_seq = Tensor::stack(inputs.clone(), 1);
    let y = layer.forward_seq_tensor(x_seq, &surrogate);
    println!("  output shape: {:?}", y.dims());

    println!("== 4. Session (init_hidden) + detach + reset ==");
    let sess = Session::new(Fhn::new(0.05, 0.25, -1.2, 0.5, 0.2, 0.02, &dev), 1, &dev);
    for x in inputs.clone() {
        let y = sess.step(x, &surrogate);
        println!("  session step: {:?}", spikes_of(&y));
    }
    sess.detach_hidden();
    sess.reset(1, &dev);
    let _ = sess.step(inputs[0].clone(), &surrogate);
    println!("  session reset ok");

    println!("== 5. Autodiff + training (Adam over Burn params) ==");
    let model = Fhn::new(0.05, 0.25, -1.2, 0.5, 0.2, 1.0, &dev);
    let mut optim = AdamConfig::new().init();
    let x_in = Tensor::stack(inputs.clone(), 1).clone().require_grad();
    let out = model.forward_seq_tensor(x_in, &surrogate);
    let loss = out.sum();
    println!(
        "  loss = {:.6}",
        loss.clone().into_data().as_slice::<f32>().unwrap()[0]
    );
    let grads = loss.backward();
    let grads = GradientsParams::from_grads(grads, &model);
    let _model = optim.step(0.05, model, grads);
    println!("  trained one step with Adam; trainable `a` updated");

    println!("== 6. truncated BPTT: chunked run with detach ==");
    let whole = layer.forward_seq(inputs.clone(), &surrogate);
    let mut st = layer.init_state(1, &dev);
    let mut chunked = Vec::new();
    for chunk in inputs.chunks(4) {
        let (ys, next) = layer.forward_seq_from(chunk.to_vec(), st, &surrogate);
        chunked.extend(ys);
        st = next.detach();
    }
    let same = whole
        .iter()
        .zip(&chunked)
        .all(|(a, b)| spikes_of(a) == spikes_of(b));
    println!("  chunked (detached) equals single pass: {same}");
}
