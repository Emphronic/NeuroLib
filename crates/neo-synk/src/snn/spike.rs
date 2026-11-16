//! Fused operation "spike with surrogate gradient" for Flex (CPU).
//!
//! The usual straight-through bundle `hard + (soft − soft.detach())` is a dozen small
//! tensor operations and the same number of nodes in the autodiff graph: comparison, reduction to float,
//! surrogate (3–4 operations), subtraction, `detach`, addition. There is one operation instead
//! over `x = mem − θ`:
//!
//! * forward: `1.0` where `x > 0`, otherwise `0.0` (exactly like `(x > 0)` in snnTorch);
//! * backward: `grad · σ'(x)`, where `σ'` is the derivative of the surrogate in closed shape
//!   ([`SurrogateKind`]).
//!
//! The operation is connected via `#[backend_extension]` and runs on Flex, including under
//! `Autodiff`. On other devices, [`spike_fused_supported`] returns `false`, and the caller
//! the code remains on the usual path (`neo-snn` does it itself).

use burn::backend::{
    Backend, Dispatch, DispatchDevice, Flex, TensorMetadata,
    autodiff::{
        Autodiff, NodeId,
        checkpoint::{base::Checkpointer, strategy::CheckpointStrategy},
        grads::Gradients,
        ops::{Backward, Ops, OpsKind},
    },
    backend_extension, get_or_init_device_settings,
    tensor::FloatTensor,
};
use burn::tensor::{Device, Tensor};

/// A surrogate with a closed derivative formula. Parameter values are the same as those of the same name
/// surrogates `neo-snn` (`ATan { alpha }`, `FastSigmoid { slope }`, ...).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum SurrogateKind {
    /// `σ = 0.5 + atan(π·α·x/2)/π`, `σ' = (α/2) / (1 + (π·α·x/2)²)`.
    ATan { alpha: f32 },
    /// `σ = 0.5 + 0.5·x/(1 + k|x|)`, `σ' = 0.5 / (1 + k|x|)²`.
    FastSigmoid { slope: f32 },
    /// `σ = sigmoid(α·x)`, `σ' = α·s·(1 − s)`.
    Sigmoid { alpha: f32 },
    /// `σ' = max(0, 1 − k|x|)`.
    Triangular { slope: f32 },
    /// `σ' = 1`.
    Straight,
}

impl SurrogateKind {
    fn encode(self) -> (u32, f32) {
        match self {
            Self::ATan { alpha } => (0, alpha),
            Self::FastSigmoid { slope } => (1, slope),
            Self::Sigmoid { alpha } => (2, alpha),
            Self::Triangular { slope } => (3, slope),
            Self::Straight => (4, 0.0),
        }
    }
}

/// Derivation of the `σ'(x)` surrogate using backend primitives.
fn derivative<B: Backend>(x: FloatTensor<B>, kind: u32, p: f32) -> FloatTensor<B> {
    match kind {
        // (α/2) / (1 + (k·x)²), k = π·α/2
        0 => {
            let k = std::f32::consts::PI * p / 2.0;
            let kx = B::float_mul_scalar(x, k.into());
            let den = B::float_add_scalar(B::float_mul(kx.clone(), kx), 1.0f32.into());
            B::float_mul_scalar(B::float_recip(den), (p / 2.0).into())
        }
        // 0.5 / (1 + k|x|)²
        1 => {
            let den = B::float_add_scalar(
                B::float_mul_scalar(B::float_abs(x), p.into()),
                1.0f32.into(),
            );
            let inv = B::float_recip(den);
            B::float_mul_scalar(B::float_mul(inv.clone(), inv), 0.5f32.into())
        }
        // α·s·(1 − s), s = 1 / (1 + exp(−α x))
        2 => {
            let e = B::float_exp(B::float_mul_scalar(x, (-p).into()));
            let s = B::float_recip(B::float_add_scalar(e, 1.0f32.into()));
            let one_minus = B::float_add_scalar(B::float_neg(s.clone()), 1.0f32.into());
            B::float_mul_scalar(B::float_mul(s, one_minus), p.into())
        }
        // max(0, 1 − k|x|)
        3 => {
            let v = B::float_add_scalar(
                B::float_neg(B::float_mul_scalar(B::float_abs(x), p.into())),
                1.0f32.into(),
            );
            B::float_clamp_min(v, 0.0f32.into())
        }
        // 1
        _ => B::float_add_scalar(B::float_mul_scalar(x, 0.0f32.into()), 1.0f32.into()),
    }
}

/// Forward: `(x > 0) as float`.
fn heaviside<B: Backend>(x: FloatTensor<B>) -> FloatTensor<B> {
    let device = TensorMetadata::device(&x);
    let bool_dtype = get_or_init_device_settings::<B>(&device).bool_dtype;
    let dtype = TensorMetadata::dtype(&x);
    let mask = B::float_greater_elem(x, 0.0f32.into(), bool_dtype);
    B::bool_into_float(mask, dtype.into())
}

/// Backend extension: spike `x > 0` with surrogate backward.
#[backend_extension(Flex, Autodiff)]
pub trait SpikeBackend: Backend {
    /// `kind`/`param` - encoded [`SurrogateKind`].
    fn spike_surrogate(x: FloatTensor<Self>, kind: u32, param: f32) -> FloatTensor<Self>;
}

impl SpikeBackend for Flex {
    fn spike_surrogate(x: FloatTensor<Self>, _kind: u32, _param: f32) -> FloatTensor<Self> {
        heaviside::<Flex>(x)
    }
}

impl<B: SpikeBackend, C: CheckpointStrategy> SpikeBackend for Autodiff<B, C> {
    fn spike_surrogate(x: FloatTensor<Self>, kind: u32, param: f32) -> FloatTensor<Self> {
        #[derive(Debug)]
        struct SpikeBackward;

        impl<B: SpikeBackend> Backward<B, 1> for SpikeBackward {
            type State = (NodeId, u32, f32);

            fn backward(
                self,
                ops: Ops<Self::State, 1>,
                grads: &mut Gradients,
                checkpointer: &mut Checkpointer,
            ) {
                let [node_x] = ops.parents;
                let grad = grads.consume::<B>(&ops.node);
                let (x_state, kind, param) = ops.state;
                if let Some(node) = node_x {
                    let x: FloatTensor<B> = checkpointer.retrieve_node_output(x_state);
                    let d = derivative::<B>(x, kind, param);
                    grads.register::<B>(node.id, B::float_mul(grad, d));
                }
            }
        }

        match SpikeBackward
            .prepare::<C>([x.node()])
            .compute_bound()
            .stateful()
        {
            OpsKind::Tracked(mut prep) => {
                let x_state = prep.checkpoint(&x);
                let out = B::spike_surrogate(x.primitive().clone(), kind, param);
                prep.finish((x_state, kind, param), out)
            }
            OpsKind::UnTracked(prep) => {
                prep.finish(B::spike_surrogate(x.into_primitive(), kind, param))
            }
        }
    }
}

/// `true`, if there is a fused operation for the tensor device (currently Flex, with or without autodiff).
pub fn spike_fused_supported(device: &Device) -> bool {
    matches!(
        device.clone().inner().as_dispatch(),
        DispatchDevice::Flex(_)
    )
}

/// Spike `x > 0` with a surrogate gradient of a given type. Forward - exactly Heaviside.
///
/// Check [`spike_fused_supported`] before calling: the operation is not defined on other devices.
pub fn spike_surrogate(x: Tensor<2>, kind: SurrogateKind) -> Tensor<2> {
    let (id, p) = kind.encode();
    Tensor::from_dispatch(Dispatch::spike_surrogate(x.into_dispatch(), id, p))
}
