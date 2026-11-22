//! Layer between [`structural::fixed`](crate::structural::fixed) and Burn layers.
//!
//! The kernel in `structural::fixed` is independent of Burn. This module uses
//! Burn idioms (`Tensor::from_data(TensorData::new(..), &device)`, `Param::map`,
//! `param.val().grad(&grads)`, `tensor_to_vec`) to connect it to Burn layers
//! and targets Burn 0.22.
//!
//! # Saving: one file
//!
//! `save_bundle` writes masks and weights of managed layers (and any additional tensors, for example
//! `linear_rec`) into one file `NFXB`; `load_bundle` returns the weights already projected onto the masks.
//! Format and checks - `structural::fixed::Bundle` (Burn is an independent part, tested).
//!
//! # How to use
//!
//! ```ignore
//! let mut ctrl = FixedShapeController::new(cfg, specs)?;
//! // 1. Initialization: set the weights to the mask.
//! let mut proj = MaskProjector::new(&ctrl, &device);
//! model.fc = proj.project_linear(0, model.fc);
//! // 2. Learning cycle.
//! for batch in data {
//!     let loss = ...; let grads = loss.backward();
//!     observe_linear_grads(&mut ctrl, 0, &model.fc, model.fc.weight.val().grad(&grads))?; // |w·g|
//!     // observe_activity(...) - batch spike frequencies (see example)
//!     let gp = GradientsParams::from_grads(grads, &model);
//!     model = optim.step(lr, model, gp);
//!     model.fc = proj.project_linear(0, model.fc);                // w ← w ⊙ mask
//!     // between batches:
//!     let w = read_linear_weights(&[&model.fc])?;
//!     let rep = ctrl.adapt(AdaptInput { step, sample, loss: Some(l) }, &refs(&w))?;
//!     if !rep.updates.is_empty() {
//!         for u in &rep.updates { model.fc = apply_layer_update(model.fc, u)?; }
//!         proj.refresh(&ctrl);
//!         model.fc = proj.project_linear(0, model.fc);
//!     }
//! }
//! ```
//!
//! The parameter identifier (`ParamId`) is saved when writing (`Param::map`),
//! therefore the optimizer state is not recreated; for changed elements
//! it remains “old” - this is a known limitation.

use burn::nn::Linear;
use burn::tensor::{Device, Tensor, TensorData};
use thiserror::Error;

use crate::error::{SnnError, tensor_to_vec};
use crate::structural::StructuralError;
use crate::structural::fixed::{Bundle, FixedShapeController, LayerBlob, LayerUpdate, NamedTensor};

/// Layer error.
#[derive(Debug, Clone, PartialEq, Error)]
pub enum BridgeError {
    /// Error working with tensors `neo-snn`.
    #[error(transparent)]
    Snn(#[from] SnnError),
    /// Structural core error.
    #[error(transparent)]
    Structural(#[from] StructuralError),
}

/// The result of the layer.
pub type BridgeResult<T> = core::result::Result<T, BridgeError>;

/// Read the weights `Linear` as flat `[in, out]`.
pub fn read_linear_weights(layers: &[&Linear]) -> BridgeResult<Vec<Vec<f32>>> {
    layers
        .iter()
        .map(|l| {
            tensor_to_vec(l.weight.val(), "fixed_shape::read_linear_weights")
                .map_err(BridgeError::from)
        })
        .collect()
}

/// Take into account the gradient of weights `Linear`: the importance of the connection `|w·∂L/∂w|`.
///
/// `grad` - the result of `lin.weight.val().grad(&grads)` (the idiom is already used in tests
/// repository); `None` (parameter did not participate in the graph) - does nothing, returns `false`.
pub fn observe_linear_grads(
    ctrl: &mut FixedShapeController,
    layer: usize,
    lin: &Linear,
    grad: Option<Tensor<2>>,
) -> BridgeResult<bool> {
    let Some(g) = grad else {
        return Ok(false);
    };
    let wv = tensor_to_vec(lin.weight.val(), "fixed_shape::observe_linear_grads(w)")?;
    let gv = tensor_to_vec(g, "fixed_shape::observe_linear_grads(g)")?;
    ctrl.observe_grads(layer, &wv, &gv)?;
    Ok(true)
}

/// Write layer changes (weights from `LayerUpdate`) to `Linear`. `ParamId` is saved.
pub fn apply_layer_update(mut lin: Linear, upd: &LayerUpdate) -> BridgeResult<Linear> {
    let dims = lin.weight.val().dims();
    let mut w = tensor_to_vec(lin.weight.val(), "fixed_shape::apply_layer_update")?;
    upd.write_weights(&mut w)?;
    lin.weight = lin.weight.map(|old| {
        let device = old.device();
        Tensor::from_data(TensorData::new(w, dims), &device)
    });
    Ok(lin)
}

/// Cuts a projected parameter off the previous autodiff graph and makes it a fresh leaf.
///
/// Burn 0.22 consumes the tape on `backward` and has no `retain_graph`, so a parameter that kept
/// the `w ⊙ mask` node would drag the already consumed graph into the next batch.
fn leaf<const D: usize>(t: Tensor<D>) -> Tensor<D> {
    t.detach().require_grad()
}

/// On-device masks for fast projection without going through the CPU.
pub struct MaskProjector {
    weight: Vec<Tensor<2>>,
    neuron: Vec<Tensor<1>>,
    shapes: Vec<[usize; 2]>,
    device: Device,
}

impl MaskProjector {
    /// Construct masks from the current controller state.
    pub fn new(ctrl: &FixedShapeController, device: &Device) -> Self {
        let mut p = Self {
            weight: vec![],
            neuron: vec![],
            shapes: vec![],
            device: device.clone(),
        };
        p.refresh(ctrl);
        p
    }

    /// Rebuild masks after structural changes (call after `apply_layer_update`).
    pub fn refresh(&mut self, ctrl: &FixedShapeController) {
        self.weight.clear();
        self.neuron.clear();
        self.shapes.clear();
        for (l, s) in ctrl.layers().iter().enumerate() {
            let sp = s.spec();
            let shape = [sp.in_features, sp.out_features];
            self.weight.push(Tensor::from_data(
                TensorData::new(ctrl.effective_mask_f32(l), shape),
                &self.device,
            ));
            self.neuron.push(Tensor::from_data(
                TensorData::new(ctrl.neuron_mask_f32(l), [sp.out_features]),
                &self.device,
            ));
            self.shapes.push(shape);
        }
    }

    /// `w ← w ⊙ mask`, `bias ← bias ⊙ neuron_mask`. Call after every `optim.step`.
    pub fn project_linear(&self, layer: usize, mut lin: Linear) -> Linear {
        let m = self.weight[layer].clone();
        lin.weight = lin.weight.map(|w| leaf(w.mul(m)));
        if let Some(b) = lin.bias.take() {
            let nm = self.neuron[layer].clone();
            lin.bias = Some(b.map(|b| leaf(b.mul(nm))));
        }
        lin
    }

    /// Projection of a matrix whose **output neurons coincide with the neurons of the `mask_layer` layer**
    /// (for example, recurrent `linear_rec` layer `RLeaky`): columns and bias are multiplied by
    /// neuron mask `mask_layer`. The individual connections of such a matrix are not structurally controlled.
    pub fn project_tied(&self, mask_layer: usize, mut lin: Linear) -> Linear {
        let nm2: Tensor<2> = self.neuron[mask_layer]
            .clone()
            .reshape([1, self.shapes[mask_layer][1]]);
        lin.weight = lin.weight.map(|w| leaf(w.mul(nm2)));
        if let Some(b) = lin.bias.take() {
            let nm = self.neuron[mask_layer].clone();
            lin.bias = Some(b.map(|b| leaf(b.mul(nm))));
        }
        lin
    }

    /// Layer neuron mask `[out]` for spike multiplication (so that the inactive neuron is silent).
    pub fn neuron_mask(&self, layer: usize) -> Tensor<1> {
        self.neuron[layer].clone()
    }

    /// the shape of the layer weight matrix is `[in, out]`.
    pub fn shape(&self, layer: usize) -> [usize; 2] {
        self.shapes[layer]
    }
}

// ─── one file: masks + weights ────────────────────── ──────────────────────

/// Weights (and bias) `Linear` as flat data for the file.
pub fn linear_blob(lin: &Linear) -> BridgeResult<LayerBlob> {
    let weights = tensor_to_vec(lin.weight.val(), "fixed_shape::linear_blob(weight)")?;
    let bias = match &lin.bias {
        Some(b) => Some(tensor_to_vec(b.val(), "fixed_shape::linear_blob(bias)")?),
        None => None,
    };
    Ok(LayerBlob { weights, bias })
}

/// An arbitrary tensor (recurrent matrix, β, ...) as named data.
pub fn named_tensor<const D: usize>(name: &str, t: Tensor<D>) -> BridgeResult<NamedTensor> {
    let shape = t.dims().to_vec();
    let data = tensor_to_vec(t, "fixed_shape::named_tensor")?;
    Ok(NamedTensor {
        name: name.to_string(),
        shape,
        data,
    })
}

/// Back to a tensor of rank `D`; rank discrepancy is a shape error.
pub fn tensor_from_named<const D: usize>(
    nt: &NamedTensor,
    device: &Device,
) -> BridgeResult<Tensor<D>> {
    if nt.shape.len() != D {
        return Err(SnnError::shape_msg(
            "fixed_shape::tensor_from_named",
            format!("rank {D}"),
            &nt.shape,
        )
        .into());
    }
    let mut shape = [0usize; D];
    shape.copy_from_slice(&nt.shape);
    Ok(Tensor::from_data(
        TensorData::new(nt.data.clone(), shape),
        device,
    ))
}

/// Write the weights (and bias, if there is one in both the layer and the file) to `Linear`. `ParamId` is saved.
/// The dimensions are checked: someone else's shape is an error, the layer does not change.
pub fn apply_blob(mut lin: Linear, blob: &LayerBlob) -> BridgeResult<Linear> {
    let dims = lin.weight.val().dims();
    if blob.weights.len() != dims[0] * dims[1] {
        return Err(SnnError::shape("fixed_shape::apply_blob", dims, [blob.weights.len()]).into());
    }
    let w = blob.weights.clone();
    lin.weight = lin.weight.map(|old| {
        let device = old.device();
        Tensor::from_data(TensorData::new(w, dims), &device)
    });
    if let (Some(b), Some(data)) = (lin.bias.take(), blob.bias.as_ref()) {
        let n = b.val().dims()[0];
        if data.len() != n {
            return Err(SnnError::shape("fixed_shape::apply_blob(bias)", [n], [data.len()]).into());
        }
        let data = data.clone();
        lin.bias = Some(b.map(|old| {
            let device = old.device();
            Tensor::from_data(TensorData::new(data, [n]), &device)
        }));
    }
    Ok(lin)
}

/// Write masks and weights of controlled layers (in the order of controller layers) and additional
/// tensors in one file (`NFXB`, atomic).
pub fn save_bundle(
    ctrl: &FixedShapeController,
    layers: &[&Linear],
    tensors: Vec<NamedTensor>,
    extra: Vec<u8>,
    path: impl AsRef<std::path::Path>,
) -> BridgeResult<()> {
    let blobs = layers
        .iter()
        .map(|l| linear_blob(l))
        .collect::<BridgeResult<Vec<_>>>()?;
    Bundle::capture(ctrl, blobs, tensors, extra)?.save_to_file(path)?;
    Ok(())
}

/// Read file: weights are already projected onto the loaded masks.
pub struct LoadedBundle {
    /// The weights of the controller layers are in order (write using [`apply_blob`]).
    pub layers: Vec<LayerBlob>,
    /// Additional tensors.
    pub tensors: Vec<NamedTensor>,
    /// Calling code data.
    pub extra: Vec<u8>,
}

/// Load the file: masks into the controller, weights into the result. In case of any error, the controller does not change.
/// After loading: `apply_blob` for each layer, `MaskProjector::refresh(&ctrl)`, `project_linear`.
pub fn load_bundle(
    ctrl: &mut FixedShapeController,
    path: impl AsRef<std::path::Path>,
) -> BridgeResult<LoadedBundle> {
    let bundle = Bundle::load_from_file(path)?;
    let layers = bundle.restore(ctrl)?;
    Ok(LoadedBundle {
        layers,
        tensors: bundle.tensors,
        extra: bundle.extra,
    })
}
