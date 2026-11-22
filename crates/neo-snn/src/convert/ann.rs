//! Description of a regular (non-spike) network for conversion: [`AnnNet`].
//!
//! Burn modules are statically typed, so "bypass arbitrary
//! `Module`" is not allowed. Instead, the network is described by a sequence of layers
//! ([`AnnLayer`]), which is convenient to assemble directly from the trained Burn modules:
//!
//! ```ignore
//! let ann = AnnNet::new()
//!     .conv2d(&model.conv1).batch_norm(&model.bn1).relu()
//!     .avg_pool2d([2, 2])
//!     .flatten()
//!     .linear(&model.fc1).relu()
//!     .linear(&model.fc2);
//! ```
//!
//! Weights are not copied “deeply” - tensors Burn are cloned cheaply.

use burn::module::Param;
use burn::nn::conv::Conv2d;
use burn::nn::{BatchNorm, Linear};
use burn::tensor::Tensor;
use burn::tensor::module::avg_pool2d;
use burn::tensor::ops::AvgPoolOptions;

use crate::error::{Result, SnnError, tensor_to_vec};

/// Network activation: tensor rank 2 (`[batch, features]`) or 4
/// (`[batch, channels, h, w]`). We need the same code to work with
/// vector layers, and with convolutional maps.
#[derive(Debug, Clone)]
pub enum Act {
    /// `[batch, features]`.
    D2(Tensor<2>),
    /// `[batch, channels, h, w]`.
    D4(Tensor<4>),
}

impl From<Tensor<2>> for Act {
    fn from(t: Tensor<2>) -> Self {
        Act::D2(t)
    }
}

impl From<Tensor<4>> for Act {
    fn from(t: Tensor<4>) -> Self {
        Act::D4(t)
    }
}

impl Act {
    /// Tensor shape.
    pub fn dims(&self) -> Vec<usize> {
        match self {
            Act::D2(t) => t.dims().to_vec(),
            Act::D4(t) => t.dims().to_vec(),
        }
    }

    /// Batch size (zero axis).
    pub fn batch(&self) -> usize {
        match self {
            Act::D2(t) => t.dims()[0],
            Act::D4(t) => t.dims()[0],
        }
    }

    /// Like `Tensor<2>`; shape error if rank is different.
    pub fn into_d2(self, op: &'static str) -> Result<Tensor<2>> {
        match self {
            Act::D2(t) => Ok(t),
            Act::D4(t) => Err(SnnError::shape_msg(op, "[batch, features]", t.dims())),
        }
    }

    /// Like `Tensor<4>`; shape error if rank is different.
    pub fn into_d4(self, op: &'static str) -> Result<Tensor<4>> {
        match self {
            Act::D4(t) => Ok(t),
            Act::D2(t) => Err(SnnError::shape_msg(op, "[batch, channels, h, w]", t.dims())),
        }
    }

    /// All values as `Vec<f32>` (in tensor memory order).
    pub fn to_vec(&self, op: &'static str) -> Result<Vec<f32>> {
        match self {
            Act::D2(t) => tensor_to_vec(t.clone(), op),
            Act::D4(t) => tensor_to_vec(t.clone(), op),
        }
    }

    /// Multiplication by a scalar.
    pub fn scale(self, k: f32) -> Act {
        match self {
            Act::D2(t) => Act::D2(t.mul_scalar(k)),
            Act::D4(t) => Act::D4(t.mul_scalar(k)),
        }
    }

    /// Element-by-element sum; the shapes must match.
    pub(crate) fn add(self, other: Act, op: &'static str) -> Result<Act> {
        match (self, other) {
            (Act::D2(a), Act::D2(b)) => Ok(Act::D2(a + b)),
            (Act::D4(a), Act::D4(b)) => Ok(Act::D4(a + b)),
            (a, b) => Err(SnnError::shape_msg(op, format!("{:?}", a.dims()), b.dims())),
        }
    }

    fn relu(self) -> Act {
        match self {
            Act::D2(t) => Act::D2(burn::tensor::activation::relu(t)),
            Act::D4(t) => Act::D4(burn::tensor::activation::relu(t)),
        }
    }

    pub(crate) fn flatten(self) -> Act {
        match self {
            Act::D4(t) => Act::D2(t.flatten::<2>(1, 3)),
            d2 => d2,
        }
    }
}

/// BatchNorm parameters taken from the trained module (output mode).
#[derive(Debug, Clone)]
pub struct BatchNormParams {
    gamma: Tensor<1>,
    beta: Tensor<1>,
    mean: Tensor<1>,
    var: Tensor<1>,
    eps: f64,
}

impl BatchNormParams {
    /// Remove parameters and running statistics from `BatchNorm` Burn.
    pub fn from_module(bn: &BatchNorm) -> Self {
        Self {
            gamma: bn.gamma.val(),
            beta: bn.beta.val(),
            mean: bn.running_mean.value(),
            var: bn.running_var.value(),
            eps: bn.epsilon,
        }
    }

    /// Assemble the shapes `[channels]` from ready-made tensors.
    pub fn new(
        gamma: Tensor<1>,
        beta: Tensor<1>,
        mean: Tensor<1>,
        var: Tensor<1>,
        eps: f64,
    ) -> Self {
        Self {
            gamma,
            beta,
            mean,
            var,
            eps,
        }
    }

    /// BN in output mode is `x * scale + shift` by channel.
    fn scale_shift(&self) -> (Tensor<1>, Tensor<1>) {
        let scale = self.gamma.clone() / (self.var.clone().add_scalar(self.eps)).sqrt();
        let shift = self.beta.clone() - self.mean.clone() * scale.clone();
        (scale, shift)
    }

    fn channels(&self) -> usize {
        self.gamma.dims()[0]
    }

    fn forward(&self, x: Act) -> Result<Act> {
        let (s, t) = self.scale_shift();
        let c = self.channels();
        match x {
            Act::D2(x) => {
                let [_, f] = x.dims();
                if f != c {
                    return Err(SnnError::Channels {
                        op: "BatchNorm",
                        expected: c,
                        actual: f,
                    });
                }
                Ok(Act::D2(x * s.reshape([1, c]) + t.reshape([1, c])))
            }
            Act::D4(x) => {
                let [_, ch, _, _] = x.dims();
                if ch != c {
                    return Err(SnnError::Channels {
                        op: "BatchNorm",
                        expected: c,
                        actual: ch,
                    });
                }
                Ok(Act::D4(
                    x * s.reshape([1, c, 1, 1]) + t.reshape([1, c, 1, 1]),
                ))
            }
        }
    }
}

#[allow(clippy::large_enum_variant)]
/// Source network layer.
#[derive(Debug, Clone)]
pub enum AnnLayer {
    /// Fully connected layer (`[batch, in] → [batch, out]`).
    Linear(Linear),
    /// Two-dimensional convolution (`[batch, c, h, w]`), including with `groups`.
    Conv2d(Conv2d),
    /// BatchNorm (calculated using running statistics; when converting
    /// collapses into the previous layer with weights).
    BatchNorm(BatchNormParams),
    /// Averaging pooling: linear operation, so applies to spikes
    /// as is. `MaxPool` is intentionally not supported - see [`AnnNet`].
    AvgPool2d {
        /// Window size.
        kernel: [usize; 2],
        /// Step.
        stride: [usize; 2],
    },
    /// `[batch, c, h, w] → [batch, c*h*w]` (for rank 2 - identity).
    Flatten,
    /// ReLU: it is this that is replaced by the spike neuron.
    Relu,
    /// Dropout and other things that do nothing in output mode.
    Identity,
}

/// Serial regular network, ready to be converted to SNN.
///
/// # What can be converted
///
/// Sequence of `Linear | Conv2d` blocks, followed by (optional)
/// goes `BatchNorm` and then `ReLU`; `AvgPool2d` are allowed between blocks,
/// `Flatten`, `Identity`. The last layer with weights can be without `ReLU` -
/// it will become a non-spike readout (logits).
///
/// Not supported because there is no exact spike equivalent:
/// `MaxPool` (replace with `AvgPool2d`), any activations except ReLU, `ReLU`
/// not immediately after the layer with weights, the layer with weights without `ReLU` in the middle of the network,
/// residual connections and other non-sequential graphs.
#[derive(Debug, Clone, Default)]
pub struct AnnNet {
    layers: Vec<AnnLayer>,
}

impl AnnNet {
    /// Empty network.
    pub fn new() -> Self {
        Self::default()
    }

    /// Add a custom layer.
    pub fn push(mut self, layer: AnnLayer) -> Self {
        self.layers.push(layer);
        self
    }

    /// Add `Linear` (weights are cloned).
    pub fn linear(self, layer: &Linear) -> Self {
        self.push(AnnLayer::Linear(layer.clone()))
    }

    /// Add `Conv2d` (weights are cloned).
    pub fn conv2d(self, layer: &Conv2d) -> Self {
        self.push(AnnLayer::Conv2d(layer.clone()))
    }

    /// Add `BatchNorm` (running statistics are taken).
    pub fn batch_norm(self, layer: &BatchNorm) -> Self {
        self.push(AnnLayer::BatchNorm(BatchNormParams::from_module(layer)))
    }

    /// Averaging pooling with a step equal to the window.
    pub fn avg_pool2d(self, kernel: [usize; 2]) -> Self {
        self.push(AnnLayer::AvgPool2d {
            kernel,
            stride: kernel,
        })
    }

    /// Averaging pooling with explicit step.
    pub fn avg_pool2d_with_stride(self, kernel: [usize; 2], stride: [usize; 2]) -> Self {
        self.push(AnnLayer::AvgPool2d { kernel, stride })
    }

    /// `Flatten` after convolutions.
    pub fn flatten(self) -> Self {
        self.push(AnnLayer::Flatten)
    }

    /// ReLU.
    pub fn relu(self) -> Self {
        self.push(AnnLayer::Relu)
    }

    /// Dropout (identity in output).
    pub fn dropout(self) -> Self {
        self.push(AnnLayer::Identity)
    }

    /// Network layers.
    pub fn layers(&self) -> &[AnnLayer] {
        &self.layers
    }

    /// Direct pass of the original ANN (in output mode).
    pub fn forward(&self, x: Act) -> Result<Act> {
        self.forward_observe(x, &mut |_, _| {})
    }

    /// Straight pass; after each `ReLU` calls `observe(index_relu, output)`.
    pub(crate) fn forward_observe(
        &self,
        mut x: Act,
        observe: &mut dyn FnMut(usize, &Act),
    ) -> Result<Act> {
        let mut relu_idx = 0;
        for layer in &self.layers {
            x = match layer {
                AnnLayer::Linear(l) => Act::D2(l.forward(x.into_d2("Linear")?)),
                AnnLayer::Conv2d(c) => Act::D4(c.forward(x.into_d4("Conv2d")?)),
                AnnLayer::BatchNorm(bn) => bn.forward(x)?,
                AnnLayer::AvgPool2d { kernel, stride } => {
                    Act::D4(pool(x.into_d4("AvgPool2d")?, *kernel, *stride))
                }
                AnnLayer::Flatten => x.flatten(),
                AnnLayer::Relu => {
                    let y = x.relu();
                    observe(relu_idx, &y);
                    relu_idx += 1;
                    y
                }
                AnnLayer::Identity => x,
            };
        }
        Ok(x)
    }

    /// Canonical shape for conversion: BatchNorm collapsed to previous
    /// layer, `Identity` removed, structure checked (see rules in description
    /// [`AnnNet`]).
    pub fn canonicalize(&self) -> Result<AnnNet> {
        let mut out: Vec<AnnLayer> = Vec::with_capacity(self.layers.len());
        for layer in &self.layers {
            match layer {
                AnnLayer::Identity => {}
                AnnLayer::BatchNorm(bn) => match out.last_mut() {
                    Some(AnnLayer::Linear(l)) => fold_into_linear(l, bn)?,
                    Some(AnnLayer::Conv2d(c)) => fold_into_conv(c, bn)?,
                    _ => {
                        return Err(invalid(
                            "BatchNorm",
                            "BatchNorm should come right after Linear or Conv2d (its weights are collapsed into this layer)",
                        ));
                    }
                },
                AnnLayer::Relu => {
                    if !matches!(out.last(), Some(AnnLayer::Linear(_) | AnnLayer::Conv2d(_))) {
                        return Err(invalid(
                            "ReLU",
                            "ReLU should go immediately after the layer with weights (Linear/Conv2d/BatchNorm): pooling or flatten between them breaks the equivalence of the spike neuron",
                        ));
                    }
                    out.push(AnnLayer::Relu);
                }
                other => out.push(other.clone()),
            }
        }
        // A layer with weights without ReLU is only allowed as the last one.
        let weighted: Vec<usize> = out
            .iter()
            .enumerate()
            .filter(|(_, l)| matches!(l, AnnLayer::Linear(_) | AnnLayer::Conv2d(_)))
            .map(|(i, _)| i)
            .collect();
        if weighted.is_empty() {
            return Err(invalid(
                "AnnNet",
                "there are no layers with weights in the network",
            ));
        }
        for (k, &i) in weighted.iter().enumerate() {
            let has_relu = matches!(out.get(i + 1), Some(AnnLayer::Relu));
            if !has_relu && k + 1 != weighted.len() {
                return Err(invalid(
                    "AnnNet",
                    "a layer with weights without ReLU is allowed only as the last one (readout); merge consecutive linear layers first",
                ));
            }
        }
        Ok(AnnNet { layers: out })
    }
}

fn invalid(what: &'static str, reason: &str) -> SnnError {
    SnnError::InvalidConfig {
        what,
        reason: reason.to_string(),
    }
}

pub(crate) fn pool(x: Tensor<4>, kernel: [usize; 2], stride: [usize; 2]) -> Tensor<4> {
    avg_pool2d(
        x,
        AvgPoolOptions {
            kernel_size: kernel,
            stride,
            padding: [(0, 0); 2],
            count_include_pad: true,
            ceil_mode: false,
        },
    )
}

fn fold_into_linear(l: &mut Linear, bn: &BatchNormParams) -> Result<()> {
    let [_, out] = l.weight.dims();
    if bn.channels() != out {
        return Err(SnnError::Channels {
            op: "BatchNorm→Linear",
            expected: out,
            actual: bn.channels(),
        });
    }
    let (s, t) = bn.scale_shift();
    let w = l.weight.val() * s.clone().reshape([1, out]);
    let b = match &l.bias {
        Some(b) => b.val() * s + t,
        None => t,
    };
    l.weight = Param::from_tensor(w);
    l.bias = Some(Param::from_tensor(b));
    Ok(())
}

fn fold_into_conv(c: &mut Conv2d, bn: &BatchNormParams) -> Result<()> {
    let [out, ..] = c.weight.dims();
    if bn.channels() != out {
        return Err(SnnError::Channels {
            op: "BatchNorm→Conv2d",
            expected: out,
            actual: bn.channels(),
        });
    }
    let (s, t) = bn.scale_shift();
    let w = c.weight.val() * s.clone().reshape([out, 1, 1, 1]);
    let b = match &c.bias {
        Some(b) => b.val() * s + t,
        None => t,
    };
    c.weight = Param::from_tensor(w);
    c.bias = Some(Param::from_tensor(b));
    Ok(())
}
