//! `QuantizedLinear` is a linear layer of `y = x·Wᵀ + b` on packed weights.
//!
//! ```text
//! QuantizedLinear
//! ├── weight : Arc<PackedBuffer> packed W [out, in] (codes + scales + zero-points)
//! ├── bias : Option<Tensor<1>> (remains in float)
//! ├── shape   : in_features, out_features
//! └── policy  : ExecutionPolicy
//! ```
//!
//! # How is matmul computed?
//!
//! | [`ExecutionPolicy`] | What's going on |
//! |---------------------|------------------------------------------------------------------------|
//! | `Auto` (default) | `Fused`, if there is a fused core for packaging, otherwise `Dequantize` |
//! | `Fused` | `neo_synk::quant::packed_matmul`: **full W in f32 is not created** |
//! | `Dequantize` | W unfolds to `Tensor` and is computed as a normal Burn `matmul` (level 1) |
//!
//! `Fused` for `Int1..Int8`/NF4/Ternary works in "integer domain"
//! (scale and zero-point are applied once per group), for Float/FP16/BF16/FP8,
//! GGUF `Q*` and K-quants - blockwise (piece of a row ≤ 1024 weights at a time).
//! Details are in `neo_synk::quant::cpu`.
//!
//! # Restrictions
//!
//! * `Fused` computes on the **host**. On CPU backends (`Device::flex()`) it is free;
//!   on GPU backends, activations are read to the host and the result is downloaded
//!   back - there use `Dequantize` or the GPU core (see `neo-synk`).
//! * `Fused` does not participate in autodiff: the result is a “leaf” tensor, gradient
//!   does not flow to `x`. For training/QLoRA use `ExecutionPolicy::Dequantize`
//!   (the gradient along `x` will go through the usual `matmul`). The weights themselves are frozen.
//!
//! # Layout of scales
//!
//! `W` is stored as `[out, in]`, row-major, scaling groups go along
//! `in`. This is a **transposed** layout relative to `burn::nn::Linear`
//! (there `[in, out]`) - [`QuantizedLinear::from_linear`] transposes itself.
//!
//! # Example
//!
//! ```ignore
//! use burn::tensor::Device;
//! use burn::nn::LinearConfig;
//! use neo_nn::quantized::QuantizedLinear;
//! use neo_quantization::QuantizeConfig;
//!
//! let device = Device::flex(); // CPU backend (feature "flex")
//! let linear = LinearConfig::new(4096, 4096).init(&device);
//!
//! let q = QuantizedLinear::from_linear(&linear, &QuantizeConfig::int4_per_group_32());
//! println!("{:.2} bits/weight, compression ×{:.1}", q.bits_per_weight(), q.compression_ratio());
//! let y = q.forward(x);
//! ```

use std::sync::Arc;

use burn::nn::Linear;
use burn::prelude::*;
use burn::tensor::TensorData;
use neo_quantization::{PackedBuffer, QuantizeConfig, quantize};
use neo_synk::quant::{KernelKind, QuantGemmError, kernel_kind, packed_matmul};

/// How `QuantizedLinear` computes matmul.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum ExecutionPolicy {
    /// `Fused`, if there is a fused core for packaging; otherwise `Dequantize`.
    #[default]
    Auto,
    /// Fused quantized GEMM by packaging (host, no autodiff).
    Fused,
    /// Expand W into a tensor and consider the usual `matmul` of the backend.
    Dequantize,
}

/// Construction errors [`QuantizedLinear`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum QuantizedLinearError {
    /// The number of weights in the package is not equal to `out · in`.
    WeightSize {
        /// Expected `out_features · in_features`.
        expected: usize,
        /// Packaged.
        got: usize,
    },
    /// The bias length is not equal to `out_features`.
    BiasSize {
        /// Expected `out_features`.
        expected: usize,
        /// Actual length.
        got: usize,
    },
    /// There is no fused path or f32 turn for this package.
    Unsupported(&'static str),
    /// Failed to read tensor data (weight or activation) to host.
    TensorRead(String),
    /// fused quantized GEMM error.
    Gemm(String),
}

impl std::fmt::Display for QuantizedLinearError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::WeightSize { expected, got } => {
                write!(
                    f,
                    "QuantizedLinear: packed weight has {got} elements, expected {expected}"
                )
            }
            Self::BiasSize { expected, got } => {
                write!(
                    f,
                    "QuantizedLinear: bias has {got} elements, expected {expected}"
                )
            }
            Self::Unsupported(why) => write!(f, "QuantizedLinear: unsupported: {why}"),
            Self::TensorRead(why) => write!(f, "QuantizedLinear: cannot read tensor data: {why}"),
            Self::Gemm(why) => write!(f, "QuantizedLinear: quantized GEMM failed: {why}"),
        }
    }
}

impl std::error::Error for QuantizedLinearError {}

/// Linear layer on packed weights. The weights are frozen; the layer is a `Module`, so
/// that it can be embedded in `#[derive(Module)]`-models. Packing in
/// model records (`record`) **not** saved - they are loaded/placed separately
/// (via `neo-quantization`).
#[derive(Module, Debug)]
pub struct QuantizedLinear {
    /// Packed `W [out, in]`. `Arc` - the layer clone does not copy the weights.
    #[module(skip)]
    weight: Arc<PackedBuffer>,
    /// Offset (remains float).
    bias: Option<Tensor<1>>,
    in_features: usize,
    out_features: usize,
    /// Execution Policy.
    #[module(skip)]
    policy: ExecutionPolicy,
}

impl QuantizedLinear {
    /// Assemble a layer from the finished `W [out, in]` package (for example, loaded
    /// from GGUF or the resulting `neo_quantization::quantize`).
    pub fn new(
        weight: PackedBuffer,
        in_features: usize,
        out_features: usize,
        bias: Option<Tensor<1>>,
    ) -> Result<Self, QuantizedLinearError> {
        if matches!(weight, PackedBuffer::GgufRaw { .. }) {
            return Err(QuantizedLinearError::Unsupported(
                "GgufRaw has no decoder in neo-quantization",
            ));
        }
        let expected = in_features * out_features;
        if weight.numel() != expected {
            return Err(QuantizedLinearError::WeightSize {
                expected,
                got: weight.numel(),
            });
        }
        if let Some(b) = &bias {
            let got = b.dims()[0];
            if got != out_features {
                return Err(QuantizedLinearError::BiasSize {
                    expected: out_features,
                    got,
                });
            }
        }
        Ok(Self {
            weight: Arc::new(weight),
            bias,
            in_features,
            out_features,
            policy: ExecutionPolicy::Auto,
        })
    }

    /// Quantize the float weights `w` of the shape `[out, in]` (row-major) by `config`.
    ///
    /// `row_len` for `PerChannel` is set to `in_features`.
    pub fn from_weights(
        w: &[f32],
        in_features: usize,
        out_features: usize,
        bias: Option<Tensor<1>>,
        config: &QuantizeConfig,
    ) -> Result<Self, QuantizedLinearError> {
        let expected = in_features * out_features;
        if w.len() != expected {
            return Err(QuantizedLinearError::WeightSize {
                expected,
                got: w.len(),
            });
        }
        Self::new(
            quantize(w, config, in_features),
            in_features,
            out_features,
            bias,
        )
    }

    /// Quantize trained `burn::nn::Linear`.
    ///
    /// In Burn, `weight` has shape `[in, out]`; here it is transposed to
    /// `[out, in]` so that the scaling groups go along `in` (the convolution axis).
    ///
    /// # Panics
    /// If the weights could not be read or packaged (the way without panic is
    /// [`try_from_linear`](Self::try_from_linear)).
    pub fn from_linear(linear: &Linear, config: &QuantizeConfig) -> Self {
        Self::try_from_linear(linear, config).unwrap_or_else(|e| panic!("{e}"))
    }

    /// Like [`from_linear`](Self::from_linear), but with `Result` instead of panic.
    pub fn try_from_linear(
        linear: &Linear,
        config: &QuantizeConfig,
    ) -> Result<Self, QuantizedLinearError> {
        let [d_in, d_out] = linear.weight.val().dims();
        let w_in_out = linear
            .weight
            .val()
            .into_data()
            .convert::<f32>()
            .try_to_vec::<f32>()
            .map_err(|e| QuantizedLinearError::TensorRead(format!("weight: {e:?}")))?;

        let mut w_out_in = vec![0.0f32; w_in_out.len()];
        for i in 0..d_in {
            for o in 0..d_out {
                w_out_in[o * d_in + i] = w_in_out[i * d_out + o];
            }
        }

        let bias = linear.bias.as_ref().map(|b| b.val());
        Self::from_weights(&w_out_in, d_in, d_out, bias, config)
    }

    /// Set execution policy.
    pub fn with_policy(mut self, policy: ExecutionPolicy) -> Self {
        self.policy = policy;
        self
    }

    /// User defined policy.
    pub fn policy(&self) -> ExecutionPolicy {
        self.policy
    }

    /// Policies that will actually work (`Auto` revealed).
    pub fn effective_policy(&self) -> ExecutionPolicy {
        match self.policy {
            ExecutionPolicy::Auto => {
                if kernel_kind(&self.weight, self.in_features) == KernelKind::DenseFallback {
                    ExecutionPolicy::Dequantize
                } else {
                    ExecutionPolicy::Fused
                }
            }
            p => p,
        }
    }

    /// What kernel level is used for this packaging.
    pub fn kernel_kind(&self) -> KernelKind {
        kernel_kind(&self.weight, self.in_features)
    }

    /// Input size.
    pub fn in_features(&self) -> usize {
        self.in_features
    }

    /// Output size.
    pub fn out_features(&self) -> usize {
        self.out_features
    }

    /// Packed weights.
    pub fn packed(&self) -> &PackedBuffer {
        &self.weight
    }

    /// Actual bits by weight, taking into account scales/zero-points.
    pub fn bits_per_weight(&self) -> f32 {
        self.weight.effective_bits_per_weight()
    }

    /// How many bytes do the packed weights take up (taking into account scales/zero-points).
    pub fn weight_bytes(&self) -> usize {
        (self.bits_per_weight() as f64 * self.weight.numel() as f64 / 8.0).ceil() as usize
    }

    /// How many times is the package smaller than the same matrix in f32.
    pub fn compression_ratio(&self) -> f32 {
        32.0 / self.bits_per_weight()
    }

    /// Expand the weights into a regular tensor `[out, in]` (for debugging and policies
    /// `Dequantize`). Creates a complete matrix in f32 - do not call it in the hot path.
    pub fn dequantize_weight(&self, device: &Device) -> Tensor<2> {
        let data = self.weight.to_f32();
        Tensor::from_data(
            TensorData::new(data, [self.out_features, self.in_features]),
            device,
        )
    }

    /// Direct pass: `[..., in] → [..., out]`.
    ///
    /// # Panics
    ///
    /// If the last axis `x` is not equal to `in_features`, or fused core/read
    /// activations ended with an error (the path without data panic is
    /// [`try_forward`](Self::try_forward)).
    pub fn forward<const D: usize>(&self, x: Tensor<D>) -> Tensor<D> {
        self.try_forward(x).unwrap_or_else(|e| panic!("{e}"))
    }

    /// Like [`forward`](Self::forward): errors reading activations and GEMM
    /// are returned as [`QuantizedLinearError`]. Shape mismatch `x`
    /// still panic (programmer error).
    pub fn try_forward<const D: usize>(
        &self,
        x: Tensor<D>,
    ) -> Result<Tensor<D>, QuantizedLinearError> {
        let mut dims = x.dims();
        assert_eq!(
            dims[D - 1],
            self.in_features,
            "QuantizedLinear: last dim of x is {}, expected in_features = {}",
            dims[D - 1],
            self.in_features
        );
        let device = x.device();
        let m: usize = dims[..D - 1].iter().product();
        let (k, n) = (self.in_features, self.out_features);

        dims[D - 1] = n;
        if m == 0 {
            return Ok(Tensor::zeros(dims, &device));
        }

        let x2 = x.reshape([m, k]);

        let y = match self.effective_policy() {
            ExecutionPolicy::Dequantize => x2.matmul(self.dequantize_weight(&device).transpose()),
            _ => {
                let xv = x2
                    .into_data()
                    .convert::<f32>()
                    .try_to_vec::<f32>()
                    .map_err(|e| QuantizedLinearError::TensorRead(format!("activations: {e:?}")))?;
                let mut out = vec![0.0f32; m * n];
                packed_matmul(&xv, m, &self.weight, n, k, &mut out)
                    .map_err(|e: QuantGemmError| QuantizedLinearError::Gemm(e.to_string()))?;
                Tensor::<2>::from_data(TensorData::new(out, [m, n]), &device)
            }
        };

        let y = match &self.bias {
            Some(b) => y.add(b.clone().reshape([1, n])),
            None => y,
        };
        Ok(y.reshape(dims))
    }
}
