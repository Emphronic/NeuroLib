//! Fourier features: deterministic representation of position or continuous
//! quantities through a set of sines and cosines.
//!
//! ```text
//!  x ──► angles = 2π · f_i · x ──► [sin(angles), cos(angles)] ──► (Linear) ──► y
//! ```
//!
//! Unlike `Embedding`, this is not a lookup by id: the input is real numbers
//! (position, time, coordinate, scalar attribute).
//!
//! # Feature layout
//!
//! Input `[..., input_dim]`, `F = num_frequencies`. Last result axis
//! (before optional projection):
//!
//! ```text
//! [ x_0 … x_{D-1} ] if include_input (D values)
//! [ sin(2π f_0 x_0) … sin(2π f_{F-1} x_0),
//!   cos(2π f_0 x_0) … cos(2π f_{F-1} x_0) ] for coordinate 0 (2F values)
//! [same for coordinate 1]
//! …
//! ```
//!
//! that is, `fourier_dim = (include_input ? D : 0) + 2·D·F`. For `D = 1`,
//! `F = 8` - 16 features, and with `include_input` - 17.
//!
//! If `fourier_dim` is too large (for example `D = 4096`, `F = 16` gives 131,072
//! feature), set `output_dim`: the features will pass through `Linear` and compress
//! to the desired width.
//!
//! # Frequencies
//!
//! Frequencies `f_i` - in cycles per unit `x`, common for all coordinates, **not
//! are trained** (regular tensor in the module, not `Param`), so backward goes
//! only through `sin`/`cos`/`mul`/`Linear` and does not require a special code.
//! Scale selection - [`FrequencyScale`].
//!
//! # Accuracy
//!
//! The calculation is carried out in `f32` (or in the float type of the backend): for large `x · f`
//! (eg positions in the hundreds of thousands with high frequencies) the phase loses accuracy.
//! Integer positions are representable exactly up to 2²⁴.
//!
//! # Communication with the transformer
//!
//! ```text
//! position ─► FourierEmbedding ─► (Linear) ─┐
//!                                           +─► Transformer
//! tokens ───► Embedding ────────────────────┘
//! ```
//!
//! For Llama/Qwen-like blocks, the main one remains RoPE; `FourierEmbedding`
//! needed where the positional mechanism is additive features, or for
//! continuous inputs. Fused core candidate in `neo-synk`: multiplication by
//! frequencies + sin/cos + projection.
//!
//! # Example
//!
//! ```ignore
//! use burn::tensor::Device;
//! use neo_nn::embedding::FourierEmbeddingConfig;
//!
//! let device = Device::flex(); // CPU backend (feature "flex")
//!
//! // Positions 0..T → [T, 2 8] features, then in d_model = 64.
//! let fourier = FourierEmbeddingConfig::new(1, 8).with_output_dim(64).init(&device);
//! let pos = fourier.position_encoding(128, &device); // [128, 64]
//! ```

use std::f32::consts::TAU;

use burn::nn::{Linear, LinearConfig};
use burn::prelude::*;
use burn::tensor::{Int, TensorData};

/// How are the frequencies `f_0 … f_{F-1}` distributed.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FrequencyScale {
    /// Evenly: `f_i = min + (max − min) · i / (F − 1)`.
    Linear,
    /// Octaves: `f_i = min · 2^i`. `max_frequency` is not used.
    Log2,
    /// Geometrically between boundaries: `log f_i` uniformly in
    /// `[log min, log max]`, that is `f_i = min · (max/min)^(i / (F − 1))`.
    Geometric,
}

/// Configuration [`FourierEmbedding`].
#[derive(Debug, Clone)]
pub struct FourierEmbeddingConfig {
    /// Number of input coordinates `D` (size of the last input axis).
    pub input_dim: usize,
    /// Number of frequencies `F` per coordinate.
    pub num_frequencies: usize,
    /// Lowest frequency (cycles per unit `x`), `> 0`.
    pub min_frequency: f32,
    /// Highest frequency; for [`FrequencyScale::Log2`] is not used.
    pub max_frequency: f32,
    /// Frequency scale.
    pub scale: FrequencyScale,
    /// Output width: `Some(n)` - features pass through `Linear(fourier_dim, n)`,
    /// `None` - issued as is.
    pub output_dim: Option<usize>,
    /// Add the original values `x` before the sin/cos attributes.
    pub include_input: bool,
    /// Offset in the output projection (if there is one).
    pub bias: bool,
}

impl FourierEmbeddingConfig {
    /// Default values: scale [`FrequencyScale::Log2`] from `1.0`
    /// (`f_i = 2^i`), no projection, no original input, `bias = true`.
    pub fn new(input_dim: usize, num_frequencies: usize) -> Self {
        Self {
            input_dim,
            num_frequencies,
            min_frequency: 1.0,
            max_frequency: 2f32
                .powi(num_frequencies.saturating_sub(1).min(i32::MAX as usize) as i32),
            scale: FrequencyScale::Log2,
            output_dim: None,
            include_input: false,
            bias: true,
        }
    }

    /// Lowest frequency.
    pub fn with_min_frequency(mut self, min_frequency: f32) -> Self {
        self.min_frequency = min_frequency;
        self
    }

    /// Highest frequency.
    pub fn with_max_frequency(mut self, max_frequency: f32) -> Self {
        self.max_frequency = max_frequency;
        self
    }

    /// Frequency scale.
    pub fn with_scale(mut self, scale: FrequencyScale) -> Self {
        self.scale = scale;
        self
    }

    /// Enable output projection `Linear(fourier_dim, output_dim)`.
    pub fn with_output_dim(mut self, output_dim: usize) -> Self {
        self.output_dim = Some(output_dim);
        self
    }

    /// Whether to add the original input before the sin/cos attributes.
    pub fn with_include_input(mut self, include_input: bool) -> Self {
        self.include_input = include_input;
        self
    }

    /// Offset in the output projection.
    pub fn with_bias(mut self, bias: bool) -> Self {
        self.bias = bias;
        self
    }

    /// Frequencies `f_0 … f_{F-1}` (cycles per unit `x`). Computed in `f64`.
    pub fn frequency_values(&self) -> Vec<f32> {
        let f = self.num_frequencies;
        let (lo, hi) = (self.min_frequency as f64, self.max_frequency as f64);
        (0..f)
            .map(|i| {
                let value = match self.scale {
                    FrequencyScale::Log2 => lo * 2f64.powi(i as i32),
                    _ if f == 1 => lo,
                    FrequencyScale::Linear => lo + (hi - lo) * i as f64 / (f - 1) as f64,
                    FrequencyScale::Geometric => {
                        (lo.ln() + (hi.ln() - lo.ln()) * i as f64 / (f - 1) as f64).exp()
                    }
                };
                value as f32
            })
            .collect()
    }

    /// Width of features before projection:
    /// `(include_input ? input_dim : 0) + 2 · input_dim · num_frequencies`.
    pub fn fourier_dim(&self) -> usize {
        let raw = if self.include_input {
            self.input_dim
        } else {
            0
        };
        raw + 2 * self.input_dim * self.num_frequencies
    }

    /// Result width: `output_dim` if specified, otherwise [`fourier_dim`](Self::fourier_dim).
    pub fn output_width(&self) -> usize {
        self.output_dim.unwrap_or_else(|| self.fourier_dim())
    }

    /// Checking parameters.
    pub fn validate(&self) -> Result<(), String> {
        if self.input_dim == 0 {
            return Err("input_dim must be > 0".into());
        }
        if self.num_frequencies == 0 {
            return Err("num_frequencies must be > 0".into());
        }
        if !(self.min_frequency.is_finite() && self.min_frequency > 0.0) {
            return Err("min_frequency must be finite and > 0".into());
        }
        if self.scale != FrequencyScale::Log2 {
            if !self.max_frequency.is_finite() {
                return Err("max_frequency must be finite".into());
            }
            if self.max_frequency < self.min_frequency {
                return Err("max_frequency should be >= min_frequency".into());
            }
        }
        if !self.frequency_values().iter().all(|f| f.is_finite()) {
            return Err("frequencies are outside the f32 range (too many octaves)".into());
        }
        if self.output_dim == Some(0) {
            return Err("output_dim must be > 0".into());
        }
        Ok(())
    }

    /// Create a layer.
    ///
    /// # Panics
    /// If the configuration is invalid (see [`validate`](Self::validate)).
    pub fn init(&self, device: &Device) -> FourierEmbedding {
        self.try_init(device)
            .unwrap_or_else(|e| panic!("FourierEmbeddingConfig: {e}"))
    }

    /// Like [`init`](Self::init), but with `Result` instead of panic.
    pub fn try_init(&self, device: &Device) -> Result<FourierEmbedding, String> {
        self.validate()?;
        let freqs = self.frequency_values();
        let n = freqs.len();
        Ok(FourierEmbedding {
            projection: self.output_dim.map(|out| {
                LinearConfig::new(self.fourier_dim(), out)
                    .with_bias(self.bias)
                    .init(device)
            }),
            frequencies: Tensor::from_data(TensorData::new(freqs, [n]), device),
            input_dim: self.input_dim,
            num_frequencies: self.num_frequencies,
            include_input: self.include_input,
        })
    }
}

/// Fourier features with optional output projection.
#[derive(Module, Debug)]
pub struct FourierEmbedding {
    /// Projection `fourier_dim → output_dim` (the only trainable parameters).
    pub projection: Option<Linear>,
    /// Frequencies `[num_frequencies]`, cycles per unit `x`. They don't study.
    pub frequencies: Tensor<1>,
    /// Size of the last input axis.
    #[module(skip)]
    pub input_dim: usize,
    /// Number of frequencies per coordinate.
    #[module(skip)]
    pub num_frequencies: usize,
    /// Whether to add the original input.
    #[module(skip)]
    pub include_input: bool,
}

impl FourierEmbedding {
    /// Width of features before projection.
    pub fn fourier_dim(&self) -> usize {
        let raw = if self.include_input {
            self.input_dim
        } else {
            0
        };
        raw + 2 * self.input_dim * self.num_frequencies
    }

    /// Accepts input of any rank `D >= 1`: `[..., input_dim]` →
    /// `[..., output_width]`. The layout is described in the module documentation.
    ///
    /// # Panics
    /// If the last input axis is not equal to `input_dim`.
    pub fn forward<const D: usize>(&self, x: Tensor<D>) -> Tensor<D> {
        assert!(D >= 1, "FourierEmbedding: need a tensor of rank >= 1");
        let mut shape = x.dims();
        assert_eq!(
            shape[D - 1],
            self.input_dim,
            "FourierEmbedding: expected last axis {}, received {}",
            self.input_dim,
            shape[D - 1]
        );
        let n: usize = shape[..D - 1].iter().product();

        let feats = self.features(x.reshape([n, self.input_dim]));
        let y = match &self.projection {
            Some(p) => p.forward(feats),
            None => feats,
        };

        shape[D - 1] = y.dims()[1];
        y.reshape(shape)
    }

    /// Scalars without feature axis: `[T]` → `[T, output_width]`. Need `input_dim == 1`.
    pub fn encode_scalars_1d(&self, x: Tensor<1>) -> Tensor<2> {
        assert_eq!(
            self.input_dim, 1,
            "FourierEmbedding: encode_scalars_* requires input_dim = 1"
        );
        let [t] = x.dims();
        self.forward(x.reshape([t, 1]))
    }

    /// Scalars without feature axis: `[B, T]` → `[B, T, output_width]`. Need `input_dim == 1`.
    pub fn encode_scalars_2d(&self, x: Tensor<2>) -> Tensor<3> {
        assert_eq!(
            self.input_dim, 1,
            "FourierEmbedding: encode_scalars_* requires input_dim = 1"
        );
        let [b, t] = x.dims();
        self.forward(x.reshape([b, t, 1]))
    }

    /// Signs of positions `0, 1, …, seq_len − 1`: `[seq_len, output_width]`.
    /// Need `input_dim == 1`.
    pub fn position_encoding(&self, seq_len: usize, device: &Device) -> Tensor<2> {
        self.position_encoding_at(0, seq_len, device)
    }

    /// Signs of positions `start, …, start + seq_len − 1` (for example, for
    /// continuation of generation with KV cache): `[seq_len, output_width]`.
    /// Need `input_dim == 1`.
    pub fn position_encoding_at(&self, start: usize, seq_len: usize, device: &Device) -> Tensor<2> {
        let pos = Tensor::<1, Int>::arange(start as i64..(start + seq_len) as i64, device).float();
        self.encode_scalars_1d(pos)
    }

    /// sin/cos-features for `[N, input_dim]` → `[N, fourier_dim]`.
    fn features(&self, x: Tensor<2>) -> Tensor<2> {
        let [n, d] = x.dims();
        let f = self.num_frequencies;

        // [N, D, 1] · [1, 1, F] → [N, D, F]
        let angles = x
            .clone()
            .reshape([n, d, 1])
            .mul(self.frequencies.clone().reshape([1, 1, f]))
            .mul_scalar(TAU);

        let sin_cos =
            Tensor::cat(vec![angles.clone().sin(), angles.cos()], 2).reshape([n, d * 2 * f]);

        if self.include_input {
            Tensor::cat(vec![x, sin_cos], 1)
        } else {
            sin_cos
        }
    }
}
