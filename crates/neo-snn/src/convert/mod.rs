//! Converting ordinary neural networks into spike networks (ANN → SNN).
//!
//! Allows you to take an already trained ANN (`Linear`/`Conv2d`/`BatchNorm`/`ReLU`/
//! `AvgPool2d`) and get a spike network without retraining: each
//! `ReLU` is replaced by the integrate-and-fire neuron, whose firing rate is
//! brings the activation value closer.
//!
//! # How it works
//!
//! 1. [`AnnNet::canonicalize`] - `BatchNorm` collapses into the previous layer,
//!    it is verified that the network is convertible.
//! 2. [`activation_scales`] - according to calibration data for each `ReLU`
//!    `λ` is taken (maximum or percentile of activation, [`NormMode`]).
//! 3. [`convert`] - weights are recalculated as `W·λ_prev/λ`, offsets as
//!    `b/λ`; the threshold of all neurons is `1`.
//! 4. [`SpikingNet::run`] - run `T` steps, the output is the average output in
//!    units of the original ANN, by steps and spike statistics.
//!
//! The last layer with weights without `ReLU` becomes the readout (current
//! averaged over time), so there is no saturation in the logits.
//!
//! # Example
//!
//! Conversion function - [`convert`]; in `neo_snn::prelude` it is available as
//! `ann_to_snn`.
//!
//! ```ignore
//! use neo_snn::prelude::*;
//!
//! let ann = AnnNet::new()
//!     .linear(&model.fc1).relu()
//!     .linear(&model.fc2).relu()
//!     .linear(&model.out);
//!
//! // Several representative batches of training data.
//! let calib = [Act::D2(train_batch)];
//! let snn = ann_to_snn(&ann, &calib, &ConvertConfig::default())?;
//!
//! let result = snn.run(&Act::D2(test_batch), 100)?;
//! let acc_by_t = result.accuracy_over_time(&labels)?; // accuracy vs number of steps
//! println!("spikes per batch: {}", result.total_spikes());
//! ```
//!
//! # Restrictions
//!
//! Only serial networks, only `ReLU`, only `AvgPool2d` (y
//! `MaxPool` there is no exact spike analogue), neurons without leakage. Accuracy
//! increases with the number of steps `T`; Usually tens to hundreds of steps are needed.

mod ann;
mod normalize;
mod snn;

#[cfg(test)]
mod tests;

pub use ann::{Act, AnnLayer, AnnNet, BatchNormParams};
pub use normalize::{NormMode, activation_scales};
pub use snn::{InputEncoding, LayerStats, SnnOutput, SnnState, SpikeConfig, SpikingNet, argmax};

use crate::error::Result;
use ann::AnnLayer as L;
use snn::{SnnLayer, scaled_conv, scaled_linear};

/// Conversion options.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct ConvertConfig {
    /// Selecting the activation scale based on calibration data.
    pub norm: NormMode,
    /// Neuron and input parameters.
    pub spike: SpikeConfig,
}

/// Convert ANN to SNN.
///
/// `calibration` — batches of inputs for selecting scales (for
/// [`NormMode::None`] may be empty).
pub fn convert(ann: &AnnNet, calibration: &[Act], cfg: &ConvertConfig) -> Result<SpikingNet> {
    cfg.spike.validate()?;
    let net = ann.canonicalize()?;
    let scales = activation_scales(&net, calibration, cfg.norm)?;

    let layers = net.layers();
    let mut out = Vec::with_capacity(layers.len());
    let mut lambdas = Vec::with_capacity(scales.len());
    let mut prev = 1.0f32; // λ input: input is supplied without normalization.
    let mut relu_idx = 0;
    let mut last_spiking = false;
    for (i, layer) in layers.iter().enumerate() {
        match layer {
            L::Linear(_) | L::Conv2d(_) => {
                let spiking = matches!(layers.get(i + 1), Some(L::Relu));
                let lam = if spiking { scales[relu_idx] } else { 1.0 };
                let (k, kb) = (prev / lam, 1.0 / lam);
                out.push(match layer {
                    L::Linear(l) => SnnLayer::Linear {
                        lin: scaled_linear(l, k, kb),
                        spiking,
                    },
                    L::Conv2d(c) => SnnLayer::Conv {
                        conv: scaled_conv(c, k, kb),
                        spiking,
                    },
                    _ => unreachable!(),
                });
                if spiking {
                    lambdas.push(lam);
                    relu_idx += 1;
                    prev = lam;
                }
                last_spiking = spiking;
            }
            L::AvgPool2d { kernel, stride } => out.push(SnnLayer::AvgPool {
                kernel: *kernel,
                stride: *stride,
            }),
            L::Flatten => out.push(SnnLayer::Flatten),
            // Relu is absorbed by the neuron, BatchNorm/Identity is removed by canonicalize.
            L::Relu | L::BatchNorm(_) | L::Identity => {}
        }
    }
    let out_scale = if last_spiking { prev } else { 1.0 };
    Ok(SpikingNet {
        layers: out,
        cfg: cfg.spike,
        scales: lambdas,
        out_scale,
    })
}
