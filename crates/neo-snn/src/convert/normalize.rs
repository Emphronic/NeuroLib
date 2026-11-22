//! Normalization to data (Diehl et al. 2015, Rueckauer et al. 2017).
//!
//! A spike neuron cannot produce more than one spike per step, so
//! activations of ANN that exceed the frequency “ceiling” are saturated. To avoid this
//! was, for each `ReLU` the scale is found from the calibration data
//! `λ` is the maximum (or high percentile) of activation, and the weights are recalculated
//! so that the firing rate of the neuron is equal to `activation / λ ∈ [0, 1]`.

use super::ann::{Act, AnnNet};
use crate::error::{Result, SnnError};

/// How to choose the scale `λ` of a layer based on calibration activations.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum NormMode {
    /// Without normalization (`λ = 1`). Suitable only if activations are already
    /// lie in `[0, 1]`.
    None,
    /// Maximum activation. Accurate, but one overshoot greatly underestimates the frequencies.
    Max,
    /// Percentile for all layer activations (in percent, `(0, 100]`). Robust
    /// version from Rueckauer et al.; `99.9` is the default value.
    Percentile(f32),
}

impl Default for NormMode {
    fn default() -> Self {
        NormMode::Percentile(99.9)
    }
}

/// Scales `λ` for each `ReLU` of the canonical network (in layer order).
///
/// `calibration` — one or more batches of inputs. For [`NormMode::None`]
/// no calibration needed. The layer on which all activations are zero (“dead”),
/// gets `λ = 1` so as not to divide by zero.
pub fn activation_scales(net: &AnnNet, calibration: &[Act], mode: NormMode) -> Result<Vec<f32>> {
    let n_relu = net
        .layers()
        .iter()
        .filter(|l| matches!(l, super::ann::AnnLayer::Relu))
        .count();
    if matches!(mode, NormMode::None) {
        return Ok(vec![1.0; n_relu]);
    }
    if let NormMode::Percentile(p) = mode
        && !(p > 0.0 && p <= 100.0)
    {
        return Err(SnnError::InvalidConfig {
            what: "NormMode::Percentile",
            reason: format!("percentile must be at (0, 100], obtained {p}"),
        });
    }
    if calibration.is_empty() {
        return Err(SnnError::EmptySequence {
            op: "activation_scales",
        });
    }

    let mut maxima = vec![0.0f32; n_relu];
    let mut values: Vec<Vec<f32>> = vec![Vec::new(); n_relu];
    let mut failure: Option<SnnError> = None;
    for batch in calibration {
        net.forward_observe(batch.clone(), &mut |idx, act| {
            if failure.is_some() {
                return;
            }
            match act.to_vec("activation_scales") {
                Ok(v) => match mode {
                    NormMode::Max => {
                        let m = v.iter().copied().fold(0.0f32, f32::max);
                        maxima[idx] = maxima[idx].max(m);
                    }
                    _ => values[idx].extend(v),
                },
                Err(e) => failure = Some(e),
            }
        })?;
        if let Some(e) = failure.take() {
            return Err(e);
        }
    }

    let scales = (0..n_relu)
        .map(|i| {
            let lam = match mode {
                NormMode::Max => maxima[i],
                NormMode::Percentile(p) => percentile(&mut values[i], p),
                NormMode::None => 1.0,
            };
            if lam.is_finite() && lam > 1e-8 {
                lam
            } else {
                1.0
            }
        })
        .collect();
    Ok(scales)
}

/// Percentile `p` (in percent) without full sorting; empty slice → 0.
pub(crate) fn percentile(values: &mut [f32], p: f32) -> f32 {
    if values.is_empty() {
        return 0.0;
    }
    let idx = (((p / 100.0) * (values.len() - 1) as f32).round() as usize).min(values.len() - 1);
    let (_, v, _) = values.select_nth_unstable_by(idx, |a, b| a.total_cmp(b));
    *v
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn percentile_picks_expected_rank() {
        let mut v: Vec<f32> = (0..=100).map(|x| x as f32).collect();
        assert_eq!(percentile(&mut v, 100.0), 100.0);
        assert_eq!(percentile(&mut v, 50.0), 50.0);
        assert_eq!(percentile(&mut v, 99.0), 99.0);
        assert_eq!(percentile(&mut [], 99.9), 0.0);
    }
}
