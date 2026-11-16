//! A small host PCA of spike waveforms (covariance + power iteration with deflation).

use crate::snippets::WaveformSnippet;

/// A fitted PCA: the mean waveform and the unit principal axes, strongest first.
#[derive(Debug, Clone, PartialEq)]
pub struct PcaModel {
    pub num_components: usize,
    pub mean: Vec<f32>,
    /// `[component][sample]`.
    pub components: Vec<Vec<f32>>,
    /// Variance along each component.
    pub variance: Vec<f32>,
}

const ITERATIONS: usize = 100;

/// Fits `num_components` principal components to the primary trace of each snippet (snippets of
/// another length than the first get a zero row) and projects every snippet: one row of
/// `num_components` scores per snippet. `None` without snippets.
pub fn extract_waveform_pca(
    snippets: &[WaveformSnippet],
    num_components: usize,
) -> Option<(PcaModel, Vec<Vec<f32>>)> {
    let first = snippets.first()?;
    let len = first.num_samples;
    if len == 0 || num_components == 0 {
        return None;
    }
    let rows: Vec<&[f32]> = snippets
        .iter()
        .map(|s| {
            if s.num_samples == len {
                s.primary_trace()
            } else {
                &[][..]
            }
        })
        .collect();
    let valid_rows: Vec<&[f32]> = rows.iter().copied().filter(|r| r.len() == len).collect();
    let used = valid_rows.len().max(1);
    let mut mean = vec![0.0f32; len];
    for r in &valid_rows {
        mean.iter_mut()
            .zip(r.iter())
            .for_each(|(m, v)| *m += v / used as f32);
    }
    // Covariance of the centred waveforms, `len × len`
    let mut cov = vec![0.0f64; len * len];
    for r in &valid_rows {
        let c: Vec<f64> = r.iter().zip(&mean).map(|(v, m)| (v - m) as f64).collect();
        for i in 0..len {
            for j in i..len {
                cov[i * len + j] += c[i] * c[j];
            }
        }
    }
    let scale = 1.0 / used.saturating_sub(1).max(1) as f64;
    for i in 0..len {
        for j in i..len {
            cov[i * len + j] *= scale;
            cov[j * len + i] = cov[i * len + j];
        }
    }
    let k = num_components.min(len);
    let (mut components, mut variance) = (Vec::with_capacity(k), Vec::with_capacity(k));
    for c in 0..k {
        // Deterministic start that is not orthogonal to the leading axis
        let mut v: Vec<f64> = (0..len).map(|i| 1.0 + ((i + c) % 7) as f64 * 0.1).collect();
        normalize(&mut v);
        let mut eig = 0.0;
        for _ in 0..ITERATIONS {
            let mut w: Vec<f64> = (0..len)
                .map(|i| (0..len).map(|j| cov[i * len + j] * v[j]).sum())
                .collect();
            eig = norm(&w);
            if eig < 1e-18 {
                break;
            }
            w.iter_mut().for_each(|x| *x /= eig);
            v = w;
        }
        // Deflate
        for i in 0..len {
            for j in 0..len {
                cov[i * len + j] -= eig * v[i] * v[j];
            }
        }
        components.push(v.iter().map(|&x| x as f32).collect::<Vec<f32>>());
        variance.push(eig as f32);
    }
    let projected = rows
        .iter()
        .map(|r| {
            if r.len() != len {
                return vec![0.0; k];
            }
            components
                .iter()
                .map(|axis| {
                    axis.iter()
                        .zip(r.iter().zip(&mean))
                        .map(|(a, (v, m))| a * (v - m))
                        .sum()
                })
                .collect()
        })
        .collect();
    Some((
        PcaModel {
            num_components: k,
            mean,
            components,
            variance,
        },
        projected,
    ))
}

fn norm(v: &[f64]) -> f64 {
    v.iter().map(|x| x * x).sum::<f64>().sqrt()
}

fn normalize(v: &mut [f64]) {
    let n = norm(v).max(1e-300);
    v.iter_mut().for_each(|x| *x /= n);
}

#[cfg(test)]
mod tests {
    use super::*;

    fn snippet(w: Vec<f32>) -> WaveformSnippet {
        WaveformSnippet {
            primary_channel: 0,
            center_sample: 0,
            subsample_offset: 0.0,
            channel_ids: vec![0],
            num_samples: w.len(),
            waveform: w,
        }
    }

    #[test]
    fn the_first_component_follows_the_dominant_variation() {
        // Waveforms differ only by the amplitude of one shape
        let shape = [0.0f32, -1.0, -3.0, -1.0, 0.0, 1.0];
        let snippets: Vec<_> = (1..=8)
            .map(|a| snippet(shape.iter().map(|v| v * a as f32).collect()))
            .collect();
        let (model, rows) = extract_waveform_pca(&snippets, 2).unwrap();
        assert_eq!(rows.len(), 8);
        let norm = model.components[0]
            .iter()
            .map(|v| v * v)
            .sum::<f32>()
            .sqrt();
        assert!((norm - 1.0).abs() < 1e-4);
        // Scores are monotonic in amplitude
        let pc1: Vec<f32> = rows.iter().map(|r| r[0]).collect();
        assert!(pc1.windows(2).all(|w| w[0] < w[1]) || pc1.windows(2).all(|w| w[0] > w[1]));
        assert!(model.variance[0] > model.variance[1] * 100.0);
    }

    #[test]
    fn nothing_without_snippets() {
        assert!(extract_waveform_pca(&[], 1).is_none());
    }
}
