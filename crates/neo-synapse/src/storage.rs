//! Loading a sorting for curation.

use std::path::Path;

use neo_core::{DspError, DspResult};
use neo_io::neuro::phy::PhyFolder;

use crate::math::lagged_dot;

/// Lag window (± samples) of the template similarity, as Kilosort's own.
const SIMILARITY_MAX_LAG: usize = 5;

/// Spikes of the sorting at `path`: a phy / Kilosort folder, or a file inside one. The similarity
/// of the templates is filled in when the folder has none.
///
/// # Errors
///
/// When `path` is not (in) a phy folder, or an array of the folder cannot be read.
pub fn load_spikes(path: &Path) -> DspResult<PhyFolder> {
    let folder = if path.is_dir() {
        Some(path)
    } else {
        path.parent().filter(|_| path.is_file())
    };
    let Some(dir) = folder.filter(|d| PhyFolder::is_phy_folder(d)) else {
        return Err(DspError::UnsupportedFormat(format!(
            "{} is not a phy / Kilosort folder (spike_times.npy, spike_clusters.npy)",
            path.display()
        )));
    };
    let mut spikes = PhyFolder::read(dir)?;
    fill_similarity(&mut spikes);
    Ok(spikes)
}

/// Computes `similar_templates` from the templates when the folder has none: the lag-maximised
/// cosine similarity of every pair of templates.
pub fn fill_similarity(folder: &mut PhyFolder) {
    if folder.similar_templates.is_some() {
        return;
    }
    folder.similar_templates = folder.templates.as_ref().filter(|t| t.count > 0).map(|t| {
        // Only the channels a template is not silent on
        let rows: Vec<Vec<(usize, Vec<f32>)>> = (0..t.count)
            .map(|i| {
                (0..t.channels)
                    .map(|c| (c, t.trace(i, c)))
                    .filter(|(_, tr)| tr.iter().any(|&v| v != 0.0))
                    .collect()
            })
            .collect();
        let norms: Vec<f32> = rows
            .iter()
            .map(|r| {
                r.iter()
                    .flat_map(|(_, tr)| tr)
                    .map(|v| v * v)
                    .sum::<f32>()
                    .sqrt()
                    .max(1e-8)
            })
            .collect();
        let max_lag = SIMILARITY_MAX_LAG.min(t.samples / 2) as isize;
        let n = t.count;
        let mut mat = vec![0.0f32; n * n];
        for i in 0..n {
            mat[i * n + i] = 1.0;
            for j in (i + 1)..n {
                let best = (-max_lag..=max_lag)
                    .map(|lag| {
                        rows[i]
                            .iter()
                            .filter_map(|(c, a)| {
                                rows[j]
                                    .iter()
                                    .find(|(cj, _)| cj == c)
                                    .map(|(_, b)| lagged_dot(a, b, lag))
                            })
                            .sum::<f32>()
                    })
                    .fold(f32::NEG_INFINITY, f32::max);
                let sim = (best / (norms[i] * norms[j])).clamp(-1.0, 1.0);
                mat[i * n + j] = sim;
                mat[j * n + i] = sim;
            }
        }
        mat
    });
}

#[cfg(test)]
mod tests {
    use super::*;
    use neo_io::neuro::templates::DenseTemplates;

    #[test]
    fn similarity_is_one_on_the_diagonal_and_high_for_equal_shapes() {
        // 3 templates, 4 samples, 2 channels: 0 and 1 share a shape, 2 differs
        let mut data = vec![0.0f32; 3 * 4 * 2];
        let set =
            |d: &mut Vec<f32>, t: usize, s: usize, c: usize, v: f32| d[(t * 4 + s) * 2 + c] = v;
        for t in 0..2 {
            set(&mut data, t, 1, 0, -5.0);
            set(&mut data, t, 2, 0, 2.0);
        }
        set(&mut data, 2, 1, 1, -4.0);
        let mut folder = PhyFolder {
            templates: Some(DenseTemplates {
                data,
                count: 3,
                samples: 4,
                channels: 2,
            }),
            ..Default::default()
        };
        fill_similarity(&mut folder);
        assert!((folder.similarity(0, 0) - 1.0).abs() < 1e-6);
        assert!(folder.similarity(0, 1) > 0.99);
        assert!(folder.similarity(0, 2).abs() < 1e-3);
        assert_eq!(folder.similarity(0, 1), folder.similarity(1, 0));
    }

    #[test]
    fn a_missing_folder_is_an_error() {
        assert!(load_spikes(Path::new("/definitely/not/here")).is_err());
    }
}
