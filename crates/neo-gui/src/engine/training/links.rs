//! The connections between two layers as the views read them: either a dense matrix (small layers)
//! or a list of links (big layers, where the dense matrix would not fit in memory), behind one
//! interface. Also the region labels a model can attach to a layer.

use std::sync::Arc;

use super::dashboard::WeightMatrix;

/// Links the state keeps per pair of layers; more are dropped (the cost of a frame grows with them).
pub const MAX_LINKS: usize = 8_000_000;

/// `from[k] → to[k]` with weight `weight[k]`, between `rows` source neurons and `cols` targets.
#[derive(Debug, Clone)]
pub struct SparseLinks {
    pub rows: usize,
    pub cols: usize,
    pub from: Arc<Vec<u32>>,
    pub to: Arc<Vec<u32>>,
    pub weight: Arc<Vec<f32>>,
}

#[derive(Debug, Clone)]
pub enum Links {
    Dense(WeightMatrix),
    Sparse(SparseLinks),
}

impl Links {
    /// Source neurons.
    pub fn rows(&self) -> usize {
        match self {
            Links::Dense(m) => m.rows,
            Links::Sparse(s) => s.rows,
        }
    }

    /// Target neurons.
    pub fn cols(&self) -> usize {
        match self {
            Links::Dense(m) => m.cols,
            Links::Sparse(s) => s.cols,
        }
    }

    /// Links stored (for a dense matrix, every cell).
    pub fn len(&self) -> usize {
        match self {
            Links::Dense(m) => m.values.len(),
            Links::Sparse(s) => s.weight.len().min(s.from.len()).min(s.to.len()),
        }
    }

    /// Calls `f(source, target, weight)` for every stored link whose ends are inside the layers.
    pub fn for_each(&self, mut f: impl FnMut(usize, usize, f32)) {
        match self {
            Links::Dense(m) => {
                if m.cols == 0 {
                    return;
                }
                for (k, &w) in m.values.iter().enumerate() {
                    f(k / m.cols, k % m.cols, w);
                }
            }
            Links::Sparse(s) => {
                for k in 0..self.len() {
                    let (i, j) = (s.from[k] as usize, s.to[k] as usize);
                    if i < s.rows && j < s.cols {
                        f(i, j, s.weight[k]);
                    }
                }
            }
        }
    }

    /// The largest |weight| over the finite ones.
    pub fn abs_max(&self) -> f32 {
        let mut m = 0.0f32;
        self.for_each(|_, _, w| {
            if w.is_finite() {
                m = m.max(w.abs());
            }
        });
        m
    }

    /// The `n` strongest (by |weight|) links out of source neuron `i`, as `(target, weight)`.
    pub fn strongest_from(&self, i: usize, n: usize) -> Vec<(usize, f32)> {
        let mut out: Vec<(usize, f32)> = Vec::new();
        match self {
            Links::Dense(m) => {
                if i < m.rows {
                    out.extend(
                        (0..m.cols).filter_map(|j| m.values.get(i * m.cols + j).map(|&w| (j, w))),
                    );
                }
            }
            Links::Sparse(_) => self.for_each(|a, j, w| {
                if a == i {
                    out.push((j, w));
                }
            }),
        }
        keep_strongest(&mut out, n);
        out
    }

    /// The `n` strongest (by |weight|) links into target neuron `j`, as `(source, weight)`.
    pub fn strongest_into(&self, j: usize, n: usize) -> Vec<(usize, f32)> {
        let mut out: Vec<(usize, f32)> = Vec::new();
        match self {
            Links::Dense(m) => {
                if j < m.cols {
                    out.extend(
                        (0..m.rows).filter_map(|i| m.values.get(i * m.cols + j).map(|&w| (i, w))),
                    );
                }
            }
            Links::Sparse(_) => self.for_each(|i, b, w| {
                if b == j {
                    out.push((i, w));
                }
            }),
        }
        keep_strongest(&mut out, n);
        out
    }

    /// The links as a small dense `rows × cols` matrix (missing ones are zero), if it is small enough:
    /// at most `max_cols` columns and `max_cells` cells. For reading what each neuron votes for.
    pub fn small_dense(&self, max_cols: usize, max_cells: usize) -> Option<Vec<f32>> {
        let (rows, cols) = (self.rows(), self.cols());
        if cols == 0 || cols > max_cols || rows.checked_mul(cols)? > max_cells {
            return None;
        }
        let mut dense = vec![0.0f32; rows * cols];
        self.for_each(|i, j, w| {
            if w.is_finite() {
                dense[i * cols + j] = w;
            }
        });
        Some(dense)
    }
}

/// Sorts by |weight|, strongest first, and keeps `n`; non-finite and zero weights go.
fn keep_strongest(links: &mut Vec<(usize, f32)>, n: usize) {
    // A zero weight is no connection: dense and sparse forms must agree on what is linked
    links.retain(|(_, w)| w.is_finite() && *w != 0.0);
    links.sort_by(|a, b| b.1.abs().total_cmp(&a.1.abs()));
    links.truncate(n);
}

/// Which region each neuron of a layer is in.
#[derive(Debug, Clone, PartialEq)]
pub struct Regions {
    pub names: Arc<Vec<String>>,
    pub labels: Arc<Vec<u16>>,
}

impl Regions {
    /// The neurons of each region (`names.len()` lists), skipping neurons beyond `units` and labels without a name.
    pub fn members(&self, units: usize) -> Vec<Vec<usize>> {
        let mut by_region = vec![Vec::new(); self.names.len()];
        for (u, &r) in self.labels.iter().enumerate().take(units) {
            if let Some(list) = by_region.get_mut(r as usize) {
                list.push(u);
            }
        }
        by_region
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dense(rows: usize, cols: usize, f: impl Fn(usize, usize) -> f32) -> Links {
        Links::Dense(WeightMatrix {
            rows,
            cols,
            values: Arc::new((0..rows * cols).map(|i| f(i / cols, i % cols)).collect()),
        })
    }

    fn sparse(rows: usize, cols: usize, links: &[(u32, u32, f32)]) -> Links {
        Links::Sparse(SparseLinks {
            rows,
            cols,
            from: Arc::new(links.iter().map(|l| l.0).collect()),
            to: Arc::new(links.iter().map(|l| l.1).collect()),
            weight: Arc::new(links.iter().map(|l| l.2).collect()),
        })
    }

    #[test]
    fn both_forms_report_their_shape_and_size() {
        let d = dense(3, 4, |_, _| 1.0);
        assert_eq!((d.rows(), d.cols(), d.len()), (3, 4, 12));
        let s = sparse(100_000, 200_000, &[(0, 1, 0.5), (5, 7, -2.0)]);
        assert_eq!((s.rows(), s.cols(), s.len()), (100_000, 200_000, 2));
        assert_eq!(s.abs_max(), 2.0);
    }

    #[test]
    fn for_each_skips_links_that_leave_the_layers_and_ragged_lists_are_cut() {
        let s = Links::Sparse(SparseLinks {
            rows: 2,
            cols: 2,
            from: Arc::new(vec![0, 5, 1]),
            to: Arc::new(vec![1, 0, 0]),
            weight: Arc::new(vec![1.0, 2.0]),
        });
        assert_eq!(s.len(), 2, "the shortest of the three lists");
        let mut seen = Vec::new();
        s.for_each(|i, j, w| seen.push((i, j, w)));
        assert_eq!(
            seen,
            vec![(0, 1, 1.0)],
            "the second link starts outside the layer"
        );
    }

    #[test]
    fn strongest_links_agree_between_dense_and_sparse() {
        let w = |i: usize, j: usize| ((i * 7 + j * 3) % 5) as f32 - 2.0;
        let d = dense(6, 6, w);
        let mut list = Vec::new();
        for i in 0..6 {
            for j in 0..6 {
                if w(i, j) != 0.0 {
                    list.push((i as u32, j as u32, w(i, j)));
                }
            }
        }
        let s = sparse(6, 6, &list);
        for i in 0..6 {
            let key = |v: Vec<(usize, f32)>| {
                v.into_iter()
                    .map(|(_, w)| w.abs() as i32)
                    .collect::<Vec<_>>()
            };
            assert_eq!(
                key(d.strongest_from(i, 3)),
                key(s.strongest_from(i, 3)),
                "row {i}"
            );
            assert_eq!(
                key(d.strongest_into(i, 3)),
                key(s.strongest_into(i, 3)),
                "col {i}"
            );
        }
        assert!(d.strongest_from(9, 3).is_empty() && s.strongest_into(9, 3).is_empty());
    }

    #[test]
    fn strongest_drops_non_finite_and_orders_by_size() {
        let s = sparse(
            1,
            4,
            &[(0, 0, 0.1), (0, 1, f32::NAN), (0, 2, -0.9), (0, 3, 0.5)],
        );
        assert_eq!(s.strongest_from(0, 2), vec![(2, -0.9), (3, 0.5)]);
    }

    #[test]
    fn small_dense_fills_missing_links_with_zero_and_refuses_big_ones() {
        let s = sparse(3, 2, &[(0, 1, 0.5), (2, 0, -1.0)]);
        assert_eq!(
            s.small_dense(4, 100).unwrap(),
            vec![0.0, 0.5, 0.0, 0.0, -1.0, 0.0]
        );
        assert!(s.small_dense(1, 100).is_none(), "too many columns");
        assert!(s.small_dense(4, 5).is_none(), "too many cells");
        assert!(
            sparse(usize::MAX, 4, &[])
                .small_dense(4, usize::MAX)
                .is_none(),
            "no overflow"
        );
    }

    #[test]
    fn regions_list_their_members_and_ignore_what_does_not_fit() {
        let r = Regions {
            names: Arc::new(vec!["optic".into(), "motor".into()]),
            labels: Arc::new(vec![0, 1, 1, 7, 0, 1]),
        };
        assert_eq!(
            r.members(5),
            vec![vec![0, 4], vec![1, 2]],
            "label 7 has no name; neuron 5 is beyond the layer"
        );
    }
}
