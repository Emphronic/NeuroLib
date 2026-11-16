//! Sparse weight matrices (CSR) for thinned and structurally plastic networks.
//!
//! The matrix `name` is stored by three ordinary tensors, so everything else is in storage
//! (CRC, mmap, sharding) works without changes:
//!
//! * `{name}.indptr` — `u64`, `rows + 1` elements;
//! * `{name}.indices` — `u32`, `nnz` column numbers (within a row in ascending order);
//! * `{name}.values` — `f32`/`f16`/`bf16`, `nnz` values; carries the attributes `sparse = "csr"`,
//!   `sparse.shape = [rows, cols]`.
//!
//! A graph node references a sparse weight with the name `{name}.values`; to check shapes
//! the logical shape `[rows, cols]` from the attribute is used.

use std::io::Write;

use crate::dtype::DType;
use crate::error::{StoreError, StoreResult};
use crate::reader::Store;
use crate::tensor::TensorRole;
use crate::value::Value;
use crate::writer::StoreWriter;

/// Matrix in CSR format.
#[derive(Debug, Clone, PartialEq)]
pub struct SparseCsr {
    /// Number of rows.
    pub rows: u64,
    /// Number of columns.
    pub cols: u64,
    /// Row boundaries in `indices`/`values`, length `rows + 1`.
    pub indptr: Vec<u64>,
    /// Column numbers.
    pub indices: Vec<u32>,
    /// Meanings.
    pub values: Vec<f32>,
}

impl SparseCsr {
    /// Number of non-zeros.
    pub fn nnz(&self) -> usize {
        self.values.len()
    }

    /// Checks the consistency of the structure.
    pub fn validate(&self) -> StoreResult<()> {
        let bad = |m: String| Err(StoreError::Invalid(format!("CSR: {m}")));
        if self.cols > u32::MAX as u64 + 1 {
            return bad(format!(
                "There are more {} columns than can fit in u32",
                self.cols
            ));
        }
        if self.indptr.len() as u64 != self.rows + 1 {
            return bad(format!(
                "indptr has {} elements, expected {}",
                self.indptr.len(),
                self.rows + 1
            ));
        }
        if self.indices.len() != self.values.len() {
            return bad("indices and values are of different lengths".into());
        }
        if self.indptr[0] != 0 || self.indptr[self.indptr.len() - 1] != self.values.len() as u64 {
            return bad("indptr must start with 0 and end with nnz".into());
        }
        for r in 0..self.rows as usize {
            let (a, b) = (self.indptr[r], self.indptr[r + 1]);
            if a > b {
                return bad(format!("indptr decreases in row {r}"));
            }
            let row = &self.indices[a as usize..b as usize];
            for (k, c) in row.iter().enumerate() {
                if *c as u64 >= self.cols {
                    return bad(format!("column {c} is out of range in row {r}"));
                }
                if k > 0 && row[k - 1] >= *c {
                    return bad(format!("columns of row {r} are not strictly ascending"));
                }
            }
        }
        Ok(())
    }

    /// Constructs a CSR from a dense matrix, discarding values starting with `|v| <= threshold`.
    pub fn from_dense(rows: usize, cols: usize, data: &[f32], threshold: f32) -> StoreResult<Self> {
        if data.len() != rows * cols {
            return Err(StoreError::ShapeMismatch {
                name: "dense".into(),
                expected: vec![(rows * cols) as u64],
                actual: vec![data.len() as u64],
            });
        }
        let mut indptr = Vec::with_capacity(rows + 1);
        let (mut indices, mut values) = (Vec::new(), Vec::new());
        indptr.push(0u64);
        for r in 0..rows {
            for c in 0..cols {
                let v = data[r * cols + c];
                if v.abs() > threshold {
                    indices.push(c as u32);
                    values.push(v);
                }
            }
            indptr.push(values.len() as u64);
        }
        let m = SparseCsr {
            rows: rows as u64,
            cols: cols as u64,
            indptr,
            indices,
            values,
        };
        m.validate()?;
        Ok(m)
    }

    /// Constructs a CSR from triples `(row, column, value)`; duplicates are an error.
    pub fn from_triplets(
        rows: usize,
        cols: usize,
        mut t: Vec<(u32, u32, f32)>,
    ) -> StoreResult<Self> {
        t.sort_by_key(|x| (x.0, x.1));
        let mut indptr = vec![0u64; rows + 1];
        let (mut indices, mut values) = (Vec::with_capacity(t.len()), Vec::with_capacity(t.len()));
        let mut prev: Option<(u32, u32)> = None;
        for (r, c, v) in t {
            if r as usize >= rows || c as usize >= cols {
                return Err(StoreError::Invalid(format!(
                    "CSR: element ({r}, {c}) outside {rows}×{cols}"
                )));
            }
            if prev == Some((r, c)) {
                return Err(StoreError::Invalid(format!(
                    "CSR: element ({r}, {c}) specified twice"
                )));
            }
            prev = Some((r, c));
            indptr[r as usize + 1] += 1;
            indices.push(c);
            values.push(v);
        }
        for r in 0..rows {
            indptr[r + 1] += indptr[r];
        }
        let m = SparseCsr {
            rows: rows as u64,
            cols: cols as u64,
            indptr,
            indices,
            values,
        };
        m.validate()?;
        Ok(m)
    }

    /// Expands into a dense matrix row by row.
    pub fn to_dense(&self) -> Vec<f32> {
        let cols = self.cols as usize;
        let mut out = vec![0f32; self.rows as usize * cols];
        for r in 0..self.rows as usize {
            for k in self.indptr[r] as usize..self.indptr[r + 1] as usize {
                out[r * cols + self.indices[k] as usize] = self.values[k];
            }
        }
        out
    }
}

impl<W: Write> StoreWriter<W> {
    /// Writes a CSR matrix with three tensors `{name}.indptr/.indices/.values`.
    /// `values_dtype` - `F32`, `F16` or `BF16`.
    pub fn add_csr(
        &mut self,
        name: &str,
        m: &SparseCsr,
        role: TensorRole,
        values_dtype: DType,
    ) -> StoreResult<()> {
        m.validate()?;
        let nnz = m.nnz() as u64;
        self.add_slice::<u64>(
            &format!("{name}.indptr"),
            &[m.rows + 1],
            TensorRole::Buffer,
            &m.indptr,
        )?;
        self.add_slice::<u32>(
            &format!("{name}.indices"),
            &[nnz],
            TensorRole::Buffer,
            &m.indices,
        )?;
        let vname = format!("{name}.values");
        self.add_f32_as(&vname, &[nnz], role, values_dtype, &m.values)?;
        self.set_tensor_attr(&vname, "sparse", "csr")?;
        self.set_tensor_attr(&vname, "sparse.shape", Value::uints(&[m.rows, m.cols]))?;
        Ok(())
    }
}

impl Store {
    /// Reads the CSR matrix written by [`StoreWriter::add_csr`].
    pub fn read_csr(&self, name: &str) -> StoreResult<SparseCsr> {
        let vname = format!("{name}.values");
        let info = self.info(&vname)?;
        if info.attrs.get("sparse").and_then(|v| v.as_str()) != Some("csr") {
            return Err(StoreError::Invalid(format!("“{name}” is not a CSR matrix")));
        }
        let shape = info
            .attrs
            .get("sparse.shape")
            .and_then(|v| v.as_u64_list())
            .filter(|s| s.len() == 2)
            .ok_or_else(|| StoreError::Invalid(format!("'{name}': no sparse.shape attribute")))?;
        let m = SparseCsr {
            rows: shape[0],
            cols: shape[1],
            indptr: self.read_vec::<u64>(&format!("{name}.indptr"))?,
            indices: self.read_vec::<u32>(&format!("{name}.indices"))?,
            values: self.read_f32(&vname)?,
        };
        m.validate()?;
        Ok(m)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dense_roundtrip() {
        let d = [0.0, 1.0, 0.0, 2.0, 0.0, 3.0];
        let m = SparseCsr::from_dense(2, 3, &d, 0.0).unwrap();
        assert_eq!(m.nnz(), 3);
        assert_eq!(m.indptr, vec![0, 1, 3]);
        assert_eq!(m.to_dense(), d.to_vec());
    }

    #[test]
    fn triplets_and_validation() {
        let m = SparseCsr::from_triplets(2, 2, vec![(1, 1, 5.0), (0, 0, 1.0)]).unwrap();
        assert_eq!(m.to_dense(), vec![1.0, 0.0, 0.0, 5.0]);
        assert!(SparseCsr::from_triplets(2, 2, vec![(0, 0, 1.0), (0, 0, 2.0)]).is_err());
        let mut bad = m.clone();
        bad.indices[0] = 9;
        assert!(bad.validate().is_err());
    }
}
