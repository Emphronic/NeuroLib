//! Unpacking quantized tensors (feature `quantization`).
//!
//! GGUF blocks are in the file as is and are read as bytes ([`Store::read_bytes`]); the
//! unpacking into `f32` lives in `neo-quantization`. This module only wraps the read
//! bytes in [`PackedBuffer`] of the desired format and calls `neo_quantization::dequantize`, so
//! the dequantization formulas exist in exactly one place.
//!
//! N-bit tensors [`DType::Bits`] do not fall here: their scales are separate
//! tensors, and [`Store::read_bits`] returns the whole codes.

use neo_quantization::dtype::DType as QType;
use neo_quantization::pack::{PackedBuffer, gguf_raw_to_typed};

use crate::codec::fmt_err;
use crate::dtype::{DType, GgufKind};
use crate::error::{StoreError, StoreResult};
use crate::reader::Store;

fn to_usize(v: u64) -> StoreResult<usize> {
    usize::try_from(v).map_err(|_| fmt_err("size does not fit in usize"))
}

/// Wraps raw GGUF blocks into [`PackedBuffer`] `kind` format via
/// `neo_quantization::pack::gguf_raw_to_typed`.
///
/// `blocks` should occupy exactly `numel / block_elems` blocks (tensors `neo-store` have a row
/// always consists of whole blocks); The block layout is checked against the `neo-quantization` table.
pub fn gguf_to_packed(kind: GgufKind, blocks: Vec<u8>, numel: usize) -> StoreResult<PackedBuffer> {
    let unsupported = || {
        StoreError::UnsupportedFeature(format!(
            "neo-quantization does not know how to unpack the {} format",
            kind.name()
        ))
    };
    let qd = QType::from_gguf_type_id(kind.gguf_id() as u32).ok_or_else(unsupported)?;
    if qd.block_elems() as u64 != kind.block_elems()
        || qd.block_bytes() as u64 != kind.block_bytes()
    {
        return Err(StoreError::UnsupportedFeature(format!(
            "the layout of the {} block in neo-store and neo-quantization is different",
            kind.name()
        )));
    }
    let n_blocks = (numel as u64) / kind.block_elems();
    if !(numel as u64).is_multiple_of(kind.block_elems())
        || blocks.len() as u64 != n_blocks * kind.block_bytes()
    {
        return Err(fmt_err(format!(
            "{} elements of format {} do not match {} B data",
            numel,
            kind.name(),
            blocks.len()
        )));
    }
    // Parsing blocks into a typed buffer lives in neo-quantization: they also know which
    // formats are implemented.
    gguf_raw_to_typed(blocks, qd, numel)
        .map_err(|m| StoreError::UnsupportedFeature(format!("{}: {m}", kind.name())))
}

/// Dequantizes the buffer; panic inside `neo-quantization` on damaged blocks turns into
/// error (data from the file is an untrusted input, especially with `verify_on_read = false`).
fn dequantize_guarded(name: &str, packed: &PackedBuffer) -> StoreResult<Vec<f32>> {
    let expected = packed.numel();
    let res = std::panic::catch_unwind(std::panic::AssertUnwindSafe(|| {
        neo_quantization::dequantize(packed)
    }));
    match res {
        Ok(v) if v.len() == expected => Ok(v),
        Ok(v) => Err(fmt_err(format!(
            "tensor '{name}': unpacked {} values instead of {expected}",
            v.len()
        ))),
        Err(_) => Err(fmt_err(format!(
            "tensor '{name}': neo-quantization could not unpack blocks (data corrupted)"
        ))),
    }
}

fn bits_error(name: &str) -> StoreError {
    StoreError::Invalid(format!(
        "tensor '{name}': N-bit tensors do not have their own scales; \
         entire codes are returned by Store::read_bits, the scales lie in neighboring tensors"
    ))
}

impl Store {
    /// Reads the tensor as [`PackedBuffer`]: block format GGUF - packed, float tensor -
    /// like `PackedBuffer::Float`.
    pub fn read_packed(&self, name: &str) -> StoreResult<PackedBuffer> {
        let info = self.info(name)?;
        match info.dtype {
            DType::Gguf(kind) => {
                let numel = to_usize(info.numel())?;
                let bytes = self.read_bytes(name)?.into_owned();
                gguf_to_packed(kind, bytes, numel)
            }
            d if d.is_float() => Ok(PackedBuffer::Float(self.read_f32(name)?)),
            other => Err(StoreError::DTypeMismatch {
                name: name.to_string(),
                expected: "float or block format GGUF".into(),
                actual: other.name(),
            }),
        }
    }

    /// Reads tensor as `Vec<f32>`: float types directly, block formats GGUF (`Q4_0`, `Q4_K`,
    /// `IQ2_XS`, `TQ2_0`, …) through `neo-quantization`.
    pub fn read_dequantized_f32(&self, name: &str) -> StoreResult<Vec<f32>> {
        let info = self.info(name)?;
        match info.dtype {
            DType::Gguf(_) => {
                let packed = self.read_packed(name)?;
                dequantize_guarded(name, &packed)
            }
            DType::Bits(_) => Err(bits_error(name)),
            _ => self.read_f32(name),
        }
    }

    /// Rows `first..first+rows` along the first axis as `Vec<f32>` without unpacking the entire tensor.
    /// For block formats, the row must consist of an integer number of blocks.
    pub fn read_rows_dequantized_f32(
        &self,
        name: &str,
        first: u64,
        rows: u64,
    ) -> StoreResult<Vec<f32>> {
        let info = self.info(name)?;
        match info.dtype {
            DType::Gguf(kind) => {
                let row_numel: u64 = info.shape.iter().skip(1).product();
                let numel = rows.checked_mul(row_numel).ok_or_else(|| {
                    StoreError::Invalid("number of elements overflows u64".into())
                })?;
                let bytes = self.read_rows_bytes(name, first, rows)?.into_owned();
                let packed = gguf_to_packed(kind, bytes, to_usize(numel)?)?;
                dequantize_guarded(name, &packed)
            }
            DType::Bits(_) => Err(bits_error(name)),
            _ => self.read_rows_f32(name, first, rows),
        }
    }
}
