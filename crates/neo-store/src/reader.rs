//! Storage read: mmap (zero copy), read from disk on demand or from memory.
//!
//! When opening, the header, footer, file length, index CRC, layout are always checked
//! tensors (alignment, boundaries, overlaps) and fingerprint. The tensor data CRC is checked when
//! each reading in its entirety ([`ReadOptions::verify_on_read`]) or at once through [`Store::verify`].

use std::borrow::Cow;
use std::collections::HashMap;
use std::fs::File;
use std::io::{Read, Seek, SeekFrom};
use std::path::Path;
use std::sync::Mutex;

use crate::codec::fmt_err;
use crate::compress::{ChunkMap, Encoding, parse_tail};
use crate::crc::{Crc32c, crc32c};
use crate::dtype::{DType, Element};
use crate::error::{StoreError, StoreResult};
use crate::format::{
    FOOTER_LEN, Footer, HEADER_LEN, Header, decode_index, fingerprint, validate_tensors,
};
use crate::graph::{Graph, GraphShapes};
use crate::half::{bf16_bits_to_f32, f16_bits_to_f32};
use crate::tensor::TensorInfo;
use crate::value::{Meta, Value};

/// Reading options.
#[derive(Debug, Clone)]
pub struct ReadOptions {
    /// Map the file into memory (if the `mmap` feature is enabled).
    pub use_mmap: bool,
    /// Check the tensor CRC on every full read.
    pub verify_on_read: bool,
    /// Check the CRC of all tensors immediately upon opening (reads the entire file).
    pub verify_on_open: bool,
}

impl Default for ReadOptions {
    fn default() -> Self {
        Self {
            use_mmap: cfg!(feature = "mmap"),
            verify_on_read: true,
            verify_on_open: false,
        }
    }
}

enum Backing {
    Mem(Vec<u8>),
    #[cfg(feature = "mmap")]
    Map(memmap2::Mmap),
    File(Mutex<File>, u64),
}

#[cfg(feature = "mmap")]
#[allow(unsafe_code)]
fn try_map(file: &File) -> Option<Backing> {
    // SAFETY: the file is read-only, and the container is unchangeable after writing (writing is always
    // goes to a new temporary file and replaces the target with renaming). Changing a file
    // by a third party process during reading - a general limitation of any mmap; consequences
    // limited by invalid data that CRC will catch, but not by Rust type safety
    // for readers working through `Cow<[u8]>`.
    unsafe { memmap2::Mmap::map(file) }.ok().map(Backing::Map)
}

#[cfg(not(feature = "mmap"))]
fn try_map(_file: &File) -> Option<Backing> {
    None
}

fn to_usize(v: u64) -> StoreResult<usize> {
    usize::try_from(v).map_err(|_| fmt_err("offset or size does not fit in usize"))
}

impl Backing {
    fn from_file(file: File, use_mmap: bool) -> StoreResult<Backing> {
        let len = file.metadata()?.len();
        if use_mmap
            && len > 0
            && let Some(b) = try_map(&file)
        {
            return Ok(b);
        }
        Ok(Backing::File(Mutex::new(file), len))
    }

    fn len(&self) -> u64 {
        match self {
            Backing::Mem(v) => v.len() as u64,
            #[cfg(feature = "mmap")]
            Backing::Map(m) => m.len() as u64,
            Backing::File(_, l) => *l,
        }
    }

    fn read(&self, off: u64, len: usize) -> StoreResult<Cow<'_, [u8]>> {
        let end = off
            .checked_add(len as u64)
            .filter(|e| *e <= self.len())
            .ok_or_else(|| fmt_err("read outside of file"))?;
        match self {
            Backing::Mem(v) => Ok(Cow::Borrowed(&v[to_usize(off)?..to_usize(end)?])),
            #[cfg(feature = "mmap")]
            Backing::Map(m) => Ok(Cow::Borrowed(&m[to_usize(off)?..to_usize(end)?])),
            Backing::File(f, _) => {
                let mut g = f.lock().unwrap_or_else(|e| e.into_inner());
                g.seek(SeekFrom::Start(off))?;
                let mut buf = vec![0u8; len];
                g.read_exact(&mut buf)?;
                Ok(Cow::Owned(buf))
            }
        }
    }
}

/// One tensor that failed CRC check.
#[derive(Debug, Clone)]
pub struct VerifyFailure {
    /// Tensor name.
    pub name: String,
    /// Recorded amount.
    pub expected: u32,
    /// The calculated amount (0 if the data could not even be unpacked - see `detail`).
    pub actual: u32,
    /// Explanation if the tensor fails a test other than sum (corrupt compression).
    pub detail: Option<String>,
}

/// Full inspection report.
#[derive(Debug, Clone, Default)]
pub struct VerifyReport {
    /// How many tensors have been tested?
    pub tensors_checked: usize,
    /// How many bytes of data were read.
    pub bytes_checked: u64,
    /// Damaged tensors.
    pub failures: Vec<VerifyFailure>,
}

impl VerifyReport {
    /// Are all the amounts added up?
    pub fn is_ok(&self) -> bool {
        self.failures.is_empty()
    }

    /// Turns the report into an error if there is damage.
    pub fn into_result(self) -> StoreResult<VerifyReport> {
        match self.failures.first() {
            None => Ok(self),
            Some(f) => Err(StoreError::Checksum {
                what: format!(
                    "tensor '{}' (total damaged: {})",
                    f.name,
                    self.failures.len()
                ),
                expected: f.expected,
                actual: f.actual,
            }),
        }
    }
}

/// Open storage.
pub struct Store {
    backing: Backing,
    header: Header,
    footer: Footer,
    meta: Meta,
    tensors: Vec<TensorInfo>,
    by_name: HashMap<String, usize>,
    graph: Option<Graph>,
    opts: ReadOptions,
}

impl Store {
    /// Opens a file with default parameters.
    pub fn open(path: impl AsRef<Path>) -> StoreResult<Store> {
        Self::open_with(path, ReadOptions::default())
    }

    /// Opens a file.
    pub fn open_with(path: impl AsRef<Path>, opts: ReadOptions) -> StoreResult<Store> {
        let file = File::open(path.as_ref())?;
        let backing = Backing::from_file(file, opts.use_mmap)?;
        Self::from_backing(backing, opts)
    }

    /// Opens a store of bytes in memory.
    pub fn from_bytes(bytes: Vec<u8>) -> StoreResult<Store> {
        Self::from_bytes_with(bytes, ReadOptions::default())
    }

    /// The same with the parameters.
    pub fn from_bytes_with(bytes: Vec<u8>, opts: ReadOptions) -> StoreResult<Store> {
        Self::from_backing(Backing::Mem(bytes), opts)
    }

    fn from_backing(backing: Backing, opts: ReadOptions) -> StoreResult<Store> {
        let len = backing.len();
        if len < HEADER_LEN + FOOTER_LEN {
            return Err(fmt_err("file shorter than minimum size"));
        }
        let header = Header::decode(&backing.read(0, HEADER_LEN as usize)?)?;
        let footer = Footer::decode(&backing.read(len - FOOTER_LEN, FOOTER_LEN as usize)?)?;
        if footer.file_len != len {
            return Err(fmt_err(format!(
                "file length {len} B does not match the recorded {} B: the file is truncated or appended",
                footer.file_len
            )));
        }
        let index_end = footer
            .index_offset
            .checked_add(footer.index_len)
            .filter(|e| *e <= len - FOOTER_LEN)
            .ok_or_else(|| fmt_err("index goes out of file"))?;
        if footer.index_offset < HEADER_LEN
            || footer.data_end < HEADER_LEN
            || footer.data_end > footer.index_offset
        {
            return Err(fmt_err("incorrect file area boundaries"));
        }
        let _ = index_end;
        let (meta, tensors, graph) = {
            let index = backing.read(footer.index_offset, to_usize(footer.index_len)?)?;
            let actual = crc32c(&index);
            if actual != footer.index_crc {
                return Err(StoreError::Checksum {
                    what: "index".into(),
                    expected: footer.index_crc,
                    actual,
                });
            }
            decode_index(&index)?
        };
        if tensors.len() as u64 != footer.tensor_count as u64 {
            return Err(fmt_err(
                "the number of tensors in the index does not match the footer",
            ));
        }
        validate_tensors(&tensors, header.alignment, footer.data_end)?;
        if fingerprint(&tensors) != footer.fingerprint {
            return Err(fmt_err("tensor set fingerprint does not match footer"));
        }
        let by_name = tensors
            .iter()
            .enumerate()
            .map(|(i, t)| (t.name.clone(), i))
            .collect();
        let store = Store {
            backing,
            header,
            footer,
            meta,
            tensors,
            by_name,
            graph,
            opts,
        };
        if store.opts.verify_on_open {
            store.verify()?.into_result()?;
        }
        Ok(store)
    }

    /// File format version `(major, minor)`.
    pub fn version(&self) -> (u16, u16) {
        (self.header.major, self.header.minor)
    }

    /// Creation time (ms since Unix epoch; 0 - not recorded).
    pub fn created_unix_ms(&self) -> u64 {
        self.header.created_unix_ms
    }

    /// File size in bytes.
    pub fn file_len(&self) -> u64 {
        self.backing.len()
    }

    /// Fingerprint of a set of tensors.
    pub fn fingerprint(&self) -> u64 {
        self.footer.fingerprint
    }

    /// File metadata.
    pub fn meta(&self) -> &Meta {
        &self.meta
    }

    /// Metadata value by key.
    pub fn meta_get(&self, key: &str) -> Option<&Value> {
        self.meta.get(key)
    }

    /// Network graph, if recorded.
    pub fn graph(&self) -> Option<&Graph> {
        self.graph.as_ref()
    }

    /// Write all tensors in the order they were written.
    pub fn tensors(&self) -> &[TensorInfo] {
        &self.tensors
    }

    /// Is there a tensor?
    pub fn contains(&self, name: &str) -> bool {
        self.by_name.contains_key(name)
    }

    /// Tensor notation.
    pub fn info(&self, name: &str) -> StoreResult<&TensorInfo> {
        self.by_name
            .get(name)
            .map(|&i| &self.tensors[i])
            .ok_or_else(|| StoreError::TensorNotFound(name.to_string()))
    }

    /// Total model parameters (weights, biases, neuron parameters, delays).
    pub fn parameter_count(&self) -> u64 {
        self.tensors
            .iter()
            .filter(|t| t.role.is_parameter())
            .map(|t| t.logical_shape().iter().product::<u64>())
            .sum()
    }

    /// Tests the graph against the weights of this store and returns the shapes of the nodes.
    pub fn validate_graph(&self) -> StoreResult<Option<GraphShapes>> {
        match &self.graph {
            None => Ok(None),
            Some(g) => {
                let lookup = |n: &str| {
                    self.by_name
                        .get(n)
                        .map(|&i| self.tensors[i].logical_shape())
                };
                g.infer_shapes(&lookup).map(Some)
            }
        }
    }

    /// Tensor bytes without copying (mmap/memory) or read from disk; compressed tensor
    /// unpacked into a new buffer. CRC (from uncompressed data) is checked if enabled
    /// `verify_on_read`.
    pub fn read_bytes(&self, name: &str) -> StoreResult<Cow<'_, [u8]>> {
        let info = self.info(name)?;
        let bytes = if info.encoding.is_compressed() {
            Cow::Owned(self.read_decoded(info, 0, info.nbytes)?)
        } else {
            self.backing.read(info.offset, to_usize(info.nbytes)?)?
        };
        if self.opts.verify_on_read {
            let actual = crc32c(&bytes);
            if actual != info.crc32c {
                return Err(StoreError::Checksum {
                    what: format!("tensor '{name}'"),
                    expected: info.crc32c,
                    actual,
                });
            }
        }
        Ok(bytes)
    }

    /// Reads the tensor as `Vec<T>`; the type on the disk must match `T::DTYPE`.
    pub fn read_vec<T: Element>(&self, name: &str) -> StoreResult<Vec<T>> {
        let info = self.info(name)?;
        if info.dtype != T::DTYPE {
            return Err(StoreError::DTypeMismatch {
                name: name.to_string(),
                expected: T::DTYPE.name(),
                actual: info.dtype.name(),
            });
        }
        let b = self.read_bytes(name)?;
        // `T::SIZE` is an associated constant, `as_chunks::<{ T::SIZE }>` is not available on stable.
        #[allow(unknown_lints, clippy::chunks_exact_to_as_chunks)]
        let values = b.chunks_exact(T::SIZE).map(T::read_le).collect();
        Ok(values)
    }

    /// Reads the float tensor (f32/f16/bf16/f64) as `Vec<f32>`.
    pub fn read_f32(&self, name: &str) -> StoreResult<Vec<f32>> {
        let info = self.info(name)?;
        let mut out = vec![0f32; to_usize(info.numel())?];
        self.read_f32_into(name, &mut out)?;
        Ok(out)
    }

    /// Reads a float tensor into a ready buffer (length must be equal to the number of elements).
    pub fn read_f32_into(&self, name: &str, out: &mut [f32]) -> StoreResult<()> {
        let info = self.info(name)?;
        if out.len() as u64 != info.numel() {
            return Err(StoreError::ShapeMismatch {
                name: name.to_string(),
                expected: vec![info.numel()],
                actual: vec![out.len() as u64],
            });
        }
        let dtype = info.dtype;
        let b = self.read_bytes(name)?;
        decode_floats(name, dtype, &b, out)
    }

    /// Reads the N-bit tensor (`DType::Bits(n)`) as whole `0..2ⁿ` codes, one per `u8`.
    /// Scales and displacements lie in neighboring tensors and are not included here.
    pub fn read_bits(&self, name: &str) -> StoreResult<Vec<u8>> {
        let info = self.info(name)?;
        let bits = match info.dtype {
            DType::Bits(n) => n,
            other => {
                return Err(StoreError::DTypeMismatch {
                    name: name.to_string(),
                    expected: "bits1..bits7".into(),
                    actual: other.name(),
                });
            }
        };
        let numel = to_usize(info.numel())?;
        let bytes = self.read_bytes(name)?;
        crate::bits::unpack_bits(&bytes, bits, numel)
    }

    /// Reads a float tensor as `Vec<f64>`.
    pub fn read_f64(&self, name: &str) -> StoreResult<Vec<f64>> {
        if self.info(name)?.dtype == DType::F64 {
            return self.read_vec::<f64>(name);
        }
        Ok(self.read_f32(name)?.into_iter().map(|v| v as f64).collect())
    }

    /// Unpacks compressed tensor chunks covering `raw_start..raw_end` (in unpacked
    /// bytes), and sends them one by one to `f(chunk offset in the tensor, raw bytes of the chunk)`.
    /// One continuous section of the file is read: from the first required chunk to the last.
    fn for_each_chunk(
        &self,
        info: &TensorInfo,
        raw_start: u64,
        raw_end: u64,
        mut f: impl FnMut(u64, &[u8]),
    ) -> StoreResult<()> {
        fn corrupt(name: &str, m: String) -> StoreError {
            fmt_err(format!("tensor '{name}': corrupted compressed data: {m}"))
        }
        let name = info.name.as_str();
        if raw_start > raw_end || raw_end > info.nbytes {
            return Err(StoreError::Invalid(format!(
                "tensor “{name}”: range {raw_start}..{raw_end} goes beyond {} B",
                info.nbytes
            )));
        }
        let stored = info.stored_bytes;
        // validate_tensors when opened guarantees stored >= 8 for compressed tensors
        let tail = self.backing.read(info.offset + (stored - 8), 8)?;
        let (chunk, count) =
            parse_tail(&tail, stored, info.nbytes).map_err(|m| corrupt(name, m))?;
        let table_len = count * 4;
        let table = self
            .backing
            .read(info.offset + (stored - 8 - table_len as u64), table_len)?;
        let map = ChunkMap::parse(chunk, count, &table, stored, info.nbytes)
            .map_err(|m| corrupt(name, m))?;
        if raw_start == raw_end {
            return Ok(());
        }
        let first = to_usize(raw_start / chunk as u64)?;
        let last = to_usize((raw_end - 1) / chunk as u64)?;
        let (body_start, body_end) = map.body_range(first, last);
        let body = self
            .backing
            .read(info.offset + body_start, to_usize(body_end - body_start)?)?;
        let es = match info.encoding {
            Encoding::ShuffleLz => info.dtype.element_bytes().unwrap_or(1) as usize,
            _ => 1,
        };
        let mut scratch = Vec::new();
        let mut raw = Vec::with_capacity(map.chunk_bytes());
        for i in first..=last {
            let sp = map.span(i);
            let rel = to_usize(sp.off - body_start)?;
            let slice = &body[rel..rel + sp.len as usize];
            raw.clear();
            map.decode_chunk(i, slice, es, &mut scratch, &mut raw)
                .map_err(|m| corrupt(name, format!("chunk {i}: {m}")))?;
            f(i as u64 * chunk as u64, &raw);
        }
        Ok(())
    }

    /// Unpacked `raw_start..raw_end` bytes of the compressed tensor.
    fn read_decoded(
        &self,
        info: &TensorInfo,
        raw_start: u64,
        raw_end: u64,
    ) -> StoreResult<Vec<u8>> {
        if raw_start > raw_end || raw_end > info.nbytes {
            return Err(StoreError::Invalid(format!(
                "tensor “{}”: range {raw_start}..{raw_end} goes beyond {} B",
                info.name, info.nbytes
            )));
        }
        let mut out = Vec::with_capacity(to_usize(raw_end - raw_start)?);
        self.for_each_chunk(info, raw_start, raw_end, |base, data| {
            let lo = raw_start.max(base) - base;
            let hi = raw_end.min(base + data.len() as u64) - base;
            out.extend_from_slice(&data[lo as usize..hi as usize]);
        })?;
        Ok(out)
    }

    /// `(offset from start of tensor data, length)` in **unpacked** bytes.
    fn rows_layout(&self, name: &str, first: u64, rows: u64) -> StoreResult<(u64, u64)> {
        let info = self.info(name)?;
        let total_rows = *info
            .shape
            .first()
            .ok_or_else(|| StoreError::Invalid(format!("tensor “{name}” is a scalar, no rows")))?;
        let end = first
            .checked_add(rows)
            .filter(|e| *e <= total_rows)
            .ok_or_else(|| {
                StoreError::Invalid(format!(
                    "the rows {first}..{first}+{rows} go beyond the {total_rows} rows of the tensor “{name}”"
                ))
            })?;
        let _ = end;
        let row_numel: u64 = info.shape[1..].iter().product();
        let row_bytes = info.dtype.nbytes(row_numel).ok_or_else(|| {
            StoreError::Invalid(format!(
                "tensor '{name}': the row does not form an integer number of blocks"
            ))
        })?;
        if matches!(info.dtype, DType::Bits(_))
            && !(row_numel * bits_of(info.dtype)).is_multiple_of(8)
        {
            return Err(StoreError::Invalid(format!(
                "tensor '{name}': the row does not occupy an integer number of bytes"
            )));
        }
        Ok((first * row_bytes, rows * row_bytes))
    }

    /// Bytes of rows `first..first+rows` along the first axis without reading the entire tensor (for a compressed
    /// tensor, only the necessary chunks are unpacked). The tensor CRC is not
    /// checked during partial reading (for this there is [`Store::verify`]).
    pub fn read_rows_bytes(&self, name: &str, first: u64, rows: u64) -> StoreResult<Cow<'_, [u8]>> {
        let (off, len) = self.rows_layout(name, first, rows)?;
        let info = self.info(name)?;
        if info.encoding.is_compressed() {
            // Only chunks containing the requested rows are unpacked
            return Ok(Cow::Owned(self.read_decoded(info, off, off + len)?));
        }
        self.backing.read(info.offset + off, to_usize(len)?)
    }

    /// Float tensor rows as `Vec<f32>` (for example, part of the embedding table).
    pub fn read_rows_f32(&self, name: &str, first: u64, rows: u64) -> StoreResult<Vec<f32>> {
        let info = self.info(name)?;
        let row_numel: u64 = info.shape.iter().skip(1).product();
        let dtype = info.dtype;
        let b = self.read_rows_bytes(name, first, rows)?;
        let mut out = vec![0f32; to_usize(rows * row_numel)?];
        decode_floats(name, dtype, &b, &mut out)?;
        Ok(out)
    }

    /// Checks the CRC of all tensors, reading data in 4 MiB chunks.
    pub fn verify(&self) -> StoreResult<VerifyReport> {
        let mut report = VerifyReport::default();
        for t in &self.tensors {
            let mut crc = Crc32c::new();
            if t.encoding.is_compressed() {
                // chunk by chunk: the entire tensor in memory is not needed
                match self.for_each_chunk(t, 0, t.nbytes, |_, data| crc.update(data)) {
                    Ok(()) => {}
                    Err(StoreError::Io(e)) => return Err(StoreError::Io(e)),
                    Err(e) => {
                        report.tensors_checked += 1;
                        report.bytes_checked += t.nbytes;
                        report.failures.push(VerifyFailure {
                            name: t.name.clone(),
                            expected: t.crc32c,
                            actual: 0,
                            detail: Some(e.to_string()),
                        });
                        continue;
                    }
                }
            } else {
                let (mut off, mut left) = (t.offset, t.nbytes);
                while left > 0 {
                    let n = left.min(4 << 20);
                    let chunk = self.backing.read(off, to_usize(n)?)?;
                    crc.update(&chunk);
                    off += n;
                    left -= n;
                }
            }
            report.tensors_checked += 1;
            report.bytes_checked += t.nbytes;
            let actual = crc.finish();
            if actual != t.crc32c {
                report.failures.push(VerifyFailure {
                    name: t.name.clone(),
                    expected: t.crc32c,
                    actual,
                    detail: None,
                });
            }
        }
        Ok(report)
    }

    /// Text summary: tensors, types, shapes, sizes, statistics.
    pub fn summary(&self) -> String {
        let mut s = String::new();
        let (ma, mi) = self.version();
        s.push_str(&format!(
            "syn v{ma}.{mi}, {} B, imprint {:016x}\n",
            self.file_len(),
            self.fingerprint()
        ));
        for (k, v) in &self.meta {
            s.push_str(&format!("  meta {k} = {v:?}\n"));
        }
        for t in &self.tensors {
            s.push_str(&format!(
                "  {:<40} {:<8} {:?} {} {} B",
                t.name,
                t.dtype.name(),
                t.shape,
                t.role.name(),
                t.nbytes
            ));
            if t.encoding.is_compressed() {
                s.push_str(&format!(
                    " → {} B in the file ({:.0} %, {})",
                    t.stored_bytes,
                    t.stored_ratio() * 100.0,
                    t.encoding.name()
                ));
            }
            if let Some(st) = &t.stats {
                s.push_str(&format!(
                    "  [min {:.4e}, max {:.4e}, zeros {}, ‖w‖₂ {:.4e}]",
                    st.min, st.max, st.zeros, st.l2
                ));
            }
            s.push('\n');
        }
        s.push_str(&format!(
            "  tensors: {}, parameters: {}\n",
            self.tensors.len(),
            self.parameter_count()
        ));
        if let Some(g) = &self.graph {
            s.push_str(&format!(
                "  graph: {} nodes, {} edges\n",
                g.nodes.len(),
                g.edges.len()
            ));
        }
        s
    }
}

fn bits_of(d: DType) -> u64 {
    match d {
        DType::Bits(n) => n as u64,
        _ => 0,
    }
}

fn decode_floats(name: &str, dtype: DType, bytes: &[u8], out: &mut [f32]) -> StoreResult<()> {
    match dtype {
        DType::F32 => {
            for (o, c) in out.iter_mut().zip(bytes.as_chunks::<4>().0) {
                *o = f32::from_le_bytes([c[0], c[1], c[2], c[3]]);
            }
        }
        DType::F64 => {
            for (o, c) in out.iter_mut().zip(bytes.as_chunks::<8>().0) {
                *o = f64::from_le_bytes([c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7]]) as f32;
            }
        }
        DType::F16 => {
            for (o, c) in out.iter_mut().zip(bytes.as_chunks::<2>().0) {
                *o = f16_bits_to_f32(u16::from_le_bytes([c[0], c[1]]));
            }
        }
        DType::BF16 => {
            for (o, c) in out.iter_mut().zip(bytes.as_chunks::<2>().0) {
                *o = bf16_bits_to_f32(u16::from_le_bytes([c[0], c[1]]));
            }
        }
        other => {
            return Err(StoreError::DTypeMismatch {
                name: name.to_string(),
                expected: "f32/f16/bf16/f64".into(),
                actual: format!(
                    "{other}{}",
                    if other.is_quantized() {
                        " (quantized type: Store::read_bits or, with the `quantization` feature, \
                         Store::read_dequantized_f32)"
                    } else {
                        ""
                    }
                ),
            });
        }
    }
    Ok(())
}
