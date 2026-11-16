//! Writing weights.
//!
//! Tensors are written in a stream: each one goes straight to the output, and only a small
//! buffer is kept in memory. The index and footer are written at [`StoreWriter::finish`]. For files there is
//! [`write_store_file`] - writing to a temporary file, `fsync` and atomic renaming: on disk
//! it is always either the old whole file or the new whole file, never half.

use std::collections::{HashMap, HashSet};
use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};
use std::time::{SystemTime, UNIX_EPOCH};

use crate::compress::{
    ChunkEncoder, Compression, DEFAULT_CHUNK_BYTES, Encoding, MAX_CHUNK_BYTES, MIN_CHUNK_BYTES,
    MIN_COMPRESS_BYTES,
};
use crate::crc::{Crc32c, crc32c};
use crate::dtype::{DType, Element};
use crate::error::{StoreError, StoreResult};
use crate::format::{
    FOOTER_LEN, FORMAT_MAJOR, FORMAT_MINOR, Footer, HEADER_LEN, Header, encode_index, fingerprint,
};
use crate::graph::Graph;
use crate::half::{f32_to_bf16_bits, f32_to_f16_bits};
use crate::tensor::{MAX_RANK, StatsAcc, TensorInfo, TensorRole};
use crate::value::{Meta, Value};

const CHUNK_BYTES: usize = 1 << 16;

/// What to do with NaN and ±Inf in float tensors.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NonFinitePolicy {
    /// Refuse to write (default): NaN in weights almost always means divergent
    /// training, and it’s good if the checkpoint doesn’t hide it. Checked already **after** casting
    /// type, so f16 overflow (> 65504) is also caught.
    Reject,
    /// Record as is; the number of such values will be included in the statistics.
    Allow,
}

/// Recording options.
#[derive(Debug, Clone)]
pub struct WriteOptions {
    /// Alignment of the beginning of each tensor, power of two 8..=65536 (default 64 -
    /// AVX-512 cache line and width; with mmap the data is aligned and in memory).
    pub alignment: u32,
    /// The policy of non-finite values.
    pub non_finite: NonFinitePolicy,
    /// Whether to calculate statistics (min/max/zeros/norm) of float tensors.
    pub compute_stats: bool,
    /// Creation time, ms since Unix era. `None` - current; `Some(0)` yields byte-for-byte
    /// reproducible files.
    pub created_unix_ms: Option<u64>,
    /// Compression of tensor data (disabled by default: tensors are read without copying).
    /// Can be changed between tensors via [`StoreWriter::set_compression`].
    pub compression: Compression,
    /// Compression chunk size, bytes: power of two from 4 KiB to 16 MiB. The smaller the chunk, the
    /// It’s cheaper to read individual rows of the compressed tensor; the more, the better the compression.
    pub compress_chunk_bytes: u32,
}

impl Default for WriteOptions {
    fn default() -> Self {
        Self {
            alignment: 64,
            non_finite: NonFinitePolicy::Reject,
            compute_stats: true,
            created_unix_ms: None,
            compression: Compression::None,
            compress_chunk_bytes: DEFAULT_CHUNK_BYTES,
        }
    }
}

impl WriteOptions {
    fn validate(&self) -> StoreResult<()> {
        if !self.alignment.is_power_of_two() || !(8..=65536).contains(&self.alignment) {
            return Err(StoreError::Invalid(format!(
                "alignment {} must be a power of two from 8 to 65536",
                self.alignment
            )));
        }
        let c = self.compress_chunk_bytes;
        if !c.is_power_of_two() || !(MIN_CHUNK_BYTES..=MAX_CHUNK_BYTES).contains(&c) {
            return Err(StoreError::Invalid(format!(
                "compression chunk size {c} must be a power of two from {MIN_CHUNK_BYTES} to {MAX_CHUNK_BYTES}"
            )));
        }
        Ok(())
    }
}

/// Summary of the recording.
#[derive(Debug, Clone)]
pub struct WriteSummary {
    /// Records of all tensors.
    pub tensors: Vec<TensorInfo>,
    /// Number of tensors.
    pub tensor_count: usize,
    /// Byte of tensor data (without alignment, unpacked).
    pub data_bytes: u64,
    /// Byte of tensor data in the file (without alignment, taking into account compression).
    pub stored_bytes: u64,
    /// File size.
    pub file_bytes: u64,
    /// Fingerprint of a set of tensors.
    pub fingerprint: u64,
}

pub(crate) fn validate_name(kind: &str, name: &str) -> StoreResult<()> {
    if name.is_empty() || name.len() > 4096 || name.contains('\0') {
        return Err(StoreError::InvalidName(format!(
            "{kind} '{name}': name is empty, longer than 4096 bytes, or contains NUL"
        )));
    }
    Ok(())
}

struct Pending {
    name: String,
    dtype: DType,
    shape: Vec<u64>,
    role: TensorRole,
    offset: u64,
    expected: u64,
    written: u64,
    crc: Crc32c,
    stats: Option<StatsAcc>,
    /// Compression encoder; `None` - the tensor is written as is.
    enc: Option<ChunkEncoder>,
}

/// Streaming storage writer.
pub struct StoreWriter<W: Write> {
    out: W,
    pos: u64,
    opts: WriteOptions,
    tensors: Vec<TensorInfo>,
    names: HashSet<String>,
    meta: Meta,
    graph: Option<Graph>,
    scratch: Vec<u8>,
}

impl<W: Write> StoreWriter<W> {
    /// Starts storage: writes the title.
    pub fn new(mut out: W, opts: WriteOptions) -> StoreResult<Self> {
        opts.validate()?;
        let created = opts.created_unix_ms.unwrap_or_else(now_ms);
        let header = Header {
            major: FORMAT_MAJOR,
            minor: FORMAT_MINOR,
            required_flags: 0,
            optional_flags: 0,
            alignment: opts.alignment,
            created_unix_ms: created,
        };
        out.write_all(&header.encode())?;
        Ok(Self {
            out,
            pos: HEADER_LEN,
            opts,
            tensors: Vec::new(),
            names: HashSet::new(),
            meta: Meta::new(),
            graph: None,
            scratch: Vec::new(),
        })
    }

    /// How many bytes have already been written.
    pub fn position(&self) -> u64 {
        self.pos
    }

    /// How many tensors have already been written.
    pub fn tensor_count(&self) -> usize {
        self.tensors.len()
    }

    /// Changes the non-finite value policy for the following tensors.
    pub fn set_non_finite_policy(&mut self, p: NonFinitePolicy) {
        self.opts.non_finite = p;
    }

    /// Changes the compression policy for the following tensors (for example, do not compress masks and states,
    /// which are read via mmap, and compress the rest).
    pub fn set_compression(&mut self, c: Compression) {
        self.opts.compression = c;
    }

    /// Specifies file metadata.
    pub fn set_meta(&mut self, key: &str, v: impl Into<Value>) -> StoreResult<()> {
        validate_name("metadata key", key)?;
        self.meta.insert(key.to_string(), v.into());
        Ok(())
    }

    /// Specifies the network graph. The structure is checked immediately, compliance with the weights is checked in `finish`.
    pub fn set_graph(&mut self, g: Graph) -> StoreResult<()> {
        g.validate_structure()?;
        self.graph = Some(g);
        Ok(())
    }

    /// Sets an attribute of an already written tensor.
    pub fn set_tensor_attr(
        &mut self,
        tensor: &str,
        key: &str,
        v: impl Into<Value>,
    ) -> StoreResult<()> {
        validate_name("attribute key", key)?;
        match self.tensors.iter_mut().find(|t| t.name == tensor) {
            Some(t) => {
                t.attrs.insert(key.to_string(), v.into());
                Ok(())
            }
            None => Err(StoreError::TensorNotFound(tensor.to_string())),
        }
    }

    fn pad_to(&mut self, align: u64) -> StoreResult<()> {
        let rem = self.pos % align;
        if rem != 0 {
            let n = (align - rem) as usize;
            self.out.write_all(&vec![0u8; n])?;
            self.pos += n as u64;
        }
        Ok(())
    }

    fn begin(
        &mut self,
        name: &str,
        dtype: DType,
        shape: &[u64],
        role: TensorRole,
    ) -> StoreResult<Pending> {
        validate_name("tensor", name)?;
        if self.names.contains(name) {
            return Err(StoreError::Duplicate(name.to_string()));
        }
        if shape.len() > MAX_RANK {
            return Err(StoreError::Invalid(format!(
                "tensor “{name}”: rank {} is greater than {MAX_RANK}",
                shape.len()
            )));
        }
        let numel = shape
            .iter()
            .try_fold(1u64, |a, d| a.checked_mul(*d))
            .ok_or_else(|| {
                StoreError::Invalid(format!("tensor '{name}': number of elements overflows u64"))
            })?;
        let expected = dtype.nbytes(numel).ok_or_else(|| {
            StoreError::Invalid(format!(
                "tensor “{name}”: {numel} elements of type {dtype} do not form an integer number of blocks"
            ))
        })?;
        self.pad_to(self.opts.alignment as u64)?;
        let stats = (self.opts.compute_stats || self.opts.non_finite == NonFinitePolicy::Reject)
            .then_some(dtype)
            .filter(|d| d.is_float())
            .map(StatsAcc::new);
        let enc = if self.opts.compression != Compression::None && expected >= MIN_COMPRESS_BYTES {
            // byte swapping makes sense for elements wider than a byte
            let es = dtype.element_bytes().filter(|e| *e > 1).unwrap_or(1) as usize;
            Some(ChunkEncoder::new(
                self.opts.compression,
                self.opts.compress_chunk_bytes as usize,
                es,
            ))
        } else {
            None
        };
        Ok(Pending {
            name: name.to_string(),
            dtype,
            shape: shape.to_vec(),
            role,
            offset: self.pos,
            expected,
            written: 0,
            crc: Crc32c::new(),
            stats,
            enc,
        })
    }

    /// `bytes` for float types is an integer number of elements.
    fn feed(&mut self, p: &mut Pending, bytes: &[u8]) -> StoreResult<()> {
        if p.written + bytes.len() as u64 > p.expected {
            return Err(StoreError::Invalid(format!(
                "tensor “{}”: there is more data than the shape {:?} ({} B) specifies",
                p.name, p.shape, p.expected
            )));
        }
        if let Some(acc) = p.stats.as_mut() {
            acc.feed(bytes);
            if self.opts.non_finite == NonFinitePolicy::Reject
                && let Some(index) = acc.first_non_finite()
            {
                return Err(StoreError::NonFinite {
                    name: p.name.clone(),
                    index,
                });
            }
        }
        match p.enc.as_mut() {
            Some(enc) => {
                enc.push(bytes, &mut self.out)?;
                self.pos = p.offset + enc.stored();
            }
            None => {
                self.out.write_all(bytes)?;
                self.pos += bytes.len() as u64;
            }
        }
        p.crc.update(bytes);
        p.written += bytes.len() as u64;
        Ok(())
    }

    fn end(&mut self, mut p: Pending) -> StoreResult<()> {
        if p.written != p.expected {
            return Err(StoreError::Invalid(format!(
                "tensor “{}”: {} B is written from {} B, which is specified by the shape {:?}",
                p.name, p.written, p.expected, p.shape
            )));
        }
        let (encoding, stored_bytes) = match p.enc.as_mut() {
            Some(enc) => {
                let e = enc.finish(&mut self.out)?;
                self.pos = p.offset + enc.stored();
                if e.is_compressed() {
                    (e, enc.stored())
                } else {
                    (Encoding::Raw, p.expected)
                }
            }
            None => (Encoding::Raw, p.expected),
        };
        let stats = if self.opts.compute_stats {
            p.stats.map(|a| a.finish())
        } else {
            None
        };
        self.names.insert(p.name.clone());
        self.tensors.push(TensorInfo {
            name: p.name,
            dtype: p.dtype,
            shape: p.shape,
            role: p.role,
            offset: p.offset,
            nbytes: p.expected,
            encoding,
            stored_bytes,
            crc32c: p.crc.finish(),
            stats,
            attrs: Meta::new(),
        });
        Ok(())
    }

    /// Writes ready bytes of a tensor of any type (including quantized).
    pub fn add_raw(
        &mut self,
        name: &str,
        dtype: DType,
        shape: &[u64],
        role: TensorRole,
        bytes: &[u8],
    ) -> StoreResult<()> {
        let mut p = self.begin(name, dtype, shape, role)?;
        if bytes.len() as u64 != p.expected {
            return Err(StoreError::Invalid(format!(
                "tensor '{name}': passed {} B, shape {shape:?} and type {dtype} require {} B",
                bytes.len(),
                p.expected
            )));
        }
        self.feed(&mut p, bytes)?;
        self.end(p)
    }

    /// Writes a slice of elements `T` (type on disk is `T::DTYPE`).
    pub fn add_slice<T: Element>(
        &mut self,
        name: &str,
        shape: &[u64],
        role: TensorRole,
        data: &[T],
    ) -> StoreResult<()> {
        let mut p = self.begin(name, T::DTYPE, shape, role)?;
        if (data.len() * T::SIZE) as u64 != p.expected {
            return Err(StoreError::ShapeMismatch {
                name: name.to_string(),
                expected: vec![p.expected / T::SIZE as u64],
                actual: vec![data.len() as u64],
            });
        }
        let per = (CHUNK_BYTES / T::SIZE).max(1);
        let mut buf = std::mem::take(&mut self.scratch);
        for chunk in data.chunks(per) {
            buf.clear();
            for v in chunk {
                v.extend_le(&mut buf);
            }
            self.feed(&mut p, &buf)?;
        }
        self.scratch = buf;
        self.end(p)
    }

    /// `f32`-weights like f32.
    pub fn add_f32(
        &mut self,
        name: &str,
        shape: &[u64],
        role: TensorRole,
        data: &[f32],
    ) -> StoreResult<()> {
        self.add_slice::<f32>(name, shape, role, data)
    }

    /// `f32`-weights stored in a more compact or wider type (`F32`, `F16`, `BF16`, `F64`).
    pub fn add_f32_as(
        &mut self,
        name: &str,
        shape: &[u64],
        role: TensorRole,
        dtype: DType,
        data: &[f32],
    ) -> StoreResult<()> {
        self.add_f32_iter(name, shape, role, dtype, data.iter().copied())
    }

    /// The same from the iterator: the weights do not need to be collected in `Vec`.
    pub fn add_f32_iter<I: IntoIterator<Item = f32>>(
        &mut self,
        name: &str,
        shape: &[u64],
        role: TensorRole,
        dtype: DType,
        iter: I,
    ) -> StoreResult<()> {
        if !matches!(dtype, DType::F32 | DType::F16 | DType::BF16 | DType::F64) {
            return Err(StoreError::Invalid(format!(
                "tensor '{name}': from f32 can only be saved to f32/f16/bf16/f64, not to {dtype}"
            )));
        }
        let mut p = self.begin(name, dtype, shape, role)?;
        let elem = dtype.element_bytes().unwrap_or(4) as usize;
        let per = (CHUNK_BYTES / elem).max(1);
        let mut buf = std::mem::take(&mut self.scratch);
        let mut it = iter.into_iter();
        let mut total = 0u64;
        loop {
            buf.clear();
            let mut n = 0usize;
            while n < per {
                match it.next() {
                    Some(v) => {
                        match dtype {
                            DType::F32 => buf.extend_from_slice(&v.to_le_bytes()),
                            DType::F64 => buf.extend_from_slice(&(v as f64).to_le_bytes()),
                            DType::F16 => buf.extend_from_slice(&f32_to_f16_bits(v).to_le_bytes()),
                            _ => buf.extend_from_slice(&f32_to_bf16_bits(v).to_le_bytes()),
                        }
                        n += 1;
                    }
                    None => break,
                }
            }
            if n == 0 {
                break;
            }
            total += n as u64;
            self.feed(&mut p, &buf)?;
        }
        self.scratch = buf;
        if total * elem as u64 != p.expected {
            return Err(StoreError::ShapeMismatch {
                name: name.to_string(),
                expected: vec![p.expected / elem as u64],
                actual: vec![total],
            });
        }
        self.end(p)
    }

    /// Boolean tensor (bytes per element).
    pub fn add_bool(
        &mut self,
        name: &str,
        shape: &[u64],
        role: TensorRole,
        data: &[bool],
    ) -> StoreResult<()> {
        let bytes: Vec<u8> = data.iter().map(|b| *b as u8).collect();
        self.add_raw(name, DType::Bool, shape, role, &bytes)
    }

    /// Scalar f32 (shape `[]`).
    pub fn add_scalar_f32(&mut self, name: &str, role: TensorRole, v: f32) -> StoreResult<()> {
        self.add_f32(name, &[], role, &[v])
    }

    /// Completes the repository: checks the graph against the weights, writes the index and footer.
    pub fn finish(mut self) -> StoreResult<(W, WriteSummary)> {
        if let Some(g) = &self.graph {
            let shapes: HashMap<&str, Vec<u64>> = self
                .tensors
                .iter()
                .map(|t| (t.name.as_str(), t.logical_shape()))
                .collect();
            let lookup = |n: &str| shapes.get(n).cloned();
            g.infer_shapes(&lookup)?;
        }
        self.meta
            .entry("syn.writer".to_string())
            .or_insert_with(|| Value::Str(format!("syn {}", env!("CARGO_PKG_VERSION"))));

        let data_end = self.pos;
        self.pad_to(8)?;
        let index_offset = self.pos;
        let index = encode_index(&self.meta, &self.tensors, self.graph.as_ref())?;
        self.out.write_all(&index)?;
        let index_len = index.len() as u64;
        self.pos += index_len;
        let fp = fingerprint(&self.tensors);
        let file_len = self.pos + FOOTER_LEN;
        let footer = Footer {
            index_offset,
            index_len,
            index_crc: crc32c(&index),
            tensor_count: self.tensors.len() as u32,
            data_end,
            file_len,
            fingerprint: fp,
        };
        self.out.write_all(&footer.encode())?;
        self.out.flush()?;

        let data_bytes: u64 = self.tensors.iter().map(|t| t.nbytes).sum();
        let stored_bytes: u64 = self.tensors.iter().map(|t| t.stored_bytes).sum();
        let summary = WriteSummary {
            tensor_count: self.tensors.len(),
            tensors: self.tensors,
            data_bytes,
            stored_bytes,
            file_bytes: file_len,
            fingerprint: fp,
        };
        Ok((self.out, summary))
    }
}

impl StoreWriter<Vec<u8>> {
    /// Writer to memory (for tests and network transmission).
    pub fn in_memory(opts: WriteOptions) -> StoreResult<Self> {
        Self::new(Vec::new(), opts)
    }
}

fn now_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|d| d.as_millis() as u64)
        .unwrap_or(0)
}

pub(crate) fn tmp_path(dest: &Path) -> PathBuf {
    let mut s = dest.as_os_str().to_owned();
    s.push(".tmp");
    PathBuf::from(s)
}

/// Writes the file atomically: data → `path.tmp` → `fsync` → rename. If there is an error
/// the temporary file is deleted, and the previous file by `path` remains untouched.
///
/// ```ignore
/// write_store_file("model.syn", WriteOptions::default(), |w| {
///     w.set_meta("model.name", "mnist-snn")?;
///     w.add_f32("fc1.weight", &[128, 784], TensorRole::Weight, &weights)?;
///     Ok(())
/// })?;
/// ```
pub fn write_store_file<F>(
    path: impl AsRef<Path>,
    opts: WriteOptions,
    f: F,
) -> StoreResult<WriteSummary>
where
    F: FnOnce(&mut StoreWriter<BufWriter<File>>) -> StoreResult<()>,
{
    let dest = path.as_ref();
    if let Some(parent) = dest.parent()
        && !parent.as_os_str().is_empty()
    {
        fs::create_dir_all(parent)?;
    }
    let tmp = tmp_path(dest);
    let result = (|| -> StoreResult<WriteSummary> {
        let file = File::create(&tmp)?;
        let mut w = StoreWriter::new(BufWriter::with_capacity(1 << 20, file), opts)?;
        f(&mut w)?;
        let (bw, summary) = w.finish()?;
        let file = bw
            .into_inner()
            .map_err(|e| StoreError::Io(e.into_error()))?;
        file.sync_all()?;
        Ok(summary)
    })();
    match result {
        Ok(summary) => {
            if let Err(e) = fs::rename(&tmp, dest) {
                let _ = fs::remove_file(&tmp);
                return Err(e.into());
            }
            Ok(summary)
        }
        Err(e) => {
            let _ = fs::remove_file(&tmp);
            Err(e)
        }
    }
}
