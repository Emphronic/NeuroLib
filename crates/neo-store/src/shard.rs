//! Sharding large models into several manifest files.
//!
//! Each shard is a full-fledged independent storage (opens [`Store`]). Manifesto
//! `{stem}.synm` stores a list of shards (name, size, fingerprint), model metadata and graph, so
//! substitution, truncation or loss of a shard is detected upon opening. The tensor lies entirely in one
//! Chardé; a new shard starts when the next tensor does not fit into the limit.

use std::collections::{HashMap, HashSet};
use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::{Path, PathBuf};

use crate::codec::{R, W, fmt_err, r_meta, w_meta};
use crate::crc::crc32c;
use crate::dtype::{DType, Element};
use crate::error::{StoreError, StoreResult};
use crate::format::{FORMAT_MAJOR, FORMAT_MINOR};
use crate::graph::{Graph, r_graph, w_graph};
use crate::reader::{ReadOptions, Store, VerifyReport};
use crate::tensor::{TensorInfo, TensorRole};
use crate::value::{Meta, Value};
use crate::writer::{StoreWriter, WriteOptions, tmp_path, validate_name};

const MANIFEST_MAGIC: [u8; 4] = *b"SYNM";

/// An entry about the shard in the manifest.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ShardEntry {
    /// The file name next to the manifest.
    pub file_name: String,
    /// File size.
    pub file_len: u64,
    /// A fingerprint of a set of shard tensors.
    pub fingerprint: u64,
    /// Number of tensors.
    pub tensor_count: u32,
}

/// The result of recording shards.
#[derive(Debug, Clone)]
pub struct ShardedSummary {
    /// Manifest path.
    pub manifest_path: PathBuf,
    /// Shards.
    pub shards: Vec<ShardEntry>,
    /// Total tensors.
    pub tensor_count: usize,
    /// Total size of shards.
    pub total_bytes: u64,
}

type FileWriter = StoreWriter<BufWriter<File>>;

struct CurShard {
    writer: FileWriter,
    tmp: PathBuf,
    dest: PathBuf,
    file_name: String,
}

/// Shared storage writer.
pub struct ShardedWriter {
    dir: PathBuf,
    stem: String,
    max_shard_bytes: u64,
    opts: WriteOptions,
    cur: Option<CurShard>,
    shards: Vec<ShardEntry>,
    meta: Meta,
    graph: Option<Graph>,
    seen: HashSet<String>,
    all: Vec<TensorInfo>,
}

impl ShardedWriter {
    /// The writer creates: `{stem}-00001.syn, …` shards and `{stem}.synm` manifest in `dir`.
    pub fn new(
        dir: impl AsRef<Path>,
        stem: &str,
        max_shard_bytes: u64,
        opts: WriteOptions,
    ) -> StoreResult<Self> {
        validate_name("set name", stem)?;
        if stem.contains(['/', '\\']) {
            return Err(StoreError::InvalidName(format!(
                "'{stem}': set name must not contain path separators"
            )));
        }
        if max_shard_bytes == 0 {
            return Err(StoreError::Invalid(
                "shard limit must be greater than zero".into(),
            ));
        }
        fs::create_dir_all(dir.as_ref())?;
        Ok(Self {
            dir: dir.as_ref().to_path_buf(),
            stem: stem.to_string(),
            max_shard_bytes,
            opts,
            cur: None,
            shards: Vec::new(),
            meta: Meta::new(),
            graph: None,
            seen: HashSet::new(),
            all: Vec::new(),
        })
    }

    /// Model metadata (ends up in the manifest).
    pub fn set_meta(&mut self, key: &str, v: impl Into<Value>) -> StoreResult<()> {
        validate_name("metadata key", key)?;
        self.meta.insert(key.to_string(), v.into());
        Ok(())
    }

    /// Network graph (checked against the weights of all shards in `finish`).
    pub fn set_graph(&mut self, g: Graph) -> StoreResult<()> {
        g.validate_structure()?;
        self.graph = Some(g);
        Ok(())
    }

    /// Returns the writer of the current shard, starting a new one if `nbytes` no longer fits.
    /// The entire API [`StoreWriter`] (including `add_csr`) is available through it; all tensors of one
    /// calls must fit into `nbytes`.
    pub fn writer_for(&mut self, nbytes: u64) -> StoreResult<&mut FileWriter> {
        let need_roll = match &self.cur {
            Some(c) => {
                c.writer.tensor_count() > 0
                    && c.writer.position().saturating_add(nbytes) > self.max_shard_bytes
            }
            None => false,
        };
        if need_roll {
            self.finish_current()?;
        }
        if self.cur.is_none() {
            self.open_next()?;
        }
        match self.cur.as_mut() {
            Some(c) => Ok(&mut c.writer),
            None => Err(StoreError::Invalid(
                "internal error: no current shard".into(),
            )),
        }
    }

    fn open_next(&mut self) -> StoreResult<()> {
        let idx = self.shards.len() + 1;
        let file_name = format!("{}-{:05}.syn", self.stem, idx);
        let dest = self.dir.join(&file_name);
        let tmp = tmp_path(&dest);
        let file = File::create(&tmp)?;
        let mut writer =
            StoreWriter::new(BufWriter::with_capacity(1 << 20, file), self.opts.clone())?;
        writer.set_meta("shard.index", idx as u64)?;
        self.cur = Some(CurShard {
            writer,
            tmp,
            dest,
            file_name,
        });
        Ok(())
    }

    fn finish_current(&mut self) -> StoreResult<()> {
        let Some(c) = self.cur.take() else {
            return Ok(());
        };
        let CurShard {
            writer,
            tmp,
            dest,
            file_name,
        } = c;
        let res = (|| -> StoreResult<crate::writer::WriteSummary> {
            let (bw, summary) = writer.finish()?;
            let file = bw
                .into_inner()
                .map_err(|e| StoreError::Io(e.into_error()))?;
            file.sync_all()?;
            Ok(summary)
        })();
        let summary = match res {
            Ok(s) => s,
            Err(e) => {
                let _ = fs::remove_file(&tmp);
                return Err(e);
            }
        };
        for t in &summary.tensors {
            if !self.seen.insert(t.name.clone()) {
                let _ = fs::remove_file(&tmp);
                return Err(StoreError::Duplicate(t.name.clone()));
            }
        }
        if let Err(e) = fs::rename(&tmp, &dest) {
            let _ = fs::remove_file(&tmp);
            return Err(e.into());
        }
        self.all.extend(summary.tensors.iter().cloned());
        self.shards.push(ShardEntry {
            file_name,
            file_len: summary.file_bytes,
            fingerprint: summary.fingerprint,
            tensor_count: summary.tensor_count as u32,
        });
        Ok(())
    }

    /// Writes a slice of `T` elements to the appropriate shard.
    pub fn add_slice<T: Element>(
        &mut self,
        name: &str,
        shape: &[u64],
        role: TensorRole,
        data: &[T],
    ) -> StoreResult<()> {
        let nbytes = (data.len() * T::SIZE) as u64;
        self.writer_for(nbytes)?
            .add_slice::<T>(name, shape, role, data)
    }

    /// Writes f32 weights to f32.
    pub fn add_f32(
        &mut self,
        name: &str,
        shape: &[u64],
        role: TensorRole,
        data: &[f32],
    ) -> StoreResult<()> {
        self.add_slice::<f32>(name, shape, role, data)
    }

    /// Writes f32 weights to the selected float type.
    pub fn add_f32_as(
        &mut self,
        name: &str,
        shape: &[u64],
        role: TensorRole,
        dtype: DType,
        data: &[f32],
    ) -> StoreResult<()> {
        let nbytes = dtype.element_bytes().unwrap_or(4) * data.len() as u64;
        self.writer_for(nbytes)?
            .add_f32_as(name, shape, role, dtype, data)
    }

    /// Writes ready bytes (including quantized ones).
    pub fn add_raw(
        &mut self,
        name: &str,
        dtype: DType,
        shape: &[u64],
        role: TensorRole,
        bytes: &[u8],
    ) -> StoreResult<()> {
        self.writer_for(bytes.len() as u64)?
            .add_raw(name, dtype, shape, role, bytes)
    }

    /// Completes the recording: closes the shard, checks the graph, atomically writes the manifest.
    pub fn finish(mut self) -> StoreResult<ShardedSummary> {
        self.finish_current()?;
        if self.shards.is_empty() {
            return Err(StoreError::Invalid(
                "there is not a single tensor in the set".into(),
            ));
        }
        if let Some(g) = &self.graph {
            let shapes: HashMap<&str, Vec<u64>> = self
                .all
                .iter()
                .map(|t| (t.name.as_str(), t.logical_shape()))
                .collect();
            let lookup = |n: &str| shapes.get(n).cloned();
            g.infer_shapes(&lookup)?;
        }
        let mut w = W::new();
        w.u16(FORMAT_MAJOR);
        w.u16(FORMAT_MINOR);
        w.len(self.shards.len());
        for s in &self.shards {
            w.str(&s.file_name);
            w.u64(s.file_len);
            w.u64(s.fingerprint);
            w.u32(s.tensor_count);
        }
        w_meta(&mut w, &self.meta);
        match &self.graph {
            Some(g) => {
                w.bool(true);
                w_graph(&mut w, g);
            }
            None => w.bool(false),
        }
        let body = w.finish()?;
        let mut file_bytes = MANIFEST_MAGIC.to_vec();
        file_bytes.extend_from_slice(&body);
        let crc = crc32c(&file_bytes);
        file_bytes.extend_from_slice(&crc.to_le_bytes());

        let manifest_path = self.dir.join(format!("{}.synm", self.stem));
        let tmp = tmp_path(&manifest_path);
        let written = (|| -> StoreResult<()> {
            let mut f = File::create(&tmp)?;
            f.write_all(&file_bytes)?;
            f.sync_all()?;
            Ok(())
        })();
        if let Err(e) = written {
            let _ = fs::remove_file(&tmp);
            return Err(e);
        }
        if let Err(e) = fs::rename(&tmp, &manifest_path) {
            let _ = fs::remove_file(&tmp);
            return Err(e.into());
        }
        let total_bytes: u64 = self.shards.iter().map(|s| s.file_len).sum();
        let tensor_count = self.all.len();
        // ShardedWriter implements Drop, so fields cannot be moved - we take them using take.
        let shards = std::mem::take(&mut self.shards);
        Ok(ShardedSummary {
            manifest_path,
            shards,
            tensor_count,
            total_bytes,
        })
    }
}

impl Drop for ShardedWriter {
    fn drop(&mut self) {
        // An incomplete shard should not leave a temporary file.
        if let Some(c) = self.cur.take() {
            let tmp = c.tmp.clone();
            drop(c);
            let _ = fs::remove_file(tmp);
        }
    }
}

/// Open sharded set.
pub struct ShardedStore {
    shards: Vec<Store>,
    entries: Vec<ShardEntry>,
    index: HashMap<String, usize>,
    meta: Meta,
    graph: Option<Graph>,
}

impl ShardedStore {
    /// Opens a set based on manifest `*.synm`.
    pub fn open(manifest: impl AsRef<Path>) -> StoreResult<ShardedStore> {
        Self::open_with(manifest, ReadOptions::default())
    }

    /// Opens a set with parameters for reading shards.
    pub fn open_with(manifest: impl AsRef<Path>, opts: ReadOptions) -> StoreResult<ShardedStore> {
        let path = manifest.as_ref();
        let dir = path.parent().map(|p| p.to_path_buf()).unwrap_or_default();
        let bytes = fs::read(path)?;
        if bytes.len() < 8 || bytes[..4] != MANIFEST_MAGIC {
            return Err(fmt_err("this is not a syn manifest (.synm)"));
        }
        let (body, tail) = bytes.split_at(bytes.len() - 4);
        let stored = u32::from_le_bytes([tail[0], tail[1], tail[2], tail[3]]);
        let actual = crc32c(body);
        if stored != actual {
            return Err(StoreError::Checksum {
                what: "manifesto".into(),
                expected: stored,
                actual,
            });
        }
        let mut r = R::new(&body[4..]);
        let major = r.u16()?;
        let minor = r.u16()?;
        if major != FORMAT_MAJOR {
            return Err(StoreError::UnsupportedVersion { major, minor });
        }
        let n = r.count(24)?;
        let mut entries = Vec::with_capacity(n);
        for _ in 0..n {
            entries.push(ShardEntry {
                file_name: r.str()?,
                file_len: r.u64()?,
                fingerprint: r.u64()?,
                tensor_count: r.u32()?,
            });
        }
        let meta = r_meta(&mut r)?;
        let graph = if r.bool()? {
            Some(r_graph(&mut r, 0)?)
        } else {
            None
        };
        if !r.is_empty() {
            return Err(fmt_err("there are extra bytes in the manifest"));
        }
        let mut shards = Vec::with_capacity(n);
        let mut index = HashMap::new();
        for (si, e) in entries.iter().enumerate() {
            let plain = Path::new(&e.file_name).file_name().and_then(|f| f.to_str());
            if plain != Some(e.file_name.as_str()) {
                return Err(fmt_err(format!("invalid shard name '{}'", e.file_name)));
            }
            let store = Store::open_with(dir.join(&e.file_name), opts.clone())?;
            if store.file_len() != e.file_len || store.fingerprint() != e.fingerprint {
                return Err(fmt_err(format!(
                    "shard “{}” does not match the manifest: replaced or changed",
                    e.file_name
                )));
            }
            for t in store.tensors() {
                if index.insert(t.name.clone(), si).is_some() {
                    return Err(fmt_err(format!(
                        "The tensor “{}” is present in several shards",
                        t.name
                    )));
                }
            }
            shards.push(store);
        }
        Ok(ShardedStore {
            shards,
            entries,
            index,
            meta,
            graph,
        })
    }

    /// Model metadata.
    pub fn meta(&self) -> &Meta {
        &self.meta
    }

    /// Network graph.
    pub fn graph(&self) -> Option<&Graph> {
        self.graph.as_ref()
    }

    /// Shards according to the manifest.
    pub fn entries(&self) -> &[ShardEntry] {
        &self.entries
    }

    /// Names of all tensors in alphabetical order.
    pub fn names(&self) -> Vec<&str> {
        let mut v: Vec<&str> = self.index.keys().map(|s| s.as_str()).collect();
        v.sort_unstable();
        v
    }

    /// The shard in which the tensor lies; All methods [`Store`] are available through it.
    pub fn store_for(&self, name: &str) -> StoreResult<&Store> {
        self.index
            .get(name)
            .map(|&i| &self.shards[i])
            .ok_or_else(|| StoreError::TensorNotFound(name.to_string()))
    }

    /// Tensor notation.
    pub fn info(&self, name: &str) -> StoreResult<&TensorInfo> {
        self.store_for(name)?.info(name)
    }

    /// Reads a float tensor as `Vec<f32>`.
    pub fn read_f32(&self, name: &str) -> StoreResult<Vec<f32>> {
        self.store_for(name)?.read_f32(name)
    }

    /// Checks the CRC of all tensors of all shards.
    pub fn verify(&self) -> StoreResult<VerifyReport> {
        let mut total = VerifyReport::default();
        for s in &self.shards {
            let r = s.verify()?;
            total.tensors_checked += r.tensors_checked;
            total.bytes_checked += r.bytes_checked;
            total.failures.extend(r.failures);
        }
        Ok(total)
    }

    /// Checks the graph against the weights of all shards.
    pub fn validate_graph(&self) -> StoreResult<Option<crate::graph::GraphShapes>> {
        match &self.graph {
            None => Ok(None),
            Some(g) => {
                let lookup = |n: &str| self.info(n).ok().map(|t| t.logical_shape());
                g.infer_shapes(&lookup).map(Some)
            }
        }
    }
}
