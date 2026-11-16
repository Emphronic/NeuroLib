//! Compression of tensor data.
//!
//! Only `std`: its own block LZ codec in LZ4 format (sequences “token, literals,
//! offset, match length", 64 KiB window) and optional byte swap
//! (byte-shuffle) in front of it. Permutation collects bytes of the same name from neighboring elements
//! in a row (all endian bytes, then all mantissas...), so f32/f16/bf16-weights are compressed
//! noticeably better. Decimated, ternary and quantized data are also well compressed by LZ itself.
//!
//! ## Compressed tensor layout
//!
//! The tensor data is cut into chunks of `chunk` raw bytes (the last one is shorter); every chunk
//! is encoded independently, so the tensor rows are read without unpacking the rest:
//!
//! ```text
//! ┌────────────┬────────────┬─────┬──────────────────┬───────────────┬──────────┐
//! │ chunk body │ chunk body │ … │ chunk lengths │ chunk: u32 │ n: u32 │
//! │ 0 │ 1 │ │ n × u32 (LE) │ raw bytes │ chunks │
//! └────────────┴────────────┴─────┴──────────────────┴───────────────┴──────────┘
//! ```
//!
//! The table of lengths is **after** the bodies, because the writer is a stream writer and does not know how to return
//! back. The most significant bit of the length (`0x8000_0000`) means “the chunk is stored as is”: incompressible
//! a chunk takes exactly the same amount as in an uncompressed tensor. Tensor size in file
//! (`stored_bytes`) is written in the index, so the trailer is read from the end.
//!
//! The checksum of the tensor in the index is calculated based on the **unpacked** data: it does not depend
//! on the storage method, and the file fingerprint of a compressed and uncompressed record of the same weights
//! matches. Any damaged chunk will either not be unpacked or will not match the CRC.

use std::io::Write;

use crate::error::{StoreError, StoreResult};

/// Default chunk size, bytes.
pub const DEFAULT_CHUNK_BYTES: u32 = 1 << 18;
/// The smallest allowable chunk size.
pub(crate) const MIN_CHUNK_BYTES: u32 = 1 << 12;
/// The largest allowable chunk size (chunk lengths fit into 31 bits).
pub(crate) const MAX_CHUNK_BYTES: u32 = 1 << 24;
/// In short, tensors do not compress: the gain does not cover the trailer.
pub(crate) const MIN_COMPRESS_BYTES: u64 = 64;

const RAW_FLAG: u32 = 0x8000_0000;
const TRAILER_FIXED: u64 = 8;

/// Write compression policy.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Default)]
pub enum Compression {
    /// Do not compress (default): tensors are read without copying via mmap.
    #[default]
    None,
    /// Compress the tensor only if the first chunk is compressed no worse than 90%. Otherwise tensor
    /// remains uncompressed and is still readable without copying.
    Auto,
    /// Compress each tensor from 64 bytes; incompressible chunks are stored as is.
    Always,
}

/// Encoding of the tensor data on disk (bytes in the index entry).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Encoding {
    /// As it is.
    Raw,
    /// Block LZ by chunks.
    Lz,
    /// Rearranging bytes by element size, then block LZ by chunks.
    ShuffleLz,
}

impl Encoding {
    /// Code on disk.
    pub fn code(self) -> u8 {
        match self {
            Encoding::Raw => 0,
            Encoding::Lz => 1,
            Encoding::ShuffleLz => 2,
        }
    }

    /// Encoding by code; `None` for unknown.
    pub fn from_code(c: u8) -> Option<Encoding> {
        match c {
            0 => Some(Encoding::Raw),
            1 => Some(Encoding::Lz),
            2 => Some(Encoding::ShuffleLz),
            _ => None,
        }
    }

    /// Encoding name.
    pub fn name(self) -> &'static str {
        match self {
            Encoding::Raw => "raw",
            Encoding::Lz => "lz",
            Encoding::ShuffleLz => "shuffle+lz",
        }
    }

    /// Is the data stored in compressed shape?
    pub fn is_compressed(self) -> bool {
        self != Encoding::Raw
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Byte swap
// ═══════════════════════════════════════════════════════════════════════════

/// `dst[b * n + i] = src[i * es + b]`, where `n = src.len() / es`. The length of `src` is a multiple of `es`.
pub(crate) fn shuffle_bytes(src: &[u8], es: usize, dst: &mut Vec<u8>) {
    debug_assert!(es > 0 && src.len().is_multiple_of(es));
    let n = src.len() / es;
    dst.clear();
    dst.resize(src.len(), 0);
    for b in 0..es {
        let plane = &mut dst[b * n..(b + 1) * n];
        for (i, o) in plane.iter_mut().enumerate() {
            *o = src[i * es + b];
        }
    }
}

/// Inverse to [`shuffle_bytes`]: `dst.len() == src.len()`, length is a multiple of `es`.
pub(crate) fn unshuffle_bytes(src: &[u8], es: usize, dst: &mut [u8]) {
    debug_assert!(es > 0 && src.len().is_multiple_of(es) && dst.len() == src.len());
    let n = src.len() / es;
    for b in 0..es {
        let plane = &src[b * n..(b + 1) * n];
        for (i, v) in plane.iter().enumerate() {
            dst[i * es + b] = *v;
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Block LZ (LZ4 block format)
// ═══════════════════════════════════════════════════════════════════════════

const MIN_MATCH: usize = 4;
/// The match cannot begin closer than this number of bytes to the end of the block.
const MF_LIMIT: usize = 12;
/// The last bytes of a block are always literals.
const LAST_LITERALS: usize = 5;
const HASH_LOG: u32 = 14;
const MAX_OFFSET: usize = 65_535;

#[inline]
fn read_u32(s: &[u8], i: usize) -> u32 {
    u32::from_le_bytes([s[i], s[i + 1], s[i + 2], s[i + 3]])
}

#[inline]
fn hash4(v: u32) -> usize {
    (v.wrapping_mul(2_654_435_761) >> (32 - HASH_LOG)) as usize
}

fn put_len(out: &mut Vec<u8>, mut n: usize) {
    while n >= 255 {
        out.push(255);
        n -= 255;
    }
    out.push(n as u8);
}

fn emit_sequence(out: &mut Vec<u8>, literals: &[u8], offset: usize, mlen: usize) {
    let lit = literals.len();
    let ml = mlen - MIN_MATCH;
    out.push(((lit.min(15) as u8) << 4) | (ml.min(15) as u8));
    if lit >= 15 {
        put_len(out, lit - 15);
    }
    out.extend_from_slice(literals);
    out.extend_from_slice(&(offset as u16).to_le_bytes());
    if ml >= 15 {
        put_len(out, ml - 15);
    }
}

fn emit_last(out: &mut Vec<u8>, literals: &[u8]) {
    let lit = literals.len();
    out.push((lit.min(15) as u8) << 4);
    if lit >= 15 {
        put_len(out, lit - 15);
    }
    out.extend_from_slice(literals);
}

/// Compression state (hash table is reused between chunks).
pub(crate) struct Lz {
    table: Vec<u32>,
}

impl Lz {
    pub fn new() -> Self {
        Self {
            table: vec![u32::MAX; 1 << HASH_LOG],
        }
    }

    /// Appends a compressed block for `src` (`src.len()` < 2³²) to `out`.
    pub fn compress(&mut self, src: &[u8], out: &mut Vec<u8>) {
        let n = src.len();
        let mut anchor = 0usize;
        if n > MF_LIMIT {
            self.table.fill(u32::MAX);
            let limit = n - MF_LIMIT;
            let match_end = n - LAST_LITERALS;
            let mut pos = 0usize;
            // “Acceleration”: the longer there are no matches, the larger the step (random mantissa).
            let mut skip = 1usize << 6;
            while pos < limit {
                let v = read_u32(src, pos);
                let h = hash4(v);
                let cand = self.table[h];
                self.table[h] = pos as u32;
                if cand != u32::MAX {
                    let c = cand as usize;
                    if pos - c <= MAX_OFFSET && read_u32(src, c) == v {
                        let mut mlen = MIN_MATCH;
                        while pos + mlen < match_end && src[c + mlen] == src[pos + mlen] {
                            mlen += 1;
                        }
                        emit_sequence(out, &src[anchor..pos], pos - c, mlen);
                        pos += mlen;
                        anchor = pos;
                        skip = 1usize << 6;
                        continue;
                    }
                }
                pos += skip >> 6;
                skip += 1;
            }
        }
        emit_last(out, &src[anchor..]);
    }
}

/// Unpacks the `src` block and adds exactly `dst_len` bytes to `out`. Any discrepancy
/// (break, out of bounds, incorrect offset, incorrect final size) - error; panic and
/// reads outside of buffers are excluded.
pub(crate) fn lz_decompress(
    src: &[u8],
    dst_len: usize,
    out: &mut Vec<u8>,
) -> Result<(), &'static str> {
    let start = out.len();
    let end_len = start
        .checked_add(dst_len)
        .ok_or("unboxing size exceeds usize")?;
    out.reserve(dst_len);
    let n = src.len();
    let mut i = 0usize;
    loop {
        if i >= n {
            return Err("block broken: no token");
        }
        let token = src[i];
        i += 1;

        let mut lit = (token >> 4) as usize;
        if lit == 15 {
            loop {
                if i >= n {
                    return Err("block broken: length of literals");
                }
                let b = src[i];
                i += 1;
                lit = lit
                    .checked_add(b as usize)
                    .ok_or("length of literals overflows usize")?;
                if b != 255 {
                    break;
                }
            }
        }
        if lit > n - i || lit > end_len - out.len() {
            return Err("literals cross block boundaries");
        }
        out.extend_from_slice(&src[i..i + lit]);
        i += lit;
        if i == n {
            break; // the last sequence is literals only
        }

        if n - i < 2 {
            return Err("block broken: no offset");
        }
        let offset = src[i] as usize | ((src[i + 1] as usize) << 8);
        i += 2;
        if offset == 0 || offset > out.len() - start {
            return Err("invalid match offset");
        }

        let mut mlen = (token & 15) as usize;
        if mlen == 15 {
            loop {
                if i >= n {
                    return Err("block torn: match length");
                }
                let b = src[i];
                i += 1;
                mlen = mlen
                    .checked_add(b as usize)
                    .ok_or("match length overflows usize")?;
                if b != 255 {
                    break;
                }
            }
        }
        mlen = mlen
            .checked_add(MIN_MATCH)
            .ok_or("match length overflows usize")?;
        if mlen > end_len - out.len() {
            return Err("the match goes beyond the chunk boundaries");
        }

        let from = out.len() - offset;
        if offset >= mlen {
            out.extend_from_within(from..from + mlen);
        } else {
            // overlapping match (short pattern repeat): copy byte by byte
            for k in 0..mlen {
                let b = out[from + k];
                out.push(b);
            }
        }
    }
    if out.len() != end_len {
        return Err("the size of the unpacked data is not the expected size");
    }
    Ok(())
}

// ═══════════════════════════════════════════════════════════════════════════
// Writing: chunk encoder
// ═══════════════════════════════════════════════════════════════════════════

/// Streaming single tensor encoder: accumulates raw bytes to chunk size, encodes a chunk and
/// immediately hands it to the output. The length table and trailer are written in [`ChunkEncoder::finish`].
pub(crate) struct ChunkEncoder {
    mode: Compression,
    chunk: usize,
    /// The size of the element for byte swapping (1 - swapping is impossible).
    es: usize,
    /// Selected encoding; `None` until the first chunk is processed.
    encoding: Option<Encoding>,
    /// `Auto` chose not to compress: all bytes are left as is.
    passthrough: bool,
    buf: Vec<u8>,
    lens: Vec<u32>,
    lz: Lz,
    shuffled: Vec<u8>,
    packed: Vec<u8>,
    stored: u64,
}

impl ChunkEncoder {
    /// `chunk` is the power of two in `MIN_CHUNK_BYTES..=MAX_CHUNK_BYTES`, `es` is the size of the element.
    pub fn new(mode: Compression, chunk: usize, es: usize) -> Self {
        Self {
            mode,
            chunk,
            es: es.max(1),
            encoding: None,
            passthrough: false,
            buf: Vec::with_capacity(chunk),
            lens: Vec::new(),
            lz: Lz::new(),
            shuffled: Vec::new(),
            packed: Vec::new(),
            stored: 0,
        }
    }

    /// How many bytes have already been handed to the output (after `finish` - the size of the tensor in the file).
    pub fn stored(&self) -> u64 {
        self.stored
    }

    pub fn push<O: Write>(&mut self, mut bytes: &[u8], out: &mut O) -> StoreResult<()> {
        while !bytes.is_empty() {
            if self.passthrough {
                out.write_all(bytes)?;
                self.stored += bytes.len() as u64;
                return Ok(());
            }
            let take = (self.chunk - self.buf.len()).min(bytes.len());
            self.buf.extend_from_slice(&bytes[..take]);
            bytes = &bytes[take..];
            if self.buf.len() == self.chunk {
                self.flush(out)?;
            }
        }
        Ok(())
    }

    /// Finishes the last chunk and trailer; returns the final encoding. `Raw` means
    /// that the tensor is written as is (there is no trailer).
    pub fn finish<O: Write>(&mut self, out: &mut O) -> StoreResult<Encoding> {
        if !self.buf.is_empty() {
            self.flush(out)?;
        }
        let enc = match (self.passthrough, self.encoding) {
            (false, Some(e)) => e,
            _ => return Ok(Encoding::Raw),
        };
        let count = u32::try_from(self.lens.len())
            .map_err(|_| StoreError::Invalid("too many chunks in one tensor".into()))?;
        for l in &self.lens {
            out.write_all(&l.to_le_bytes())?;
        }
        out.write_all(&(self.chunk as u32).to_le_bytes())?;
        out.write_all(&count.to_le_bytes())?;
        self.stored += 4 * self.lens.len() as u64 + TRAILER_FIXED;
        Ok(enc)
    }

    fn flush<O: Write>(&mut self, out: &mut O) -> StoreResult<()> {
        let raw = std::mem::take(&mut self.buf);
        let res = self.flush_chunk(&raw, out);
        self.buf = raw;
        self.buf.clear();
        res
    }

    fn flush_chunk<O: Write>(&mut self, raw: &[u8], out: &mut O) -> StoreResult<()> {
        if self.encoding.is_none() && !self.passthrough {
            self.decide(raw);
        }
        if self.passthrough {
            out.write_all(raw)?;
            self.stored += raw.len() as u64;
            return Ok(());
        }
        self.packed.clear();
        if self.encoding == Some(Encoding::ShuffleLz) {
            shuffle_bytes(raw, self.es, &mut self.shuffled);
            self.lz.compress(&self.shuffled, &mut self.packed);
        } else {
            self.lz.compress(raw, &mut self.packed);
        }
        if self.packed.len() < raw.len() {
            out.write_all(&self.packed)?;
            self.lens.push(self.packed.len() as u32);
            self.stored += self.packed.len() as u64;
        } else {
            out.write_all(raw)?;
            self.lens.push(raw.len() as u32 | RAW_FLAG);
            self.stored += raw.len() as u64;
        }
        Ok(())
    }

    /// Based on the first chunk, it selects the encoding: it tries LZ with and without permutation.
    fn decide(&mut self, raw: &[u8]) {
        let mut best = Encoding::Lz;
        self.packed.clear();
        self.lz.compress(raw, &mut self.packed);
        let mut best_len = self.packed.len();
        if self.es > 1 && raw.len().is_multiple_of(self.es) {
            shuffle_bytes(raw, self.es, &mut self.shuffled);
            self.packed.clear();
            self.lz.compress(&self.shuffled, &mut self.packed);
            if self.packed.len() < best_len {
                best = Encoding::ShuffleLz;
                best_len = self.packed.len();
            }
        }
        if self.mode == Compression::Auto && best_len * 10 > raw.len() * 9 {
            self.passthrough = true;
        } else {
            self.encoding = Some(best);
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Read: chunk table
// ═══════════════════════════════════════════════════════════════════════════

#[derive(Debug, Clone, Copy)]
pub(crate) struct Span {
    /// The offset of the chunk body from the beginning of the tensor in the file.
    pub off: u64,
    /// Chunk body length (without flag).
    pub len: u32,
    /// The chunk is stored as is.
    pub raw: bool,
}

/// Parsed table of compressed tensor chunks.
pub(crate) struct ChunkMap {
    chunk: usize,
    raw_len: u64,
    spans: Vec<Span>,
}

/// Checks the last 8 bytes of the compressed tensor; returns `(chunk, number of chunks)`.
pub(crate) fn parse_tail(tail: &[u8], stored: u64, raw_len: u64) -> Result<(usize, usize), String> {
    if tail.len() != TRAILER_FIXED as usize {
        return Err("The compression trailer is the wrong length".into());
    }
    let chunk = u32::from_le_bytes([tail[0], tail[1], tail[2], tail[3]]);
    let count = u32::from_le_bytes([tail[4], tail[5], tail[6], tail[7]]);
    if !(MIN_CHUNK_BYTES..=MAX_CHUNK_BYTES).contains(&chunk) || !chunk.is_multiple_of(8) {
        return Err(format!("invalid chunk size {chunk}"));
    }
    let expected = raw_len.div_ceil(chunk as u64);
    if count as u64 != expected {
        return Err(format!(
            "the number of chunks {count} does not match the size of the data ({expected} chunks)"
        ));
    }
    if (count as u64) * 4 + TRAILER_FIXED > stored {
        return Err("chunk table does not fit into tensor".into());
    }
    Ok((chunk as usize, count as usize))
}

impl ChunkMap {
    /// Parses the length table (`count * 4` bytes, goes before the trailer).
    pub fn parse(
        chunk: usize,
        count: usize,
        table: &[u8],
        stored: u64,
        raw_len: u64,
    ) -> Result<ChunkMap, String> {
        if table.len() != count * 4 {
            return Err("chunk table has incorrect length".into());
        }
        if chunk == 0 || count as u64 != raw_len.div_ceil(chunk as u64) {
            return Err("the number of chunks does not match the data size".into());
        }
        let body_total = stored
            .checked_sub(count as u64 * 4 + TRAILER_FIXED)
            .ok_or_else(|| "chunk table does not fit into tensor".to_string())?;
        let mut map = ChunkMap {
            chunk,
            raw_len,
            spans: Vec::with_capacity(count),
        };
        let mut off = 0u64;
        for i in 0..count {
            let v = u32::from_le_bytes([
                table[i * 4],
                table[i * 4 + 1],
                table[i * 4 + 2],
                table[i * 4 + 3],
            ]);
            let raw = v & RAW_FLAG != 0;
            let len = v & !RAW_FLAG;
            let raw_chunk = map.chunk_raw_len(i) as u64;
            if raw && len as u64 != raw_chunk {
                return Err(format!(
                    "chunk {i}: uncompressed chunk occupies {len} B instead of {raw_chunk} B"
                ));
            }
            if !raw && len == 0 {
                return Err(format!("chunk {i}: empty body"));
            }
            // In an LZ block of `L` bytes, no more than `256·L` bytes of data fit: this cuts off
            // files that promise gigantic size with a tiny body (unpacking bombs).
            if !raw && raw_chunk > len as u64 * 256 {
                return Err(format!("chunk {i}: impossible compression ratio"));
            }
            map.spans.push(Span { off, len, raw });
            off += len as u64;
        }
        if off != body_total {
            return Err("chunk lengths do not match the size of the tensor in the file".into());
        }
        Ok(map)
    }

    /// The raw chunk size is `i`.
    pub fn chunk_raw_len(&self, i: usize) -> usize {
        let start = i as u64 * self.chunk as u64;
        (self.raw_len - start).min(self.chunk as u64) as usize
    }

    pub fn chunk_bytes(&self) -> usize {
        self.chunk
    }

    pub fn span(&self, i: usize) -> Span {
        self.spans[i]
    }

    /// Range of file offsets of chunk bodies `first..=last` (from the beginning of the tensor).
    pub fn body_range(&self, first: usize, last: usize) -> (u64, u64) {
        let a = self.spans[first];
        let b = self.spans[last];
        (a.off, b.off + b.len as u64)
    }

    /// Unpacks chunk `i` from its body `body` and writes raw bytes to `out`.
    /// `es > 1` — the tensor is written with bytes rearranged; `scratch` is a reused buffer.
    pub fn decode_chunk(
        &self,
        i: usize,
        body: &[u8],
        es: usize,
        scratch: &mut Vec<u8>,
        out: &mut Vec<u8>,
    ) -> Result<(), String> {
        let sp = self.spans[i];
        if body.len() != sp.len as usize {
            return Err("chunk body has incorrect length".into());
        }
        if sp.raw {
            out.extend_from_slice(body);
            return Ok(());
        }
        let raw_len = self.chunk_raw_len(i);
        if es > 1 {
            if !raw_len.is_multiple_of(es) {
                return Err("chunk size is not a multiple of the element size".into());
            }
            scratch.clear();
            lz_decompress(body, raw_len, scratch).map_err(|m| m.to_string())?;
            let start = out.len();
            out.resize(start + raw_len, 0);
            unshuffle_bytes(scratch, es, &mut out[start..]);
        } else {
            lz_decompress(body, raw_len, out).map_err(|m| m.to_string())?;
        }
        Ok(())
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Tests
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    struct Rng(u64);

    impl Rng {
        fn next(&mut self) -> u64 {
            // xorshift64*
            self.0 ^= self.0 >> 12;
            self.0 ^= self.0 << 25;
            self.0 ^= self.0 >> 27;
            self.0.wrapping_mul(0x2545_F491_4F6C_DD1D)
        }
        fn below(&mut self, n: u64) -> u64 {
            self.next() % n
        }
    }

    fn lz_roundtrip(data: &[u8]) {
        let mut lz = Lz::new();
        let mut packed = Vec::new();
        lz.compress(data, &mut packed);
        let mut back = Vec::new();
        lz_decompress(&packed, data.len(), &mut back).unwrap();
        assert_eq!(back, data, "length {}", data.len());
    }

    #[test]
    fn lz_roundtrips_edge_sizes_and_patterns() {
        let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
        for len in 0..80usize {
            lz_roundtrip(&vec![0u8; len]);
            let noisy: Vec<u8> = (0..len).map(|_| rng.below(3) as u8).collect();
            lz_roundtrip(&noisy);
        }
        lz_roundtrip(&vec![0u8; 70_000]);
        lz_roundtrip(&b"abcdefgh".repeat(5000));
        let sparse: Vec<u8> = (0..100_000)
            .map(|_| if rng.below(4) == 0 { 1 } else { 0 })
            .collect();
        lz_roundtrip(&sparse);
        let random: Vec<u8> = (0..5000).map(|_| rng.below(256) as u8).collect();
        lz_roundtrip(&random);
    }

    #[test]
    fn lz_shrinks_repetitive_data() {
        let data = vec![7u8; 10_000];
        let mut packed = Vec::new();
        Lz::new().compress(&data, &mut packed);
        assert!(packed.len() < 100, "{}", packed.len());
    }

    #[test]
    fn lz_decoder_rejects_garbage_without_panicking() {
        let mut rng = Rng(12345);
        let src: Vec<u8> = (0..3000)
            .map(|_| [0u8, 0, 0, 1, 2][rng.below(5) as usize])
            .collect();
        let mut packed = Vec::new();
        Lz::new().compress(&src, &mut packed);
        for _ in 0..3000 {
            let mut bad = packed.clone();
            for _ in 0..1 + rng.below(3) {
                let at = rng.below(bad.len() as u64) as usize;
                bad[at] = rng.below(256) as u8;
            }
            let mut out = Vec::new();
            let _ = lz_decompress(&bad, src.len(), &mut out); // Ok or Err, but don't panic
        }
        // truncation
        for cut in 0..packed.len().min(64) {
            let mut out = Vec::new();
            let _ = lz_decompress(&packed[..cut], src.len(), &mut out);
        }
        // incorrect expected size
        let mut out = Vec::new();
        assert!(lz_decompress(&packed, src.len() + 1, &mut out).is_err());
        let mut out = Vec::new();
        assert!(lz_decompress(&packed, src.len() - 1, &mut out).is_err());
    }

    #[test]
    fn shuffle_roundtrip() {
        let mut rng = Rng(7);
        for es in [2usize, 4, 8] {
            let data: Vec<u8> = (0..es * 1000).map(|_| rng.below(256) as u8).collect();
            let mut sh = Vec::new();
            shuffle_bytes(&data, es, &mut sh);
            let mut back = vec![0u8; data.len()];
            unshuffle_bytes(&sh, es, &mut back);
            assert_eq!(back, data);
        }
        let mut sh = Vec::new();
        shuffle_bytes(&[1, 2, 3, 4, 5, 6, 7, 8], 4, &mut sh);
        assert_eq!(sh, vec![1, 5, 2, 6, 3, 7, 4, 8]);
    }

    /// Encodes `data` in unevenly sized chunks and decodes back through `ChunkMap`.
    fn container_roundtrip(
        data: &[u8],
        es: usize,
        mode: Compression,
        chunk: usize,
    ) -> (Encoding, u64) {
        let mut enc = ChunkEncoder::new(mode, chunk, es);
        let mut out: Vec<u8> = Vec::new();
        for piece in data.chunks(777) {
            enc.push(piece, &mut out).unwrap();
        }
        let e = enc.finish(&mut out).unwrap();
        assert_eq!(out.len() as u64, enc.stored());
        if e == Encoding::Raw {
            assert_eq!(out, data);
            return (e, out.len() as u64);
        }
        let stored = out.len() as u64;
        let (c, n) = parse_tail(&out[out.len() - 8..], stored, data.len() as u64).unwrap();
        assert_eq!(c, chunk);
        let table_start = out.len() - 8 - n * 4;
        let map = ChunkMap::parse(
            c,
            n,
            &out[table_start..out.len() - 8],
            stored,
            data.len() as u64,
        )
        .unwrap();
        let shuffle_es = if e == Encoding::ShuffleLz { es } else { 1 };
        let mut back = Vec::new();
        let mut scratch = Vec::new();
        for i in 0..n {
            let sp = map.span(i);
            let body = &out[sp.off as usize..sp.off as usize + sp.len as usize];
            map.decode_chunk(i, body, shuffle_es, &mut scratch, &mut back)
                .unwrap();
        }
        assert_eq!(back, data);
        (e, stored)
    }

    #[test]
    fn container_roundtrips_for_every_size_class() {
        let mut rng = Rng(99);
        let chunk = MIN_CHUNK_BYTES as usize;
        for len in [64usize, 100, 4095, 4096, 4097, 8192, 20_000, 3 * 4096 + 8] {
            let len = len / 8 * 8;
            let zeros = vec![0u8; len];
            let random: Vec<u8> = (0..len).map(|_| rng.below(256) as u8).collect();
            for es in [1usize, 2, 4, 8] {
                for mode in [Compression::Auto, Compression::Always] {
                    container_roundtrip(&zeros, es, mode, chunk);
                    container_roundtrip(&random, es, mode, chunk);
                }
            }
        }
    }

    #[test]
    fn auto_leaves_incompressible_data_raw_and_always_does_not() {
        let mut rng = Rng(5);
        let random: Vec<u8> = (0..20_000).map(|_| rng.below(256) as u8).collect();
        let chunk = MIN_CHUNK_BYTES as usize;
        let (e, stored) = container_roundtrip(&random, 4, Compression::Auto, chunk);
        assert_eq!(e, Encoding::Raw);
        assert_eq!(stored, random.len() as u64);
        let (e, stored) = container_roundtrip(&random, 4, Compression::Always, chunk);
        assert!(e.is_compressed());
        // each chunk is stored as is: overhead - only table and trailer
        assert_eq!(stored, random.len() as u64 + 4 * 5 + 8);
    }

    #[test]
    fn shuffle_is_chosen_when_it_helps() {
        // f32 with a common exponent and almost identical high bytes
        let mut data = Vec::new();
        let mut rng = Rng(3);
        for _ in 0..8192 {
            let v = 1.0f32 + (rng.below(1 << 10) as f32) * f32::EPSILON;
            data.extend_from_slice(&v.to_le_bytes());
        }
        let (e, stored) =
            container_roundtrip(&data, 4, Compression::Auto, DEFAULT_CHUNK_BYTES as usize);
        assert_eq!(e, Encoding::ShuffleLz);
        assert!(stored < data.len() as u64 * 3 / 4, "{stored}");
    }

    #[test]
    fn map_rejects_inconsistent_tables() {
        // tail: chunk = 4096, n = 2, but data for 1 chunk
        let mut tail = Vec::new();
        tail.extend_from_slice(&4096u32.to_le_bytes());
        tail.extend_from_slice(&2u32.to_le_bytes());
        assert!(parse_tail(&tail, 100, 4096).is_err());
        assert!(parse_tail(&tail, 100, 8192).is_ok());
        // chunk out of bounds
        let mut bad = Vec::new();
        bad.extend_from_slice(&7u32.to_le_bytes());
        bad.extend_from_slice(&1u32.to_le_bytes());
        assert!(parse_tail(&bad, 100, 7).is_err());
        // table: lengths do not match body sizes
        let mut table = Vec::new();
        table.extend_from_slice(&10u32.to_le_bytes());
        assert!(ChunkMap::parse(4096, 1, &table, 10 + 4 + 8, 100).is_ok());
        assert!(ChunkMap::parse(4096, 1, &table, 11 + 4 + 8, 100).is_err());
        // "bomb": 4096 bytes from one body byte
        let mut bomb = Vec::new();
        bomb.extend_from_slice(&1u32.to_le_bytes());
        assert!(ChunkMap::parse(4096, 1, &bomb, 1 + 4 + 8, 4096).is_err());
        // uncompressed chunk of incorrect length
        let mut rawbad = Vec::new();
        rawbad.extend_from_slice(&(50u32 | RAW_FLAG).to_le_bytes());
        assert!(ChunkMap::parse(4096, 1, &rawbad, 50 + 4 + 8, 100).is_err());
    }
}
