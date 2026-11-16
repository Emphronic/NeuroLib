//! File layout: header, footer, index.
//!
//! ```text
//! ┌──────────────────────────────┐ 0
//! │ Header (64 B) │ magic, version, flags, alignment, CRC
//! ├──────────────────────────────┤ 64
//! │ tensor data │ each tensor from an address that is a multiple of the alignment (default 64)
//! │ …                            │
//! ├──────────────────────────────┤ data_end (+ padding up to 8)
//! │ Index │ metadata, tensor records, graph; CRC-32C in the footer
//! ├──────────────────────────────┤
//! │ Footer (64 B) │ index offset/length/CRC, file length, fingerprint
//! └──────────────────────────────┘
//! ```
//!
//! The tensor can be stored compressed (encoding byte in the index entry, see [`crate::compress`]):
//! then it ranks `stored_bytes` instead of `nbytes`, and the CRC is still calculated according to
//! unpacked data. Uncompressed tensors are read without copying, as before.
//!
//! The footer at the end allows you to write the file in a stream (the index is known only at the end) and immediately
//! detect a truncated file: the length from the footer must match the actual one.

use crate::codec::{R, W, fmt_err, r_meta, w_meta};
use crate::compress::Encoding;
use crate::crc::{Fnv64, crc32c};
use crate::dtype::DType;
use crate::error::{StoreError, StoreResult};
use crate::graph::{Graph, r_graph, w_graph};
use crate::tensor::{MAX_RANK, TensorInfo, TensorRole, TensorStats};
use crate::value::Meta;

/// Older version of the format. A file with a different older version cannot be read.
pub const FORMAT_MAJOR: u16 = 1;
/// Minor version of the format (backwards compatible additions). 1—compression of tensor data.
pub const FORMAT_MINOR: u16 = 1;

pub(crate) const HEADER_LEN: u64 = 64;
pub(crate) const FOOTER_LEN: u64 = 64;
/// PNG style magic: breaks during text transcoding (CRLF, 7 bits).
pub(crate) const MAGIC: [u8; 8] = *b"\x89SYN\r\n\x1a\n";
pub(crate) const FOOTER_MAGIC: [u8; 8] = *b"SYNEND\0\0";
/// Bits of must-have features this reader knows (none yet).
pub(crate) const KNOWN_REQUIRED_FLAGS: u32 = 0;

#[derive(Debug, Clone)]
pub(crate) struct Header {
    pub major: u16,
    pub minor: u16,
    pub required_flags: u32,
    pub optional_flags: u32,
    pub alignment: u32,
    pub created_unix_ms: u64,
}

impl Header {
    pub fn encode(&self) -> [u8; 64] {
        let mut b = [0u8; 64];
        b[0..8].copy_from_slice(&MAGIC);
        b[8..10].copy_from_slice(&self.major.to_le_bytes());
        b[10..12].copy_from_slice(&self.minor.to_le_bytes());
        b[12..16].copy_from_slice(&self.required_flags.to_le_bytes());
        b[16..20].copy_from_slice(&self.optional_flags.to_le_bytes());
        b[20..24].copy_from_slice(&self.alignment.to_le_bytes());
        b[24..32].copy_from_slice(&self.created_unix_ms.to_le_bytes());
        let crc = crc32c(&b[..60]);
        b[60..64].copy_from_slice(&crc.to_le_bytes());
        b
    }

    pub fn decode(b: &[u8]) -> StoreResult<Header> {
        if b.len() != 64 {
            return Err(fmt_err("The header should be 64 bytes"));
        }
        if b[0..8] != MAGIC {
            return Err(fmt_err("this is not a syn format file (wrong magic)"));
        }
        let stored = u32::from_le_bytes([b[60], b[61], b[62], b[63]]);
        let actual = crc32c(&b[..60]);
        if stored != actual {
            return Err(StoreError::Checksum {
                what: "title".into(),
                expected: stored,
                actual,
            });
        }
        let h = Header {
            major: u16::from_le_bytes([b[8], b[9]]),
            minor: u16::from_le_bytes([b[10], b[11]]),
            required_flags: u32::from_le_bytes([b[12], b[13], b[14], b[15]]),
            optional_flags: u32::from_le_bytes([b[16], b[17], b[18], b[19]]),
            alignment: u32::from_le_bytes([b[20], b[21], b[22], b[23]]),
            created_unix_ms: u64::from_le_bytes([
                b[24], b[25], b[26], b[27], b[28], b[29], b[30], b[31],
            ]),
        };
        if h.major != FORMAT_MAJOR {
            return Err(StoreError::UnsupportedVersion {
                major: h.major,
                minor: h.minor,
            });
        }
        let unknown = h.required_flags & !KNOWN_REQUIRED_FLAGS;
        if unknown != 0 {
            return Err(StoreError::UnsupportedFeature(format!(
                "the file requires {unknown:#x} capabilities unknown to this reader"
            )));
        }
        if !h.alignment.is_power_of_two() || !(8..=65536).contains(&h.alignment) {
            return Err(fmt_err(format!("invalid alignment {}", h.alignment)));
        }
        Ok(h)
    }
}

#[derive(Debug, Clone)]
pub(crate) struct Footer {
    pub index_offset: u64,
    pub index_len: u64,
    pub index_crc: u32,
    pub tensor_count: u32,
    pub data_end: u64,
    pub file_len: u64,
    pub fingerprint: u64,
}

impl Footer {
    pub fn encode(&self) -> [u8; 64] {
        let mut b = [0u8; 64];
        b[0..8].copy_from_slice(&FOOTER_MAGIC);
        b[8..16].copy_from_slice(&self.index_offset.to_le_bytes());
        b[16..24].copy_from_slice(&self.index_len.to_le_bytes());
        b[24..28].copy_from_slice(&self.index_crc.to_le_bytes());
        b[28..32].copy_from_slice(&self.tensor_count.to_le_bytes());
        b[32..40].copy_from_slice(&self.data_end.to_le_bytes());
        b[40..48].copy_from_slice(&self.file_len.to_le_bytes());
        b[48..56].copy_from_slice(&self.fingerprint.to_le_bytes());
        let crc = crc32c(&b[..60]);
        b[60..64].copy_from_slice(&crc.to_le_bytes());
        b
    }

    pub fn decode(b: &[u8]) -> StoreResult<Footer> {
        if b.len() != 64 {
            return Err(fmt_err("the footer should occupy 64 bytes"));
        }
        if b[0..8] != FOOTER_MAGIC {
            return Err(fmt_err(
                "footer not found: file is truncated or recording was not completed",
            ));
        }
        let stored = u32::from_le_bytes([b[60], b[61], b[62], b[63]]);
        let actual = crc32c(&b[..60]);
        if stored != actual {
            return Err(StoreError::Checksum {
                what: "footer".into(),
                expected: stored,
                actual,
            });
        }
        let u64_at = |o: usize| {
            u64::from_le_bytes([
                b[o],
                b[o + 1],
                b[o + 2],
                b[o + 3],
                b[o + 4],
                b[o + 5],
                b[o + 6],
                b[o + 7],
            ])
        };
        Ok(Footer {
            index_offset: u64_at(8),
            index_len: u64_at(16),
            index_crc: u32::from_le_bytes([b[24], b[25], b[26], b[27]]),
            tensor_count: u32::from_le_bytes([b[28], b[29], b[30], b[31]]),
            data_end: u64_at(32),
            file_len: u64_at(40),
            fingerprint: u64_at(48),
        })
    }
}

/// Fingerprint of a set of tensors: names, types, shapes, sizes and CRCs; does not depend on the recording order.
pub(crate) fn fingerprint(tensors: &[TensorInfo]) -> u64 {
    let mut order: Vec<&TensorInfo> = tensors.iter().collect();
    order.sort_by(|a, b| a.name.cmp(&b.name));
    let mut h = Fnv64::new();
    for t in order {
        h.write(t.name.as_bytes());
        h.write(&[0]);
        h.write(&t.dtype.code().to_le_bytes());
        h.write(&(t.shape.len() as u32).to_le_bytes());
        for d in &t.shape {
            h.write(&d.to_le_bytes());
        }
        h.write(&t.nbytes.to_le_bytes());
        h.write(&t.crc32c.to_le_bytes());
    }
    h.finish()
}

pub(crate) fn w_tensor(w: &mut W, t: &TensorInfo) {
    w.str(&t.name);
    w.u16(t.dtype.code());
    w.u8(t.role.code());
    w.u8(t.encoding.code()); // 0 = as is, otherwise compression (see crate::compress)
    w.u8(t.shape.len() as u8);
    for d in &t.shape {
        w.u64(*d);
    }
    w.u64(t.offset);
    w.u64(t.nbytes);
    if t.encoding.is_compressed() {
        w.u64(t.stored_bytes); // how long does the tensor take up in the file?
    }
    w.u32(t.crc32c);
    match &t.stats {
        Some(s) => {
            w.bool(true);
            w.f64(s.min);
            w.f64(s.max);
            w.u64(s.zeros);
            w.u64(s.non_finite);
            w.f64(s.l2);
        }
        None => w.bool(false),
    }
    w_meta(w, &t.attrs);
}

pub(crate) fn r_tensor(r: &mut R) -> StoreResult<TensorInfo> {
    let name = r.str()?;
    let code = r.u16()?;
    let dtype = DType::from_code(code)
        .ok_or_else(|| fmt_err(format!("tensor '{name}': unknown type {code:#06x}")))?;
    let role = TensorRole::from_code(r.u8()?);
    let encoding_code = r.u8()?;
    let encoding = Encoding::from_code(encoding_code).ok_or_else(|| {
        StoreError::UnsupportedFeature(format!(
            "tensor '{name}': {encoding_code} data encoding is not supported"
        ))
    })?;
    let ndim = r.u8()? as usize;
    if ndim > MAX_RANK {
        return Err(fmt_err(format!(
            "tensor “{name}”: rank {ndim} is greater than {MAX_RANK}"
        )));
    }
    let mut shape = Vec::with_capacity(ndim);
    for _ in 0..ndim {
        shape.push(r.u64()?);
    }
    let offset = r.u64()?;
    let nbytes = r.u64()?;
    let stored_bytes = if encoding.is_compressed() {
        r.u64()?
    } else {
        nbytes
    };
    let crc = r.u32()?;
    let stats = if r.bool()? {
        Some(TensorStats {
            min: r.f64()?,
            max: r.f64()?,
            zeros: r.u64()?,
            non_finite: r.u64()?,
            l2: r.f64()?,
        })
    } else {
        None
    };
    let attrs = r_meta(r)?;
    Ok(TensorInfo {
        name,
        dtype,
        shape,
        role,
        offset,
        nbytes,
        encoding,
        stored_bytes,
        crc32c: crc,
        stats,
        attrs,
    })
}

pub(crate) fn encode_index(
    meta: &Meta,
    tensors: &[TensorInfo],
    graph: Option<&Graph>,
) -> StoreResult<Vec<u8>> {
    let mut w = W::new();
    w_meta(&mut w, meta);
    w.len(tensors.len());
    for t in tensors {
        w_tensor(&mut w, t);
    }
    match graph {
        Some(g) => {
            w.bool(true);
            w_graph(&mut w, g);
        }
        None => w.bool(false),
    }
    w.finish()
}

pub(crate) type DecodedIndex = (Meta, Vec<TensorInfo>, Option<Graph>);

pub(crate) fn decode_index(bytes: &[u8]) -> StoreResult<DecodedIndex> {
    let mut r = R::new(bytes);
    let meta = r_meta(&mut r)?;
    let n = r.count(34)?;
    let mut tensors = Vec::with_capacity(n);
    for _ in 0..n {
        tensors.push(r_tensor(&mut r)?);
    }
    let graph = if r.bool()? {
        Some(r_graph(&mut r, 0)?)
    } else {
        None
    };
    if !r.is_empty() {
        return Err(fmt_err("there are extra bytes in the index"));
    }
    Ok((meta, tensors, graph))
}

/// Checks tensor entries against the file layout.
pub(crate) fn validate_tensors(
    tensors: &[TensorInfo],
    alignment: u32,
    data_end: u64,
) -> StoreResult<()> {
    let mut spans: Vec<(u64, u64, &str)> = Vec::with_capacity(tensors.len());
    let mut names = std::collections::HashSet::with_capacity(tensors.len());
    for t in tensors {
        if !names.insert(t.name.as_str()) {
            return Err(fmt_err(format!("tensor '{}' occurs twice", t.name)));
        }
        let numel = t
            .shape
            .iter()
            .try_fold(1u64, |a, d| a.checked_mul(*d))
            .ok_or_else(|| {
                fmt_err(format!(
                    "tensor '{}': number of elements overflows u64",
                    t.name
                ))
            })?;
        let expected = t.dtype.nbytes(numel).ok_or_else(|| {
            fmt_err(format!(
                "tensor “{}”: {numel} elements of type {} do not form an integer number of blocks",
                t.name, t.dtype
            ))
        })?;
        if expected != t.nbytes {
            return Err(fmt_err(format!(
                "tensor “{}”: size {} B does not converge with the shape {:?} and type {} ({expected} B)",
                t.name, t.nbytes, t.shape, t.dtype
            )));
        }
        if t.encoding.is_compressed() {
            if t.nbytes == 0 {
                return Err(fmt_err(format!(
                    "tensor '{}': empty tensor cannot be compressed",
                    t.name
                )));
            }
            if t.encoding == Encoding::ShuffleLz && t.dtype.element_bytes().is_none_or(|e| e < 2) {
                return Err(fmt_err(format!(
                    "tensor '{}': byte swap does not apply to type {}",
                    t.name, t.dtype
                )));
            }
            if t.stored_bytes < 8 {
                return Err(fmt_err(format!(
                    "tensor '{}': compressed data shorter than trailer",
                    t.name
                )));
            }
        } else if t.stored_bytes != t.nbytes {
            return Err(fmt_err(format!(
                "tensor '{}': the size in the file {} B is not equal to the data size {} B",
                t.name, t.stored_bytes, t.nbytes
            )));
        }
        if t.offset < HEADER_LEN || t.offset % alignment as u64 != 0 {
            return Err(fmt_err(format!(
                "tensor '{}': offset {} is not aligned or is in the header",
                t.name, t.offset
            )));
        }
        let end = t
            .offset
            .checked_add(t.stored_bytes)
            .filter(|e| *e <= data_end)
            .ok_or_else(|| fmt_err(format!("tensor '{}' goes out of data area", t.name)))?;
        spans.push((t.offset, end, t.name.as_str()));
    }
    spans.sort_by_key(|s| (s.0, s.1));
    for pair in spans.windows(2) {
        if pair[1].0 < pair[0].1 {
            return Err(fmt_err(format!(
                "tensors '{}' and '{}' overlap",
                pair[0].2, pair[1].2
            )));
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn header_and_footer_roundtrip() {
        let h = Header {
            major: FORMAT_MAJOR,
            minor: 7,
            required_flags: 0,
            optional_flags: 5,
            alignment: 64,
            created_unix_ms: 123,
        };
        let back = Header::decode(&h.encode()).unwrap();
        assert_eq!(
            (back.minor, back.optional_flags, back.created_unix_ms),
            (7, 5, 123)
        );
        let mut bad = h.encode();
        bad[30] ^= 1;
        assert!(matches!(
            Header::decode(&bad),
            Err(StoreError::Checksum { .. })
        ));

        let f = Footer {
            index_offset: 100,
            index_len: 20,
            index_crc: 9,
            tensor_count: 2,
            data_end: 96,
            file_len: 184,
            fingerprint: 77,
        };
        let fb = Footer::decode(&f.encode()).unwrap();
        assert_eq!(
            (fb.index_offset, fb.file_len, fb.fingerprint),
            (100, 184, 77)
        );
    }

    #[test]
    fn newer_major_and_unknown_required_flags_rejected() {
        let mut h = Header {
            major: FORMAT_MAJOR + 1,
            minor: 0,
            required_flags: 0,
            optional_flags: 0,
            alignment: 64,
            created_unix_ms: 0,
        };
        assert!(matches!(
            Header::decode(&h.encode()),
            Err(StoreError::UnsupportedVersion { .. })
        ));
        h.major = FORMAT_MAJOR;
        h.required_flags = 0x10;
        assert!(matches!(
            Header::decode(&h.encode()),
            Err(StoreError::UnsupportedFeature(_))
        ));
    }
}
