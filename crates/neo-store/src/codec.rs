//! Low-level index coding: little-endian, length `u32`, protection against “bombs” in the data.

use crate::error::{StoreError, StoreResult};
use crate::value::{Meta, Value};

/// Maximum nesting depth of values and subgraphs when reading.
pub(crate) const MAX_DEPTH: usize = 32;

pub(crate) fn fmt_err(m: impl Into<String>) -> StoreError {
    StoreError::Format(m.into())
}

/// Byte writer. Length overflow (> 4 GiB) is stored and returned from `finish`.
pub(crate) struct W {
    buf: Vec<u8>,
    overflow: bool,
}

impl W {
    pub fn new() -> Self {
        Self {
            buf: Vec::new(),
            overflow: false,
        }
    }
    pub fn finish(self) -> StoreResult<Vec<u8>> {
        if self.overflow {
            return Err(StoreError::Invalid(
                "a string or list longer than 4 GiB does not fit into the index".into(),
            ));
        }
        Ok(self.buf)
    }
    pub fn u8(&mut self, v: u8) {
        self.buf.push(v);
    }
    pub fn bool(&mut self, v: bool) {
        self.buf.push(v as u8);
    }
    pub fn u16(&mut self, v: u16) {
        self.buf.extend_from_slice(&v.to_le_bytes());
    }
    pub fn u32(&mut self, v: u32) {
        self.buf.extend_from_slice(&v.to_le_bytes());
    }
    pub fn u64(&mut self, v: u64) {
        self.buf.extend_from_slice(&v.to_le_bytes());
    }
    pub fn i64(&mut self, v: i64) {
        self.buf.extend_from_slice(&v.to_le_bytes());
    }
    pub fn f64(&mut self, v: f64) {
        self.buf.extend_from_slice(&v.to_bits().to_le_bytes());
    }
    pub fn len(&mut self, n: usize) {
        match u32::try_from(n) {
            Ok(v) => self.u32(v),
            Err(_) => {
                self.overflow = true;
                self.u32(0);
            }
        }
    }
    pub fn str(&mut self, s: &str) {
        self.len(s.len());
        self.buf.extend_from_slice(s.as_bytes());
    }
    pub fn bytes(&mut self, b: &[u8]) {
        self.len(b.len());
        self.buf.extend_from_slice(b);
    }
}

/// Byte reader with bounds checking.
pub(crate) struct R<'a> {
    b: &'a [u8],
    p: usize,
}

impl<'a> R<'a> {
    pub fn new(b: &'a [u8]) -> Self {
        Self { b, p: 0 }
    }
    pub fn remaining(&self) -> usize {
        self.b.len() - self.p
    }
    pub fn take(&mut self, n: usize) -> StoreResult<&'a [u8]> {
        let end = self
            .p
            .checked_add(n)
            .filter(|e| *e <= self.b.len())
            .ok_or_else(|| fmt_err("unexpected end of data"))?;
        let s = &self.b[self.p..end];
        self.p = end;
        Ok(s)
    }
    fn arr<const N: usize>(&mut self) -> StoreResult<[u8; N]> {
        let mut a = [0u8; N];
        a.copy_from_slice(self.take(N)?);
        Ok(a)
    }
    pub fn u8(&mut self) -> StoreResult<u8> {
        Ok(self.take(1)?[0])
    }
    pub fn bool(&mut self) -> StoreResult<bool> {
        match self.u8()? {
            0 => Ok(false),
            1 => Ok(true),
            v => Err(fmt_err(format!("invalid bool {v}"))),
        }
    }
    pub fn u16(&mut self) -> StoreResult<u16> {
        Ok(u16::from_le_bytes(self.arr()?))
    }
    pub fn u32(&mut self) -> StoreResult<u32> {
        Ok(u32::from_le_bytes(self.arr()?))
    }
    pub fn u64(&mut self) -> StoreResult<u64> {
        Ok(u64::from_le_bytes(self.arr()?))
    }
    pub fn i64(&mut self) -> StoreResult<i64> {
        Ok(i64::from_le_bytes(self.arr()?))
    }
    pub fn f64(&mut self) -> StoreResult<f64> {
        Ok(f64::from_bits(u64::from_le_bytes(self.arr()?)))
    }
    /// Number of collection elements; cannot exceed what will fit in the remainder of the data.
    pub fn count(&mut self, min_item_bytes: usize) -> StoreResult<usize> {
        let n = self.u32()? as usize;
        if n.saturating_mul(min_item_bytes.max(1)) > self.remaining() {
            return Err(fmt_err("collection size exceeds data size"));
        }
        Ok(n)
    }
    pub fn str(&mut self) -> StoreResult<String> {
        let n = self.u32()? as usize;
        let raw = self.take(n)?;
        String::from_utf8(raw.to_vec()).map_err(|_| fmt_err("string is not UTF-8"))
    }
    pub fn bytes(&mut self) -> StoreResult<Vec<u8>> {
        let n = self.u32()? as usize;
        Ok(self.take(n)?.to_vec())
    }
    pub fn is_empty(&self) -> bool {
        self.remaining() == 0
    }
}

pub(crate) fn w_value(w: &mut W, v: &Value) {
    match v {
        Value::Bool(b) => {
            w.u8(0);
            w.bool(*b);
        }
        Value::I64(x) => {
            w.u8(1);
            w.i64(*x);
        }
        Value::U64(x) => {
            w.u8(2);
            w.u64(*x);
        }
        Value::F64(x) => {
            w.u8(3);
            w.f64(*x);
        }
        Value::Str(s) => {
            w.u8(4);
            w.str(s);
        }
        Value::Bytes(b) => {
            w.u8(5);
            w.bytes(b);
        }
        Value::List(l) => {
            w.u8(6);
            w.len(l.len());
            for x in l {
                w_value(w, x);
            }
        }
    }
}

pub(crate) fn r_value(r: &mut R, depth: usize) -> StoreResult<Value> {
    if depth > MAX_DEPTH {
        return Err(fmt_err("nesting of values too deep"));
    }
    Ok(match r.u8()? {
        0 => Value::Bool(r.bool()?),
        1 => Value::I64(r.i64()?),
        2 => Value::U64(r.u64()?),
        3 => Value::F64(r.f64()?),
        4 => Value::Str(r.str()?),
        5 => Value::Bytes(r.bytes()?),
        6 => {
            let n = r.count(1)?;
            let mut l = Vec::with_capacity(n);
            for _ in 0..n {
                l.push(r_value(r, depth + 1)?);
            }
            Value::List(l)
        }
        t => return Err(fmt_err(format!("unknown value tag {t}"))),
    })
}

pub(crate) fn w_meta(w: &mut W, m: &Meta) {
    w.len(m.len());
    for (k, v) in m {
        w.str(k);
        w_value(w, v);
    }
}

pub(crate) fn r_meta(r: &mut R) -> StoreResult<Meta> {
    let n = r.count(5)?;
    let mut m = Meta::new();
    for _ in 0..n {
        let k = r.str()?;
        let v = r_value(r, 0)?;
        if m.insert(k.clone(), v).is_some() {
            return Err(fmt_err(format!("duplicate metadata key '{k}'")));
        }
    }
    Ok(m)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn meta_roundtrip() {
        let mut m = Meta::new();
        m.insert("a".into(), Value::Bool(true));
        m.insert("b".into(), Value::ints(&[1, -2, 3]));
        m.insert("c".into(), Value::Str("hi".into()));
        m.insert("d".into(), Value::F64(f64::INFINITY));
        m.insert("e".into(), Value::Bytes(vec![1, 2, 3]));
        let mut w = W::new();
        w_meta(&mut w, &m);
        let bytes = w.finish().unwrap();
        let mut r = R::new(&bytes);
        let back = r_meta(&mut r).unwrap();
        assert!(r.is_empty());
        assert_eq!(back, m);
    }

    #[test]
    fn hostile_lengths_are_rejected() {
        // "a billion elements" in a tiny buffer
        let bytes = [0xff, 0xff, 0xff, 0x3f];
        let mut r = R::new(&bytes);
        assert!(r.count(1).is_err());
        let mut r = R::new(&bytes);
        assert!(r.str().is_err());
    }

    #[test]
    fn deep_nesting_is_rejected() {
        let mut w = W::new();
        for _ in 0..(MAX_DEPTH + 5) {
            w.u8(6);
            w.u32(1);
        }
        w.u8(0);
        w.bool(true);
        let bytes = w.finish().unwrap();
        assert!(r_value(&mut R::new(&bytes), 0).is_err());
    }
}
