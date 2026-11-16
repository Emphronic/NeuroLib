//! Typed metadata: values and ordered vocabularies.

use std::collections::BTreeMap;

/// Metadata value.
#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    /// Boolean.
    Bool(bool),
    /// Significant whole.
    I64(i64),
    /// Unsigned integer.
    U64(u64),
    /// Floating point number (including NaN/Inf).
    F64(f64),
    /// UTF-8 string.
    Str(String),
    /// Arbitrary bytes (for example, optimizer state).
    Bytes(Vec<u8>),
    /// List of values.
    List(Vec<Value>),
}

/// An ordered dictionary of metadata. Ordering by key makes the file deterministic.
pub type Meta = BTreeMap<String, Value>;

impl Value {
    /// List of signed integers.
    pub fn ints(v: &[i64]) -> Value {
        Value::List(v.iter().map(|x| Value::I64(*x)).collect())
    }

    /// List of unsigned integers.
    pub fn uints(v: &[u64]) -> Value {
        Value::List(v.iter().map(|x| Value::U64(*x)).collect())
    }

    /// Like bool.
    pub fn as_bool(&self) -> Option<bool> {
        match self {
            Value::Bool(b) => Some(*b),
            _ => None,
        }
    }

    /// Like i64 (allows `U64` if it fits).
    pub fn as_i64(&self) -> Option<i64> {
        match self {
            Value::I64(v) => Some(*v),
            Value::U64(v) => i64::try_from(*v).ok(),
            _ => None,
        }
    }

    /// Like u64 (allows non-negative `I64`).
    pub fn as_u64(&self) -> Option<u64> {
        match self {
            Value::U64(v) => Some(*v),
            Value::I64(v) => u64::try_from(*v).ok(),
            _ => None,
        }
    }

    /// Like f64 (allows integers).
    pub fn as_f64(&self) -> Option<f64> {
        match self {
            Value::F64(v) => Some(*v),
            Value::I64(v) => Some(*v as f64),
            Value::U64(v) => Some(*v as f64),
            _ => None,
        }
    }

    /// As a string.
    pub fn as_str(&self) -> Option<&str> {
        match self {
            Value::Str(s) => Some(s.as_str()),
            _ => None,
        }
    }

    /// Like a list.
    pub fn as_list(&self) -> Option<&[Value]> {
        match self {
            Value::List(l) => Some(l.as_slice()),
            _ => None,
        }
    }

    /// List of integers; `None` if it's not a list of integers.
    pub fn as_i64_list(&self) -> Option<Vec<i64>> {
        self.as_list()?.iter().map(|v| v.as_i64()).collect()
    }

    /// Unsigned list; `None` if it is not a list of non-negative integers.
    pub fn as_u64_list(&self) -> Option<Vec<u64>> {
        self.as_list()?.iter().map(|v| v.as_u64()).collect()
    }

    /// An integer or a list of integers (the scalar becomes a list of one element).
    pub fn as_i64_scalar_or_list(&self) -> Option<Vec<i64>> {
        self.as_i64_list()
            .or_else(|| self.as_i64().map(|x| vec![x]))
    }
}

impl From<bool> for Value {
    fn from(v: bool) -> Self {
        Value::Bool(v)
    }
}
impl From<i64> for Value {
    fn from(v: i64) -> Self {
        Value::I64(v)
    }
}
impl From<i32> for Value {
    fn from(v: i32) -> Self {
        Value::I64(v as i64)
    }
}
impl From<u32> for Value {
    fn from(v: u32) -> Self {
        Value::U64(v as u64)
    }
}
impl From<u64> for Value {
    fn from(v: u64) -> Self {
        Value::U64(v)
    }
}
impl From<usize> for Value {
    fn from(v: usize) -> Self {
        Value::U64(v as u64)
    }
}
impl From<f64> for Value {
    fn from(v: f64) -> Self {
        Value::F64(v)
    }
}
impl From<f32> for Value {
    fn from(v: f32) -> Self {
        Value::F64(v as f64)
    }
}
impl From<&str> for Value {
    fn from(v: &str) -> Self {
        Value::Str(v.to_string())
    }
}
impl From<String> for Value {
    fn from(v: String) -> Self {
        Value::Str(v)
    }
}
impl From<Vec<u8>> for Value {
    fn from(v: Vec<u8>) -> Self {
        Value::Bytes(v)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn accessors() {
        assert_eq!(Value::U64(5).as_i64(), Some(5));
        assert_eq!(Value::I64(-1).as_u64(), None);
        assert_eq!(Value::I64(3).as_f64(), Some(3.0));
        assert_eq!(Value::ints(&[1, 2]).as_i64_list(), Some(vec![1, 2]));
        assert_eq!(Value::I64(4).as_i64_scalar_or_list(), Some(vec![4]));
        assert_eq!(Value::Str("a".into()).as_i64_list(), None);
    }
}
