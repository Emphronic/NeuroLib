//! Types of tensor elements and traits [`Element`].
//!
//! Type codes are part of the format on disk and never change. Quantized GGUF formats
//! use the same numeric identifiers as GGUF/`neo-quantization`
//! (`0x0200 | gguf_type_id`), so the blocks are in the file as is, without repacking.

use std::fmt;

/// Block quantized formats GGUF.
#[allow(non_camel_case_types)]
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum GgufKind {
    /// 4 bits, 32 elements per block.
    Q4_0,
    /// 4 bits + minimum.
    Q4_1,
    /// 5 bits
    Q5_0,
    /// 5 bits + minimum.
    Q5_1,
    /// 8 bit.
    Q8_0,
    /// 8 bits + sum.
    Q8_1,
    /// K-quantum 2 bits (super block 256).
    Q2_K,
    /// K-quantum 3 bits.
    Q3_K,
    /// K-quantum 4 bits.
    Q4_K,
    /// K-quantum 5 bits.
    Q5_K,
    /// K-quantum 6 bits.
    Q6_K,
    /// K-quantum 8 bit.
    Q8_K,
    /// IQ2_XXS.
    IQ2_XXS,
    /// IQ2_XS.
    IQ2_XS,
    /// IQ3_XXS.
    IQ3_XXS,
    /// IQ1_S.
    IQ1_S,
    /// IQ4_NL.
    IQ4_NL,
    /// IQ3_S.
    IQ3_S,
    /// IQ4_XS.
    IQ4_XS,
    /// IQ1_M.
    IQ1_M,
    /// Ternary TQ1_0.
    TQ1_0,
    /// Ternary TQ2_0.
    TQ2_0,
}

impl GgufKind {
    /// All supported formats.
    pub const ALL: [GgufKind; 22] = [
        GgufKind::Q4_0,
        GgufKind::Q4_1,
        GgufKind::Q5_0,
        GgufKind::Q5_1,
        GgufKind::Q8_0,
        GgufKind::Q8_1,
        GgufKind::Q2_K,
        GgufKind::Q3_K,
        GgufKind::Q4_K,
        GgufKind::Q5_K,
        GgufKind::Q6_K,
        GgufKind::Q8_K,
        GgufKind::IQ2_XXS,
        GgufKind::IQ2_XS,
        GgufKind::IQ3_XXS,
        GgufKind::IQ1_S,
        GgufKind::IQ4_NL,
        GgufKind::IQ3_S,
        GgufKind::IQ4_XS,
        GgufKind::IQ1_M,
        GgufKind::TQ1_0,
        GgufKind::TQ2_0,
    ];

    /// (gguf id, elements in the block, byte in the block, name)
    const fn row(self) -> (u16, u64, u64, &'static str) {
        match self {
            GgufKind::Q4_0 => (2, 32, 18, "Q4_0"),
            GgufKind::Q4_1 => (3, 32, 20, "Q4_1"),
            GgufKind::Q5_0 => (6, 32, 22, "Q5_0"),
            GgufKind::Q5_1 => (7, 32, 24, "Q5_1"),
            GgufKind::Q8_0 => (8, 32, 34, "Q8_0"),
            GgufKind::Q8_1 => (9, 32, 36, "Q8_1"),
            GgufKind::Q2_K => (10, 256, 84, "Q2_K"),
            GgufKind::Q3_K => (11, 256, 110, "Q3_K"),
            GgufKind::Q4_K => (12, 256, 144, "Q4_K"),
            GgufKind::Q5_K => (13, 256, 176, "Q5_K"),
            GgufKind::Q6_K => (14, 256, 210, "Q6_K"),
            GgufKind::Q8_K => (15, 256, 292, "Q8_K"),
            GgufKind::IQ2_XXS => (16, 256, 66, "IQ2_XXS"),
            GgufKind::IQ2_XS => (17, 256, 74, "IQ2_XS"),
            GgufKind::IQ3_XXS => (18, 256, 98, "IQ3_XXS"),
            GgufKind::IQ1_S => (19, 256, 50, "IQ1_S"),
            GgufKind::IQ4_NL => (20, 32, 18, "IQ4_NL"),
            GgufKind::IQ3_S => (21, 256, 110, "IQ3_S"),
            GgufKind::IQ4_XS => (23, 256, 136, "IQ4_XS"),
            GgufKind::IQ1_M => (29, 256, 56, "IQ1_M"),
            GgufKind::TQ1_0 => (34, 256, 54, "TQ1_0"),
            GgufKind::TQ2_0 => (35, 256, 66, "TQ2_0"),
        }
    }

    /// ID GGUF.
    pub const fn gguf_id(self) -> u16 {
        self.row().0
    }

    /// Format by identifier GGUF.
    pub fn from_gguf_id(id: u16) -> Option<GgufKind> {
        GgufKind::ALL.iter().copied().find(|k| k.gguf_id() == id)
    }

    /// Elements in a block.
    pub const fn block_elems(self) -> u64 {
        self.row().1
    }

    /// Byte per block.
    pub const fn block_bytes(self) -> u64 {
        self.row().2
    }

    /// Format name.
    pub const fn name(self) -> &'static str {
        self.row().3
    }
}

/// Tensor element type.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum DType {
    /// IEEE-754 binary64.
    F64,
    /// IEEE-754 binary32.
    F32,
    /// IEEE-754 binary16.
    F16,
    /// bfloat16.
    BF16,
    /// i64.
    I64,
    /// i32.
    I32,
    /// i16.
    I16,
    /// i8.
    I8,
    /// u64.
    U64,
    /// u32.
    U32,
    /// u16.
    U16,
    /// u8.
    U8,
    /// bool, one byte per element (0 or 1).
    Bool,
    /// Integers of `n` bits (1..=7), packed into a little-endian bit stream: element `i` occupies
    /// bits `[i·n, (i+1)·n)` of the stream, bit 0 of each byte is the least significant. Scales and offsets
    /// are stored by separate tensors. For ternary and binary weights.
    Bits(u8),
    /// Block quantized format GGUF.
    Gguf(GgufKind),
}

impl DType {
    /// Packed type on `n` bits (`1..=7`).
    pub fn bits(n: u8) -> Option<DType> {
        (1..=7).contains(&n).then_some(DType::Bits(n))
    }

    /// Stable code on disk.
    pub fn code(self) -> u16 {
        match self {
            DType::F64 => 1,
            DType::F32 => 2,
            DType::F16 => 3,
            DType::BF16 => 4,
            DType::I64 => 5,
            DType::I32 => 6,
            DType::I16 => 7,
            DType::I8 => 8,
            DType::U64 => 9,
            DType::U32 => 10,
            DType::U16 => 11,
            DType::U8 => 12,
            DType::Bool => 13,
            DType::Bits(n) => 0x0100 | n as u16,
            DType::Gguf(k) => 0x0200 | k.gguf_id(),
        }
    }

    /// Type by code; `None` for unknown.
    pub fn from_code(c: u16) -> Option<DType> {
        match c {
            1 => Some(DType::F64),
            2 => Some(DType::F32),
            3 => Some(DType::F16),
            4 => Some(DType::BF16),
            5 => Some(DType::I64),
            6 => Some(DType::I32),
            7 => Some(DType::I16),
            8 => Some(DType::I8),
            9 => Some(DType::U64),
            10 => Some(DType::U32),
            11 => Some(DType::U16),
            12 => Some(DType::U8),
            13 => Some(DType::Bool),
            0x0101..=0x0107 => Some(DType::Bits((c & 0xff) as u8)),
            0x0200..=0x02ff => GgufKind::from_gguf_id(c & 0xff).map(DType::Gguf),
            _ => None,
        }
    }

    /// The element size in bytes for fixed-size types.
    pub fn element_bytes(self) -> Option<u64> {
        match self {
            DType::F64 | DType::I64 | DType::U64 => Some(8),
            DType::F32 | DType::I32 | DType::U32 => Some(4),
            DType::F16 | DType::BF16 | DType::I16 | DType::U16 => Some(2),
            DType::I8 | DType::U8 | DType::Bool => Some(1),
            DType::Bits(_) | DType::Gguf(_) => None,
        }
    }

    /// How many bytes are occupied by `numel` elements; `None` if the number is not a multiple of the block
    /// (for GGUF) or an overflow occurred.
    pub fn nbytes(self, numel: u64) -> Option<u64> {
        match self {
            DType::Bits(b) => {
                let bits = numel.checked_mul(b as u64)?;
                Some(bits.div_ceil(8))
            }
            DType::Gguf(k) => {
                let e = k.block_elems();
                if !numel.is_multiple_of(e) {
                    None
                } else {
                    (numel / e).checked_mul(k.block_bytes())
                }
            }
            other => numel.checked_mul(other.element_bytes()?),
        }
    }

    /// Real type (f64/f32/f16/bf16).
    pub fn is_float(self) -> bool {
        matches!(self, DType::F64 | DType::F32 | DType::F16 | DType::BF16)
    }

    /// Quantized/packed type.
    pub fn is_quantized(self) -> bool {
        matches!(self, DType::Bits(_) | DType::Gguf(_))
    }

    /// Human-readable name.
    pub fn name(self) -> String {
        match self {
            DType::F64 => "f64".into(),
            DType::F32 => "f32".into(),
            DType::F16 => "f16".into(),
            DType::BF16 => "bf16".into(),
            DType::I64 => "i64".into(),
            DType::I32 => "i32".into(),
            DType::I16 => "i16".into(),
            DType::I8 => "i8".into(),
            DType::U64 => "u64".into(),
            DType::U32 => "u32".into(),
            DType::U16 => "u16".into(),
            DType::U8 => "u8".into(),
            DType::Bool => "bool".into(),
            DType::Bits(n) => format!("bits{n}"),
            DType::Gguf(k) => k.name().to_string(),
        }
    }
}

impl fmt::Display for DType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.name())
    }
}

/// A Rust type that can be written to a tensor and read from it as is.
pub trait Element: Copy + 'static {
    /// Type on disk.
    const DTYPE: DType;
    /// The element's size in bytes.
    const SIZE: usize;
    /// Completes the little-endian presentation.
    fn extend_le(self, out: &mut Vec<u8>);
    /// Reads from the first `SIZE` bytes.
    fn read_le(b: &[u8]) -> Self;
}

macro_rules! impl_element {
    ($t:ty, $d:expr) => {
        impl Element for $t {
            const DTYPE: DType = $d;
            const SIZE: usize = std::mem::size_of::<$t>();
            #[inline]
            fn extend_le(self, out: &mut Vec<u8>) {
                out.extend_from_slice(&self.to_le_bytes());
            }
            #[inline]
            fn read_le(b: &[u8]) -> Self {
                let mut a = [0u8; std::mem::size_of::<$t>()];
                a.copy_from_slice(&b[..std::mem::size_of::<$t>()]);
                <$t>::from_le_bytes(a)
            }
        }
    };
}

impl_element!(f64, DType::F64);
impl_element!(f32, DType::F32);
impl_element!(i64, DType::I64);
impl_element!(i32, DType::I32);
impl_element!(i16, DType::I16);
impl_element!(i8, DType::I8);
impl_element!(u64, DType::U64);
impl_element!(u32, DType::U32);
impl_element!(u16, DType::U16);
impl_element!(u8, DType::U8);

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn codes_roundtrip() {
        let mut all = vec![
            DType::F64,
            DType::F32,
            DType::F16,
            DType::BF16,
            DType::I64,
            DType::I32,
            DType::I16,
            DType::I8,
            DType::U64,
            DType::U32,
            DType::U16,
            DType::U8,
            DType::Bool,
        ];
        for n in 1..=7u8 {
            all.push(DType::Bits(n));
        }
        for k in GgufKind::ALL {
            all.push(DType::Gguf(k));
        }
        for d in all {
            assert_eq!(DType::from_code(d.code()), Some(d), "{d}");
        }
        assert_eq!(DType::from_code(0), None);
        assert_eq!(DType::from_code(0x0100), None);
        assert_eq!(DType::from_code(0x0108), None);
        assert_eq!(DType::from_code(0x02ff), None);
    }

    #[test]
    fn sizes() {
        assert_eq!(DType::F32.nbytes(10), Some(40));
        assert_eq!(DType::Bits(2).nbytes(5), Some(2));
        assert_eq!(DType::Bits(1).nbytes(9), Some(2));
        assert_eq!(DType::Gguf(GgufKind::Q4_0).nbytes(64), Some(36));
        assert_eq!(DType::Gguf(GgufKind::Q4_0).nbytes(40), None);
        assert_eq!(DType::Gguf(GgufKind::Q4_K).nbytes(512), Some(288));
        assert_eq!(DType::U64.nbytes(u64::MAX), None);
    }

    #[test]
    fn element_roundtrip() {
        let mut v = Vec::new();
        1.5f32.extend_le(&mut v);
        (-7i16).extend_le(&mut v);
        assert_eq!(f32::read_le(&v[..4]), 1.5);
        assert_eq!(i16::read_le(&v[4..]), -7);
    }
}
