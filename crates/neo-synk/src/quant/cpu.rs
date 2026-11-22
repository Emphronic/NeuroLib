//! CPU fused quantized GEMM: `out[m, n] = x[m, k] · Wᵀ`, `W: [n, k]` packed.
//!
//! # Three levels (see [`KernelKind`])
//!
//! 1. [`KernelKind::IntDomain`] - **real quantized GEMM** for `Int1..Int8`,
//!    NF4 and Ternary. Codes can be read directly from the packaging, group scale
//!    is taken out of the inner loop, and the zero-point is collapsed into a sum
//!    group activations:
//!
//!    ```text
//!    Σ_i  w_i · x_i  =  scale · ( a · Σ_i raw_i·x_i  +  b · Σ_i x_i )
//!    ```
//!
//!    where `raw_i` is the raw code (or grid level), and `(a, b)` is the affine
//!    converting code to value (symmetric: `b = -bias`; asymmetric:
//!    `b = -zero_point`; 1 bit: `a = 2, b = -1`; NF4/ternary: `a = 1, b = 0`).
//!    Multiplication by `scale` and subtraction by `zero_point` are done **once per
//!    group**, and not for each weight. The sums `Σ x_i` are taken from the prefix sums
//!    activations computed once per row `x`.
//!
//! 2. [`KernelKind::RowDecode`] — blockwise dequant + GEMM: Float/FP16/BF16/FP8
//!    and GGUF (`Q4_0..Q8_1`), K-quants. A piece of a row (≤ [`CHUNK`] weights)
//!    is unpacked into a small buffer and immediately collapsed with activations.
//!
//! 3. [`KernelKind::DenseFallback`] - IQ-series, TQ, custom: without yet
//!    row-by-row decoder, we expand the entire matrix once (level 1).
//!
//! # Work layout
//!
//! Activations are processed in blocks of [`M_BLOCK`] rows. Inside a row block
//! weights are distributed to rayon streams; each piece of the weight row is decoded
//! **once** and is reused for all rows of the `x` block, so the price
//! decoding is divided by `M_BLOCK` (in decode mode `m = 1` it is, of course,
//! not divided - only the throughput of reading the scales is important).

use neo_quantization::pack::{
    self, IntNPacked, NF4_LEVELS, PackedBuffer, bf16_to_f32, fp16_to_f32,
};
use neo_quantization::pack::{Calibration, Nf4Packed, TernaryPacked};
use rayon::prelude::*;

/// Maximum weights decoded at one time (4 KiB buffer - fits in L1).
pub const CHUNK: usize = 1024;

/// How many activation rows are processed in one pass through the weights.
pub const M_BLOCK: usize = 128;

/// Which way will the matmul be calculated for this package?
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum KernelKind {
    /// Integer area: codes read from package, scale/zero-point
    /// apply once per group. This is quantized GEMM.
    IntDomain,
    /// Blockwise decoder + convolution. The complete matrix is not created.
    RowDecode,
    /// There is no row-by-row decoder: the matrix is unfolded entirely.
    DenseFallback,
}

/// Errors [`packed_matmul`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum QuantGemmError {
    /// The sizes don't match.
    ShapeMismatch {
        /// What exactly didn't add up?
        what: &'static str,
        /// Expected number of elements.
        expected: usize,
        /// Actual number of elements.
        got: usize,
    },
    /// There is no single path for this package (for example, `GgufRaw`).
    Unsupported(&'static str),
}

impl std::fmt::Display for QuantGemmError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            QuantGemmError::ShapeMismatch {
                what,
                expected,
                got,
            } => {
                write!(f, "quantized gemm: {what}: expected {expected}, got {got}")
            }
            QuantGemmError::Unsupported(why) => write!(f, "quantized gemm: unsupported: {why}"),
        }
    }
}

impl std::error::Error for QuantGemmError {}

/// Which path will [`packed_matmul`] choose to pack `w` with `in_features = k`.
pub fn kernel_kind(w: &PackedBuffer, k: usize) -> KernelKind {
    match w {
        PackedBuffer::Int1(_)
        | PackedBuffer::Int2(_)
        | PackedBuffer::Int3(_)
        | PackedBuffer::Int4(_)
        | PackedBuffer::Int5(_)
        | PackedBuffer::Int6(_)
        | PackedBuffer::Int7(_)
        | PackedBuffer::Int8(_)
        | PackedBuffer::Nf4(_)
        | PackedBuffer::Ternary(_) => KernelKind::IntDomain,

        PackedBuffer::Float(_)
        | PackedBuffer::Bf16(_)
        | PackedBuffer::Fp16(_)
        | PackedBuffer::Fp8(_) => KernelKind::RowDecode,

        PackedBuffer::Gguf(p) => {
            let be = p.dtype.block_elems();
            if pack::gguf::is_supported(p.dtype) && be > 0 && k.is_multiple_of(be) {
                KernelKind::RowDecode
            } else {
                KernelKind::DenseFallback
            }
        }
        PackedBuffer::KQuant(p) => {
            if pack::kquant::is_supported(p.dtype) && k.is_multiple_of(pack::kquant::QK_K) {
                KernelKind::RowDecode
            } else {
                KernelKind::DenseFallback
            }
        }
        _ => KernelKind::DenseFallback,
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Public entry points
// ═══════════════════════════════════════════════════════════════════════════

/// `out[m, n] = x[m, k] · Wᵀ`, where `W` - `[n, k]`, row-major, packed into `w`.
///
/// * `x` — `m·k` values, row-major;
/// * `out` — `m·n` values, row-major (completely overwritten).
///
/// Scaling groups are calculated by **flat** weight index
/// (`idx / group_size`) as in `neo-quantization`, so groups may not
/// divide `k` entirely - the kernel cuts the row along the group boundaries itself.
pub fn packed_matmul(
    x: &[f32],
    m: usize,
    w: &PackedBuffer,
    n: usize,
    k: usize,
    out: &mut [f32],
) -> Result<(), QuantGemmError> {
    check(x.len(), m * k, "x.len() != m*k")?;
    check(out.len(), m * n, "out.len() != m*n")?;
    check(w.numel(), n * k, "packed.numel() != n*k")?;

    if m == 0 || n == 0 || k == 0 {
        out.fill(0.0);
        return Ok(());
    }

    match w {
        PackedBuffer::Int1(p) => run(x, m, n, k, out, &IntNDec(p.as_intn())),
        PackedBuffer::Int2(p) => run(x, m, n, k, out, &IntNDec(p.as_intn())),
        PackedBuffer::Int3(p) => run(x, m, n, k, out, &IntNDec(p.as_intn())),
        PackedBuffer::Int4(p) => run(x, m, n, k, out, &IntNDec(p.as_intn())),
        PackedBuffer::Int5(p) => run(x, m, n, k, out, &IntNDec(p.as_intn())),
        PackedBuffer::Int6(p) => run(x, m, n, k, out, &IntNDec(p.as_intn())),
        PackedBuffer::Int7(p) => run(x, m, n, k, out, &IntNDec(p.as_intn())),
        PackedBuffer::Int8(p) => run(x, m, n, k, out, &IntNDec(p.as_intn())),
        PackedBuffer::Nf4(p) => run(x, m, n, k, out, &Nf4Dec(p)),
        PackedBuffer::Ternary(p) => run(x, m, n, k, out, &TernaryDec(p)),

        PackedBuffer::Float(v) => run(x, m, n, k, out, &FloatDec(v)),
        PackedBuffer::Bf16(p) => run(x, m, n, k, out, &Bf16Dec(&p.data)),
        PackedBuffer::Fp16(p) => run(x, m, n, k, out, &Fp16Dec(&p.data)),
        PackedBuffer::Fp8(_) => run(x, m, n, k, out, &PointDec(w)),

        PackedBuffer::GgufRaw { .. } => {
            return Err(QuantGemmError::Unsupported(
                "GgufRaw has no decoder in neo-quantization",
            ));
        }

        _ => match kernel_kind(w, k) {
            KernelKind::RowDecode => match w {
                PackedBuffer::Gguf(p) => run(x, m, n, k, out, &GgufDec(p)),
                PackedBuffer::KQuant(p) => run(x, m, n, k, out, &KQuantDec(p)),
                _ => unreachable!("RowDecode is only reported for Gguf/KQuant here"),
            },
            _ => {
                // Level 1: expand entirely (IQ/TQ/custom).
                let dense = w.to_f32();
                run(x, m, n, k, out, &FloatDec(&dense));
            }
        },
    }
    Ok(())
}

/// Standard (level 1): full dequant + naive matmul with f64 accumulation.
/// Needed for tests and as a “golden” result; slow and memory hungry.
pub fn dequant_matmul_reference(
    x: &[f32],
    m: usize,
    w: &PackedBuffer,
    n: usize,
    k: usize,
) -> Vec<f32> {
    let dense = w.to_f32();
    assert_eq!(
        dense.len(),
        n * k,
        "dequant_matmul_reference: bad packed size"
    );
    assert_eq!(x.len(), m * k, "dequant_matmul_reference: bad x size");
    let mut out = vec![0.0f32; m * n];
    for i in 0..m {
        for r in 0..n {
            let mut acc = 0.0f64;
            for c in 0..k {
                acc += x[i * k + c] as f64 * dense[r * k + c] as f64;
            }
            out[i * n + r] = acc as f32;
        }
    }
    out
}

fn check(got: usize, expected: usize, what: &'static str) -> Result<(), QuantGemmError> {
    if got == expected {
        Ok(())
    } else {
        Err(QuantGemmError::ShapeMismatch {
            what,
            expected,
            got,
        })
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Decoders
// ═══════════════════════════════════════════════════════════════════════════

/// How to read packaging in pieces.
///
/// Weight value with flat index `i` from group `g`:
/// `w_i = scale_g · (a_g · raw_i + b_g)`, where `raw_i` gives [`decode_raw`].
///
/// [`decode_raw`]: SegDecoder::decode_raw
trait SegDecoder: Sync {
    /// Flat index scaling group size (`usize::MAX` - one).
    fn group_size(&self) -> usize;
    /// `(scale, a, b)` group `g`.
    fn affine(&self, g: usize) -> (f32, f32, f32);
    /// Are activation amounts needed (`b != 0` for at least one group).
    fn needs_sum(&self) -> bool;
    /// Raw values of weights `[start, start + buf.len())` (flat index).
    fn decode_raw(&self, start: usize, buf: &mut [f32]);
}

/// Reader of `bits`-bit codes (bit-cursor from the least significant bit of the first byte -
/// same layout as `neo-quantization::pack::intn`).
struct IntNDec<'a>(&'a IntNPacked);

impl SegDecoder for IntNDec<'_> {
    fn group_size(&self) -> usize {
        self.0.group_size
    }

    fn affine(&self, g: usize) -> (f32, f32, f32) {
        let p = self.0;
        let scale = p.scales[g];
        match p.calibration {
            Calibration::Symmetric => {
                if p.bits == 1 {
                    // raw ∈ {0,1} ↔ {-1,+1}
                    (scale, 2.0, -1.0)
                } else {
                    (scale, 1.0, -(IntNPacked::symmetric_bias(p.bits) as f32))
                }
            }
            Calibration::Asymmetric => (scale, 1.0, -(p.zero_points[g] as f32)),
        }
    }

    fn needs_sum(&self) -> bool {
        true
    }

    fn decode_raw(&self, start: usize, buf: &mut [f32]) {
        let codes = &self.0.codes;
        match self.0.bits {
            8 => {
                for (i, o) in buf.iter_mut().enumerate() {
                    *o = codes[start + i] as f32;
                }
            }
            4 => {
                for (i, o) in buf.iter_mut().enumerate() {
                    let idx = start + i;
                    let byte = codes[idx >> 1];
                    *o = if idx & 1 == 0 { byte & 0x0F } else { byte >> 4 } as f32;
                }
            }
            2 => {
                for (i, o) in buf.iter_mut().enumerate() {
                    let idx = start + i;
                    *o = ((codes[idx >> 2] >> ((idx & 3) * 2)) & 0b11) as f32;
                }
            }
            1 => {
                for (i, o) in buf.iter_mut().enumerate() {
                    let idx = start + i;
                    *o = ((codes[idx >> 3] >> (idx & 7)) & 1) as f32;
                }
            }
            bits => {
                for (i, o) in buf.iter_mut().enumerate() {
                    *o = read_bits(codes, start + i, bits) as f32;
                }
            }
        }
    }
}

/// General case: element `idx` occupies `bits` bits, can cross a byte.
#[inline]
fn read_bits(buf: &[u8], idx: usize, bits: u8) -> u32 {
    let bit_offset = idx * bits as usize;
    let mut remaining = bits;
    let mut byte_idx = bit_offset / 8;
    let mut bit_in_byte = bit_offset % 8;
    let mut result: u32 = 0;
    let mut shift = 0u32;
    while remaining > 0 {
        let space = 8 - bit_in_byte;
        let take = remaining.min(space as u8);
        let mask = ((1u32 << take) - 1) as u8;
        let chunk = (buf[byte_idx] >> bit_in_byte) & mask;
        result |= (chunk as u32) << shift;
        shift += take as u32;
        remaining -= take;
        byte_idx += 1;
        bit_in_byte = 0;
    }
    result
}

/// NF4: 4-bit index in `NF4_LEVELS`, scale - absmax group.
struct Nf4Dec<'a>(&'a Nf4Packed);

impl SegDecoder for Nf4Dec<'_> {
    fn group_size(&self) -> usize {
        self.0.group_size
    }
    fn affine(&self, g: usize) -> (f32, f32, f32) {
        (self.0.scales[g], 1.0, 0.0)
    }
    fn needs_sum(&self) -> bool {
        false
    }
    fn decode_raw(&self, start: usize, buf: &mut [f32]) {
        let codes = &self.0.codes;
        for (i, o) in buf.iter_mut().enumerate() {
            let idx = start + i;
            let byte = codes[idx >> 1];
            let nib = if idx & 1 == 0 { byte & 0x0F } else { byte >> 4 };
            *o = NF4_LEVELS[nib as usize];
        }
    }
}

/// Ternary {-1, 0, +1}: 2 bits per weight, alpha group scale.
struct TernaryDec<'a>(&'a TernaryPacked);

impl SegDecoder for TernaryDec<'_> {
    fn group_size(&self) -> usize {
        self.0.group_size
    }
    fn affine(&self, g: usize) -> (f32, f32, f32) {
        (self.0.scales[g], 1.0, 0.0)
    }
    fn needs_sum(&self) -> bool {
        false
    }
    fn decode_raw(&self, start: usize, buf: &mut [f32]) {
        const LUT: [f32; 4] = [0.0, 1.0, -1.0, 0.0]; // 00→0, 01→+1, 10→-1, 11→0
        let bits = &self.0.bits;
        for (i, o) in buf.iter_mut().enumerate() {
            let idx = start + i;
            *o = LUT[((bits[idx >> 2] >> ((idx & 3) * 2)) & 0b11) as usize];
        }
    }
}

// ── formats without group scale: value = what the decoder gave ──────────

macro_rules! plain_dec {
    ($name:ident, |$s:ident, $start:ident, $buf:ident| $body:block, $($gen:tt)*) => {
        struct $name<'a>($($gen)*);
        impl SegDecoder for $name<'_> {
            fn group_size(&self) -> usize { usize::MAX }
            fn affine(&self, _g: usize) -> (f32, f32, f32) { (1.0, 1.0, 0.0) }
            fn needs_sum(&self) -> bool { false }
            fn decode_raw(&self, $start: usize, $buf: &mut [f32]) {
                let $s = self;
                $body
            }
        }
    };
}

plain_dec!(
    FloatDec,
    |s, start, buf| {
        buf.copy_from_slice(&s.0[start..start + buf.len()]);
    },
    &'a [f32]
);

plain_dec!(
    Bf16Dec,
    |s, start, buf| {
        for (i, o) in buf.iter_mut().enumerate() {
            *o = bf16_to_f32(s.0[start + i]);
        }
    },
    &'a [u16]
);

plain_dec!(
    Fp16Dec,
    |s, start, buf| {
        for (i, o) in buf.iter_mut().enumerate() {
            *o = fp16_to_f32(s.0[start + i]);
        }
    },
    &'a [u16]
);

// FP8: `Fp8Packed::get_f32` has O(1), takes into account tensor-wide scale.
plain_dec!(
    PointDec,
    |s, start, buf| {
        for (i, o) in buf.iter_mut().enumerate() {
            *o = s.0.get_f32(start + i);
        }
    },
    &'a PackedBuffer
);

/// GGUF simple blocks: for each piece we assemble a sub-package from the necessary blocks.
/// `start` and `buf.len()` are multiples of the block size (the engine guarantees: chunks
/// start with multiples of `CHUNK` offsets, and `k` is a multiple of the block - see.
/// [`kernel_kind`]).
struct GgufDec<'a>(&'a pack::GgufPacked);

impl SegDecoder for GgufDec<'_> {
    fn group_size(&self) -> usize {
        usize::MAX
    }
    fn affine(&self, _g: usize) -> (f32, f32, f32) {
        (1.0, 1.0, 0.0)
    }
    fn needs_sum(&self) -> bool {
        false
    }
    fn decode_raw(&self, start: usize, buf: &mut [f32]) {
        let (be, bb) = (self.0.dtype.block_elems(), self.0.dtype.block_bytes());
        debug_assert!(start.is_multiple_of(be) && buf.len().is_multiple_of(be));
        let blocks = &self.0.blocks[(start / be) * bb..((start + buf.len()) / be) * bb];
        let sub = pack::GgufPacked {
            dtype: self.0.dtype,
            blocks: blocks.to_vec(),
            numel: buf.len(),
        };
        buf.copy_from_slice(&sub.to_f32());
    }
}

/// K-quant super-blocks (256 weights): same as [`GgufDec`].
struct KQuantDec<'a>(&'a pack::KQuantPacked);

impl SegDecoder for KQuantDec<'_> {
    fn group_size(&self) -> usize {
        usize::MAX
    }
    fn affine(&self, _g: usize) -> (f32, f32, f32) {
        (1.0, 1.0, 0.0)
    }
    fn needs_sum(&self) -> bool {
        false
    }
    fn decode_raw(&self, start: usize, buf: &mut [f32]) {
        let (be, bb) = (pack::kquant::QK_K, self.0.dtype.block_bytes());
        debug_assert!(start.is_multiple_of(be) && buf.len().is_multiple_of(be));
        let blocks = &self.0.blocks[(start / be) * bb..((start + buf.len()) / be) * bb];
        let sub = pack::KQuantPacked {
            dtype: self.0.dtype,
            blocks: blocks.to_vec(),
            numel: buf.len(),
        };
        buf.copy_from_slice(&sub.to_f32());
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Engine
// ═══════════════════════════════════════════════════════════════════════════

/// Dot product with 8 independent accumulators (good
/// vectorized and accumulates less error than a single accumulator).
#[inline]
fn dot(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let mut acc = [0.0f32; 8];
    let (ca, ra) = a.as_chunks::<8>();
    let (cb, rb) = b.as_chunks::<8>();
    for (xa, xb) in ca.iter().zip(cb) {
        for l in 0..8 {
            acc[l] += xa[l] * xb[l];
        }
    }
    let mut tail = 0.0f32;
    for (xa, xb) in ra.iter().zip(rb) {
        tail += xa * xb;
    }
    acc.iter().sum::<f32>() + tail
}

fn run<D: SegDecoder>(x: &[f32], m: usize, n: usize, k: usize, out: &mut [f32], dec: &D) {
    let gs = dec.group_size();
    let needs_sum = dec.needs_sum();

    for m0 in (0..m).step_by(M_BLOCK) {
        let mb = M_BLOCK.min(m - m0);
        let xb = &x[m0 * k..(m0 + mb) * k];

        // Prefix activation sums (f64): Σ x[c0..c1] = P[c1] - P[c0].
        let prefix: Vec<f64> = if needs_sum {
            let mut p = vec![0.0f64; mb * (k + 1)];
            for i in 0..mb {
                let row = &xb[i * k..(i + 1) * k];
                let pr = &mut p[i * (k + 1)..(i + 1) * (k + 1)];
                let mut s = 0.0f64;
                for (c, &v) in row.iter().enumerate() {
                    s += v as f64;
                    pr[c + 1] = s;
                }
            }
            p
        } else {
            Vec::new()
        };

        // tmp[r * mb + i] = y[m0 + i, r]; each thread owns its own row.
        let mut tmp = vec![0.0f32; n * mb];
        tmp.par_chunks_mut(mb).enumerate().for_each_init(
            || vec![0.0f32; CHUNK],
            |buf, (r, orow)| {
                let base = r * k;
                let mut c = 0usize;
                while c < k {
                    let flat = base + c;
                    let g = if gs == usize::MAX { 0 } else { flat / gs };
                    // Until the end of the group, until the end of the row and no more than CHUNK.
                    let to_group_end = if gs == usize::MAX {
                        usize::MAX
                    } else {
                        (g + 1) * gs - flat
                    };
                    let len = (k - c).min(to_group_end).min(CHUNK);

                    dec.decode_raw(flat, &mut buf[..len]);
                    let (scale, a, b) = dec.affine(g);

                    for i in 0..mb {
                        let xs = &xb[i * k + c..i * k + c + len];
                        let d = dot(&buf[..len], xs);
                        let acc = if needs_sum && b != 0.0 {
                            let sx =
                                (prefix[i * (k + 1) + c + len] - prefix[i * (k + 1) + c]) as f32;
                            a * d + b * sx
                        } else {
                            a * d
                        };
                        orow[i] += scale * acc;
                    }
                    c += len;
                }
            },
        );

        for r in 0..n {
            for i in 0..mb {
                out[(m0 + i) * n + r] = tmp[r * mb + i];
            }
        }
    }
}
