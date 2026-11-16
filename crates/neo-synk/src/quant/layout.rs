//! Layout of packaged weights for the GPU core.
//!
//! The GPU core (`gpu::qgemm_kernel`) reads codes through `u32` words (in WGSL/SPIR-V
//! there are no byte arrays), so the codes from [`neo_quantization`] are repacked
//! from bytes to words little-endian. The bit-cursor layout is **the same**:
//! element `idx` occupies bits `[idx·bits, (idx+1)·bits)` of the stream from the low
//! bit of the first word.
//!
//! The layout does not depend on CubeCL - it can be compiled and tested on the host
//! ([`emulate_gpu_matmul`] repeats kernel arithmetic on the CPU).

use neo_quantization::pack::{Calibration, IntNPacked, NF4_LEVELS, PackedBuffer};

/// What the kernel needs: codes, scales, and the code → value affine scheme.
///
/// Weight value from the `g` group for the raw code `raw`:
///
/// * `use_lut == false`: `scales[g] · (a · raw + offsets[g])`;
/// * `use_lut == true`:  `scales[g] · lut[raw]`.
#[derive(Debug, Clone, PartialEq)]
pub struct GpuQuantLayout {
    /// Codes packed into `u32`-words (+1 stock word for reading “across the border”).
    pub words: Vec<u32>,
    /// Group scale.
    pub scales: Vec<f32>,
    /// Affine group bias (`-zero_point` / `-bias`); for LUT mode - zeros.
    pub offsets: Vec<f32>,
    /// Table `raw → value` (NF4, ternary); empty in affine mode
    /// (for the core it is expanded to one element).
    pub lut: Vec<f32>,
    /// Code width in bits.
    pub bits: u32,
    /// Use `lut` instead of an affine scheme.
    pub use_lut: bool,
    /// Multiplier `a` of the affine mapping (1, and for 1 bit - 2).
    pub a: f32,
    /// Group size by flat index (for per-tensor - all weights).
    pub group_size: u32,
    /// Row weights (`out_features`).
    pub n: u32,
    /// Column weights (`in_features`).
    pub k: u32,
}

/// Assemble the layout for the GPU. `None`, if the format is not supported by the kernel
/// (`Int1..Int8`, NF4, Ternary are supported) or the sizes do not fit into `u32`.
pub fn gpu_layout(w: &PackedBuffer, n: usize, k: usize) -> Option<GpuQuantLayout> {
    let numel = n.checked_mul(k)?;
    if numel == 0 || w.numel() != numel {
        return None;
    }

    let (codes, bits, scales, group_size, offsets, a, lut): (
        &[u8],
        u32,
        &[f32],
        usize,
        Vec<f32>,
        f32,
        Vec<f32>,
    ) = if let Some(p) = w.as_intn() {
        let (a, offsets) = intn_affine(p);
        (
            &p.codes,
            p.bits as u32,
            &p.scales,
            p.group_size,
            offsets,
            a,
            Vec::new(),
        )
    } else {
        match w {
            PackedBuffer::Nf4(p) => (
                &p.codes,
                4,
                &p.scales,
                p.group_size,
                vec![0.0; p.scales.len()],
                1.0,
                NF4_LEVELS.to_vec(),
            ),
            PackedBuffer::Ternary(p) => (
                &p.bits,
                2,
                &p.scales,
                p.group_size,
                vec![0.0; p.scales.len()],
                1.0,
                vec![0.0, 1.0, -1.0, 0.0],
            ),
            _ => return None,
        }
    };

    // Bit indices and flat indices are stored in u32.
    if (numel as u128) * (bits as u128) >= u32::MAX as u128 {
        return None;
    }

    let group_size = if group_size == usize::MAX {
        numel
    } else {
        group_size.min(numel)
    };
    if group_size == 0 {
        return None;
    }

    let mut words: Vec<u32> = codes
        .chunks(4)
        .map(|c| {
            let mut b = [0u8; 4];
            b[..c.len()].copy_from_slice(c);
            u32::from_le_bytes(b)
        })
        .collect();
    words.push(0); // stock: kernel reads word `w + 1` for cross-border codes

    Some(GpuQuantLayout {
        words,
        scales: scales.to_vec(),
        offsets,
        use_lut: !lut.is_empty(),
        lut: if lut.is_empty() { vec![0.0] } else { lut },
        bits,
        a,
        group_size: group_size as u32,
        n: n as u32,
        k: k as u32,
    })
}

fn intn_affine(p: &IntNPacked) -> (f32, Vec<f32>) {
    let groups = p.scales.len();
    match p.calibration {
        Calibration::Symmetric => {
            if p.bits == 1 {
                (2.0, vec![-1.0; groups])
            } else {
                (
                    1.0,
                    vec![-(IntNPacked::symmetric_bias(p.bits) as f32); groups],
                )
            }
        }
        Calibration::Asymmetric => (1.0, p.zero_points.iter().map(|&z| -(z as f32)).collect()),
    }
}

/// Reads `bits` bit from number `idx·bits` - an exact copy of what the kernel does.
#[inline]
fn read_code(words: &[u32], idx: u32, bits: u32) -> u32 {
    let bitpos = idx * bits;
    let w = (bitpos / 32) as usize;
    let sh = bitpos % 32;
    let mut v = words[w] >> sh;
    if sh + bits > 32 {
        v |= words[w + 1] << (32 - sh);
    }
    v & ((1u32 << bits) - 1)
}

/// CPU emulation of the GPU core: the same codes, groups and formulas “code → value”,
/// but the order of summation is different (the kernel divides the row between threads).
/// Needed to check the layout without a GPU. Result `[m, n]`.
pub fn emulate_gpu_matmul(l: &GpuQuantLayout, x: &[f32], m: usize) -> Vec<f32> {
    let (n, k, gs) = (l.n, l.k, l.group_size);
    assert_eq!(x.len(), m * k as usize);
    let mut out = vec![0.0f32; m * n as usize];
    for i in 0..m as u32 {
        for r in 0..n {
            let mut acc = 0.0f32;
            let mut c = 0u32;
            while c < k {
                let flat = r * k + c;
                let g = flat / gs;
                let to_end = (g + 1) * gs - flat;
                let seg = (k - c).min(to_end);
                let (mut dot, mut sx) = (0.0f32, 0.0f32);
                for j in 0..seg {
                    let raw = read_code(&l.words, flat + j, l.bits);
                    let xv = x[(i * k + c + j) as usize];
                    if l.use_lut {
                        dot += l.lut[raw as usize] * xv;
                    } else {
                        dot += raw as f32 * xv;
                        sx += xv;
                    }
                }
                let sc = l.scales[g as usize];
                acc += if l.use_lut {
                    sc * dot
                } else {
                    sc * (l.a * dot + l.offsets[g as usize] * sx)
                };
                c += seg;
            }
            out[(i * n + r) as usize] = acc;
        }
    }
    out
}
