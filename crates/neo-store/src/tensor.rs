//! Description of the tensor in the index: shape, type, role, checksum, statistics.

use crate::compress::Encoding;
use crate::dtype::DType;
use crate::half::{bf16_bits_to_f32, f16_bits_to_f32};
use crate::value::Meta;

/// Maximum tensor rank.
pub const MAX_RANK: usize = 32;

/// Purpose of the tensor. Unknown codes turn into [`TensorRole::Other`] when read.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TensorRole {
    /// Synapse weights.
    Weight,
    /// Offsets.
    Bias,
    /// Neuron parameters (tau, thresholds, reset potentials, etc.).
    NeuronParam,
    /// State (membrane potentials, currents).
    State,
    /// Delays.
    Delay,
    /// Masks (for example, thinning).
    Mask,
    /// Other buffers (normalization statistics, etc.).
    Buffer,
    /// Optimizer state.
    Optimizer,
    /// The rest.
    Other,
}

impl TensorRole {
    /// Code on disk.
    pub fn code(self) -> u8 {
        match self {
            TensorRole::Weight => 0,
            TensorRole::Bias => 1,
            TensorRole::NeuronParam => 2,
            TensorRole::State => 3,
            TensorRole::Delay => 4,
            TensorRole::Mask => 5,
            TensorRole::Buffer => 6,
            TensorRole::Optimizer => 7,
            TensorRole::Other => 255,
        }
    }

    /// Role by code (unknown code → `Other`).
    pub fn from_code(c: u8) -> TensorRole {
        match c {
            0 => TensorRole::Weight,
            1 => TensorRole::Bias,
            2 => TensorRole::NeuronParam,
            3 => TensorRole::State,
            4 => TensorRole::Delay,
            5 => TensorRole::Mask,
            6 => TensorRole::Buffer,
            7 => TensorRole::Optimizer,
            _ => TensorRole::Other,
        }
    }

    /// Role name.
    pub fn name(self) -> &'static str {
        match self {
            TensorRole::Weight => "weight",
            TensorRole::Bias => "bias",
            TensorRole::NeuronParam => "neuron_param",
            TensorRole::State => "state",
            TensorRole::Delay => "delay",
            TensorRole::Mask => "mask",
            TensorRole::Buffer => "buffer",
            TensorRole::Optimizer => "optimizer",
            TensorRole::Other => "other",
        }
    }

    /// Is the tensor included in the model's trainable parameters?
    pub fn is_parameter(self) -> bool {
        matches!(
            self,
            TensorRole::Weight | TensorRole::Bias | TensorRole::NeuronParam | TensorRole::Delay
        )
    }
}

/// Statistics of the float tensor, calculated during recording (no need to read the data).
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct TensorStats {
    /// Minimum among finite values.
    pub min: f64,
    /// Maximum among finite values.
    pub max: f64,
    /// Number of zeros.
    pub zeros: u64,
    /// Number NaN and ±Inf.
    pub non_finite: u64,
    /// Euclidean norm of finite values.
    pub l2: f64,
}

/// An entry about the tensor.
#[derive(Debug, Clone, PartialEq)]
pub struct TensorInfo {
    /// Unique name.
    pub name: String,
    /// Item type.
    pub dtype: DType,
    /// shape (empty - scalar).
    pub shape: Vec<u64>,
    /// Purpose.
    pub role: TensorRole,
    /// Data offset from the beginning of the file.
    pub offset: u64,
    /// Data size in bytes (logical, after unpacking).
    pub nbytes: u64,
    /// Encoding of data on disk.
    pub encoding: Encoding,
    /// How many bytes the tensor occupies in the file: for uncompressed it is equal to `nbytes`.
    pub stored_bytes: u64,
    /// CRC-32C data (unpacked: independent of encoding).
    pub crc32c: u32,
    /// Statistics (for float tensors, if enabled).
    pub stats: Option<TensorStats>,
    /// Arbitrary tensor attributes.
    pub attrs: Meta,
}

impl TensorInfo {
    /// Number of elements.
    pub fn numel(&self) -> u64 {
        self.shape.iter().fold(1u64, |a, d| a.saturating_mul(*d))
    }

    /// Logical shape: for sparse tensor values - the shape of the original matrix,
    /// for others it’s normal.
    pub fn logical_shape(&self) -> Vec<u64> {
        self.attrs
            .get("sparse.shape")
            .and_then(|v| v.as_u64_list())
            .unwrap_or_else(|| self.shape.clone())
    }

    /// Is the tensor compressed on disk?
    pub fn is_compressed(&self) -> bool {
        self.encoding.is_compressed()
    }

    /// The size of the file is relative to the unpacked one (`stored_bytes / nbytes`; 1.0 - not compressed).
    pub fn stored_ratio(&self) -> f64 {
        if self.nbytes == 0 {
            1.0
        } else {
            self.stored_bytes as f64 / self.nbytes as f64
        }
    }

    /// The proportion of zeros according to statistics, if any.
    pub fn sparsity(&self) -> Option<f64> {
        let n = self.numel();
        self.stats
            .filter(|_| n > 0)
            .map(|s| s.zeros as f64 / n as f64)
    }
}

/// Statistics storage; feeds on pieces of whole elements.
pub(crate) struct StatsAcc {
    dtype: DType,
    min: f64,
    max: f64,
    zeros: u64,
    non_finite: u64,
    sum_sq: f64,
    seen: u64,
    first_non_finite: Option<u64>,
}

impl StatsAcc {
    pub fn new(dtype: DType) -> Self {
        Self {
            dtype,
            min: f64::INFINITY,
            max: f64::NEG_INFINITY,
            zeros: 0,
            non_finite: 0,
            sum_sq: 0.0,
            seen: 0,
            first_non_finite: None,
        }
    }

    #[inline]
    fn visit(&mut self, v: f64) {
        if !v.is_finite() {
            self.non_finite += 1;
            if self.first_non_finite.is_none() {
                self.first_non_finite = Some(self.seen);
            }
        } else {
            if v < self.min {
                self.min = v;
            }
            if v > self.max {
                self.max = v;
            }
            if v == 0.0 {
                self.zeros += 1;
            }
            self.sum_sq += v * v;
        }
        self.seen += 1;
    }

    /// `bytes` is an integer number of elements of type `dtype`.
    pub fn feed(&mut self, bytes: &[u8]) {
        match self.dtype {
            DType::F32 => {
                for c in bytes.as_chunks::<4>().0 {
                    self.visit(f32::from_le_bytes([c[0], c[1], c[2], c[3]]) as f64);
                }
            }
            DType::F64 => {
                for c in bytes.as_chunks::<8>().0 {
                    self.visit(f64::from_le_bytes([
                        c[0], c[1], c[2], c[3], c[4], c[5], c[6], c[7],
                    ]));
                }
            }
            DType::F16 => {
                for c in bytes.as_chunks::<2>().0 {
                    self.visit(f16_bits_to_f32(u16::from_le_bytes([c[0], c[1]])) as f64);
                }
            }
            DType::BF16 => {
                for c in bytes.as_chunks::<2>().0 {
                    self.visit(bf16_bits_to_f32(u16::from_le_bytes([c[0], c[1]])) as f64);
                }
            }
            _ => {}
        }
    }

    pub fn first_non_finite(&self) -> Option<u64> {
        self.first_non_finite
    }

    pub fn finish(self) -> TensorStats {
        let (min, max) = if self.min > self.max {
            (0.0, 0.0)
        } else {
            (self.min, self.max)
        };
        TensorStats {
            min,
            max,
            zeros: self.zeros,
            non_finite: self.non_finite,
            l2: self.sum_sq.sqrt(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn stats_over_f32() {
        let mut acc = StatsAcc::new(DType::F32);
        let mut bytes = Vec::new();
        for v in [0.0f32, 3.0, -4.0, 0.0] {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        acc.feed(&bytes);
        assert_eq!(acc.first_non_finite(), None);
        let s = acc.finish();
        assert_eq!((s.min, s.max, s.zeros, s.non_finite), (-4.0, 3.0, 2, 0));
        assert!((s.l2 - 5.0).abs() < 1e-12);
    }

    #[test]
    fn non_finite_is_located() {
        let mut acc = StatsAcc::new(DType::F32);
        let mut bytes = Vec::new();
        for v in [1.0f32, f32::NAN, f32::INFINITY] {
            bytes.extend_from_slice(&v.to_le_bytes());
        }
        acc.feed(&bytes);
        assert_eq!(acc.first_non_finite(), Some(1));
        assert_eq!(acc.finish().non_finite, 2);
    }
}
