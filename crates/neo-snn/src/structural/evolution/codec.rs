//! Mini codec for genomes and populations: little-endian, no external dependencies.
//! As in `structural::checkpoint`, the format is versioned, and any departure from the borders
//! buffers are an error, not a panic.

use crate::scalar::Scalar;
use crate::state::ResetMode;

use super::super::error::{StructuralError, StructuralResult};
use super::super::model::{
    AdExParams, AdaptiveLifParams, HodgkinHuxleyParams, IzhikevichParams, LifParams, NeuronModel,
};

pub(crate) fn err(msg: impl Into<String>) -> StructuralError {
    StructuralError::param("evolution", msg.into())
}

/// Writer. `wide` — width of real values on disk: `false` — `f32` (4 bytes,
/// format versions 1–2), `true` - `f64` (8 bytes). Values of any `T: Scalar`
/// are reduced to the selected width (writing `f64` in a narrow format loses accuracy).
#[derive(Default)]
pub(crate) struct W {
    pub buf: Vec<u8>,
    pub wide: bool,
}

impl W {
    pub fn with_width(wide: bool) -> Self {
        Self {
            buf: Vec::new(),
            wide,
        }
    }
    /// Real value in the selected width.
    pub fn sc<T: Scalar>(&mut self, v: T) {
        if self.wide {
            self.buf
                .extend_from_slice(&v.to_f64().to_bits().to_le_bytes());
        } else {
            self.f32(v.to_f64() as f32);
        }
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
    pub fn f32(&mut self, v: f32) {
        self.buf.extend_from_slice(&v.to_bits().to_le_bytes());
    }
    pub fn len(&mut self, n: usize) {
        self.u64(n as u64);
    }
    pub fn lif<T: Scalar>(&mut self, p: &LifParams<T>) {
        self.sc(p.beta);
        self.sc(p.threshold);
        self.u8(match p.reset {
            ResetMode::Subtract => 0,
            ResetMode::Zero => 1,
            ResetMode::None => 2,
        });
    }
    pub fn model<T: Scalar>(&mut self, m: &NeuronModel<T>) {
        self.u8(m.tag());
        match m {
            NeuronModel::Lif(p) => self.lif(p),
            NeuronModel::AdaptiveLif(p) => {
                self.sc(p.beta);
                self.sc(p.rho);
                self.sc(p.adapt_increment);
                self.sc(p.threshold);
                self.u8(match p.reset {
                    ResetMode::Subtract => 0,
                    ResetMode::Zero => 1,
                    ResetMode::None => 2,
                });
            }
            NeuronModel::Izhikevich(p) => {
                self.sc(p.a);
                self.sc(p.b);
                self.sc(p.c);
                self.sc(p.d);
                self.sc(p.threshold);
                self.len(p.substeps);
            }
            NeuronModel::AdEx(p) => {
                for v in [
                    p.c, p.g_l, p.e_l, p.v_t, p.delta_t, p.a, p.tau_w, p.b, p.v_reset, p.v_peak,
                    p.dt, p.exp_clip,
                ] {
                    self.sc(v);
                }
                self.len(p.substeps);
            }
            NeuronModel::HodgkinHuxley(p) => {
                for v in [
                    p.c,
                    p.g_na,
                    p.g_k,
                    p.g_l,
                    p.e_na,
                    p.e_k,
                    p.e_l,
                    p.v_rest,
                    p.spike_threshold,
                    p.dt,
                ] {
                    self.sc(v);
                }
                self.len(p.substeps);
            }
        }
    }
}

pub(crate) struct R<'a> {
    b: &'a [u8],
    at: usize,
    /// Width of real values on disk (see [`W`]).
    pub wide: bool,
}

impl<'a> R<'a> {
    pub fn new(b: &'a [u8]) -> Self {
        Self {
            b,
            at: 0,
            wide: false,
        }
    }
    fn take(&mut self, n: usize) -> StructuralResult<&'a [u8]> {
        let end = self
            .at
            .checked_add(n)
            .filter(|e| *e <= self.b.len())
            .ok_or_else(|| err("unexpected end of data"))?;
        let s = &self.b[self.at..end];
        self.at = end;
        Ok(s)
    }
    pub fn u8(&mut self) -> StructuralResult<u8> {
        Ok(self.take(1)?[0])
    }
    pub fn bool(&mut self) -> StructuralResult<bool> {
        match self.u8()? {
            0 => Ok(false),
            1 => Ok(true),
            x => Err(err(format!("invalid bool {x}"))),
        }
    }
    pub fn u16(&mut self) -> StructuralResult<u16> {
        Ok(u16::from_le_bytes(
            self.take(2)?.try_into().map_err(|_| err("u16"))?,
        ))
    }
    pub fn u32(&mut self) -> StructuralResult<u32> {
        Ok(u32::from_le_bytes(
            self.take(4)?.try_into().map_err(|_| err("u32"))?,
        ))
    }
    pub fn u64(&mut self) -> StructuralResult<u64> {
        Ok(u64::from_le_bytes(
            self.take(8)?.try_into().map_err(|_| err("u64"))?,
        ))
    }
    pub fn f32(&mut self) -> StructuralResult<f32> {
        Ok(f32::from_bits(self.u32()?))
    }
    /// A real value at the width specified by the format.
    pub fn sc<T: Scalar>(&mut self) -> StructuralResult<T> {
        if self.wide {
            Ok(T::from_f64(f64::from_bits(self.u64()?)))
        } else {
            Ok(T::from_f64(self.f32()? as f64))
        }
    }
    /// Collection length; no more than `max`, so that damaged data does not require gigabytes.
    pub fn len(&mut self, max: usize) -> StructuralResult<usize> {
        let n = self.u64()?;
        if n > max as u64 {
            return Err(err(format!("length {n} exceeds permissible {max}")));
        }
        Ok(n as usize)
    }
    pub fn lif<T: Scalar>(&mut self) -> StructuralResult<LifParams<T>> {
        let (beta, threshold) = (self.sc()?, self.sc()?);
        let reset = match self.u8()? {
            0 => ResetMode::Subtract,
            1 => ResetMode::Zero,
            2 => ResetMode::None,
            x => return Err(err(format!("unknown reset {x}"))),
        };
        Ok(LifParams {
            beta,
            threshold,
            reset,
        })
    }
    pub fn model<T: Scalar>(&mut self) -> StructuralResult<NeuronModel<T>> {
        Ok(match self.u8()? {
            0 => NeuronModel::Lif(self.lif()?),
            1 => NeuronModel::AdaptiveLif(AdaptiveLifParams {
                beta: self.sc()?,
                rho: self.sc()?,
                adapt_increment: self.sc()?,
                threshold: self.sc()?,
                reset: match self.u8()? {
                    0 => ResetMode::Subtract,
                    1 => ResetMode::Zero,
                    2 => ResetMode::None,
                    x => return Err(err(format!("unknown reset {x}"))),
                },
            }),
            2 => NeuronModel::Izhikevich(IzhikevichParams {
                a: self.sc()?,
                b: self.sc()?,
                c: self.sc()?,
                d: self.sc()?,
                threshold: self.sc()?,
                substeps: self.len(MAX_ITEMS)?,
            }),
            3 => NeuronModel::AdEx(AdExParams {
                c: self.sc()?,
                g_l: self.sc()?,
                e_l: self.sc()?,
                v_t: self.sc()?,
                delta_t: self.sc()?,
                a: self.sc()?,
                tau_w: self.sc()?,
                b: self.sc()?,
                v_reset: self.sc()?,
                v_peak: self.sc()?,
                dt: self.sc()?,
                exp_clip: self.sc()?,
                substeps: self.len(MAX_ITEMS)?,
            }),
            4 => NeuronModel::HodgkinHuxley(HodgkinHuxleyParams {
                c: self.sc()?,
                g_na: self.sc()?,
                g_k: self.sc()?,
                g_l: self.sc()?,
                e_na: self.sc()?,
                e_k: self.sc()?,
                e_l: self.sc()?,
                v_rest: self.sc()?,
                spike_threshold: self.sc()?,
                dt: self.sc()?,
                substeps: self.len(MAX_ITEMS)?,
            }),
            x => return Err(err(format!("unknown neuron model {x}"))),
        })
    }
    pub fn finished(&self) -> bool {
        self.at == self.b.len()
    }
}

/// Collection size limit when reading.
pub(crate) const MAX_ITEMS: usize = 1 << 22;
