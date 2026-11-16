//! Saving and loading STDP layers on top of the standard Burn store
//! (`burn::store::ModuleRecord`, burnpack format).
//!
//! # Why this is needed
//!
//! STDP layers ([`StdpLinear`](super::StdpLinear),
//! [`StdpConv1d`](super::StdpConv1d), [`StdpConv2d`](super::StdpConv2d))
//! are deliberately not `Module`s: learned weights are part of the EXPLICIT
//! state, not a parameter moved by the optimizer. Because of this,
//! `Module::into_record` does not "see" them, and learned weights were lost
//! on restart.
//!
//! # What is saved
//!
//! A single file contains everything needed to restore a layer without the
//! original configuration:
//!
//! * **weights** (`w` / `kernel`);
//! * **traces**: fast pre/post, slow (Triplet), eligibility
//!   (RewardModulated), as well as membrane and bias;
//! * **thresholds and hyperparameters**: `threshold`, `beta`, the STDP rule
//!   with all `a_*`/`tau_*`, the weight bounding mode, the initialization
//!   range and the layer geometry (channels, kernel, stride, padding,
//!   dilation, groups).
//!
//! "Layer + state" together are the records [`StdpLinearRecord`],
//! [`StdpConv1dRecord`], [`StdpConv2dRecord`]. These are ordinary structs
//! with public fields (they can be built and edited by hand); they are
//! converted to disk and bytes by `into_bytes` / `from_bytes` / `save` /
//! `load`.
//!
//! # Format layout
//!
//! The new Burn store saves only `Module` parameters and, when loading,
//! requires a module "skeleton" with the same shapes. The record is
//! therefore packed into a private `Module` of flat (1-D) tensors:
//!
//! * `version` — format version ([`RECORD_VERSION`]);
//! * `ints` — integers: record kind, geometry, rule/bound kinds and
//!   their parameter lengths (an `Int` tensor);
//! * `floats` — `threshold`, `beta`, `init_range`, rule and bound
//!   parameters, bias homeostasis;
//! * `shapes` — original shapes of the state tensors (per slot);
//! * slots `w`, `mem`, `pre_trace`, `post_trace`, `pre_trace_slow`,
//!   `post_trace_slow`, `eligibility`, `bias` — the tensors themselves,
//!   flattened.
//!
//! Loading makes three passes over the same bytes: first only the version
//! is read, then the metadata, then a skeleton is built with shapes from
//! `shapes` and all tensors are loaded. Any mismatch (foreign version,
//! truncated file, extra or missing tensors) yields
//! [`SnnError::Record`], not a panic.
//!
//! # Usage
//!
//! ```ignore
//! use neo_snn::prelude::*;
//!
//! // save
//! layer.to_record(&state).save("stdp_conv2d.bpk")?;
//!
//! // load: the layer and state are restored in full
//! let record = StdpConv2dRecord::load("stdp_conv2d.bpk", &device)?;
//! let (layer, state) = StdpConv2d::from_record(record)?;
//!
//! // the same through memory
//! let bytes = layer.to_record(&state).into_bytes()?;
//! let record = StdpConv2dRecord::from_bytes(bytes, &device)?;
//! ```
//!
//! # Format compatibility
//!
//! The [`StdpHyper::version`] field is the format version; loading a record
//! with a different version returns [`SnnError::Record`]. Each tensor's
//! shape is checked against the layer geometry in `from_record`, so a
//! corrupted or foreign file gives a clear error instead of a panic inside
//! Burn on the first `step`.

use burn::module::{Module, Param, ParamId};
use burn::store::ModuleRecord;
use burn::tensor::{Bytes, Device, Int, Tensor, TensorData};

use crate::error::{Result, SnnError};
use crate::functional::plasticity::WeightBound;

use super::stdp_conv1d::StdpConv1dState;
use super::stdp_conv2d::StdpConv2dState;
use super::stdp_linear::StdpState;

/// Current record format version.
pub const RECORD_VERSION: u32 = 1;

/// STDP layer hyperparameters as primitives suitable for serialization
/// (rule and bound enums are encoded as a numeric kind + parameters).
#[derive(Clone, Debug, PartialEq)]
pub struct StdpHyper {
    /// Record format version ([`RECORD_VERSION`]).
    pub version: u32,
    /// Spike threshold of the postsynaptic LIF.
    pub threshold: f32,
    /// Membrane decay β.
    pub beta: f32,
    /// Rule kind: 0 — Pair, 1 — Triplet, 2 — RewardModulated.
    pub rule_kind: u8,
    /// Rule parameters in the order of the variant's fields.
    pub rule_params: Vec<f32>,
    /// Weight bound kind: 0 — None, 1 — Hard, 2 — Soft.
    pub bound_kind: u8,
    /// `[w_min, w_max]` for Hard/Soft, empty for None.
    pub bound_params: Vec<f32>,
    /// Range of the uniform weight initialization.
    pub init_range: [f32; 2],
}

pub(crate) fn encode_bound(bound: WeightBound) -> (u8, Vec<f32>) {
    match bound {
        WeightBound::None => (0, Vec::new()),
        WeightBound::Hard { w_min, w_max } => (1, vec![w_min, w_max]),
        WeightBound::Soft { w_min, w_max } => (2, vec![w_min, w_max]),
    }
}

pub(crate) fn decode_bound(kind: u8, params: &[f32]) -> Result<WeightBound> {
    match (kind, params) {
        (0, []) => Ok(WeightBound::None),
        (1, &[w_min, w_max]) => Ok(WeightBound::Hard { w_min, w_max }),
        (2, &[w_min, w_max]) => Ok(WeightBound::Soft { w_min, w_max }),
        _ => Err(record_err(
            "WeightBound",
            format!(
                "invalid weight bound: kind {kind}, {} parameters",
                params.len()
            ),
        )),
    }
}

pub(crate) fn record_err(what: &'static str, reason: impl Into<String>) -> SnnError {
    SnnError::Record {
        what,
        reason: reason.into(),
    }
}

/// Format version check.
pub(crate) fn check_version(what: &'static str, hyper: &StdpHyper) -> Result<()> {
    if hyper.version == RECORD_VERSION {
        Ok(())
    } else {
        Err(record_err(
            what,
            format!(
                "format version {} is not supported (expected {RECORD_VERSION})",
                hyper.version
            ),
        ))
    }
}

/// Check a record tensor's shape against the expected one.
pub(crate) fn check_dims<const D: usize>(
    what: &'static str,
    field: &str,
    got: [usize; D],
    want: [usize; D],
) -> Result<()> {
    if got == want {
        Ok(())
    } else {
        Err(record_err(
            what,
            format!("{field}: shape {got:?}, expected {want:?}"),
        ))
    }
}

/// Check that optional tensors are consistent with the rule kind.
pub(crate) fn check_optional<const D: usize>(
    what: &'static str,
    field: &str,
    present: &Option<Tensor<D>>,
    expected: bool,
    rule_name: &str,
) -> Result<()> {
    match (present.is_some(), expected) {
        (true, false) => Err(record_err(
            what,
            format!("{field} is present, but rule {rule_name} does not use it"),
        )),
        (false, true) => Err(record_err(
            what,
            format!("{field} is missing, but rule {rule_name} requires it"),
        )),
        _ => Ok(()),
    }
}

/// Record of [`StdpLinear`](super::StdpLinear) together with its state.
#[derive(Clone, Debug)]
pub struct StdpLinearRecord {
    /// Number of input features.
    pub in_features: usize,
    /// Number of output features.
    pub out_features: usize,
    /// Thresholds and hyperparameters.
    pub hyper: StdpHyper,
    /// Weights, traces, membrane.
    pub state: StdpState,
}

/// Record of [`StdpConv1d`](super::StdpConv1d) together with its state.
#[derive(Clone, Debug)]
pub struct StdpConv1dRecord {
    /// Number of input channels.
    pub in_channels: usize,
    /// Number of output channels.
    pub out_channels: usize,
    /// Kernel size.
    pub kernel_size: usize,
    /// Stride.
    pub stride: usize,
    /// Thresholds and hyperparameters.
    pub hyper: StdpHyper,
    /// Kernel, traces, membrane.
    pub state: StdpConv1dState,
}

/// Record of [`StdpConv2d`](super::StdpConv2d) together with its state.
#[derive(Clone, Debug)]
pub struct StdpConv2dRecord {
    /// Number of input channels.
    pub c_in: usize,
    /// Number of output channels.
    pub c_out: usize,
    /// Kernel height.
    pub kernel_h: usize,
    /// Kernel width.
    pub kernel_w: usize,
    /// Convolution stride.
    pub stride: usize,
    /// Zero padding.
    pub padding: usize,
    /// Dilation.
    pub dilation: usize,
    /// Number of groups.
    pub groups: usize,
    /// Whether there is a bias.
    pub bias: bool,
    /// Bias homeostasis: empty or `[target_rate, lr]`.
    pub bias_homeostasis: Vec<f32>,
    /// Thresholds and hyperparameters.
    pub hyper: StdpHyper,
    /// Weights, traces, membrane, bias.
    pub state: StdpConv2dState,
}

// ---------------------------------------------------------------------
// Packing into flat tensors and (de)serialization via ModuleRecord.
// ---------------------------------------------------------------------

/// Number of state tensor slots in the format.
const SLOTS: usize = 8;
/// Maximum tensor rank in a slot.
const MAX_RANK: usize = 4;
/// Length of the integer metadata block.
const INTS_LEN: usize = 16;
/// Length of the floating-point metadata block.
const FLOATS_LEN: usize = 16;
/// Length of the shapes block: per slot, rank + `MAX_RANK` dimensions.
const SHAPES_LEN: usize = SLOTS * (1 + MAX_RANK);
/// Maximum number of rule parameters (Triplet has 8).
const MAX_RULE_PARAMS: usize = 8;
/// Upper bound on the number of elements of a single tensor when loading:
/// protection against a huge skeleton allocation from corrupted metadata.
const MAX_NUMEL: usize = 1 << 32;

// Indices in `ints`: 0 record kind, 1 rule_kind, 2 bound_kind, 3 number of
// rule parameters, 4 number of bound parameters, 5 number of homeostasis
// parameters, 6 bias flag, 7.. geometry (up to 8 numbers).
const I_KIND: usize = 0;
const I_RULE_KIND: usize = 1;
const I_BOUND_KIND: usize = 2;
const I_N_RULE: usize = 3;
const I_N_BOUND: usize = 4;
const I_N_HOMEO: usize = 5;
const I_BIAS: usize = 6;
const I_GEOM: usize = 7;

// Indices in `floats`: 0 threshold, 1 beta, 2..4 init_range,
// 4..12 rule parameters, 12..14 bound, 14..16 homeostasis.
const F_RULE: usize = 4;
const F_BOUND: usize = 12;
const F_HOMEO: usize = 14;

const KIND_LINEAR: i64 = 1;
const KIND_CONV1D: i64 = 2;
const KIND_CONV2D: i64 = 3;

// Slot numbers.
pub(crate) const S_W: usize = 0;
pub(crate) const S_MEM: usize = 1;
pub(crate) const S_PRE: usize = 2;
pub(crate) const S_POST: usize = 3;
pub(crate) const S_PRE_SLOW: usize = 4;
pub(crate) const S_POST_SLOW: usize = 5;
pub(crate) const S_ELIG: usize = 6;
pub(crate) const S_BIAS: usize = 7;

type Flat = Option<(Vec<usize>, Tensor<1>)>;

#[derive(Module, Debug)]
struct VersionPart {
    version: Param<Tensor<1, Int>>,
}

#[derive(Module, Debug)]
struct MetaPart {
    version: Param<Tensor<1, Int>>,
    ints: Param<Tensor<1, Int>>,
    floats: Param<Tensor<1>>,
    shapes: Param<Tensor<1, Int>>,
}

#[derive(Module, Debug)]
struct FullPart {
    version: Param<Tensor<1, Int>>,
    ints: Param<Tensor<1, Int>>,
    floats: Param<Tensor<1>>,
    shapes: Param<Tensor<1, Int>>,
    w: Option<Param<Tensor<1>>>,
    mem: Option<Param<Tensor<1>>>,
    pre_trace: Option<Param<Tensor<1>>>,
    post_trace: Option<Param<Tensor<1>>>,
    pre_trace_slow: Option<Param<Tensor<1>>>,
    post_trace_slow: Option<Param<Tensor<1>>>,
    eligibility: Option<Param<Tensor<1>>>,
    bias: Option<Param<Tensor<1>>>,
}

fn int_tensor(v: Vec<i64>, device: &Device) -> Param<Tensor<1, Int>> {
    let n = v.len();
    Param::initialized(
        ParamId::new(),
        Tensor::<1, Int>::from_data(TensorData::new(v, [n]), device),
    )
}

fn float_tensor(v: Vec<f32>, device: &Device) -> Param<Tensor<1>> {
    let n = v.len();
    Param::from_tensor(Tensor::<1>::from_data(TensorData::new(v, [n]), device))
}

fn read_ints(t: Tensor<1, Int>, what: &'static str) -> Result<Vec<i64>> {
    t.into_data()
        .convert::<i64>()
        .try_to_vec::<i64>()
        .map_err(|e| record_err(what, format!("integer metadata is unreadable: {e:?}")))
}

fn read_floats(t: Tensor<1>, what: &'static str) -> Result<Vec<f32>> {
    t.into_data()
        .convert::<f32>()
        .try_to_vec::<f32>()
        .map_err(|e| {
            record_err(
                what,
                format!("floating-point metadata is unreadable: {e:?}"),
            )
        })
}

/// Flatten a tensor into a slot (the shape is stored separately).
pub(crate) fn flatten<const D: usize>(t: &Tensor<D>) -> Flat {
    let dims = t.dims().to_vec();
    let n = dims.iter().product::<usize>();
    Some((dims, t.clone().reshape([n])))
}

pub(crate) fn flatten_opt<const D: usize>(t: &Option<Tensor<D>>) -> Flat {
    t.as_ref().and_then(flatten)
}

/// Restore a required tensor of rank `D` from a slot.
pub(crate) fn unflatten<const D: usize>(
    what: &'static str,
    field: &str,
    slot: Flat,
) -> Result<Tensor<D>> {
    unflatten_opt::<D>(what, field, slot)?
        .ok_or_else(|| record_err(what, format!("{field}: tensor is missing from the record")))
}

/// Restore an optional tensor of rank `D` from a slot.
pub(crate) fn unflatten_opt<const D: usize>(
    what: &'static str,
    field: &str,
    slot: Flat,
) -> Result<Option<Tensor<D>>> {
    match slot {
        None => Ok(None),
        Some((dims, flat)) => {
            let dims: [usize; D] = dims.as_slice().try_into().map_err(|_| {
                record_err(what, format!("{field}: rank {}, expected {D}", dims.len()))
            })?;
            Ok(Some(flat.reshape(dims)))
        }
    }
}

/// Unpacked record contents: metadata + flat tensors.
pub(crate) struct Packed {
    kind: i64,
    version: u32,
    ints: Vec<i64>,
    floats: Vec<f32>,
    pub(crate) slots: [Flat; SLOTS],
}

impl Packed {
    /// Start packing: hyperparameters, record kind and geometry.
    pub(crate) fn new(
        what: &'static str,
        kind: i64,
        hyper: &StdpHyper,
        bias: bool,
        homeostasis: &[f32],
        geometry: &[usize],
    ) -> Result<Self> {
        if hyper.rule_params.len() > MAX_RULE_PARAMS
            || hyper.bound_params.len() > 2
            || homeostasis.len() > 2
        {
            return Err(record_err(
                what,
                "too many rule/bound/homeostasis parameters",
            ));
        }
        if geometry.len() > INTS_LEN - I_GEOM {
            return Err(record_err(what, "too many geometry numbers"));
        }
        let mut ints = vec![0i64; INTS_LEN];
        ints[I_KIND] = kind;
        ints[I_RULE_KIND] = i64::from(hyper.rule_kind);
        ints[I_BOUND_KIND] = i64::from(hyper.bound_kind);
        ints[I_N_RULE] = hyper.rule_params.len() as i64;
        ints[I_N_BOUND] = hyper.bound_params.len() as i64;
        ints[I_N_HOMEO] = homeostasis.len() as i64;
        ints[I_BIAS] = i64::from(bias);
        for (k, &g) in geometry.iter().enumerate() {
            ints[I_GEOM + k] = g as i64;
        }
        let mut floats = vec![0f32; FLOATS_LEN];
        floats[0] = hyper.threshold;
        floats[1] = hyper.beta;
        floats[2..4].copy_from_slice(&hyper.init_range);
        floats[F_RULE..F_RULE + hyper.rule_params.len()].copy_from_slice(&hyper.rule_params);
        floats[F_BOUND..F_BOUND + hyper.bound_params.len()].copy_from_slice(&hyper.bound_params);
        floats[F_HOMEO..F_HOMEO + homeostasis.len()].copy_from_slice(homeostasis);
        Ok(Self {
            kind,
            version: hyper.version,
            ints,
            floats,
            slots: Default::default(),
        })
    }

    pub(crate) fn kind_linear() -> i64 {
        KIND_LINEAR
    }
    pub(crate) fn kind_conv1d() -> i64 {
        KIND_CONV1D
    }
    pub(crate) fn kind_conv2d() -> i64 {
        KIND_CONV2D
    }

    /// Hyperparameters, bias flag, homeostasis and geometry (`n_geom` numbers).
    pub(crate) fn meta(
        &self,
        what: &'static str,
        n_geom: usize,
    ) -> Result<(StdpHyper, bool, Vec<f32>, Vec<usize>)> {
        let count = |i: usize, max: usize, name: &str| -> Result<usize> {
            let n = self.ints[i];
            if (0..=max as i64).contains(&n) {
                Ok(n as usize)
            } else {
                Err(record_err(what, format!("{name}: invalid length {n}")))
            }
        };
        let n_rule = count(I_N_RULE, MAX_RULE_PARAMS, "rule_params")?;
        let n_bound = count(I_N_BOUND, 2, "bound_params")?;
        let n_homeo = count(I_N_HOMEO, 2, "bias_homeostasis")?;
        let small = |i: usize, name: &str| -> Result<u8> {
            u8::try_from(self.ints[i]).map_err(|_| {
                record_err(what, format!("{name}: value {} out of range", self.ints[i]))
            })
        };
        let hyper = StdpHyper {
            version: self.version,
            threshold: self.floats[0],
            beta: self.floats[1],
            rule_kind: small(I_RULE_KIND, "rule_kind")?,
            rule_params: self.floats[F_RULE..F_RULE + n_rule].to_vec(),
            bound_kind: small(I_BOUND_KIND, "bound_kind")?,
            bound_params: self.floats[F_BOUND..F_BOUND + n_bound].to_vec(),
            init_range: [self.floats[2], self.floats[3]],
        };
        let geometry = self.ints[I_GEOM..I_GEOM + n_geom]
            .iter()
            .map(|&g| {
                usize::try_from(g)
                    .map_err(|_| record_err(what, format!("geometry: negative value {g}")))
            })
            .collect::<Result<Vec<_>>>()?;
        Ok((
            hyper,
            self.ints[I_BIAS] != 0,
            self.floats[F_HOMEO..F_HOMEO + n_homeo].to_vec(),
            geometry,
        ))
    }

    fn into_part(self, device: &Device) -> Result<FullPart> {
        let mut shapes = vec![0i64; SHAPES_LEN];
        let mut it = self.slots.into_iter().enumerate();
        let mut flat: [Option<Param<Tensor<1>>>; SLOTS] = Default::default();
        for (i, slot) in &mut it {
            if let Some((dims, t)) = slot {
                let base = i * (1 + MAX_RANK);
                shapes[base] = dims.len() as i64;
                for (k, &d) in dims.iter().enumerate() {
                    shapes[base + 1 + k] = d as i64;
                }
                flat[i] = Some(Param::from_tensor(t));
            }
        }
        let [
            w,
            mem,
            pre_trace,
            post_trace,
            pre_trace_slow,
            post_trace_slow,
            eligibility,
            bias,
        ] = flat;
        Ok(FullPart {
            version: int_tensor(vec![i64::from(self.version)], device),
            ints: int_tensor(self.ints, device),
            floats: float_tensor(self.floats, device),
            shapes: int_tensor(shapes, device),
            w,
            mem,
            pre_trace,
            post_trace,
            pre_trace_slow,
            post_trace_slow,
            eligibility,
            bias,
        })
    }

    /// Pack into a `ModuleRecord` (burnpack): saving to bytes or a file.
    pub(crate) fn into_module_record(self, what: &'static str) -> Result<ModuleRecord> {
        // Take the device from the first state tensor; metadata goes there too.
        let device = self
            .slots
            .iter()
            .flatten()
            .next()
            .map(|(_, t)| t.device())
            .ok_or_else(|| record_err(what, "no state tensors in the record"))?;
        Ok(self.into_part(&device)?.into_record())
    }

    /// Serialize into burnpack bytes.
    pub(crate) fn into_bytes(self, what: &'static str) -> Result<Bytes> {
        self.into_module_record(what)?
            .into_bytes()
            .map_err(|e| record_err(what, e.to_string()))
    }

    /// Read burnpack bytes: version → metadata → tensors.
    pub(crate) fn from_bytes(what: &'static str, bytes: Bytes, device: &Device) -> Result<Self> {
        let rec = ModuleRecord::from_bytes(bytes).map_err(|e| record_err(what, e.to_string()))?;
        Self::from_module_record(what, rec, device)
    }

    /// Read a `ModuleRecord`: version → metadata → tensors.
    pub(crate) fn from_module_record(
        what: &'static str,
        rec: ModuleRecord,
        device: &Device,
    ) -> Result<Self> {
        let err = |e: burn::store::RecordError| record_err(what, e.to_string());

        // Pass 1: version only.
        let skeleton = VersionPart {
            version: int_tensor(vec![0], device),
        };
        let part = skeleton
            .try_load_record(rec.clone().allow_unused(true))
            .map_err(err)?;
        let version = read_ints(part.version.val(), what)?;
        let version = u32::try_from(version[0]).unwrap_or(u32::MAX);
        if version != RECORD_VERSION {
            return Err(record_err(
                what,
                format!("format version {version} is not supported (expected {RECORD_VERSION})"),
            ));
        }

        // Pass 2: metadata.
        let skeleton = MetaPart {
            version: int_tensor(vec![0], device),
            ints: int_tensor(vec![0; INTS_LEN], device),
            floats: float_tensor(vec![0.0; FLOATS_LEN], device),
            shapes: int_tensor(vec![0; SHAPES_LEN], device),
        };
        let meta = skeleton
            .try_load_record(rec.clone().allow_unused(true))
            .map_err(err)?;
        let ints = read_ints(meta.ints.val(), what)?;
        let floats = read_floats(meta.floats.val(), what)?;
        let shapes = read_ints(meta.shapes.val(), what)?;

        // Pass 3: skeleton with shapes from metadata and all tensors.
        let mut dims_of: Vec<Option<Vec<usize>>> = Vec::with_capacity(SLOTS);
        for i in 0..SLOTS {
            let base = i * (1 + MAX_RANK);
            let rank = shapes[base];
            if !(0..=MAX_RANK as i64).contains(&rank) {
                return Err(record_err(what, format!("slot {i}: invalid rank {rank}")));
            }
            if rank == 0 {
                dims_of.push(None);
                continue;
            }
            let dims = shapes[base + 1..base + 1 + rank as usize]
                .iter()
                .map(|&d| {
                    usize::try_from(d)
                        .map_err(|_| record_err(what, format!("slot {i}: negative dimension {d}")))
                })
                .collect::<Result<Vec<_>>>()?;
            let numel = dims
                .iter()
                .try_fold(1usize, |a, &d| a.checked_mul(d))
                .filter(|&n| n <= MAX_NUMEL);
            if numel.is_none() {
                return Err(record_err(
                    what,
                    format!("slot {i}: shape too large {dims:?}"),
                ));
            }
            dims_of.push(Some(dims));
        }
        let zeros = |i: usize| {
            dims_of[i].as_ref().map(|d| {
                Param::from_tensor(Tensor::<1>::zeros([d.iter().product::<usize>()], device))
            })
        };
        let skeleton = FullPart {
            version: int_tensor(vec![0], device),
            ints: int_tensor(vec![0; INTS_LEN], device),
            floats: float_tensor(vec![0.0; FLOATS_LEN], device),
            shapes: int_tensor(vec![0; SHAPES_LEN], device),
            w: zeros(S_W),
            mem: zeros(S_MEM),
            pre_trace: zeros(S_PRE),
            post_trace: zeros(S_POST),
            pre_trace_slow: zeros(S_PRE_SLOW),
            post_trace_slow: zeros(S_POST_SLOW),
            eligibility: zeros(S_ELIG),
            bias: zeros(S_BIAS),
        };
        let full = skeleton.try_load_record(rec).map_err(err)?;
        let parts = [
            full.w,
            full.mem,
            full.pre_trace,
            full.post_trace,
            full.pre_trace_slow,
            full.post_trace_slow,
            full.eligibility,
            full.bias,
        ];
        let mut slots: [Flat; SLOTS] = Default::default();
        for (i, (p, d)) in parts.into_iter().zip(dims_of).enumerate() {
            slots[i] = p.zip(d).map(|(p, d)| (d, p.val()));
        }
        Ok(Self {
            kind: ints[I_KIND],
            version,
            ints,
            floats,
            slots,
        })
    }

    /// Check that the file contains the expected layer kind.
    pub(crate) fn expect_kind(&self, what: &'static str, kind: i64) -> Result<()> {
        if self.kind == kind {
            Ok(())
        } else {
            Err(record_err(
                what,
                format!("the file contains a different layer kind ({})", self.kind),
            ))
        }
    }
}

// ---------------------------------------------------------------------
// Public record methods: bytes and file.
// ---------------------------------------------------------------------

fn path_err(what: &'static str, e: impl core::fmt::Display) -> SnnError {
    record_err(what, e.to_string())
}

impl StdpLinearRecord {
    fn pack(self) -> Result<Packed> {
        let mut p = Packed::new(
            "StdpLinear",
            Packed::kind_linear(),
            &self.hyper,
            false,
            &[],
            &[self.in_features, self.out_features],
        )?;
        let s = &self.state;
        p.slots[S_W] = flatten(&s.w);
        p.slots[S_MEM] = flatten(&s.mem);
        p.slots[S_PRE] = flatten(&s.pre_trace);
        p.slots[S_POST] = flatten(&s.post_trace);
        p.slots[S_PRE_SLOW] = flatten_opt(&s.pre_trace_slow);
        p.slots[S_POST_SLOW] = flatten_opt(&s.post_trace_slow);
        p.slots[S_ELIG] = flatten_opt(&s.eligibility);
        Ok(p)
    }

    fn unpack(what: &'static str, mut p: Packed) -> Result<Self> {
        p.expect_kind(what, Packed::kind_linear())?;
        let (hyper, _bias, _homeo, geom) = p.meta(what, 2)?;
        let state = super::stdp_linear::StdpState {
            w: unflatten::<2>(what, "w", std::mem::take(&mut p.slots[S_W]))?,
            mem: unflatten::<2>(what, "mem", std::mem::take(&mut p.slots[S_MEM]))?,
            pre_trace: unflatten::<2>(what, "pre_trace", std::mem::take(&mut p.slots[S_PRE]))?,
            post_trace: unflatten::<2>(what, "post_trace", std::mem::take(&mut p.slots[S_POST]))?,
            pre_trace_slow: unflatten_opt::<2>(
                what,
                "pre_trace_slow",
                std::mem::take(&mut p.slots[S_PRE_SLOW]),
            )?,
            post_trace_slow: unflatten_opt::<2>(
                what,
                "post_trace_slow",
                std::mem::take(&mut p.slots[S_POST_SLOW]),
            )?,
            eligibility: unflatten_opt::<2>(
                what,
                "eligibility",
                std::mem::take(&mut p.slots[S_ELIG]),
            )?,
        };
        Ok(Self {
            in_features: geom[0],
            out_features: geom[1],
            hyper,
            state,
        })
    }

    /// Serialize the record into bytes (burnpack).
    pub fn into_bytes(self) -> Result<Bytes> {
        self.pack()?.into_bytes("StdpLinear")
    }

    /// Read the record from bytes; tensors are created on `device`.
    pub fn from_bytes(bytes: Bytes, device: &Device) -> Result<Self> {
        Self::unpack(
            "StdpLinear",
            Packed::from_bytes("StdpLinear", bytes, device)?,
        )
    }

    /// Save the record to a file (burnpack, recommended extension `.bpk`).
    pub fn save(self, path: impl AsRef<std::path::Path>) -> Result<()> {
        self.pack()?
            .into_module_record("StdpLinear")?
            .save(path)
            .map_err(|e| path_err("StdpLinear", e))
    }

    /// Load the record from a file; tensors are created on `device`.
    pub fn load(path: impl AsRef<std::path::Path>, device: &Device) -> Result<Self> {
        let rec = ModuleRecord::load(path).map_err(|e| path_err("StdpLinear", e))?;
        Self::unpack(
            "StdpLinear",
            Packed::from_module_record("StdpLinear", rec, device)?,
        )
    }
}

impl StdpConv1dRecord {
    fn pack(self) -> Result<Packed> {
        let mut p = Packed::new(
            "StdpConv1d",
            Packed::kind_conv1d(),
            &self.hyper,
            false,
            &[],
            &[
                self.in_channels,
                self.out_channels,
                self.kernel_size,
                self.stride,
            ],
        )?;
        let s = &self.state;
        p.slots[S_W] = flatten(&s.kernel);
        p.slots[S_MEM] = flatten(&s.mem);
        p.slots[S_PRE] = flatten(&s.pre_trace);
        p.slots[S_POST] = flatten(&s.post_trace);
        p.slots[S_PRE_SLOW] = flatten_opt(&s.pre_trace_slow);
        p.slots[S_POST_SLOW] = flatten_opt(&s.post_trace_slow);
        p.slots[S_ELIG] = flatten_opt(&s.eligibility);
        Ok(p)
    }

    fn unpack(what: &'static str, mut p: Packed) -> Result<Self> {
        p.expect_kind(what, Packed::kind_conv1d())?;
        let (hyper, _bias, _homeo, geom) = p.meta(what, 4)?;
        let state = super::stdp_conv1d::StdpConv1dState {
            kernel: unflatten::<3>(what, "kernel", std::mem::take(&mut p.slots[S_W]))?,
            mem: unflatten::<2>(what, "mem", std::mem::take(&mut p.slots[S_MEM]))?,
            pre_trace: unflatten::<2>(what, "pre_trace", std::mem::take(&mut p.slots[S_PRE]))?,
            post_trace: unflatten::<2>(what, "post_trace", std::mem::take(&mut p.slots[S_POST]))?,
            pre_trace_slow: unflatten_opt::<2>(
                what,
                "pre_trace_slow",
                std::mem::take(&mut p.slots[S_PRE_SLOW]),
            )?,
            post_trace_slow: unflatten_opt::<2>(
                what,
                "post_trace_slow",
                std::mem::take(&mut p.slots[S_POST_SLOW]),
            )?,
            eligibility: unflatten_opt::<2>(
                what,
                "eligibility",
                std::mem::take(&mut p.slots[S_ELIG]),
            )?,
        };
        Ok(Self {
            in_channels: geom[0],
            out_channels: geom[1],
            kernel_size: geom[2],
            stride: geom[3],
            hyper,
            state,
        })
    }

    /// Serialize the record into bytes (burnpack).
    pub fn into_bytes(self) -> Result<Bytes> {
        self.pack()?.into_bytes("StdpConv1d")
    }

    /// Read the record from bytes; tensors are created on `device`.
    pub fn from_bytes(bytes: Bytes, device: &Device) -> Result<Self> {
        Self::unpack(
            "StdpConv1d",
            Packed::from_bytes("StdpConv1d", bytes, device)?,
        )
    }

    /// Save the record to a file (burnpack, recommended extension `.bpk`).
    pub fn save(self, path: impl AsRef<std::path::Path>) -> Result<()> {
        self.pack()?
            .into_module_record("StdpConv1d")?
            .save(path)
            .map_err(|e| path_err("StdpConv1d", e))
    }

    /// Load the record from a file; tensors are created on `device`.
    pub fn load(path: impl AsRef<std::path::Path>, device: &Device) -> Result<Self> {
        let rec = ModuleRecord::load(path).map_err(|e| path_err("StdpConv1d", e))?;
        Self::unpack(
            "StdpConv1d",
            Packed::from_module_record("StdpConv1d", rec, device)?,
        )
    }
}

impl StdpConv2dRecord {
    fn pack(self) -> Result<Packed> {
        let mut p = Packed::new(
            "StdpConv2d",
            Packed::kind_conv2d(),
            &self.hyper,
            self.bias,
            &self.bias_homeostasis,
            &[
                self.c_in,
                self.c_out,
                self.kernel_h,
                self.kernel_w,
                self.stride,
                self.padding,
                self.dilation,
                self.groups,
            ],
        )?;
        let s = &self.state;
        p.slots[S_W] = flatten(&s.w);
        p.slots[S_MEM] = flatten(&s.mem);
        p.slots[S_PRE] = flatten(&s.pre_trace);
        p.slots[S_POST] = flatten(&s.post_trace);
        p.slots[S_PRE_SLOW] = flatten_opt(&s.pre_trace_slow);
        p.slots[S_POST_SLOW] = flatten_opt(&s.post_trace_slow);
        p.slots[S_ELIG] = flatten_opt(&s.eligibility);
        p.slots[S_BIAS] = flatten_opt(&s.bias);
        Ok(p)
    }

    fn unpack(what: &'static str, mut p: Packed) -> Result<Self> {
        p.expect_kind(what, Packed::kind_conv2d())?;
        let (hyper, bias, bias_homeostasis, g) = p.meta(what, 8)?;
        let state = super::stdp_conv2d::StdpConv2dState {
            w: unflatten::<2>(what, "w", std::mem::take(&mut p.slots[S_W]))?,
            mem: unflatten::<2>(what, "mem", std::mem::take(&mut p.slots[S_MEM]))?,
            pre_trace: unflatten::<2>(what, "pre_trace", std::mem::take(&mut p.slots[S_PRE]))?,
            post_trace: unflatten::<2>(what, "post_trace", std::mem::take(&mut p.slots[S_POST]))?,
            pre_trace_slow: unflatten_opt::<2>(
                what,
                "pre_trace_slow",
                std::mem::take(&mut p.slots[S_PRE_SLOW]),
            )?,
            post_trace_slow: unflatten_opt::<2>(
                what,
                "post_trace_slow",
                std::mem::take(&mut p.slots[S_POST_SLOW]),
            )?,
            eligibility: unflatten_opt::<2>(
                what,
                "eligibility",
                std::mem::take(&mut p.slots[S_ELIG]),
            )?,
            bias: unflatten_opt::<1>(what, "bias", std::mem::take(&mut p.slots[S_BIAS]))?,
        };
        Ok(Self {
            c_in: g[0],
            c_out: g[1],
            kernel_h: g[2],
            kernel_w: g[3],
            stride: g[4],
            padding: g[5],
            dilation: g[6],
            groups: g[7],
            bias,
            bias_homeostasis,
            hyper,
            state,
        })
    }

    /// Serialize the record into bytes (burnpack).
    pub fn into_bytes(self) -> Result<Bytes> {
        self.pack()?.into_bytes("StdpConv2d")
    }

    /// Read the record from bytes; tensors are created on `device`.
    pub fn from_bytes(bytes: Bytes, device: &Device) -> Result<Self> {
        Self::unpack(
            "StdpConv2d",
            Packed::from_bytes("StdpConv2d", bytes, device)?,
        )
    }

    /// Save the record to a file (burnpack, recommended extension `.bpk`).
    pub fn save(self, path: impl AsRef<std::path::Path>) -> Result<()> {
        self.pack()?
            .into_module_record("StdpConv2d")?
            .save(path)
            .map_err(|e| path_err("StdpConv2d", e))
    }

    /// Load the record from a file; tensors are created on `device`.
    pub fn load(path: impl AsRef<std::path::Path>, device: &Device) -> Result<Self> {
        let rec = ModuleRecord::load(path).map_err(|e| path_err("StdpConv2d", e))?;
        Self::unpack(
            "StdpConv2d",
            Packed::from_module_record("StdpConv2d", rec, device)?,
        )
    }
}
