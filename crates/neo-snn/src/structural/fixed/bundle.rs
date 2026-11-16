//! One file: structure masks + layer weights + arbitrary named tensors.
//!
//! `NFXB` v1 format (little-endian, FNV-1a checksum at the end):
//!
//! ```text
//! "NFXB" u32 version
//! u32 masks length · masks (NFXM format from FixedShapeController::save_masks)
//! u32 layer count · per layer: u64 n_w · f32×n_w · u8 has_bias · [u64 n_b · f32×n_b]
//! u32 tensor count · per tensor: u32 name length · UTF-8 name · u32 rank · u64×rank · f32×elements
//! u64 extra length · extra
//! u64 FNV-1a of everything above
//! ```
//!
//! The module does not depend on Burn: tensors come in and go out as flat `Vec<f32>`
//! (see `neo_snn::fixed_shape` for `Linear`).
//!
//! Why separate from `save_masks`: masks without weights are useless, and weights without masks do not
//! restore the network after structural changes. The file is written atomically (temporary file + rename).

use std::path::Path;

use super::controller::FixedShapeController;
use crate::structural::error::{StructuralError, StructuralResult};

const MAGIC: &[u8; 4] = b"NFXB";
/// Format version.
pub const BUNDLE_VERSION: u32 = 1;

fn err(m: impl Into<String>) -> StructuralError {
    StructuralError::Checkpoint(m.into())
}

fn fnv(data: &[u8]) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for b in data {
        h ^= *b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

/// Weights of one controlled layer.
#[derive(Debug, Clone, PartialEq)]
pub struct LayerBlob {
    /// `[in, out]` matrix, flat (`i*out + o`).
    pub weights: Vec<f32>,
    /// Bias `[out]`, if present.
    pub bias: Option<Vec<f32>>,
}

/// Arbitrary named tensor (recurrent matrix, β, …).
#[derive(Debug, Clone, PartialEq)]
pub struct NamedTensor {
    /// Name.
    pub name: String,
    /// Shape.
    pub shape: Vec<usize>,
    /// Data, flat.
    pub data: Vec<f32>,
}

/// File contents.
#[derive(Debug, Clone, PartialEq)]
pub struct Bundle {
    /// Masks (`NFXM` format).
    pub masks: Vec<u8>,
    /// Controller layers in order.
    pub layers: Vec<LayerBlob>,
    /// Other tensors.
    pub tensors: Vec<NamedTensor>,
    /// Caller data (optimizer, epoch number, …).
    pub extra: Vec<u8>,
}

impl Bundle {
    /// Assemble a bundle from the controller and weights. Dimensions are checked.
    pub fn capture(
        ctrl: &FixedShapeController,
        layers: Vec<LayerBlob>,
        tensors: Vec<NamedTensor>,
        extra: Vec<u8>,
    ) -> StructuralResult<Self> {
        check_layers(ctrl, &layers)?;
        for t in &tensors {
            let n: usize = t.shape.iter().product();
            if n != t.data.len() {
                return Err(StructuralError::InputSize {
                    op: "Bundle::capture(tensor)",
                    expected: n,
                    actual: t.data.len(),
                });
            }
        }
        Ok(Self {
            masks: ctrl.save_masks(),
            layers,
            tensors,
            extra,
        })
    }

    /// Find a tensor by name.
    pub fn tensor(&self, name: &str) -> Option<&NamedTensor> {
        self.tensors.iter().find(|t| t.name == name)
    }

    /// Restore the controller: load the masks and return the layer weights already projected onto the mask
    /// (and the bias of inactive neurons zeroed). If anything does not match — an error, the controller is untouched.
    pub fn restore(&self, ctrl: &mut FixedShapeController) -> StructuralResult<Vec<LayerBlob>> {
        check_layers(ctrl, &self.layers)?;
        let backup = ctrl.save_masks();
        ctrl.load_masks(&self.masks)?;
        let mut out = self.layers.clone();
        let res = (|| {
            for (l, blob) in out.iter_mut().enumerate() {
                ctrl.project_weights(l, &mut blob.weights)?;
                if let Some(b) = blob.bias.as_mut() {
                    ctrl.project_bias(l, b)?;
                }
            }
            Ok::<(), StructuralError>(())
        })();
        if let Err(e) = res {
            ctrl.load_masks(&backup)?;
            return Err(e);
        }
        Ok(out)
    }

    /// Encode.
    pub fn to_bytes(&self) -> Vec<u8> {
        let mut v = MAGIC.to_vec();
        v.extend_from_slice(&BUNDLE_VERSION.to_le_bytes());
        v.extend_from_slice(&(self.masks.len() as u32).to_le_bytes());
        v.extend_from_slice(&self.masks);
        v.extend_from_slice(&(self.layers.len() as u32).to_le_bytes());
        for l in &self.layers {
            let n_w = l.weights.len();
            v.extend_from_slice(&(n_w as u64).to_le_bytes());
            put_f32s(&mut v, &l.weights);
            match &l.bias {
                Some(b) => {
                    v.push(1);
                    v.extend_from_slice(&(b.len() as u64).to_le_bytes());
                    put_f32s(&mut v, b);
                }
                None => v.push(0),
            }
        }
        v.extend_from_slice(&(self.tensors.len() as u32).to_le_bytes());
        for t in &self.tensors {
            v.extend_from_slice(&(t.name.len() as u32).to_le_bytes());
            v.extend_from_slice(t.name.as_bytes());
            v.extend_from_slice(&(t.shape.len() as u32).to_le_bytes());
            for d in &t.shape {
                v.extend_from_slice(&(*d as u64).to_le_bytes());
            }
            put_f32s(&mut v, &t.data);
        }
        v.extend_from_slice(&(self.extra.len() as u64).to_le_bytes());
        v.extend_from_slice(&self.extra);
        let h = fnv(&v);
        v.extend_from_slice(&h.to_le_bytes());
        v
    }

    /// Decode and verify integrity.
    pub fn from_bytes(bytes: &[u8]) -> StructuralResult<Self> {
        if bytes.len() < 4 + 4 + 8 || &bytes[..4] != MAGIC {
            return Err(err("not NFXB"));
        }
        let (body, tail) = bytes.split_at(bytes.len() - 8);
        if fnv(body).to_le_bytes() != tail {
            return Err(err("checksum mismatch"));
        }
        let mut r = Reader { b: body, p: 4 };
        let ver = r.u32()?;
        if ver != BUNDLE_VERSION {
            return Err(err(format!(
                "version {ver} is not supported (expected {BUNDLE_VERSION})"
            )));
        }
        let ml = r.u32()? as usize;
        let masks = r.take(ml)?.to_vec();
        let n_layers = r.u32()? as usize;
        let mut layers = Vec::with_capacity(n_layers.min(1024));
        for _ in 0..n_layers {
            let n_w = r.u64()? as usize;
            let weights = r.f32s(n_w)?;
            let bias = if r.u8()? == 1 {
                let n_b = r.u64()? as usize;
                Some(r.f32s(n_b)?)
            } else {
                None
            };
            layers.push(LayerBlob { weights, bias });
        }
        let n_t = r.u32()? as usize;
        let mut tensors = Vec::with_capacity(n_t.min(1024));
        for _ in 0..n_t {
            let nl = r.u32()? as usize;
            let name = String::from_utf8(r.take(nl)?.to_vec())
                .map_err(|_| err("tensor name is not UTF-8"))?;
            let rank = r.u32()? as usize;
            if rank > 8 {
                return Err(err("rank too large"));
            }
            let shape: Vec<usize> = (0..rank)
                .map(|_| r.u64().map(|d| d as usize))
                .collect::<StructuralResult<_>>()?;
            let n = shape
                .iter()
                .try_fold(1usize, |a, &d| a.checked_mul(d))
                .ok_or_else(|| err("shape overflow"))?;
            tensors.push(NamedTensor {
                name,
                shape,
                data: r.f32s(n)?,
            });
        }
        let el = r.u64()? as usize;
        let extra = r.take(el)?.to_vec();
        if r.p != body.len() {
            return Err(err("trailing data at the end"));
        }
        Ok(Self {
            masks,
            layers,
            tensors,
            extra,
        })
    }

    /// Write atomically: to a temporary file nearby, then rename.
    pub fn save_to_file(&self, path: impl AsRef<Path>) -> StructuralResult<()> {
        let path = path.as_ref();
        let mut tmp = path.as_os_str().to_owned();
        tmp.push(".tmp");
        let tmp = std::path::PathBuf::from(tmp);
        std::fs::write(&tmp, self.to_bytes())
            .map_err(|e| err(format!("write {}: {e}", tmp.display())))?;
        std::fs::rename(&tmp, path).map_err(|e| {
            let _ = std::fs::remove_file(&tmp);
            err(format!("rename to {}: {e}", path.display()))
        })
    }

    /// Read from a file.
    pub fn load_from_file(path: impl AsRef<Path>) -> StructuralResult<Self> {
        let path = path.as_ref();
        let bytes =
            std::fs::read(path).map_err(|e| err(format!("read {}: {e}", path.display())))?;
        Self::from_bytes(&bytes)
    }
}

fn check_layers(ctrl: &FixedShapeController, layers: &[LayerBlob]) -> StructuralResult<()> {
    if layers.len() != ctrl.layers().len() {
        return Err(StructuralError::InputSize {
            op: "Bundle(layers)",
            expected: ctrl.layers().len(),
            actual: layers.len(),
        });
    }
    for (l, (blob, st)) in layers.iter().zip(ctrl.layers()).enumerate() {
        let sp = st.spec();
        if blob.weights.len() != sp.weight_len() {
            return Err(StructuralError::InputSize {
                op: "Bundle(layer weights)",
                expected: sp.weight_len(),
                actual: blob.weights.len(),
            });
        }
        if let Some(b) = &blob.bias {
            if b.len() != sp.out_features {
                return Err(StructuralError::InputSize {
                    op: "Bundle(layer bias)",
                    expected: sp.out_features,
                    actual: b.len(),
                });
            }
        }
        if blob
            .weights
            .iter()
            .chain(blob.bias.iter().flatten())
            .any(|v| !v.is_finite())
        {
            return Err(StructuralError::InvalidParameter {
                what: "weights",
                reason: format!("layer {l}: non-numeric value"),
            });
        }
    }
    Ok(())
}

fn put_f32s(v: &mut Vec<u8>, xs: &[f32]) {
    for x in xs {
        v.extend_from_slice(&x.to_le_bytes());
    }
}

struct Reader<'a> {
    b: &'a [u8],
    p: usize,
}

impl<'a> Reader<'a> {
    fn take(&mut self, n: usize) -> StructuralResult<&'a [u8]> {
        let end = self
            .p
            .checked_add(n)
            .filter(|&e| e <= self.b.len())
            .ok_or_else(|| err("file is truncated"))?;
        let s = &self.b[self.p..end];
        self.p = end;
        Ok(s)
    }
    fn u8(&mut self) -> StructuralResult<u8> {
        Ok(self.take(1)?[0])
    }
    fn u32(&mut self) -> StructuralResult<u32> {
        Ok(u32::from_le_bytes(
            self.take(4)?.try_into().expect("4 bytes"),
        ))
    }
    fn u64(&mut self) -> StructuralResult<u64> {
        Ok(u64::from_le_bytes(
            self.take(8)?.try_into().expect("8 bytes"),
        ))
    }
    fn f32s(&mut self, n: usize) -> StructuralResult<Vec<f32>> {
        let bytes = n.checked_mul(4).ok_or_else(|| err("size overflow"))?;
        Ok(self
            .take(bytes)?
            .chunks_exact(4)
            .map(|c| f32::from_le_bytes(c.try_into().expect("4 bytes")))
            .collect())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::structural::fixed::{AdaptInput, FixedShapeConfig, LayerRole, LayerSpec};

    fn ctrl() -> FixedShapeController {
        let mut cfg = FixedShapeConfig::default();
        cfg.schedule.min_interval_steps = 0;
        cfg.pruning.min_age_steps = 0;
        FixedShapeController::new(
            cfg,
            vec![
                LayerSpec::dense("h", 4, 5, LayerRole::Hidden)
                    .with_initial_active(3)
                    .with_density(0.6),
                LayerSpec::dense("o", 5, 2, LayerRole::Output).fed_by(0),
            ],
        )
        .unwrap()
    }

    fn blobs(c: &FixedShapeController) -> Vec<LayerBlob> {
        (0..2)
            .map(|l| {
                let sp = c.layers()[l].spec();
                let mut w = vec![0.0; sp.weight_len()];
                c.init_weights(l, &mut w, |i, o| 0.1 + 0.01 * (i * 7 + o) as f32)
                    .unwrap();
                LayerBlob {
                    weights: w,
                    bias: Some((0..sp.out_features).map(|o| 0.5 + o as f32).collect()),
                }
            })
            .collect()
    }

    #[test]
    fn roundtrip_bytes_and_file() {
        let c = ctrl();
        let tensors = vec![NamedTensor {
            name: "rec".into(),
            shape: vec![2, 3],
            data: vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0],
        }];
        let b = Bundle::capture(&c, blobs(&c), tensors, b"epoch=7".to_vec()).unwrap();
        assert_eq!(Bundle::from_bytes(&b.to_bytes()).unwrap(), b);
        let path = std::env::temp_dir().join(format!("neo_bundle_{}.nfxb", std::process::id()));
        b.save_to_file(&path).unwrap();
        assert!(
            !path.with_extension("nfxb.tmp").exists(),
            "temporary file removed"
        );
        assert_eq!(Bundle::load_from_file(&path).unwrap(), b);
        std::fs::remove_file(&path).ok();
        assert_eq!(b.tensor("rec").unwrap().shape, vec![2, 3]);
        assert!(b.tensor("missing").is_none());
    }

    #[test]
    fn restore_brings_masks_back_after_structure_changed() {
        let mut c = ctrl();
        let blobs0 = blobs(&c);
        let bundle = Bundle::capture(&c, blobs0.clone(), vec![], vec![]).unwrap();
        let masks_at_save = (c.effective_mask(0), c.effective_mask(1));
        // The structure changes: activate a reserve neuron (plateau) in another controller with the same layers.
        let mut other = ctrl();
        other
            .observe_activity(0, &[1.0; 4], &[0.3, 0.3, 0.3, 0.0, 0.0], 1)
            .unwrap();
        let w: Vec<&[f32]> = blobs0.iter().map(|b| b.weights.as_slice()).collect();
        for s in 0..12u64 {
            let _ = other.adapt(
                AdaptInput {
                    step: s,
                    sample: s,
                    loss: Some(1.0),
                },
                &w,
            );
        }
        assert_ne!(
            (other.effective_mask(0), other.effective_mask(1)),
            masks_at_save,
            "structure should have changed"
        );
        let restored = bundle.restore(&mut other).unwrap();
        assert_eq!(
            (other.effective_mask(0), other.effective_mask(1)),
            masks_at_save
        );
        // Weights are already projected: zeros outside the mask; bias of inactive neurons is zero.
        for (l, b) in restored.iter().enumerate() {
            let m = other.effective_mask(l);
            assert!(b.weights.iter().zip(&m).all(|(x, &mm)| mm || *x == 0.0));
        }
        let na = other.layers()[0].neuron_active().to_vec();
        for (o, &a) in na.iter().enumerate() {
            if !a {
                assert_eq!(restored[0].bias.as_ref().unwrap()[o], 0.0);
            }
        }
        let _ = &mut c;
    }

    #[test]
    fn corruption_truncation_and_shape_mismatch_are_typed_errors() {
        let c = ctrl();
        let b = Bundle::capture(&c, blobs(&c), vec![], vec![1, 2, 3]).unwrap();
        let bytes = b.to_bytes();
        let mut bad = bytes.clone();
        bad[bytes.len() / 2] ^= 0x40;
        assert!(matches!(
            Bundle::from_bytes(&bad),
            Err(StructuralError::Checkpoint(_))
        ));
        assert!(matches!(
            Bundle::from_bytes(&bytes[..bytes.len() - 9]),
            Err(StructuralError::Checkpoint(_))
        ));
        assert!(matches!(
            Bundle::from_bytes(b"XXXXXXXXXXXXXXXX"),
            Err(StructuralError::Checkpoint(_))
        ));
        // Foreign architecture: the weights do not fit.
        let mut wrong = FixedShapeController::new(
            FixedShapeConfig::default(),
            vec![LayerSpec::dense("x", 3, 3, LayerRole::Output)],
        )
        .unwrap();
        assert!(matches!(
            b.restore(&mut wrong),
            Err(StructuralError::InputSize { .. })
        ));
        // Wrong size during assembly and non-numeric values.
        let mut l = blobs(&c);
        l[0].weights.pop();
        assert!(matches!(
            Bundle::capture(&c, l, vec![], vec![]),
            Err(StructuralError::InputSize { .. })
        ));
        let mut l = blobs(&c);
        l[1].weights[0] = f32::NAN;
        assert!(matches!(
            Bundle::capture(&c, l, vec![], vec![]),
            Err(StructuralError::InvalidParameter { .. })
        ));
        let bt = vec![NamedTensor {
            name: "t".into(),
            shape: vec![2, 2],
            data: vec![0.0; 3],
        }];
        assert!(matches!(
            Bundle::capture(&c, blobs(&c), bt, vec![]),
            Err(StructuralError::InputSize { .. })
        ));
    }

    #[test]
    fn failed_restore_leaves_controller_untouched() {
        let c = ctrl();
        let mut b = Bundle::capture(&c, blobs(&c), vec![], vec![]).unwrap();
        let mut target = ctrl();
        let before = (target.effective_mask(0), target.effective_mask(1));
        b.masks[10] ^= 0xFF; // masks corrupted → load_masks will fail
        assert!(b.restore(&mut target).is_err());
        assert_eq!((target.effective_mask(0), target.effective_mask(1)), before);
    }
}
