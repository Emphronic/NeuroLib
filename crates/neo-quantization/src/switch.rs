// neo-quantization/src/switch.rs
//
// Fast switching of weight quantization scheme while the application is running
// (not to be confused with convert.rs - it does ONE-TIME offline conversion
// checkpoint; this file is about changing the scheme of an already loaded model in runtime,
// for example, for A/B quality comparison of ternary vs int8 without restarting).
//
// ## Two modes (see discussion with the user - both are needed, configurable)
//
//   `SwitchMode::OnTheFly`
//     We keep ONLY a float-master copy of each tensor in memory. When changing
//     schemes - repack from the master copy again. Slower for each
//     switching (full quantize() pass through all weights), but saves
//     memory: 1 copy of the scales in float + the current packed version.
//
//   `SwitchMode::Cached`
//     When accessing the scheme for the first time, we quantize and CACHE the packed version.
//     Repeated switching to an already seen scheme - instant swap
//     pointer (HashMap lookup), without recalculation. We pay in memory for each
//     ever requested scheme at the same time.
//
// In both modes the API is the same (`WeightRegistry::set_active_scheme`) -
// The mode only affects what happens “under the hood” when switching.

use std::collections::HashMap;
use std::sync::{RwLock, RwLockReadGuard, RwLockWriteGuard};

use crate::config::QuantizeConfig;
use crate::pack::PackedBuffer;
use crate::quantize::quantize;

// ═══════════════════════════════════════════════════════════════════════════
// § 1 SWITCHING MODE
// ═══════════════════════════════════════════════════════════════════════════

/// Switching strategy for the weight quantization scheme.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum SwitchMode {
    /// Repack float master copies every time you switch.
    /// Saves memory, it costs more time to change the scheme.
    OnTheFly,
    /// Cache every packed version ever requested. Instant
    /// switching between already seen schemes is more expensive in memory.
    Cached,
}

// ═══════════════════════════════════════════════════════════════════════════
// § 2 RECORDING ONE TENSOR
// ═══════════════════════════════════════════════════════════════════════════

/// State of one tensor in the registry: float master + optional cache
/// packed versions by config tag.
struct TensorEntry {
    /// Float master copy is the source of truth for repacking. Stored
    /// always (needed both for `OnTheFly` and for initially filling the cache
    /// in `Cached`), even if the active scheme is now quantized.
    master: Vec<f32>,
    /// `row_len`, needed for `ScaleGranularity::PerChannel` when repacking.
    row_len: usize,
    /// Cache of packed versions by scheme tag (`PackedScheme::tag()`). Used
    /// only in `Cached` mode - in `OnTheFly` it is always empty between calls
    /// (see `WeightRegistry::activate` - cleared immediately after the shift).
    cache: HashMap<&'static str, PackedBuffer>,
    /// Tag of the current active scheme (key in `cache`, or synthetic "f32"
    /// master-passthrough tag).
    active_tag: &'static str,
}

// ═══════════════════════════════════════════════════════════════════════════
// § 3 REGISTER
// ═══════════════════════════════════════════════════════════════════════════

/// Register of model weights with fast switching of the quantization scheme.
///
/// Thread safe (`RwLock` on internal storage) - scheme switching
/// and reading the active buffer can be called from different threads (for example,
/// the background thread re-quantizes the next scheme while the current forward pass
/// reads active).
pub struct WeightRegistry {
    mode: SwitchMode,
    entries: RwLock<HashMap<String, TensorEntry>>,
}

impl WeightRegistry {
    /// Read lock. Poisoned `RwLock` (panic in someone else's thread when
    /// record retention) does not make the registry unusable: the inside is normal
    /// `HashMap`, each record of which is inserted entirely, so the data
    /// remain consistent and we continue to work rather than panic
    /// every next call.
    fn read_entries(&self) -> RwLockReadGuard<'_, HashMap<String, TensorEntry>> {
        self.entries
            .read()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    /// Write lock; see [`Self::read_entries`] about poisoning.
    fn write_entries(&self) -> RwLockWriteGuard<'_, HashMap<String, TensorEntry>> {
        self.entries
            .write()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }

    /// A new empty registry with a specified switching mode.
    pub fn new(mode: SwitchMode) -> Self {
        WeightRegistry {
            mode,
            entries: RwLock::new(HashMap::new()),
        }
    }

    /// Register a tensor using float data. Initial active scheme -
    /// the float copy itself (passthrough) until it is explicitly activated
    /// another scheme via `activate`.
    pub fn register(&self, name: impl Into<String>, data: Vec<f32>, row_len: usize) {
        let entry = TensorEntry {
            master: data,
            row_len,
            cache: HashMap::new(),
            active_tag: "f32",
        };
        self.write_entries().insert(name.into(), entry);
    }

    /// Activate the given scheme for the tensor `name`. Returns `false`,
    /// if the tensor is not registered.
    ///
    /// The behavior depends on `self.mode`:
    ///   • `Cached` — if a version for this scheme is already in the cache, instant
    ///                  swap; otherwise, we quantize it once and add it to the cache.
    ///   • `OnTheFly` — we always quantize again from `master` (the cache does not grow).
    pub fn activate(&self, name: &str, config: &QuantizeConfig) -> bool {
        let tag = config.scheme.tag();
        let mut entries = self.write_entries();
        let Some(entry) = entries.get_mut(name) else {
            return false;
        };

        match self.mode {
            SwitchMode::Cached => {
                if !entry.cache.contains_key(tag) {
                    let packed = quantize(&entry.master, config, entry.row_len);
                    entry.cache.insert(tag, packed);
                }
                entry.active_tag = tag;
            }
            SwitchMode::OnTheFly => {
                let packed = quantize(&entry.master, config, entry.row_len);
                // In OnTheFly we keep no more than one packed version at a time -
                // We displace the old one so as not to consume memory beyond the active one.
                entry.cache.clear();
                entry.cache.insert(tag, packed);
                entry.active_tag = tag;
            }
        }
        true
    }

    /// Activate a given scheme for ALL registered tensors
    /// with one call - the typical scenario "switch the entire model to
    /// ternary for this run."
    pub fn activate_all(&self, config: &QuantizeConfig) {
        let names: Vec<String> = self.read_entries().keys().cloned().collect();
        for name in names {
            self.activate(&name, config);
        }
    }

    /// Get a copy of the active packed buffer for the tensor (clone -
    /// `PackedBuffer` is not `Copy`; for access without cloning on the hot path
    /// use `with_active`).
    pub fn get_active(&self, name: &str) -> Option<PackedBuffer> {
        self.with_active(name, |buf| buf.clone())
    }

    /// Execute `f` on the active buffer without cloning - holds
    /// read-lock for the duration of the call to `f`, so `f` should be fast and
    /// Do not try to take the write-lock of this registry yourself (otherwise deadlock).
    pub fn with_active<R>(&self, name: &str, f: impl FnOnce(&PackedBuffer) -> R) -> Option<R> {
        let entries = self.read_entries();
        let entry = entries.get(name)?;

        if entry.active_tag == "f32" {
            // Special case: active "scheme" - raw float passthrough,
            // not in cache (we avoid duplicating master in cache).
            let buf = PackedBuffer::Float(entry.master.clone());
            return Some(f(&buf));
        }

        entry.cache.get(entry.active_tag).map(f)
    }

    /// Number of registered tensors.
    pub fn len(&self) -> usize {
        self.read_entries().len()
    }

    /// `true` if the registry is empty.
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Free the entire cache of packed versions for the tensor, leaving only
    /// master + active scheme (useful after `activate_all` if there was
    /// experimental switching through many schemes in `Cached` mode -
    /// frees memory from versions that are no longer needed).
    pub fn trim_cache(&self, name: &str) {
        let mut entries = self.write_entries();
        if let Some(entry) = entries.get_mut(name) {
            let active = entry.active_tag;
            entry.cache.retain(|tag, _| *tag == active);
        }
    }

    /// The total size of all active packed buffers (not counting master) —
    /// a rough estimate of the model's current "working" footprint in memory.
    pub fn active_footprint_bytes(&self) -> usize {
        let entries = self.read_entries();
        entries
            .values()
            .filter_map(|e| e.cache.get(e.active_tag))
            .map(|buf| buf.data_bytes())
            .sum()
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// § 4 TESTS
// ═══════════════════════════════════════════════════════════════════════════

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_register_and_default_float() {
        let reg = WeightRegistry::new(SwitchMode::Cached);
        reg.register("w", vec![1.0, 2.0, 3.0], 0);
        let buf = reg.get_active("w").unwrap();
        assert!(matches!(buf, PackedBuffer::Float(_)));
    }

    #[test]
    fn test_activate_ternary_then_switch_back_cached() {
        let reg = WeightRegistry::new(SwitchMode::Cached);
        reg.register("w", vec![0.9, -0.9, 0.1, -0.1], 0);

        let ternary_cfg = QuantizeConfig::ternary_per_tensor();
        assert!(reg.activate("w", &ternary_cfg));
        let buf = reg.get_active("w").unwrap();
        assert!(matches!(buf, PackedBuffer::Ternary(_)));

        let int8_cfg = QuantizeConfig::int8_per_group_32();
        reg.activate("w", &int8_cfg);
        let buf2 = reg.get_active("w").unwrap();
        assert!(matches!(buf2, PackedBuffer::Int8(_)));

        // Switching back to ternary in Cached mode should not require
        // recalculation - we simply check that the result is correct (we cannot
        // directly verify “not recomputed”, so we check idempotency).
        reg.activate("w", &ternary_cfg);
        let buf3 = reg.get_active("w").unwrap();
        assert!(matches!(buf3, PackedBuffer::Ternary(_)));
    }

    #[test]
    fn test_on_the_fly_does_not_grow_cache() {
        let reg = WeightRegistry::new(SwitchMode::OnTheFly);
        reg.register("w", vec![0.9, -0.9, 0.1, -0.1], 0);

        reg.activate("w", &QuantizeConfig::ternary_per_tensor());
        reg.activate("w", &QuantizeConfig::int8_per_group_32());
        reg.activate("w", &QuantizeConfig::int4_per_group_32());

        // In OnTheFly, the cache always contains exactly 1 entry (active).
        let entries = reg.entries.read().unwrap();
        let entry = entries.get("w").unwrap();
        assert_eq!(entry.cache.len(), 1);
    }

    #[test]
    fn test_activate_unknown_tensor_returns_false() {
        let reg = WeightRegistry::new(SwitchMode::Cached);
        assert!(!reg.activate("nonexistent", &QuantizeConfig::ternary_per_tensor()));
    }

    #[test]
    fn test_activate_all() {
        let reg = WeightRegistry::new(SwitchMode::Cached);
        reg.register("a", vec![1.0, -1.0], 0);
        reg.register("b", vec![2.0, -2.0], 0);
        reg.activate_all(&QuantizeConfig::ternary_per_tensor());

        assert!(matches!(
            reg.get_active("a").unwrap(),
            PackedBuffer::Ternary(_)
        ));
        assert!(matches!(
            reg.get_active("b").unwrap(),
            PackedBuffer::Ternary(_)
        ));
    }

    #[test]
    fn test_trim_cache_keeps_only_active() {
        let reg = WeightRegistry::new(SwitchMode::Cached);
        reg.register("w", vec![0.5; 32], 0);
        reg.activate("w", &QuantizeConfig::ternary_per_tensor());
        reg.activate("w", &QuantizeConfig::int8_per_group_32());
        reg.trim_cache("w");

        let entries = reg.entries.read().unwrap();
        let entry = entries.get("w").unwrap();
        assert_eq!(entry.cache.len(), 1);
        assert_eq!(entry.active_tag, "int8");
    }

    #[test]
    fn test_with_active_no_clone_path() {
        let reg = WeightRegistry::new(SwitchMode::Cached);
        reg.register("w", vec![1.0, 2.0, 3.0, 4.0], 0);
        reg.activate("w", &QuantizeConfig::ternary_per_tensor());
        let numel = reg.with_active("w", |buf| buf.numel()).unwrap();
        assert_eq!(numel, 4);
    }
}
