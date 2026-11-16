//! Resistance to bad inputs and to panic in a neighboring thread: operations
//! report typed errors instead of unwrapping.

use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::Arc;

use neo_quantization::{quantize, DType, PackedScheme, QuantizeConfig, SwitchMode, WeightRegistry};

fn gguf(d: DType) -> QuantizeConfig {
    let mut cfg = QuantizeConfig::float();
    cfg.scheme = PackedScheme::Gguf(d);
    cfg
}

/// Panic in the thread holding the write-lock poisons `RwLock`. Register
/// obliged to remain working, and not panic in every next call.
#[test]
fn weight_registry_survives_a_poisoned_lock() {
    let reg = Arc::new(WeightRegistry::new(SwitchMode::Cached));
    reg.register("w", vec![0.5; 64], 64);

    let r2 = Arc::clone(&reg);
    // `activate` with a deliberately panicky scheme: panic occurs under write-lock.
    let crashed = std::thread::spawn(move || {
        let mut cfg = QuantizeConfig::float();
        cfg.scheme = PackedScheme::Gguf(DType::F32); // quantize() panics: not block format
        r2.activate("w", &cfg);
    })
    .join();
    assert!(crashed.is_err(), "the flow had to fall");

    // The registry is still working.
    assert_eq!(reg.len(), 1);
    assert!(reg.get_active("w").is_some());
    assert!(reg.activate("w", &QuantizeConfig::int8_per_group_32()));
}

/// NaN in input IQ1 used to fail sorting (`partial_cmp().unwrap()`).
#[test]
fn iq1_does_not_panic_on_nan_input() {
    let mut data: Vec<f32> = (0..256)
        .map(|i| ((i * 7 % 31) as f32 - 15.0) / 10.0)
        .collect();
    data[3] = f32::NAN;
    for d in [DType::IQ1_S, DType::IQ1_M] {
        let r = catch_unwind(AssertUnwindSafe(|| quantize(&data, &gguf(d), 0)));
        assert!(r.is_ok(), "{d:?}: panic on NaN in input");
    }
}
