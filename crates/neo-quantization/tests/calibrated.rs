//! Comparison of algorithms with calibration on the same data.
//!
//! This file is not only a test, but also a reproducible measurement: numbers
//! are printed, so `cargo test --test calibrated -- --nocapture`
//! shows exactly what each algorithm does on your machine.

use neo_quantization::algo::{
    awq_quantize, gptq_quantize, output_mse, round_to_nearest, AwqConfig, GptqConfig,
};

fn lcg(n: usize, seed: u64) -> Vec<f32> {
    let mut s = seed;
    (0..n)
        .map(|_| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((s >> 40) as f32) / ((1u64 << 24) as f32) - 0.5
        })
        .collect()
}

/// Activations similar to real ones: channels are correlated with each other and
/// some of them are systematically more active than others. Exactly this
/// GPTQ and AWQ structures give gains; there is nothing for both of them in white noise
/// use.
fn realistic_activations(n_samples: usize, in_f: usize, seed: u64) -> Vec<f32> {
    let base = lcg(n_samples * in_f, seed);
    let mut x = vec![0.0f32; n_samples * in_f];
    for s in 0..n_samples {
        for c in 0..in_f {
            let prev = if c == 0 { 0.0 } else { x[s * in_f + c - 1] };
            let outlier = if c % 16 == 0 { 12.0 } else { 1.0 };
            x[s * in_f + c] = (0.6 * prev + base[s * in_f + c]) * outlier;
        }
    }
    x
}

#[test]
fn gptq_and_awq_beat_plain_rounding() {
    let (out_f, in_f) = (32, 128);
    let bits = 4u8;
    let group = 32usize;

    let w = lcg(out_f * in_f, 1001);
    let x = realistic_activations(1024, in_f, 1002);

    let rtn = round_to_nearest(&w, out_f, in_f, bits, group);
    let gptq = gptq_quantize(
        &w,
        out_f,
        in_f,
        &x,
        &GptqConfig::int4_g128().with_group_size(group),
    )
    .expect("calibration is enough");
    let awq = awq_quantize(
        &w,
        out_f,
        in_f,
        &x,
        &AwqConfig::int4_g128().with_group_size(group),
    )
    .expect("calibration is enough");

    let e_rtn = output_mse(&w, &rtn, &x, out_f, in_f);
    let e_gptq = output_mse(&w, &gptq.weights, &x, out_f, in_f);
    let e_awq = output_mse(&w, &awq.weights, &x, out_f, in_f);

    println!("layer output error at 4 bits, group {group}:");
    println!("  round-to-nearest : {e_rtn:.6e}");
    println!(
        "  GPTQ : {e_gptq:.6e} ({:.1}% of base)",
        100.0 * e_gptq / e_rtn
    );
    println!(
        "  AWQ : {e_awq:.6e} ({:.1}% of base, α={})",
        100.0 * e_awq / e_rtn,
        awq.alpha
    );

    assert!(e_gptq < e_rtn, "GPTQ didn't handle rounding");
    assert!(e_awq < e_rtn, "AWQ did not beat rounding");
}

/// Both algorithms must refuse to work without calibration, and not silently
/// degenerate into ordinary rounding under its own name.
#[test]
fn both_algorithms_refuse_to_run_without_calibration() {
    let w = lcg(64, 1);
    assert!(gptq_quantize(&w, 4, 16, &[], &GptqConfig::default()).is_err());
    assert!(awq_quantize(&w, 4, 16, &[], &AwqConfig::default()).is_err());
}

/// The result of both algorithms remains within reasonable limits of the original weights.
/// A discrepancy of orders of magnitude would indicate an error in the reverse scaling.
#[test]
fn quantized_weights_stay_in_range() {
    let (out_f, in_f) = (16, 64);
    let w = lcg(out_f * in_f, 2001);
    let x = realistic_activations(512, in_f, 2002);
    let amax = w.iter().fold(0.0f32, |a, v| a.max(v.abs()));

    for (name, q) in [
        (
            "GPTQ",
            gptq_quantize(
                &w,
                out_f,
                in_f,
                &x,
                &GptqConfig::default().with_group_size(32),
            )
            .unwrap()
            .weights,
        ),
        (
            "AWQ",
            awq_quantize(
                &w,
                out_f,
                in_f,
                &x,
                &AwqConfig::default().with_group_size(32),
            )
            .unwrap()
            .weights,
        ),
    ] {
        let qmax = q.iter().fold(0.0f32, |a, v| a.max(v.abs()));
        assert!(
            q.iter().all(|v| v.is_finite()),
            "{name}: non-finite weights"
        );
        assert!(
            qmax <= amax * 2.0,
            "{name}: maximum increased from {amax} to {qmax} - looks like a reverse scaling error"
        );
    }
}
