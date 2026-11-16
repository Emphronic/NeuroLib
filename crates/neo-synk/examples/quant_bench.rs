//! Benchmark fused quantized GEMM versus “full dequant + matmul”.
//!
//! Launch: `cargo run --release -p neo-synk --example quant_bench`

use std::time::Instant;

use neo_quantization::pack::Calibration;
use neo_quantization::scheme::{PackedScheme, ScaleGranularity};
use neo_quantization::{QuantizeConfig, RoundingMode, quantize};
use neo_synk::quant::{kernel_kind, packed_matmul};

fn gen_vec(len: usize, seed: u64) -> Vec<f32> {
    let mut s = seed;
    (0..len)
        .map(|_| {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            ((s >> 40) as f32 / (1u64 << 24) as f32) - 0.5
        })
        .collect()
}

/// Naive “path 1”: dequant the entire matrix + single-threaded matmul.
fn dequant_then_matmul(
    x: &[f32],
    m: usize,
    w: &neo_quantization::PackedBuffer,
    n: usize,
    k: usize,
) -> Vec<f32> {
    let dense = w.to_f32();
    let mut out = vec![0.0f32; m * n];
    for i in 0..m {
        for r in 0..n {
            let (xr, wr) = (&x[i * k..(i + 1) * k], &dense[r * k..(r + 1) * k]);
            out[i * n + r] = xr.iter().zip(wr).map(|(a, b)| a * b).sum();
        }
    }
    out
}

fn main() {
    let (n, k) = (4096usize, 4096usize);
    let w = gen_vec(n * k, 1);
    println!(
        "W: {n}x{k} ({:.0} MiB in f32)",
        (n * k * 4) as f64 / 1048576.0
    );

    for (name, scheme) in [
        (
            "int4 sym  g32",
            PackedScheme::Int4(
                ScaleGranularity::PerGroup { group_size: 32 },
                Calibration::Symmetric,
            ),
        ),
        (
            "int4 asym g32",
            PackedScheme::Int4(
                ScaleGranularity::PerGroup { group_size: 32 },
                Calibration::Asymmetric,
            ),
        ),
        (
            "int8 sym  g32",
            PackedScheme::Int8(
                ScaleGranularity::PerGroup { group_size: 32 },
                Calibration::Symmetric,
            ),
        ),
        (
            "nf4       g64",
            PackedScheme::Nf4(ScaleGranularity::PerGroup { group_size: 64 }),
        ),
    ] {
        let cfg = QuantizeConfig {
            scheme,
            rounding: RoundingMode::Nearest,
            symmetric: true,
            outlier_clip_percentile: 0.0,
        };
        let packed = quantize(&w, &cfg, k);
        println!(
            "\n{name}: {:.2} bit/weight, {:.1} MiB, core {:?}",
            packed.effective_bits_per_weight(),
            packed.effective_bits_per_weight() as f64 * (n * k) as f64 / 8.0 / 1048576.0,
            kernel_kind(&packed, k)
        );

        for m in [1usize, 8, 64] {
            let x = gen_vec(m * k, 2);
            let mut out = vec![0.0f32; m * n];

            packed_matmul(&x, m, &packed, n, k, &mut out).unwrap(); // warming up
            let reps = if m == 1 { 20 } else { 5 };
            let t = Instant::now();
            for _ in 0..reps {
                packed_matmul(&x, m, &packed, n, k, &mut out).unwrap();
            }
            let fused = t.elapsed().as_secs_f64() / reps as f64;

            let t = Instant::now();
            let reference = dequant_then_matmul(&x, m, &packed, n, k);
            let naive = t.elapsed().as_secs_f64();

            let err = out
                .iter()
                .zip(&reference)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);
            println!(
                "  m={m:>3}: fused {:>7.1} ms | dequant+matmul {:>8.1} ms | x{:>5.1} | max|diff|={err:.1e}",
                fused * 1e3,
                naive * 1e3,
                naive / fused
            );
        }
    }
}
