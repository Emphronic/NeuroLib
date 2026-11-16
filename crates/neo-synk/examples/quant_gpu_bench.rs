//! Benchmark quantized GEMM on wgpu (Vulkan) against CPU and against dense mathmul on GPU.
//!
//! Launch:
//!   cargo run --release -p neo-synk --features wgpu,fusion --example quant_gpu_bench
//!
//! Environment Variables:
//!   BENCH_DEVICE=default — take the default adapter (otherwise the first discrete GPU);
//!   WGPU_BACKEND=vulkan - force Vulkan if possible (PowerShell: `$env:WGPU_BACKEND="vulkan"`);
//!   BENCH_N=1024 — size W (n = k), default 4096;
//!   BENCH_BATCH=1 - batch runs with one sync at the end (by default sync after
//!                            each call: burst mode drops RX 5500 XT);
//!   BENCH_BURN_GPU=1 - additionally measure the dense-mathmul Burn on the GPU (gpu-dense columns).
//!                            Disabled by default: on some drivers (for example RX 5500 XT)
//!                            matmul/Burn copy pipelines do not pass wgpu validation and
//!                            the process crashes; the quantized path is not affected.
//!
//! Columns (time per call, `y = x·Wᵀ`, W: 4096×4096):
//!   cpu - `packed_matmul` (rayon, fused by packaging);
//!   gpu-quant - CubeCL-kernel according to packed codes (weights on the device, x too);
//!   gpu-dense — `x.matmul(Wᵀ)` on GPUs with expanded f32 weights (only with BENCH_BURN_GPU=1).
//! Loading x and reading the result are not included in the measurements (except for a one-time accuracy check).
//!
//! GPU path works: on wgpu (discrete GPU, Vulkan, W 4096×4096) gpu-quant matches
//! with cpu (max|diff| ≤ ~1e-4) and faster than fused-CPU `packed_matmul` by ×2…×21 times
//! (int4/int8/nf4; winnings increase with m: m=1 ≈ ×2–3, m=256 ≈ ×10–21).

use std::time::Instant;

use burn::tensor::{Device, DeviceKind, Tensor, TensorData};

use neo_quantization::pack::Calibration;
use neo_quantization::scheme::{PackedScheme, ScaleGranularity};
use neo_quantization::{QuantizeConfig, RoundingMode, quantize};
use neo_synk::quant::{gpu_layout, packed_matmul, quant_matmul_tensor, upload_weights};

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

fn sync(device: &Device) {
    let _ = device.sync();
}

/// Average call time on GPU: warm up (with shader compilation), then `reps`
/// launches in a row and one sync at the end.
///
/// Default sync after EVERY call: on RX 5500 XT (wgpu 29 / CubeCL 0.10)
/// a long queue of runs without sync crashes the process with a validation error
/// "Buffer... is invalid." `BENCH_BATCH=1` - return batch mode (one sync
/// at the end; the call results are kept alive until sync).
fn time_gpu<T, F: FnMut() -> T>(device: &Device, reps: usize, mut f: F) -> f64 {
    let each = std::env::var("BENCH_BATCH").as_deref() != Ok("1");
    let mut keep = Vec::with_capacity(reps + 3);
    for _ in 0..3 {
        keep.push(f());
        if each {
            sync(device);
        }
    }
    sync(device);
    keep.clear();
    let t = Instant::now();
    for _ in 0..reps {
        keep.push(f());
        if each {
            sync(device);
        }
    }
    sync(device);
    let dt = t.elapsed().as_secs_f64() / reps as f64;
    drop(keep);
    dt
}

fn pick_device() -> Device {
    match std::env::var("BENCH_DEVICE").as_deref() {
        Ok("default") => Device::wgpu(DeviceKind::DefaultDevice),
        _ => Device::wgpu(DeviceKind::DiscreteGpu(0)),
    }
}

fn main() {
    let device = pick_device();
    let dense_on = std::env::var("BENCH_BURN_GPU").as_deref() == Ok("1");
    println!("device: {device:?}");
    if !dense_on {
        println!("(gpu-dense disabled; enable: BENCH_BURN_GPU=1)");
    }

    // BENCH_N=1024 - smaller size (diagnostics); default is 4096.
    let n: usize = std::env::var("BENCH_N")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(4096);
    let k = n;
    let w = gen_vec(n * k, 1);
    println!(
        "W: {n}x{k} ({:.1} MiB in f32)",
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
        let bits = packed.effective_bits_per_weight() as f64;
        let packed_bytes = bits * (n * k) as f64 / 8.0;
        println!(
            "\n{name}: {bits:.2} bit/weight, {:.1} MiB",
            packed_bytes / 1048576.0
        );

        // GPU: packed weights and dense standard (expanded once, outside of measurements).
        let layout = gpu_layout(&packed, n, k).expect("gpu layout");
        let t = Instant::now();
        let wq = upload_weights(&layout, &device);
        sync(&device);
        println!(
            "  loading packaging on GPU: {:.1} ms",
            t.elapsed().as_secs_f64() * 1e3
        );
        let w_dense = dense_on.then(|| {
            Tensor::<2>::from_data(TensorData::new(packed.to_f32(), [n, k]), &device).transpose()
        });

        println!(
            "        |    cpu ms | gpu-quant ms | gpu-dense ms | quant/cpu | quant/dense | GB/s(W) | max|diff| gpu vs cpu"
        );
        for m in [1usize, 8, 64, 256] {
            let x = gen_vec(m * k, 2);
            let xt = Tensor::<2>::from_data(TensorData::new(x.clone(), [m, k]), &device);

            // CPU
            let mut out = vec![0.0f32; m * n];
            packed_matmul(&x, m, &packed, n, k, &mut out).unwrap();
            let cpu_reps = match m {
                1 => 20,
                8 => 5,
                64 => 3,
                _ => 2,
            };
            let t = Instant::now();
            for _ in 0..cpu_reps {
                packed_matmul(&x, m, &packed, n, k, &mut out).unwrap();
            }
            let cpu = t.elapsed().as_secs_f64() / cpu_reps as f64;

            // GPU
            let gpu_reps = match m {
                1 => 50,
                8 => 30,
                64 => 10,
                _ => 5,
            };
            let gq = time_gpu(&device, gpu_reps, || quant_matmul_tensor(xt.clone(), &wq));
            let gd = w_dense
                .as_ref()
                .map(|wd| time_gpu(&device, gpu_reps, || xt.clone().matmul(wd.clone())));
            let gd_ms = gd.map_or("-".to_string(), |v| format!("{:.2}", v * 1e3));
            let gd_x = gd.map_or("-".to_string(), |v| format!("x{:.1}", v / gq));

            // accuracy (reading the result once)
            let got = quant_matmul_tensor(xt.clone(), &wq)
                .into_data()
                .try_to_vec::<f32>()
                .unwrap();
            let err = got
                .iter()
                .zip(&out)
                .map(|(a, b)| (a - b).abs())
                .fold(0.0f32, f32::max);

            println!(
                "  m={m:>3} | {:>9.2} | {:>12.2} | {gd_ms:>12} | x{:>8.1} | {gd_x:>11} | {:>7.1} | {err:.1e}",
                cpu * 1e3,
                gq * 1e3,
                cpu / gq,
                packed_bytes / gq / 1e9,
            );
        }
    }
}
