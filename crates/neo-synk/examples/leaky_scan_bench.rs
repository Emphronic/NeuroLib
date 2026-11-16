//! Benchmark fused leaky scan (CubeCL, wgpu) against `leaky_scan_reference`.
//!
//! Launch:
//!   cargo run --release -p neo-synk --features wgpu,fusion --example leaky_scan_bench
//!
//! Environment Variables:
//!   BENCH_DEVICE=default — default adapter (otherwise the first discrete GPU);
//!   WGPU_BACKEND=vulkan - force Vulkan if possible;
//!   BENCH_BURN_GPU=1 — additionally measure `leaky_scan_reference` on the GPU (ref-gpu column).
//!                            Disabled by default: on some drivers (for example RX 5500 XT)
//!                            pipelines of ordinary tensor operations Burn do not pass validation
//!                            wgpu and the process crashes; The fused core is not affected by this.
//!
//! Columns:
//!   fused — one CubeCL core: stream per pair (batch, feature), sequential in time;
//!   ref-gpu - `leaky_scan_reference` on the same wgpu (only with BENCH_BURN_GPU=1);
//!   ref-cpu - `leaky_scan_reference` on Flex (CPU).
//! The kernel is parallelized only by batch features, so when their product is small
//! (last lines) it loses in terms of GPU load - this can be seen in the table.
//!
//! The GPU path works: on the wgpu (discrete GPU) the fused core matches the standard
//! (max|diff| ≈ 7e-7) and faster than `ref-cpu` by ×16…×570 times; the best case is large
//! batch·features (≥ 8192 threads), the worst is batch=1, features=32 (32 threads).

use std::time::Instant;

use burn::tensor::{Device, DeviceKind, Tensor, TensorData};

use neo_synk::snn::gpu::leaky_scan_fused;
use neo_synk::snn::leaky_scan_reference;

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

fn time_gpu<T, F: FnMut() -> T>(device: &Device, reps: usize, mut f: F) -> f64 {
    for _ in 0..3 {
        let _ = f();
    }
    sync(device);
    let t = Instant::now();
    for _ in 0..reps {
        let _ = f();
    }
    sync(device);
    t.elapsed().as_secs_f64() / reps as f64
}

fn pick_device() -> Device {
    match std::env::var("BENCH_DEVICE").as_deref() {
        Ok("default") => Device::wgpu(DeviceKind::DefaultDevice),
        _ => Device::wgpu(DeviceKind::DiscreteGpu(0)),
    }
}

fn main() {
    let device = pick_device();
    let cdev = Device::flex();
    let ref_gpu_on = std::env::var("BENCH_BURN_GPU").as_deref() == Ok("1");
    println!("device: {device:?}");
    println!("beta = 0.9, mem0 = 0");
    if !ref_gpu_on {
        println!("(ref-gpu disabled; enable: BENCH_BURN_GPU=1)");
    }
    println!();
    println!(
        "   batch  steps  feats | threads |  fused ms | ref-gpu ms | ref-cpu ms | ref-gpu/fused | ref-cpu/fused | max|diff|"
    );

    for (b, t, f) in [
        (32usize, 128usize, 256usize),
        (32, 512, 256),
        (16, 1024, 512),
        (4, 4096, 128),
        (1, 8192, 32),
    ] {
        let x = gen_vec(b * t * f, 3);
        let beta_v = vec![0.9f32];
        let mem0_v = vec![0.0f32; b * f];

        let mk3 = |d: &Device| Tensor::<3>::from_data(TensorData::new(x.clone(), [b, t, f]), d);
        let mk2 = |d: &Device| Tensor::<2>::from_data(TensorData::new(mem0_v.clone(), [b, f]), d);
        let mk1 = |d: &Device| Tensor::<1>::from_data(TensorData::new(beta_v.clone(), [1]), d);
        let (xg, mg, bg) = (mk3(&device), mk2(&device), mk1(&device));
        let (xc, mc, bc) = (mk3(&cdev), mk2(&cdev), mk1(&cdev));

        let fu = time_gpu(&device, 20, || {
            leaky_scan_fused(xg.clone(), mg.clone(), bg.clone())
        });
        let rg = ref_gpu_on.then(|| {
            time_gpu(&device, 5, || {
                leaky_scan_reference(xg.clone(), mg.clone(), bg.clone())
            })
        });

        // CPU: one warm-up + 2 measurements.
        let _ = leaky_scan_reference(xc.clone(), mc.clone(), bc.clone());
        let t0 = Instant::now();
        for _ in 0..2 {
            let _ = leaky_scan_reference(xc.clone(), mc.clone(), bc.clone());
        }
        let rc = t0.elapsed().as_secs_f64() / 2.0;

        // accuracy: fused (GPU) vs reference on CPU
        let a = leaky_scan_fused(xg.clone(), mg.clone(), bg.clone())
            .into_data()
            .try_to_vec::<f32>()
            .unwrap();
        let r = leaky_scan_reference(xc.clone(), mc.clone(), bc.clone())
            .into_data()
            .try_to_vec::<f32>()
            .unwrap();
        let err = a
            .iter()
            .zip(&r)
            .map(|(p, q)| (p - q).abs())
            .fold(0.0f32, f32::max);
        let rg_ms = rg.map_or("-".to_string(), |v| format!("{:.3}", v * 1e3));
        let rg_x = rg.map_or("-".to_string(), |v| format!("x{:.1}", v / fu));

        println!(
            "  {b:>6} {t:>6} {f:>6} | {:>7} | {:>9.3} | {rg_ms:>10} | {:>10.3} | {rg_x:>13} | x{:>12.1} | {err:.1e}",
            b * f,
            fu * 1e3,
            rc * 1e3,
            rc / fu,
        );
    }
}
