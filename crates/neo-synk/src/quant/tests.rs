//! Fused quantized GEMM (CPU) tests: the result must match the “full”
//! dequant + naive matmul" on all formats and shapes.

use super::cpu::M_BLOCK;
use super::*;
use neo_quantization::config::RoundingMode;
use neo_quantization::dtype::DType;
use neo_quantization::pack::{Calibration, PackedBuffer};
use neo_quantization::scheme::{PackedScheme, ScaleGranularity};
use neo_quantization::{QuantizeConfig, quantize};

// ───────────────────────── assistants ─────────────────────────

/// Deterministic generator of ~N(0,1)-like numbers (sum of 4 uniform ones).
fn gen_vec(len: usize, seed: u64) -> Vec<f32> {
    let mut s = seed
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    let mut next = move || {
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((s >> 40) as f32 / (1u64 << 24) as f32) - 0.5
    };
    (0..len)
        .map(|_| (next() + next() + next() + next()) * 1.7)
        .collect()
}

fn cfg(scheme: PackedScheme) -> QuantizeConfig {
    QuantizeConfig {
        scheme,
        rounding: RoundingMode::Nearest,
        symmetric: !matches!(
            scheme,
            PackedScheme::Int2(_, Calibration::Asymmetric)
                | PackedScheme::Int3(_, Calibration::Asymmetric)
                | PackedScheme::Int4(_, Calibration::Asymmetric)
                | PackedScheme::Int5(_, Calibration::Asymmetric)
                | PackedScheme::Int6(_, Calibration::Asymmetric)
                | PackedScheme::Int7(_, Calibration::Asymmetric)
                | PackedScheme::Int8(_, Calibration::Asymmetric)
        ),
        outlier_clip_percentile: 0.0,
    }
}

fn int(bits: u8, g: ScaleGranularity, c: Calibration) -> PackedScheme {
    match bits {
        2 => PackedScheme::Int2(g, c),
        3 => PackedScheme::Int3(g, c),
        4 => PackedScheme::Int4(g, c),
        5 => PackedScheme::Int5(g, c),
        6 => PackedScheme::Int6(g, c),
        7 => PackedScheme::Int7(g, c),
        8 => PackedScheme::Int8(g, c),
        _ => unreachable!(),
    }
}

/// Compare the fused core with the standard. Tolerance is relative to the scale of the result.
fn assert_matches_reference(w: &PackedBuffer, m: usize, n: usize, k: usize, label: &str) {
    let x = gen_vec(m * k, 7);
    let mut got = vec![0.0f32; m * n];
    packed_matmul(&x, m, w, n, k, &mut got).unwrap();
    let want = dequant_matmul_reference(&x, m, w, n, k);

    let scale = want.iter().fold(1e-3f32, |a, v| a.max(v.abs()));
    for (i, (g, e)) in got.iter().zip(&want).enumerate() {
        assert!(
            (g - e).abs() <= 2e-4 * scale + 1e-5,
            "{label}: [{i}] fused={g} reference={e} (scale {scale})"
        );
    }
}

fn pack(data: &[f32], scheme: PackedScheme, k: usize) -> PackedBuffer {
    quantize(data, &cfg(scheme), k)
}

// ───────────────────────── IntDomain ─────────────────────────

#[test]
fn intn_all_bits_symmetric_and_asymmetric_match_reference() {
    let (m, n, k) = (5, 9, 64);
    let w = gen_vec(n * k, 1);
    for bits in 2..=8u8 {
        for calib in [Calibration::Symmetric, Calibration::Asymmetric] {
            for g in [
                ScaleGranularity::PerGroup { group_size: 32 },
                ScaleGranularity::PerChannel,
                ScaleGranularity::PerTensor,
            ] {
                let p = pack(&w, int(bits, g, calib), k);
                assert_eq!(kernel_kind(&p, k), KernelKind::IntDomain);
                assert_matches_reference(&p, m, n, k, &format!("int{bits} {calib:?} {g:?}"));
            }
        }
    }
}

#[test]
fn int1_matches_reference() {
    let (m, n, k) = (4, 7, 96);
    let w = gen_vec(n * k, 2);
    for g in [
        ScaleGranularity::PerChannel,
        ScaleGranularity::PerGroup { group_size: 32 },
        ScaleGranularity::PerTensor,
    ] {
        let p = pack(&w, PackedScheme::Int1(g), k);
        assert_matches_reference(&p, m, n, k, &format!("int1 {g:?}"));
    }
}

/// Group does not divide `k`: groups cross row boundaries.
#[test]
fn groups_straddling_rows_match_reference() {
    let (m, n, k) = (3, 11, 64);
    let w = gen_vec(n * k, 3);
    for gs in [48usize, 7, 100, 1000] {
        for calib in [Calibration::Symmetric, Calibration::Asymmetric] {
            let p = pack(
                &w,
                int(4, ScaleGranularity::PerGroup { group_size: gs }, calib),
                k,
            );
            assert_matches_reference(&p, m, n, k, &format!("int4 gs={gs} {calib:?}"));
        }
    }
}

/// `k` is not a multiple of 8 and not a multiple of packing bytes - check for tails.
#[test]
fn odd_k_matches_reference() {
    let (m, n) = (3, 5);
    for k in [1usize, 3, 7, 13, 37, 101] {
        let w = gen_vec(n * k, 4);
        for bits in [3u8, 4, 5, 8] {
            let p = pack(
                &w,
                int(
                    bits,
                    ScaleGranularity::PerGroup { group_size: 16 },
                    Calibration::Asymmetric,
                ),
                k,
            );
            assert_matches_reference(&p, m, n, k, &format!("int{bits} k={k}"));
        }
        let p = pack(
            &w,
            PackedScheme::Nf4(ScaleGranularity::PerGroup { group_size: 16 }),
            k,
        );
        assert_matches_reference(&p, m, n, k, &format!("nf4 k={k}"));
    }
}

/// Long row: group and row are larger than `CHUNK`, the piece is cut several times.
#[test]
fn rows_longer_than_chunk_match_reference() {
    let (m, n, k) = (2, 3, 3000);
    let w = gen_vec(n * k, 5);
    for g in [
        ScaleGranularity::PerChannel,
        ScaleGranularity::PerTensor,
        ScaleGranularity::PerGroup { group_size: 128 },
    ] {
        for calib in [Calibration::Symmetric, Calibration::Asymmetric] {
            let p = pack(&w, int(4, g, calib), k);
            assert_matches_reference(&p, m, n, k, &format!("long int4 {g:?} {calib:?}"));
        }
    }
}

/// `m` is greater than `M_BLOCK`: several activation blocks.
#[test]
fn many_activation_rows_cross_m_blocks() {
    let (m, n, k) = (M_BLOCK * 2 + 13, 6, 64);
    let w = gen_vec(n * k, 6);
    let p = pack(
        &w,
        int(
            4,
            ScaleGranularity::PerGroup { group_size: 32 },
            Calibration::Asymmetric,
        ),
        k,
    );
    assert_matches_reference(&p, m, n, k, "m > M_BLOCK");
}

#[test]
fn single_row_decode_shape_matches_reference() {
    let (m, n, k) = (1, 64, 128);
    let w = gen_vec(n * k, 8);
    let p = pack(
        &w,
        int(
            4,
            ScaleGranularity::PerGroup { group_size: 32 },
            Calibration::Symmetric,
        ),
        k,
    );
    assert_matches_reference(&p, m, n, k, "m = 1");
}

#[test]
fn nf4_and_ternary_match_reference() {
    let (m, n, k) = (4, 8, 128);
    let w = gen_vec(n * k, 9);
    for g in [
        ScaleGranularity::PerGroup { group_size: 64 },
        ScaleGranularity::PerChannel,
        ScaleGranularity::PerTensor,
    ] {
        let nf4 = pack(&w, PackedScheme::Nf4(g), k);
        assert_eq!(kernel_kind(&nf4, k), KernelKind::IntDomain);
        assert_matches_reference(&nf4, m, n, k, &format!("nf4 {g:?}"));

        let t = pack(&w, PackedScheme::Ternary(g), k);
        assert_eq!(kernel_kind(&t, k), KernelKind::IntDomain);
        assert_matches_reference(&t, m, n, k, &format!("ternary {g:?}"));
    }
}

// ───────────────────────── RowDecode ─────────────────────────

#[test]
fn float_half_and_fp8_match_reference() {
    let (m, n, k) = (3, 5, 70);
    let w = gen_vec(n * k, 10);
    for s in [PackedScheme::Float] {
        let p = pack(&w, s, k);
        assert_eq!(kernel_kind(&p, k), KernelKind::RowDecode);
        assert_matches_reference(&p, m, n, k, "float");
    }
    let bf = PackedBuffer::Bf16(neo_quantization::pack::quantize_bf16(&w));
    let fp = PackedBuffer::Fp16(neo_quantization::pack::quantize_fp16(&w));
    let f8 = PackedBuffer::Fp8(neo_quantization::pack::quantize_fp8(
        &w,
        neo_quantization::pack::Fp8Variant::E4M3,
    ));
    for (p, l) in [(&bf, "bf16"), (&fp, "fp16"), (&f8, "fp8")] {
        assert_eq!(kernel_kind(p, k), KernelKind::RowDecode);
        assert_matches_reference(p, m, n, k, l);
    }
}

#[test]
fn gguf_simple_blocks_match_reference() {
    let (m, n, k) = (3, 4, 96); // 96 = 3 blocks of 32
    let w = gen_vec(n * k, 11);
    for dt in [
        DType::Q4_0,
        DType::Q4_1,
        DType::Q5_0,
        DType::Q5_1,
        DType::Q8_0,
    ] {
        let p = pack(&w, PackedScheme::Gguf(dt), k);
        assert_eq!(kernel_kind(&p, k), KernelKind::RowDecode, "{dt:?}");
        assert_matches_reference(&p, m, n, k, &format!("{dt:?}"));
    }
}

#[test]
fn gguf_long_rows_cross_chunk_boundary() {
    let (m, n, k) = (2, 2, 2048 + 64); // 2112 = 66 blocks, a piece is cut into 1024 + 1024 + 64
    let w = gen_vec(n * k, 12);
    let p = pack(&w, PackedScheme::Gguf(DType::Q4_0), k);
    assert_matches_reference(&p, m, n, k, "Q4_0 long");
}

#[test]
fn kquants_match_reference() {
    let (m, n, k) = (3, 3, 512); // 2 super-blocks per row
    let w = gen_vec(n * k, 13);
    for dt in [
        DType::Q2_K,
        DType::Q3_K,
        DType::Q4_K,
        DType::Q5_K,
        DType::Q6_K,
        DType::Q8_K,
    ] {
        let p = pack(&w, PackedScheme::Gguf(dt), k);
        assert_eq!(kernel_kind(&p, k), KernelKind::RowDecode, "{dt:?}");
        assert_matches_reference(&p, m, n, k, &format!("{dt:?}"));
    }
}

// ───────────────────────── DenseFallback and errors ─────────────────────────

#[test]
fn unsupported_row_layout_falls_back_to_dense_and_still_matches() {
    // k is not a multiple of the block GGUF → a progressive decoder is not possible.
    let (m, n, k) = (2, 2, 48);
    let w = gen_vec(n * k, 14);
    let p = pack(&w, PackedScheme::Gguf(DType::Q8_0), 32);
    // numel = 96 = 3 blocks; rows of 48 cross the blocks.
    assert_eq!(kernel_kind(&p, k), KernelKind::DenseFallback);
    assert_matches_reference(&p, m, n, k, "Q8_0 k=48 dense fallback");
}

#[test]
fn iq_series_uses_dense_fallback() {
    let (m, n, k) = (2, 2, 64);
    let w = gen_vec(n * k, 15);
    let p = pack(&w, PackedScheme::Gguf(DType::IQ4_NL), k);
    assert_eq!(kernel_kind(&p, k), KernelKind::DenseFallback);
    assert_matches_reference(&p, m, n, k, "IQ4_NL");
}

#[test]
fn shape_mismatches_are_reported() {
    let (n, k) = (4, 32);
    let w = gen_vec(n * k, 16);
    let p = pack(
        &w,
        int(4, ScaleGranularity::PerChannel, Calibration::Symmetric),
        k,
    );
    let x = gen_vec(2 * k, 17);
    let mut out = vec![0.0; 2 * n];

    // x is shorter
    assert!(matches!(
        packed_matmul(&x[..k], 2, &p, n, k, &mut out),
        Err(QuantGemmError::ShapeMismatch { .. })
    ));
    // out is the wrong size
    assert!(matches!(
        packed_matmul(&x, 2, &p, n, k, &mut out[..3]),
        Err(QuantGemmError::ShapeMismatch { .. })
    ));
    // packaging is not the right shape
    assert!(matches!(
        packed_matmul(&x, 2, &p, n + 1, k, &mut vec![0.0; 2 * (n + 1)]),
        Err(QuantGemmError::ShapeMismatch { .. })
    ));
}

#[test]
fn gguf_raw_is_unsupported() {
    let p = PackedBuffer::GgufRaw {
        bytes: vec![0; 18],
        dtype: DType::Q4_0,
        numel: 32,
    };
    let x = vec![0.0f32; 32];
    let mut out = vec![0.0f32; 1];
    assert!(matches!(
        packed_matmul(&x, 1, &p, 1, 32, &mut out),
        Err(QuantGemmError::Unsupported(_))
    ));
}

#[test]
fn empty_dimensions_give_zeros() {
    let (n, k) = (3, 32);
    let w = gen_vec(n * k, 18);
    let p = pack(
        &w,
        int(4, ScaleGranularity::PerChannel, Calibration::Symmetric),
        k,
    );
    let mut out: Vec<f32> = vec![];
    packed_matmul(&[], 0, &p, n, k, &mut out).unwrap(); // m = 0
}

/// The result is overwritten rather than accumulated on top of the old data.
#[test]
fn output_is_overwritten() {
    let (m, n, k) = (2, 3, 32);
    let w = gen_vec(n * k, 19);
    let p = pack(
        &w,
        int(8, ScaleGranularity::PerChannel, Calibration::Symmetric),
        k,
    );
    let x = gen_vec(m * k, 20);
    let mut a = vec![123.0f32; m * n];
    let mut b = vec![-7.0f32; m * n];
    packed_matmul(&x, m, &p, n, k, &mut a).unwrap();
    packed_matmul(&x, m, &p, n, k, &mut b).unwrap();
    assert_eq!(a, b);
}

// ───────────────────────── quality relative to float ─────────────────────────

/// The Fused core should not add errors beyond the quantization itself: on int8
/// the result is close to float matmul.
#[test]
fn int8_result_is_close_to_float_matmul() {
    let (m, n, k) = (4, 16, 256);
    let w = gen_vec(n * k, 21);
    let x = gen_vec(m * k, 22);

    let mut exact = vec![0.0f32; m * n];
    for i in 0..m {
        for r in 0..n {
            exact[i * n + r] = (0..k).map(|c| x[i * k + c] * w[r * k + c]).sum();
        }
    }

    let p = pack(
        &w,
        int(
            8,
            ScaleGranularity::PerGroup { group_size: 64 },
            Calibration::Symmetric,
        ),
        k,
    );
    let mut got = vec![0.0f32; m * n];
    packed_matmul(&x, m, &p, n, k, &mut got).unwrap();

    let norm = exact.iter().map(|v| v * v).sum::<f32>().sqrt();
    let err = got
        .iter()
        .zip(&exact)
        .map(|(a, b)| (a - b) * (a - b))
        .sum::<f32>()
        .sqrt();
    assert!(err / norm < 0.05, "relative L2 error {}", err / norm);
}

// ───────────────────────── GPU layout (kernel emulation on CPU) ─────────────────────────

mod gpu_layout_tests {
    use super::*;

    fn check_layout(p: &PackedBuffer, m: usize, n: usize, k: usize, label: &str) {
        let l = gpu_layout(p, n, k).unwrap_or_else(|| panic!("{label}: no gpu layout"));
        let x = gen_vec(m * k, 31);
        let got = emulate_gpu_matmul(&l, &x, m);
        let want = dequant_matmul_reference(&x, m, p, n, k);
        let scale = want.iter().fold(1e-3f32, |a, v| a.max(v.abs()));
        for (i, (g, e)) in got.iter().zip(&want).enumerate() {
            assert!(
                (g - e).abs() <= 2e-4 * scale + 1e-5,
                "{label}: [{i}] emu={g} ref={e}"
            );
        }
    }

    #[test]
    fn layout_emulation_matches_reference_for_all_int_bits() {
        let (m, n, k) = (3, 7, 64);
        let w = gen_vec(n * k, 30);
        for bits in 2..=8u8 {
            for calib in [Calibration::Symmetric, Calibration::Asymmetric] {
                for g in [
                    ScaleGranularity::PerGroup { group_size: 32 },
                    ScaleGranularity::PerChannel,
                    ScaleGranularity::PerTensor,
                ] {
                    let p = pack(&w, int(bits, g, calib), k);
                    check_layout(&p, m, n, k, &format!("int{bits} {calib:?} {g:?}"));
                }
            }
        }
        let p = pack(&w, PackedScheme::Int1(ScaleGranularity::PerChannel), k);
        check_layout(&p, m, n, k, "int1");
    }

    /// Codes crossing the `u32` boundary (3/5/6/7 bits), and groups across row boundaries.
    #[test]
    fn layout_handles_codes_crossing_word_boundaries_and_straddling_groups() {
        let (m, n, k) = (2, 9, 37);
        let w = gen_vec(n * k, 32);
        for bits in [3u8, 5, 6, 7] {
            for gs in [7usize, 48, 100] {
                let p = pack(
                    &w,
                    int(
                        bits,
                        ScaleGranularity::PerGroup { group_size: gs },
                        Calibration::Asymmetric,
                    ),
                    k,
                );
                check_layout(&p, m, n, k, &format!("int{bits} gs={gs}"));
            }
        }
    }

    #[test]
    fn layout_emulation_matches_reference_for_nf4_and_ternary() {
        let (m, n, k) = (3, 5, 96);
        let w = gen_vec(n * k, 33);
        for g in [
            ScaleGranularity::PerGroup { group_size: 32 },
            ScaleGranularity::PerTensor,
        ] {
            check_layout(&pack(&w, PackedScheme::Nf4(g), k), m, n, k, "nf4");
            check_layout(&pack(&w, PackedScheme::Ternary(g), k), m, n, k, "ternary");
        }
    }

    #[test]
    fn layout_is_none_for_unsupported_formats() {
        let (n, k) = (2, 64);
        let w = gen_vec(n * k, 34);
        assert!(gpu_layout(&pack(&w, PackedScheme::Float, k), n, k).is_none());
        assert!(gpu_layout(&pack(&w, PackedScheme::Gguf(DType::Q4_0), k), n, k).is_none());
        // incorrect shape
        let p = pack(
            &w,
            int(4, ScaleGranularity::PerChannel, Calibration::Symmetric),
            k,
        );
        assert!(gpu_layout(&p, n + 1, k).is_none());
    }
}

// ───────────────────────── GPU (CubeCL) ─────────────────────────
//
// Run: `cargo test -p neo-synk --features wgpu gpu_` (or `--features cuda`).
// The backend is selected in the same way as in Burn, with one type alias `G` below, and
// checks are generalized according to `B: QuantMatmulBackend`. // `wgpu`/`cuda`; The kernels themselves and `QuantMatmulBackend` are always compiled.
//
// For wgpu you need Vulkan/Metal/DX12; on a machine without a GPU - software Vulkan
// (Mesa lavapipe: `mesa-vulkan-drivers`).

// ───────────────────── GPU (CubeCL) ─────────────────────
// Run: `cargo test -p neo-synk --features wgpu gpu_` (on a machine without a GPU
// Vulkan software works - lavapipe).
#[cfg(any(feature = "wgpu", feature = "cuda", feature = "rocm"))]
mod gpu_tests {
    use super::*;
    use crate::quant::{quant_matmul_tensor, upload_weights};
    use burn::tensor::{Device, Tensor, TensorData};

    fn device() -> Device {
        #[cfg(feature = "cuda")]
        return Device::cuda(0usize);
        #[cfg(all(feature = "wgpu", not(feature = "cuda")))]
        return Device::wgpu(burn::tensor::DeviceKind::DefaultDevice);
        #[cfg(all(feature = "rocm", not(any(feature = "cuda", feature = "wgpu"))))]
        return Device::rocm(0usize);
    }

    fn run(device: &Device, p: &PackedBuffer, m: usize, n: usize, k: usize, x: &[f32]) -> Vec<f32> {
        let layout = gpu_layout(p, n, k).expect("gpu layout");
        let w = upload_weights(&layout, device);
        let xt = Tensor::<2>::from_data(TensorData::new(x.to_vec(), [m, k]), device);
        quant_matmul_tensor(xt, &w)
            .into_data()
            .try_to_vec::<f32>()
            .unwrap()
    }

    fn check(p: &PackedBuffer, m: usize, n: usize, k: usize, label: &str) {
        let x = gen_vec(m * k, 41);
        let got = run(&device(), p, m, n, k, &x);
        let want = dequant_matmul_reference(&x, m, p, n, k);
        assert_eq!(got.len(), want.len(), "{label}");
        let scale = want.iter().fold(1e-3f32, |a, v| a.max(v.abs()));
        for (i, (g, e)) in got.iter().zip(&want).enumerate() {
            assert!(
                (g - e).abs() <= 1e-3 * scale + 1e-4,
                "{label}: [{i}] gpu={g} ref={e} (scale {scale})"
            );
        }
    }

    #[test]
    fn gpu_matches_reference_for_all_int_bits() {
        let (m, n, k) = (4, 10, 64);
        let w = gen_vec(n * k, 40);
        for bits in 2..=8u8 {
            for calib in [Calibration::Symmetric, Calibration::Asymmetric] {
                for g in [
                    ScaleGranularity::PerGroup { group_size: 32 },
                    ScaleGranularity::PerChannel,
                    ScaleGranularity::PerTensor,
                ] {
                    check(
                        &pack(&w, int(bits, g, calib), k),
                        m,
                        n,
                        k,
                        &format!("int{bits} {calib:?} {g:?}"),
                    );
                }
            }
        }
        check(
            &pack(&w, PackedScheme::Int1(ScaleGranularity::PerChannel), k),
            m,
            n,
            k,
            "int1",
        );
    }

    #[test]
    fn gpu_handles_boundary_crossing_codes_and_straddling_groups() {
        let (m, n, k) = (3, 9, 37);
        let w = gen_vec(n * k, 42);
        for bits in [3u8, 5, 6, 7] {
            for gs in [7usize, 48] {
                let p = pack(
                    &w,
                    int(
                        bits,
                        ScaleGranularity::PerGroup { group_size: gs },
                        Calibration::Asymmetric,
                    ),
                    k,
                );
                check(&p, m, n, k, &format!("int{bits} gs={gs}"));
            }
        }
    }

    #[test]
    fn gpu_matches_reference_for_nf4_and_ternary() {
        let (m, n, k) = (3, 6, 96);
        let w = gen_vec(n * k, 43);
        let g = ScaleGranularity::PerGroup { group_size: 32 };
        check(&pack(&w, PackedScheme::Nf4(g), k), m, n, k, "nf4");
        check(&pack(&w, PackedScheme::Ternary(g), k), m, n, k, "ternary");
    }

    #[test]
    fn gpu_decode_shape_and_long_rows() {
        let w = gen_vec(8 * 2048, 44);
        let p = pack(
            &w,
            int(
                4,
                ScaleGranularity::PerGroup { group_size: 128 },
                Calibration::Symmetric,
            ),
            2048,
        );
        check(&p, 1, 8, 2048, "m=1 k=2048");
        check(&p, 33, 8, 2048, "m=33 k=2048");
    }
}
