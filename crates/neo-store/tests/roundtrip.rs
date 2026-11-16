mod common;

use common::TmpDir;
use neo_store::*;

fn repro() -> WriteOptions {
    WriteOptions {
        created_unix_ms: Some(0),
        ..WriteOptions::default()
    }
}

fn sample_bytes() -> Vec<u8> {
    let mut w = StoreWriter::in_memory(repro()).unwrap();
    w.set_meta("model.name", "demo").unwrap();
    w.set_meta("train.steps", 1234u64).unwrap();
    let a: Vec<f32> = (0..12).map(|i| i as f32 * 0.5 - 2.0).collect();
    w.add_f32("a", &[3, 4], TensorRole::Weight, &a).unwrap();
    w.add_f32_as(
        "h",
        &[4],
        TensorRole::Bias,
        DType::F16,
        &[1.0, -2.0, 0.5, 0.0],
    )
    .unwrap();
    w.add_f32_as("b", &[2], TensorRole::Buffer, DType::BF16, &[1.0, 2.0])
        .unwrap();
    w.add_slice::<i32>("ints", &[3], TensorRole::Other, &[1, -2, 3])
        .unwrap();
    w.add_bool("mask", &[3], TensorRole::Mask, &[true, false, true])
        .unwrap();
    w.add_scalar_f32("scalar", TensorRole::NeuronParam, 0.9)
        .unwrap();
    w.add_raw(
        "tern",
        DType::Bits(2),
        &[5],
        TensorRole::Weight,
        &[0b1001_1000, 0b01],
    )
    .unwrap();
    w.add_raw(
        "q4",
        DType::Gguf(GgufKind::Q4_0),
        &[64],
        TensorRole::Weight,
        &[7u8; 36],
    )
    .unwrap();
    let (bytes, summary) = w.finish().unwrap();
    assert_eq!(summary.tensor_count, 8);
    assert_eq!(summary.file_bytes as usize, bytes.len());
    bytes
}

#[test]
fn memory_roundtrip_all_types() {
    let s = Store::from_bytes(sample_bytes()).unwrap();
    assert_eq!(s.version(), (FORMAT_MAJOR, FORMAT_MINOR));
    assert_eq!(
        s.meta_get("model.name").and_then(|v| v.as_str()),
        Some("demo")
    );
    assert_eq!(
        s.meta_get("train.steps").and_then(|v| v.as_u64()),
        Some(1234)
    );
    assert!(s.meta_get("syn.writer").is_some());

    let a = s.read_f32("a").unwrap();
    assert_eq!(a.len(), 12);
    assert_eq!(a[0], -2.0);
    assert_eq!(a[11], 3.5);
    assert_eq!(s.read_f32("h").unwrap(), vec![1.0, -2.0, 0.5, 0.0]);
    assert_eq!(s.read_f32("b").unwrap(), vec![1.0, 2.0]);
    assert_eq!(s.read_vec::<i32>("ints").unwrap(), vec![1, -2, 3]);
    assert_eq!(s.read_bytes("mask").unwrap().to_vec(), vec![1u8, 0, 1]);
    assert_eq!(s.info("scalar").unwrap().shape, Vec::<u64>::new());
    assert!((s.read_f32("scalar").unwrap()[0] - 0.9).abs() < 1e-7);
    assert_eq!(s.read_bytes("tern").unwrap().len(), 2);
    assert_eq!(s.read_bytes("q4").unwrap().len(), 36);

    // each tensor is aligned
    for t in s.tensors() {
        assert_eq!(t.offset % 64, 0, "{}", t.name);
    }
    // statistics without reading data
    let st = s.info("a").unwrap().stats.unwrap();
    assert_eq!((st.min, st.max, st.zeros), (-2.0, 3.5, 1));
    // quantized type is not readable as f32
    assert!(matches!(
        s.read_f32("q4"),
        Err(StoreError::DTypeMismatch { .. })
    ));
    assert!(matches!(
        s.read_vec::<f32>("ints"),
        Err(StoreError::DTypeMismatch { .. })
    ));
    assert!(matches!(
        s.read_f32("no"),
        Err(StoreError::TensorNotFound(_))
    ));
    assert!(s.verify().unwrap().is_ok());
    assert_eq!(s.parameter_count(), 12 + 4 + 1 + 5 + 64);
    assert!(s.summary().contains("model.name"));
}

#[test]
fn output_is_reproducible() {
    assert_eq!(sample_bytes(), sample_bytes());
}

#[test]
fn rows_can_be_read_partially() {
    let mut w = StoreWriter::in_memory(repro()).unwrap();
    let data: Vec<f32> = (0..20).map(|i| i as f32).collect();
    w.add_f32("emb", &[5, 4], TensorRole::Weight, &data)
        .unwrap();
    let (bytes, _) = w.finish().unwrap();
    let s = Store::from_bytes(bytes).unwrap();
    assert_eq!(
        s.read_rows_f32("emb", 2, 2).unwrap(),
        (8..16).map(|i| i as f32).collect::<Vec<_>>()
    );
    assert!(s.read_rows_f32("emb", 4, 2).is_err());
}

#[test]
fn iterator_and_slice_writers_agree() {
    let mut w = StoreWriter::in_memory(repro()).unwrap();
    let data: Vec<f32> = (0..100_000).map(|i| (i % 97) as f32).collect();
    w.add_f32("x", &[100_000], TensorRole::Weight, &data)
        .unwrap();
    w.add_f32_iter(
        "y",
        &[100_000],
        TensorRole::Weight,
        DType::F32,
        data.iter().copied(),
    )
    .unwrap();
    assert!(matches!(
        w.add_f32_iter(
            "z",
            &[10],
            TensorRole::Weight,
            DType::F32,
            data.iter().copied().take(9)
        ),
        Err(StoreError::ShapeMismatch { .. })
    ));
    let (bytes, _) = w.finish().unwrap();
    let s = Store::from_bytes(bytes).unwrap();
    assert_eq!(s.info("x").unwrap().crc32c, s.info("y").unwrap().crc32c);
    assert_eq!(s.read_f32("y").unwrap(), data);
}

#[test]
fn bad_writes_are_rejected() {
    let mut w = StoreWriter::in_memory(repro()).unwrap();
    w.add_f32("a", &[2], TensorRole::Weight, &[1.0, 2.0])
        .unwrap();
    assert!(matches!(
        w.add_f32("a", &[2], TensorRole::Weight, &[1.0, 2.0]),
        Err(StoreError::Duplicate(_))
    ));
    assert!(matches!(
        w.add_f32("b", &[3], TensorRole::Weight, &[1.0, 2.0]),
        Err(StoreError::ShapeMismatch { .. })
    ));
    assert!(matches!(
        w.add_f32("", &[1], TensorRole::Weight, &[1.0]),
        Err(StoreError::InvalidName(_))
    ));
    assert!(
        w.add_raw(
            "q",
            DType::Gguf(GgufKind::Q4_0),
            &[40],
            TensorRole::Weight,
            &[0; 40]
        )
        .is_err()
    );
    assert!(matches!(
        w.add_f32("n", &[1], TensorRole::Weight, &[f32::NAN]),
        Err(StoreError::NonFinite { index: 0, .. })
    ));
    // f16 overflows to inf - caught after type cast
    assert!(matches!(
        w.add_f32_as("o", &[1], TensorRole::Weight, DType::F16, &[70000.0]),
        Err(StoreError::NonFinite { .. })
    ));
    // Allow policy skips and counts in statistics
    w.set_non_finite_policy(NonFinitePolicy::Allow);
    w.add_f32("n2", &[2], TensorRole::State, &[f32::NEG_INFINITY, 1.0])
        .unwrap();
    let (bytes, _) = w.finish().unwrap();
    let s = Store::from_bytes(bytes).unwrap();
    assert_eq!(s.info("n2").unwrap().stats.unwrap().non_finite, 1);
    assert!(!s.contains("n"));
}

#[test]
fn corruption_is_detected() {
    let good = sample_bytes();
    let s = Store::from_bytes(good.clone()).unwrap();
    let off = s.info("a").unwrap().offset as usize;

    // corruption of tensor data
    let mut bad = good.clone();
    bad[off + 3] ^= 0x40;
    let s2 = Store::from_bytes(bad.clone()).unwrap();
    assert!(matches!(s2.read_f32("a"), Err(StoreError::Checksum { .. })));
    let rep = s2.verify().unwrap();
    assert_eq!(rep.failures.len(), 1);
    assert_eq!(rep.failures[0].name, "a");
    assert!(rep.into_result().is_err());
    assert!(
        s2.read_f32("h").is_ok(),
        "the neighboring tensor was not affected"
    );
    let lax = Store::from_bytes_with(
        bad.clone(),
        ReadOptions {
            verify_on_read: false,
            ..ReadOptions::default()
        },
    )
    .unwrap();
    assert!(lax.read_f32("a").is_ok());
    assert!(
        Store::from_bytes_with(
            bad,
            ReadOptions {
                verify_on_open: true,
                ..ReadOptions::default()
            }
        )
        .is_err()
    );

    // index corruption
    let n = good.len();
    let idx_off = u64::from_le_bytes(good[n - 64 + 8..n - 64 + 16].try_into().unwrap()) as usize;
    let mut bad = good.clone();
    bad[idx_off + 2] ^= 1;
    assert!(matches!(
        Store::from_bytes(bad),
        Err(StoreError::Checksum { .. })
    ));

    // truncation and appending
    assert!(Store::from_bytes(good[..n - 10].to_vec()).is_err());
    assert!(Store::from_bytes(good[..100].to_vec()).is_err());
    let mut longer = good.clone();
    longer.push(0);
    assert!(Store::from_bytes(longer).is_err());

    // someone else's file and text corruption of magic
    let mut bad = good.clone();
    bad[4] = b'\n';
    assert!(Store::from_bytes(bad).is_err());
    assert!(Store::from_bytes(vec![0u8; 200]).is_err());
}

#[test]
fn file_roundtrip_with_and_without_mmap() {
    let dir = TmpDir::new("file");
    let path = dir.path("m.syn");
    let data: Vec<f32> = (0..1000).map(|i| (i as f32).sin()).collect();
    let summary = write_store_file(&path, WriteOptions::default(), |w| {
        w.set_meta("k", true)?;
        w.add_f32("w", &[10, 100], TensorRole::Weight, &data)
    })
    .unwrap();
    assert_eq!(std::fs::metadata(&path).unwrap().len(), summary.file_bytes);
    assert!(!dir.path("m.syn.tmp").exists());

    for use_mmap in [true, false] {
        let s = Store::open_with(
            &path,
            ReadOptions {
                use_mmap,
                ..ReadOptions::default()
            },
        )
        .unwrap();
        assert_eq!(s.read_f32("w").unwrap(), data);
        assert!(s.verify().unwrap().is_ok());
        assert_eq!(s.fingerprint(), summary.fingerprint);
    }
}

#[test]
fn failed_write_keeps_the_old_file() {
    let dir = TmpDir::new("atomic");
    let path = dir.path("m.syn");
    write_store_file(&path, WriteOptions::default(), |w| {
        w.add_f32("w", &[1], TensorRole::Weight, &[1.0])
    })
    .unwrap();
    let before = std::fs::read(&path).unwrap();
    let r = write_store_file(&path, WriteOptions::default(), |w| {
        w.add_f32("w", &[1], TensorRole::Weight, &[2.0])?;
        w.add_f32("bad", &[1], TensorRole::Weight, &[f32::NAN])
    });
    assert!(r.is_err());
    assert_eq!(std::fs::read(&path).unwrap(), before);
    assert!(!dir.path("m.syn.tmp").exists());
}

#[test]
fn invalid_alignment_is_rejected() {
    for a in [0u32, 7, 12, 100_000] {
        let o = WriteOptions {
            alignment: a,
            ..WriteOptions::default()
        };
        assert!(StoreWriter::in_memory(o).is_err(), "{a}");
    }
    let o = WriteOptions {
        alignment: 4096,
        ..WriteOptions::default()
    };
    let mut w = StoreWriter::in_memory(o).unwrap();
    w.add_f32("a", &[1], TensorRole::Weight, &[1.0]).unwrap();
    w.add_f32("b", &[1], TensorRole::Weight, &[1.0]).unwrap();
    let (bytes, _) = w.finish().unwrap();
    let s = Store::from_bytes(bytes).unwrap();
    assert_eq!(s.info("b").unwrap().offset % 4096, 0);
}
