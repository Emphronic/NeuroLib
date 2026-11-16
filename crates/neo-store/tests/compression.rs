mod common;

use common::TmpDir;
use neo_store::*;

fn opts(c: Compression) -> WriteOptions {
    WriteOptions {
        created_unix_ms: Some(0),
        compression: c,
        compress_chunk_bytes: 4096,
        ..WriteOptions::default()
    }
}

/// Thinned weights: Compresses well.
fn sparse(n: usize) -> Vec<f32> {
    (0..n)
        .map(|i| {
            if i % 10 == 0 {
                ((i % 97) as f32) * 0.125 - 6.0
            } else {
                0.0
            }
        })
        .collect()
}

/// Noise-like weights: not compressed (all bytes are random except the bit guaranteeing finiteness).
fn noisy(n: usize) -> Vec<f32> {
    let mut x = 0x2545_F491_4F6C_DD1Du64;
    (0..n)
        .map(|_| {
            x ^= x >> 12;
            x ^= x << 25;
            x ^= x >> 27;
            let bits = (x.wrapping_mul(0x9E37_79B9_7F4A_7C15) >> 32) as u32;
            f32::from_bits(bits & 0xBFFF_FFFF) // bit 30 cleared: order ≤ 127, value finite
        })
        .collect()
}

fn write_all(c: Compression) -> (Vec<u8>, WriteSummary) {
    let mut w = StoreWriter::in_memory(opts(c)).unwrap();
    w.add_f32("sparse", &[100, 64], TensorRole::Weight, &sparse(6400))
        .unwrap();
    w.add_f32_as(
        "half",
        &[64, 32],
        TensorRole::Weight,
        DType::F16,
        &sparse(2048),
    )
    .unwrap();
    w.add_slice::<i32>("ints", &[2000], TensorRole::Other, &vec![7i32; 2000])
        .unwrap();
    w.add_bool("mask", &[1000], TensorRole::Mask, &vec![true; 1000])
        .unwrap();
    w.add_raw(
        "q4",
        DType::Gguf(GgufKind::Q4_0),
        &[4096],
        TensorRole::Weight,
        &vec![9u8; 2304],
    )
    .unwrap();
    w.add_f32("noise", &[2048], TensorRole::Weight, &noisy(2048))
        .unwrap();
    w.add_scalar_f32("scalar", TensorRole::NeuronParam, 0.5)
        .unwrap();
    w.finish().unwrap()
}

#[test]
fn compressed_store_reads_back_identically() {
    let (plain, _) = write_all(Compression::None);
    let (packed, summary) = write_all(Compression::Always);
    assert!(packed.len() < plain.len());
    assert!(summary.stored_bytes < summary.data_bytes);

    let a = Store::from_bytes(plain).unwrap();
    let b = Store::from_bytes(packed).unwrap();
    assert_eq!(b.version(), (FORMAT_MAJOR, FORMAT_MINOR));
    for t in a.tensors() {
        assert_eq!(
            a.read_bytes(&t.name).unwrap().to_vec(),
            b.read_bytes(&t.name).unwrap().to_vec(),
            "{}",
            t.name
        );
        let tb = b.info(&t.name).unwrap();
        assert_eq!(
            tb.crc32c, t.crc32c,
            "CRC does not depend on compression: {}",
            t.name
        );
        assert_eq!(tb.nbytes, t.nbytes);
    }
    assert_eq!(a.fingerprint(), b.fingerprint());

    let sp = b.info("sparse").unwrap();
    assert!(sp.is_compressed());
    assert!(
        sp.stored_bytes < sp.nbytes / 2,
        "{} / {}",
        sp.stored_bytes,
        sp.nbytes
    );
    assert!(sp.stored_ratio() < 0.5);
    assert_eq!(b.read_f32("sparse").unwrap(), sparse(6400));
    assert_eq!(b.read_f32("half").unwrap(), a.read_f32("half").unwrap());
    assert_eq!(b.read_vec::<i32>("ints").unwrap(), vec![7i32; 2000]);
    // tensors shorter than 64 bytes remain as is
    assert_eq!(b.info("scalar").unwrap().encoding, Encoding::Raw);
    assert!(b.verify().unwrap().is_ok());
    assert!(b.summary().contains("in the file"));
    for t in b.tensors() {
        assert_eq!(t.offset % 64, 0, "{}", t.name);
    }
}

#[test]
fn auto_keeps_incompressible_tensors_raw() {
    let (bytes, _) = write_all(Compression::Auto);
    let s = Store::from_bytes(bytes).unwrap();
    assert!(s.info("sparse").unwrap().is_compressed());
    let noise = s.info("noise").unwrap();
    assert_eq!(noise.encoding, Encoding::Raw);
    assert_eq!(noise.stored_bytes, noise.nbytes);
    assert_eq!(s.read_f32("noise").unwrap(), noisy(2048));
}

#[test]
fn rows_of_a_compressed_tensor_are_read_partially() {
    let data: Vec<f32> = sparse(100 * 64);
    let mut w = StoreWriter::in_memory(opts(Compression::Always)).unwrap();
    w.add_f32("emb", &[100, 64], TensorRole::Weight, &data)
        .unwrap();
    let (bytes, _) = w.finish().unwrap();
    let s = Store::from_bytes(bytes).unwrap();
    assert!(s.info("emb").unwrap().is_compressed());
    // line - 256 B, chunk - 4096 B: ranges cross chunk boundaries
    for (first, rows) in [
        (0u64, 1u64),
        (0, 100),
        (15, 2),
        (3, 40),
        (99, 1),
        (50, 0),
        (16, 16),
    ] {
        let got = s.read_rows_f32("emb", first, rows).unwrap();
        let lo = (first * 64) as usize;
        let hi = lo + (rows * 64) as usize;
        assert_eq!(got, data[lo..hi].to_vec(), "{first}+{rows}");
    }
    assert!(s.read_rows_f32("emb", 99, 2).is_err());
}

#[test]
fn corruption_of_compressed_data_is_detected() {
    let (good, _) = write_all(Compression::Always);
    let s = Store::from_bytes(good.clone()).unwrap();
    let info = s.info("sparse").unwrap().clone();
    let (off, stored) = (info.offset as usize, info.stored_bytes as usize);

    // damage to body, table and trailer
    for at in [3usize, stored / 3, stored - 20, stored - 9, stored - 1] {
        let mut bad = good.clone();
        bad[off + at] ^= 0x5A;
        let s2 = Store::from_bytes(bad).unwrap();
        assert!(s2.read_f32("sparse").is_err(), "offset {at}");
        let rep = s2.verify().unwrap();
        assert_eq!(rep.failures.len(), 1, "offset {at}");
        assert_eq!(rep.failures[0].name, "sparse");
        assert!(
            s2.read_f32("half").is_ok(),
            "the neighboring tensor was not affected"
        );
    }
}

#[test]
fn file_roundtrip_with_and_without_mmap() {
    let dir = TmpDir::new("compressed");
    let path = dir.path("c.syn");
    let data = sparse(20_000);
    let summary = write_store_file(&path, opts(Compression::Auto), |w| {
        w.add_f32("w", &[200, 100], TensorRole::Weight, &data)
    })
    .unwrap();
    assert_eq!(std::fs::metadata(&path).unwrap().len(), summary.file_bytes);
    for use_mmap in [true, false] {
        let s = Store::open_with(
            &path,
            ReadOptions {
                use_mmap,
                verify_on_open: true,
                ..ReadOptions::default()
            },
        )
        .unwrap();
        assert!(s.info("w").unwrap().is_compressed());
        assert_eq!(s.read_f32("w").unwrap(), data);
        assert_eq!(
            s.read_rows_f32("w", 10, 5).unwrap(),
            data[1000..1500].to_vec()
        );
        assert_eq!(s.fingerprint(), summary.fingerprint);
    }
}

#[test]
fn compression_can_be_switched_per_tensor() {
    let mut w = StoreWriter::in_memory(opts(Compression::None)).unwrap();
    w.add_f32("a", &[2000], TensorRole::Weight, &sparse(2000))
        .unwrap();
    w.set_compression(Compression::Always);
    w.add_f32("b", &[2000], TensorRole::Weight, &sparse(2000))
        .unwrap();
    w.set_compression(Compression::None);
    w.add_f32("c", &[2000], TensorRole::Weight, &sparse(2000))
        .unwrap();
    let (bytes, _) = w.finish().unwrap();
    let s = Store::from_bytes(bytes).unwrap();
    assert!(!s.info("a").unwrap().is_compressed());
    assert!(s.info("b").unwrap().is_compressed());
    assert!(!s.info("c").unwrap().is_compressed());
    for n in ["a", "b", "c"] {
        assert_eq!(s.read_f32(n).unwrap(), sparse(2000));
    }
}

#[test]
fn streaming_from_an_iterator_compresses_too() {
    let data = sparse(10_000);
    let mut w = StoreWriter::in_memory(opts(Compression::Always)).unwrap();
    w.add_f32_iter(
        "it",
        &[10_000],
        TensorRole::Weight,
        DType::F32,
        data.iter().copied(),
    )
    .unwrap();
    let (bytes, _) = w.finish().unwrap();
    let s = Store::from_bytes(bytes).unwrap();
    assert!(s.info("it").unwrap().is_compressed());
    assert_eq!(s.read_f32("it").unwrap(), data);
}

#[test]
fn invalid_chunk_size_is_rejected() {
    for c in [0u32, 1000, 2048, 3 * 4096, 1 << 25] {
        let o = WriteOptions {
            compress_chunk_bytes: c,
            ..WriteOptions::default()
        };
        assert!(StoreWriter::in_memory(o).is_err(), "{c}");
    }
    let o = WriteOptions {
        compress_chunk_bytes: 1 << 16,
        ..WriteOptions::default()
    };
    assert!(StoreWriter::in_memory(o).is_ok());
}

#[test]
fn sharded_store_supports_compression() {
    let dir = TmpDir::new("sharded-compressed");
    let mut w = ShardedWriter::new(&dir.0, "m", 1 << 20, opts(Compression::Always)).unwrap();
    w.add_f32("a", &[3000], TensorRole::Weight, &sparse(3000))
        .unwrap();
    w.add_f32("b", &[3000], TensorRole::Weight, &sparse(3000))
        .unwrap();
    let summary = w.finish().unwrap();
    let s = ShardedStore::open(&summary.manifest_path).unwrap();
    assert_eq!(s.read_f32("a").unwrap(), sparse(3000));
    assert!(s.info("b").unwrap().is_compressed());
    assert!(s.verify().unwrap().is_ok());
}
