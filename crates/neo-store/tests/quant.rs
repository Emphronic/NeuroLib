#![cfg(feature = "quantization")]

mod common;

use neo_quantization::dtype::DType as QType;
use neo_quantization::pack::gguf::quantize_gguf;
use neo_quantization::pack::kquant::quantize_kquant;
use neo_quantization::pack::tq::quantize_tq2_0;
use neo_quantization::pack::{PackedBuffer, TqPacked};
use neo_store::quant::gguf_to_packed;
use neo_store::*;

fn weights(n: usize) -> Vec<f32> {
    (0..n).map(|i| ((i as f32) * 0.37).sin() * 0.8).collect()
}

fn store_with(c: Compression, name: &str, kind: GgufKind, shape: &[u64], blocks: &[u8]) -> Store {
    let o = WriteOptions {
        compression: c,
        compress_chunk_bytes: 4096,
        ..WriteOptions::default()
    };
    let mut w = StoreWriter::in_memory(o).unwrap();
    w.add_raw(name, DType::Gguf(kind), shape, TensorRole::Weight, blocks)
        .unwrap();
    let (bytes, _) = w.finish().unwrap();
    Store::from_bytes(bytes).unwrap()
}

#[test]
fn simple_gguf_blocks_are_dequantized_by_neo_quantization() {
    let data = weights(256);
    for (q, kind) in [(QType::Q8_0, GgufKind::Q8_0), (QType::Q4_0, GgufKind::Q4_0)] {
        let packed = quantize_gguf(&data, q);
        let expected = neo_quantization::dequantize(&PackedBuffer::Gguf(packed.clone()));
        for c in [Compression::None, Compression::Always] {
            let s = store_with(c, "w", kind, &[4, 64], &packed.blocks);
            assert_eq!(s.read_dequantized_f32("w").unwrap(), expected, "{kind:?}");
            // rows of 64 elements = 2 blocks
            assert_eq!(
                s.read_rows_dequantized_f32("w", 1, 2).unwrap(),
                expected[64..192].to_vec()
            );
            assert!(matches!(s.read_packed("w").unwrap(), PackedBuffer::Gguf(_)));
        }
    }
}

#[test]
fn kquant_and_ternary_blocks_are_dequantized() {
    let data = weights(512);
    let kq = quantize_kquant(&data, QType::Q4_K);
    let expected = neo_quantization::dequantize(&PackedBuffer::KQuant(kq.clone()));
    let s = store_with(
        Compression::Always,
        "k",
        GgufKind::Q4_K,
        &[2, 256],
        &kq.blocks,
    );
    assert_eq!(s.read_dequantized_f32("k").unwrap(), expected);

    let tq = quantize_tq2_0(&data[..256]);
    let expected = neo_quantization::dequantize(&PackedBuffer::Tq(TqPacked::Tq2_0(tq.clone())));
    let s = store_with(Compression::None, "t", GgufKind::TQ2_0, &[256], &tq.blocks);
    assert_eq!(s.read_dequantized_f32("t").unwrap(), expected);
}

#[test]
fn float_tensors_and_errors() {
    let mut w = StoreWriter::in_memory(WriteOptions::default()).unwrap();
    w.add_f32("f", &[3], TensorRole::Weight, &[1.0, 2.0, 3.0])
        .unwrap();
    w.add_slice::<i32>("i", &[2], TensorRole::Other, &[1, 2])
        .unwrap();
    w.add_raw(
        "b",
        DType::Bits(2),
        &[4],
        TensorRole::Weight,
        &[0b1110_0100],
    )
    .unwrap();
    let (bytes, _) = w.finish().unwrap();
    let s = Store::from_bytes(bytes).unwrap();
    assert_eq!(s.read_dequantized_f32("f").unwrap(), vec![1.0, 2.0, 3.0]);
    assert!(matches!(
        s.read_packed("f").unwrap(),
        PackedBuffer::Float(_)
    ));
    assert!(s.read_dequantized_f32("i").is_err());
    assert!(s.read_dequantized_f32("b").is_err()); // N-bit tensors have separate scales
    assert!(s.read_packed("i").is_err());
}

#[test]
fn wrong_block_data_is_rejected() {
    assert!(gguf_to_packed(GgufKind::Q4_0, vec![0u8; 35], 64).is_err());
    assert!(gguf_to_packed(GgufKind::Q4_0, vec![0u8; 36], 40).is_err());
    assert!(gguf_to_packed(GgufKind::Q4_0, vec![0u8; 36], 64).is_ok());
}
