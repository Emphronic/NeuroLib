use neo_store::*;

#[test]
fn bits_tensor_is_unpacked_to_codes() {
    let mut w = StoreWriter::in_memory(WriteOptions::default()).unwrap();
    w.add_raw(
        "tern",
        DType::Bits(2),
        &[5],
        TensorRole::Weight,
        &[0b1001_1000, 0b01],
    )
    .unwrap();
    w.add_f32("f", &[1], TensorRole::Weight, &[1.0]).unwrap();
    let (bytes, _) = w.finish().unwrap();
    let s = Store::from_bytes(bytes).unwrap();
    assert_eq!(s.read_bits("tern").unwrap(), vec![0, 2, 1, 2, 1]);
    assert!(matches!(
        s.read_bits("f"),
        Err(StoreError::DTypeMismatch { .. })
    ));
    assert!(matches!(
        s.read_f32("tern"),
        Err(StoreError::DTypeMismatch { .. })
    ));
}

#[test]
fn pack_then_store_then_unpack() {
    let codes: Vec<u8> = (0..101).map(|i| (i * 3 % 8) as u8).collect();
    let packed = pack_bits(&codes, 3).unwrap();
    let mut w = StoreWriter::in_memory(WriteOptions::default()).unwrap();
    w.add_raw("w", DType::Bits(3), &[101], TensorRole::Weight, &packed)
        .unwrap();
    let (bytes, _) = w.finish().unwrap();
    let s = Store::from_bytes(bytes).unwrap();
    assert_eq!(s.read_bits("w").unwrap(), codes);
}
