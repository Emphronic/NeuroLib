mod common;

use common::TmpDir;
use neo_store::*;

fn build(dir: &TmpDir) -> ShardedSummary {
    let mut w = ShardedWriter::new(&dir.0, "m", 50, WriteOptions::default()).unwrap();
    w.set_meta("model.name", "sharded").unwrap();
    w.add_f32("a", &[3, 4], TensorRole::Weight, &[1.0; 12])
        .unwrap();
    w.add_f32("b", &[2, 3], TensorRole::Weight, &[2.0; 6])
        .unwrap();
    w.add_f32_as("c", &[4], TensorRole::Bias, DType::F16, &[3.0; 4])
        .unwrap();
    let mut g = Graph::new();
    g.add_node(Node::input("in", &[4]))
        .add_node(Node::linear("fc1", "a"))
        .add_node(Node::linear("fc2", "b"))
        .add_node(Node::output("out", &[2]));
    g.chain(&["in", "fc1", "fc2", "out"]);
    w.set_graph(g).unwrap();
    w.finish().unwrap()
}

#[test]
fn sharded_roundtrip() {
    let dir = TmpDir::new("shard1");
    let summary = build(&dir);
    assert!(summary.shards.len() >= 3, "{:?}", summary.shards);
    assert_eq!(summary.tensor_count, 3);
    assert!(summary.manifest_path.exists());

    let s = ShardedStore::open(&summary.manifest_path).unwrap();
    assert_eq!(s.names(), vec!["a", "b", "c"]);
    assert_eq!(s.read_f32("a").unwrap(), vec![1.0; 12]);
    assert_eq!(s.read_f32("c").unwrap(), vec![3.0; 4]);
    assert_eq!(
        s.meta().get("model.name").and_then(|v| v.as_str()),
        Some("sharded")
    );
    assert!(s.validate_graph().unwrap().is_some());
    assert!(s.verify().unwrap().is_ok());
    assert!(matches!(
        s.read_f32("zzz"),
        Err(StoreError::TensorNotFound(_))
    ));
    // each shard is an independent storage
    let first = &summary.shards[0].file_name;
    assert!(Store::open(dir.path(first)).is_ok());
}

#[test]
fn tampered_shard_is_detected() {
    let dir = TmpDir::new("shard2");
    let summary = build(&dir);
    let victim = dir.path(&summary.shards[1].file_name);
    let mut bytes = std::fs::read(&victim).unwrap();
    bytes.push(0);
    std::fs::write(&victim, bytes).unwrap();
    assert!(ShardedStore::open(&summary.manifest_path).is_err());

    // lost shard
    let dir = TmpDir::new("shard3");
    let summary = build(&dir);
    std::fs::remove_file(dir.path(&summary.shards[0].file_name)).unwrap();
    assert!(ShardedStore::open(&summary.manifest_path).is_err());

    // damaged manifest
    let dir = TmpDir::new("shard4");
    let summary = build(&dir);
    let mut m = std::fs::read(&summary.manifest_path).unwrap();
    m[10] ^= 1;
    std::fs::write(&summary.manifest_path, m).unwrap();
    assert!(matches!(
        ShardedStore::open(&summary.manifest_path),
        Err(StoreError::Checksum { .. })
    ));
}

#[test]
fn duplicate_across_shards_and_bad_names() {
    let dir = TmpDir::new("shard5");
    let mut w = ShardedWriter::new(&dir.0, "m", 100, WriteOptions::default()).unwrap();
    w.add_f32("a", &[12], TensorRole::Weight, &[1.0; 12])
        .unwrap();
    w.add_f32("a", &[12], TensorRole::Weight, &[1.0; 12])
        .unwrap();
    assert!(matches!(w.finish(), Err(StoreError::Duplicate(_))));
    assert!(ShardedWriter::new(&dir.0, "a/b", 100, WriteOptions::default()).is_err());
    assert!(ShardedWriter::new(&dir.0, "ok", 0, WriteOptions::default()).is_err());
}
