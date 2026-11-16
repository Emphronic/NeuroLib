mod common;

use common::TmpDir;
use neo_store::*;

fn lif_params(w: &mut StoreWriter<Vec<u8>>, prefix: &str, n: usize) {
    for (port, count) in [
        ("tau", n),
        ("r", 1),
        ("v_leak", n),
        ("v_reset", 1),
        ("v_threshold", n),
    ] {
        w.add_f32(
            &format!("{prefix}.{port}"),
            &[count as u64],
            TensorRole::NeuronParam,
            &vec![1.0; count],
        )
        .unwrap();
    }
}

fn mlp_graph(hidden: u64) -> Graph {
    let mut g = Graph::new();
    g.add_node(Node::input("in", &[4]))
        .add_node(Node::affine("fc1", "fc1.weight", "fc1.bias"))
        .add_node(Node::neuron("lif1", NodeOp::Lif, "lif1"))
        .add_node(Node::linear("fc2", "fc2.weight"))
        .add_node(Node::output("out", &[2]));
    g.chain(&["in", "fc1", "lif1", "fc2", "out"]);
    let _ = hidden;
    g
}

fn write_mlp(hidden: usize) -> StoreResult<(Vec<u8>, WriteSummary)> {
    let mut w = StoreWriter::in_memory(WriteOptions::default())?;
    w.add_f32(
        "fc1.weight",
        &[hidden as u64, 4],
        TensorRole::Weight,
        &vec![0.1; hidden * 4],
    )?;
    w.add_f32(
        "fc1.bias",
        &[hidden as u64],
        TensorRole::Bias,
        &vec![0.0; hidden],
    )?;
    lif_params(&mut w, "lif1", hidden);
    w.add_f32("fc2.weight", &[2, 8], TensorRole::Weight, &[0.2; 16])?;
    w.set_graph(mlp_graph(hidden as u64))?;
    w.finish()
}

#[test]
fn graph_is_stored_and_validated_against_weights() {
    let (bytes, _) = write_mlp(8).unwrap();
    let s = Store::from_bytes(bytes).unwrap();
    let g = s.graph().unwrap();
    assert_eq!(g.nodes.len(), 5);
    assert_eq!(g.node("lif1").unwrap().op, NodeOp::Lif);
    let shapes = s.validate_graph().unwrap().unwrap();
    assert_eq!(shapes.nodes["fc1"].output, vec![8]);
    assert_eq!(shapes.nodes["out"].output, vec![2]);
}

#[test]
fn mismatched_weights_fail_at_write_time() {
    // fc2.weight expects 8 inputs, and fc1 now outputs 6
    let err = write_mlp(6).unwrap_err();
    assert!(matches!(err, StoreError::Graph(_)), "{err}");
}

#[test]
fn missing_tensor_and_cycles_fail() {
    let mut w = StoreWriter::in_memory(WriteOptions::default()).unwrap();
    w.add_f32("fc1.weight", &[8, 4], TensorRole::Weight, &[0.0; 32])
        .unwrap();
    let mut g = Graph::new();
    g.add_node(Node::input("in", &[4]))
        .add_node(Node::linear("fc1", "fc1.weight"))
        .add_node(Node::linear("ghost", "ghost.weight"))
        .add_node(Node::output("out", &[8]));
    g.chain(&["in", "fc1", "out"]);
    g.connect("fc1", "ghost");
    w.set_graph(g).unwrap();
    assert!(matches!(w.finish(), Err(StoreError::Graph(_))));

    let mut g = Graph::new();
    g.add_node(Node::new("a", NodeOp::Custom("X".into())))
        .add_node(Node::new("b", NodeOp::Custom("X".into())));
    g.connect("a", "b");
    g.connect("b", "a");
    let mut w = StoreWriter::in_memory(WriteOptions::default()).unwrap();
    assert!(w.set_graph(g).is_err());
}

#[test]
fn recurrent_lif_layer_with_delay() {
    let mut w = StoreWriter::in_memory(WriteOptions::default()).unwrap();
    w.add_f32("ff.weight", &[8, 4], TensorRole::Weight, &[0.1; 32])
        .unwrap();
    w.add_f32("rec.weight", &[8, 8], TensorRole::Weight, &[0.0; 64])
        .unwrap();
    lif_params(&mut w, "lif", 8);
    let mut g = Graph::new();
    g.add_node(Node::input("in", &[4]))
        .add_node(Node::linear("ff", "ff.weight"))
        .add_node(Node::neuron("lif", NodeOp::Lif, "lif"))
        .add_node(Node::linear("rec", "rec.weight"))
        .add_node(Node::output("out", &[8]));
    g.chain(&["in", "ff", "lif", "out"]);
    g.connect_edge(Edge::new("lif", "rec").with_delay(1));
    g.connect_edge(Edge::new("rec", "lif").as_recurrent());
    w.set_graph(g).unwrap();
    let (bytes, _) = w.finish().unwrap();
    let s = Store::from_bytes(bytes).unwrap();
    let shapes = s.validate_graph().unwrap().unwrap();
    assert_eq!(shapes.nodes["rec"].output, vec![8]);
    assert_eq!(
        s.graph()
            .unwrap()
            .edges
            .iter()
            .filter(|e| e.recurrent)
            .count(),
        1
    );
}

#[test]
fn recurrent_edge_with_wrong_size_fails() {
    let mut w = StoreWriter::in_memory(WriteOptions::default()).unwrap();
    w.add_f32("ff.weight", &[8, 4], TensorRole::Weight, &[0.1; 32])
        .unwrap();
    w.add_f32("rec.weight", &[5, 8], TensorRole::Weight, &[0.0; 40])
        .unwrap();
    lif_params(&mut w, "lif", 8);
    let mut g = Graph::new();
    g.add_node(Node::input("in", &[4]))
        .add_node(Node::linear("ff", "ff.weight"))
        .add_node(Node::neuron("lif", NodeOp::Lif, "lif"))
        .add_node(Node::linear("rec", "rec.weight"))
        .add_node(Node::output("out", &[8]));
    g.chain(&["in", "ff", "lif", "out"]);
    g.connect("lif", "rec");
    g.connect_edge(Edge::new("rec", "lif").as_recurrent()); // rec gives 5, lif expects 8
    w.set_graph(g).unwrap();
    assert!(matches!(w.finish(), Err(StoreError::Graph(_))));
}

#[test]
fn sparse_roundtrip_and_graph_uses_logical_shape() {
    let dir = TmpDir::new("sparse");
    let path = dir.path("s.syn");
    let mut dense = vec![0f32; 6 * 5];
    dense[1] = 0.5;
    dense[8] = -1.5;
    dense[29] = 2.0;
    let csr = SparseCsr::from_dense(6, 5, &dense, 0.0).unwrap();
    write_store_file(&path, WriteOptions::default(), |w| {
        w.add_csr("fc.weight", &csr, TensorRole::Weight, DType::F32)?;
        let mut g = Graph::new();
        g.add_node(Node::input("in", &[5]))
            .add_node(Node::linear("fc", "fc.weight.values"))
            .add_node(Node::output("out", &[6]));
        g.chain(&["in", "fc", "out"]);
        w.set_graph(g)
    })
    .unwrap();
    let s = Store::open(&path).unwrap();
    let back = s.read_csr("fc.weight").unwrap();
    assert_eq!(back, csr);
    assert_eq!(back.to_dense(), dense);
    assert_eq!(
        s.info("fc.weight.values").unwrap().logical_shape(),
        vec![6, 5]
    );
    assert!(s.validate_graph().unwrap().is_some());
    assert!(s.read_csr("nope").is_err());
}
