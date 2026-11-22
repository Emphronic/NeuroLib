mod common;

use common::TmpDir;
use neo_store::*;
use serde_json::{Value as Json, json};

fn parse(s: &str) -> Json {
    serde_json::from_str(s).expect("export must be valid JSON")
}

fn lif_params(w: &mut StoreWriter<Vec<u8>>, prefix: &str, n: usize) {
    // tau, r, v_reset are scalars (stretched to the number of neurons), the rest are vectors
    w.add_scalar_f32(&format!("{prefix}.tau"), TensorRole::NeuronParam, 0.5)
        .unwrap();
    w.add_scalar_f32(&format!("{prefix}.r"), TensorRole::NeuronParam, 1.0)
        .unwrap();
    w.add_f32(
        &format!("{prefix}.v_leak"),
        &[n as u64],
        TensorRole::NeuronParam,
        &vec![0.0; n],
    )
    .unwrap();
    w.add_scalar_f32(&format!("{prefix}.v_reset"), TensorRole::NeuronParam, 0.0)
        .unwrap();
    w.add_f32(
        &format!("{prefix}.v_threshold"),
        &[n as u64],
        TensorRole::NeuronParam,
        &vec![1.0; n],
    )
    .unwrap();
}

fn mlp_store(c: Compression) -> Store {
    let opts = WriteOptions {
        compression: c,
        compress_chunk_bytes: 4096,
        ..WriteOptions::default()
    };
    let mut w = StoreWriter::in_memory(opts).unwrap();
    w.add_f32("fc1.weight", &[8, 4], TensorRole::Weight, &[0.1; 32])
        .unwrap();
    w.add_f32("fc1.bias", &[8], TensorRole::Bias, &[0.25; 8])
        .unwrap();
    lif_params(&mut w, "lif1", 8);
    w.add_f32("fc2.weight", &[2, 8], TensorRole::Weight, &[0.2; 16])
        .unwrap();
    let mut g = Graph::new();
    g.add_node(Node::input("in", &[4]))
        .add_node(Node::affine("fc1", "fc1.weight", "fc1.bias"))
        .add_node(Node::neuron("lif1", NodeOp::Lif, "lif1"))
        .add_node(Node::linear("fc2", "fc2.weight"))
        .add_node(Node::output("out", &[2]));
    g.chain(&["in", "fc1", "lif1", "fc2", "out"]);
    w.set_graph(g).unwrap();
    let (bytes, _) = w.finish().unwrap();
    Store::from_bytes(bytes).unwrap()
}

#[test]
fn mlp_is_exported_in_nir_shape() {
    let s = mlp_store(Compression::None);
    let out = s.to_nir_json(&NirJsonOptions::default()).unwrap();
    assert!(out.warnings.is_empty(), "{:?}", out.warnings);
    let v = parse(&out.json);
    assert_eq!(v["version"], "1.0");
    assert_eq!(v["nodes"]["in"]["type"], "Input");
    assert_eq!(v["nodes"]["in"]["input_type"]["input"], json!([4]));
    assert_eq!(v["nodes"]["out"]["type"], "Output");
    assert_eq!(v["nodes"]["out"]["output_type"]["output"], json!([2]));
    // Affine: weight [8, 4] nested arrays + bias
    assert_eq!(v["nodes"]["fc1"]["type"], "Affine");
    let w = v["nodes"]["fc1"]["weight"].as_array().unwrap();
    assert_eq!(w.len(), 8);
    assert!(w.iter().all(|row| row.as_array().unwrap().len() == 4));
    assert!((w[0][0].as_f64().unwrap() - 0.1).abs() < 1e-12);
    assert_eq!(v["nodes"]["fc1"]["bias"].as_array().unwrap().len(), 8);
    // Linear without bias remains Linear
    assert_eq!(v["nodes"]["fc2"]["type"], "Linear");
    assert!(v["nodes"]["fc2"].get("bias").is_none());
    // scalar parameters of neurons are stretched to 8 neurons
    assert_eq!(v["nodes"]["lif1"]["type"], "LIF");
    for port in ["tau", "r", "v_leak", "v_reset", "v_threshold"] {
        assert_eq!(
            v["nodes"]["lif1"][port].as_array().unwrap().len(),
            8,
            "{port}"
        );
    }
    assert_eq!(v["nodes"]["lif1"]["tau"][3], 0.5);
    assert_eq!(
        v["edges"],
        json!([
            ["in", "fc1"],
            ["fc1", "lif1"],
            ["lif1", "fc2"],
            ["fc2", "out"]
        ])
    );
}

#[test]
fn compact_and_pretty_describe_the_same_document() {
    let s = mlp_store(Compression::None);
    let pretty = s.to_nir_json(&NirJsonOptions::default()).unwrap().json;
    let compact = s
        .to_nir_json(&NirJsonOptions {
            pretty: false,
            ..NirJsonOptions::default()
        })
        .unwrap()
        .json;
    assert!(pretty.len() > compact.len());
    assert!(!compact.trim_end().contains('\n'));
    assert_eq!(parse(&pretty), parse(&compact));
}

#[test]
fn compression_does_not_change_the_export() {
    let a = mlp_store(Compression::None)
        .to_nir_json(&NirJsonOptions::default())
        .unwrap();
    let b = mlp_store(Compression::Always)
        .to_nir_json(&NirJsonOptions::default())
        .unwrap();
    assert_eq!(a.json, b.json);
}

#[test]
fn conv_flatten_and_linear_get_nir_fields() {
    let mut w = StoreWriter::in_memory(WriteOptions::default()).unwrap();
    w.add_f32("c.weight", &[2, 1, 3, 3], TensorRole::Weight, &[0.5; 18])
        .unwrap();
    w.add_f32("fc.weight", &[3, 8], TensorRole::Weight, &[0.5; 24])
        .unwrap();
    let mut g = Graph::new();
    g.add_node(Node::input("in", &[1, 4, 4]))
        .add_node(Node::new("conv", NodeOp::Conv2d).param("weight", "c.weight"))
        .add_node(Node::new("flat", NodeOp::Flatten))
        .add_node(Node::linear("fc", "fc.weight"))
        .add_node(Node::output("out", &[3]));
    g.chain(&["in", "conv", "flat", "fc", "out"]);
    w.set_graph(g).unwrap();
    let (bytes, _) = w.finish().unwrap();
    let s = Store::from_bytes(bytes).unwrap();

    let v = parse(&s.to_nir_json(&NirJsonOptions::default()).unwrap().json);
    let conv = &v["nodes"]["conv"];
    assert_eq!(conv["type"], "Conv2d");
    assert_eq!(conv["input_shape"], json!([4, 4]));
    assert_eq!(conv["stride"], 1);
    assert_eq!(conv["padding"], 0);
    assert_eq!(conv["dilation"], 1);
    assert_eq!(conv["groups"], 1);
    assert_eq!(conv["weight"].as_array().unwrap().len(), 2);
    assert_eq!(conv["weight"][0][0][0].as_array().unwrap().len(), 3);
    let flat = &v["nodes"]["flat"];
    assert_eq!(flat["type"], "Flatten");
    assert_eq!(flat["input_type"]["input"], json!([2, 2, 2]));
    assert_eq!(flat["start_dim"], 0);
    assert_eq!(flat["end_dim"], -1);
}

#[test]
fn nested_graph_becomes_nir_graph() {
    let mut w = StoreWriter::in_memory(WriteOptions::default()).unwrap();
    w.add_f32("w", &[2, 2], TensorRole::Weight, &[1.0, 2.0, 3.0, 4.0])
        .unwrap();
    let mut inner = Graph::new();
    inner
        .add_node(Node::input("i", &[2]))
        .add_node(Node::linear("lin", "w"))
        .add_node(Node::output("o", &[2]));
    inner.chain(&["i", "lin", "o"]);
    let mut g = Graph::new();
    g.add_node(Node::input("in", &[2]))
        .add_node(Node::with_subgraph("blk", inner))
        .add_node(Node::output("out", &[2]));
    g.chain(&["in", "blk", "out"]);
    w.set_graph(g).unwrap();
    let (bytes, _) = w.finish().unwrap();
    let s = Store::from_bytes(bytes).unwrap();

    let v = parse(&s.to_nir_json(&NirJsonOptions::default()).unwrap().json);
    let blk = &v["nodes"]["blk"];
    assert_eq!(blk["type"], "NIRGraph");
    assert_eq!(blk["nodes"]["lin"]["type"], "Linear");
    assert_eq!(blk["nodes"]["lin"]["weight"], json!([[1, 2], [3, 4]]));
    assert_eq!(blk["edges"], json!([["i", "lin"], ["lin", "o"]]));
}

fn store_with_alpha() -> Store {
    let mut w = StoreWriter::in_memory(WriteOptions::default()).unwrap();
    w.add_f32("a.tau", &[2], TensorRole::NeuronParam, &[0.1, 0.2])
        .unwrap();
    let mut g = Graph::new();
    g.add_node(Node::input("in", &[2]))
        .add_node(
            Node::new("alpha", NodeOp::Alpha)
                .param("tau", "a.tau")
                .attr("steps", 3i64),
        )
        .add_node(Node::output("out", &[2]));
    g.chain(&["in", "alpha", "out"]);
    w.set_graph(g).unwrap();
    let (bytes, _) = w.finish().unwrap();
    Store::from_bytes(bytes).unwrap()
}

#[test]
fn nodes_without_nir_equivalent_are_rejected_or_passed_through() {
    let s = store_with_alpha();
    let err = s.to_nir_json(&NirJsonOptions::default()).unwrap_err();
    assert!(matches!(err, StoreError::UnsupportedFeature(_)), "{err}");

    let out = s
        .to_nir_json(&NirJsonOptions {
            unsupported: Unsupported::Passthrough,
            ..NirJsonOptions::default()
        })
        .unwrap();
    assert_eq!(out.warnings.len(), 1);
    let v = parse(&out.json);
    assert_eq!(v["nodes"]["alpha"]["type"], "Alpha");
    let tau = v["nodes"]["alpha"]["tau"].as_array().unwrap();
    assert!((tau[0].as_f64().unwrap() - 0.1).abs() < 1e-9);
    assert!((tau[1].as_f64().unwrap() - 0.2).abs() < 1e-9);
    assert_eq!(v["nodes"]["alpha"]["steps"], 3);
}

#[test]
fn edge_delay_is_an_error_unless_lossy() {
    let mut w = StoreWriter::in_memory(WriteOptions::default()).unwrap();
    w.add_f32("w", &[2, 2], TensorRole::Weight, &[1.0; 4])
        .unwrap();
    let mut g = Graph::new();
    g.add_node(Node::input("in", &[2]))
        .add_node(Node::linear("fc", "w"))
        .add_node(Node::output("out", &[2]));
    g.connect_edge(Edge::new("in", "fc").with_delay(2));
    g.connect("fc", "out");
    w.set_graph(g).unwrap();
    let (bytes, _) = w.finish().unwrap();
    let s = Store::from_bytes(bytes).unwrap();

    assert!(matches!(
        s.to_nir_json(&NirJsonOptions::default()),
        Err(StoreError::UnsupportedFeature(_))
    ));
    let out = s
        .to_nir_json(&NirJsonOptions {
            lossy: true,
            ..NirJsonOptions::default()
        })
        .unwrap();
    assert_eq!(out.warnings.len(), 1);
    assert_eq!(
        parse(&out.json)["edges"],
        json!([["in", "fc"], ["fc", "out"]])
    );
}

#[test]
fn non_finite_weights_cannot_be_exported() {
    let mut w = StoreWriter::in_memory(WriteOptions::default()).unwrap();
    w.set_non_finite_policy(NonFinitePolicy::Allow);
    w.add_f32("w", &[1, 1], TensorRole::Weight, &[f32::NAN])
        .unwrap();
    let mut g = Graph::new();
    g.add_node(Node::input("in", &[1]))
        .add_node(Node::linear("fc", "w"))
        .add_node(Node::output("out", &[1]));
    g.chain(&["in", "fc", "out"]);
    w.set_graph(g).unwrap();
    let (bytes, _) = w.finish().unwrap();
    let s = Store::from_bytes(bytes).unwrap();
    assert!(matches!(
        s.to_nir_json(&NirJsonOptions::default()),
        Err(StoreError::Invalid(_))
    ));
}

#[test]
fn store_without_graph_has_nothing_to_export() {
    let mut w = StoreWriter::in_memory(WriteOptions::default()).unwrap();
    w.add_f32("w", &[1], TensorRole::Weight, &[1.0]).unwrap();
    let (bytes, _) = w.finish().unwrap();
    let s = Store::from_bytes(bytes).unwrap();
    assert!(matches!(
        s.to_nir_json(&NirJsonOptions::default()),
        Err(StoreError::Graph(_))
    ));
}

#[test]
fn file_and_sharded_export() {
    let dir = TmpDir::new("nir-json");
    let path = dir.path("m.json");
    let s = mlp_store(Compression::None);
    let warnings = s.write_nir_json(&path, &NirJsonOptions::default()).unwrap();
    assert!(warnings.is_empty());
    assert!(!dir.path("m.json.tmp").exists());
    let text = std::fs::read_to_string(&path).unwrap();
    assert_eq!(parse(&text)["nodes"]["fc1"]["type"], "Affine");

    let mut w = ShardedWriter::new(&dir.0, "sh", 50, WriteOptions::default()).unwrap();
    w.add_f32("a", &[3, 4], TensorRole::Weight, &[1.0; 12])
        .unwrap();
    w.add_f32("b", &[2, 3], TensorRole::Weight, &[2.0; 6])
        .unwrap();
    let mut g = Graph::new();
    g.add_node(Node::input("in", &[4]))
        .add_node(Node::linear("fc1", "a"))
        .add_node(Node::linear("fc2", "b"))
        .add_node(Node::output("out", &[2]));
    g.chain(&["in", "fc1", "fc2", "out"]);
    w.set_graph(g).unwrap();
    let summary = w.finish().unwrap();
    let sh = ShardedStore::open(&summary.manifest_path).unwrap();
    let v = parse(&sh.to_nir_json(&NirJsonOptions::default()).unwrap().json);
    assert_eq!(v["nodes"]["fc2"]["weight"], json!([[2, 2, 2], [2, 2, 2]]));
}

// ─── import: NIR JSON → syn ───────────────────────────────── ─────────────────────────────────

fn import_to_store(text: &str, opts: &NirImportOptions) -> (Store, Vec<String>) {
    let mut w = StoreWriter::in_memory(WriteOptions::default()).unwrap();
    let warnings = w.add_nir_json(text, opts).unwrap();
    let (bytes, _) = w.finish().unwrap();
    (Store::from_bytes(bytes).unwrap(), warnings)
}

/// The second export differs only in metadata with the NIR version.
fn without_metadata(mut v: Json) -> Json {
    v.as_object_mut().unwrap().remove("metadata");
    v
}

#[test]
fn export_then_import_reproduces_graph_and_weights() {
    let original = mlp_store(Compression::None);
    let json1 = original
        .to_nir_json(&NirJsonOptions::default())
        .unwrap()
        .json;

    let (back, warnings) = import_to_store(&json1, &NirImportOptions::default());
    assert!(warnings.is_empty(), "{warnings:?}");
    back.validate_graph().unwrap().unwrap();
    for name in ["fc1.weight", "fc1.bias", "fc2.weight", "lif1.v_leak"] {
        assert_eq!(
            back.read_f32(name).unwrap(),
            original.read_f32(name).unwrap(),
            "{name}"
        );
    }
    // scalar parameters of neurons were returned as vectors of 8 neurons
    assert_eq!(back.info("lif1.tau").unwrap().shape, vec![8]);
    assert_eq!(back.info("fc1.weight").unwrap().role, TensorRole::Weight);
    assert_eq!(back.info("fc1.bias").unwrap().role, TensorRole::Bias);
    assert_eq!(back.info("lif1.tau").unwrap().role, TensorRole::NeuronParam);
    assert_eq!(back.graph().unwrap().node("lif1").unwrap().op, NodeOp::Lif);
    assert_eq!(
        back.graph().unwrap().node("fc1").unwrap().op,
        NodeOp::Affine
    );

    // and the second export gives the same document (plus the NIR version in the metadata)
    let json2 = back.to_nir_json(&NirJsonOptions::default()).unwrap().json;
    let v2 = parse(&json2);
    assert_eq!(v2["metadata"]["nir.version"], "1.0");
    assert_eq!(without_metadata(v2), parse(&json1));
}

#[test]
fn hand_written_nir_document_is_imported() {
    let text = r#"{
        "version": "1.0",
        "nodes": {
            "input": {"type": "Input", "input_type": {"input": [3]}},
            "fc": {"type": "Linear", "weight": [[1, 2, 3], [4, 5, 6]]},
            "lif": {"type": "LIF", "tau": [0.1, 0.1], "r": [1, 1], "v_leak": [0, 0],
                    "v_reset": [0, 0], "v_threshold": [1, 1]},
            "output": {"type": "Output", "output_type": {"output": [2]}}
        },
        "edges": [["input", "fc"], ["fc", "lif"], ["lif", "output"]]
    }"#;
    let (s, warnings) = import_to_store(text, &NirImportOptions::default());
    assert!(warnings.is_empty(), "{warnings:?}");
    let info = s.info("fc.weight").unwrap();
    assert_eq!(info.shape, vec![2, 3]);
    assert_eq!(
        s.read_f32("fc.weight").unwrap(),
        vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    );
    let shapes = s.validate_graph().unwrap().unwrap();
    assert_eq!(shapes.nodes["lif"].output, vec![2]);
    assert_eq!(s.graph().unwrap().edges.len(), 3);
}

#[test]
fn precision_is_selectable() {
    let text = r#"{"nodes": {
        "i": {"type": "Input", "input_type": {"input": [1]}},
        "fc": {"type": "Linear", "weight": [[0.1]]},
        "o": {"type": "Output", "output_type": {"output": [1]}}},
        "edges": [["i", "fc"], ["fc", "o"]]}"#;
    let opts = NirImportOptions {
        dtype: DType::F64,
        ..NirImportOptions::default()
    };
    let (s, _) = import_to_store(text, &opts);
    assert_eq!(s.info("fc.weight").unwrap().dtype, DType::F64);
    assert_eq!(s.read_f64("fc.weight").unwrap(), vec![0.1f64]);
}

#[test]
fn nested_graph_survives_a_roundtrip_with_path_prefixed_names() {
    let mut w = StoreWriter::in_memory(WriteOptions::default()).unwrap();
    w.add_f32("w", &[2, 2], TensorRole::Weight, &[1.0, 2.0, 3.0, 4.0])
        .unwrap();
    let mut inner = Graph::new();
    inner
        .add_node(Node::input("i", &[2]))
        .add_node(Node::linear("lin", "w"))
        .add_node(Node::output("o", &[2]));
    inner.chain(&["i", "lin", "o"]);
    let mut g = Graph::new();
    g.add_node(Node::input("in", &[2]))
        .add_node(Node::with_subgraph("blk", inner))
        .add_node(Node::output("out", &[2]));
    g.chain(&["in", "blk", "out"]);
    w.set_graph(g).unwrap();
    let (bytes, _) = w.finish().unwrap();
    let s = Store::from_bytes(bytes).unwrap();
    let json1 = s.to_nir_json(&NirJsonOptions::default()).unwrap().json;

    let (back, _) = import_to_store(&json1, &NirImportOptions::default());
    assert_eq!(
        back.read_f32("blk/lin.weight").unwrap(),
        vec![1.0, 2.0, 3.0, 4.0]
    );
    back.validate_graph().unwrap().unwrap();
    let json2 = back.to_nir_json(&NirJsonOptions::default()).unwrap().json;
    assert_eq!(without_metadata(parse(&json2)), parse(&json1));
}

#[test]
fn passthrough_export_is_imported_back_with_passthrough() {
    let s = store_with_alpha();
    let json = s
        .to_nir_json(&NirJsonOptions {
            unsupported: Unsupported::Passthrough,
            ..NirJsonOptions::default()
        })
        .unwrap()
        .json;
    let strict = parse_nir_json(&json, &NirImportOptions::default());
    assert!(matches!(strict, Err(StoreError::UnsupportedFeature(_))));
    let opts = NirImportOptions {
        unknown: Unsupported::Passthrough,
        ..NirImportOptions::default()
    };
    let imp = parse_nir_json(&json, &opts).unwrap();
    assert_eq!(imp.graph.node("alpha").unwrap().op, NodeOp::Alpha);
    assert_eq!(
        imp.graph.node("alpha").unwrap().attrs["steps"],
        Value::I64(3)
    );
    assert!(
        imp.tensors
            .iter()
            .any(|t| t.name == "alpha.tau" && t.shape == vec![2])
    );
}

#[test]
fn bad_documents_are_reported() {
    let opts = NirImportOptions::default();
    // not JSON
    assert!(matches!(
        parse_nir_json("{", &opts),
        Err(StoreError::Invalid(_))
    ));
    assert!(matches!(
        parse_nir_json("[1,2]", &opts),
        Err(StoreError::Graph(_))
    ));
    // LIF without tau
    let no_tau = r#"{"nodes": {"n": {"type": "LIF", "r": [1], "v_leak": [0],
                      "v_reset": [0], "v_threshold": [1]}}, "edges": []}"#;
    let e = parse_nir_json(no_tau, &opts).unwrap_err();
    assert!(
        matches!(e, StoreError::Graph(_)) && e.to_string().contains("tau"),
        "{e}"
    );
    // non-rectangular array
    let ragged = r#"{"nodes": {"fc": {"type": "Linear", "weight": [[1, 2], [3]]}}, "edges": []}"#;
    assert!(parse_nir_json(ragged, &opts).is_err());
    // a graph with an edge to a non-existent node is rejected when written
    let dangling = r#"{"nodes": {"i": {"type": "Input", "input_type": {"input": [1]}}},
                       "edges": [["i", "ghost"]]}"#;
    let mut w = StoreWriter::in_memory(WriteOptions::default()).unwrap();
    assert!(w.add_nir_json(dangling, &opts).is_err());
    // the discrepancy between the shape of the weights and the input is caught by checking the graph
    let mismatch = r#"{"nodes": {
        "i": {"type": "Input", "input_type": {"input": [4]}},
        "fc": {"type": "Linear", "weight": [[1, 2, 3]]},
        "o": {"type": "Output", "output_type": {"output": [1]}}},
        "edges": [["i", "fc"], ["fc", "o"]]}"#;
    let mut w = StoreWriter::in_memory(WriteOptions::default()).unwrap();
    w.add_nir_json(mismatch, &opts).unwrap();
    assert!(matches!(w.finish(), Err(StoreError::Graph(_))));
    // storage type must be float
    let int_opts = NirImportOptions {
        dtype: DType::I32,
        ..NirImportOptions::default()
    };
    assert!(parse_nir_json("{\"nodes\": {}}", &int_opts).is_err());
}

#[test]
fn file_conversion_writes_a_valid_syn() {
    let dir = TmpDir::new("nir-import");
    let json_path = dir.path("in.json");
    let syn_path = dir.path("out.syn");
    let original = mlp_store(Compression::None);
    std::fs::write(
        &json_path,
        original
            .to_nir_json(&NirJsonOptions::default())
            .unwrap()
            .json,
    )
    .unwrap();
    let (summary, warnings) = nir_json_file_to_syn(
        &json_path,
        &syn_path,
        WriteOptions::default(),
        &NirImportOptions::default(),
    )
    .unwrap();
    assert!(warnings.is_empty());
    assert_eq!(summary.tensor_count, 8);
    let s = Store::open(&syn_path).unwrap();
    assert!(s.verify().unwrap().is_ok());
    assert_eq!(
        s.read_f32("fc2.weight").unwrap(),
        original.read_f32("fc2.weight").unwrap()
    );
}
