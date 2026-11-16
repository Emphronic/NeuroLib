//! Import NIR JSON → graph and weights `neo-store`: reverse operation to
//! [`Store::to_nir_json`](crate::Store::to_nir_json).
//!
//! Reads JSON of the same type that the export writes (dictionary `nodes`, list of pairs `edges`,
//! optional `metadata` and `version`), including JSON obtained from the Python package NIR
//! (`nir.NIRGraph.to_dict()` and the like) if it has such a structure. Exact field names
//! It is worth checking with the version of NIR you are using.
//!
//! ## Where the weights go
//!
//! Each node parameter becomes a separate tensor with the name `"node". "port"`
//! (`fc1.weight`, `lif1.tau`); in subgraphs the path is added to the name: `blk/lin.weight`.
//! A node refers to a tensor by this name, just like handcrafted graphs. The role of the tensor
//! output from the port: `weight` - weight, `bias` - offset, `delay` - delay, the rest -
//! neuron parameters.
//!
//! ## Node matching
//!
//! Reverse to export table: `Input`/`Output` (shape from `input_type`/`output_type`),
//! `Linear`, `Affine`, `Conv1d`, `Conv2d`, `Flatten`, `SumPool2d`, `AvgPool2d`, `Scale`,
//! `Threshold`, `Delay`, `I`, `IF`, `LI`, `LIF`, `CubaLI`, `CubaLIF`, and `NIRGraph` becomes
//! nested subgraph. Fields that are not in the node specification (for example, `input_shape` in
//! convolutions: the shape is derived from the graph), are ignored, the remaining extra fields are reported
//! warning. If the file does not have `start_dim`/`end_dim` at `Flatten` or `stride`/`padding`
//! for convolution, the defaults are `neo-store`, not NIR.
//!
//! Node of unknown type - error. With [`Unsupported::Passthrough`] it is imported as
//! custom operation: arrays of numbers become tensor parameters, the remaining fields become
//! attributes (a scalar without an array, even if it was a parameter, will become an attribute).
//!
//! All weights come in JSON as numbers; storage type is specified [`NirImportOptions::dtype`]
//! (default `F32`). The graph is checked against the weights when the record is completed, so that
//! inconsistent shapes are identified immediately.

use std::collections::HashSet;
use std::fs;
use std::io::Write;
use std::path::Path;

use crate::dtype::DType;
use crate::error::{StoreError, StoreResult};
use crate::graph::{Edge, Graph, Node, NodeOp};
use crate::nir_json::Unsupported;
use crate::shard::ShardedWriter;
use crate::tensor::TensorRole;
use crate::value::{Meta, Value};
use crate::writer::{StoreWriter, WriteOptions, WriteSummary, write_store_file};

// ═══════════════════════════════════════════════════════════════════════════
// JSON parsing (std only)
// ═══════════════════════════════════════════════════════════════════════════

const MAX_JSON_DEPTH: usize = 128;

#[derive(Debug, Clone, PartialEq)]
enum Jv {
    Null,
    Bool(bool),
    Int(i64),
    UInt(u64),
    Float(f64),
    Str(String),
    Arr(Vec<Jv>),
    Obj(Vec<(String, Jv)>),
}

struct Parser<'a> {
    text: &'a str,
    s: &'a [u8],
    pos: usize,
}

impl Parser<'_> {
    fn err<T>(&self, msg: &str) -> StoreResult<T> {
        let upto = &self.s[..self.pos.min(self.s.len())];
        let line = upto.iter().filter(|b| **b == b'\n').count() + 1;
        let col = upto.len() - upto.iter().rposition(|b| *b == b'\n').map_or(0, |i| i + 1) + 1;
        Err(StoreError::Invalid(format!(
            "NIR JSON: {msg} (line {line}, column {col})"
        )))
    }

    fn peek(&self) -> Option<u8> {
        self.s.get(self.pos).copied()
    }

    fn ws(&mut self) {
        while matches!(self.peek(), Some(b' ' | b'\t' | b'\n' | b'\r')) {
            self.pos += 1;
        }
    }

    fn digits(&mut self) -> usize {
        let start = self.pos;
        while matches!(self.peek(), Some(b'0'..=b'9')) {
            self.pos += 1;
        }
        self.pos - start
    }

    fn value(&mut self, depth: usize) -> StoreResult<Jv> {
        if depth > MAX_JSON_DEPTH {
            return self.err("too deep nesting");
        }
        self.ws();
        match self.peek() {
            None => self.err("unexpected end of text"),
            Some(b'{') => self.object(depth),
            Some(b'[') => self.array(depth),
            Some(b'"') => Ok(Jv::Str(self.string()?)),
            Some(b't') => self.literal("true", Jv::Bool(true)),
            Some(b'f') => self.literal("false", Jv::Bool(false)),
            Some(b'n') => self.literal("null", Jv::Null),
            Some(b'N' | b'I') => self.err("NaN and Infinity are not allowed in JSON"),
            Some(b'-' | b'0'..=b'9') => self.number(),
            Some(_) => self.err("unexpected symbol"),
        }
    }

    fn literal(&mut self, word: &str, v: Jv) -> StoreResult<Jv> {
        if self.s[self.pos..].starts_with(word.as_bytes()) {
            self.pos += word.len();
            Ok(v)
        } else {
            self.err("unknown word")
        }
    }

    fn number(&mut self) -> StoreResult<Jv> {
        let start = self.pos;
        if self.peek() == Some(b'-') {
            self.pos += 1;
        }
        match self.peek() {
            Some(b'0') => self.pos += 1,
            Some(b'1'..=b'9') => {
                self.digits();
            }
            _ => return self.err("incorrect number"),
        }
        let mut float = false;
        if self.peek() == Some(b'.') {
            self.pos += 1;
            float = true;
            if self.digits() == 0 {
                return self.err("numbers are needed after the dot");
            }
        }
        if matches!(self.peek(), Some(b'e' | b'E')) {
            self.pos += 1;
            float = true;
            if matches!(self.peek(), Some(b'+' | b'-')) {
                self.pos += 1;
            }
            if self.digits() == 0 {
                return self.err("the exponent needs numbers");
            }
        }
        let txt = &self.text[start..self.pos];
        if !float {
            if let Ok(i) = txt.parse::<i64>() {
                return Ok(Jv::Int(i));
            }
            if let Ok(u) = txt.parse::<u64>() {
                return Ok(Jv::UInt(u));
            }
        }
        match txt.parse::<f64>() {
            Ok(f) if f.is_finite() => Ok(Jv::Float(f)),
            _ => self.err("number out of range f64"),
        }
    }

    fn hex4(&mut self) -> StoreResult<u32> {
        if self.pos + 4 > self.s.len() {
            return self.err("break of \\u sequence");
        }
        let mut v = 0u32;
        for i in 0..4 {
            match (self.s[self.pos + i] as char).to_digit(16) {
                Some(d) => v = v * 16 + d,
                None => return self.err("invalid \\u sequence"),
            }
        }
        self.pos += 4;
        Ok(v)
    }

    fn string(&mut self) -> StoreResult<String> {
        self.pos += 1; // opening quote
        let mut out = String::new();
        loop {
            let run = self.pos;
            while let Some(b) = self.peek() {
                if b == b'"' || b == b'\\' || b < 0x20 {
                    break;
                }
                self.pos += 1;
            }
            out.push_str(&self.text[run..self.pos]);
            match self.peek() {
                None => return self.err("the line is not closed"),
                Some(b'"') => {
                    self.pos += 1;
                    return Ok(out);
                }
                Some(b'\\') => {
                    self.pos += 1;
                    let Some(c) = self.peek() else {
                        return self.err("broken escape sequence");
                    };
                    self.pos += 1;
                    match c {
                        b'"' => out.push('"'),
                        b'\\' => out.push('\\'),
                        b'/' => out.push('/'),
                        b'b' => out.push('\u{8}'),
                        b'f' => out.push('\u{c}'),
                        b'n' => out.push('\n'),
                        b'r' => out.push('\r'),
                        b't' => out.push('\t'),
                        b'u' => {
                            let hi = self.hex4()?;
                            let cp = if (0xD800..0xDC00).contains(&hi) {
                                if self.peek() == Some(b'\\')
                                    && self.s.get(self.pos + 1) == Some(&b'u')
                                {
                                    self.pos += 2;
                                    let lo = self.hex4()?;
                                    if !(0xDC00..0xE000).contains(&lo) {
                                        return self.err("unfaithful surrogate couple");
                                    }
                                    0x10000 + ((hi - 0xD800) << 10) + (lo - 0xDC00)
                                } else {
                                    return self.err("single surrogate");
                                }
                            } else if (0xDC00..0xE000).contains(&hi) {
                                return self.err("single surrogate");
                            } else {
                                hi
                            };
                            match char::from_u32(cp) {
                                Some(ch) => out.push(ch),
                                None => return self.err("invalid Unicode character"),
                            }
                        }
                        _ => return self.err("unknown escape sequence"),
                    }
                }
                Some(_) => return self.err("control character inside a string"),
            }
        }
    }

    fn array(&mut self, depth: usize) -> StoreResult<Jv> {
        self.pos += 1;
        let mut items = Vec::new();
        self.ws();
        if self.peek() == Some(b']') {
            self.pos += 1;
            return Ok(Jv::Arr(items));
        }
        loop {
            items.push(self.value(depth + 1)?);
            self.ws();
            match self.peek() {
                Some(b',') => self.pos += 1,
                Some(b']') => {
                    self.pos += 1;
                    return Ok(Jv::Arr(items));
                }
                _ => return self.err("expected ',' or ']'"),
            }
        }
    }

    fn object(&mut self, depth: usize) -> StoreResult<Jv> {
        self.pos += 1;
        let mut fields: Vec<(String, Jv)> = Vec::new();
        self.ws();
        if self.peek() == Some(b'}') {
            self.pos += 1;
            return Ok(Jv::Obj(fields));
        }
        loop {
            self.ws();
            if self.peek() != Some(b'"') {
                return self.err("key string expected");
            }
            let key = self.string()?;
            self.ws();
            if self.peek() != Some(b':') {
                return self.err("expected ':'");
            }
            self.pos += 1;
            let v = self.value(depth + 1)?;
            if fields.iter().any(|(k, _)| *k == key) {
                return self.err(&format!("duplicate key '{key}'"));
            }
            fields.push((key, v));
            self.ws();
            match self.peek() {
                Some(b',') => self.pos += 1,
                Some(b'}') => {
                    self.pos += 1;
                    return Ok(Jv::Obj(fields));
                }
                _ => return self.err("expected ',' or '}'"),
            }
        }
    }
}

fn parse_json(text: &str) -> StoreResult<Jv> {
    let text = text.strip_prefix('\u{feff}').unwrap_or(text);
    let mut p = Parser {
        text,
        s: text.as_bytes(),
        pos: 0,
    };
    let v = p.value(0)?;
    p.ws();
    if p.pos != p.s.len() {
        return p.err("extra data after the document");
    }
    Ok(v)
}

// ═══════════════════════════════════════════════════════════════════════════
// Import
// ═══════════════════════════════════════════════════════════════════════════

/// Import options.
#[derive(Debug, Clone)]
pub struct NirImportOptions {
    /// Nodes of unknown type and operations without an analogue in NIR.
    pub unknown: Unsupported,
    /// Weight storage type: `F32` (default), `F64`, `F16` or `BF16`.
    pub dtype: DType,
}

impl Default for NirImportOptions {
    fn default() -> Self {
        Self {
            unknown: Unsupported::Error,
            dtype: DType::F32,
        }
    }
}

/// Tensor read from node NIR.
#[derive(Debug, Clone, PartialEq)]
pub struct ImportedTensor {
    /// Name `"node". "port" (in subgraphs - with path `subgraph/`).
    pub name: String,
    /// shape.
    pub shape: Vec<u64>,
    /// Role.
    pub role: TensorRole,
    /// Values are in row order.
    pub data: Vec<f64>,
}

/// Parsed NIR JSON document: graph, weights and comments.
#[derive(Debug, Clone)]
pub struct NirImported {
    /// Graph; nodes refer to tensors from [`NirImported::tensors`] by name.
    pub graph: Graph,
    /// Weights and parameters.
    pub tensors: Vec<ImportedTensor>,
    /// What had to be missed or brought closer.
    pub warnings: Vec<String>,
    /// Type of storage of scales.
    pub dtype: DType,
}

fn bad(msg: String) -> StoreError {
    StoreError::Graph(msg)
}

fn get<'a>(fields: &'a [(String, Jv)], key: &str) -> Option<&'a Jv> {
    fields.iter().find(|(k, _)| k == key).map(|(_, v)| v)
}

fn jv_to_value(v: &Jv) -> Option<Value> {
    Some(match v {
        Jv::Null | Jv::Obj(_) => return None,
        Jv::Bool(b) => Value::Bool(*b),
        Jv::Int(i) => Value::I64(*i),
        Jv::UInt(u) => Value::U64(*u),
        Jv::Float(f) => Value::F64(*f),
        Jv::Str(s) => Value::Str(s.clone()),
        Jv::Arr(items) => Value::List(items.iter().filter_map(jv_to_value).collect()),
    })
}

fn is_numeric(v: &Jv) -> bool {
    match v {
        Jv::Int(_) | Jv::UInt(_) | Jv::Float(_) => true,
        Jv::Arr(items) => items.iter().all(is_numeric),
        _ => false,
    }
}

fn fill(v: &Jv, shape: &[u64], depth: usize, out: &mut Vec<f64>) -> Result<(), String> {
    if depth == shape.len() {
        return match v {
            Jv::Int(i) => {
                out.push(*i as f64);
                Ok(())
            }
            Jv::UInt(u) => {
                out.push(*u as f64);
                Ok(())
            }
            Jv::Float(f) => {
                out.push(*f);
                Ok(())
            }
            _ => Err(
                "a number was expected (the array is not rectangular or does not contain numbers)"
                    .into(),
            ),
        };
    }
    match v {
        Jv::Arr(items) if items.len() as u64 == shape[depth] => {
            for it in items {
                fill(it, shape, depth + 1, out)?;
            }
            Ok(())
        }
        _ => Err("non-rectangular array".into()),
    }
}

/// The shape and values of a nested numeric array (or number).
fn to_tensor(v: &Jv) -> Result<(Vec<u64>, Vec<f64>), String> {
    let mut shape = Vec::new();
    let mut cur = v;
    while let Jv::Arr(items) = cur {
        shape.push(items.len() as u64);
        match items.first() {
            Some(first) => cur = first,
            None => break,
        }
    }
    let mut data = Vec::new();
    fill(v, &shape, 0, &mut data)?;
    Ok((shape, data))
}

fn shape_ints(v: &Jv, what: &str) -> StoreResult<Vec<u64>> {
    let Jv::Arr(items) = v else {
        return Err(bad(format!("{what}: shape must be an array of integers")));
    };
    items
        .iter()
        .map(|d| match d {
            Jv::Int(i) if *i >= 0 => Ok(*i as u64),
            Jv::UInt(u) => Ok(*u),
            _ => Err(bad(format!(
                "{what}: the shape must consist of non-negative integers"
            ))),
        })
        .collect()
}

fn io_shape(f: &[(String, Jv)], key: &str, ctx: &str) -> StoreResult<Vec<u64>> {
    let what = format!("{ctx}.{key}");
    match get(f, key) {
        Some(Jv::Obj(m)) if m.len() == 1 => shape_ints(&m[0].1, &what),
        Some(Jv::Obj(_)) => Err(StoreError::UnsupportedFeature(format!(
            "{what}: Multiple I/O ports are not supported"
        ))),
        Some(v) if matches!(v, Jv::Arr(_)) => shape_ints(v, &what),
        _ => Err(bad(format!("{what}: no shape"))),
    }
}

fn role_for(port: &str) -> TensorRole {
    match port {
        "weight" => TensorRole::Weight,
        "bias" => TensorRole::Bias,
        "delay" => TensorRole::Delay,
        _ => TensorRole::NeuronParam,
    }
}

fn flatten_meta(prefix: &str, fields: &[(String, Jv)], out: &mut Meta, warnings: &mut Vec<String>) {
    for (k, v) in fields {
        let key = if prefix.is_empty() {
            k.clone()
        } else {
            format!("{prefix}.{k}")
        };
        match v {
            Jv::Obj(inner) => flatten_meta(&key, inner, out, warnings),
            other => match jv_to_value(other) {
                Some(val) => {
                    out.insert(key, val);
                }
                None => warnings.push(format!("metadata.{key}: null value missing")),
            },
        }
    }
}

struct Importer<'a> {
    opts: &'a NirImportOptions,
    tensors: Vec<ImportedTensor>,
    names: HashSet<String>,
    warnings: Vec<String>,
}

impl Importer<'_> {
    fn tensor(&mut self, ctx: &str, port: &str, v: &Jv) -> StoreResult<String> {
        let name = format!("{ctx}.{port}");
        if !self.names.insert(name.clone()) {
            return Err(bad(format!("repeating tensor name '{name}'")));
        }
        let (shape, data) = to_tensor(v).map_err(|m| bad(format!("{name}: {m}")))?;
        self.tensors.push(ImportedTensor {
            name: name.clone(),
            shape,
            role: role_for(port),
            data,
        });
        Ok(name)
    }

    fn graph(&mut self, obj: &[(String, Jv)], path: &str) -> StoreResult<Graph> {
        let Some(Jv::Obj(nodes)) = get(obj, "nodes") else {
            return Err(bad(format!("{path}graph: no nodes object")));
        };
        let mut g = Graph::new();
        for (name, v) in nodes {
            let Jv::Obj(fields) = v else {
                return Err(bad(format!("{path}{name}: node must be an object")));
            };
            g.nodes.push(self.node(name, fields, path)?);
        }
        match get(obj, "edges") {
            Some(Jv::Arr(edges)) => {
                for e in edges {
                    match e {
                        Jv::Arr(pair) if pair.len() >= 2 => match (&pair[0], &pair[1]) {
                            (Jv::Str(a), Jv::Str(b)) => {
                                if pair.len() > 2 {
                                    self.warnings.push(format!(
                                        "{path}edge {a} → {b}: extra elements are skipped"
                                    ));
                                }
                                g.edges.push(Edge::new(a.clone(), b.clone()));
                            }
                            _ => return Err(bad(format!("{path}edge must be a pair of names"))),
                        },
                        _ => return Err(bad(format!("{path}edge must be a pair of names"))),
                    }
                }
            }
            None => {}
            Some(_) => return Err(bad(format!("{path}edges must be an array"))),
        }
        if let Some(Jv::Obj(m)) = get(obj, "metadata") {
            flatten_meta("", m, &mut g.meta, &mut self.warnings);
        }
        Ok(g)
    }

    fn node(&mut self, name: &str, f: &[(String, Jv)], path: &str) -> StoreResult<Node> {
        let ctx = format!("{path}{name}");
        let Some(Jv::Str(ty)) = get(f, "type") else {
            return Err(bad(format!(
                "{ctx}: node does not have a string field type"
            )));
        };
        match ty.as_str() {
            "Input" => {
                let shape = io_shape(f, "input_type", &ctx)?;
                Ok(Node::new(name, NodeOp::Input).attr("shape", Value::uints(&shape)))
            }
            "Output" => {
                let shape = io_shape(f, "output_type", &ctx)?;
                Ok(Node::new(name, NodeOp::Output).attr("shape", Value::uints(&shape)))
            }
            "NIRGraph" => {
                let g = self.graph(f, &format!("{ctx}/"))?;
                Ok(Node::with_subgraph(name, g))
            }
            "Subgraph" => Err(bad(format!(
                "{ctx}: The nested graph in NIR is called NIRGraph"
            ))),
            other => {
                let op = NodeOp::from_name(other);
                let open = matches!(
                    op,
                    NodeOp::Custom(_)
                        | NodeOp::Alpha
                        | NodeOp::Synaptic
                        | NodeOp::Izhikevich
                        | NodeOp::AdaptiveLif
                );
                if open {
                    self.open_node(name, other, op, f, &ctx)
                } else {
                    self.known_node(name, other, op, f, &ctx)
                }
            }
        }
    }

    fn known_node(
        &mut self,
        name: &str,
        ty: &str,
        op: NodeOp,
        f: &[(String, Jv)],
        ctx: &str,
    ) -> StoreResult<Node> {
        let spec = op.spec();
        let attr_keys: &[&str] = match op {
            NodeOp::Conv1d | NodeOp::Conv2d => &["stride", "padding", "dilation", "groups"],
            NodeOp::Flatten => &["start_dim", "end_dim"],
            NodeOp::SumPool2d | NodeOp::AvgPool2d => &["kernel_size", "stride", "padding"],
            _ => &[],
        };
        let mut node = Node::new(name, op);
        for port in spec.required.iter().chain(spec.optional.iter()) {
            match get(f, port) {
                Some(v) => {
                    let t = self.tensor(ctx, port, v)?;
                    node.params.insert((*port).to_string(), t);
                }
                None if spec.required.contains(port) => {
                    return Err(bad(format!(
                        "{ctx}: node {ty} does not have parameter '{port}'"
                    )));
                }
                None => {}
            }
        }
        for key in attr_keys {
            if let Some(val) = get(f, key).and_then(jv_to_value) {
                node.attrs.insert((*key).to_string(), val);
            }
        }
        for (k, _) in f {
            let used = k == "type"
                || spec.required.contains(&k.as_str())
                || spec.optional.contains(&k.as_str())
                || attr_keys.contains(&k.as_str())
                || matches!(k.as_str(), "input_shape" | "input_type" | "output_type");
            if !used {
                self.warnings
                    .push(format!("{ctx}: The '{k}' field of node {ty} is not used"));
            }
        }
        Ok(node)
    }

    fn open_node(
        &mut self,
        name: &str,
        ty: &str,
        op: NodeOp,
        f: &[(String, Jv)],
        ctx: &str,
    ) -> StoreResult<Node> {
        if self.opts.unknown == Unsupported::Error {
            return Err(StoreError::UnsupportedFeature(format!(
                "{ctx}: Node type '{ty}' is unknown (Unsupported::Passthrough imports it\
                 as a custom operation)"
            )));
        }
        let mut node = Node::new(name, op);
        for (k, v) in f {
            if k == "type" {
                continue;
            }
            if matches!(v, Jv::Arr(_)) && is_numeric(v) {
                let t = self.tensor(ctx, k, v)?;
                node.params.insert(k.clone(), t);
            } else if let Some(val) = jv_to_value(v) {
                node.attrs.insert(k.clone(), val);
            } else {
                self.warnings
                    .push(format!("{ctx}: field '{k}' (null or object) missing"));
            }
        }
        self.warnings
            .push(format!("{ctx}: type '{ty}' imported as is"));
        Ok(node)
    }
}

/// Parses NIR JSON without accessing storage.
pub fn parse_nir_json(text: &str, opts: &NirImportOptions) -> StoreResult<NirImported> {
    if !opts.dtype.is_float() {
        return Err(StoreError::Invalid(format!(
            "weight storage type {} not float",
            opts.dtype
        )));
    }
    let root = parse_json(text)?;
    let Jv::Obj(fields) = &root else {
        return Err(bad("root NIR JSON must be an object".into()));
    };
    let mut imp = Importer {
        opts,
        tensors: Vec::new(),
        names: HashSet::new(),
        warnings: Vec::new(),
    };
    let mut graph = imp.graph(fields, "")?;
    if let Some(Jv::Str(v)) = get(fields, "version") {
        graph
            .meta
            .insert("nir.version".to_string(), Value::Str(v.clone()));
    }
    Ok(NirImported {
        graph,
        tensors: imp.tensors,
        warnings: imp.warnings,
        dtype: opts.dtype,
    })
}

impl NirImported {
    /// Writes the weights and graph to the writer of a single file.
    pub fn write_into<W: Write>(&self, w: &mut StoreWriter<W>) -> StoreResult<()> {
        for t in &self.tensors {
            if self.dtype == DType::F64 {
                w.add_slice::<f64>(&t.name, &t.shape, t.role, &t.data)?;
            } else {
                let data: Vec<f32> = t.data.iter().map(|x| *x as f32).collect();
                w.add_f32_as(&t.name, &t.shape, t.role, self.dtype, &data)?;
            }
        }
        w.set_graph(self.graph.clone())
    }

    /// Writes the weights and graph into a sharded writer.
    pub fn write_into_sharded(&self, w: &mut ShardedWriter) -> StoreResult<()> {
        for t in &self.tensors {
            if self.dtype == DType::F64 {
                w.add_slice::<f64>(&t.name, &t.shape, t.role, &t.data)?;
            } else {
                let data: Vec<f32> = t.data.iter().map(|x| *x as f32).collect();
                w.add_f32_as(&t.name, &t.shape, t.role, self.dtype, &data)?;
            }
        }
        w.set_graph(self.graph.clone())
    }
}

impl<W: Write> StoreWriter<W> {
    /// Adds a graph and weights from NIR JSON to the storage; returns warnings. Tensors
    /// are called `"node". "port"`.
    pub fn add_nir_json(
        &mut self,
        text: &str,
        opts: &NirImportOptions,
    ) -> StoreResult<Vec<String>> {
        let imp = parse_nir_json(text, opts)?;
        imp.write_into(self)?;
        Ok(imp.warnings)
    }
}

impl ShardedWriter {
    /// Same as [`StoreWriter::add_nir_json`] for sharded notation.
    pub fn add_nir_json(
        &mut self,
        text: &str,
        opts: &NirImportOptions,
    ) -> StoreResult<Vec<String>> {
        let imp = parse_nir_json(text, opts)?;
        imp.write_into_sharded(self)?;
        Ok(imp.warnings)
    }
}

/// Converts NIR JSON (text) to a `.syn` file; returns the write summary and warnings.
pub fn nir_json_to_syn(
    text: &str,
    syn_path: impl AsRef<Path>,
    write: WriteOptions,
    opts: &NirImportOptions,
) -> StoreResult<(WriteSummary, Vec<String>)> {
    let imp = parse_nir_json(text, opts)?;
    let summary = write_store_file(syn_path, write, |w| imp.write_into(w))?;
    Ok((summary, imp.warnings))
}

/// Same as [`nir_json_to_syn`], but JSON is read from a file.
pub fn nir_json_file_to_syn(
    json_path: impl AsRef<Path>,
    syn_path: impl AsRef<Path>,
    write: WriteOptions,
    opts: &NirImportOptions,
) -> StoreResult<(WriteSummary, Vec<String>)> {
    let text = fs::read_to_string(json_path)?;
    nir_json_to_syn(&text, syn_path, write, opts)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn parse(s: &str) -> StoreResult<Jv> {
        parse_json(s)
    }

    #[test]
    fn numbers_keep_their_kind() {
        assert_eq!(parse("1").unwrap(), Jv::Int(1));
        assert_eq!(parse("-7").unwrap(), Jv::Int(-7));
        assert_eq!(parse("18446744073709551615").unwrap(), Jv::UInt(u64::MAX));
        assert_eq!(parse("1.5").unwrap(), Jv::Float(1.5));
        assert_eq!(parse("1e3").unwrap(), Jv::Float(1000.0));
        assert_eq!(parse("-0.25E-1").unwrap(), Jv::Float(-0.025));
    }

    #[test]
    fn strings_and_escapes() {
        assert_eq!(
            parse(r#""a\"b\\c\n\u00e9""#).unwrap(),
            Jv::Str("a\"b\\c\né".into())
        );
        assert_eq!(parse(r#""\ud83d\ude00""#).unwrap(), Jv::Str("😀".into()));
        assert_eq!(parse("\"hello\"").unwrap(), Jv::Str("hello".into()));
        assert_eq!(parse("\u{feff}[]").unwrap(), Jv::Arr(vec![]));
    }

    #[test]
    fn structure() {
        let v = parse(r#" {"a": [1, 2.5, true, null], "b": {"c": "x"}} "#).unwrap();
        assert_eq!(
            v,
            Jv::Obj(vec![
                (
                    "a".into(),
                    Jv::Arr(vec![Jv::Int(1), Jv::Float(2.5), Jv::Bool(true), Jv::Null])
                ),
                ("b".into(), Jv::Obj(vec![("c".into(), Jv::Str("x".into()))])),
            ])
        );
    }

    #[test]
    fn malformed_documents_are_rejected() {
        for bad in [
            "",
            "[",
            "[1,]",
            "{\"a\":1,}",
            "{\"a\":1,\"a\":2}",
            "01",
            "1.",
            "1e",
            "-",
            "+1",
            "NaN",
            "Infinity",
            "[1] x",
            "\"abc",
            "\"\\x\"",
            "\"\\ud800\"",
            "\"\\udc00\"",
            "\"a\nb\"",
            "{1:2}",
            "tru",
            "1e999",
        ] {
            assert!(parse(bad).is_err(), "{bad:?} should be rejected");
        }
    }

    #[test]
    fn error_reports_position() {
        let e = parse("{\n  \"a\": ?\n}").unwrap_err().to_string();
        assert!(e.contains("line 2"), "{e}");
    }

    #[test]
    fn deep_nesting_does_not_overflow_the_stack() {
        let deep = "[".repeat(10_000);
        assert!(parse(&deep).is_err());
        let ok = format!("{}{}", "[".repeat(100), "]".repeat(100));
        assert!(parse(&ok).is_ok());
    }

    #[test]
    fn tensors_from_nested_arrays() {
        let (s, d) = to_tensor(&parse("[[1,2,3],[4,5,6]]").unwrap()).unwrap();
        assert_eq!((s, d), (vec![2, 3], vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0]));
        let (s, d) = to_tensor(&parse("0.5").unwrap()).unwrap();
        assert_eq!((s, d), (vec![], vec![0.5]));
        let (s, _) = to_tensor(&parse("[]").unwrap()).unwrap();
        assert_eq!(s, vec![0]);
        assert!(to_tensor(&parse("[[1,2],[3]]").unwrap()).is_err());
        assert!(to_tensor(&parse("[1,\"a\"]").unwrap()).is_err());
        assert!(to_tensor(&parse("[[1],2]").unwrap()).is_err());
    }
}
