//! Export graph and weights to JSON in the spirit of NIR.
//!
//! The original NIR is stored in HDF5; here is a text “safety” copy, which is convenient
//! read with your eyes and parse with any JSON parser. The structure repeats the NIR dictionary:
//!
//! ```text
//! {
//!   "version": "1.0",
//!   "nodes": { "fc1": {"type": "Affine", "weight": [[…]], "bias": […]}, … },
//!   "edges": [["in", "fc1"], ["fc1", "lif1"], …],
//!   "metadata": { … } // only if the graph has metadata
//! }
//! ```
//!
//! This is **not** the official NIR format: NIR does not have standard JSON, and the exact field names are
//! check with the version of the specification used (the version is written in `version` and specified in
//! [`NirJsonOptions`]). Weights are nested in nodes in nested arrays, so for large
//! network file turns out to be many times larger than `.syn`.
//!
//! ## Node matching
//!
//! | `neo-store` | NIR JSON |
//! |---|---|
//! | `Input` / `Output` | `Input` with `input_type: {"input": shape}` / `Output` with `output_type: {"output": shape}` |
//! | `Linear` without bias | `Linear` (`weight`) |
//! | `Linear` with bias, `Affine` | `Affine` (`weight`, `bias`) |
//! | `Conv1d`, `Conv2d` | the same, plus `input_shape` (spatial part of the input), `stride`, `padding`, `dilation`, `groups` |
//! | `Flatten` | `Flatten` with `input_type`, `start_dim`, `end_dim` |
//! | `SumPool2d`, `AvgPool2d` | the same (`kernel_size`, `stride`, `padding`) |
//! | `Scale`, `Threshold`, `Delay` | the same (`scale`, `threshold`, `delay`) |
//! | `I`, `IF`, `LI`, `LIF`, `CubaLI`, `CubaLIF` | the same, parameters - arrays by neurons |
//! | `Subgraph` | `NIRGraph` (nested `nodes` and `edges`) |
//! | `Alpha`, `Synaptic`, `Izhikevich`, `AdaptiveLIF`, custom | no analogue: error or [`Unsupported::Passthrough`] |
//!
//! The scalar parameters of neurons are stretched to a vector by the number of neurons if the node shapes
//! known (at the top level of the graph); in nested subgraphs they remain as is.
//!
//! ## What is lost
//!
//! The edge `neo-store` can have ports and a delay in steps. In NIR, an edge is just a pair of nodes, and
//! the delay is expressed by the `Delay` node in time (not in steps), so these edges are by default
//! give an error. With [`NirJsonOptions::lossy`] they are exported as normal pairs, and the loss
//! falls into [`NirJson::warnings`]. Recurrent edge marking itself does not lose anything:
//! in NIR cycles are defined by ordinary edges.

use std::collections::{BTreeMap, HashSet};
use std::fmt::{Display, Write as _};
use std::fs::{self, File};
use std::io::Write as _;
use std::path::Path;

use crate::dtype::DType;
use crate::error::{StoreError, StoreResult};
use crate::graph::{Graph, GraphShapes, Node, NodeOp, NodeShapes};
use crate::reader::Store;
use crate::shard::ShardedStore;
use crate::value::Value;
use crate::writer::tmp_path;

/// What to do with nodes that do not have an analogue in NIR.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum Unsupported {
    /// Refuse to export (default).
    #[default]
    Error,
    /// Write down the node as it is: `type` is the name of the operation, parameters and attributes are the fields of the node.
    /// The reader of NIR most likely will not understand such a node.
    Passthrough,
}

/// Export options.
#[derive(Debug, Clone)]
pub struct NirJsonOptions {
    /// Indentations and line breaks (tensors are still written in one line).
    pub pretty: bool,
    /// Field value `version`.
    pub version: String,
    /// Nodes without an analogue in NIR.
    pub unsupported: Unsupported,
    /// Allow loss of information (delays and edge ports, byte metadata): instead
    /// errors in [`NirJson::warnings`] a warning appears.
    pub lossy: bool,
}

impl Default for NirJsonOptions {
    fn default() -> Self {
        Self {
            pretty: true,
            version: "1.0".to_string(),
            unsupported: Unsupported::Error,
            lossy: false,
        }
    }
}

/// Export result.
#[derive(Debug, Clone)]
pub struct NirJson {
    /// JSON document.
    pub json: String,
    /// What had to be lost or brought closer (empty if the export is accurate).
    pub warnings: Vec<String>,
}

// ═══════════════════════════════════════════════════════════════════════════
// JSON tree and record
// ═══════════════════════════════════════════════════════════════════════════

enum Data {
    F32(Vec<f32>),
    F64(Vec<f64>),
}

struct Tensor {
    shape: Vec<u64>,
    data: Data,
}

impl Tensor {
    fn numel(&self) -> usize {
        match &self.data {
            Data::F32(v) => v.len(),
            Data::F64(v) => v.len(),
        }
    }

    /// Multiplies a single value to a vector of length `n`.
    fn broadcast(self, n: usize) -> Tensor {
        let data = match self.data {
            Data::F32(v) => Data::F32(vec![v[0]; n]),
            Data::F64(v) => Data::F64(vec![v[0]; n]),
        };
        Tensor {
            shape: vec![n as u64],
            data,
        }
    }
}

enum J {
    Bool(bool),
    Num(String),
    Str(String),
    Arr(Vec<J>),
    Obj(Vec<(String, J)>),
    Tensor(Tensor),
}

fn num(v: impl Display) -> J {
    J::Num(v.to_string())
}

fn shape_j(shape: &[u64]) -> J {
    J::Arr(shape.iter().map(num).collect())
}

fn push_str_escaped(out: &mut String, s: &str) {
    out.push('"');
    for c in s.chars() {
        match c {
            '"' => out.push_str("\\\""),
            '\\' => out.push_str("\\\\"),
            '\n' => out.push_str("\\n"),
            '\r' => out.push_str("\\r"),
            '\t' => out.push_str("\\t"),
            c if (c as u32) < 0x20 => {
                let _ = write!(out, "\\u{:04x}", c as u32);
            }
            c => out.push(c),
        }
    }
    out.push('"');
}

fn write_nested<T: Display>(out: &mut String, shape: &[u64], data: &[T]) {
    match shape.split_first() {
        None => {
            if let Some(v) = data.first() {
                let _ = write!(out, "{v}");
            }
        }
        Some((&n, rest)) => {
            out.push('[');
            if rest.is_empty() {
                for (i, v) in data.iter().enumerate() {
                    if i > 0 {
                        out.push(',');
                    }
                    let _ = write!(out, "{v}");
                }
            } else if n > 0 {
                let step = data.len() / n as usize;
                for i in 0..n as usize {
                    if i > 0 {
                        out.push(',');
                    }
                    write_nested(out, rest, &data[i * step..(i + 1) * step]);
                }
            }
            out.push(']');
        }
    }
}

fn indent(out: &mut String, level: usize) {
    for _ in 0..level {
        out.push_str("  ");
    }
}

fn is_scalar(j: &J) -> bool {
    matches!(j, J::Bool(_) | J::Num(_) | J::Str(_))
}

fn write_j(j: &J, out: &mut String, pretty: bool, level: usize) {
    match j {
        J::Bool(b) => out.push_str(if *b { "true" } else { "false" }),
        J::Num(s) => out.push_str(s),
        J::Str(s) => push_str_escaped(out, s),
        J::Tensor(t) => match &t.data {
            Data::F32(v) => write_nested(out, &t.shape, v),
            Data::F64(v) => write_nested(out, &t.shape, v),
        },
        J::Arr(items) => {
            if items.is_empty() {
                out.push_str("[]");
            } else if !pretty || items.iter().all(is_scalar) {
                out.push('[');
                for (i, it) in items.iter().enumerate() {
                    if i > 0 {
                        out.push_str(if pretty { ", " } else { "," });
                    }
                    write_j(it, out, pretty, level);
                }
                out.push(']');
            } else {
                out.push_str("[\n");
                for (i, it) in items.iter().enumerate() {
                    if i > 0 {
                        out.push_str(",\n");
                    }
                    indent(out, level + 1);
                    write_j(it, out, pretty, level + 1);
                }
                out.push('\n');
                indent(out, level);
                out.push(']');
            }
        }
        J::Obj(fields) => {
            if fields.is_empty() {
                out.push_str("{}");
            } else if !pretty {
                out.push('{');
                for (i, (k, v)) in fields.iter().enumerate() {
                    if i > 0 {
                        out.push(',');
                    }
                    push_str_escaped(out, k);
                    out.push(':');
                    write_j(v, out, false, 0);
                }
                out.push('}');
            } else {
                out.push_str("{\n");
                for (i, (k, v)) in fields.iter().enumerate() {
                    if i > 0 {
                        out.push_str(",\n");
                    }
                    indent(out, level + 1);
                    push_str_escaped(out, k);
                    out.push_str(": ");
                    write_j(v, out, true, level + 1);
                }
                out.push('\n');
                indent(out, level);
                out.push('}');
            }
        }
    }
}

// ═══════════════════════════════════════════════════════════════════════════
// Export
// ═══════════════════════════════════════════════════════════════════════════

type Reader<'a> = &'a dyn Fn(&str) -> StoreResult<Tensor>;

struct Exporter<'a> {
    read: Reader<'a>,
    opts: &'a NirJsonOptions,
    warnings: Vec<String>,
}

fn unsupported(msg: String) -> StoreError {
    StoreError::UnsupportedFeature(msg)
}

fn finite(name: &str, t: &Tensor) -> StoreResult<()> {
    let ok = match &t.data {
        Data::F32(v) => v.iter().all(|x| x.is_finite()),
        Data::F64(v) => v.iter().all(|x| x.is_finite()),
    };
    if ok {
        Ok(())
    } else {
        Err(StoreError::Invalid(format!(
            "tensor '{name}': NaN or infinity is not representable in JSON"
        )))
    }
}

fn read_tensor(store: &Store, name: &str) -> StoreResult<Tensor> {
    let info = store.info(name)?;
    if info.attrs.contains_key("sparse.shape") {
        return Err(unsupported(format!(
            "tensor '{name}' is sparse (CSR): exporting sparse weights to NIR JSON is not supported"
        )));
    }
    let shape = info.shape.clone();
    let data = match info.dtype {
        DType::F64 => Data::F64(store.read_f64(name)?),
        d if d.is_float() => Data::F32(store.read_f32(name)?),
        other => {
            return Err(StoreError::DTypeMismatch {
                name: name.to_string(),
                expected: "float (f64/f32/f16/bf16)".into(),
                actual: other.name(),
            });
        }
    };
    let t = Tensor { shape, data };
    finite(name, &t)?;
    Ok(t)
}

impl Exporter<'_> {
    fn value(&mut self, v: &Value, ctx: &str) -> StoreResult<Option<J>> {
        Ok(Some(match v {
            Value::Bool(b) => J::Bool(*b),
            Value::I64(x) => num(x),
            Value::U64(x) => num(x),
            Value::F64(x) => {
                if !x.is_finite() {
                    return Err(StoreError::Invalid(format!(
                        "{ctx}: NaN or infinity is not representable in JSON"
                    )));
                }
                num(x)
            }
            Value::Str(s) => J::Str(s.clone()),
            Value::Bytes(_) => {
                if self.opts.lossy {
                    self.warnings.push(format!("{ctx}: byte value missing"));
                    return Ok(None);
                }
                return Err(unsupported(format!(
                    "{ctx}: byte metadata is not representable in JSON (enable lossy)"
                )));
            }
            Value::List(items) => {
                let mut out = Vec::with_capacity(items.len());
                for (i, it) in items.iter().enumerate() {
                    if let Some(j) = self.value(it, &format!("{ctx}[{i}]"))? {
                        out.push(j);
                    }
                }
                J::Arr(out)
            }
        }))
    }

    fn meta(&mut self, meta: &BTreeMap<String, Value>, ctx: &str) -> StoreResult<Vec<(String, J)>> {
        let mut out = Vec::new();
        for (k, v) in meta {
            if let Some(j) = self.value(v, &format!("{ctx}.{k}"))? {
                out.push((k.clone(), j));
            }
        }
        Ok(out)
    }

    fn attr(&mut self, node: &Node, key: &str, ctx: &str) -> StoreResult<Option<J>> {
        match node.attrs.get(key) {
            Some(v) => self.value(v, &format!("{ctx}.{key}")),
            None => Ok(None),
        }
    }

    fn attr_or(&mut self, node: &Node, key: &str, default: i64, ctx: &str) -> StoreResult<J> {
        Ok(self.attr(node, key, ctx)?.unwrap_or_else(|| num(default)))
    }

    /// Tensor parameter `port`; a single value is multiplied to `broadcast` neurons.
    fn param(
        &mut self,
        node: &Node,
        port: &str,
        broadcast: Option<usize>,
    ) -> StoreResult<Option<J>> {
        let Some(name) = node.params.get(port) else {
            return Ok(None);
        };
        let mut t = (self.read)(name)?;
        if let Some(n) = broadcast
            && t.numel() == 1
            && n > 1
            && node.op.is_neuron()
        {
            t = t.broadcast(n);
        }
        Ok(Some(J::Tensor(t)))
    }

    fn graph(
        &mut self,
        g: &Graph,
        shapes: Option<&GraphShapes>,
        path: &str,
        depth: usize,
    ) -> StoreResult<Vec<(String, J)>> {
        if depth > crate::codec::MAX_DEPTH {
            return Err(StoreError::Graph("too deep nesting of subgraphs".into()));
        }
        let mut nodes = Vec::with_capacity(g.nodes.len());
        for n in &g.nodes {
            let ns = shapes.and_then(|s| s.nodes.get(&n.name));
            let ctx = format!("{path}{}", n.name);
            nodes.push((n.name.clone(), self.node(n, ns, &ctx, depth)?));
        }

        let mut edges = Vec::with_capacity(g.edges.len());
        let mut seen = HashSet::new();
        for e in &g.edges {
            let plain_ports = e.src_port == "output" && e.dst_port == "input";
            if !plain_ports || e.delay != 0 {
                let what = format!(
                    "{path} edge {} → {}: ports “{}” → “{}” and delay {} steps are not expressible in NIR",
                    e.src, e.dst, e.src_port, e.dst_port, e.delay
                );
                if self.opts.lossy {
                    self.warnings.push(what);
                } else {
                    return Err(unsupported(format!("{what} (enable lossy)")));
                }
            }
            if seen.insert((e.src.as_str(), e.dst.as_str())) {
                edges.push(J::Arr(vec![J::Str(e.src.clone()), J::Str(e.dst.clone())]));
            }
        }

        let mut out = vec![
            ("nodes".to_string(), J::Obj(nodes)),
            ("edges".to_string(), J::Arr(edges)),
        ];
        if !g.meta.is_empty() {
            let m = self.meta(&g.meta, &format!("{path}metadata"))?;
            if !m.is_empty() {
                out.push(("metadata".to_string(), J::Obj(m)));
            }
        }
        Ok(out)
    }

    fn node(
        &mut self,
        n: &Node,
        ns: Option<&NodeShapes>,
        ctx: &str,
        depth: usize,
    ) -> StoreResult<J> {
        let neurons = ns.map(|s| s.output.iter().product::<u64>() as usize);
        let mut f: Vec<(String, J)> = Vec::new();
        let ty: String;

        match &n.op {
            NodeOp::Input | NodeOp::Output => {
                let shape = n
                    .attrs
                    .get("shape")
                    .and_then(|v| v.as_u64_list())
                    .ok_or_else(|| {
                        StoreError::Graph(format!("{ctx}: node does not have a shape attribute"))
                    })?;
                let (key, port) = if n.op == NodeOp::Input {
                    ("input_type", "input")
                } else {
                    ("output_type", "output")
                };
                ty = n.op.name().to_string();
                f.push((
                    key.to_string(),
                    J::Obj(vec![(port.to_string(), shape_j(&shape))]),
                ));
            }
            NodeOp::Linear | NodeOp::Affine => {
                let has_bias = n.params.contains_key("bias");
                ty = if has_bias { "Affine" } else { "Linear" }.to_string();
                f.extend(
                    self.param(n, "weight", None)?
                        .map(|j| ("weight".to_string(), j)),
                );
                f.extend(
                    self.param(n, "bias", None)?
                        .map(|j| ("bias".to_string(), j)),
                );
            }
            NodeOp::Conv1d | NodeOp::Conv2d => {
                ty = n.op.name().to_string();
                match ns.map(|s| s.input.as_slice()) {
                    Some([_, spatial @ ..]) if !spatial.is_empty() => {
                        let j = if spatial.len() == 1 {
                            num(spatial[0])
                        } else {
                            shape_j(spatial)
                        };
                        f.push(("input_shape".to_string(), j));
                    }
                    _ => self.warnings.push(format!(
                        "{ctx}: input shape unknown, input_shape not recorded"
                    )),
                }
                f.push(("weight".to_string(), self.need(n, "weight", ctx)?));
                f.extend(
                    self.param(n, "bias", None)?
                        .map(|j| ("bias".to_string(), j)),
                );
                f.push(("stride".to_string(), self.attr_or(n, "stride", 1, ctx)?));
                f.push(("padding".to_string(), self.attr_or(n, "padding", 0, ctx)?));
                f.push(("dilation".to_string(), self.attr_or(n, "dilation", 1, ctx)?));
                f.push(("groups".to_string(), self.attr_or(n, "groups", 1, ctx)?));
            }
            NodeOp::Flatten => {
                ty = "Flatten".to_string();
                if let Some(s) = ns {
                    f.push((
                        "input_type".to_string(),
                        J::Obj(vec![("input".to_string(), shape_j(&s.input))]),
                    ));
                }
                f.push((
                    "start_dim".to_string(),
                    self.attr_or(n, "start_dim", 0, ctx)?,
                ));
                f.push(("end_dim".to_string(), self.attr_or(n, "end_dim", -1, ctx)?));
            }
            NodeOp::SumPool2d | NodeOp::AvgPool2d => {
                ty = n.op.name().to_string();
                for key in ["kernel_size", "stride", "padding"] {
                    f.extend(self.attr(n, key, ctx)?.map(|j| (key.to_string(), j)));
                }
            }
            NodeOp::Scale | NodeOp::Threshold | NodeOp::Delay => {
                ty = n.op.name().to_string();
                let port = match n.op {
                    NodeOp::Scale => "scale",
                    NodeOp::Threshold => "threshold",
                    _ => "delay",
                };
                f.push((port.to_string(), self.need(n, port, ctx)?));
            }
            op if op.is_neuron() && !n.params.is_empty() && !is_open(op) => {
                ty = op.name().to_string();
                for port in n.params.keys() {
                    f.extend(self.param(n, port, neurons)?.map(|j| (port.clone(), j)));
                }
            }
            NodeOp::Subgraph => {
                ty = "NIRGraph".to_string();
                let sg = n.subgraph.as_deref().ok_or_else(|| {
                    StoreError::Graph(format!("{ctx}: Subgraph node does not have a subgraph"))
                })?;
                f.extend(self.graph(sg, None, &format!("{ctx}/"), depth + 1)?);
            }
            other => {
                // Alpha, Synaptic, Izhikevich, AdaptiveLIF, custom operations
                // (and neurons without parameters)
                if self.opts.unsupported == Unsupported::Error && is_open(other) {
                    return Err(unsupported(format!(
                        "{ctx}: operation {} has no counterpart in NIR (Unsupported::Passthrough \
                         will keep the node as is)",
                        other.name()
                    )));
                }
                if is_open(other) {
                    self.warnings.push(format!(
                        "{ctx}: operation {} is written as is, reader NIR will not understand it",
                        other.name()
                    ));
                }
                ty = other.name().to_string();
                for port in n.params.keys() {
                    f.extend(self.param(n, port, neurons)?.map(|j| (port.clone(), j)));
                }
                for (k, v) in &n.attrs {
                    if let Some(j) = self.value(v, &format!("{ctx}.{k}"))? {
                        f.push((k.clone(), j));
                    }
                }
            }
        }

        let mut fields = vec![("type".to_string(), J::Str(ty))];
        fields.extend(f);
        Ok(J::Obj(fields))
    }

    /// Required parameter.
    fn need(&mut self, n: &Node, port: &str, ctx: &str) -> StoreResult<J> {
        self.param(n, port, None)?
            .ok_or_else(|| StoreError::Graph(format!("{ctx}: parameter “{port}” is not specified")))
    }
}

/// Operations with arbitrary parameters that are not in NIR.
fn is_open(op: &NodeOp) -> bool {
    matches!(
        op,
        NodeOp::Alpha
            | NodeOp::Synaptic
            | NodeOp::Izhikevich
            | NodeOp::AdaptiveLif
            | NodeOp::Custom(_)
    )
}

fn export(
    graph: &Graph,
    shapes: Option<&GraphShapes>,
    read: Reader<'_>,
    opts: &NirJsonOptions,
) -> StoreResult<NirJson> {
    let mut ex = Exporter {
        read,
        opts,
        warnings: Vec::new(),
    };
    let mut fields = vec![("version".to_string(), J::Str(opts.version.clone()))];
    fields.extend(ex.graph(graph, shapes, "", 0)?);
    let mut json = String::new();
    write_j(&J::Obj(fields), &mut json, opts.pretty, 0);
    json.push('\n');
    Ok(NirJson {
        json,
        warnings: ex.warnings,
    })
}

fn write_atomic(path: &Path, text: &str) -> StoreResult<()> {
    if let Some(parent) = path.parent()
        && !parent.as_os_str().is_empty()
    {
        fs::create_dir_all(parent)?;
    }
    let tmp = tmp_path(path);
    let res = (|| -> StoreResult<()> {
        let mut f = File::create(&tmp)?;
        f.write_all(text.as_bytes())?;
        f.sync_all()?;
        Ok(())
    })();
    match res.and_then(|_| fs::rename(&tmp, path).map_err(StoreError::from)) {
        Ok(()) => Ok(()),
        Err(e) => {
            let _ = fs::remove_file(&tmp);
            Err(e)
        }
    }
}

fn no_graph() -> StoreError {
    StoreError::Graph("there is no graph in the storage: there is nothing to export to NIR".into())
}

impl Store {
    /// Exports the graph and weights to JSON in the spirit of NIR (see [module](crate::nir_json)). The graph
    /// is pre-checked against the weights; shapes are needed for `input_shape` and for broadcasting
    /// scalar neuron parameters.
    pub fn to_nir_json(&self, opts: &NirJsonOptions) -> StoreResult<NirJson> {
        let graph = self.graph().ok_or_else(no_graph)?;
        let shapes = self.validate_graph()?;
        export(
            graph,
            shapes.as_ref(),
            &|name| read_tensor(self, name),
            opts,
        )
    }

    /// Writes [`Store::to_nir_json`] to a file atomically (to a temporary file and renaming);
    /// returns warnings.
    pub fn write_nir_json(
        &self,
        path: impl AsRef<Path>,
        opts: &NirJsonOptions,
    ) -> StoreResult<Vec<String>> {
        let out = self.to_nir_json(opts)?;
        write_atomic(path.as_ref(), &out.json)?;
        Ok(out.warnings)
    }
}

impl ShardedStore {
    /// Same as [`Store::to_nir_json`] for a sharded set.
    pub fn to_nir_json(&self, opts: &NirJsonOptions) -> StoreResult<NirJson> {
        let graph = self.graph().ok_or_else(no_graph)?;
        let shapes = self.validate_graph()?;
        export(
            graph,
            shapes.as_ref(),
            &|name| read_tensor(self.store_for(name)?, name),
            opts,
        )
    }

    /// Same as [`Store::write_nir_json`] for a sharded set.
    pub fn write_nir_json(
        &self,
        path: impl AsRef<Path>,
        opts: &NirJsonOptions,
    ) -> StoreResult<Vec<String>> {
        let out = self.to_nir_json(opts)?;
        write_atomic(path.as_ref(), &out.json)?;
        Ok(out.warnings)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn strings_are_escaped() {
        let mut s = String::new();
        push_str_escaped(&mut s, "a\"b\\c\nd\u{1}é");
        assert_eq!(s, "\"a\\\"b\\\\c\\nd\\u0001é\"");
    }

    #[test]
    fn tensors_are_nested_by_shape() {
        let mut s = String::new();
        write_nested(&mut s, &[2, 3], &[1.0f32, 2.0, 3.0, 4.5, 5.0, 6.0]);
        assert_eq!(s, "[[1,2,3],[4.5,5,6]]");
        let mut s = String::new();
        write_nested(&mut s, &[], &[0.25f64]);
        assert_eq!(s, "0.25");
        let mut s = String::new();
        write_nested(&mut s, &[0, 3], &[] as &[f32]);
        assert_eq!(s, "[]");
        let mut s = String::new();
        write_nested(&mut s, &[2, 0], &[] as &[f32]);
        assert_eq!(s, "[[],[]]");
        let mut s = String::new();
        write_nested(&mut s, &[2, 2, 1], &[1.0f32, 2.0, 3.0, 4.0]);
        assert_eq!(s, "[[[1],[2]],[[3],[4]]]");
    }

    #[test]
    fn compact_and_pretty_writers() {
        let j = J::Obj(vec![
            ("a".into(), J::Arr(vec![num(1), num(2)])),
            ("b".into(), J::Obj(vec![("c".into(), J::Bool(true))])),
            ("e".into(), J::Arr(vec![])),
        ]);
        let mut c = String::new();
        write_j(&j, &mut c, false, 0);
        assert_eq!(c, "{\"a\":[1,2],\"b\":{\"c\":true},\"e\":[]}");
        let mut p = String::new();
        write_j(&j, &mut p, true, 0);
        assert_eq!(
            p,
            "{\n  \"a\": [1, 2],\n  \"b\": {\n    \"c\": true\n  },\n  \"e\": []\n}"
        );
    }
}
