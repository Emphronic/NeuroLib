//! The network graph is analogous to NIR (Neuromorphic Intermediate Representation), but more strict.
//!
//! Differences from NIR:
//! * weights are not nested in nodes, but are stored in storage as named tensors; node refers to
//!   them by name (`params`: port → tensor name). One tensor can be reused;
//! * when writing, the graph **is checked against the weights**: the tensor exists, the shape
//!   matches the shape of the node input, and neuron parameters are a scalar or a vector of length “number of neurons”;
//! * the shapes of all nodes are inferred (Linear/Affine/Conv1d/Conv2d/Flatten/pooling/subgraphs);
//! * cycles are allowed only through edges marked `recurrent`; an edge has a delay in steps;
//! * nested subgraphs, arbitrary node and graph attributes, custom operations.
//!
//! shapes - for one time step, without a batch (as in NIR).

use std::collections::{BTreeMap, HashMap, HashSet, VecDeque};

use crate::codec::{MAX_DEPTH, R, W, fmt_err, r_meta, w_meta};
use crate::error::{StoreError, StoreResult};
use crate::value::{Meta, Value};

fn gerr(m: impl Into<String>) -> StoreError {
    StoreError::Graph(m.into())
}

/// Node type.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum NodeOp {
    /// Input (`attrs["shape"]`).
    Input,
    /// Output (`attrs["shape"]`).
    Output,
    /// Linear layer: `weight [out, in]`, optional `bias [out]`.
    Linear,
    /// Affine layer: `weight [out, in]`, `bias [out]`.
    Affine,
    /// Convolution 1D: input `[C, L]`, `weight [OC, C/groups, K]`.
    Conv1d,
    /// 2D convolution: input `[C, H, W]`, `weight [OC, C/groups, KH, KW]`.
    Conv2d,
    /// Flattening dimensions (`start_dim`, `end_dim`).
    Flatten,
    /// 2D sum pooling.
    SumPool2d,
    /// 2D average pooling.
    AvgPool2d,
    /// Scale: `scale`.
    Scale,
    /// Threshold: `threshold`.
    Threshold,
    /// Delay: `delay`.
    Delay,
    /// Integrator (NIR `I`): `r`.
    Integrator,
    /// Integrate-and-Fire: `r`, `v_reset`, `v_threshold`.
    If,
    /// Leaky integrator: `tau`, `r`, `v_leak`.
    Li,
    /// Leaky integrate-and-fire: `tau`, `r`, `v_leak`, `v_reset`, `v_threshold`.
    Lif,
    /// Current-based LI: `tau_syn`, `tau_mem`, `r`, `v_leak`, `w_in`.
    CubaLi,
    /// Current-based LIF: as `CubaLi` plus `v_reset`, `v_threshold`.
    CubaLif,
    /// Alpha neuron (arbitrary named parameters).
    Alpha,
    /// Synaptic (double exponential).
    Synaptic,
    /// Izhikevich neuron.
    Izhikevich,
    /// Adaptive LIF.
    AdaptiveLif,
    /// Nested graph (`Node::subgraph`).
    Subgraph,
    /// Custom operation; parameters are arbitrary, the shape is inferred
    /// or set by `attrs["output_shape"]`.
    Custom(String),
}

pub(crate) struct OpSpec {
    pub(crate) required: &'static [&'static str],
    pub(crate) optional: &'static [&'static str],
    pub(crate) open: bool,
}

impl NodeOp {
    /// Code on disk.
    pub fn code(&self) -> u16 {
        match self {
            NodeOp::Input => 1,
            NodeOp::Output => 2,
            NodeOp::Linear => 3,
            NodeOp::Affine => 4,
            NodeOp::Conv1d => 5,
            NodeOp::Conv2d => 6,
            NodeOp::Flatten => 7,
            NodeOp::SumPool2d => 8,
            NodeOp::AvgPool2d => 9,
            NodeOp::Scale => 10,
            NodeOp::Threshold => 11,
            NodeOp::Delay => 12,
            NodeOp::Integrator => 20,
            NodeOp::If => 21,
            NodeOp::Li => 22,
            NodeOp::Lif => 23,
            NodeOp::CubaLi => 24,
            NodeOp::CubaLif => 25,
            NodeOp::Alpha => 26,
            NodeOp::Synaptic => 27,
            NodeOp::Izhikevich => 28,
            NodeOp::AdaptiveLif => 29,
            NodeOp::Subgraph => 100,
            NodeOp::Custom(_) => 0xFFFF,
        }
    }

    pub(crate) fn from_code(c: u16) -> Option<NodeOp> {
        Some(match c {
            1 => NodeOp::Input,
            2 => NodeOp::Output,
            3 => NodeOp::Linear,
            4 => NodeOp::Affine,
            5 => NodeOp::Conv1d,
            6 => NodeOp::Conv2d,
            7 => NodeOp::Flatten,
            8 => NodeOp::SumPool2d,
            9 => NodeOp::AvgPool2d,
            10 => NodeOp::Scale,
            11 => NodeOp::Threshold,
            12 => NodeOp::Delay,
            20 => NodeOp::Integrator,
            21 => NodeOp::If,
            22 => NodeOp::Li,
            23 => NodeOp::Lif,
            24 => NodeOp::CubaLi,
            25 => NodeOp::CubaLif,
            26 => NodeOp::Alpha,
            27 => NodeOp::Synaptic,
            28 => NodeOp::Izhikevich,
            29 => NodeOp::AdaptiveLif,
            100 => NodeOp::Subgraph,
            _ => return None,
        })
    }

    /// Operation name (for canonical ones - as in NIR).
    pub fn name(&self) -> &str {
        match self {
            NodeOp::Input => "Input",
            NodeOp::Output => "Output",
            NodeOp::Linear => "Linear",
            NodeOp::Affine => "Affine",
            NodeOp::Conv1d => "Conv1d",
            NodeOp::Conv2d => "Conv2d",
            NodeOp::Flatten => "Flatten",
            NodeOp::SumPool2d => "SumPool2d",
            NodeOp::AvgPool2d => "AvgPool2d",
            NodeOp::Scale => "Scale",
            NodeOp::Threshold => "Threshold",
            NodeOp::Delay => "Delay",
            NodeOp::Integrator => "I",
            NodeOp::If => "IF",
            NodeOp::Li => "LI",
            NodeOp::Lif => "LIF",
            NodeOp::CubaLi => "CubaLI",
            NodeOp::CubaLif => "CubaLIF",
            NodeOp::Alpha => "Alpha",
            NodeOp::Synaptic => "Synaptic",
            NodeOp::Izhikevich => "Izhikevich",
            NodeOp::AdaptiveLif => "AdaptiveLIF",
            NodeOp::Subgraph => "Subgraph",
            NodeOp::Custom(s) => s.as_str(),
        }
    }

    /// Operation by name; the unknown name becomes `Custom`.
    pub fn from_name(n: &str) -> NodeOp {
        match n {
            "Input" => NodeOp::Input,
            "Output" => NodeOp::Output,
            "Linear" => NodeOp::Linear,
            "Affine" => NodeOp::Affine,
            "Conv1d" => NodeOp::Conv1d,
            "Conv2d" => NodeOp::Conv2d,
            "Flatten" => NodeOp::Flatten,
            "SumPool2d" => NodeOp::SumPool2d,
            "AvgPool2d" => NodeOp::AvgPool2d,
            "Scale" => NodeOp::Scale,
            "Threshold" => NodeOp::Threshold,
            "Delay" => NodeOp::Delay,
            "I" => NodeOp::Integrator,
            "IF" => NodeOp::If,
            "LI" => NodeOp::Li,
            "LIF" => NodeOp::Lif,
            "CubaLI" => NodeOp::CubaLi,
            "CubaLIF" => NodeOp::CubaLif,
            "Alpha" => NodeOp::Alpha,
            "Synaptic" => NodeOp::Synaptic,
            "Izhikevich" => NodeOp::Izhikevich,
            "AdaptiveLIF" => NodeOp::AdaptiveLif,
            "Subgraph" => NodeOp::Subgraph,
            other => NodeOp::Custom(other.to_string()),
        }
    }

    /// Neural operation (the shape is preserved, the parameters are a scalar or a vector across neurons).
    pub fn is_neuron(&self) -> bool {
        matches!(
            self,
            NodeOp::Integrator
                | NodeOp::If
                | NodeOp::Li
                | NodeOp::Lif
                | NodeOp::CubaLi
                | NodeOp::CubaLif
                | NodeOp::Alpha
                | NodeOp::Synaptic
                | NodeOp::Izhikevich
                | NodeOp::AdaptiveLif
        )
    }

    pub(crate) fn spec(&self) -> OpSpec {
        let (required, optional, open): (&'static [&'static str], &'static [&'static str], bool) =
            match self {
                NodeOp::Linear => (&["weight"], &["bias"], false),
                NodeOp::Affine => (&["weight", "bias"], &[], false),
                NodeOp::Conv1d | NodeOp::Conv2d => (&["weight"], &["bias"], false),
                NodeOp::Scale => (&["scale"], &[], false),
                NodeOp::Threshold => (&["threshold"], &[], false),
                NodeOp::Delay => (&["delay"], &[], false),
                NodeOp::Integrator => (&["r"], &[], false),
                NodeOp::If => (&["r", "v_reset", "v_threshold"], &[], false),
                NodeOp::Li => (&["tau", "r", "v_leak"], &[], false),
                NodeOp::Lif => (
                    &["tau", "r", "v_leak", "v_reset", "v_threshold"],
                    &[],
                    false,
                ),
                NodeOp::CubaLi => (&["tau_syn", "tau_mem", "r", "v_leak", "w_in"], &[], false),
                NodeOp::CubaLif => (
                    &[
                        "tau_syn",
                        "tau_mem",
                        "r",
                        "v_leak",
                        "v_reset",
                        "v_threshold",
                        "w_in",
                    ],
                    &[],
                    false,
                ),
                NodeOp::Alpha
                | NodeOp::Synaptic
                | NodeOp::Izhikevich
                | NodeOp::AdaptiveLif
                | NodeOp::Custom(_) => (&[], &[], true),
                _ => (&[], &[], false),
            };
        OpSpec {
            required,
            optional,
            open,
        }
    }
}

/// Graph node.
#[derive(Debug, Clone, PartialEq)]
pub struct Node {
    /// A unique name within the graph.
    pub name: String,
    /// Operation.
    pub op: NodeOp,
    /// Hyperparameters (stride, padding, shape, ...).
    pub attrs: Meta,
    /// Parameter port → tensor name in storage.
    pub params: BTreeMap<String, String>,
    /// Nested graph (only for `NodeOp::Subgraph`).
    pub subgraph: Option<Box<Graph>>,
}

impl Node {
    /// Empty node.
    pub fn new(name: impl Into<String>, op: NodeOp) -> Self {
        Self {
            name: name.into(),
            op,
            attrs: Meta::new(),
            params: BTreeMap::new(),
            subgraph: None,
        }
    }

    /// Input with the shape for one step.
    pub fn input(name: impl Into<String>, shape: &[u64]) -> Self {
        Self::new(name, NodeOp::Input).attr("shape", Value::uints(shape))
    }

    /// Output with the shape for one step.
    pub fn output(name: impl Into<String>, shape: &[u64]) -> Self {
        Self::new(name, NodeOp::Output).attr("shape", Value::uints(shape))
    }

    /// Linear layer.
    pub fn linear(name: impl Into<String>, weight: impl Into<String>) -> Self {
        Self::new(name, NodeOp::Linear).param("weight", weight)
    }

    /// Affine layer.
    pub fn affine(
        name: impl Into<String>,
        weight: impl Into<String>,
        bias: impl Into<String>,
    ) -> Self {
        Self::new(name, NodeOp::Affine)
            .param("weight", weight)
            .param("bias", bias)
    }

    /// A neural node in which all required parameters are under names
    /// `"{prefix}.{port}"` (for example, `lif1.tau`, `lif1.v_threshold`).
    pub fn neuron(name: impl Into<String>, op: NodeOp, prefix: &str) -> Self {
        let mut n = Self::new(name, op);
        for port in n.op.spec().required {
            n.params
                .insert((*port).to_string(), format!("{prefix}.{port}"));
        }
        n
    }

    /// Nested graph.
    pub fn with_subgraph(name: impl Into<String>, g: Graph) -> Self {
        let mut n = Self::new(name, NodeOp::Subgraph);
        n.subgraph = Some(Box::new(g));
        n
    }

    /// Binds a parameter to a tensor.
    pub fn param(mut self, port: &str, tensor: impl Into<String>) -> Self {
        self.params.insert(port.to_string(), tensor.into());
        self
    }

    /// Specifies an attribute.
    pub fn attr(mut self, key: &str, v: impl Into<Value>) -> Self {
        self.attrs.insert(key.to_string(), v.into());
        self
    }
}

/// Graph edge.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct Edge {
    /// Source node.
    pub src: String,
    /// Source output port.
    pub src_port: String,
    /// Destination node.
    pub dst: String,
    /// Destination input port.
    pub dst_port: String,
    /// Transmission delay in steps.
    pub delay: u32,
    /// Feedback (recurrent) connection: does not participate in topological order.
    pub recurrent: bool,
}

impl Edge {
    /// Default edge between ports (`output` → `input`).
    pub fn new(src: impl Into<String>, dst: impl Into<String>) -> Self {
        Self {
            src: src.into(),
            src_port: "output".to_string(),
            dst: dst.into(),
            dst_port: "input".to_string(),
            delay: 0,
            recurrent: false,
        }
    }

    /// With a delay.
    pub fn with_delay(mut self, steps: u32) -> Self {
        self.delay = steps;
        self
    }

    /// Marks an edge as recurrent.
    pub fn as_recurrent(mut self) -> Self {
        self.recurrent = true;
        self
    }
}

/// Node shapes for one step.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct NodeShapes {
    /// The input shape (for `Input` is the same as the output).
    pub input: Vec<u64>,
    /// Output shape.
    pub output: Vec<u64>,
}

/// Result of shape inference.
#[derive(Debug, Clone, Default)]
pub struct GraphShapes {
    /// Shapes by top-level node names.
    pub nodes: BTreeMap<String, NodeShapes>,
}

/// Network graph.
#[derive(Debug, Clone, Default, PartialEq)]
pub struct Graph {
    /// Nodes.
    pub nodes: Vec<Node>,
    /// Edges.
    pub edges: Vec<Edge>,
    /// Graph metadata.
    pub meta: Meta,
}

type Lookup<'a> = &'a dyn Fn(&str) -> Option<Vec<u64>>;

fn numel(shape: &[u64]) -> StoreResult<u64> {
    shape
        .iter()
        .try_fold(1u64, |a, d| a.checked_mul(*d))
        .ok_or_else(|| gerr("number of shape elements exceeds u64"))
}

impl Graph {
    /// Empty graph.
    pub fn new() -> Self {
        Self::default()
    }

    /// Adds a node.
    pub fn add_node(&mut self, node: Node) -> &mut Self {
        self.nodes.push(node);
        self
    }

    /// Connects `src` → `dst` with a default edge.
    pub fn connect(&mut self, src: &str, dst: &str) -> &mut Self {
        self.edges.push(Edge::new(src, dst));
        self
    }

    /// Adds a complete edge.
    pub fn connect_edge(&mut self, edge: Edge) -> &mut Self {
        self.edges.push(edge);
        self
    }

    /// Connects in a row: `a → b → c …`.
    pub fn chain(&mut self, names: &[&str]) -> &mut Self {
        for w in names.windows(2) {
            self.edges.push(Edge::new(w[0], w[1]));
        }
        self
    }

    /// Node by name.
    pub fn node(&self, name: &str) -> Option<&Node> {
        self.nodes.iter().find(|n| n.name == name)
    }

    /// The names of all tensors referenced by the graph (including subgraphs).
    pub fn tensor_refs(&self) -> Vec<&str> {
        let mut out = Vec::new();
        for n in &self.nodes {
            out.extend(n.params.values().map(|s| s.as_str()));
            if let Some(sg) = &n.subgraph {
                out.extend(sg.tensor_refs());
            }
        }
        out
    }

    /// Checking structure without weights: names, edges, ports, cycles, subgraphs.
    pub fn validate_structure(&self) -> StoreResult<()> {
        self.validate_structure_at("", 0)
    }

    fn validate_structure_at(&self, path: &str, depth: usize) -> StoreResult<()> {
        if depth > MAX_DEPTH {
            return Err(gerr("too deep nesting of subgraphs"));
        }
        let idx = self.check_structure(path)?;
        self.topo(&idx, path)?;
        for n in &self.nodes {
            match (&n.op, &n.subgraph) {
                (NodeOp::Subgraph, Some(sg)) => {
                    sg.validate_structure_at(&format!("{path}{}/", n.name), depth + 1)?
                }
                (NodeOp::Subgraph, None) => {
                    return Err(gerr(format!(
                        "{path}{}: the Subgraph node does not have a subgraph",
                        n.name
                    )));
                }
                (_, Some(_)) => {
                    return Err(gerr(format!(
                        "{path}{}: the subgraph is valid only for the Subgraph node",
                        n.name
                    )));
                }
                _ => {}
            }
        }
        Ok(())
    }

    fn check_structure<'a>(&'a self, path: &str) -> StoreResult<HashMap<&'a str, usize>> {
        let mut idx: HashMap<&str, usize> = HashMap::new();
        for (i, n) in self.nodes.iter().enumerate() {
            if n.name.is_empty() || n.name.len() > 256 || n.name.contains('\0') {
                return Err(StoreError::InvalidName(format!(
                    "{path}'{}': host name is empty, longer than 256 bytes, or contains NUL",
                    n.name
                )));
            }
            if idx.insert(n.name.as_str(), i).is_some() {
                return Err(gerr(format!("{path}{}: node name is repeated", n.name)));
            }
        }
        let mut seen: HashSet<&Edge> = HashSet::new();
        for e in &self.edges {
            for end in [&e.src, &e.dst] {
                if !idx.contains_key(end.as_str()) {
                    return Err(gerr(format!(
                        "{path}edge {} → {}: no node “{end}”",
                        e.src, e.dst
                    )));
                }
            }
            if e.src_port.is_empty() || e.dst_port.is_empty() {
                return Err(gerr(format!(
                    "{path}edge {} → {}: empty port name",
                    e.src, e.dst
                )));
            }
            if !seen.insert(e) {
                return Err(gerr(format!("{path} edge {} → {} repeats", e.src, e.dst)));
            }
        }
        Ok(idx)
    }

    /// Topological order over non-recurrent edges.
    fn topo(&self, idx: &HashMap<&str, usize>, path: &str) -> StoreResult<Vec<usize>> {
        let n = self.nodes.len();
        let mut deg = vec![0usize; n];
        let mut adj: Vec<Vec<usize>> = vec![Vec::new(); n];
        for e in self.edges.iter().filter(|e| !e.recurrent) {
            let (s, d) = (idx[e.src.as_str()], idx[e.dst.as_str()]);
            deg[d] += 1;
            adj[s].push(d);
        }
        let mut queue: VecDeque<usize> = (0..n).filter(|&i| deg[i] == 0).collect();
        let mut order = Vec::with_capacity(n);
        while let Some(u) = queue.pop_front() {
            order.push(u);
            for &v in &adj[u] {
                deg[v] -= 1;
                if deg[v] == 0 {
                    queue.push_back(v);
                }
            }
        }
        if order.len() != n {
            let stuck: Vec<&str> = (0..n)
                .filter(|&i| deg[i] > 0)
                .take(8)
                .map(|i| self.nodes[i].name.as_str())
                .collect();
            return Err(gerr(format!(
                "{path}cycle without recurrent marking through nodes {stuck:?}; \
                 mark feedback `Edge::as_recurrent`"
            )));
        }
        Ok(order)
    }

    /// Infers the shapes of all nodes and checks the graph against the weights (`lookup` is the tensor shape by name).
    pub fn infer_shapes(
        &self,
        lookup: &dyn Fn(&str) -> Option<Vec<u64>>,
    ) -> StoreResult<GraphShapes> {
        self.infer(lookup, "", 0)
    }

    fn infer(&self, lookup: Lookup<'_>, path: &str, depth: usize) -> StoreResult<GraphShapes> {
        if depth > MAX_DEPTH {
            return Err(gerr("too deep nesting of subgraphs"));
        }
        let idx = self.check_structure(path)?;
        let order = self.topo(&idx, path)?;
        let n = self.nodes.len();
        let mut incoming: Vec<Vec<usize>> = vec![Vec::new(); n];
        for (ei, e) in self.edges.iter().enumerate().filter(|(_, e)| !e.recurrent) {
            incoming[idx[e.dst.as_str()]].push(ei);
        }
        let mut ins: Vec<Vec<u64>> = vec![Vec::new(); n];
        let mut outs: Vec<Option<Vec<u64>>> = vec![None; n];
        let (mut has_in, mut has_out) = (false, false);

        for &u in &order {
            let node = &self.nodes[u];
            let ctx = format!("{path}{}", node.name);
            check_params(node, &ctx, lookup)?;
            let in_shape = if node.op == NodeOp::Input {
                has_in = true;
                if !incoming[u].is_empty() {
                    return Err(gerr(format!(
                        "{ctx}: Input node cannot have incoming edges"
                    )));
                }
                declared_shape(node, &ctx)?
            } else {
                let mut shapes = Vec::new();
                for &ei in &incoming[u] {
                    let s = idx[self.edges[ei].src.as_str()];
                    shapes.push(
                        outs[s]
                            .clone()
                            .ok_or_else(|| gerr(format!("{ctx}: source not yet calculated")))?,
                    );
                }
                merge_inputs(node, &shapes, &ctx)?
            };
            let out = match node.op {
                NodeOp::Output => {
                    has_out = true;
                    let d = declared_shape(node, &ctx)?;
                    if numel(&d)? != numel(&in_shape)? {
                        return Err(gerr(format!(
                            "{ctx}: the declared output shape {d:?} does not match the input shape {in_shape:?}"
                        )));
                    }
                    d
                }
                _ => node_output(node, &in_shape, lookup, &ctx, path, depth)?,
            };
            ins[u] = in_shape;
            outs[u] = Some(out);
        }
        if !has_in {
            return Err(gerr(format!("{path}there is no Input node in the graph")));
        }
        if !has_out {
            return Err(gerr(format!("{path}there is no Output node in the graph")));
        }
        for e in self.edges.iter().filter(|e| e.recurrent) {
            let (s, d) = (idx[e.src.as_str()], idx[e.dst.as_str()]);
            let so = outs[s].clone().unwrap_or_default();
            if numel(&so)? != numel(&ins[d])? {
                return Err(gerr(format!(
                    "{path} recurrent edge {} → {}: output {so:?} does not converge with input {:?}",
                    e.src, e.dst, ins[d]
                )));
            }
        }
        let mut res = GraphShapes::default();
        for (i, node) in self.nodes.iter().enumerate() {
            res.nodes.insert(
                node.name.clone(),
                NodeShapes {
                    input: ins[i].clone(),
                    output: outs[i].clone().unwrap_or_default(),
                },
            );
        }
        Ok(res)
    }
}

fn declared_shape(node: &Node, ctx: &str) -> StoreResult<Vec<u64>> {
    let s = node
        .attrs
        .get("shape")
        .and_then(|v| v.as_u64_list())
        .ok_or_else(|| {
            gerr(format!(
                "{ctx}: the shape attribute is needed - a list of positive integers"
            ))
        })?;
    if s.is_empty() || s.contains(&0) {
        return Err(gerr(format!(
            "{ctx}: shape {s:?} must be non-empty, without zeros"
        )));
    }
    Ok(s)
}

fn merge_inputs(node: &Node, shapes: &[Vec<u64>], ctx: &str) -> StoreResult<Vec<u64>> {
    let first = shapes
        .first()
        .ok_or_else(|| gerr(format!("{ctx}: node has no incoming edges")))?;
    if shapes.iter().all(|s| s == first) {
        return Ok(first.clone());
    }
    let flat_ok = matches!(node.op, NodeOp::Linear | NodeOp::Affine | NodeOp::Flatten);
    let n0 = numel(first)?;
    if flat_ok {
        let mut same = true;
        for s in shapes {
            same &= numel(s)? == n0;
        }
        if same {
            return Ok(vec![n0]);
        }
    }
    Err(gerr(format!(
        "{ctx}: the shapes of the incoming edges are different: {shapes:?}"
    )))
}

fn check_params(node: &Node, ctx: &str, lookup: Lookup<'_>) -> StoreResult<()> {
    let spec = node.op.spec();
    for req in spec.required {
        if !node.params.contains_key(*req) {
            return Err(gerr(format!(
                "{ctx}: the required parameter “{req}” is not specified"
            )));
        }
    }
    for (port, tensor) in &node.params {
        let known =
            spec.required.contains(&port.as_str()) || spec.optional.contains(&port.as_str());
        if !known && !spec.open {
            return Err(gerr(format!(
                "{ctx}: Operation {} does not have parameter '{port}'",
                node.op.name()
            )));
        }
        if lookup(tensor).is_none() {
            return Err(gerr(format!(
                "{ctx}: tensor '{tensor}' (parameter '{port}') is missing from storage"
            )));
        }
    }
    Ok(())
}

fn param_shape(
    node: &Node,
    port: &str,
    lookup: Lookup<'_>,
    ctx: &str,
) -> StoreResult<Option<Vec<u64>>> {
    match node.params.get(port) {
        None => Ok(None),
        Some(t) => lookup(t)
            .map(Some)
            .ok_or_else(|| gerr(format!("{ctx}: tensor '{t}' not found"))),
    }
}

fn attr_ints(node: &Node, key: &str, n: usize, default: i64, ctx: &str) -> StoreResult<Vec<i64>> {
    match node.attrs.get(key) {
        None => Ok(vec![default; n]),
        Some(v) => {
            let l = v.as_i64_scalar_or_list().ok_or_else(|| {
                gerr(format!(
                    "{ctx}: attribute '{key}' must be an integer or a list of integers"
                ))
            })?;
            if l.len() == 1 {
                Ok(vec![l[0]; n])
            } else if l.len() == n {
                Ok(l)
            } else {
                Err(gerr(format!(
                    "{ctx}: The attribute '{key}' should have {n} value(s), not {}",
                    l.len()
                )))
            }
        }
    }
}

enum Pad {
    Same,
    Fixed(Vec<i64>),
}

fn attr_pad(node: &Node, n: usize, ctx: &str) -> StoreResult<Pad> {
    match node.attrs.get("padding") {
        Some(Value::Str(s)) => match s.as_str() {
            "same" => Ok(Pad::Same),
            "valid" => Ok(Pad::Fixed(vec![0; n])),
            other => Err(gerr(format!(
                "{ctx}: padding '{other}' is not supported (same/valid/number)"
            ))),
        },
        _ => Ok(Pad::Fixed(attr_ints(node, "padding", n, 0, ctx)?)),
    }
}

fn out_len(
    ctx: &str,
    len: u64,
    k: u64,
    stride: i64,
    dil: i64,
    pad: &Pad,
    dim: usize,
) -> StoreResult<u64> {
    if stride < 1 || dil < 1 {
        return Err(gerr(format!("{ctx}: stride and dilation must be ≥ 1")));
    }
    let (s, d) = (stride as u64, dil as u64);
    let eff = d
        .checked_mul(k.saturating_sub(1))
        .and_then(|x| x.checked_add(1))
        .ok_or_else(|| gerr(format!("{ctx}: Kernel size overflow")))?;
    let padded = match pad {
        Pad::Same => {
            if s != 1 {
                return Err(gerr(format!("{ctx}: padding 'same' requires stride = 1")));
            }
            len + (eff - 1)
        }
        Pad::Fixed(p) => {
            let pv = p[dim];
            if pv < 0 {
                return Err(gerr(format!("{ctx}: padding cannot be negative")));
            }
            len.checked_add(2 * pv as u64)
                .ok_or_else(|| gerr(format!("{ctx}: overflow padding")))?
        }
    };
    if padded < eff {
        return Err(gerr(format!(
            "{ctx}: kernel ({eff}) is larger than input with padding ({padded})"
        )));
    }
    Ok((padded - eff) / s + 1)
}

fn broadcastable(node: &Node, n_in: u64, lookup: Lookup<'_>, ctx: &str) -> StoreResult<()> {
    for (port, t) in &node.params {
        let s = lookup(t).ok_or_else(|| gerr(format!("{ctx}: tensor '{t}' not found")))?;
        let k = numel(&s)?;
        if k != 1 && k != n_in {
            return Err(gerr(format!(
                "{ctx}: parameter '{port}' (tensor '{t}', shape {s:?}) must be a scalar or have {n_in} elements"
            )));
        }
    }
    Ok(())
}

fn node_output(
    node: &Node,
    inp: &[u64],
    lookup: Lookup<'_>,
    ctx: &str,
    path: &str,
    depth: usize,
) -> StoreResult<Vec<u64>> {
    let n_in = numel(inp)?;
    match &node.op {
        NodeOp::Input | NodeOp::Output => Ok(inp.to_vec()),
        NodeOp::Linear | NodeOp::Affine => {
            let w = param_shape(node, "weight", lookup, ctx)?.unwrap_or_default();
            if w.len() != 2 {
                return Err(gerr(format!(
                    "{ctx}: weight should be of the shape [out, in], not {w:?}"
                )));
            }
            if w[1] != n_in {
                return Err(gerr(format!(
                    "{ctx}: weight {w:?} expects {} inputs, and on input {n_in}",
                    w[1]
                )));
            }
            if let Some(b) = param_shape(node, "bias", lookup, ctx)?
                && numel(&b)? != w[0]
            {
                return Err(gerr(format!(
                    "{ctx}: bias {b:?} does not converge with out = {}",
                    w[0]
                )));
            }
            Ok(vec![w[0]])
        }
        NodeOp::Conv1d => conv(node, inp, lookup, ctx, 1),
        NodeOp::Conv2d => conv(node, inp, lookup, ctx, 2),
        NodeOp::Flatten => {
            if inp.is_empty() {
                return Ok(vec![1]);
            }
            let rank = inp.len() as i64;
            let norm = |v: i64| if v < 0 { v + rank } else { v };
            let sd = norm(
                node.attrs
                    .get("start_dim")
                    .and_then(|v| v.as_i64())
                    .unwrap_or(0),
            );
            let ed = norm(
                node.attrs
                    .get("end_dim")
                    .and_then(|v| v.as_i64())
                    .unwrap_or(-1),
            );
            if sd < 0 || ed < sd || ed >= rank {
                return Err(gerr(format!(
                    "{ctx}: start_dim/end_dim out of range for shape {inp:?}"
                )));
            }
            let (sd, ed) = (sd as usize, ed as usize);
            let mut out = inp[..sd].to_vec();
            out.push(numel(&inp[sd..=ed])?);
            out.extend_from_slice(&inp[ed + 1..]);
            Ok(out)
        }
        NodeOp::SumPool2d | NodeOp::AvgPool2d => {
            if inp.len() != 3 {
                return Err(gerr(format!(
                    "{ctx}: pooling is waiting for input [C, H, W], not {inp:?}"
                )));
            }
            if !node.attrs.contains_key("kernel_size") {
                return Err(gerr(format!("{ctx}: need kernel_size attribute")));
            }
            let k = attr_ints(node, "kernel_size", 2, 1, ctx)?;
            let s = if node.attrs.contains_key("stride") {
                attr_ints(node, "stride", 2, 1, ctx)?
            } else {
                k.clone()
            };
            let pad = Pad::Fixed(attr_ints(node, "padding", 2, 0, ctx)?);
            if k.iter().any(|v| *v < 1) {
                return Err(gerr(format!("{ctx}: kernel_size must be ≥ 1")));
            }
            let h = out_len(ctx, inp[1], k[0] as u64, s[0], 1, &pad, 0)?;
            let w = out_len(ctx, inp[2], k[1] as u64, s[1], 1, &pad, 1)?;
            Ok(vec![inp[0], h, w])
        }
        NodeOp::Scale | NodeOp::Threshold | NodeOp::Delay => {
            broadcastable(node, n_in, lookup, ctx)?;
            Ok(inp.to_vec())
        }
        op if op.is_neuron() => {
            broadcastable(node, n_in, lookup, ctx)?;
            Ok(inp.to_vec())
        }
        NodeOp::Subgraph => {
            let sg = node
                .subgraph
                .as_ref()
                .ok_or_else(|| gerr(format!("{ctx}: Subgraph node does not have a subgraph")))?;
            let inner = sg.infer(lookup, &format!("{path}{}/", node.name), depth + 1)?;
            let i = find_unique(sg, &NodeOp::Input, ctx)?;
            let o = find_unique(sg, &NodeOp::Output, ctx)?;
            let i_shape = &inner.nodes[&i.name].output;
            if numel(i_shape)? != n_in {
                return Err(gerr(format!(
                    "{ctx}: the input of the subgraph {i_shape:?} does not converge with the input of {inp:?}"
                )));
            }
            Ok(inner.nodes[&o.name].output.clone())
        }
        NodeOp::Custom(_) => match node.attrs.get("output_shape") {
            Some(v) => v
                .as_u64_list()
                .ok_or_else(|| gerr(format!("{ctx}: output_shape must be a list of integers"))),
            None => Ok(inp.to_vec()),
        },
        _ => Ok(inp.to_vec()),
    }
}

fn find_unique<'a>(sg: &'a Graph, op: &NodeOp, ctx: &str) -> StoreResult<&'a Node> {
    let mut it = sg.nodes.iter().filter(|n| &n.op == op);
    match (it.next(), it.next()) {
        (Some(n), None) => Ok(n),
        _ => Err(gerr(format!(
            "{ctx}: there must be exactly one node {} in the subgraph",
            op.name()
        ))),
    }
}

fn conv(
    node: &Node,
    inp: &[u64],
    lookup: Lookup<'_>,
    ctx: &str,
    dims: usize,
) -> StoreResult<Vec<u64>> {
    if inp.len() != dims + 1 {
        return Err(gerr(format!(
            "{ctx}: convolution {dims}D waits for input of rank {}, received {inp:?}",
            dims + 1
        )));
    }
    let w = param_shape(node, "weight", lookup, ctx)?.unwrap_or_default();
    if w.len() != dims + 2 {
        return Err(gerr(format!(
            "{ctx}: weight should have rank {}, not {w:?}",
            dims + 2
        )));
    }
    let groups = node
        .attrs
        .get("groups")
        .and_then(|v| v.as_i64())
        .unwrap_or(1);
    if groups < 1 {
        return Err(gerr(format!("{ctx}: groups must be ≥ 1")));
    }
    let g = groups as u64;
    let (c, oc) = (inp[0], w[0]);
    if c % g != 0 || oc % g != 0 || w[1] != c / g {
        return Err(gerr(format!(
            "{ctx}: weight {w:?} does not converge with {c} input channels and groups = {groups}"
        )));
    }
    if let Some(b) = param_shape(node, "bias", lookup, ctx)?
        && numel(&b)? != oc
    {
        return Err(gerr(format!(
            "{ctx}: bias {b:?} does not converge with {oc} output channels"
        )));
    }
    let stride = attr_ints(node, "stride", dims, 1, ctx)?;
    let dil = attr_ints(node, "dilation", dims, 1, ctx)?;
    let pad = attr_pad(node, dims, ctx)?;
    let mut out = vec![oc];
    for d in 0..dims {
        out.push(out_len(
            ctx,
            inp[1 + d],
            w[2 + d],
            stride[d],
            dil[d],
            &pad,
            d,
        )?);
    }
    Ok(out)
}

// ---------------- coding ----------------

pub(crate) fn w_graph(w: &mut W, g: &Graph) {
    w.len(g.nodes.len());
    for n in &g.nodes {
        w.str(&n.name);
        w.u16(n.op.code());
        if let NodeOp::Custom(s) = &n.op {
            w.str(s);
        }
        w_meta(w, &n.attrs);
        w.len(n.params.len());
        for (k, v) in &n.params {
            w.str(k);
            w.str(v);
        }
        match &n.subgraph {
            Some(sg) => {
                w.bool(true);
                w_graph(w, sg);
            }
            None => w.bool(false),
        }
    }
    w.len(g.edges.len());
    for e in &g.edges {
        w.str(&e.src);
        w.str(&e.src_port);
        w.str(&e.dst);
        w.str(&e.dst_port);
        w.u32(e.delay);
        w.bool(e.recurrent);
    }
    w_meta(w, &g.meta);
}

pub(crate) fn r_graph(r: &mut R, depth: usize) -> StoreResult<Graph> {
    if depth > MAX_DEPTH {
        return Err(fmt_err("too deep nesting of subgraphs"));
    }
    let nn = r.count(15)?;
    let mut nodes = Vec::with_capacity(nn);
    for _ in 0..nn {
        let name = r.str()?;
        let code = r.u16()?;
        let op = if code == 0xFFFF {
            NodeOp::Custom(r.str()?)
        } else {
            NodeOp::from_code(code).ok_or_else(|| fmt_err(format!("unknown operation {code}")))?
        };
        let attrs = r_meta(r)?;
        let np = r.count(8)?;
        let mut params = BTreeMap::new();
        for _ in 0..np {
            let k = r.str()?;
            let v = r.str()?;
            params.insert(k, v);
        }
        let subgraph = if r.bool()? {
            Some(Box::new(r_graph(r, depth + 1)?))
        } else {
            None
        };
        nodes.push(Node {
            name,
            op,
            attrs,
            params,
            subgraph,
        });
    }
    let ne = r.count(21)?;
    let mut edges = Vec::with_capacity(ne);
    for _ in 0..ne {
        edges.push(Edge {
            src: r.str()?,
            src_port: r.str()?,
            dst: r.str()?,
            dst_port: r.str()?,
            delay: r.u32()?,
            recurrent: r.bool()?,
        });
    }
    let meta = r_meta(r)?;
    Ok(Graph { nodes, edges, meta })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn shapes(map: &'static [(&'static str, &'static [u64])]) -> impl Fn(&str) -> Option<Vec<u64>> {
        move |n: &str| map.iter().find(|(k, _)| *k == n).map(|(_, v)| v.to_vec())
    }

    fn mlp() -> Graph {
        let mut g = Graph::new();
        g.add_node(Node::input("in", &[4]))
            .add_node(Node::affine("fc1", "fc1.weight", "fc1.bias"))
            .add_node(Node::neuron("lif1", NodeOp::Lif, "lif1"))
            .add_node(Node::linear("fc2", "fc2.weight"))
            .add_node(Node::output("out", &[2]));
        g.chain(&["in", "fc1", "lif1", "fc2", "out"]);
        g
    }

    const MLP_SHAPES: &[(&str, &[u64])] = &[
        ("fc1.weight", &[8, 4]),
        ("fc1.bias", &[8]),
        ("lif1.tau", &[8]),
        ("lif1.r", &[1]),
        ("lif1.v_leak", &[8]),
        ("lif1.v_reset", &[1]),
        ("lif1.v_threshold", &[8]),
        ("fc2.weight", &[2, 8]),
    ];

    #[test]
    fn mlp_infers() {
        let s = mlp().infer_shapes(&shapes(MLP_SHAPES)).unwrap();
        assert_eq!(s.nodes["fc1"].output, vec![8]);
        assert_eq!(s.nodes["out"].output, vec![2]);
    }

    #[test]
    fn missing_and_wrong_weights_are_caught() {
        let mut g = mlp();
        g.nodes[1].params.insert("weight".into(), "nope".into());
        assert!(g.infer_shapes(&shapes(MLP_SHAPES)).is_err());
        const BAD: &[(&str, &[u64])] = &[
            ("fc1.weight", &[8, 5]),
            ("fc1.bias", &[8]),
            ("lif1.tau", &[8]),
            ("lif1.r", &[1]),
            ("lif1.v_leak", &[8]),
            ("lif1.v_reset", &[1]),
            ("lif1.v_threshold", &[8]),
            ("fc2.weight", &[2, 8]),
        ];
        assert!(mlp().infer_shapes(&shapes(BAD)).is_err());
    }

    #[test]
    fn cycle_needs_recurrent_mark() {
        let mut g = mlp();
        g.connect("lif1", "fc1");
        assert!(g.validate_structure().is_err());
        let mut g = mlp();
        g.connect_edge(Edge::new("lif1", "fc1").as_recurrent());
        // fc1 accepts 4 inputs, and lif1 outputs 8 → the recurrence shape does not converge
        assert!(g.infer_shapes(&shapes(MLP_SHAPES)).is_err());
    }

    #[test]
    fn conv_and_flatten() {
        let mut g = Graph::new();
        g.add_node(Node::input("in", &[2, 8, 8]))
            .add_node(
                Node::new("c", NodeOp::Conv2d)
                    .param("weight", "c.w")
                    .attr("padding", "same"),
            )
            .add_node(Node::new("p", NodeOp::AvgPool2d).attr("kernel_size", 2i64))
            .add_node(Node::new("f", NodeOp::Flatten))
            .add_node(Node::output("out", &[4 * 4 * 4]));
        g.chain(&["in", "c", "p", "f", "out"]);
        const S: &[(&str, &[u64])] = &[("c.w", &[4, 2, 3, 3])];
        let s = g.infer_shapes(&shapes(S)).unwrap();
        assert_eq!(s.nodes["c"].output, vec![4, 8, 8]);
        assert_eq!(s.nodes["p"].output, vec![4, 4, 4]);
        assert_eq!(s.nodes["f"].output, vec![64]);
    }

    #[test]
    fn graph_codec_roundtrip() {
        let mut g = mlp();
        g.meta.insert("name".into(), Value::Str("mlp".into()));
        g.connect_edge(Edge::new("in", "out").with_delay(3));
        let inner = mlp();
        g.add_node(Node::with_subgraph("sub", inner));
        g.add_node(Node::new("x", NodeOp::Custom("MyOp".into())));
        let mut w = W::new();
        w_graph(&mut w, &g);
        let bytes = w.finish().unwrap();
        let mut r = R::new(&bytes);
        let back = r_graph(&mut r, 0).unwrap();
        assert!(r.is_empty());
        assert_eq!(back, g);
    }
}
