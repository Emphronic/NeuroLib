//! Genome: description of the network in the shape of genes with historical markers (innovation numbers).
//!
//! This is a “blueprint” of the network, not the network itself: unlike [`Topology`], the genome has no state
//! simulations, and gene identifiers are global for all evolution. Thanks to this, two genomes
//! can be aligned to markers (crossover, compatibility distance) even after they
//! grew up differently. The usual [`Topology`] ([`Genome::to_topology`]) is built from the genome, and from
//! trained network - genome ([`Genome::from_topology`]), therefore the evolution of architectures and
//! structural plasticity within the “life” of an individual work on the same graph.

use std::collections::BTreeMap;

use super::super::error::{StructuralError, StructuralResult};
use super::super::model::{LifParams, NeuronModel, NeuronPolarity, NeuronRole};
use super::super::topology::{Birth, NeuronId, Origin, ParamLimits, Topology, TopologyLimits};
use super::codec::{MAX_ITEMS, R, W, err};
use crate::scalar::Scalar;

/// Historical gene marker.
pub type Innov = u64;

/// Neuron gene.
#[derive(Debug, Clone, PartialEq)]
pub struct NodeGene<T = f32> {
    /// Marker.
    pub innov: Innov,
    /// Role. Inputs and outputs exist in any genome and are not removed.
    pub kind: NeuronRole,
    /// Physiological polarity.
    pub polarity: NeuronPolarity,
    /// Dynamics model (not used at inputs).
    pub model: NeuronModel<T>,
}

/// Synapse gene.
#[derive(Debug, Clone, PartialEq)]
pub struct ConnGene<T = f32> {
    /// Marker.
    pub innov: Innov,
    /// Source neuron marker.
    pub src: Innov,
    /// Receiver neuron marker.
    pub dst: Innov,
    /// Weight.
    pub weight: T,
    /// Delay in steps.
    pub delay: u16,
    /// Whether the gene is expressed (a disabled gene remains in the genome for alignment).
    pub enabled: bool,
}

/// Genome. Genes are stored in ascending order of marker, markers are unique.
#[derive(Debug, Clone, PartialEq)]
pub struct Genome<T = f32> {
    /// Neuron genes.
    pub nodes: Vec<NodeGene<T>>,
    /// Synapse genes.
    pub conns: Vec<ConnGene<T>>,
}

impl<T> Default for Genome<T> {
    fn default() -> Self {
        Self {
            nodes: Vec::new(),
            conns: Vec::new(),
        }
    }
}

/// Register of historical markers of all evolution: the same structural novelty
/// (link `a → b`; separation of link with marker `c`), which arose in different individuals, receives one
/// and the same marker - otherwise the genes of different individuals could not be compared.
#[derive(Debug, Clone, PartialEq, Default)]
pub struct InnovationTracker {
    next: Innov,
    conns: BTreeMap<(Innov, Innov), Innov>,
    splits: BTreeMap<Innov, (Innov, Innov, Innov)>,
}

impl InnovationTracker {
    /// A register whose free tokens begin with `first_free`.
    pub fn new(first_free: Innov) -> Self {
        Self {
            next: first_free,
            ..Default::default()
        }
    }

    /// New, not registered anywhere marker.
    pub fn fresh(&mut self) -> Innov {
        let i = self.next;
        self.next += 1;
        i
    }

    /// Communication marker `src → dst` (created upon first request).
    pub fn conn(&mut self, src: Innov, dst: Innov) -> Innov {
        if let Some(i) = self.conns.get(&(src, dst)) {
            return *i;
        }
        let i = self.fresh();
        self.conns.insert((src, dst), i);
        i
    }

    /// Markers `(neuron, input→neuron, neuron→output)` when dividing a connection `conn`.
    pub fn split(&mut self, conn: Innov, src: Innov, dst: Innov) -> (Innov, Innov, Innov) {
        if let Some(t) = self.splits.get(&conn) {
            return *t;
        }
        let node = self.fresh();
        let (a, b) = (self.conn(src, node), self.conn(node, dst));
        self.splits.insert(conn, (node, a, b));
        (node, a, b)
    }

    /// How many tokens are issued?
    pub fn issued(&self) -> Innov {
        self.next
    }

    pub(crate) fn write(&self, w: &mut W) {
        w.u64(self.next);
        w.len(self.conns.len());
        for ((a, b), i) in &self.conns {
            w.u64(*a);
            w.u64(*b);
            w.u64(*i);
        }
        w.len(self.splits.len());
        for (c, (n, a, b)) in &self.splits {
            w.u64(*c);
            w.u64(*n);
            w.u64(*a);
            w.u64(*b);
        }
    }

    pub(crate) fn read(r: &mut R) -> StructuralResult<Self> {
        let mut t = Self::new(r.u64()?);
        for _ in 0..r.len(MAX_ITEMS)? {
            let (a, b, i) = (r.u64()?, r.u64()?, r.u64()?);
            t.conns.insert((a, b), i);
        }
        for _ in 0..r.len(MAX_ITEMS)? {
            let (c, n, a, b) = (r.u64()?, r.u64()?, r.u64()?, r.u64()?);
            t.splits.insert(c, (n, a, b));
        }
        Ok(t)
    }
}

impl<T: Scalar> Genome<T> {
    /// Bring the signs of the connection weights to the polarity of the source neurons
    /// (Dale's principle): excitatory and modulating - `≥ 0`, inhibitory - `≤ 0`.
    /// The magnitudes are preserved; the number of modified genes is returned. Evolution
    /// does this after every mutation and crossover, so no change in
    /// polarity and weight mutation can break the genome invariant.
    pub fn normalize_dale(&mut self) -> usize {
        let pol: BTreeMap<Innov, NeuronPolarity> =
            self.nodes.iter().map(|n| (n.innov, n.polarity)).collect();
        let mut changed = 0;
        for c in &mut self.conns {
            if let Some(p) = pol.get(&c.src) {
                let w = p.resign(c.weight);
                if w != c.weight {
                    c.weight = w;
                    changed += 1;
                }
            }
        }
        changed
    }

    /// Minimal genome: `inputs` inputs, `outputs` outputs, no connections. Inputs are received
    /// markers `0..inputs`, outputs – `inputs..inputs+outputs` (for all genomes of the same evolution
    /// they are the same). `output_model` sets the output dynamics; `LifParams` accepted
    /// thanks to `From<LifParams>`.
    pub fn bare(inputs: usize, outputs: usize, output_model: impl Into<NeuronModel<T>>) -> Self {
        let output_model = output_model.into();
        let mut nodes = Vec::with_capacity(inputs + outputs);
        for i in 0..inputs {
            nodes.push(NodeGene {
                innov: i as Innov,
                kind: NeuronRole::Input,
                polarity: NeuronPolarity::Excitatory,
                model: NeuronModel::Lif(LifParams::<T>::default()),
            });
        }
        for k in 0..outputs {
            nodes.push(NodeGene {
                innov: (inputs + k) as Innov,
                kind: NeuronRole::Output,
                polarity: NeuronPolarity::Excitatory,
                model: output_model,
            });
        }
        Self {
            nodes,
            conns: Vec::new(),
        }
    }

    /// Neuron gene by marker.
    pub fn node(&self, innov: Innov) -> Option<&NodeGene<T>> {
        self.nodes
            .binary_search_by_key(&innov, |n| n.innov)
            .ok()
            .map(|i| &self.nodes[i])
    }

    /// Gene connection by marker.
    pub fn conn(&self, innov: Innov) -> Option<&ConnGene<T>> {
        self.conns
            .binary_search_by_key(&innov, |c| c.innov)
            .ok()
            .map(|i| &self.conns[i])
    }

    /// Is there a `src → dst` connection (including disconnected ones).
    pub fn has_conn(&self, src: Innov, dst: Innov) -> bool {
        self.conns.iter().any(|c| c.src == src && c.dst == dst)
    }

    /// Insert the link gene, maintaining the order. `false`, if the marker already exists.
    pub fn insert_conn(&mut self, c: ConnGene<T>) -> bool {
        match self.conns.binary_search_by_key(&c.innov, |x| x.innov) {
            Ok(_) => false,
            Err(i) => {
                self.conns.insert(i, c);
                true
            }
        }
    }

    /// Insert the neuron gene, maintaining the order. `false`, if the marker already exists.
    pub fn insert_node(&mut self, n: NodeGene<T>) -> bool {
        match self.nodes.binary_search_by_key(&n.innov, |x| x.innov) {
            Ok(_) => false,
            Err(i) => {
                self.nodes.insert(i, n);
                true
            }
        }
    }

    /// Number of hidden neurons.
    pub fn hidden_count(&self) -> usize {
        self.nodes
            .iter()
            .filter(|n| n.kind == NeuronRole::Hidden)
            .count()
    }

    /// Number of expressed connections.
    pub fn enabled_count(&self) -> usize {
        self.conns.iter().filter(|c| c.enabled).count()
    }

    /// Checking internal invariants: order and uniqueness of markers, existence
    /// After all, nothing is included in the input, there are no duplicate connections.
    ///
    /// The signs of the weights are checked according to the Dale principle (excitatory and modulating
    /// source - `w ≥ 0`, inhibitory - `w ≤ 0`): the genome by default lives in
    /// E/I mode. For legacy genomes with free signs there is
    /// [`validate_with`](Self::validate_with).
    pub fn validate(&self) -> StructuralResult<()> {
        self.validate_with(true)
    }

    /// Same as [`validate`](Self::validate), but checking characters according to the principle
    /// Deila includes `enforce_dale` (take `limits.enforce_dale`).
    pub fn validate_with(&self, enforce_dale: bool) -> StructuralResult<()> {
        if !self.nodes.windows(2).all(|w| w[0].innov < w[1].innov) {
            return Err(err("neuronal genes are out of order or repeated"));
        }
        if !self.conns.windows(2).all(|w| w[0].innov < w[1].innov) {
            return Err(err("connection genes are not ordered or repeated"));
        }
        let mut pairs = std::collections::BTreeSet::new();
        for c in &self.conns {
            let dst = self.node(c.dst).ok_or_else(|| {
                err(format!(
                    "connection {} leads to non-existent neuron {}",
                    c.innov, c.dst
                ))
            })?;
            self.node(c.src).ok_or_else(|| {
                err(format!(
                    "connection {} comes from non-existent neuron {}",
                    c.innov, c.src
                ))
            })?;
            if dst.kind == NeuronRole::Input {
                return Err(err(format!(
                    "connection {} enters the input neuron",
                    c.innov
                )));
            }
            if !c.weight.is_finite() {
                return Err(err(format!("link weight {} is not a number", c.innov)));
            }
            if enforce_dale {
                let src = self
                    .node(c.src)
                    .map_or(NeuronPolarity::Excitatory, |n| n.polarity);
                let ok = if src == NeuronPolarity::Inhibitory {
                    c.weight <= T::ZERO
                } else {
                    c.weight >= T::ZERO
                };
                if !ok {
                    return Err(err(format!(
                        "connection {}: Dale principle violated ({} source, weight {})",
                        c.innov,
                        src.as_str(),
                        c.weight
                    )));
                }
            }
            if !pairs.insert((c.src, c.dst)) {
                return Err(err(format!("repeated connection {} → {}", c.src, c.dst)));
            }
        }
        Ok(())
    }

    /// NEAT compatibility distance: `c1·E/N + c2·D/N + c3·W̄`, where `E` are extra genes (marker
    /// greater than the maximum in another genome), `D` - unpaired within the range, `W̄` - average
    /// the difference in the weights of paired genes, `N` is the size of the larger genome (1 if it is small: up to 20 genes
    /// normalization only inflates the differences).
    pub fn distance(&self, other: &Genome<T>, c1: T, c2: T, c3: T) -> T {
        let (a, b) = (&self.conns, &other.conns);
        let (max_a, max_b) = (
            a.last().map_or(0, |c| c.innov),
            b.last().map_or(0, |c| c.innov),
        );
        let (mut i, mut j) = (0, 0);
        let (mut excess, mut disjoint, mut matching, mut wdiff) = (0u32, 0u32, 0u32, T::ZERO);
        while i < a.len() || j < b.len() {
            match (a.get(i), b.get(j)) {
                (Some(x), Some(y)) if x.innov == y.innov => {
                    matching += 1;
                    wdiff += (x.weight - y.weight).abs();
                    (i, j) = (i + 1, j + 1);
                }
                (Some(x), Some(y)) if x.innov < y.innov => {
                    disjoint += 1;
                    i += 1;
                }
                (Some(_), Some(_)) => {
                    disjoint += 1;
                    j += 1;
                }
                (Some(x), None) => {
                    if x.innov > max_b {
                        excess += 1
                    } else {
                        disjoint += 1
                    }
                    i += 1;
                }
                (None, Some(y)) => {
                    if y.innov > max_a {
                        excess += 1
                    } else {
                        disjoint += 1
                    }
                    j += 1;
                }
                (None, None) => break,
            }
        }
        let n = a.len().max(b.len());
        let n = if n < 20 { T::ONE } else { T::from_usize(n) };
        let wbar = if matching > 0 {
            wdiff / T::from_usize(matching as usize)
        } else {
            T::ZERO
        };
        c1 * T::from_usize(excess as usize) / n
            + c2 * T::from_usize(disjoint as usize) / n
            + c3 * wbar
    }

    /// Build a topology: neurons are in marker order (so outputs are in the order of their
    /// markers), expressed connections. Error - if the genome violates `limits`.
    pub fn to_topology(
        &self,
        limits: TopologyLimits,
        param_limits: ParamLimits<T>,
    ) -> StructuralResult<Topology<T>> {
        let mut t = Topology::new(limits, param_limits);
        let birth = Birth::default();
        let mut ids: BTreeMap<Innov, NeuronId> = BTreeMap::new();
        for n in &self.nodes {
            ids.insert(
                n.innov,
                t.add_neuron_with(n.kind, n.polarity, n.model, Origin::Initial, birth)?,
            );
        }
        for c in self.conns.iter().filter(|c| c.enabled) {
            let (s, d) = (ids.get(&c.src), ids.get(&c.dst));
            let (Some(&s), Some(&d)) = (s, d) else {
                return Err(StructuralError::param(
                    "genome",
                    format!("connection {} without ends", c.innov),
                ));
            };
            t.add_synapse(s, d, c.weight, c.delay, Origin::Initial, birth)?;
        }
        Ok(t)
    }

    /// Genome according to a ready-made topology (for example, a trained network): inputs and outputs are received
    /// standard markers, hidden neurons - new ones from `tracker`, connections - markers at a pair of ends.
    /// `tracker` must be created with `first_free ≥ inputs + outputs`.
    pub fn from_topology(t: &Topology<T>, tracker: &mut InnovationTracker) -> Self {
        let inputs = t.inputs();
        let outputs = t.outputs();
        let mut map: BTreeMap<NeuronId, Innov> = BTreeMap::new();
        let mut nodes = Vec::new();
        for (k, id) in inputs.iter().enumerate() {
            map.insert(*id, k as Innov);
        }
        for (k, id) in outputs.iter().enumerate() {
            map.insert(*id, (inputs.len() + k) as Innov);
        }
        for n in t.neurons() {
            let innov = *map.entry(n.id).or_insert_with(|| tracker.fresh());
            nodes.push(NodeGene {
                innov,
                kind: n.role,
                polarity: n.polarity,
                model: n.model,
            });
        }
        nodes.sort_by_key(|n| n.innov);
        let mut g = Genome {
            nodes,
            conns: Vec::new(),
        };
        for s in t.synapses() {
            let (a, b) = (map[&s.src], map[&s.dst]);
            let innov = tracker.conn(a, b);
            g.insert_conn(ConnGene {
                innov,
                src: a,
                dst: b,
                weight: s.weight,
                delay: s.delay,
                enabled: s.is_active(),
            });
        }
        g
    }

    pub(crate) fn write(&self, w: &mut W) {
        w.len(self.nodes.len());
        for n in &self.nodes {
            w.u64(n.innov);
            w.u8(n.kind.code());
            w.u8(n.polarity.code());
            w.model(&n.model);
        }
        w.len(self.conns.len());
        for c in &self.conns {
            w.u64(c.innov);
            w.u64(c.src);
            w.u64(c.dst);
            w.sc(c.weight);
            w.u16(c.delay);
            w.bool(c.enabled);
        }
    }

    pub(crate) fn read(r: &mut R, version: u32) -> StructuralResult<Self> {
        let mut g = Genome::<T>::default();
        for _ in 0..r.len(MAX_ITEMS)? {
            let innov = r.u64()?;
            let kind =
                NeuronRole::from_code(r.u8()?).ok_or_else(|| err("unknown role of the neuron"))?;
            let (polarity, model) = if version >= 2 {
                let polarity = NeuronPolarity::from_code(r.u8()?)
                    .ok_or_else(|| err("unknown neuron polarity"))?;
                (polarity, r.model()?)
            } else {
                (NeuronPolarity::Excitatory, NeuronModel::Lif(r.lif::<T>()?))
            };
            g.nodes.push(NodeGene {
                innov,
                kind,
                polarity,
                model,
            });
        }
        for _ in 0..r.len(MAX_ITEMS)? {
            let (innov, src, dst) = (r.u64()?, r.u64()?, r.u64()?);
            g.conns.push(ConnGene {
                innov,
                src,
                dst,
                weight: r.sc()?,
                delay: r.u16()?,
                enabled: r.bool()?,
            });
        }
        // Signs are not checked: an existing genome could be legacy
        // (free signs); the population normalizes it according to its `limits`.
        g.validate_with(false)?;
        Ok(g)
    }

    /// Genome in bytes (`NGEN` magic). For `f32` - version 2 (values are 4 bytes, as before),
    /// for `f64` - version 3: after the version comes the width of the values (8) and the values without loss
    /// accuracy.
    pub fn to_bytes(&self) -> Vec<u8> {
        let wide = std::mem::size_of::<T>() > 4;
        let mut w = W::with_width(wide);
        w.buf.extend_from_slice(b"NGEN");
        if wide {
            w.u32(3);
            w.u8(8);
        } else {
            w.u32(2);
        }
        self.write(&mut w);
        w.buf
    }

    /// Read the genome; version 1 (LIF only) migrates to the current one. Values are converted to `T`
    /// (`f64` → `f32` loses accuracy, `f32` → `f64` is accurate).
    /// Corrupt or unsupported data is an error.
    pub fn from_bytes(b: &[u8]) -> StructuralResult<Self> {
        if b.len() < 8 || &b[..4] != b"NGEN" {
            return Err(err("this is not a genome (no NGEN magic)"));
        }
        let mut r = R::new(&b[4..]);
        let v = r.u32()?;
        if v == 0 || v > 3 {
            return Err(err(format!("genome version {v} is not supported")));
        }
        if v == 3 {
            r.wide = match r.u8()? {
                4 => false,
                8 => true,
                x => return Err(err(format!("value width {x} is not supported"))),
            };
        }
        let g = Self::read(&mut r, v)?;
        if !r.finished() {
            return Err(err("extra data after the genome"));
        }
        Ok(g)
    }
}
