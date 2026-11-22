//! Structural plasticity controller for fixed-shape layers.
//!
//! Independent of Burn: works with flat `f32` slices. Tensors are read and
//! written by the glue layer (`burn_glue`).

use std::collections::{BTreeMap, HashSet, VecDeque};

use super::config::FixedShapeConfig;
use super::layer::*;
use crate::structural::error::{StructuralError, StructuralResult};
use crate::structural::mutation::{ActionKind, EventLog, Outcome, StructuralEvent};
use crate::structural::rng::Rng64;
use crate::structural::stats::StructuralStats;

/// Input of a single [`FixedShapeController::adapt`] call.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct AdaptInput {
    /// Global training step number.
    pub step: u64,
    /// Sample/batch number (for the log).
    pub sample: u64,
    /// Current loss (smoothed if possible), if measured.
    pub loss: Option<f32>,
}

/// Why no changes were made.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FixedSkip {
    /// The next analysis iteration has not arrived.
    NotScheduled,
    /// Too early after the previous change.
    TooSoon,
    /// The previous change is being verified.
    Verifying,
    /// Nothing to change.
    NothingToDo,
}

/// Outcome of a single call.
#[derive(Debug, Clone, PartialEq)]
pub struct FixedAdaptReport {
    /// Metrics after the call.
    pub stats: StructuralStats,
    /// New log events.
    pub events: Vec<StructuralEvent>,
    /// What to write into the tensors (weights) — apply **before** the next forward pass.
    /// After writing the weights, mask projection is mandatory.
    pub updates: Vec<LayerUpdate>,
    /// Loss before the last change.
    pub loss_before: Option<f32>,
    /// Loss during verification.
    pub loss_after: Option<f32>,
    /// This is a rollback of the previous change.
    pub rolled_back: bool,
    /// Reason why no changes were made.
    pub skipped: Option<FixedSkip>,
}

#[derive(Debug, Clone)]
struct Pending {
    inverse: Vec<LayerUpdate>,
    loss_before: Option<f32>,
    calls_left: u64,
    events: Vec<(u64, ActionKind)>,
}

#[derive(Default)]
struct Builder {
    fwd: BTreeMap<usize, LayerUpdate>,
    inv: BTreeMap<usize, LayerUpdate>,
    seen_syn: HashSet<(usize, usize)>,
    seen_neu: HashSet<(usize, usize)>,
}

impl Builder {
    fn finish(self) -> (Vec<LayerUpdate>, Vec<LayerUpdate>) {
        (
            self.fwd.into_values().collect(),
            self.inv.into_values().collect(),
        )
    }
}

struct Op {
    kind: ActionKind,
    action: String,
    reason: String,
    before: (usize, usize),
    after: (usize, usize),
}

/// Controller.
#[derive(Debug)]
pub struct FixedShapeController {
    cfg: FixedShapeConfig,
    layers: Vec<LayerState>,
    rng: Rng64,
    log: EventLog,
    calls: u64,
    last_change_step: Option<u64>,
    loss_hist: VecDeque<f32>,
    pending: Option<Pending>,
    cooldown: BTreeMap<ActionKind, (u64, u64)>,
}

impl FixedShapeController {
    /// Create a controller. Validates the configuration and layer consistency.
    pub fn new(cfg: FixedShapeConfig, specs: Vec<LayerSpec>) -> StructuralResult<Self> {
        cfg.validate()?;
        if specs.is_empty() {
            return Err(StructuralError::InvalidConfig(
                "at least one layer is required".into(),
            ));
        }
        for (k, s) in specs.iter().enumerate() {
            s.validate(k, specs.len())?;
            if let Some(src) = s.input_from
                && specs[src].out_features != s.in_features
            {
                return Err(StructuralError::InvalidConfig(format!(
                    "layer {k} \"{}\": in_features {} ≠ out_features {} of source layer {src}",
                    s.name, s.in_features, specs[src].out_features
                )));
            }
        }
        let mut rng = Rng64::new(cfg.seed);
        let layers = specs
            .into_iter()
            .map(|s| LayerState::new(s, &mut rng))
            .collect();
        let log = EventLog::new(cfg.log_capacity);
        Ok(Self {
            cfg,
            layers,
            rng,
            log,
            calls: 0,
            last_change_step: None,
            loss_hist: VecDeque::new(),
            pending: None,
            cooldown: BTreeMap::new(),
        })
    }

    /// Layers.
    pub fn layers(&self) -> &[LayerState] {
        &self.layers
    }

    /// Configuration.
    pub fn config(&self) -> &FixedShapeConfig {
        &self.cfg
    }

    /// Event log.
    pub fn log(&self) -> &EventLog {
        &self.log
    }

    /// Whether the previous change is being verified.
    pub fn is_verifying(&self) -> bool {
        self.pending.is_some()
    }

    fn alive_inputs(&self, l: usize) -> Vec<bool> {
        match self.layers[l].spec.input_from {
            None => vec![true; self.layers[l].spec.in_features],
            Some(k) => self.layers[k].neuron_active.clone(),
        }
    }

    /// Effective mask of layer `[in, out]`: the connection exists, the target neuron
    /// is active and the source (input) is alive.
    pub fn effective_mask(&self, l: usize) -> Vec<bool> {
        let s = &self.layers[l];
        let alive = self.alive_inputs(l);
        let out = s.spec.out_features;
        let mut m = vec![false; s.spec.weight_len()];
        for (i, &a) in alive.iter().enumerate() {
            if !a {
                continue;
            }
            for o in 0..out {
                m[i * out + o] = s.syn_active[i * out + o] && s.neuron_active[o];
            }
        }
        m
    }

    /// Layer mask as `f32` (1.0/0.0) — for multiplying with the weight tensor.
    pub fn effective_mask_f32(&self, l: usize) -> Vec<f32> {
        self.effective_mask(l)
            .into_iter()
            .map(|b| if b { 1.0 } else { 0.0 })
            .collect()
    }

    /// Neuron mask of a layer as `f32` — for multiplying bias and spikes.
    pub fn neuron_mask_f32(&self, l: usize) -> Vec<f32> {
        self.layers[l]
            .neuron_active
            .iter()
            .map(|&b| if b { 1.0 } else { 0.0 })
            .collect()
    }

    /// Zero out forbidden elements of the layer weights (projection after an optimizer step).
    pub fn project_weights(&self, l: usize, w: &mut [f32]) -> StructuralResult<()> {
        let n = self.layers[l].spec.weight_len();
        if w.len() != n {
            return Err(StructuralError::InputSize {
                op: "project_weights",
                expected: n,
                actual: w.len(),
            });
        }
        for (x, m) in w.iter_mut().zip(self.effective_mask(l)) {
            if !m {
                *x = 0.0;
            }
        }
        Ok(())
    }

    /// Zero out the bias of inactive neurons.
    pub fn project_bias(&self, l: usize, b: &mut [f32]) -> StructuralResult<()> {
        let n = self.layers[l].spec.out_features;
        if b.len() != n {
            return Err(StructuralError::InputSize {
                op: "project_bias",
                expected: n,
                actual: b.len(),
            });
        }
        for (x, &a) in b.iter_mut().zip(&self.layers[l].neuron_active) {
            if !a {
                *x = 0.0;
            }
        }
        Ok(())
    }

    /// Initial weight initialization: write `init(i, o)` into the allowed positions
    /// and zero out the rest.
    pub fn init_weights(
        &self,
        l: usize,
        w: &mut [f32],
        mut init: impl FnMut(usize, usize) -> f32,
    ) -> StructuralResult<()> {
        let out = self.layers[l].spec.out_features;
        let n = self.layers[l].spec.weight_len();
        if w.len() != n {
            return Err(StructuralError::InputSize {
                op: "init_weights",
                expected: n,
                actual: w.len(),
            });
        }
        for (idx, m) in self.effective_mask(l).into_iter().enumerate() {
            w[idx] = if m { init(idx / out, idx % out) } else { 0.0 };
        }
        Ok(())
    }

    /// Account for activity: `pre` is `[batch, in]`, `post` is `[batch, out]`
    /// (spike rates per step over the window, values 0..1).
    pub fn observe_activity(
        &mut self,
        l: usize,
        pre: &[f32],
        post: &[f32],
        batch: usize,
    ) -> StructuralResult<()> {
        let (inp, out) = (
            self.layers[l].spec.in_features,
            self.layers[l].spec.out_features,
        );
        if batch == 0 || pre.len() != batch * inp {
            return Err(StructuralError::InputSize {
                op: "observe_activity(pre)",
                expected: batch * inp,
                actual: pre.len(),
            });
        }
        if post.len() != batch * out {
            return Err(StructuralError::InputSize {
                op: "observe_activity(post)",
                expected: batch * out,
                actual: post.len(),
            });
        }
        let a = self.cfg.ema;
        let st = &mut self.layers[l].stats;
        let inv = 1.0 / batch as f32;
        let mut pre_m = vec![0.0f32; inp];
        let mut post_m = vec![0.0f32; out];
        let mut co = vec![0.0f32; inp * out];
        for b in 0..batch {
            let p = &pre[b * inp..(b + 1) * inp];
            let q = &post[b * out..(b + 1) * out];
            for (i, &pi) in p.iter().enumerate() {
                pre_m[i] += pi * inv;
                if pi != 0.0 {
                    let row = &mut co[i * out..(i + 1) * out];
                    for (c, &qo) in row.iter_mut().zip(q) {
                        *c += pi * qo * inv;
                    }
                }
            }
            for (o, &qo) in q.iter().enumerate() {
                post_m[o] += qo * inv;
            }
        }
        let first = st.activity_samples == 0;
        let ema =
            |old: &mut f32, new: f32| *old = if first { new } else { *old + a * (new - *old) };
        pre_m
            .iter()
            .zip(st.pre_rate.iter_mut())
            .for_each(|(n, o)| ema(o, *n));
        post_m
            .iter()
            .zip(st.post_rate.iter_mut())
            .for_each(|(n, o)| ema(o, *n));
        co.iter()
            .zip(st.coactivity.iter_mut())
            .for_each(|(n, o)| ema(o, *n));
        st.activity_samples += 1;
        Ok(())
    }

    /// Account for layer gradients: connection importance `|w·g|`.
    pub fn observe_grads(&mut self, l: usize, w: &[f32], g: &[f32]) -> StructuralResult<()> {
        let n = self.layers[l].spec.weight_len();
        if w.len() != n || g.len() != n {
            return Err(StructuralError::InputSize {
                op: "observe_grads",
                expected: n,
                actual: w.len().min(g.len()),
            });
        }
        let a = self.cfg.ema;
        let st = &mut self.layers[l].stats;
        let first = st.grad_samples == 0;
        for ((s, &wi), &gi) in st.saliency.iter_mut().zip(w).zip(g) {
            let v = (wi * gi).abs();
            *s = if first { v } else { *s + a * (v - *s) };
        }
        st.grad_samples += 1;
        Ok(())
    }

    // ---- mutations with inverse change accounting ----------------------------------------

    #[allow(clippy::too_many_arguments)]
    fn set_syn(
        &mut self,
        b: &mut Builder,
        w: &[&[f32]],
        l: usize,
        idx: usize,
        active: bool,
        weight: f32,
        step: u64,
    ) {
        if b.seen_syn.insert((l, idx)) {
            let inv = b.inv.entry(l).or_insert_with(|| LayerUpdate {
                layer: l,
                ..Default::default()
            });
            inv.syn.push(SynChange {
                idx,
                active: self.layers[l].syn_active[idx],
                weight: w[l][idx],
            });
        }
        b.fwd
            .entry(l)
            .or_insert_with(|| LayerUpdate {
                layer: l,
                ..Default::default()
            })
            .syn
            .push(SynChange {
                idx,
                active,
                weight,
            });
        let s = &mut self.layers[l];
        if active && !s.syn_active[idx] {
            s.syn_born[idx] = step;
            s.stats.reset_pair(idx);
        }
        s.syn_active[idx] = active;
    }

    fn set_neuron(&mut self, b: &mut Builder, l: usize, o: usize, active: bool, step: u64) {
        if b.seen_neu.insert((l, o)) {
            let inv = b.inv.entry(l).or_insert_with(|| LayerUpdate {
                layer: l,
                ..Default::default()
            });
            inv.neurons.push(NeuronChange {
                o,
                active: self.layers[l].neuron_active[o],
            });
        }
        b.fwd
            .entry(l)
            .or_insert_with(|| LayerUpdate {
                layer: l,
                ..Default::default()
            })
            .neurons
            .push(NeuronChange { o, active });
        let spec = self.layers[l].spec.clone();
        let s = &mut self.layers[l];
        if active && !s.neuron_active[o] {
            s.neuron_born[o] = step;
            s.stats.reset_neuron(o, &spec);
        }
        s.neuron_active[o] = active;
    }

    /// Apply `LayerUpdate` to the masks (for rollback). Age is reset.
    fn apply_to_masks(&mut self, ups: &[LayerUpdate], step: u64) {
        for u in ups {
            let spec = self.layers[u.layer].spec.clone();
            for c in &u.syn {
                let s = &mut self.layers[u.layer];
                if c.active && !s.syn_active[c.idx] {
                    s.syn_born[c.idx] = step;
                }
                s.syn_active[c.idx] = c.active;
            }
            for c in &u.neurons {
                let s = &mut self.layers[u.layer];
                if c.active && !s.neuron_active[c.o] {
                    s.neuron_born[c.o] = step;
                }
                s.neuron_active[c.o] = c.active;
            }
            let _ = spec;
        }
    }

    fn counts(&self) -> (usize, usize) {
        let first_in = self
            .layers
            .iter()
            .find(|l| l.spec.input_from.is_none())
            .map_or(0, |l| l.spec.in_features);
        let neurons = first_in
            + self
                .layers
                .iter()
                .map(|l| l.active_neurons())
                .sum::<usize>();
        let syn = (0..self.layers.len())
            .map(|l| self.effective_mask(l).iter().filter(|&&m| m).count())
            .sum();
        (neurons, syn)
    }

    fn cooling(&self, kind: ActionKind, step: u64) -> bool {
        self.cooldown
            .get(&kind)
            .is_some_and(|&(until, _)| step < until)
    }

    fn plateau(&self) -> bool {
        let p = self.cfg.neurons.plateau_patience;
        if self.loss_hist.len() < p {
            return false;
        }
        let v: Vec<f32> = self.loss_hist.iter().rev().take(p).copied().collect(); // new → old
        let half = p / 2;
        let new: f32 = v[..half].iter().sum::<f32>() / half as f32;
        let old: f32 = v[p - half..].iter().sum::<f32>() / half as f32;
        let last = v[0];
        last >= self.cfg.neurons.min_loss && old - new < self.cfg.neurons.plateau_min_delta
    }

    // ---- main call ----------------------------------------------------------------------

    /// One call between batches. `weights[l]` is the current weight of layer `l` as a flat `[in, out]`
    /// array (after projection). Returns the changes that **must be written into the tensors**.
    ///
    /// Atomicity: if writing to the tensors fails, call [`Self::revert_report`].
    pub fn adapt(
        &mut self,
        input: AdaptInput,
        weights: &[&[f32]],
    ) -> StructuralResult<FixedAdaptReport> {
        if weights.len() != self.layers.len() {
            return Err(StructuralError::InputSize {
                op: "adapt(weights)",
                expected: self.layers.len(),
                actual: weights.len(),
            });
        }
        for (l, w) in weights.iter().enumerate() {
            let n = self.layers[l].spec.weight_len();
            if w.len() != n {
                return Err(StructuralError::InputSize {
                    op: "adapt(weights[l])",
                    expected: n,
                    actual: w.len(),
                });
            }
        }
        self.calls += 1;
        if let Some(l) = input.loss.filter(|v| v.is_finite()) {
            if self.loss_hist.len() == self.cfg.loss_window {
                self.loss_hist.pop_front();
            }
            self.loss_hist.push_back(l);
        }
        let mut rep = FixedAdaptReport {
            stats: self.stats(input, Some(weights)),
            events: vec![],
            updates: vec![],
            loss_before: None,
            loss_after: None,
            rolled_back: false,
            skipped: None,
        };

        // 1. Verification of the previous change.
        if let Some(p) = self.pending.as_mut() {
            p.calls_left = p.calls_left.saturating_sub(1);
            if p.calls_left > 0 {
                rep.skipped = Some(FixedSkip::Verifying);
                return Ok(rep);
            }
            let p = self.pending.take().expect("pending checked above");
            rep.loss_before = p.loss_before;
            rep.loss_after = input.loss;
            let worse = match (p.loss_before, input.loss) {
                (Some(b), Some(a))
                    if self.cfg.rollback.enabled && b.is_finite() && a.is_finite() =>
                {
                    a > b * (1.0 + self.cfg.rollback.rel_tolerance)
                        + self.cfg.rollback.abs_tolerance
                }
                (Some(_), Some(a)) if self.cfg.rollback.enabled && !a.is_finite() => true,
                _ => false,
            };
            if worse {
                let la = input.loss.unwrap_or(f32::NAN);
                self.apply_to_masks(&p.inverse, input.step);
                for (seq, kind) in &p.events {
                    self.log.reclassify_rolled_back(*seq, *kind, la);
                    let (until, len) = self.cooldown.get(kind).copied().unwrap_or((0, 0));
                    let _ = until;
                    let len = if len == 0 {
                        self.cfg.rollback.cooldown_steps
                    } else {
                        (len * 2).min(self.cfg.rollback.max_cooldown_steps)
                    };
                    self.cooldown.insert(*kind, (input.step + len, len));
                }
                rep.updates = p.inverse;
                rep.rolled_back = true;
                rep.events = self.drain_events_since(rep.stats.step, &p.events);
                rep.stats = self.stats(input, Some(weights));
                self.last_change_step = Some(input.step);
                return Ok(rep);
            }
            // Accepted: the per-type pause is reset.
            for (_, kind) in &p.events {
                self.cooldown.remove(kind);
            }
        }

        // 2. Schedule.
        if self
            .calls
            .checked_rem(self.cfg.schedule.analyze_every)
            .unwrap_or(0)
            != 0
        {
            rep.skipped = Some(FixedSkip::NotScheduled);
            return Ok(rep);
        }
        if self
            .last_change_step
            .is_some_and(|s| input.step < s + self.cfg.schedule.min_interval_steps)
        {
            rep.skipped = Some(FixedSkip::TooSoon);
            return Ok(rep);
        }

        // 3. Plan and apply to the masks.
        let mut b = Builder::default();
        let mut ops: Vec<Op> = vec![];
        let max_ops = self.cfg.schedule.max_ops_per_interval;
        self.plan_neurons(&mut b, &mut ops, weights, input, max_ops);
        let used = ops.len();
        if self.cfg.pruning.enabled && used < max_ops {
            let budget = (max_ops / 2).max(1).min(max_ops - used);
            self.plan_prune(&mut b, &mut ops, weights, input, budget);
        }
        if self.cfg.growth.enabled && ops.len() < max_ops {
            let budget = (max_ops - ops.len()).min(self.cfg.growth.max_new_per_interval);
            self.plan_grow(&mut b, &mut ops, weights, input, budget);
        }
        if ops.is_empty() {
            rep.skipped = Some(FixedSkip::NothingToDo);
            return Ok(rep);
        }
        let (fwd, inv) = b.finish();
        let mut seqs = vec![];
        for op in ops {
            let seq = self.log.push(StructuralEvent {
                seq: 0,
                step: input.step,
                sample: input.sample,
                kind: op.kind,
                action: op.action,
                reason: op.reason,
                outcome: Outcome::Applied,
                loss_before: input.loss,
                loss_after: None,
                neurons_before: op.before.0,
                neurons_after: op.after.0,
                synapses_before: op.before.1,
                synapses_after: op.after.1,
            });
            seqs.push((seq, op.kind));
        }
        rep.events = self
            .log
            .iter()
            .filter(|e| seqs.iter().any(|(s, _)| *s == e.seq))
            .cloned()
            .collect();
        rep.updates = fwd;
        rep.loss_before = input.loss;
        self.last_change_step = Some(input.step);
        self.pending = Some(Pending {
            inverse: inv,
            loss_before: input.loss,
            calls_left: self.cfg.rollback.verify_after_calls.max(1),
            events: seqs,
        });
        rep.stats = self.stats(input, None);
        Ok(rep)
    }

    fn drain_events_since(&self, _step: u64, rolled: &[(u64, ActionKind)]) -> Vec<StructuralEvent> {
        self.log
            .iter()
            .filter(|e| rolled.iter().any(|(s, _)| *s == e.seq))
            .cloned()
            .collect()
    }

    /// Cancel the changes from the report if writing to the tensors failed.
    ///
    /// Restores the masks and clears the pending verification. The weights are **not** touched:
    /// the caller has not written the new values.
    pub fn revert_report(&mut self, step: u64) {
        if let Some(p) = self.pending.take() {
            self.apply_to_masks(&p.inverse, step);
            for (seq, kind) in p.events {
                self.log.reclassify_rolled_back(seq, kind, f32::NAN);
            }
        }
    }

    // ---- planners -----------------------------------------------------------------------

    fn plan_prune(
        &mut self,
        b: &mut Builder,
        ops: &mut Vec<Op>,
        w: &[&[f32]],
        inp: AdaptInput,
        budget: usize,
    ) {
        let pc = self.cfg.pruning.clone();
        if budget == 0 || self.cooling(ActionKind::DisableSynapse, inp.step) {
            return;
        }
        let mut left = budget;
        for l in 0..self.layers.len() {
            if left == 0 {
                break;
            }
            let st = &self.layers[l].stats;
            let enough = if pc.use_saliency {
                st.grad_samples
            } else {
                st.activity_samples.max(st.grad_samples)
            };
            if enough < pc.min_observed_samples {
                continue;
            }
            let out = self.layers[l].spec.out_features;
            let mask = self.effective_mask(l);
            let imp = |idx: usize| {
                if pc.use_saliency {
                    st.saliency[idx]
                } else {
                    w[l][idx].abs()
                }
            };
            let active: Vec<usize> = (0..mask.len()).filter(|&i| mask[i]).collect();
            if active.len() <= pc.min_synapses_per_layer {
                continue;
            }
            let mean = active.iter().map(|&i| imp(i)).sum::<f32>() / active.len() as f32;
            if mean <= 0.0 || !mean.is_finite() {
                continue;
            }
            let thr = pc.rel_threshold * mean;
            let mut cand: Vec<(f32, usize)> = active
                .iter()
                .filter(|&&i| {
                    inp.step.saturating_sub(self.layers[l].syn_born[i]) >= pc.min_age_steps
                })
                .map(|&i| (imp(i), i))
                .filter(|&(v, _)| v <= thr)
                .collect();
            cand.sort_by(|a, c| a.0.total_cmp(&c.0).then(a.1.cmp(&c.1)));
            let frac_cap = ((active.len() as f32) * pc.max_fraction_per_interval).ceil() as usize;
            let mut indeg: Vec<usize> = (0..out)
                .map(|o| {
                    (0..self.layers[l].spec.in_features)
                        .filter(|&i| mask[i * out + o])
                        .count()
                })
                .collect();
            let mut remaining = active.len();
            let mut taken = 0usize;
            for (v, idx) in cand {
                if taken >= frac_cap || left == 0 || remaining <= pc.min_synapses_per_layer {
                    break;
                }
                let o = idx % out;
                if indeg[o] <= pc.min_in_degree {
                    continue;
                }
                let before = self.counts();
                self.set_syn(b, w, l, idx, false, 0.0, inp.step);
                indeg[o] -= 1;
                remaining -= 1;
                taken += 1;
                left -= 1;
                ops.push(Op {
                    kind: ActionKind::DisableSynapse,
                    action: format!(
                        "disable connection {}[{}→{}] (w={:.3}, importance {:.2e})",
                        self.layers[l].spec.name,
                        idx / out,
                        o,
                        w[l][idx],
                        v
                    ),
                    reason: format!(
                        "importance ≤ {:.0}% of the layer mean ({:.2e})",
                        pc.rel_threshold * 100.0,
                        mean
                    ),
                    before,
                    after: self.counts(),
                });
            }
        }
    }

    fn plan_grow(
        &mut self,
        b: &mut Builder,
        ops: &mut Vec<Op>,
        w: &[&[f32]],
        inp: AdaptInput,
        budget: usize,
    ) {
        let gc = self.cfg.growth.clone();
        if budget == 0 || self.cooling(ActionKind::AddSynapse, inp.step) {
            return;
        }
        let mut cand: Vec<(f32, usize, usize)> = vec![]; // (score, layer, idx)
        for l in 0..self.layers.len() {
            let s = &self.layers[l];
            if s.stats.activity_samples == 0 {
                continue;
            }
            let out = s.spec.out_features;
            let alive = self.alive_inputs(l);
            let eff = self.effective_mask(l);
            let possible = alive.iter().filter(|&&a| a).count() * s.active_neurons();
            let mut have = eff.iter().filter(|&&m| m).count();
            let cap = (gc.max_density * possible as f32).floor() as usize;
            let mut per_layer: Vec<(f32, usize)> = vec![];
            for (i, &a) in alive.iter().enumerate() {
                if !a {
                    continue;
                }
                for o in 0..out {
                    let idx = i * out + o;
                    if s.neuron_active[o]
                        && !s.syn_active[idx]
                        && s.stats.coactivity[idx] >= gc.min_score
                        && !b.seen_syn.contains(&(l, idx))
                    {
                        per_layer.push((s.stats.coactivity[idx], idx));
                    }
                }
            }
            per_layer.sort_by(|a, c| c.0.total_cmp(&a.0).then(a.1.cmp(&c.1)));
            for (sc, idx) in per_layer {
                if have >= cap {
                    break;
                }
                cand.push((sc, l, idx));
                have += 1;
            }
        }
        cand.sort_by(|a, c| c.0.total_cmp(&a.0).then(a.1.cmp(&c.1)).then(a.2.cmp(&c.2)));
        for (sc, l, idx) in cand.into_iter().take(budget) {
            let out = self.layers[l].spec.out_features;
            let wt = gc.init_weight + gc.weight_jitter * self.rng.symmetric();
            let before = self.counts();
            self.set_syn(b, w, l, idx, true, wt, inp.step);
            ops.push(Op {
                kind: ActionKind::AddSynapse,
                action: format!(
                    "enable connection {}[{}→{}] (w₀={:.3})",
                    self.layers[l].spec.name,
                    idx / out,
                    idx % out,
                    wt
                ),
                reason: format!("co-activity {:.2e} ≥ {:.2e}", sc, gc.min_score),
                before,
                after: self.counts(),
            });
        }
    }

    fn deps_of(&self, l: usize) -> Vec<usize> {
        (0..self.layers.len())
            .filter(|&d| self.layers[d].spec.input_from == Some(l))
            .collect()
    }

    fn clear_neuron_links(&mut self, b: &mut Builder, w: &[&[f32]], l: usize, o: usize, step: u64) {
        let (inp, out) = (
            self.layers[l].spec.in_features,
            self.layers[l].spec.out_features,
        );
        for i in 0..inp {
            let idx = i * out + o;
            if self.layers[l].syn_active[idx] {
                self.set_syn(b, w, l, idx, false, 0.0, step);
            }
        }
        for d in self.deps_of(l) {
            let od = self.layers[d].spec.out_features;
            for o2 in 0..od {
                let idx = o * od + o2;
                if self.layers[d].syn_active[idx] {
                    self.set_syn(b, w, d, idx, false, 0.0, step);
                }
            }
        }
    }

    fn plan_neurons(
        &mut self,
        b: &mut Builder,
        ops: &mut Vec<Op>,
        w: &[&[f32]],
        inp: AdaptInput,
        max_ops: usize,
    ) {
        let nc = self.cfg.neurons.clone();
        if !nc.enabled {
            return;
        }
        // Deactivation of silent neurons.
        if nc.deactivate_silent && !self.cooling(ActionKind::RemoveNeuron, inp.step) {
            let mut left = nc.max_deactivate;
            for l in 0..self.layers.len() {
                if self.layers[l].spec.role != LayerRole::Hidden {
                    continue;
                }
                let mut cand: Vec<(f32, usize)> = (0..self.layers[l].spec.out_features)
                    .filter(|&o| {
                        let s = &self.layers[l];
                        s.neuron_active[o]
                            && s.stats.activity_samples >= nc.min_observed_samples
                            && s.stats.post_rate[o] < nc.neuron_min_rate
                            && inp.step.saturating_sub(s.neuron_born[o]) >= nc.min_age_steps
                    })
                    .map(|o| (self.layers[l].stats.post_rate[o], o))
                    .collect();
                cand.sort_by(|a, c| a.0.total_cmp(&c.0).then(a.1.cmp(&c.1)));
                for (rate, o) in cand {
                    if left == 0
                        || ops.len() >= max_ops
                        || self.layers[l].active_neurons() <= nc.min_active_per_layer
                    {
                        break;
                    }
                    let before = self.counts();
                    self.clear_neuron_links(b, w, l, o, inp.step);
                    self.set_neuron(b, l, o, false, inp.step);
                    left -= 1;
                    ops.push(Op {
                        kind: ActionKind::RemoveNeuron,
                        action: format!(
                            "return neuron {}[{}] to the reserve",
                            self.layers[l].spec.name, o
                        ),
                        reason: format!("rate {:.2e} < {:.2e}", rate, nc.neuron_min_rate),
                        before,
                        after: self.counts(),
                    });
                }
            }
        }
        // Activation of a reserve neuron on a plateau.
        if !self.cooling(ActionKind::AddNeuron, inp.step) && self.plateau() {
            let mut left = nc.max_activate;
            while left > 0 && ops.len() < max_ops {
                let pick = (0..self.layers.len())
                    .filter(|&l| self.layers[l].spec.role == LayerRole::Hidden)
                    .map(|l| {
                        (
                            self.layers[l].spec.out_features - self.layers[l].active_neurons(),
                            l,
                        )
                    })
                    .filter(|&(r, _)| r > 0)
                    .max_by_key(|&(r, l)| (r, std::cmp::Reverse(l)));
                let Some((_, l)) = pick else { break };
                let o = (0..self.layers[l].spec.out_features)
                    .find(|&o| !self.layers[l].neuron_active[o])
                    .expect("a reserve exists");
                let alive = self.alive_inputs(l);
                let out = self.layers[l].spec.out_features;
                let mut ins: Vec<(f32, usize)> = (0..self.layers[l].spec.in_features)
                    .filter(|&i| alive[i])
                    .map(|i| {
                        (
                            self.layers[l].stats.pre_rate[i] + 1e-6 * self.rng.next_f32(),
                            i,
                        )
                    })
                    .collect();
                ins.sort_by(|a, c| c.0.total_cmp(&a.0));
                ins.truncate(nc.new_fan_in.max(1));
                if ins.is_empty() {
                    break;
                }
                let before = self.counts();
                self.clear_neuron_links(b, w, l, o, inp.step);
                self.set_neuron(b, l, o, true, inp.step);
                for &(_, i) in &ins {
                    let wt = nc.init_weight * (0.5 + self.rng.next_f32());
                    self.set_syn(b, w, l, i * out + o, true, wt, inp.step);
                }
                let mut fan_out = 0usize;
                for d in self.deps_of(l) {
                    let od = self.layers[d].spec.out_features;
                    let mut targets: Vec<usize> = (0..od)
                        .filter(|&o2| self.layers[d].neuron_active[o2])
                        .collect();
                    for k in (1..targets.len()).rev() {
                        targets.swap(k, self.rng.below(k + 1));
                    }
                    for &o2 in targets.iter().take(nc.new_fan_out) {
                        let wt = nc.init_weight * self.rng.symmetric();
                        self.set_syn(b, w, d, o * od + o2, true, wt, inp.step);
                        fan_out += 1;
                    }
                }
                left -= 1;
                // The new capacity changes the loss dynamics: the plateau must be detected anew.
                self.loss_hist.clear();
                ops.push(Op {
                    kind: ActionKind::AddNeuron,
                    action: format!(
                        "activate reserve neuron {}[{}] (inputs {}, outputs {})",
                        self.layers[l].spec.name,
                        o,
                        ins.len(),
                        fan_out
                    ),
                    reason: "loss plateau: capacity is insufficient".to_string(),
                    before,
                    after: self.counts(),
                });
            }
        }
    }

    // ---- statistics ---------------------------------------------------------------------

    /// Structural metrics (for telemetry and the GUI). `weights` is needed for `mean_abs_weight`.
    pub fn stats(&self, input: AdaptInput, weights: Option<&[&[f32]]>) -> StructuralStats {
        let mut s = StructuralStats {
            step: input.step,
            sample: input.sample,
            ..Default::default()
        };
        let (neurons, _) = self.counts();
        s.neurons = neurons;
        s.inputs = self
            .layers
            .iter()
            .find(|l| l.spec.input_from.is_none())
            .map_or(0, |l| l.spec.in_features);
        let (mut cap, mut act, mut rate_sum, mut nrate) = (0usize, 0usize, 0.0f32, 0usize);
        let (mut wsum, mut wcnt) = (0.0f64, 0usize);
        for (l, ls) in self.layers.iter().enumerate() {
            let alive = self.alive_inputs(l);
            cap += alive.iter().filter(|&&a| a).count() * ls.active_neurons();
            let m = self.effective_mask(l);
            let out = ls.spec.out_features;
            act += m.iter().filter(|&&x| x).count();
            match ls.spec.role {
                LayerRole::Hidden => s.hidden += ls.active_neurons(),
                LayerRole::Output => s.outputs += ls.active_neurons(),
            }
            for o in 0..out {
                if !ls.neuron_active[o] {
                    continue;
                }
                let indeg = (0..ls.spec.in_features).filter(|&i| m[i * out + o]).count();
                s.max_in_degree = s.max_in_degree.max(indeg);
                if indeg > 0 {
                    s.active_neurons += 1;
                }
                let r = ls.stats.post_rate[o];
                if ls.stats.activity_samples > 0 {
                    rate_sum += r;
                    nrate += 1;
                    if r < self.cfg.silent_rate {
                        s.silent_neurons += 1;
                    }
                    if r > self.cfg.overactive_rate {
                        s.overactive_neurons += 1;
                    }
                }
            }
            for i in 0..ls.spec.in_features {
                let od = (0..out).filter(|&o| m[i * out + o]).count();
                s.max_out_degree = s.max_out_degree.max(od);
            }
            if let Some(w) = weights {
                for (x, &mm) in w[l].iter().zip(&m) {
                    if mm {
                        wsum += x.abs() as f64;
                        wcnt += 1;
                    }
                }
            }
        }
        s.active_neurons += s.inputs;
        s.synapses = act;
        s.active_synapses = act;
        s.disabled_synapses = cap.saturating_sub(act);
        s.mean_rate = if nrate > 0 {
            rate_sum / nrate as f32
        } else {
            0.0
        };
        s.activity_cost = rate_sum;
        s.mean_abs_weight = if wcnt > 0 {
            (wsum / wcnt as f64) as f32
        } else {
            0.0
        };
        s.density = if cap > 0 {
            act as f32 / cap as f32
        } else {
            0.0
        };
        s.state_bytes = self
            .layers
            .iter()
            .map(|l| l.spec.weight_len() * (4 * 3 + 1 + 8) + l.spec.out_features * 9)
            .sum();
        s
    }

    /// Total weight elements across all layers (capacity) and neurons (including inputs).
    pub fn capacity(&self) -> (usize, usize) {
        let n = self
            .layers
            .iter()
            .find(|l| l.spec.input_from.is_none())
            .map_or(0, |l| l.spec.in_features)
            + self
                .layers
                .iter()
                .map(|l| l.spec.out_features)
                .sum::<usize>();
        (n, self.layers.iter().map(|l| l.spec.weight_len()).sum())
    }

    // ---- mask saving --------------------------------------------------------------------

    /// Serialize the masks (NFXM v1 format, FNV-1a). Statistics and the log are not saved.
    pub fn save_masks(&self) -> Vec<u8> {
        let mut v = b"NFXM".to_vec();
        v.extend_from_slice(&1u32.to_le_bytes());
        v.extend_from_slice(&(self.layers.len() as u32).to_le_bytes());
        for l in &self.layers {
            v.extend_from_slice(&(l.spec.in_features as u64).to_le_bytes());
            v.extend_from_slice(&(l.spec.out_features as u64).to_le_bytes());
            v.extend(l.syn_active.iter().map(|&b| b as u8));
            v.extend(l.neuron_active.iter().map(|&b| b as u8));
        }
        let h = fnv(&v);
        v.extend_from_slice(&h.to_le_bytes());
        v
    }

    /// Load masks into a controller with the same layer dimensions.
    pub fn load_masks(&mut self, bytes: &[u8]) -> StructuralResult<()> {
        let bad = |m: &str| Err(StructuralError::Checkpoint(m.to_string()));
        if bytes.len() < 20 || &bytes[..4] != b"NFXM" {
            return bad("not NFXM");
        }
        let (body, tail) = bytes.split_at(bytes.len() - 8);
        if fnv(body).to_le_bytes() != tail {
            return bad("checksum mismatch");
        }
        if u32::from_le_bytes(body[4..8].try_into().unwrap()) != 1 {
            return bad("unknown version");
        }
        let n = u32::from_le_bytes(body[8..12].try_into().unwrap()) as usize;
        if n != self.layers.len() {
            return bad("layer count mismatch");
        }
        let mut p = 12usize;
        let mut parsed: Vec<(Vec<bool>, Vec<bool>)> = vec![];
        for l in &self.layers {
            if body.len() < p + 16 {
                return bad("truncated");
            }
            let i = u64::from_le_bytes(body[p..p + 8].try_into().unwrap()) as usize;
            let o = u64::from_le_bytes(body[p + 8..p + 16].try_into().unwrap()) as usize;
            p += 16;
            if i != l.spec.in_features || o != l.spec.out_features {
                return bad("layer dimensions mismatch");
            }
            if body.len() < p + i * o + o {
                return bad("truncated");
            }
            parsed.push((
                body[p..p + i * o].iter().map(|&b| b != 0).collect(),
                body[p + i * o..p + i * o + o]
                    .iter()
                    .map(|&b| b != 0)
                    .collect(),
            ));
            p += i * o + o;
        }
        for (l, (s, nn)) in self.layers.iter_mut().zip(parsed) {
            l.syn_active = s;
            l.neuron_active = nn;
        }
        self.pending = None;
        Ok(())
    }
}

fn fnv(data: &[u8]) -> u64 {
    let mut h = 0xcbf2_9ce4_8422_2325u64;
    for b in data {
        h ^= *b as u64;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}
