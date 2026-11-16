//! Operations on synapses: creation, deletion, modification, redirection,
//! disconnection and normalization of connectivity.
//!
//! All operations are atomic: if an error occurs, the topology does not change. Recurrent
//! connections are allowed if `TopologyLimits::allow_recurrent == true`.

use crate::scalar::Scalar;
use std::collections::BTreeSet;

use super::error::{StructuralError, StructuralResult};
use super::topology::{
    Birth, NeuronId, NeuronRole, Origin, Synapse, SynapseId, SynapseState, Topology,
};

/// Changeable synapse parameters.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct SynapseMutation<T = f32> {
    /// New weight (will be limited to `weight_abs_max`).
    pub weight: Option<T>,
    /// New delay.
    pub delay: Option<u16>,
}

/// Rules for normalizing connectivity.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DegreePolicy {
    /// Maximum incoming degree.
    pub max_in: usize,
    /// Maximum outgoing degree.
    pub max_out: usize,
    /// Maximum density `S / N²`; `None` - do not limit.
    pub max_density: Option<f32>,
}

impl<T: Scalar> Topology<T> {
    /// Check whether it is possible to create the `src → dst` synapse without creating it.
    ///
    /// `ignore` is a synapse that is not taken into account (needed for redirection and
    /// changing the delay: it “frees up” its place).
    pub fn check_add_synapse(
        &self,
        src: NeuronId,
        dst: NeuronId,
        delay: u16,
        ignore: Option<SynapseId>,
    ) -> StructuralResult<()> {
        if !self.neurons.contains_key(&src) {
            return Err(StructuralError::NoSuchNeuron(src));
        }
        let d = self
            .neurons
            .get(&dst)
            .ok_or(StructuralError::NoSuchNeuron(dst))?;
        if d.role == NeuronRole::Input {
            return Err(StructuralError::InputCannotReceive(dst));
        }
        if delay > self.param_limits.delay_max {
            return Err(StructuralError::param(
                "delay",
                format!("{delay} > delay_max {}", self.param_limits.delay_max),
            ));
        }
        if src == dst && !self.limits.allow_self_loops {
            return Err(StructuralError::SelfLoopForbidden(src));
        }
        for sid in self.out_synapses(src) {
            if Some(*sid) == ignore {
                continue;
            }
            if let Some(x) = self.synapses.get(sid) {
                if x.dst == dst && (!self.limits.allow_duplicates || x.delay == delay) {
                    return Err(StructuralError::DuplicateSynapse { src, dst });
                }
            }
        }
        if !self.limits.allow_recurrent && (src == dst || self.reaches(dst, src, ignore)) {
            return Err(StructuralError::RecurrenceForbidden { src, dst });
        }
        if ignore.is_none() && self.synapses.len() >= self.limits.max_synapses {
            return Err(StructuralError::LimitExceeded {
                what: "max_synapses",
                limit: self.limits.max_synapses,
            });
        }
        let ign = ignore.and_then(|i| self.synapses.get(&i));
        let ign_in = ign.is_some_and(|x| x.dst == dst) as usize;
        let ign_out = ign.is_some_and(|x| x.src == src) as usize;
        if self.in_synapses(dst).len() - ign_in >= self.limits.max_in_degree {
            return Err(StructuralError::LimitExceeded {
                what: "max_in_degree",
                limit: self.limits.max_in_degree,
            });
        }
        if self.out_synapses(src).len() - ign_out >= self.limits.max_out_degree {
            return Err(StructuralError::LimitExceeded {
                what: "max_out_degree",
                limit: self.limits.max_out_degree,
            });
        }
        Ok(())
    }

    fn clamp_weight(&self, w: T) -> StructuralResult<T> {
        if !w.is_finite() {
            return Err(StructuralError::param(
                "weight",
                format!("non-numeric weight {w}"),
            ));
        }
        let m = self.param_limits.weight_abs_max;
        Ok(w.clamp(-m, m))
    }

    /// Create a synapse. Weight is limited to `weight_abs_max`.
    pub fn add_synapse(
        &mut self,
        src: NeuronId,
        dst: NeuronId,
        weight: T,
        delay: u16,
        origin: Origin,
        birth: Birth,
    ) -> StructuralResult<SynapseId> {
        let w = self.clamp_weight(weight)?;
        self.check_add_synapse(src, dst, delay, None)?;
        // Dale's principle: the sign of the new connection determines the polarity of the source.
        let w = self.dale_new_weight(src, w);
        Ok(self.raw_insert_synapse(src, dst, w, delay, birth, origin))
    }

    /// Remove the synapse and return it.
    pub fn remove_synapse(&mut self, id: SynapseId) -> StructuralResult<Synapse<T>> {
        self.raw_remove_synapse(id)
            .ok_or(StructuralError::NoSuchSynapse(id))
    }

    /// Set weight; returns the actually applied value (after
    /// restrictions).
    pub fn set_synapse_weight(&mut self, id: SynapseId, weight: T) -> StructuralResult<T> {
        let w = self.clamp_weight(weight)?;
        let src = self
            .synapses
            .get(&id)
            .ok_or(StructuralError::NoSuchSynapse(id))?
            .src;
        // Dale's Principle: A weight cannot cross zero (become zero).
        let w = self.dale_clamp_weight(src, w);
        if let Some(s) = self.synapses.get_mut(&id) {
            s.weight = w;
        }
        Ok(w)
    }

    /// Change the available synapse parameters (weight and/or delay).
    pub fn mutate_synapse(&mut self, id: SynapseId, m: SynapseMutation<T>) -> StructuralResult<()> {
        let (src, dst, old_delay) = {
            let s = self
                .synapses
                .get(&id)
                .ok_or(StructuralError::NoSuchSynapse(id))?;
            (s.src, s.dst, s.delay)
        };
        let new_w = match m.weight {
            Some(w) => Some(self.dale_clamp_weight(src, self.clamp_weight(w)?)),
            None => None,
        };
        if let Some(d) = m.delay {
            if d != old_delay {
                self.check_add_synapse(src, dst, d, Some(id))?;
            }
        }
        if let Some(s) = self.synapses.get_mut(&id) {
            if let Some(w) = new_w {
                s.weight = w;
            }
            if let Some(d) = m.delay {
                s.delay = d;
            }
        }
        Ok(())
    }

    /// Redirect the synapse to another source and/or destination. ID,
    /// weight and condition are preserved, usage statistics are reset to zero (this
    /// actually a new connection).
    pub fn rewire_synapse(
        &mut self,
        id: SynapseId,
        new_src: Option<NeuronId>,
        new_dst: Option<NeuronId>,
    ) -> StructuralResult<()> {
        let (old_src, old_dst, delay) = {
            let s = self
                .synapses
                .get(&id)
                .ok_or(StructuralError::NoSuchSynapse(id))?;
            (s.src, s.dst, s.delay)
        };
        let src = new_src.unwrap_or(old_src);
        let dst = new_dst.unwrap_or(old_dst);
        if src == old_src && dst == old_dst {
            return Ok(());
        }
        self.check_add_synapse(src, dst, delay, Some(id))?;
        if let Some(v) = self.out_adj.get_mut(&old_src) {
            v.retain(|x| *x != id);
        }
        if let Some(v) = self.in_adj.get_mut(&old_dst) {
            v.retain(|x| *x != id);
        }
        self.out_adj.entry(src).or_default().push(id);
        self.in_adj.entry(dst).or_default().push(id);
        // Dale's principle: communication is redirected to another source - a sign
        // the weight is taken from its polarity (the modulus is preserved).
        let new_weight =
            self.dale_new_weight(src, self.synapses.get(&id).map_or(T::ZERO, |s| s.weight));
        if let Some(s) = self.synapses.get_mut(&id) {
            s.src = src;
            s.dst = dst;
            s.weight = new_weight;
            let trace_step = s.stats.trace_step;
            s.stats = Default::default();
            s.stats.trace_step = trace_step;
            s.elig = T::ZERO;
        }
        self.touch();
        Ok(())
    }

    /// Disable a synapse (soft removal): it stops transmitting spikes, but
    /// remains in the column until it is finally deleted.
    pub fn disable_synapse(&mut self, id: SynapseId, step: u64) -> StructuralResult<()> {
        let s = self
            .synapses
            .get_mut(&id)
            .ok_or(StructuralError::NoSuchSynapse(id))?;
        if s.is_active() {
            s.state = SynapseState::Disabled { since_step: step };
        }
        Ok(())
    }

    /// Return the disabled synapse to work.
    pub fn enable_synapse(&mut self, id: SynapseId) -> StructuralResult<()> {
        let s = self
            .synapses
            .get_mut(&id)
            .ok_or(StructuralError::NoSuchSynapse(id))?;
        s.state = SynapseState::Active;
        Ok(())
    }

    /// Bring connectivity into the `policy` framework by removing the least useful synapses
    /// (disabled ones first). `utility` — utility rating: the less, the more
    /// removed earlier. Returns removed identifiers.
    ///
    /// The function does not know about the minimum allowable connectivity of individual
    /// neurons; To do this, use pruning with its protections.
    pub fn normalize_connectivity(
        &mut self,
        policy: &DegreePolicy,
        utility: &dyn Fn(&Synapse<T>) -> T,
    ) -> StructuralResult<Vec<SynapseId>> {
        let key = |t: &Topology<T>, sid: &SynapseId| -> (u8, T, SynapseId) {
            let s = &t.synapses[sid];
            (s.is_active() as u8, utility(s), *sid)
        };
        let order = |a: &(u8, T, SynapseId), b: &(u8, T, SynapseId)| {
            a.0.cmp(&b.0)
                .then(a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
                .then(a.2.cmp(&b.2))
        };
        let mut removed: BTreeSet<SynapseId> = BTreeSet::new();
        let ids: Vec<NeuronId> = self.neurons.keys().copied().collect();
        for n in &ids {
            let mut list: Vec<_> = self
                .in_synapses(*n)
                .iter()
                .filter(|s| !removed.contains(s))
                .map(|s| key(self, s))
                .collect();
            if list.len() > policy.max_in {
                list.sort_by(order);
                removed.extend(list.iter().take(list.len() - policy.max_in).map(|k| k.2));
            }
        }
        for n in &ids {
            let mut list: Vec<_> = self
                .out_synapses(*n)
                .iter()
                .filter(|s| !removed.contains(s))
                .map(|s| key(self, s))
                .collect();
            if list.len() > policy.max_out {
                list.sort_by(order);
                removed.extend(list.iter().take(list.len() - policy.max_out).map(|k| k.2));
            }
        }
        if let Some(d) = policy.max_density {
            let n = self.neurons.len() as f32;
            let allowed = (d.max(0.0) * n * n).floor() as usize;
            let mut rest: Vec<_> = self
                .synapses
                .keys()
                .filter(|s| !removed.contains(s))
                .map(|s| key(self, s))
                .collect();
            if rest.len() > allowed {
                rest.sort_by(order);
                removed.extend(rest.iter().take(rest.len() - allowed).map(|k| k.2));
            }
        }
        let out: Vec<SynapseId> = removed.into_iter().collect();
        self.transaction(|t| {
            for id in &out {
                t.remove_synapse(*id)?;
            }
            Ok(())
        })?;
        Ok(out)
    }
}
