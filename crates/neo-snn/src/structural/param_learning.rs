//! Learning physical neuron parameters from a measured loss.
//!
//! Unlike [`NeuronMutation`](super::NeuronMutation) and evolutionary
//! mutation, this mechanism does not merely perturb parameters: it evaluates
//! the candidate on a probe copy of the network and accepts a change only with a real
//! loss improvement. It therefore works for all five models, including AdEx and
//! Hodgkin–Huxley, and needs no per-model code.
//!
//! # Algorithm
//!
//! 1. A baseline copy of the network is measured via the external `eval`.
//! 2. For the selected neuron, the [`NeuronParam`](super::NeuronParam) values it
//!    supports are iterated (with rotation between calls so that
//!    all parameters are covered on a limited budget).
//! 3. For each parameter, steps `+δ` and `−δ` are tried, where
//!    `δ = |value|·step + min_step`; the candidate is applied to the probe copy.
//! 4. The best candidate by loss becomes [`StructuralAction::MutateNeuron`];
//!    the controller's common mechanism with verification and rollback applies it.
//!
//! The number of probe evaluations is bounded by [`ParamLearningConfig::max_probes_per_interval`].
//! The mechanism is disabled by default: without an external quality metric,
//! parameter learning makes no sense.

use super::error::{StructuralError, StructuralResult};
use super::model::NeuronParam;
use super::mutation::{Reason, StructuralAction};
use super::network::DynamicNetwork;
use super::neuron_ops::NeuronMutation;
use super::topology::{NeuronId, NeuronRole};
use crate::scalar::Scalar;

/// Physical parameter learning settings.
#[derive(Debug, Clone, PartialEq)]
pub struct ParamLearningConfig<T = f32> {
    /// Whether the mechanism is enabled.
    pub enabled: bool,
    /// Minimum number of observation steps for a neuron.
    pub min_observed_steps: u64,
    /// Maximum number of neurons processed per interval.
    pub max_neurons_per_interval: usize,
    /// Maximum number of parameters per neuron (the rest are processed on later
    /// calls thanks to rotation).
    pub max_params_per_neuron: usize,
    /// Maximum number of probe evaluations per interval, including the baseline.
    pub max_probes_per_interval: usize,
    /// Relative step size.
    pub step: T,
    /// Absolute minimum step (important for parameters near zero).
    pub min_step: T,
    /// Minimum loss improvement required to accept a change.
    pub min_improvement: T,
    /// Whether to consider input neurons (meaningful in `InputMode::Current`).
    pub include_inputs: bool,
    /// Whether to consider output neurons.
    pub include_outputs: bool,
}

impl<T: Scalar> Default for ParamLearningConfig<T> {
    fn default() -> Self {
        Self {
            enabled: false,
            min_observed_steps: 200,
            max_neurons_per_interval: 4,
            max_params_per_neuron: 4,
            max_probes_per_interval: 32,
            step: T::c(0.05),
            min_step: T::c(1e-3),
            min_improvement: T::c(1e-4),
            include_inputs: false,
            include_outputs: true,
        }
    }
}

impl<T: Scalar> ParamLearningConfig<T> {
    /// Validate the settings.
    pub fn validate(&self) -> StructuralResult<()> {
        let bad = |m: &str| Err(StructuralError::InvalidConfig(m.to_string()));
        if !(self.step.is_finite()
            && self.step > T::ZERO
            && self.min_step.is_finite()
            && self.min_step >= T::ZERO
            && self.min_improvement.is_finite()
            && self.min_improvement >= T::ZERO)
        {
            return bad("param_learning: step > 0, min_step >= 0, min_improvement >= 0");
        }
        if self.enabled
            && (self.max_neurons_per_interval == 0
                || self.max_params_per_neuron == 0
                || self.max_probes_per_interval < 2)
        {
            return bad(
                "param_learning: max_neurons_per_interval > 0, max_params_per_neuron > 0, and max_probes_per_interval >= 2 are required",
            );
        }
        Ok(())
    }

    fn includes(&self, role: NeuronRole) -> bool {
        match role {
            NeuronRole::Input => self.include_inputs,
            NeuronRole::Output => self.include_outputs,
            NeuronRole::Hidden => true,
        }
    }
}

/// Propose physical parameter updates based on the real loss.
///
/// Returns at most one [`StructuralAction::MutateNeuron`] per neuron.
/// The function does not modify the source network: all candidates are evaluated on
/// [`DynamicNetwork::probe_copy`], and the controller (or
/// calling code) applies the actions through the common mechanism with rollback.
pub fn propose_param_updates<T: Scalar>(
    net: &DynamicNetwork<T>,
    cfg: &ParamLearningConfig<T>,
    eval: &mut dyn FnMut(&mut DynamicNetwork<T>) -> T,
) -> Vec<(StructuralAction<T>, Reason)> {
    if !cfg.enabled
        || cfg.max_neurons_per_interval == 0
        || cfg.max_params_per_neuron == 0
        || cfg.max_probes_per_interval < 2
    {
        return Vec::new();
    }

    let mut neurons: Vec<_> = net
        .topology()
        .neurons()
        .filter(|n| cfg.includes(n.role) && n.stats.steps >= cfg.min_observed_steps)
        .collect();
    if neurons.is_empty() {
        return Vec::new();
    }
    // Rotation of the starting position between calls: on a small budget different
    // neurons get a chance, not only the first ones by id.
    let offset = (net.step_count() as usize) % neurons.len();
    neurons.rotate_left(offset);
    neurons.truncate(cfg.max_neurons_per_interval);

    let mut baseline_probe = net.probe_copy();
    let baseline = eval(&mut baseline_probe);
    if !baseline.is_finite() {
        return Vec::new();
    }

    let mut budget = cfg.max_probes_per_interval - 1;
    let mut proposals: Vec<(T, u64, NeuronParam, T)> = Vec::new();
    for (index, neuron) in neurons.into_iter().enumerate() {
        let mut params = neuron.model.params();
        if params.is_empty() {
            continue;
        }
        let start = (net.step_count() as usize + index) % params.len();
        params.rotate_left(start);
        params.truncate(cfg.max_params_per_neuron.min(params.len()));

        let mut best: Option<(T, NeuronParam, T)> = None;
        for (param, value) in params {
            if budget == 0 {
                break;
            }
            let delta = (value.abs() * cfg.step + cfg.min_step).max(cfg.min_step);
            for sign in [T::ONE, -T::ONE] {
                if budget == 0 {
                    break;
                }
                let candidate = value + sign * delta;
                if !candidate.is_finite() || candidate == value {
                    continue;
                }
                let mutation = NeuronMutation {
                    params: vec![(param, candidate)],
                    ..Default::default()
                };
                let mut probe = net.probe_copy();
                let Ok(outcome) = probe.topology_mut().mutate_neuron(neuron.id, mutation) else {
                    continue;
                };
                let actual = outcome.new.param(param).unwrap_or(candidate);
                let loss = eval(&mut probe);
                budget -= 1;
                if !loss.is_finite() || actual == value {
                    continue;
                }
                let better = best.is_none_or(|(b, _, _)| loss < b - T::c(1e-6));
                if better {
                    best = Some((loss, param, actual));
                }
            }
        }
        if let Some((loss, param, value)) = best {
            let improvement = baseline - loss;
            if improvement + T::c(1e-6) >= cfg.min_improvement {
                proposals.push((improvement, neuron.id.0, param, value));
            }
        }
        if budget == 0 {
            break;
        }
    }

    proposals.sort_by(|a, b| {
        b.0.partial_cmp(&a.0)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.1.cmp(&b.1))
    });
    proposals
        .into_iter()
        .map(|(score, id, param, value)| {
            (
                StructuralAction::MutateNeuron {
                    id: NeuronId(id),
                    mutation: NeuronMutation {
                        params: vec![(param, value)],
                        ..Default::default()
                    },
                },
                Reason::ParameterLearning {
                    parameter: param.as_str(),
                    score: score.to_f32(),
                },
            )
        })
        .collect()
}
