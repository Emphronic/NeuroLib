//! Automatic selection of a neuron model based on efficiency and specialization.
//!
//! The system does not try to “predict” the quality of the task for each model - it
//! would require a separate simulation. Instead, it evaluates how
//! the current activity of the neuron matches the candidate's specialization, and
//! subtracts the cost of the model (memory/computation). This assessment:
//!
//! * encourages cheap models (LIF) for neurons with normal activity;
//! * offers slowly adapting ALIF for rare, “slow” neurons;
//! * suggests Izhikevich/AdEx/HH for neurons with high or rich
//!   dynamics if their winnings cover the price.
//!
//! The decision is made by [`StructuralPlasticityController`](super::StructuralPlasticityController)
//! on the same safe boundary as other structural changes:
//! [`propose_model_switches`] only suggests actions
//! [`StructuralAction::ChangeNeuronModel`],
//! but uses their common mechanism with check and rollback.

use super::error::{StructuralError, StructuralResult};
use super::model::{NeuronModel, StateMigration};
use super::mutation::{Reason, StructuralAction};
use super::network::DynamicNetwork;
use super::topology::{Neuron, NeuronRole, ParamLimits};
use crate::scalar::Scalar;

/// Candidate for model change.
#[derive(Debug, Clone, PartialEq)]
pub struct ModelCandidate<T: Scalar = f32> {
    /// Model and its parameters. The parameters are used as initial ones after
    /// switching; the trained parameters of the current model are not carried over.
    pub model: NeuronModel<T>,
    /// Relative price of the model (memory and calculations per step, `≥ 0`).
    pub cost: T,
    /// Desired spike frequency for the specialization (`None` - ignore).
    pub target_rate: Option<T>,
    /// Frequency tolerance width; outside contribution
    /// specialization is zero.
    pub rate_tolerance: T,
}

impl<T: Scalar> ModelCandidate<T> {
    /// Candidate with no frequency preference.
    pub fn new(model: impl Into<NeuronModel<T>>, cost: T) -> Self {
        Self {
            model: model.into(),
            cost,
            target_rate: None,
            rate_tolerance: T::c(0.05),
        }
    }

    /// Set the desired frequency and tolerance width.
    pub fn with_target_rate(mut self, rate: T, tolerance: T) -> Self {
        self.target_rate = Some(rate);
        self.rate_tolerance = tolerance;
        self
    }
}

/// Automatic model selection settings.
#[derive(Debug, Clone, PartialEq)]
pub struct ModelSelectionConfig<T: Scalar = f32> {
    /// Is automatic selection enabled?
    pub enabled: bool,
    /// Minimum steps for monitoring a neuron.
    pub min_observed_steps: u64,
    /// Maximum switchings per interval.
    pub max_switches_per_interval: usize,
    /// Minimum score win to switch model.
    pub min_improvement: T,
    /// Weight of the model price in the assessment.
    pub efficiency_weight: T,
    /// Weight of specialization matching in assessment.
    pub specialization_weight: T,
    /// Evaluate candidates based on real loss on probe copies, and not just
    /// heuristics based on frequency and price. If the evaluator is not passed, it is used
    /// heuristic fallback route.
    pub loss_evaluation: bool,
    /// Minimum improvement loss (`loss_to − loss_candidate`) to
    /// accept the switch. Valid only at `loss_evaluation`.
    pub loss_min_improvement: T,
    /// Maximum probe scores per interval, including the base score. Zero or
    /// the absence of an evaluator includes a heuristic fallback.
    pub max_probes_per_interval: usize,
    /// Switchover state migration policy.
    pub migration: StateMigration,
    /// Whether to consider input neurons.
    pub include_inputs: bool,
    /// Whether to consider output neurons.
    pub include_outputs: bool,
    /// Candidates. If the rating is equal, the cheaper model wins;
    /// The listing order allows ties only if the price is equal.
    pub candidates: Vec<ModelCandidate<T>>,
}

impl<T: Scalar> Default for ModelSelectionConfig<T> {
    fn default() -> Self {
        Self {
            enabled: false,
            min_observed_steps: 200,
            max_switches_per_interval: 1,
            min_improvement: T::c(0.25),
            efficiency_weight: T::ONE,
            specialization_weight: T::c(2.0),
            loss_evaluation: true,
            loss_min_improvement: T::c(1e-4),
            max_probes_per_interval: 32,
            migration: StateMigration::Compatible,
            include_inputs: false,
            include_outputs: false,
            candidates: vec![
                ModelCandidate::new(NeuronModel::Lif(Default::default()), T::ZERO),
                ModelCandidate::new(
                    NeuronModel::AdaptiveLif(super::model::AdaptiveLifParams::slow_adaptation()),
                    T::c(0.1),
                )
                .with_target_rate(T::c(0.01), T::c(0.05)),
                ModelCandidate::new(
                    NeuronModel::Izhikevich(super::model::IzhikevichParams::regular_spiking()),
                    T::c(0.3),
                )
                .with_target_rate(T::c(0.08), T::c(0.08)),
                ModelCandidate::new(
                    NeuronModel::AdEx(super::model::AdExParams::default()),
                    T::c(0.4),
                )
                .with_target_rate(T::c(0.05), T::c(0.05)),
                ModelCandidate::new(
                    NeuronModel::HodgkinHuxley(super::model::HodgkinHuxleyParams::default()),
                    T::c(0.8),
                )
                .with_target_rate(T::c(0.1), T::c(0.1)),
            ],
        }
    }
}

impl<T: Scalar> ModelSelectionConfig<T> {
    /// Check candidate settings and parameters.
    pub fn validate(&self, limits: &ParamLimits<T>) -> StructuralResult<()> {
        let bad = |m: &str| Err(StructuralError::InvalidConfig(m.to_string()));
        let finite = |v: T| v.is_finite();
        if !(finite(self.min_improvement)
            && self.min_improvement >= T::ZERO
            && finite(self.efficiency_weight)
            && self.efficiency_weight >= T::ZERO
            && finite(self.specialization_weight)
            && self.specialization_weight >= T::ZERO
            && finite(self.loss_min_improvement)
            && self.loss_min_improvement >= T::ZERO)
        {
            return bad(
                "model_selection: weights, min_improvement and loss_min_improvement must be finite and >= 0",
            );
        }
        for c in &self.candidates {
            if !(finite(c.cost) && c.cost >= T::ZERO) {
                return bad("model_selection: candidate cost must be finite and >= 0");
            }
            if let Some(rate) = c.target_rate {
                if !(finite(rate) && rate >= T::ZERO) {
                    return bad("model_selection: target_rate must be finite and >= 0");
                }
                if !(finite(c.rate_tolerance) && c.rate_tolerance > T::ZERO) {
                    return bad("model_selection: rate_tolerance must be > 0");
                }
            }
            c.model.validate(limits)?;
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

fn same_variant<T: Scalar>(a: &NeuronModel<T>, b: &NeuronModel<T>) -> bool {
    a.tag() == b.tag()
}

fn candidate_score<T: Scalar>(
    candidate: &ModelCandidate<T>,
    neuron: &Neuron<T>,
    cfg: &ModelSelectionConfig<T>,
) -> T {
    let efficiency = -candidate.cost * cfg.efficiency_weight;
    let specialization = candidate.target_rate.map_or(T::ZERO, |target| {
        let tolerance = candidate.rate_tolerance.max(T::c(1e-6));
        let fit =
            (T::ONE - (neuron.stats.rate_ema - target).abs() / tolerance).clamp(T::ZERO, T::ONE);
        fit * cfg.specialization_weight
    });
    efficiency + specialization
}

/// Suggest switching models for neurons that have a gain
/// specialization/efficiency exceeds `min_improvement`.
///
/// Returns at most `max_switches_per_interval` actions, sorted
/// in descending order of winnings. The function does not change anything: it applies the actions
/// controller (or calling code) via
/// [`apply_action`](super::apply_action).
pub fn propose_model_switches<T: Scalar>(
    net: &DynamicNetwork<T>,
    cfg: &ModelSelectionConfig<T>,
) -> Vec<(StructuralAction<T>, Reason)> {
    if !cfg.enabled || cfg.max_switches_per_interval == 0 || cfg.candidates.is_empty() {
        return Vec::new();
    }
    let mut proposals: Vec<(T, u64, NeuronModel<T>)> = Vec::new();
    for n in net.topology().neurons() {
        if !cfg.includes(n.role) || n.stats.steps < cfg.min_observed_steps {
            continue;
        }
        let current_score = cfg
            .candidates
            .iter()
            .find(|c| same_variant(&c.model, &n.model))
            .map_or(T::ZERO, |c| candidate_score(c, n, cfg));
        // Ties by score are resolved in favor of the cheaper model: otherwise
        // error f32 would decide between Izhikevich and Hodgkin–Huxley there,
        // where their specialization coincides.
        let mut best: Option<(T, T, NeuronModel<T>)> = None;
        for c in &cfg.candidates {
            if same_variant(&c.model, &n.model) {
                continue;
            }
            let score = candidate_score(c, n, cfg);
            let better = best.clone().is_none_or(|(b, cost, _)| {
                score > b + T::c(1e-6) || ((score - b).abs() <= T::c(1e-6) && c.cost < cost)
            });
            if better {
                best = Some((score, c.cost, c.model.clone()));
            }
        }
        let Some((score, _, model)) = best else {
            continue;
        };
        let improvement = score - current_score;
        if improvement + T::c(1e-6) < cfg.min_improvement {
            continue;
        }
        proposals.push((improvement, n.id.0, model));
    }
    proposals.sort_by(|a, b| {
        b.0.partial_cmp(&a.0)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.1.cmp(&b.1))
    });
    proposals.truncate(cfg.max_switches_per_interval);
    proposals
        .into_iter()
        .map(|(improvement, id, model)| {
            (
                StructuralAction::ChangeNeuronModel {
                    id: super::topology::NeuronId(id),
                    model,
                    migration: cfg.migration,
                },
                Reason::ModelSelection {
                    score: improvement.to_f32(),
                },
            )
        })
        .collect()
}

/// Offer switching models based on real loss on probe copies.
///
/// Unlike [`propose_model_switches`], this function does not solve one by one
/// heuristics: for each candidate [`DynamicNetwork::probe_copy`] is created
/// (plasticity is turned off, the state is reset), a change is applied in the copy
/// model, and `eval` measures the loss. Gain - `loss_to − loss_candidate`;
/// the best candidate is selected, and if the loss is equal, the cheaper one is selected.
///
/// The number of calls to `eval` is limited to `max_probes_per_interval` (including
/// basic assessment), so the expensive assessor does not block learning. If
/// `loss_evaluation` is off, the estimator did not return a finite value, or the budget
/// less than two samples, heuristic is used
/// [`propose_model_switches`] - behavior remains defined even without
/// external assessment.
pub fn propose_model_switches_evaluated<T: Scalar>(
    net: &DynamicNetwork<T>,
    cfg: &ModelSelectionConfig<T>,
    eval: &mut dyn FnMut(&mut DynamicNetwork<T>) -> T,
) -> Vec<(StructuralAction<T>, Reason)> {
    if !cfg.enabled || cfg.max_switches_per_interval == 0 || cfg.candidates.is_empty() {
        return Vec::new();
    }
    if !cfg.loss_evaluation || cfg.max_probes_per_interval < 2 {
        return propose_model_switches(net, cfg);
    }

    // First, we order the neurons using heuristics: it does not decide the outcome, but only
    // chooses who to spend a limited sample budget on.
    #[allow(clippy::type_complexity)]
    let mut plans: Vec<(T, u64, Vec<(T, T, NeuronModel<T>)>)> = Vec::new();
    for n in net.topology().neurons() {
        if !cfg.includes(n.role) || n.stats.steps < cfg.min_observed_steps {
            continue;
        }
        let mut candidates: Vec<(T, T, NeuronModel<T>)> = cfg
            .candidates
            .iter()
            .filter(|c| !same_variant(&c.model, &n.model))
            .map(|c| (candidate_score(c, n, cfg), c.cost, c.model.clone()))
            .collect();
        if candidates.is_empty() {
            continue;
        }
        candidates.sort_by(|a, b| {
            b.0.partial_cmp(&a.0)
                .unwrap_or(std::cmp::Ordering::Equal)
                .then(a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal))
        });
        plans.push((candidates[0].0, n.id.0, candidates));
    }
    plans.sort_by(|a, b| {
        b.0.partial_cmp(&a.0)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.1.cmp(&b.1))
    });

    let mut baseline_probe = net.probe_copy();
    let baseline = eval(&mut baseline_probe);
    if !baseline.is_finite() {
        return propose_model_switches(net, cfg);
    }
    let mut remaining = cfg.max_probes_per_interval - 1;
    let mut proposals: Vec<(T, u64, NeuronModel<T>)> = Vec::new();
    let mut any_finite = false;
    for (_, id, candidates) in plans {
        let id = super::topology::NeuronId(id);
        let mut best: Option<(T, T, NeuronModel<T>)> = None;
        for (_, cost, model) in candidates {
            if remaining == 0 {
                break;
            }
            let mut probe = net.probe_copy();
            remaining -= 1;
            if probe
                .change_neuron_model(id, model.clone(), cfg.migration)
                .is_err()
            {
                continue;
            }
            let loss = eval(&mut probe);
            if !loss.is_finite() {
                continue;
            }
            any_finite = true;
            let better = best.clone().is_none_or(|(b, bcost, _)| {
                loss < b - T::c(1e-6) || ((loss - b).abs() <= T::c(1e-6) && cost < bcost)
            });
            if better {
                best = Some((loss, cost, model));
            }
        }
        if let Some((loss, _, model)) = best {
            let improvement = baseline - loss;
            if improvement + T::c(1e-6) >= cfg.loss_min_improvement {
                proposals.push((improvement, id.0, model));
            }
        }
        if remaining == 0 {
            break;
        }
    }
    if !any_finite {
        return propose_model_switches(net, cfg);
    }
    proposals.sort_by(|a, b| {
        b.0.partial_cmp(&a.0)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then(a.1.cmp(&b.1))
    });
    proposals.truncate(cfg.max_switches_per_interval);
    proposals
        .into_iter()
        .map(|(improvement, id, model)| {
            (
                StructuralAction::ChangeNeuronModel {
                    id: super::topology::NeuronId(id),
                    model,
                    migration: cfg.migration,
                },
                Reason::ModelSelection {
                    score: improvement.to_f32(),
                },
            )
        })
        .collect()
}
