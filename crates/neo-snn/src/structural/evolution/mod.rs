//! Architecture evolution (NEAT-like) on top of structural plasticity.
//!
//! Two mechanisms operate on different time scales and on the same graph:
//!
//! | | what changes | when | module |
//! |---|---|---|---|
//! | structural plasticity | topology of ONE network | within a "lifetime", between samples | [`StructuralPlasticityController`](super::StructuralPlasticityController) |
//! | architecture evolution | topologies of a POPULATION | between generations | this module |
//!
//! * [`Genome`] — a network blueprint with historical markers (innovation numbers), converted to
//!   [`Topology`](super::Topology) and back ([`Genome::to_topology`], [`Genome::from_topology`]).
//! * [`Population`] — evaluation, speciation by compatibility distance, fitness sharing,
//!   species stagnation, elitism, crossover with marker alignment, mutations: weights, LIF parameters,
//!   new connection, new neuron by splitting a connection, enable/disable.
//! * Fitness is computed by your function `Fn(&mut DynamicNetwork) -> f32`. Inside it the network
//!   can be trained (STDP, structural plasticity controller): the architecture is searched by evolution,
//!   the weights by training. A complexity penalty ([`ComplexityPenalty`]) keeps networks small.
//! * All limits of [`TopologyLimits`](super::TopologyLimits) and
//!   [`ParamLimits`](super::ParamLimits) are enforced for every offspring; an invalid genome
//!   receives [`INVALID_FITNESS`] instead of crashing the evolution.
//! * Determinism: one `seed` — one evolution run; the number of evaluation threads does not change the result;
//!   [`Population::save`]/[`Population::load`] allow stopping and resuming without differences.
//!
//! By default evaluation runs on the CPU engine. The batched dense baseline ([`dense`](super::dense)) and its
//! Burn mirror (`neo_snn::dense_burn`) prepare evaluation on the device, but the Burn part is untested
//! (see `STRUCTURAL_INTEGRATION.md`); GPU execution is not claimed.

mod codec;
mod genome;
mod operators;
mod population;

#[cfg(test)]
mod tests;

pub use genome::{ConnGene, Genome, Innov, InnovationTracker, NodeGene};
pub use operators::{MutationConfig, MutationSummary};
pub use population::{
    Champion, ComplexityPenalty, EvolutionConfig, GenerationReport, INVALID_FITNESS, Individual,
    InitialConnectivity, Population, SelectionConfig, SpeciationConfig, Species, build_network,
};
