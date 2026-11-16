//! A report on the generational evolution of architectures for telemetry (`structural::evolution`).

use crate::structural::evolution::GenerationReport;

/// The result of one generation is what the Evolution tab in the GUI shows.
///
/// Numbers are absolute; reports have `generation`, so report loss if overflowed
/// The queue is visible to the consumer as a gap in numbers.
#[derive(Debug, Clone, PartialEq)]
pub struct EvolutionReport {
    /// Generation number (from scratch).
    pub generation: u64,
    /// How many generations are planned (0 - unlimited).
    pub generations: u64,
    /// Better fitness without penalty.
    pub best_raw: f32,
    /// The best, taking into account the penalty for complexity and division by type.
    pub best_effective: f32,
    /// Average fitness.
    pub mean_raw: f32,
    /// Adaptability of a champion of all time.
    pub champion_raw: f32,
    /// Hidden neurons in the best individual.
    pub best_hidden: usize,
    /// Included connections in the best individual.
    pub best_conns: usize,
    /// Average number of hidden neurons per population.
    pub mean_hidden: f32,
    /// Sizes of species.
    pub species_sizes: Vec<usize>,
    /// Individuals with an invalid genome (received `INVALID_FITNESS`).
    pub invalid: usize,
    /// Current compatibility threshold.
    pub threshold: f32,
    /// Population size.
    pub population: usize,
    /// The goal has been achieved.
    pub solved: bool,
}

impl EvolutionReport {
    /// Collect a report from a population report.
    pub fn from_generation(
        r: &GenerationReport,
        generations: u64,
        population: usize,
        solved: bool,
    ) -> Self {
        Self {
            generation: r.generation,
            generations,
            best_raw: r.best_raw,
            best_effective: r.best_effective,
            mean_raw: r.mean_raw,
            champion_raw: r.champion_raw,
            best_hidden: r.best_hidden,
            best_conns: r.best_conns,
            mean_hidden: r.mean_hidden,
            species_sizes: r.species_sizes.clone(),
            invalid: r.invalid,
            threshold: r.threshold,
            population,
            solved,
        }
    }
}
