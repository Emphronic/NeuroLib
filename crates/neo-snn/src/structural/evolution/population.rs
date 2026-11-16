//! Population: Estimation, Speciation, Selection and Reproduction (NEAT).
//!
//! Generation cycle ([`Population::step`]):
//! 1. **score** - each genome turns into [`DynamicNetwork`], custom function
//!    returns fitness (more is better). The function receives `&mut` network, so
//!    inside the “life” of an individual, the network can learn (STDP, [`StructuralPlasticityController`](super::super::StructuralPlasticityController)):
//!    evolution changes architecture, plasticity changes weight;
//! 2. **species** - genomes are grouped by compatibility distance, so that the new structure
//!    competes with its own people, and not with the entire population;
//! 3. **selection** - fitness is divided by the size of the species; species not improved
//!    `stagnation_generations` generations are dying out (except for the best);
//! 4. **reproduction** - elite, crossover of the best individuals of the species, mutation.
//!
//! Everything is determined at one `seed`; the number of evaluation threads does not affect the result.

use super::super::error::{StructuralError, StructuralResult};
use super::super::model::{LifParams, NeuronModel, NeuronRole};
use super::super::network::{DynamicNetwork, EngineConfig};
use super::super::rng::Rng64;
use super::super::topology::{ParamLimits, Topology, TopologyLimits};
use super::codec::{MAX_ITEMS, R, W, err};
use super::genome::{ConnGene, Genome, InnovationTracker};
use super::operators::{Env, MutationConfig, MutationSummary, crossover, mutate};
use crate::scalar::Scalar;

/// Invalid genome fitness (not built, or the function did not return a number).
pub const INVALID_FITNESS: f32 = -1.0e9;

/// Initial connections of the minimal genome.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum InitialConnectivity {
    /// Every input is connected to every output.
    Full,
    /// Each input→output pair is associated with a probability `p`.
    Sparse(f32),
    /// There are no connections: everything will grow through mutations.
    Empty,
}

/// Speciation.
#[derive(Debug, Clone, PartialEq)]
pub struct SpeciationConfig {
    /// Weight of extra genes.
    pub c1: f32,
    /// Weight of unpaired genes.
    pub c2: f32,
    /// Weight of the average difference of weights.
    pub c3: f32,
    /// Compatibility threshold: closer - one type.
    pub threshold: f32,
    /// Desired number of species: the threshold is adjusted in steps of `threshold_step` (`None` - do not change).
    pub target_species: Option<usize>,
    /// Threshold adjustment step per generation.
    pub threshold_step: f32,
}

impl Default for SpeciationConfig {
    fn default() -> Self {
        Self {
            c1: 1.0,
            c2: 1.0,
            c3: 0.4,
            threshold: 3.0,
            target_species: Some(6),
            threshold_step: 0.2,
        }
    }
}

/// Selection and reproduction.
#[derive(Debug, Clone, PartialEq)]
pub struct SelectionConfig {
    /// The proportion of the best individuals of a species allowed to reproduce.
    pub survival_rate: f32,
    /// A species no smaller than this size carries its best into the next generation unchanged.
    pub elitism_min_species: usize,
    /// The probability of a crossover (otherwise a mutated copy of one parent).
    pub crossover_rate: f32,
    /// Generations without improvement, after which the species becomes extinct.
    pub stagnation_generations: u32,
    /// How many of the best species do not die out from stagnation?
    pub keep_species: usize,
}

impl Default for SelectionConfig {
    fn default() -> Self {
        Self {
            survival_rate: 0.3,
            elitism_min_species: 5,
            crossover_rate: 0.75,
            stagnation_generations: 15,
            keep_species: 2,
        }
    }
}

/// Penalty for difficulty: selection is based on `fitness - penalty`, reports show the original one.
#[derive(Debug, Clone, Copy, PartialEq, Default)]
pub struct ComplexityPenalty {
    /// For each hidden neuron.
    pub per_hidden: f32,
    /// For every connection expressed.
    pub per_conn: f32,
}

/// Evolution settings.
#[derive(Debug, Clone, PartialEq)]
pub struct EvolutionConfig<T = f32> {
    /// Inputs (the same for all individuals).
    pub inputs: usize,
    /// Outputs.
    pub outputs: usize,
    /// Population size.
    pub population: usize,
    /// Grain.
    pub seed: u64,
    /// Initial connections.
    pub initial: InitialConnectivity,
    /// Parameters of output neurons (used if `output_model` = `None`).
    pub output_params: LifParams<T>,
    /// Model of output neurons; `None` - LIF with `output_params`.
    pub output_model: Option<NeuronModel<T>>,
    /// Mutations.
    pub mutation: MutationConfig<T>,
    /// Types.
    pub speciation: SpeciationConfig,
    /// Selection.
    pub selection: SelectionConfig,
    /// Difficulty penalty.
    pub complexity: ComplexityPenalty,
    /// Topology limits: any descendant satisfies them.
    pub limits: TopologyLimits,
    /// Limits of parameters and weights.
    pub param_limits: ParamLimits<T>,
    /// Network engine parameters (rule of plasticity within “life”).
    pub engine: EngineConfig<T>,
    /// Evaluation threads (0 or 1 - one thread). Does not affect the result.
    pub threads: usize,
}

impl<T: Scalar> EvolutionConfig<T> {
    /// Default settings for the `inputs → outputs` network.
    pub fn new(inputs: usize, outputs: usize) -> Self {
        let limits = TopologyLimits {
            max_neurons: 64,
            max_synapses: 256,
            ..TopologyLimits::default()
        };
        Self {
            inputs,
            outputs,
            population: 100,
            seed: 1,
            initial: InitialConnectivity::Full,
            output_params: LifParams::new(T::c(0.9), T::ONE),
            output_model: None,
            mutation: MutationConfig::default(),
            speciation: SpeciationConfig::default(),
            selection: SelectionConfig::default(),
            complexity: ComplexityPenalty::default(),
            limits,
            param_limits: ParamLimits::default(),
            engine: EngineConfig::default(),
            threads: 1,
        }
    }

    /// Checking settings.
    pub fn validate(&self) -> StructuralResult<()> {
        let bad =
            |what: &'static str, why: &str| Err(StructuralError::param(what, why.to_string()));
        let prob = |p: f32| p.is_finite() && (0.0..=1.0).contains(&p);
        let probt = |p: T| prob(p.to_f32());
        if self.inputs == 0 || self.outputs == 0 {
            return bad(
                "inputs/outputs",
                "you need at least one input and one output",
            );
        }
        if self.population < 2 {
            return bad("population", "you need at least two individuals");
        }
        if self.inputs + self.outputs > self.limits.max_neurons {
            return bad("limits.max_neurons", "less than inputs and outputs");
        }
        let m = &self.mutation;
        if ![
            m.weight_mutate_prob,
            m.weight_replace_prob,
            m.add_conn_prob,
            m.add_node_prob,
            m.toggle_prob,
            m.param_mutate_prob,
            m.model_mutate_prob,
            m.polarity_mutate_prob,
        ]
        .into_iter()
        .all(probt)
            || !prob(self.selection.survival_rate)
            || !prob(self.selection.crossover_rate)
        {
            return bad("probability", "must be in [0, 1]");
        }
        if self.selection.survival_rate <= 0.0 {
            return bad("survival_rate", "must be > 0");
        }
        if !(self.speciation.threshold.is_finite() && self.speciation.threshold > 0.0) {
            return bad("speciation.threshold", "must be > 0");
        }
        if !(m.beta_power.is_finite()
            && m.beta_power >= T::ZERO
            && m.threshold_power.is_finite()
            && m.threshold_power >= T::ZERO
            && m.param_power.is_finite()
            && m.param_power >= T::ZERO
            && m.param_min_step.is_finite()
            && m.param_min_step >= T::ZERO)
        {
            return bad(
                "mutation.power",
                "beta_power/threshold_power/param_power/param_min_step must be finite and >= 0",
            );
        }
        if let InitialConnectivity::Sparse(p) = self.initial {
            if !prob(p) {
                return bad("initial", "probability Sparse outside [0, 1]");
            }
        }
        if !(self.complexity.per_hidden >= 0.0 && self.complexity.per_conn >= 0.0) {
            return bad("complexity", "penalties cannot be negative");
        }
        if let Some(m) = &self.output_model {
            m.validate(&self.param_limits)?;
        }
        if let Some(m) = &self.mutation.split_model {
            m.validate(&self.param_limits)?;
        }
        for m in &self.mutation.model_candidates {
            m.validate(&self.param_limits)?;
        }
        Ok(())
    }
}

/// Individual of a generation.
#[derive(Debug, Clone, PartialEq)]
pub struct Individual<T = f32> {
    /// Genome.
    pub genome: Genome<T>,
    /// Fitness returned from assessment (`NaN` before assessment).
    pub raw: f32,
    /// Fitness minus the difficulty penalty (based on selection).
    pub effective: f32,
    /// Was the genome valid?
    pub valid: bool,
}

/// View.
#[derive(Debug, Clone, PartialEq)]
pub struct Species<T = f32> {
    /// Number (not repeated).
    pub id: u32,
    /// Representative: new individuals are compared with it.
    pub representative: Genome<T>,
    /// Indices of individuals of the current generation.
    pub members: Vec<usize>,
    /// The best `effective` for the entire life of the species.
    pub best: f32,
    /// Generations without improvement `best`.
    pub stagnation: u32,
    /// Age in generations.
    pub age: u32,
}

/// The best genome ever.
#[derive(Debug, Clone, PartialEq)]
pub struct Champion<T = f32> {
    /// Genome.
    pub genome: Genome<T>,
    /// Fitness.
    pub raw: f32,
    /// Accommodation minus penalty.
    pub effective: f32,
    /// The generation in which it appeared.
    pub generation: u64,
}

/// The result of a generation.
#[derive(Debug, Clone, PartialEq)]
pub struct GenerationReport {
    /// Generation number (from scratch).
    pub generation: u64,
    /// Best initial fitness of a generation.
    pub best_raw: f32,
    /// Best fitness minus penalty.
    pub best_effective: f32,
    /// Average initial fitness of eligible individuals.
    pub mean_raw: f32,
    /// Hidden neurons in the best individual.
    pub best_hidden: usize,
    /// Pronounced connections in the best individual.
    pub best_conns: usize,
    /// Average number of hidden neurons per population.
    pub mean_hidden: f32,
    /// Sizes of species.
    pub species_sizes: Vec<usize>,
    /// Invalid genomes.
    pub invalid: usize,
    /// Current compatibility threshold.
    pub threshold: f32,
    /// Best fitness ever.
    pub champion_raw: f32,
}

/// Population.
#[derive(Debug, Clone)]
pub struct Population<T = f32> {
    cfg: EvolutionConfig<T>,
    rng: Rng64,
    tracker: InnovationTracker,
    individuals: Vec<Individual<T>>,
    species: Vec<Species<T>>,
    next_species: u32,
    threshold: f32,
    generation: u64,
    champion: Option<Champion<T>>,
}

impl<T: Scalar> Population<T> {
    /// An initial population of minimal genomes with random weights.
    pub fn new(cfg: EvolutionConfig<T>) -> StructuralResult<Self> {
        cfg.validate()?;
        let mut rng = Rng64::new(cfg.seed);
        let mut tracker = InnovationTracker::new((cfg.inputs + cfg.outputs) as u64);
        let mut individuals = Vec::with_capacity(cfg.population);
        for _ in 0..cfg.population {
            let output_model = cfg
                .output_model
                .unwrap_or(NeuronModel::Lif(cfg.output_params));
            let mut g = Genome::bare(cfg.inputs, cfg.outputs, output_model);
            for i in 0..cfg.inputs {
                for o in 0..cfg.outputs {
                    let connect = match cfg.initial {
                        InitialConnectivity::Full => true,
                        InitialConnectivity::Sparse(p) => rng.chance(p),
                        InitialConnectivity::Empty => false,
                    };
                    if connect {
                        let (src, dst) = (i as u64, (cfg.inputs + o) as u64);
                        let w = (rng.symmetric_as::<T>() * cfg.mutation.init_weight).clamp(
                            -cfg.param_limits.weight_abs_max,
                            cfg.param_limits.weight_abs_max,
                        );
                        let innov = tracker.conn(src, dst);
                        g.insert_conn(ConnGene {
                            innov,
                            src,
                            dst,
                            weight: w,
                            delay: 0,
                            enabled: true,
                        });
                    }
                }
            }
            if cfg.limits.enforce_dale {
                g.normalize_dale();
            }
            individuals.push(Individual {
                genome: g,
                raw: f32::NAN,
                effective: f32::NAN,
                valid: false,
            });
        }
        let threshold = cfg.speciation.threshold;
        Ok(Self {
            cfg,
            rng,
            tracker,
            individuals,
            species: Vec::new(),
            next_species: 0,
            threshold,
            generation: 0,
            champion: None,
        })
    }

    /// A population of copies of `template` (for example, a trained network converted into the genome
    /// [`Genome::from_topology`]): the first individual is the template itself, the rest are its mutants.
    /// `tracker` is the one that created the template.
    pub fn seeded(
        cfg: EvolutionConfig<T>,
        template: Genome<T>,
        tracker: InnovationTracker,
    ) -> StructuralResult<Self> {
        let mut p = Self::new(cfg)?;
        template.validate_with(p.cfg.limits.enforce_dale)?;
        let inputs = template
            .nodes
            .iter()
            .filter(|n| n.kind == NeuronRole::Input)
            .count();
        let outputs = template
            .nodes
            .iter()
            .filter(|n| n.kind == NeuronRole::Output)
            .count();
        if (inputs, outputs) != (p.cfg.inputs, p.cfg.outputs) {
            return Err(err(format!(
                "template {inputs}→{outputs} does not fit the settings {}→{}",
                p.cfg.inputs, p.cfg.outputs
            )));
        }
        p.tracker = tracker;
        for (k, ind) in p.individuals.iter_mut().enumerate() {
            ind.genome = template.clone();
            if k > 0 {
                let env = Env {
                    cfg: &p.cfg.mutation,
                    limits: &p.cfg.limits,
                    plimits: &p.cfg.param_limits,
                };
                mutate(&mut ind.genome, &env, &mut p.tracker, &mut p.rng);
            }
        }
        Ok(p)
    }

    /// Settings.
    pub fn config(&self) -> &EvolutionConfig<T> {
        &self.cfg
    }

    /// Number of the next generation (how many have already been played).
    pub fn generation(&self) -> u64 {
        self.generation
    }

    /// Individuals of the current generation.
    pub fn individuals(&self) -> &[Individual<T>] {
        &self.individuals
    }

    /// Types (after the first `step`).
    pub fn species(&self) -> &[Species<T>] {
        &self.species
    }

    /// The best genome ever.
    pub fn champion(&self) -> Option<&Champion<T>> {
        self.champion.as_ref()
    }

    /// Register of markers.
    pub fn tracker(&self) -> &InnovationTracker {
        &self.tracker
    }

    /// Network of the best genome (for additional training, display, checkpoint of structural plasticity).
    pub fn champion_network(&self) -> StructuralResult<Option<DynamicNetwork<T>>> {
        self.champion
            .as_ref()
            .map(|c| self.build(&c.genome))
            .transpose()
    }

    /// Build a network from the genome using population settings.
    pub fn build(&self, g: &Genome<T>) -> StructuralResult<DynamicNetwork<T>> {
        build_network(g, &self.cfg)
    }

    fn penalty(&self, g: &Genome<T>) -> f32 {
        self.cfg.complexity.per_hidden * g.hidden_count() as f32
            + self.cfg.complexity.per_conn * g.enabled_count() as f32
    }

    /// Rate everyone. The order of the results does not depend on the number of threads.
    fn evaluate<F: Fn(&mut DynamicNetwork<T>) -> f32 + Sync>(&mut self, fitness: &F) {
        let genomes: Vec<&Genome<T>> = self.individuals.iter().map(|i| &i.genome).collect();
        let cfg = &self.cfg;
        let one = |g: &Genome<T>| -> Option<f32> {
            let mut net = build_network(g, cfg).ok()?;
            let f = fitness(&mut net);
            f.is_finite().then_some(f)
        };
        let threads = cfg.threads.max(1).min(genomes.len());
        let results: Vec<Option<f32>> = if threads <= 1 {
            genomes.iter().map(|g| one(g)).collect()
        } else {
            let chunk = genomes.len().div_ceil(threads);
            std::thread::scope(|s| {
                let handles: Vec<_> = genomes
                    .chunks(chunk)
                    .map(|part| s.spawn(|| part.iter().map(|g| one(g)).collect::<Vec<_>>()))
                    .collect();
                handles
                    .into_iter()
                    .flat_map(|h| h.join().expect("assessment flow"))
                    .collect()
            })
        };
        for (ind, r) in self.individuals.iter_mut().zip(results) {
            match r {
                Some(f) => {
                    ind.raw = f;
                    ind.valid = true;
                }
                None => {
                    ind.raw = INVALID_FITNESS;
                    ind.valid = false;
                }
            }
        }
        for i in 0..self.individuals.len() {
            let pen = self.penalty(&self.individuals[i].genome);
            let ind = &mut self.individuals[i];
            ind.effective = if ind.valid {
                ind.raw - pen
            } else {
                INVALID_FITNESS
            };
        }
    }

    fn speciate(&mut self) {
        for s in &mut self.species {
            s.members.clear();
        }
        let sp = &self.cfg.speciation;
        for (i, ind) in self.individuals.iter().enumerate() {
            let found = self.species.iter().position(|s| {
                s.representative
                    .distance(
                        &ind.genome,
                        T::from_f32(sp.c1),
                        T::from_f32(sp.c2),
                        T::from_f32(sp.c3),
                    )
                    .to_f32()
                    < self.threshold
            });
            match found {
                Some(k) => self.species[k].members.push(i),
                None => {
                    self.species.push(Species {
                        id: self.next_species,
                        representative: ind.genome.clone(),
                        members: vec![i],
                        best: f32::NEG_INFINITY,
                        stagnation: 0,
                        age: 0,
                    });
                    self.next_species += 1;
                }
            }
        }
        self.species.retain(|s| !s.members.is_empty());
        if let Some(target) = sp.target_species {
            if self.species.len() < target {
                self.threshold = (self.threshold - sp.threshold_step).max(0.3);
            } else if self.species.len() > target {
                self.threshold += sp.threshold_step;
            }
        }
    }

    fn best_of(&self, members: &[usize]) -> usize {
        let mut best = members[0];
        for &m in &members[1..] {
            if self.individuals[m].effective > self.individuals[best].effective {
                best = m;
            }
        }
        best
    }

    /// Update species and champion statistics, collect a report.
    fn account(&mut self) -> GenerationReport {
        let (mut best_i, mut sum, mut valid) = (0usize, 0.0f32, 0usize);
        for (i, ind) in self.individuals.iter().enumerate() {
            if ind.effective > self.individuals[best_i].effective {
                best_i = i;
            }
            if ind.valid {
                sum += ind.raw;
                valid += 1;
            }
        }
        let best = self.individuals[best_i].clone();
        if best.valid
            && self
                .champion
                .as_ref()
                .is_none_or(|c| best.effective > c.effective)
        {
            self.champion = Some(Champion {
                genome: best.genome.clone(),
                raw: best.raw,
                effective: best.effective,
                generation: self.generation,
            });
        }
        for k in 0..self.species.len() {
            let members = self.species[k].members.clone();
            let b = self.best_of(&members);
            let eff = self.individuals[b].effective;
            let rep = self.individuals[b].genome.clone();
            let s = &mut self.species[k];
            if eff > s.best + 1e-6 {
                s.best = eff;
                s.stagnation = 0;
            } else {
                s.stagnation += 1;
            }
            s.age += 1;
            s.representative = rep;
        }
        GenerationReport {
            generation: self.generation,
            best_raw: best.raw,
            best_effective: best.effective,
            mean_raw: if valid > 0 {
                sum / valid as f32
            } else {
                INVALID_FITNESS
            },
            best_hidden: best.genome.hidden_count(),
            best_conns: best.genome.enabled_count(),
            mean_hidden: self
                .individuals
                .iter()
                .map(|i| i.genome.hidden_count() as f32)
                .sum::<f32>()
                / self.individuals.len() as f32,
            species_sizes: self.species.iter().map(|s| s.members.len()).collect(),
            invalid: self.individuals.len() - valid,
            threshold: self.threshold,
            champion_raw: self.champion.as_ref().map_or(INVALID_FITNESS, |c| c.raw),
        }
    }

    /// The next generation: extinction, distribution of descendants by species, reproduction.
    fn reproduce(&mut self) {
        let n = self.cfg.population;
        let sel = self.cfg.selection.clone();
        // Stagnating species die out, except for the best according to `best`
        let mut order: Vec<usize> = (0..self.species.len()).collect();
        order.sort_by(|&a, &b| {
            self.species[b]
                .best
                .total_cmp(&self.species[a].best)
                .then(a.cmp(&b))
        });
        let protected: Vec<usize> = order
            .iter()
            .take(sel.keep_species.max(1))
            .copied()
            .collect();
        let alive: Vec<usize> = (0..self.species.len())
            .filter(|k| {
                protected.contains(k) || self.species[*k].stagnation < sel.stagnation_generations
            })
            .collect();

        // The proportion of a species is the average fitness of its individuals, shifted so as to be ≥ 0
        let min_eff = self
            .individuals
            .iter()
            .map(|i| i.effective)
            .fold(f32::INFINITY, f32::min);
        let share: Vec<f32> = alive
            .iter()
            .map(|&k| {
                let m = &self.species[k].members;
                m.iter()
                    .map(|&i| self.individuals[i].effective - min_eff)
                    .sum::<f32>()
                    / m.len() as f32
            })
            .collect();
        let total: f32 = share.iter().sum();
        let exact: Vec<f32> = if total > f32::EPSILON {
            share.iter().map(|s| s / total * n as f32).collect()
        } else {
            vec![n as f32 / alive.len() as f32; alive.len()]
        };
        let mut alloc: Vec<usize> = exact.iter().map(|e| e.floor() as usize).collect();
        let mut rest = n - alloc.iter().sum::<usize>();
        let mut by_fraction: Vec<usize> = (0..alive.len()).collect();
        by_fraction.sort_by(|&a, &b| {
            (exact[b] - exact[b].floor())
                .total_cmp(&(exact[a] - exact[a].floor()))
                .then(a.cmp(&b))
        });
        for &k in by_fraction.iter().cycle().take(rest) {
            alloc[k] += 1;
            rest -= 1;
        }
        debug_assert_eq!(rest, 0);

        let global_best = {
            let all: Vec<usize> = (0..self.individuals.len()).collect();
            self.best_of(&all)
        };
        let mut next: Vec<Individual<T>> = Vec::with_capacity(n);
        let fresh = |g: Genome<T>| Individual {
            genome: g,
            raw: f32::NAN,
            effective: f32::NAN,
            valid: false,
        };
        for (pos, &k) in alive.iter().enumerate() {
            let mut left = alloc[pos];
            if left == 0 {
                continue;
            }
            let mut members = self.species[k].members.clone();
            members.sort_by(|&a, &b| {
                self.individuals[b]
                    .effective
                    .total_cmp(&self.individuals[a].effective)
                    .then(a.cmp(&b))
            });
            if (members.len() >= sel.elitism_min_species || members.contains(&global_best))
                && left > 0
            {
                let elite = if members.contains(&global_best) {
                    global_best
                } else {
                    members[0]
                };
                next.push(fresh(self.individuals[elite].genome.clone()));
                left -= 1;
            }
            let keep = ((members.len() as f32 * sel.survival_rate).ceil() as usize)
                .clamp(1, members.len());
            let parents = &members[..keep];
            for _ in 0..left {
                let a = parents[self.rng.below(parents.len())];
                let mut child = if parents.len() >= 2 && self.rng.chance(sel.crossover_rate) {
                    let mut b = parents[self.rng.below(parents.len())];
                    if b == a {
                        b = parents[(parents.iter().position(|&x| x == a).unwrap_or(0) + 1)
                            % parents.len()];
                    }
                    let (ia, ib) = (&self.individuals[a], &self.individuals[b]);
                    crossover(
                        &ia.genome,
                        ia.effective,
                        &ib.genome,
                        ib.effective,
                        &self.cfg.limits,
                        &mut self.rng,
                    )
                } else {
                    self.individuals[a].genome.clone()
                };
                let env = Env {
                    cfg: &self.cfg.mutation,
                    limits: &self.cfg.limits,
                    plimits: &self.cfg.param_limits,
                };
                mutate(&mut child, &env, &mut self.tracker, &mut self.rng);
                next.push(fresh(child));
            }
        }
        debug_assert_eq!(next.len(), n);
        // Extinct species are passing away; the rest have representatives (for comparison in speciate)
        let alive_species: Vec<Species<T>> =
            alive.iter().map(|&k| self.species[k].clone()).collect();
        self.species = alive_species;
        self.individuals = next;
    }

    /// One generation: assessment → species → report → reproduction.
    pub fn step<F: Fn(&mut DynamicNetwork<T>) -> f32 + Sync>(
        &mut self,
        fitness: &F,
    ) -> GenerationReport {
        self.evaluate(fitness);
        self.speciate();
        let report = self.account();
        self.reproduce();
        self.generation += 1;
        report
    }

    /// Up to `max_generations` generations; stops when initial fitness is best
    /// reached `target`. `on_generation` is called after each. Returns the generation number,
    /// achieved the goal.
    pub fn run<F: Fn(&mut DynamicNetwork<T>) -> f32 + Sync>(
        &mut self,
        fitness: &F,
        max_generations: u64,
        target: Option<f32>,
        mut on_generation: impl FnMut(&GenerationReport),
    ) -> Option<u64> {
        for _ in 0..max_generations {
            let r = self.step(fitness);
            on_generation(&r);
            if target.is_some_and(|t| r.best_raw >= t) {
                return Some(r.generation);
            }
        }
        None
    }

    /// Mutates the genome the same way reproduction does (for external experiments and tests).
    pub fn mutate_genome(&mut self, g: &mut Genome<T>) -> MutationSummary {
        let env = Env {
            cfg: &self.cfg.mutation,
            limits: &self.cfg.limits,
            plimits: &self.cfg.param_limits,
        };
        mutate(g, &env, &mut self.tracker, &mut self.rng)
    }

    /// Save state (`NPOP` magic; version 2 for `f32`, version 3 for `f64` - after version
    /// is the width of the gene values). Settings are not saved: they are transferred
    /// when loading; continuation with the same settings gives the same course of evolution.
    pub fn save(&self) -> Vec<u8> {
        let wide = std::mem::size_of::<T>() > 4;
        let mut w = W::with_width(wide);
        w.buf.extend_from_slice(b"NPOP");
        if wide {
            w.u32(3);
            w.u8(8);
        } else {
            w.u32(2);
        }
        w.u64(self.generation);
        w.u64(self.rng.state());
        self.tracker.write(&mut w);
        w.u32(self.next_species);
        w.f32(self.threshold);
        w.len(self.individuals.len());
        for i in &self.individuals {
            i.genome.write(&mut w);
            w.f32(i.raw);
            w.f32(i.effective);
            w.bool(i.valid);
        }
        w.len(self.species.len());
        for s in &self.species {
            w.u32(s.id);
            s.representative.write(&mut w);
            w.f32(s.best);
            w.u32(s.stagnation);
            w.u32(s.age);
            w.len(s.members.len());
            s.members.iter().for_each(|m| w.u64(*m as u64));
        }
        match &self.champion {
            None => w.bool(false),
            Some(c) => {
                w.bool(true);
                c.genome.write(&mut w);
                w.f32(c.raw);
                w.f32(c.effective);
                w.u64(c.generation);
            }
        }
        w.buf
    }

    /// Restore the population. `cfg` must match in inputs, outputs and population size.
    pub fn load(bytes: &[u8], cfg: EvolutionConfig<T>) -> StructuralResult<Self> {
        cfg.validate()?;
        if bytes.len() < 8 || &bytes[..4] != b"NPOP" {
            return Err(err("this is not a population (no NPOP magic)"));
        }
        let mut r = R::new(&bytes[4..]);
        let version = r.u32()?;
        if version == 0 || version > 3 {
            return Err(err(format!(
                "population version {version} is not supported"
            )));
        }
        if version == 3 {
            r.wide = match r.u8()? {
                4 => false,
                8 => true,
                x => return Err(err(format!("value width {x} is not supported"))),
            };
        }
        let generation = r.u64()?;
        let rng = Rng64::from_state(r.u64()?);
        let tracker = InnovationTracker::read(&mut r)?;
        let next_species = r.u32()?;
        let threshold = r.f32()?;
        let mut individuals: Vec<Individual<T>> = Vec::new();
        for _ in 0..r.len(MAX_ITEMS)? {
            let genome = Genome::<T>::read(&mut r, version)?;
            individuals.push(Individual {
                genome,
                raw: r.f32()?,
                effective: r.f32()?,
                valid: r.bool()?,
            });
        }
        if individuals.len() != cfg.population {
            return Err(err(format!(
                "in the data of {} individuals, in the settings of {}",
                individuals.len(),
                cfg.population
            )));
        }
        let mut species: Vec<Species<T>> = Vec::new();
        for _ in 0..r.len(MAX_ITEMS)? {
            let id = r.u32()?;
            let representative = Genome::<T>::read(&mut r, version)?;
            let (best, stagnation, age) = (r.f32()?, r.u32()?, r.u32()?);
            let mut members = Vec::new();
            for _ in 0..r.len(MAX_ITEMS)? {
                let m = r.u64()? as usize;
                if m >= individuals.len() {
                    return Err(err("member of a species outside the population"));
                }
                members.push(m);
            }
            species.push(Species {
                id,
                representative,
                members,
                best,
                stagnation,
                age,
            });
        }
        let champion = if r.bool()? {
            let genome = Genome::<T>::read(&mut r, version)?;
            Some(Champion {
                genome,
                raw: r.f32()?,
                effective: r.f32()?,
                generation: r.u64()?,
            })
        } else {
            None
        };
        if !r.finished() {
            return Err(err("extra data after population"));
        }
        let (inputs, outputs) = individuals
            .first()
            .map(|i| {
                (
                    i.genome
                        .nodes
                        .iter()
                        .filter(|n| n.kind == NeuronRole::Input)
                        .count(),
                    i.genome
                        .nodes
                        .iter()
                        .filter(|n| n.kind == NeuronRole::Output)
                        .count(),
                )
            })
            .unwrap_or((0, 0));
        if (inputs, outputs) != (cfg.inputs, cfg.outputs) {
            return Err(err("inputs/outputs in the data do not match the settings"));
        }
        let mut individuals = individuals;
        let mut species = species;
        let mut champion = champion;
        if cfg.limits.enforce_dale {
            // A population preserved before Dale's principle is reduced to E/I.
            individuals.iter_mut().for_each(|i| {
                i.genome.normalize_dale();
            });
            species.iter_mut().for_each(|s| {
                s.representative.normalize_dale();
            });
            if let Some(c) = champion.as_mut() {
                c.genome.normalize_dale();
            }
        }
        Ok(Self {
            cfg,
            rng,
            tracker,
            individuals,
            species,
            next_species,
            threshold,
            generation,
            champion,
        })
    }
}

/// Network from the genome: topology according to settings limits, engine with their plasticity parameters.
pub fn build_network<T: Scalar>(
    g: &Genome<T>,
    cfg: &EvolutionConfig<T>,
) -> StructuralResult<DynamicNetwork<T>> {
    let topo: Topology<T> = g.to_topology(cfg.limits, cfg.param_limits)?;
    DynamicNetwork::new(topo, cfg.engine.clone())
}
