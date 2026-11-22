//! Architecture evolution tests.
#![allow(clippy::field_reassign_with_default)]

use super::operators::{Env, crossover, enforce_acyclic, fits, mutate};
use super::*;
use crate::structural::{
    AdaptiveLifParams, DynamicNetwork, NeuronKind, NeuronModel, NeuronPolarity, Rng64,
    TopologyLimits,
};

fn cfg(inputs: usize, outputs: usize) -> EvolutionConfig {
    let mut c = EvolutionConfig::new(inputs, outputs);
    c.population = 40;
    c.seed = 11;
    c
}

fn env(c: &EvolutionConfig) -> Env<'_> {
    Env {
        cfg: &c.mutation,
        limits: &c.limits,
        plimits: &c.param_limits,
    }
}

fn tracker_for(c: &EvolutionConfig) -> InnovationTracker {
    InnovationTracker::new((c.inputs + c.outputs) as u64)
}

/// A genome that has increased its structure through a series of mutations.
fn grown(
    c: &EvolutionConfig,
    tracker: &mut InnovationTracker,
    rng: &mut Rng64,
    rounds: usize,
) -> Genome {
    let mut g = Genome::bare(c.inputs, c.outputs, c.output_params);
    for i in 0..c.inputs {
        for o in 0..c.outputs {
            let (s, d) = (i as u64, (c.inputs + o) as u64);
            let innov = tracker.conn(s, d);
            g.insert_conn(ConnGene {
                innov,
                src: s,
                dst: d,
                weight: 0.5,
                delay: 0,
                enabled: true,
            });
        }
    }
    let mut m = c.clone();
    m.mutation.add_node_prob = 0.6;
    m.mutation.add_conn_prob = 0.8;
    for _ in 0..rounds {
        mutate(&mut g, &env(&m), tracker, rng);
    }
    g
}

#[test]
fn genome_and_topology_round_trip() {
    let c = cfg(3, 2);
    let mut tr = tracker_for(&c);
    let g = grown(&c, &mut tr, &mut Rng64::new(1), 30);
    g.validate().unwrap();
    assert!(
        g.hidden_count() > 0,
        "in 30 rounds at least one neuron has grown"
    );
    let t = g.to_topology(c.limits, c.param_limits).unwrap();
    t.validate().unwrap();
    assert_eq!(t.neuron_count(), g.nodes.len());
    assert_eq!(t.synapse_count(), g.enabled_count());
    assert_eq!((t.inputs().len(), t.outputs().len()), (3, 2));
    // Topology → genome → topology gives the same structure
    let g2 = Genome::from_topology(&t, &mut tracker_for(&c));
    g2.validate().unwrap();
    let t2 = g2.to_topology(c.limits, c.param_limits).unwrap();
    assert_eq!(
        (t2.neuron_count(), t2.synapse_count()),
        (t.neuron_count(), t.synapse_count())
    );
    let weights = |t: &crate::structural::Topology| {
        let mut w: Vec<_> = t.synapses().map(|s| (s.weight * 1e4) as i64).collect();
        w.sort();
        w
    };
    assert_eq!(weights(&t), weights(&t2));
}

#[test]
fn disabled_synapses_become_disabled_genes() {
    let c = cfg(2, 1);
    let mut tr = tracker_for(&c);
    let mut g = Genome::bare(2, 1, c.output_params);
    for i in 0..2u64 {
        let innov = tr.conn(i, 2);
        g.insert_conn(ConnGene {
            innov,
            src: i,
            dst: 2,
            weight: 0.4,
            delay: 0,
            enabled: true,
        });
    }
    let mut t = g.to_topology(c.limits, c.param_limits).unwrap();
    let sid = t.synapses().next().unwrap().id;
    t.disable_synapse(sid, 0).unwrap();
    let g2 = Genome::from_topology(&t, &mut tracker_for(&c));
    assert_eq!((g2.conns.len(), g2.enabled_count()), (2, 1));
}

#[test]
fn serialization_round_trips_and_rejects_garbage() {
    let c = cfg(2, 2);
    let g = grown(&c, &mut tracker_for(&c), &mut Rng64::new(5), 25);
    let bytes = g.to_bytes().unwrap();
    assert_eq!(Genome::<f32>::from_bytes(&bytes).unwrap(), g);
    assert!(
        Genome::<f32>::from_bytes(&bytes[..bytes.len() - 1]).is_err(),
        "trimmed data"
    );
    assert!(Genome::<f32>::from_bytes(b"XXXX\x01\0\0\0").is_err());
    let mut bad = bytes.clone();
    bad[4] = 9;
    assert!(Genome::<f32>::from_bytes(&bad).is_err(), "unknown version");
    let mut extra = bytes.clone();
    extra.push(0);
    assert!(Genome::<f32>::from_bytes(&extra).is_err(), "extra data");
    for n in 0..bytes.len().min(300) {
        assert!(
            Genome::<f32>::from_bytes(&bytes[..n]).is_err(),
            "prefix {n} byte accepted"
        );
    }
}

#[test]
fn distance_is_zero_for_equal_symmetric_and_grows_with_difference() {
    let c = cfg(3, 2);
    let (mut tr, mut rng) = (tracker_for(&c), Rng64::new(2));
    let a = grown(&c, &mut tr, &mut rng, 10);
    let b = grown(&c, &mut tr, &mut rng, 40);
    let d = |x: &Genome, y: &Genome| x.distance(y, 1.0, 1.0, 0.4);
    assert_eq!(d(&a, &a), 0.0);
    assert!((d(&a, &b) - d(&b, &a)).abs() < 1e-6);
    assert!(d(&a, &b) > 0.0);
    let mut shifted = a.clone();
    shifted.conns.iter_mut().for_each(|c| c.weight += 0.5);
    assert!(
        (d(&a, &shifted) - 0.4 * 0.5).abs() < 1e-5,
        "weights only: c3·W̄"
    );
}

#[test]
fn mutations_always_respect_limits_and_stay_valid() {
    for allow_recurrent in [true, false] {
        let mut c = cfg(3, 2);
        c.limits = TopologyLimits {
            max_neurons: 12,
            max_synapses: 20,
            max_in_degree: 5,
            max_out_degree: 6,
            allow_self_loops: false,
            allow_recurrent,
            allow_duplicates: false,
            enforce_dale: true,
        };
        c.mutation.add_node_prob = 0.5;
        c.mutation.add_conn_prob = 0.9;
        c.mutation.toggle_prob = 0.1;
        c.mutation.param_mutate_prob = 0.5;
        let (mut tr, mut rng) = (tracker_for(&c), Rng64::new(9));
        let mut g = Genome::bare(3, 2, c.output_params);
        for _ in 0..500 {
            mutate(&mut g, &env(&c), &mut tr, &mut rng);
            g.validate().unwrap();
            assert!(
                fits(&g, &c.limits),
                "limits violated: {} neurons, {} connections",
                g.nodes.len(),
                g.enabled_count()
            );
            let t = g
                .to_topology(c.limits, c.param_limits)
                .expect("the genome is being built");
            t.validate().unwrap();
            for n in g.nodes.iter().filter(|n| n.kind != NeuronKind::Input) {
                if let Some(beta) = n.model.beta() {
                    assert!((c.param_limits.beta_min..=c.param_limits.beta_max).contains(&beta));
                    assert!(
                        (c.param_limits.threshold_min..=c.param_limits.threshold_max)
                            .contains(&n.model.threshold())
                    );
                }
            }
            assert!(
                g.conns
                    .iter()
                    .all(|c0| c0.weight.abs() <= c.param_limits.weight_abs_max)
            );
        }
        assert!(g.hidden_count() > 0 && g.nodes.len() <= 12);
    }
}

#[test]
fn add_node_splits_a_connection_nearly_preserving_behaviour() {
    let mut c = cfg(1, 1);
    c.mutation.add_node_prob = 1.0;
    c.mutation.add_conn_prob = 0.0;
    c.mutation.weight_mutate_prob = 0.0;
    c.mutation.toggle_prob = 0.0;
    c.mutation.param_mutate_prob = 0.0;
    let mut tr = tracker_for(&c);
    let mut g = Genome::bare(1, 1, c.output_params);
    let innov = tr.conn(0, 1);
    g.insert_conn(ConnGene {
        innov,
        src: 0,
        dst: 1,
        weight: 1.2,
        delay: 0,
        enabled: true,
    });
    let s = mutate(&mut g, &env(&c), &mut tr, &mut Rng64::new(1));
    assert_eq!((s.nodes_added, s.conns_added), (1, 2));
    assert!(
        !g.conn(innov).unwrap().enabled,
        "the original link is disabled but remains in the genome"
    );
    let relay = g
        .nodes
        .iter()
        .find(|n| n.kind == NeuronKind::Hidden)
        .unwrap()
        .innov;
    let (a, b) = (
        g.conns.iter().find(|c| c.dst == relay).unwrap(),
        g.conns.iter().find(|c| c.src == relay).unwrap(),
    );
    assert_eq!((a.src, b.dst, b.weight), (0, 1, 1.2));
    // The input is spiked at each step → the output receives the same current, one step later
    let spikes = |g: &Genome| {
        let mut net = build_network(g, &c).unwrap();
        net.begin_sample();
        (0..12)
            .map(|_| net.step(&[1.0]).unwrap().output_spikes[0] as u32)
            .sum::<u32>()
    };
    let mut direct = Genome::bare(1, 1, c.output_params);
    direct.insert_conn(ConnGene {
        innov,
        src: 0,
        dst: 1,
        weight: 1.2,
        delay: 0,
        enabled: true,
    });
    let (with, without) = (spikes(&g), spikes(&direct));
    assert!(
        without >= 5,
        "control: feedforward causes spikes ({without})"
    );
    assert!(
        with.abs_diff(without) <= 1,
        "with relay {with}, without {without}"
    );
}

#[test]
fn the_same_novelty_gets_the_same_marker() {
    let c = cfg(2, 1);
    let mut tr = tracker_for(&c);
    let (a, b) = (tr.conn(0, 2), tr.conn(0, 2));
    assert_eq!(a, b);
    let s1 = tr.split(a, 0, 2);
    assert_eq!(
        s1,
        tr.split(a, 0, 2),
        "division of the same connection - the same markers"
    );
    assert_ne!(tr.conn(1, 2), a);
    assert!(tr.issued() > s1.0);
}

#[test]
fn crossover_aligns_by_markers_and_inherits_extra_genes_from_the_fitter() {
    let c = cfg(2, 1);
    let mut tr = tracker_for(&c);
    let mut base = Genome::bare(2, 1, c.output_params);
    for i in 0..2u64 {
        let innov = tr.conn(i, 2);
        base.insert_conn(ConnGene {
            innov,
            src: i,
            dst: 2,
            weight: 0.1,
            delay: 0,
            enabled: true,
        });
    }
    let mut rng = Rng64::new(3);
    let mut big = base.clone();
    let mut m = c.clone();
    m.mutation.add_node_prob = 1.0;
    m.mutation.weight_mutate_prob = 0.0;
    mutate(&mut big, &env(&m), &mut tr, &mut rng);
    assert_eq!(big.hidden_count(), 1);
    let hidden = big
        .nodes
        .iter()
        .find(|n| n.kind == NeuronKind::Hidden)
        .unwrap()
        .innov;
    let child = crossover(&big, 2.0, &base, 1.0, &c.limits, &mut rng);
    child.validate().unwrap();
    assert!(
        child.node(hidden).is_some(),
        "an extra neuron from a more adapted one"
    );
    let child = crossover(&big, 1.0, &base, 2.0, &c.limits, &mut rng);
    child.validate().unwrap();
    assert!(
        child.node(hidden).is_none(),
        "other people's extra genes are not taken"
    );
    assert_eq!(child.conns.len(), base.conns.len());
    let child = crossover(&big, 1.0, &base, 1.0, &c.limits, &mut rng);
    assert!(
        child.node(hidden).is_none(),
        "in case of equality, the smaller genome wins"
    );
    build_network(&child, &c).unwrap();
}

#[test]
fn crossover_never_creates_a_cycle_when_recurrence_is_forbidden() {
    let mut c = cfg(1, 2);
    c.limits.allow_recurrent = false;
    let mk = |forward: bool| {
        let mut g = Genome::bare(1, 2, c.output_params);
        g.insert_node(NodeGene {
            innov: 10,
            kind: NeuronKind::Hidden,
            polarity: NeuronPolarity::Excitatory,
            model: NeuronModel::Lif(c.output_params),
        });
        g.insert_node(NodeGene {
            innov: 11,
            kind: NeuronKind::Hidden,
            polarity: NeuronPolarity::Excitatory,
            model: NeuronModel::Lif(c.output_params),
        });
        let (s, d) = if forward { (10, 11) } else { (11, 10) };
        g.insert_conn(ConnGene {
            innov: if forward { 20 } else { 21 },
            src: s,
            dst: d,
            weight: 0.5,
            delay: 0,
            enabled: true,
        });
        g
    };
    let (a, b) = (mk(true), mk(false));
    let mut union = a.clone();
    union.insert_conn(b.conns[0].clone());
    enforce_acyclic(&mut union);
    assert_eq!(union.enabled_count(), 1, "the cycle is broken");
    let mut rng = Rng64::new(1);
    for _ in 0..50 {
        let child = crossover(&a, 1.0, &b, 1.0, &c.limits, &mut rng);
        assert!(child.to_topology(c.limits, c.param_limits).is_ok());
    }
}

#[test]
fn config_validation_catches_nonsense() {
    assert!(cfg(2, 1).validate().is_ok());
    let mut c = cfg(2, 1);
    c.population = 1;
    assert!(c.validate().is_err());
    let mut c = cfg(0, 1);
    c.population = 10;
    assert!(c.validate().is_err());
    let mut c = cfg(2, 1);
    c.mutation.add_conn_prob = 1.5;
    assert!(c.validate().is_err());
    let mut c = cfg(2, 1);
    c.limits.max_neurons = 2;
    assert!(Population::new(c).is_err());
}

/// Problem for mechanics only: average weight of connections.
fn mean_weight(net: &mut DynamicNetwork) -> f32 {
    let t = net.topology();
    t.synapses().map(|s| s.weight).sum::<f32>() / t.synapse_count().max(1) as f32
}

#[test]
fn evolution_is_deterministic_and_independent_of_thread_count() {
    let run = |threads: usize| {
        let mut c = cfg(3, 2);
        c.threads = threads;
        let mut p = Population::new(c).unwrap();
        let reports: Vec<_> = (0..8).map(|_| p.step(&mean_weight)).collect();
        (reports, p.champion().unwrap().genome.clone())
    };
    let (r1, g1) = run(1);
    let (r1b, g1b) = run(1);
    let (r4, g4) = run(4);
    assert_eq!((&r1, &g1), (&r1b, &g1b));
    assert_eq!(r1, r4, "the number of threads does not change the result");
    assert_eq!(g1, g4);
}

#[test]
fn population_keeps_its_size_and_every_individual_is_valid() {
    let mut c = cfg(3, 2);
    c.limits.max_neurons = 10;
    c.limits.max_synapses = 16;
    c.mutation.add_node_prob = 0.4;
    c.mutation.add_conn_prob = 0.6;
    let n = c.population;
    let mut p = Population::new(c.clone()).unwrap();
    for _ in 0..25 {
        let r = p.step(&mean_weight);
        assert_eq!(r.invalid, 0);
        assert_eq!(p.individuals().len(), n);
        assert_eq!(r.species_sizes.iter().sum::<usize>(), n);
        for ind in p.individuals() {
            ind.genome.validate().unwrap();
            assert!(fits(&ind.genome, &c.limits));
            build_network(&ind.genome, &c).unwrap();
        }
    }
    assert!(p.champion().is_some());
}

#[test]
fn the_champion_never_gets_worse() {
    let mut p = Population::new(cfg(3, 2)).unwrap();
    let mut best = f32::NEG_INFINITY;
    for _ in 0..15 {
        let r = p.step(&mean_weight);
        assert!(r.champion_raw >= best);
        best = r.champion_raw;
        assert!(r.champion_raw >= r.best_raw);
    }
}

#[test]
fn invalid_genomes_get_the_floor_fitness_instead_of_breaking_the_run() {
    let mut p = Population::new(cfg(2, 1)).unwrap();
    let r = p.step(&|_: &mut DynamicNetwork| f32::NAN);
    assert_eq!(r.invalid, p.config().population);
    assert!(p.champion().is_none());
    let r = p.step(&mean_weight);
    assert_eq!(r.invalid, 0);
}

#[test]
fn complexity_penalty_keeps_networks_small_when_structure_does_not_help() {
    // Fitness does not depend on the structure: without a penalty the network grows by drift, with a penalty it does not.
    let flat = |_: &mut DynamicNetwork| 1.0f32;
    let hidden_after = |penalty: f32| {
        let mut c = cfg(3, 2);
        c.complexity = ComplexityPenalty {
            per_hidden: penalty,
            per_conn: penalty / 4.0,
        };
        c.mutation.add_node_prob = 0.3;
        c.mutation.add_conn_prob = 0.5;
        let mut p = Population::new(c).unwrap();
        let mut last = 0.0;
        for _ in 0..30 {
            last = p.step(&flat).mean_hidden;
        }
        last
    };
    let (free, taxed) = (hidden_after(0.0), hidden_after(0.2));
    assert!(
        taxed < free * 0.6,
        "without penalty {free}, with penalty {taxed}"
    );
}

#[test]
fn run_stops_at_the_target() {
    let mut p = Population::new(cfg(2, 1)).unwrap();
    let mut seen = 0;
    let hit = p.run(&|_: &mut DynamicNetwork| 5.0, 50, Some(4.0), |_| seen += 1);
    assert_eq!((hit, seen), (Some(0), 1));
    let none = p.run(&|_: &mut DynamicNetwork| 0.0, 3, Some(1.0), |_| seen += 1);
    assert_eq!((none, seen), (None, 4));
}

#[test]
fn stagnant_species_die_but_the_best_ones_stay() {
    let mut c = cfg(3, 2);
    c.selection.stagnation_generations = 3;
    c.selection.keep_species = 1;
    c.speciation.threshold = 0.5;
    c.speciation.target_species = None;
    c.mutation.add_node_prob = 0.5;
    let mut p = Population::new(c).unwrap();
    let flat = |_: &mut DynamicNetwork| 1.0f32;
    let mut max_age = 0;
    for _ in 0..25 {
        p.step(&flat);
        max_age = max_age.max(p.species().iter().map(|s| s.age).max().unwrap_or(0));
        assert!(!p.species().is_empty());
    }
    assert!(max_age >= 3, "species have time to live");
}

#[test]
fn save_and_load_continue_identically() {
    let c = cfg(3, 2);
    let mut a = Population::new(c.clone()).unwrap();
    for _ in 0..6 {
        a.step(&mean_weight);
    }
    let bytes = a.save().unwrap();
    let mut b = Population::load(&bytes, c.clone()).unwrap();
    assert_eq!(b.generation(), a.generation());
    assert_eq!(b.champion(), a.champion());
    for _ in 0..6 {
        assert_eq!(a.step(&mean_weight), b.step(&mean_weight));
    }
    assert_eq!(a.save().unwrap(), b.save().unwrap());
    let mut other = c.clone();
    other.population += 1;
    assert!(Population::load(&bytes, other).is_err());
    let mut other = c.clone();
    other.inputs += 1;
    assert!(Population::load(&bytes, other).is_err());
    assert!(Population::load(&bytes[..bytes.len() / 2], c.clone()).is_err());
    assert!(Population::load(b"NPOP", c).is_err());
}

#[test]
fn a_trained_network_can_seed_a_population() {
    let c = cfg(2, 1);
    let mut tr = tracker_for(&c);
    let mut g = Genome::bare(2, 1, c.output_params);
    for i in 0..2u64 {
        let innov = tr.conn(i, 2);
        g.insert_conn(ConnGene {
            innov,
            src: i,
            dst: 2,
            weight: 0.7,
            delay: 0,
            enabled: true,
        });
    }
    let topo = g.to_topology(c.limits, c.param_limits).unwrap();
    let mut tr2 = tracker_for(&c);
    let template = Genome::from_topology(&topo, &mut tr2);
    let p = Population::seeded(c.clone(), template.clone(), tr2).unwrap();
    assert_eq!(
        p.individuals()[0].genome,
        template,
        "the first individual is the template itself"
    );
    assert!(p.individuals().iter().all(|i| i.genome.validate().is_ok()));
    let wrong = Genome::bare(3, 1, c.output_params);
    assert!(Population::seeded(c, wrong, InnovationTracker::new(10)).is_err());
}

// ───────────────────────── heterogeneous genes ─────────────────────────

fn write_genome_v1(w: &mut super::codec::W, g: &Genome) {
    w.len(g.nodes.len());
    for n in &g.nodes {
        w.u64(n.innov);
        w.u8(match n.kind {
            NeuronKind::Input => 0,
            NeuronKind::Hidden => 1,
            NeuronKind::Output => 2,
        });
        let lif = match &n.model {
            NeuronModel::Lif(p) => *p,
            _ => c_lif_default(),
        };
        w.lif(&lif);
    }
    w.len(g.conns.len());
    for c0 in &g.conns {
        w.u64(c0.innov);
        w.u64(c0.src);
        w.u64(c0.dst);
        w.f32(c0.weight);
        w.u16(c0.delay);
        w.bool(c0.enabled);
    }
}

fn genome_v1_bytes(g: &Genome) -> Vec<u8> {
    use super::codec::W;
    let mut w = W::default();
    w.buf.extend_from_slice(b"NGEN");
    w.u32(1);
    write_genome_v1(&mut w, g);
    w.buf
}

fn population_v1_bytes(p: &Population) -> Vec<u8> {
    use super::codec::W;
    let mut w = W::default();
    w.buf.extend_from_slice(b"NPOP");
    w.u32(1);
    w.u64(p.generation());
    w.u64(1); // the state of the PRNG is not important for migration
    p.tracker().write(&mut w);
    w.u32(0); // next_species
    w.f32(0.5); // speciation threshold
    w.len(p.individuals().len());
    for i in p.individuals() {
        write_genome_v1(&mut w, &i.genome);
        w.f32(i.raw);
        w.f32(i.effective);
        w.bool(i.valid);
    }
    w.len(p.species().len());
    for s in p.species() {
        w.u32(s.id);
        write_genome_v1(&mut w, &s.representative);
        w.f32(s.best);
        w.u32(s.stagnation);
        w.u32(s.age);
        w.len(s.members.len());
        s.members.iter().for_each(|m| w.u64(*m as u64));
    }
    match p.champion() {
        None => w.bool(false),
        Some(ch) => {
            w.bool(true);
            write_genome_v1(&mut w, &ch.genome);
            w.f32(ch.raw);
            w.f32(ch.effective);
            w.u64(ch.generation);
        }
    }
    w.buf
}

fn c_lif_default() -> crate::structural::LifParams {
    crate::structural::LifParams::default()
}

#[test]
fn population_v1_migrates_genomes_to_lif_excitatory() {
    let c = cfg(2, 1);
    let mut p = Population::new(c.clone()).unwrap();
    p.step(&mean_weight);
    let bytes = population_v1_bytes(&p);
    let loaded = Population::load(&bytes, c).unwrap();
    assert_eq!(loaded.generation(), p.generation());
    assert_eq!(loaded.individuals().len(), p.individuals().len());
    for i in loaded.individuals() {
        assert!(
            i.genome
                .nodes
                .iter()
                .all(|n| n.model.is_lif() && n.polarity == NeuronPolarity::Excitatory)
        );
    }
    for s in loaded.species() {
        assert!(
            s.representative
                .nodes
                .iter()
                .all(|n| n.model.is_lif() && n.polarity == NeuronPolarity::Excitatory)
        );
    }
    if let Some(ch) = loaded.champion() {
        assert!(
            ch.genome
                .nodes
                .iter()
                .all(|n| n.model.is_lif() && n.polarity == NeuronPolarity::Excitatory)
        );
    }
}

#[test]
fn genome_v1_migrates_to_lif_excitatory() {
    let c = cfg(1, 1);
    let mut g = Genome::bare(1, 1, c.output_params);
    g.insert_conn(ConnGene {
        innov: 2,
        src: 0,
        dst: 1,
        weight: 0.5,
        delay: 0,
        enabled: true,
    });
    let old = genome_v1_bytes(&g);
    let migrated = Genome::<f32>::from_bytes(&old).unwrap();
    assert_eq!(migrated.nodes.len(), 2);
    assert!(
        migrated
            .nodes
            .iter()
            .all(|n| n.polarity == NeuronPolarity::Excitatory && n.model.is_lif())
    );
    assert_eq!(migrated.conns.len(), 1);
}

#[test]
fn genome_v2_roundtrips_model_and_polarity() {
    let c = cfg(1, 1);
    let mut g = Genome::bare(1, 1, c.output_params);
    g.nodes[1].model = NeuronModel::AdaptiveLif(AdaptiveLifParams::slow_adaptation());
    g.nodes[1].polarity = NeuronPolarity::Inhibitory;
    g.insert_conn(ConnGene {
        innov: 2,
        src: 0,
        dst: 1,
        weight: 0.5,
        delay: 0,
        enabled: true,
    });
    let bytes = g.to_bytes().unwrap();
    assert_eq!(&bytes[4..8], &2u32.to_le_bytes());
    let back = Genome::<f32>::from_bytes(&bytes).unwrap();
    assert_eq!(back, g);
    let t = back.to_topology(c.limits, c.param_limits).unwrap();
    let o = t
        .neurons()
        .find(|n| n.kind() == NeuronKind::Output)
        .unwrap();
    assert!(matches!(o.model, NeuronModel::AdaptiveLif(_)));
    assert_eq!(o.polarity, NeuronPolarity::Inhibitory);
}

#[test]
fn model_mutation_learns_neuron_types() {
    let mut c = cfg(1, 1);
    c.mutation.weight_mutate_prob = 0.0;
    c.mutation.add_conn_prob = 0.0;
    c.mutation.add_node_prob = 0.0;
    c.mutation.toggle_prob = 0.0;
    c.mutation.param_mutate_prob = 0.0;
    c.mutation.model_mutate_prob = 1.0;
    c.mutation.model_candidates = vec![
        NeuronModel::AdaptiveLif(AdaptiveLifParams::slow_adaptation()),
        NeuronModel::Izhikevich(crate::structural::IzhikevichParams::fast_spiking()),
    ];
    c.mutation.polarity_mutate_prob = 1.0;
    let mut tr = tracker_for(&c);
    let mut g = Genome::bare(1, 1, c.output_params);
    let summary = mutate(&mut g, &env(&c), &mut tr, &mut Rng64::new(7));
    assert!(
        summary.models_changed >= 1,
        "the model of the output neuron has changed"
    );
    let output = g
        .nodes
        .iter()
        .find(|n| n.kind == NeuronKind::Output)
        .unwrap();
    assert!(!output.model.is_lif());
    g.validate().unwrap();

    // The polarity is chosen randomly; on a series of genomes the change is required
    // happen at least once.
    let mut polarity_changed = summary.polarities_changed > 0;
    for seed in 0..32 {
        let mut g = Genome::bare(1, 1, c.output_params);
        let s = mutate(&mut g, &env(&c), &mut tr, &mut Rng64::new(seed));
        polarity_changed |= s.polarities_changed > 0;
    }
    assert!(polarity_changed, "polarity mutation is triggered");
}

#[test]
fn split_model_creates_specialized_alif_hidden_neuron() {
    let mut c = cfg(1, 1);
    c.mutation.add_node_prob = 1.0;
    c.mutation.add_conn_prob = 0.0;
    c.mutation.weight_mutate_prob = 0.0;
    c.mutation.toggle_prob = 0.0;
    c.mutation.param_mutate_prob = 0.0;
    c.mutation.split_model = Some(NeuronModel::AdaptiveLif(
        AdaptiveLifParams::slow_adaptation(),
    ));
    let mut tr = tracker_for(&c);
    let mut g = Genome::bare(1, 1, c.output_params);
    let innov = tr.conn(0, 1);
    g.insert_conn(ConnGene {
        innov,
        src: 0,
        dst: 1,
        weight: 1.0,
        delay: 0,
        enabled: true,
    });
    mutate(&mut g, &env(&c), &mut tr, &mut Rng64::new(1));
    let hidden = g
        .nodes
        .iter()
        .find(|n| n.kind == NeuronKind::Hidden)
        .expect("neuron created");
    assert!(matches!(hidden.model, NeuronModel::AdaptiveLif(_)));
    g.validate().unwrap();
    let t = g.to_topology(c.limits, c.param_limits).unwrap();
    t.validate().unwrap();
}

#[test]
fn evolution_never_violates_dale_even_when_polarity_mutates() {
    let mut c = cfg(3, 2);
    c.mutation.polarity_mutate_prob = 0.5;
    c.mutation.add_node_prob = 0.5;
    c.mutation.add_conn_prob = 0.8;
    assert!(c.limits.enforce_dale);
    let mut tracker = tracker_for(&c);
    let mut rng = Rng64::new(5);
    let mut g = Genome::bare(c.inputs, c.outputs, c.output_params);
    let mut inhibitory_seen = false;
    for _ in 0..60 {
        mutate(&mut g, &env(&c), &mut tracker, &mut rng);
        // The genome has already been normalized: repeated normalization does not change anything.
        assert_eq!(g.clone().normalize_dale(), 0);
        let t = g.to_topology(c.limits, c.param_limits).unwrap();
        t.validate().unwrap();
        inhibitory_seen |= t.ei_stats().inhibitory > 0;
    }
    assert!(
        inhibitory_seen,
        "polarity mutation should produce inhibitory neurons"
    );
    // Direct damage to a character in the genome is corrected by normalization and does not reach the phenotype.
    if let Some(conn) = g.conns.first_mut() {
        conn.weight = -conn.weight - 0.1;
    }
    g.normalize_dale();
    assert!(
        g.to_topology(c.limits, c.param_limits)
            .unwrap()
            .validate()
            .is_ok()
    );
}

#[test]
fn f64_genome_roundtrips_without_precision_loss() {
    use super::super::model::LifParams;
    use super::super::topology::ParamLimits;
    let mut g: Genome<f64> = Genome::bare(2, 1, LifParams::<f64>::default());
    g.insert_conn(ConnGene {
        innov: 10,
        src: 0,
        dst: 2,
        weight: 0.123_456_789_012_345_68,
        delay: 1,
        enabled: true,
    });
    let bytes = g.to_bytes().unwrap();
    let back = Genome::<f64>::from_bytes(&bytes).unwrap();
    assert_eq!(back, g);
    assert_eq!(back.conns[0].weight, 0.123_456_789_012_345_68_f64);
    // f64 → f32: readable, but with loss of precision
    let narrow = Genome::<f32>::from_bytes(&bytes).unwrap();
    assert_eq!(narrow.conns[0].weight, 0.123_456_789_012_345_68_f64 as f32);
    // f32 -> f64: version 2 read and expanded
    let g32: Genome<f32> = Genome::bare(2, 1, LifParams::<f32>::default());
    let widened = Genome::<f64>::from_bytes(&g32.to_bytes().unwrap()).unwrap();
    assert_eq!(widened.nodes.len(), 3);
    // Distance and mutation work on f64
    assert_eq!(g.distance(&g, 1.0, 1.0, 0.4), 0.0);
    let cfg = MutationConfig::<f64>::default();
    let (limits, pl) = (TopologyLimits::default(), ParamLimits::<f64>::default());
    let env = Env {
        cfg: &cfg,
        limits: &limits,
        plimits: &pl,
    };
    let mut tracker = InnovationTracker::new(100);
    let mut rng = Rng64::new(3);
    for _ in 0..50 {
        mutate(&mut g, &env, &mut tracker, &mut rng);
        g.validate().unwrap();
    }
}

#[test]
fn population_evolves_saves_and_loads_on_f64() {
    let mk = || {
        let mut c = EvolutionConfig::<f64>::new(3, 2);
        c.population = 30;
        c.seed = 5;
        c
    };
    let fit = |net: &mut DynamicNetwork<f64>| -> f32 {
        let t = net.topology();
        (t.synapses().map(|s| s.weight).sum::<f64>() / t.synapse_count().max(1) as f64) as f32
    };
    let mut a = Population::<f64>::new(mk()).unwrap();
    for _ in 0..6 {
        a.step(&fit);
    }
    for i in a.individuals() {
        i.genome.validate().unwrap();
    }
    let bytes = a.save().unwrap();
    let mut b = Population::<f64>::load(&bytes, mk()).unwrap();
    let ra = a.step(&fit);
    let rb = b.step(&fit);
    assert_eq!(ra, rb, "after loading the evolution is the same");
    assert_eq!(a.champion().unwrap().genome, b.champion().unwrap().genome);
    // champion → f64 network
    let net = a.champion_network().unwrap().unwrap();
    net.topology().validate().unwrap();
}
