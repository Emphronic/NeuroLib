//! Integration tests of the evolution of architectures based on spike XOR.
//!
//! Task: two “bits” (bit 1 - the input is spiked with probability 0.9, bit 0 is silent) and a bias input;
//! two ways out, the answer is the one that spikes more often. Tests do NOT claim that evolution is better at any
//! task: it is this and precisely these grains that are checked, plus mechanics (determinism, checkpoint).
#![allow(clippy::field_reassign_with_default)]

use neo_snn::structural::evolution::*;
use neo_snn::structural::{
    DynamicNetwork, Metrics, Rng64, StructuralConfig, StructuralPlasticityController,
    load_checkpoint, save_checkpoint,
};

const STEPS: usize = 20;
const INPUTS: usize = 3;

fn pattern(a: bool, b: bool, seed: u64) -> Vec<[f32; INPUTS]> {
    let mut rng = Rng64::new(seed);
    (0..STEPS)
        .map(|_| {
            [
                (a && rng.chance(0.9)) as u8 as f32,
                (b && rng.chance(0.9)) as u8 as f32,
                1.0,
            ]
        })
        .collect()
}

/// `(accuracy, average normalized stock)`.
fn score(net: &mut DynamicNetwork, seed_base: u64, repeats: usize) -> (f32, f32) {
    let (mut correct, mut margin_sum, mut n) = (0usize, 0.0f32, 0usize);
    for rep in 0..repeats {
        for (k, (a, b)) in [(false, false), (false, true), (true, false), (true, true)]
            .into_iter()
            .enumerate()
        {
            let class = (a ^ b) as usize;
            net.begin_sample();
            let mut count = [0i32; 2];
            for x in pattern(a, b, seed_base + (rep * 4 + k) as u64) {
                if let Ok(o) = net.step(&x) {
                    for (j, s) in o.output_spikes.iter().enumerate().take(2) {
                        count[j] += *s as i32;
                    }
                }
            }
            let margin = (count[class] - count[1 - class]) as f32;
            correct += (margin > 0.0) as usize;
            margin_sum += (margin / 5.0).clamp(-1.0, 1.0);
            n += 1;
        }
    }
    (correct as f32 / n as f32, margin_sum / n as f32)
}

fn fitness(net: &mut DynamicNetwork) -> f32 {
    let (acc, margin) = score(net, 1_000, 3);
    acc + 0.1 * margin
}

fn config(seed: u64) -> EvolutionConfig {
    let mut c = EvolutionConfig::new(INPUTS, 2);
    c.population = 100;
    c.seed = seed;
    c.threads = 4;
    c.complexity = ComplexityPenalty {
        per_hidden: 0.01,
        per_conn: 0.002,
    };
    c
}

#[test]
fn xor_is_solved_by_growing_structure_and_generalizes() {
    // The grains are fixed: this is a regression test, not an assessment of reliability (proportion of solved runs
    // for grains, see the report and the example `neat_xor`: some starts get stuck at 3/4)
    for seed in [2u64, 5] {
        let mut pop = Population::new(config(seed)).unwrap();
        let start_best = {
            let r = pop.step(&fitness);
            r.best_raw
        };
        let solved = pop.run(&fitness, 60, Some(1.0), |_| {});
        assert!(
            solved.is_some(),
            "seed {seed}: not solved within 60 generations (start {start_best:.2}, champion {:.3})",
            pop.champion().unwrap().raw
        );
        let champ = pop.champion().unwrap().clone();
        assert!(
            champ.genome.hidden_count() >= 1,
            "seed {seed}: solution without hidden neurons"
        );
        assert!(
            champ.raw > start_best,
            "seed {seed}: better than the starting population"
        );
        // Champion's network on data that evolution has never seen
        let mut net = pop.champion_network().unwrap().unwrap();
        let (acc, _) = score(&mut net, 777_000, 25);
        assert!(acc >= 0.9, "seed {seed}: on new data accuracy {acc}");
        // All restrictions are met
        assert!(champ.genome.nodes.len() <= pop.config().limits.max_neurons);
        champ.genome.validate().unwrap();
    }
}

#[test]
fn without_structural_mutations_the_task_stays_unsolved() {
    // Control: the same weights and parameters, but the topology does not grow
    for seed in [3u64, 4] {
        let mut c = config(seed);
        c.mutation.add_node_prob = 0.0;
        c.mutation.add_conn_prob = 0.0;
        let mut pop = Population::new(c).unwrap();
        let solved = pop.run(&fitness, 40, Some(1.0), |_| {});
        assert!(
            solved.is_none(),
            "seed {seed}: solved without structure growth"
        );
        assert!(pop.champion().unwrap().genome.hidden_count() == 0);
        let mut net = pop.champion_network().unwrap().unwrap();
        let (acc, _) = score(&mut net, 777_000, 25);
        assert!(acc <= 0.8, "seed {seed}: control gave accuracy {acc}");
    }
}

#[test]
fn a_stopped_run_resumes_identically_and_threads_do_not_matter() {
    let mut a = Population::new(config(5)).unwrap();
    for _ in 0..3 {
        a.step(&fitness);
    }
    let bytes = a.save().unwrap();
    let mut one_thread = config(5);
    one_thread.threads = 1;
    let mut b = Population::load(&bytes, one_thread).unwrap();
    for g in 3..6 {
        let (ra, rb) = (a.step(&fitness), b.step(&fitness));
        assert_eq!(
            ra, rb,
            "generation {g}: the course of evolution has diverged"
        );
    }
    assert_eq!(a.champion(), b.champion());
}

#[test]
fn the_evolved_network_continues_under_structural_plasticity_and_checkpoints() {
    let mut pop = Population::new(config(3)).unwrap();
    pop.run(&fitness, 30, Some(1.0), |_| {});
    let mut net = pop.champion_network().unwrap().unwrap();
    let (n0, s0) = (
        net.topology().neuron_count(),
        net.topology().synapse_count(),
    );
    // The evolved network is a regular DynamicNetwork: it is controlled by a structural plasticity controller
    let mut sc = StructuralConfig::default();
    sc.topology = pop.config().limits;
    let mut ctl = StructuralPlasticityController::new(sc).unwrap();
    for _ in 0..4 {
        for (k, (a, b)) in [(false, false), (false, true), (true, false), (true, true)]
            .into_iter()
            .enumerate()
        {
            net.begin_sample();
            for x in pattern(a, b, 50 + k as u64) {
                net.step(&x).unwrap();
            }
        }
        let (acc, _) = score(&mut net.probe_copy(), 1_000, 1);
        ctl.adapt(
            &mut net,
            Metrics {
                task_loss: 1.0 - acc,
                reward: None,
                accuracy: Some(acc),
            },
            None,
        )
        .unwrap();
        net.topology().validate().unwrap();
    }
    let bytes = save_checkpoint(&net, &ctl, b"").unwrap();
    let ck = load_checkpoint(&bytes).unwrap();
    assert_eq!(ck.network.topology(), net.topology());
    assert!(net.topology().neuron_count() <= pop.config().limits.max_neurons && n0 > 0 && s0 > 0);
    // And vice versa: its structure can be returned to the genome
    let mut tracker = InnovationTracker::new(1000);
    let back = Genome::from_topology(net.topology(), &mut tracker);
    back.validate().unwrap();
}
