//! Evolution of the spike network architecture on the XOR problem (NEAT-like, `structural::evolution`).
//!
//! Input: two “bits” (bit 1 - the input is spliced with a probability of 0.9 per step, bit 0 - silent) and
//! constant bias input. Output: two neurons, the answer is the one that spikes more often
//! (the first one is XOR = 0, the second one is XOR = 1). Networks start minimal (only inputs → outputs)
//! and grow by mutations. Control (`NEAT_CONTROL=1`, height prohibited: only weights and parameters)
//! rests on 3 correct answers out of 4, with growth the problem is solved; see the output from both runs.
//!
//! Launch: `cargo run --release --example neat_xor -p neo-snn`
#![allow(clippy::field_reassign_with_default)]

use neo_snn::structural::evolution::*;
use neo_snn::structural::{DynamicNetwork, Rng64};

const STEPS: usize = 20;
const INPUTS: usize = 3; // a, b, bias

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

/// `(accuracy, average normalized stock)` on `repeats` repetitions of each of the four samples.
pub fn score(net: &mut DynamicNetwork, seed_base: u64, repeats: usize) -> (f32, f32) {
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

fn main() {
    let mut cfg = EvolutionConfig::new(INPUTS, 2);
    let env_num = |k: &str, d: u64| {
        std::env::var(k)
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(d)
    };
    cfg.population = env_num("NEAT_POP", 150) as usize;
    cfg.seed = env_num("NEAT_SEED", 3);
    cfg.threads = 4;
    cfg.complexity = ComplexityPenalty {
        per_hidden: 0.01,
        per_conn: 0.002,
    };
    // Control: NEAT_CONTROL=1 disables growth (weights and parameters only) - to see what the structure gives
    if std::env::var_os("NEAT_CONTROL").is_some() {
        cfg.mutation.add_node_prob = 0.0;
        cfg.mutation.add_conn_prob = 0.0;
    }
    let mut pop = Population::new(cfg).expect("settings");

    // Fitness: Accuracy + small stock reward (training seeds)
    let fitness = |net: &mut DynamicNetwork| {
        let (acc, margin) = score(net, 1_000, 3);
        acc + 0.1 * margin
    };
    let solved = pop.run(&fitness, 200, Some(1.05), |r| {
        if r.generation % 5 == 0 || r.best_raw >= 1.05 {
            println!(
                "generation {:3}: best {:.3} average {:.3} | the best: hidden {}, connections {} | species {} (threshold {:.1})",
                r.generation, r.best_raw, r.mean_raw, r.best_hidden, r.best_conns, r.species_sizes.len(), r.threshold
            );
        }
    });
    let champion = pop.champion().expect("champion");
    println!(
        "\nchampion: generation {}, fitness {:.3}, hidden {}, connections {}",
        champion.generation,
        champion.raw,
        champion.genome.hidden_count(),
        champion.genome.enabled_count()
    );
    match solved {
        Some(g) => println!("goal achieved in generation {g}"),
        None => println!("goal not achieved in 200 generations"),
    }
    // Testing on seeds that evolution has never seen
    let mut net = pop.champion_network().unwrap().expect("champion network");
    let (acc, margin) = score(&mut net, 777_000, 25);
    println!("on new data (100 runs): accuracy {acc:.2}, margin {margin:.2}");
    for c in champion.genome.conns.iter().filter(|c| c.enabled) {
        println!(
            "  connection {} → {}: weight {:+.2}",
            c.src, c.dst, c.weight
        );
    }
}
