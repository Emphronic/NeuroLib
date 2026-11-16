//! Demonstration of the evolution of architectures through a telemetry channel: the stream sends [`EvolutionReport`] to
//! every generation; GUI (Evolution tab) draws fitness curves, complexity of the best
//! networks and types. The problem is XOR on a spike network, as in `examples/neat_xor.rs`.
//!
//! Runs on the CPU engine [`crate::structural`], without Burn and GPU.

use std::time::Duration;

use crate::structural::evolution::{ComplexityPenalty, EvolutionConfig, Population};
use crate::structural::{DynamicNetwork, Rng64};

use super::channel::TrainTx;
use super::evolution::EvolutionReport;

const STEPS: usize = 20;
const INPUTS: usize = 3;

/// Demonstration options.
#[derive(Debug, Clone)]
pub struct DemoConfig {
    /// Maximum generations.
    pub generations: u64,
    /// Population size.
    pub population: usize,
    /// Grain.
    pub seed: u64,
    /// Evaluation streams.
    pub threads: usize,
    /// Pause after generation (for GUI; in tests - zero).
    pub pace: Duration,
    /// Fitness goal: Once reached, the flow ends.
    pub target: f32,
}

impl Default for DemoConfig {
    fn default() -> Self {
        Self {
            generations: 120,
            population: 100,
            seed: 2,
            threads: 4,
            pace: Duration::from_millis(250),
            target: 1.05,
        }
    }
}

/// What the demonstration returned.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct DemoSummary {
    /// How many generations have passed?
    pub generations_done: u64,
    /// Better fitness.
    pub best: f32,
    /// Has the goal been achieved?
    pub solved: bool,
    /// Stopped from GUI.
    pub stopped: bool,
}

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

fn fitness(net: &mut DynamicNetwork) -> f32 {
    let (mut correct, mut margin_sum, mut n) = (0usize, 0.0f32, 0usize);
    for rep in 0..3usize {
        for (k, (a, b)) in [(false, false), (false, true), (true, false), (true, true)]
            .into_iter()
            .enumerate()
        {
            let class = (a ^ b) as usize;
            net.begin_sample();
            let mut count = [0i32; 2];
            for x in pattern(a, b, 1_000 + (rep * 4 + k) as u64) {
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
    correct as f32 / n as f32 + 0.1 * margin_sum / n as f32
}

/// Start evolution by sending reports to `tx`. Blocks the thread: run in a separate one.
pub fn run(tx: &TrainTx, cfg: &DemoConfig) -> DemoSummary {
    let mut ec = EvolutionConfig::new(INPUTS, 2);
    ec.population = cfg.population;
    ec.seed = cfg.seed;
    ec.threads = cfg.threads;
    ec.complexity = ComplexityPenalty {
        per_hidden: 0.01,
        per_conn: 0.002,
    };
    let mut pop = match Population::new(ec) {
        Ok(p) => p,
        Err(_) => {
            tx.start();
            tx.finish();
            return DemoSummary {
                generations_done: 0,
                best: f32::NAN,
                solved: false,
                stopped: false,
            };
        }
    };
    tx.start();
    let mut sum = DemoSummary {
        generations_done: 0,
        best: f32::NEG_INFINITY,
        solved: false,
        stopped: false,
    };
    for g in 0..cfg.generations {
        let r = pop.step(&fitness);
        let solved = r.best_raw >= cfg.target;
        sum.generations_done = g + 1;
        sum.best = sum.best.max(r.best_raw);
        sum.solved = solved;
        tx.evolution(EvolutionReport::from_generation(
            &r,
            cfg.generations,
            cfg.population,
            solved,
        ));
        if !cfg.pace.is_zero() {
            std::thread::sleep(cfg.pace);
        }
        if tx.checkpoint() {
            sum.stopped = true;
            break;
        }
        if solved {
            break;
        }
    }
    tx.finish();
    sum
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::telemetry::{ChannelConfig, TrainEvent, channel};

    #[test]
    fn streams_one_report_per_generation_and_finishes() {
        let (tx, rx) = channel(ChannelConfig::new("evo"));
        let cfg = DemoConfig {
            generations: 4,
            population: 30,
            threads: 2,
            pace: Duration::ZERO,
            ..DemoConfig::default()
        };
        let s = run(&tx, &cfg);
        assert_eq!(s.generations_done, 4);
        assert!(!s.stopped);
        let mut gens = vec![];
        let (mut started, mut finished) = (false, false);
        while let Ok(e) = rx.events.try_recv() {
            match e {
                TrainEvent::Started { .. } => started = true,
                TrainEvent::Evolution(r) => {
                    assert_eq!(r.population, 30);
                    assert!(!r.species_sizes.is_empty());
                    gens.push(r.generation);
                }
                TrainEvent::Finished { stopped } => finished = !stopped,
                _ => {}
            }
        }
        assert!(started && finished);
        assert_eq!(gens, vec![0, 1, 2, 3]);
    }

    #[test]
    fn stop_interrupts_at_generation_boundary() {
        let (tx, rx) = channel(ChannelConfig::new("evo"));
        rx.control().stop();
        let cfg = DemoConfig {
            generations: 50,
            population: 20,
            threads: 1,
            pace: Duration::ZERO,
            ..DemoConfig::default()
        };
        let s = run(&tx, &cfg);
        assert!(s.stopped);
        assert_eq!(s.generations_done, 1);
    }
}
