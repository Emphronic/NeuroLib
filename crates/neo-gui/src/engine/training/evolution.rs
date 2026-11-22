//! What the dashboard knows about an ARCHITECTURE SEARCH (`neo_snn::structural::evolution`):
//! folded from the [`EvolutionReport`]s `neo_snn::telemetry` delivers, one per generation.
//! Pure data, like [`super::state`] and [`super::structure`]: no UI type, tested headless.
//!
//! A hole in the generation numbers means the search thread had to drop a report (its queue
//! was full); it is counted and shown, not hidden.

use std::sync::Arc;

use neo_snn::telemetry::{EvolutionReport, Finding, History, Severity};

/// Generations kept per history.
pub const EVOLUTION_HISTORY: usize = 2048;
/// Generations without a better champion after which the search is called stuck.
pub const STAGNATION_GENERATIONS: u64 = 30;
/// Share of invalid genomes that is a problem, not noise.
pub const INVALID_PROBLEM: f32 = 0.25;

#[derive(Debug, Clone)]
pub struct EvolutionState {
    pub best: History,
    pub mean: History,
    pub champion: History,
    pub best_hidden: History,
    pub best_conns: History,
    pub species: History,
    /// The latest report.
    pub last: Option<Arc<EvolutionReport>>,
    /// Reports folded in.
    pub reports: u64,
    /// Generations missing: their reports were dropped by a full queue.
    pub missed: u64,
    /// Generation of the latest report that raised the champion.
    pub last_improvement: u64,
    best_champion: f32,
    last_generation: Option<u64>,
}

impl Default for EvolutionState {
    fn default() -> Self {
        Self::new()
    }
}

impl EvolutionState {
    pub fn new() -> Self {
        Self {
            best: History::new(EVOLUTION_HISTORY),
            mean: History::new(EVOLUTION_HISTORY),
            champion: History::new(EVOLUTION_HISTORY),
            best_hidden: History::new(EVOLUTION_HISTORY),
            best_conns: History::new(EVOLUTION_HISTORY),
            species: History::new(EVOLUTION_HISTORY),
            last: None,
            reports: 0,
            missed: 0,
            last_improvement: 0,
            best_champion: f32::NEG_INFINITY,
            last_generation: None,
        }
    }

    /// No report yet: the Evolution tab has nothing to show (the run is not an architecture search).
    pub fn is_empty(&self) -> bool {
        self.reports == 0
    }

    /// Folds one report in.
    pub fn apply(&mut self, r: &Arc<EvolutionReport>) {
        if let Some(prev) = self.last_generation {
            // A generation already folded in (a report delivered twice, or an old one arriving
            // late) adds no point, as in the structure state
            if r.generation <= prev {
                return;
            }
            if r.generation > prev + 1 {
                self.missed += r.generation - prev - 1;
            }
        }
        self.last_generation = Some(r.generation);
        self.reports += 1;
        let ok = |v: f32| if v.is_finite() { v } else { 0.0 };
        self.best.push(ok(r.best_raw));
        self.mean.push(ok(r.mean_raw));
        self.champion.push(ok(r.champion_raw));
        self.best_hidden.push(r.best_hidden as f32);
        self.best_conns.push(r.best_conns as f32);
        self.species.push(r.species_sizes.len() as f32);
        if r.champion_raw.is_finite() && r.champion_raw > self.best_champion + 1e-6 {
            self.best_champion = r.champion_raw;
            self.last_improvement = r.generation;
        }
        self.last = Some(r.clone());
    }

    /// Generations since the champion last improved (0 before the first report).
    pub fn stagnation(&self) -> u64 {
        self.last
            .as_ref()
            .map_or(0, |r| r.generation.saturating_sub(self.last_improvement))
    }

    /// The facts of the tab's header: `(label, value)` pairs, in the order shown. Empty before the
    /// first report.
    pub fn summary(&self) -> Vec<(&'static str, String)> {
        let Some(r) = &self.last else {
            return Vec::new();
        };
        let progress = if r.generations > 0 {
            format!("{} of {}", r.generation + 1, r.generations)
        } else {
            (r.generation + 1).to_string()
        };
        let mut out = vec![
            ("Generation", progress),
            (
                "Fitness",
                format!(
                    "best {:.3} · mean {:.3} · champion {:.3}",
                    r.best_raw, r.mean_raw, r.champion_raw
                ),
            ),
            (
                "Best network",
                format!("{} hidden · {} connections", r.best_hidden, r.best_conns),
            ),
            (
                "Population",
                format!(
                    "{} · mean hidden {:.1} · {} invalid",
                    r.population, r.mean_hidden, r.invalid
                ),
            ),
            (
                "Species",
                format!(
                    "{} · compatibility threshold {:.1}",
                    r.species_sizes.len(),
                    r.threshold
                ),
            ),
            (
                "Since last improvement",
                format!("{} generations", self.stagnation()),
            ),
        ];
        if self.missed > 0 {
            out.push(("Reports lost", self.missed.to_string()));
        }
        out
    }

    /// The verdict on the search, worst first; never empty once a report has come.
    pub fn findings(&self) -> Vec<Finding> {
        let Some(r) = &self.last else {
            return Vec::new();
        };
        let mut out = Vec::new();
        let invalid_share = r.invalid as f32 / r.population.max(1) as f32;
        if invalid_share >= INVALID_PROBLEM {
            out.push(Finding {
                severity: Severity::Problem,
                what: format!("{} of {} genomes are invalid ({:.0}%)", r.invalid, r.population, invalid_share * 100.0),
                advice: "loosen the topology/parameter limits or lower the mutation rates: invalid offspring are wasted evaluations".into(),
            });
        }
        if r.solved {
            out.push(Finding {
                severity: Severity::Ok,
                what: format!("target reached at generation {}", r.generation + 1),
                advice: String::new(),
            });
        } else if self.stagnation() >= STAGNATION_GENERATIONS {
            out.push(Finding {
                severity: Severity::Notice,
                what: format!("the champion has not improved for {} generations", self.stagnation()),
                advice: "raise the population or the add-node / add-connection rates, or lower the complexity penalty".into(),
            });
        }
        if r.population >= 50 && r.generation >= 10 && r.species_sizes.len() == 1 {
            out.push(Finding {
                severity: Severity::Notice,
                what: "a single species: no architectural diversity".into(),
                advice: "lower the compatibility threshold so that different structures can survive side by side".into(),
            });
        }
        if self.missed > 0 {
            out.push(Finding {
                severity: Severity::Notice,
                what: format!(
                    "{} generation reports were lost (the queue was full)",
                    self.missed
                ),
                advice: "the curves have gaps; the champion line is still exact".into(),
            });
        }
        if out.iter().all(|f| f.severity == Severity::Ok) && !r.solved {
            out.push(Finding {
                severity: Severity::Ok,
                what: format!(
                    "the search is progressing: best {:.3}, {} species",
                    r.best_raw,
                    r.species_sizes.len()
                ),
                advice: String::new(),
            });
        }
        out.sort_by_key(|a| std::cmp::Reverse(a.severity));
        out
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn rep(generation: u64, best: f32, champion: f32) -> Arc<EvolutionReport> {
        Arc::new(EvolutionReport {
            generation,
            generations: 100,
            best_raw: best,
            best_effective: best,
            mean_raw: best * 0.5,
            champion_raw: champion,
            best_hidden: 1,
            best_conns: 5,
            mean_hidden: 0.7,
            species_sizes: vec![10, 8, 6],
            invalid: 0,
            threshold: 3.0,
            population: 24,
            solved: false,
        })
    }

    #[test]
    fn empty_until_first_report_and_findings_empty() {
        let s = EvolutionState::new();
        assert!(s.is_empty());
        assert!(s.findings().is_empty());
        assert_eq!(s.stagnation(), 0);
    }

    #[test]
    fn summary_is_empty_then_describes_the_latest_generation() {
        let mut s = EvolutionState::new();
        assert!(s.summary().is_empty());
        s.apply(&rep(6, 0.8, 0.8));
        let sum = s.summary();
        assert_eq!(sum[0], ("Generation", "7 of 100".to_string()));
        assert!(sum[1].1.contains("best 0.800"));
        assert!(sum[4].1.starts_with("3 "), "{:?}", sum[4]);
        assert!(!sum.iter().any(|(k, _)| *k == "Reports lost"));
    }

    #[test]
    fn folds_histories_and_tracks_improvement() {
        let mut s = EvolutionState::new();
        s.apply(&rep(0, 0.5, 0.5));
        s.apply(&rep(1, 0.6, 0.6));
        s.apply(&rep(2, 0.55, 0.6));
        assert_eq!(s.reports, 3);
        assert_eq!(s.best.values().len(), 3);
        assert_eq!(s.last_improvement, 1);
        assert_eq!(s.stagnation(), 1);
        assert_eq!(s.missed, 0);
        let f = s.findings();
        assert_eq!(f.len(), 1);
        assert_eq!(f[0].severity, Severity::Ok);
    }

    #[test]
    fn gap_in_generations_is_counted_and_reported() {
        let mut s = EvolutionState::new();
        s.apply(&rep(0, 0.5, 0.5));
        s.apply(&rep(4, 0.6, 0.6));
        assert_eq!(s.missed, 3);
        let f = s.findings();
        assert!(
            f.iter()
                .any(|f| f.severity == Severity::Notice && f.what.contains("lost"))
        );
        assert!(
            f.iter().all(|f| f.severity != Severity::Ok),
            "the all-fine line must go"
        );
    }

    #[test]
    fn stagnation_invalid_and_single_species_are_flagged() {
        let mut s = EvolutionState::new();
        for g in 0..40 {
            s.apply(&rep(g, 0.9, 0.9));
        }
        let mut r = (*rep(40, 0.9, 0.9)).clone();
        r.invalid = 20;
        r.population = 60;
        r.species_sizes = vec![60];
        s.apply(&Arc::new(r));
        let f = s.findings();
        assert_eq!(f[0].severity, Severity::Problem, "{f:?}");
        assert!(f.iter().any(|f| f.what.contains("has not improved")));
        assert!(f.iter().any(|f| f.what.contains("single species")));
    }

    #[test]
    fn a_repeated_or_late_report_is_ignored() {
        let mut s = EvolutionState::new();
        s.apply(&rep(3, 0.5, 0.5));
        s.apply(&rep(3, 0.9, 0.9));
        s.apply(&rep(2, 0.9, 0.9));
        assert_eq!((s.reports, s.best.values().len(), s.missed), (1, 1, 0));
        assert_eq!(s.last.as_ref().unwrap().best_raw, 0.5);
    }

    #[test]
    fn the_verdict_counts_generations_like_the_header() {
        let mut s = EvolutionState::new();
        let mut r = (*rep(6, 1.1, 1.1)).clone();
        r.solved = true;
        s.apply(&Arc::new(r));
        assert_eq!(s.summary()[0].1, "7 of 100");
        assert!(
            s.findings()[0].what.contains("generation 7"),
            "{:?}",
            s.findings()
        );
    }

    #[test]
    fn solved_run_reports_success_not_stagnation() {
        let mut s = EvolutionState::new();
        for g in 0..80 {
            s.apply(&rep(g, 0.9, 0.9));
        }
        let mut r = (*rep(80, 1.1, 1.1)).clone();
        r.solved = true;
        s.apply(&Arc::new(r));
        let f = s.findings();
        assert_eq!(f.len(), 1);
        assert!(f[0].what.contains("target reached"));
    }

    #[test]
    fn nonfinite_values_do_not_poison_histories() {
        let mut s = EvolutionState::new();
        s.apply(&rep(0, f32::NAN, f32::NEG_INFINITY));
        assert!(s.best.values().iter().all(|v| v.is_finite()));
        assert!(s.champion.values().iter().all(|v| v.is_finite()));
    }
}
