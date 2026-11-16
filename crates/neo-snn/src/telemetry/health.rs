//! Verdict on the state of the network - in simple terms.
//!
//! Raster and curves show DATA, but you still need to be able to read them.
//! This module does what an experienced person would do by looking at the screen:
//! reduces the picture to one conclusion - “everything is fine”, “the layer is silent”,
//! “neurons are spiked in one gulp” - and tells you what to do about it.
//!
//! No I/O: functions receive numbers and return text, so
//! The logic is verified by regular tests.

use super::frame::SpikeFrame;
use super::history::History;

/// Output Severity - Defines the color and icon.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum Severity {
    /// It's okay.
    Ok,
    /// Worth paying attention, but learning continues.
    Notice,
    /// Problem: The training probably won't add up.
    Problem,
}

/// One point of verdict: what’s wrong and what to do about it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct Finding {
    /// How serious is this?
    pub severity: Severity,
    /// What is visible is one short phrase.
    pub what: String,
    /// What to try. Empty if there is nothing to do.
    pub advice: String,
}

impl Finding {
    fn new(severity: Severity, what: impl Into<String>, advice: impl Into<String>) -> Self {
        Self {
            severity,
            what: what.into(),
            advice: advice.into(),
        }
    }
}

/// Assess the state of the network using a snapshot of activity and historical metrics.
///
/// Returns items from most severe to mildest. The list is not
/// can be empty: if there are no problems, it contains one item `Severity::Ok`.
pub fn assess(
    frame: &SpikeFrame,
    loss: &History,
    accuracy: &History,
    grad: &History,
) -> Vec<Finding> {
    let mut out: Vec<Finding> = Vec::new();

    // ──Learning metrics ──────────────────────── ────────────────────────
    // Loss diverges or not a number: more important than anything else.
    if let Some(t) = loss.trend() {
        if t > 0.25 {
            out.push(Finding::new(
                Severity::Problem,
                "the loss is growing",
                "reduce learning rate (lr) by 2–10 times",
            ));
        }
    }

    if let Some(g) = grad.recent_mean(8) {
        if g > 1e3 {
            out.push(Finding::new(
                Severity::Problem,
                "the gradient is very large",
                "enable gradient norm constraint or reduce lr",
            ));
        } else if g < 1e-7 {
            out.push(Finding::new(
                Severity::Problem,
                "the gradient is almost zero - the network does not learn",
                "check if the layer is silent (see raster); raise β or lower the threshold",
            ));
        }
    }

    // ── Spike activity ─────────────────────── ───────────────────────
    if !frame.is_empty() {
        let mean = frame.mean_rate();
        let silent = frame.silent_fraction();
        let saturated = frame.saturated_units();

        if mean == 0.0 {
            out.push(Finding::new(
                Severity::Problem,
                "the layer does not spike: not a single neuron fired",
                "lower the threshold, increase the input weight or β - the gradient does not go through the silent layer",
            ));
        } else if silent > 0.5 {
            out.push(Finding::new(
                Severity::Problem,
                format!("{:.0}% of neurons are silent", silent * 100.0),
                "lower the threshold or raise the initial weights - silent neurons do not learn",
            ));
        } else if silent > 0.25 {
            out.push(Finding::new(
                Severity::Notice,
                format!("{:.0}% of neurons are silent", silent * 100.0),
                "if the share grows, lower the threshold",
            ));
        }

        if mean > 0.9 {
            out.push(Finding::new(
                Severity::Problem,
                "neurons spike at almost every step",
                "raise the threshold or reduce the weight of the input - the output no longer depends on the input",
            ));
        } else if saturated > 0 && frame.units > 0 {
            out.push(Finding::new(
                Severity::Notice,
                format!("{saturated} neur. spike at every step"),
                "they do not differentiate between inputs; raise the threshold",
            ));
        }

        // Burst synchronization: only if there are enough spikes,
        // otherwise one random spike gives a huge ratio.
        if mean > 0.05 && frame.burstiness() > 3.0 {
            out.push(Finding::new(
                Severity::Notice,
                "neurons spike in volleys (synchronously)",
                "add lateral inhibition or input noise",
            ));
        }
    }

    // ── Summary ────────────────────────────── ──────────────────────────────
    if out.is_empty() {
        let learning = match (loss.trend(), accuracy.recent_mean(8)) {
            (Some(t), _) if t < -0.03 => "loss falls",
            (_, Some(a)) if a >= 0.99 => "maximum accuracy",
            _ => "no signs of problems",
        };
        out.push(Finding::new(
            Severity::Ok,
            format!("everything is fine - {learning}"),
            "",
        ));
    }

    out.sort_by(|a, b| b.severity.cmp(&a.severity));
    out
}

/// Overall Severity: Highest among the items.
pub fn overall(findings: &[Finding]) -> Severity {
    findings
        .iter()
        .map(|f| f.severity)
        .max()
        .unwrap_or(Severity::Ok)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn hist(values: &[f32]) -> History {
        let mut h = History::new(64);
        for &v in values {
            h.push(v);
        }
        h
    }

    fn frame(steps: usize, units: usize, f: impl Fn(usize, usize) -> bool) -> SpikeFrame {
        SpikeFrame::from_bits(
            steps,
            units,
            (0..steps)
                .flat_map(|t| (0..units).map(move |u| (t, u)))
                .map(|(t, u)| f(t, u))
                .collect(),
        )
    }

    fn none() -> History {
        History::new(4)
    }

    #[test]
    fn healthy_network_gets_ok() {
        // each neuron spikes every other time, with a shift → smoothly and vividly
        let fr = frame(24, 8, |t, u| (t + u) % 3 == 0);
        let loss = hist(&(0..30).map(|i| 1.0 / (1.0 + i as f32)).collect::<Vec<_>>());
        let f = assess(&fr, &loss, &none(), &none());
        assert_eq!(overall(&f), Severity::Ok, "{f:?}");
        assert_eq!(f.len(), 1);
    }

    #[test]
    fn dead_layer_is_a_problem_with_advice() {
        let fr = frame(24, 8, |_, _| false);
        let f = assess(&fr, &none(), &none(), &none());
        assert_eq!(overall(&f), Severity::Problem);
        assert!(f[0].what.contains("does not spike"));
        assert!(!f[0].advice.is_empty(), "problem without advice is useless");
    }

    #[test]
    fn mostly_silent_layer_is_flagged() {
        // only neuron 0 is alive out of 8 → silent 87%
        let fr = frame(24, 8, |t, u| u == 0 && t % 2 == 0);
        let f = assess(&fr, &none(), &none(), &none());
        assert_eq!(overall(&f), Severity::Problem);
        assert!(f.iter().any(|x| x.what.contains("are silent")));
    }

    #[test]
    fn saturated_layer_is_flagged() {
        let fr = frame(24, 8, |_, _| true);
        let f = assess(&fr, &none(), &none(), &none());
        assert_eq!(overall(&f), Severity::Problem);
    }

    #[test]
    fn synchronous_bursts_are_noticed() {
        // all neurons are spiked simultaneously at steps 0 and 12, otherwise silence
        let fr = frame(24, 8, |t, _| t == 0 || t == 12);
        let f = assess(&fr, &none(), &none(), &none());
        assert!(f.iter().any(|x| x.what.contains("in volleys")), "{f:?}");
    }

    #[test]
    fn rising_loss_is_a_problem() {
        let fr = frame(24, 8, |t, u| (t + u) % 3 == 0);
        let loss = hist(&(0..30).map(|i| 0.1 + i as f32 * 0.1).collect::<Vec<_>>());
        let f = assess(&fr, &loss, &none(), &none());
        assert_eq!(overall(&f), Severity::Problem);
        assert!(f.iter().any(|x| x.what.contains("the loss is growing")));
    }

    #[test]
    fn vanishing_gradient_is_a_problem() {
        let fr = frame(24, 8, |t, u| (t + u) % 3 == 0);
        let g = hist(&[1e-9; 20]);
        let f = assess(&fr, &none(), &none(), &g);
        assert!(f.iter().any(|x| x.what.contains("does not learn")));
    }

    #[test]
    fn problems_sort_before_notices() {
        // dead layer + volleys are impossible at the same time, so we take
        // growing loss (Problem) and volleys (Notice)
        let fr = frame(24, 8, |t, _| t == 0 || t == 12);
        let loss = hist(&(0..30).map(|i| 0.1 + i as f32 * 0.1).collect::<Vec<_>>());
        let f = assess(&fr, &loss, &none(), &none());
        assert!(f.len() >= 2);
        assert!(f[0].severity >= f[1].severity);
        assert_eq!(f[0].severity, Severity::Problem);
    }

    #[test]
    fn empty_input_is_ok_not_panic() {
        let f = assess(&SpikeFrame::default(), &none(), &none(), &none());
        assert_eq!(overall(&f), Severity::Ok);
    }
}
