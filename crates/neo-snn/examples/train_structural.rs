//! Structural plasticity through a telemetry channel, without GUI: the learning flow sends reports, and
//! the consumer (here - the console, in `neo-gui` - the Structure tab) prints counters, events from
//! reasons and verdict.
//!
//! Launch: `cargo run --release --example train_structural -p neo-snn`
use std::time::Duration;

use neo_snn::telemetry::structural_demo::{self, DemoConfig};
use neo_snn::telemetry::{ChannelConfig, History, TrainEvent, assess_structure, channel};

fn main() {
    let (tx, rx) = channel(ChannelConfig::new("structural plasticity"));
    let worker = std::thread::spawn(move || {
        structural_demo::run(
            &tx,
            &DemoConfig {
                pace: Duration::from_millis(20),
                ..Default::default()
            },
        )
    });

    let mut loss = History::new(256);
    let mut recent = Vec::new();
    while let Ok(event) = rx.events.recv_blocking() {
        match event {
            TrainEvent::Started { title } => println!("▶ {title}"),
            TrainEvent::Step(m) => loss.push(m.loss),
            TrainEvent::Structure(r) => {
                let s = &r.stats;
                println!(
                    "epoch {:2}: neurons {}/{}, synapses {} (active {}), rate {:.3}, loss {}",
                    r.progress.epoch + 1,
                    s.neurons,
                    r.limits.max_neurons,
                    s.synapses,
                    s.active_synapses,
                    s.mean_rate,
                    loss.last().map_or("–".into(), |l| format!("{l:.3}")),
                );
                for e in &r.events {
                    println!("    {e}");
                    recent.push(e.clone());
                }
                for f in assess_structure(s, r.limits, &recent, &loss)
                    .iter()
                    .filter(|f| f.severity != neo_snn::telemetry::Severity::Ok)
                {
                    println!("    ! {} — {}", f.what, f.advice);
                }
            }
            TrainEvent::Finished { stopped } => {
                println!("■ {}", if stopped { "stopped" } else { "ready" });
                break;
            }
            _ => {}
        }
    }
    let s = worker.join().expect("learning flow");
    println!(
        "loss {:.3} → {:.3}, accuracy {:.2}, operations applied: {}",
        s.start_loss, s.end_loss, s.end_accuracy, s.applied
    );
}
