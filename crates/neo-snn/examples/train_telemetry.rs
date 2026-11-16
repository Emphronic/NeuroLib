//! Training SNN with telemetry in the console (replacement of `train_tui`).
//!
//! Training takes place in a separate thread and writes to the `neo_snn::telemetry` channel; main thread reads
//! events and prints a line for the epoch and a verdict about the network state at the end. Exactly the same channel
//! reads the Training tab in `neo-gui`, where instead of lines there are curves, a raster of spikes and health.
//!
//! Launch: `cargo run -p neo-snn --example train_telemetry --features demo --release`
//! With picture: `cargo run -p neo-gui --release`, Training tab, Start button.

use std::thread;

use burn::tensor::Device;
use neo_snn::demo::{self, DemoConfig};
use neo_snn::telemetry::{
    ChannelConfig, History, Severity, SpikeFrame, TrainEvent, assess, channel,
};

fn main() {
    let (tx, rx) = channel(ChannelConfig::new("two-layer SNN synthetic"));
    let cfg = DemoConfig {
        step_delay_ms: 0,
        ..DemoConfig::default()
    };

    let trainer = thread_with_device(cfg, tx);

    let (mut loss, mut accuracy, mut grad) =
        (History::new(512), History::new(512), History::new(512));
    let mut frame = SpikeFrame::default();
    let (mut epoch_loss, mut epoch_acc, mut in_epoch) = (0.0f32, 0.0f32, 0usize);

    while let Ok(event) = rx.events.recv_blocking() {
        match event {
            TrainEvent::Started { title } => println!("▶ {title}"),
            TrainEvent::Step(m) => {
                loss.push(m.loss);
                if let Some(a) = m.accuracy {
                    accuracy.push(a);
                    epoch_acc += a;
                }
                if let Some(g) = m.grad_norm {
                    grad.push(g);
                }
                epoch_loss += m.loss;
                in_epoch += 1;
                if m.progress.batch + 1 == m.progress.batches {
                    let n = in_epoch.max(1) as f32;
                    println!(
                        "epoch {:>3}/{} loss {:.4} accuracy {:>5.1}%",
                        m.progress.epoch + 1,
                        m.progress.epochs,
                        epoch_loss / n,
                        100.0 * epoch_acc / n
                    );
                    (epoch_loss, epoch_acc, in_epoch) = (0.0, 0.0, 0);
                }
            }
            // The verdict on the health of the network is made by the hidden layer; other layers and weights are for GUI
            TrainEvent::Spikes {
                layer, frame: f, ..
            } if layer == "hidden" => frame = (*f).clone(),
            TrainEvent::Spikes { .. }
            | TrainEvent::Weights { .. }
            | TrainEvent::Links { .. }
            | TrainEvent::Regions { .. }
            | TrainEvent::Activity { .. }
            | TrainEvent::Structure(_)
            | TrainEvent::Evolution(_) => {}
            TrainEvent::Finished { stopped } => {
                println!(
                    "{}",
                    if stopped {
                        "■ stopped"
                    } else {
                        "■ training completed"
                    }
                );
                break;
            }
        }
    }

    println!("\nNetwork status:");
    for f in assess(&frame, &loss, &accuracy, &grad) {
        let mark = match f.severity {
            Severity::Ok => "✓",
            Severity::Notice => "•",
            Severity::Problem => "✗",
        };
        if f.advice.is_empty() {
            println!("  {mark} {}", f.what);
        } else {
            println!("  {mark} {} — {}", f.what, f.advice);
        }
    }
    let summary = trainer.join().expect("the flow of training has fallen");
    println!(
        "\nSteps: {}, last loss: {:.4}",
        summary.steps_done, summary.last_loss
    );
}

fn thread_with_device(
    cfg: DemoConfig,
    tx: neo_snn::telemetry::TrainTx,
) -> thread::JoinHandle<demo::DemoSummary> {
    thread::spawn(move || {
        let device = Device::flex().autodiff();
        demo::run(&device, &cfg, &tx)
    })
}
