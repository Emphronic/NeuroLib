//! Bench: forward+backward MultiStepLeaky / LeakyParallel on Flex.
//! `cargo run -p neo-snn --example bench_multistep --release`
use burn::tensor::{Device, Distribution, Tensor};
use neo_snn::neuron::{LeakyConfig, LeakyParallel, MultiStepLeaky};
use neo_snn::state::ResetMode;
use neo_snn::surrogate::ATan;
use std::time::Instant;

fn main() {
    let device = Device::flex().autodiff();
    let sg = ATan::default();
    for &(b, t, f) in &[(32usize, 50usize, 128usize), (64, 100, 256)] {
        for mode in [ResetMode::Subtract, ResetMode::Zero] {
            let cfg = LeakyConfig::new(0.9).with_reset_mode(mode);
            let ms = MultiStepLeaky::new(&cfg, &device);
            let lp = LeakyParallel::new(&cfg, &device);
            let x: Tensor<3> = Tensor::random([b, t, f], Distribution::Normal(0.5, 1.0), &device);
            for (name, run) in [
                (
                    "MultiStep",
                    Box::new(|x: Tensor<3>| ms.forward(x, &sg))
                        as Box<dyn Fn(Tensor<3>) -> Tensor<3>>,
                ),
                ("Parallel ", Box::new(|x: Tensor<3>| lp.forward(x, &sg))),
            ] {
                for _ in 0..2 {
                    let _ = run(x.clone().require_grad()).sum().backward();
                }
                let n = 8;
                let (mut fw, mut bw) = (0.0, 0.0);
                for _ in 0..n {
                    let s = Instant::now();
                    let y = run(x.clone().require_grad()).sum();
                    let _ = y.clone().into_data();
                    fw += s.elapsed().as_secs_f64();
                    let s = Instant::now();
                    let _ = y.backward();
                    bw += s.elapsed().as_secs_f64();
                }
                println!(
                    "B{b} T{t} F{f} {mode:?} {name}: fwd {:.1} ms, bwd {:.1} ms",
                    fw / n as f64 * 1e3,
                    bw / n as f64 * 1e3
                );
            }
        }
    }
}
