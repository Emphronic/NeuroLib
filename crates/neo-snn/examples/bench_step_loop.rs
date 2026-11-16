use burn::tensor::{Device, Distribution, Tensor};
use neo_snn::neuron::{Leaky, LeakyConfig, MultiStepLeaky, split_time};
use neo_snn::state::NeuronState;
use neo_snn::surrogate::ATan;
use std::time::Instant;
fn main() {
    let device = Device::flex().autodiff();
    let sg = ATan::default();
    let (b, t, f) = (64usize, 100usize, 256usize);
    let lif = Leaky::new(&LeakyConfig::new(0.9), &device);
    let ms = MultiStepLeaky::from_layer(lif.clone());
    let x: Tensor<3> = Tensor::random([b, t, f], Distribution::Normal(0.5, 1.0), &device);
    let step_loop = |x: Tensor<3>, with_beta: bool, split: bool| {
        let mut st = NeuronState::zeros_mem_only(&device, b, f);
        let beta = lif.beta_tensor([b, f]);
        let mut outs = Vec::with_capacity(t);
        let xs = if split {
            split_time(x.clone())
        } else {
            (0..t)
                .map(|k| {
                    x.clone()
                        .slice([0..b, k..k + 1, 0..f])
                        .squeeze_dims::<2>(&[1])
                })
                .collect()
        };
        for xt in xs {
            let (s, ns) = if with_beta {
                lif.step_with_beta(xt, &st, &beta, &sg)
            } else {
                lif.step(xt, &st, &sg)
            };
            st = ns;
            outs.push(s);
        }
        Tensor::stack::<3>(outs, 1)
    };
    type F<'a> = Box<dyn Fn(Tensor<3>) -> Tensor<3> + 'a>;
    let cases: Vec<(&str, F)> = vec![
        (
            "Leaky::step loop       ",
            Box::new(|x| step_loop(x, false, false)),
        ),
        (
            "Leaky::step_with_beta  ",
            Box::new(|x| step_loop(x, true, false)),
        ),
        (
            "step_with_beta+split_time",
            Box::new(|x| step_loop(x, true, true)),
        ),
        ("MultiStepLeaky         ", Box::new(|x| ms.forward(x, &sg))),
    ];
    for (name, run) in cases {
        for _ in 0..2 {
            let _ = run(x.clone().require_grad()).sum().backward();
        }
        let (mut fw, mut bw) = (0.0, 0.0);
        for _ in 0..6 {
            let s = Instant::now();
            let y = run(x.clone().require_grad()).sum();
            let _ = y.clone().into_data();
            fw += s.elapsed().as_secs_f64();
            let s = Instant::now();
            let _ = y.backward();
            bw += s.elapsed().as_secs_f64();
        }
        println!(
            "{name} fwd {:.1} ms, bwd {:.1} ms",
            fw / 6.0 * 1e3,
            bw / 6.0 * 1e3
        );
    }
}
