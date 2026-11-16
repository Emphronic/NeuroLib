use burn::tensor::{Device, Distribution, Tensor};
use neo_snn::functional::{max_over_time, sum_over_time};
use std::time::Instant;
fn main() {
    let device = Device::flex().autodiff();
    let x: Tensor<3> = Tensor::random([64, 200, 128], Distribution::Normal(0.0, 1.0), &device);
    let t = |name: &str, f: &dyn Fn(Tensor<3>) -> Tensor<2>| {
        for _ in 0..2 {
            let _ = f(x.clone().require_grad()).sum().backward();
        }
        let (mut a, mut b) = (0.0, 0.0);
        for _ in 0..6 {
            let s = Instant::now();
            let y = f(x.clone().require_grad()).sum();
            let _ = y.clone().into_data();
            a += s.elapsed().as_secs_f64();
            let s = Instant::now();
            let _ = y.backward();
            b += s.elapsed().as_secs_f64();
        }
        println!("{name}: fwd {:.1} ms, bwd {:.1} ms", a / 6e-3, b / 6e-3);
    };
    t("sum_over_time", &|x| sum_over_time(x));
    t("max_over_time", &|x| max_over_time(x));
}
