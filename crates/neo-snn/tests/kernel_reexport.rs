//! `leaky_scan_forward` is available as `neo_snn::leaky_scan_forward` (i.e. `neurolib::snn::...`),
//! and on Flex it matches the reference `leaky_scan_reference`.
use burn::tensor::{Device, Distribution, Tensor};
use neo_snn::{leaky_scan_forward, leaky_scan_reference};

#[test]
fn leaky_scan_forward_is_reexported_and_matches_reference() {
    let device = Device::flex();
    let x: Tensor<3> = Tensor::random([2, 11, 3], Distribution::Normal(0.0, 1.0), &device);
    let m0: Tensor<2> = Tensor::random([2, 3], Distribution::Normal(0.0, 1.0), &device);
    let beta = Tensor::<1>::from_floats([0.8f32], &device);

    let a = leaky_scan_forward(x.clone(), m0.clone(), beta.clone());
    let b = leaky_scan_reference(x, m0, beta);
    let d = (a - b).abs().max().into_scalar::<f32>();
    assert!(d < 1e-6, "max|diff| = {d}");
}

#[test]
fn kernels_alias_exposes_neo_synk_snn() {
    // The path `neo_snn::kernels::leaky_scan_forward` is the same symbol.
    let device = Device::flex();
    let out = neo_snn::kernels::leaky_scan_forward(
        Tensor::<3>::zeros([1, 2, 2], &device),
        Tensor::<2>::zeros([1, 2], &device),
        Tensor::<1>::from_floats([0.5f32], &device),
    );
    assert_eq!(out.dims(), [1, 2, 2]);
}
