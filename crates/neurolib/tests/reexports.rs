//! Checks that each feature opens its module. Launch: `cargo test -p neurolib --features full`.

#[cfg(feature = "snn")]
#[test]
fn snn() {
    let _: Option<neurolib::snn::SnnError> = None;
}

#[cfg(feature = "nn")]
#[test]
fn nn() {
    let _: Option<neurolib::nn::NnError> = None;
}

#[cfg(feature = "quant")]
#[test]
fn quant() {
    let _: Option<neurolib::quant::DType> = None;
}

#[cfg(feature = "kernels")]
#[test]
fn kernels() {
    #[allow(unused_imports)]
    use neurolib::kernels::{quant, snn};
}

#[cfg(feature = "store")]
#[test]
fn store() {
    let _: Option<neurolib::store::StoreError> = None;
}

#[cfg(feature = "core")]
#[test]
fn core_crate() {
    let _: Option<neurolib::core::DspError> = None;
}

#[cfg(feature = "io")]
#[test]
fn io() {
    let _ = neurolib::io::formats;
}

#[cfg(feature = "view")]
#[test]
fn view() {
    let _: Option<neurolib::view::View> = None;
}

#[cfg(feature = "synapse")]
#[test]
fn synapse() {
    let _: Option<neurolib::synapse::PcaModel> = None;
}
