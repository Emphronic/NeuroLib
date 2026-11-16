// File: crates/neo-snn/tests/attention.rs
// New file: minimum test coverage for attention/{spiking_self_attention,
// spike_driven_self_attention, mlp, block}.rs - before this there was a section at all
// without #[cfg(test)]. Covers: shape preservation [T,B,N,D], finiteness of the
// output, non-trivial output (not all zeros), panic at num_heads == 0 /
// dim % num_heads != 0, and that the residual connection in SpikingTransformerBlock
// actually participates in the result (and is not discarded by mistake).
//
// Launch: `cargo test -p neo-snn --test attention`

use burn::tensor::Device;
use burn::tensor::Tensor;

use neo_snn::prelude::*;

fn dev() -> Device {
    Device::flex()
}

/// Deterministic but non-trivial (not all zeros/not all ones) input
/// `[T, B, N, D]` - sine over a flat index gives a scatter of values around
/// default spike threshold (1.0) so that SN(·) does not degenerate into a constant.
fn spike_input(t: usize, b: usize, n: usize, d: usize) -> Tensor<4> {
    let total = t * b * n * d;
    let data: Vec<f32> = (0..total).map(|i| (i as f32 * 0.37).sin().abs()).collect();
    Tensor::<1>::from_floats(data.as_slice(), &dev()).reshape([t, b, n, d])
}

fn to_vec4(x: Tensor<4>) -> Vec<f32> {
    x.into_data().try_to_vec::<f32>().unwrap()
}

// ---------------------------------------------------------------------
// Shape preservation
// ---------------------------------------------------------------------

#[test]
fn ssa_preserves_shape() {
    let device = dev();
    let layer = SpikingSelfAttentionConfig::new(8, 2).init(&device);
    let surrogate = ATan::default();
    let x = spike_input(3, 2, 5, 8);

    let out = layer.forward(x, &surrogate);
    assert_eq!(out.dims(), [3, 2, 5, 8]);
}

#[test]
fn sdsa_preserves_shape() {
    let device = dev();
    let layer = SpikeDrivenSelfAttentionConfig::new(8, 2).init(&device);
    let surrogate = ATan::default();
    let x = spike_input(3, 2, 5, 8);

    let out = layer.forward(x, &surrogate);
    assert_eq!(out.dims(), [3, 2, 5, 8]);
}

#[test]
fn mlp_preserves_shape() {
    let device = dev();
    let layer = SpikingMlpConfig::new(8).init(&device);
    let surrogate = ATan::default();
    let x = spike_input(3, 2, 5, 8);

    let out = layer.forward(x, &surrogate);
    assert_eq!(out.dims(), [3, 2, 5, 8]);
}

#[test]
fn transformer_block_preserves_shape_ssa() {
    let device = dev();
    let block = SpikingTransformerBlockConfig::new(8, 2).init(&device);
    let surrogate = ATan::default();
    let x = spike_input(3, 2, 5, 8);

    let out = block.forward(x, &surrogate);
    assert_eq!(out.dims(), [3, 2, 5, 8]);
}

#[test]
fn transformer_block_preserves_shape_sdsa() {
    let device = dev();
    let block = SpikingTransformerBlockConfig::new(8, 2)
        .with_sdsa()
        .init(&device);
    let surrogate = ATan::default();
    let x = spike_input(3, 2, 5, 8);

    let out = block.forward(x, &surrogate);
    assert_eq!(out.dims(), [3, 2, 5, 8]);
}

// ---------------------------------------------------------------------
// Finite output
// ---------------------------------------------------------------------

#[test]
fn ssa_forward_is_finite() {
    let device = dev();
    let layer = SpikingSelfAttentionConfig::new(8, 2).init(&device);
    let surrogate = ATan::default();
    let x = spike_input(3, 2, 5, 8);

    let out = to_vec4(layer.forward(x, &surrogate));
    assert!(
        out.iter().all(|v| v.is_finite()),
        "non-finite values obtained: {out:?}"
    );
}

#[test]
fn sdsa_forward_is_finite() {
    let device = dev();
    let layer = SpikeDrivenSelfAttentionConfig::new(8, 2).init(&device);
    let surrogate = ATan::default();
    let x = spike_input(3, 2, 5, 8);

    let out = to_vec4(layer.forward(x, &surrogate));
    assert!(
        out.iter().all(|v| v.is_finite()),
        "non-finite values obtained: {out:?}"
    );
}

// ---------------------------------------------------------------------
// Non-trivial output (the forward must not degenerate into a constant zero)
// ---------------------------------------------------------------------

#[test]
fn ssa_forward_is_not_trivially_zero() {
    let device = dev();
    // Default threshold (1.0) with Linear initialized to ±1/√8 and input
    // |sin| ≤ 1 gives almost no Q/K/V spikes: the output is zero not because of a
    // layer bug, but because the threshold is unreachable. Lower it to verify
    // that the signal reaches the output.
    let layer = SpikingSelfAttentionConfig::new(8, 2)
        .with_threshold(0.1)
        .init(&device);
    let surrogate = ATan::default();
    let x = spike_input(3, 2, 5, 8);

    let out = to_vec4(layer.forward(x, &surrogate));
    assert!(
        out.iter().any(|&v| v != 0.0),
        "SSA issued all zeros on a non-trivial input"
    );
}

#[test]
fn sdsa_forward_is_not_trivially_zero() {
    let device = dev();
    // Default threshold (1.0) with Linear initialized to ±1/√8 and input
    // |sin| ≤ 1 gives almost no Q/K/V spikes: the output is zero not because of a
    // layer bug, but because the threshold is unreachable. Lower it to verify
    // that the signal reaches the output.
    let layer = SpikeDrivenSelfAttentionConfig::new(8, 2)
        .with_threshold(0.1)
        .init(&device);
    let surrogate = ATan::default();
    let x = spike_input(3, 2, 5, 8);

    let out = to_vec4(layer.forward(x, &surrogate));
    assert!(
        out.iter().any(|&v| v != 0.0),
        "SDSA returned all zeros on a non-trivial input"
    );
}

// ---------------------------------------------------------------------
// Configuration: num_heads == 0 / dim % num_heads != 0 must panic
// ---------------------------------------------------------------------

#[test]
#[should_panic(expected = "num_heads")]
fn ssa_config_panics_on_zero_heads() {
    let _ = SpikingSelfAttentionConfig::new(8, 0);
}

#[test]
#[should_panic(expected = "num_heads")]
fn sdsa_config_panics_on_zero_heads() {
    let _ = SpikeDrivenSelfAttentionConfig::new(8, 0);
}

#[test]
#[should_panic(expected = "num_heads")]
fn block_config_panics_on_zero_heads() {
    let _ = SpikingTransformerBlockConfig::new(8, 0);
}

#[test]
#[should_panic(expected = "must be divisible by")]
fn ssa_config_panics_on_indivisible_dim() {
    let _ = SpikingSelfAttentionConfig::new(9, 2);
}

#[test]
#[should_panic(expected = "must be divisible by")]
fn sdsa_config_panics_on_indivisible_dim() {
    let _ = SpikeDrivenSelfAttentionConfig::new(9, 2);
}

#[test]
#[should_panic(expected = "must be divisible by")]
fn block_config_panics_on_indivisible_dim() {
    let _ = SpikingTransformerBlockConfig::new(9, 2);
}

#[test]
#[should_panic(expected = "dim")]
fn ssa_config_panics_on_zero_dim() {
    let _ = SpikingSelfAttentionConfig::new(0, 1);
}

#[test]
#[should_panic(expected = "hidden_dim")]
fn mlp_config_panics_on_zero_hidden_dim() {
    let _ = SpikingMlpConfig::new(8).with_hidden_dim(0);
}

// Test for the participation of residual connections (`transformer_block_residual_changes_output`)
// lives as #[cfg(test)] inside src/attention/block.rs, not here: it
// need access to private `SpikingAttentionVariant::forward_pre_spike`,
// which the external integration test does not have. See block.rs.
