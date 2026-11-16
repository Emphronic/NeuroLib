//! Saving/loading STDP layers with a standard Burn recorder: weights, traces,
//! thresholds and hyperparameters survive writing and reading, and damaged ones
//! records give `SnnError::Record`, not panic.

use burn::tensor::Device;
use burn::tensor::Tensor;

use neo_snn::layers::record::RECORD_VERSION;
use neo_snn::prelude::*;

/// Write → bytes → write (in memory).
trait RoundTrip: Sized {
    fn round_trip(self, device: &Device) -> Self;
}

macro_rules! impl_round_trip {
    ($($t:ty),*) => {$(
        impl RoundTrip for $t {
            fn round_trip(self, device: &Device) -> Self {
                let bytes = self.into_bytes().expect("recording");
                <$t>::from_bytes(bytes, device).expect("reading")
            }
        }
    )*};
}
impl_round_trip!(StdpLinearRecord, StdpConv1dRecord, StdpConv2dRecord);

fn dev() -> Device {
    Device::flex()
}

fn pseudo(n: usize, seed: usize) -> Vec<f32> {
    (0..n)
        .map(|i| (((i + seed) * 37 % 101) as f32) / 101.0)
        .collect()
}

fn t(data: Vec<f32>, shape: [usize; 4]) -> Tensor<4> {
    Tensor::<1>::from_floats(data.as_slice(), &dev()).reshape(shape)
}

fn vec_of<const D: usize>(x: Tensor<D>) -> Vec<f32> {
    x.into_data().try_to_vec::<f32>().unwrap()
}

fn roundtrip<R: RoundTrip>(record: R) -> R {
    record.round_trip(&dev())
}

fn triplet() -> StdpRule {
    StdpRule::Triplet {
        a2_plus: 0.05,
        a2_minus: 0.03,
        a3_plus: 0.02,
        a3_minus: 0.01,
        tau_pre: 20.0,
        tau_post: 25.0,
        tau_pre_slow: 100.0,
        tau_post_slow: 120.0,
    }
}

fn reward() -> StdpRule {
    StdpRule::RewardModulated {
        a_plus: 0.1,
        a_minus: 0.05,
        tau_pre: 20.0,
        tau_post: 20.0,
        tau_e: 200.0,
    }
}

fn pair() -> StdpRule {
    StdpRule::Pair {
        a_plus: 0.05,
        a_minus: 0.03,
        tau_pre: 20.0,
        tau_post: 20.0,
    }
}

// ------------------------------------------------------------------ linear

#[test]
fn stdp_linear_roundtrip_preserves_weights_traces_thresholds_and_behaviour() {
    for rule in [pair(), triplet(), reward()] {
        let layer = StdpLinearConfig::new(5, 3, rule)
            .with_threshold(0.37)
            .with_beta(0.81)
            .with_bound(WeightBound::Hard {
                w_min: -0.2,
                w_max: 0.9,
            })
            .init();
        let sur = FastSigmoid::default();
        let mut state = StdpState::zeros(&layer, &dev(), 2);
        for k in 0..6 {
            let x = Tensor::<1>::from_floats(pseudo(10, k).as_slice(), &dev()).reshape([2, 5]);
            state = layer.step(x, &state, &sur).1;
        }

        let loaded = roundtrip(layer.to_record(&state));
        let (layer2, state2) = StdpLinear::from_record(loaded).expect("from_record");

        assert_eq!(layer2.threshold(), 0.37);
        assert_eq!(layer2.beta(), 0.81);
        assert_eq!(vec_of(state.w.clone()), vec_of(state2.w.clone()));
        assert_eq!(vec_of(state.mem.clone()), vec_of(state2.mem.clone()));
        assert_eq!(
            vec_of(state.pre_trace.clone()),
            vec_of(state2.pre_trace.clone())
        );
        assert_eq!(state.eligibility.is_some(), state2.eligibility.is_some());
        assert_eq!(
            state.pre_trace_slow.is_some(),
            state2.pre_trace_slow.is_some()
        );

        // Behavioral check: the next step is the same for the original and the copy
        // - this means, as a rule, the boundaries and thresholds have been restored.
        let x = Tensor::<1>::from_floats(pseudo(10, 99).as_slice(), &dev()).reshape([2, 5]);
        let (s_a, n_a) = layer.step(x.clone(), &state, &sur);
        let (s_b, n_b) = layer2.step(x, &state2, &sur);
        assert_eq!(vec_of(s_a), vec_of(s_b));
        assert_eq!(vec_of(n_a.w), vec_of(n_b.w));
    }
}

#[test]
fn stdp_linear_rejects_foreign_version_and_wrong_shapes() {
    let layer = StdpLinearConfig::new(4, 2, pair()).init();
    let state = StdpState::zeros(&layer, &dev(), 1);

    let mut rec = layer.to_record(&state);
    rec.hyper.version = RECORD_VERSION + 1;
    let err = StdpLinear::from_record(rec).unwrap_err();
    assert!(matches!(err, SnnError::Record { .. }), "{err:?}");
    assert!(err.to_string().contains("version"));

    let mut rec = layer.to_record(&state);
    rec.out_features = 3; // does not match the shape of the scales
    let err = StdpLinear::from_record(rec).unwrap_err();
    assert!(err.to_string().contains('w'), "{err}");

    let mut rec = layer.to_record(&state);
    rec.hyper.rule_kind = 7;
    assert!(StdpLinear::from_record(rec).is_err());

    // Pair rule, but the entry contains eligibility - inconsistency.
    let mut rec = layer.to_record(&state);
    rec.state.eligibility = Some(Tensor::zeros([2, 4], &dev()));
    let err = StdpLinear::from_record(rec).unwrap_err();
    assert!(err.to_string().contains("eligibility"), "{err}");
}

// ------------------------------------------------------------------ conv1d

#[test]
fn stdp_conv1d_roundtrip_matches_original_behaviour() {
    let layer = StdpConv1dConfig::new(
        2,
        3,
        3,
        StdpConv1dRule::Pair {
            a_plus: 0.05,
            a_minus: 0.03,
            tau_pre: 20.0,
            tau_post: 20.0,
        },
    )
    .init();
    let sur = FastSigmoid::default();
    let eff = layer.num_positions(6);
    let mut state = StdpConv1dState::zeros(&layer, &dev(), eff);
    for k in 0..4 {
        let x = Tensor::<1>::from_floats(pseudo(12, k).as_slice(), &dev()).reshape([1, 2, 6]);
        state = layer.step(x, &state, &sur).1;
    }
    let (layer2, state2) =
        StdpConv1d::from_record(roundtrip(layer.to_record(&state))).expect("from_record");
    assert_eq!(vec_of(state.kernel.clone()), vec_of(state2.kernel.clone()));

    let x = Tensor::<1>::from_floats(pseudo(12, 50).as_slice(), &dev()).reshape([1, 2, 6]);
    let (a, na) = layer.step(x.clone(), &state, &sur);
    let (b, nb) = layer2.step(x, &state2, &sur);
    assert_eq!(vec_of(a), vec_of(b));
    assert_eq!(vec_of(na.kernel), vec_of(nb.kernel));
}

// ------------------------------------------------------------------ conv2d

#[test]
fn stdp_conv2d_roundtrip_with_groups_bias_and_homeostasis() {
    for rule in [pair(), triplet(), reward()] {
        let cfg = StdpConv2dConfig::new(4, 6, (2, 2), rule)
            .with_groups(2)
            .with_stride(1)
            .with_padding(1)
            .with_bias_homeostasis(0.2, 0.05)
            .with_threshold(0.45)
            .with_bound(WeightBound::Soft {
                w_min: 0.0,
                w_max: 1.0,
            });
        let layer = cfg.init();
        let sur = FastSigmoid::default();
        let mut state = StdpConv2dState::zeros(&layer, &dev(), 1, 3, 3);
        for k in 0..5 {
            state = layer.step(t(pseudo(36, k), [1, 4, 3, 3]), &state, &sur).1;
        }
        let (layer2, state2) =
            StdpConv2d::from_record(roundtrip(layer.to_record(&state))).expect("from_record");

        assert_eq!(layer2.groups(), 2);
        assert_eq!(layer2.padding(), 1);
        assert!(layer2.has_bias());
        assert_eq!(layer2.threshold(), 0.45);
        assert_eq!(vec_of(state.w.clone()), vec_of(state2.w.clone()));
        assert_eq!(
            vec_of(state.bias.clone().unwrap()),
            vec_of(state2.bias.clone().unwrap())
        );

        let x = t(pseudo(36, 77), [1, 4, 3, 3]);
        let (a, na) = layer.step(x.clone(), &state, &sur);
        let (b, nb) = layer2.step(x, &state2, &sur);
        assert_eq!(vec_of(a), vec_of(b));
        assert_eq!(vec_of(na.w), vec_of(nb.w));
        assert_eq!(vec_of(na.bias.unwrap()), vec_of(nb.bias.unwrap()));
    }
}

#[test]
fn stdp_conv2d_rejects_inconsistent_records() {
    let layer = StdpConv2dConfig::new(2, 2, (2, 2), pair())
        .with_bias()
        .init();
    let state = StdpConv2dState::zeros(&layer, &dev(), 1, 3, 3);

    let mut rec = layer.to_record(&state);
    rec.bias = false; // the state carries bias, and the layer is written without it
    assert!(StdpConv2d::from_record(rec).is_err());

    let mut rec = layer.to_record(&state);
    rec.groups = 3; // does not share channels
    let err = StdpConv2d::from_record(rec).unwrap_err();
    assert!(matches!(err, SnnError::InvalidConfig { .. }), "{err:?}");

    let mut rec = layer.to_record(&state);
    rec.kernel_h = 3; // the shape of the scales diverges from the geometry
    assert!(StdpConv2d::from_record(rec).is_err());

    let mut rec = layer.to_record(&state);
    rec.bias_homeostasis = vec![0.1];
    assert!(StdpConv2d::from_record(rec).is_err());
}
