//! Prelude: all public crate types and functions needed in everyday life
//! work.
//!
//! ```ignore
//! use neo_snn::prelude::*;
//! ```
//!
//! The completeness of the prelude is checked by the `tests/prelude.rs` test: it reads
//! `pub use` from root modules and requires each name to be
//! re-exported here (or explicitly added to the test exception list).
//!
//! `Result` crate is deliberately NOT re-exported: glob-import would obscure
//! `std::result::Result` from the user. Use [`crate::Result`].

pub use crate::attention::{
    AttentionKind, SpikeDrivenSelfAttention, SpikeDrivenSelfAttentionConfig, SpikeSequence,
    SpikingAttentionVariant, SpikingMlp, SpikingMlpConfig, SpikingSelfAttention,
    SpikingSelfAttentionConfig, SpikingTransformerBlock, SpikingTransformerBlockConfig,
};
pub use crate::bntt::{BatchNormTT1d, BatchNormTT2d};
/// Conversion ANN → SNN (`neo_snn::convert::convert`); in prelude under the name
/// `ann_to_snn` so that the glob import does not take the too generic name `convert`.
pub use crate::convert::convert as ann_to_snn;
pub use crate::convert::{
    Act, AnnLayer, AnnNet, BatchNormParams, ConvertConfig, InputEncoding, LayerStats, NormMode,
    SnnOutput, SnnState, SpikeConfig, SpikingNet, activation_scales, argmax,
};
pub use crate::encoding::{
    burst_encode, delta_encode, latency_encode, periodic_encode, rate_encode,
};
pub use crate::error::SnnError;
pub use crate::event_driven::{
    EventDrivenFullSKAN, EventDrivenKANSynapse, EventDrivenSpikingKANLayer,
};
pub use crate::functional::{
    ActivityStats, FnTrace, Identity, Monitor, NoWeightDependence, PopulationReduction,
    RasterStyle, TraceNonlinearity, WeightBound, WeightDependence, accuracy_rate,
    accuracy_temporal, activity_histogram, apply_reward, ce_count_loss, ce_max_membrane_loss,
    ce_rate_loss, ce_temporal_loss, clip_weights, eligibility_trace_update, first_spike_times,
    grad_norm, l1_spike_rate, l2_spike_rate, lateral_inhibition, lateral_inhibition_soft,
    max_over_time, mean_over_time, mse_count_loss, mse_membrane_loss, mse_temporal_loss,
    pair_update, pair_update_with_f, pair_update_with_f_and_dep, population_accuracy_rate,
    population_accuracy_temporal, population_ce_count_loss, population_ce_max_membrane_loss,
    population_ce_rate_loss, population_rates, population_reduce, population_reduce_seq,
    population_spike_counts, rate_bounds_penalty, spike_counts, spike_raster, spike_raster_labeled,
    sum_over_time, time_slice, triplet_update, triplet_update_with_f,
    triplet_update_with_f_and_dep, update_trace, winner_take_all,
};
pub use crate::kan::{KANLinear, KANLinearConfig};
pub use crate::layers::{
    BiasHomeostasis, ConvLstmState, DelayKernel, DelayState, RECORD_VERSION, SConv2dLSTM,
    SpikingConv2d, StdpConv1d, StdpConv1dConfig, StdpConv1dRecord, StdpConv1dRule, StdpConv1dState,
    StdpConv2d, StdpConv2dConfig, StdpConv2dRecord, StdpConv2dState, StdpHyper, StdpLinear,
    StdpLinearConfig, StdpLinearRecord, StdpRule, StdpState, SynapticDelay, SynapticDelayConfig,
};
pub use crate::neuron::ThresholdDistribution;
pub use crate::neuron::{
    AdEx, AdExConfig, AdExPreset, AdExState, AdaptiveLeaky, AdaptiveLeakyConfig, Alpha,
    GradedLeaky, GradedLeakyConfig, HodgkinHuxley, HodgkinHuxleyConfig, HodgkinHuxleyState,
    Izhikevich, IzhikevichConfig, IzhikevichPreset, IzhikevichState, Lapicque, Leaky, LeakyConfig,
    LeakyParallel, MultiStepLeaky, MultiStepRLeaky, RLeaky, RSynaptic, Synaptic,
    apply_graded_spikes, expand_levels, hh_rates, hh_rates_f32, hh_steady_state,
    hh_steady_state_as, multi_level_spike, spike_with_scalar_threshold, spike_with_threshold,
    split_ternary, split_time,
};
pub use crate::neuron::{
    StochasticThresholdLeaky, StochasticThresholdLeakyConfig, StochasticThresholdState,
};
pub use crate::probabilistic::{
    ProbabilityState, RateEstimate, RngSource, ThresholdPosterior, expected_calibration_error,
    forward_seq_prob, rate_estimate, spike_nll,
};
pub use crate::quant::{StateQuant, StateQuantScheme};
pub use crate::reservoir::{
    EchoStateNetwork, EchoStateNetworkConfig, LiquidFeature, LiquidState, LiquidStateMachine,
    LiquidStateMachineConfig, Pooling, RidgeReadout, accuracy, estimate_spectral_radius,
    flatten_time, nmse, one_hot, pool_time,
};
pub use crate::slstm::SLSTM;
pub use crate::spiking_layer::{
    Deferred, RecurrentState, Session, SpikingLayer, SpikingLayerExt, StepInput,
};
pub use crate::state::{
    LayerState, NeuronState, ResetMode, ResetOrder, apply_reset, apply_reset_tensor,
};
/// `Stateful`/`StatefulRecurrent` - optional "init_hidden" mode, see.
/// module documentation [`crate::stateful`]. `Detach` - auxiliary
/// trait, only needed so that `Stateful::detach_hidden` is available without
/// explicit `use neo_snn::stateful::Detach`.
pub use crate::stateful::{Detach, Stateful, StatefulRecurrent};
pub use crate::structural::analysis::{
    ActivityHistory, CausalValidation, ConnectivityEstimate, ConnectivityEstimator,
    ReconstructedConnectivity, SpectralConnectivity, validate_causal,
};
pub use crate::surrogate::{
    ATan, FastSigmoid, Heaviside, LeakySpikeOperator, SigmoidSurrogate, SparseFastSigmoid,
    SpikeOperator, SpikeRateEscape, StochasticSpikeOperator, StraightThrough, Surrogate,
    Triangular,
};
pub use crate::temporal::{
    DualMemoryConfig, DualMemoryLayer, DualMemoryState, FastState, MemoryFeedback, SlowMemory,
    SlowMemoryConfig, SlowMemoryState,
};
