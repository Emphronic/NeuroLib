//! Composite layers (Conv+LIF, spiking ConvLSTM, synaptic delay,
//! STDP-trained linear layer, etc.).

/// Saving/loading STDP layers via the standard Burn `Record`.
pub mod record;
mod sconv2dlstm;
mod spiking_conv2d;
mod stdp_conv1d;
mod stdp_conv2d;
mod stdp_linear;
mod synaptic_delay;

pub use record::{RECORD_VERSION, StdpConv1dRecord, StdpConv2dRecord, StdpHyper, StdpLinearRecord};
pub use sconv2dlstm::{ConvLstmState, SConv2dLSTM};
pub use spiking_conv2d::SpikingConv2d;
pub use stdp_conv1d::{StdpConv1d, StdpConv1dConfig, StdpConv1dRule, StdpConv1dState};
pub use stdp_conv2d::{BiasHomeostasis, StdpConv2d, StdpConv2dConfig, StdpConv2dState};
pub use stdp_linear::{StdpLinear, StdpLinearConfig, StdpRule, StdpState};
pub use synaptic_delay::{DelayKernel, DelayState, SynapticDelay, SynapticDelayConfig};
