//! Spike input coding: rate, latency, delta, periodic, burst.

mod burst;
mod delta;
mod latency;
mod periodic;
mod rate;

pub use burst::burst_encode;
pub use delta::delta_encode;
pub use latency::latency_encode;
pub use periodic::periodic_encode;
pub use rate::rate_encode;
