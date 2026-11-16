//! Spike LSTM: An LSTM whose output passes through a spike (surrogate) nonlinearity.
//! Input `[batch, seq_len, input_size]` → spike output `[batch, seq_len, hidden_size]`.

use burn::module::Module;
use burn::nn::{Lstm, LstmConfig};
use burn::tensor::Device;
use burn::tensor::Tensor;

use crate::surrogate::Surrogate;

/// Spike LSTM: LSTM with spike nonlinearity on hidden output
/// every step of time.
#[derive(Module, Debug)]
pub struct SLSTM {
    /// Internal LSTM.
    pub lstm: Lstm,
    /// Spike threshold.
    pub threshold: f32,
}

impl SLSTM {
    /// Create a layer.
    pub fn new(d_input: usize, d_hidden: usize, threshold: f32, device: &Device) -> Self {
        let lstm = LstmConfig::new(d_input, d_hidden, true).init(device);
        Self { lstm, threshold }
    }

    /// Forward pass: LSTM then spike on the hidden output of each step.
    /// Input `[batch, seq_len, d_input]`. Output `[batch, seq_len, d_hidden]`
    /// (spikes).
    pub fn forward<S: Surrogate + ?Sized>(&self, x: Tensor<3>, surrogate: &S) -> Tensor<3> {
        // In burn 0.21 LstmState became rank 2 (was 3 in 0.14) - signature
        // forward is synchronized automatically by the burn::nn::Lstm type,
        // here it is enough to pass None for the initial state.
        let (out, _state) = self.lstm.forward(x, None);
        let [batch, seq, hidden] = out.dims();

        let flat = out.reshape([batch * seq, hidden]);
        let x = flat.sub_scalar(self.threshold);

        let hard = x.clone().greater_elem(0.0).float().detach();
        let soft = surrogate.surrogate_forward(x);
        let spiked_flat = hard.clone() + (soft.clone() - soft.detach());

        spiked_flat.reshape([batch, seq, hidden])
    }
}
