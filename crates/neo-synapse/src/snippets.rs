use serde::{Deserialize, Serialize};

/// A spike waveform cut from a recording across one or several channels.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct WaveformSnippet {
    pub primary_channel: usize,
    pub center_sample: u64,
    /// Sub-sample offset in `[-0.5, 0.5]` samples (0 when not estimated).
    pub subsample_offset: f32,
    /// Channel of each row of `waveform`.
    pub channel_ids: Vec<usize>,
    pub num_samples: usize,
    /// `[channels, samples]`, row-major.
    pub waveform: Vec<f32>,
}

impl WaveformSnippet {
    pub fn num_channels(&self) -> usize {
        self.channel_ids.len()
    }

    /// Samples of the primary channel (its row; the first row when it is not among `channel_ids`).
    pub fn primary_trace(&self) -> &[f32] {
        let row = self
            .channel_ids
            .iter()
            .position(|&c| c == self.primary_channel)
            .unwrap_or(0);
        self.waveform
            .get(row * self.num_samples..(row + 1) * self.num_samples)
            .unwrap_or(&[])
    }
}
