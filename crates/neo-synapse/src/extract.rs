//! Spike waveforms read from a recording.

use neo_core::RecordingSource;

use crate::snippets::WaveformSnippet;

/// Reads the window `[center − pre, center + post)` on `channels` around each of `centers` from a
/// recording (files read in place, chunk by chunk, not loaded). Spikes whose window leaves the
/// recording, or whose read fails, are skipped. With `subtract_mean`, each channel's mean over the
/// window is removed (raw recordings carry per-channel offsets). `primary_channel` is
/// `channels[0]`.
pub fn read_snippets(
    source: &dyn RecordingSource,
    centers: &[u64],
    channels: &[usize],
    pre_samples: usize,
    post_samples: usize,
    subtract_mean: bool,
) -> Vec<WaveformSnippet> {
    let total = source.info().samples;
    let len = pre_samples + post_samples;
    if channels.is_empty() || len == 0 {
        return Vec::new();
    }
    let mut buf = vec![0.0f32; channels.len() * len];
    let mut out = Vec::with_capacity(centers.len());
    for &center in centers {
        let Some(start) = center.checked_sub(pre_samples as u64) else {
            continue;
        };
        if start + len as u64 > total
            || source
                .read(channels, start..start + len as u64, &mut buf)
                .is_err()
        {
            continue;
        }
        let mut waveform = buf.clone();
        if subtract_mean {
            for row in waveform.chunks_exact_mut(len) {
                let mean = row.iter().sum::<f32>() / len as f32;
                row.iter_mut().for_each(|v| *v -= mean);
            }
        }
        out.push(WaveformSnippet {
            primary_channel: channels[0],
            center_sample: center,
            subsample_offset: 0.0,
            channel_ids: channels.to_vec(),
            num_samples: len,
            waveform,
        });
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use neo_core::MemoryRecording;

    #[test]
    fn snippets_from_a_recording() {
        // 2 channels, 100 samples: channel c, sample t holds 10c + t
        let data: Vec<f32> = (0..2)
            .flat_map(|c| (0..100).map(move |t| (10 * c + t) as f32))
            .collect();
        let rec = MemoryRecording::new("r", data, 2, 1000.0).unwrap();
        let s = read_snippets(&rec, &[1, 50, 98], &[1, 0], 2, 3, false);
        assert_eq!(s.len(), 1, "windows leaving the recording are skipped");
        assert_eq!(s[0].channel_ids, vec![1, 0]);
        assert_eq!(&s[0].waveform[..5], &[58.0, 59.0, 60.0, 61.0, 62.0]);
        assert_eq!(&s[0].waveform[5..], &[48.0, 49.0, 50.0, 51.0, 52.0]);
        let centered = read_snippets(&rec, &[50], &[0], 2, 3, true);
        assert_eq!(centered[0].waveform, vec![-2.0, -1.0, 0.0, 1.0, 2.0]);
    }
}
