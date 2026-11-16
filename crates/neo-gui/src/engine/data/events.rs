//! Spike events for the timeline event track.
//!
//! Events come only from a sorting the user opens in Curation (or a spike extraction run with
//! their own parameters); nothing is detected when a recording opens. Until then the store is
//! empty.

/// Collection of spike events across channels.
#[derive(Debug, Clone, Default)]
pub struct SpikeEventStore {
    /// All event times, sorted ascending.
    #[cfg_attr(not(test), allow(dead_code))]
    pub times_sec: Vec<f64>,
    /// Per-channel sorted event times (for windowed lookups while rendering).
    by_channel: Vec<Vec<f64>>,
}

impl SpikeEventStore {
    #[cfg_attr(not(test), allow(dead_code))]
    pub fn len(&self) -> usize {
        self.times_sec.len()
    }

    /// Number of events detected on `channel`.
    pub fn count(&self, channel: usize) -> usize {
        self.by_channel.get(channel).map_or(0, Vec::len)
    }

    /// Event times of `channel` within `[t0, t1]`, found by binary search.
    pub fn in_window(&self, channel: usize, t0: f64, t1: f64) -> &[f64] {
        let Some(times) = self.by_channel.get(channel) else {
            return &[];
        };
        let lo = times.partition_point(|&t| t < t0);
        let hi = times.partition_point(|&t| t <= t1);
        &times[lo..hi.max(lo)]
    }

    /// The spikes of a sorting, each on the recording channel where its cluster's template is
    /// largest (the cluster's best channel). Spikes of every cluster count, whatever its group.
    pub fn from_sorting(data: &crate::engine::curation::SortingData) -> Self {
        let rate = data.sample_rate().max(f64::MIN_POSITIVE);
        let s = &data.sorting;
        let channels = s
            .channel_map
            .iter()
            .copied()
            .max()
            .map_or(0, |m| m + 1)
            .max(s.channels());
        let mut by_channel: Vec<Vec<f64>> = vec![Vec::new(); channels];
        let home: std::collections::BTreeMap<_, _> = data
            .clusters
            .keys()
            .map(|&c| {
                (
                    c,
                    s.channel_map
                        .get(data.best_channel(c))
                        .copied()
                        .unwrap_or(data.best_channel(c)),
                )
            })
            .collect();
        for (&t, c) in s.spike_times.iter().zip(&s.spike_clusters) {
            if let Some(list) = home.get(c).and_then(|&ch| by_channel.get_mut(ch)) {
                list.push(t as f64 / rate);
            }
        }
        by_channel
            .iter_mut()
            .for_each(|v| v.sort_by(f64::total_cmp));
        let mut all: Vec<f64> = by_channel.iter().flatten().copied().collect();
        all.sort_by(f64::total_cmp);
        Self {
            times_sec: all,
            by_channel,
        }
    }

    /// Negative threshold crossings per channel (`-4.5 * sigma_n`, with sigma_n = median(|x|) /
    /// 0.6745; local-minimum peak; 2 ms spacing: the deeper of two troughs closer than that
    /// stays), reading whole channels. Test fixture only: the app never detects on its own;
    /// user-run extraction replaces it.
    #[cfg(test)]
    pub fn detect(source: &dyn neo_view::SignalBackend) -> Self {
        const THRESHOLD_FACTOR: f32 = 4.5;
        const REFRACTORY_SEC: f64 = 0.002;
        let info = source.info();
        let sample_rate = info.sample_rate_hz();
        let refractory = (sample_rate * REFRACTORY_SEC) as usize;
        let samples = info.samples as usize;

        let mut by_channel = Vec::with_capacity(info.channel_count());
        let mut all: Vec<f64> = Vec::new();
        let mut trace = vec![0.0f32; samples];

        for ch in 0..info.channel_count() {
            if source.read(&[ch], 0..samples as u64, &mut trace).is_err() {
                by_channel.push(Vec::new());
                continue;
            }
            // Channels are detected independently so the refractory state never leaks across them
            let mut abs: Vec<f32> = trace.iter().map(|v| v.abs()).collect();
            let mid = abs.len() / 2;
            let sigma_n = if abs.is_empty() {
                0.0
            } else {
                *abs.select_nth_unstable_by(mid, f32::total_cmp).1 / 0.6745
            };
            let threshold = -THRESHOLD_FACTOR * sigma_n;
            let mut kept: Vec<usize> = Vec::new();
            for i in 0..samples {
                let x = trace[i];
                let is_min = x < threshold
                    && (i == 0 || x <= trace[i - 1])
                    && (i + 1 == samples || x < trace[i + 1]);
                if !is_min {
                    continue;
                }
                match kept.last().copied() {
                    Some(last) if i - last < refractory => {
                        if x < trace[last] {
                            *kept.last_mut().unwrap() = i;
                        }
                    }
                    _ => kept.push(i),
                }
            }
            let times: Vec<f64> = kept.iter().map(|&i| i as f64 / sample_rate).collect();
            all.extend_from_slice(&times);
            by_channel.push(times);
        }

        all.sort_by(f64::total_cmp);
        Self {
            times_sec: all,
            by_channel,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::engine::data::Dataset;

    #[test]
    fn test_spike_event_detection() {
        let sample_rate = 10_000.0;
        let samples = 2_000;
        let mut data = vec![1.0f32; 2 * samples];

        // Inject two clear spikes on ch 0 separated by > 2ms (20 samples)
        data[500] = -100.0; // Within 2 ms of a deeper trough -> dropped
        data[505] = -120.0;
        data[600] = -100.0; // More than 2 ms away -> detected

        // Inject one clear spike on ch 1
        data[samples + 1000] = -110.0;
        let ds = Dataset::from_samples("test", data, 2, sample_rate);

        let store = SpikeEventStore::detect(ds.signal().as_ref());
        assert_eq!(store.len(), 3);
        assert!((store.times_sec[0] - 0.0505).abs() < 1e-6);
        assert!((store.times_sec[1] - 0.060).abs() < 1e-6);
        assert!((store.times_sec[2] - 0.100).abs() < 1e-6);

        assert_eq!(store.in_window(0, 0.0, 1.0), &[0.0505, 0.060]);
        assert_eq!(store.in_window(1, 0.0, 1.0), &[0.100]);
        assert_eq!(store.in_window(0, 0.055, 1.0), &[0.060]);
        assert_eq!(store.in_window(1, 0.0, 0.05), &[] as &[f64]);
        assert_eq!(store.in_window(5, 0.0, 1.0), &[] as &[f64]);
    }
}
