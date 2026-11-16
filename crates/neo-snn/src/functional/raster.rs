//! A text spike raster.
//!
//! An analog of `snntorch.spikeplot.raster`, but without a plotting library:
//! the functions return a `String` without drawing anywhere or opening
//! anything.
//!
//! ## Why this way and not through a plotting crate
//!
//! A spike raster is a "neuron × time" table filled with zeros and ones. It
//! needs no axes with ticks, no antialiasing, and no color scale: exactly one
//! bit per cell is enough. A text grid conveys that bit completely and can be
//! printed anywhere — to a terminal, a log, a test, a file.
//!
//! A module with no I/O also does not impose a graphics dependency or terminal
//! control on the crate: where to put it is up to the calling code. A live
//! graphical training raster is the Training tab in `neo-gui`
//! (see [`crate::telemetry`]).
//!
//! The backend must have `f32` as its element type.

use burn::tensor::Tensor;

/// Characters used to draw the raster.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RasterStyle {
    /// Spike character.
    pub on: char,
    /// Silence character.
    pub off: char,
}

impl Default for RasterStyle {
    fn default() -> Self {
        Self {
            on: '█', off: '·'
        }
    }
}

impl RasterStyle {
    /// ASCII only — for terminals and logs where Unicode breaks the layout.
    pub fn ascii() -> Self {
        Self { on: '#', off: '.' }
    }
}

/// Raster of a single sample from the batch: a row per neuron, a column per
/// time step.
///
/// `spikes` — `[batch, num_steps, units]`, `sample` — sample index.
/// A cell counts as a spike when its value is greater than 0.5.
///
/// Rows are NOT numbered and there is no header — this is a raw grid that is
/// convenient to frame in any way. Numbering is added by
/// [`spike_raster_labeled`].
///
/// # Example
///
/// ```ignore
/// let train = rate_encode(data, 16, &device);
/// println!("{}", spike_raster(train, 0, RasterStyle::default()));
/// ```
pub fn spike_raster(spikes: Tensor<3>, sample: usize, style: RasterStyle) -> String {
    let [batch, steps, units] = spikes.dims();
    assert!(
        sample < batch,
        "spike_raster: sample {sample} outside batch of size {batch}"
    );

    let flat = crate::error::or_panic(crate::error::tensor_to_vec(spikes, "spike_raster"));

    let base = sample * steps * units;
    let mut out = String::with_capacity(units * (steps + 1));

    for u in 0..units {
        for t in 0..steps {
            let v = flat[base + t * units + u];
            out.push(if v > 0.5 { style.on } else { style.off });
        }
        if u + 1 < units {
            out.push('\n');
        }
    }
    out
}

/// The same raster, but with neuron numbers on the left and firing rates on
/// the right.
///
/// The rate is the fraction of steps on which the neuron fired. It is not here
/// for decoration: it immediately shows a silent layer (all zeros) and a layer
/// firing at every step (all ones) — the two degenerations for which
/// [`reg`](super::reg) exists.
pub fn spike_raster_labeled(spikes: Tensor<3>, sample: usize, style: RasterStyle) -> String {
    let [batch, steps, units] = spikes.dims();
    assert!(
        sample < batch,
        "spike_raster_labeled: sample {sample} outside batch of size {batch}"
    );

    let flat = crate::error::or_panic(crate::error::tensor_to_vec(spikes, "spike_raster_labeled"));

    let base = sample * steps * units;
    let width = units.saturating_sub(1).to_string().len().max(1);
    let mut out = String::new();

    for u in 0..units {
        let mut fired = 0usize;
        let mut row = String::with_capacity(steps);
        for t in 0..steps {
            let v = flat[base + t * units + u];
            if v > 0.5 {
                fired += 1;
                row.push(style.on);
            } else {
                row.push(style.off);
            }
        }
        let rate = if steps == 0 {
            0.0
        } else {
            fired as f32 / steps as f32
        };
        out.push_str(&format!("{u:>width$} │{row}│ {rate:.2}\n"));
    }
    out
}

/// Histogram of total activity over time steps.
///
/// Shows HOW MANY neurons fired at each step. It catches what is not visible
/// in the per-neuron raster: layer synchronization, when the whole population
/// fires in bursts instead of encoding the input.
///
/// `width` — length of the longest bar in characters.
pub fn activity_histogram(spikes: Tensor<3>, sample: usize, width: usize) -> String {
    let [batch, steps, units] = spikes.dims();
    assert!(
        sample < batch,
        "activity_histogram: sample {sample} outside batch of size {batch}"
    );

    let flat = crate::error::or_panic(crate::error::tensor_to_vec(spikes, "activity_histogram"));

    let base = sample * steps * units;
    let mut counts = Vec::with_capacity(steps);
    for t in 0..steps {
        let mut n = 0usize;
        for u in 0..units {
            if flat[base + t * units + u] > 0.5 {
                n += 1;
            }
        }
        counts.push(n);
    }

    let peak = counts.iter().copied().max().unwrap_or(0).max(1);
    let step_width = steps.saturating_sub(1).to_string().len().max(1);

    let mut out = String::new();
    for (t, n) in counts.iter().enumerate() {
        let len = n * width / peak;
        out.push_str(&format!(
            "t{t:>step_width$} │{:<width$}│ {n}/{units}\n",
            "█".repeat(len)
        ));
    }
    out
}
