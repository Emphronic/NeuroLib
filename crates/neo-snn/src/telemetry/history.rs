//! Metric ring buffer.

/// The history of a scalar metric with bounded length.
///
/// The training goes on in thousands of steps, the screen is eighty columns. Store all
/// history to show the last hundred values, there is no need: the buffer holds
/// exactly as much as may be needed for drawing, and old
/// values are supplanted.
#[derive(Debug, Clone)]
pub struct History {
    values: Vec<f32>,
    capacity: usize,
    total: usize,
    sum: f64,
    best: Option<f32>,
}

impl History {
    /// New buffer of specified capacity (minimum 1).
    pub fn new(capacity: usize) -> Self {
        Self {
            values: Vec::with_capacity(capacity.max(1)),
            capacity: capacity.max(1),
            total: 0,
            sum: 0.0,
            best: None,
        }
    }

    /// Add value. Non-numeric (`NaN`, `inf`) are ignored: one
    /// such a value would otherwise stretch the graph scale to infinity and
    /// would make the whole story unreadable.
    pub fn push(&mut self, value: f32) {
        if !value.is_finite() {
            return;
        }
        if self.values.len() == self.capacity {
            self.values.remove(0);
        }
        self.values.push(value);
        self.total += 1;
        self.sum += value as f64;
        self.best = Some(match self.best {
            Some(b) if b <= value => b,
            _ => value,
        });
    }

    /// Values are in order of arrival.
    pub fn values(&self) -> &[f32] {
        &self.values
    }

    /// Last value.
    pub fn last(&self) -> Option<f32> {
        self.values.last().copied()
    }

    /// Minimum for the entire time (not only in the buffer).
    pub fn best(&self) -> Option<f32> {
        self.best
    }

    /// All-time average.
    pub fn mean(&self) -> f32 {
        if self.total == 0 {
            0.0
        } else {
            (self.sum / self.total as f64) as f32
        }
    }

    /// How many values are taken over the entire time.
    pub fn total(&self) -> usize {
        self.total
    }

    /// Is the buffer empty?
    pub fn is_empty(&self) -> bool {
        self.values.is_empty()
    }

    /// Average of recent `n` values (less if history is shorter).
    ///
    /// `None` if the buffer is empty. Used where FRESH is important
    /// picture, and not the average for the entire training - for example, the current norm
    /// gradient, which a thousand steps ago says nothing about
    /// happening now.
    pub fn recent_mean(&self, n: usize) -> Option<f32> {
        if self.values.is_empty() {
            return None;
        }
        let start = self.values.len().saturating_sub(n.max(1));
        let tail = &self.values[start..];
        Some(tail.iter().sum::<f32>() / tail.len() as f32)
    }

    /// Curve direction: average of the second half of the buffer minus the average
    /// first, normalized to the average scale of values.
    ///
    /// A positive value means the metric is growing, a negative value means it is falling.
    /// Normalization to scale is important because both loss and accuracy
    /// different order of magnitude: without it, the same threshold “grows”
    /// would work differently on different metrics. `None`, if data
    /// less than four points - the trend for two or three values is too noisy,
    /// to draw conclusions from it.
    pub fn trend(&self) -> Option<f32> {
        if self.values.len() < 4 {
            return None;
        }
        let mid = self.values.len() / 2;
        let (first, second) = self.values.split_at(mid);
        let mean = |s: &[f32]| s.iter().sum::<f32>() / s.len() as f32;
        let a = mean(first);
        let b = mean(second);

        let scale = (a.abs() + b.abs()) / 2.0;
        if scale < f32::EPSILON {
            return Some(0.0);
        }
        Some((b - a) / scale)
    }

    /// The boundaries of the values in the buffer are the scale for the graph.
    pub fn bounds(&self) -> (f32, f32) {
        let mut lo = f32::INFINITY;
        let mut hi = f32::NEG_INFINITY;
        for &v in &self.values {
            lo = lo.min(v);
            hi = hi.max(v);
        }
        if !lo.is_finite() || !hi.is_finite() {
            return (0.0, 1.0);
        }
        if (hi - lo).abs() < f32::EPSILON {
            // Constant metric: flat line in the middle instead of division
            // to zero during normalization.
            (lo - 0.5, hi + 0.5)
        } else {
            (lo, hi)
        }
    }
}
