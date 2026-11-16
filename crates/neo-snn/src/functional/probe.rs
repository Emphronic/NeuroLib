//! Activity and gradient monitors for debugging training.
//!
//! ## How this differs from PyTorch hooks
//!
//! In PyTorch a monitor is attached to a module with a hook and fires on its
//! own. In burn, modules are ordinary structs with no hook mechanism; there is
//! nothing to intercept: `forward` calls your code, and you cannot insert an
//! observer from outside. That is why the monitor here is EXPLICIT — you pass
//! it a tensor at the point in the loop that interests you.
//!
//! This is one line more verbose, but more honest: you can see exactly what is
//! measured and when, and there is no hidden GPU→CPU synchronization in an
//! unexpected place. Synchronization happens exactly when [`Monitor::record`]
//! is called.
//!
//! ## What to watch
//!
//! * the firing rate drifted to 0 — the layer went silent, there is no
//!   gradient beyond it;
//! * the rate drifted to 1 — spikes stopped encoding anything;
//! * the gradient norm drops by an order of magnitude from layer to layer —
//!   depth-wise attenuation, time to change the surrogate (see
//!   [`FastSigmoid`](crate::surrogate::FastSigmoid)) or enable
//!   [`BatchNormTT1d`](crate::bntt::BatchNormTT1d).
//!
//! The backend must have `f32` as its element type.

use burn::tensor::Tensor;

/// Summary over a single tensor.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct ActivityStats {
    /// Mean value. For spikes this is the firing rate.
    pub mean: f32,
    /// Maximum absolute value — catches diverging membranes.
    pub abs_max: f32,
    /// Fraction of exactly zero elements: sparsity.
    pub zero_fraction: f32,
    /// Number of elements the summary was computed over.
    pub count: usize,
}

impl ActivityStats {
    /// Compute a summary over a tensor of any rank.
    #[track_caller]
    pub fn of<const D: usize>(tensor: Tensor<D>) -> Self {
        crate::error::or_panic(Self::try_of(tensor))
    }

    /// [`ActivityStats::of`] without panicking: tensor data that cannot be
    /// read as `f32` yields [`SnnError::Data`](crate::SnnError::Data).
    pub fn try_of<const D: usize>(tensor: Tensor<D>) -> crate::error::Result<Self> {
        let flat_vec = crate::error::tensor_to_vec(tensor, "ActivityStats")?;
        let flat: &[f32] = &flat_vec;

        let count = flat.len();
        if count == 0 {
            return Ok(Self {
                mean: 0.0,
                abs_max: 0.0,
                zero_fraction: 0.0,
                count: 0,
            });
        }

        let mut sum = 0.0f64;
        let mut abs_max = 0.0f32;
        let mut zeros = 0usize;
        for &v in flat {
            sum += v as f64;
            if v.abs() > abs_max {
                abs_max = v.abs();
            }
            if v == 0.0 {
                zeros += 1;
            }
        }

        Ok(Self {
            mean: (sum / count as f64) as f32,
            abs_max,
            zero_fraction: zeros as f32 / count as f32,
            count,
        })
    }
}

/// Accumulator of summaries over time steps or training iterations.
///
/// ```ignore
/// let mut mon = Monitor::new("hidden");
/// for t in 0..num_steps {
///     let (spikes, state) = layer.step(x, &state, &surrogate);
///     mon.record(spikes.clone());
/// }
/// println!("{}: rate {:.3}", mon.name(), mon.mean_rate());
/// ```
#[derive(Debug, Clone, Default)]
pub struct Monitor {
    name: String,
    history: Vec<ActivityStats>,
}

impl Monitor {
    /// A new monitor with a name that will appear in the log.
    pub fn new(name: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            history: Vec::new(),
        }
    }

    /// Monitor name.
    pub fn name(&self) -> &str {
        &self.name
    }

    /// Record the next tensor. This is where data is read from the device.
    pub fn record<const D: usize>(&mut self, tensor: Tensor<D>) {
        self.history.push(ActivityStats::of(tensor));
    }

    /// All accumulated summaries in order.
    pub fn history(&self) -> &[ActivityStats] {
        &self.history
    }

    /// Number of accumulated records.
    pub fn len(&self) -> usize {
        self.history.len()
    }

    /// Whether there are no records.
    pub fn is_empty(&self) -> bool {
        self.history.is_empty()
    }

    /// Mean firing rate over all records.
    ///
    /// Zero if there are no records.
    pub fn mean_rate(&self) -> f32 {
        if self.history.is_empty() {
            return 0.0;
        }
        let s: f64 = self.history.iter().map(|h| h.mean as f64).sum();
        (s / self.history.len() as f64) as f32
    }

    /// Maximum `abs_max` over all records — whether anything is diverging.
    pub fn peak(&self) -> f32 {
        self.history.iter().fold(0.0f32, |a, h| a.max(h.abs_max))
    }

    /// Clear the history while keeping the name.
    pub fn clear(&mut self) {
        self.history.clear();
    }
}

/// L2 norm of a tensor, computed on the host.
///
/// Applied to gradients AFTER `backward()`: burn returns them as ordinary
/// tensors through `GradientsParams`, so a separate hook mechanism is not
/// needed for this.
#[track_caller]
pub fn grad_norm<const D: usize>(grad: Tensor<D>) -> f32 {
    crate::error::or_panic(try_grad_norm(grad))
}

/// [`grad_norm`] without panicking (see [`ActivityStats::try_of`]).
pub fn try_grad_norm<const D: usize>(grad: Tensor<D>) -> crate::error::Result<f32> {
    let flat = crate::error::tensor_to_vec(grad, "grad_norm")?;
    let sum: f64 = flat.iter().map(|&v| (v as f64) * (v as f64)).sum();
    Ok(sum.sqrt() as f32)
}
