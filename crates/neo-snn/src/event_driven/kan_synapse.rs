//! Delta-gated KAN synapse: recalculation only with |Δtrace| > δ.

use burn::module::Module;
use burn::tensor::Device;
use burn::tensor::Tensor;

use crate::kan::KANLinear;

/// Event-driven KAN with optional delta gating.
///
/// # Delta gating
///
/// With `use_delta_gating` enabled, the layer compares the current trace with
/// `last_trace` element by element. If `|trace - last_trace| <= delta` for a
/// batch/input element, its previous output is reused; otherwise the KAN is
/// recalculated for that element. The `delta` and `use_delta_gating` fields
/// therefore affect the result instead of being inert configuration.
///
/// # Gating is semantic, not computational
///
/// `forward` calls `self.kan.forward(trace)` unconditionally for the entire
/// batch and only afterwards selects which rows to take from the fresh result
/// and which to replace with `last_output`. For a batch of 1000 examples with
/// the gate open for only 100, the KAN still runs for all 1000 and 900 results
/// are discarded. A sparse implementation would first gather the active batch
/// indexes, call `KANLinear::forward` only on those rows, and scatter the result
/// back. That is not implemented: `KANLinear::forward` (like most Burn tensor
/// operations) expects tensors of a fixed, known shape, while the number of
/// active rows is known only at runtime. The event-driven semantics are
/// implemented and covered by tests; skipping the KAN computation itself for
/// inactive rows is a separate optimization that does not change the `forward`
/// signature.
#[derive(Module, Debug)]
pub struct EventDrivenKANSynapse {
    /// Internal KAN layer.
    pub kan: KANLinear,
    /// Trace change threshold, below which recalculation is skipped.
    pub delta: f32,
    /// Is delta gating enabled?
    pub use_delta_gating: bool,
}

impl EventDrivenKANSynapse {
    /// Create a layer.
    pub fn new(
        in_features: usize,
        out_features: usize,
        grid_size: usize,
        delta: f32,
        use_delta_gating: bool,
        device: &Device,
    ) -> Self {
        let kan = KANLinear::new(
            in_features,
            out_features,
            grid_size,
            3,
            [-2.0, 2.0],
            0.5,
            0.1,
            device,
        );
        Self {
            kan,
            delta,
            use_delta_gating,
        }
    }

    /// Direct passage.
    ///
    /// Returns `(output, reference_trace, reference_output)` - the last two
    /// are intended to be saved in [`crate::state::LayerState`] and
    /// transfer to the next step as `last_trace`/`last_output`.
    ///
    /// Reference values are updated ONLY for batch examples that have
    /// the gate has opened. If `last_trace` were rewritten every step,
    /// slow input drift (each step less than `delta`) never
    /// would accumulate to the threshold, and the output would forever be stuck at
    /// outdated meaning. The comparison is with the trace at the time of the LAST
    /// recalculation, and not with the trace of the previous step.
    pub fn forward(
        &self,
        trace: Tensor<2>,
        last_trace: Option<Tensor<2>>,
        last_output: Option<Tensor<2>>,
    ) -> (Tensor<2>, Tensor<2>, Tensor<2>) {
        let fresh = self.kan.forward(trace.clone());

        match (self.use_delta_gating, last_trace, last_output) {
            (true, Some(lt), Some(lo)) => {
                // The gate is computed PER INPUT (in_features), and is applied to
                // OUTPUT (out_features) - the dimensions are generally different,
                // so the gate is aggregated to one scalar for example
                // batch: "has the input of this example changed enough,
                // to recalculate KAN". max by characteristics - not mean -
                // because an event in ANY input channel must
                // lead to recalculation, and not blurred by averaging over
                // others who may not have changed at all.
                let changed = (trace.clone() - lt.clone())
                    .abs()
                    .greater_elem(self.delta)
                    .float()
                    .max_dim(1); // [batch, 1]
                let keep = changed.clone().mul_scalar(-1.0).add_scalar(1.0);

                let out = fresh.mul(changed.clone()) + lo.mul(keep.clone());

                // The reference trace moves only where the gate has opened.
                let new_ref_trace = trace.mul(changed) + lt.mul(keep);

                (out.clone(), new_ref_trace, out)
            }
            // First step (or gating off): reference values = current.
            _ => (fresh.clone(), trace, fresh),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use burn::tensor::Device;

    fn dev() -> Device {
        Device::flex()
    }

    fn t2(data: Vec<f32>, shape: [usize; 2]) -> Tensor<2> {
        Tensor::<1>::from_floats(data.as_slice(), &dev()).reshape(shape)
    }

    fn vec2(t: Tensor<2>) -> Vec<f32> {
        t.into_data().try_to_vec::<f32>().unwrap()
    }

    /// Gating contract (see "Known restriction" in the documentation
    /// module): with `use_delta_gating=true` and trace NOT gone beyond
    /// `delta` from the reference, the output must be EXACTLY `last_output`, even
    /// if the KAN inside was recalculated on the new `trace` - that is, closed
    /// the gate observably "masks" the result no matter what
    /// does `KANLinear::forward` under the hood (dense calculation or
    /// someday real sparse way). This fixes SEMANTICS,
    /// which the test can see - the very fact that `kan.forward`
    /// is called on the entire batch, of course, not by an external test
    /// observed and remains a documented limitation, not
    /// verifiable invariant.
    #[test]
    fn closed_gate_returns_exact_last_output_regardless_of_fresh_kan_result() {
        let device = dev();
        let synapse = EventDrivenKANSynapse::new(2, 2, 4, 0.5, true, &device);

        let last_trace = t2(vec![0.0, 0.0], [1, 2]);
        let last_output = t2(vec![0.42, -0.17], [1, 2]); // arbitrary "previous" value.

        // The new trace differs from the reference one by less than delta=0.5 -
        // the gate must remain closed.
        let trace = t2(vec![0.1, -0.05], [1, 2]);

        let (out, ref_trace, ref_output) =
            synapse.forward(trace, Some(last_trace.clone()), Some(last_output.clone()));

        let out = vec2(out);
        let want = vec2(last_output.clone());
        for (i, (&a, &b)) in out.iter().zip(want.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-6,
                "element {i}: a closed gate must return last_output unchanged: {a} vs {b}"
            );
        }

        // The reference trace/output should not have moved either.
        let ref_trace = vec2(ref_trace);
        let want_trace = vec2(last_trace);
        for (i, (&a, &b)) in ref_trace.iter().zip(want_trace.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-6,
                "ref_trace[{i}] should not have moved: {a} vs {b}"
            );
        }
        let ref_output = vec2(ref_output);
        for (i, (&a, &b)) in ref_output.iter().zip(want.iter()).enumerate() {
            assert!(
                (a - b).abs() < 1e-6,
                "ref_output[{i}] must match last_output: {a} vs {b}"
            );
        }
    }

    /// Open gate (change above `delta` in at least one channel)
    /// must recalculate the output via KAN and shift the reference trace to
    /// NEW meaning - otherwise "slow drift" would never
    /// fixed (see `forward` documentation).
    #[test]
    fn open_gate_updates_reference_trace_to_new_value() {
        let device = dev();
        let synapse = EventDrivenKANSynapse::new(2, 2, 4, 0.5, true, &device);

        let last_trace = t2(vec![0.0, 0.0], [1, 2]);
        let last_output = t2(vec![0.0, 0.0], [1, 2]);
        // Change 1.0 in the first channel exceeds delta=0.5 -> gate is open.
        let trace = t2(vec![1.0, 0.0], [1, 2]);

        let (_, ref_trace, _) = synapse.forward(trace.clone(), Some(last_trace), Some(last_output));

        let ref_trace = vec2(ref_trace);
        let want_trace = vec2(trace);
        for (&a, &b) in ref_trace.iter().zip(want_trace.iter()) {
            assert!(
                (a - b).abs() < 1e-6,
                "an open gate must shift ref_trace to a new trace: {a} vs {b}"
            );
        }
    }
}
