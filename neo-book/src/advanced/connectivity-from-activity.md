# Connectivity from activity

The paper [**Connectivity from activity in spiking neural networks**](https://arxiv.org/abs/2611.21592)
(arXiv:2611.21592, Wakhloo & Mizera, Columbia University, Nov 2026) — a theoretical work with
mathematical derivation and a numerical check on LIF networks. P0 — recommended: "the most
interesting for your project so far".

## What is it

The idea: a network of ~1000 neurons is observed — spike times, autocorrelations,
cross-correlations, frequency characteristics — without knowing the internal structure. Can the
synaptic connectivity be recovered? The authors show that under certain conditions — yes.

## The method — spectral connectivity

For two neurons \(i\) and \(j\) in a frequency regime the authors get the relation:

\[
C_{ij}(\omega)\approx
V_{ij}(\omega)C_{jj}(\omega)
+
C_{ii}(\omega)V_{ji}^{*}(\omega)
\]

Where:
- \(C_{ij}\) — the cross-spectral density (the observed cross-correlation of the activity);
- \(C_{ii},C_{jj}\) — the autocorrelations;
- \(V_{ij}, V_{ji}\) — the effective connections.

This lets you estimate the effective synaptic coupling using the observed activity. Indirect
paths through other neurons are accounted for — the relation is the linear response of the
network.

Numerical results (LIF networks): 0.86 correlation between the true and reconstructed
excitatory-connection strengths, 0.99 for inhibitory (network 784–1000–10, T=64).

Important limitation: a large network in stationary asynchronous activity is assumed — no
guarantee for arbitrary dynamic networks. It is a recovery of *effective* connectivity, not a
universal method for single-synapse proof of cause.

## In NeoLib — `structural::analysis`

```rust
use neurolib::snn::structural::analysis::{
    ActivityHistory, ConnectivityEstimate, SpectralConnectivity, validate_causal,
};
```

Pipeline — `Activity monitoring → Connectivity estimation → Candidate connections → Causal
validation → Rewire/prune/strengthen`.

### 1. Activity monitoring

```rust
let mut history = ActivityHistory::from_network(&net);
for x in steps {
    net.step(x)?;
    history.record_step(&net);  // the spikes of ALL neurons of the last step
}
```

[`ActivityHistory`](../../../crates/neo-snn/src/structural/analysis/mod.rs) — per step, one bit per
neuron (`ids` order — `Topology::neurons()`); spike trains are available via
`ActivityHistory::spike_train`.

### 2. Connectivity estimation — `SpectralConnectivity` (the paper's method)

```rust
use neurolib::snn::structural::analysis::{ConnectivityEstimator, SpectralConnectivity};

let estimates = SpectralConnectivity::default().estimate(&history);
// Vec<ConnectivityEstimate> — { source, target, estimated_weight, confidence }
```

The paper's relation \(C_{ij}(\omega)\approx V_{ij}(\omega)C_{jj}(\omega)+C_{ii}(\omega)V_{ji}^{*}(\omega)\)
is implemented as: for each ordered pair (i, j), the effective coupling \(V_{ij}\) is the
least-squares projection of the cross-correlation \(C_{ij}\) onto the target's autocorrelation
\(C_jj\) (the time-domain dual of the spectral relation); the confidence is the squared cosine
between the two.

### 3. Candidate connections — `ReconstructedConnectivity`

```rust
use neurolib::snn::structural::analysis::reconstruct;

let recon = reconstruct(estimates, net.topology().neurons().map(|n| n.id).collect());
let top = recon.top(5);                       // strongest 5
let strong = recon.with_confidence(0.8);      // above a threshold
```

### 4. Causal validation — temporarily disable and measure the behavior change

```rust
use neurolib::snn::structural::analysis::validate_causal;

let steps: Vec<Vec<f32>> = /* the episode inputs */;
let validation = validate_causal(&mut net, &top[0], &steps, episodes)?;
// validation.abs_output_change / .rate_with / .rate_without / .has_effect()
```

Disables the candidate connection (`Topology::disable_synapse`), runs the same episodes,
re-enables, and measures the change in output spikes.

### 5. Rewire / prune / strengthen — via the structural engine

The candidates become structural actions — see
[Structural plasticity](./structural-plasticity.md):

- **Strengthen**: `net.topology_mut().set_synapse_weight(id, weight)`;
- **Prune**: `net.topology_mut().remove_synapse(id)` — when the estimated connection is weak or
  the causal validation shows no effect;
- **Rewire**: `StructuralAction::AddSynapse { src, dst, weight, delay }` via
  `apply_action`/`grow_synapses` (see [Structural plasticity](./structural-plasticity.md)).

## Tests

Unit tests in the modules — `cargo test -p neo-snn --lib structural::analysis`:
`autocorr_counts_pairs`, `cross_corr_counts_pairs`, `estimates_all_pairs_sorted_by_confidence`,
`silent_neurons_give_zero_estimates`, `missing_candidate_is_default`, ...

## Notes

- The estimator is independent of the structural engine — it works on any `ActivityHistory`.
- `SpectralConnectivity::new(max_lag)` — the window of the correlation functions (default 100
  steps); larger — for slower dynamics.
- The estimation is of the *effective* connectivity — for direct proofs of cause use the causal
  validation stage (step 4).
