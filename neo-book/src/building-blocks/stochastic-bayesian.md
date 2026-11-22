# Stochastic Bayesian SNN

The paper **Device Stochasticity Directly Implements Bayesian Sampling in Spiking Neurons**
(Advanced Science, Nov 3 2026, Wiley Online Library) replaces the fixed threshold of an LIF neuron with a
*stochastic* threshold: the neuron draws its threshold from a distribution, so the Bayesian view of the
threshold lives in the neuron itself — the same input gives different spikes, and the distribution over spikes
reflects the uncertainty about the neuron's excitability.

The paper has two parts; both are in NeuroLib:

1. **Stochastic threshold** — [`StochasticThresholdLeaky`](../../../crates/neo-snn/src/neuron/stochastic_threshold.rs)
   (`neurolib::snn::StochasticThresholdLeaky`) — the neuron itself;
2. **Rate Estimation (RE)** — instead of modeling every spike during training, the probabilities of firing are
   propagated — [`rate_estimate`](../../../crates/neo-snn/src/probabilistic/rate_estimation.rs).

## The paper — key ideas

### Stochastic threshold — Bayesian sampling at the neuron level

The LIF update becomes:

```text
S(t) = H(V(t) − θ(t))       θ(t) ~ N(μ, σ) — drawn per neuron (or per timestep)
```

instead of the fixed `S(t) = H(V(t) − V_th)`. "For each neuron, parameters of the threshold distribution —
mean and variance — are trained." The distribution is the posterior over "how excitable is this neuron" —
that is the Bayesian part: running the stochastic neuron step-by-step *is* sampling from the posterior;
[`ThresholdPosterior`](#bayesian-inference) below gives the same posterior without running spikes.

Hardware analogue: the paper uses the physical stochasticity of an MTJ (magnetic tunnel junction) — one
physical device per neuron generates the threshold draws. Software:
[`RngSource`](../../../crates/neo-snn/src/probabilistic/rng.rs) — a seeded pseudo-random generator
(`rand` 0.8) — makes runs reproducible.

### Rate Estimation (RE) — probabilities instead of spikes

Instead of explicitly modeling each spike during training, RE propagates the probability of firing per step.
For the sum of independent Bernoulli input events the authors use the mean/variance approximation:

```text
μ        = P·W
σ²       = (P ⊙ (1−P))·(W ⊙ W)
P_spike  ≈ Φ((μ − θ)/(σ + ε))      — Φ via erf (Burn `Tensor::erf`)
```

where `P` — input spike probabilities, `W` — connection weights, `θ` — threshold, `ε` — a small guard
against `σ = 0`.

The paper's results (MNIST, network 784–1000–10, T=64): RE ≈ 5.05 s/epoch vs 112.71 s for surrogate-gradient
training — about 22× faster in that experiment; weight/input noise robustness checked; probabilistic
predictions evaluated with NLL and ECE.

## The NeuroLib API

```rust
use neurolib::snn::{
    neuron::{StochasticThresholdLeaky, StochasticThresholdLeakyConfig, ThresholdDistribution},
    spiking_layer::Session,
    probabilistic::{
        ProbabilityState, RateEstimate, RngSource, ThresholdPosterior,
        forward_seq_prob, rate_estimate, spike_nll, expected_calibration_error,
    },
    state::ResetMode,
    surrogate::Triangular,
};
```

### StochasticThresholdLeaky — the stochastic-threshold neuron

```rust
use neurolib::snn::neuron::{StochasticThresholdLeaky, StochasticThresholdLeakyConfig, ThresholdDistribution};
use neurolib::snn::state::ResetMode;
use burn::tensor::Device;

let config = StochasticThresholdLeakyConfig {
    beta_init: 0.9,
    distribution: ThresholdDistribution::new(1.0, 0.5), // θ ~ N(1.0, 0.5)
    sample_per_timestep: true,                          // re-draw per step (hardware analogue)
    seed: Some(42),                                     // reproducible runs
    reset_mode: ResetMode::Subtract,
};
let layer = StochasticThresholdLeaky::new(&config, &Device::flex());
```

- implements the `SpikingLayer` contract — `Session` works as for every layer:
```rust
let session = Session::new(layer, batch, &device);
let spikes = session.step(x_t, &Triangular::new(1.0));
```
- or the explicit-state API — `StochasticThresholdLeaky::step(x, &state, surrogate) -> (spikes, state)`;
- `state.rng` — the RNG state (reproducible runs); `state.threshold_sample` — the drawn θ.

Training through the stochastic neuron is like `Leaky`: the straight-through surrogate gradient flows
through the membrane. The threshold draws themselves are host-side (RNG) — not differentiable; the
differentiable path is RE + `ThresholdPosterior` (below).

### Rate Estimation — the probabilistic forward

```rust
use neurolib::snn::probabilistic::{forward_seq_prob, rate_estimate};

// inputs — the input spike probabilities P [batch, in]; weights — W [in, out]
let probs: Vec<ProbabilityState> = forward_seq_prob(&layer, inputs, &weights, 1e-6);
let est: RateEstimate = rate_estimate(&inputs[0], &weights, &threshold, 1e-6);
// est.mean, est.variance, est.prob_spike
```

`rate_estimate` is the paper's RE: `μ = P·W`, `σ² = (P⊙(1−P))·(W⊙W)`, `P_spike ≈ Φ((μ−θ)/(σ+ε))`.

### Bayesian inference — ThresholdPosterior

```rust
use neurolib::snn::probabilistic::ThresholdPosterior;

let posterior = ThresholdPosterior::from_layer(&layer, [batch, units], &device);
let thresholds = posterior.sample(&mut rng);          // draw θ ~ posterior — Bayesian sampling
let combined = ThresholdPosterior::combine(&a, &b);    // product of Gaussians
```

### Metrics

```rust
use neurolib::snn::probabilistic::{expected_calibration_error, spike_nll};

let nll = spike_nll(&prob, &spikes, 1e-6);                // NLL
let ece = expected_calibration_error(&prob, &spikes, 10); // ECE
```

## Tests

Unit tests live in the modules — e.g. `cargo test -p neo-snn --lib stochastic_threshold`:

- `probabilistic::rng::tests` — reproducible seeds, normal samples;
- `probabilistic::rate_estimation::tests` — the paper's formulas (`rate_estimate_matches_paper_formulas`:
  μ = 0.5, σ² = 0.25, Φ(−1) ≈ 0.1587);
- `probabilistic::probability_state::tests` — RE agrees with `rate_estimate`;
- `probabilistic::bayesian_inference::tests` — posterior sampling spread, product of Gaussians;
- `probabilistic::metrics::tests` — NLL of a half-prediction is `ln 2`, ECE of confident-but-wrong;
- `neuron::stochastic_threshold::tests` — same seed reproduces the run, per-step re-sampling, binary
  spikes, the `SpikingLayer` path.

## Notes

- The stochastic-threshold neuron is the "device stochasticity" of the paper — every step draws a fresh
  threshold per neuron, so the same input gives different spikes; with a fixed threshold you get the
  classic LIF raster (see [Neuron models](./neurons.md)).
- RE is most useful when inputs are Bernoulli spike probabilities — e.g. encoded inputs or the output of an
  upstream stochastic-threshold layer.
- Both parts are independent: you can run the stochastic neuron and train with surrogate (like `Leaky`), or
  train with RE probabilities and read them via [`ProbabilityState`](#rate-estimation--the-probabilistic-forward).
