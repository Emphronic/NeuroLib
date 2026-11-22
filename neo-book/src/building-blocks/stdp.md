# STDP

`StdpLinear` — fully connected `Linear → LIF` layer trained by STDP instead of backprop. Rules (`StdpRule`):

- `Pair { a_plus, a_minus, tau_pre, tau_post }` — classic pair-based STDP: `a_plus`/`a_minus` potentiation/depression
  coefficients, `tau_pre`/`tau_post` constant attenuation (in steps) of the pre-/post-synaptic traces;
- `Triplet { a2_plus, a2_minus, a3_plus, a3_minus, tau_pre, tau_post, tau_pre_slow, tau_post_slow }` — frequency-dependent
  plasticity: two fast traces + one slow trace, triplet coefficients;
- `RewardModulated { a_plus, a_minus, tau_pre, tau_post, tau_e }` — R-STDP: co-activity accumulates into an
  eligibility trace with attenuation `tau_e`; weights change only when `apply_reward` is called with a scalar reward.

Plasticity math: `neo-snn::functional::plasticity` — `pair_update`, `triplet_update`, `eligibility_trace_update`,
`update_trace`, `apply_reward`.

## How it differs from other layers

- All other layers train through Burn `Param` + backprop with surrogate gradients. `StdpLinear` is the intentional
  exception: its weights are the usual `Tensor<2>`, part of the **explicit state** (`StdpState`), not the `Module`
  parameter.
- Weight update is a local co-activity rule, called at each step explicitly — there is no optimizer pass.
- `StdpLinear` does NOT implement `burn::module::Module` — nothing to wrap in the module tree; `step`/`forward_seq`
  return output AND the new state.

## Layers

- `StdpLinear` (fully connected), `StdpConv1d`, `StdpConv2d` — 1D/2D spiking convs trained by STDP
  (`StdpConv1dConfig/State/Rule`, `StdpConv2dConfig/State/Rule`, `BiasHomeostasis` for conv2d);
- `SpikingConv2d` — spiking conv (backprop, not STDP);
- `SConv2dLSTM` — spiking conv LSTM.

Online and sequence modes: `StdpLinear::step` — one step with explicit state (on-chip training); `StdpLinear::
forward_sequence` — run `[batch, time, in_features]`, STDP updates accumulated inside.

Saving: `StdpLinearRecord` — STDP layers saved via the standard Burn `Record` (see
[Weight storage](./storage.md)).
