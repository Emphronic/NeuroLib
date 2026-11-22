# ANN → SNN conversion

`neo-snn::convert` — conversion of ready-made regular networks into spike networks (ANN → SNN). In the prelude
under the name `ann_to_snn`: `convert`, `Act`, `AnnLayer`, `AnnNet`, `BatchNormParams`, `ConvertConfig`,
`InputEncoding`, `LayerStats`, `NormMode`, `SnnOutput`, `SnnState`, `SpikeConfig`, `SpikingNet`,
`activation_scales`, `argmax`.

The conversion is configuration-based (`ConvertConfig`: activation type, encodings, norms — see `NormMode`,
`InputEncoding`, `SpikeConfig`) — see the crate for the exact fields. Input encoding for the SNN is via
[Spikes & encoding](../building-blocks/spikes.md) (`rate_encode` and friends).

Related: the inverse direction — STDP training of the resulting SNN — see [STDP](../building-blocks/stdp.md).
