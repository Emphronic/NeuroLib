# Inference

Read spikes out of a sequence with the functional API — `mean_over_time`, `sum_over_time`, `spike_counts`,
`population_reduce`, `raster` (`spike_raster`) — and measure quality with `accuracy_rate`, `accuracy_temporal`,
`population_accuracy_rate`, `first_spike_times`.

Quantized inference: [`ExecutionPolicy::Fused`](../building-blocks/quantization.md) reads directly from packed
weights (no matrix expansion); `Dequantize` expands the weights to a tensor and uses the normal matmul.

State quantization of the membrane: [State quantization](../advanced/state-quantization.md).
