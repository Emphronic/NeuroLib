# Basic Workflow: From Spikes to Inference

The typical flow of a NeuroLib program:

1. **Model** — choose and configure spiking layers or composite layers ([Model](./model.md));
2. **Data** — open a recording or encode input into spikes ([Data](./data.md));
3. **Training** — run a loop: forward step with surrogate, losses, STDP updates, telemetry ([Training](./training.md));
4. **Backend** — pick the compute runtime: `wgpu` / `cuda` / `rocm` / `cpu` ([Backend](./backend.md));
5. **Inference** — read spikes, decode to rates/labels, measure quality ([Inference](./inference.md)).

The pieces fit together as in the [crate map](../crate-map.md).
