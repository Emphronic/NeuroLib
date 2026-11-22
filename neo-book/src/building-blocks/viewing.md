# Viewing

`neo-view` — preparing continuous multi-channel signals for viewing, wherever the viewer runs (a local app next to
the recording, or a remote viewer served over the network):

- `envelope` — the min/max reduction: `[min, max]` per column keeps every spike peak visible at any zoom; host-side
  (`min_max_decimate`), and on a CubeCL device (`envelope_on_device`, feature `device`);
- `pyramid` — the min/max pyramid of a whole recording (`Pyramid`, `PyramidBuilder` — in memory or in a file) with
  background building (`OnProgress`/`Progress`); `PyramidIdentity` for direct access;
- `signal` — the signal as a viewer sees it (`SignalBackend`): `LocalSignal` (a recording + its pyramid in this
  process) or a remote session answering the same views;
- `view` — `View` (what a viewer asks for) and `Envelope` (what it gets), locally or remotely;
- `Block`/`Columns` — the units of the envelope.

Rendering and UI belong to the app (neo-gui); file formats to neo-io.
