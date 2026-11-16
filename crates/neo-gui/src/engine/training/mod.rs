//! The training dashboard's engine: no UI type, so it is tested headless. [`state`] folds the
//! events a training thread sends over `neo_snn::telemetry` into histories, spike-frame history,
//! weights and the health verdict; [`plot`] draws the curves and the spike raster, and
//! [`dashboard`] the all-in-one overview (network, raster, accuracy, weights, rates, PCA) and
//! [`network`] the full-size glowing network (curves, spike beads, bloom) and [`brain`] the map of which
//! neurons are in use (regions, usage grid, synapse trace, raster), into a [`Frame`]
//! (`super::canvas`), the same way the Explore renderers do.
//!
//! [`structure`] and [`structure_plot`] do the same for structural plasticity (`neo_snn::structural`):
//! the counters of neurons and synapses over the run, every change with its reason and the loss
//! around it, and the graph of the network as it is now.
//!
//! [`evolution`] and [`evolution_plot`] do the same for an architecture search
//! (`neo_snn::structural::evolution`): fitness of the best / mean / champion, the size of the best
//! network and the number of species, one point per generation.
//!
//! This replaces the terminal dashboard of `neo-snn` (`tui`, on ratatui): the same three things
//! (curves, a spike raster, a verdict on the network's health), drawn on the CPU into images the
//! GUI shows, while the training itself runs on its own thread.

pub mod brain;
pub mod dashboard;
pub mod evolution;
pub mod evolution_plot;
pub mod links;
pub mod network;
pub mod plot;
pub mod state;
pub mod structure;
pub mod structure_plot;

pub use brain::{BrainCamera, BrainInfo, BrainLayout, render_brain};
pub use dashboard::{DashboardInfo, PanelKind, PanelRect, panel_rects, render_dashboard};
pub use evolution_plot::{EvolutionRequest, render_evolution};
pub use network::{NetCamera, NetLayout, render_network};
pub use plot::{
    CurvePanel, CurveRequest, RasterLayout, RasterRequest, RasterStats, render_curves,
    render_raster,
};
pub use state::{RunStatus, TrainingState};
pub use structure::{EventRow, MarkerKind};
pub use structure_plot::{
    CountsRequest, TopologyLayout, TopologyRequest, marker_color, render_counts, render_topology,
};
