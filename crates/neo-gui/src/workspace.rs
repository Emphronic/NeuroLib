//! The workspaces of the title bar: Explore and Curation are built; Sorting and Pipeline say what
//! they will hold (they are planned). Training is the live dashboard of a `neo-snn` run (curves,
//! spike raster, health), fed by a training thread over `neo_snn::telemetry`.

use gpui_kit::assets::IconName;
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
pub enum Workspace {
    /// Traces, heatmaps, channels and the timeline: the entry point.
    #[default]
    Explore,
    /// Define a spike-sorting run (planned).
    Sorting,
    /// A node graph of processing steps (planned).
    Pipeline,
    /// Inspect and judge sorting results: clusters, similar, waveforms, correlograms.
    Curation,
    /// Loss / accuracy / gradient curves, the spike raster and the health of a training run.
    Training,
}

impl Workspace {
    pub const ALL: [Workspace; 5] = [
        Workspace::Explore,
        Workspace::Sorting,
        Workspace::Pipeline,
        Workspace::Curation,
        Workspace::Training,
    ];

    pub fn title(self) -> &'static str {
        match self {
            Workspace::Explore => "Explore",
            Workspace::Sorting => "Sorting",
            Workspace::Pipeline => "Pipeline",
            Workspace::Curation => "Curation",
            Workspace::Training => "Training",
        }
    }

    /// One line for the tab's tooltip.
    pub fn hint(self) -> &'static str {
        match self {
            Workspace::Explore => "Look at the activity: traces, heatmaps, channels, timeline",
            Workspace::Sorting => "Define a spike-sorting run",
            Workspace::Pipeline => "Snap processing steps together in a node graph",
            Workspace::Curation => {
                "Inspect and judge sorting results: clusters, waveforms, correlograms"
            }
            Workspace::Training => "Watch a spiking network learn: curves, spike raster, health",
        }
    }

    /// What a workspace that is not built yet will hold.
    pub fn planned(self) -> Option<&'static str> {
        match self {
            Workspace::Explore | Workspace::Curation | Workspace::Training => None,
            Workspace::Sorting => Some(
                "Choose the channels and the part of the recording to sort, the sorter and its parameters, \
                 then run it and follow its progress here.",
            ),
            Workspace::Pipeline => Some(
                "Build a processing pipeline as a graph: filters, re-referencing, detection and sorting \
                 steps snap together, and each step's output can be shown in a view.",
            ),
        }
    }

    pub fn icon(self) -> IconName {
        match self {
            Workspace::Explore => IconName::AudioWaveform,
            Workspace::Sorting => IconName::SlidersHorizontal,
            Workspace::Pipeline => IconName::Workflow,
            Workspace::Curation => IconName::Microscope,
            Workspace::Training => IconName::Activity,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_workspace_is_listed_once_and_only_the_unbuilt_ones_are_planned() {
        assert_eq!(Workspace::ALL.len(), 5);
        let planned: Vec<_> = Workspace::ALL
            .iter()
            .filter(|w| w.planned().is_some())
            .copied()
            .collect();
        assert_eq!(planned, vec![Workspace::Sorting, Workspace::Pipeline]);
        let mut titles: Vec<_> = Workspace::ALL.iter().map(|w| w.title()).collect();
        titles.sort();
        titles.dedup();
        assert_eq!(titles.len(), Workspace::ALL.len());
    }

    #[test]
    fn a_saved_session_from_before_training_still_loads() {
        let w: Workspace = serde_json::from_str("\"Explore\"").unwrap();
        assert_eq!(w, Workspace::Explore);
        let w: Workspace = serde_json::from_str("\"Training\"").unwrap();
        assert_eq!(w, Workspace::Training);
        let w: Workspace = serde_json::from_str("\"Curation\"").unwrap();
        assert_eq!(w, Workspace::Curation);
    }
}
