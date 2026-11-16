//! Views: thin renderers of the view models, forwarding input to them.

pub mod curation;
pub mod explore;
pub mod panels;
// Plot kit (ticks, transforms, lasso, grid painters) is ahead of the views that use it.
#[allow(dead_code)]
pub mod plot;
pub mod trace;
pub mod training;

pub use curation::CurationView;
pub use explore::ExploreView;
pub use training::TrainingView;
