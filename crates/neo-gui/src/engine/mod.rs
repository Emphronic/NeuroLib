//! Everything without a UI type: the open recording and its sources, the shared timeline, the
//! time-view state, the rasterizers and the threads that run them. The GPUI layer (store, view
//! models, views) builds on this; `--snapshot` uses it alone.

pub mod canvas;
#[allow(dead_code)] // cache of derived curation data: not used by the panels yet
pub mod compute;
pub mod curation;
pub mod data;
pub mod palette;
pub mod ticks;
pub mod time;
pub mod training;
pub mod work_pool;
