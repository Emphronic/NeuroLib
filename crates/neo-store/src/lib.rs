//! # neo-store - `syn` weight format (`.syn` extension)
//!
//! Binary container for weights and topology of neural networks (primarily SNN). Conceived as
//! analogue of the **NIR** format (node graph + parameters), but without HDF5 and with a number of improvements:
//!
//! | | NIR (HDF5) | neo-store |
//! |---|---|---|
//! | dependencies | HDF5 (C-library) | only `std` (+ optional `memmap2`) |
//! | integrity | no | CRC-32C of header, index and **each tensor**, file length, fingerprint |
//! | writing | free-form | streaming, atomic (tmp → fsync → rename), byte-for-byte reproducible |
//! | reading | via the library | mmap without copying, partial row reads, on-demand verification |
//! | types | f32/f64 | f64/f32/f16/bf16, integer, bool, N-bit (ternary/binary), 22 block formats GGUF |
//! | sparsity | no | CSR (thinned and structurally plastic networks) |
//! | compression | gzip inside HDF5 | optional, per chunk: block LZ (LZ4 format) with byte swapping, no dependencies; rows are read without unpacking the entire tensor |
//! | quantized tensors | no | bytes as is; `read_bits` for N-bit, `read_dequantized_f32` (feature `quantization`) via `neo-quantization` for GGUF blocks |
//! | exchange with NIR | HDF5 | `Store::to_nir_json` and `StoreWriter::add_nir_json`: JSON in the spirit of NIR (graph and weights), without HDF5 |
//! | graph | nodes and edges without validation | shape inference, validation against weights, cycles only through `recurrent`, edge delays, subgraphs |
//! | NaN/Inf | are recorded silently | default write failure with index indication |
//! | large models | one file | shards with manifest and substitution check |
//! | versions | — | major/minor, required and optional feature flags |
//! | inspection | need code | min/max/zeros/norm of each tensor in the index without reading data, CLI `synctl` |
//!
//! ## Record
//!
//! ```no_run
//! use neo_store::*;
//!
//! # fn main() -> StoreResult<()> {
//! let w1 = vec![0.1f32; 8 * 4];
//! write_store_file("model.syn", WriteOptions::default(), |w| {
//!     w.set_meta("model.name", "demo")?;
//!     w.add_f32("fc1.weight", &[8, 4], TensorRole::Weight, &w1)?;
//!     w.add_f32("fc1.bias", &[8], TensorRole::Bias, &[0.0; 8])?;
//!
//!     let mut g = Graph::new();
//!     g.add_node(Node::input("in", &[4]))
//!         .add_node(Node::affine("fc1", "fc1.weight", "fc1.bias"))
//!         .add_node(Node::output("out", &[8]));
//!     g.chain(&["in", "fc1", "out"]);
//!     w.set_graph(g)?; // validates shapes when writing finishes
//!     Ok(())
//! })?;
//!
//! let store = Store::open("model.syn")?;
//! let weights = store.read_f32("fc1.weight")?;
//! # let _ = weights;
//! # Ok(()) }
//! ```
//!
//! ## Compression
//!
//! By default, tensors are written as is and read via mmap without copying. Compression
//! enabled by policy [`Compression`] in [`WriteOptions`] or on the fly via
//! [`StoreWriter::set_compression`]:
//!
//! ```no_run
//! use neo_store::*;
//!
//! # fn main() -> StoreResult<()> {
//! let opts = WriteOptions {
//!     compression: Compression::Auto, // compress if the first chunk wins at least 10%
//!     ..WriteOptions::default()
//! };
//! write_store_file("model.syn", opts, |w| {
//!     w.add_f32("fc1.weight", &[128, 784], TensorRole::Weight, &vec![0.0f32; 128 * 784])?;
//!     Ok(())
//! })?;
//! # Ok(()) }
//! ```
//!
//! The compressed tensor is read using the same methods as a regular one; CRC is computed over the unpacked
//! data, so the file fingerprint is independent of compression.
//!
//! The file format is described in [`format`](crate::format) and in `README.md` crate.

#![deny(unsafe_code)]
#![warn(missing_docs)]

pub mod bits;
mod codec;
pub mod compress;
pub mod crc;
pub mod dtype;
pub mod error;
pub mod format;
pub mod graph;
pub mod half;
pub mod nir_import;
pub mod nir_json;
#[cfg(feature = "quantization")]
pub mod quant;
pub mod reader;
pub mod shard;
pub mod sparse;
pub mod tensor;
pub mod value;
pub mod writer;

pub use bits::{pack_bits, unpack_bits};
pub use compress::{Compression, Encoding};
pub use dtype::{DType, Element, GgufKind};
pub use error::{StoreError, StoreResult};
pub use format::{FORMAT_MAJOR, FORMAT_MINOR};
pub use graph::{Edge, Graph, GraphShapes, Node, NodeOp, NodeShapes};
pub use nir_import::{
    ImportedTensor, NirImportOptions, NirImported, nir_json_file_to_syn, nir_json_to_syn,
    parse_nir_json,
};
pub use nir_json::{NirJson, NirJsonOptions, Unsupported};
pub use reader::{ReadOptions, Store, VerifyFailure, VerifyReport};
pub use shard::{ShardEntry, ShardedStore, ShardedSummary, ShardedWriter};
pub use sparse::SparseCsr;
pub use tensor::{MAX_RANK, TensorInfo, TensorRole, TensorStats};
pub use value::{Meta, Value};
pub use writer::{NonFinitePolicy, StoreWriter, WriteOptions, WriteSummary, write_store_file};
