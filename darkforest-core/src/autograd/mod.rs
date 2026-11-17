//! Autograd engine: dynamic tape-based reverse-mode automatic differentiation.
//!
//! Features:
//! - Iterative, non-recursive topological sweep engine (`engine::backward`).
//! - Node trait with apply and next_edges.
//! - SavedTensor tracking storage version counter for in-place mutation detection.
//! - `sum_to_shape` for broadcasting backwards.
//! - Thread-local `no_grad()` guard.

pub mod broadcast;
pub mod checkpoint;
pub mod edge;
pub mod engine;
pub mod grad_fn;
pub mod meta;
pub mod node;
pub mod ops;
pub mod saved_tensor;
pub mod tape;
pub mod value;

pub use broadcast::sum_to_shape;
pub use edge::Edge;
pub use engine::backward;
pub use meta::{AccumulateGrad, AutogradMeta};
pub use node::{next_node_id, Node};
pub use saved_tensor::SavedTensor;
pub use tape::{is_grad_enabled, no_grad, set_grad_enabled};
pub use value::Value;
