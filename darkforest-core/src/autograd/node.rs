use std::sync::atomic::{AtomicUsize, Ordering};
use anyhow::Result;
use crate::autograd::edge::Edge;
use crate::tensor::Tensor;

static NEXT_NODE_ID: AtomicUsize = AtomicUsize::new(1);

pub fn next_node_id() -> usize {
    NEXT_NODE_ID.fetch_add(1, Ordering::Relaxed)
}

/// Node trait for dynamic tape autograd.
/// Represents a backward function node in the execution tape.
pub trait Node: Send + Sync {
    /// Unique identifier for topological sweep and ready-queue deduplication.
    fn id(&self) -> usize;

    /// Node name for profiling and debugging (e.g., "AddBackward0").
    fn name(&self) -> &'static str;

    /// Apply the backward operation.
    /// `grads`: Upstream gradients for each output produced by this op.
    /// Returns gradients for each input edge in `next_edges`.
    fn apply(&self, grads: Vec<Option<Tensor>>) -> Result<Vec<Option<Tensor>>>;

    /// Downstream edges in the backward graph (i.e. predecessors in the forward pass).
    fn next_edges(&self) -> &[Edge];
}
