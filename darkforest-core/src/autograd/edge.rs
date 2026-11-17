use std::sync::Arc;
use crate::autograd::node::Node;

/// An Edge connects the gradient output of one backward Node to an input slot
/// of the next backward Node in the tape.
#[derive(Clone)]
pub struct Edge {
    pub node: Arc<dyn Node>,
    pub input_nr: usize,
}

impl Edge {
    pub fn new(node: Arc<dyn Node>, input_nr: usize) -> Self {
        Self { node, input_nr }
    }
}
