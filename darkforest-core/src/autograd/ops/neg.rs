use anyhow::Result;
use crate::autograd::edge::Edge;
use crate::autograd::node::{next_node_id, Node};
use crate::tensor::Tensor;

pub struct NegBackward {
    id: usize,
    next_edges: Vec<Edge>,
}

impl NegBackward {
    pub fn new(next_edges: Vec<Edge>) -> Self {
        Self {
            id: next_node_id(),
            next_edges,
        }
    }
}

impl Node for NegBackward {
    fn id(&self) -> usize {
        self.id
    }

    fn name(&self) -> &'static str {
        "NegBackward0"
    }

    fn apply(&self, grads: Vec<Option<Tensor>>) -> Result<Vec<Option<Tensor>>> {
        let grad = match grads.into_iter().next().flatten() {
            Some(g) => g,
            None => return Ok(vec![None]),
        };

        Ok(vec![Some(grad.neg()?)])
    }

    fn next_edges(&self) -> &[Edge] {
        &self.next_edges
    }
}
