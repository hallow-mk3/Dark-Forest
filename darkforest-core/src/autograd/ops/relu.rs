use anyhow::Result;
use crate::autograd::edge::Edge;
use crate::autograd::node::{next_node_id, Node};
use crate::autograd::saved_tensor::SavedTensor;
use crate::dispatch::iterator::TensorIterator;
use crate::tensor::Tensor;

pub struct ReLUBackward {
    id: usize,
    input: SavedTensor,
    next_edges: Vec<Edge>,
}

impl ReLUBackward {
    pub fn new(input: &Tensor, next_edges: Vec<Edge>) -> Self {
        Self {
            id: next_node_id(),
            input: SavedTensor::new(input),
            next_edges,
        }
    }
}

impl Node for ReLUBackward {
    fn id(&self) -> usize {
        self.id
    }

    fn name(&self) -> &'static str {
        "ReluBackward0"
    }

    fn apply(&self, grads: Vec<Option<Tensor>>) -> Result<Vec<Option<Tensor>>> {
        let grad = match grads.into_iter().next().flatten() {
            Some(g) => g,
            None => return Ok(vec![None]),
        };

        let input = self.input.unpack()?;
        let mask = TensorIterator::unary_f32(&input, |x| if x > 0.0 { 1.0 } else { 0.0 })?;
        let in_grad = grad.mul(&mask)?;
        Ok(vec![Some(in_grad)])
    }

    fn next_edges(&self) -> &[Edge] {
        &self.next_edges
    }
}
