use anyhow::Result;
use crate::autograd::broadcast::sum_to_shape;
use crate::autograd::edge::Edge;
use crate::autograd::node::{next_node_id, Node};
use crate::tensor::Tensor;

pub struct SubBackward {
    id: usize,
    shape_a: Vec<usize>,
    shape_b: Vec<usize>,
    needs_input_grad: [bool; 2],
    next_edges: Vec<Edge>,
}

impl SubBackward {
    pub fn new(shape_a: Vec<usize>, shape_b: Vec<usize>, needs_input_grad: [bool; 2], next_edges: Vec<Edge>) -> Self {
        Self {
            id: next_node_id(),
            shape_a,
            shape_b,
            needs_input_grad,
            next_edges,
        }
    }
}

impl Node for SubBackward {
    fn id(&self) -> usize {
        self.id
    }

    fn name(&self) -> &'static str {
        "SubBackward0"
    }

    fn apply(&self, grads: Vec<Option<Tensor>>) -> Result<Vec<Option<Tensor>>> {
        let grad = match grads.into_iter().next().flatten() {
            Some(g) => g,
            None => return Ok(vec![None; self.next_edges.len()]),
        };

        let mut out = Vec::with_capacity(self.next_edges.len());
        if self.needs_input_grad[0] {
            let ga = sum_to_shape(&grad, &self.shape_a)?;
            out.push(Some(ga));
        }
        if self.needs_input_grad[1] {
            let neg_grad = grad.neg()?;
            let gb = sum_to_shape(&neg_grad, &self.shape_b)?;
            out.push(Some(gb));
        }
        Ok(out)
    }

    fn next_edges(&self) -> &[Edge] {
        &self.next_edges
    }
}
