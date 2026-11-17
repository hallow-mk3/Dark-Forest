use anyhow::Result;
use crate::autograd::broadcast::sum_to_shape;
use crate::autograd::edge::Edge;
use crate::autograd::node::{next_node_id, Node};
use crate::autograd::saved_tensor::SavedTensor;
use crate::tensor::Tensor;

pub struct DivBackward {
    id: usize,
    a: SavedTensor,
    b: SavedTensor,
    needs_input_grad: [bool; 2],
    next_edges: Vec<Edge>,
}

impl DivBackward {
    pub fn new(a: &Tensor, b: &Tensor, needs_input_grad: [bool; 2], next_edges: Vec<Edge>) -> Self {
        Self {
            id: next_node_id(),
            a: SavedTensor::new(a),
            b: SavedTensor::new(b),
            needs_input_grad,
            next_edges,
        }
    }
}

impl Node for DivBackward {
    fn id(&self) -> usize {
        self.id
    }

    fn name(&self) -> &'static str {
        "DivBackward0"
    }

    fn apply(&self, grads: Vec<Option<Tensor>>) -> Result<Vec<Option<Tensor>>> {
        let grad = match grads.into_iter().next().flatten() {
            Some(g) => g,
            None => return Ok(vec![None; self.next_edges.len()]),
        };

        let a = self.a.unpack()?;
        let b = self.b.unpack()?;
        let mut out = Vec::with_capacity(self.next_edges.len());

        if self.needs_input_grad[0] {
            // ga = grad / b
            let ga = sum_to_shape(&grad.div(&b)?, a.shape())?;
            out.push(Some(ga));
        }

        if self.needs_input_grad[1] {
            // gb = -grad * a / (b * b)
            let b_sq = b.mul(&b)?;
            let neg_grad = grad.neg()?;
            let num = neg_grad.mul(&a)?;
            let gb_full = num.div(&b_sq)?;
            let gb = sum_to_shape(&gb_full, b.shape())?;
            out.push(Some(gb));
        }

        Ok(out)
    }

    fn next_edges(&self) -> &[Edge] {
        &self.next_edges
    }
}
