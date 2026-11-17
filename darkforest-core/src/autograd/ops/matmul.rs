use anyhow::Result;
use crate::autograd::broadcast::sum_to_shape;
use crate::autograd::edge::Edge;
use crate::autograd::node::{next_node_id, Node};
use crate::autograd::saved_tensor::SavedTensor;
use crate::tensor::Tensor;

pub struct MatmulBackward {
    id: usize,
    a: SavedTensor,
    b: SavedTensor,
    needs_input_grad: [bool; 2],
    next_edges: Vec<Edge>,
}

impl MatmulBackward {
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

impl Node for MatmulBackward {
    fn id(&self) -> usize {
        self.id
    }

    fn name(&self) -> &'static str {
        "MmBackward0"
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
            // dA = grad @ b.T
            let b_t = b.transpose(b.ndim() - 2, b.ndim() - 1)?;
            let ga_raw = grad.matmul(&b_t)?;
            let ga = sum_to_shape(&ga_raw, a.shape())?;
            out.push(Some(ga));
        }

        if self.needs_input_grad[1] {
            // dB = a.T @ grad
            let a_t = a.transpose(a.ndim() - 2, a.ndim() - 1)?;
            let gb_raw = a_t.matmul(&grad)?;
            let gb = sum_to_shape(&gb_raw, b.shape())?;
            out.push(Some(gb));
        }

        Ok(out)
    }

    fn next_edges(&self) -> &[Edge] {
        &self.next_edges
    }
}
