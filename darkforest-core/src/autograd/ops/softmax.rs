use anyhow::Result;
use crate::autograd::edge::Edge;
use crate::autograd::node::{next_node_id, Node};
use crate::autograd::saved_tensor::SavedTensor;
use crate::dispatch::iterator::TensorIterator;
use crate::tensor::Tensor;

pub struct LogSoftmaxBackward {
    id: usize,
    output: SavedTensor,
    dim: isize,
    next_edges: Vec<Edge>,
}

impl LogSoftmaxBackward {
    pub fn new(output: &Tensor, dim: isize, next_edges: Vec<Edge>) -> Self {
        Self {
            id: next_node_id(),
            output: SavedTensor::new(output),
            dim,
            next_edges,
        }
    }
}

impl Node for LogSoftmaxBackward {
    fn id(&self) -> usize {
        self.id
    }

    fn name(&self) -> &'static str {
        "LogSoftmaxBackward0"
    }

    fn apply(&self, grads: Vec<Option<Tensor>>) -> Result<Vec<Option<Tensor>>> {
        let grad = match grads.into_iter().next().flatten() {
            Some(g) => g,
            None => return Ok(vec![None]),
        };

        let output = self.output.unpack()?;
        let ndim = output.ndim();
        let d = if self.dim < 0 {
            (ndim as isize + self.dim) as usize
        } else {
            self.dim as usize
        };

        // sum_grad = sum(grad, dim=d, keepdim=true)
        let sum_grad = grad.sum(Some(&[d]), true)?;

        // exp(output)
        let exp_out = TensorIterator::unary_f32(&output, |x| x.exp())?;

        // in_grad = grad - exp(output) * sum_grad
        let term = exp_out.mul(&sum_grad)?;
        let in_grad = grad.sub(&term)?;

        Ok(vec![Some(in_grad)])
    }

    fn next_edges(&self) -> &[Edge] {
        &self.next_edges
    }
}
