use anyhow::Result;
use crate::autograd::broadcast::sum_to_shape;
use crate::autograd::edge::Edge;
use crate::autograd::node::{next_node_id, Node};
use crate::tensor::Tensor;

pub struct ViewBackward {
    id: usize,
    input_shape: Vec<usize>,
    next_edges: Vec<Edge>,
}

impl ViewBackward {
    pub fn new(input_shape: Vec<usize>, next_edges: Vec<Edge>) -> Self {
        Self {
            id: next_node_id(),
            input_shape,
            next_edges,
        }
    }
}

impl Node for ViewBackward {
    fn id(&self) -> usize {
        self.id
    }

    fn name(&self) -> &'static str {
        "ViewBackward0"
    }

    fn apply(&self, grads: Vec<Option<Tensor>>) -> Result<Vec<Option<Tensor>>> {
        let grad = match grads.into_iter().next().flatten() {
            Some(g) => g,
            None => return Ok(vec![None]),
        };

        let in_grad = grad.reshape(self.input_shape.clone())?;
        Ok(vec![Some(in_grad)])
    }

    fn next_edges(&self) -> &[Edge] {
        &self.next_edges
    }
}

pub struct TransposeBackward {
    id: usize,
    dim0: usize,
    dim1: usize,
    next_edges: Vec<Edge>,
}

impl TransposeBackward {
    pub fn new(dim0: usize, dim1: usize, next_edges: Vec<Edge>) -> Self {
        Self {
            id: next_node_id(),
            dim0,
            dim1,
            next_edges,
        }
    }
}

impl Node for TransposeBackward {
    fn id(&self) -> usize {
        self.id
    }

    fn name(&self) -> &'static str {
        "TransposeBackward0"
    }

    fn apply(&self, grads: Vec<Option<Tensor>>) -> Result<Vec<Option<Tensor>>> {
        let grad = match grads.into_iter().next().flatten() {
            Some(g) => g,
            None => return Ok(vec![None]),
        };

        let in_grad = grad.transpose(self.dim0, self.dim1)?;
        Ok(vec![Some(in_grad)])
    }

    fn next_edges(&self) -> &[Edge] {
        &self.next_edges
    }
}

pub struct PermuteBackward {
    id: usize,
    inv_dims: Vec<usize>,
    next_edges: Vec<Edge>,
}

impl PermuteBackward {
    pub fn new(dims: &[usize], next_edges: Vec<Edge>) -> Self {
        let mut inv_dims = vec![0; dims.len()];
        for (i, &d) in dims.iter().enumerate() {
            inv_dims[d] = i;
        }
        Self {
            id: next_node_id(),
            inv_dims,
            next_edges,
        }
    }
}

impl Node for PermuteBackward {
    fn id(&self) -> usize {
        self.id
    }

    fn name(&self) -> &'static str {
        "PermuteBackward0"
    }

    fn apply(&self, grads: Vec<Option<Tensor>>) -> Result<Vec<Option<Tensor>>> {
        let grad = match grads.into_iter().next().flatten() {
            Some(g) => g,
            None => return Ok(vec![None]),
        };

        let in_grad = grad.permute(&self.inv_dims)?;
        Ok(vec![Some(in_grad)])
    }

    fn next_edges(&self) -> &[Edge] {
        &self.next_edges
    }
}

pub struct ExpandBackward {
    id: usize,
    input_shape: Vec<usize>,
    next_edges: Vec<Edge>,
}

impl ExpandBackward {
    pub fn new(input_shape: Vec<usize>, next_edges: Vec<Edge>) -> Self {
        Self {
            id: next_node_id(),
            input_shape,
            next_edges,
        }
    }
}

impl Node for ExpandBackward {
    fn id(&self) -> usize {
        self.id
    }

    fn name(&self) -> &'static str {
        "ExpandBackward0"
    }

    fn apply(&self, grads: Vec<Option<Tensor>>) -> Result<Vec<Option<Tensor>>> {
        let grad = match grads.into_iter().next().flatten() {
            Some(g) => g,
            None => return Ok(vec![None]),
        };

        let in_grad = sum_to_shape(&grad, &self.input_shape)?;
        Ok(vec![Some(in_grad)])
    }

    fn next_edges(&self) -> &[Edge] {
        &self.next_edges
    }
}

pub struct FlipBackward {
    id: usize,
    dims: Vec<usize>,
    next_edges: Vec<Edge>,
}

impl FlipBackward {
    pub fn new(dims: &[usize], next_edges: Vec<Edge>) -> Self {
        Self {
            id: next_node_id(),
            dims: dims.to_vec(),
            next_edges,
        }
    }
}

impl Node for FlipBackward {
    fn id(&self) -> usize {
        self.id
    }

    fn name(&self) -> &'static str {
        "FlipBackward0"
    }

    fn apply(&self, grads: Vec<Option<Tensor>>) -> Result<Vec<Option<Tensor>>> {
        let grad = match grads.into_iter().next().flatten() {
            Some(g) => g,
            None => return Ok(vec![None]),
        };

        let in_grad = grad.flip(&self.dims)?;
        Ok(vec![Some(in_grad)])
    }

    fn next_edges(&self) -> &[Edge] {
        &self.next_edges
    }
}
