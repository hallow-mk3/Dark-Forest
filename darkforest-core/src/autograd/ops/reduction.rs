use anyhow::Result;
use crate::autograd::edge::Edge;
use crate::autograd::node::{next_node_id, Node};
use crate::tensor::Tensor;

pub struct SumBackward {
    id: usize,
    input_shape: Vec<usize>,
    dims: Option<Vec<usize>>,
    keepdim: bool,
    next_edges: Vec<Edge>,
}

impl SumBackward {
    pub fn new(
        input_shape: Vec<usize>,
        dims: Option<Vec<usize>>,
        keepdim: bool,
        next_edges: Vec<Edge>,
    ) -> Self {
        Self {
            id: next_node_id(),
            input_shape,
            dims,
            keepdim,
            next_edges,
        }
    }
}

impl Node for SumBackward {
    fn id(&self) -> usize {
        self.id
    }

    fn name(&self) -> &'static str {
        "SumBackward0"
    }

    fn apply(&self, grads: Vec<Option<Tensor>>) -> Result<Vec<Option<Tensor>>> {
        let grad = match grads.into_iter().next().flatten() {
            Some(g) => g,
            None => return Ok(vec![None]),
        };

        let mut g = grad;
        match &self.dims {
            None => {
                // Scalar reduction
                let ones = vec![1usize; self.input_shape.len()];
                g = g.reshape(ones)?.expand(&self.input_shape)?;
            }
            Some(dims) => {
                if !self.keepdim {
                    let mut sorted_dims = dims.clone();
                    sorted_dims.sort_unstable();
                    for &d in &sorted_dims {
                        g = g.unsqueeze(d as i64)?;
                    }
                }
                g = g.expand(&self.input_shape)?;
            }
        }

        Ok(vec![Some(g)])
    }

    fn next_edges(&self) -> &[Edge] {
        &self.next_edges
    }
}

pub struct MeanBackward {
    id: usize,
    input_shape: Vec<usize>,
    dims: Option<Vec<usize>>,
    keepdim: bool,
    count: f32,
    next_edges: Vec<Edge>,
}

impl MeanBackward {
    pub fn new(
        input_shape: Vec<usize>,
        dims: Option<Vec<usize>>,
        keepdim: bool,
        count: f32,
        next_edges: Vec<Edge>,
    ) -> Self {
        Self {
            id: next_node_id(),
            input_shape,
            dims,
            keepdim,
            count,
            next_edges,
        }
    }
}

impl Node for MeanBackward {
    fn id(&self) -> usize {
        self.id
    }

    fn name(&self) -> &'static str {
        "MeanBackward0"
    }

    fn apply(&self, grads: Vec<Option<Tensor>>) -> Result<Vec<Option<Tensor>>> {
        let sum_node = SumBackward::new(
            self.input_shape.clone(),
            self.dims.clone(),
            self.keepdim,
            vec![],
        );
        let mut sum_grads = sum_node.apply(grads)?;
        if let Some(Some(g)) = sum_grads.first_mut() {
            *g = g.div_scalar(self.count)?;
        }
        Ok(sum_grads)
    }

    fn next_edges(&self) -> &[Edge] {
        &self.next_edges
    }
}
