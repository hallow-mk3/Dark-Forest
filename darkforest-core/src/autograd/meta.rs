use std::sync::{Arc, Mutex, Weak};
use anyhow::Result;
use crate::autograd::edge::Edge;
use crate::autograd::node::{next_node_id, Node};
use crate::tensor::Tensor;

pub struct AutogradMeta {
    pub requires_grad: bool,
    pub is_leaf: bool,
    pub grad_fn: Option<Arc<dyn Node>>,
    pub output_nr: usize,
    pub grad: Option<Tensor>,
    pub grad_accumulator: Option<Weak<AccumulateGrad>>,
}

impl std::fmt::Debug for AutogradMeta {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("AutogradMeta")
            .field("requires_grad", &self.requires_grad)
            .field("is_leaf", &self.is_leaf)
            .field("output_nr", &self.output_nr)
            .field("grad_fn", &self.grad_fn.as_ref().map(|n| n.name()))
            .finish()
    }
}

impl AutogradMeta {
    pub fn new_leaf(requires_grad: bool) -> Self {
        Self {
            requires_grad,
            is_leaf: true,
            grad_fn: None,
            output_nr: 0,
            grad: None,
            grad_accumulator: None,
        }
    }

    pub fn new_node(grad_fn: Arc<dyn Node>, output_nr: usize) -> Self {
        Self {
            requires_grad: true,
            is_leaf: false,
            grad_fn: Some(grad_fn),
            output_nr,
            grad: None,
            grad_accumulator: None,
        }
    }
}

pub struct AccumulateGrad {
    id: usize,
    meta: Weak<Mutex<AutogradMeta>>,
}

impl AccumulateGrad {
    pub fn new(meta: Weak<Mutex<AutogradMeta>>) -> Self {
        Self {
            id: next_node_id(),
            meta,
        }
    }
}

impl Node for AccumulateGrad {
    fn id(&self) -> usize {
        self.id
    }

    fn name(&self) -> &'static str {
        "AccumulateGrad"
    }

    fn apply(&self, grads: Vec<Option<Tensor>>) -> Result<Vec<Option<Tensor>>> {
        if let Some(meta_arc) = self.meta.upgrade() {
            if let Some(Some(new_grad)) = grads.into_iter().next() {
                let mut guard = meta_arc.lock().unwrap();
                if let Some(existing) = &guard.grad {
                    guard.grad = Some(existing.add(&new_grad)?);
                } else {
                    guard.grad = Some(new_grad);
                }
            }
        }
        Ok(vec![])
    }

    fn next_edges(&self) -> &[Edge] {
        &[]
    }
}
