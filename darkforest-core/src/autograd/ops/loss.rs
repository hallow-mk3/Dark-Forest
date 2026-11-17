use anyhow::Result;
use crate::autograd::edge::Edge;
use crate::autograd::node::{next_node_id, Node};
use crate::autograd::saved_tensor::SavedTensor;
use crate::dispatch::cpu::softmax::softmax;
use crate::tensor::Tensor;

pub struct CrossEntropyBackward {
    id: usize,
    logits: SavedTensor,
    target: Tensor,
    next_edges: Vec<Edge>,
}

impl CrossEntropyBackward {
    pub fn new(logits: &Tensor, target: &Tensor, next_edges: Vec<Edge>) -> Self {
        Self {
            id: next_node_id(),
            logits: SavedTensor::new(logits),
            target: target.clone(),
            next_edges,
        }
    }
}

impl Node for CrossEntropyBackward {
    fn id(&self) -> usize {
        self.id
    }

    fn name(&self) -> &'static str {
        "NllLossBackward0"
    }

    fn apply(&self, grads: Vec<Option<Tensor>>) -> Result<Vec<Option<Tensor>>> {
        let grad = match grads.into_iter().next().flatten() {
            Some(g) => g,
            None => return Ok(vec![None]),
        };

        let logits = self.logits.unpack()?;
        let probs = softmax(&logits, -1)?;

        let batch_size = logits.shape()[0];
        let num_classes = logits.shape()[1];
        let targets = self.target.to_vec_i64()?;

        let probs_strides = probs.strides().to_vec();
        let probs_ptr = probs.data_ptr() as *mut f32;
        let probs_off = probs.storage_offset() as isize;

        let scale = 1.0f32 / (batch_size as f32);

        // grad_loss is a scalar
        let grad_scalar = if grad.numel() == 1 {
            unsafe { *(grad.data_ptr() as *const f32).offset(grad.storage_offset() as isize) }
        } else {
            1.0f32
        };

        for (b, &label) in targets.iter().enumerate() {
            let c = label as usize;
            for class_i in 0..num_classes {
                let off = probs_off + (b as isize) * probs_strides[0] + (class_i as isize) * probs_strides[1];
                unsafe {
                    let p = *probs_ptr.offset(off);
                    let target_val = if class_i == c { 1.0f32 } else { 0.0f32 };
                    *probs_ptr.offset(off) = (p - target_val) * scale * grad_scalar;
                }
            }
        }

        Ok(vec![Some(probs)])
    }

    fn next_edges(&self) -> &[Edge] {
        &self.next_edges
    }
}
