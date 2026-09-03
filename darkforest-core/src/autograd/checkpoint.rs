//! Gradient checkpointing (activation recomputation) support for memory-efficient training.
//!
//! Replaces storing forward activations for every intermediate layer in VRAM
//! with re-running forward pass during the backward step under `no_grad()`.

use crate::autograd::grad_fn::{GradFn, GradFnBox};
use crate::autograd::tape::{is_grad_enabled, no_grad};
use crate::autograd::Value;
use crate::tensor::Tensor;
use anyhow::Result;
use std::sync::Arc;

pub struct CheckpointGradFn<F>
where
    F: Fn(&Value) -> Result<Value> + Send + Sync + 'static,
{
    pub forward_fn: Arc<F>,
    pub input: Value,
}

impl<F> GradFn for CheckpointGradFn<F>
where
    F: Fn(&Value) -> Result<Value> + Send + Sync + 'static,
{
    fn backward(&self, grad_output: &Tensor) -> Vec<Tensor> {
        // Recompute the forward graph with gradients enabled from the saved input
        let recomputed_in = Value::leaf(self.input.tensor());
        let recomputed_out =
            (self.forward_fn)(&recomputed_in).expect("Checkpoint recomputation forward failed");

        // Seed gradient into recomputed output and run backward
        recomputed_out._backward_recursive(grad_output);

        let in_grad = recomputed_in.grad_tensor().unwrap_or_else(|| {
            Tensor::zeros_device(self.input.shape(), self.input.tensor().device.clone())
        });

        vec![in_grad]
    }

    fn required_inputs(&self) -> Vec<usize> {
        vec![0]
    }
}

/// Execute `forward_fn(input)` with gradient checkpointing.
///
/// Forward activations are not stored on the autograd tape; during backward,
/// `forward_fn` is re-run to reconstruct the local computation graph and backpropagate.
pub fn checkpoint<F>(forward_fn: F, input: &Value) -> Result<Value>
where
    F: Fn(&Value) -> Result<Value> + Send + Sync + 'static,
{
    if !is_grad_enabled() {
        return forward_fn(input);
    }

    let forward_fn_arc = Arc::new(forward_fn);

    // Compute the output without saving autograd graph
    let out_val = no_grad(|| (forward_fn_arc)(input))?;
    let out_tensor = out_val.tensor();

    let grad_fn: GradFnBox = Box::new(CheckpointGradFn {
        forward_fn: forward_fn_arc,
        input: input.clone(),
    });

    Ok(Value::from_op(out_tensor, grad_fn, vec![input.clone()]))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_checkpoint_recomputation_matches_standard() {
        let x_data = vec![1.0, 2.0, 3.0, 4.0];
        let x_std = Value::leaf(Tensor::from_vec(x_data.clone(), vec![2, 2]).unwrap());
        let x_chk = Value::leaf(Tensor::from_vec(x_data, vec![2, 2]).unwrap());

        let forward_block = |val: &Value| -> Result<Value> {
            let s = val.scale(2.0)?;
            let a = s.gelu()?;
            a.scale(3.0)
        };

        // Standard forward & backward
        let out_std = forward_block(&x_std).unwrap();
        let loss_std = out_std.sum().unwrap();
        loss_std.backward();

        // Checkpointed forward & backward
        let out_chk = checkpoint(forward_block, &x_chk).unwrap();
        let loss_chk = out_chk.sum().unwrap();
        loss_chk.backward();

        let out_std_data = out_std.tensor().to_vec();
        let out_chk_data = out_chk.tensor().to_vec();
        assert_eq!(out_std_data, out_chk_data);

        let grad_std = x_std.grad();
        let grad_chk = x_chk.grad();
        assert_eq!(grad_std.len(), grad_chk.len());
        for (g1, g2) in grad_std.iter().zip(grad_chk.iter()) {
            assert!(
                (g1 - g2).abs() < 1e-5,
                "Checkpoint gradient must match standard gradient: {} vs {}",
                g1,
                g2
            );
        }
    }
}
