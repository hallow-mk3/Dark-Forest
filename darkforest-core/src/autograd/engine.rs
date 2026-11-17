use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::Arc;
use anyhow::{anyhow, Result};
use crate::autograd::node::Node;
use crate::tensor::Tensor;

/// Execute reverse-mode automatic differentiation starting from a scalar or tensor output.
///
/// Uses an iterative topological sweep:
/// 1. Discovers the backward DAG starting from `tensor.gradient_edge()` and counts in-degrees (dependencies).
/// 2. Executes nodes using a ready queue (`VecDeque`), guaranteeing all incoming gradient edges
///    for a node have arrived and accumulated before `node.apply()` is invoked.
/// 3. Completely non-recursive, preventing stack overflows on deep graphs.
pub fn backward(tensor: &Tensor, gradient: Option<Tensor>, _retain_graph: bool) -> Result<()> {
    if !tensor.requires_grad() {
        return Err(anyhow!(
            "element 0 of tensors does not require grad and does not have a grad_fn"
        ));
    }

    let initial_grad = match gradient {
        Some(g) => {
            if g.shape() != tensor.shape() {
                return Err(anyhow!(
                    "mismatch in shape: grad output has shape {:?}, but tensor has shape {:?}",
                    g.shape(),
                    tensor.shape()
                ));
            }
            g
        }
        None => {
            if tensor.numel() != 1 {
                return Err(anyhow!(
                    "grad can be implicitly created only for scalar outputs"
                ));
            }
            Tensor::ones_device(tensor.shape().clone(), tensor.device())
        }
    };

    let root_edge = match tensor.gradient_edge() {
        Some(edge) => edge,
        None => return Ok(()),
    };

    // Phase 1: Dependency counting (Graph Traversal)
    let mut dependencies: HashMap<usize, usize> = HashMap::new();
    let mut visited: HashSet<usize> = HashSet::new();
    let mut q = VecDeque::new();

    q.push_back(root_edge.node.clone());
    visited.insert(root_edge.node.id());

    while let Some(node) = q.pop_front() {
        for edge in node.next_edges() {
            let target_id = edge.node.id();
            *dependencies.entry(target_id).or_insert(0) += 1;
            if visited.insert(target_id) {
                q.push_back(edge.node.clone());
            }
        }
    }

    // Phase 2: Ready Queue Execution
    let mut ready_queue: VecDeque<Arc<dyn Node>> = VecDeque::new();
    let mut pending_grads: HashMap<usize, HashMap<usize, Tensor>> = HashMap::new();

    // Initialize root node
    pending_grads
        .entry(root_edge.node.id())
        .or_default()
        .insert(root_edge.input_nr, initial_grad);
    ready_queue.push_back(root_edge.node);

    while let Some(node) = ready_queue.pop_front() {
        let node_id = node.id();

        // Gather accumulated gradients for this node's inputs
        let input_grads_map = pending_grads.remove(&node_id).unwrap_or_default();
        let max_input_nr = input_grads_map.keys().copied().max().unwrap_or(0);
        let mut grad_inputs = Vec::with_capacity(max_input_nr + 1);
        for i in 0..=max_input_nr {
            grad_inputs.push(input_grads_map.get(&i).cloned());
        }

        // Apply backward computation
        let grad_outputs = node.apply(grad_inputs)?;

        // Route outputs to next edges
        for (edge, maybe_grad) in node.next_edges().iter().zip(grad_outputs.into_iter()) {
            let target_id = edge.node.id();

            if let Some(grad) = maybe_grad {
                let target_map = pending_grads.entry(target_id).or_default();
                if let Some(existing) = target_map.get_mut(&edge.input_nr) {
                    *existing = existing.add(&grad)?;
                } else {
                    target_map.insert(edge.input_nr, grad);
                }
            }

            if let Some(count) = dependencies.get_mut(&target_id) {
                *count -= 1;
                if *count == 0 {
                    ready_queue.push_back(edge.node.clone());
                }
            }
        }
    }

    Ok(())
}
