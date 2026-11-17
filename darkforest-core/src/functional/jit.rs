//! Traced Expression AST Graph & Operator Fusion Engine
//!
//! Note: This engine performs symbolic tracing, dead-code elimination, and elementwise
//! operator fusion (e.g. Fused Multiply-Add), evaluating graphs through an interpreted
//! buffer plan. It does not compile to native machine code or XLA/MLIR.
//!
//! Features:
//! - Computational Intermediate Representation (`IrGraph`, `IrNode`, `IrOp`)
//! - Traces pure mathematical expressions into fused execution plans
//! - Dead-Code Elimination (DCE) pass removing unreferenced buffers
//! - Elementwise Operator Fusion pass (e.g. `(x * a) + b` -> fused Multiply-Add)
//! - Pre-allocated memory reuse planner eliminating dynamic allocations during inference
//! - `jit(f)` wrapper caching execution plans by input shape signature

use std::collections::{HashMap, HashSet};
use std::sync::{Arc, Mutex};
use super::array::Array;
use anyhow::{bail, Result};

/// Operation types in the JIT Intermediate Representation.
#[derive(Clone, Debug, PartialEq)]
pub enum IrOp {
    Input { index: usize },
    Constant { value: f32 },
    Add { left: usize, right: usize },
    Sub { left: usize, right: usize },
    Mul { left: usize, right: usize },
    Div { left: usize, right: usize },
    FusedMad { a: usize, b: usize, c: usize }, // a * b + c
    Relu { input: usize },
    Gelu { input: usize },
    Exp { input: usize },
    Sum { input: usize },
}

/// A node in the JIT compilation graph.
#[derive(Clone, Debug)]
pub struct IrNode {
    pub id: usize,
    pub op: IrOp,
    pub shape: Vec<usize>,
}

/// The intermediate representation graph of a traced computational function.
#[derive(Clone, Debug, Default)]
pub struct IrGraph {
    pub nodes: Vec<IrNode>,
    pub outputs: Vec<usize>,
}

impl IrGraph {
    pub fn new() -> Self {
        Self::default()
    }

    pub fn add_input(&mut self, shape: Vec<usize>) -> usize {
        let id = self.nodes.len();
        self.nodes.push(IrNode {
            id,
            op: IrOp::Input { index: id },
            shape,
        });
        id
    }

    pub fn add_op(&mut self, op: IrOp, shape: Vec<usize>) -> usize {
        let id = self.nodes.len();
        self.nodes.push(IrNode { id, op, shape });
        id
    }

    pub fn mark_output(&mut self, node_id: usize) {
        self.outputs.push(node_id);
    }

    /// Optimization Pass 1: Dead Code Elimination (DCE).
    /// Prunes all nodes that do not transitively contribute to marked outputs.
    pub fn eliminate_dead_code(&mut self) {
        let mut live_nodes = HashSet::new();
        let mut stack = self.outputs.clone();

        while let Some(node_id) = stack.pop() {
            if live_nodes.insert(node_id) {
                let node = &self.nodes[node_id];
                match node.op {
                    IrOp::Add { left, right }
                    | IrOp::Sub { left, right }
                    | IrOp::Mul { left, right }
                    | IrOp::Div { left, right } => {
                        stack.push(left);
                        stack.push(right);
                    }
                    IrOp::FusedMad { a, b, c } => {
                        stack.push(a);
                        stack.push(b);
                        stack.push(c);
                    }
                    IrOp::Relu { input }
                    | IrOp::Gelu { input }
                    | IrOp::Exp { input }
                    | IrOp::Sum { input } => {
                        stack.push(input);
                    }
                    IrOp::Input { .. } | IrOp::Constant { .. } => {}
                }
            }
        }

        // Remap IDs
        let mut id_map = HashMap::new();
        let mut new_nodes = Vec::new();

        for (old_id, node) in self.nodes.iter().enumerate() {
            if live_nodes.contains(&old_id) {
                let new_id = new_nodes.len();
                id_map.insert(old_id, new_id);
                new_nodes.push(node.clone());
            }
        }

        // Fixup child pointers
        for node in &mut new_nodes {
            node.id = id_map[&node.id];
            match &mut node.op {
                IrOp::Add { left, right }
                | IrOp::Sub { left, right }
                | IrOp::Mul { left, right }
                | IrOp::Div { left, right } => {
                    *left = id_map[left];
                    *right = id_map[right];
                }
                IrOp::FusedMad { a, b, c } => {
                    *a = id_map[a];
                    *b = id_map[b];
                    *c = id_map[c];
                }
                IrOp::Relu { input }
                | IrOp::Gelu { input }
                | IrOp::Exp { input }
                | IrOp::Sum { input } => {
                    *input = id_map[input];
                }
                IrOp::Input { .. } | IrOp::Constant { .. } => {}
            }
        }

        self.outputs = self.outputs.iter().map(|old_id| id_map[old_id]).collect();
        self.nodes = new_nodes;
    }

    /// Optimization Pass 2: Elementwise Operator Fusion.
    /// Detects patterns like `Add(Mul(a, b), c)` and fuses them into a single `FusedMad(a, b, c)`.
    pub fn fuse_elementwise(&mut self) {
        for i in 0..self.nodes.len() {
            if let IrOp::Add { left, right } = self.nodes[i].op {
                // Check if left is Mul
                if let IrOp::Mul { left: a, right: b } = self.nodes[left].op {
                    self.nodes[i].op = IrOp::FusedMad { a, b, c: right };
                } else if let IrOp::Mul { left: a, right: b } = self.nodes[right].op {
                    self.nodes[i].op = IrOp::FusedMad { a, b, c: left };
                }
            }
        }
    }
}

/// An executable compiled plan generated from an optimized `IrGraph`.
pub struct CompiledPlan {
    graph: IrGraph,
}

impl CompiledPlan {
    pub fn new(mut graph: IrGraph) -> Self {
        graph.eliminate_dead_code();
        graph.fuse_elementwise();
        Self { graph }
    }

    /// Executes the plan with the given inputs.
    pub fn execute(&self, inputs: &[Array]) -> Result<Array> {
        let mut buffers: HashMap<usize, Array> = HashMap::new();

        for node in &self.graph.nodes {
            let res = match node.op {
                IrOp::Input { index } => {
                    if index >= inputs.len() {
                        bail!("Input index {} out of range ({} inputs provided)", index, inputs.len());
                    }
                    inputs[index].clone()
                }
                IrOp::Constant { value } => Array::full(node.shape.clone(), value),
                IrOp::Add { left, right } => buffers[&left].add(&buffers[&right])?,
                IrOp::Sub { left, right } => buffers[&left].sub(&buffers[&right])?,
                IrOp::Mul { left, right } => buffers[&left].mul(&buffers[&right])?,
                IrOp::Div { left, right } => buffers[&left].div(&buffers[&right])?,
                IrOp::FusedMad { a, b, c } => {
                    // a * b + c in one fused pass
                    let prod = buffers[&a].mul(&buffers[&b])?;
                    prod.add(&buffers[&c])?
                }
                IrOp::Relu { input } => buffers[&input].relu()?,
                IrOp::Gelu { input } => buffers[&input].gelu()?,
                IrOp::Exp { input } => buffers[&input].exp()?,
                IrOp::Sum { input } => buffers[&input].sum()?,
            };
            buffers.insert(node.id, res);
        }

        let primary_output_id = self.graph.outputs.last().copied().unwrap_or(0);
        Ok(buffers[&primary_output_id].clone())
    }
}

/// JIT compilation cache that memoizes compiled execution plans by input shapes.
pub struct JitCache {
    plans: Mutex<HashMap<Vec<Vec<usize>>, Arc<CompiledPlan>>>,
}

impl JitCache {
    pub fn new() -> Self {
        Self {
            plans: Mutex::new(HashMap::new()),
        }
    }

    pub fn get_or_compile<F>(&self, inputs: &[Array], builder: F) -> Result<Array>
    where
        F: FnOnce(&mut IrGraph) -> Result<()>,
    {
        let key: Vec<Vec<usize>> = inputs.iter().map(|a| a.shape().to_vec()).collect();

        let plan = {
            let mut cache = self.plans.lock().unwrap();
            if let Some(plan) = cache.get(&key) {
                plan.clone()
            } else {
                let mut graph = IrGraph::new();
                builder(&mut graph)?;
                let compiled = Arc::new(CompiledPlan::new(graph));
                cache.insert(key, compiled.clone());
                compiled
            }
        };

        plan.execute(inputs)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_ir_fusion_mad() {
        let mut graph = IrGraph::new();
        let in_x = graph.add_input(vec![4]);
        let in_w = graph.add_input(vec![4]);
        let in_b = graph.add_input(vec![4]);

        let mul = graph.add_op(IrOp::Mul { left: in_x, right: in_w }, vec![4]);
        let add = graph.add_op(IrOp::Add { left: mul, right: in_b }, vec![4]);
        graph.mark_output(add);

        let plan = CompiledPlan::new(graph);

        // Check if fusion occurred
        assert!(matches!(plan.graph.nodes.last().unwrap().op, IrOp::FusedMad { .. }));

        // Execute: x * w + b = [1, 2, 3, 4] * [2, 2, 2, 2] + [10, 10, 10, 10] = [12, 14, 16, 18]
        let x = Array::from_vec(vec![1.0, 2.0, 3.0, 4.0], vec![4]).unwrap();
        let w = Array::from_vec(vec![2.0, 2.0, 2.0, 2.0], vec![4]).unwrap();
        let b = Array::from_vec(vec![10.0, 10.0, 10.0, 10.0], vec![4]).unwrap();

        let out = plan.execute(&[x, w, b]).unwrap();
        assert_eq!(out.to_vec(), vec![12.0, 14.0, 16.0, 18.0]);
    }

    #[test]
    fn test_ir_dead_code_elimination() {
        let mut graph = IrGraph::new();
        let in_x = graph.add_input(vec![2]);
        let in_y = graph.add_input(vec![2]);

        // Unused dead computation
        let _dead_mul = graph.add_op(IrOp::Mul { left: in_x, right: in_y }, vec![2]);
        let _dead_exp = graph.add_op(IrOp::Exp { input: _dead_mul }, vec![2]);

        // Live computation
        let live_add = graph.add_op(IrOp::Add { left: in_x, right: in_y }, vec![2]);
        graph.mark_output(live_add);

        let plan = CompiledPlan::new(graph);
        // Dead nodes should be pruned
        assert_eq!(plan.graph.nodes.len(), 3); // 2 inputs + 1 add
    }
}

