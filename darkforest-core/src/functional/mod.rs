//! Dark Forest Functional Transformations Engine (JAX equivalent & beyond)
//!
//! This module provides a complete functional programming paradigm for scientific
//! computing and machine learning in Rust, matching and enhancing JAX's core transformations:
//!
//! 1. **NumPy-like API (`jnp`, `Array`)**: Immutable arrays with referential transparency,
//!    zero-copy Copy-on-Write (COW), and indexed functional updates (`arr.at(idx).set/add/mul/min/max(...)`).
//! 2. **Explicit Randomness (`random`)**: Pure, stateless PRNG tracking via 128-bit `Key` with
//!    `split()`, `split_n()`, and `fold_in()`.
//! 3. **Automatic Differentiation (`diff`)**: Forward-mode (`jvp` with dual numbers), Reverse-mode
//!    (`vjp`), exact gradients (`grad`, `value_and_grad`), Jacobians, and Hessians (`hessian`).
//! 4. **Automatic Vectorization (`vmap`)**: Lifts single-sample operations into batched multi-threaded
//!    executions with configurable input/output axes.
//! 5. **Just-In-Time Compilation & Fusion (`jit`)**: Tracing Intermediate Representation (`IrGraph`),
//!    Dead Code Elimination (DCE), elementwise Multiply-Add (MAD) fusion, and pre-allocated execution plans.
//! 6. **Parallelization & Sharding (`sharding`)**: Distributed `DeviceMesh`, `ShardedArray`, `pmap`,
//!    and collective reductions (`p_sum`, `p_mean`).
//! 7. **Debugging & NaN Trapping (`debug`)**: Zero-overhead NaN/Inf traps (`debug_check_nans`),
//!    pipeline inspection (`debug_print`), and microsecond profilers (`ProfileTrace`).
//! 8. **Functional Optimizers (`optax`)**: Pure state transformations for AdamW, gradient clipping,
//!    and weight decay.

pub mod array;
pub mod debug;
pub mod diff;
pub mod jit;
pub mod optax;
pub mod random;
pub mod sharding;
pub mod vmap;

// Top-level re-exports for ergonomic use: `use darkforest_core::functional::*;`
pub use array::{jnp, Array, IndexSpec};
pub use debug::{debug_check_nans, debug_print, ArrayTelemetry, ProfileTrace};
pub use diff::{grad, hessian, jacobian, jvp, value_and_grad, vjp, Dual};
pub use jit::{CompiledPlan, IrGraph, IrNode, IrOp, JitCache};
pub use optax::{clip_by_global_norm, AdamState, AdamWOpt};
pub use random::{bernoulli, normal, rademacher, randint, uniform, Key};
pub use sharding::{p_mean, p_sum, pmap, DeviceMesh, ShardedArray, Sharding};
pub use vmap::{vmap, vmap2};

