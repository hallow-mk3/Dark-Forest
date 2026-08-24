//! Autograd engine: dynamic tape-based reverse-mode AD.
//!
//! Design principles:
//!  - Each forward op creates a `GradFn` that knows how to compute its inputs' gradients
//!    from its output's gradient.
//!  - Tensors that `requires_grad = true` accumulate gradients in `tensor.grad`.
//!  - `backward()` takes a `Value` (the scalar loss) and walks the tape in reverse
//!    topological order, calling each GradFn and accumulating into input grads.
//!
//! Phase 2 note: GradFn is a trait object — quantization hooks can be added as wrapper
//! GradFns without touching the core engine.

pub mod checkpoint;
pub mod grad_fn;
pub mod tape;
pub mod value;

pub use checkpoint::checkpoint;
pub use tape::{is_grad_enabled, no_grad};
pub use value::Value;
