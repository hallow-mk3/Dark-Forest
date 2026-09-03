//! CPU operations: add, matmul, softmax, layernorm.
//!
//! All ops follow the same contract:
//!   forward(inputs) → output Tensor
//!   backward(output_grad, saved_inputs) → Vec<input_grad>
//!
//! Ops here are correctness-first, not performance-optimised.
//! GPU-accelerated versions live in darkforest-cuda.

pub mod activation;
pub mod add;
pub mod index;
pub mod layernorm;
pub mod matmul;
pub mod norm;
pub mod reduction;
pub mod shape;
pub mod softmax;

pub use activation::{
    elu, elu_backward, gelu, gelu_backward, hardswish, hardswish_backward, leaky_relu,
    leaky_relu_backward, log_sigmoid, log_sigmoid_backward, mish, mish_backward, relu,
    relu_backward, selu, selu_backward, sigmoid, sigmoid_backward, silu, silu_backward, softplus,
    softplus_backward, tanh_act, tanh_backward,
};
pub use add::{add, add_backward};
pub use index::{gather, index_select, masked_fill, scatter_add, slice_dim, where_op};
pub use layernorm::{layernorm, layernorm_backward};
pub use matmul::{matmul, matmul_backward};
pub use norm::{batch_norm, group_norm, instance_norm, rms_norm, BatchNormStats};
pub use reduction::{
    argmax, argmin, max, max_dim, mean, mean_backward, mean_dim, min, min_dim, std_dev, sum_dim,
    var,
};
pub use shape::{cat, chunk, expand, flatten, narrow, permute, split, squeeze, stack, unsqueeze};
pub use softmax::{softmax, softmax_backward};
