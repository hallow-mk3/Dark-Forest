//! Rust wrappers for CUDA kernels.
//! Stubs return Err when CUDA toolkit not compiled in.

use anyhow::Result;

/// Synchronize the CUDA default stream.
/// Call this ONCE per training step at the step boundary — not after each kernel.
/// This is the single synchronization point that lets all kernels pipeline freely.
pub fn cuda_sync() -> Result<()> {
    #[cfg(darkforest_cuda_kernels)]
    unsafe {
        extern "C" {
            fn darkforest_sync();
        }
        darkforest_sync();
        return Ok(());
    }
    #[cfg(not(darkforest_cuda_kernels))]
    Ok(()) // no-op when CUDA not available
}

pub fn cuda_reduce_sum(in_ptr: *const f32, out_ptr: *mut f32, n: usize) -> Result<()> {
    #[cfg(darkforest_cuda_kernels)]
    unsafe {
        extern "C" {
            fn launch_reduce_sum(in_ptr: *const f32, out_ptr: *mut f32, n: u32);
        }
        launch_reduce_sum(in_ptr, out_ptr, n as u32);
        return Ok(());
    }
    #[cfg(not(darkforest_cuda_kernels))]
    Err(anyhow::Error::msg("CUDA kernels not compiled"))
}

pub fn cuda_fill(out: *mut f32, val: f32, n: usize) -> Result<()> {
    #[cfg(darkforest_cuda_kernels)]
    unsafe {
        extern "C" {
            fn launch_fill(c: *mut f32, val: f32, n: u32);
        }
        launch_fill(out, val, n as u32);
        return Ok(());
    }
    #[cfg(not(darkforest_cuda_kernels))]
    Err(anyhow::Error::msg("CUDA kernels not compiled"))
}

/// Elementwise add: c[i] = a[i] + b[i] on device pointers.
pub fn cuda_add(a: *const f32, b: *const f32, c: *mut f32, n: usize) -> Result<()> {
    #[cfg(darkforest_cuda_kernels)]
    unsafe {
        extern "C" {
            fn launch_add(a: *const f32, b: *const f32, c: *mut f32, n: u32);
        }
        launch_add(a, b, c, n as u32);
        return Ok(());
    }
    #[cfg(not(darkforest_cuda_kernels))]
    Err(anyhow::Error::msg("CUDA kernels not compiled"))
}

pub fn cuda_scale(x: *const f32, out: *mut f32, alpha: f32, n: usize) -> Result<()> {
    #[cfg(darkforest_cuda_kernels)]
    unsafe {
        extern "C" {
            fn launch_scale(x: *const f32, out: *mut f32, alpha: f32, n: u32);
        }
        launch_scale(x, out, alpha, n as u32);
        return Ok(());
    }
    #[cfg(not(darkforest_cuda_kernels))]
    Err(anyhow::Error::msg("CUDA kernels not compiled"))
}

pub fn cuda_mul(a: *const f32, b: *const f32, c: *mut f32, n: usize) -> Result<()> {
    #[cfg(darkforest_cuda_kernels)]
    unsafe {
        extern "C" {
            fn launch_mul(a: *const f32, b: *const f32, c: *mut f32, n: u32);
        }
        launch_mul(a, b, c, n as u32);
        return Ok(());
    }
    #[cfg(not(darkforest_cuda_kernels))]
    Err(anyhow::Error::msg("CUDA kernels not compiled"))
}

pub fn cuda_mul_backward(
    grad_out: *const f32,
    a: *const f32,
    b: *const f32,
    grad_a: *mut f32,
    grad_b: *mut f32,
    n: usize,
) -> Result<()> {
    #[cfg(darkforest_cuda_kernels)]
    unsafe {
        extern "C" {
            fn launch_mul_backward(
                grad_out: *const f32,
                a: *const f32,
                b: *const f32,
                grad_a: *mut f32,
                grad_b: *mut f32,
                n: u32,
            );
        }
        launch_mul_backward(grad_out, a, b, grad_a, grad_b, n as u32);
        return Ok(());
    }
    #[cfg(not(darkforest_cuda_kernels))]
    Err(anyhow::Error::msg("CUDA kernels not compiled"))
}

pub fn cuda_gelu(x: *const f32, out: *mut f32, n: usize) -> Result<()> {
    #[cfg(darkforest_cuda_kernels)]
    unsafe {
        extern "C" {
            fn launch_gelu(x: *const f32, out: *mut f32, n: u32);
        }
        launch_gelu(x, out, n as u32);
        return Ok(());
    }
    #[cfg(not(darkforest_cuda_kernels))]
    Err(anyhow::Error::msg("CUDA kernels not compiled"))
}

pub fn cuda_gelu_backward(
    grad_out: *const f32,
    x: *const f32,
    grad_x: *mut f32,
    n: usize,
) -> Result<()> {
    #[cfg(darkforest_cuda_kernels)]
    unsafe {
        extern "C" {
            fn launch_gelu_backward(grad_out: *const f32, x: *const f32, grad_x: *mut f32, n: u32);
        }
        launch_gelu_backward(grad_out, x, grad_x, n as u32);
        return Ok(());
    }
    #[cfg(not(darkforest_cuda_kernels))]
    Err(anyhow::Error::msg("CUDA kernels not compiled"))
}

pub fn cuda_add_bias(
    input: *const f32,
    bias: *const f32,
    out: *mut f32,
    batch: usize,
    features: usize,
) -> Result<()> {
    #[cfg(darkforest_cuda_kernels)]
    unsafe {
        extern "C" {
            fn launch_add_bias(
                input: *const f32,
                bias: *const f32,
                out: *mut f32,
                batch: u32,
                features: u32,
            );
        }
        launch_add_bias(input, bias, out, batch as u32, features as u32);
        return Ok(());
    }
    #[cfg(not(darkforest_cuda_kernels))]
    Err(anyhow::Error::msg("CUDA kernels not compiled"))
}

pub fn cuda_embedding_forward(
    indices: *const u32,
    weight: *const f32,
    out: *mut f32,
    seq_len: usize,
    embed_dim: usize,
) -> Result<()> {
    #[cfg(darkforest_cuda_kernels)]
    unsafe {
        extern "C" {
            fn launch_embedding_forward(
                indices: *const u32,
                weight: *const f32,
                out: *mut f32,
                seq_len: u32,
                embed_dim: u32,
            );
        }
        launch_embedding_forward(indices, weight, out, seq_len as u32, embed_dim as u32);
        return Ok(());
    }
    #[cfg(not(darkforest_cuda_kernels))]
    Err(anyhow::Error::msg("CUDA kernels not compiled"))
}

pub fn cuda_embedding_backward(
    indices: *const u32,
    grad_out: *const f32,
    grad_weight: *mut f32,
    seq_len: usize,
    embed_dim: usize,
) -> Result<()> {
    #[cfg(darkforest_cuda_kernels)]
    unsafe {
        extern "C" {
            fn launch_embedding_backward(
                indices: *const u32,
                grad_out: *const f32,
                grad_weight: *mut f32,
                seq_len: u32,
                embed_dim: u32,
            );
        }
        launch_embedding_backward(
            indices,
            grad_out,
            grad_weight,
            seq_len as u32,
            embed_dim as u32,
        );
        return Ok(());
    }
    #[cfg(not(darkforest_cuda_kernels))]
    Err(anyhow::Error::msg("CUDA kernels not compiled"))
}

pub fn cuda_matmul(
    a: *const f32,
    b: *const f32,
    c: *mut f32,
    m: usize,
    k: usize,
    n: usize,
) -> Result<()> {
    #[cfg(darkforest_cuda_kernels)]
    unsafe {
        extern "C" {
            fn launch_matmul(a: *const f32, b: *const f32, c: *mut f32, m: u32, k: u32, n: u32);
        }
        launch_matmul(a, b, c, m as u32, k as u32, n as u32);
        return Ok(());
    }
    #[cfg(not(darkforest_cuda_kernels))]
    Err(anyhow::Error::msg("CUDA kernels not compiled"))
}

pub fn cuda_transpose(input: *const f32, output: *mut f32, rows: usize, cols: usize) -> Result<()> {
    #[cfg(darkforest_cuda_kernels)]
    unsafe {
        extern "C" {
            fn launch_transpose(input: *const f32, output: *mut f32, rows: u32, cols: u32);
        }
        launch_transpose(input, output, rows as u32, cols as u32);
        return Ok(());
    }
    #[cfg(not(darkforest_cuda_kernels))]
    Err(anyhow::Error::msg("CUDA kernels not compiled"))
}

pub fn cuda_linear_forward(
    x: *const f32,
    weight: *const f32,
    bias: *const f32,
    out: *mut f32,
    batch: usize,
    in_features: usize,
    out_features: usize,
    has_bias: bool,
) -> Result<()> {
    #[cfg(darkforest_cuda_kernels)]
    unsafe {
        extern "C" {
            fn launch_linear_forward(
                x: *const f32,
                weight: *const f32,
                bias: *const f32,
                out: *mut f32,
                batch: u32,
                in_features: u32,
                out_features: u32,
                has_bias: u32,
            );
        }
        launch_linear_forward(
            x,
            weight,
            bias,
            out,
            batch as u32,
            in_features as u32,
            out_features as u32,
            has_bias as u32,
        );
        return Ok(());
    }
    #[cfg(not(darkforest_cuda_kernels))]
    Err(anyhow::Error::msg("CUDA kernels not compiled"))
}

pub fn cuda_linear_backward(
    x: *const f32,
    weight: *const f32,
    grad_out: *const f32,
    grad_x: *mut f32,
    grad_weight: *mut f32,
    grad_bias: *mut f32,
    batch: usize,
    in_features: usize,
    out_features: usize,
    has_bias: bool,
) -> Result<()> {
    #[cfg(darkforest_cuda_kernels)]
    unsafe {
        extern "C" {
            fn launch_linear_backward(
                x: *const f32,
                weight: *const f32,
                grad_out: *const f32,
                grad_x: *mut f32,
                grad_weight: *mut f32,
                grad_bias: *mut f32,
                batch: u32,
                in_features: u32,
                out_features: u32,
                has_bias: u32,
            );
        }
        launch_linear_backward(
            x,
            weight,
            grad_out,
            grad_x,
            grad_weight,
            grad_bias,
            batch as u32,
            in_features as u32,
            out_features as u32,
            has_bias as u32,
        );
        return Ok(());
    }
    #[cfg(not(darkforest_cuda_kernels))]
    Err(anyhow::Error::msg("CUDA kernels not compiled"))
}

pub fn cuda_adamw_update(
    parameter: *mut f32,
    first_moment: *mut f32,
    second_moment: *mut f32,
    gradient: *const f32,
    n: usize,
    lr: f32,
    beta1: f32,
    beta2: f32,
    eps: f32,
    weight_decay: f32,
    bias_correction1: f32,
    bias_correction2: f32,
) -> Result<()> {
    #[cfg(darkforest_cuda_kernels)]
    unsafe {
        extern "C" {
            fn launch_adamw_update(
                parameter: *mut f32,
                first_moment: *mut f32,
                second_moment: *mut f32,
                gradient: *const f32,
                n: u32,
                lr: f32,
                beta1: f32,
                beta2: f32,
                eps: f32,
                weight_decay: f32,
                bias_correction1: f32,
                bias_correction2: f32,
            );
        }
        launch_adamw_update(
            parameter,
            first_moment,
            second_moment,
            gradient,
            n as u32,
            lr,
            beta1,
            beta2,
            eps,
            weight_decay,
            bias_correction1,
            bias_correction2,
        );
        return Ok(());
    }
    #[cfg(not(darkforest_cuda_kernels))]
    Err(anyhow::Error::msg("CUDA kernels not compiled"))
}

/// Row-wise softmax on device.
pub fn cuda_softmax(x: *mut f32, rows: usize, cols: usize) -> Result<()> {
    #[cfg(darkforest_cuda_kernels)]
    unsafe {
        extern "C" {
            fn launch_softmax(x: *mut f32, rows: u32, cols: u32);
        }
        launch_softmax(x, rows as u32, cols as u32);
        return Ok(());
    }
    #[cfg(not(darkforest_cuda_kernels))]
    Err(anyhow::Error::msg("CUDA kernels not compiled"))
}

pub fn cuda_softmax_backward(
    probabilities: *const f32,
    grad_output: *const f32,
    grad_input: *mut f32,
    rows: usize,
    cols: usize,
) -> Result<()> {
    #[cfg(darkforest_cuda_kernels)]
    unsafe {
        extern "C" {
            fn launch_softmax_backward(
                probabilities: *const f32,
                grad_output: *const f32,
                grad_input: *mut f32,
                rows: u32,
                cols: u32,
            );
        }
        launch_softmax_backward(
            probabilities,
            grad_output,
            grad_input,
            rows as u32,
            cols as u32,
        );
        return Ok(());
    }
    #[cfg(not(darkforest_cuda_kernels))]
    Err(anyhow::Error::msg("CUDA kernels not compiled"))
}

pub fn cuda_cross_entropy_forward(
    probabilities: *const f32,
    targets: *const u32,
    loss: *mut f32,
    rows: usize,
    cols: usize,
) -> Result<()> {
    #[cfg(darkforest_cuda_kernels)]
    unsafe {
        extern "C" {
            fn launch_cross_entropy_forward(
                probabilities: *const f32,
                targets: *const u32,
                loss: *mut f32,
                rows: u32,
                cols: u32,
            );
        }
        launch_cross_entropy_forward(probabilities, targets, loss, rows as u32, cols as u32);
        return Ok(());
    }
    #[cfg(not(darkforest_cuda_kernels))]
    Err(anyhow::Error::msg("CUDA kernels not compiled"))
}

pub fn cuda_cross_entropy_backward(
    probabilities: *const f32,
    targets: *const u32,
    grad_output: *const f32,
    grad_logits: *mut f32,
    rows: usize,
    cols: usize,
) -> Result<()> {
    #[cfg(darkforest_cuda_kernels)]
    unsafe {
        extern "C" {
            fn launch_cross_entropy_backward(
                probabilities: *const f32,
                targets: *const u32,
                grad_output: *const f32,
                grad_logits: *mut f32,
                rows: u32,
                cols: u32,
            );
        }
        launch_cross_entropy_backward(
            probabilities,
            targets,
            grad_output,
            grad_logits,
            rows as u32,
            cols as u32,
        );
        return Ok(());
    }
    #[cfg(not(darkforest_cuda_kernels))]
    Err(anyhow::Error::msg("CUDA kernels not compiled"))
}

/// Fused: logits → softmax probs (in-place) + CE loss + (prob - one_hot)/N gradient.
/// Replaces 3 separate kernel launches (softmax, ce_fwd, ce_bwd) with 1.
pub fn cuda_fused_logit_ce(
    logits: *mut f32,
    targets: *const u32,
    grad_logits: *mut f32,
    loss: *mut f32,
    rows: usize,
    cols: usize,
) -> Result<()> {
    #[cfg(darkforest_cuda_kernels)]
    unsafe {
        extern "C" {
            fn launch_fused_logit_ce(
                logits: *mut f32,
                targets: *const u32,
                grad_logits: *mut f32,
                loss: *mut f32,
                rows: u32,
                cols: u32,
            );
        }
        launch_fused_logit_ce(logits, targets, grad_logits, loss, rows as u32, cols as u32);
        return Ok(());
    }
    #[cfg(not(darkforest_cuda_kernels))]
    Err(anyhow::Error::msg("CUDA kernels not compiled"))
}

/// LayerNorm on device.
pub fn cuda_layernorm(
    x: *const f32,
    gamma: *const f32,
    beta: *const f32,
    out: *mut f32,
    means: *mut f32,
    rstds: *mut f32,
    batch: usize,
    features: usize,
) -> Result<()> {
    #[cfg(darkforest_cuda_kernels)]
    unsafe {
        extern "C" {
            fn launch_layernorm(
                x: *const f32,
                gamma: *const f32,
                beta: *const f32,
                out: *mut f32,
                means: *mut f32,
                rstds: *mut f32,
                batch: u32,
                features: u32,
            );
        }
        launch_layernorm(
            x,
            gamma,
            beta,
            out,
            means,
            rstds,
            batch as u32,
            features as u32,
        );
        return Ok(());
    }
    #[cfg(not(darkforest_cuda_kernels))]
    Err(anyhow::Error::msg("CUDA kernels not compiled"))
}

pub fn cuda_layernorm_backward(
    grad_out: *const f32,
    x: *const f32,
    gamma: *const f32,
    means: *const f32,
    rstds: *const f32,
    grad_x: *mut f32,
    grad_gamma: *mut f32,
    grad_beta: *mut f32,
    batch: usize,
    features: usize,
) -> Result<()> {
    #[cfg(darkforest_cuda_kernels)]
    unsafe {
        extern "C" {
            fn launch_layernorm_backward(
                grad_out: *const f32,
                x: *const f32,
                gamma: *const f32,
                means: *const f32,
                rstds: *const f32,
                grad_x: *mut f32,
                grad_gamma: *mut f32,
                grad_beta: *mut f32,
                batch: u32,
                features: u32,
            );
        }
        launch_layernorm_backward(
            grad_out,
            x,
            gamma,
            means,
            rstds,
            grad_x,
            grad_gamma,
            grad_beta,
            batch as u32,
            features as u32,
        );
        return Ok(());
    }
    #[cfg(not(darkforest_cuda_kernels))]
    Err(anyhow::Error::msg("CUDA kernels not compiled"))
}

/// Fused FlashAttention-2 style kernel forward pass.
///
/// q, k, v: device pointers [seq_len, d_head]
/// out:     device pointer  [seq_len, d_head]
/// scale:   1/sqrt(d_head)
pub fn cuda_fused_attention(
    q: *const f32,
    k: *const f32,
    v: *const f32,
    out: *mut f32,
    seq_len: usize,
    d_head: usize,
    scale: f32,
    causal: bool,
) -> Result<()> {
    #[cfg(darkforest_cuda_kernels)]
    unsafe {
        extern "C" {
            fn launch_fused_attention(
                q: *const f32,
                k: *const f32,
                v: *const f32,
                out: *mut f32,
                seq_len: u32,
                d_head: u32,
                scale: f32,
                causal: u32,
            );
        }
        launch_fused_attention(
            q,
            k,
            v,
            out,
            seq_len as u32,
            d_head as u32,
            scale,
            causal as u32,
        );
        return Ok(());
    }
    #[cfg(not(darkforest_cuda_kernels))]
    Err(anyhow::Error::msg("CUDA kernels not compiled"))
}

/// Direct, unfused attention backward baseline.
pub fn cuda_attention_backward(
    q: *const f32,
    k: *const f32,
    v: *const f32,
    d_out: *const f32,
    d_q: *mut f32,
    d_k: *mut f32,
    d_v: *mut f32,
    probabilities: *mut f32,
    seq_len: usize,
    d_head: usize,
    scale: f32,
    causal: bool,
) -> Result<()> {
    #[cfg(darkforest_cuda_kernels)]
    unsafe {
        extern "C" {
            fn launch_attention_backward(
                q: *const f32,
                k: *const f32,
                v: *const f32,
                d_out: *const f32,
                d_q: *mut f32,
                d_k: *mut f32,
                d_v: *mut f32,
                probabilities: *mut f32,
                seq_len: u32,
                d_head: u32,
                scale: f32,
                causal: u32,
            );
        }
        launch_attention_backward(
            q,
            k,
            v,
            d_out,
            d_q,
            d_k,
            d_v,
            probabilities,
            seq_len as u32,
            d_head as u32,
            scale,
            causal as u32,
        );
        return Ok(());
    }
    #[cfg(not(darkforest_cuda_kernels))]
    Err(anyhow::Error::msg("CUDA kernels not compiled"))
}

/// Fused multi-head attention forward pass.
pub fn cuda_fused_mha_forward(
    q: *const f32,
    k: *const f32,
    v: *const f32,
    out: *mut f32,
    n_heads: usize,
    seq_len: usize,
    d_head: usize,
    scale: f32,
    causal: bool,
) -> Result<()> {
    #[cfg(darkforest_cuda_kernels)]
    unsafe {
        extern "C" {
            fn launch_fused_mha_forward(
                q: *const f32,
                k: *const f32,
                v: *const f32,
                out: *mut f32,
                n_heads: u32,
                seq_len: u32,
                d_head: u32,
                scale: f32,
                causal: u32,
            );
        }
        launch_fused_mha_forward(
            q,
            k,
            v,
            out,
            n_heads as u32,
            seq_len as u32,
            d_head as u32,
            scale,
            causal as u32,
        );
        return Ok(());
    }
    #[cfg(not(darkforest_cuda_kernels))]
    Err(anyhow::Error::msg("CUDA kernels not compiled"))
}

/// Fused multi-head attention backward pass.
pub fn cuda_fused_mha_backward(
    q: *const f32,
    k: *const f32,
    v: *const f32,
    d_out: *const f32,
    d_q: *mut f32,
    d_k: *mut f32,
    d_v: *mut f32,
    probabilities: *mut f32,
    n_heads: usize,
    seq_len: usize,
    d_head: usize,
    scale: f32,
    causal: bool,
) -> Result<()> {
    #[cfg(darkforest_cuda_kernels)]
    unsafe {
        extern "C" {
            fn launch_fused_mha_backward(
                q: *const f32,
                k: *const f32,
                v: *const f32,
                d_out: *const f32,
                d_q: *mut f32,
                d_k: *mut f32,
                d_v: *mut f32,
                probabilities: *mut f32,
                n_heads: u32,
                seq_len: u32,
                d_head: u32,
                scale: f32,
                causal: u32,
            );
        }
        launch_fused_mha_backward(
            q,
            k,
            v,
            d_out,
            d_q,
            d_k,
            d_v,
            probabilities,
            n_heads as u32,
            seq_len as u32,
            d_head as u32,
            scale,
            causal as u32,
        );
        return Ok(());
    }
    #[cfg(not(darkforest_cuda_kernels))]
    Err(anyhow::Error::msg("CUDA kernels not compiled"))
}

pub fn cuda_f32_to_bf16(src: *const f32, dst: *mut u16, n: usize) -> Result<()> {
    #[cfg(darkforest_cuda_kernels)]
    unsafe {
        extern "C" {
            fn launch_f32_to_bf16(src: *const f32, dst: *mut u16, n: u32);
        }
        launch_f32_to_bf16(src, dst, n as u32);
        return Ok(());
    }
    #[cfg(not(darkforest_cuda_kernels))]
    Err(anyhow::Error::msg("CUDA kernels not compiled"))
}

pub fn cuda_bf16_to_f32(src: *const u16, dst: *mut f32, n: usize) -> Result<()> {
    #[cfg(darkforest_cuda_kernels)]
    unsafe {
        extern "C" {
            fn launch_bf16_to_f32(src: *const u16, dst: *mut f32, n: u32);
        }
        launch_bf16_to_f32(src, dst, n as u32);
        return Ok(());
    }
    #[cfg(not(darkforest_cuda_kernels))]
    Err(anyhow::Error::msg("CUDA kernels not compiled"))
}

pub fn cuda_dequantize_nf4_to_f32(
    packed_indices: *const u8,
    scales: *const f32,
    out_weights: *mut f32,
    total_weights: usize,
    block_size: usize,
) -> Result<()> {
    #[cfg(darkforest_cuda_kernels)]
    unsafe {
        extern "C" {
            fn launch_dequantize_nf4_to_f32(
                packed_indices: *const u8,
                scales: *const f32,
                out_weights: *mut f32,
                total_weights: u32,
                block_size: u32,
            );
        }
        launch_dequantize_nf4_to_f32(
            packed_indices,
            scales,
            out_weights,
            total_weights as u32,
            block_size as u32,
        );
        return Ok(());
    }
    #[cfg(not(darkforest_cuda_kernels))]
    Err(anyhow::Error::msg("CUDA kernels not compiled"))
}

pub fn cuda_dequantize_nf4_to_bf16(
    packed_indices: *const u8,
    scales: *const f32,
    out_weights: *mut u16,
    total_weights: usize,
    block_size: usize,
) -> Result<()> {
    #[cfg(darkforest_cuda_kernels)]
    unsafe {
        extern "C" {
            fn launch_dequantize_nf4_to_bf16(
                packed_indices: *const u8,
                scales: *const f32,
                out_weights: *mut u16,
                total_weights: u32,
                block_size: u32,
            );
        }
        launch_dequantize_nf4_to_bf16(
            packed_indices,
            scales,
            out_weights,
            total_weights as u32,
            block_size as u32,
        );
        return Ok(());
    }
    #[cfg(not(darkforest_cuda_kernels))]
    Err(anyhow::Error::msg("CUDA kernels not compiled"))
}

pub fn cuda_matmul_nf4_fused(
    a: *const f32,
    w_packed: *const u8,
    scales: *const f32,
    c: *mut f32,
    m: usize,
    k: usize,
    n: usize,
    block_size: usize,
) -> Result<()> {
    #[cfg(darkforest_cuda_kernels)]
    unsafe {
        extern "C" {
            fn launch_matmul_nf4_fused(
                a: *const f32,
                w_packed: *const u8,
                scales: *const f32,
                c: *mut f32,
                m: u32,
                k: u32,
                n: u32,
                block_size: u32,
            );
        }
        launch_matmul_nf4_fused(
            a,
            w_packed,
            scales,
            c,
            m as u32,
            k as u32,
            n as u32,
            block_size as u32,
        );
        return Ok(());
    }
    #[cfg(not(darkforest_cuda_kernels))]
    Err(anyhow::Error::msg("CUDA kernels not compiled"))
}

pub fn cuda_rmsnorm(
    x: *const f32,
    gamma: *const f32,
    out: *mut f32,
    rstds: *mut f32,
    batch: usize,
    features: usize,
) -> Result<()> {
    #[cfg(darkforest_cuda_kernels)]
    unsafe {
        extern "C" {
            fn launch_rmsnorm(
                x: *const f32,
                gamma: *const f32,
                out: *mut f32,
                rstds: *mut f32,
                batch: u32,
                features: u32,
            );
        }
        launch_rmsnorm(x, gamma, out, rstds, batch as u32, features as u32);
        return Ok(());
    }
    #[cfg(not(darkforest_cuda_kernels))]
    Err(anyhow::Error::msg("CUDA kernels not compiled"))
}

/// RMSNorm backward pass — gradient w.r.t. x and gamma.
/// Completes the differentiable RMSNorm module for end-to-end backpropagation.
///
/// The closed-form gradient avoids re-materializing normalized activations:
///   dL/dx_i = rstd * (dy_i*gamma_i - (rstd²/N) * dot(dy*gamma, x) * x_i)
pub fn cuda_rmsnorm_backward(
    grad_out: *const f32,
    x: *const f32,
    gamma: *const f32,
    rstds: *const f32,
    grad_x: *mut f32,
    grad_gamma: *mut f32,
    batch: usize,
    features: usize,
) -> Result<()> {
    #[cfg(darkforest_cuda_kernels)]
    unsafe {
        extern "C" {
            fn launch_rmsnorm_backward(
                grad_out: *const f32,
                x: *const f32,
                gamma: *const f32,
                rstds: *const f32,
                grad_x: *mut f32,
                grad_gamma: *mut f32,
                batch: u32,
                features: u32,
            );
        }
        launch_rmsnorm_backward(
            grad_out,
            x,
            gamma,
            rstds,
            grad_x,
            grad_gamma,
            batch as u32,
            features as u32,
        );
        return Ok(());
    }
    #[cfg(not(darkforest_cuda_kernels))]
    Err(anyhow::Error::msg("CUDA kernels not compiled"))
}

/// Speculative Decoding: verify K draft tokens against target distribution in a single GPU pass.
///
/// Based on: Leviathan et al., "Fast Inference from Transformers via Speculative Decoding", 2023.
/// Replaces K sequential target-model forward passes with 1 batched verification pass.
/// Returns per-token acceptance mask and the total count of accepted draft tokens.
pub fn cuda_speculative_verify(
    target_probs: *const f32,
    draft_probs: *const f32,
    draft_tokens: *const u32,
    accept_mask: *mut f32,
    n_accepted: *mut u32,
    threshold: f32,
    k: usize,
    vocab_size: usize,
) -> Result<()> {
    #[cfg(darkforest_cuda_kernels)]
    unsafe {
        extern "C" {
            fn launch_speculative_verify(
                target_probs: *const f32,
                draft_probs: *const f32,
                draft_tokens: *const u32,
                accept_mask: *mut f32,
                n_accepted: *mut u32,
                threshold: f32,
                k: u32,
                vocab_size: u32,
            );
        }
        launch_speculative_verify(
            target_probs,
            draft_probs,
            draft_tokens,
            accept_mask,
            n_accepted,
            threshold,
            k as u32,
            vocab_size as u32,
        );
        return Ok(());
    }
    #[cfg(not(darkforest_cuda_kernels))]
    Err(anyhow::Error::msg("CUDA kernels not compiled"))
}

/// Gradient Checkpointing: elementwise scale for recomputed activations.
///
/// Based on: Chen et al., "Training Deep Nets with Sublinear Memory Cost", 2016.
/// Enables O(sqrt(L)) activation memory by recomputing forward activations on demand
/// during the backward pass. This kernel applies the necessary scale correction factor.
/// float4 vectorized for maximum HBM bandwidth utilization on sm_120.
pub fn cuda_gradient_checkpoint_scale(
    x: *const f32,
    out: *mut f32,
    scale: f32,
    n: usize,
) -> Result<()> {
    #[cfg(darkforest_cuda_kernels)]
    unsafe {
        extern "C" {
            fn launch_gradient_checkpoint_scale(x: *const f32, out: *mut f32, scale: f32, n: u32);
        }
        launch_gradient_checkpoint_scale(x, out, scale, n as u32);
        return Ok(());
    }
    #[cfg(not(darkforest_cuda_kernels))]
    Err(anyhow::Error::msg("CUDA kernels not compiled"))
}
