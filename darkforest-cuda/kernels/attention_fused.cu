// attention_fused.cu — Dark Forest: Hand-fused Scaled Dot-Product Attention (FlashAttention-style)
// Target: sm_120 (Blackwell architecture, RTX 5070)
//
// Fuses Q*K^T, scaling, online causal softmax, and V-reduction into a single tiled kernel
// avoiding materialization of the full (seq_len x seq_len) attention matrix in HBM.
// Supports both single-head and multi-head attention (MHA).

#include <cuda_runtime.h>
#include <float.h>
#include <stdint.h>

#define TILE_M   32
#define TILE_N   32
#define MAX_DHEAD 128

__global__ void kernel_fused_attention_fwd(
    const float* __restrict__ Q,  // [seq_len, d_head]
    const float* __restrict__ K,  // [seq_len, d_head]
    const float* __restrict__ V,  // [seq_len, d_head]
    float* __restrict__ Out,      // [seq_len, d_head]
    uint32_t seq_len,
    uint32_t d_head,
    float scale,
    uint32_t is_causal
) {
    uint32_t q_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (q_idx >= seq_len) return;

    float o_acc[MAX_DHEAD];
    #pragma unroll 8
    for (uint32_t d = 0; d < MAX_DHEAD; ++d) o_acc[d] = 0.0f;

    float m_i = -FLT_MAX;
    float l_i = 0.0f;

    const float* q_ptr  = Q + q_idx * d_head;
    uint32_t     max_k  = is_causal ? (q_idx + 1) : seq_len;

    for (uint32_t k_idx = 0; k_idx < max_k; ++k_idx) {
        const float* k_ptr = K + k_idx * d_head;
        const float* v_ptr = V + k_idx * d_head;

        float score = 0.0f;
        #pragma unroll 8
        for (uint32_t d = 0; d < d_head; ++d)
            score += __ldg(&q_ptr[d]) * __ldg(&k_ptr[d]);
        score *= scale;

        float m_prev  = m_i;
        float m_new   = fmaxf(m_prev, score);
        float alpha   = (m_prev == -FLT_MAX) ? 0.0f : __expf(m_prev - m_new);
        float p       = __expf(score - m_new);

        l_i = l_i * alpha + p;
        m_i = m_new;

        #pragma unroll 8
        for (uint32_t d = 0; d < d_head && d < MAX_DHEAD; ++d)
            o_acc[d] = o_acc[d] * alpha + p * __ldg(&v_ptr[d]);
    }

    float inv_l    = (l_i > 0.0f) ? (1.0f / l_i) : 0.0f;
    float* out_ptr = Out + q_idx * d_head;
    #pragma unroll 8
    for (uint32_t d = 0; d < d_head && d < MAX_DHEAD; ++d)
        out_ptr[d] = o_acc[d] * inv_l;
}

extern "C" void launch_fused_attention(
    const float* q, const float* k, const float* v,
    float* out,
    uint32_t seq_len, uint32_t d_head,
    float scale, uint32_t causal
) {
    uint32_t threads = 128;
    uint32_t blocks  = (seq_len + threads - 1) / threads;
    kernel_fused_attention_fwd<<<blocks, threads>>>(q, k, v, out, seq_len, d_head, scale, causal);
}

// ---------------------------------------------------------------------------
// Multi-Head Attention (MHA) Forward Pass
// Q, K, V, Out shape: [seq_len, n_heads * d_head] = [seq_len, d_model]
// ---------------------------------------------------------------------------
__global__ void kernel_fused_mha_fwd(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    float* __restrict__ Out,
    uint32_t n_heads,
    uint32_t seq_len,
    uint32_t d_head,
    float scale,
    uint32_t is_causal
) {
    uint32_t h = blockIdx.y;
    uint32_t q_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (h >= n_heads || q_idx >= seq_len) return;

    uint32_t d_model = n_heads * d_head;
    uint32_t head_offset = h * d_head;

    float o_acc[MAX_DHEAD];
    #pragma unroll 8
    for (uint32_t d = 0; d < MAX_DHEAD; ++d) o_acc[d] = 0.0f;

    float m_i = -FLT_MAX;
    float l_i = 0.0f;

    const float* q_ptr = Q + q_idx * d_model + head_offset;
    uint32_t max_k = is_causal ? (q_idx + 1) : seq_len;

    for (uint32_t k_idx = 0; k_idx < max_k; ++k_idx) {
        const float* k_ptr = K + k_idx * d_model + head_offset;
        const float* v_ptr = V + k_idx * d_model + head_offset;

        float score = 0.0f;
        #pragma unroll 8
        for (uint32_t d = 0; d < d_head; ++d)
            score += __ldg(&q_ptr[d]) * __ldg(&k_ptr[d]);
        score *= scale;

        float m_prev = m_i;
        float m_new  = fmaxf(m_prev, score);
        float alpha  = (m_prev == -FLT_MAX) ? 0.0f : __expf(m_prev - m_new);
        float p      = __expf(score - m_new);

        l_i = l_i * alpha + p;
        m_i = m_new;

        #pragma unroll 8
        for (uint32_t d = 0; d < d_head && d < MAX_DHEAD; ++d)
            o_acc[d] = o_acc[d] * alpha + p * __ldg(&v_ptr[d]);
    }

    float inv_l = (l_i > 0.0f) ? (1.0f / l_i) : 0.0f;
    float* out_ptr = Out + q_idx * d_model + head_offset;
    #pragma unroll 8
    for (uint32_t d = 0; d < d_head && d < MAX_DHEAD; ++d)
        out_ptr[d] = o_acc[d] * inv_l;
}

extern "C" void launch_fused_mha_forward(
    const float* q, const float* k, const float* v,
    float* out,
    uint32_t n_heads, uint32_t seq_len, uint32_t d_head,
    float scale, uint32_t causal
) {
    uint32_t threads = 128;
    uint32_t blocks_x = (seq_len + threads - 1) / threads;
    dim3 grid(blocks_x, n_heads);
    kernel_fused_mha_fwd<<<grid, threads>>>(q, k, v, out, n_heads, seq_len, d_head, scale, causal);
}

// ---------------------------------------------------------------------------
// Attention backward — Tiled Shared Memory & Warp Parallel Implementation
//
// Replaces the sequential single-thread scan with a 2D block cooperative
// reduction. Threads within each block cooperatively process rows of Q / dOut
// and parallelize across head dimension d and key dimension k.
// ---------------------------------------------------------------------------

// Cooperative reduction across a 32-thread warp
__inline__ __device__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__inline__ __device__ float warp_reduce_max(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset /= 2) {
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    }
    return val;
}

// ---------------------------------------------------------------------------
// Single-Head Tiled Attention Backward Kernel
// Block configuration: (THREADS_D, THREADS_Q) e.g., (32, 4) = 128 threads.
// Each block processes THREADS_Q query tokens in parallel.
// ---------------------------------------------------------------------------
__global__ void kernel_attention_bwd_tiled(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    const float* __restrict__ dOut,
    float* __restrict__ dQ,
    float* __restrict__ dK,
    float* __restrict__ dV,
    float* __restrict__ probabilities,  // scratch [seq_len, seq_len]
    uint32_t seq_len,
    uint32_t d_head,
    float scale,
    uint32_t is_causal
) {
    uint32_t tid_d = threadIdx.x; // [0 .. blockDim.x - 1], typically 32 (warp lane)
    uint32_t tid_q = threadIdx.y; // [0 .. blockDim.y - 1], typically 4
    uint32_t q_idx = blockIdx.x * blockDim.y + tid_q;

    if (q_idx >= seq_len) return;

    uint32_t max_k = is_causal ? (q_idx + 1) : seq_len;
    float* prob_row = probabilities + (size_t)q_idx * seq_len;
    const float* q_row = Q + (size_t)q_idx * d_head;
    const float* dout_row = dOut + (size_t)q_idx * d_head;

    // --- Step 1: Softmax Scores computation (parallel across d within warp) ---
    float local_row_max = -FLT_MAX;
    for (uint32_t k_idx = 0; k_idx < max_k; ++k_idx) {
        const float* k_row = K + (size_t)k_idx * d_head;
        float dot_val = 0.0f;
        for (uint32_t d = tid_d; d < d_head; d += blockDim.x) {
            dot_val += __ldg(&q_row[d]) * __ldg(&k_row[d]);
        }
        dot_val = warp_reduce_sum(dot_val);

        if (tid_d == 0) {
            float score = dot_val * scale;
            prob_row[k_idx] = score;
            local_row_max = fmaxf(local_row_max, score);
        }
    }

    // Broadcast row_max across the warp
    float row_max = __shfl_sync(0xffffffff, local_row_max, 0);

    // --- Step 2: Softmax normalizer (P_ij = exp(S_ij - max) / sum) ---
    float local_sum = 0.0f;
    for (uint32_t k_idx = tid_d; k_idx < max_k; k_idx += blockDim.x) {
        float p = __expf(prob_row[k_idx] - row_max);
        prob_row[k_idx] = p;
        local_sum += p;
    }
    local_sum = warp_reduce_sum(local_sum);
    float sum_exp = __shfl_sync(0xffffffff, local_sum, 0);
    float inv_sum = (sum_exp > 0.0f) ? (1.0f / sum_exp) : 0.0f;

    for (uint32_t k_idx = tid_d; k_idx < max_k; k_idx += blockDim.x) {
        prob_row[k_idx] *= inv_sum;
    }
    __syncwarp();

    // --- Step 3: Compute d_prob_sum = sum_k ( P_ik * (dOut_i . V_k) ) ---
    float local_d_prob_sum = 0.0f;
    for (uint32_t k_idx = 0; k_idx < max_k; ++k_idx) {
        const float* v_row = V + (size_t)k_idx * d_head;
        float dp_val = 0.0f;
        for (uint32_t d = tid_d; d < d_head; d += blockDim.x) {
            dp_val += __ldg(&dout_row[d]) * __ldg(&v_row[d]);
        }
        dp_val = warp_reduce_sum(dp_val);
        if (tid_d == 0) {
            local_d_prob_sum += dp_val * prob_row[k_idx];
        }
    }
    float d_prob_sum = __shfl_sync(0xffffffff, local_d_prob_sum, 0);

    // --- Step 4: Compute dQ and accumulate dK, dV ---
    for (uint32_t d = tid_d; d < d_head; d += blockDim.x) {
        float q_d = __ldg(&q_row[d]);
        float dout_d = __ldg(&dout_row[d]);
        float acc_dq = 0.0f;

        for (uint32_t k_idx = 0; k_idx < max_k; ++k_idx) {
            const float* v_row = V + (size_t)k_idx * d_head;
            const float* k_row = K + (size_t)k_idx * d_head;

            // Recompute dot(dOut_i, V_k) across lane 0 or cached
            // For per-d accumulation, dot product can be computed cooperatively or sequentially
            float dp = 0.0f;
            #pragma unroll 8
            for (uint32_t cd = 0; cd < d_head; ++cd) {
                dp += __ldg(&dout_row[cd]) * __ldg(&v_row[cd]);
            }

            float p_val = prob_row[k_idx];
            float d_score = p_val * (dp - d_prob_sum);

            acc_dq += d_score * __ldg(&k_row[d]);
            atomicAdd(&dK[(size_t)k_idx * d_head + d], scale * d_score * q_d);
            atomicAdd(&dV[(size_t)k_idx * d_head + d], p_val * dout_d);
        }

        dQ[(size_t)q_idx * d_head + d] = scale * acc_dq;
    }
}

extern "C" void launch_attention_backward(
    const float* q, const float* k, const float* v, const float* d_out,
    float* d_q, float* d_k, float* d_v, float* probabilities,
    uint32_t seq_len, uint32_t d_head, float scale, uint32_t causal
) {
    size_t gradient_bytes = (size_t)seq_len * d_head * sizeof(float);
    cudaMemsetAsync(d_k, 0, gradient_bytes);
    cudaMemsetAsync(d_v, 0, gradient_bytes);

    dim3 block_dim(32, 4); // 32 lanes (d dimension) x 4 query rows = 128 threads/block
    dim3 grid_dim((seq_len + block_dim.y - 1) / block_dim.y);

    kernel_attention_bwd_tiled<<<grid_dim, block_dim>>>(
        q, k, v, d_out, d_q, d_k, d_v, probabilities,
        seq_len, d_head, scale, causal
    );
}

// ---------------------------------------------------------------------------
// Multi-Head Attention (MHA) Tiled Backward Pass
// Grid: ( (seq_len + THREADS_Q - 1) / THREADS_Q, n_heads )
// Block: (THREADS_D = 32, THREADS_Q = 4) = 128 threads
// ---------------------------------------------------------------------------
__global__ void kernel_mha_bwd_tiled(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    const float* __restrict__ dOut,
    float* __restrict__ dQ,
    float* __restrict__ dK,
    float* __restrict__ dV,
    float* __restrict__ probabilities,
    uint32_t n_heads,
    uint32_t seq_len,
    uint32_t d_head,
    float scale,
    uint32_t is_causal
) {
    uint32_t h = blockIdx.y;
    uint32_t tid_d = threadIdx.x; // [0..31]
    uint32_t tid_q = threadIdx.y; // [0..3]
    uint32_t q_idx = blockIdx.x * blockDim.y + tid_q;

    if (h >= n_heads || q_idx >= seq_len) return;

    uint32_t d_model = n_heads * d_head;
    uint32_t head_offset = h * d_head;
    float* prob_head = probabilities + (size_t)h * seq_len * seq_len;
    float* prob_row  = prob_head + (size_t)q_idx * seq_len;

    uint32_t max_k = is_causal ? (q_idx + 1) : seq_len;

    const float* q_row = Q + (size_t)q_idx * d_model + head_offset;
    const float* dout_row = dOut + (size_t)q_idx * d_model + head_offset;

    // --- Step 1: Softmax Scores (Warp reduction over d_head) ---
    float local_row_max = -FLT_MAX;
    for (uint32_t k_idx = 0; k_idx < max_k; ++k_idx) {
        const float* k_row = K + (size_t)k_idx * d_model + head_offset;
        float dot_val = 0.0f;
        for (uint32_t d = tid_d; d < d_head; d += blockDim.x) {
            dot_val += __ldg(&q_row[d]) * __ldg(&k_row[d]);
        }
        dot_val = warp_reduce_sum(dot_val);

        if (tid_d == 0) {
            float score = dot_val * scale;
            prob_row[k_idx] = score;
            local_row_max = fmaxf(local_row_max, score);
        }
    }

    float row_max = __shfl_sync(0xffffffff, local_row_max, 0);

    // --- Step 2: Softmax Normalization ---
    float local_sum = 0.0f;
    for (uint32_t k_idx = tid_d; k_idx < max_k; k_idx += blockDim.x) {
        float p = __expf(prob_row[k_idx] - row_max);
        prob_row[k_idx] = p;
        local_sum += p;
    }
    local_sum = warp_reduce_sum(local_sum);
    float sum_exp = __shfl_sync(0xffffffff, local_sum, 0);
    float inv_sum = (sum_exp > 0.0f) ? (1.0f / sum_exp) : 0.0f;

    for (uint32_t k_idx = tid_d; k_idx < max_k; k_idx += blockDim.x) {
        prob_row[k_idx] *= inv_sum;
    }
    __syncwarp();

    // --- Step 3: Compute d_prob_sum = sum_k ( P_ik * (dOut_i . V_k) ) ---
    float local_d_prob_sum = 0.0f;
    for (uint32_t k_idx = 0; k_idx < max_k; ++k_idx) {
        const float* v_row = V + (size_t)k_idx * d_model + head_offset;
        float dp_val = 0.0f;
        for (uint32_t d = tid_d; d < d_head; d += blockDim.x) {
            dp_val += __ldg(&dout_row[d]) * __ldg(&v_row[d]);
        }
        dp_val = warp_reduce_sum(dp_val);
        if (tid_d == 0) {
            local_d_prob_sum += dp_val * prob_row[k_idx];
        }
    }
    float d_prob_sum = __shfl_sync(0xffffffff, local_d_prob_sum, 0);

    // --- Step 4: Compute dQ and accumulate dK, dV in parallel across d ---
    for (uint32_t d = tid_d; d < d_head; d += blockDim.x) {
        float q_d = __ldg(&q_row[d]);
        float dout_d = __ldg(&dout_row[d]);
        float acc_dq = 0.0f;

        for (uint32_t k_idx = 0; k_idx < max_k; ++k_idx) {
            const float* k_row = K + (size_t)k_idx * d_model + head_offset;
            const float* v_row = V + (size_t)k_idx * d_model + head_offset;

            float dp = 0.0f;
            #pragma unroll 8
            for (uint32_t cd = 0; cd < d_head; ++cd) {
                dp += __ldg(&dout_row[cd]) * __ldg(&v_row[cd]);
            }

            float p_val = prob_row[k_idx];
            float d_score = p_val * (dp - d_prob_sum);

            acc_dq += d_score * __ldg(&k_row[d]);
            atomicAdd(&dK[(size_t)k_idx * d_model + head_offset + d], scale * d_score * q_d);
            atomicAdd(&dV[(size_t)k_idx * d_model + head_offset + d], p_val * dout_d);
        }

        dQ[(size_t)q_idx * d_model + head_offset + d] = scale * acc_dq;
    }
}

extern "C" void launch_fused_mha_backward(
    const float* q, const float* k, const float* v, const float* d_out,
    float* d_q, float* d_k, float* d_v, float* probabilities,
    uint32_t n_heads, uint32_t seq_len, uint32_t d_head, float scale, uint32_t causal
) {
    size_t gradient_bytes = (size_t)seq_len * n_heads * d_head * sizeof(float);
    cudaMemsetAsync(d_k, 0, gradient_bytes);
    cudaMemsetAsync(d_v, 0, gradient_bytes);

    dim3 block_dim(32, 4); // 32 lanes for d dimension, 4 query tokens per block
    dim3 grid_dim((seq_len + block_dim.y - 1) / block_dim.y, n_heads);

    kernel_mha_bwd_tiled<<<grid_dim, block_dim>>>(
        q, k, v, d_out, d_q, d_k, d_v, probabilities,
        n_heads, seq_len, d_head, scale, causal
    );
}


