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
// Attention backward — correctness baseline (materializes probabilities)
// ---------------------------------------------------------------------------
__global__ void kernel_attention_bwd(
    const float* __restrict__ Q,
    const float* __restrict__ K,
    const float* __restrict__ V,
    const float* __restrict__ dOut,
    float* __restrict__ dQ,
    float* __restrict__ dK,
    float* __restrict__ dV,
    float* __restrict__ probabilities,
    uint32_t seq_len,
    uint32_t d_head,
    float scale,
    uint32_t is_causal
) {
    uint32_t q_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (q_idx >= seq_len) return;

    uint32_t max_k = is_causal ? (q_idx + 1) : seq_len;

    float row_max = -FLT_MAX;
    for (uint32_t k_idx = 0; k_idx < max_k; ++k_idx) {
        float score = 0.0f;
        for (uint32_t d = 0; d < d_head; ++d)
            score += __ldg(&Q[q_idx * d_head + d]) * __ldg(&K[k_idx * d_head + d]);
        score *= scale;
        probabilities[q_idx * seq_len + k_idx] = score;
        row_max = fmaxf(row_max, score);
    }

    float normalizer = 0.0f;
    for (uint32_t k_idx = 0; k_idx < max_k; ++k_idx) {
        float probability = __expf(probabilities[q_idx * seq_len + k_idx] - row_max);
        probabilities[q_idx * seq_len + k_idx] = probability;
        normalizer += probability;
    }
    for (uint32_t k_idx = 0; k_idx < max_k; ++k_idx)
        probabilities[q_idx * seq_len + k_idx] /= normalizer;

    float d_probability_sum = 0.0f;
    for (uint32_t k_idx = 0; k_idx < max_k; ++k_idx) {
        float d_probability = 0.0f;
        for (uint32_t d = 0; d < d_head; ++d)
            d_probability += __ldg(&dOut[q_idx * d_head + d]) * __ldg(&V[k_idx * d_head + d]);
        d_probability_sum += d_probability * probabilities[q_idx * seq_len + k_idx];
    }

    for (uint32_t d = 0; d < d_head; ++d) {
        float d_q = 0.0f;
        for (uint32_t k_idx = 0; k_idx < max_k; ++k_idx) {
            float d_probability = 0.0f;
            for (uint32_t od = 0; od < d_head; ++od)
                d_probability += __ldg(&dOut[q_idx * d_head + od]) * __ldg(&V[k_idx * d_head + od]);
            float d_score = probabilities[q_idx * seq_len + k_idx]
                          * (d_probability - d_probability_sum);
            d_q += d_score * __ldg(&K[k_idx * d_head + d]);
            atomicAdd(&dK[k_idx * d_head + d], scale * d_score * __ldg(&Q[q_idx * d_head + d]));
            atomicAdd(&dV[k_idx * d_head + d],
                      probabilities[q_idx * seq_len + k_idx] * __ldg(&dOut[q_idx * d_head + d]));
        }
        dQ[q_idx * d_head + d] = scale * d_q;
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
    uint32_t threads = 128;
    uint32_t blocks  = (seq_len + threads - 1) / threads;
    kernel_attention_bwd<<<blocks, threads>>>(
        q, k, v, d_out, d_q, d_k, d_v, probabilities,
        seq_len, d_head, scale, causal
    );
}

// ---------------------------------------------------------------------------
// Multi-Head Attention (MHA) Backward Pass
// Q, K, V, dOut, dQ, dK, dV shape: [seq_len, n_heads * d_head] = [seq_len, d_model]
// Probabilities shape: [n_heads, seq_len, seq_len]
// ---------------------------------------------------------------------------
__global__ void kernel_mha_bwd(
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
    uint32_t q_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (h >= n_heads || q_idx >= seq_len) return;

    uint32_t d_model = n_heads * d_head;
    uint32_t head_offset = h * d_head;
    float* prob_head = probabilities + (size_t)h * seq_len * seq_len;

    uint32_t max_k = is_causal ? (q_idx + 1) : seq_len;

    const float* q_row = Q + q_idx * d_model + head_offset;
    const float* dout_row = dOut + q_idx * d_model + head_offset;

    float row_max = -FLT_MAX;
    for (uint32_t k_idx = 0; k_idx < max_k; ++k_idx) {
        const float* k_row = K + k_idx * d_model + head_offset;
        float score = 0.0f;
        for (uint32_t d = 0; d < d_head; ++d)
            score += __ldg(&q_row[d]) * __ldg(&k_row[d]);
        score *= scale;
        prob_head[q_idx * seq_len + k_idx] = score;
        row_max = fmaxf(row_max, score);
    }

    float normalizer = 0.0f;
    for (uint32_t k_idx = 0; k_idx < max_k; ++k_idx) {
        float probability = __expf(prob_head[q_idx * seq_len + k_idx] - row_max);
        prob_head[q_idx * seq_len + k_idx] = probability;
        normalizer += probability;
    }
    for (uint32_t k_idx = 0; k_idx < max_k; ++k_idx)
        prob_head[q_idx * seq_len + k_idx] /= normalizer;

    float d_probability_sum = 0.0f;
    // Precompute per-kv-key dot product of dOut with V (same for all d)
    float d_probs[2048]; // supports seq_len <= 2048
    for (uint32_t k_idx = 0; k_idx < max_k; ++k_idx) {
        const float* v_row = V + k_idx * d_model + head_offset;
        float dp = 0.0f;
        #pragma unroll 8
        for (uint32_t d = 0; d < d_head; ++d)
            dp += __ldg(&dout_row[d]) * __ldg(&v_row[d]);
        d_probs[k_idx] = dp;
        d_probability_sum += dp * prob_head[q_idx * seq_len + k_idx];
    }

    // Parallelize inner computation
    for (uint32_t d = 0; d < d_head; ++d) {
        float d_q = 0.0f;
        #pragma unroll 8
        for (uint32_t k_idx = 0; k_idx < max_k; ++k_idx) {
            const float* k_row = K + k_idx * d_model + head_offset;
            const float* v_row = V + k_idx * d_model + head_offset;
            float p_val = prob_head[q_idx * seq_len + k_idx];
            float d_score = p_val * (d_probs[k_idx] - d_probability_sum);
            d_q += d_score * __ldg(&k_row[d]);
            atomicAdd(&dK[k_idx * d_model + head_offset + d], scale * d_score * __ldg(&q_row[d]));
            atomicAdd(&dV[k_idx * d_model + head_offset + d], p_val * __ldg(&dout_row[d]));
        }
        dQ[q_idx * d_model + head_offset + d] = scale * d_q;
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
    uint32_t threads = 128;
    uint32_t blocks_x = (seq_len + threads - 1) / threads;
    dim3 grid(blocks_x, n_heads);
    kernel_mha_bwd<<<grid, threads>>>(
        q, k, v, d_out, d_q, d_k, d_v, probabilities,
        n_heads, seq_len, d_head, scale, causal
    );
}


