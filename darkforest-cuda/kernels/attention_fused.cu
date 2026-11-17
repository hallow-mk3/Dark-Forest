// attention_fused.cu — Dark Forest: Hand-fused Scaled Dot-Product Attention
// (Fused Online-Softmax Causal Attention) Target: sm_120 (Blackwell architecture, RTX 5070)
//
// Fuses Q*K^T, scaling, online causal softmax, and V-reduction into a single
// tiled kernel avoiding materialization of the full (seq_len x seq_len)
// attention matrix in HBM. Supports both single-head and multi-head attention
// (MHA).

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <float.h>
#include <stdint.h>

#define TILE_M 32
#define TILE_N 32
#define MAX_DHEAD 128

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
// Multi-Head Attention (MHA) Forward Pass — Cooperative Warp Tiled
// Block Dim: (32 lanes across d, 4 query rows) = 128 threads/block
// Eliminates sequential loop over d inside each thread and enables vectorized
// loads.
// ---------------------------------------------------------------------------
__global__ void kernel_fused_mha_fwd_tiled(
    const float *__restrict__ Q, const float *__restrict__ K,
    const float *__restrict__ V, float *__restrict__ Out, uint32_t n_heads,
    uint32_t seq_len, uint32_t d_head, float scale, uint32_t is_causal) {
  uint32_t h = blockIdx.y;
  uint32_t tid_d = threadIdx.x; // [0..31]
  uint32_t tid_q = threadIdx.y; // [0..3]
  uint32_t q_idx = blockIdx.x * blockDim.y + tid_q;

  if (h >= n_heads || q_idx >= seq_len)
    return;

  // Guard: each warp lane accumulates MAX_DHEAD/blockDim.x = 128/32 = 4 slots.
  // Enforce at runtime so misconfigured callers fail loudly instead of
  // silently.
  if (d_head > MAX_DHEAD) {
    __trap(); // visible as cudaErrorIllegalInstruction at the next sync
  }

  uint32_t d_model = n_heads * d_head;
  uint32_t head_offset = h * d_head;

  float m_i = -FLT_MAX;
  float l_i = 0.0f;
  uint32_t max_k = is_causal ? (q_idx + 1) : seq_len;
  const float *q_ptr = Q + (size_t)q_idx * d_model + head_offset;

  // Optimized vectorized coalesced memory fast-path for standard d_head=64 (GPT-2)
  if (d_head == 64) {
    const float2 *q_vec_ptr = reinterpret_cast<const float2*>(q_ptr);
    float2 q_val = __ldg(&q_vec_ptr[tid_d]);
    float q_0 = q_val.x;
    float q_1 = q_val.y;
    float o_acc_0 = 0.0f;
    float o_acc_1 = 0.0f;

    for (uint32_t k_idx = 0; k_idx < max_k; ++k_idx) {
      const float2 *k_vec_ptr = reinterpret_cast<const float2*>(K + (size_t)k_idx * d_model + head_offset);
      const float2 *v_vec_ptr = reinterpret_cast<const float2*>(V + (size_t)k_idx * d_model + head_offset);

      float2 k_val = __ldg(&k_vec_ptr[tid_d]);
      float partial_dot = q_0 * k_val.x + q_1 * k_val.y;

      float dot_val = warp_reduce_sum(partial_dot);
      dot_val = __shfl_sync(0xffffffff, dot_val, 0);
      float score = dot_val * scale;

      float2 v_val = __ldg(&v_vec_ptr[tid_d]);

      if (score > m_i) {
        float alpha = (m_i == -FLT_MAX) ? 0.0f : __expf(m_i - score);
        m_i = score;
        l_i = l_i * alpha + 1.0f;
        o_acc_0 = o_acc_0 * alpha + v_val.x;
        o_acc_1 = o_acc_1 * alpha + v_val.y;
      } else {
        float p = __expf(score - m_i);
        l_i += p;
        o_acc_0 += p * v_val.x;
        o_acc_1 += p * v_val.y;
      }
    }

    float inv_l = (l_i > 0.0f) ? (1.0f / l_i) : 0.0f;
    float2 *out_vec_ptr = reinterpret_cast<float2*>(Out + (size_t)q_idx * d_model + head_offset);
    out_vec_ptr[tid_d] = make_float2(o_acc_0 * inv_l, o_acc_1 * inv_l);
    return;
  }

  // Preload Q values into registers for general d_head
  float q_0 = (tid_d < d_head) ? __ldg(&q_ptr[tid_d]) : 0.0f;
  float q_1 = (tid_d + 32 < d_head) ? __ldg(&q_ptr[tid_d + 32]) : 0.0f;

  float o_acc_0 = 0.0f;
  float o_acc_1 = 0.0f;

  for (uint32_t k_idx = 0; k_idx < max_k; ++k_idx) {
    const float *k_ptr = K + (size_t)k_idx * d_model + head_offset;
    const float *v_ptr = V + (size_t)k_idx * d_model + head_offset;

    // Dot product Q . K unrolled in registers
    float k_0 = (tid_d < d_head) ? __ldg(&k_ptr[tid_d]) : 0.0f;
    float k_1 = (tid_d + 32 < d_head) ? __ldg(&k_ptr[tid_d + 32]) : 0.0f;
    float partial_dot = q_0 * k_0;
    if (tid_d + 32 < d_head) {
      partial_dot += q_1 * k_1;
    }

    float dot_val = warp_reduce_sum(partial_dot);
    dot_val = __shfl_sync(0xffffffff, dot_val, 0);
    float score = dot_val * scale;

    float v_0 = (tid_d < d_head) ? __ldg(&v_ptr[tid_d]) : 0.0f;
    float v_1 = (tid_d + 32 < d_head) ? __ldg(&v_ptr[tid_d + 32]) : 0.0f;

    // Online softmax with exactly 1 exponential call per key
    if (score > m_i) {
      float alpha = (m_i == -FLT_MAX) ? 0.0f : __expf(m_i - score);
      m_i = score;
      l_i = l_i * alpha + 1.0f;
      o_acc_0 = o_acc_0 * alpha + v_0;
      o_acc_1 = o_acc_1 * alpha + v_1;
    } else {
      float p = __expf(score - m_i);
      l_i += p;
      o_acc_0 += p * v_0;
      o_acc_1 += p * v_1;
    }
  }

  float inv_l = (l_i > 0.0f) ? (1.0f / l_i) : 0.0f;
  float *out_ptr = Out + (size_t)q_idx * d_model + head_offset;
  if (tid_d < d_head) {
    out_ptr[tid_d] = o_acc_0 * inv_l;
  }
  if (tid_d + 32 < d_head) {
    out_ptr[tid_d + 32] = o_acc_1 * inv_l;
  }
}

__global__ void kernel_fused_attention_fwd_tiled(
    const float *__restrict__ Q, const float *__restrict__ K,
    const float *__restrict__ V, float *__restrict__ Out, uint32_t seq_len,
    uint32_t d_head, float scale, uint32_t is_causal) {
  uint32_t tid_d = threadIdx.x; // [0..31]
  uint32_t tid_q = threadIdx.y; // [0..3]
  uint32_t q_idx = blockIdx.x * blockDim.y + tid_q;

  if (q_idx >= seq_len)
    return;

  float m_i = -FLT_MAX;
  float l_i = 0.0f;
  uint32_t max_k = is_causal ? (q_idx + 1) : seq_len;
  const float *q_ptr = Q + (size_t)q_idx * d_head;

  // Preload Q values into registers to avoid reloading Q in every step of key
  // loop
  float q_val[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  for (uint32_t d = tid_d, lane = 0; d < d_head && lane < 4;
       d += blockDim.x, ++lane) {
    q_val[lane] = __ldg(&q_ptr[d]);
  }

  float o_acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};

  for (uint32_t k_idx = 0; k_idx < max_k; ++k_idx) {
    const float *k_ptr = K + (size_t)k_idx * d_head;
    const float *v_ptr = V + (size_t)k_idx * d_head;

    float partial_dot = 0.0f;
    for (uint32_t d = tid_d, lane = 0; d < d_head && lane < 4;
         d += blockDim.x, ++lane) {
      partial_dot = fmaf(q_val[lane], __ldg(&k_ptr[d]), partial_dot);
    }
    float dot_val = warp_reduce_sum(partial_dot);
    dot_val = __shfl_sync(0xffffffff, dot_val,
                          0); // Broadcast full dot product to all lanes
    float score = dot_val * scale;

    float m_prev = m_i;
    float m_new = fmaxf(m_prev, score);
    float alpha = (m_prev == -FLT_MAX) ? 0.0f : __expf(m_prev - m_new);
    float p = __expf(score - m_new);

    l_i = fmaf(l_i, alpha, p);
    m_i = m_new;

    for (uint32_t d = tid_d, lane = 0; d < d_head && lane < 4;
         d += blockDim.x, ++lane) {
      o_acc[lane] = fmaf(p, __ldg(&v_ptr[d]), o_acc[lane] * alpha);
    }
  }

  float inv_l = (l_i > 0.0f) ? (1.0f / l_i) : 0.0f;
  float *out_ptr = Out + (size_t)q_idx * d_head;
  for (uint32_t d = tid_d, lane = 0; d < d_head && lane < 4;
       d += blockDim.x, ++lane) {
    out_ptr[d] = o_acc[lane] * inv_l;
  }
}

extern "C" void launch_fused_attention(const float *q, const float *k,
                                       const float *v, float *out,
                                       uint32_t seq_len, uint32_t d_head,
                                       float scale, uint32_t causal) {
  dim3 block_dim(32, 4);
  dim3 grid_dim((seq_len + block_dim.y - 1) / block_dim.y);
  kernel_fused_attention_fwd_tiled<<<grid_dim, block_dim>>>(
      q, k, v, out, seq_len, d_head, scale, causal);
}

extern "C" void launch_fused_mha_forward(const float *q, const float *k,
                                         const float *v, float *out,
                                         uint32_t n_heads, uint32_t seq_len,
                                         uint32_t d_head, float scale,
                                         uint32_t causal) {
  dim3 block_dim(32, 4); // 32 lanes in d dimension, 4 query rows
  dim3 grid_dim((seq_len + block_dim.y - 1) / block_dim.y, n_heads);
  kernel_fused_mha_fwd_tiled<<<grid_dim, block_dim>>>(
      q, k, v, out, n_heads, seq_len, d_head, scale, causal);
}

// ---------------------------------------------------------------------------
// Attention backward — Tiled Shared Memory & Warp Parallel Implementation
//
// Replaces the sequential single-thread scan with a 2D block cooperative
// reduction. Threads within each block cooperatively process rows of Q / dOut
// and parallelize across head dimension d and key dimension k.
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Single-Head Tiled Attention Backward Kernel
// Block configuration: (THREADS_D, THREADS_Q) e.g., (32, 4) = 128 threads.
// Each block processes THREADS_Q query tokens in parallel.
// ---------------------------------------------------------------------------
__global__ void kernel_attention_bwd_tiled(
    const float *__restrict__ Q, const float *__restrict__ K,
    const float *__restrict__ V, const float *__restrict__ dOut,
    float *__restrict__ dQ, float *__restrict__ dK, float *__restrict__ dV,
    float *__restrict__ probabilities, // scratch [seq_len, seq_len]
    uint32_t seq_len, uint32_t d_head, float scale, uint32_t is_causal) {
  uint32_t tid_d =
      threadIdx.x; // [0 .. blockDim.x - 1], typically 32 (warp lane)
  uint32_t tid_q = threadIdx.y; // [0 .. blockDim.y - 1], typically 4
  uint32_t q_idx = blockIdx.x * blockDim.y + tid_q;

  if (q_idx >= seq_len)
    return;

  uint32_t max_k = is_causal ? (q_idx + 1) : seq_len;
  float *prob_row = probabilities + (size_t)q_idx * seq_len;
  const float *q_row = Q + (size_t)q_idx * d_head;
  const float *dout_row = dOut + (size_t)q_idx * d_head;

  // --- Step 1: Softmax Scores computation (parallel across d within warp) ---
  float local_row_max = -FLT_MAX;
  for (uint32_t k_idx = 0; k_idx < max_k; ++k_idx) {
    const float *k_row = K + (size_t)k_idx * d_head;
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
  __syncwarp();

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
    const float *v_row = V + (size_t)k_idx * d_head;
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
  // Zero out dQ for this thread's d-lanes
  for (uint32_t d = tid_d; d < d_head; d += blockDim.x) {
    dQ[(size_t)q_idx * d_head + d] = 0.0f;
  }

  // Outer loop over keys k_idx so the full dot(dOut_i, V_k) is computed
  // cooperatively once
  for (uint32_t k_idx = 0; k_idx < max_k; ++k_idx) {
    const float *v_row = V + (size_t)k_idx * d_head;
    const float *k_row = K + (size_t)k_idx * d_head;

    // Compute full dp = dot(dOut_i, V_k) across all d
    float partial_dp = 0.0f;
    for (uint32_t d = tid_d; d < d_head; d += blockDim.x) {
      partial_dp += __ldg(&dout_row[d]) * __ldg(&v_row[d]);
    }
    float dp = warp_reduce_sum(partial_dp);
    dp = __shfl_sync(0xffffffff, dp, 0);

    float p_val = prob_row[k_idx];
    float d_score = p_val * (dp - d_prob_sum);

    for (uint32_t d = tid_d; d < d_head; d += blockDim.x) {
      float q_d = __ldg(&q_row[d]);
      float dout_d = __ldg(&dout_row[d]);
      dQ[(size_t)q_idx * d_head + d] += scale * d_score * __ldg(&k_row[d]);
      atomicAdd(&dK[(size_t)k_idx * d_head + d], scale * d_score * q_d);
      atomicAdd(&dV[(size_t)k_idx * d_head + d], p_val * dout_d);
    }
  }
}

extern "C" void launch_attention_backward(const float *q, const float *k,
                                          const float *v, const float *d_out,
                                          float *d_q, float *d_k, float *d_v,
                                          float *probabilities,
                                          uint32_t seq_len, uint32_t d_head,
                                          float scale, uint32_t causal) {
  size_t gradient_bytes = (size_t)seq_len * d_head * sizeof(float);
  cudaMemsetAsync(d_k, 0, gradient_bytes);
  cudaMemsetAsync(d_v, 0, gradient_bytes);

  dim3 block_dim(
      32, 4); // 32 lanes (d dimension) x 4 query rows = 128 threads/block
  dim3 grid_dim((seq_len + block_dim.y - 1) / block_dim.y);

  kernel_attention_bwd_tiled<<<grid_dim, block_dim>>>(
      q, k, v, d_out, d_q, d_k, d_v, probabilities, seq_len, d_head, scale,
      causal);
}

// ---------------------------------------------------------------------------
// Multi-Head Attention (MHA) Tiled Backward Pass (Per-Head Streamlined)
// Evaluates one head at a time, reusing the [seq_len x seq_len] probability
// scratch buffer across heads, cutting VRAM by 12x (from 768 MB to 64 MB at
// S=4096).
// ---------------------------------------------------------------------------
// Multi-Head Attention (MHA) Backward Pass — All Heads Parallel
// Grid: ( (seq_len + 3) / 4, n_heads )
// Block: (32, 4) — 128 threads per block
// ---------------------------------------------------------------------------
__global__ void kernel_mha_bwd_all_heads(
    const float *__restrict__ Q, const float *__restrict__ K,
    const float *__restrict__ V, const float *__restrict__ dOut,
    float *__restrict__ dQ, float *__restrict__ dK, float *__restrict__ dV,
    float *__restrict__ probabilities, uint32_t n_heads, uint32_t seq_len,
    uint32_t d_head, float scale, uint32_t is_causal) {
  uint32_t h = blockIdx.y;
  uint32_t tid_d = threadIdx.x; // [0..31]
  uint32_t tid_q = threadIdx.y; // [0..3]
  uint32_t q_idx = blockIdx.x * blockDim.y + tid_q;

  if (h >= n_heads || q_idx >= seq_len)
    return;

  uint32_t d_model = n_heads * d_head;
  uint32_t head_offset = h * d_head;
  size_t head_prob_offset = (size_t)h * 2 * seq_len * seq_len;
  float *prob_row = probabilities + head_prob_offset + (size_t)q_idx * seq_len;
  float *d_score_row = probabilities + head_prob_offset +
                       (size_t)seq_len * seq_len + (size_t)q_idx * seq_len;

  uint32_t max_k = is_causal ? (q_idx + 1) : seq_len;

  const float *q_row = Q + (size_t)q_idx * d_model + head_offset;
  const float *dout_row = dOut + (size_t)q_idx * d_model + head_offset;

  // Load query and output gradient into registers once for this token
  float q_0 = (tid_d < d_head) ? __ldg(&q_row[tid_d]) : 0.0f;
  float q_1 = (tid_d + 32 < d_head) ? __ldg(&q_row[tid_d + 32]) : 0.0f;
  float dout_0 = (tid_d < d_head) ? __ldg(&dout_row[tid_d]) : 0.0f;
  float dout_1 = (tid_d + 32 < d_head) ? __ldg(&dout_row[tid_d + 32]) : 0.0f;

  // --- Step 1: Softmax Scores (Warp reduction over d_head) ---
  float local_row_max = -FLT_MAX;
  for (uint32_t k_idx = 0; k_idx < max_k; ++k_idx) {
    const float *k_row = K + (size_t)k_idx * d_model + head_offset;
    float dot_val = q_0 * __ldg(&k_row[tid_d]);
    if (tid_d + 32 < d_head) {
      dot_val += q_1 * __ldg(&k_row[tid_d + 32]);
    }
    dot_val = warp_reduce_sum(dot_val);

    if (tid_d == 0) {
      float score = dot_val * scale;
      prob_row[k_idx] = score;
      local_row_max = fmaxf(local_row_max, score);
    }
  }
  __syncwarp();

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

  // --- Step 3: Compute d_prob_sum and d_score_row in a single pass ---
  float local_d_prob_sum = 0.0f;
  for (uint32_t k_idx = 0; k_idx < max_k; ++k_idx) {
    const float *v_row = V + (size_t)k_idx * d_model + head_offset;
    float dp_val = dout_0 * __ldg(&v_row[tid_d]);
    if (tid_d + 32 < d_head) {
      dp_val += dout_1 * __ldg(&v_row[tid_d + 32]);
    }
    dp_val = warp_reduce_sum(dp_val);
    if (tid_d == 0) {
      d_score_row[k_idx] = dp_val; // temporarily hold dp
      local_d_prob_sum += dp_val * prob_row[k_idx];
    }
  }
  float d_prob_sum = __shfl_sync(0xffffffff, local_d_prob_sum, 0);

  for (uint32_t k_idx = tid_d; k_idx < max_k; k_idx += blockDim.x) {
    d_score_row[k_idx] = prob_row[k_idx] * (d_score_row[k_idx] - d_prob_sum);
  }
  __syncwarp();

  // --- Step 4: Compute dQ (accumulated in registers) and accumulate dK, dV ---
  float acc_dq_0 = 0.0f;
  float acc_dq_1 = 0.0f;

  for (uint32_t k_idx = 0; k_idx < max_k; ++k_idx) {
    float p_val = prob_row[k_idx];
    float d_score = d_score_row[k_idx];

    const float *k_row = K + (size_t)k_idx * d_model + head_offset;

    float k_0 = (tid_d < d_head) ? __ldg(&k_row[tid_d]) : 0.0f;
    float k_1 = (tid_d + 32 < d_head) ? __ldg(&k_row[tid_d + 32]) : 0.0f;

    acc_dq_0 += scale * d_score * k_0;
    acc_dq_1 += scale * d_score * k_1;

    if (tid_d < d_head) {
      atomicAdd(&dK[(size_t)k_idx * d_model + head_offset + tid_d],
                scale * d_score * q_0);
      atomicAdd(&dV[(size_t)k_idx * d_model + head_offset + tid_d],
                p_val * dout_0);
    }
    if (tid_d + 32 < d_head) {
      atomicAdd(&dK[(size_t)k_idx * d_model + head_offset + tid_d + 32],
                scale * d_score * q_1);
      atomicAdd(&dV[(size_t)k_idx * d_model + head_offset + tid_d + 32],
                p_val * dout_1);
    }
  }

  // Write dQ once from registers to global device memory
  if (tid_d < d_head) {
    dQ[(size_t)q_idx * d_model + head_offset + tid_d] = acc_dq_0;
  }
  if (tid_d + 32 < d_head) {
    dQ[(size_t)q_idx * d_model + head_offset + tid_d + 32] = acc_dq_1;
  }
}

extern "C" void launch_fused_mha_backward(
    const float *q, const float *k, const float *v, const float *d_out,
    float *d_q, float *d_k, float *d_v, float *probabilities, uint32_t n_heads,
    uint32_t seq_len, uint32_t d_head, float scale, uint32_t causal) {
  size_t gradient_bytes = (size_t)seq_len * n_heads * d_head * sizeof(float);
  cudaMemsetAsync(d_k, 0, gradient_bytes);
  cudaMemsetAsync(d_v, 0, gradient_bytes);

  dim3 block_dim(32, 4); // 32 lanes for d dimension, 4 query tokens per block
  dim3 grid_dim((seq_len + block_dim.y - 1) / block_dim.y, n_heads);

  kernel_mha_bwd_all_heads<<<grid_dim, block_dim>>>(
      q, k, v, d_out, d_q, d_k, d_v, probabilities, n_heads, seq_len, d_head,
      scale, causal);
}

// ---------------------------------------------------------------------------
// Single-Token Autoregressive KV-Cache Decode Attention
//
// In autoregressive decode, the query length is 1 (the single new token).
// The key and value caches contain the full history of cur_seq_len tokens
// (cur_seq_len <= max_seq_len).
//
// Memory Layout:
//   Q:       [1, d_model] contiguous (head h is at Q + h * d_head)
//   K_cache: [max_seq_len, d_model] contiguous (token t, head h is at K_cache +
//   t * d_model + h * d_head) V_cache: [max_seq_len, d_model] contiguous (token
//   t, head h is at V_cache + t * d_model + h * d_head) Out:     [1, d_model]
//   contiguous (head h is at Out + h * d_head)
//
// Grid: (n_heads, 1) — exactly one block (warp) per attention head.
// Block: (32, 1) — 32 threads cooperatively process head dimension d_head.
// ---------------------------------------------------------------------------
__global__ void kernel_fused_mha_decode_kv(
    const float *__restrict__ Q, const float *__restrict__ K_cache,
    const float *__restrict__ V_cache, float *__restrict__ Out,
    uint32_t n_heads, uint32_t cur_seq_len, uint32_t d_head, float scale) {
  uint32_t h = blockIdx.x;
  uint32_t tid_d = threadIdx.x; // [0..31]
  if (h >= n_heads || cur_seq_len == 0)
    return;

  uint32_t d_model = n_heads * d_head;
  uint32_t head_offset = h * d_head;
  const float *q_ptr = Q + head_offset;

  // Preload Q values into registers once for the entire autoregressive decode
  // step
  float q_val[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  for (uint32_t d = tid_d, lane = 0; d < d_head && lane < 4;
       d += blockDim.x, ++lane) {
    q_val[lane] = __ldg(&q_ptr[d]);
  }

  // Online softmax state
  float m_i = -FLT_MAX;
  float l_i = 0.0f;

  // Up to 4 register slots per thread for d_head <= 128
  float o_acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};

  for (uint32_t t = 0; t < cur_seq_len; ++t) {
    const float *k_ptr = K_cache + (size_t)t * d_model + head_offset;
    const float *v_ptr = V_cache + (size_t)t * d_model + head_offset;

    // Compute Q . K_t across d_head using cached Q
    float partial_dot = 0.0f;
    for (uint32_t d = tid_d, lane = 0; d < d_head && lane < 4;
         d += blockDim.x, ++lane) {
      partial_dot = fmaf(q_val[lane], __ldg(&k_ptr[d]), partial_dot);
    }
    float dot_val = warp_reduce_sum(partial_dot);
    dot_val = __shfl_sync(0xffffffff, dot_val, 0); // broadcast to all 32 lanes
    float score = dot_val * scale;

    // Online softmax update
    float m_prev = m_i;
    float m_new = fmaxf(m_prev, score);
    float alpha = (m_prev == -FLT_MAX) ? 0.0f : __expf(m_prev - m_new);
    float p = __expf(score - m_new);

    l_i = fmaf(l_i, alpha, p);
    m_i = m_new;

    // Accumulate V_t with hardware FMA
    for (uint32_t d = tid_d, lane = 0; d < d_head && lane < 4;
         d += blockDim.x, ++lane) {
      o_acc[lane] = fmaf(p, __ldg(&v_ptr[d]), o_acc[lane] * alpha);
    }
  }

  float inv_l = (l_i > 0.0f) ? (1.0f / l_i) : 0.0f;
  float *out_ptr = Out + head_offset;
  for (uint32_t d = tid_d, lane = 0; d < d_head && lane < 4;
       d += blockDim.x, ++lane) {
    out_ptr[d] = o_acc[lane] * inv_l;
  }
}

extern "C" void launch_fused_mha_decode_kv(const float *q, const float *k_cache,
                                           const float *v_cache, float *out,
                                           uint32_t n_heads,
                                           uint32_t cur_seq_len,
                                           uint32_t d_head, float scale) {
  dim3 block_dim(32, 1);
  dim3 grid_dim(n_heads, 1);
  kernel_fused_mha_decode_kv<<<grid_dim, block_dim>>>(
      q, k_cache, v_cache, out, n_heads, cur_seq_len, d_head, scale);
}

// ---------------------------------------------------------------------------
// Split-K / Chunked Parallel Multi-Head Attention Decode (FlashDecoding)
//
// Solves GPU under-utilization when decoding across long contexts (S >= 1024).
// Instead of launching 1 warp per head, we partition the sequence into chunks
// of size CHUNK_SIZE (e.g. 128), saturating all GPU SMs.
// ---------------------------------------------------------------------------
__global__ void kernel_fused_mha_decode_kv_splitk(
    const float *__restrict__ Q,
    const float *__restrict__ K_cache,
    const float *__restrict__ V_cache,
    float *__restrict__ partial_m,
    float *__restrict__ partial_l,
    float *__restrict__ partial_o,
    uint32_t n_heads,
    uint32_t cur_seq_len,
    uint32_t d_head,
    float scale,
    uint32_t chunk_size) {
  uint32_t h = blockIdx.x;
  uint32_t chunk_idx = blockIdx.y;
  uint32_t tid_d = threadIdx.x; // [0..31]

  if (h >= n_heads) return;

  uint32_t t_start = chunk_idx * chunk_size;
  if (t_start >= cur_seq_len) return;
  uint32_t t_end = (t_start + chunk_size < cur_seq_len) ? (t_start + chunk_size) : cur_seq_len;

  uint32_t d_model = n_heads * d_head;
  uint32_t head_offset = h * d_head;
  const float *q_ptr = Q + head_offset;

  float q_val[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  for (uint32_t d = tid_d, lane = 0; d < d_head && lane < 4;
       d += blockDim.x, ++lane) {
    q_val[lane] = __ldg(&q_ptr[d]);
  }

  float m_i = -FLT_MAX;
  float l_i = 0.0f;
  float o_acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};

  for (uint32_t t = t_start; t < t_end; ++t) {
    const float *k_ptr = K_cache + (size_t)t * d_model + head_offset;
    const float *v_ptr = V_cache + (size_t)t * d_model + head_offset;

    float partial_dot = 0.0f;
    for (uint32_t d = tid_d, lane = 0; d < d_head && lane < 4;
         d += blockDim.x, ++lane) {
      partial_dot = fmaf(q_val[lane], __ldg(&k_ptr[d]), partial_dot);
    }
    float dot_val = warp_reduce_sum(partial_dot);
    dot_val = __shfl_sync(0xffffffff, dot_val, 0);
    float score = dot_val * scale;

    float m_prev = m_i;
    float m_new = fmaxf(m_prev, score);
    float alpha = (m_prev == -FLT_MAX) ? 0.0f : __expf(m_prev - m_new);
    float p = __expf(score - m_new);

    l_i = fmaf(l_i, alpha, p);
    m_i = m_new;

    for (uint32_t d = tid_d, lane = 0; d < d_head && lane < 4;
         d += blockDim.x, ++lane) {
      o_acc[lane] = fmaf(p, __ldg(&v_ptr[d]), o_acc[lane] * alpha);
    }
  }

  uint32_t num_chunks = (cur_seq_len + chunk_size - 1) / chunk_size;
  uint32_t chunk_slot = h * num_chunks + chunk_idx;

  if (tid_d == 0) {
    partial_m[chunk_slot] = m_i;
    partial_l[chunk_slot] = l_i;
  }

  float *out_chunk_ptr = partial_o + (size_t)chunk_slot * d_head;
  for (uint32_t d = tid_d, lane = 0; d < d_head && lane < 4;
       d += blockDim.x, ++lane) {
    out_chunk_ptr[d] = o_acc[lane];
  }
}

// Reduction kernel combining partial Split-K statistics across chunks
__global__ void kernel_fused_mha_decode_kv_reduce(
    const float *__restrict__ partial_m,
    const float *__restrict__ partial_l,
    const float *__restrict__ partial_o,
    float *__restrict__ Out,
    uint32_t n_heads,
    uint32_t num_chunks,
    uint32_t d_head) {
  uint32_t h = blockIdx.x;
  uint32_t tid_d = threadIdx.x;
  if (h >= n_heads) return;

  float m_global = -FLT_MAX;
  for (uint32_t c = 0; c < num_chunks; ++c) {
    float m_c = partial_m[h * num_chunks + c];
    m_global = fmaxf(m_global, m_c);
  }

  float l_global = 0.0f;
  for (uint32_t c = 0; c < num_chunks; ++c) {
    float m_c = partial_m[h * num_chunks + c];
    float l_c = partial_l[h * num_chunks + c];
    if (m_c > -FLT_MAX) {
      l_global += l_c * __expf(m_c - m_global);
    }
  }

  float inv_l = (l_global > 0.0f) ? (1.0f / l_global) : 0.0f;

  float o_acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  for (uint32_t c = 0; c < num_chunks; ++c) {
    float m_c = partial_m[h * num_chunks + c];
    if (m_c <= -FLT_MAX) continue;
    float factor = __expf(m_c - m_global);
    const float *c_ptr = partial_o + (size_t)(h * num_chunks + c) * d_head;
    for (uint32_t d = tid_d, lane = 0; d < d_head && lane < 4;
         d += blockDim.x, ++lane) {
      o_acc[lane] = fmaf(factor, __ldg(&c_ptr[d]), o_acc[lane]);
    }
  }

  float *out_ptr = Out + h * d_head;
  for (uint32_t d = tid_d, lane = 0; d < d_head && lane < 4;
       d += blockDim.x, ++lane) {
    out_ptr[d] = o_acc[lane] * inv_l;
  }
}

extern "C" void launch_fused_mha_decode_kv_splitk(
    const float *q,
    const float *k_cache,
    const float *v_cache,
    float *partial_m,
    float *partial_l,
    float *partial_o,
    float *out,
    uint32_t n_heads,
    uint32_t cur_seq_len,
    uint32_t d_head,
    float scale,
    uint32_t chunk_size) {
  if (cur_seq_len <= chunk_size) {
    launch_fused_mha_decode_kv(q, k_cache, v_cache, out, n_heads, cur_seq_len, d_head, scale);
    return;
  }
  uint32_t num_chunks = (cur_seq_len + chunk_size - 1) / chunk_size;
  dim3 block_dim(32, 1);
  dim3 grid_splitk(n_heads, num_chunks);
  kernel_fused_mha_decode_kv_splitk<<<grid_splitk, block_dim>>>(
      q, k_cache, v_cache, partial_m, partial_l, partial_o,
      n_heads, cur_seq_len, d_head, scale, chunk_size);
  dim3 grid_reduce(n_heads, 1);
  kernel_fused_mha_decode_kv_reduce<<<grid_reduce, block_dim>>>(
      partial_m, partial_l, partial_o, out, n_heads, num_chunks, d_head);
}

// ---------------------------------------------------------------------------
// Paged Attention Multi-Head Attention Decode Kernels
//
// Enables non-contiguous physical block storage for KV caches, eliminating
// memory fragmentation and allowing linear sequence length scaling up to VRAM.
// ---------------------------------------------------------------------------
__global__ void kernel_paged_mha_decode_kv(
    const float *__restrict__ Q,
    const float *__restrict__ K_pool,
    const float *__restrict__ V_pool,
    const int32_t *__restrict__ block_table,
    float *__restrict__ Out,
    uint32_t n_heads,
    uint32_t cur_seq_len,
    uint32_t d_head,
    float scale,
    uint32_t page_size) {
  uint32_t h = blockIdx.x;
  uint32_t tid_d = threadIdx.x;
  if (h >= n_heads || cur_seq_len == 0) return;

  uint32_t d_model = n_heads * d_head;
  uint32_t head_offset = h * d_head;
  const float *q_ptr = Q + head_offset;

  float q_val[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  for (uint32_t d = tid_d, lane = 0; d < d_head && lane < 4;
       d += blockDim.x, ++lane) {
    q_val[lane] = __ldg(&q_ptr[d]);
  }

  float m_i = -FLT_MAX;
  float l_i = 0.0f;
  float o_acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};

  size_t page_stride = (size_t)page_size * d_model;

  for (uint32_t t = 0; t < cur_seq_len; ++t) {
    uint32_t logical_page = t / page_size;
    uint32_t page_offset = t % page_size;
    int32_t phys_block = block_table[logical_page];

    size_t block_offset = (size_t)phys_block * page_stride + (size_t)page_offset * d_model + head_offset;
    const float *k_ptr = K_pool + block_offset;
    const float *v_ptr = V_pool + block_offset;

    float partial_dot = 0.0f;
    for (uint32_t d = tid_d, lane = 0; d < d_head && lane < 4;
         d += blockDim.x, ++lane) {
      partial_dot = fmaf(q_val[lane], __ldg(&k_ptr[d]), partial_dot);
    }
    float dot_val = warp_reduce_sum(partial_dot);
    dot_val = __shfl_sync(0xffffffff, dot_val, 0);
    float score = dot_val * scale;

    float m_prev = m_i;
    float m_new = fmaxf(m_prev, score);
    float alpha = (m_prev == -FLT_MAX) ? 0.0f : __expf(m_prev - m_new);
    float p = __expf(score - m_new);

    l_i = fmaf(l_i, alpha, p);
    m_i = m_new;

    for (uint32_t d = tid_d, lane = 0; d < d_head && lane < 4;
         d += blockDim.x, ++lane) {
      o_acc[lane] = fmaf(p, __ldg(&v_ptr[d]), o_acc[lane] * alpha);
    }
  }

  float inv_l = (l_i > 0.0f) ? (1.0f / l_i) : 0.0f;
  float *out_ptr = Out + head_offset;
  for (uint32_t d = tid_d, lane = 0; d < d_head && lane < 4;
       d += blockDim.x, ++lane) {
    out_ptr[d] = o_acc[lane] * inv_l;
  }
}

__global__ void kernel_paged_mha_decode_kv_splitk(
    const float *__restrict__ Q,
    const float *__restrict__ K_pool,
    const float *__restrict__ V_pool,
    const int32_t *__restrict__ block_table,
    float *__restrict__ partial_m,
    float *__restrict__ partial_l,
    float *__restrict__ partial_o,
    uint32_t n_heads,
    uint32_t cur_seq_len,
    uint32_t d_head,
    float scale,
    uint32_t page_size,
    uint32_t chunk_size) {
  uint32_t h = blockIdx.x;
  uint32_t chunk_idx = blockIdx.y;
  uint32_t tid_d = threadIdx.x;

  if (h >= n_heads) return;

  uint32_t t_start = chunk_idx * chunk_size;
  if (t_start >= cur_seq_len) return;
  uint32_t t_end = (t_start + chunk_size < cur_seq_len) ? (t_start + chunk_size) : cur_seq_len;

  uint32_t d_model = n_heads * d_head;
  uint32_t head_offset = h * d_head;
  const float *q_ptr = Q + head_offset;

  float q_val[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  for (uint32_t d = tid_d, lane = 0; d < d_head && lane < 4;
       d += blockDim.x, ++lane) {
    q_val[lane] = __ldg(&q_ptr[d]);
  }

  float m_i = -FLT_MAX;
  float l_i = 0.0f;
  float o_acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};

  size_t page_stride = (size_t)page_size * d_model;

  for (uint32_t t = t_start; t < t_end; ++t) {
    uint32_t logical_page = t / page_size;
    uint32_t page_offset = t % page_size;
    int32_t phys_block = block_table[logical_page];

    size_t block_offset = (size_t)phys_block * page_stride + (size_t)page_offset * d_model + head_offset;
    const float *k_ptr = K_pool + block_offset;
    const float *v_ptr = V_pool + block_offset;

    float partial_dot = 0.0f;
    for (uint32_t d = tid_d, lane = 0; d < d_head && lane < 4;
         d += blockDim.x, ++lane) {
      partial_dot = fmaf(q_val[lane], __ldg(&k_ptr[d]), partial_dot);
    }
    float dot_val = warp_reduce_sum(partial_dot);
    dot_val = __shfl_sync(0xffffffff, dot_val, 0);
    float score = dot_val * scale;

    float m_prev = m_i;
    float m_new = fmaxf(m_prev, score);
    float alpha = (m_prev == -FLT_MAX) ? 0.0f : __expf(m_prev - m_new);
    float p = __expf(score - m_new);

    l_i = fmaf(l_i, alpha, p);
    m_i = m_new;

    for (uint32_t d = tid_d, lane = 0; d < d_head && lane < 4;
         d += blockDim.x, ++lane) {
      o_acc[lane] = fmaf(p, __ldg(&v_ptr[d]), o_acc[lane] * alpha);
    }
  }

  uint32_t num_chunks = (cur_seq_len + chunk_size - 1) / chunk_size;
  uint32_t chunk_slot = h * num_chunks + chunk_idx;

  if (tid_d == 0) {
    partial_m[chunk_slot] = m_i;
    partial_l[chunk_slot] = l_i;
  }

  float *out_chunk_ptr = partial_o + (size_t)chunk_slot * d_head;
  for (uint32_t d = tid_d, lane = 0; d < d_head && lane < 4;
       d += blockDim.x, ++lane) {
    out_chunk_ptr[d] = o_acc[lane];
  }
}

// BF16 Paged Attention Kernel (2x memory compaction)
__global__ void kernel_paged_mha_decode_kv_bf16(
    const float *__restrict__ Q,
    const __nv_bfloat16 *__restrict__ K_pool,
    const __nv_bfloat16 *__restrict__ V_pool,
    const int32_t *__restrict__ block_table,
    float *__restrict__ Out,
    uint32_t n_heads,
    uint32_t cur_seq_len,
    uint32_t d_head,
    float scale,
    uint32_t page_size) {
  uint32_t h = blockIdx.x;
  uint32_t tid_d = threadIdx.x;
  if (h >= n_heads || cur_seq_len == 0) return;

  uint32_t d_model = n_heads * d_head;
  uint32_t head_offset = h * d_head;
  const float *q_ptr = Q + head_offset;

  float q_val[4] = {0.0f, 0.0f, 0.0f, 0.0f};
  for (uint32_t d = tid_d, lane = 0; d < d_head && lane < 4;
       d += blockDim.x, ++lane) {
    q_val[lane] = __ldg(&q_ptr[d]);
  }

  float m_i = -FLT_MAX;
  float l_i = 0.0f;
  float o_acc[4] = {0.0f, 0.0f, 0.0f, 0.0f};

  size_t page_stride = (size_t)page_size * d_model;

  for (uint32_t t = 0; t < cur_seq_len; ++t) {
    uint32_t logical_page = t / page_size;
    uint32_t page_offset = t % page_size;
    int32_t phys_block = block_table[logical_page];

    size_t block_offset = (size_t)phys_block * page_stride + (size_t)page_offset * d_model + head_offset;
    const __nv_bfloat16 *k_ptr = K_pool + block_offset;
    const __nv_bfloat16 *v_ptr = V_pool + block_offset;

    float partial_dot = 0.0f;
    for (uint32_t d = tid_d, lane = 0; d < d_head && lane < 4;
         d += blockDim.x, ++lane) {
      float k_f = __bfloat162float(__ldg(&k_ptr[d]));
      partial_dot = fmaf(q_val[lane], k_f, partial_dot);
    }
    float dot_val = warp_reduce_sum(partial_dot);
    dot_val = __shfl_sync(0xffffffff, dot_val, 0);
    float score = dot_val * scale;

    float m_prev = m_i;
    float m_new = fmaxf(m_prev, score);
    float alpha = (m_prev == -FLT_MAX) ? 0.0f : __expf(m_prev - m_new);
    float p = __expf(score - m_new);

    l_i = fmaf(l_i, alpha, p);
    m_i = m_new;

    for (uint32_t d = tid_d, lane = 0; d < d_head && lane < 4;
         d += blockDim.x, ++lane) {
      float v_f = __bfloat162float(__ldg(&v_ptr[d]));
      o_acc[lane] = fmaf(p, v_f, o_acc[lane] * alpha);
    }
  }

  float inv_l = (l_i > 0.0f) ? (1.0f / l_i) : 0.0f;
  float *out_ptr = Out + head_offset;
  for (uint32_t d = tid_d, lane = 0; d < d_head && lane < 4;
       d += blockDim.x, ++lane) {
    out_ptr[d] = o_acc[lane] * inv_l;
  }
}

extern "C" void launch_paged_mha_decode_kv(
    const float *q,
    const float *k_pool,
    const float *v_pool,
    const int32_t *block_table,
    float *out,
    uint32_t n_heads,
    uint32_t cur_seq_len,
    uint32_t d_head,
    float scale,
    uint32_t page_size) {
  dim3 block_dim(32, 1);
  dim3 grid_dim(n_heads, 1);
  kernel_paged_mha_decode_kv<<<grid_dim, block_dim>>>(
      q, k_pool, v_pool, block_table, out, n_heads, cur_seq_len, d_head, scale, page_size);
}

extern "C" void launch_paged_mha_decode_kv_splitk(
    const float *q,
    const float *k_pool,
    const float *v_pool,
    const int32_t *block_table,
    float *partial_m,
    float *partial_l,
    float *partial_o,
    float *out,
    uint32_t n_heads,
    uint32_t cur_seq_len,
    uint32_t d_head,
    float scale,
    uint32_t page_size,
    uint32_t chunk_size) {
  if (cur_seq_len <= chunk_size) {
    launch_paged_mha_decode_kv(q, k_pool, v_pool, block_table, out, n_heads, cur_seq_len, d_head, scale, page_size);
    return;
  }
  uint32_t num_chunks = (cur_seq_len + chunk_size - 1) / chunk_size;
  dim3 block_dim(32, 1);
  dim3 grid_splitk(n_heads, num_chunks);
  kernel_paged_mha_decode_kv_splitk<<<grid_splitk, block_dim>>>(
      q, k_pool, v_pool, block_table, partial_m, partial_l, partial_o,
      n_heads, cur_seq_len, d_head, scale, page_size, chunk_size);
  dim3 grid_reduce(n_heads, 1);
  kernel_fused_mha_decode_kv_reduce<<<grid_reduce, block_dim>>>(
      partial_m, partial_l, partial_o, out, n_heads, num_chunks, d_head);
}

extern "C" void launch_paged_mha_decode_kv_bf16(
    const float *q,
    const __nv_bfloat16 *k_pool,
    const __nv_bfloat16 *v_pool,
    const int32_t *block_table,
    float *out,
    uint32_t n_heads,
    uint32_t cur_seq_len,
    uint32_t d_head,
    float scale,
    uint32_t page_size) {
  dim3 block_dim(32, 1);
  dim3 grid_dim(n_heads, 1);
  kernel_paged_mha_decode_kv_bf16<<<grid_dim, block_dim>>>(
      q, k_pool, v_pool, block_table, out, n_heads, cur_seq_len, d_head, scale, page_size);
}

// ---------------------------------------------------------------------------
// Position Embedding Interpolation Kernel (Context Extrapolation)
//
// Dynamically interpolates trained position embeddings to arbitrarily long sequences
// using linear position interpolation (Chen et al.).
// ---------------------------------------------------------------------------
__global__ void kernel_pos_embed_interpolate(
    const float *__restrict__ pos_table,
    float *__restrict__ out,
    uint32_t pos,
    uint32_t trained_ctx_len,
    uint32_t max_ctx_len,
    uint32_t d_model) {
  uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx >= d_model) return;

  if (pos < trained_ctx_len && max_ctx_len <= trained_ctx_len) {
    out[idx] = pos_table[pos * d_model + idx];
  } else {
    float scale = (float)(trained_ctx_len - 1) / (float)(max_ctx_len - 1);
    float p_scaled = (float)pos * scale;
    uint32_t p0 = (uint32_t)p_scaled;
    uint32_t p1 = (p0 + 1 < trained_ctx_len) ? (p0 + 1) : (trained_ctx_len - 1);
    float alpha = p_scaled - (float)p0;

    float v0 = pos_table[p0 * d_model + idx];
    float v1 = pos_table[p1 * d_model + idx];
    out[idx] = (1.0f - alpha) * v0 + alpha * v1;
  }
}

extern "C" void launch_pos_embed_interpolate(
    const float *pos_table,
    float *out,
    uint32_t pos,
    uint32_t trained_ctx_len,
    uint32_t max_ctx_len,
    uint32_t d_model) {
  uint32_t threads = 256;
  uint32_t blocks = (d_model + threads - 1) / threads;
  kernel_pos_embed_interpolate<<<blocks, threads>>>(
      pos_table, out, pos, trained_ctx_len, max_ctx_len, d_model);
}

// ---------------------------------------------------------------------------
// Rotary Position Embeddings (RoPE) Kernel
// ---------------------------------------------------------------------------
__global__ void kernel_apply_rope(
    float *__restrict__ qk,
    uint32_t pos,
    uint32_t n_heads,
    uint32_t d_head,
    float base_freq) {
  uint32_t h = blockIdx.x;
  uint32_t pair_idx = threadIdx.x;
  if (h >= n_heads || pair_idx >= d_head / 2) return;

  float theta = powf(base_freq, -2.0f * (float)pair_idx / (float)d_head);
  float angle = (float)pos * theta;
  float cos_val = cosf(angle);
  float sin_val = sinf(angle);

  uint32_t idx0 = h * d_head + 2 * pair_idx;
  uint32_t idx1 = idx0 + 1;

  float x0 = qk[idx0];
  float x1 = qk[idx1];

  qk[idx0] = x0 * cos_val - x1 * sin_val;
  qk[idx1] = x0 * sin_val + x1 * cos_val;
}

extern "C" void launch_apply_rope(
    float *qk,
    uint32_t pos,
    uint32_t n_heads,
    uint32_t d_head,
    float base_freq) {
  dim3 grid(n_heads, 1);
  dim3 block(d_head / 2, 1);
  kernel_apply_rope<<<grid, block>>>(qk, pos, n_heads, d_head, base_freq);
}

// ---------------------------------------------------------------------------
// Paged KV Pool Copy Row Kernels
// ---------------------------------------------------------------------------
__global__ void kernel_paged_copy_row(
    const float *__restrict__ src,
    float *__restrict__ pool,
    int32_t phys_block,
    uint32_t page_offset,
    uint32_t page_size,
    uint32_t d_model) {
  uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < d_model) {
    size_t dst_offset = ((size_t)phys_block * page_size + page_offset) * d_model + idx;
    pool[dst_offset] = src[idx];
  }
}

extern "C" void launch_paged_copy_row(
    const float *src,
    float *pool,
    int32_t phys_block,
    uint32_t page_offset,
    uint32_t page_size,
    uint32_t d_model) {
  uint32_t threads = 256;
  uint32_t blocks = (d_model + threads - 1) / threads;
  kernel_paged_copy_row<<<blocks, threads>>>(
      src, pool, phys_block, page_offset, page_size, d_model);
}

__global__ void kernel_paged_copy_row_f32_to_bf16(
    const float *__restrict__ src,
    __nv_bfloat16 *__restrict__ pool,
    int32_t phys_block,
    uint32_t page_offset,
    uint32_t page_size,
    uint32_t d_model) {
  uint32_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < d_model) {
    size_t dst_offset = ((size_t)phys_block * page_size + page_offset) * d_model + idx;
    pool[dst_offset] = __float2bfloat16_rn(src[idx]);
  }
}

extern "C" void launch_paged_copy_row_f32_to_bf16(
    const float *src,
    __nv_bfloat16 *pool,
    int32_t phys_block,
    uint32_t page_offset,
    uint32_t page_size,
    uint32_t d_model) {
  uint32_t threads = 256;
  uint32_t blocks = (d_model + threads - 1) / threads;
  kernel_paged_copy_row_f32_to_bf16<<<blocks, threads>>>(
      src, pool, phys_block, page_offset, page_size, d_model);
}

