// softmax.cu — Dark Forest: row-wise online softmax + cross-entropy kernels
// Target: sm_120 (Blackwell, RTX 5070)
//
// v3 changes:
//   - Fused kernel_fused_logit_ce: computes softmax + CE loss + gradient in ONE pass per row.
//     Avoids writing & reading back the full [seq, vocab] probability matrix.
//   - Softmax block size increased to 512 for wide vocab.

#include <cuda_runtime.h>
#include <float.h>
#include <stdint.h>

#define WARP_SIZE 32

__device__ __forceinline__ float warp_reduce_max(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
        val = fmaxf(val, __shfl_down_sync(0xFFFFFFFF, val, offset));
    return val;
}

__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    return val;
}

// ---------------------------------------------------------------------------
// Softmax — one block per row, warp-level reductions
// ---------------------------------------------------------------------------
__global__ void kernel_softmax(float* __restrict__ x, uint32_t rows, uint32_t cols) {
    extern __shared__ float shared[];

    uint32_t row = blockIdx.x;
    if (row >= rows) return;

    float* row_ptr   = x + (size_t)row * cols;
    uint32_t n_warps = (blockDim.x + WARP_SIZE - 1) / WARP_SIZE;

    // Phase 1: row max
    float thread_max = -FLT_MAX;
    for (uint32_t i = threadIdx.x; i < cols; i += blockDim.x)
        thread_max = fmaxf(thread_max, row_ptr[i]);

    float warp_max = warp_reduce_max(thread_max);
    if (threadIdx.x % WARP_SIZE == 0) shared[threadIdx.x / WARP_SIZE] = warp_max;
    __syncthreads();

    if (threadIdx.x < WARP_SIZE) {
        float v = (threadIdx.x < n_warps) ? shared[threadIdx.x] : -FLT_MAX;
        v = warp_reduce_max(v);
        if (threadIdx.x == 0) shared[0] = v;
    }
    __syncthreads();
    float row_max = shared[0];

    // Phase 2: exp + partial sum
    float thread_sum = 0.0f;
    for (uint32_t i = threadIdx.x; i < cols; i += blockDim.x) {
        float val = __expf(row_ptr[i] - row_max);
        row_ptr[i] = val;
        thread_sum += val;
    }

    float warp_sum = warp_reduce_sum(thread_sum);
    if (threadIdx.x % WARP_SIZE == 0) shared[threadIdx.x / WARP_SIZE] = warp_sum;
    __syncthreads();

    if (threadIdx.x < WARP_SIZE) {
        float v = (threadIdx.x < n_warps) ? shared[threadIdx.x] : 0.0f;
        v = warp_reduce_sum(v);
        if (threadIdx.x == 0) shared[0] = v;
    }
    __syncthreads();
    float inv_sum = 1.0f / shared[0];

    for (uint32_t i = threadIdx.x; i < cols; i += blockDim.x)
        row_ptr[i] *= inv_sum;
}

extern "C" void launch_softmax(float* x, uint32_t rows, uint32_t cols) {
    uint32_t threads = max(32u, min(cols, 512u));
    uint32_t n_warps = (threads + WARP_SIZE - 1) / WARP_SIZE;
    uint32_t shmem   = n_warps * sizeof(float);
    kernel_softmax<<<rows, threads, shmem>>>(x, rows, cols);
}

// ---------------------------------------------------------------------------
// Softmax backward — block-wide multi-warp parallel dot product per row
// ---------------------------------------------------------------------------
__global__ void kernel_softmax_backward(
    const float* __restrict__ probabilities,
    const float* __restrict__ grad_output,
    float* __restrict__ grad_input,
    uint32_t cols
) {
    extern __shared__ float s_warp_dots[];
    uint32_t row = blockIdx.x;
    const float* prob_row = probabilities + (size_t)row * cols;
    const float* go_row   = grad_output   + (size_t)row * cols;
    uint32_t n_warps = (blockDim.x + WARP_SIZE - 1) / WARP_SIZE;

    float thread_dot = 0.0f;
    for (uint32_t col = threadIdx.x; col < cols; col += blockDim.x)
        thread_dot += prob_row[col] * go_row[col];

    float warp_dot = warp_reduce_sum(thread_dot);
    if (threadIdx.x % WARP_SIZE == 0) {
        s_warp_dots[threadIdx.x / WARP_SIZE] = warp_dot;
    }
    __syncthreads();

    if (threadIdx.x < WARP_SIZE) {
        float v = (threadIdx.x < n_warps) ? s_warp_dots[threadIdx.x] : 0.0f;
        v = warp_reduce_sum(v);
        if (threadIdx.x == 0) s_warp_dots[0] = v;
    }
    __syncthreads();
    float dot = s_warp_dots[0];

    for (uint32_t col = threadIdx.x; col < cols; col += blockDim.x)
        grad_input[(size_t)row * cols + col] = prob_row[col] * (go_row[col] - dot);
}

extern "C" void launch_softmax_backward(
    const float* probabilities,
    const float* grad_output,
    float* grad_input,
    uint32_t rows,
    uint32_t cols
) {
    uint32_t threads = max(32u, min(cols, 512u));
    uint32_t n_warps = (threads + WARP_SIZE - 1) / WARP_SIZE;
    uint32_t shmem   = n_warps * sizeof(float);
    kernel_softmax_backward<<<rows, threads, shmem>>>(probabilities, grad_output, grad_input, cols);
}

// ---------------------------------------------------------------------------
// Cross-entropy forward — parallel reduction over rows
// ---------------------------------------------------------------------------
__global__ void kernel_cross_entropy_forward(
    const float* __restrict__ probabilities,
    const uint32_t* __restrict__ targets,
    float* __restrict__ loss,
    uint32_t rows,
    uint32_t cols
) {
    uint32_t row = blockIdx.x * blockDim.x + threadIdx.x;
    float local_loss = 0.0f;
    if (row < rows) {
        float prob = fmaxf(probabilities[(size_t)row * cols + targets[row]], 1e-12f);
        local_loss = -__logf(prob);
    }
    float warp_loss = warp_reduce_sum(local_loss);
    if (threadIdx.x % WARP_SIZE == 0)
        atomicAdd(loss, warp_loss);
}

__global__ void kernel_ce_divide(float* loss, float inv_rows) {
    if (threadIdx.x == 0 && blockIdx.x == 0) *loss *= inv_rows;
}

extern "C" void launch_cross_entropy_forward(
    const float* probabilities,
    const uint32_t* targets,
    float* loss,
    uint32_t rows,
    uint32_t cols
) {
    cudaMemsetAsync(loss, 0, sizeof(float));
    uint32_t threads = 256;
    uint32_t blocks  = (rows + threads - 1) / threads;
    kernel_cross_entropy_forward<<<blocks, threads>>>(probabilities, targets, loss, rows, cols);
    kernel_ce_divide<<<1, 1>>>(loss, 1.0f / (float)rows);
}

// ---------------------------------------------------------------------------
// Cross-entropy backward — parallel flat indexing
// ---------------------------------------------------------------------------
__global__ void kernel_cross_entropy_backward(
    const float* __restrict__ probabilities,
    const uint32_t* __restrict__ targets,
    const float* __restrict__ grad_output,
    float* __restrict__ grad_logits,
    uint32_t rows,
    uint32_t cols
) {
    uint32_t index = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t total = rows * cols;
    if (index >= total) return;
    uint32_t row = index / cols;
    uint32_t col = index % cols;
    grad_logits[index] = *grad_output
        * (probabilities[index] - (col == targets[row] ? 1.0f : 0.0f)) / rows;
}

extern "C" void launch_cross_entropy_backward(
    const float* probabilities,
    const uint32_t* targets,
    const float* grad_output,
    float* grad_logits,
    uint32_t rows,
    uint32_t cols
) {
    uint32_t total   = rows * cols;
    uint32_t threads = 256;
    uint32_t blocks  = (total + threads - 1) / threads;
    kernel_cross_entropy_backward<<<blocks, threads>>>(
        probabilities, targets, grad_output, grad_logits, rows, cols);
}

// ---------------------------------------------------------------------------
// FUSED: Logit → Softmax + Cross-entropy forward + gradient backward in ONE pass
//
// Inputs:
//   logits [rows, cols]       — overwritten with softmax probs
//   targets [rows]            — token class indices
// Outputs:
//   grad_logits [rows, cols]  — (prob - one_hot) / rows
//   loss (scalar, atomicAdd)  — mean cross-entropy
//
// Each block handles exactly one row. Shared mem: n_warps floats.
// This replaces: launch_softmax + launch_cross_entropy_forward +
//                launch_cross_entropy_backward (3 kernel launches, 3 global passes)
// ---------------------------------------------------------------------------
__global__ void kernel_fused_logit_ce(
    float* __restrict__ logits,
    const uint32_t* __restrict__ targets,
    float* __restrict__ grad_logits,
    float* __restrict__ loss,
    uint32_t rows,
    uint32_t cols
) {
    extern __shared__ float smem[];

    uint32_t row = blockIdx.x;
    if (row >= rows) return;

    float* row_ptr  = logits      + (size_t)row * cols;
    float* grad_ptr = grad_logits + (size_t)row * cols;
    uint32_t target  = targets[row];
    uint32_t n_warps = (blockDim.x + WARP_SIZE - 1) / WARP_SIZE;
    float inv_rows   = 1.0f / (float)rows;

    // Pass 1: max reduction
    float tmax = -FLT_MAX;
    for (uint32_t i = threadIdx.x; i < cols; i += blockDim.x)
        tmax = fmaxf(tmax, row_ptr[i]);
    float wmax = warp_reduce_max(tmax);
    if (threadIdx.x % WARP_SIZE == 0) smem[threadIdx.x / WARP_SIZE] = wmax;
    __syncthreads();
    if (threadIdx.x < n_warps) {
        float v = smem[threadIdx.x];
        v = warp_reduce_max(v);
        if (threadIdx.x == 0) smem[0] = v;
    }
    __syncthreads();
    float row_max = smem[0];

    // Pass 2: exp + sum reduction
    float tsum = 0.0f;
    for (uint32_t i = threadIdx.x; i < cols; i += blockDim.x) {
        float e = __expf(row_ptr[i] - row_max);
        row_ptr[i] = e;
        tsum += e;
    }
    float wsum = warp_reduce_sum(tsum);
    if (threadIdx.x % WARP_SIZE == 0) smem[threadIdx.x / WARP_SIZE] = wsum;
    __syncthreads();
    if (threadIdx.x < n_warps) {
        float v = smem[threadIdx.x];
        v = warp_reduce_sum(v);
        if (threadIdx.x == 0) smem[0] = v;
    }
    __syncthreads();
    float inv_sum = 1.0f / smem[0];

    // Pass 3: normalize -> write probs, write grad, accumulate loss
    float local_loss = 0.0f;
    for (uint32_t i = threadIdx.x; i < cols; i += blockDim.x) {
        float p = row_ptr[i] * inv_sum;
        row_ptr[i]  = p;
        float one_hot = (i == target) ? 1.0f : 0.0f;
        grad_ptr[i] = (p - one_hot) * inv_rows;
        if (i == target) local_loss = -__logf(fmaxf(p, 1e-12f)) * inv_rows;
    }
    float wloss = warp_reduce_sum(local_loss);
    if (threadIdx.x % WARP_SIZE == 0) smem[threadIdx.x / WARP_SIZE] = wloss;
    __syncthreads();
    if (threadIdx.x < n_warps) {
        float v = smem[threadIdx.x];
        v = warp_reduce_sum(v);
        if (threadIdx.x == 0) atomicAdd(loss, v);
    }
}

extern "C" void launch_fused_logit_ce(
    float* logits,
    const uint32_t* targets,
    float* grad_logits,
    float* loss,
    uint32_t rows,
    uint32_t cols
) {
    cudaMemsetAsync(loss, 0, sizeof(float));
    uint32_t threads = max(32u, min(cols, 512u));
    uint32_t n_warps = (threads + WARP_SIZE - 1) / WARP_SIZE;
    uint32_t shmem   = n_warps * sizeof(float);
    kernel_fused_logit_ce<<<rows, threads, shmem>>>(logits, targets, grad_logits, loss, rows, cols);
}
