// layernorm.cu — Dark Forest: LayerNorm forward and backward kernels
// Target: sm_120 (Blackwell, RTX 5070)
//
// Changes from v1:
//   - All cudaDeviceSynchronize() removed from launchers.
//   - Warp reduction now correctly handles multi-warp blocks via shared memory.
//   - Increased thread count to 256 for better occupancy on large feature dims.

#include <cuda_runtime.h>
#include <float.h>
#include <stdint.h>

#define WARP_SIZE 32
#define LN_EPS    1e-5f

__device__ __forceinline__ float warp_reduce_sum_ln(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    return val;
}

// ---------------------------------------------------------------------------
// LayerNorm forward — one block per batch row
// ---------------------------------------------------------------------------
__global__ void kernel_layernorm(
    const float* __restrict__ x,
    const float* __restrict__ gamma,
    const float* __restrict__ beta,
    float* __restrict__ out,
    float* __restrict__ means,
    float* __restrict__ rstds,
    uint32_t batch,
    uint32_t features
) {
    uint32_t b = blockIdx.x;
    if (b >= batch) return;

    extern __shared__ float smem[];   // [n_warps] for reduction

    const float* row_x  = x   + b * features;
    float*       row_out = out + b * features;
    uint32_t n_warps = (blockDim.x + WARP_SIZE - 1) / WARP_SIZE;

    // ---- Mean ----
    float tsum = 0.0f;
    for (uint32_t i = threadIdx.x; i < features; i += blockDim.x)
        tsum += row_x[i];

    float wsum = warp_reduce_sum_ln(tsum);
    if (threadIdx.x % WARP_SIZE == 0) smem[threadIdx.x / WARP_SIZE] = wsum;
    __syncthreads();

    if (threadIdx.x < WARP_SIZE) {
        float v = (threadIdx.x < n_warps) ? smem[threadIdx.x] : 0.0f;
        v = warp_reduce_sum_ln(v);
        if (threadIdx.x == 0) smem[0] = v;
    }
    __syncthreads();

    float mean = smem[0] / (float)features;
    if (threadIdx.x == 0 && means) means[b] = mean;

    // ---- Variance ----
    float tvar = 0.0f;
    for (uint32_t i = threadIdx.x; i < features; i += blockDim.x) {
        float diff = row_x[i] - mean;
        tvar += diff * diff;
    }

    float wvar = warp_reduce_sum_ln(tvar);
    if (threadIdx.x % WARP_SIZE == 0) smem[threadIdx.x / WARP_SIZE] = wvar;
    __syncthreads();

    if (threadIdx.x < WARP_SIZE) {
        float v = (threadIdx.x < n_warps) ? smem[threadIdx.x] : 0.0f;
        v = warp_reduce_sum_ln(v);
        if (threadIdx.x == 0) smem[0] = v;
    }
    __syncthreads();

    float rstd = rsqrtf(smem[0] / (float)features + LN_EPS);
    if (threadIdx.x == 0 && rstds) rstds[b] = rstd;
    if (threadIdx.x == 0) smem[0] = rstd;  // broadcast rstd
    __syncthreads();
    rstd = smem[0];

    // ---- Normalize ----
    for (uint32_t i = threadIdx.x; i < features; i += blockDim.x) {
        float x_hat = (row_x[i] - mean) * rstd;
        float g  = gamma ? __ldg(&gamma[i]) : 1.0f;
        float bt = beta  ? __ldg(&beta[i])  : 0.0f;
        row_out[i] = x_hat * g + bt;
    }
}

extern "C" void launch_layernorm(
    const float* x, const float* gamma, const float* beta,
    float* out, float* means, float* rstds,
    uint32_t batch, uint32_t features
) {
    uint32_t threads = min(features, 256u);
    if (threads < WARP_SIZE) threads = WARP_SIZE;
    uint32_t n_warps = (threads + WARP_SIZE - 1) / WARP_SIZE;
    uint32_t shmem   = n_warps * sizeof(float);
    kernel_layernorm<<<batch, threads, shmem>>>(
        x, gamma, beta, out, means, rstds, batch, features);
}

// ---------------------------------------------------------------------------
// LayerNorm backward — one block per batch row
// ---------------------------------------------------------------------------
__global__ void kernel_layernorm_backward(
    const float* __restrict__ grad_out,
    const float* __restrict__ x,
    const float* __restrict__ gamma,
    const float* __restrict__ means,
    const float* __restrict__ rstds,
    float* __restrict__ grad_x,
    float* __restrict__ grad_gamma,
    float* __restrict__ grad_beta,
    uint32_t batch,
    uint32_t features
) {
    uint32_t b = blockIdx.x;
    if (b >= batch) return;

    extern __shared__ float smem[];   // [2 * n_warps] for two reductions
    uint32_t n_warps = (blockDim.x + WARP_SIZE - 1) / WARP_SIZE;

    const float* row_x    = x        + b * features;
    const float* row_dout = grad_out  + b * features;
    float*       row_gx   = grad_x    + b * features;

    float mean = means[b];
    float rstd = rstds[b];

    float t_ds  = 0.0f;
    float t_dsx = 0.0f;

    for (uint32_t i = threadIdx.x; i < features; i += blockDim.x) {
        float x_hat = (row_x[i] - mean) * rstd;
        float dout  = row_dout[i];
        float g     = gamma ? __ldg(&gamma[i]) : 1.0f;
        float ds    = dout * g;
        t_ds  += ds;
        t_dsx += ds * x_hat;
        if (grad_gamma) atomicAdd(&grad_gamma[i], dout * x_hat);
        if (grad_beta)  atomicAdd(&grad_beta[i],  dout);
    }

    // Reduce ds
    float wds = warp_reduce_sum_ln(t_ds);
    if (threadIdx.x % WARP_SIZE == 0) smem[threadIdx.x / WARP_SIZE] = wds;
    __syncthreads();
    if (threadIdx.x < WARP_SIZE) {
        float v = (threadIdx.x < n_warps) ? smem[threadIdx.x] : 0.0f;
        v = warp_reduce_sum_ln(v);
        if (threadIdx.x == 0) smem[0] = v;
    }
    __syncthreads();

    // Reduce dsx
    float wdsx = warp_reduce_sum_ln(t_dsx);
    if (threadIdx.x % WARP_SIZE == 0) smem[n_warps + threadIdx.x / WARP_SIZE] = wdsx;
    __syncthreads();
    if (threadIdx.x < WARP_SIZE) {
        float v = (threadIdx.x < n_warps) ? smem[n_warps + threadIdx.x] : 0.0f;
        v = warp_reduce_sum_ln(v);
        if (threadIdx.x == 0) smem[1] = v;
    }
    __syncthreads();

    float sum_ds  = smem[0];
    float sum_dsx = smem[1];
    float f       = (float)features;

    for (uint32_t i = threadIdx.x; i < features; i += blockDim.x) {
        float x_hat = (row_x[i] - mean) * rstd;
        float dout  = row_dout[i];
        float g     = gamma ? __ldg(&gamma[i]) : 1.0f;
        float ds    = dout * g;
        row_gx[i]   = (rstd / f) * (f * ds - sum_ds - x_hat * sum_dsx);
    }
}

extern "C" void launch_layernorm_backward(
    const float* grad_out, const float* x, const float* gamma,
    const float* means, const float* rstds,
    float* grad_x, float* grad_gamma, float* grad_beta,
    uint32_t batch, uint32_t features
) {
    uint32_t threads = min(features, 256u);
    if (threads < WARP_SIZE) threads = WARP_SIZE;
    uint32_t n_warps = (threads + WARP_SIZE - 1) / WARP_SIZE;
    // Need 2*n_warps slots: one set for ds, one for dsx
    uint32_t shmem   = (n_warps * 2 + 2) * sizeof(float);
    kernel_layernorm_backward<<<batch, threads, shmem>>>(
        grad_out, x, gamma, means, rstds, grad_x, grad_gamma, grad_beta, batch, features);
}
