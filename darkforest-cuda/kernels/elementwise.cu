// elementwise.cu — Dark Forest CUDA kernels: elementwise operations
// Target: sm_120 (Blackwell, RTX 5070)
//
// All cudaDeviceSynchronize() calls removed. Use darkforest_sync() once per
// step from the Rust caller.

#include <cuda_runtime.h>
#include <stdint.h>

// ---------------------------------------------------------------------------
// Fast GPU reduction sum
// ---------------------------------------------------------------------------
__device__ __forceinline__ float warp_reduce_sum_elem(float val) {
    #pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xFFFFFFFF, val, offset);
    return val;
}

__global__ void kernel_reduce_sum(const float* __restrict__ in, float* __restrict__ out, uint32_t n) {
    extern __shared__ float sdata[];
    uint32_t tid = threadIdx.x;
    uint32_t i = blockIdx.x * (blockDim.x * 2) + threadIdx.x;
    float my_sum = (i < n) ? in[i] : 0.0f;
    if (i + blockDim.x < n) my_sum += in[i + blockDim.x];
    sdata[tid] = my_sum;
    __syncthreads();

    for (uint32_t s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] = my_sum = my_sum + sdata[tid + s];
        }
        __syncthreads();
    }
    if (tid == 0) {
        atomicAdd(out, sdata[0]);
    }
}

extern "C" void launch_reduce_sum(const float* in, float* out, uint32_t n) {
    cudaMemsetAsync(out, 0, sizeof(float));
    uint32_t threads = 256;
    uint32_t blocks = (n + (threads * 2) - 1) / (threads * 2);
    if (blocks == 0) blocks = 1;
    uint32_t smem = threads * sizeof(float);
    kernel_reduce_sum<<<blocks, threads, smem>>>(in, out, n);
}
__global__ void kernel_fill(float* __restrict__ c, float val, uint32_t n) {
    uint32_t idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    if (idx + 4 <= n) {
        float4 vc;
        vc.x = val;
        vc.y = val;
        vc.z = val;
        vc.w = val;
        *reinterpret_cast<float4*>(c + idx) = vc;
    } else {
        for (uint32_t i = idx; i < n; ++i) c[i] = val;
    }
}

extern "C" void launch_fill(float* c, float val, uint32_t n) {
    uint32_t threads = 256;
    uint32_t blocks  = (n / 4 + threads - 1) / threads;
    if (blocks == 0) blocks = 1;
    kernel_fill<<<blocks, threads>>>(c, val, n);
}

// ---------------------------------------------------------------------------
// Elementwise add: c[i] = a[i] + b[i]  — float4 vectorized (128-bit loads)
// ---------------------------------------------------------------------------
__global__ void kernel_add(const float* __restrict__ a,
                            const float* __restrict__ b,
                            float* __restrict__ c,
                            uint32_t n) {
    uint32_t idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    if (idx + 4 <= n) {
        float4 va = *reinterpret_cast<const float4*>(a + idx);
        float4 vb = *reinterpret_cast<const float4*>(b + idx);
        float4 vc;
        vc.x = va.x + vb.x;
        vc.y = va.y + vb.y;
        vc.z = va.z + vb.z;
        vc.w = va.w + vb.w;
        *reinterpret_cast<float4*>(c + idx) = vc;
    } else {
        for (uint32_t i = idx; i < n; ++i) c[i] = a[i] + b[i];
    }
}

extern "C" void launch_add(const float* a, const float* b, float* c, uint32_t n) {
    uint32_t threads = 256;
    uint32_t blocks  = (n / 4 + threads - 1) / threads;
    if (blocks == 0) blocks = 1;
    kernel_add<<<blocks, threads>>>(a, b, c, n);
}

// ---------------------------------------------------------------------------
// Elementwise scale: c[i] = alpha * a[i] — float4 vectorized
// ---------------------------------------------------------------------------
__global__ void kernel_scale(const float* __restrict__ a,
                              float alpha,
                              float* __restrict__ c,
                              uint32_t n) {
    uint32_t idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    if (idx + 4 <= n) {
        float4 va = *reinterpret_cast<const float4*>(a + idx);
        float4 vc;
        vc.x = va.x * alpha;
        vc.y = va.y * alpha;
        vc.z = va.z * alpha;
        vc.w = va.w * alpha;
        *reinterpret_cast<float4*>(c + idx) = vc;
    } else {
        for (uint32_t i = idx; i < n; ++i) c[i] = a[i] * alpha;
    }
}

extern "C" void launch_scale(const float* a, float alpha, float* c, uint32_t n) {
    uint32_t threads = 256;
    uint32_t blocks  = (n / 4 + threads - 1) / threads;
    if (blocks == 0) blocks = 1;
    kernel_scale<<<blocks, threads>>>(a, alpha, c, n);
}

// ---------------------------------------------------------------------------
// Elementwise multiply: c[i] = a[i] * b[i] — float4 vectorized
// ---------------------------------------------------------------------------
__global__ void kernel_mul(const float* __restrict__ a,
                            const float* __restrict__ b,
                            float* __restrict__ c,
                            uint32_t n) {
    uint32_t idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    if (idx + 4 <= n) {
        float4 va = *reinterpret_cast<const float4*>(a + idx);
        float4 vb = *reinterpret_cast<const float4*>(b + idx);
        float4 vc;
        vc.x = va.x * vb.x;
        vc.y = va.y * vb.y;
        vc.z = va.z * vb.z;
        vc.w = va.w * vb.w;
        *reinterpret_cast<float4*>(c + idx) = vc;
    } else {
        for (uint32_t i = idx; i < n; ++i) c[i] = a[i] * b[i];
    }
}

extern "C" void launch_mul(const float* a, const float* b, float* c, uint32_t n) {
    uint32_t threads = 256;
    uint32_t blocks  = (n / 4 + threads - 1) / threads;
    if (blocks == 0) blocks = 1;
    kernel_mul<<<blocks, threads>>>(a, b, c, n);
}

__global__ void kernel_mul_backward(const float* __restrict__ grad_out,
                                    const float* __restrict__ a,
                                    const float* __restrict__ b,
                                    float* __restrict__ grad_a,
                                    float* __restrict__ grad_b,
                                    uint32_t n) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float g = grad_out[i];
        if (grad_a) grad_a[i] = g * b[i];
        if (grad_b) grad_b[i] = g * a[i];
    }
}

extern "C" void launch_mul_backward(const float* grad_out,
                                    const float* a,
                                    const float* b,
                                    float* grad_a,
                                    float* grad_b,
                                    uint32_t n) {
    uint32_t threads = 256;
    uint32_t blocks  = (n + threads - 1) / threads;
    kernel_mul_backward<<<blocks, threads>>>(grad_out, a, b, grad_a, grad_b, n);
}

// ---------------------------------------------------------------------------
// GELU: x * 0.5 * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
// ---------------------------------------------------------------------------
#define SQRT_2_OVER_PI 0.7978845608028654f
#define GELU_COEFF     0.044715f

__global__ void kernel_gelu(const float* __restrict__ x, float* __restrict__ out, uint32_t n) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float v     = x[i];
        float inner = SQRT_2_OVER_PI * (v + GELU_COEFF * v * v * v);
        out[i]      = 0.5f * v * (1.0f + tanhf(inner));
    }
}

extern "C" void launch_gelu(const float* x, float* out, uint32_t n) {
    uint32_t threads = 256;
    uint32_t blocks  = (n + threads - 1) / threads;
    kernel_gelu<<<blocks, threads>>>(x, out, n);
}

__global__ void kernel_gelu_backward(const float* __restrict__ grad_out,
                                     const float* __restrict__ x,
                                     float* __restrict__ grad_x,
                                     uint32_t n) {
    uint32_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float v       = x[i];
        float inner   = SQRT_2_OVER_PI * (v + GELU_COEFF * v * v * v);
        float tanh_v  = tanhf(inner);
        float sech2   = 1.0f - tanh_v * tanh_v;
        float d_inner = SQRT_2_OVER_PI * (1.0f + 3.0f * GELU_COEFF * v * v);
        float dy_dx   = 0.5f * (1.0f + tanh_v) + 0.5f * v * sech2 * d_inner;
        grad_x[i]     = grad_out[i] * dy_dx;
    }
}

extern "C" void launch_gelu_backward(const float* grad_out,
                                     const float* x,
                                     float* grad_x,
                                     uint32_t n) {
    uint32_t threads = 256;
    uint32_t blocks  = (n + threads - 1) / threads;
    kernel_gelu_backward<<<blocks, threads>>>(grad_out, x, grad_x, n);
}

// ---------------------------------------------------------------------------
// Add bias: out[row, col] = input[row, col] + bias[col]
// Uses float4 vectorized loads when col is aligned.
// ---------------------------------------------------------------------------
__global__ void kernel_add_bias(const float* __restrict__ input,
                                const float* __restrict__ bias,
                                float* __restrict__ out,
                                uint32_t batch,
                                uint32_t features) {
    uint32_t col = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row < batch && col < features) {
        out[row * features + col] = input[row * features + col] + __ldg(&bias[col]);
    }
}

extern "C" void launch_add_bias(const float* input,
                                const float* bias,
                                float* out,
                                uint32_t batch,
                                uint32_t features) {
    dim3 threads(32, 8);
    dim3 blocks((features + 31) / 32, (batch + 7) / 8);
    kernel_add_bias<<<blocks, threads>>>(input, bias, out, batch, features);
}

// ---------------------------------------------------------------------------
// Embedding lookup forward & backward
// ---------------------------------------------------------------------------
__global__ void kernel_embedding_forward(const uint32_t* __restrict__ indices,
                                         const float* __restrict__ weight,
                                         float* __restrict__ out,
                                         uint32_t seq_len,
                                         uint32_t embed_dim) {
    uint32_t d   = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t pos = blockIdx.y * blockDim.y + threadIdx.y;
    if (pos < seq_len && d < embed_dim) {
        uint32_t idx = indices[pos];
        out[pos * embed_dim + d] = __ldg(&weight[idx * embed_dim + d]);
    }
}

extern "C" void launch_embedding_forward(const uint32_t* indices,
                                         const float* weight,
                                         float* out,
                                         uint32_t seq_len,
                                         uint32_t embed_dim) {
    dim3 threads(32, 8);
    dim3 blocks((embed_dim + 31) / 32, (seq_len + 7) / 8);
    kernel_embedding_forward<<<blocks, threads>>>(indices, weight, out, seq_len, embed_dim);
}

__global__ void kernel_embedding_backward(const uint32_t* __restrict__ indices,
                                          const float* __restrict__ grad_out,
                                          float* __restrict__ grad_weight,
                                          uint32_t seq_len,
                                          uint32_t embed_dim) {
    uint32_t d   = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t pos = blockIdx.y * blockDim.y + threadIdx.y;
    if (pos < seq_len && d < embed_dim) {
        uint32_t idx = indices[pos];
        atomicAdd(&grad_weight[idx * embed_dim + d], grad_out[pos * embed_dim + d]);
    }
}

extern "C" void launch_embedding_backward(const uint32_t* indices,
                                          const float* grad_out,
                                          float* grad_weight,
                                          uint32_t seq_len,
                                          uint32_t embed_dim) {
    dim3 threads(32, 8);
    dim3 blocks((embed_dim + 31) / 32, (seq_len + 7) / 8);
    kernel_embedding_backward<<<blocks, threads>>>(indices, grad_out, grad_weight, seq_len, embed_dim);
}
