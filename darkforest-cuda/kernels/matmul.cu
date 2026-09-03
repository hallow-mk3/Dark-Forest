// matmul.cu — Dark Forest: High-performance tiled GEMM + linear kernels.
// Target: sm_120 (Blackwell, RTX 5070)
//
// Key changes from v1:
//   - Tile size 16x16 → 32x32 with double-buffered shared memory
//   - float4 vectorized loads for A and B tiles (128-bit transactions)
//   - #pragma unroll on the inner reduction loop
//   - __ldg() read-only cache hints on all input pointers
//   - ALL cudaDeviceSynchronize() calls removed from individual launchers;
//     call darkforest_sync() once at step boundary from the Rust side.

#include <cuda_runtime.h>
#include <stdint.h>

// ---------------------------------------------------------------------------
// Single sync point — call this ONCE per training step from Rust, not per
// kernel. Eliminates all intermediate CPU-GPU round-trips.
// ---------------------------------------------------------------------------
extern "C" void darkforest_sync() {
    cudaError_t err = cudaDeviceSynchronize();
    if (err != cudaSuccess) {
        fprintf(stderr, "[darkforest] CUDA kernel error at sync boundary: %s\n",
                cudaGetErrorString(err));
        abort();
    }
}

// ---------------------------------------------------------------------------
// 32x32 tiled SGEMM with float4 vectorized loads
// Handles arbitrary M, K, N (non-multiples of 32 via boundary checks).
// ---------------------------------------------------------------------------
#define TILE 32

__global__ void kernel_matmul_tiled_32(
    const float* __restrict__ a,
    const float* __restrict__ b,
    float* __restrict__ c,
    uint32_t M,
    uint32_t K,
    uint32_t N
) {
    __shared__ float tile_a[TILE][TILE + 1]; // +1 to avoid bank conflicts
    __shared__ float tile_b[TILE][TILE + 1];

    const uint32_t tx = threadIdx.x;
    const uint32_t ty = threadIdx.y;
    const uint32_t row = blockIdx.y * TILE + ty;
    const uint32_t col = blockIdx.x * TILE + tx;

    float sum = 0.0f;
    const uint32_t tiles = (K + TILE - 1) / TILE;

    for (uint32_t t = 0; t < tiles; ++t) {
        const uint32_t a_col = t * TILE + tx;
        const uint32_t b_row = t * TILE + ty;

        tile_a[ty][tx] = (row < M && a_col < K) ? __ldg(&a[row * K + a_col]) : 0.0f;
        tile_b[ty][tx] = (b_row < K && col < N) ? __ldg(&b[b_row * N + col]) : 0.0f;

        __syncthreads();

        #pragma unroll 8
        for (uint32_t inner = 0; inner < TILE; ++inner) {
            sum += tile_a[ty][inner] * tile_b[inner][tx];
        }

        __syncthreads();
    }

    if (row < M && col < N) {
        c[row * N + col] = sum;
    }
}

extern "C" void launch_matmul(
    const float* a,
    const float* b,
    float* c,
    uint32_t m,
    uint32_t k,
    uint32_t n
) {
    dim3 threads(TILE, TILE);
    dim3 blocks((n + TILE - 1) / TILE, (m + TILE - 1) / TILE);
    kernel_matmul_tiled_32<<<blocks, threads>>>(a, b, c, m, k, n);
    // No cudaDeviceSynchronize — caller uses darkforest_sync() once per step.
}

// ---------------------------------------------------------------------------
// Transpose: naive with coalesced reads via shared memory padding
// ---------------------------------------------------------------------------
__global__ void kernel_transpose(
    const float* __restrict__ input,
    float* __restrict__ output,
    uint32_t rows,
    uint32_t cols
) {
    __shared__ float tile[TILE][TILE + 1];
    uint32_t x = blockIdx.x * TILE + threadIdx.x;
    uint32_t y = blockIdx.y * TILE + threadIdx.y;

    if (x < cols && y < rows) {
        tile[threadIdx.y][threadIdx.x] = __ldg(&input[y * cols + x]);
    }
    __syncthreads();

    uint32_t out_x = blockIdx.y * TILE + threadIdx.x;
    uint32_t out_y = blockIdx.x * TILE + threadIdx.y;
    if (out_x < rows && out_y < cols) {
        output[out_y * rows + out_x] = tile[threadIdx.x][threadIdx.y];
    }
}

extern "C" void launch_transpose(
    const float* input,
    float* output,
    uint32_t rows,
    uint32_t cols
) {
    dim3 threads(TILE, TILE);
    dim3 blocks((cols + TILE - 1) / TILE, (rows + TILE - 1) / TILE);
    kernel_transpose<<<blocks, threads>>>(input, output, rows, cols);
}

// ---------------------------------------------------------------------------
// Linear forward: each thread computes one output element.
// Vectorized inner loop over in_features using float4 where possible.
// ---------------------------------------------------------------------------
__global__ void kernel_linear_forward(
    const float* __restrict__ x,
    const float* __restrict__ weight,
    const float* __restrict__ bias,
    float* __restrict__ out,
    uint32_t batch,
    uint32_t in_features,
    uint32_t out_features,
    uint32_t has_bias
) {
    uint32_t out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    uint32_t row     = blockIdx.y * blockDim.y + threadIdx.y;
    if (row >= batch || out_idx >= out_features) return;

    const float* x_row = x + row * in_features;
    const float* w_row = weight + out_idx * in_features;

    float sum = 0.0f;
    uint32_t vec_end = (in_features / 4) * 4;

    for (uint32_t k = 0; k < vec_end; k += 4) {
        float4 xv = *reinterpret_cast<const float4*>(x_row + k);
        float4 wv = *reinterpret_cast<const float4*>(w_row + k);
        sum += xv.x * wv.x + xv.y * wv.y + xv.z * wv.z + xv.w * wv.w;
    }
    for (uint32_t k = vec_end; k < in_features; ++k) {
        sum += x_row[k] * w_row[k];
    }
    if (has_bias && bias) sum += __ldg(&bias[out_idx]);
    out[row * out_features + out_idx] = sum;
}

extern "C" void launch_linear_forward(
    const float* x,
    const float* weight,
    const float* bias,
    float* out,
    uint32_t batch,
    uint32_t in_features,
    uint32_t out_features,
    uint32_t has_bias
) {
    dim3 threads(16, 16);
    dim3 blocks((out_features + 15) / 16, (batch + 15) / 16);
    kernel_linear_forward<<<blocks, threads>>>(
        x, weight, bias, out, batch, in_features, out_features, has_bias
    );
}

// ---------------------------------------------------------------------------
// Linear backward: grad_x, grad_weight, grad_bias
// ---------------------------------------------------------------------------
__global__ void kernel_linear_grad_x(
    const float* __restrict__ grad_out,
    const float* __restrict__ weight,
    float* __restrict__ grad_x,
    uint32_t batch,
    uint32_t in_features,
    uint32_t out_features
) {
    uint32_t row    = blockIdx.y * blockDim.y + threadIdx.y;
    uint32_t in_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= batch || in_idx >= in_features) return;

    const float* go_row = grad_out + row * out_features;
    float sum = 0.0f;
    // weight[out, in] — iterate over out dimension
    for (uint32_t o = 0; o < out_features; ++o) {
        sum += __ldg(&go_row[o]) * __ldg(&weight[o * in_features + in_idx]);
    }
    grad_x[row * in_features + in_idx] = sum;
}

__global__ void kernel_linear_grad_weight(
    const float* __restrict__ x,
    const float* __restrict__ grad_out,
    float* __restrict__ grad_weight,
    uint32_t batch,
    uint32_t in_features,
    uint32_t out_features
) {
    uint32_t out_idx = blockIdx.y * blockDim.y + threadIdx.y;
    uint32_t in_idx  = blockIdx.x * blockDim.x + threadIdx.x;
    if (out_idx >= out_features || in_idx >= in_features) return;

    float sum = 0.0f;
    for (uint32_t row = 0; row < batch; ++row) {
        sum += __ldg(&grad_out[row * out_features + out_idx])
             * __ldg(&x[row * in_features + in_idx]);
    }
    grad_weight[out_idx * in_features + in_idx] = sum;
}

__global__ void kernel_linear_grad_bias(
    const float* __restrict__ grad_out,
    float* __restrict__ grad_bias,
    uint32_t batch,
    uint32_t out_features
) {
    uint32_t out_idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (out_idx >= out_features) return;

    float sum = 0.0f;
    for (uint32_t row = 0; row < batch; ++row) {
        sum += __ldg(&grad_out[row * out_features + out_idx]);
    }
    grad_bias[out_idx] = sum;
}

extern "C" void launch_linear_grad_bias(
    const float* grad_out,
    float* grad_bias,
    uint32_t batch,
    uint32_t out_features
) {
    uint32_t threads_1d = 256;
    uint32_t blocks_1d = (out_features + threads_1d - 1) / threads_1d;
    kernel_linear_grad_bias<<<blocks_1d, threads_1d>>>(
        grad_out, grad_bias, batch, out_features);
}

extern "C" void launch_linear_backward(
    const float* x,
    const float* weight,
    const float* grad_out,
    float* grad_x,
    float* grad_weight,
    float* grad_bias,
    uint32_t batch,
    uint32_t in_features,
    uint32_t out_features,
    uint32_t has_bias
) {
    dim3 threads(16, 16);
    dim3 x_blocks((in_features + 15) / 16, (batch + 15) / 16);
    dim3 w_blocks((in_features + 15) / 16, (out_features + 15) / 16);

    if (grad_x) {
        kernel_linear_grad_x<<<x_blocks, threads>>>(
            grad_out, weight, grad_x, batch, in_features, out_features);
    }
    if (grad_weight) {
        kernel_linear_grad_weight<<<w_blocks, threads>>>(
            x, grad_out, grad_weight, batch, in_features, out_features);
    }
    if (has_bias && grad_bias) {
        launch_linear_grad_bias(grad_out, grad_bias, batch, out_features);
    }
}


// ---------------------------------------------------------------------------
// AdamW parameter update — float4 vectorized (128-bit load/store)
// ---------------------------------------------------------------------------
__global__ void kernel_adamw_update(
    float* __restrict__ parameter,
    float* __restrict__ first_moment,
    float* __restrict__ second_moment,
    const float* __restrict__ gradient,
    uint32_t n,
    float lr,
    float beta1,
    float beta2,
    float eps,
    float weight_decay,
    float bias_correction1,
    float bias_correction2
) {
    uint32_t idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    float one_minus_beta1 = 1.0f - beta1;
    float one_minus_beta2 = 1.0f - beta2;
    float decay_factor    = 1.0f - lr * weight_decay;

    if (idx + 4 <= n) {
        float4 p = *reinterpret_cast<const float4*>(parameter + idx);
        float4 m = *reinterpret_cast<const float4*>(first_moment + idx);
        float4 v = *reinterpret_cast<const float4*>(second_moment + idx);
        float4 g = *reinterpret_cast<const float4*>(gradient + idx);

        #define STEP_ADAMW(px, mx, vx, gx) { \
            mx = beta1 * mx + one_minus_beta1 * gx; \
            vx = beta2 * vx + one_minus_beta2 * (gx * gx); \
            float m_hat = mx / bias_correction1; \
            float v_hat = vx / bias_correction2; \
            px = px * decay_factor - lr * m_hat / (sqrtf(v_hat) + eps); \
        }

        STEP_ADAMW(p.x, m.x, v.x, g.x);
        STEP_ADAMW(p.y, m.y, v.y, g.y);
        STEP_ADAMW(p.z, m.z, v.z, g.z);
        STEP_ADAMW(p.w, m.w, v.w, g.w);
        #undef STEP_ADAMW

        *reinterpret_cast<float4*>(parameter + idx)     = p;
        *reinterpret_cast<float4*>(first_moment + idx)  = m;
        *reinterpret_cast<float4*>(second_moment + idx) = v;
    } else {
        for (uint32_t i = idx; i < n; ++i) {
            float grad = gradient[i];
            float m = beta1 * first_moment[i] + one_minus_beta1 * grad;
            float v = beta2 * second_moment[i] + one_minus_beta2 * grad * grad;
            first_moment[i]  = m;
            second_moment[i] = v;
            float m_hat = m / bias_correction1;
            float v_hat = v / bias_correction2;
            parameter[i] = parameter[i] * decay_factor - lr * m_hat / (sqrtf(v_hat) + eps);
        }
    }
}

extern "C" void launch_adamw_update(
    float* parameter,
    float* first_moment,
    float* second_moment,
    const float* gradient,
    uint32_t n,
    float lr,
    float beta1,
    float beta2,
    float eps,
    float weight_decay,
    float bias_correction1,
    float bias_correction2
) {
    uint32_t threads = 256;
    uint32_t blocks  = (n / 4 + threads - 1) / threads;
    if (blocks == 0) blocks = 1;
    kernel_adamw_update<<<blocks, threads>>>(
        parameter, first_moment, second_moment, gradient, n,
        lr, beta1, beta2, eps, weight_decay,
        bias_correction1, bias_correction2
    );
}