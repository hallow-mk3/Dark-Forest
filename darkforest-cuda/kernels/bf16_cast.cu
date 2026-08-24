// bf16_cast.cu — Dark Forest: F32 <-> BF16 cast kernels for mixed-precision GEMM
// Target: sm_120 (Blackwell, RTX 5070)
//
// These kernels cast FP32 tensors to BF16 before feeding into cublasGemmEx,
// and cast BF16 outputs back to FP32 for elementwise ops.
// Uses float2 / short2 vectorized loads (64-bit aligned) for max throughput.

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <stdint.h>

// ---------------------------------------------------------------------------
// F32 -> BF16 cast: vectorized 2-element per thread
// ---------------------------------------------------------------------------
__global__ void kernel_f32_to_bf16(
    const float* __restrict__ src,
    __nv_bfloat16* __restrict__ dst,
    uint32_t n
) {
    uint32_t idx = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
    if (idx + 1 < n) {
        // Load 2 floats at once (64-bit)
        float2 v = *reinterpret_cast<const float2*>(src + idx);
        __nv_bfloat162 r = __float22bfloat162_rn(v);
        *reinterpret_cast<__nv_bfloat162*>(dst + idx) = r;
    } else if (idx < n) {
        dst[idx] = __float2bfloat16_rn(src[idx]);
    }
}

extern "C" void launch_f32_to_bf16(
    const float* src,
    __nv_bfloat16* dst,
    uint32_t n
) {
    uint32_t pairs = (n + 1) / 2;
    uint32_t threads = 256;
    uint32_t blocks = (pairs + threads - 1) / threads;
    kernel_f32_to_bf16<<<blocks, threads>>>(src, dst, n);
}

// ---------------------------------------------------------------------------
// BF16 -> F32 cast: vectorized 2-element per thread
// ---------------------------------------------------------------------------
__global__ void kernel_bf16_to_f32(
    const __nv_bfloat16* __restrict__ src,
    float* __restrict__ dst,
    uint32_t n
) {
    uint32_t idx = (blockIdx.x * blockDim.x + threadIdx.x) * 2;
    if (idx + 1 < n) {
        __nv_bfloat162 v = *reinterpret_cast<const __nv_bfloat162*>(src + idx);
        float2 r = __bfloat1622float2(v);
        *reinterpret_cast<float2*>(dst + idx) = r;
    } else if (idx < n) {
        dst[idx] = __bfloat162float(src[idx]);
    }
}

extern "C" void launch_bf16_to_f32(
    const __nv_bfloat16* src,
    float* dst,
    uint32_t n
) {
    uint32_t pairs = (n + 1) / 2;
    uint32_t threads = 256;
    uint32_t blocks = (pairs + threads - 1) / threads;
    kernel_bf16_to_f32<<<blocks, threads>>>(src, dst, n);
}
