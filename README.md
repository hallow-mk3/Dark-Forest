# Dark Forest

Dark Forest is a Rust-first ML runtime aimed at building a smaller, more explicit, and more controllable alternative to heavy-framework training stacks. It is designed around three product goals:

- predictable autograd and device residency
- tighter control over CUDA execution paths
- a simpler mental model for production training and experimentation

This project is not a claim that it is universally "better than PyTorch" across every workload. It is an honest attempt to build a more opinionated stack for the workloads we care about: fast experimentation, deterministic numeric behavior, and explicit execution boundaries.

## Product Architecture

```
┌────────────────────────────────────────────────────────┐
│  Python Benchmark Harness (benchmark/run_pytorch_*.py) │
├────────────────────────────────────────────────────────┤
│  Rust Core Autograd Engine (darkforest-core)           │
│  - Reverse-mode dynamic tape autodiff                  │
│  - Thread-safe Arc/RwLock tensor storage               │
│  - Numerical finite-difference gradient checker        │
│  - In-place AdamW optimizer with decoupled decay       │
├────────────────────────────────────────────────────────┤
│  CUDA Layer (darkforest-cuda)                          │
│  - Vectorized elementwise ops (float4)                 │
│  - Online row-wise softmax (warp reduction)            │
│  - Fused FlashAttention-2 style kernel (sm_120)        │
│  - LayerNorm with warp reduction                       │
└────────────────────────────────────────────────────────┘
```

## Milestone Verification & Test Suite

### 1. Run Gradient Checker
Numerical verification comparing analytical reverse-mode gradients against finite differences ($\delta = 10^{-4}$):
```bash
cargo run -p darkforest-core --bin grad_check
```

### 2. Run All Unit Tests
```bash
cargo test -p darkforest-core
```

### 3. Run GPT-2 Small Training Baseline
```bash
cargo run -p darkforest-core --release --bin train_gpt2_small
```

### 4. Run PyTorch Comparison Benchmark
```bash
python benchmark/run_pytorch_baseline.py
```

The current benchmark is a reproducibility baseline for the reduced 4-layer,
256-wide PyTorch model configured in the script, not a GPT-2-small (125M)
comparison. It uses a fixed seed, 10 warmup steps, and five measured trials;
the reported timing and throughput are medians, while the timing range is also
printed to expose run-to-run variance. Peak memory is reported from
`torch.cuda.max_memory_allocated()` after the warmup phase.

#### Benchmark reality check

The project currently has three different benchmark layers, and they should not be
mixed together:

1. A full-model PyTorch reference baseline for training-step timing.
2. An apples-to-apples CPU matrix benchmark that matches the same work shape for
   both PyTorch and Dark Forest.
3. The actual product milestone: a CUDA-vs-CUDA throughput measurement on the
   mixed-mode training path.

The first two are useful for understanding correctness and runtime behavior, but
neither is a product-level claim that Dark Forest is meaningfully faster than
PyTorch in general. The CPU matmul path in the core library is intentionally a
naive, correctness-first implementation; PyTorch's CPU backend is typically
linked against vendor-optimized BLAS. A CPU-only comparison therefore does not
answer the real question, which is whether the Rust + CUDA path produces stable
training throughput on real GPU workloads.

The immediate next requirement is a stable CUDA throughput run on a machine with
NVIDIA tooling present, logged across 200+ steps with rolling-average timing. Until
that result is measured and reproduced, no claim of performance superiority should
be made.

### Current Verification Status

| Area | Status |
| --- | --- |
| Core library unit tests | Verified: 14 tests pass with `cargo test -p darkforest-core --lib` |
| Toy CPU training loop | Verified: 1,000 steps complete; `training_loss.csv` is written |
| Toy loss convergence | Observed: current run decreased from 4.9598 to 3.7343 (minimum 3.2373) |
| Toy CUDA-attention training | Verified: 1,000 steps complete with CUDA attention; loss 5.0945 -> 4.2104 (minimum 3.5095) |
| Resident CUDA attention working set | Verified: model retains device buffers across steps; latest toy run loss 5.5491 -> 3.6979 (minimum 3.3793) |
| CUDA device tensor storage | Verified: persistent `DeviceTensor` host/device round-trip passes on the RTX 5070 |
| CUDA device matmul | Verified: `DeviceTensor::matmul` matches the CPU reference on the RTX 5070 |
| CUDA Linear forward/autograd | Verified: GPU matmul forward plus finite-difference checks for input, weight, and bias gradients |
| CUDA Linear toy training | Verified: 1,000 steps complete; loss 5.2529 -> 2.9742 (minimum 2.6193) |
| CUDA Linear backward reductions | Verified: CUDA dX/dW/dB kernels pass the existing finite-difference gate |
| CUDA Linear full toy training | Verified: 1,000 steps with CUDA forward/backward; loss 5.1412 -> 3.0664 (minimum 2.6468) |
| CUDA AdamW update primitive | Verified: device parameter/moment update matches the CPU AdamW formula for one step |
| Persistent CUDA AdamW integration | Verified: two-step state test and 1,000-step toy training pass under `--features cuda` |
| GPT-2-small training | Not demonstrated; the executable uses `GPT2Config::tiny()` |
| Fused attention forward correctness | Verified: CUDA test passes for causal seq_len=5, d_head=4 within 2e-4 |
| Attention backward correctness | Verified: direct unfused CUDA backward matches CPU gradients and finite differences |
| Fused attention backward | Not implemented; the verified backward path materializes an attention workspace |
| CUDA attention model integration | Verified for the toy model: attention routes CUDA forward/backward under `--features cuda` |
| Dark Forest vs PyTorch comparison | Not available; only the reduced PyTorch baseline is measured |

The CUDA crate exposes host-slice wrappers and CPU-reference correctness tests.
The test has now run successfully with CUDA Toolkit 13.3, MSVC, and the RTX 5070
Laptop GPU. The suite covers fused forward plus direct unfused backward, including
finite-difference checks for Q, K, and V. To repeat it, initialize the Visual Studio x64 developer environment,
ensure `CUDA_PATH` points to the toolkit, and run:
```bash
cargo test -p darkforest-cuda --features cuda_kernels
```
The test is gated on successful kernel compilation; a `running 0 tests` result
means `nvcc` was unavailable and does not verify the kernel.

### Device Residency Design

The device-residency work uses a global single-device rule for each training
step: tensors participating in one operation must share a device. Binary
operations now reject mixed CPU/GPU inputs instead of silently downloading a
GPU tensor to the host.

CUDA storage is persistent in `Tensor`, and matrix multiplication plus softmax
now dispatch both forward and backward computation without host round-trips.
The remaining transformer path is still mixed-mode: cross-entropy reduction and
some surrounding model operations continue to use CPU-owned values. GPU builds
also require an x64 Visual Studio developer environment so `nvcc` can find
`cl.exe`.

The core CUDA attention path is deliberately mixed-mode for this milestone:
the existing CPU tensor storage and non-attention operations remain unchanged,
while attention uploads Q/K/V for CUDA forward and backward and downloads the
results. The attention module now retains its CUDA buffers across training steps,
including Q/K/V and gradient workspaces, but CPU projections still refresh their
contents every step. This validates a first residency boundary, but the remaining
CPU/GPU transfers make it unsuitable for performance claims; the latest run was
roughly 216-258 ms per logged step.

The CUDA crate also exposes an explicit `DeviceTensor` storage primitive, and
core `Tensor` provides `to_cuda()` and `from_cuda()` under `--features cuda`.
This is the first global device-storage boundary; automatic dispatch for every
core operation and mixed-device validation are still pending, so CPU remains
the default execution mode.

The first dispatched device operation is rank-2 matmul. Linear-layer routing is
now uses that device matmul under `--features cuda`, with a custom autograd
node for input, weight, and bias gradients. The forward projection is on GPU,
and its dX/dW/dB reductions now execute on CUDA. The resulting gradients are
downloaded into the existing CPU autograd accumulator, and parameter storage,
gradient accumulation, and optimizer updates remain CPU-side; this is not yet a
fully device-resident training path or a performance benchmark.

The CUDA crate now also exposes an AdamW update primitive over device-resident
parameters, moments, and gradients. It is validated but not connected to the
CPU-owned `AdamW` optimizer yet; doing that efficiently requires persistent
device optimizer state rather than uploading and downloading every parameter on
each step.

The core `AdamW` now creates persistent CUDA optimizer state under
`--features cuda` and dispatches parameter updates to the GPU. Updated parameter
values are still downloaded to the CPU tensor store after each step so the
existing autograd and parameter APIs remain coherent; the latest integrated toy
run reached loss 4.9349 -> 2.9324 (minimum 2.7455), with logged steps around
188-200 ms. This is an integration result, not a final performance benchmark.

CPU mirror synchronization is now an explicit `AdamW::sync_parameters_to_cpu()`
boundary. It remains called after each CUDA optimizer step because non-Linear
CPU operations still read the CPU tensor store; deferring it before those ops
are device-dispatched would change model behavior.

To run the toy training wiring check from a Visual Studio x64 developer shell:
```bash
cargo run -p darkforest-core --release --features cuda --bin train_gpt2_small
```
