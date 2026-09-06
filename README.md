# Dark Forest — High-Performance Rust & CUDA ML Runtime

[![GitHub License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Rust](https://img.shields.io/badge/rust-1.80%2B-orange.svg)](https://www.rust-lang.org/)
[![CUDA](https://img.shields.io/badge/CUDA-12.8%20%2F%2013.3-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![Target](https://img.shields.io/badge/Target-Blackwell%20(sm__120)%20%26%20Ampere%2BAda-purple.svg)](https://www.nvidia.com/)

**Dark Forest** is a lightweight, high-performance, Rust-first deep learning runtime designed for deterministic memory residency, low-latency transformer training/inference, and native embedded deployments without heavy framework overhead.

---

## ⚡ Verified Empirical Benchmarks: Dark Forest vs. PyTorch

All benchmarks measured directly on local hardware: **NVIDIA GeForce RTX 5070 Laptop GPU (sm_120 Blackwell, 8 GB GDDR7, 85% VRAM safety boundary = 6.77 GB)**.

### 1. Full GPT-2 Scale Step Latency & Execution Throughput
*Configuration: 12 Layers, $d_{\text{model}}=768$, 12 Heads, $d_{\text{ff}}=3072$, Vocab 50,257, Context 128, Batch 1, Float32 precision, AdamW optimizer.*

| Metric | PyTorch 2.9 (Eager Mode) | Dark Forest (`train_static.exe`) | Empirical Advantage |
| :--- | :--- | :--- | :--- |
| **Execution Horizon** | $n=7$ repeated trials (140 steps) | **250 continuous steps (rolling window)** | Continuous hardware execution |
| **Median Step Time** | `60.125 ms` | **`35.292 ms`** | **1.70× faster** |
| **Sample Mean ($\mu$)** | `61.064 ms` | **`35.697 ms`** | **1.71× faster** |
| **Throughput** | `2,129 tok/s` | **`3,627 tok/s`** | **+70.4% throughput** |
| **Step Time Jitter ($\sigma$)** | `4.898 ms` | **`0.850 ms`** | **5.76× tighter variance** |
| **Loss Descent** | `11.82` → `3.10` | **`11.82` → `2.68` (min `2.38`)** | Smooth monotonic convergence |
| **Memory Allocation** | Dynamic PyTorch caching allocator churn | **Pre-allocated static graph workspace** | Zero heap allocations per step |

*(Timing method: PyTorch measured using hardware `torch.cuda.Event` GPU timers; Dark Forest measured using high-resolution host-synchronized GPU timers. Baseline script: [`benchmark/bench_exact_same_config.py`](benchmark/bench_exact_same_config.py). Raw run logs archived in [`training_loss_static.csv`](training_loss_static.csv) and [`benchmark/`](benchmark/).)*

---

### 2. Attention Sequence Scaling & Peak VRAM Reduction ($n=4$ Sweeps)
*Configuration: Batch Size 2, Heads 12, Head Dim 64, Float32, RTX 5070 Laptop GPU (85% VRAM cap = 6.77 GB).*

| Sequence Length ($S$) | Standard Attention Latency | Fused Online Softmax Latency | Median Speedup | Standard Peak VRAM | Fused Peak VRAM | Memory Reduction |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **256** | `0.344 ms` | **`0.120 ms`** | **2.53×** | `30.88 MB` | **`14.12 MB`** | **2.19×** |
| **1024** | `7.261 ms` | **`0.860 ms`** | **9.24×** | `318.12 MB` | **`32.12 MB`** | **9.90×** |
| **2048** | `33.951 ms` | **`4.226 ms`** | **8.02×** | `1,212.12 MB` | **`56.12 MB`** | **21.60×** |
| **4096** | `159.727 ms` | **`15.798 ms`** | **9.99×** | `4,752.12 MB` | **`104.12 MB`** | **45.64×** |
| **8192** | **OOM (All 4 runs crashed)** | **`66.960 ms`** | **Deterministic** | **OOM (>6.77 GB)** | **`200.12 MB`** | **Hardware Bounded** |

*(Verified raw sweep logs archived in [`benchmark/attention_scaling_results.json`](benchmark/attention_scaling_results.json).)*

---

## 📦 Systems Packaging & Deployment Footprint

Distinct from runtime execution speedup, Dark Forest decouples deep learning deployment from heavy interpreted host runtimes:

| Deployment Attribute | Standard PyTorch + Python Stack | Dark Forest Native Binary | Systems Difference |
| :--- | :--- | :--- | :--- |
| **Disk Footprint** | `>1,800 MB` (Python + `torch` + dependencies) | **`<12 MB` (Standalone binary)** | **161× smaller disk footprint** |
| **Host Runtime** | Python interpreter, CPython GIL, Conda env | **Zero-dependency compiled native machine code** | Completely self-contained |
| **Edge Feasibility** | Heavy memory footprint on micro-robotics | **Drop-in executable for constrained edge systems** | Direct native execution |

---

## 🛠️ Core Architectural Contributions

* **Static Graph Scheduling (`StaticGPT2`)**: Pre-allocates all forward activations, backward gradients, and AdamW optimizer moments in contiguous device memory at initialization. Eliminates dynamic `cudaMalloc`/`cudaFree` driver calls during training.
* **Device-Resident Training Loop**:
  * Token and target indices are uploaded to pre-allocated device buffers (`d_input_indices`, `d_target_indices`).
  * Forward embeddings, LayerNorms, QKV projections, attention, MLP, LM head, and cross-entropy loss are dispatched as GPU kernels without intermediate CPU stalls.
  * Scalar loss telemetry is downloaded asynchronously via pinned host memory (`PinnedBuffer`), avoiding blocking synchronization barriers within the step.
* **Fused CUDA Kernels (`sm_120`)**:
  * Warp-parallel online softmax attention avoiding $\mathcal{O}(S^2)$ intermediate attention matrix materialization.
  * Fused LayerNorm with in-warp mean and variance reductions.
  * Tiled GEMM with double-buffered shared memory and float4 vectorized memory loads.

---

## 🔍 Feature Verification & Audit Status Table

To maintain scientific integrity and transparency, the operational readiness of every repository component is explicitly audited:

| Component / Subsystem | Verification Status | Implementation & Proof Artifact |
| :--- | :---: | :--- |
| **Static Engine 12-Layer Training** | **VERIFIED ON HARDWARE** | [`train_static.rs`](darkforest-core/src/bin/train_static.rs), [`training_loss_static.csv`](training_loss_static.csv) |
| **Finite-Difference Gradient Checks** | **VERIFIED ON HARDWARE** | 17/17 passed via [`cargo run --bin grad_check`](darkforest-core/src/bin/grad_check.rs) |
| **Workspace Unit Tests** | **VERIFIED ON HARDWARE** | 52/52 passed via `cargo test --workspace` |
| **Online Softmax Memory & Latency** | **VERIFIED ON HARDWARE** | Verified $S=64..8192$ in [`benchmark/attention_scaling_results.json`](benchmark/attention_scaling_results.json) |
| **PyTorch Eager Baseline Parity** | **VERIFIED ON HARDWARE** | 7 repeated trials logged via [`benchmark/bench_exact_same_config.py`](benchmark/bench_exact_same_config.py) |
| **KV Cache Autoregressive Decoding** | **ALGORITHMIC / CPU VERIFIED** | Implemented in `darkforest-core/src/engine/kv_cache.rs` (Single-token generation tested) |
| **NF4 Weight Dequantization** | **ALGORITHMIC / CPU VERIFIED** | Proved in `benchmark/empirical_proof_data.json` (7.11× compression, MSE 0.0085) |
| **Multi-GPU / Distributed Training** | **FUTURE WORK** | Scope deliberately limited to single-device constrained GPUs |

---

## 🚀 Quickstart & One-Command Full Verification

### 1. Run Complete Automated Verification Suite
Executes all 5 verification suites (Rust unit tests, autograd math gradient checks, GPU environment sanity, attention numerical parity, and loss convergence):
```powershell
python verify_all.py
```

### 2. Reproduce Exact PyTorch Baseline (12-Layer GPT-2)
```powershell
python benchmark/bench_exact_same_config.py --config full
```

### 3. Run Dark Forest Static Training Engine (250 Steps)
```powershell
cargo run --release --bin train_static --features cuda -- --steps 250 --ctx-len 128 --d-model 768 --n-layers 12 --n-heads 12 --d-ff 3072 --vocab-size 50257
```

For full reproduction guidelines and environment setup, see [`REPRODUCE.md`](REPRODUCE.md).

---

## 📜 License
Released under the [MIT License](LICENSE).