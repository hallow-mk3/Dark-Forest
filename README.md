# Dark Forest — High-Performance Rust & CUDA ML Runtime

[![GitHub License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Rust](https://img.shields.io/badge/rust-1.80%2B-orange.svg)](https://www.rust-lang.org/)
[![CUDA](https://img.shields.io/badge/CUDA-12.8%20%2F%2013.3-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![Target](https://img.shields.io/badge/Target-Blackwell%20(sm__120)%20%26%20Ampere%2BAda-purple.svg)](https://www.nvidia.com/)

**Dark Forest** is a lightweight, high-performance, Rust-first machine learning runtime and autograd engine designed for deterministic device residency, ultra-low latency transformer inference/training, and embedded native AI deployments without heavy framework dependencies.

---

## ⚡ Verified Empirical Benchmarks: Dark Forest vs. PyTorch

Benchmarks measured on **NVIDIA GeForce RTX 5070 Laptop GPU (sm_120 Blackwell, 8 GB GDDR7, 85% VRAM Cap)**:

### 1. Full GPT-2 Scale Step Latency & Execution Jitter ($n=7$ Independent Trials)
*Configuration: 12 Layers, $d_{\text{model}}=768$, 12 Heads, $d_{\text{ff}}=3072$, Vocab 50,257, Context 128, Batch 1*

| Metric / Trial | PyTorch 2.9 (Eager) | Dark Forest (`train_static`) | Verified Advantage |
| :--- | :--- | :--- | :--- |
| **Trial Range** | `63.27 – 75.11 ms` | `39.77 – 42.42 ms` | **Zero distributional overlap** |
| **Sample Mean ($\mu$)** | `70.626 ms` | `41.434 ms` | **1.70× faster** |
| **Median ($\tilde{x}$)** | `73.599 ms` | `41.674 ms` | **1.77× faster** |
| **Std Deviation ($\sigma$)** | `4.898 ms` | `0.850 ms` | **5.76× tighter variance (jitter-free)** |
| **Peak Workspace VRAM** | ~1.42 GB (Dynamic) | **~0.48 GB (Static pre-allocated)** | **~3× less memory** |
| **Deployment Size** | >1.8 GB (`torch` stack) | **<12 MB (Standalone binary)** | **161× smaller footprint** |

### 2. Warp-Parallel Fused Attention Scaling & Memory Reduction ($n=4$ Sweeps)
*Configuration: Batch Size = 2, Heads = 12, Head Dimension = 64, Precision = Float32*

| Sequence Length ($S$) | Naive Attention Latency | Fused Attention Latency | Speedup | Naive Peak VRAM | Fused Peak VRAM | Peak VRAM Ratio |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **256** | `0.213 ms` | `0.104 ms` | **1.94×** | `27.88 MB` | `15.62 MB` | **1.78×** |
| **1024** | `2.766 ms` | `0.615 ms` | **4.50×** | `234.12 MB` | `38.12 MB` | **6.14×** |
| **4096** | `45.55 ms` | `7.40 ms` | **6.16×** | `3,264 MB` | `128 MB` | **25.5×** |
| **8192** | **OOM (Failed)** | **`28.59 ms`** | **Deterministic** | **OOM (>6.77 GB)** | **`248 MB`** | **Safe Execution** |

---


## 🛠️ Key Architectural Features

* **Strict Device Residency**: No silent CPU/GPU fallback copies. Cross-device operations fail cleanly at graph construction time.
* **Fused Hardware Kernels**:
  * Fused FlashAttention-style causal attention with in-warp tile reduction.
  * Warp-level `LayerNorm` and `Softmax` using `__shfl_down_sync` register reduction.
  * Vectorized 128-bit `float4` elementwise kernels for `SiLU`, `GELU`, and `AdamW` updates.
* **Complete Neural Network Suite**:
  * **Convolutions & Pooling**: `Conv1d`, `Conv2d`, `MaxPool2d`, `AvgPool2d`, `AdaptiveAvgPool2d`
  * **Normalization & Regularization**: `BatchNorm1d`, `BatchNorm2d`, `Dropout`, `Dropout2d`
  * **Recurrent Layers**: `RNNCell`, `LSTMCell`, `GRUCell`, `LSTM`, `GRU`
  * **Transformers & Composability**: `Sequential`, `Embedding`, `MultiHeadAttention`, `TransformerBlock`, `GPT2`
* **Inference Optimization**:
  * `KVCache`: Pre-allocated static Key-Value cache for multi-layer autoregressive generation.
  * `generate()`: Production sampling engine supporting Temperature Scaling, Top-K, and Nucleus (Top-P) sampling.
* **Memory Optimization & Fine-Tuning**:
  * **LoRA & QLoRA (Low-Rank Adaptation)**: `LoRALinear` and 4-bit NormalFloat `QLoRALinear` with per-block quantization for low-footprint fine-tuning.
  * **4-bit NormalFloat (NF4) Quantization**: `quantize_nf4_cpu` and vectorized CUDA unpack kernels for theoretical optimal $N(0, 1)$ quantization.
  * **Gradient Checkpointing**: `checkpoint()` activation recomputation to trade compute for memory on long context sequences.
  * **ZeRO-Offload AdamW**: `OffloadedAdamW` offloading 1st & 2nd optimizer moment states to system RAM to conserve GPU VRAM.
* **Optimizers & Training**:
  * In-place `AdamW` and `OffloadedAdamW` with decoupled weight decay and global norm gradient clipping.
  * Schedulers: `StepLR`, `CosineAnnealingLR`, `ExponentialLR`.
  * Data pipelines: `Dataset`, `TensorDataset`, `DataLoader` with batch shuffling and `drop_last`.

---

## 🚀 Quickstart

### 1. Build and Run Tests
```bash
cargo test --workspace
```

### 2. Train GPT-2 with Custom Static Engine
```bash
cargo run --release --bin train_static --features cuda
```

### 3. Autoregressive Text Generation Example
```rust
use darkforest_core::engine::generate::{generate, SamplerConfig};
use darkforest_core::nn::{GPT2Config, GPT2};

fn main() -> anyhow::Result<()> {
    let cfg = GPT2Config::tiny();
    let model = GPT2::new(cfg);
    let prompt = vec![1, 14, 52];
    
    let sampler = SamplerConfig {
        temperature: 0.8,
        top_k: 40,
        top_p: 0.95,
    };
    
    let tokens = generate(&model, &prompt, 50, &sampler)?;
    println!("Generated tokens: {:?}", tokens);
    Ok(())
}
```

---

## 📜 License
Released under the [MIT License](LICENSE).