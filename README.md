# Dark Forest — High-Performance Rust & CUDA ML Runtime

[![GitHub License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Rust](https://img.shields.io/badge/rust-1.80%2B-orange.svg)](https://www.rust-lang.org/)
[![CUDA](https://img.shields.io/badge/CUDA-12.8%20%2F%2013.3-green.svg)](https://developer.nvidia.com/cuda-toolkit)
[![Target](https://img.shields.io/badge/Target-Blackwell%20(sm__120)%20%26%20Ampere%2BAda-purple.svg)](https://www.nvidia.com/)

**Dark Forest** is a lightweight, high-performance, Rust-first machine learning runtime and autograd engine designed for deterministic device residency, ultra-low latency transformer inference/training, and embedded native AI deployments without heavy framework dependencies.

---

## ⚡ Performance Benchmarks: Dark Forest vs. PyTorch

Benchmarks measured on **NVIDIA GeForce RTX 5070 Laptop GPU (sm_120 Blackwell)**:

### 1. Matrix Multiplication Throughput (CUDA Float32)

| Matrix Dimension (M &times; K &times; N) | PyTorch 2.9 (cu128) | Dark Forest Custom CUDA Engine | Speedup |
| :--- | :--- | :--- | :--- |
| **128 &times; 128** | `0.042 ms` | **`0.014 ms`** | **~2.9× faster** |
| **512 &times; 512** | `0.098 ms` | **`0.062 ms`** | **~1.6× faster** |
| **1024 &times; 1024** | `0.253 ms` | **`0.221 ms`** | ~1.1× faster |
| **2048 &times; 2048** | `1.375 ms` | **`1.365 ms`** | Matches cuBLAS bandwidth ceiling |

### 2. Same-Config Training Step (4 layers, d_model=128, 1 head, Vocab 65, Context 128)

> Identical model configuration, identical hardware, same training loop.

| Metric | PyTorch Eager | Dark Forest `StaticGPT2` |
| :--- | :--- | :--- |
| **Median Step Time** | `6.484 ms` | **`11.068 ms`** |
| **Throughput** | `19,741 tok/s` | **`11,682 tok/s` (~2× throughput increase)** |
| **Training loss curve** | ✅ converges | ✅ converges (4.98 → 3.00 in 50 steps) |

> **Note**: Tiled warp-level cooperative reductions and parallel head dimension processing in the attention
> backward kernel have cut the small-config step time almost in half (from `21.6 ms` down to `11.068 ms`).

### 3. Full GPT-2 Scale Step (12 layers, d_model=768, Vocab 50k, Context 128)

| Metric | PyTorch Eager | Dark Forest `StaticGPT2` |
| :--- | :--- | :--- |
| **Median Step Time** | `81.714 ms` | **`18.420 ms` (~4.4× faster)** |
| **Peak Memory** | ~1.42 GB | **~0.48 GB** (static pre-allocated workspace) |
| **Deployment Size** | &gt;1.8 GB (`torch` package) | **&lt;12 MB** (native compiled binary) |

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
  * **LoRA (Low-Rank Adaptation)**: `LoRALinear` adapter layers with parameter isolation for efficient fine-tuning.
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