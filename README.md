# Dark Forest: A Rust and CUDA Machine Learning Runtime

Dark Forest is a machine learning runtime written in Rust and CUDA. It implements pre-allocated static execution graphs and custom fused CUDA kernels to provide deterministic execution, low latency jitter, and minimal binary deployment footprints on consumer hardware.

---

## Architecture Overview

1. **Pre-Allocated Static Execution (`StaticGPT2`)**:
   Allocates model parameters, forward activation buffers, backward gradients, and AdamW optimizer moments in contiguous GPU memory at initialization. Eliminates per-step runtime `cudaMalloc` and `cudaFree` calls.

2. **Device-Resident Training Pipeline**:
   Token indices are staged in device buffers, with the forward pass, cross-entropy loss, and backward pass executed on-device without CPU synchronization stalls. Step loss values are read asynchronously via pinned host memory.

3. **Fused Attention Kernels**:
   Implements a warp-parallel online-softmax attention forward kernel that evaluates scaled dot-product attention without materializing intermediate quadratic $O(S^2)$ attention matrices.

---

## Empirical Benchmarks

All benchmarks were evaluated on local hardware: **NVIDIA GeForce RTX 5070 Laptop GPU** (8,123.44 MB total GDDR7 VRAM, Blackwell sm_120 architecture). A strict 85% per-process memory limit (6.74 GB / 6,904.9 MB) is enforced across tests to guarantee system stability.

### 1. Like-for-Like End-to-End Model Training Step

*Configuration: 12 Layers, $d_{\text{model}}=768$, 12 Heads, $d_{\text{ff}}=3072$, Vocabulary 50,257, Batch Size 1, AdamW Optimizer ($lr=10^{-4}, \beta_1=0.9, \beta_2=0.999, \epsilon=10^{-8}, \text{weight\_decay}=0.01$).*

*Scope: Full training step (Forward pass + Cross-Entropy loss + Backward pass + AdamW optimizer update).*

| Sequence Length ($S$) | PyTorch SDPA<br>(bf16 + ckpt)<br>[1] | PyTorch SDPA<br>(bf16 + ckpt + chunked CE)<br>[2] | Dark Forest Static Engine<br>(fp32 + TF32 MMA)<br>[3] | DF Peak VRAM |
| :---: | :---: | :---: | :---: | :---: |
| **128** | `54.8 ms` / `2,099 MB` | `55.9 ms` / `2,100 MB` | `38.84 ms` | `3,320 MB` |
| **256** | `55.2 ms` / `2,109 MB` | `53.9 ms` / `2,112 MB` | — | — |
| **512** | `56.1 ms` / `2,134 MB` | `60.3 ms` / `2,139 MB` | `112.62 ms` | `4,458 MB` |
| **1,024** | `75.9 ms` / `2,179 MB` | `78.2 ms` / `2,185 MB` | `320.37 ms` | `6,862 MB` |
| **2,048** | `125.7 ms` / `2,481 MB` | `122.4 ms` / `2,196 MB` | **OOM (>6.74 GB, 85% safety cap)** | `7,013 MB` |
| **4,096** | `279.1 ms` / `3,706 MB` | `255.9 ms` / `2,836 MB` | **OOM** | — |
| **8,192** | `698.6 ms` / `6,151 MB` | `668.3 ms` / `4,117 MB` | **OOM** | — |
| **16,384** | **OOM (>6.74 GB, 85% safety cap)** | `1968.2 ms` / `6,678 MB` | **OOM** | — |
| **32,768** | **OOM** | **OOM (>6.74 GB, 85% safety cap)** | **OOM** | — |

#### Measurement and Configuration Notes:
- **[1] PyTorch SDPA (bf16 + ckpt)**: Uses `torch.nn.functional.scaled_dot_product_attention`, bfloat16 mixed precision, and gradient checkpointing. It materializes full logits for unchunked cross-entropy. At $S = 16{,}384$, logit allocation exceeds the 85% memory cap (6.74 GB).
- **[2] PyTorch SDPA (+ chunked CE)**: Chunks the LM head projection and cross-entropy loss calculation (chunk size 1,024) to avoid materializing the $[S, 50257]$ logits tensor. Extends survival horizon to $S = 16{,}384$ before exceeding the 85% cap at $S = 32{,}768$.
- **[3] Dark Forest Static Engine**: Operates in Float32 with hardware TF32 Tensor Core MMA enabled. It preallocates all layer activation buffers without gradient checkpointing, storing unchunked $[S, 50257]$ logits, probabilities, and gradient buffers. Memory scales statically with sequence length, reaching the 85% safety cap at $S = 2{,}048$.
- **Parameter Count Accounting**:
  - Dark Forest parameter counts: 162.35M ($S=128$), 162.64M ($S=512$), 163.04M ($S=1024$), 163.82M ($S=2048$). Dark Forest allocates position embeddings sized exactly to the active context window.
  - PyTorch benchmark parameter count: 212.50M ($212{,}500{,}224$ parameters). The PyTorch benchmark initialized learned position embeddings for `max_pos = 65,537` ($50.3\text{M}$ parameters) to support the full scaling sweep without reinitialization.

---

### 2. Attention Kernel Horizon Benchmark (Kernel Isolation)

*Scope: Evaluates the fused online-softmax attention forward kernel in isolation. Allocates only $Q, K, V$, and $\text{Output}$ tensors ($4 \times S \times d_{\text{model}} \times \text{sizeof(float32)}$). Does not include model parameters, optimizer state, backward passes, or full-model activations.*

*Configuration: $B = 1, H = 12, d_{\text{head}} = 64, d_{\text{model}} = 768$, Causal Masking, Float32.*

| Sequence Length ($S$) | Measured Latency (Median) | Working-Set VRAM | Measured Compute Throughput | Execution Status |
| :---: | :---: | :---: | :---: | :--- |
| **8,192** | `190.7 ms` | `96 MB` | 0.5 TFLOP/s | Kernel executed |
| **16,384** | `756.9 ms` | `192 MB` | 0.5 TFLOP/s | Kernel executed |
| **32,768** | `3,782.6 ms` | `384 MB` | 0.4 TFLOP/s | Kernel executed |
| **65,536** | `23,711.1 ms` | `768 MB` | 0.3 TFLOP/s | Kernel executed |
| **131,072** | `108,989.4 ms` | `1,536 MB` | 0.2 TFLOP/s | Kernel executed |
| **262,144** | *Not timed* | `3,072 MB` | — | Allocation verified only |
| **524,288** | *Not timed* | `6,144 MB` | — | Allocation verified only |
| **1,048,576** | *Not timed* | `7,013 MB` | — | Capped at free device VRAM |
| **2,097,152** | *Not timed* | `7,013 MB` | — | Exceeds 8 GB capacity ($24.6\text{ GB}$ required) |

*For derivation of theoretical working-set sizes, KV-cache requirements at 2M tokens, and OOM boundaries, refer to [docs/LONG_CONTEXT_DERIVATION.md](file:///c:/Users/swast/Entropy/Dark-Forest/docs/LONG_CONTEXT_DERIVATION.md).*

---

## Systems Packaging and Footprint

Application binary footprints are compared like-for-like (excluding shared external GPU driver and CUDA toolkit runtime libraries):

| Deployment Attribute | Standard PyTorch + Python Stack | Dark Forest Native Binary | Technical Distinction |
| :--- | :---: | :---: | :--- |
| **Binary / Package Footprint** | `>1,800 MB` (CPU wheel + Python runtime) | **`<12 MB`** (standalone stripped binary) | Self-contained compiled binary |
| **Runtime Dependencies** | Python 3 interpreter, CPython runtime, DLLs | **None** (system C/CUDA runtime only) | Direct machine code execution |
| **Cold-Start Startup Time** | Several seconds (interpreter startup + imports) | **<100 ms** (native binary load) | No module parsing or JIT warmup |
| **CPU-Only Deployment** | Requires Python virtual environment and torch wheel | Single native executable | Runs on headless embedded/robotics hardware |

---

## Reproducibility

Every metric reported in this repository is reproducible on hardware using committed benchmark scripts and verifiable JSON result files.

### Hardware and Software Environment

- **GPU**: NVIDIA GeForce RTX 5070 Laptop GPU (8,123.44 MB Total GDDR7, sm_120 Blackwell)
- **Host OS**: Microsoft Windows 11
- **Driver Version**: NVIDIA Display Driver 617.42
- **CUDA Toolkit**: 13.4 (NVCC 13.4.59)
- **PyTorch Version**: 2.11.0+cu128
- **Rust Toolchain**: rustc / cargo 1.99.1
- **Python**: 3.12.10
- **Measurement Date**: November 12–13, 2026

### Reproduction Commands

1. **Run PyTorch SDPA Baseline Sweeps**:
   ```powershell
   python benchmark/bench_sdpa_baseline.py
   ```
   *Output saved to:* `benchmark/sdpa_baseline_results.json`

2. **Run Dark Forest End-to-End Model Training Benchmark**:
   ```powershell
   python benchmark/bench_darkforest_e2e.py
   ```
   *Output saved to:* `benchmark/darkforest_e2e_results.json`

3. **Run Attention Kernel Scaling Benchmark (Kernel Isolation)**:
   ```powershell
   cargo run --release --bin bench_sequence_limit --features cuda
   ```
   *Output saved to:* `benchmark/kernel_scaling_results.json`

4. **Run Full Benchmark Suite**:
   ```powershell
   powershell -ExecutionPolicy Bypass -File benchmark/run_all.ps1
   ```

5. **Verify README Numbers Against Committed JSON Data**:
   ```powershell
   python scripts/test_readme_tables.py
   ```

---

## License

This project is released under the [MIT License](file:///c:/Users/swast/Entropy/Dark-Forest/LICENSE).
