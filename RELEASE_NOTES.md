# Dark Forest v1.0.0 Release Notes

Dark Forest is a machine learning runtime written in Rust and native CUDA. It implements pre-allocated static execution graphs and custom fused CUDA kernels to provide deterministic execution, low latency variance, and minimal binary footprints on consumer hardware.

---

## 1. What is Dark Forest

Dark Forest provides an alternative transformer training and inference engine built directly in Rust and CUDA C++:
- **Pre-Allocated Static Graph Execution (`StaticGPT2`)**: Allocates model parameters, forward activation buffers, backward gradients, and AdamW optimizer moments in contiguous GPU memory at initialization. Eliminates runtime `cudaMalloc` and `cudaFree` driver calls during training steps.
- **Fused Online-Softmax Attention**: Evaluates scaled dot-product attention without materializing intermediate quadratic $O(S^2)$ attention matrices in device memory.
- **Autoregressive KV-Cache Decoding (`generate_fast`)**: Caches Key and Value projections for single-token autoregressive generation without recomputing full context.
- **Low-Rank Adaptation (`StaticLoRAGPT2`)**: Supports LoRA adapter training with optional 4-bit NF4 quantized base weights and exact on-device weight merging ($W \mathrel{+}= \frac{\alpha}{r}BA$).
- **Native Deployment**: Compiles to a standalone native binary (<12 MB) without requiring Python interpreters or runtime packages.

---

## 2. Hardware and Evaluation Environment

All empirical measurements reported for v1.0.0 were evaluated on:
- **GPU**: NVIDIA GeForce RTX 5070 Laptop GPU (8,123.44 MB total GDDR7 VRAM, sm_120 Blackwell architecture)
- **Host OS**: Microsoft Windows 11
- **Driver Version**: NVIDIA Display Driver 617.42
- **CUDA Toolkit**: 13.4 (NVCC 13.4.59)
- **PyTorch Version**: 2.11.0+cu128
- **Rust Toolchain**: rustc / cargo 1.99.1
- **Python**: 3.12.10
- **Safety Limit**: Strict 85% per-process memory limit (6.74 GB / 6,904.9 MB) enforced across all comparative runs.

---

## 3. What Was Measured

### End-to-End Model Training Step
Full training step (Forward pass + Cross-Entropy loss + Backward pass + AdamW optimizer update):
- **Model Configuration**: 12 Layers, $d_{\text{model}}=768$, 12 Heads, $d_{\text{ff}}=3072$, Vocabulary 50,257, Batch Size 1, AdamW ($lr=10^{-4}, \beta_1=0.9, \beta_2=0.999, \epsilon=10^{-8}, \text{weight\_decay}=0.01$).
- **Dark Forest Static Engine (Float32, TF32 MMA)**:
  - $S = 128$: 38.84 ms median step time, 3,320.0 MB peak VRAM.
  - $S = 512$: 112.62 ms median step time, 4,458.0 MB peak VRAM.
  - $S = 1{,}024$: 320.37 ms median step time, 6,861.9 MB peak VRAM.
  - $S = 2{,}048$: OOM (Exceeded 85% VRAM safety cap, 7,013.0 MB).
- **PyTorch SDPA Baseline (bf16 + Gradient Checkpointing + Unchunked CE)**:
  - $S = 128$: 54.8 ms / 2,099 MB.
  - $S = 8{,}192$: 698.6 ms / 6,151 MB.
  - $S = 16{,}384$: OOM (>6.74 GB, 85% safety cap).
- **PyTorch SDPA Baseline (bf16 + Gradient Checkpointing + Chunked CE)**:
  - $S = 16{,}384$: 1,968.2 ms / 6,678 MB.
  - $S = 32{,}768$: OOM (>6.74 GB, 85% safety cap).

### Isolated Attention Kernel Horizon
Fused online-softmax forward attention pass in isolation (allocating only $Q, K, V, \text{Output}$ tensors):
- $S = 8{,}192$: 190.7 ms, 96.0 MB working set.
- $S = 16{,}384$: 756.9 ms, 192.0 MB working set.
- $S = 32{,}768$: 3,782.6 ms, 384.0 MB working set.
- $S = 65{,}536$: 23,711.1 ms, 768.0 MB working set.
- $S = 131{,}072$: 108,989.4 ms, 1,536.0 MB working set.
- $S \ge 262{,}144$: Allocation verified; kernel execution omitted due to $O(S^2)$ latency scaling.
- $S = 2{,}097{,}152$: Exceeds physical 8 GB GPU capacity ($24.6\text{ GB}$ required for $Q, K, V, \text{Out}$ in fp32); verified graceful OOM handling.

---

## 4. Known Limitations

1. **Fixed Static Allocation Horizon**: Static graph pre-allocation requires fixing the sequence length at initialization. Dynamically changing sequence length requires re-allocating workspace buffers.
2. **Memory Footprint in Float32**: `StaticGPT2` pre-allocates all forward activations, backward gradients, and AdamW moments in Float32 without gradient checkpointing. It has a high base VRAM floor (~3.3 GB at $S=128$) and reaches the 8 GB device limit at $S=2{,}048$.
3. **No Dynamic JIT Compiler Backend**: Dark Forest does not currently feature a Triton or TorchInductor equivalent JIT compiler for arbitrary dynamic op fusions.
4. **Isolated Kernel vs Full Training Step Distinctions**: Isolated kernel scaling metrics (such as the 96 MB working set at $S=8,192$) must not be directly compared against full training step VRAM.

---

## 5. How to Reproduce

1. **Install Python bindings**:
   ```powershell
   pip install -e darkforest-py
   ```
2. **Execute PyTorch SDPA baseline sweeps**:
   ```powershell
   python benchmark/bench_sdpa_baseline.py
   ```
3. **Execute Dark Forest end-to-end model training benchmark**:
   ```powershell
   python benchmark/bench_darkforest_e2e.py
   ```
4. **Execute attention kernel scaling benchmark**:
   ```powershell
   cargo run --release --bin bench_sequence_limit --features cuda
   ```
5. **Validate all numbers in README against JSON results**:
   ```powershell
   python scripts/test_readme_tables.py
   ```
6. **Execute complete reproduction pipeline in one command**:
   ```powershell
   powershell -ExecutionPolicy Bypass -File benchmark/run_all.ps1
   ```
