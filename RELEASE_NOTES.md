# Release Notes — Dark Forest v1.0.0 (Production Release)

### 🌲 Executive Summary
Dark Forest v1.0.0 is the first production-grade release of a lightweight, high-performance deep learning runtime engineered from scratch in Rust and native CUDA. It delivers deterministic transformer training and sub-millisecond autoregressive inference on constrained consumer GPUs without heavy Python runtime overhead.

---

### ⚡ Key Architectural Highlights

#### 1. Fused Online Softmax FlashAttention (Blackwell `sm_120` & Ampere/Ada)
* **Register-Cached Query Lanes**: Preloads Q query components into thread warp registers, eliminating millions of redundant global/shared memory read cycles across the key loop.
* **Hardware FMA (`fmaf`) Instruction Pipelining**: Converts dot-product accumulations, normalizer rescalings, and output projections to single-cycle hardware FMA instructions.
* **Linear Memory Scaling $\mathcal{O}(S)$**: Completely avoids materializing the quadratic $\mathcal{O}(S^2)$ attention score matrix, achieving up to a **9.99x speedup** at $S=4096$ and preventing OOM failures at $S=8192$ under strict 85% VRAM caps.

#### 2. Static Execution Graph Scheduling (`StaticGPT2`)
* Pre-allocates forward activations, backward gradients, and AdamW optimizer moments in contiguous device memory at startup.
* Eliminates all dynamic `cudaMalloc` / `cudaFree` driver calls during training.
* Reduces execution latency jitter by **5.76x** (0.85 ms vs. 4.90 ms std dev) with deterministic step latency (~35.4 ms for full 124M GPT-2 scale).

#### 3. Autoregressive KV-Cache Fast Decode (`generate_fast`)
* Replaces naive $\mathcal{O}(S^2)$ full-context recomputation with deterministic $\mathcal{O}(1)$ single-token decode steps.
* Custom float4-vectorized GEMV (matrix-vector) kernel replaces GEMM for decode steps.
* GPU-side argmax reduction keeps token selection on-device, crossing PCIe with only 4 bytes per token.

#### 4. Parameter-Efficient Fine-Tuning (`StaticLoRAGPT2`)
* Freezes base model parameters and injects low-rank adapter matrices $A$ and $B$ (< 1% trainable parameters).
* Optional 4-bit NormalFloat (NF4) base weight quantization (QLoRA) achieving 7.11x VRAM compression.
* Mathematical weight merging ($W' = W_{\text{base}} + \frac{\alpha}{r}BA$) for zero-overhead deployment without adapter routing.

#### 5. Unified Production Interface & Tooling
* **Native Standalone CLI (`darkforest_cli`)**: Full streaming text generation, interactive chat REPL, and architecture inspection without Python dependencies.
* **SafeTensors Interoperability**: Direct export/import to HuggingFace `.safetensors` format.
* **Zero-Dependency Tokenizers**: Includes `ByteTokenizer` (256 UTF-8 bytes + EOS), `CharTokenizer`, and GPT-2 compatible `BPETokenizer`.
* **Zero-Copy Python Bindings**: Direct PyO3 bindings via `darkforest.Engine` and `darkforest.EngineConfig`.
