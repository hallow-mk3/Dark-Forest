# Long-Context Sequence Length Derivation and Hardware Limits

This document details the derivation, theoretical memory requirements, and empirical measurements associated with the 2,097,152-token sequence claim previously referenced in the Dark Forest repository.

---

## 1. Origin of the 2,097,152-Token Claim

The sequence length $S = 2{,}097{,}152$ originates in the kernel-isolation benchmark located at `darkforest-core/src/bin/bench_sequence_limit.rs`.

In that benchmark, sequence lengths are swept in powers of two up to $S = 2{,}097{,}152$ ($2^{21}$). The benchmark was designed to probe the limits of the fused online-softmax attention kernel implementation.

### Experimental Setup of `bench_sequence_limit.rs`

- **Scope**: Isolated attention forward pass only.
- **Tensors Allocated**: Exactly four tensors: $Q$, $K$, $V$, and $\text{Output}$.
- **Data Type**: Float32 (4 bytes per element).
- **Dimensions**: Batch size $B = 1$, heads $H = 12$, head dimension $d_{\text{head}} = 64$, model dimension $d_{\text{model}} = 768$.
- **Excluded Components**:
  - Model weights, biases, and LayerNorm parameters.
  - Backward-pass activation buffers and gradients.
  - Optimizer state (e.g. AdamW first and second moments).
  - Logits tensor and cross-entropy loss computation.
  - Key-Value cache ($K/V$ tensors across 12 layers).

---

## 2. Theoretical Memory Arithmetic

### A. Isolated Attention Kernel (Working Set for $Q, K, V, \text{Out}$)

Each tensor has shape $[S, d_{\text{model}}]$ with element size $\text{sizeof}(\text{float32}) = 4\text{ bytes}$:

$$\text{Bytes per tensor} = S \times d_{\text{model}} \times 4 = S \times 768 \times 4 = 3{,}072 \times S\text{ bytes}$$

$$\text{Working set for 4 tensors} = 4 \times 3{,}072 \times S = 12{,}288 \times S\text{ bytes}$$

Evaluating across sequence lengths:
- **$S = 8{,}192$**: $12{,}288 \times 8{,}192 = 100{,}663{,}296\text{ B} = 96.0\text{ MB}$
- **$S = 16{,}384$**: $12{,}288 \times 16{,}384 = 201{,}326{,}592\text{ B} = 192.0\text{ MB}$
- **$S = 131{,}072$**: $12{,}288 \times 131{,}072 = 1{,}610{,}612{,}736\text{ B} = 1{,}536.0\text{ MB}$
- **$S = 262{,}144$**: $12{,}288 \times 262{,}144 = 3{,}221{,}225{,}472\text{ B} = 3{,}072.0\text{ MB}$
- **$S = 524{,}288$**: $12{,}288 \times 524{,}288 = 6{,}442{,}450{,}944\text{ B} = 6{,}144.0\text{ MB}$
- **$S = 1{,}048{,}576$**: $12{,}288 \times 1{,}048{,}576 = 12{,}884{,}901{,}888\text{ B} = 12{,}288.0\text{ MB}$ ($\approx 12.29\text{ GB}$)
- **$S = 2{,}097{,}152$**: $12{,}288 \times 2{,}097{,}152 = 25{,}769{,}803{,}776\text{ B} = 24{,}576.0\text{ MB}$ ($\approx 24.58\text{ GB}$)

**Finding**: At $S = 2{,}097{,}152$, the four isolated tensors require **24,576 MB (~24.6 GB)** of contiguous device memory in Float32. This exceeds the total physical capacity of an 8 GB GPU (8,123 MB) by a factor of $3.02\times$.

### B. Full GPT-2 Training Step at $S = 2{,}097{,}152$

For a full 12-layer GPT-2 model (124M-163M parameters) in Float32:
- Parameters (weights + biases): $\approx 650\text{ MB}$
- AdamW optimizer moments ($m, v$): $\approx 1{,}300\text{ MB}$
- Parameter gradients: $\approx 650\text{ MB}$
- Logits tensor ($[S, \text{vocab\_size}]$ with vocab 50,257):
  $$2{,}097{,}152 \times 50{,}257 \times 4\text{ bytes} \approx 421.6\text{ GB}$$
- Cross-entropy gradient tensor ($d\text{logits}$): $\approx 421.6\text{ GB}$
- Forward activations (12 layers, uncheckpointed): $> 100\text{ GB}$

**Finding**: Full-model Float32 training at 2M tokens requires roughly **1 Terabyte** of memory. Even with gradient checkpointing and chunked cross-entropy in BF16, activation storage exceeds 8 GB by more than an order of magnitude.

### C. Autoregressive Inference KV-Cache at $S = 2{,}097{,}152$

In standard autoregressive decoding with a pre-allocated Key-Value cache across $L = 12$ layers:

$$\text{KV-Cache bytes} = 2 \times L \times S \times d_{\text{model}} \times \text{bytes\_per\_element}$$

- **Float32 (4 bytes/element)**:
  $$2 \times 12 \times 2{,}097{,}152 \times 768 \times 4 = 154{,}618{,}822{,}656\text{ bytes} \approx 144.0\text{ GB}$$
- **BF16 / FP16 (2 bytes/element)**:
  $$144.0\text{ GB} / 2 = 72.0\text{ GB}$$
- **4-bit Quantized Cache (0.5 bytes/element)**:
  $$144.0\text{ GB} / 8 = 18.0\text{ GB}$$

**Finding**: Holding a full uncompressed KV-cache for 2,097,152 tokens requires **144 GB** in FP32, **72 GB** in BF16, and **18 GB** in 4-bit quantization. None of these fit in 8 GB GDDR7 without hierarchical paging, sliding-window attention, or holographic context compaction (`darkforest-core/src/engine/holographic.rs`).

---

## 3. Empirical Hardware Measurement

The benchmark binary `darkforest-core/src/bin/bench_sequence_limit.rs` was executed on the target hardware:
- **Device**: NVIDIA GeForce RTX 5070 Laptop GPU (8,123.44 MB total VRAM).
- **Available Free VRAM at Launch**: 7,013.00 MB.

### Execution Log and Measurements

| Sequence Length ($S$) | Measured Latency | Measured VRAM Allocated | Status |
| :--- | :---: | :---: | :--- |
| **8,192** | 190.72 ms | 96.00 MB | ✅ Kernel executed |
| **16,384** | 756.93 ms | 192.00 MB | ✅ Kernel executed |
| **32,768** | 3,782.57 ms | 384.00 MB | ✅ Kernel executed |
| **65,536** | 23,711.05 ms | 768.00 MB | ✅ Kernel executed |
| **131,072** | 108,989.41 ms | 1,536.00 MB | ✅ Kernel executed |
| **262,144** | Not timed | 3,072.00 MB | Allocation verified only |
| **524,288** | Not timed | 6,144.00 MB | Allocation verified only |
| **1,048,576** | Not timed | 7,013.00 MB | Allocation capped at free capacity |
| **2,097,152** | Not timed | 7,013.00 MB | Allocation capped at free capacity |

### Observations:
1. **Execution Cutoff**: The attention kernel is explicitly timed only up to $S = 131{,}072$ (`run_kernel = s <= 131_072`).
2. **Allocation at $S \ge 1{,}048{,}576$**: The reported allocation of 7,013.00 MB at $S = 1{,}048{,}576$ and $S = 2{,}097{,}152$ is identical because it reflects the maximum free device memory available to the CUDA allocator before further allocation requests are rejected.
3. The kernel **was never launched or executed** at $S = 2{,}097{,}152$.

---

## 4. Remediation and Reporting Protocol

In accordance with strict technical rigor:
1. **Comparison Tables**: The 2,097,152-token figure must **not** appear in any comparative table alongside full-model training baselines (such as PyTorch SDPA). The two measure completely different scopes: an allocation test of four tensors vs a full model training step with 12 layers, activations, gradients, and optimizer state.
2. **Kernel Horizon Documentation**: The verified upper boundary for timed execution of the fused online-softmax attention kernel on an 8 GB RTX 5070 Laptop GPU is **$S = 131{,}072$** (latency: 108.99 s, working-set VRAM: 1,536 MB).
3. **Context Length Claims**: Any reference to $S = 2{,}097{,}152$ must state that it represents theoretical $O(S)$ scaling arithmetic that exceeds 8 GB device capacity ($24.6\text{ GB}$ required) and was never executed on hardware.

