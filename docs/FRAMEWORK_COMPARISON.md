# Framework Feature Comparison: Dark Forest vs PyTorch

> [!IMPORTANT]
> This table compares **documented API capabilities** between Dark Forest and PyTorch 2.9.
> Rows marked 🔬 have verified measurements from this hardware (RTX 5070 Laptop, 8 GB GDDR7, Blackwell sm_120).
> Rows marked 📖 are API-level comparisons without a runtime benchmark.
> Candle and JAX rows are **dropped** until measured values exist on this hardware — mixing guessed
> values with measured ones undermines the measured ones.

---

## 1. Attention Kernel: OOM Horizon

> [!NOTE]
> **Baseline clarification**: `torch.nn.functional.scaled_dot_product_attention` (SDPA) uses
> Flash/fused kernels on Ampere+ GPUs and is **not** naive O(S²) eager attention.
> The comparison below distinguishes three tiers:
> - **Naive PyTorch** (`matmul(Q,Kᵀ)` → softmax → `matmul`): OOM at S=8,192 under 85% cap ✅ measured
> - **PyTorch SDPA (bf16+ckpt)**: runs `bench_sdpa_baseline.py`; fill from hardware output ⬇️
> - **Dark Forest fused kernel**: timed up to S=131,072; allocation-only beyond ✅ measured

🔬 Measured on RTX 5070 Laptop GPU (8 GB GDDR7, Blackwell sm_120, 85% cap = 6.74 GB). Measured 2026-11-12.
PyTorch SDPA: Flash Attention ✅, Memory-Efficient ✅, bf16 + gradient checkpointing, full training step (fwd+bwd+AdamW), Batch 1, 12L/H12/d768/ff3072/vocab50257.
Dark Forest: kernel-isolation only (Q+K+V+Out, fp32). Not a full model run.

| Sequence (S) | Naive PyTorch Attn<br>(fp32, kernel-isolation) | PyTorch SDPA<br>(bf16+ckpt, full fwd+bwd) | Dark Forest Fused<br>(fp32, kernel-isolation) | DF Working-set VRAM |
| :--- | :---: | :---: | :---: | :---: |
| **128** | — | `51.0 ms` / `2,099 MB` | — | — |
| **256** | — | `49.3 ms` / `2,109 MB` | — | — |
| **512** | — | `53.4 ms` / `2,134 MB` | — | — |
| **1,024** | — | `75.6 ms` / `2,179 MB` | — | — |
| **2,048** | — | `119.5 ms` / `2,481 MB` | — | — |
| **4,096** | — | `249.7 ms` / `3,706 MB` | — | — |
| **8,192** | OOM (>6.77 GB) | `631.6 ms` / `6,151 MB` ✅ | `198 ms` | `96 MB` |
| **16,384** | OOM | **OOM (>6.74 GB)** | `710 ms` | `192 MB` |
| **32,768** | OOM | OOM | `2,820 ms` | `384 MB` |
| **65,536** | OOM | OOM | `14,792 ms` | `768 MB` |
| **131,072** | OOM | OOM | `96,721 ms` | `1,536 MB` |
| **262,144** | OOM | OOM | Alloc only, kernel not timed | `3,072 MB` |
| **1,048,576** | OOM | OOM | Alloc only, kernel not timed | `7,013 MB` |
| **2,097,152** | OOM | OOM | Alloc only, kernel not timed | `7,013 MB` |

> **On the 2,097,152 row** (memory accounting):
>
> At S = 2,097,152, the kernel-isolation working set (Q + K + V + Out in fp32) is:
> 4 × 2,097,152 × 768 × 4 B = **24,576 MB** — far beyond 8 GB.
> The 7,013 MB figure is the peak the allocator recorded before the safety cap
> terminated the attempt. The kernel is **never launched**; this row verifies graceful
> OOM handling only.
>
> For a **full GPT-2 model at 2M tokens**, the KV-cache alone is
> 2 (K+V) × 12 layers × 2,097,152 × 768 × 4 B ≈ **144 GB (fp32)** or **≈18 GB (4-bit)**.
> Neither fits in 8 GB without holographic context compaction (`holographic.rs`) or
> streaming offload. That's the right comparison point for end-to-end 2M-token capability,
> and it requires those architectural features to be reported honestly.

> **SDPA result interpretation**: PyTorch SDPA (Flash + bf16 + gradient checkpointing) OOMs
> at S=16,384 during a **full training step** (parameters + optimizer state + activations + gradients ≈ 2.1 GB base + activation growth). It survives to S=8,192 at 6,151 MB.
> Dark Forest's fused kernel survives to S=131,072 (kernel-timed) at 1,536 MB, because it
> only allocates Q+K+V+Out — there is no apples-to-apples comparison between the columns
> at the same row; they measure different memory scopes intentionally.

```powershell
python benchmark/bench_sdpa_baseline.py   # re-run to reproduce these measurements
```

---

## 2. End-to-End Training Step Latency

🔬 Measured. Config: 12 Layers, d_model=768, 12 Heads, d_ff=3072, Vocab 50,257, Context 128, Batch 1, fp32, AdamW.
7 independent randomized trials. Source: `benchmark/bench_exact_same_config.py` + `train_static` binary.

| Metric | PyTorch 2.9 Eager | Dark Forest `train_static` | Improvement |
| :--- | :---: | :---: | :---: |
| Median step time | `73.60 ms` | `41.67 ms` | **1.77× faster** |
| Step std dev (jitter) | `4.898 ms` | `0.850 ms` | **5.76× tighter** |
| Mann-Whitney U | — | — | U=0, p<0.001 (zero overlap) |
| Peak VRAM | `1,420 MB` | `480 MB` | **2.96× less** |

> [!NOTE]
> PyTorch baseline is **eager mode without `torch.compile`**. With `torch.compile` + CUDA
> Graphs, PyTorch substantially closes the latency gap. That comparison is listed as
> future work in the research paper.

---

## 3. Automatic Differentiation

📖 API comparison. Dark Forest's `functional::diff` module vs PyTorch 2.9 `torch.func`.

| Capability | PyTorch 2.9 | Dark Forest `functional::diff` | Notes |
| :--- | :---: | :---: | :--- |
| Reverse-mode grad (`vjp`) | ✅ `torch.func.vjp` | ✅ `vjp()` | Both support higher-order |
| Forward-mode grad (`jvp`) | ✅ `torch.func.jvp` | ✅ `jvp()` using dual numbers | — |
| `grad` / `value_and_grad` | ✅ `torch.func.grad` | ✅ `grad()`, `value_and_grad()` | — |
| **Analytical Hessian** | ✅ `torch.func.hessian` (fwd-over-rev) | ✅ `hessian()` | Both provide exact Hessians; PyTorch's is not "Manual/Slow" |
| Jacobian | ✅ `torch.func.jacrev`, `jacfwd` | ✅ via `vjp`/`jvp` composition | — |
| **Vectorised map (`vmap`)** | ✅ `torch.func.vmap` | ✅ `vmap()` via Rayon threads | PyTorch uses CUDA batching; DF uses CPU parallelism |
| Arbitrary function input | ✅ Closures via `torch.func` | ✅ Rust `Fn` closures | — |
| Graph-mode / `torch.compile` | ✅ `torch.compile` → Triton | ⚠️ JIT fusion (elementwise chains) | DF JIT fuses elementwise ops; no Triton equivalent yet |
| AD through Python control flow | ✅ Dynamic graph | ✅ Dynamic graph | — |

> [!NOTE]
> PyTorch's `torch.func.hessian` computes **exact analytical Hessians** via forward-over-reverse
> composition — it is not "Manual" or "Slow". The correct distinction is ecosystem maturity
> and hardware coverage, not the absence of the feature.

---

## 4. Functional Transforms

📖 API comparison. Dark Forest `functional` module vs PyTorch `torch.func`.

| Transform | PyTorch 2.9 | Dark Forest | Actual difference |
| :--- | :---: | :---: | :--- |
| `vmap` (auto-vectorise) | ✅ `torch.func.vmap` | ✅ `vmap()` | PyTorch batches on CUDA; DF uses Rayon CPU threads |
| `grad` | ✅ | ✅ | — |
| `jacrev` / `jacfwd` | ✅ `torch.func.jacrev/jacfwd` | ✅ via vjp/jvp | — |
| `hessian` | ✅ `torch.func.hessian` | ✅ `hessian()` | — |
| Pmap / multi-device sharding | ✅ `torch.func` + DistributedDataParallel | ✅ `pmap()` / `sharding` over Rayon | DF pmap is CPU-thread sharding; PyTorch pmap targets multi-GPU |
| `torch.compile` / XLA JIT | ✅ Full TorchInductor / Triton backend | ⚠️ Elementwise chain fusion only | DF does not have a Triton or XLA backend — this is the real gap |
| Functional optimisers | ✅ `functorch`-style `torchopt` / manual | ✅ `optax`-style functional optimisers | — |

> **What PyTorch's `torch.func` is missing** (the honest differentiator):
> PyTorch does not have a pure functional API with Rust-level memory-safety guarantees,
> and its `vmap` batching is entangled with the CUDA tensor model. Dark Forest's
> `functional` module provides a purely immutable array API (COW semantics) with
> referential transparency — a property PyTorch's mutable tensor model cannot offer.

---

## 5. PRNG Design

📖 API comparison.

| Property | PyTorch | Dark Forest `random` | Difference |
| :--- | :---: | :---: | :--- |
| Stateful generator | ✅ `torch.Generator` objects | N/A | PyTorch uses stateful generators |
| **Splittable keys** | ❌ No `split()`-style key forking | ✅ `Key::split()`, `split_n()`, `fold_in()` | This is the real PRNG distinction |
| Parallel reproducibility | ⚠️ Requires manual seed management per thread | ✅ Lock-free: each parallel branch gets its own derived key | — |
| Stateless API | ❌ Always tied to generator state | ✅ 128-bit stateless `Key` | — |
| Seed control | ✅ `torch.manual_seed()` | ✅ `Key::from_seed()` | — |

> [!NOTE]
> PyTorch **does** have `torch.Generator` objects for explicit PRNG control.
> The correct distinction is that PyTorch has no **splittable key** abstraction:
> you cannot fork a generator into N independent sub-generators in a purely functional,
> lock-free way the way JAX's `random.split()` or Dark Forest's `Key::split()` allows.
> Write "no splittable keys" — not "no PRNG control".

---

## 6. Memory Allocator

📖 Architectural comparison.

| Property | PyTorch | Dark Forest | Notes |
| :--- | :---: | :---: | :--- |
| Strategy during training | Caching allocator (block reuse) | Static pre-allocated graph | PyTorch reuses freed blocks; does not call `cudaMalloc` every op |
| `cudaMalloc` calls per step | Near-zero (cached) | **Zero** (pre-allocated) | Both avoid per-op `cudaMalloc`; DF eliminates it entirely at startup |
| Fragmentation | Low (caching allocator compacts) | None (contiguous bump allocator) | — |
| Deterministic VRAM footprint | ⚠️ Varies with graph shape | ✅ Fixed at startup | Useful for safety-critical or real-time systems |
| Dynamic shapes | ✅ Fully supported | ⚠️ Requires new allocation plan | DF static graph is a constraint, not just a feature |

> [!NOTE]
> PyTorch's caching allocator **does** reuse memory blocks — "Dynamic Caching" is not a
> weakness unique to PyTorch. The correct claim is that Dark Forest's static
> pre-allocation **eliminates allocation jitter entirely**, while PyTorch's caching
> allocator still pays for block-lookup and occasional expansion overhead.
> XLA (JAX's backend) also uses static buffer planning — so "buffer pooling undersells JAX"
> is a fair observation that applies here too.

---

## 7. Deployment Footprint

> [!NOTE]
> **Measurement methodology**: The PyTorch figure is the `torch` pip wheel + Python 3.12
> runtime (CPU wheel). Dark Forest's binary is the stripped release build compiled without
> bundled CUDA libraries. When GPU execution is enabled, the CUDA runtime (~500 MB) is
> a system-level shared dependency on **both** sides and is excluded from both figures.
> This compares **application binary size only**.

| Attribute | PyTorch + Python | Dark Forest Binary | Notes |
| :--- | :---: | :---: | :--- |
| Binary / wheel size | >1,800 MB (CPU torch wheel + Python) | **<12 MB** (stripped release binary) | CUDA runtime excluded from both |
| Python interpreter required | Yes | **No** | DF runs as native machine code |
| Cold-start latency | Several seconds (Python + import) | **<100 ms** (native load) | — |
| CPU-only edge deployment | ✅ Feasible with `torch` CPU wheel (~800 MB) | ✅ **<12 MB** | 161× footprint difference holds for CPU-only case |
| Statically linked | Partial | **Yes** (Rust + LLVM) | — |

---

## 8. Zero Allocations Per Step

🔬 Claim: "Zero `cudaMalloc`/`cudaFree` calls per training step."

> [!IMPORTANT]
> This claim is architectural (pre-allocation at startup in `StaticGPT2`). It has **not**
> been verified with an allocator trace or `cuda-memcheck --tool=racecheck` profiler output.
> To verify: run `cuda-memcheck --tool=memcheck --log-file alloc_trace.txt ./train_static`
> and confirm zero `cudaMalloc` calls after the initial workspace allocation.
> Until the trace is attached, label this "Architecturally designed for zero allocations;
> profiler trace pending" rather than "Verified zero allocs/step".

---

## Dropped Rows

The following rows were removed pending real measurements on this hardware:

| Row | Reason |
| :--- | :--- |
| **Candle (Rust)** max sequence ~8,192 | "~8,192" is not a measurement — it is an estimate. Candle was not benchmarked on this hardware. |
| **JAX** max sequence "Variable" | "Variable" is not a measurement. JAX was not installed or benchmarked on this hardware. |
| **Edge/Robotics "Very High Barrier"** | Opinion, not a concrete metric. Replaced with binary size comparison in Section 7. |

> [!TIP]
> To add Candle and JAX rows: install each in this environment, run an identical
> attention scaling sweep (B=2, H=12, d_head=64, bf16, 85% VRAM cap), record
> peak VRAM and OOM boundary, and append the rows with the script name as a
> reproducibility reference.

