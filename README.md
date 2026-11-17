# Dark Forest

**A GPT-2 training runtime in Rust and CUDA. One ~12 MB binary. No Python. No allocator surprises.**

Dark Forest trains a 12-layer GPT-2 end to end (forward, loss, backward, AdamW) on an 8 GB laptop GPU, using a single executable instead of a 1.8 GB Python stack. Every byte of GPU memory is claimed once at startup, so after that, nothing allocates, nothing frees, and nothing stalls.

It is not faster than PyTorch (see [the honest numbers](#the-honest-numbers)). It is a different bet: **predictable, tiny, and fully understandable.**

---

## Why this exists

| | PyTorch + Python | Dark Forest |
|---|---|---|
| Deployment size | > 1,800 MB | **< 12 MB** |
| Cold start | Several seconds | **< 100 ms** |
| Runtime dependencies | Python, CPython, DLLs | **None** (system C/CUDA runtime only) |
| GPU memory behavior | Allocator-managed, grows and fragments | **Allocated once, static for the whole run** |
| Per-step `cudaMalloc`/`cudaFree` | Yes | **Zero** |

*Footprints exclude the shared GPU driver and CUDA toolkit libraries on both sides.*

If you care about deterministic latency, shipping to headless or embedded hardware, or simply seeing what a training loop looks like with nothing hidden, this is for you.

---

## What's inside

**Static execution graph.** Parameters, activations, gradients, and AdamW moments live in one contiguous GPU allocation made at init. No per-step allocation means low jitter and no out-of-memory surprises mid-run.

**Fully device-resident training.** Token indices, forward pass, cross-entropy, and backward pass all run on the GPU with no CPU sync stalls. Loss is read back asynchronously through pinned host memory.

**Fused attention that never builds the S×S matrix.** A warp-parallel online-softmax kernel computes scaled dot-product attention without materializing the O(S²) score matrix. In isolation it runs at sequence length **131,072 using just 1.5 GB**.

**Fused chunked cross-entropy + shared backward scratch.** Together these cut peak VRAM at S=1,024 from 6,862 MB to 4,182 MB and let training reach **S=4,096** on an 8 GB card under a strict 85% memory cap.

**Verified correct.** Matches PyTorch across 5 full backward + AdamW steps. Initial eval loss is 10.81 against the theoretical ln(50,257) = 10.825.

---

## The honest numbers

Hardware: RTX 5070 Laptop GPU (8 GB, Blackwell sm_120), with a hard 85% per-process memory cap (6.74 GB). Model: 12 layers, d_model 768, 12 heads, d_ff 3072, vocab 50,257, batch size 1. A full training step is timed.

| Seq length | PyTorch SDPA (bf16 + ckpt) | Dark Forest (fp32 + TF32) | DF peak VRAM |
|---:|---:|---:|---:|
| 128 | 54.8 ms / 2,099 MB | 51.06 ms | 3,170 MB |
| 512 | 56.1 ms / 2,134 MB | 185.16 ms | 3,694 MB |
| 1,024 | 75.9 ms / 2,179 MB | 380.78 ms | 4,182 MB |
| 2,048 | 125.7 ms / 2,481 MB | 849.60 ms | 4,832 MB |
| 4,096 | 279.1 ms / 3,706 MB | 2,652.66 ms | 6,751 MB |
| 8,192 | 698.6 ms / 6,151 MB | OOM | n/a |

<sub>PyTorch also supports 16,384 with chunked cross-entropy (1,968 ms / 6,678 MB). Full table with every column and the chunked-CE variant is in [Full results](#full-results).</sub>

**What this says:**

- At short sequences (S=128) Dark Forest is on par with PyTorch. Past that, PyTorch is roughly 5x faster at S=1,024 and about 10x faster at S=4,096.
- PyTorch uses less memory because it runs bf16 with activation checkpointing. Dark Forest runs fp32 (TF32 tensor-core matmuls) and keeps activations resident.
- Dark Forest's ceiling on this card is S=4,096. PyTorch reaches 8,192 (16,384 with chunked CE).
- The attention kernel sustains only 0.2 to 0.5 TFLOP/s. Production kernels reach tens. There is a lot of room to improve here, and that's where contributions would matter most.

*Laptop thermals cause roughly ±10% run-to-run variation.*

---

## Roadmap: where the gap closes

1. **bf16 / mixed precision**, which closes most of the memory and throughput gap by itself
2. **Activation checkpointing**, taking activation memory from O(L·S) to O(S)
3. **Tensor-core attention**, lifting the kernel from 0.5 TFLOP/s toward hardware limits
4. **Backward-pass fusion** to cut memory traffic

---

## Quick start

```bash
# Kernel-isolation benchmark (Rust + CUDA)
cargo run --release --bin bench_sequence_limit --features cuda

# End-to-end training benchmark
python benchmark/bench_darkforest_e2e.py

# PyTorch baseline for comparison
python benchmark/bench_sdpa_baseline.py

# Everything (Windows)
powershell -ExecutionPolicy Bypass -File benchmark/run_all.ps1

# Check that the README tables match the committed JSON
python scripts/test_readme_tables.py
```

Every number in this README comes from committed scripts and JSON result files in `benchmark/`.

---

## Full results

### End-to-end training step (all configurations)

| Seq length | PyTorch SDPA (bf16 + ckpt) | PyTorch SDPA (bf16 + ckpt + chunked CE) | Dark Forest (fp32 + TF32) | DF peak VRAM |
|---:|---:|---:|---:|---:|
| 128 | 54.8 ms / 2,099 MB | 55.9 ms / 2,100 MB | 51.06 ms | 3,170 MB |
| 256 | 55.2 ms / 2,109 MB | 53.9 ms / 2,112 MB | n/a | n/a |
| 512 | 56.1 ms / 2,134 MB | 60.3 ms / 2,139 MB | 185.16 ms | 3,694 MB |
| 1,024 | 75.9 ms / 2,179 MB | 78.2 ms / 2,185 MB | 380.78 ms | 4,182 MB |
| 2,048 | 125.7 ms / 2,481 MB | 122.4 ms / 2,196 MB | 849.60 ms | 4,832 MB |
| 4,096 | 279.1 ms / 3,706 MB | 255.9 ms / 2,836 MB | 2,652.66 ms | 6,751 MB |
| 8,192 | 698.6 ms / 6,151 MB | 668.3 ms / 4,117 MB | OOM (>6.74 GB cap) | 7,013 MB |
| 16,384 | OOM | 1,968.2 ms / 6,678 MB | OOM | n/a |
| 32,768 | OOM | OOM | OOM | n/a |

**Parameter counts.** Dark Forest sizes position embeddings to the active context, giving 162.35M (S=128) to 163.82M (S=2,048) parameters. The PyTorch baseline was initialized once with 65,537 learned positions to cover the whole sweep, so it counts 212.5M parameters. The extra ~50M are position embeddings and do not affect per-step compute.

### Attention kernel in isolation

Forward pass only. Allocates just Q, K, V, and output. B=1, H=12, d_head=64, causal, fp32.

| Seq length | Median latency | Working set | Throughput |
|---:|---:|---:|---:|
| 8,192 | 190.7 ms | 96 MB | 0.5 TFLOP/s |
| 16,384 | 756.9 ms | 192 MB | 0.5 TFLOP/s |
| 32,768 | 3,782.6 ms | 384 MB | 0.4 TFLOP/s |
| 65,536 | 23,711.1 ms | 768 MB | 0.3 TFLOP/s |
| 131,072 | 108,989.4 ms | 1,536 MB | 0.2 TFLOP/s |

At identical shapes and precision (S=8,192), PyTorch SDPA also allocates 96 MB, so memory is a tie. Sequence lengths of 262,144 and above are not reported because the kernel was never run there; derivations are in `docs/LONG_CONTEXT_DERIVATION.md`.

### Environment

| | |
|---|---|
| GPU | NVIDIA GeForce RTX 5070 Laptop (8,123 MB GDDR7, sm_120) |
| OS | Windows 11 |
| Driver | NVIDIA 617.42 |
| CUDA | 13.4 (NVCC 13.4.59) |
| PyTorch | 2.11.0+cu128 |
| Rust | rustc / cargo 1.99.1 |
| Python | 3.12.10 |
| Measured | November 12 to 13, 2026 |

---

## License

MIT
