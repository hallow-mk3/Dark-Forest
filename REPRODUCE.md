# Reproduction Guide: Dark Forest vs. PyTorch Baseline

This document explains exactly how to run the benchmarks I used to test the 12-layer GPT-2 training speed on my local hardware (NVIDIA GeForce RTX 5070 Laptop GPU, sm_120 Blackwell architecture, CUDA 13.3).

---

## 1. Environment Prerequisites

- **OS**: Windows 10/11 (x86_64)
- **Rust Toolchain**: `stable-x86_64-pc-windows-msvc`
- **C++ Compiler**: Microsoft Visual Studio 2022 C++ Build Tools (`cl.exe`)
- **CUDA Toolkit**: 12.8 or 13.3 with `nvcc.exe` on PATH
- **Python**: 3.10+ with `torch` (CUDA-enabled)

---

## 2. Step-by-Step Reproduction

### Step 1: Run PyTorch 2.9 Eager Baseline (12-Layer GPT-2)
This runs 7 independent trials of 20 steps each (140 total steps) and measures median step latency using GPU CUDA events:
```powershell
python benchmark/bench_exact_same_config.py --config full
```
**Witnessed Output**:
- Median of Medians: `~60.125 ms`
- Sample Mean: `~61.064 ms`
- Throughput: `~2,129 tok/s`

---

### Step 2: Build Dark Forest with Native CUDA Kernels
This compiles all 7 custom CUDA kernels (`.cu`) targeting `sm_120` and links against `cudart.lib` and `cublas.lib`:
```powershell
cargo build --release --bin train_static --features cuda
```

---

### Step 3: Run Dark Forest Static Engine (250 Steps)
This executes 250 contiguous training steps on the identical 124M-parameter architecture (d_model=768, 12 layers, 12 heads, d_ff=3072, vocab 50257, context 128, batch 1) with rolling 25-step averages:
```powershell
cargo run --release --bin train_static --features cuda -- --steps 250 --ctx-len 128 --d-model 768 --n-layers 12 --n-heads 12 --d-ff 3072 --vocab-size 50257
```
**Witnessed Output**:
- Initial Loss: `11.84` -> Final Loss: `2.68` (monotonic descent, min `2.38`–`2.46`)
- Median Step Time: `34.45 ms – 35.69 ms` (sustained AC) / `68.54 ms` (throttled)
- Throughput: `3,592 – 3,627 tok/s` (sustained AC) / `1,825 tok/s` (throttled)
- Speedup vs PyTorch: **1.70x – 2.05x faster** (mean ~1.91x across power/thermal states)

---

## 3. Raw Data Artifacts

- [`training_loss_static.csv`](training_loss_static.csv): Contiguous step, loss, rolling execution time, and step latency log.
- [`benchmark/bench_exact_same_config.py`](benchmark/bench_exact_same_config.py): Standard PyTorch baseline harness.
- [`darkforest-core/src/bin/train_static.rs`](darkforest-core/src/bin/train_static.rs): Static zero-allocation training engine source.
