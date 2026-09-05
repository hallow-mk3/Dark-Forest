# Reproduction Guide: Dark Forest vs. PyTorch Baseline

This document provides exact, reproducible commands to evaluate the 12-layer GPT-2 training benchmark on local hardware (tested on an NVIDIA GeForce RTX 5070 Laptop GPU, sm_120 Blackwell architecture, CUDA 13.3).

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
Runs 7 independent trials of 20 steps each (140 total steps) measuring median step latency using GPU CUDA events:
```powershell
python benchmark/bench_exact_same_config.py --config full
```
**Witnessed Output**:
- Median of Medians: `~60.125 ms`
- Sample Mean: `~61.064 ms`
- Throughput: `~2,129 tok/s`

---

### Step 2: Build Dark Forest with Native CUDA Kernels
Compiles all 7 custom CUDA kernels (`.cu`) targeting `sm_120` and links against `cudart.lib` and `cublas.lib`:
```powershell
cargo build --release --bin train_static --features cuda
```

---

### Step 3: Run Dark Forest Static Engine (250 Steps)
Executes 250 contiguous training steps on the identical 124M-parameter architecture ($d_{\text{model}}=768$, 12 layers, 12 heads, $d_{\text{ff}}=3072$, vocab 50,257, context 128, batch 1) with rolling 25-step averages:
```powershell
cargo run --release --bin train_static --features cuda -- --steps 250 --ctx-len 128 --d-model 768 --n-layers 12 --n-heads 12 --d-ff 3072 --vocab-size 50257
```
**Witnessed Output**:
- Initial Loss: `11.84` → Final Loss: `2.68` (monotonic descent)
- Median Step Time: `~35.292 ms`
- Mean Step Time: `~35.697 ms`
- Throughput: `~3,627 tok/s`
- Speedup vs PyTorch: **1.70× faster**

---

## 3. Raw Data Artifacts

- [`training_loss_static.csv`](file:///c:/Users/Swasthik%20Shetty/Dark%20Forest/training_loss_static.csv): Contiguous step, loss, rolling execution time, and step latency log.
- [`benchmark/bench_exact_same_config.py`](file:///c:/Users/Swasthik%20Shetty/Dark%20Forest/benchmark/bench_exact_same_config.py): Standard PyTorch baseline harness.
- [`darkforest-core/src/bin/train_static.rs`](file:///c:/Users/Swasthik%20Shetty/Dark%20Forest/darkforest-core/src/bin/train_static.rs): Static zero-allocation training engine source.
