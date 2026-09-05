PROJECT: Hyper-optimized ML framework (codename: TBD)

GOAL
Build a machine learning framework — Python bindings over a Rust core with 
hand-fused CUDA kernels — purpose-built for NVIDIA GPUs (targeting Ada/Blackwell 
architecture, e.g. RTX 5070). The framework must support training from scratch, 
fine-tuning, and inference, with an emphasis on running larger-than-normal models 
on consumer-grade single-GPU hardware through aggressive memory optimization.

This is NOT a general-purpose, hardware-portable framework like PyTorch. 
NVIDIA-only is a deliberate constraint — optimize hard for it, don't abstract 
around it.

ARCHITECTURE
┌─────────────────────────────────────┐
│  Python API layer (PyTorch-like)     │  ergonomic user-facing API
├─────────────────────────────────────┤
│  Rust core                           │  autograd engine, memory
│                                       │  management, graph scheduling
├─────────────────────────────────────┤
│  CUDA kernels (hand-fused)           │  tuned for target SM architecture
└─────────────────────────────────────┘

- Python bindings via PyO3 (Rust) or similar
- Rust core owns: tensor struct, autograd graph, memory allocator/pool, 
  scheduling, device management
- CUDA layer: hand-written fused kernels, starting with fused attention

CORE MEMORY OPTIMIZATION TECHNIQUES TO IMPLEMENT (in priority order)
1. Mixed precision (fp16/bf16) as default compute dtype
2. Gradient checkpointing (activation recomputation)
3. 4-bit/8-bit quantization for weights (QLoRA-style)
4. LoRA / parameter-efficient fine-tuning as a first-class training mode
5. VRAM <-> system RAM offloading for optimizer states (ZeRO-Offload-style)
6. Fused CUDA kernels for attention (FlashAttention-style tiling to reduce 
   memory reads/writes, not just FLOPs)

PHASE 1 SCOPE (this is what to build first — nothing more)
Build ONLY a minimal proof of concept to validate the core approach before any 
API design work:

1. A Rust-based autograd engine supporting: tensor ops (add, matmul, softmax, 
   layernorm), reverse-mode automatic differentiation, a computation graph 
   builder
2. CUDA kernel bindings for the above ops, callable from Rust
3. One hand-fused CUDA kernel: fused scaled-dot-product attention (forward + 
   backward passes)
4. A GPT-2-small-scale transformer (~125M params) defined using this engine, 
   trainable from scratch on a single RTX 5070
5. A benchmark harness comparing: (a) training step time, (b) peak VRAM usage, 
   (c) tokens/sec — against PyTorch eager-mode running the identical model/data 
   on the same hardware

DELIVERABLE FOR PHASE 1
A working repo that can:
- Train the small transformer on a toy dataset (e.g. tinyshakespeare or similar) 
  for a few thousand steps without crashing
- Print a benchmark comparison table vs PyTorch baseline
- Include a README documenting the architecture, how to run it, and current 
  results (even if the framework doesn't yet beat PyTorch — the point of Phase 
  1 is signal, not victory)

EXPLICITLY OUT OF SCOPE FOR PHASE 1
- Full Python API design/ergonomics
- Multi-GPU or distributed training
- Quantization, LoRA, offloading (these come in Phase 2, once the core engine 
  is validated)
- Support for any architecture other than a basic transformer decoder
- Model serving/deployment tooling

TECH STACK
- Rust (core engine, memory management, autograd)
- CUDA C++ (kernels), targeting compute capability for Blackwell/Ada 
  (confirm exact SM version for RTX 5070 before writing kernels)
- PyO3 for Python bindings (used only in the benchmark harness for Phase 1, 
  full API comes later)
- cuBLAS/cuDNN may be used as a fallback baseline to compare hand-written 
  kernels against — don't reinvent GEMM, focus fused-kernel effort on 
  attention and memory-bound ops

WORKING STYLE
- Work incrementally: get a naive unfused version running correctly first 
  (correctness > performance initially), verify gradients numerically 
  (gradient checking against finite differences), THEN optimize with fused 
  kernels
- Benchmark after every major change, not just at the end
- Flag early if a design decision (e.g. Rust autograd graph representation) 
  will make Phase 2 features (quantization, offloading) significantly harder — 
  raise it before continuing rather than after


Confirm the RTX 5070's compute capability (SM version) before kernel work starts — Blackwell consumer cards have specific tensor core generations that affect kernel design significantly, and getting this wrong wastes real time.
Gradient checking is not optional — a subtly wrong autograd engine will train something that looks plausible but is silently broken, and you won't catch it without numerical verification against finite differences.
If you're using Antigravity or Claude Code specifically, you may want to paste this in and then explicitly ask it to break Phase 1 into a task list before writing any code — that gives you a checkpoint to sanity-check the plan before the agent starts generating files.