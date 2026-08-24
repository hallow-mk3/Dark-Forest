# Dark Forest Product Strategy

## Product thesis

Dark Forest should not try to out-implement PyTorch feature-for-feature. Instead, it should win on the workloads where explicit control, deterministic execution, and minimal hidden memory movement matter more than a massive ecosystem.

### Where it is stronger

- explicit device residency rules reduce surprise copies
- Rust core gives memory safety and predictable execution
- a smaller API surface is easier to reason about and ship reliably
- CUDA execution is more clearly controlled and easier to profile

### Where it is not yet competitive

- ecosystem breadth: PyTorch has broad model support, tooling, and integrations
- distributed training and multi-GPU orchestration
- end-user ergonomics and notebook-first workflows
- broad compiler/runtime support across hardware and OS environments

## Initial ship target

The first ship target is not a general replacement for PyTorch. It is a narrower product:

- CPU + CUDA training runtime for single-device experiments
- Rust-first autograd engine with explicit Tensor device semantics
- Python bindings for model definition and benchmark integration
- deterministic numerical checks and clear failure modes

## Roadmap

### Phase 1: stable alpha

- stable CPU autograd passes
- CUDA fallback path when GPU toolchain is unavailable
- Python package scaffolding via maturin
- clear install and build instructions
- release notes and supported environment matrix

### Phase 2: product-grade runtime

- fully explicit device transfer policies
- optimizer state pinned to device when applicable
- fused CPU/GPU execution boundaries for common ops
- a model zoo with small, tested architectures
- benchmarking harness with median/variance reporting

### Phase 3: real-world adoption

- user-facing training APIs inspired by PyTorch but smaller
- serialization and checkpointing support
- mixed precision and memory profiling
- reproducible examples for fine-tuning and benchmarking

## Product positioning

Instead of "PyTorch replacement," the cleaner positioning is:

> Dark Forest is a Rust-native ML runtime for controlled, device-aware training workloads that value performance clarity and execution determinism over framework breadth.

That is a defensible product story and a much more realistic target than claiming universal superiority over PyTorch.
