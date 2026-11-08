# Dark-Forest Project Rules

## Project Overview
**Dark-Forest** is a specialized high-performance AI training framework and optimization engine (cloned from `https://github.com/hallow-mk3/Dark-Forest`). It delivers hardware-accelerated model training, CUDA kernels, core training loops, and distributed computation pipelines.

---

## Strict Isolation Rules
- **Autonomous Training Core**: Dark-Forest operates as an independent repository with its own Git tree, Rust crates (`darkforest-core`, `darkforest-cuda`), Python bindings (`darkforest-py`), and benchmark suites.
- **Zero Cross-Contamination**:
  - No training scripts, data loaders, or CUDA kernels may import files or modules from sibling directories (`Entropy`, `Optimus`, `Flow`, etc.).
  - Dark-Forest must remain completely standalone and buildable solely via its own `Cargo.toml`, `requirements.txt`, or Python package setup.
- **Data & Checkpoint Isolation**: Datasets (`data/`), model weights, checkpoints, logs, and benchmark results must remain inside this folder or point to explicitly configured external data volumes, never referencing paths in sibling folders.
- **Git Separation**: Do not commit parent workspace files into Dark-Forest's git history, and do not track Dark-Forest changes inside sibling project trees.

