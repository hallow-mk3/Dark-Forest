"""CUDA benchmark gate for Dark Forest.

This script is intentionally strict: it only runs the actual CUDA throughput path
when the current environment has a working NVIDIA CUDA runtime and torch CUDA
support available. Otherwise it exits cleanly with a skip message instead of
silently producing a meaningless CPU-only result.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def has_cuda() -> bool:
    try:
        import torch

        return bool(torch.cuda.is_available())
    except Exception:
        return False


def run_darkforest_cuda_benchmark(steps: int = 200, ctx_len: int = 16) -> int:
    cmd = [
        "cargo",
        "run",
        "-q",
        "-p",
        "darkforest-core",
        "--release",
        "--features",
        "cuda",
        "--bin",
        "train_gpt2_small",
        "--",
        "--steps",
        str(steps),
        "--ctx-len",
        str(ctx_len),
        "--device",
        "cuda",
    ]

    print("Running Dark Forest CUDA benchmark...")
    print("Command:", " ".join(cmd))
    completed = subprocess.run(cmd, cwd=str(ROOT), text=True)
    return completed.returncode


def main() -> int:
    if not has_cuda():
        print("CUDA benchmark skipped: no CUDA-enabled runtime detected in this environment.")
        print("This benchmark is a hard gate for the real product milestone; no CUDA means no final claim.")
        return 0

    print("CUDA detected. Running the real Dark Forest GPU benchmark path.")
    return run_darkforest_cuda_benchmark()


if __name__ == "__main__":
    sys.exit(main())
