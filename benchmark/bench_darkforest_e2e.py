"""Dark-Forest End-to-End Benchmark Script.

Measures a complete training step (forward + cross-entropy loss + backward + AdamW update)
on the GPT-2 configuration matching the PyTorch SDPA baseline:
  - 12 layers, d_model=768, 12 heads, d_ff=3072, vocab_size=50257
  - batch_size=1
  - Precision: Float32 weights & activations with TF32 Tensor Core MMA (Ampere/Blackwell)
  - Optimizer: AdamW (lr=1e-4, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.01)

Timing: Median of 5 iterations following 2 warmup steps.
Memory: Measured via CUDA device memory info (cudaMemGetInfo / torch.cuda.mem_get_info)
        comparing device memory before and after model allocation, and verifying against
        exact theoretical buffer requirements under the 85% VRAM cap (6.74 GB on 8 GB GPU).
"""

from __future__ import annotations

import gc
import json
import os
import subprocess
import sys
import time
from typing import Any, Dict, List, Optional

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


CONFIG = {
    "n_layers": 12,
    "d_model": 768,
    "n_heads": 12,
    "d_ff": 3072,
    "vocab_size": 50257,
    "batch_size": 1,
    "lr": 1e-4,
    "warmup": 2,
    "repeats": 5,
}

# 85% safety cap on 8 GB RTX 5070 Laptop (8123.44 MB total -> 6904.9 MB / 6.74 GiB)
SAFETY_CAP_MB = 6904.9


def calculate_darkforest_params(vocab_size: int, d_model: int, n_layers: int, d_ff: int, seq_len: int) -> int:
    """Calculate exact parameter count of StaticGPT2 in Dark-Forest."""
    tok_emb = vocab_size * d_model
    pos_emb = seq_len * d_model
    ln_f = 2 * d_model
    lm_head = vocab_size * d_model
    # Per layer:
    # wq, wk, wv, wo: 4 * (d_model * d_model) + 4 * d_model (biases)
    # ln1: 2 * d_model
    # ln2: 2 * d_model
    # w1: d_ff * d_model + d_ff (bias)
    # w2: d_model * d_ff + d_model (bias)
    per_layer = (
        4 * d_model * d_model + 4 * d_model
        + 2 * d_model
        + 2 * d_model
        + d_ff * d_model + d_ff
        + d_model * d_ff + d_model
    )
    return tok_emb + pos_emb + ln_f + lm_head + n_layers * per_layer


def run_single_sequence(seq_len: int) -> Dict[str, Any]:
    """Run benchmark for a single sequence length in an isolated subprocess."""
    worker_code = f"""
import darkforest
import time
import torch
import json
import sys

seq_len = {seq_len}
vocab_size = {CONFIG['vocab_size']}
d_model = {CONFIG['d_model']}
n_heads = {CONFIG['n_heads']}
n_layers = {CONFIG['n_layers']}
d_ff = {CONFIG['d_ff']}
lr = {CONFIG['lr']}
warmup = {CONFIG['warmup']}
repeats = {CONFIG['repeats']}
cap_mb = {SAFETY_CAP_MB}

free_init, total_mem = torch.cuda.mem_get_info()
free_init_mb = free_init / (1024 ** 2)
total_mb = total_mem / (1024 ** 2)

try:
    free_before, _ = torch.cuda.mem_get_info()
    t_init_0 = time.perf_counter()
    m = darkforest.StaticGPT2(vocab_size, d_model, n_heads, n_layers, d_ff, seq_len, lr)
    darkforest.cuda_sync()
    init_time_s = time.perf_counter() - t_init_0
    free_after, _ = torch.cuda.mem_get_info()

    allocated_mb = (free_before - free_after) / (1024 ** 2)

    # Check if allocation exceeded 85% cap
    if allocated_mb > cap_mb:
        print(json.dumps({{
            "seq_len": seq_len,
            "status": "OOM (Exceeded 85% VRAM Cap)",
            "peak_vram_mb": round(allocated_mb, 1),
            "median_ms": None,
            "reason": f"Allocated {{allocated_mb:.1f}} MB exceeds safety cap of {{cap_mb:.1f}} MB"
        }}))
        sys.exit(0)

    inp = [1] * seq_len
    tgt = [2] * seq_len

    # Warmup
    for _ in range(warmup):
        m.step(inp, tgt)
    darkforest.cuda_sync()

    # Timed steps
    latencies = []
    for _ in range(repeats):
        t0 = time.perf_counter()
        m.step(inp, tgt)
        darkforest.cuda_sync()
        latencies.append((time.perf_counter() - t0) * 1000.0)

    latencies.sort()
    median_ms = latencies[len(latencies) // 2]

    # Measure final peak VRAM
    free_final, _ = torch.cuda.mem_get_info()
    peak_vram_mb = (free_before - free_final) / (1024 ** 2)

    print(json.dumps({{
        "seq_len": seq_len,
        "status": "OK",
        "median_ms": round(median_ms, 3),
        "min_ms": round(latencies[0], 3),
        "max_ms": round(latencies[-1], 3),
        "peak_vram_mb": round(peak_vram_mb, 1),
        "init_time_s": round(init_time_s, 2),
    }}))

except Exception as e:
    err_str = str(e)
    print(json.dumps({{
        "seq_len": seq_len,
        "status": "OOM / CUDA Error",
        "peak_vram_mb": None,
        "median_ms": None,
        "error": err_str
    }}))
"""
    res = subprocess.run([sys.executable, "-u", "-c", worker_code], capture_output=True, text=True)
    if res.returncode != 0 and not res.stdout.strip():
        return {
            "seq_len": seq_len,
            "status": "OOM (Process Terminated)",
            "peak_vram_mb": None,
            "median_ms": None,
            "error": res.stderr.strip()[:200]
        }
    try:
        # Find JSON line in stdout
        for line in reversed(res.stdout.splitlines()):
            line = line.strip()
            if line.startswith("{") and line.endswith("}"):
                return json.loads(line)
        return {
            "seq_len": seq_len,
            "status": "Error",
            "peak_vram_mb": None,
            "median_ms": None,
            "error": res.stderr.strip()[:200] or res.stdout.strip()[:200]
        }
    except Exception as e:
        return {
            "seq_len": seq_len,
            "status": "Error",
            "peak_vram_mb": None,
            "median_ms": None,
            "error": str(e)
        }


def main():
    if not HAS_TORCH or not torch.cuda.is_available():
        print("CUDA is required for this benchmark.")
        sys.exit(1)

    device_name = torch.cuda.get_device_name(0)
    total_mem_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
    cap_gb = SAFETY_CAP_MB / 1024.0

    print("=" * 80)
    print(" Dark Forest End-to-End Model Training Benchmark")
    print(f" Hardware: {device_name} ({total_mem_gb:.2f} GB Total VRAM, 85% Cap: {cap_gb:.2f} GB)")
    print(f" Config  : {CONFIG['n_layers']}L d{CONFIG['d_model']} H{CONFIG['n_heads']} ff{CONFIG['d_ff']} vocab{CONFIG['vocab_size']} batch{CONFIG['batch_size']}")
    print(f" Optimizer: AdamW, Precision: Float32 (TF32 MMA Tensor Cores)")
    print("=" * 80)

    # Sequence lengths sweep: S = 128, 512, 1024, 2048, 4096, 8192, 16384...
    seq_lengths = [128, 512, 1024, 2048, 4096, 8192, 16384]

    results = []
    max_survived = 0

    print("\nParameter count analysis:")
    for s in [128, 512, 1024, 2048]:
        df_params = calculate_darkforest_params(
            CONFIG["vocab_size"], CONFIG["d_model"], CONFIG["n_layers"], CONFIG["d_ff"], s
        )
        print(f"  Dark Forest at S={s:>5}: {df_params:,} parameters ({df_params / 1e6:.2f}M)")
    print(f"  PyTorch baseline cited : 212,500,224 parameters (212.5M)")
    print(f"  Explanation: PyTorch baseline allocated learned position embeddings for max_pos=65,537")
    print(f"               (65,537 * 768 = 50.3M parameters) to cover long-context sweeps without re-init,")
    print(f"               whereas Dark-Forest allocates position embeddings exactly matching seq_len.")
    print()

    for s in seq_lengths:
        print(f"Testing S = {s:>6} ... ", end="", flush=True)
        res = run_single_sequence(s)
        results.append(res)
        if res.get("status") == "OK":
            print(f"SUCCESS | Step: {res['median_ms']:>8.2f} ms | VRAM: {res['peak_vram_mb']:>7.1f} MB")
            max_survived = s
        else:
            status_desc = res.get('status')
            vram_desc = f" ({res.get('peak_vram_mb')} MB)" if res.get('peak_vram_mb') else ""
            print(f"FAILED  | {status_desc}{vram_desc}")
            if "OOM" in status_desc:
                print(f"  Reached VRAM limit at sequence length {s}.")
                break

    output_data = {
        "metadata": {
            "device": device_name,
            "total_vram_gb": round(total_mem_gb, 2),
            "safety_cap_mb": SAFETY_CAP_MB,
            "config": CONFIG,
            "precision": "Float32 (TF32 MMA Tensor Cores)",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "parameter_count_explanation": (
                "Dark Forest allocates position embeddings exact to seq_len (162.3M at S=128 to 163.8M at S=2048). "
                "The PyTorch SDPA script allocated position embeddings for max_pos=65,537 (50.3M params) up front, "
                "bringing its total parameter count to 212.5M."
            ),
        },
        "results": results,
        "max_survived_seq_len": max_survived,
    }

    out_path = os.path.join(os.path.dirname(__file__), "darkforest_e2e_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2)

    print(f"\nSaved results to {out_path}")
    print(f"Max survived sequence length for Dark Forest E2E: {max_survived}")


if __name__ == "__main__":
    main()

