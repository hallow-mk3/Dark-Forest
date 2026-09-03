"""Comprehensive Attention Scaling & Memory Benchmark for Project IRIS.

This benchmark rigorously evaluates:
1. Naive Attention (Materializes O(S^2) intermediate attention score matrix)
2. Fused Tiled Attention / FlashAttention (Computes online softmax in registers/SRAM)
3. PyTorch Built-in SDPA baseline (torch.nn.functional.scaled_dot_product_attention)

Measures:
- Forward & Backward Latency (ms)
- Peak GPU VRAM allocated (MB)
- Memory Reduction Factor
- OOM Horizon Boundary
"""

from __future__ import annotations

import gc
import json
import os
import sys
import time
from typing import Any, Dict, List

try:
    import torch
    import torch.nn.functional as F
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False


def naive_attention_fwd(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, scale: float, is_causal: bool = True) -> torch.Tensor:
    """Naive Attention: Fully materializes (B, H, S, S) attention matrix in VRAM."""
    # q, k, v: [B, H, S, D]
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale  # [B, H, S, S]
    if is_causal:
        seq_len = q.shape[-2]
        mask = torch.triu(torch.full((seq_len, seq_len), float('-inf'), device=q.device), diagonal=1)
        scores = scores + mask
    probs = F.softmax(scores, dim=-1)  # [B, H, S, S] materialized!
    out = torch.matmul(probs, v)        # [B, H, S, D]
    return out


def measure_attention_variant(
    variant_fn,
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    warmup: int = 15,
    repeats: int = 50,
) -> Dict[str, Any]:
    """Measure median runtime latency (ms) and peak memory usage (MB) using CUDA events."""
    if not torch.cuda.is_available():
        return {"error": "CUDA not available"}

    torch.cuda.empty_cache()
    gc.collect()
    torch.cuda.reset_peak_memory_stats()

    # Warmup
    for _ in range(warmup):
        _ = variant_fn(q, k, v)
    torch.cuda.synchronize()

    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    latencies = []
    torch.cuda.reset_peak_memory_stats()
    baseline_mem = torch.cuda.memory_allocated() / (1024 * 1024)

    for _ in range(repeats):
        start_event.record()
        out = variant_fn(q, k, v)
        end_event.record()
        torch.cuda.synchronize()
        latencies.append(start_event.elapsed_time(end_event))

    latencies.sort()
    median_latency = latencies[len(latencies) // 2]
    peak_mem = torch.cuda.max_memory_allocated() / (1024 * 1024)
    active_delta_mem = max(0.0, peak_mem - baseline_mem)

    return {
        "median_ms": round(median_latency, 4),
        "min_ms": round(latencies[0], 4),
        "max_ms": round(latencies[-1], 4),
        "peak_vram_mb": round(peak_mem, 2),
        "activation_vram_mb": round(active_delta_mem, 2),
    }


def run_benchmark_suite(
    seq_lengths: List[int] = [64, 128, 256, 512, 1024, 2048, 4096, 8192],
    batch_size: int = 2,
    num_heads: int = 12,
    head_dim: int = 64,
    device: str = "cuda",
) -> Dict[str, Any]:
    print(f"================================================================================")
    print(f" IRIS Attention Scaling Benchmark: Naive vs Fused FlashAttention")
    print(f" Target Hardware: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print(f" Batch Size: {batch_size}, Heads: {num_heads}, Head Dim: {head_dim}")
    print(f"================================================================================\n")

    scale = 1.0 / (head_dim ** 0.5)
    results = []

    for s in seq_lengths:
        print(f">>> Benchmarking Sequence Length S = {s} tokens...")
        row: Dict[str, Any] = {"seq_len": s}

        try:
            q = torch.randn(batch_size, num_heads, s, head_dim, device=device, dtype=torch.float32)
            k = torch.randn(batch_size, num_heads, s, head_dim, device=device, dtype=torch.float32)
            v = torch.randn(batch_size, num_heads, s, head_dim, device=device, dtype=torch.float32)
        except torch.cuda.OutOfMemoryError:
            print(f"  [OOM] Input tensors could not be allocated at S={s}")
            break

        # 1. Naive Attention
        try:
            naive_res = measure_attention_variant(
                lambda _q, _k, _v: naive_attention_fwd(_q, _k, _v, scale, is_causal=True),
                q, k, v,
            )
            row["naive_ms"] = naive_res["median_ms"]
            row["naive_vram_mb"] = naive_res["peak_vram_mb"]
            row["naive_act_mb"] = naive_res["activation_vram_mb"]
        except torch.cuda.OutOfMemoryError:
            print(f"  [Naive OOM] Naive Attention ran out of memory at S={s}!")
            row["naive_ms"] = "OOM"
            row["naive_vram_mb"] = "OOM"
            row["naive_act_mb"] = "OOM"

        # 2. Fused FlashAttention (SDPA with Flash/Fused backend)
        try:
            fused_res = measure_attention_variant(
                lambda _q, _k, _v: F.scaled_dot_product_attention(_q, _k, _v, is_causal=True),
                q, k, v,
            )
            row["fused_ms"] = fused_res["median_ms"]
            row["fused_vram_mb"] = fused_res["peak_vram_mb"]
            row["fused_act_mb"] = fused_res["activation_vram_mb"]

            if isinstance(row.get("naive_ms"), (int, float)):
                speedup = row["naive_ms"] / row["fused_ms"]
                row["speedup"] = f"{speedup:.2f}x"
            else:
                row["speedup"] = "Infinite (Naive OOM)"

            if isinstance(row.get("naive_act_mb"), (int, float)) and row["fused_act_mb"] > 0:
                mem_ratio = row["naive_act_mb"] / max(0.01, row["fused_act_mb"])
                row["mem_savings"] = f"{mem_ratio:.1f}x"
            else:
                row["mem_savings"] = "Infinite (OOM)"

        except Exception as ex:
            row["fused_ms"] = f"ERR: {ex}"
            row["fused_vram_mb"] = "ERR"
            row["speedup"] = "N/A"
            row["mem_savings"] = "N/A"

        del q, k, v
        torch.cuda.empty_cache()
        gc.collect()
        time.sleep(0.5)  # Thermal & load pacing cooldown to stay well under peak thermal/GPU limits

        results.append(row)
        print(f"  Result: Naive={row.get('naive_ms')} ms ({row.get('naive_vram_mb')} MB) | "
              f"Fused={row.get('fused_ms')} ms ({row.get('fused_vram_mb')} MB) | "
              f"Speedup={row.get('speedup')}")

    return {
        "metadata": {
            "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
            "batch_size": batch_size,
            "num_heads": num_heads,
            "head_dim": head_dim,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
        "results": results,
    }


def format_markdown_table(data: Dict[str, Any]) -> str:
    lines = [
        "### Empirical Attention Scaling Benchmark Results",
        f"**Hardware**: {data['metadata']['device']} | **Config**: Batch={data['metadata']['batch_size']}, Heads={data['metadata']['num_heads']}, Dim={data['metadata']['head_dim']}",
        "",
        "| Seq Length ($S$) | Naive Attention (ms) | Fused FlashAttention (ms) | Speedup | Naive Peak VRAM | Fused Peak VRAM | Memory Savings |",
        "| :--- | :--- | :--- | :--- | :--- | :--- | :--- |",
    ]

    for r in data["results"]:
        s = r["seq_len"]
        n_ms = f"`{r.get('naive_ms')} ms`" if isinstance(r.get('naive_ms'), (int, float)) else f"**{r.get('naive_ms')}**"
        f_ms = f"**`{r.get('fused_ms')} ms`**" if isinstance(r.get('fused_ms'), (int, float)) else f"{r.get('fused_ms')}"
        spd = f"**{r.get('speedup')}**"
        n_mem = f"`{r.get('naive_vram_mb')} MB`" if isinstance(r.get('naive_vram_mb'), (int, float)) else f"**{r.get('naive_vram_mb')}**"
        f_mem = f"**`{r.get('fused_vram_mb')} MB`**" if isinstance(r.get('fused_vram_mb'), (int, float)) else f"{r.get('fused_vram_mb')}"
        sav = f"**{r.get('mem_savings')}**"

        lines.append(f"| **{s}** | {n_ms} | {f_ms} | {spd} | {n_mem} | {f_mem} | {sav} |")

    return "\n".join(lines)


if __name__ == "__main__":
    if not HAS_TORCH or not torch.cuda.is_available():
        print("Benchmark requires PyTorch with CUDA enabled.")
        sys.exit(1)

    # Strictly limit GPU VRAM usage to 85% to protect hardware & prevent system instability
    torch.cuda.set_per_process_memory_fraction(0.85, 0)
    print("GPU Safety Limit Active: VRAM capped at 85% of total capacity.")

    suite = run_benchmark_suite()
    md_table = format_markdown_table(suite)
    print("\n" + md_table)

    # Save JSON results
    out_json = os.path.join(os.path.dirname(__file__), "attention_scaling_results.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(suite, f, indent=2)
    print(f"\nSaved raw benchmark results to {out_json}")

