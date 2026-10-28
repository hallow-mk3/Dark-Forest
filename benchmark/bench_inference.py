"""bench_inference.py -- Dark Forest KV-Cache vs. Naive Generation Latency Benchmark.

Measures autoregressive inference speed (ms/token, tokens/sec) for the
StaticGPT2 KV-cache fast path vs. naive full-context recomputation at
multiple token-generation budgets, then writes results to JSON.

Usage:
    python benchmark/bench_inference.py
    python benchmark/bench_inference.py --config small   # d_model=256, 4 layers
    python benchmark/bench_inference.py --config gpt2   # full GPT-2 scale
"""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

CONFIGS = {
    "small": {
        "d_model": 256, "n_layers": 4, "n_heads": 4, "d_ff": 1024,
        "ctx_len": 128, "vocab_size": None, "steps": 50,
        "description": "Small (d=256, 4L)",
    },
    "medium": {
        "d_model": 512, "n_layers": 6, "n_heads": 8, "d_ff": 2048,
        "ctx_len": 128, "vocab_size": None, "steps": 50,
        "description": "Medium (d=512, 6L)",
    },
    "gpt2": {
        "d_model": 768, "n_layers": 12, "n_heads": 12, "d_ff": 3072,
        "ctx_len": 128, "vocab_size": 50257, "steps": 20,
        "description": "GPT-2 Scale (d=768, 12L)",
    },
}

TOKEN_BUDGETS = [16, 32, 64, 128]


def run_generate_compare(cfg: dict, n_tokens: int) -> dict | None:
    cmd = [
        "cargo", "run", "--release", "--bin", "train_static",
        "--features", "cuda", "--",
        "--steps", str(cfg["steps"]),
        "--ctx-len", str(cfg["ctx_len"]),
        "--d-model", str(cfg["d_model"]),
        "--n-layers", str(cfg["n_layers"]),
        "--n-heads", str(cfg["n_heads"]),
        "--d-ff", str(cfg["d_ff"]),
        "--generate-compare", str(n_tokens),
    ]
    if cfg.get("vocab_size"):
        cmd += ["--vocab-size", str(cfg["vocab_size"])]

    print(f"  Running: cargo run ... --generate-compare {n_tokens}")
    t0 = time.perf_counter()
    result = subprocess.run(cmd, capture_output=True, text=True,
                            cwd=Path(__file__).parent.parent)
    wall = time.perf_counter() - t0

    if result.returncode != 0:
        print(f"  [FAIL] returncode={result.returncode}")
        print(result.stderr[-2000:])
        return None

    out = result.stdout
    all_speed = re.findall(r"Speed:\s+([\d.]+)\s+ms/token\s+\|\s+([\d.]+)\s+tok/s", out)
    speedup_match = re.search(r"Speedup:\s+([\d.]+)x faster", out)

    if len(all_speed) < 1:
        print(f"  [WARN] Could not parse output. Tail:\n{out[-600:]}")
        return None

    fast_ms = float(all_speed[0][0])
    fast_tps = float(all_speed[0][1])
    naive_ms = float(all_speed[1][0]) if len(all_speed) >= 2 else None
    naive_tps = float(all_speed[1][1]) if len(all_speed) >= 2 else None
    speedup = float(speedup_match.group(1)) if speedup_match else (
        naive_ms / fast_ms if naive_ms else None
    )

    return {
        "n_tokens": n_tokens,
        "fast_ms_per_token": fast_ms,
        "fast_tps": fast_tps,
        "naive_ms_per_token": naive_ms,
        "naive_tps": naive_tps,
        "speedup": speedup,
        "wall_time_s": wall,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", choices=list(CONFIGS.keys()), default="small")
    parser.add_argument("--tokens", nargs="+", type=int, default=TOKEN_BUDGETS)
    parser.add_argument("--output", default="benchmark/inference_results.json")
    args = parser.parse_args()

    cfg = CONFIGS[args.config]
    print("=" * 72)
    print(f"  Dark Forest -- Inference Latency Benchmark")
    print(f"  Config: {cfg['description']}")
    print(f"  Token budgets: {args.tokens}")
    print("=" * 72)

    results = {"config": args.config, "model": cfg, "runs": []}

    for n_tok in args.tokens:
        print(f"\n[Token budget = {n_tok}]")
        r = run_generate_compare(cfg, n_tok)
        if r:
            results["runs"].append(r)
            print(f"  Fast: {r['fast_ms_per_token']:.3f} ms/token ({r['fast_tps']:.1f} tok/s)", end="")
            if r["naive_ms_per_token"]:
                print(f"  |  Naive: {r['naive_ms_per_token']:.3f} ms/token  |  Speedup: {r['speedup']:.2f}x")
            else:
                print()

    print("\n" + "=" * 72)
    print("  KV-Cache Inference Speedup Summary")
    print("=" * 72)
    print(f"  {'Tokens':>8}  {'Fast ms/tok':>12}  {'Fast tok/s':>10}  {'Naive ms/tok':>13}  {'Speedup':>8}")
    print("  " + "-" * 65)
    for r in results["runs"]:
        naive_str = f"{r['naive_ms_per_token']:>13.3f}" if r["naive_ms_per_token"] else "          N/A"
        sp_str = f"{r['speedup']:>8.2f}x" if r["speedup"] else "       N/A"
        print(f"  {r['n_tokens']:>8}  {r['fast_ms_per_token']:>12.3f}  {r['fast_tps']:>10.1f}  {naive_str}  {sp_str}")
    print("=" * 72)

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\n  Results saved to: {out_path}")
    return 0 if results["runs"] else 1


if __name__ == "__main__":
    sys.exit(main())
