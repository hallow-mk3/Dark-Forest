"""
Dark Forest vs PyTorch — Automated Side-by-Side Verification Script
Runs both engines sequentially and prints an objective side-by-side performance table.
"""

import subprocess
import sys
import re
import os

def run_pytorch_benchmark():
    print("=" * 65)
    print(" 1/2 RUNNING PYTORCH BENCHMARK (12L GPT-2)...")
    print("=" * 65)
    cmd = [sys.executable, "bench_pytorch.py"]
    res = subprocess.run(cmd, capture_output=True, text=True)
    out = res.stdout + "\n" + res.stderr
    print(out)
    
    match = re.search(r"Eager median:\s+([0-9\.]+)\s+ms", out)
    if match:
        return float(match.group(1))
    match2 = re.search(r"Median step:\s+([0-9\.]+)\s+ms", out)
    if match2:
        return float(match2.group(1))
    return None

def run_darkforest_benchmark():
    print("=" * 65)
    print(" 2/2 RUNNING DARK FOREST STATIC CUDA BENCHMARK (12L GPT-2)...")
    print("=" * 65)
    
    # Path to release binary
    exe_path = os.path.join(os.path.dirname(__file__), "target", "release", "train_static.exe")
    if os.path.exists(exe_path):
        cmd = [
            exe_path,
            "--steps", "50", "--ctx-len", "128", "--d-model", "768",
            "--n-layers", "12", "--n-heads", "12", "--d-ff", "3072",
            "--vocab-size", "50257"
        ]
    else:
        cmd = [
            "cargo", "run", "-p", "darkforest-core", "--release",
            "--bin", "train_static", "--features", "cuda", "--",
            "--steps", "50", "--ctx-len", "128", "--d-model", "768",
            "--n-layers", "12", "--n-heads", "12", "--d-ff", "3072",
            "--vocab-size", "50257"
        ]
    env = os.environ.copy()
    env["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    res = subprocess.run(cmd, capture_output=True, text=True, env=env)
    out = res.stdout + "\n" + res.stderr
    print(out)
    
    match = re.search(r"Median step:\s+([0-9\.]+)\s+ms", out)
    if match:
        return float(match.group(1))
    return None

def main():
    print("\nDARK FOREST VS PYTORCH VERIFICATION HARNESS")
    print("Hardware: RTX 5070 Laptop GPU | Model: GPT-2 124M (12L, 768d, 50257v, ctx 128)\n")
    
    pt_med = run_pytorch_benchmark()
    df_med = run_darkforest_benchmark()
    
    print("\n" + "=" * 65)
    print(" FINAL VERIFICATION SUMMARY")
    print("=" * 65)
    if pt_med and df_med:
        speedup = pt_med / df_med
        print(f" PyTorch (Eager) Median Step Time : {pt_med:7.2f} ms (1.00x)")
        print(f" Dark Forest (Static CUDA Engine) : {df_med:7.2f} ms ({speedup:.2f}x faster)")
        print("-" * 65)
        if speedup > 1.0:
            print(f" RESULT: PASS -> Dark Forest is {speedup:.2f}x FASTER than PyTorch.")
        else:
            print(" RESULT: Dark Forest is slower than PyTorch.")
    else:
        print(" Could not parse median timing from one of the benchmarks.")
    print("=" * 65 + "\n")

if __name__ == "__main__":
    main()
