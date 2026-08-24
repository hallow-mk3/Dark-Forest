import subprocess
import time
import json
import statistics

print("==========================================================")
print(" Dark Forest vs. PyTorch Comprehensive Benchmark Suite")
print("==========================================================")

# 1. Run Matrix Multiplication Benchmark across Sizes
sizes = [
    (128, 128),
    (512, 512),
    (1024, 1024),
    (2048, 2048),
]

import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
print(f"Target Hardware: {device_name}")
print(f"PyTorch Version: {torch.__version__}")
print("----------------------------------------------------------")

results = []

for rows, cols in sizes:
    # PyTorch Matmul Benchmark
    a_torch = torch.randn(rows, cols, device=device, dtype=torch.float32)
    b_torch = torch.randn(cols, rows, device=device, dtype=torch.float32)

    # Warmup
    for _ in range(10):
        c = torch.matmul(a_torch, b_torch)
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    torch_times = []
    n_iters = 100
    for _ in range(n_iters):
        t0 = time.perf_counter()
        c = torch.matmul(a_torch, b_torch)
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        t1 = time.perf_counter()
        torch_times.append((t1 - t0) * 1000.0)

    pt_median = statistics.median(torch_times)
    pt_mean = statistics.mean(torch_times)

    print(f"[MatMul {rows}x{cols}] PyTorch: {pt_median:.4f} ms")
    results.append({
        "workload": f"MatMul {rows}x{cols}",
        "pytorch_median_ms": round(pt_median, 4),
        "pytorch_mean_ms": round(pt_mean, 4),
    })

print("\nBenchmark measurements completed.")
with open("benchmark_results.json", "w") as f:
    json.dump({"device": device_name, "benchmarks": results}, f, indent=2)
print("Saved benchmark_results.json")
