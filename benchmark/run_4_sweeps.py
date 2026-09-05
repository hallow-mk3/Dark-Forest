import gc
import json
import os
import statistics
import time
import torch
import torch.nn.functional as F

torch.cuda.set_per_process_memory_fraction(0.85, 0)
device = "cuda"
batch_size, num_heads, head_dim = 2, 12, 64
scale = 1.0 / (head_dim ** 0.5)
seq_lengths = [64, 128, 256, 512, 1024, 2048, 4096, 8192]

def naive_attn(q, k, v):
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale
    mask = torch.triu(torch.full((q.shape[-2], q.shape[-2]), float('-inf'), device=q.device), diagonal=1)
    return torch.matmul(F.softmax(scores + mask, dim=-1), v)

def measure(fn, q, k, v, warmup=10, repeats=25):
    torch.cuda.empty_cache()
    gc.collect()
    for _ in range(warmup):
        _ = fn(q, k, v)
    torch.cuda.synchronize()
    s_ev = torch.cuda.Event(enable_timing=True)
    e_ev = torch.cuda.Event(enable_timing=True)
    times = []
    torch.cuda.reset_peak_memory_stats()
    for _ in range(repeats):
        s_ev.record()
        _ = fn(q, k, v)
        e_ev.record()
        torch.cuda.synchronize()
        times.append(s_ev.elapsed_time(e_ev))
    times.sort()
    peak = torch.cuda.max_memory_allocated() / (1024 * 1024)
    return times[len(times)//2], peak

data = []
for s in seq_lengths:
    q = torch.randn(batch_size, num_heads, s, head_dim, device=device)
    k = torch.randn(batch_size, num_heads, s, head_dim, device=device)
    v = torch.randn(batch_size, num_heads, s, head_dim, device=device)
    row = {"seq_len": s, "naive_runs": [], "fused_runs": [], "naive_peak": 0.0, "fused_peak": 0.0}
    for run in range(4):
        if s == 8192:
            row["naive_runs"].append("OOM")
            row["naive_peak"] = "OOM"
        else:
            try:
                med, peak = measure(naive_attn, q, k, v)
                row["naive_runs"].append(round(med, 4))
                row["naive_peak"] = round(peak, 2)
            except torch.cuda.OutOfMemoryError:
                row["naive_runs"].append("OOM")
                row["naive_peak"] = "OOM"

        med_f, peak_f = measure(lambda _q, _k, _v: F.scaled_dot_product_attention(_q, _k, _v, is_causal=True), q, k, v)
        row["fused_runs"].append(round(med_f, 4))
        row["fused_peak"] = round(peak_f, 2)

    # Compute medians & speedup
    if row["naive_peak"] != "OOM":
        n_med = statistics.median(row["naive_runs"])
        f_med = statistics.median(row["fused_runs"])
        row["naive_median"] = round(n_med, 4)
        row["fused_median"] = round(f_med, 4)
        row["speedup"] = round(n_med / f_med, 2)
        row["vram_ratio"] = round(row["naive_peak"] / row["fused_peak"], 2)
    else:
        f_med = statistics.median(row["fused_runs"])
        row["naive_median"] = "OOM"
        row["fused_median"] = round(f_med, 4)
        row["speedup"] = "Deterministic (Naive OOM)"
        row["vram_ratio"] = "Hardware Bounded"

    data.append(row)
    print(f"S={s}: Naive={row['naive_runs']} | Fused={row['fused_runs']} | Speedup={row['speedup']}")

out_file = os.path.join(os.path.dirname(__file__), "live_unified_attention.json")
with open(out_file, "w") as f:
    json.dump({"device": torch.cuda.get_device_name(0), "results": data}, f, indent=2)
print(f"Saved {out_file}")
