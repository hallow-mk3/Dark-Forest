"""
PyTorch Baseline & Dark Forest Benchmark Script

This benchmark is intentionally split into two modes:
1) a full-model GPT-2 baseline for end-to-end PyTorch reference timing
2) a same-workload matrix benchmark between PyTorch and Dark Forest

The second mode is the apples-to-apples comparison that the project should rely on.

Timing method: CUDA events (torch.cuda.Event) for GPU-accurate sub-microsecond timing.
Python time.perf_counter measures kernel-dispatch + sync latency; CUDA events measure
only actual GPU execution time.
"""

import os
import time
import statistics

import torch
import torch.nn as nn

try:
    import darkforest
except ImportError:
    darkforest = None

class PyTorchGPT2Small(nn.Module):
    def __init__(self, vocab_size=50257, d_model=768, n_heads=1, n_layers=12, max_len=1024):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_len, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=n_heads, dim_feedforward=3072,
            activation='gelu', batch_first=True, norm_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, idx):
        B, T = idx.shape
        pos = torch.arange(0, T, dtype=torch.long, device=idx.device)
        x = self.tok_emb(idx) + self.pos_emb(pos)
        mask = nn.Transformer.generate_square_subsequent_mask(T, device=idx.device)
        x = self.transformer(x, mask=mask, is_causal=True)
        x = self.ln_f(x)
        logits = self.head(x)
        return logits

def benchmark_pytorch(steps=20, seq_len=128, batch_size=2, warmup_steps=10, repeats=5):
    torch.manual_seed(0)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cpu":
        torch.set_num_threads(max(1, os.cpu_count() or 1))
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass

    print(f"\n=======================================================")
    print(f" PyTorch Baseline Benchmark (Device: {device})")
    print(f"=======================================================")
    print(f"Config: vocab=256, d_model=128, n_heads=1, n_layers=4, seq_len={seq_len}, batch_size={batch_size}, ff=512, max_len=512")

    model = PyTorchGPT2Small(vocab_size=256, d_model=128, n_heads=1, n_layers=4, max_len=512).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    inputs = torch.randint(0, 256, (batch_size, seq_len), device=device)
    targets = torch.randint(0, 256, (batch_size, seq_len), device=device)

    for _ in range(warmup_steps):
        optimizer.zero_grad()
        out = model(inputs)
        loss = loss_fn(out.view(-1, 256), targets.view(-1))
        loss.backward()
        optimizer.step()

    step_times = []
    peak_vrams = []
    throughputs = []
    for _ in range(repeats):
        if device == "cuda":
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()

        t0 = time.perf_counter()
        for _ in range(steps):
            optimizer.zero_grad()
            out = model(inputs)
            loss = loss_fn(out.view(-1, 256), targets.view(-1))
            loss.backward()
            optimizer.step()

        if device == "cuda":
            torch.cuda.synchronize()
            peak_vram = torch.cuda.max_memory_allocated() / (1024 * 1024)
        else:
            peak_vram = 0.0

        total_time = time.perf_counter() - t0
        step_times.append((total_time / steps) * 1000.0)
        peak_vrams.append(peak_vram)
        throughputs.append((steps * batch_size * seq_len) / total_time)

    step_time_ms = statistics.median(step_times)
    peak_vram = max(peak_vrams)
    tokens_per_sec = statistics.median(throughputs)

    print(f"Step Time:      {step_time_ms:.2f} ms")
    print(f"Peak VRAM:      {peak_vram:.2f} MB")
    print(f"Throughput:     {tokens_per_sec:.1f} tokens/sec")
    print(f"Timing range:   {min(step_times):.2f}-{max(step_times):.2f} ms/step ({repeats} trials)")
    print(f"=======================================================\n")
    return {"step_time_ms": step_time_ms, "peak_vram_mb": peak_vram, "tokens_per_sec": tokens_per_sec}


def _cuda_event_time_ms(fn, warmup=5, repeats=20):
    """Run fn() and return a list of per-call GPU execution times in ms using CUDA events.

    CUDA events record timestamps on the GPU timeline with ~0.5 us resolution.
    This strips Python dispatch overhead and measures only actual GPU work.
    """
    use_cuda = torch.cuda.is_available()

    if not use_cuda:
        for _ in range(warmup):
            fn()
        times = []
        for _ in range(repeats):
            t0 = time.perf_counter()
            fn()
            times.append((time.perf_counter() - t0) * 1000.0)
        return times

    # Warmup
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()

    times = []
    for _ in range(repeats):
        start = torch.cuda.Event(enable_timing=True)
        end   = torch.cuda.Event(enable_timing=True)
        start.record()
        fn()
        end.record()
        torch.cuda.synchronize()
        times.append(start.elapsed_time(end))  # milliseconds

    return times


def benchmark_pytorch_matrix(steps=20, rows=256, cols=256, warmup_steps=5, repeats=20):
    torch.manual_seed(0)
    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
        torch.set_num_threads(max(1, os.cpu_count() or 1))
        try:
            torch.set_num_interop_threads(1)
        except RuntimeError:
            pass

    a = torch.randn(rows, cols, device=device, dtype=torch.float32, requires_grad=True)
    b = torch.randn(cols, rows, device=device, dtype=torch.float32, requires_grad=True)

    def one_step():
        out = a @ b
        loss = out.sum()
        loss.backward()
        with torch.no_grad():
            if a.grad is not None:
                a.grad.zero_()
            if b.grad is not None:
                b.grad.zero_()

    step_times = _cuda_event_time_ms(one_step, warmup=warmup_steps, repeats=repeats)

    median_ms = statistics.median(step_times)
    mean_ms   = statistics.mean(step_times)
    timing_method = "CUDA events (GPU-accurate)" if device == "cuda" else "wall-clock (CPU)"
    print(f"\n=======================================================")
    print(f" PyTorch Matrix Benchmark (same workload as Dark Forest)")
    print(f" Device:        {device}")
    print(f" Matrix:        {rows} x {cols} fp32 matmul + sum + backward")
    print(f" Timing:        {timing_method}")
    print(f" Median step:   {median_ms:.4f} ms")
    print(f" Mean step:     {mean_ms:.4f} ms")
    print(f" Min/Max:       {min(step_times):.4f}/{max(step_times):.4f} ms ({repeats} trials)")
    print(f"=======================================================\n")
    return {"rows": rows, "cols": cols, "median_ms": median_ms, "mean_ms": mean_ms,
            "min_ms": min(step_times), "max_ms": max(step_times)}


def benchmark_darkforest(steps=20, rows=256, cols=256):
    if darkforest is None:
        raise RuntimeError("darkforest is not installed in this Python environment")

    result = darkforest.benchmark_matrix_step(steps=steps, rows=rows, cols=cols)
    print(f"\n=======================================================")
    print(" Dark Forest Matrix Benchmark")
    print(f"=======================================================")
    print(f"Steps:          {int(result['steps'])}")
    print(f"Mean step:      {result['mean_ms']:.4f} ms")
    print(f"Median step:    {result['median_ms']:.4f} ms")
    print(f"Min/Max step:   {result['min_ms']:.4f}/{result['max_ms']:.4f} ms")
    print(f"Matrix shape:   {int(result['rows'])} x {int(result['cols'])}")
    print(f"=======================================================\n")
    return result


if __name__ == "__main__":
    print("Running full-model PyTorch baseline ...")
    pytorch_result = benchmark_pytorch()

    print("Running apples-to-apples matrix comparison (CUDA event timing) ...")
    pytorch_matrix = benchmark_pytorch_matrix(steps=20, rows=256, cols=256, repeats=20)
    darkforest_result = benchmark_darkforest(steps=20, rows=256, cols=256)

    print("Comparison summary:")
    print(f"  PyTorch matrix median step:     {pytorch_matrix['median_ms']:.4f} ms  (CUDA events)")
    print(f"  Dark Forest matrix median step: {darkforest_result['median_ms']:.4f} ms")
    print(f"  Full-model PyTorch baseline:    {pytorch_result['step_time_ms']:.2f} ms per step")
