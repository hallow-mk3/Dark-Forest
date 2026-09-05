"""Apples-to-Apples Transformer Architecture Benchmark for Project IRIS.

This benchmark provides rigorous baseline comparisons across two distinct model tiers:
1. Full GPT-2 (124M Parameters):
   - Layers: 12, d_model: 768, heads: 12, d_ff: 3072, vocab: 50257, context: 128, batch: 1
   - Used for Research Paper Experiment 2 (Full-Scale Step Latency & Variance).

2. Small Transformer (Experiment 2 Micro-Tier):
   - Layers: 4, d_model: 128, heads: 1, d_ff: 512, vocab: 65, context: 128, batch: 1
   - Used for character-level rapid prototyping.

Timing Method: High-precision CUDA events (torch.cuda.Event) with GPU synchronization.
"""

from __future__ import annotations

import argparse
import statistics
import time
import torch
import torch.nn as nn


class GPT2Block(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int):
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, batch_first=True)
        self.ln2 = nn.LayerNorm(d_model)
        self.mlp = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        t = x.size(1)
        mask = torch.triu(torch.ones(t, t, device=x.device, dtype=torch.bool), diagonal=1)
        a, _ = self.attn(self.ln1(x), self.ln1(x), self.ln1(x), attn_mask=mask, need_weights=False)
        x = x + a
        return x + self.mlp(self.ln2(x))


class GPT2(nn.Module):
    def __init__(self, vocab_size: int, ctx_len: int, d_model: int, n_layers: int, n_heads: int, d_ff: int):
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(ctx_len, d_model)
        self.blocks = nn.ModuleList([GPT2Block(d_model, n_heads, d_ff) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias=False)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        b, t = idx.shape
        pos = torch.arange(t, device=idx.device).unsqueeze(0)
        x = self.tok_emb(idx) + self.pos_emb(pos)
        for block in self.blocks:
            x = block(x)
        return self.lm_head(self.ln_f(x))


def run_benchmark(
    tier: str,
    vocab_size: int,
    ctx_len: int,
    d_model: int,
    n_layers: int,
    n_heads: int,
    d_ff: int,
    batch_size: int = 1,
    steps: int = 50,
    warmup: int = 10,
    repeats: int = 7,
):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        torch.cuda.set_per_process_memory_fraction(0.85, 0)

    print(f"================================================================================")
    print(f" Benchmark Tier: {tier}")
    print(f" Device        : {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print(f" Config        : vocab={vocab_size}, ctx={ctx_len}, d_model={d_model}, layers={n_layers}, heads={n_heads}, d_ff={d_ff}, batch={batch_size}")
    print(f"================================================================================")

    model = GPT2(vocab_size, ctx_len, d_model, n_layers, n_heads, d_ff).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()

    torch.manual_seed(42)
    inputs = torch.randint(0, vocab_size, (batch_size, ctx_len), device=device)
    targets = torch.randint(0, vocab_size, (batch_size, ctx_len), device=device)

    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)

    # Global warmup
    for _ in range(warmup):
        optimizer.zero_grad(set_to_none=True)
        logits = model(inputs)
        loss = criterion(logits.view(-1, vocab_size), targets.view(-1))
        loss.backward()
        optimizer.step()
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    trial_medians = []
    trial_means = []

    for trial in range(1, repeats + 1):
        times = []
        for _ in range(steps):
            s.record()
            optimizer.zero_grad(set_to_none=True)
            logits = model(inputs)
            loss = criterion(logits.view(-1, vocab_size), targets.view(-1))
            loss.backward()
            optimizer.step()
            e.record()
            torch.cuda.synchronize()
            times.append(s.elapsed_time(e))

        med = statistics.median(times)
        mean = statistics.mean(times)
        trial_medians.append(med)
        trial_means.append(mean)
        print(f"  Trial {trial}/{repeats}: Median = {med:.3f} ms | Mean = {mean:.3f} ms")

    overall_median = statistics.median(trial_medians)
    overall_mean = statistics.mean(trial_medians)
    overall_std = statistics.stdev(trial_medians) if len(trial_medians) > 1 else 0.0

    print(f"\n--- Summary ({repeats} Repeated Trials) ---")
    print(f"Median of Medians: {overall_median:.3f} ms")
    print(f"Mean of Medians  : {overall_mean:.3f} ms")
    print(f"Std Dev (sigma)  : {overall_std:.3f} ms")
    print(f"Range            : [{min(trial_medians):.3f}, {max(trial_medians):.3f}] ms")
    print(f"Throughput       : {(batch_size * ctx_len / (overall_median / 1000.0)):.1f} tok/s\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Apples-to-Apples Transformer Benchmark")
    parser.add_argument(
        "--config",
        choices=["full", "small", "both"],
        default="both",
        help="'full' = 12-layer GPT-2 (Experiment 2); 'small' = 4-layer tiny; 'both' = runs both",
    )
    args = parser.parse_args()

    if args.config in ("small", "both"):
        run_benchmark(
            tier="Small Micro-Tier (4-Layer, d=128, H=1, Vocab=65)",
            vocab_size=65,
            ctx_len=128,
            d_model=128,
            n_layers=4,
            n_heads=1,
            d_ff=512,
            batch_size=1,
            steps=30,
            warmup=5,
            repeats=5,
        )

    if args.config in ("full", "both"):
        run_benchmark(
            tier="Full GPT-2 Architecture (12-Layer, d=768, H=12, Vocab=50257)",
            vocab_size=50257,
            ctx_len=128,
            d_model=768,
            n_layers=12,
            n_heads=12,
            d_ff=3072,
            batch_size=1,
            steps=20,
            warmup=5,
            repeats=7,
        )
