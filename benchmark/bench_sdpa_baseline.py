"""
PyTorch SDPA + bf16 + gradient-checkpointing baseline for Dark Forest comparison.

This is the STRONG baseline the reviewer demands:
  - scaled_dot_product_attention (uses Flash/fused kernels under the hood on Ampere+)
  - bfloat16 precision (native Blackwell support)
  - gradient checkpointing (activation recompute, extends OOM horizon significantly)
  - Identical config: 12 layers, d_model=768, 12 heads, d_ff=3072, vocab=50257

Usage:
    python benchmark/bench_sdpa_baseline.py

Outputs:
  - Per-sequence VRAM and latency table (forward only & fwd+bwd)
  - Max sequence that fits under the 85% VRAM cap (6.77 GB on 8 GB)
  - JSON results saved to benchmark/sdpa_baseline_results.json
"""

from __future__ import annotations

import gc
import json
import math
import os
import sys
import time
from typing import Any, Dict, List, Optional

try:
    import torch
    import torch.nn as nn
    import torch.nn.functional as F
    from torch.utils.checkpoint import checkpoint as grad_checkpoint
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False
    print("ERROR: PyTorch not found. Install with: pip install torch --index-url https://download.pytorch.org/whl/cu128")
    sys.exit(1)

# ---------------------------------------------------------------------------
# Minimal GPT-2–scale transformer using SDPA (the strong baseline)
# ---------------------------------------------------------------------------

class SDPASelfAttention(nn.Module):
    """Multi-head self-attention using torch.nn.functional.scaled_dot_product_attention.

    On Ampere/Ada/Blackwell with PyTorch >= 2.0, SDPA dispatches to Flash Attention
    kernels automatically — it is NOT naive O(S^2) eager attention.
    """

    def __init__(self, d_model: int, n_heads: int) -> None:
        super().__init__()
        assert d_model % n_heads == 0
        self.n_heads = n_heads
        self.d_head = d_model // n_heads
        self.qkv = nn.Linear(d_model, 3 * d_model, bias=False)
        self.proj = nn.Linear(d_model, d_model, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, S, D = x.shape
        qkv = self.qkv(x)                                               # [B, S, 3D]
        q, k, v = qkv.chunk(3, dim=-1)                                  # [B, S, D] each

        # Reshape to [B, H, S, d_head] for SDPA
        def reshape(t: torch.Tensor) -> torch.Tensor:
            return t.view(B, S, self.n_heads, self.d_head).transpose(1, 2)

        q, k, v = reshape(q), reshape(k), reshape(v)

        # Flash/fused path: is_causal=True avoids materialising the mask
        out = F.scaled_dot_product_attention(q, k, v, is_causal=True)   # [B, H, S, d_head]
        out = out.transpose(1, 2).contiguous().view(B, S, D)
        return self.proj(out)


class TransformerBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, d_ff: int) -> None:
        super().__init__()
        self.ln1 = nn.LayerNorm(d_model)
        self.attn = SDPASelfAttention(d_model, n_heads)
        self.ln2 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_ff, bias=False),
            nn.GELU(),
            nn.Linear(d_ff, d_model, bias=False),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.attn(self.ln1(x))
        x = x + self.ff(self.ln2(x))
        return x


class SDPATransformer(nn.Module):
    """GPT-2–scale (124 M) transformer using SDPA + optional gradient checkpointing."""

    def __init__(
        self,
        vocab_size: int = 50257,
        d_model: int = 768,
        n_heads: int = 12,
        n_layers: int = 12,
        d_ff: int = 3072,
        max_pos: int = 2048,         # RoPE would remove this limit; here we use learned embeddings capped at max_pos
        use_checkpointing: bool = False,
    ) -> None:
        super().__init__()
        self.tok_emb = nn.Embedding(vocab_size, d_model)
        self.pos_emb = nn.Embedding(max_pos, d_model)
        self.blocks = nn.ModuleList(
            [TransformerBlock(d_model, n_heads, d_ff) for _ in range(n_layers)]
        )
        self.ln_f = nn.LayerNorm(d_model)
        self.head = nn.Linear(d_model, vocab_size, bias=False)
        self.use_checkpointing = use_checkpointing

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        B, S = idx.shape
        pos = torch.arange(S, device=idx.device, dtype=torch.long)
        x = self.tok_emb(idx) + self.pos_emb(pos)

        for block in self.blocks:
            if self.use_checkpointing and self.training:
                x = grad_checkpoint(block, x, use_reentrant=False)
            else:
                x = block(x)

        x = self.ln_f(x)
        return self.head(x)


# ---------------------------------------------------------------------------
# Measurement helpers
# ---------------------------------------------------------------------------

def _cuda_reset() -> None:
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        gc.collect()
        torch.cuda.reset_peak_memory_stats()


def _peak_vram_mb() -> float:
    return torch.cuda.max_memory_allocated() / (1024 ** 2)


def measure_sequence(
    model: SDPATransformer,
    optimizer: torch.optim.Optimizer,
    seq_len: int,
    batch_size: int,
    vocab_size: int,
    dtype: torch.dtype,
    warmup: int = 3,
    repeats: int = 10,
    device: str = "cuda",
    chunked_ce: bool = False,
    chunk_size: int = 1024,
) -> Dict[str, Any]:
    """Return timing and peak VRAM for one (seq_len, model) combination.

    Returns dict with status="OK" on success or status="OOM (reason)" on failure.
    """
    model.train()

    def one_step() -> torch.Tensor:
        optimizer.zero_grad(set_to_none=True)
        idx = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
        tgt = torch.randint(0, vocab_size, (batch_size, seq_len), device=device)
        with torch.autocast(device_type="cuda", dtype=dtype):
            if chunked_ce:
                pos = torch.arange(seq_len, device=device, dtype=torch.long)
                x = model.tok_emb(idx) + model.pos_emb(pos)
                for block in model.blocks:
                    if model.use_checkpointing and model.training:
                        x = grad_checkpoint(block, x, use_reentrant=False)
                    else:
                        x = block(x)
                x = model.ln_f(x)

                total_loss = torch.tensor(0.0, device=device, dtype=torch.float32)
                for i in range(0, seq_len, chunk_size):
                    end = min(i + chunk_size, seq_len)
                    x_chunk = x[:, i:end, :]
                    logits_chunk = model.head(x_chunk)
                    tgt_chunk = tgt[:, i:end]
                    loss_chunk = F.cross_entropy(
                        logits_chunk.view(-1, vocab_size), tgt_chunk.view(-1), reduction="sum"
                    )
                    total_loss = total_loss + loss_chunk
                loss = total_loss / seq_len
            else:
                logits = model(idx)                            # [B, S, V]
                loss = F.cross_entropy(
                    logits.view(-1, vocab_size), tgt.view(-1)
                )
        loss.backward()
        optimizer.step()
        return loss

    def classify_oom(err: Exception) -> str:
        err_msg = str(err)
        if "allowed" in err_msg:
            return "OOM (>6.74 GB, 85% safety cap)"
        total_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        return f"OOM (>{total_gb:.2f} GB, GPU capacity)"

    # Warmup
    try:
        for _ in range(warmup):
            one_step()
        torch.cuda.synchronize()
    except torch.cuda.OutOfMemoryError as e:
        return {"seq_len": seq_len, "status": classify_oom(e), "peak_vram_mb": None}

    _cuda_reset()

    latencies: List[float] = []
    try:
        for _ in range(repeats):
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            one_step()
            end.record()
            torch.cuda.synchronize()
            latencies.append(start.elapsed_time(end))
    except torch.cuda.OutOfMemoryError as e:
        return {"seq_len": seq_len, "status": classify_oom(e), "peak_vram_mb": None}

    latencies.sort()
    return {
        "seq_len": seq_len,
        "status": "OK",
        "median_ms": round(latencies[len(latencies) // 2], 3),
        "min_ms": round(latencies[0], 3),
        "max_ms": round(latencies[-1], 3),
        "peak_vram_mb": round(_peak_vram_mb(), 1),
        "repeats": repeats,
    }


# ---------------------------------------------------------------------------
# Main sweep
# ---------------------------------------------------------------------------

def run_sdpa_sweep(
    seq_lengths: List[int],
    batch_size: int = 1,
    dtype_name: str = "bf16",
    use_checkpointing: bool = True,
    chunked_ce: bool = False,
    device: str = "cuda",
) -> Dict[str, Any]:

    dtype = torch.bfloat16 if dtype_name == "bf16" else torch.float32
    vocab_size, d_model, n_heads, n_layers, d_ff = 50257, 768, 12, 12, 3072
    max_pos = max(seq_lengths) + 1

    ce_label = " (+ chunked CE)" if chunked_ce else ""
    print("=" * 80)
    print(f"  PyTorch SDPA Baseline  |  dtype={dtype_name}  |  ckpt={use_checkpointing}{ce_label}")
    print(f"  Config: {n_layers}L d{d_model} H{n_heads} ff{d_ff} vocab{vocab_size} batch{batch_size}")
    print(f"  Hardware: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print("=" * 80)

    model = SDPATransformer(
        vocab_size=vocab_size,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        d_ff=d_ff,
        max_pos=max_pos,
        use_checkpointing=use_checkpointing,
    ).to(device=device, dtype=dtype)

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

    param_mb = sum(p.numel() * p.element_size() for p in model.parameters()) / (1024 ** 2)
    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")
    print(f"Parameter memory: {param_mb:.1f} MB ({dtype_name})")
    print()

    results = []
    max_survived = 0

    for s in seq_lengths:
        print(f"  S = {s:>8,} ... ", end="", flush=True)
        _cuda_reset()
        res = measure_sequence(
            model=model,
            optimizer=optimizer,
            seq_len=s,
            batch_size=batch_size,
            vocab_size=vocab_size,
            dtype=dtype,
            device=device,
            chunked_ce=chunked_ce,
        )
        if res.get("status") != "OK":
            status = res["status"]
            print(f"{status}")
            results.append({"seq_len": s, "status": status, "peak_vram_mb": None})
            break
        else:
            print(f"  {res['median_ms']:>8.1f} ms  |  {res['peak_vram_mb']:>7.1f} MB peak VRAM")
            results.append(res)
            max_survived = s

    print(f"\n  Max sequence that survived: {max_survived:,} tokens")
    print()

    return {
        "metadata": {
            "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU",
            "dtype": dtype_name,
            "checkpointing": use_checkpointing,
            "chunked_ce": chunked_ce,
            "batch_size": batch_size,
            "config": {
                "n_layers": n_layers, "d_model": d_model, "n_heads": n_heads,
                "d_ff": d_ff, "vocab_size": vocab_size,
            },
            "sdpa_backend": "torch.nn.functional.scaled_dot_product_attention (Flash/fused on Ampere+)",
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        },
        "results": results,
        "max_survived_seq_len": max_survived,
    }


def print_markdown_table(data: Dict[str, Any]) -> None:
    meta = data["metadata"]
    ce_str = " + chunked CE" if meta.get("chunked_ce") else ""
    print(f"\n### PyTorch SDPA Baseline (dtype={meta['dtype']}, checkpointing={meta['checkpointing']}{ce_str})")
    print(f"**Hardware**: {meta['device']} | **Batch**: {meta['batch_size']}")
    print()
    print("| Seq Length (S) | PyTorch SDPA Latency (median) | Peak VRAM | Status |")
    print("| :--- | :---: | :---: | :---: |")
    for r in data["results"]:
        s = f"{r['seq_len']:,}"
        if r.get("status") != "OK":
            print(f"| **{s}** | — | — | **{r.get('status')}** |")
        else:
            print(f"| **{s}** | `{r['median_ms']:.1f} ms` | `{r['peak_vram_mb']:.0f} MB` | ✅ |")
    print(f"\n**Max survived**: {data['max_survived_seq_len']:,} tokens")


if __name__ == "__main__":
    if not HAS_TORCH or not torch.cuda.is_available():
        print("This benchmark requires PyTorch with CUDA. Exiting.")
        sys.exit(1)

    # Safety cap: 85% of VRAM
    torch.cuda.set_per_process_memory_fraction(0.85, 0)
    cap_gb = torch.cuda.get_device_properties(0).total_memory * 0.85 / (1024**3)
    print(f"VRAM safety cap: 85% = {cap_gb:.2f} GB\n")

    seq_lengths = [
        128, 256, 512, 1024, 2048, 4096, 8192,
        16_384, 32_768, 65_536,
    ]

    # 1. SDPA + bf16 + checkpointing (unchunked CE)
    results_ckpt = run_sdpa_sweep(
        seq_lengths=seq_lengths,
        batch_size=1,
        dtype_name="bf16",
        use_checkpointing=True,
        chunked_ce=False,
    )
    print_markdown_table(results_ckpt)

    # 2. SDPA + bf16 + checkpointing (+ chunked CE)
    results_chunked_ce = run_sdpa_sweep(
        seq_lengths=seq_lengths,
        batch_size=1,
        dtype_name="bf16",
        use_checkpointing=True,
        chunked_ce=True,
    )
    print_markdown_table(results_chunked_ce)

    # 3. Without checkpointing
    results_no_ckpt = run_sdpa_sweep(
        seq_lengths=seq_lengths,
        batch_size=1,
        dtype_name="bf16",
        use_checkpointing=False,
        chunked_ce=False,
    )
    print_markdown_table(results_no_ckpt)

    out_path = os.path.join(os.path.dirname(__file__), "sdpa_baseline_results.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "with_checkpointing": results_ckpt,
                "with_chunked_ce": results_chunked_ce,
                "without_checkpointing": results_no_ckpt,
            },
            f, indent=2,
        )
    print(f"\nResults saved to {out_path}")

