"""
PyTorch GPT-2 training step benchmark — fair comparison with Dark Forest
Config: 12 layers, d_model=768, n_heads=12, d_ff=3072, vocab=50257, ctx=128, batch=1
Tests: eager mode AND torch.compile (PyTorch's best mode)
"""

import torch
import torch.nn as nn
import statistics
import sys

VOCAB    = 50257
CTX      = 128
D_MODEL  = 768
N_LAYERS = 12
N_HEADS  = 12
D_FF     = 3072
LR       = 3e-4
N_STEPS  = 50
WARMUP   = 5
DTYPE    = torch.float32

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device : {torch.cuda.get_device_name(0)}")
print(f"PyTorch: {torch.__version__}")
print(f"Config : vocab={VOCAB}, ctx={CTX}, d_model={D_MODEL}, n_layers={N_LAYERS}")

class GPT2Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1  = nn.LayerNorm(D_MODEL)
        self.attn = nn.MultiheadAttention(D_MODEL, N_HEADS, batch_first=True, dtype=DTYPE)
        self.ln2  = nn.LayerNorm(D_MODEL)
        self.mlp  = nn.Sequential(
            nn.Linear(D_MODEL, D_FF, dtype=DTYPE),
            nn.GELU(),
            nn.Linear(D_FF, D_MODEL, dtype=DTYPE),
        )
    def forward(self, x):
        T = x.size(1)
        mask = torch.triu(torch.ones(T, T, device=x.device, dtype=torch.bool), diagonal=1)
        a, _ = self.attn(self.ln1(x), self.ln1(x), self.ln1(x), attn_mask=mask, need_weights=False)
        x = x + a
        x = x + self.mlp(self.ln2(x))
        return x

class GPT2(nn.Module):
    def __init__(self):
        super().__init__()
        self.tok_emb = nn.Embedding(VOCAB, D_MODEL)
        self.pos_emb = nn.Embedding(CTX,   D_MODEL)
        self.blocks  = nn.ModuleList([GPT2Block() for _ in range(N_LAYERS)])
        self.ln_f    = nn.LayerNorm(D_MODEL)
        self.lm_head = nn.Linear(D_MODEL, VOCAB, bias=False, dtype=DTYPE)
    def forward(self, idx):
        B, T = idx.shape
        pos  = torch.arange(T, device=idx.device).unsqueeze(0)
        x    = self.tok_emb(idx) + self.pos_emb(pos)
        for block in self.blocks:
            x = block(x)
        return self.lm_head(self.ln_f(x))

def benchmark(model, label):
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR,
                                   betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()
    torch.manual_seed(42)
    tokens  = torch.randint(0, VOCAB, (CTX + 1,), device=device)
    inputs  = tokens[:CTX].unsqueeze(0)
    targets = tokens[1:CTX+1].unsqueeze(0)

    s = torch.cuda.Event(enable_timing=True)
    e = torch.cuda.Event(enable_timing=True)
    times = []
    torch.cuda.synchronize()

    for step in range(N_STEPS + WARMUP):
        s.record()
        optimizer.zero_grad(set_to_none=True)
        logits = model(inputs)
        loss   = criterion(logits.view(-1, VOCAB), targets.view(-1))
        loss.backward()
        optimizer.step()
        e.record()
        torch.cuda.synchronize()
        ms = s.elapsed_time(e)
        if step >= WARMUP:
            times.append(ms)
            if step == WARMUP or step == N_STEPS + WARMUP - 1:
                print(f"  [{label}] step {step-WARMUP+1:3d} | loss {loss.item():.4f} | {ms:.2f}ms")

    median = statistics.median(times)
    mean   = statistics.mean(times)
    print(f"  [{label}] Median: {median:.3f} ms | Mean: {mean:.3f} ms")
    return median

# -- Eager --
print("\n-- Eager mode ---------------------------------------------")
m_eager = GPT2().to(device)
med_eager = benchmark(m_eager, "eager")

# -- torch.compile --
print("\n-- torch.compile (reduce-overhead) ------------------------")
m_compile = GPT2().to(device)
try:
    m_compile = torch.compile(m_compile, mode="reduce-overhead")
    med_compile = benchmark(m_compile, "compile")
except Exception as ex:
    print(f"  torch.compile failed: {ex}")
    med_compile = None

# -- Summary --
print()
print("=" * 55)
print(" PyTorch GPT-2 Benchmark Summary")
print("=" * 55)
print(f" Eager median:     {med_eager:.3f} ms")
if med_compile:
    print(f" Compiled median:  {med_compile:.3f} ms")
print("=" * 55)
print(" Dark Forest target: see train_static output")
