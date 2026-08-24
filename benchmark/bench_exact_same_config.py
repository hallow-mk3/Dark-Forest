import torch
import torch.nn as nn
import statistics
import time

VOCAB    = 65
CTX      = 128
D_MODEL  = 128
N_LAYERS = 4
N_HEADS  = 1
D_FF     = 512
LR       = 3e-4
N_STEPS  = 100
WARMUP   = 5
DTYPE    = torch.float32

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device : {torch.cuda.get_device_name(0)}")
print(f"PyTorch: {torch.__version__}")
print(f"Exact apples-to-apples configuration:")
print(f"vocab={VOCAB}, ctx={CTX}, d_model={D_MODEL}, n_layers={N_LAYERS}, n_heads={N_HEADS}, d_ff={D_FF}")

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

model = GPT2().to(device)
optimizer = torch.optim.AdamW(model.parameters(), lr=LR, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01)
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

median = statistics.median(times)
mean   = statistics.mean(times)
print(f"\n[Apples-to-Apples PyTorch Eager Result]")
print(f"Median: {median:.3f} ms | Mean: {mean:.3f} ms | Tok/s: {(CTX / (median/1000.0)):.0f}")
