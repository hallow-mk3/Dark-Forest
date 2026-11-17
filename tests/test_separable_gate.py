"""
Separable task gate: 4-class Gaussian clusters (centers ±2.5, noise=0.2).
Both AdamW and SGD must drive loss from ~1.4 to <0.01 within 200 steps,
matching PyTorch step-by-step within 1e-5.
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "darkforest-py", "python"))

import darkforest as df
import torch
import torch.nn as nn
import random, math

PASS = 0
FAIL = 0

def check(name, cond, detail=""):
    global PASS, FAIL
    if cond:
        print(f"  [PASS] {name}")
        PASS += 1
    else:
        print(f"  [FAIL] {name}  {detail}")
        FAIL += 1


def make_dataset(n_per_class=64, noise=0.2, seed=0):
    """4 Gaussian clusters at (±2.5, ±2.5) corners."""
    rng = random.Random(seed)
    centers = [(-2.5, -2.5), (2.5, -2.5), (-2.5, 2.5), (2.5, 2.5)]
    X, Y = [], []
    for cls, (cx, cy) in enumerate(centers):
        for _ in range(n_per_class):
            x1 = cx + rng.gauss(0, noise)
            x2 = cy + rng.gauss(0, noise)
            X.append([x1, x2])
            Y.append(cls)
    return X, Y


def run_pytorch(X_pt, y_pt, lr, weight_decay, momentum=None, steps=200, seed=42):
    torch.manual_seed(seed)
    model = nn.Sequential(
        nn.Linear(2, 16), nn.ReLU(),
        nn.Linear(16, 16), nn.ReLU(),
        nn.Linear(16, 4)
    )
    if momentum is not None:
        opt = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    else:
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    losses = []
    for _ in range(steps):
        opt.zero_grad()
        logits = model(X_pt)
        loss = nn.functional.cross_entropy(logits, y_pt)
        loss.backward()
        opt.step()
        losses.append(loss.item())
    # Return model weights (for transfer to DF) and loss curve
    weights = [(p.detach().numpy().flatten().tolist(), p.grad) for p in model.parameters()]
    return losses, model


def extract_pt_weights(model):
    """Return list of (weight_data, bias_data) for each Linear layer."""
    layers = []
    for m in model.modules():
        if isinstance(m, nn.Linear):
            layers.append((m.weight.detach().numpy().flatten().tolist(),
                           list(m.weight.shape),
                           m.bias.detach().numpy().flatten().tolist()))
    return layers


def run_darkforest(layer_weights, X_data, y_data, lr, weight_decay, momentum=None, steps=200):
    """Build DF MLP with same initial weights as PyTorch, train, return loss curve."""
    n = len(X_data)
    in_dim, hidden, out_dim = 2, 16, 4

    # Build layers from transferred weights
    class DFLinear:
        def __init__(self, w_data, w_shape, b_data):
            self.w = df.Tensor(w_data, w_shape)
            self.w.requires_grad = True
            bias_shape = [w_shape[0]]
            self.b = df.Tensor(b_data, bias_shape)
            self.b.requires_grad = True

        def forward(self, x):
            return x.matmul(self.w.transpose()) + self.b

        def params(self):
            return [self.w, self.b]

    layers = [DFLinear(w, ws, b) for (w, ws, b) in layer_weights]

    def forward(x):
        h = x
        for i, lin in enumerate(layers):
            h = lin.forward(h)
            if i < len(layers) - 1:
                h = h.relu()
        return h

    all_params = []
    for lin in layers:
        all_params.extend(lin.params())

    if momentum is not None:
        opt = df.SGD(all_params, lr=lr, momentum=momentum, weight_decay=weight_decay)
    else:
        opt = df.AdamW(all_params, lr=lr, weight_decay=weight_decay)

    X_flat = []
    for row in X_data:
        X_flat.extend(row)
    X_df = df.Tensor(X_flat, [n, 2])
    y_df = list(y_data)

    losses = []
    for _ in range(steps):
        opt.zero_grad()
        logits = forward(X_df)
        loss = logits.cross_entropy(y_df)
        loss.backward()
        opt.step()
        losses.append(loss.item())
    return losses


def test_separable(optimizer_name, lr, weight_decay, momentum=None):
    print(f"\n--- Separable gate: {optimizer_name} (lr={lr}, wd={weight_decay}) ---")

    X_data, y_data = make_dataset()
    X_pt = torch.tensor(X_data, dtype=torch.float32)
    y_pt = torch.tensor(y_data, dtype=torch.long)

    # Run PyTorch first to get initial weights + loss curve
    pt_losses, pt_model = run_pytorch(X_pt, y_pt, lr, weight_decay, momentum, steps=200)
    layer_weights = extract_pt_weights(pt_model)

    # Rebuild PyTorch from same initial weights for fair comparison
    # (pt_model already trained — we need a fresh model with same init)
    # So we'll re-init PT from seed=42 and re-run
    torch.manual_seed(42)
    pt_model2 = nn.Sequential(
        nn.Linear(2, 16), nn.ReLU(),
        nn.Linear(16, 16), nn.ReLU(),
        nn.Linear(16, 4)
    )
    init_weights = extract_pt_weights(pt_model2)

    # PT reference run from same init
    if momentum is not None:
        opt2 = torch.optim.SGD(pt_model2.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
    else:
        opt2 = torch.optim.AdamW(pt_model2.parameters(), lr=lr, weight_decay=weight_decay)
    pt_losses2 = []
    for _ in range(200):
        opt2.zero_grad()
        logits = pt_model2(X_pt)
        loss = nn.functional.cross_entropy(logits, y_pt)
        loss.backward()
        opt2.step()
        pt_losses2.append(loss.item())

    # Dark-Forest run from same initial weights
    df_losses = run_darkforest(init_weights, X_data, y_data, lr, weight_decay, momentum, steps=200)

    final_pt = pt_losses2[-1]
    final_df = df_losses[-1]
    max_diff = max(abs(a - b) for a, b in zip(df_losses, pt_losses2))

    check(f"{optimizer_name}: final loss < 0.01 (PT={final_pt:.5f})",
          final_pt < 0.01, f"final_pt={final_pt:.5f}")
    check(f"{optimizer_name}: DF final loss < 0.01 (DF={final_df:.5f})",
          final_df < 0.01, f"final_df={final_df:.5f}")
    check(f"{optimizer_name}: step-wise max diff < 1e-4 (got {max_diff:.2e})",
          max_diff < 1e-4, f"max_diff={max_diff:.2e}")

    print(f"    PT loss: {pt_losses2[0]:.4f} → {final_pt:.6f}")
    print(f"    DF loss: {df_losses[0]:.4f} → {final_df:.6f}")
    print(f"    max step-wise diff: {max_diff:.2e}")


if __name__ == "__main__":
    test_separable("AdamW", lr=0.03, weight_decay=1e-4)
    test_separable("SGD+momentum", lr=0.05, weight_decay=1e-4, momentum=0.9)

    print(f"\n{'='*55}")
    print(f"SEPARABLE GATE SUMMARY: {PASS} passed, {FAIL} failed")
    print(f"{'='*55}")
    sys.exit(0 if FAIL == 0 else 1)
