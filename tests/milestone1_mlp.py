#!/usr/bin/env python3
"""Milestone 1 Gate Verification: MLP Training Parity against PyTorch.

Exit criteria:
1. Copy PyTorch's initial weights into Dark-Forest.
2. Train an MLP for 200 steps on identical batches.
3. Require loss to stay within 1e-4 of PyTorch's at EVERY step.
4. Test both SGD and AdamW optimizers.
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Ensure darkforest package is importable
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
df_py_path = os.path.join(repo_root, "darkforest-py", "python")
if df_py_path not in sys.path:
    sys.path.insert(0, df_py_path)

import darkforest as df

# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------

class PyTorchMLP(nn.Module):
    def __init__(self, in_features=8, hidden1=16, hidden2=8, out_features=4):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, out_features)

    def forward(self, x):
        h1 = F.relu(self.fc1(x))
        h2 = F.relu(self.fc2(h1))
        return self.fc3(h2)


class DarkForestLinear:
    def __init__(self, weight: df.Tensor, bias: df.Tensor):
        self.weight = weight
        self.bias = bias
        self.weight.requires_grad = True
        self.bias.requires_grad = True

    def forward(self, x: df.Tensor) -> df.Tensor:
        # W^T: [in_features, out_features]
        wt = self.weight.transpose(0, 1)
        # x @ W^T: [batch, out_features]
        xw = x.matmul(wt)
        # + bias
        return xw.add(self.bias)


class DarkForestMLP:
    def __init__(self, fc1: DarkForestLinear, fc2: DarkForestLinear, fc3: DarkForestLinear):
        self.fc1 = fc1
        self.fc2 = fc2
        self.fc3 = fc3

    def forward(self, x: df.Tensor) -> df.Tensor:
        h1 = self.fc1.forward(x).relu()
        h2 = self.fc2.forward(h1).relu()
        return self.fc3.forward(h2)

    def parameters(self):
        return [
            self.fc1.weight, self.fc1.bias,
            self.fc2.weight, self.fc2.bias,
            self.fc3.weight, self.fc3.bias,
        ]


def build_models(seed=42):
    torch.manual_seed(seed)
    np.random.seed(seed)

    pt_mlp = PyTorchMLP(in_features=8, hidden1=16, hidden2=8, out_features=4)

    # Copy weights to Dark-Forest
    def copy_param(pt_tensor):
        data = pt_tensor.detach().cpu().numpy().flatten().tolist()
        shape = list(pt_tensor.shape)
        return df.Tensor.from_list(data, shape)

    fc1 = DarkForestLinear(copy_param(pt_mlp.fc1.weight), copy_param(pt_mlp.fc1.bias))
    fc2 = DarkForestLinear(copy_param(pt_mlp.fc2.weight), copy_param(pt_mlp.fc2.bias))
    fc3 = DarkForestLinear(copy_param(pt_mlp.fc3.weight), copy_param(pt_mlp.fc3.bias))
    df_mlp = DarkForestMLP(fc1, fc2, fc3)

    return pt_mlp, df_mlp


# ---------------------------------------------------------------------------
# Training Loop Test
# ---------------------------------------------------------------------------

def test_mlp_training(optimizer_name="sgd", steps=200, lr=0.02, tolerance=1e-4):
    print(f"\n=======================================================")
    print(f"Running Milestone 1 MLP Gate: {optimizer_name.upper()} (steps={steps}, lr={lr}, tol={tolerance})")
    print(f"=======================================================")

    pt_mlp, df_mlp = build_models(seed=1337)

    if optimizer_name == "sgd":
        pt_opt = torch.optim.SGD(pt_mlp.parameters(), lr=lr, momentum=0.0, weight_decay=0.0)
        df_opt = df.SGD(lr=lr, momentum=0.0, weight_decay=0.0)
    elif optimizer_name == "adamw":
        pt_opt = torch.optim.AdamW(pt_mlp.parameters(), lr=lr, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.01)
        df_opt = df.AdamW(lr=lr, beta1=0.9, beta2=0.999, eps=1e-8, weight_decay=0.01)
    else:
        raise ValueError(f"Unknown optimizer {optimizer_name}")

    batch_size = 16
    in_features = 8
    num_classes = 4

    rng = np.random.RandomState(42)
    max_diff = 0.0
    sum_diff = 0.0

    for step in range(steps):
        # Generate identical batch
        x_data = rng.randn(batch_size, in_features).astype(np.float32)
        y_data = rng.randint(0, num_classes, size=(batch_size,), dtype=np.int64)

        # PyTorch step
        pt_opt.zero_grad()
        pt_x = torch.from_numpy(x_data)
        pt_y = torch.from_numpy(y_data)
        pt_logits = pt_mlp(pt_x)
        pt_loss = F.cross_entropy(pt_logits, pt_y)
        pt_loss.backward()
        pt_opt.step()
        pt_val = pt_loss.item()

        # Dark-Forest step
        df_opt.zero_grad(df_mlp.parameters())
        df_x = df.Tensor.from_list(x_data.flatten().tolist(), [batch_size, in_features])
        df_y = df.Tensor.from_list_i64(y_data.tolist(), [batch_size])
        df_logits = df_mlp.forward(df_x)
        df_loss = df_logits.cross_entropy(df_y)
        df_loss.backward()
        df_opt.step(df_mlp.parameters())
        df_val = df_loss.to_list()[0]

        diff = abs(df_val - pt_val)
        max_diff = max(max_diff, diff)
        sum_diff += diff

        if step % 25 == 0 or step == steps - 1 or diff > tolerance:
            print(f"  Step {step:3d} | PyTorch: {pt_val:.6f} | DarkForest: {df_val:.6f} | Diff: {diff:.2e}")

        assert diff <= tolerance, (
            f"Gate Failure at step {step}: |DarkForest ({df_val:.6f}) - PyTorch ({pt_val:.6f})| = {diff:.2e} > {tolerance}"
        )

    mean_diff = sum_diff / steps
    print(f"\n>> {optimizer_name.upper()} PASSED {steps} STEPS! Max Diff = {max_diff:.2e}, Mean Diff = {mean_diff:.2e}")
    return max_diff, mean_diff


if __name__ == "__main__":
    # Test 1: SGD 200 steps
    max_sgd, mean_sgd = test_mlp_training("sgd", steps=200, lr=0.02, tolerance=1e-4)

    # Test 2: AdamW 200 steps
    max_adamw, mean_adamw = test_mlp_training("adamw", steps=200, lr=0.005, tolerance=1e-4)

    print("\n" + "=" * 60)
    print("ALL MILESTONE 1 GATES PASSED! Dark-Forest loss matches PyTorch within 1e-4 across all 200 steps!")
    print("=" * 60)
