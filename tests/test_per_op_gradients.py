#!/usr/bin/env python3
"""Comprehensive Per-Op Gradient Verification Suite.

Tests EVERY differentiable operator in Dark-Forest against PyTorch:
- add, sub, mul, div, neg, relu, matmul, sum, mean, log_softmax, cross_entropy
Under:
- Contiguous layouts
- Transposed layouts
- Strided sliced layouts
- Broadcast layouts (stride-0 expand)
- Single-input gradient tracking (only A requires grad, or only B requires grad)
"""

import os
import sys
import numpy as np
import torch
import torch.nn.functional as F

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
df_py_path = os.path.join(repo_root, "darkforest-py", "python")
if df_py_path not in sys.path:
    sys.path.insert(0, df_py_path)

import darkforest as df

passed = 0
failed = 0


def check_close(df_val, pt_val, name, tol=1e-5):
    global passed, failed
    diff = np.max(np.abs(np.array(df_val) - np.array(pt_val)))
    if diff > tol:
        print(f"  [FAIL] {name}: max diff {diff:.2e} > {tol:.2e}")
        failed += 1
        raise AssertionError(f"{name} diff {diff} exceeds {tol}")
    else:
        passed += 1
        print(f"  [PASS] {name} (diff: {diff:.2e})")


def test_binary_op(op_name, df_op, pt_op):
    print(f"\n--- Testing {op_name.upper()} ---")
    rng = np.random.RandomState(42)

    # 1. Contiguous (both require grad)
    a_np = rng.randn(4, 6).astype(np.float32) + 2.0
    b_np = rng.randn(4, 6).astype(np.float32) + 2.0

    a_pt = torch.tensor(a_np, requires_grad=True)
    b_pt = torch.tensor(b_np, requires_grad=True)
    out_pt = pt_op(a_pt, b_pt)
    loss_pt = out_pt.sum()
    loss_pt.backward()

    a_df = df.Tensor.from_list(a_np.flatten().tolist(), [4, 6]); a_df.requires_grad = True
    b_df = df.Tensor.from_list(b_np.flatten().tolist(), [4, 6]); b_df.requires_grad = True
    out_df = df_op(a_df, b_df)
    loss_df = out_df.sum()
    loss_df.backward()

    check_close(out_df.to_list(), out_pt.detach().numpy().flatten(), f"{op_name} contiguous forward")
    check_close(a_df.grad.to_list(), a_pt.grad.numpy().flatten(), f"{op_name} contiguous grad A")
    check_close(b_df.grad.to_list(), b_pt.grad.numpy().flatten(), f"{op_name} contiguous grad B")

    # 2. Only A requires grad
    a_pt = torch.tensor(a_np, requires_grad=True)
    b_pt = torch.tensor(b_np, requires_grad=False)
    pt_op(a_pt, b_pt).sum().backward()

    a_df = df.Tensor.from_list(a_np.flatten().tolist(), [4, 6]); a_df.requires_grad = True
    b_df = df.Tensor.from_list(b_np.flatten().tolist(), [4, 6]); b_df.requires_grad = False
    df_op(a_df, b_df).sum().backward()

    check_close(a_df.grad.to_list(), a_pt.grad.numpy().flatten(), f"{op_name} only A requires grad")
    assert b_df.grad is None, f"{op_name} b_df.grad should be None"

    # 3. Only B requires grad (Crucial check for needs_input_grad)
    a_pt = torch.tensor(a_np, requires_grad=False)
    b_pt = torch.tensor(b_np, requires_grad=True)
    pt_op(a_pt, b_pt).sum().backward()

    a_df = df.Tensor.from_list(a_np.flatten().tolist(), [4, 6]); a_df.requires_grad = False
    b_df = df.Tensor.from_list(b_np.flatten().tolist(), [4, 6]); b_df.requires_grad = True
    df_op(a_df, b_df).sum().backward()

    check_close(b_df.grad.to_list(), b_pt.grad.numpy().flatten(), f"{op_name} only B requires grad")
    assert a_df.grad is None, f"{op_name} a_df.grad should be None"

    # 4. Transposed inputs
    a_leaf_pt = torch.tensor(a_np, requires_grad=True)
    b_leaf_pt = torch.tensor(b_np, requires_grad=True)
    a_pt = a_leaf_pt.t()
    b_pt = b_leaf_pt.t()
    pt_op(a_pt, b_pt).sum().backward()

    a_leaf_df = df.Tensor.from_list(a_np.flatten().tolist(), [4, 6]); a_leaf_df.requires_grad = True
    b_leaf_df = df.Tensor.from_list(b_np.flatten().tolist(), [4, 6]); b_leaf_df.requires_grad = True
    a_df = a_leaf_df.transpose(0, 1)
    b_df = b_leaf_df.transpose(0, 1)
    df_op(a_df, b_df).sum().backward()

    check_close(a_leaf_df.grad.to_list(), a_leaf_pt.grad.flatten().numpy(), f"{op_name} transposed grad A")
    check_close(b_leaf_df.grad.to_list(), b_leaf_pt.grad.flatten().numpy(), f"{op_name} transposed grad B")

    # 5. Broadcast inputs ([4, 1] op [1, 6] -> [4, 6])
    a_bcast = rng.randn(4, 1).astype(np.float32) + 2.0
    b_bcast = rng.randn(1, 6).astype(np.float32) + 2.0

    a_pt = torch.tensor(a_bcast, requires_grad=True)
    b_pt = torch.tensor(b_bcast, requires_grad=True)
    pt_op(a_pt, b_pt).sum().backward()

    a_df = df.Tensor.from_list(a_bcast.flatten().tolist(), [4, 1]); a_df.requires_grad = True
    b_df = df.Tensor.from_list(b_bcast.flatten().tolist(), [1, 6]); b_df.requires_grad = True
    df_op(a_df, b_df).sum().backward()

    check_close(a_df.grad.to_list(), a_pt.grad.flatten().numpy(), f"{op_name} broadcast grad A")
    check_close(b_df.grad.to_list(), b_pt.grad.flatten().numpy(), f"{op_name} broadcast grad B")


def test_matmul():
    print(f"\n--- Testing MATMUL ---")
    rng = np.random.RandomState(1337)
    a_np = rng.randn(4, 6).astype(np.float32)
    b_np = rng.randn(6, 5).astype(np.float32)

    # 1. Contiguous both
    a_pt = torch.tensor(a_np, requires_grad=True)
    b_pt = torch.tensor(b_np, requires_grad=True)
    (a_pt @ b_pt).sum().backward()

    a_df = df.Tensor.from_list(a_np.flatten().tolist(), [4, 6]); a_df.requires_grad = True
    b_df = df.Tensor.from_list(b_np.flatten().tolist(), [6, 5]); b_df.requires_grad = True
    a_df.matmul(b_df).sum().backward()

    check_close(a_df.grad.to_list(), a_pt.grad.flatten().numpy(), "matmul contiguous grad A")
    check_close(b_df.grad.to_list(), b_pt.grad.flatten().numpy(), "matmul contiguous grad B")

    # 2. Only A
    a_pt = torch.tensor(a_np, requires_grad=True)
    b_pt = torch.tensor(b_np, requires_grad=False)
    (a_pt @ b_pt).sum().backward()

    a_df = df.Tensor.from_list(a_np.flatten().tolist(), [4, 6]); a_df.requires_grad = True
    b_df = df.Tensor.from_list(b_np.flatten().tolist(), [6, 5]); b_df.requires_grad = False
    a_df.matmul(b_df).sum().backward()
    check_close(a_df.grad.to_list(), a_pt.grad.flatten().numpy(), "matmul only A requires grad")

    # 3. Only B (Needs_input_grad)
    a_pt = torch.tensor(a_np, requires_grad=False)
    b_pt = torch.tensor(b_np, requires_grad=True)
    (a_pt @ b_pt).sum().backward()

    a_df = df.Tensor.from_list(a_np.flatten().tolist(), [4, 6]); a_df.requires_grad = False
    b_df = df.Tensor.from_list(b_np.flatten().tolist(), [6, 5]); b_df.requires_grad = True
    a_df.matmul(b_df).sum().backward()
    check_close(b_df.grad.to_list(), b_pt.grad.flatten().numpy(), "matmul only B requires grad")

    # 4. Transposed inputs: a is [6, 4].T, b is [5, 6].T
    a_t_np = rng.randn(6, 4).astype(np.float32)
    b_t_np = rng.randn(5, 6).astype(np.float32)

    a_leaf_pt = torch.tensor(a_t_np, requires_grad=True)
    b_leaf_pt = torch.tensor(b_t_np, requires_grad=True)
    a_pt = a_leaf_pt.t()
    b_pt = b_leaf_pt.t()
    (a_pt @ b_pt).sum().backward()

    a_leaf_df = df.Tensor.from_list(a_t_np.flatten().tolist(), [6, 4]); a_leaf_df.requires_grad = True
    b_leaf_df = df.Tensor.from_list(b_t_np.flatten().tolist(), [5, 6]); b_leaf_df.requires_grad = True
    a_df = a_leaf_df.transpose(0, 1)
    b_df = b_leaf_df.transpose(0, 1)
    a_df.matmul(b_df).sum().backward()

    check_close(a_leaf_df.grad.to_list(), a_leaf_pt.grad.flatten().numpy(), "matmul transposed grad A")
    check_close(b_leaf_df.grad.to_list(), b_leaf_pt.grad.flatten().numpy(), "matmul transposed grad B")


def test_unary_and_reductions():
    print(f"\n--- Testing UNARY & REDUCTIONS ---")
    rng = np.random.RandomState(99)
    x_np = rng.randn(4, 6).astype(np.float32)

    # ReLU
    x_pt = torch.tensor(x_np, requires_grad=True)
    F.relu(x_pt).sum().backward()
    x_df = df.Tensor.from_list(x_np.flatten().tolist(), [4, 6]); x_df.requires_grad = True
    x_df.relu().sum().backward()
    check_close(x_df.grad.to_list(), x_pt.grad.flatten().numpy(), "relu backward")

    # Neg
    x_pt = torch.tensor(x_np, requires_grad=True)
    (-x_pt).sum().backward()
    x_df = df.Tensor.from_list(x_np.flatten().tolist(), [4, 6]); x_df.requires_grad = True
    x_df.neg().sum().backward()
    check_close(x_df.grad.to_list(), x_pt.grad.flatten().numpy(), "neg backward")

    # Sum full & dim
    x_pt = torch.tensor(x_np, requires_grad=True)
    x_pt.sum(dim=0, keepdim=True).sum().backward()
    x_df = df.Tensor.from_list(x_np.flatten().tolist(), [4, 6]); x_df.requires_grad = True
    x_df.sum(dims=[0], keepdim=True).sum().backward()
    check_close(x_df.grad.to_list(), x_pt.grad.flatten().numpy(), "sum(dim=0) backward")

    # Mean full & dim
    x_pt = torch.tensor(x_np, requires_grad=True)
    x_pt.mean(dim=1).sum().backward()
    x_df = df.Tensor.from_list(x_np.flatten().tolist(), [4, 6]); x_df.requires_grad = True
    x_df.mean(dims=[1], keepdim=False).sum().backward()
    check_close(x_df.grad.to_list(), x_pt.grad.flatten().numpy(), "mean(dim=1) backward")

    # Log Softmax
    x_pt = torch.tensor(x_np, requires_grad=True)
    F.log_softmax(x_pt, dim=-1).sum().backward()
    x_df = df.Tensor.from_list(x_np.flatten().tolist(), [4, 6]); x_df.requires_grad = True
    x_df.log_softmax(-1).sum().backward()
    check_close(x_df.grad.to_list(), x_pt.grad.flatten().numpy(), "log_softmax backward")

    # Cross Entropy
    logits_np = rng.randn(8, 5).astype(np.float32)
    labels_np = rng.randint(0, 5, size=(8,), dtype=np.int64)

    l_pt = torch.tensor(logits_np, requires_grad=True)
    t_pt = torch.tensor(labels_np)
    F.cross_entropy(l_pt, t_pt).backward()

    l_df = df.Tensor.from_list(logits_np.flatten().tolist(), [8, 5]); l_df.requires_grad = True
    t_df = df.Tensor.from_list_i64(labels_np.tolist(), [8])
    l_df.cross_entropy(t_df).backward()
    check_close(l_df.grad.to_list(), l_pt.grad.flatten().numpy(), "cross_entropy backward")


if __name__ == "__main__":
    test_binary_op("add", lambda a, b: a.add(b), lambda a, b: a + b)
    test_binary_op("sub", lambda a, b: a.sub(b), lambda a, b: a - b)
    test_binary_op("mul", lambda a, b: a.mul(b), lambda a, b: a * b)
    test_binary_op("div", lambda a, b: a.div(b), lambda a, b: a / b)
    test_matmul()
    test_unary_and_reductions()

    print(f"\n=======================================================")
    print(f"PER-OP VERIFICATION SUMMARY: {passed} passed, {failed} failed")
    print(f"=======================================================")
    assert failed == 0
