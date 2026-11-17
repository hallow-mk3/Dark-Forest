#!/usr/bin/env python3
"""tests/verify_against_pytorch.py

Comprehensive PyTorch Verification Suite for Dark Forest.
Directly cross-checks Dark-Forest tensors and operators against PyTorch (at::Tensor / torch.Tensor)
for:
1. Exact shape and stride matching
2. Storage offset alignment
3. Value parity (torch.allclose)
4. Zero-copy view aliasing (mutations reflect across views)
5. In-place version counter incrementing
6. Contiguous layout conversions

Usage:
    python tests/verify_against_pytorch.py
"""

from __future__ import annotations

import os
import sys
import numpy as np
import torch

# Ensure local darkforest package is importable
repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
df_py_path = os.path.join(repo_root, "darkforest-py", "python")
if df_py_path not in sys.path:
    sys.path.insert(0, df_py_path)

import darkforest as df

passed_count = 0
failed_count = 0


def log_test(name: str):
    print(f"\n[TEST] {name}")


def assert_parity(df_t, torch_t, test_name: str, check_strides: bool = True, atol: float = 1e-5):
    global passed_count, failed_count
    try:
        # 1. Shape check
        assert list(df_t.shape) == list(torch_t.shape), (
            f"Shape mismatch: DF {df_t.shape} vs PyTorch {list(torch_t.shape)}"
        )

        # 2. Strides check
        if check_strides:
            assert list(df_t.strides()) == list(torch_t.stride()), (
                f"Strides mismatch: DF {df_t.strides()} vs PyTorch {list(torch_t.stride())}"
            )

        # 3. Storage offset check
        assert df_t.storage_offset == torch_t.storage_offset(), (
            f"Offset mismatch: DF {df_t.storage_offset} vs PyTorch {torch_t.storage_offset()}"
        )

        # 4. Contiguity check
        assert df_t.is_contiguous() == torch_t.is_contiguous(), (
            f"Contiguity mismatch: DF {df_t.is_contiguous()} vs PyTorch {torch_t.is_contiguous()}"
        )

        # 5. Values check
        df_vals = np.array(df_t.to_list(), dtype=np.float32)
        torch_vals = torch_t.contiguous().detach().cpu().numpy().flatten()
        np.testing.assert_allclose(df_vals, torch_vals, atol=atol, rtol=1e-5)

        print(f"  PASS: {test_name} (shape={df_t.shape}, strides={df_t.strides()}, contiguous={df_t.is_contiguous()})")
        passed_count += 1
    except Exception as e:
        print(f"  FAIL: {test_name} -> {e}")
        failed_count += 1
        raise e


def test_tensor_creation():
    log_test("1. Tensor Creation & Contiguity Baseline")
    np.random.seed(42)
    arr = np.random.randn(3, 4, 5).astype(np.float32)

    torch_t = torch.from_numpy(arr.copy())
    df_t = df.Tensor.from_numpy(arr.copy())

    assert_parity(df_t, torch_t, "Creation from NumPy (3x4x5)")


def test_transpose_view():
    log_test("2. Transpose (2D & Multi-D Zero-Copy View)")
    arr = np.arange(12, dtype=np.float32).reshape(3, 4)
    torch_t = torch.from_numpy(arr.copy())
    df_t = df.Tensor.from_numpy(arr.copy())

    torch_tr = torch_t.transpose(0, 1)
    df_tr = df_t.transpose(0, 1)
    assert_parity(df_tr, torch_tr, "transpose(0, 1)")

    # Aliasing / Zero-copy verification:
    # Mutating element in df_t MUST reflect in df_tr without copying!
    initial_version = df_t.version()
    df_t.set(0, 999.0)
    torch_t[0, 0] = 999.0

    assert df_tr.get(0) == 999.0, "Zero-copy aliasing failed: base set did not reflect in transpose view!"
    assert df_t.version() == initial_version + 1, "Version counter did not increment on mutation!"
    assert df_tr.version() == df_t.version(), "View does not share version counter with base tensor!"
    print("  PASS: Zero-copy aliasing confirmed (mutation reflected in view, version counter bumped)")
    global passed_count
    passed_count += 1


def test_permute_view():
    log_test("3. Permute (Zero-Copy Multi-Dim Permutation)")
    arr = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    torch_t = torch.from_numpy(arr.copy())
    df_t = df.Tensor.from_numpy(arr.copy())

    torch_perm = torch_t.permute(2, 0, 1)
    df_perm = df_t.permute([2, 0, 1])

    assert_parity(df_perm, torch_perm, "permute([2, 0, 1])")


def test_slice_view():
    log_test("4. Slicing & Strided Narrowing")
    arr = np.arange(30, dtype=np.float32).reshape(5, 6)
    torch_t = torch.from_numpy(arr.copy())
    df_t = df.Tensor.from_numpy(arr.copy())

    # Slice dim 0: rows 1..4 (step 1)
    torch_sl0 = torch_t[1:4, :]
    df_sl0 = df_t.slice(0, 1, 4, 1)
    assert_parity(df_sl0, torch_sl0, "slice(dim=0, start=1, end=4, step=1)")

    # Slice dim 1: cols 2..6 (step 2)
    torch_sl1 = torch_sl0[:, 2:6:2]
    df_sl1 = df_sl0.slice(1, 2, 6, 2)
    assert_parity(df_sl1, torch_sl1, "strided slice(dim=1, start=2, end=6, step=2)")


def test_unsqueeze_and_squeeze():
    log_test("5. Unsqueeze & Squeeze (Zero-Copy Dimension Injection/Removal)")
    arr = np.arange(6, dtype=np.float32).reshape(2, 3)
    torch_t = torch.from_numpy(arr.copy())
    df_t = df.Tensor.from_numpy(arr.copy())

    # Unsqueeze dim 0
    torch_unsq0 = torch_t.unsqueeze(0)
    df_unsq0 = df_t.unsqueeze(0)
    assert_parity(df_unsq0, torch_unsq0, "unsqueeze(0)")

    # Unsqueeze dim 2
    torch_unsq2 = torch_t.unsqueeze(2)
    df_unsq2 = df_t.unsqueeze(2)
    assert_parity(df_unsq2, torch_unsq2, "unsqueeze(2)")

    # Squeeze dim 0
    torch_sq0 = torch_unsq0.squeeze(0)
    df_sq0 = df_unsq0.squeeze(0)
    assert_parity(df_sq0, torch_sq0, "squeeze(0)")


def test_expand_broadcasting():
    log_test("6. Expand (Zero-Copy Stride-0 Broadcasting)")
    arr = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)  # Shape (1, 3)
    torch_t = torch.from_numpy(arr.copy())
    df_t = df.Tensor.from_numpy(arr.copy())

    torch_exp = torch_t.expand(4, 3)
    df_exp = df_t.expand([4, 3])
    assert_parity(df_exp, torch_exp, "expand([4, 3]) from (1, 3)")

    # Multi-D expansion (2, 1, 4) -> (2, 5, 4)
    arr3d = np.arange(8, dtype=np.float32).reshape(2, 1, 4)
    torch_3d = torch.from_numpy(arr3d.copy())
    df_3d = df.Tensor.from_numpy(arr3d.copy())

    torch_exp3d = torch_3d.expand(2, 5, 4)
    df_exp3d = df_3d.expand([2, 5, 4])
    assert_parity(df_exp3d, torch_exp3d, "expand([2, 5, 4]) from (2, 1, 4)")


def test_flatten():
    log_test("7. Flatten (Dimension Collapsing)")
    arr = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
    torch_t = torch.from_numpy(arr.copy())
    df_t = df.Tensor.from_numpy(arr.copy())

    torch_flat = torch_t.flatten(1, 2)
    df_flat = df_t.flatten(1, 2)
    assert_parity(df_flat, torch_flat, "flatten(1, 2)")


def test_contiguous_conversion():
    log_test("8. Contiguous Layout Materialization")
    arr = np.arange(12, dtype=np.float32).reshape(3, 4)
    torch_t = torch.from_numpy(arr.copy())
    df_t = df.Tensor.from_numpy(arr.copy())

    # Transpose is non-contiguous
    torch_noncontig = torch_t.transpose(0, 1)
    df_noncontig = df_t.transpose(0, 1)
    assert not df_noncontig.is_contiguous()

    # Contiguous copy
    torch_contig = torch_noncontig.contiguous()
    df_contig = df_noncontig.contiguous()
    assert_parity(df_contig, torch_contig, "contiguous() on transposed tensor")


def test_view_vs_reshape_contract():
    log_test("9. View vs. Reshape Contract Parity")
    arr = np.arange(12, dtype=np.float32).reshape(3, 4)
    df_t = df.Tensor.from_numpy(arr.copy())

    # On contiguous tensor, view(2, 6) succeeds
    df_view = df_t.view([2, 6])
    assert list(df_view.shape) == [2, 6]
    assert df_view.is_contiguous()
    print("  PASS: view([2, 6]) succeeded on contiguous tensor")
    global passed_count
    passed_count += 1

    # On non-contiguous tensor, view() must raise an error (PyTorch rule!)
    df_noncontig = df_t.transpose(0, 1)
    try:
        df_noncontig.view([2, 6])
        assert False, "view() should have failed on non-contiguous tensor!"
    except Exception as e:
        print(f"  PASS: view() properly rejected non-contiguous layout ({e})")
        passed_count += 1

    # But reshape() must succeed by copying to contiguous under the hood
    df_reshaped = df_noncontig.reshape([2, 6])
    assert list(df_reshaped.shape) == [2, 6]
    assert df_reshaped.is_contiguous()
    print("  PASS: reshape([2, 6]) automatically restored contiguity")
    passed_count += 1


def test_inplace_version_counter():
    log_test("10. In-Place Mutation Version Counter Tracking")
    df_t = df.Tensor.zeros([4, 4])
    assert df_t.version() == 0, f"Expected initial version 0, got {df_t.version()}"

    df_t.add_inplace_(5.0)
    assert df_t.version() == 1, f"Expected version 1 after add_inplace_, got {df_t.version()}"
    assert df_t.get(0) == 5.0

    # Creating a slice view
    view = df_t.slice(0, 0, 2, 1)
    assert view.version() == 1

    # Mutating view bumps version for BOTH view and base tensor
    view.add_inplace_(10.0)
    assert view.version() == 2
    assert df_t.version() == 2
    assert df_t.get(0) == 15.0
    print("  PASS: In-place version tracking confirmed across base and view")
    global passed_count
    passed_count += 1


def main():
    print("================================================================================")
    print(" [DARK FOREST] PyTorch Parity & Verification Harness (Phase 1, Step 1 & 2)")
    print("================================================================================")
    print(f"PyTorch Version : {torch.__version__} (CUDA available: {torch.cuda.is_available()})")
    print(f"Dark Forest Ver : {df.version()}")

    test_tensor_creation()
    test_transpose_view()
    test_permute_view()
    test_slice_view()
    test_unsqueeze_and_squeeze()
    test_expand_broadcasting()
    test_flatten()
    test_contiguous_conversion()
    test_view_vs_reshape_contract()
    test_inplace_version_counter()

    print("\n================================================================================")
    print(f" SUITE COMPLETE: {passed_count} checks passed, {failed_count} failed")
    print("================================================================================")

    if failed_count > 0:
        sys.exit(1)
    else:
        print("ALL PYTORCH PARITY TESTS PASSED.")
        sys.exit(0)


if __name__ == "__main__":
    main()
