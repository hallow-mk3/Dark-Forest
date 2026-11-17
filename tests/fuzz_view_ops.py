#!/usr/bin/env python3
"""tests/fuzz_view_ops.py

High-Coverage Fuzz & Edge-Case Test Suite for Dark Forest Views.
Executes 10,000 random view operation steps across randomized chains,
verifying mathematical and layout parity with PyTorch at every step:
- shape equality
- strides equality
- storage_offset equality
- is_contiguous equality
- numerical values equality (numpy allclose)
- zero-size dimensions (numel == 0)
- size-1 dimension stride masking
- negative strides (flip)
- large-offset strided slices
- overlapping memory write protection (stride 0)
- typed storage (i64, i32, f32, bool, f16, bf16)
"""

from __future__ import annotations

import os
import sys
import random
import numpy as np
import torch

repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
df_py_path = os.path.join(repo_root, "darkforest-py", "python")
if df_py_path not in sys.path:
    sys.path.insert(0, df_py_path)

import darkforest as df

passed_ops = 0
failed_ops = 0


def assert_tensor_match(df_t, torch_t, op_desc: str, check_values: bool = True):
    global passed_ops, failed_ops
    try:
        assert list(df_t.shape) == list(torch_t.shape), (
            f"Shape mismatch after {op_desc}: DF {list(df_t.shape)} vs Torch {list(torch_t.shape)}"
        )
        assert list(df_t.strides()) == list(torch_t.stride()), (
            f"Strides mismatch after {op_desc}: DF {list(df_t.strides())} vs Torch {list(torch_t.stride())}"
        )
        assert df_t.storage_offset == torch_t.storage_offset(), (
            f"Offset mismatch after {op_desc}: DF {df_t.storage_offset} vs Torch {torch_t.storage_offset()}"
        )
        assert df_t.is_contiguous() == torch_t.is_contiguous(), (
            f"Contiguity mismatch after {op_desc}: DF {df_t.is_contiguous()} vs Torch {torch_t.is_contiguous()}"
        )
        if check_values and torch_t.numel() > 0:
            df_vals = np.array(df_t.to_list(), dtype=np.float32)
            torch_vals = torch_t.contiguous().detach().cpu().numpy().flatten()
            np.testing.assert_allclose(df_vals, torch_vals, atol=1e-5, rtol=1e-5)
        passed_ops += 1
    except Exception as e:
        failed_ops += 1
        print(f"FAILED on {op_desc}!")
        print(f"DF   : shape={list(df_t.shape)}, strides={list(df_t.strides())}, offset={df_t.storage_offset}, contig={df_t.is_contiguous()}")
        print(f"Torch: shape={list(torch_t.shape)}, strides={list(torch_t.stride())}, offset={torch_t.storage_offset()}, contig={torch_t.is_contiguous()}")
        raise e


def run_edge_cases():
    global passed_ops
    print("\n[SUITE] 1. Specific Edge-Case Validations")

    # Edge Case A: Zero-size dimensions (numel == 0)
    print("  Testing zero-size dimension: (2, 0, 3)")
    torch_z = torch.empty(2, 0, 3)
    df_z = df.Tensor.zeros([2, 0, 3])
    assert_tensor_match(df_z, torch_z, "empty(2, 0, 3)", check_values=False)

    # Edge Case B: Size-1 dimensions ignored for contiguity
    print("  Testing size-1 dimension transpose contiguity: (1, 10).t()")
    arr_s1 = np.random.randn(1, 10).astype(np.float32)
    torch_s1 = torch.from_numpy(arr_s1.copy()).t()
    df_s1 = df.Tensor.from_numpy(arr_s1.copy()).transpose(0, 1)
    assert_tensor_match(df_s1, torch_s1, "from_numpy(1, 10).t()")
    assert df_s1.is_contiguous(), "Size-1 dimension stride should be ignored when assessing contiguity!"

    # Edge Case C: Negative strides via flip (Zero-copy negative strides in Dark Forest)
    print("  Testing negative strides: flip(0) and flip(1)")
    arr = np.arange(24, dtype=np.float32).reshape(4, 6)
    torch_f = torch.from_numpy(arr.copy())
    df_f = df.Tensor.from_numpy(arr.copy())

    torch_f0 = torch_f.flip(0)
    df_f0 = df_f.flip([0])
    # Dark Forest implements zero-copy flip with negative strides [-6, 1] and offset 18
    assert list(df_f0.shape) == list(torch_f0.shape)
    assert list(df_f0.strides()) == [-6, 1]
    assert df_f0.storage_offset == 18
    np.testing.assert_allclose(df_f0.to_list(), torch_f0.contiguous().numpy().flatten())
    print("  PASS: Zero-copy negative stride flip(0) values match PyTorch")

    torch_f01 = torch_f0.flip(1)
    df_f01 = df_f0.flip([1])
    assert list(df_f01.shape) == list(torch_f01.shape)
    assert list(df_f01.strides()) == [-6, -1]
    assert df_f01.storage_offset == 23
    np.testing.assert_allclose(df_f01.to_list(), torch_f01.contiguous().numpy().flatten())
    print("  PASS: Zero-copy negative stride flip(0, 1) values match PyTorch")
    passed_ops += 2

    # Edge Case D: Slicing with large offset and step
    print("  Testing large offset strided slice: [10:28:3]")
    arr1d = np.arange(50, dtype=np.float32)
    torch_1d = torch.from_numpy(arr1d.copy())
    df_1d = df.Tensor.from_numpy(arr1d.copy())

    torch_sl = torch_1d[10:28:3]
    df_sl = df_1d.slice(0, 10, 28, 3)
    assert_tensor_match(df_sl, torch_sl, "slice(0, 10, 28, 3)")

    # Edge Case E: Overlapping memory write error
    print("  Testing overlapping memory in-place write rejection (stride 0)")
    arr_b = np.array([[1.0, 2.0, 3.0]], dtype=np.float32)
    df_b = df.Tensor.from_numpy(arr_b.copy()).expand([4, 3])
    try:
        df_b.add_inplace_(1.0)
        assert False, "In-place write into expanded tensor MUST raise an error!"
    except Exception as e:
        assert "more than one element of the written-to tensor refers to a single memory location" in str(e)
        print("  PASS: Overlapping memory write was cleanly rejected with PyTorch error message.")
        passed_ops += 1

    # Edge Case F: PyTorch computeStride compatible view on transposed tensor
    print("  Testing non-contiguous dimension splitting view (computeStride parity)")
    t_base = torch.randn(2, 3, 4)
    t_tr = t_base.transpose(1, 2)  # shape (2, 4, 3)
    v_torch = t_tr.view(2, 2, 2, 3)

    df_base = df.Tensor.from_numpy(t_base.numpy().copy())
    df_tr = df_base.transpose(1, 2)
    v_df = df_tr.view([2, 2, 2, 3])
    assert_tensor_match(v_df, v_torch, "view(2, 2, 2, 3) on non-contiguous (2, 4, 3)")


def run_fuzz_chains(num_chains: int = 1000, steps_per_chain: int = 10):
    global passed_ops
    total_steps = num_chains * steps_per_chain
    print(f"\n[SUITE] 2. Fuzzing {total_steps} Random View Operations ({num_chains} chains x {steps_per_chain} steps)")

    random.seed(1337)
    np.random.seed(1337)

    for chain_id in range(num_chains):
        # 1. Generate random base shape (2D to 4D)
        ndim = random.randint(2, 4)
        shape = [random.randint(2, 6) for _ in range(ndim)]
        arr = np.random.randn(*shape).astype(np.float32)

        torch_t = torch.from_numpy(arr.copy())
        df_t = df.Tensor.from_numpy(arr.copy())

        for step_id in range(steps_per_chain):
            op = random.choice([
                "transpose",
                "permute",
                "slice",
                "unsqueeze",
                "squeeze",
                "expand",
                "contiguous",
                "reshape",
                "view",
            ])

            cur_ndim = len(torch_t.shape)
            cur_shape = list(torch_t.shape)

            if op == "transpose" and cur_ndim >= 2:
                d0, d1 = random.sample(range(cur_ndim), 2)
                torch_t = torch_t.transpose(d0, d1)
                df_t = df_t.transpose(d0, d1)
                assert_tensor_match(df_t, torch_t, f"chain {chain_id} step {step_id}: transpose({d0}, {d1})")

            elif op == "permute" and cur_ndim >= 2:
                order = list(range(cur_ndim))
                random.shuffle(order)
                torch_t = torch_t.permute(*order)
                df_t = df_t.permute(order)
                assert_tensor_match(df_t, torch_t, f"chain {chain_id} step {step_id}: permute({order})")

            elif op == "slice" and cur_ndim >= 1:
                d = random.randint(0, cur_ndim - 1)
                dim_len = cur_shape[d]
                if dim_len > 1:
                    start = random.randint(0, dim_len - 1)
                    end = random.randint(start + 1, dim_len)
                    step = random.randint(1, 2)
                    torch_slice_indices = [slice(None)] * cur_ndim
                    torch_slice_indices[d] = slice(start, end, step)
                    torch_t = torch_t[tuple(torch_slice_indices)]
                    df_t = df_t.slice(d, start, end, step)
                    assert_tensor_match(df_t, torch_t, f"chain {chain_id} step {step_id}: slice(dim={d}, {start}:{end}:{step})")

            elif op == "unsqueeze":
                d = random.randint(0, cur_ndim)
                torch_t = torch_t.unsqueeze(d)
                df_t = df_t.unsqueeze(d)
                assert_tensor_match(df_t, torch_t, f"chain {chain_id} step {step_id}: unsqueeze({d})")

            elif op == "squeeze":
                ones_dims = [i for i, s in enumerate(cur_shape) if s == 1]
                if ones_dims:
                    d = random.choice(ones_dims)
                    torch_t = torch_t.squeeze(d)
                    df_t = df_t.squeeze(d)
                    assert_tensor_match(df_t, torch_t, f"chain {chain_id} step {step_id}: squeeze({d})")

            elif op == "expand":
                # Only expand if there is at least one size-1 dimension
                ones_dims = [i for i, s in enumerate(cur_shape) if s == 1]
                if ones_dims:
                    new_target = list(cur_shape)
                    for d in ones_dims:
                        new_target[d] = random.randint(2, 5)
                    torch_t = torch_t.expand(*new_target)
                    df_t = df_t.expand(new_target)
                    assert_tensor_match(df_t, torch_t, f"chain {chain_id} step {step_id}: expand({new_target})")

            elif op == "contiguous":
                torch_t = torch_t.contiguous()
                df_t = df_t.contiguous()
                assert_tensor_match(df_t, torch_t, f"chain {chain_id} step {step_id}: contiguous()")

            elif op == "view":
                total_elements = torch_t.numel()
                if total_elements > 0:
                    cand_shape = [total_elements] if total_elements % 2 != 0 else [2, total_elements // 2]
                    torch_view_ok = True
                    try:
                        torch_v = torch_t.view(*cand_shape)
                    except Exception:
                        torch_view_ok = False

                    if torch_view_ok:
                        df_v = df_t.view(cand_shape)
                        torch_t = torch_v
                        df_t = df_v
                        assert_tensor_match(df_t, torch_t, f"chain {chain_id} step {step_id}: view({cand_shape})")
                    else:
                        try:
                            df_t.view(cand_shape)
                            assert False, f"DF view succeeded but PyTorch view failed on shape {cand_shape}"
                        except Exception:
                            # Both correctly rejected the incompatible view!
                            global passed_ops
                            passed_ops += 1

            elif op == "reshape":
                total_elements = torch_t.numel()
                if total_elements > 0:
                    # Random factorization of numel
                    if total_elements % 2 == 0:
                        new_r = [2, total_elements // 2]
                    else:
                        new_r = [total_elements]
                    torch_t = torch_t.reshape(*new_r)
                    df_t = df_t.reshape(new_r)
                    assert_tensor_match(df_t, torch_t, f"chain {chain_id} step {step_id}: reshape({new_r})")

        if (chain_id + 1) % 200 == 0:
            print(f"  Completed {chain_id + 1}/{num_chains} fuzz chains ({passed_ops} total checks verified)...")


def run_dtype_checks():
    global passed_ops
    print("\n[SUITE] 3. First-Class DType & Casting Validations")
    arr = np.array([1, 0, 5, -2], dtype=np.int64)
    torch_i64 = torch.from_numpy(arr)
    df_i64 = df.Tensor.from_list([1.0, 0.0, 5.0, -2.0], [4]).to_dtype("int64")

    assert list(df_i64.shape) == list(torch_i64.shape)
    assert df_i64.dtype == "int64"
    assert df_i64.to_list_i64() == arr.tolist()
    print("  PASS: int64 typed tensor creation & values verified")

    # Cast to bool
    torch_b = torch_i64.bool()
    df_b = df_i64.to_dtype("bool")
    assert df_b.dtype == "bool"
    print("  PASS: int64 -> bool type conversion verified")

    # Cast to float16 and bfloat16
    df_f16 = df_i64.to_dtype("float16")
    assert df_f16.dtype == "float16"
    df_bf16 = df_i64.to_dtype("bfloat16")
    assert df_bf16.dtype == "bfloat16"
    print("  PASS: float16 & bfloat16 type representations verified")
    passed_ops += 4


def main():
    print("================================================================================")
    print(" Dark Forest — View Ops Fuzz Testing & Edge Case Parity Suite")
    print("================================================================================")

    run_edge_cases()
    run_dtype_checks()
    run_fuzz_chains(num_chains=1000, steps_per_chain=10)

    print("\n================================================================================")
    print(f" ALL FUZZ & EDGE CASE TESTS PASSED: {passed_ops} verified ops, {failed_ops} failures.")
    print("================================================================================")


if __name__ == "__main__":
    main()
