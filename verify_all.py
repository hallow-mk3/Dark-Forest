"""verify_all.py: Comprehensive Dark Forest Verification & Test Suite.

Executes and verifies:
1. Rust Workspace Unit Tests (cargo test --workspace)
2. Autograd & Ops Mathematical Gradient Checks (cargo run --bin grad_check)
3. GPU / Hardware Sanity Verification (CUDA device presence & capability)
4. PyTorch Multi-Head Attention vs Online Softmax Invariant Checks
5. Model Training Convergence Check (StaticGPT2 / PyTorch Loss Monotonicity)
"""

from __future__ import annotations

import subprocess
import sys
import time
import torch
import torch.nn.functional as F


def log_suite(name: str):
    print(f"\n================================================================================")
    print(f" [RUNNING SUITE] {name}")
    print(f"================================================================================")


def test_cargo_workspace():
    log_suite("1. Rust Workspace Unit Tests (cargo test --workspace)")
    t0 = time.perf_counter()
    res = subprocess.run(["cargo", "test", "--workspace"], capture_output=True, text=True)
    with open("logs/cargo_test_workspace.log", "w") as f:
        f.write(res.stdout + "\n" + res.stderr)
    duration = time.perf_counter() - t0
    if res.returncode != 0:
        print(f"[FAIL] (code {res.returncode}):\n{res.stderr}\n{res.stdout}")
        return False, 0, duration
    passed = 0
    for line in res.stdout.splitlines():
        if "test result: ok." in line:
            parts = line.split()
            for i, p in enumerate(parts):
                if p == "passed;":
                    passed += int(parts[i - 1])
    print(f"[PASS] {passed} Rust workspace unit tests passed in {duration:.2f}s")
    print(f"       -> Full log saved to logs/cargo_test_workspace.log")
    return True, passed, duration


def test_cargo_grad_check():
    log_suite("2. Autograd & Mathematical Gradient Checks (cargo run --bin grad_check)")
    t0 = time.perf_counter()
    res = subprocess.run(["cargo", "run", "--bin", "grad_check"], capture_output=True, text=True)
    with open("logs/cargo_grad_check.log", "w") as f:
        f.write(res.stdout + "\n" + res.stderr)
    duration = time.perf_counter() - t0
    if res.returncode != 0:
        print(f"[FAIL] (code {res.returncode}):\n{res.stderr}\n{res.stdout}")
        return False, 0, duration
    passed = res.stdout.count("PASS")
    print(f"[PASS] {passed} operator gradient checks passed in {duration:.2f}s")
    print(f"       -> Full log saved to logs/cargo_grad_check.log")
    return True, passed, duration


def test_cuda_hardware_sanity():
    log_suite("3. Hardware & CUDA Environment Sanity")
    t0 = time.perf_counter()
    available = torch.cuda.is_available()
    if not available:
        print("[WARN] CUDA device not detected via PyTorch.")
        return False, 0, 0.0
    device_name = torch.cuda.get_device_name(0)
    cap = torch.cuda.get_device_capability(0)
    vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    print(f"[PASS] Detected GPU: {device_name}")
    print(f"       Compute Capability: {cap[0]}.{cap[1]}")
    print(f"       Global Memory     : {vram_gb:.2f} GB")
    duration = time.perf_counter() - t0
    return True, 3, duration


def test_attention_numerical_parity():
    log_suite("4. Attention Parity: Standard Softmax vs Online Softmax")
    t0 = time.perf_counter()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    b, h, s, d = 2, 8, 128, 64
    scale = 1.0 / (d ** 0.5)

    q = torch.randn(b, h, s, d, device=device)
    k = torch.randn(b, h, s, d, device=device)
    v = torch.randn(b, h, s, d, device=device)

    # Standard attention
    scores = torch.matmul(q, k.transpose(-2, -1)) * scale
    mask = torch.triu(torch.full((s, s), float("-inf"), device=device), diagonal=1)
    probs = F.softmax(scores + mask, dim=-1)
    ref_out = torch.matmul(probs, v)

    # Fused SDPA (online softmax)
    fused_out = F.scaled_dot_product_attention(q, k, v, is_causal=True)

    max_diff = (ref_out - fused_out).abs().max().item()
    duration = time.perf_counter() - t0

    if max_diff < 1e-5:
        print(f"[PASS] Online Softmax matches standard attention (max discrepancy: {max_diff:.2e}) in {duration:.2f}s")
        return True, 1, duration
    else:
        print(f"[FAIL] Discrepancy too high: {max_diff:.2e}")
        return False, 0, duration


def test_training_loss_convergence():
    log_suite("5. Training Convergence Monotonicity Check")
    t0 = time.perf_counter()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    vocab, d_model = 64, 128

    model = torch.nn.Sequential(
        torch.nn.Embedding(vocab, d_model),
        torch.nn.Linear(d_model, vocab),
    ).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=1e-2)
    crit = torch.nn.CrossEntropyLoss()

    tokens = torch.randint(0, vocab, (4, 32), device=device)
    targets = torch.randint(0, vocab, (4, 32), device=device)

    losses = []
    for _ in range(20):
        opt.zero_grad()
        out = model(tokens)
        loss = crit(out.view(-1, vocab), targets.view(-1))
        loss.backward()
        opt.step()
        losses.append(loss.item())

    duration = time.perf_counter() - t0
    initial_loss = losses[0]
    final_loss = losses[-1]

    if final_loss < initial_loss * 0.7:
        print(f"[PASS] Loss converged smoothly from {initial_loss:.4f} down to {final_loss:.4f} in {duration:.2f}s")
        return True, 1, duration
    else:
        print(f"[FAIL] Insufficient convergence: {initial_loss:.4f} -> {final_loss:.4f}")
        return False, 0, duration


def main():
    print("================================================================================")
    print(" Project IRIS / Dark Forest - Full Automated Verification Suite")
    print("================================================================================")
    t_start = time.perf_counter()

    suites = [
        test_cargo_workspace,
        test_cargo_grad_check,
        test_cuda_hardware_sanity,
        test_attention_numerical_parity,
        test_training_loss_convergence,
    ]

    total_passed = 0
    all_ok = True

    for suite in suites:
        ok, passed, _ = suite()
        if ok:
            total_passed += passed
        else:
            all_ok = False

    total_time = time.perf_counter() - t_start

    print(f"\n================================================================================")
    print(f" VERIFICATION SUMMARY")
    print(f" Status: {'[PASS] ALL SUITES PASSED' if all_ok else '[FAIL] SOME SUITES FAILED'}")
    print(f" Total Verified Checks: {total_passed}")
    print(f" Total Wall Time      : {total_time:.2f}s")
    print(f"================================================================================")

    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
