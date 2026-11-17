"""
Test autograd special behaviors:
- no_grad context manager
- detach()
- retain_graph=True / re-backward error
- tensor used twice in graph (x*x)
- gradient accumulation (two backward calls without zero_grad)
"""
import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "darkforest-py", "python"))

import darkforest as df
import torch
import math

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


# ---------------------------------------------------------------------------
# 1. no_grad
# ---------------------------------------------------------------------------
def test_no_grad():
    print("\n--- no_grad context manager ---")

    a = df.Tensor([1.0, 2.0, 3.0], [3])
    a.requires_grad = True

    # Inside no_grad, ops should NOT build grad_fn
    with df.no_grad():
        b = a + a
    check("no_grad: output requires_grad is False", not b.requires_grad,
          f"got requires_grad={b.requires_grad}")

    # Outside no_grad, ops should resume building grad_fn
    c = a + a
    check("after no_grad: output requires_grad is True", c.requires_grad,
          f"got requires_grad={c.requires_grad}")

    # PyTorch agrees
    a_pt = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
    with torch.no_grad():
        b_pt = a_pt + a_pt
    check("PyTorch parity: no_grad output rg=False", not b_pt.requires_grad)
    c_pt = a_pt + a_pt
    check("PyTorch parity: after no_grad output rg=True", c_pt.requires_grad)


# ---------------------------------------------------------------------------
# 2. detach
# ---------------------------------------------------------------------------
def test_detach():
    print("\n--- detach ---")

    a = df.Tensor([1.0, 2.0, 3.0], [3])
    a.requires_grad = True
    b = a * df.Tensor([2.0, 2.0, 2.0], [3])  # b = 2*a, grad_fn set
    d = b.detach()

    check("detach: requires_grad is False", not d.requires_grad,
          f"got {d.requires_grad}")

    # Op on detached tensor doesn't propagate back to a
    e = d + df.Tensor([1.0, 1.0, 1.0], [3])
    check("op on detach: output requires_grad is False", not e.requires_grad,
          f"got {e.requires_grad}")

    # a.grad should be None (no backward through detached path)
    check("a.grad is None before backward", a.grad is None)


# ---------------------------------------------------------------------------
# 3. retain_graph — first backward keeps graph, second without retain fails
# ---------------------------------------------------------------------------
def test_retain_graph():
    print("\n--- retain_graph ---")

    a = df.Tensor([1.0, 2.0, 3.0], [3])
    a.requires_grad = True
    loss = (a * a).sum_all()

    # First backward with retain_graph=True
    loss.backward(retain_graph=True)
    grad1 = list(a.grad.data())
    check("first backward with retain_graph: grad is [2,4,6]",
          [abs(g - e) < 1e-6 for g, e in zip(grad1, [2.0, 4.0, 6.0])],
          f"got {grad1}")

    # zero grad and do second backward (graph still alive)
    a.zero_grad()
    loss.backward(retain_graph=False)
    grad2 = list(a.grad.data())
    check("second backward (no retain): grad is [2,4,6]",
          [abs(g - e) < 1e-6 for g, e in zip(grad2, [2.0, 4.0, 6.0])],
          f"got {grad2}")

    # Third backward should raise (graph freed)
    a.zero_grad()
    raised = False
    try:
        loss.backward(retain_graph=False)
    except Exception:
        raised = True
    check("third backward without retain raises", raised)


# ---------------------------------------------------------------------------
# 4. Tensor used twice in one graph  (x*x → grad = 2x)
# ---------------------------------------------------------------------------
def test_tensor_used_twice():
    print("\n--- tensor used twice (x*x) ---")

    data = [1.0, 2.0, 3.0, 4.0]

    # Dark-Forest
    x = df.Tensor(data, [4])
    x.requires_grad = True
    y = x * x
    s = y.sum_all()
    s.backward()
    df_grad = list(x.grad.data())

    # PyTorch
    x_pt = torch.tensor(data, requires_grad=True)
    y_pt = x_pt * x_pt
    y_pt.sum().backward()
    pt_grad = x_pt.grad.tolist()

    diffs = [abs(a - b) for a, b in zip(df_grad, pt_grad)]
    check("x*x grad matches PyTorch [2,4,6,8]", max(diffs) < 1e-5,
          f"DF={df_grad}  PT={pt_grad}  max_diff={max(diffs):.2e}")


# ---------------------------------------------------------------------------
# 5. Gradient accumulation (two backward calls, no zero_grad between them)
# ---------------------------------------------------------------------------
def test_gradient_accumulation():
    print("\n--- gradient accumulation ---")

    data = [1.0, 2.0, 3.0]

    # Dark-Forest: two separate forward+backward without zero_grad
    x = df.Tensor(data, [3])
    x.requires_grad = True

    # First pass: loss = sum(x)   → grad = [1,1,1]
    loss1 = x.sum_all()
    loss1.backward()
    g1 = list(x.grad.data())

    # Second pass: loss = sum(x)  → accumulates → grad = [2,2,2]
    loss2 = x.sum_all()
    loss2.backward()
    g2 = list(x.grad.data())

    check("after 1st backward: grad=[1,1,1]",
          all(abs(g - 1.0) < 1e-6 for g in g1), f"got {g1}")
    check("after 2nd backward (no zero_grad): grad=[2,2,2]",
          all(abs(g - 2.0) < 1e-6 for g in g2), f"got {g2}")

    # PyTorch parity
    x_pt = torch.tensor(data, requires_grad=True)
    x_pt.sum().backward()
    g1_pt = x_pt.grad.tolist()
    x_pt.sum().backward()
    g2_pt = x_pt.grad.tolist()

    check("PyTorch parity grad after 1st", all(abs(a-b)<1e-6 for a,b in zip(g1,g1_pt)))
    check("PyTorch parity grad after 2nd", all(abs(a-b)<1e-6 for a,b in zip(g2,g2_pt)))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    test_no_grad()
    test_detach()
    test_retain_graph()
    test_tensor_used_twice()
    test_gradient_accumulation()

    print(f"\n{'='*55}")
    print(f"AUTOGRAD BEHAVIOR SUMMARY: {PASS} passed, {FAIL} failed")
    print(f"{'='*55}")
    sys.exit(0 if FAIL == 0 else 1)
