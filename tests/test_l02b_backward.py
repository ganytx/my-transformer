"""2B 检查 · 手写反向传播

对拍对象有两个：autograd（实现无关、但也是黑盒）和中心差分（数学定义，最硬）。
两个都对得上，才说明梯度真的推对了。
"""

from __future__ import annotations

import torch

import harness as H
from src import l02b_backward as M


def manual_grads(x, y, W1, b1, W2, b2):
    out = H.call(M, "manual_grads", x, y, W1, b1, W2, b2)
    if not isinstance(out, dict):
        raise AssertionError(f"manual_grads 应该返回 dict，实际是 {type(out).__name__}。")
    missing = [k for k in ("W1", "b1", "W2", "b2") if k not in out]
    if missing:
        raise AssertionError(f"返回的 dict 里缺少这些键：{missing}。应该正好四把：W1 / b1 / W2 / b2。")
    return out


def _loss(x, y, W1, b1, W2, b2):
    z1 = x @ W1.transpose(-1, -2) + b1
    h = torch.relu(z1)
    z2 = h @ W2.transpose(-1, -2) + b2
    return -torch.log_softmax(z2, -1).gather(1, y[:, None]).mean()


def _autograd(x, y, W1, b1, W2, b2):
    ps = [t.clone().requires_grad_(True) for t in (W1, b1, W2, b2)]
    _loss(x, y, *ps).backward()
    return dict(zip(("W1", "b1", "W2", "b2"), [p.grad for p in ps]))


def _numeric(f, p, eps=1e-6):
    base = p.detach().clone()
    flat = base.view(-1)
    out = torch.zeros(flat.numel(), dtype=base.dtype)
    for i in range(flat.numel()):
        old = flat[i].item()
        flat[i] = old + eps
        hi = f(base).item()
        flat[i] = old - eps
        lo = f(base).item()
        flat[i] = old
        out[i] = (hi - lo) / (2 * eps)
    return out.view(base.shape)


def _case(B=8, D=3, hd=5, C=4, seed=1):
    W1 = H.randn(hd, D, seed=seed)
    b1 = H.randn(hd, seed=seed + 1)
    W2 = H.randn(C, hd, seed=seed + 2)
    b2 = H.randn(C, seed=seed + 3)
    x = H.randn(B, D, seed=seed + 4)
    y = torch.randint(0, C, (B,), generator=torch.Generator().manual_seed(seed + 5))
    return x, y, W1, b1, W2, b2


# =========================================================================== #
def test_01_grad_shapes():
    """01 四个梯度的形状与参数完全一致"""
    args = _case()
    g = manual_grads(*args)
    for name, param in zip(("W1", "b1", "W2", "b2"), args[2:]):
        H.expect_tensor(g[name], f"∂L/∂{name}")
        H.assert_shape(g[name], tuple(param.shape), f"∂L/∂{name}")
        H.assert_dtype(g[name], param.dtype, f"∂L/∂{name}")


def test_02_matches_autograd():
    """02 手算梯度与 autograd 完全一致"""
    args = _case(B=8, D=3, hd=5, C=4, seed=11)
    g = manual_grads(*args)
    ref = _autograd(*args)
    for name in ("W1", "b1", "W2", "b2"):
        H.assert_close(g[name], ref[name], what=f"∂L/∂{name}", atol=1e-9, rtol=1e-7,
                       case="B=8, D=3, H=5, C=4，float64")


def test_03_matches_autograd_other_shapes():
    """03 换一组形状、包括 B=1，仍然与 autograd 一致"""
    for B, D, hd, C, seed in [(1, 2, 3, 2, 21), (17, 6, 4, 7, 22), (4, 1, 1, 1, 23)]:
        args = _case(B=B, D=D, hd=hd, C=C, seed=seed)
        g = manual_grads(*args)
        ref = _autograd(*args)
        for name in ("W1", "b1", "W2", "b2"):
            H.assert_close(g[name], ref[name], what=f"∂L/∂{name}（B={B},D={D},H={hd},C={C}）",
                           atol=1e-9, rtol=1e-7)


def test_04_numeric_grad_w2():
    """04 ∂L/∂W2 与中心差分一致（不看 autograd 的独立验证）"""
    x, y, W1, b1, W2, b2 = _case(B=5, D=2, hd=3, C=2, seed=31)
    g = manual_grads(x, y, W1, b1, W2, b2)
    num = _numeric(lambda W: _loss(x, y, W1, b1, W, b2), W2)
    H.assert_close(g["W2"], num, what="∂L/∂W2 的数值梯度", atol=1e-8, rtol=1e-5,
                   case="中心差分 (f(x+h)-f(x-h))/2h，h=1e-6")


def test_05_numeric_grad_w1():
    """05 ∂L/∂W1 与中心差分一致（要穿过 ReLU 和一层 W2）"""
    x, y, W1, b1, W2, b2 = _case(B=5, D=2, hd=3, C=2, seed=41)
    g = manual_grads(x, y, W1, b1, W2, b2)
    num = _numeric(lambda W: _loss(x, y, W, b1, W2, b2), W1)
    H.assert_close(g["W1"], num, what="∂L/∂W1 的数值梯度", atol=1e-8, rtol=1e-5,
                   case="这一条同时检查了 ReLU 的开关方向和 W2 的回传")
    num_b1 = _numeric(lambda b: _loss(x, y, W1, b, W2, b2), b1)
    H.assert_close(g["b1"], num_b1, what="∂L/∂b1 的数值梯度", atol=1e-8, rtol=1e-5)


def test_06_pure_function():
    """06 不修改传入的张量"""
    args = _case(seed=51)
    before = [t.clone() for t in args]
    manual_grads(*args)
    names = ("x", "y", "W1", "b1", "W2", "b2")
    for name, a, b in zip(names, args, before):
        if not torch.equal(a, b):
            raise AssertionError(f"manual_grads 修改了输入 {name}。求梯度不应该动原参数。")


def test_07_no_autograd():
    """07 没有偷偷用 autograd（这一讲就是要你自己走链式法则）"""
    H.assert_no_builtin(
        M,
        [".backward(", "torch.autograd", "autograd", "requires_grad"],
        why="手写反向传播的意义就在于不借助自动微分",
    )
    H.assert_no_loops(M, ["manual_grads"], why="这一讲要求整块张量运算")
