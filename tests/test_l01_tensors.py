"""L01 检查 · 张量与广播

检查按六类组织：契约 / 数值 / 性质 / 边界 / 梯度 / 实现约束。
每个测试的 docstring 就是 check.py 里显示给你的那一行。

这些测试是"规格说明书"，不是敌人：读它们能知道到底要求什么。
"""

from __future__ import annotations

import math

import pytest
import torch

import harness as H
from src import l01_tensors as M


# --------------------------------------------------------------------------- #
# 调用包装：接口缺失 / 未实现 -> 可读的失败信息
# --------------------------------------------------------------------------- #
def softmax(x, dim=-1):
    return H.call(M, "softmax", x, dim)


def log_softmax(x, dim=-1):
    return H.call(M, "log_softmax", x, dim)


def sqdist(x):
    return H.call(M, "pairwise_sqdist", x)


def dist(x):
    return H.call(M, "pairwise_dist", x)


# --------------------------------------------------------------------------- #
# 黄金用例：输入在这里登记，期望输出由 .private/gen_golden.py 用参考实现算好
# --------------------------------------------------------------------------- #
X_SM3 = H.golden_input("l01", "softmax_3d", lambda r, x: r.softmax(x, -1),
                       H.randn(2, 3, 5, seed=1))
X_SM0 = H.golden_input("l01", "softmax_dim0", lambda r, x: r.softmax(x, 0),
                       H.randn(4, 3, seed=2))
X_LSM3 = H.golden_input("l01", "log_softmax_3d", lambda r, x: r.log_softmax(x, -1),
                        H.randn(2, 4, 6, seed=3))
X_LSM1 = H.golden_input("l01", "log_softmax_dim1", lambda r, x: r.log_softmax(x, 1),
                        H.randn(2, 3, 4, seed=4))
X_D2 = H.golden_input("l01", "sqdist", lambda r, x: r.pairwise_sqdist(x),
                      H.randn(3, 7, 4, seed=5))
X_D = H.golden_input("l01", "dist", lambda r, x: r.pairwise_dist(x),
                     H.randn(3, 7, 4, seed=6))


# =========================================================================== #
# 契约
# =========================================================================== #
def test_01_softmax_shape_dtype():
    """01 softmax 保持 shape 与 dtype，且不修改输入"""
    x = H.randn(2, 3, 5, seed=11, dtype=torch.float32)
    x0 = x.clone()
    y = softmax(x, -1)
    H.expect_tensor(y, "softmax 的返回值")
    H.assert_shape(y, (2, 3, 5), "softmax 的返回值")
    H.assert_dtype(y, torch.float32, "softmax 的返回值")
    if not torch.equal(x, x0):
        raise AssertionError(
            "softmax 修改了输入张量。\n"
            "  判题会把同一份输入重复喂给你，原地修改会导致后面的测试出现莫名其妙的错误。\n"
            "  检查 x -= ...、x.div_()、x.exp_() 这类带下划线的原地操作。"
        )


def test_02_log_softmax_shape_dtype():
    """02 log_softmax 保持 shape 与 dtype"""
    x = H.randn(3, 4, seed=12, dtype=torch.float32)
    y = log_softmax(x, -1)
    H.expect_tensor(y, "log_softmax 的返回值")
    H.assert_shape(y, (3, 4), "log_softmax 的返回值")
    H.assert_dtype(y, torch.float32, "log_softmax 的返回值")


# =========================================================================== #
# 数值
# =========================================================================== #
def test_03_softmax_golden_3d():
    """03 softmax 与参考实现数值一致（dim=-1，三维输入）"""
    H.assert_close(softmax(X_SM3, -1), H.golden("l01", "softmax_3d"),
                   what="softmax(x, dim=-1)",
                   case="x 形状 (2,3,5)，标准正态随机数")


def test_04_softmax_golden_dim0():
    """04 softmax 与参考实现数值一致（dim=0）"""
    H.assert_close(softmax(X_SM0, 0), H.golden("l01", "softmax_dim0"),
                   what="softmax(x, dim=0)",
                   case="x 形状 (4,3)，沿第 0 维归一化")


def test_05_log_softmax_golden():
    """05 log_softmax 与参考实现数值一致（dim=-1 与 dim=1）"""
    H.assert_close(log_softmax(X_LSM3, -1), H.golden("l01", "log_softmax_3d"),
                   what="log_softmax(x, dim=-1)", case="x 形状 (2,4,6)")
    H.assert_close(log_softmax(X_LSM1, 1), H.golden("l01", "log_softmax_dim1"),
                   what="log_softmax(x, dim=1)", case="x 形状 (2,3,4)")


def test_06_sqdist_golden():
    """06 平方距离矩阵与参考实现数值一致"""
    H.assert_close(sqdist(X_D2), H.golden("l01", "sqdist"),
                   what="pairwise_sqdist(x)", case="x 形状 (3,7,4)")


def test_07_dist_golden():
    """07 距离矩阵与参考实现数值一致"""
    H.assert_close(dist(X_D), H.golden("l01", "dist"),
                   what="pairwise_dist(x)", case="x 形状 (3,7,4)")


# =========================================================================== #
# 性质
# =========================================================================== #
def test_08_softmax_normalizes():
    """08 softmax 沿指定轴的概率和为 1 且非负"""
    for dim in (-1, 0, 1):
        x = H.randn(3, 4, 5, seed=20 + abs(dim))
        y = softmax(x, dim)
        H.assert_finite(y, f"softmax(x, dim={dim})")
        s = y.sum(dim=dim)
        H.assert_close(s, torch.ones_like(s), what=f"softmax(x, dim={dim}) 沿该轴的求和",
                       atol=1e-9, rtol=1e-9, case=f"输入形状 (3,4,5)，dim={dim}")
        if (y < 0).any():
            idx = (y < 0).nonzero()[0].tolist()
            raise AssertionError(
                f"softmax 出现了负概率，位置 {tuple(idx)}，值 {y[tuple(idx)].item():.6g}。\n"
                f"  概率必须落在 [0,1]。检查分母是否用了 sum 之外的东西，或者有没有减错符号。"
            )


def test_09_softmax_shift_invariant():
    """09 softmax 平移不变，大数值输入不溢出"""
    x = torch.tensor([[1000.0, 1001.0, 999.0, 1000.5]], dtype=torch.float64)
    y = softmax(x, -1)
    H.assert_finite(y, "softmax 在 x≈1000 时的输出")
    H.assert_close(softmax(x - 1000.0, -1), y, what="softmax(x - 1000) 与 softmax(x)",
                   atol=1e-12, rtol=1e-12,
                   case="softmax 的分子分母同乘一个常数不改变结果，所以减去最大值不影响答案")

    big = torch.tensor([[1e4, 1e4 + 1.0, 1e4 - 1.0]], dtype=torch.float32)
    yb = softmax(big, -1)
    H.assert_finite(yb, "softmax(float32, x≈1e4) 的输出")
    H.assert_close(yb.sum(-1), torch.ones(1, dtype=torch.float32),
                   what="softmax(float32, 1e4) 的概率和", atol=1e-6, rtol=1e-6)


def test_10_softmax_slices_independent():
    """10 softmax 的各切片互相独立（不是全局归一化）"""
    x = H.randn(5, 4, seed=30)
    y = softmax(x, -1)
    x2 = x.clone()
    x2[0] = x2[0] + 100.0
    y2 = softmax(x2, -1)
    H.assert_close(y2[1:], y[1:], what="只改动第 0 行后其余行的 softmax",
                   atol=1e-12, rtol=1e-12,
                   case="沿 dim 之外的维度是批量方向：改一行不应该影响别的行")


def test_11_softmax_extreme_values():
    """11 softmax 在量级悬殊的输入上仍给出合法概率"""
    x = torch.tensor([[0.0, 1e9], [-1e9, 0.0], [5.0, 5.0]], dtype=torch.float64)
    y = softmax(x, -1)
    H.assert_finite(y, "softmax 在 1e9 量级输入下的输出")
    H.assert_close(y.sum(-1), torch.ones(3, dtype=torch.float64),
                   what="每行概率和", atol=1e-12, rtol=1e-12)
    H.assert_close(y, torch.tensor([[0.0, 1.0], [0.0, 1.0], [0.5, 0.5]], dtype=torch.float64),
                   what="极端输入下的概率", atol=1e-12, rtol=1e-12,
                   case="exp(-1e9) 下溢成 0 是正常的；不能出现 nan")


def test_12_log_softmax_relations():
    """12 log_softmax 等于 log(softmax) 且沿轴 exp 后和为 1"""
    x = H.randn(3, 4, seed=40)
    ls = log_softmax(x, -1)
    H.assert_finite(ls, "log_softmax 的输出")
    H.assert_close(ls.exp().sum(-1), torch.ones(3, dtype=torch.float64),
                   what="exp(log_softmax) 沿轴求和", atol=1e-10, rtol=1e-10)
    H.assert_close(ls, softmax(x, -1).log(), what="log_softmax(x) 与 log(softmax(x))",
                   atol=1e-10, rtol=1e-10)
    if (ls > 1e-12).any():
        raise AssertionError("log_softmax 出现了正数。log 概率必须 ≤ 0，检查是不是忘了取负数。")


def test_13_log_softmax_stable():
    """13 log_softmax 在极端输入下仍然有限"""
    x = torch.tensor([[1e4, 1e4 + 1.0, 1e4 - 1.0], [-1e9, 0.0, -1e9]], dtype=torch.float64)
    ls = log_softmax(x, -1)
    H.assert_finite(ls, "log_softmax 在 1e4 / -1e9 量级输入下的输出")
    H.assert_close(ls.exp().sum(-1), torch.ones(2, dtype=torch.float64),
                   what="exp(log_softmax) 每行求和", atol=1e-12, rtol=1e-12)
    if (ls > 1e-12).any():
        raise AssertionError(
            f"log_softmax 出现了正数（最大 {ls.max().item():.3e}）：log 概率必须 ≤ 0。\n"
            f"  检查是不是漏了归一化项，或者符号写反了。"
        )
    if not torch.equal(ls.argmax(-1), x.argmax(-1)):
        raise AssertionError(
            "log_softmax 的最大值位置和输入的最大值位置不一致。\n"
            "  概率最大的位置应该正好是原输入最大的位置。"
        )


def test_14_sqdist_properties():
    """14 距离矩阵对称、对角线为 0、非负"""
    x = H.randn(2, 6, 3, seed=50)
    d2 = sqdist(x)
    H.assert_shape(d2, (2, 6, 6), "pairwise_sqdist 的返回值")
    H.assert_finite(d2, "pairwise_sqdist 的返回值")
    H.assert_close(d2, d2.transpose(-1, -2), what="距离矩阵的对称性", atol=1e-12, rtol=1e-12,
                   case="d(i,j) 必须等于 d(j,i)")
    diag = torch.diagonal(d2, dim1=-2, dim2=-1)
    if (diag.abs() > 1e-12).any():
        raise AssertionError(
            f"对角线不为 0（最大 {diag.abs().max().item():.3e}）。\n"
            "  ||a-a||² 严格等于 0。用展开式 (x²+y²-2xy) 算时，浮点误差可能给出 -1e-16，"
            "这种情况下开方会变成 nan —— 记得把结果 clamp 到 ≥ 0。"
        )
    if (d2 < -1e-12).any():
        raise AssertionError("出现了负的平方距离。展开式在浮点下可能算出极小负数，需要 clamp。")


def test_15_sqdist_matches_definition():
    """15 平方距离与定义式（逐项相减再平方求和）一致"""
    x = H.randn(2, 5, 3, seed=51)
    d2 = sqdist(x)

    def reference(x):
        b, n, _ = x.shape
        out = torch.zeros(b, n, n, dtype=x.dtype)
        for bi in range(b):
            for i in range(n):
                for j in range(n):
                    out[bi, i, j] = ((x[bi, i] - x[bi, j]) ** 2).sum()
        return out

    H.assert_close(d2, reference(x), what="pairwise_sqdist(x)", atol=1e-9, rtol=1e-9,
                   case="x 形状 (2,5,3)")


def test_16_dist_definition():
    """16 距离矩阵等于平方距离开方，且恒为非负"""
    x = H.randn(2, 5, 3, seed=52)
    d = dist(x)
    H.assert_shape(d, (2, 5, 5), "pairwise_dist 的返回值")
    H.assert_finite(d, "pairwise_dist 的返回值")
    H.assert_close(d, torch.sqrt(torch.clamp(sqdist(x), min=0.0)),
                   what="pairwise_dist(x) 与 sqrt(pairwise_sqdist(x))", atol=1e-10, rtol=1e-10)
    if (d < 0).any():
        raise AssertionError("距离不能是负数。开方之前要 clamp 到 ≥ 0。")


def test_17_sqdist_edge_shapes():
    """17 距离矩阵在 B=1 / N=1 / D=1 的边界形状上正确"""
    for shape in [(1, 1, 3), (2, 1, 1), (1, 4, 1), (2, 3, 1)]:
        x = H.randn(*shape, seed=60 + shape[1])
        d2 = sqdist(x)
        H.assert_shape(d2, (shape[0], shape[1], shape[1]), f"pairwise_sqdist(x[{shape}]) 的返回值")
        H.assert_finite(d2, f"pairwise_sqdist(x[{shape}])")
        if shape[1] == 1 and d2.abs().max() > 1e-12:
            raise AssertionError(f"N=1 时矩阵应该全 0，实际最大 {d2.abs().max().item():.3e}")


# =========================================================================== #
# 梯度
# =========================================================================== #
def test_18_softmax_gradcheck():
    """18 softmax 的梯度与数值梯度一致（可反向传播）"""
    x = H.randn(3, 4, seed=70).requires_grad_(True)
    ok = torch.autograd.gradcheck(lambda t: softmax(t, -1), (x,), eps=1e-6, atol=1e-4, rtol=1e-2)
    if not ok:
        raise AssertionError("softmax 的自动求导结果与数值梯度不一致。")


def test_19_log_softmax_gradcheck():
    """19 log_softmax 的梯度与数值梯度一致"""
    x = H.randn(2, 5, seed=71).requires_grad_(True)
    ok = torch.autograd.gradcheck(lambda t: log_softmax(t, -1), (x,), eps=1e-6, atol=1e-4, rtol=1e-2)
    if not ok:
        raise AssertionError("log_softmax 的自动求导结果与数值梯度不一致。")


def test_20_sqdist_gradcheck():
    """20 平方距离的梯度与数值梯度一致"""
    x = H.randn(2, 3, 2, seed=72).requires_grad_(True)
    ok = torch.autograd.gradcheck(lambda t: sqdist(t), (x,), eps=1e-6, atol=1e-4, rtol=1e-2)
    if not ok:
        raise AssertionError("pairwise_sqdist 的自动求导结果与数值梯度不一致。")


def test_21_sqdist_gradient_reaches_input():
    """21 平方距离能把梯度传到输入张量"""
    x = H.randn(2, 4, 3, seed=73).requires_grad_(True)
    loss = sqdist(x).sum()
    loss.backward()
    if x.grad is None:
        raise AssertionError("梯度没有传到输入。检查是否在中间步骤用了 .item() / .detach() / torch.no_grad()。")
    H.assert_finite(x.grad, "输入张量的梯度")


# =========================================================================== #
# 反作弊
# =========================================================================== #
def test_22_no_forbidden_builtins():
    """22 没有直接调用 PyTorch 内置的 softmax / logsumexp / cdist"""
    H.assert_no_builtin(
        M,
        [
            "torch.softmax", ".softmax(", "torch.nn.functional.softmax",
            "torch.log_softmax", ".log_softmax(", "logsumexp",
            "torch.cdist", ".cdist(",
        ],
        why="这一关的意义就是把公式自己翻译成张量运算",
    )


def test_23_no_python_loops():
    """23 没有使用 for / while / 推导式（必须整块张量运算）"""
    H.assert_no_loops(M, ["softmax", "log_softmax", "pairwise_sqdist", "pairwise_dist"],
                      why="判题会在大 shape 上跑，Python 循环会让它慢几十倍，而且这关想教的就是广播")
