"""L02 检查 · 训练闭环

契约 / 数值 / 性质 / 边界 / 梯度 / 实现约束 / 真的能学会。
"""

from __future__ import annotations

import math

import pytest
import torch
import torch.nn as nn

import harness as H
from src import l02_train as M


# --------------------------------------------------------------------------- #
def MLP(d_in, d_hidden, d_out):
    return H.call(M, "MLP", d_in, d_hidden, d_out)


def cross_entropy(logits, targets):
    return H.call(M, "cross_entropy", logits, targets)


def train_step(model, x, y, opt):
    return H.call(M, "train_step", model, x, y, opt)


def train_model(model, x, y, steps=300, lr=0.1):
    return H.call(M, "train_model", model, x, y, steps, lr)


def _linears(model):
    return [m for m in model.modules() if isinstance(m, nn.Linear)]


def _toy(n=300, seed=7, spread=0.6):
    """三个高斯团，2 维 3 类 —— 线性模型不够，MLP 足够。"""
    g = torch.Generator().manual_seed(seed)
    centers = torch.tensor([[-2.5, -2.5], [2.5, -2.5], [0.0, 2.5]], dtype=torch.float32)
    lab = torch.randint(0, 3, (n,), generator=g)
    x = centers[lab] + spread * torch.randn(n, 2, generator=g)
    return x, lab


# --------------------------------------------------------------------------- #
# 黄金用例
# --------------------------------------------------------------------------- #
LOGITS = H.randn(6, 5, seed=301)
TARGETS = torch.tensor([0, 4, 2, 2, 1, 3])
H.golden_input("l02", "ce", lambda r, inp: r.cross_entropy(inp[0], inp[1]), (LOGITS, TARGETS))


# =========================================================================== #
# 一、MLP 的结构与前向
# =========================================================================== #
def test_01_mlp_structure():
    """01 MLP 用两个 nn.Linear，权重形状是 (out, in)"""
    m = MLP(3, 5, 4)
    if not isinstance(m, nn.Module):
        raise AssertionError("MLP 必须继承 torch.nn.Module（否则 .parameters() 和 .to() 都用不了）。")
    linears = _linears(m)
    if len(linears) != 2:
        raise AssertionError(
            f"应该正好有 2 个 nn.Linear 子模块，实际找到 {len(linears)} 个。\n"
            f"  用 nn.Linear 建层，不要自己用 torch.randn 造权重——那样 .parameters() 里没有它们，优化器就更新不到。"
        )
    l1, l2 = linears
    H.assert_shape(l1.weight, (5, 3), "第一层 (d_in=3 -> d_hidden=5) 的 weight")
    H.assert_shape(l1.bias, (5,), "第一层的 bias")
    H.assert_shape(l2.weight, (4, 5), "第二层 (d_hidden=5 -> d_out=4) 的 weight")
    H.assert_shape(l2.bias, (4,), "第二层的 bias")

    total = sum(p.numel() for p in m.parameters())
    want = 3 * 5 + 5 + 5 * 4 + 4
    if total != want:
        raise AssertionError(f"参数量应该是 {want}，实际 {total}。是不是多建了层，或者 bias 没开？")


def test_02_mlp_forward_shape():
    """02 MLP 前向输出 (B, d_out)，dtype 保持"""
    m = MLP(3, 5, 4)
    x = H.randn(7, 3, seed=310, dtype=torch.float32)
    y = m(x)
    H.expect_tensor(y, "MLP 的输出")
    H.assert_shape(y, (7, 4), "MLP 的输出")
    H.assert_dtype(y, torch.float32, "MLP 的输出")


def test_03_mlp_matches_manual():
    """03 注入固定权重后，MLP 前向与手写公式一致"""
    m = MLP(3, 5, 4).double()
    linears = _linears(m)
    W1, b1 = H.randn(5, 3, seed=401), H.randn(5, seed=402)
    W2, b2 = H.randn(4, 5, seed=403), H.randn(4, seed=404)
    with torch.no_grad():
        linears[0].weight.copy_(W1)
        linears[0].bias.copy_(b1)
        linears[1].weight.copy_(W2)
        linears[1].bias.copy_(b2)

    x = H.randn(7, 3, seed=405)
    got = m(x)
    want = torch.relu(x @ W1.T + b1) @ W2.T + b2
    H.assert_close(got, want, what="MLP 前向输出",
                   case="relu(x @ W1.T + b1) @ W2.T + b2，激活是 ReLU，顺序是先激活再进第二层")


# =========================================================================== #
# 二、交叉熵
# =========================================================================== #
def test_04_ce_golden():
    """04 交叉熵与参考实现数值一致"""
    got = cross_entropy(LOGITS, TARGETS)
    H.expect_tensor(got, "cross_entropy 的返回值")
    H.assert_close(got.reshape(1), H.golden("l02", "ce").reshape(1),
                   what="cross_entropy(logits, targets)", case="B=6, C=5，标准正态 logits")


def test_05_ce_is_scalar():
    """05 交叉熵返回 0 维标量"""
    got = cross_entropy(LOGITS, TARGETS)
    if got.dim() != 0:
        raise AssertionError(
            f"应该返回 0 维标量（对整个 batch 取平均），实际 shape 是 {tuple(got.shape)}。\n"
            f"  已经在 batch 上平均过了的话，检查是不是用了 .sum() 而不是 .mean() 并且忘了除。"
        )


def test_06_ce_uniform_logits():
    """06 均匀 logits 时交叉熵等于 ln(C)"""
    c = 5
    logits = torch.zeros(3, c, dtype=torch.float64)
    got = cross_entropy(logits, torch.tensor([0, 2, 4]))
    H.assert_close(got.reshape(1), torch.tensor([math.log(c)], dtype=torch.float64),
                   what="均匀 logits 的交叉熵", atol=1e-12, rtol=1e-12,
                   case="完全猜不出时损失应该正好是 ln(类别数)")


def test_07_ce_gradient_is_softmax_minus_onehot():
    """07 交叉熵对 logits 的梯度等于 (softmax - onehot)/B"""
    logits = H.randn(4, 5, seed=501).requires_grad_(True)
    targets = torch.tensor([0, 3, 1, 2])
    loss = cross_entropy(logits, targets)
    loss.backward()
    if logits.grad is None:
        raise AssertionError("cross_entropy 没有把梯度传给 logits：中间是不是用了 .item() / .detach() / .long()？")
    onehot = torch.zeros(4, 5, dtype=torch.float64)
    onehot[torch.arange(4), targets] = 1.0
    want = (torch.softmax(logits.detach(), -1) - onehot) / 4
    H.assert_close(logits.grad, want, what="交叉熵对 logits 的梯度", atol=1e-9, rtol=1e-7,
                   case="这是交叉熵最漂亮的性质：梯度就是预测概率减去正确答案")


def test_08_ce_extreme_logits():
    """08 交叉熵在 1e4 量级 logits 下仍然有限"""
    logits = torch.tensor([[1e4, 0.0, -1e4]], dtype=torch.float64)
    got = cross_entropy(logits, torch.tensor([2]))
    H.assert_finite(got.reshape(1), "极端 logits 下的交叉熵")
    H.assert_close(got.reshape(1), torch.tensor([2e4], dtype=torch.float64),
                   what="答案错得离谱时的交叉熵", atol=1.0, rtol=1e-9,
                   case="正确答案的概率下溢到 0 时，损失应该是有限的大数，不是 inf")


def test_09_ce_batch_one():
    """09 交叉熵在 B=1 的边界上正确"""
    logits = H.randn(1, 3, seed=502)
    got = cross_entropy(logits, torch.tensor([1]))
    logp = logits - logits.max(-1, keepdim=True).values
    logp = logp - logp.exp().sum(-1, keepdim=True).log()
    want = -logp[0, 1]
    H.assert_close(got.reshape(1), want.reshape(1), what="B=1 时的交叉熵", atol=1e-12, rtol=1e-12)


# =========================================================================== #
# 三、一次参数更新
# =========================================================================== #
def test_10_train_step_returns_float():
    """10 train_step 返回 Python float"""
    x, y = _toy(n=32)
    m = MLP(2, 8, 3)
    opt = torch.optim.SGD(m.parameters(), lr=0.1)
    ret = train_step(m, x, y, opt)
    if not isinstance(ret, float):
        raise AssertionError(
            f"train_step 应该返回 Python 的 float，实际是 {type(ret).__name__}。\n"
            f"  张量用 .item() 取出数值。为什么不在训练循环里一直拿着张量？"
            f"因为那会让计算图一直被引用，显存/内存越跑越大。"
        )


def test_11_train_step_lr_zero_keeps_params():
    """11 lr=0 时参数一点都不能变"""
    x, y = _toy(n=32)
    m = MLP(2, 8, 3)
    before = [p.detach().clone() for p in m.parameters()]
    opt = torch.optim.SGD(m.parameters(), lr=0.0)
    train_step(m, x, y, opt)
    for i, (p, b) in enumerate(zip(m.parameters(), before)):
        if not torch.equal(p, b):
            raise AssertionError(
                f"lr=0 时第 {i} 个参数仍然变了（最大变化 {(p - b).abs().max().item():.3e}）。\n"
                f"  多半是手动写了 p -= lr * p.grad 却没乘 lr，或者用了 lr=1。"
            )


def test_12_train_step_updates_and_has_grads():
    """12 train_step 会真的更新参数，且每个参数都拿到了梯度"""
    x, y = _toy(n=64)
    m = MLP(2, 8, 3)
    before = [p.detach().clone() for p in m.parameters()]
    opt = torch.optim.SGD(m.parameters(), lr=0.5)
    train_step(m, x, y, opt)

    no_grad = [n for n, p in m.named_parameters() if p.grad is None]
    if no_grad:
        raise AssertionError(
            f"这些参数没有梯度：{no_grad}\n"
            f"  说明前向时它们没参与计算，或者计算被 torch.no_grad() / .detach() 截断了。"
        )
    changed = [not torch.equal(p, b) for p, b in zip(m.parameters(), before)]
    if not any(changed):
        raise AssertionError("跑完一步之后所有参数都没变——检查是否调用了 optimizer.step()。")


def test_13_train_step_reduces_loss():
    """13 连续调用 train_step，loss 明显下降"""
    x, y = _toy(n=200)
    m = MLP(2, 32, 3)
    opt = torch.optim.SGD(m.parameters(), lr=0.1)
    first = train_step(m, x, y, opt)
    last = first
    for _ in range(79):
        last = train_step(m, x, y, opt)
    if not (last < first * 0.8):
        raise AssertionError(
            f"训练 80 步之后 loss 几乎没降：{first:.4f} -> {last:.4f}（初始 ln3≈1.0986）。\n"
            f"  三个最常见的原因：① 忘了 zero_grad()；② 忘了 step()；③ 学习率太小/太大。\n"
            f"  先把 loss 曲线的前 5 个数打出来看看。"
        )


def test_14_train_model_converges():
    """14 训练 400 步之后训练集准确率超过 95%"""
    x, y = _toy(n=300)
    m = MLP(2, 32, 3)
    hist = train_model(m, x, y, steps=400, lr=0.2)
    if not isinstance(hist, list) or len(hist) != 400:
        raise AssertionError(f"train_model 应该返回长度 400 的 list，实际是 {type(hist).__name__}，长度 {len(hist) if hasattr(hist, '__len__') else '?'}。")
    if not all(isinstance(v, float) for v in hist):
        raise AssertionError("loss 列表里应该是 Python float，不是张量。")
    with torch.no_grad():
        acc = (m(x).argmax(dim=-1) == y).float().mean().item()
    if acc < 0.95:
        raise AssertionError(
            f"训练 400 步后训练集准确率只有 {acc:.1%}（loss {hist[0]:.3f} -> {hist[-1]:.3f}）。\n"
            f"  这三团数据非常好分，正常应该 >99%。检查：激活函数有没有加？两层之间的维度对不对？"
            f"优化器拿到的是不是 model.parameters()？"
        )


# =========================================================================== #
# 四、实现约束
# =========================================================================== #
def test_15_no_forbidden_loss():
    """15 没有直接调用 PyTorch 的交叉熵 / log_softmax"""
    H.assert_no_builtin(
        M,
        [
            "F.cross_entropy", "functional.cross_entropy", "nn.CrossEntropyLoss",
            "F.nll_loss", "functional.nll_loss", "nn.NLLLoss",
            "F.log_softmax", "functional.log_softmax", "torch.log_softmax",
        ],
        why="这一讲要你把 log_softmax + 取对应项 + 平均 这条链自己走一遍",
    )


def test_16_uses_own_log_softmax():
    """16 cross_entropy 复用了第 1 讲写的 log_softmax"""
    src = H.module_source(M)
    if "log_softmax" not in src:
        raise AssertionError(
            "没看到你调用自己写的 log_softmax。\n"
            f"  在 src/l02_train.py 顶部 `from src.l01_tensors import log_softmax`，然后直接用。\n"
            f"  第 1 讲那 23 项检查就是为这一天准备的。"
        )


def test_17_zero_grad_prevents_accumulation():
    """17 train_step 不会把上一步的梯度累加进来（漏掉 zero_grad 会被抓）"""
    x, y = _toy(n=32, seed=91)
    m = MLP(2, 8, 3)
    opt = torch.optim.SGD(m.parameters(), lr=0.1)

    train_step(m, x, y, opt)                       # 第 1 步
    p1 = [p.detach().clone() for p in m.parameters()]

    # 在 p1 这个点上，用 PyTorch 自己的交叉熵算出"真实的当前梯度"
    m.zero_grad()
    torch.nn.functional.cross_entropy(m(x), y).backward()
    g = [p.grad.detach().clone() for p in m.parameters()]
    # 注意：故意不清空 g。正确的 train_step 应该自己把它清掉。

    train_step(m, x, y, opt)                       # 第 2 步
    p2 = [p.detach().clone() for p in m.parameters()]

    for i, (a, b, gi) in enumerate(zip(p1, p2, g)):
        delta = b - a
        want = -0.1 * gi
        err = (delta - want).abs().max().item()
        scale = want.abs().max().item() + 1e-12
        if err > 1e-6 + 1e-4 * scale:
            raise AssertionError(
                f"第 2 步的更新量和「-学习率 × 当前梯度」对不上（第 {i} 个参数，最大偏差 {err:.3e}）。\n"
                f"  期望更新量最大 {scale:.3e}，实际 {delta.abs().max().item():.3e}。\n"
                f"  这是漏掉 optimizer.zero_grad() 的典型症状：.grad 里还留着上一步的值，"
                f"这一步走的是「历史梯度之和」，不是「当前梯度」。\n"
                f"  修法：train_step 的第一行调用 optimizer.zero_grad()。"
            )
