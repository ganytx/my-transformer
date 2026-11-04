"""L05 检查 · 单头因果注意力

核心只有三条：算得对、不偷看未来、知道自己在前还是在后。
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

import harness as H
from src import l05_attention as M


def attn(q, k, v, causal=True):
    return H.call(M, "scaled_dot_product_attention", q, k, v, causal)


def pe(t, d):
    return H.call(M, "sinusoidal_pe", t, d)


def make_attn(d_model):
    return H.call(M, "SingleHeadAttention", d_model)


# --------------------------------------------------------------------------- #
# 黄金用例
# --------------------------------------------------------------------------- #
Q = H.randn(2, 5, 4, seed=601)
K = H.randn(2, 5, 4, seed=602)
V = H.randn(2, 5, 4, seed=603)
Q64 = H.randn(2, 6, 64, seed=604)
K64 = H.randn(2, 6, 64, seed=605)
V64 = H.randn(2, 6, 64, seed=606)

H.golden_input("l05", "attn_causal",
               lambda r, t: r.scaled_dot_product_attention(t[0], t[1], t[2], True), (Q, K, V))
H.golden_input("l05", "attn_full",
               lambda r, t: r.scaled_dot_product_attention(t[0], t[1], t[2], False), (Q, K, V))
H.golden_input("l05", "attn_large_d",
               lambda r, t: r.scaled_dot_product_attention(t[0], t[1], t[2], True), (Q64, K64, V64))
H.golden_input("l05", "pe", lambda r, t: r.sinusoidal_pe(6, 8), None)


# =========================================================================== #
def test_01_shape_dtype():
    """01 输出 (B,T,d)，dtype 不变，输入未被修改"""
    q, k, v = (H.randn(2, 5, 4, seed=610, dtype=torch.float32),
               H.randn(2, 5, 4, seed=611, dtype=torch.float32),
               H.randn(2, 5, 4, seed=612, dtype=torch.float32))
    before = [t.clone() for t in (q, k, v)]
    out = attn(q, k, v)
    H.expect_tensor(out, "注意力输出")
    H.assert_shape(out, (2, 5, 4), "注意力输出")
    H.assert_dtype(out, torch.float32, "注意力输出")
    for name, a, b in zip("qkv", (q, k, v), before):
        if not torch.equal(a, b):
            raise AssertionError(f"注意力修改了输入 {name}。掩码应该用 masked_fill 生成新张量。")


def test_02_golden_causal():
    """02 因果注意力的数值与参考实现一致"""
    H.assert_close(attn(Q, K, V, True), H.golden("l05", "attn_causal"),
                   what="causal 注意力输出", case="B=2, T=5, d=4")


def test_03_golden_full():
    """03 非因果注意力的数值与参考实现一致"""
    H.assert_close(attn(Q, K, V, False), H.golden("l05", "attn_full"),
                   what="非因果注意力输出", case="同一组 q/k/v，causal=False")


def test_04_scaling():
    """04 d=64 时数值仍然正确（检查有没有除以 √d）"""
    H.assert_close(attn(Q64, K64, V64, True), H.golden("l05", "attn_large_d"),
                   what="d=64 时的注意力输出", case="B=2, T=6, d=64",
                   atol=1e-9, rtol=1e-7)


def test_05_causality_future():
    """05 改了未来位置，过去位置的输出一点都不能变"""
    q, k, v = H.randn(1, 6, 5, seed=620), H.randn(1, 6, 5, seed=621), H.randn(1, 6, 5, seed=622)
    base = attn(q, k, v, True)
    q2, k2, v2 = q.clone(), k.clone(), v.clone()
    q2[:, 4:] = H.randn(1, 2, 5, seed=623)
    k2[:, 4:] = H.randn(1, 2, 5, seed=624)
    v2[:, 4:] = H.randn(1, 2, 5, seed=625)
    after = attn(q2, k2, v2, True)
    H.assert_close(after[:, :4], base[:, :4], what="改变了第 4、5 个位置之后，前 4 个位置的输出",
                   atol=1e-12, rtol=1e-12,
                   case="因果注意力：位置 i 只能看到 0..i，绝不能受未来影响")


def test_06_causality_past():
    """06 改了最早的位置，后面的输出必须跟着变（不是把输入整个忽略了）"""
    q, k, v = H.randn(1, 6, 5, seed=630), H.randn(1, 6, 5, seed=631), H.randn(1, 6, 5, seed=632)
    base = attn(q, k, v, True)
    q2, k2, v2 = q.clone(), k.clone(), v.clone()
    q2[:, 0] += 3.0
    k2[:, 0] += 3.0
    v2[:, 0] += 3.0
    after = attn(q2, k2, v2, True)
    if torch.allclose(after[:, -1], base[:, -1], atol=1e-9):
        raise AssertionError(
            "改了第 0 个位置，最后一个位置的输出却没变。\n"
            "  说明你的掩码把整行都屏蔽掉了（比如把 > 和 >= 弄反），或者 softmax 之后输出恒为常数。"
        )
    if torch.allclose(after[:, 0], base[:, 0], atol=1e-9):
        raise AssertionError("位置 0 的输入变了，它的输出却没变——检查 v 有没有真正参与加权求和。")


def test_07_single_token():
    """07 T=1 时，输出必须正好等于 v（只能看自己，权重为 1）"""
    q = H.randn(3, 1, 5, seed=640)
    k = H.randn(3, 1, 5, seed=641)
    v = H.randn(3, 1, 5, seed=642)
    out = attn(q, k, v, True)
    H.assert_close(out, v, what="T=1 时的注意力输出", atol=1e-12, rtol=1e-12)


def test_08_permutation_equivariant():
    """08 常量输入下，各位置输出完全相同（注意力自己不知道顺序）"""
    d = 5
    x = H.randn(1, 1, d, seed=650).expand(1, 6, d).contiguous()
    h = make_attn(d)
    h = h.double()
    with torch.no_grad():
        out = h(x)
    H.assert_shape(out, (1, 6, d), "SingleHeadAttention 的输出")
    for t in range(1, 6):
        H.assert_close(out[:, t], out[:, 0], what=f"第 {t} 个位置的输出（输入一模一样）",
                       atol=1e-10, rtol=1e-10,
                       case="输入 6 个位置完全相同，注意力没有位置信息，输出必然相同")


def test_09_pe_properties():
    """09 位置编码的形状与数值性质"""
    pe_ = pe(6, 8)
    H.expect_tensor(pe_, "sinusoidal_pe 的返回值")
    H.assert_shape(pe_, (6, 8), "sinusoidal_pe 的返回值")
    if (pe_.abs() > 1 + 1e-6).any():
        raise AssertionError("位置编码里出现了绝对值大于 1 的数，sin/cos 的范围不对。")
    pos0 = torch.tensor([0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0])
    H.assert_close(pe_[0], pos0, what="第 0 行的位置编码", atol=1e-6, rtol=1e-6,
                   case="pos=0 时 sin(0)=0、cos(0)=1，所以是 0,1,0,1…")
    # 第 0 列应该是 sin(pos)，因为 2i/d=0 时频率是 1
    want = torch.sin(torch.arange(6, dtype=torch.float32))
    H.assert_close(pe_[:, 0], want, what="第 0 列", atol=1e-5, rtol=1e-5,
                   case="i=0 时 10000^(0/d)=1，所以频率是 1，第 0 列就是 sin(pos)")


def test_10_pe_golden():
    """10 位置编码与参考实现完全一致（含后半段的低频列）"""
    H.assert_close(pe(6, 8), H.golden("l05", "pe"), what="sinusoidal_pe(6, 8)",
                   atol=1e-6, rtol=1e-6, case="T=6, d=8")


def test_11_pe_breaks_symmetry():
    """11 加上位置编码之后，常量序列的各位置输出不再相同"""
    d = 5
    x = H.randn(1, 1, d, seed=660).expand(1, 6, d).contiguous()
    h = make_attn(d).double()
    pe_ = pe(6, d).double()
    with torch.no_grad():
        out = h(x + pe_[None, :, :])
    spread = (out[0] - out[0, 0]).abs().max().item()
    if spread < 1e-6:
        raise AssertionError(
            "加上位置编码之后，各位置的输出还是完全一样。\n"
            "  检查位置编码有没有真的加进去（注意 shape 要能和 (B,T,d) 广播），"
            "以及 pe 的量级是不是被别的东西盖掉了。"
        )


def test_12_param_count():
    """12 参数量是 4*d*d"""
    d = 12
    h = make_attn(d)
    total = sum(p.numel() for p in h.parameters())
    if total != 4 * d * d:
        raise AssertionError(
            f"参数量应该是 4*d*d = {4 * d * d}，实际 {total}。\n"
            f"  四个无 bias 的 (d,d) 线性层：Wq、Wk、Wv、Wo。bias=False 别忘了。"
        )


def test_13_no_builtin():
    """13 没有直接调用 PyTorch 的注意力 / softmax 实现"""
    H.assert_no_builtin(
        M,
        ["F.scaled_dot_product_attention", "functional.scaled_dot_product_attention",
         "nn.MultiheadAttention", "torch.softmax",
         ".softmax(", "F.softmax", "functional.softmax"],
        why="这一讲的 20 行就是整门课的核心，不能调现成的",
    )


def test_14_uniform_scores():
    """14 全零打分时，因果输出等于前缀平均，非因果输出等于全局平均"""
    q = torch.zeros(2, 4, 3, dtype=torch.float64)
    k = H.randn(2, 4, 3, seed=670)
    v = H.randn(2, 4, 3, seed=671)
    prefix_mean = torch.stack([v[:, :i + 1].mean(1) for i in range(4)], dim=1)
    H.assert_close(attn(q, k, v, True), prefix_mean, what="各位置的前缀平均")
    H.assert_close(attn(q, k, v, False), v.mean(1, keepdim=True).expand_as(v),
                   what="非因果模式的全局平均")


def test_15_pe_odd_dimensions():
    """15 d=1、3、7 的每一列位置编码都符合公式"""
    for d in (1, 3, 7):
        expected = torch.tensor([
            [(math.sin if col % 2 == 0 else math.cos)(
                pos / 10000 ** (2 * (col // 2) / d)) for col in range(d)]
            for pos in range(9)
        ], dtype=torch.float64)
        H.assert_close(pe(9, d), expected, what=f"奇数维位置编码 d={d}",
                       atol=1e-6, rtol=1e-6)


def test_16_module_forward_backward():
    """16 整个单头模块的输出、输入梯度及四组权重梯度与逐位置计算一致"""
    d = 3
    model = make_attn(d).double()
    # 使用可辨认的非对称权重，避免漏掉 Wo、转置方向错误被偶然掩盖。
    layers = [getattr(model, name) for name in ("Wq", "Wk", "Wv", "Wo")]
    with torch.no_grad():
        for i, layer in enumerate(layers):
            layer.weight.copy_(H.randn(d, d, seed=680 + i) * 0.4)
    x = H.randn(2, 4, d, seed=684).requires_grad_()
    xr = x.detach().clone().requires_grad_()
    weights = [layer.weight.detach().clone().requires_grad_() for layer in layers]
    # 独立参照：逐个 query 只访问合法前缀，不构造三角掩码。
    q, k, v = [xr @ w.T for w in weights[:3]]
    rows = []
    for i in range(x.shape[1]):
        scores = (q[:, i:i + 1] * k[:, :i + 1]).sum(-1) / math.sqrt(d)
        probs = torch.softmax(scores, dim=-1)
        rows.append((probs.unsqueeze(-1) * v[:, :i + 1]).sum(1))
    expected = torch.stack(rows, dim=1) @ weights[3].T
    actual = model(x)
    H.assert_close(actual, expected, what="SingleHeadAttention 完整前向")
    upstream = H.randn(2, 4, d, seed=685)
    got_grads = torch.autograd.grad((actual * upstream).sum(),
                                    [x] + [layer.weight for layer in layers])
    ref_grads = torch.autograd.grad((expected * upstream).sum(), [xr] + weights)
    for name, got, want in zip(("x", "Wq", "Wk", "Wv", "Wo"), got_grads, ref_grads):
        H.assert_close(got, want, what=f"{name} 的梯度")


def test_17_attention_gradcheck():
    """17 两种注意力模式的 q/k/v 反向梯度通过有限差分检查"""
    for causal in (True, False):
        inputs = tuple(H.randn(1, 3, 2, seed=690 + i).requires_grad_()
                       for i in range(3))
        assert torch.autograd.gradcheck(
            lambda q, k, v: attn(q, k, v, causal), inputs,
            eps=1e-6, atol=1e-5, rtol=1e-3,
        ), f"causal={causal}：自动求导与有限差分不一致"
