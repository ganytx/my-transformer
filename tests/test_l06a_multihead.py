"""L06A 判题：轴的语义、逐头计算、因果性与梯度。"""

import math

import pytest
import torch
import torch.nn as nn

import harness as H
from src import l06a_multihead as M
from src.l05_attention import SingleHeadAttention


def split(x, heads):
    return H.call(M, "split_heads", x, heads)


def merge(x):
    return H.call(M, "merge_heads", x)


def make(c, heads):
    return H.make(M, "MultiHeadAttention", c, heads)


def fixed_model(c, heads):
    model = make(c, heads).double()
    with torch.no_grad():
        for i, name in enumerate(("Wq", "Wk", "Wv", "Wo")):
            getattr(model, name).weight.copy_(H.randn(c, c, seed=710 + i) / math.sqrt(c))
    return model


def reference(x, weights, heads):
    """逐头、逐位置取合法前缀，与学生的合并 batch 算法独立。"""
    c = x.shape[-1]
    d = c // heads
    q, k, v = [x @ w.T for w in weights[:3]]
    outputs = []
    for head in range(heads):
        sl = slice(head * d, (head + 1) * d)
        rows = []
        for t in range(x.shape[1]):
            scores = (q[:, t:t + 1, sl] * k[:, :t + 1, sl]).sum(-1) / math.sqrt(d)
            prob = torch.softmax(scores, dim=-1)
            rows.append((prob.unsqueeze(-1) * v[:, :t + 1, sl]).sum(1))
        outputs.append(torch.stack(rows, dim=1))
    return torch.cat(outputs, dim=-1) @ weights[3].T


def test_01_split_numbered():
    """01 拆头的元素位置正确，不能只把目标 shape 改对"""
    x = torch.arange(8, dtype=torch.float64).reshape(1, 2, 4)
    want = torch.tensor([[[[0, 1], [4, 5]], [[2, 3], [6, 7]]]], dtype=x.dtype)
    H.assert_equal_tensor(split(x, 2), want, what="编号样例拆头结果")


def test_02_merge_numbered():
    """02 合头按每个位置拼接各头，支持不连续输入"""
    x = torch.arange(24, dtype=torch.float64).reshape(2, 3, 2, 2).transpose(1, 2)
    H.assert_equal_tensor(merge(x), torch.arange(24, dtype=x.dtype).reshape(2, 3, 4))


def test_03_roundtrip_and_grad():
    """03 不同形状拆合互逆，保持 dtype、输入和梯度"""
    for b, t, c, heads in ((2, 3, 12, 3), (1, 1, 5, 5), (3, 2, 7, 1)):
        x = H.randn(b, c, t, seed=720).transpose(1, 2).requires_grad_()
        before = x.detach().clone()
        separated = split(x, heads)
        H.assert_shape(separated, (b, heads, t, c // heads))
        out = merge(separated)
        H.assert_dtype(out, x.dtype)
        assert out.device == x.device, "拆合头不能改变 device"
        H.assert_equal_tensor(out, x)
        H.assert_equal_tensor(x, before, what="调用后的输入")
        probe = H.randn(b, t, c, seed=721)
        grad, = torch.autograd.grad((out * probe).sum(), x)
        H.assert_close(grad, probe, what="拆合头组合的输入梯度")


def test_04_invalid_configuration():
    """04 非法头数或模型宽度必须抛 ValueError"""
    for c, heads in ((6, 4), (6, 0), (6, -2), (0, 2), (-6, 2)):
        with pytest.raises(ValueError):
            make(c, heads)
    for heads in (0, -1, 4):
        with pytest.raises(ValueError):
            split(torch.zeros(1, 2, 6), heads)


def test_05_structure():
    """05 四个无偏置 C→C 投影，总参数 4*C*C，与头数无关"""
    for heads in (1, 3, 12):
        model = make(12, heads)
        assert model.n_heads == heads and model.head_dim == 12 // heads
        for name in ("Wq", "Wk", "Wv", "Wo"):
            layer = getattr(model, name)
            assert isinstance(layer, nn.Linear), f"{name} 应为 nn.Linear"
            assert layer.bias is None, f"{name} 应设置 bias=False"
            H.assert_shape(layer.weight, (12, 12), what=name)
        assert sum(p.numel() for p in model.parameters()) == 4 * 12 * 12


def test_06_forward_reference():
    """06 多种形状与逐头独立参照一致，缩放应使用每头宽度"""
    for b, t, c, heads in ((2, 5, 12, 3), (1, 3, 8, 2), (2, 4, 5, 5)):
        model = fixed_model(c, heads)
        x = H.randn(b, t, c, seed=730)
        weights = [getattr(model, name).weight for name in ("Wq", "Wk", "Wv", "Wo")]
        H.assert_close(model(x), reference(x, weights, heads), what="多头输出",
                       case=f"B={b}, T={t}, C={c}, H={heads}")


def test_07_one_head():
    """07 H=1 且权重相同时，与第五讲单头模块一致"""
    multi = fixed_model(5, 1)
    single = SingleHeadAttention(5).double()
    with torch.no_grad():
        for name in ("Wq", "Wk", "Wv", "Wo"):
            getattr(single, name).weight.copy_(getattr(multi, name).weight)
    x = H.randn(2, 4, 5, seed=740)
    H.assert_close(multi(x), single(x), what="单头退化情况")


def test_08_single_token():
    """08 T=1 时输出等于 Wo(Wv(x))，仍需保留输出投影"""
    model = fixed_model(8, 4)
    x = H.randn(2, 1, 8, seed=741)
    H.assert_close(model(x), model.Wo(model.Wv(x)), what="单 token 输出")


def test_09_causal():
    """09 修改未来 token 不得影响过去位置"""
    model = fixed_model(8, 2)
    x = H.randn(2, 6, 8, seed=750)
    changed = x.clone()
    changed[:, 3:] = H.randn(2, 3, 8, seed=751) * 5
    H.assert_close(model(x)[:, :3], model(changed)[:, :3],
                   what="未来变化前后的前缀输出", atol=1e-12, rtol=1e-12)


def test_10_backward_reference():
    """10 输入及四组参数的梯度与独立参照一致"""
    model = fixed_model(6, 3)
    x = H.randn(2, 4, 6, seed=760).requires_grad_()
    xr = x.detach().clone().requires_grad_()
    params = [getattr(model, name).weight for name in ("Wq", "Wk", "Wv", "Wo")]
    copies = [p.detach().clone().requires_grad_() for p in params]
    probe = H.randn(2, 4, 6, seed=761)
    got = torch.autograd.grad((model(x) * probe).sum(), [x] + params)
    want = torch.autograd.grad((reference(xr, copies, 3) * probe).sum(), [xr] + copies)
    for name, a, e in zip(("x", "Wq", "Wk", "Wv", "Wo"), got, want):
        H.assert_close(a, e, what=f"{name} 梯度")


def test_11_dtype_and_noncontiguous():
    """11 float32/float64 非连续输入可运行，且输入未被修改"""
    for dtype in (torch.float32, torch.float64):
        model = make(6, 2).to(dtype=dtype)
        x = H.randn(2, 6, 4, seed=770, dtype=dtype).transpose(1, 2)
        before = x.clone()
        out = model(x)
        H.assert_shape(out, x.shape)
        H.assert_dtype(out, dtype)
        H.assert_finite(out)
        H.assert_equal_tensor(x, before, what="原输入")
        H.assert_close(out, model(x.contiguous()), what="连续与非连续输入的输出",
                       atol=1e-6, rtol=1e-5)


def test_12_reuse_helpers(monkeypatch):
    """12 前向复用拆合函数，并用 (B*H,T,D) 调用第五讲"""
    seen = {"split": 0, "merge": 0, "attention": []}
    original_split, original_merge = M.split_heads, M.merge_heads
    original_attn = M.scaled_dot_product_attention

    def tracked_split(*args, **kwargs):
        seen["split"] += 1
        return original_split(*args, **kwargs)

    def tracked_merge(*args, **kwargs):
        seen["merge"] += 1
        return original_merge(*args, **kwargs)

    def tracked_attn(q, k, v, causal=True):
        seen["attention"].append((tuple(q.shape), tuple(k.shape), tuple(v.shape), causal))
        return original_attn(q, k, v, causal)

    monkeypatch.setattr(M, "split_heads", tracked_split)
    monkeypatch.setattr(M, "merge_heads", tracked_merge)
    monkeypatch.setattr(M, "scaled_dot_product_attention", tracked_attn)
    make(12, 3)(torch.zeros(2, 5, 12))
    assert seen["split"] >= 3 and seen["merge"] >= 1, "前向需要复用拆头/合头函数"
    assert seen["attention"] == [((6, 5, 4), (6, 5, 4), (6, 5, 4), True)], (
        "应将每个头作为独立样本，一次调用 L05；期望 (B*H,T,D)=(6,5,4)，causal=True"
    )


def test_13_no_shortcuts():
    """13 不使用内置注意力/softmax，也不循环遍历头"""
    H.assert_no_builtin(M, ["nn.MultiheadAttention", "F.scaled_dot_product_attention",
                            "functional.scaled_dot_product_attention", "torch.softmax",
                            ".softmax(", "F.softmax", "functional.softmax"])
    H.assert_no_loops(M, ["split_heads", "merge_heads", "forward", "__init__"])
