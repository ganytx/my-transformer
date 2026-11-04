"""6B2：FFN、Pre-LN 顺序、残差路径与完整梯度。"""
import math

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

import harness as H
from src import l06b2_block as M
from src.l06a_multihead import MultiHeadAttention
from src.l06b1_layernorm import LayerNorm


def make_ff(c):
    return H.make(M, "FeedForward", c)


def make_block(c, heads, eps=1e-5):
    return H.make(M, "TransformerBlock", c, heads, eps=eps)


def fixed(model):
    model = model.double()
    with torch.no_grad():
        for i, (name, p) in enumerate(model.named_parameters()):
            value = H.randn(*p.shape, seed=901 + i) * 0.2
            if name.endswith("weight") and p.ndim == 1:
                value = value + 1
            p.copy_(value)
    return model


def ff_reference(x, p, prefix=""):
    hidden = x @ p[prefix + "fc1.weight"].T + p[prefix + "fc1.bias"]
    activated = hidden * (1 + torch.erf(hidden / math.sqrt(2))) / 2
    return activated @ p[prefix + "fc2.weight"].T + p[prefix + "fc2.bias"]


def block_reference(x, p, heads, eps):
    """使用内置 LN、erf GELU、逐头逐位置注意力，独立于学生的组装。"""
    c = x.shape[-1]
    d = c // heads
    normed = F.layer_norm(x, (c,), p["ln1.weight"], p["ln1.bias"], eps)
    q, k, v = [normed @ p[f"attn.W{name}.weight"].T for name in ("q", "k", "v")]
    outputs = []
    for h in range(heads):
        sl = slice(h * d, (h + 1) * d)
        rows = []
        for t in range(x.shape[1]):
            scores = (q[:, t:t + 1, sl] * k[:, :t + 1, sl]).sum(-1) / math.sqrt(d)
            rows.append((torch.softmax(scores, -1).unsqueeze(-1) * v[:, :t + 1, sl]).sum(1))
        outputs.append(torch.stack(rows, 1))
    u = x + torch.cat(outputs, -1) @ p["attn.Wo.weight"].T
    normed_u = F.layer_norm(u, (c,), p["ln2.weight"], p["ln2.bias"], eps)
    return u + ff_reference(normed_u, p, "ffn.")


def test_01_ff_structure():
    """01 FFN 是 C→4C→C，两层有 bias，中间为 GELU"""
    c = 5
    ff = make_ff(c)
    assert isinstance(ff.fc1, nn.Linear) and isinstance(ff.fc2, nn.Linear)
    assert isinstance(ff.act, nn.GELU) and ff.act.approximate == "none"
    for layer, shape in ((ff.fc1, (4 * c, c)), (ff.fc2, (c, 4 * c))):
        H.assert_shape(layer.weight, shape)
        assert layer.bias is not None, "两层 FFN Linear 都需要 bias"
        H.assert_shape(layer.bias, (shape[0],))
    assert sum(p.numel() for p in ff.parameters()) == 8 * c * c + 5 * c


def test_02_ff_values():
    """02 FFN 数值与矩阵公式和 erf GELU 独立参照一致"""
    for shape in ((2, 3, 5), (1, 1, 1), (3, 2, 4)):
        ff = fixed(make_ff(shape[-1]))
        x = H.randn(*shape, seed=920)
        H.assert_close(ff(x), ff_reference(x, dict(ff.named_parameters())), what="FFN 输出")


def test_03_ff_token_independence():
    """03 FFN 逐 token 计算，修改其他位置不影响当前输出"""
    ff = fixed(make_ff(4))
    x = H.randn(2, 3, 4, seed=921)
    changed = x.clone()
    changed[0, 2] += 5
    changed[1] *= 10
    H.assert_close(ff(x)[0, :2], ff(changed)[0, :2], what="FFN 的未修改位置")


def test_04_block_structure():
    """04 复用已有模块，LN 参数独立，传递 eps，总参数量正确"""
    c = 6
    block = make_block(c, 3, eps=0.07)
    assert isinstance(block.ln1, LayerNorm) and isinstance(block.ln2, LayerNorm)
    assert isinstance(block.attn, MultiHeadAttention)
    assert isinstance(block.ffn, M.FeedForward)
    assert block.attn.n_heads == 3
    assert block.ln1.eps == block.ln2.eps == 0.07, "两份 LN 都应接收 eps"
    assert block.ln1 is not block.ln2, "两份 LN 应为独立实例"
    for name in ("weight", "bias"):
        assert getattr(block.ln1, name).data_ptr() != getattr(block.ln2, name).data_ptr(), (
            f"LN1/LN2 的 {name} 不能共享存储"
        )
    assert sum(p.numel() for p in block.parameters()) == 12 * c * c + 9 * c


def test_05_invalid_configuration():
    """05 非法宽度、头数或 eps 抛 ValueError"""
    for c in (0, -2):
        with pytest.raises(ValueError):
            make_ff(c)
    for c, h, eps in ((0, 1, 1e-5), (-2, 1, 1e-5), (4, 0, 1e-5),
                      (4, -1, 1e-5), (4, 3, 1e-5), (4, 2, 0), (4, 2, -0.1)):
        with pytest.raises(ValueError):
            make_block(c, h, eps)


def test_06_block_reference():
    """06 完整块符合两次 Pre-LN 残差公式，覆盖 T=1/C=1/H=1"""
    for b, t, c, heads in ((2, 4, 6, 3), (1, 3, 5, 1), (2, 1, 4, 2), (1, 2, 1, 1)):
        block = fixed(make_block(c, heads, eps=0.03))
        x = H.randn(b, t, c, seed=930)
        want = block_reference(x, dict(block.named_parameters()), heads, eps=0.03)
        H.assert_close(block(x), want, what="Transformer 块输出",
                       case=f"B={b}, T={t}, C={c}, H={heads}，检查归一化和残差顺序")


def test_07_zero_branches_identity():
    """07 两个分支为零时，整个块是恒等映射，输入梯度直接传回"""
    block = fixed(make_block(4, 2))
    with torch.no_grad():
        block.attn.Wo.weight.zero_()
        block.ffn.fc2.weight.zero_()
        block.ffn.fc2.bias.zero_()
    x = H.randn(2, 3, 4, seed=940).requires_grad_()
    out = block(x)
    H.assert_close(out, x, what="分支为零时的输出",
                   case="残差主路径应保留原输入，不能漏掉相加或用 LN(x) 替代 x")
    probe = H.randn(2, 3, 4, seed=941)
    gradient, = torch.autograd.grad((out * probe).sum(), x)
    H.assert_close(gradient, probe, what="恒等映射的输入梯度",
                   case="不要 detach 残差主路径")


def test_08_causality():
    """08 完整块仍保持因果性，未来 token 不影响前缀"""
    block = fixed(make_block(6, 2))
    x = H.randn(2, 5, 6, seed=950)
    changed = x.clone()
    changed[:, 3:] = H.randn(2, 2, 6, seed=951) * 4
    H.assert_close(block(x)[:, :3], block(changed)[:, :3], what="过去位置的输出",
                   atol=1e-12, rtol=1e-12)


def test_09_all_gradients_reference():
    """09 输入和所有参数的梯度与独立参照一致"""
    block = fixed(make_block(4, 2, eps=0.02))
    x = H.randn(2, 3, 4, seed=960).requires_grad_()
    xr = x.detach().clone().requires_grad_()
    params = dict(block.named_parameters())
    copies = {name: p.detach().clone().requires_grad_() for name, p in params.items()}
    probe = H.randn(2, 3, 4, seed=961)
    got = torch.autograd.grad((block(x) * probe).sum(), [x] + list(params.values()))
    want_out = block_reference(xr, copies, 2, eps=0.02)
    want = torch.autograd.grad((want_out * probe).sum(), [xr] + list(copies.values()))
    for name, a, e in zip(["x"] + list(params), got, want):
        H.assert_close(a, e, what=f"{name} 的梯度")


def test_10_gradcheck():
    """10 完整块的输入梯度通过有限差分检查"""
    block = fixed(make_block(4, 2, eps=0.02))
    x = H.randn(1, 2, 4, seed=970).requires_grad_()
    assert torch.autograd.gradcheck(block, (x,), eps=1e-6, atol=1e-5, rtol=1e-3)


def test_11_dtype_layout():
    """11 FFN 和块保持 shape/dtype/device，支持非连续输入"""
    for dtype in (torch.float32, torch.float64):
        for model in (make_ff(6), make_block(6, 3)):
            model = model.to(dtype=dtype)
            x = H.randn(2, 6, 3, seed=980, dtype=dtype).transpose(1, 2)
            out = model(x)
            H.assert_shape(out, x.shape)
            H.assert_dtype(out, dtype)
            assert out.device == x.device
            H.assert_finite(out)
            H.assert_close(out, model(x.contiguous()), what="连续与非连续输入",
                           atol=2e-6, rtol=2e-6)


def test_12_no_mutation():
    """12 前向不修改输入或参数，叶子输入可反向传播"""
    for model in (fixed(make_ff(4)), fixed(make_block(4, 2))):
        x = H.randn(2, 3, 4, seed=990).requires_grad_()
        before = x.detach().clone()
        params = {n: p.detach().clone() for n, p in model.named_parameters()}
        out = model(x)
        H.assert_equal_tensor(x, before, what="前向后的输入")
        for name, p in model.named_parameters():
            H.assert_equal_tensor(p, params[name], what=f"前向后的参数 {name}")
        out.square().mean().backward()
        H.assert_finite(x.grad, what="输入梯度")


def test_13_train_eval():
    """13 本题无 dropout，train/eval 输出一致，参数不被重建"""
    block = fixed(make_block(6, 2))
    x = H.randn(2, 3, 6, seed=991)
    params = dict(block.named_parameters())
    a = block.train()(x)
    b = block.eval()(x)
    H.assert_close(a, b, what="train/eval 输出")
    assert all(dict(block.named_parameters()).get(n) is p for n, p in params.items())


def test_14_no_shortcuts():
    """14 不使用现成 Transformer/归一化/注意力，不用循环"""
    H.assert_no_builtin(M, ["nn.Transformer", "nn.MultiheadAttention", "nn.LayerNorm",
                            ".layer_norm(", "F.scaled_dot_product_attention",
                            "functional.scaled_dot_product_attention", "torch.softmax", ".softmax("])
    H.assert_no_loops(M, ["__init__", "forward"])
