"""7A：模型注册、完整 logits、对齐标签和训练闭环。"""
import math

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

import harness as H
from src import l07a_gpt as M
from src.l06b1_layernorm import LayerNorm
from src.l06b2_block import TransformerBlock
from test_l06b2_block import block_reference, fixed


def make(**changes):
    config = dict(vocab_size=7, block_size=6, d_model=6, n_heads=2, n_layers=2)
    config.update(changes)
    return H.make(M, "MiniGPT", **config)


def loss(logits, targets):
    return H.call(M, "language_model_loss", logits, targets)


def reference(idx, p, pe, heads, layers):
    x = p["token_emb.weight"][idx] + pe[:idx.shape[1]]
    for i in range(layers):
        prefix = f"blocks.{i}."
        bp = {name[len(prefix):]: value for name, value in p.items() if name.startswith(prefix)}
        x = block_reference(x, bp, heads, eps=1e-5)
    x = F.layer_norm(x, (x.shape[-1],), p["ln_f.weight"], p["ln_f.bias"], 1e-5)
    return x @ p["lm_head.weight"].T


def test_01_registration():
    """01 注册独立的块、embedding、最终 LN 和输出层，参数量正确"""
    model = make()
    assert model.vocab_size == 7 and model.block_size == 6
    assert isinstance(model.token_emb, nn.Embedding)
    H.assert_shape(model.token_emb.weight, (7, 6))
    assert isinstance(model.blocks, nn.ModuleList) and len(model.blocks) == 2
    assert all(isinstance(b, TransformerBlock) for b in model.blocks)
    assert all(b.attn.n_heads == 2 for b in model.blocks)
    assert isinstance(model.ln_f, LayerNorm) and model.ln_f.eps == 1e-5
    assert isinstance(model.lm_head, nn.Linear) and model.lm_head.bias is None
    H.assert_shape(model.lm_head.weight, (7, 6))
    assert model.token_emb.weight.data_ptr() != model.lm_head.weight.data_ptr(), "本关不共享输入输出权重"
    block_params = [dict(b.named_parameters()) for b in model.blocks]
    assert model.blocks[0] is not model.blocks[1], "重复同一个块会共享权重"
    for name in block_params[0]:
        assert block_params[0][name].data_ptr() != block_params[1][name].data_ptr(), name
    assert sum(p.numel() for p in model.parameters()) == 2 * 7 * 6 + 2 * (12 * 6 * 6 + 9 * 6) + 2 * 6


def test_02_position_buffer():
    """02 正弦位置表是持久化 buffer，不是参数，奇数 C 也正确"""
    model = make(d_model=5, n_heads=1)
    assert "pe" in dict(model.named_buffers()), "用 register_buffer 注册 pe"
    assert "pe" not in dict(model.named_parameters()) and not model.pe.requires_grad
    assert "pe" in model.state_dict(), "本关要求保存位置表"
    want = torch.tensor([[(math.sin if c % 2 == 0 else math.cos)(
        t / 10000 ** (2 * (c // 2) / 5)) for c in range(5)] for t in range(6)])
    H.assert_close(model.pe, want, what="位置编码表", atol=1e-6, rtol=1e-6)


def test_03_invalid_configuration():
    """03 非正配置或不整除的头数抛 ValueError"""
    for name in ("vocab_size", "block_size", "d_model", "n_heads", "n_layers"):
        for value in (0, -1):
            with pytest.raises(ValueError):
                make(**{name: value})
    with pytest.raises(ValueError):
        make(d_model=5, n_heads=2)


def test_04_lengths_and_shape():
    """04 支持 T=1 到最大长度，空序列和超长序列抛 ValueError"""
    model = make().double()
    for t in (1, 3, 6):
        out = model(torch.arange(2 * t).reshape(2, t) % 7)
        H.assert_shape(out, (2, t, 7))
        H.assert_dtype(out, torch.float64)
        H.assert_finite(out)
    for t in (0, 7):
        with pytest.raises(ValueError):
            model(torch.zeros(1, t, dtype=torch.long))


def test_05_logits_reference():
    """05 完整 logits 与独立公式一致，检查位置、各块顺序、最终 LN 和 head"""
    for c, h, n in ((6, 2, 2), (5, 1, 1), (1, 1, 1)):
        model = fixed(make(d_model=c, n_heads=h, n_layers=n))
        idx = torch.arange(8).reshape(2, 4) % 7
        want = reference(idx, dict(model.named_parameters()), model.pe, h, n)
        H.assert_close(model(idx), want, what="完整 GPT logits", case=f"C={c},H={h},N={n}")


def test_06_causal_prefix():
    """06 未来 token 不改变前缀 logits，单独计算前缀也一致"""
    model = fixed(make())
    idx = torch.tensor([[0, 1, 2, 3, 4], [1, 3, 5, 0, 2]])
    changed = idx.clone()
    changed[:, 3:] = (changed[:, 3:] + 2) % 7
    base = model(idx)[:, :3]
    H.assert_close(model(changed)[:, :3], base, what="修改未来后的前缀", atol=1e-12, rtol=1e-12)
    H.assert_close(model(idx[:, :3]), base, what="单独计算的前缀", atol=1e-10, rtol=1e-10)


def test_07_aligned_loss():
    """07 loss 使用所有已对齐位置，不再次移位，也不只算最后一项"""
    logits = torch.tensor([[[6., 0., -1.], [0., 4., -2.], [-1., 0., 2.]],
                           [[0., 3., -1.], [-2., 1., 0.], [2., 0., -3.]]], dtype=torch.float64)
    targets = torch.tensor([[0, 1, 2], [2, 1, 0]])
    out = loss(logits, targets)
    H.assert_shape(out, ())
    H.assert_close(out, F.cross_entropy(logits.reshape(-1, 3), targets.reshape(-1)),
                   what="所有 B*T 位置的平均交叉熵")
    H.assert_close(loss(torch.zeros_like(logits), targets), torch.tensor(math.log(3), dtype=logits.dtype))


def test_08_loss_gradient_layout():
    """08 loss 支持非连续输入，logits 梯度与独立参照一致，不改输入"""
    logits = H.randn(2, 5, 3, seed=1020).transpose(1, 2).requires_grad_()
    targets = (torch.arange(6).reshape(3, 2) % 5).T
    before, target_before = logits.detach().clone(), targets.clone()
    ref = logits.detach().clone().requires_grad_()
    actual = loss(logits, targets)
    want = F.cross_entropy(ref.reshape(-1, 5), targets.reshape(-1))
    H.assert_close(actual, want, what="非连续 logits 的 loss")
    got_grad, = torch.autograd.grad(actual, logits)
    ref_grad, = torch.autograd.grad(want, ref)
    H.assert_close(got_grad, ref_grad, what="loss 对 logits 的梯度")
    H.assert_equal_tensor(logits, before)
    H.assert_equal_tensor(targets, target_before)


def test_09_all_parameter_gradients():
    """09 从交叉熵到所有层参数的完整梯度与独立参照一致"""
    model = fixed(make(vocab_size=5, d_model=4, n_heads=2, n_layers=2))
    idx = torch.arange(6).reshape(2, 3) % 5
    targets = (idx + 1) % 5
    params = dict(model.named_parameters())
    copies = {k: p.detach().clone().requires_grad_() for k, p in params.items()}
    actual = loss(model(idx), targets)
    expected_logits = reference(idx, copies, model.pe, 2, 2)
    expected_loss = F.cross_entropy(expected_logits.reshape(-1, 5), targets.reshape(-1))
    H.assert_close(actual, expected_loss, what="完整模型的 loss")
    got = torch.autograd.grad(actual, list(params.values()))
    want = torch.autograd.grad(expected_loss, list(copies.values()))
    for name, a, e in zip(params, got, want):
        H.assert_close(a, e, what=f"{name} 梯度")


def test_10_dtype_and_layout():
    """10 模型和位置表一起转换 dtype，支持非连续 token 输入"""
    model = make()
    idx = (torch.arange(8).reshape(4, 2) % 7).T
    before = idx.clone()
    for dtype in (torch.float32, torch.float64):
        model = model.to(dtype=dtype)
        H.assert_dtype(model.pe, dtype, what="位置表")
        out = model(idx)
        H.assert_dtype(out, dtype)
        assert out.device == idx.device
        H.assert_close(out, model(idx.contiguous()), what="非连续输入输出", atol=2e-6, rtol=2e-6)
        H.assert_equal_tensor(idx, before)


def test_11_state_dict_roundtrip():
    """11 state_dict 能恢复完整模型，前向不重新创建参数或修改位置表"""
    model = fixed(make())
    params = dict(model.named_parameters())
    saved = {k: v.clone() for k, v in model.state_dict().items()}
    other = make().double()
    other.load_state_dict(saved)
    idx = torch.tensor([[0, 4, 2, 1]])
    H.assert_close(other(idx), model(idx), what="恢复后的 logits")
    assert all(dict(model.named_parameters()).get(n) is p for n, p in params.items())
    for name, value in model.state_dict().items():
        H.assert_equal_tensor(value, saved[name], what=f"前向后的 {name}")


def test_12_reuse_functions(monkeypatch):
    """12 复用第五讲位置编码与第二讲交叉熵"""
    original_pe, original_ce = M.sinusoidal_pe, M.cross_entropy
    seen = {"pe": [], "ce": []}

    def track_pe(t, d_model):
        seen["pe"].append((t, d_model))
        return original_pe(t, d_model)

    def track_ce(logits, targets):
        seen["ce"].append((tuple(logits.shape), tuple(targets.shape)))
        return original_ce(logits, targets)

    monkeypatch.setattr(M, "sinusoidal_pe", track_pe)
    monkeypatch.setattr(M, "cross_entropy", track_ce)
    make()
    loss(torch.zeros(2, 3, 7), torch.zeros(2, 3, dtype=torch.long))
    assert seen["pe"] == [(6, 6)], "构造时复用 sinusoidal_pe(S,C)"
    assert seen["ce"] == [((6, 7), (6,))], "将 B*T 个样本交给 L02 cross_entropy"


def test_13_tiny_training():
    """13 小模型能在固定的循环字符 batch 上降低 loss 并学会预测"""
    torch.manual_seed(1040)
    model = make(vocab_size=4, block_size=4, d_model=8, n_heads=2, n_layers=1)
    idx = (torch.arange(4)[None, :] + torch.arange(4)[:, None]) % 4
    targets = (idx + 1) % 4
    optimizer = torch.optim.Adam(model.parameters(), lr=0.02)
    initial = loss(model(idx), targets).item()
    for _ in range(60):
        optimizer.zero_grad()
        value = loss(model(idx), targets)
        value.backward()
        optimizer.step()
    with torch.no_grad():
        logits = model(idx)
        final = loss(logits, targets).item()
        accuracy = (logits.argmax(-1) == targets).float().mean().item()
    assert final < initial * 0.5 and accuracy >= 0.9, (
        f"固定 batch 未学会：loss {initial:.4f}→{final:.4f}，准确率 {accuracy:.1%}"
    )


def test_14_no_shortcuts():
    """14 不调用现成 Transformer/交叉熵，不对 logits 先做 softmax"""
    H.assert_no_builtin(M, ["nn.Transformer", "nn.CrossEntropyLoss", "F.cross_entropy",
                            "functional.cross_entropy", "torch.softmax", ".softmax("])
    H.assert_no_loops(M, ["language_model_loss"])
