"""L04 检查 · 上下文窗口

核心问题：它到底有没有用上"前面的字符"？
"""

from __future__ import annotations

import torch
import torch.nn as nn

import harness as H
from src import l04_context as M

TEXT = (H.ROOT / "data" / "alice.txt").read_text("utf-8")[:60000]
_CACHE: dict = {}


def _vocab():
    if "stoi" not in _CACHE:
        stoi = {c: i for i, c in enumerate(sorted(set(TEXT)))}
        itos = {i: c for c, i in stoi.items()}
        data = torch.tensor([stoi[c] for c in TEXT], dtype=torch.long)
        _CACHE.update(stoi=stoi, itos=itos, data=data)
    return _CACHE["stoi"], _CACHE["itos"], _CACHE["data"]


def get_batch(data, bs, k, gen=None):
    return H.call(M, "get_batch", data, bs, k, gen)


def make_model(v, k, d=16, hid=128, seed=0):
    torch.manual_seed(seed)
    return H.call(M, "ContextLM", v, k, d, hid)


def trained():
    if "model" not in _CACHE:
        stoi, itos, data = _vocab()
        torch.manual_seed(0)
        model = H.call(M, "ContextLM", len(stoi), 8, 16, 128)
        hist = H.call(M, "train_context_model", model, data, 2000, 32, 8, 3e-3)
        _CACHE.update(model=model, hist=hist)
    return _CACHE["model"], _CACHE["hist"]


# =========================================================================== #
def test_01_get_batch_shape():
    """01 get_batch 返回 (B,k) 和 (B,)，dtype 是 long"""
    _, _, data = _vocab()
    x, y = get_batch(data, 16, 8)
    H.expect_tensor(x, "x")
    H.assert_shape(x, (16, 8), "x")
    H.assert_shape(y, (16,), "y")
    H.assert_dtype(x, torch.long, "x")
    H.assert_dtype(y, torch.long, "y")


def test_02_get_batch_content():
    """02 y 是窗口后面紧跟的那一个字符，x 是连续的一段"""
    data = torch.arange(100, dtype=torch.long)
    x, y = get_batch(data, 32, 5)
    if (x[:, 1:] - x[:, :-1] != 1).any():
        raise AssertionError(
            "x 的每一行必须是连续的一段（下标相差 1）。\n"
            "  常见错误：用 torch.stack 拼的时候步长写错，或者行内顺序乱了。"
        )
    if not torch.equal(y, x[:, -1] + 1):
        raise AssertionError(
            "y 必须正好是 x 窗口后面的那一个字符（y = x[:,-1] + 1）。\n"
            "  检查起点范围是不是 i + block_size + 1 <= len(data)，以及 y 取的是 data[i+block_size]。"
        )
    if int(x.min()) < 0 or int(x.max()) >= 100:
        raise AssertionError("窗口越界了：起点范围要保证 i + block_size + 1 <= len(data)。")


def test_03_forward_shape_and_embedding():
    """03 前向输出 (B,V)，并且真的有 Embedding 层"""
    v, k, d = 20, 8, 16
    m = make_model(v, k, d)
    if not any(isinstance(mod, nn.Embedding) for mod in m.modules()):
        raise AssertionError(
            "没找到 nn.Embedding。这一讲的重点就是「每个字符一个向量」，必须用 embedding 查表。\n"
            "  nn.Embedding(V, d) 等价于一张 (V, d) 的表，输入 (B,k) 的整数，输出 (B,k,d)。"
        )
    idx = torch.randint(0, v, (5, k))
    logits = m(idx)
    H.expect_tensor(logits, "ContextLM 的输出")
    H.assert_shape(logits, (5, v), "ContextLM 的输出")


def test_04_param_count():
    """04 参数量与公式一致"""
    v, k, d, hid = 20, 8, 16, 128
    m = make_model(v, k, d, hid)
    want = v * d + (k * d) * hid + hid + hid * v + v
    got = sum(p.numel() for p in m.parameters())
    if got != want:
        raise AssertionError(
            f"参数量应该是 {want}，实际 {got}。\n"
            f"  公式：V*d + (k*d)*n_hidden + n_hidden + n_hidden*V + V\n"
            f"       = {v}*{d} + ({k}*{d})*{hid} + {hid} + {hid}*{v} + {v}\n"
            f"  差得不多的话，多半是 fc1 的输入维度写成了 d 而不是 k*d —— 那就等于没用上上下文。"
        )


def test_05_uses_all_positions():
    """05 用的是整个窗口：同样结尾、不同更早字符，输出必须不同"""
    v, k = 20, 8
    m = make_model(v, k, seed=7)
    a = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 9]])
    b = torch.tensor([[8, 8, 8, 8, 8, 6, 7, 9]])       # 只改前面的位置，最后两个相同
    la, lb = m(a), m(b)
    if torch.allclose(la, lb, atol=1e-6):
        raise AssertionError(
            "改了窗口里的前几个字符，输出却完全没变——说明你只用了最后一个位置（那就是 bigram）。\n"
            "  前向里要把 (B,k,d) 拼成 (B,k*d) 再送进 fc1，不能只取 e[:, -1, :]。"
        )


def test_06_loss_beats_bigram():
    """06 训练 2000 步后 loss 明显低于 bigram 的 2.6"""
    _, hist = trained()
    if not isinstance(hist, list) or len(hist) != 2000:
        raise AssertionError(f"train_context_model 应返回 2000 个 float，实际 {len(hist) if hasattr(hist,'__len__') else '?'}")
    first, last = hist[0], hist[-1]
    if last > 2.5:
        raise AssertionError(
            f"训练 2000 步后 loss 是 {last:.3f}（初始 {first:.3f}）。\n"
            f"  bigram 在同样的数据上能到 2.60，一个看 8 个字符的模型应该明显低于它。\n"
            f"  低于 2.5 才算「用上了上下文」。如果卡在 3 以上：检查 embedding 维度、"
            f"fc1 的输入维度是不是 k*d、以及有没有真的把窗口拼起来。"
        )
    if last > 0.75 * first:
        raise AssertionError(f"loss 从 {first:.3f} 只降到 {last:.3f}，训练没起作用。")


def test_07_accuracy():
    """07 随机批上的下一个字符准确率超过 35%"""
    model, _ = trained()
    _, _, data = _vocab()
    g = torch.Generator().manual_seed(99)
    x, y = get_batch(data, 512, 8, g)
    with torch.no_grad():
        acc = (model(x).argmax(dim=-1) == y).float().mean().item()
    if acc < 0.35:
        raise AssertionError(
            f"准确率只有 {acc:.1%}。参考实现大约 45%；bigram 大约 30%。\n"
            f"  低于 35% 说明上下文没起作用，或者训练步数/学习率不对。"
        )


def test_08_sample_shape():
    """08 采样输出 (B, T+n)，前缀保留，取值不越界"""
    v, k = 20, 8
    m = make_model(v, k, seed=3)
    idx = torch.tensor([[1, 2, 3, 4, 5, 6, 7, 8, 9], [9, 8, 7, 6, 5, 4, 3, 2, 1]])
    torch.manual_seed(0)
    out = H.call(M, "sample_context", m, idx, 12, k, 1.0)
    H.expect_tensor(out, "sample_context 的返回值")
    H.assert_shape(out, (2, 21), "sample_context 的返回值")
    if not torch.equal(out[:, :9], idx):
        raise AssertionError("采样改掉了给定前缀，前缀必须原样保留。")
    if int(out.min()) < 0 or int(out.max()) >= v:
        raise AssertionError("采样结果越界了。")


def test_09_sample_uses_window():
    """09 采样确实用滑动窗口：第 9 步之后仍能继续跑（不会因长度不对报错）"""
    v, k = 20, 8
    m = make_model(v, k, seed=4)
    idx = torch.tensor([[3] * 9])
    torch.manual_seed(0)
    try:
        out = H.call(M, "sample_context", m, idx, 40, k, 0.9)
    except Exception as exc:  # noqa: BLE001
        raise AssertionError(
            f"生成长一点就报错了：{type(exc).__name__}: {exc}\n"
            f"  常见原因：把整个 out 都喂给模型，而模型只接受长度为 {k} 的输入（fc1 的维度是 k*d）。\n"
            f"  应该每次只取 out[:, -{k}:]。"
        ) from None
    H.assert_shape(out, (1, 49), "长采样的结果")


def test_10_no_builtin_loss():
    """10 损失复用自己写的 cross_entropy"""
    H.assert_no_builtin(
        M,
        ["F.cross_entropy", "functional.cross_entropy", "nn.CrossEntropyLoss",
         "F.nll_loss", "nn.NLLLoss"],
        why="第 2 讲那 17 项检查就是为这一天准备的",
    )


def test_11_short_start():
    """11 开头比窗口短时不报错（左边补第一个字符）"""
    v, k = 20, 8
    m = make_model(v, k, seed=6)
    idx = torch.tensor([[3, 4, 5]])                    # 只有 3 个字符，窗口是 8
    torch.manual_seed(0)
    try:
        out = H.call(M, "sample_context", m, idx, 9, k, 1.0)
    except Exception as exc:  # noqa: BLE001
        raise AssertionError(
            f"开头只有 3 个字符、窗口是 {k}，直接报错了：{type(exc).__name__}: {exc}\n"
            f"  模型永远要求输入长度正好是 {k}。开头不够长时，用第一个字符在左边重复补齐。\n"
            f"  例：idx = [[3,4,5]] -> [[3,3,3,3,3,3,4,5]] 再送进模型。\n"
            f"  注意返回值仍然是 (B, 原长度 + n)，补出来的部分不要出现在结果里。"
        ) from None
    H.assert_shape(out, (1, 12), "短开头的采样输出")
    if not torch.equal(out[:, :3], idx):
        raise AssertionError("前缀被改掉了。补齐只用于喂模型，返回结果里只保留原来的前缀。")
