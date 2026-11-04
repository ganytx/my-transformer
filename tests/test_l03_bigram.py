"""L03 检查 · 字符级 bigram 语言模型

从"能不能正确编码"到"训练完是不是真的学到了字符转移"。
"""

from __future__ import annotations

import math

import torch

import harness as H
from src import l03_bigram as M

TEXT = (H.ROOT / "data" / "alice.txt").read_text("utf-8")
_CACHE: dict = {}


def build_vocab(text=TEXT):
    out = H.call(M, "build_vocab", text)
    if not (isinstance(out, tuple) and len(out) == 2):
        raise AssertionError(f"build_vocab 应该返回 (stoi, itos) 两个字典，实际返回 {type(out).__name__}")
    return out


def encode(text, stoi):
    return H.call(M, "encode", text, stoi)


def decode(ids, itos):
    return H.call(M, "decode", ids, itos)


def make_model(vocab_size, seed=0):
    torch.manual_seed(seed)
    return H.call(M, "BigramLM", vocab_size)


def trained():
    """整套检查共用一次训练结果，避免重复跑 2000 步。"""
    if "model" not in _CACHE:
        stoi, itos = build_vocab()
        torch.manual_seed(0)
        model = H.call(M, "BigramLM", len(stoi))
        hist = H.call(M, "train_bigram", model, H.call(M, "encode", TEXT[:60000], stoi),
                      2000, 32, 8, 0.5)
        _CACHE.update(model=model, hist=hist, stoi=stoi, itos=itos)
    return _CACHE["model"], _CACHE["hist"], _CACHE["stoi"], _CACHE["itos"]


# =========================================================================== #
def test_01_build_vocab():
    """01 词表覆盖全部字符，且两个映射互逆"""
    stoi, itos = build_vocab()
    chars = set(TEXT)
    missing = chars - set(stoi)
    if missing:
        raise AssertionError(f"词表漏了 {len(missing)} 个字符，例如 {sorted(missing)[:5]}。")
    if len(stoi) != len(chars):
        raise AssertionError(f"词表大小应该是 {len(chars)}（去重后的字符数），实际 {len(stoi)}。")
    if set(stoi.values()) != set(range(len(stoi))):
        raise AssertionError("stoi 的值必须是 0..V-1 连续整数，不能有空洞或重复。")
    for ch, i in list(stoi.items())[:50]:
        if itos.get(i) != ch:
            raise AssertionError(f"stoi 和 itos 对不上：stoi[{ch!r}]={i}，但 itos[{i}]={itos.get(i)!r}。")


def test_02_encode_decode():
    """02 encode/decode 往返一致，dtype 是 long"""
    stoi, itos = build_vocab()
    s = TEXT[:200]
    ids = encode(s, stoi)
    H.expect_tensor(ids, "encode 的返回值")
    H.assert_shape(ids, (len(s),), "encode 的返回值")
    H.assert_dtype(ids, torch.long, "encode 的返回值")
    if decode(ids, itos) != s:
        raise AssertionError("decode(encode(s)) 不等于 s。检查两个表是不是用反了。")


def test_03_forward_shape():
    """03 前向输出 (B,T,V)，参数量正好 V*V"""
    v = 20
    m = make_model(v)
    idx = torch.randint(0, v, (4, 7))
    logits = m(idx)
    H.expect_tensor(logits, "BigramLM 的输出")
    H.assert_shape(logits, (4, 7, v), "BigramLM 的输出")
    total = sum(p.numel() for p in m.parameters())
    if total != v * v:
        raise AssertionError(
            f"参数量应该是 V*V = {v * v}，实际 {total}。\n"
            f"  这个模型只有一个 V×V 的表，没有 bias、没有隐藏层。"
        )


def test_04_ignores_context():
    """04 只依赖当前字符：同一个字符的不同出现位置，logits 必须相同"""
    v = 20
    m = make_model(v)
    idx = torch.tensor([[3, 5, 7],
                        [9, 5, 1]])          # 第 0 行第 1 列、第 1 行第 1 列都是 5
    logits = m(idx)
    a, b = logits[0, 1], logits[1, 1]
    if not torch.allclose(a, b, atol=1e-6):
        raise AssertionError(
            "同一个字符 5 在两个不同上下文里得到了不同的 logits。\n"
            "  bigram 的定义就是「下一个字符只取决于当前字符」，所以这两个位置必须完全一样。\n"
            "  检查你的前向是不是把整段 idx 当成输入喂进了什么会混上下文的东西（比如加了一层隐藏层或平均）。"
        )
    # 另外验证：位置不同但字符相同，结果也应相同
    idx2 = torch.tensor([[5, 5, 5]])
    lg = m(idx2)
    if not torch.allclose(lg[0, 0], lg[0, 2], atol=1e-6):
        raise AssertionError("同一行里两个相同的字符得到了不同的 logits，前向有问题。")


def test_05_training_reduces_loss():
    """05 训练 2000 步后 loss 显著下降"""
    _, hist, stoi, _ = trained()
    if not isinstance(hist, list) or len(hist) != 2000:
        raise AssertionError(f"train_bigram 应该返回 2000 个 float，实际 {type(hist).__name__} / {len(hist) if hasattr(hist,'__len__') else '?'}")
    if not all(isinstance(x, float) for x in hist):
        raise AssertionError("loss 列表里应该是 Python float，不是张量。")
    first, last = hist[0], hist[-1]
    base = math.log(len(stoi))
    if first > base * 1.15:
        raise AssertionError(
            f"初始 loss {first:.3f} 比 ln(V)={base:.3f} 还大不少，说明初始化或损失算错了。"
        )
    if last > 3.2:
        raise AssertionError(
            f"训练 2000 步后 loss 还有 {last:.3f}（初始 {first:.4f}，ln(V)={base:.3f}）。\n"
            f"  bigram 在这份数据上应该能压到 2.8 左右。检查：学习率是不是太小、batch 抽样范围对不对、"
            f"损失有没有对 batch 取平均。"
        )
    if last > 0.8 * first:
        raise AssertionError(f"loss 从 {first:.3f} 只降到 {last:.3f}，训练没有真正起作用。")


def test_06_learned_bigram_distribution():
    """06 训练后每个字符的 top-1 预测与真实语料的 bigram 统计基本一致"""
    model, _, stoi, _ = trained()
    v = len(stoi)
    data = encode(TEXT[:60000], stoi)
    counts = torch.zeros(v, v)
    counts.index_put_((data[:-1], data[1:]), torch.ones(len(data) - 1), accumulate=True)
    emp = counts.argmax(dim=1)
    idx = torch.arange(v).reshape(1, v)
    with torch.no_grad():
        pred = model(idx)[0].argmax(dim=1)
    mask = counts.sum(dim=1) >= 20
    agree = (emp[mask] == pred[mask]).float().mean().item()
    if agree < 0.75:
        raise AssertionError(
            f"模型对「当前字符后面最可能跟什么」的判断，只有 {agree:.1%} 与语料统计一致。\n"
            f"  这通常意味着：训练步数不够、学习率太小，或者 batch 的 x/y 错位了一格。"
        )


def test_07_sample_shape_and_prefix():
    """07 采样输出 (B, T+n)，原有前缀原样保留"""
    v = 20
    m = make_model(v, seed=3)
    idx = torch.tensor([[1, 2, 3], [4, 5, 6]])
    torch.manual_seed(0)
    out = H.call(M, "sample", m, idx, 12, 1.0)
    H.expect_tensor(out, "sample 的返回值")
    H.assert_shape(out, (2, 15), "sample 的返回值")
    H.assert_dtype(out, torch.long, "sample 的返回值")
    if not torch.equal(out[:, :3], idx):
        raise AssertionError("采样把给定的前缀改掉了，前缀必须原样保留。")
    if (out < 0).any() or (out >= v).any():
        raise AssertionError("采样结果的取值超出了词表范围。")


def test_08_temperature_zero_is_greedy():
    """08 温度趋近 0 时，采样等于贪婪解码"""
    v = 20
    m = make_model(v, seed=4)
    torch.manual_seed(123)
    with torch.no_grad():                     # 让打分表有差异，否则温度无从体现
        for p in m.parameters():
            p.normal_(0.0, 3.0)

    def greedy(model, start, n):
        seq = [int(start)]
        with torch.no_grad():
            for _ in range(n):
                logits = model(torch.tensor([[seq[-1]]]))
                seq.append(int(logits[0, -1].argmax()))
        return seq

    start = 7
    torch.manual_seed(0)
    got = H.call(M, "sample", m, torch.tensor([[start]]), 25, 1e-3)
    want = greedy(m, start, 25)
    if got[0].tolist() != want:
        raise AssertionError(
            f"temperature=1e-3 时应该几乎必然选概率最大的字符，但结果不同。\n"
            f"  期望前几个：{want[:8]}\n  实际前几个：{got[0].tolist()[:8]}\n"
            f"  检查温度是不是除在了 logits 上（logits / temperature），而不是对概率做别的事。"
        )


def test_09_sample_deterministic():
    """09 同一个随机种子下，采样结果可复现"""
    v = 20
    m = make_model(v, seed=5)
    idx = torch.tensor([[2, 3]])
    torch.manual_seed(11)
    a = H.call(M, "sample", m, idx, 10, 1.0)
    torch.manual_seed(11)
    b = H.call(M, "sample", m, idx, 10, 1.0)
    if not torch.equal(a, b):
        raise AssertionError(
            "同一个种子两次采样结果不同。sampling 必须用 torch.multinomial（或等价的按概率抽样），"
            "不能用 argmax 或自己对概率做阈值判断。"
        )


def test_10_generated_text_is_in_vocab():
    """10 训练后生成 200 个字符，全部落在词表内且不是常数重复"""
    model, _, stoi, itos = trained()
    start = encode(TEXT[:1], stoi).reshape(1, 1)
    torch.manual_seed(0)
    out = H.call(M, "sample", model, start, 200, 0.8)
    ids = out[0].tolist()
    if any(i < 0 or i >= len(stoi) for i in ids):
        raise AssertionError("生成的 token 越界了。")
    s = decode(out[0], itos)
    if len(set(s[1:])) < 10:
        raise AssertionError(
            f"生成的文本只有 {len(set(s[1:]))} 种不同字符，说明采样退化成了一直输出同一个字符。\n"
            f"  输出片段：{s[:60]!r}"
        )


def test_11_no_builtin_loss():
    """11 损失用的是第 2 讲写的 cross_entropy，不是 PyTorch 现成的"""
    H.assert_no_builtin(
        M,
        ["F.cross_entropy", "functional.cross_entropy", "nn.CrossEntropyLoss",
         "F.nll_loss", "nn.NLLLoss", "torch.log_softmax"],
        why="第 2 讲那 17 项检查就是为这一天准备的，直接复用",
    )
