"""7B：随机窗口、优化器更新语义与自回归采样。"""
import copy

import pytest
import torch
import torch.nn as nn

import harness as H
from src import l07b_train_generate as M
from src.l07a_gpt import MiniGPT, language_model_loss


def batch(data, b, t):
    return H.call(M, "get_batch", data, b, t)


def train(model, data, steps=2, batch_size=2, block_size=3, lr=0.003):
    return H.call(M, "train_language_model", model, data, steps, batch_size, block_size, lr)


def generate(model, idx, n, temperature=1.0):
    return H.call(M, "generate", model, idx, n, temperature)


def tiny():
    return MiniGPT(5, 4, 8, 2, 1)


class ProbeLM(nn.Module):
    """每个位置偏好其输入 token+1，记录实际输入和运行环境。"""
    def __init__(self, block_size=3, vocab_size=5):
        super().__init__()
        self.block_size, self.vocab_size = block_size, vocab_size
        self.anchor = nn.Parameter(torch.zeros(()))
        self.calls = []

    def forward(self, idx):
        assert idx.shape[1] <= self.block_size, "生成时未裁剪上下文"
        self.calls.append((idx.clone(), self.training, torch.is_grad_enabled()))
        logits = torch.full((*idx.shape, self.vocab_size), -5., device=idx.device)
        logits.scatter_(-1, ((idx + 1) % self.vocab_size).unsqueeze(-1), 5.)
        return logits + self.anchor


class ConstantLM(ProbeLM):
    def __init__(self):
        super().__init__(block_size=4, vocab_size=4)

    def forward(self, idx):
        super().forward(idx)
        return torch.tensor([0., 0.1, 0.2, -0.3], device=idx.device).expand(*idx.shape, 4) + self.anchor


def test_01_batch_values():
    """01 随机窗口连续、标签右移一位、dtype/device 正确且语料未修改"""
    data = torch.arange(17)
    before = data.clone()
    x, y = batch(data, 8, 5)
    H.assert_shape(x, (8, 5))
    H.assert_shape(y, (8, 5))
    H.assert_dtype(x, torch.long)
    H.assert_dtype(y, torch.long)
    assert x.device == y.device == data.device
    H.assert_equal_tensor(x[:, 1:] - x[:, :-1], torch.ones(8, 4, dtype=torch.long))
    H.assert_equal_tensor(y, x + 1)
    assert int(x.min()) >= 0 and int(y.max()) < len(data)
    H.assert_equal_tensor(data, before)


def test_02_batch_boundaries_seed():
    """02 最短语料、T=1、最后合法起点和相同种子可复现"""
    x, y = batch(torch.arange(5), 3, 4)
    H.assert_equal_tensor(x, torch.arange(4).expand(3, 4))
    H.assert_equal_tensor(y, torch.arange(1, 5).expand(3, 4))
    torch.manual_seed(1101)
    a = batch(torch.arange(6), 256, 4)
    assert set(a[0][:, 0].tolist()) == {0, 1}, "必须允许最后一个合法起点"
    torch.manual_seed(1101)
    b = batch(torch.arange(6), 256, 4)
    H.assert_equal_tensor(a[0], b[0])
    x, y = batch(torch.arange(4), 5, 1)
    H.assert_shape(x, (5, 1))
    H.assert_equal_tensor(y, x + 1)


def test_03_batch_invalid():
    """03 非正 B/T 或语料不足时抛 ValueError"""
    for length, b, t in ((5, 0, 2), (5, -1, 2), (5, 2, 0), (5, 2, -1), (3, 2, 3)):
        with pytest.raises(ValueError):
            batch(torch.arange(length), b, t)


def test_04_exact_two_updates(monkeypatch):
    """04 两步 AdamW 更新与独立训练过程一致，检查清梯度、重采样及优化器状态"""
    torch.manual_seed(1110)
    model = tiny().double().eval()
    ref = copy.deepcopy(model).train()
    data = torch.arange(20) % 5
    before_data = data.clone()
    xs = [torch.tensor([[0, 1, 2], [1, 2, 3]]), torch.tensor([[4, 0, 1], [2, 3, 4]])]
    ys = [(x + 1) % 5 for x in xs]
    seen = {"batch": 0, "loss": 0}
    original_loss = M.language_model_loss

    def scripted_batch(given, batch_size, block_size):
        assert given is data and (batch_size, block_size) == (2, 3)
        i = seen["batch"]
        seen["batch"] += 1
        assert i < 2, "每个训练 step 取一次 batch"
        return xs[i].clone(), ys[i].clone()

    def counted_loss(logits, targets):
        seen["loss"] += 1
        return original_loss(logits, targets)

    monkeypatch.setattr(M, "get_batch", scripted_batch)
    monkeypatch.setattr(M, "language_model_loss", counted_loss)
    # 旧梯度故意非零：训练首步也必须清理，不能仅在步尾清理。
    for p in model.parameters():
        p.grad = torch.full_like(p, 0.25)
    optimizer = torch.optim.AdamW(ref.parameters(), lr=0.003, weight_decay=0.01)
    expected = []
    for x, y in zip(xs, ys):
        optimizer.zero_grad()
        value = language_model_loss(ref(x), y)
        expected.append(value.item())
        value.backward()
        optimizer.step()
    got = train(model, data)
    assert isinstance(got, list) and len(got) == 2 and all(isinstance(v, float) for v in got)
    H.assert_close(torch.tensor(got, dtype=torch.float64), torch.tensor(expected, dtype=torch.float64),
                   what="每步更新前的 loss")
    assert seen == {"batch": 2, "loss": 2}, "每步调用 get_batch 和课程 loss"
    assert model.training, "训练函数应进入 train 模式"
    for (name, p), (_, expected_p) in zip(model.named_parameters(), ref.named_parameters()):
        H.assert_close(p, expected_p, what=f"两步后参数 {name}")
    H.assert_equal_tensor(data, before_data)


def test_05_training_boundaries():
    """05 零步不改参数；非法步数、窗口、lr 或语料抛 ValueError"""
    model = tiny()
    data = torch.arange(20) % 5
    before = {n: p.clone() for n, p in model.named_parameters()}
    assert train(model, data, steps=0) == []
    for n, p in model.named_parameters():
        H.assert_equal_tensor(p, before[n])
    for kwargs in (dict(steps=-1), dict(batch_size=0), dict(block_size=0),
                   dict(block_size=5), dict(lr=0), dict(lr=-0.1)):
        with pytest.raises(ValueError):
            train(model, data, **kwargs)
    with pytest.raises(ValueError):
        train(model, data[:3], block_size=3)


def test_06_real_training_and_generation():
    """06 真实 Mini-GPT 学会循环字符，并可生成超过窗口长度的续文"""
    torch.manual_seed(1120)
    model = MiniGPT(4, 4, 8, 2, 1)
    data = torch.arange(128) % 4
    idx = (torch.arange(4)[None] + torch.arange(4)[:, None]) % 4
    with torch.no_grad():
        initial = language_model_loss(model(idx), (idx + 1) % 4).item()
    history = train(model, data, steps=70, batch_size=8, block_size=4, lr=0.02)
    assert len(history) == 70 and all(isinstance(v, float) for v in history)
    with torch.no_grad():
        final = language_model_loss(model(idx), (idx + 1) % 4).item()
    assert final < 0.25 * initial, f"循环语料 loss 未显著下降：{initial:.4f}→{final:.4f}"
    prefix = torch.tensor([[0, 1]])
    before = {n: v.clone() for n, v in model.state_dict().items()}
    out = generate(model, prefix, 10, 0)
    H.assert_equal_tensor(out, (torch.arange(12) % 4)[None], what="学会循环后的贪婪续文")
    for n, v in model.state_dict().items():
        H.assert_equal_tensor(v, before[n], what=f"生成后的 {n}")


def test_07_greedy_context():
    """07 只取最后位置，每步使用最近 S 个 token，输出保留完整前缀"""
    model = ProbeLM()
    prefix = torch.tensor([[0, 2, 1, 3, 0], [4, 1, 3, 2, 4]])
    before = prefix.clone()
    out = generate(model, prefix, 6, 0)
    H.assert_shape(out, (2, 11))
    H.assert_dtype(out, torch.long)
    assert out.device == prefix.device
    expected = prefix.clone()
    assert len(model.calls) == 6
    for seen, _, _ in model.calls:
        H.assert_equal_tensor(seen, expected[:, -3:], what="本步模型输入")
        expected = torch.cat([expected, (expected[:, -1:] + 1) % 5], dim=1)
    H.assert_equal_tensor(out, expected, what="完整生成序列")
    H.assert_equal_tensor(prefix, before)


def test_08_temperature_and_sample(monkeypatch):
    """08 温度作用于 logits，概率正确，并使用 multinomial 的返回结果"""
    original_softmax = M.softmax
    for temperature in (0.5, 2.0):
        seen = {"softmax": 0, "sample": 0}

        def tracked_softmax(*args, **kwargs):
            seen["softmax"] += 1
            return original_softmax(*args, **kwargs)

        def fake_sample(probs, num_samples, **kwargs):
            seen["sample"] += 1
            want = torch.softmax(torch.tensor([0., 0.1, 0.2, -0.3]) / temperature, -1).expand(2, 4)
            H.assert_close(probs, want, what="温度调整后的采样概率", atol=1e-6, rtol=1e-6)
            assert num_samples == 1
            return torch.tensor([[2], [0]], device=probs.device)

        monkeypatch.setattr(M, "softmax", tracked_softmax)
        monkeypatch.setattr(torch, "multinomial", fake_sample)
        prefix = torch.tensor([[0, 1], [2, 3]])
        out = generate(ConstantLM(), prefix, 1, temperature)
        H.assert_equal_tensor(out[:, -1], torch.tensor([2, 0]))
        assert seen == {"softmax": 1, "sample": 1}, "正温度需调用课程 softmax 与 multinomial"


def test_09_sampling_reference_seed():
    """09 固定种子的多步随机采样与独立概率参照一致"""
    prefix = torch.tensor([[0], [2]])
    probs = torch.softmax(torch.tensor([0., 0.1, 0.2, -0.3]) / 0.7, -1).expand(2, 4)
    torch.manual_seed(1130)
    expected = torch.cat([prefix] + [torch.multinomial(probs, 1) for _ in range(8)], dim=1)
    torch.manual_seed(1130)
    out = generate(ConstantLM(), prefix, 8, 0.7)
    H.assert_equal_tensor(out, expected)
    torch.manual_seed(1130)
    H.assert_equal_tensor(generate(ConstantLM(), prefix, 8, 0.7), out)


def test_10_mode_and_no_grad():
    """10 生成期间 eval/no_grad，结束后恢复原 train/eval 状态"""
    for mode in (True, False):
        model = ProbeLM().train(mode)
        generate(model, torch.tensor([[1, 2]]), 3, 0)
        assert model.training == mode, "生成后应恢复调用前的模式"
        assert all(not training and not grad_enabled for _, training, grad_enabled in model.calls), (
            "eval 不会自动关闭梯度，必须同时禁用 autograd"
        )


def test_11_restore_on_error():
    """11 模型前向发生异常时也恢复原模式"""
    class BrokenLM(ProbeLM):
        def forward(self, idx):
            raise RuntimeError("deliberate forward failure")

    model = BrokenLM().train()
    with pytest.raises(RuntimeError, match="deliberate forward failure"):
        generate(model, torch.tensor([[1]]), 1, 0)
    assert model.training, "可用 try/finally 恢复模式"


def test_12_generation_boundaries():
    """12 生成零步不调用模型；空前缀、负步数或负温度抛 ValueError"""
    model = ProbeLM().train()
    prefix = torch.tensor([[1, 2, 3, 4]])
    H.assert_equal_tensor(generate(model, prefix, 0, 1), prefix)
    assert model.calls == [] and model.training
    for idx, n, temp in ((prefix, -1, 1), (prefix, 1, -0.1), (prefix[:, :0], 1, 1)):
        with pytest.raises(ValueError):
            generate(model, idx, n, temp)


def test_13_short_prefix():
    """13 单 token 前缀可逐步增长到满窗口，非连续前缀也可生成"""
    model = ProbeLM()
    prefix = torch.tensor([[2], [4]])
    out = generate(model, prefix, 5, 0)
    want = (prefix + torch.arange(6)[None]) % 5
    H.assert_equal_tensor(out, want)
    assert [x.shape[1] for x, _, _ in model.calls] == [1, 2, 3, 3, 3]
    noncontiguous = torch.tensor([[1, 3], [2, 4], [0, 2]]).T
    out = generate(ProbeLM(), noncontiguous, 1, 0)
    H.assert_equal_tensor(out[:, :3], noncontiguous)


def test_14_no_shortcuts():
    """14 复用课程损失/softmax，批量取样不逐 token 循环"""
    H.assert_no_builtin(M, ["F.cross_entropy", "functional.cross_entropy", "nn.CrossEntropyLoss",
                            "torch.softmax", ".softmax("])
    H.assert_no_loops(M, ["get_batch"])


def test_15_tiny_positive_temperature(monkeypatch):
    """15 很小的正温度仍须采样：并列 logits 不会变成唯一最大值"""
    class TiedLM(ProbeLM):
        def __init__(self):
            super().__init__(block_size=2, vocab_size=2)

        def forward(self, idx):
            super().forward(idx)
            return torch.zeros((*idx.shape, 2), device=idx.device) + self.anchor

    calls = []

    def choose_second(probs, num_samples, **kwargs):
        calls.append(probs.clone())
        H.assert_close(probs, torch.tensor([[0.5, 0.5]], device=probs.device),
                       what="并列分数在正温度下的概率")
        assert num_samples == 1
        return torch.ones((1, 1), dtype=torch.long, device=probs.device)

    monkeypatch.setattr(torch, "multinomial", choose_second)
    prefix = torch.tensor([[1]])
    greedy = generate(TiedLM(), prefix, 1, 0)
    assert greedy[0, -1].item() == 0 and not calls, "零温度应直接 argmax"
    sampled = generate(TiedLM(), prefix, 1, 1e-9)
    assert len(calls) == 1, (
        "temperature=1e-9 仍为正数，应调用 multinomial。"
        "logits=[0,0] 在任何正温度下都是 [0.5,0.5]，不能用 argmax 替代。"
    )
    assert sampled[0, -1].item() == 1, "应使用采样返回的 token"
