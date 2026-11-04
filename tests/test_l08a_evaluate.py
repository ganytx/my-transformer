"""8A：确定的窗口覆盖、token 加权、评估状态保护。"""
import math

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

import harness as H
from src import l08a_evaluate as M
from src.l07a_gpt import MiniGPT


def batches(data, b, t):
    return H.call(M, "make_eval_batches", data, b, t)


def evaluate(model, values):
    return H.call(M, "evaluate", model, values)


class TableLM(nn.Module):
    def __init__(self, table):
        super().__init__()
        self.table = nn.Parameter(table.clone())
        self.calls = []

    def forward(self, idx):
        self.calls.append((self.training, torch.is_grad_enabled()))
        return self.table[idx]


def assert_report(report, loss, count):
    assert isinstance(report, dict) and {"loss", "perplexity", "n_tokens"} <= set(report)
    assert type(report["loss"]) is float and type(report["perplexity"]) is float
    assert type(report["n_tokens"]) is int and report["n_tokens"] == count
    assert report["loss"] == pytest.approx(loss, abs=1e-9, rel=1e-7)
    assert report["perplexity"] == pytest.approx(math.exp(loss), abs=1e-9, rel=1e-7)


def test_01_example():
    """01 按编号样例切分完整窗口和最后短窗口"""
    result = batches(torch.arange(12), 2, 4)
    assert isinstance(result, list) and len(result) == 2
    H.assert_equal_tensor(result[0][0], torch.tensor([[0, 1, 2, 3], [4, 5, 6, 7]]))
    H.assert_equal_tensor(result[0][1], torch.tensor([[1, 2, 3, 4], [5, 6, 7, 8]]))
    H.assert_equal_tensor(result[1][0], torch.tensor([[8, 9, 10]]))
    H.assert_equal_tensor(result[1][1], torch.tensor([[9, 10, 11]]))


def test_02_every_target_once():
    """02 每个目标恰好覆盖一次，完整小批和短窗口都不丢弃"""
    for length, b, t, shapes in (
        (2, 3, 8, [(1, 1)]), (13, 2, 4, [(2, 4), (1, 4)]),
        (20, 2, 4, [(2, 4), (2, 4), (1, 3)]), (6, 2, 1, [(2, 1), (2, 1), (1, 1)]),
    ):
        data = torch.arange(length)
        result = batches(data, b, t)
        assert [tuple(x.shape) for x, y in result] == shapes
        for x, y in result:
            H.assert_shape(y, x.shape)
            H.assert_dtype(x, torch.long)
            H.assert_dtype(y, torch.long)
            assert x.device == y.device == data.device
        H.assert_equal_tensor(torch.cat([x.reshape(-1) for x, _ in result]), data[:-1])
        H.assert_equal_tensor(torch.cat([y.reshape(-1) for _, y in result]), data[1:])


def test_03_deterministic_noncontiguous():
    """03 非连续语料也正确，切分不消耗随机数、不修改输入"""
    data = torch.arange(40)[::2]
    before, rng = data.clone(), torch.get_rng_state().clone()
    a = batches(data, 3, 4)
    H.assert_equal_tensor(torch.get_rng_state(), rng, what="切分后的 RNG 状态")
    b = batches(data, 3, 4)
    assert len(a) == len(b)
    for (ax, ay), (bx, by) in zip(a, b):
        H.assert_equal_tensor(ax, bx)
        H.assert_equal_tensor(ay, by)
    H.assert_equal_tensor(torch.cat([y.reshape(-1) for _, y in a]), data[1:])
    H.assert_equal_tensor(data, before)


def test_04_invalid_data():
    """04 太短语料、非正 batch 或窗口大小抛 ValueError"""
    for length, b, t in ((0, 2, 3), (1, 2, 3), (5, 0, 3), (5, -1, 3), (5, 2, 0), (5, 2, -1)):
        with pytest.raises(ValueError):
            batches(torch.arange(length), b, t)


def test_05_weighted_mean():
    """05 三个 loss=1 和一个 loss=3，整体 loss=1.5，不能平均 batch"""
    probs = torch.tensor([[math.exp(-1), 1 - math.exp(-1)],
                          [math.exp(-3), 1 - math.exp(-3)]], dtype=torch.float64)
    model = TableLM(probs.log())
    values = [(torch.tensor([[0, 0, 0]]), torch.tensor([[0, 0, 0]])),
              (torch.tensor([[1]]), torch.tensor([[0]]))]
    assert_report(evaluate(model, values), 1.5, 4)


def test_06_uniform_perplexity():
    """06 V 类均匀概率的 loss=ln(V)，perplexity=V"""
    model = TableLM(torch.zeros(5, 5, dtype=torch.float64))
    values = [(torch.tensor([[0, 1], [2, 3]]), torch.tensor([[4, 2], [0, 1]]))]
    assert_report(evaluate(model, values), math.log(5), 4)


def test_07_independent_reference():
    """07 不同 B/T 的批次与逐 token 负对数概率参照一致"""
    table = H.randn(5, 5, seed=1201)
    model = TableLM(table)
    values = [(torch.tensor([[0, 1, 2], [3, 4, 0]]), torch.tensor([[1, 4, 2], [0, 3, 1]])),
              (torch.tensor([[2, 3]]), torch.tensor([[4, 0]]))]
    log_probs = torch.log_softmax(table, dim=-1)
    total, count = 0., 0
    for x, y in values:
        for token, target in zip(x.flatten(), y.flatten()):
            total -= log_probs[token, target].item()
            count += 1
    assert_report(evaluate(model, values), total / count, count)


def test_08_modes_and_iterable():
    """08 可消费一次性 iterable，期间 eval/no_grad，恢复原模式"""
    for mode in (True, False):
        model = TableLM(torch.zeros(3, 3, dtype=torch.float64)).train(mode)
        values = ((torch.tensor([[0, 1]]), torch.tensor([[1, 2]])) for _ in range(3))
        assert_report(evaluate(model, values), math.log(3), 6)
        assert model.training == mode
        assert len(model.calls) == 3 and all(not m and not g for m, g in model.calls)


def test_09_invalid_and_error_restore():
    """09 空评估、空 batch 和前向异常时均正确处理并恢复模式"""
    model = TableLM(torch.zeros(3, 3, dtype=torch.float64)).train()
    for values in ([], [(torch.empty(1, 0, dtype=torch.long), torch.empty(1, 0, dtype=torch.long))],
                   [(torch.empty(0, 2, dtype=torch.long), torch.empty(0, 2, dtype=torch.long))]):
        with pytest.raises(ValueError):
            evaluate(model, values)
        assert model.training, "错误退出也需恢复模式"

    class Broken(TableLM):
        def forward(self, idx):
            raise RuntimeError("deliberate eval failure")

    broken = Broken(torch.zeros(2, 2)).train()
    with pytest.raises(RuntimeError, match="deliberate eval failure"):
        evaluate(broken, [(torch.tensor([[0]]), torch.tensor([[1]]))])
    assert broken.training


def test_10_no_side_effects():
    """10 评估不改参数、已有梯度、输入和 RNG 状态"""
    model = TableLM(H.randn(4, 4, seed=1210))
    model.table.grad = H.randn(4, 4, seed=1211)
    parameter, gradient = model.table.detach().clone(), model.table.grad.clone()
    gradient_object = model.table.grad
    x, y = torch.tensor([[0, 1, 2]]), torch.tensor([[1, 2, 3]])
    original_x, original_y = x.clone(), y.clone()
    rng = torch.get_rng_state().clone()
    evaluate(model, [(x, y)])
    H.assert_equal_tensor(model.table, parameter)
    assert model.table.grad is gradient_object, "评估不应清理或替换已有梯度"
    H.assert_equal_tensor(model.table.grad, gradient)
    H.assert_equal_tensor(x, original_x)
    H.assert_equal_tensor(y, original_y)
    H.assert_equal_tensor(torch.get_rng_state(), rng)


def test_11_real_model_batch_invariance():
    """11 真实 GPT 改 batch 分组不改窗口协议，结果应一致"""
    torch.manual_seed(1220)
    model = MiniGPT(5, 4, 8, 2, 1).double()
    data = torch.arange(23) % 5
    a = evaluate(model, batches(data, 1, 4))
    b = evaluate(model, batches(data, 4, 4))
    assert a["n_tokens"] == b["n_tokens"] == 22
    assert a["loss"] == pytest.approx(b["loss"], abs=1e-9, rel=1e-7)
    with torch.no_grad():
        sums = [F.cross_entropy(model(x).reshape(-1, 5), y.reshape(-1), reduction="sum").item()
                for x, y in batches(data, 2, 4)]
    assert_report(a, sum(sums) / 22, 22)


def test_12_large_perplexity():
    """12 极大有限 loss 的 perplexity 为正 inf，不抛 OverflowError"""
    model = TableLM(torch.tensor([[0., -1000.]], dtype=torch.float64))
    report = evaluate(model, [(torch.tensor([[0]]), torch.tensor([[1]]))])
    assert report["loss"] == pytest.approx(1000.) and report["n_tokens"] == 1
    assert type(report["perplexity"]) is float and report["perplexity"] == float("inf")


def test_13_reuse_loss(monkeypatch):
    """13 每批复用 language_model_loss，且能处理非连续 batch"""
    original = M.language_model_loss
    calls = []

    def tracked(logits, targets):
        calls.append(tuple(targets.shape))
        return original(logits, targets)

    monkeypatch.setattr(M, "language_model_loss", tracked)
    x = torch.tensor([[0, 1], [2, 3], [1, 0]]).T
    y = (x + 1) % 4
    model = TableLM(torch.zeros(4, 4, dtype=torch.float64))
    assert_report(evaluate(model, [(x, y)]), math.log(4), 6)
    assert calls == [(2, 3)]


def test_14_no_shortcuts():
    """14 不直接调用内置交叉熵或偷偷训练"""
    H.assert_no_builtin(M, ["F.cross_entropy", "functional.cross_entropy", "nn.CrossEntropyLoss",
                            ".backward(", "torch.optim."])
