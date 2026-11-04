"""6B1：每 token 特征归一化、可训练仿射参数和梯度。"""

import math

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

import harness as H
from src import l06b1_layernorm as M


def make(c, eps=1e-5):
    return H.make(M, "LayerNorm", c, eps=eps)


def configured(c, eps=1e-5):
    layer = make(c, eps).double()
    with torch.no_grad():
        layer.weight.copy_(H.randn(c, seed=801))
        layer.bias.copy_(H.randn(c, seed=802))
    return layer


def test_01_parameters():
    """01 weight/bias 是 (C,) 可训练参数，初值为 1/0，总参数量 2*C"""
    layer = make(5, eps=0.02)
    assert layer.eps == 0.02, "保存调用者传入的 eps"
    params = dict(layer.named_parameters())
    assert set(params) == {"weight", "bias"}, "只注册 weight/bias 两个参数"
    for name in ("weight", "bias"):
        p = getattr(layer, name)
        assert isinstance(p, nn.Parameter) and p.requires_grad, f"{name} 应是可训练 nn.Parameter"
        H.assert_shape(p, (5,), what=name)
    H.assert_equal_tensor(layer.weight, torch.ones(5), what="初始 gamma")
    H.assert_equal_tensor(layer.bias, torch.zeros(5), what="初始 beta")
    assert sum(p.numel() for p in layer.parameters()) == 10


def test_02_invalid_configuration():
    """02 非正宽度或 eps 抛 ValueError"""
    for c, eps in ((0, 1e-5), (-1, 1e-5), (3, 0), (3, -0.1)):
        with pytest.raises(ValueError):
            make(c, eps)


def test_03_hand_calculation():
    """03 [1,3] 的均值为 2、方差为 1，方差分母是 C"""
    layer = make(2, eps=0.25).double()
    x = torch.tensor([[[1., 3.]]], dtype=torch.float64)
    expected = torch.tensor([[[-1., 1.]]], dtype=x.dtype) / math.sqrt(1.25)
    H.assert_close(layer(x), expected, what="手算样例",
                   case="mu=2, var=1；检查方差除以 C，以及 eps 在根号内")


def test_04_reference_shapes():
    """04 多种形状、dtype 的输出与独立内置参照一致"""
    for dtype in (torch.float32, torch.float64):
        for shape in ((2, 3, 7), (1, 4, 2), (3, 1, 5)):
            layer = configured(shape[-1]).to(dtype=dtype)
            x = H.randn(*shape, seed=810, dtype=dtype)
            expected = F.layer_norm(x, (shape[-1],), layer.weight, layer.bias, layer.eps)
            out = layer(x)
            H.assert_shape(out, shape)
            H.assert_dtype(out, dtype)
            assert out.device == x.device, "输出 device 应与输入一致"
            tol = 2e-6 if dtype == torch.float32 else 1e-9
            H.assert_close(out, expected, what="LayerNorm 数值", atol=tol, rtol=tol)


def test_05_affine_parameters():
    """05 每通道 gamma/beta 参与输出，并广播到所有位置"""
    layer = make(3, eps=0.1).double()
    with torch.no_grad():
        layer.weight.copy_(torch.tensor([2., -1., 0.]))
        layer.bias.copy_(torch.tensor([1., 3., -4.]))
    x = torch.tensor([[[0., 2., 4.], [2., 4., 6.]]], dtype=torch.float64)
    z = torch.tensor([-2., 0., 2.], dtype=x.dtype) / math.sqrt(8 / 3 + 0.1)
    want = (z * layer.weight + layer.bias).expand_as(x)
    H.assert_close(layer(x), want, what="仿射变换后的输出")


def test_06_token_independence():
    """06 修改其他样本或位置，不影响当前 token 的输出"""
    layer = configured(4)
    x = H.randn(2, 3, 4, seed=820)
    before = layer(x)
    changed = x.clone()
    changed[0, 2] = torch.tensor([100., -20., 3., 7.])
    changed[1] = H.randn(3, 4, seed=821) * 10
    H.assert_close(layer(changed)[0, :2], before[0, :2], what="未修改 token 的输出",
                   atol=1e-12, rtol=1e-12,
                   case="只能沿 C 轴归一化；沿 T/B 或整块归一化会混入其他 token")


def test_07_constant_and_single_channel():
    """07 常量向量和 C=1 输出等于 beta，前向和反向均有限"""
    for c in (1, 5):
        layer = configured(c)
        x = torch.tensor([[[2.], [-3.]], [[7.], [0.]]], dtype=torch.float64)
        x = x.expand(2, 2, c).clone().requires_grad_()
        out = layer(x)
        H.assert_close(out, layer.bias.expand_as(x), what=f"常量输入 C={c}")
        probe = H.randn(2, 2, c, seed=830)
        grads = torch.autograd.grad((out * probe).sum(), (x, layer.weight, layer.bias))
        for grad in grads:
            H.assert_finite(grad, what="常量输入时的梯度")


def test_08_epsilon_inside_root():
    """08 方差很小时仍按 sqrt(var+eps) 计算"""
    layer = make(2, eps=0.01).double()
    x = torch.tensor([[[0., 0.002]]], dtype=torch.float64)
    want = torch.tensor([[[-0.001, 0.001]]], dtype=x.dtype) / math.sqrt(1e-6 + 0.01)
    H.assert_close(layer(x), want, what="小方差样例",
                   case="不要漏 eps，不要把 eps 放到 sqrt(var) 外面")


def test_09_shift_invariance():
    """09 每个 token 的所有通道加同一个常数，输出不变"""
    layer = configured(5)
    x = H.randn(2, 3, 5, seed=840)
    shifts = torch.tensor([[[2.], [-3.], [5.]], [[7.], [0.], [-1.]]], dtype=x.dtype)
    H.assert_close(layer(x + shifts), layer(x), what="平移前后输出")


def test_10_noncontiguous_no_mutation():
    """10 支持非连续输入，不修改输入或前向期间的参数"""
    layer = configured(5)
    x = H.randn(2, 5, 3, seed=850).transpose(1, 2)
    before = x.clone()
    params = {name: p.detach().clone() for name, p in layer.named_parameters()}
    out = layer(x)
    H.assert_close(out, layer(x.contiguous()), what="非连续/连续输入")
    H.assert_equal_tensor(x, before, what="调用后的输入")
    for name, p in layer.named_parameters():
        H.assert_equal_tensor(p, params[name], what=f"调用后的 {name}")


def test_11_backward_reference():
    """11 输入、gamma、beta 梯度与独立参照一致"""
    layer = configured(4, eps=0.03)
    x = H.randn(2, 3, 4, seed=860).requires_grad_()
    xr = x.detach().clone().requires_grad_()
    wr = layer.weight.detach().clone().requires_grad_()
    br = layer.bias.detach().clone().requires_grad_()
    probe = H.randn(2, 3, 4, seed=861)
    got = torch.autograd.grad((layer(x) * probe).sum(), (x, layer.weight, layer.bias))
    expected = F.layer_norm(xr, (4,), wr, br, layer.eps)
    want = torch.autograd.grad((expected * probe).sum(), (xr, wr, br))
    for name, a, e in zip(("x", "gamma", "beta"), got, want):
        H.assert_close(a, e, what=f"{name} 梯度")


def test_12_gradcheck():
    """12 输入反向梯度通过有限差分校验"""
    layer = configured(3, eps=0.02)
    x = H.randn(1, 2, 3, seed=870).requires_grad_()
    assert torch.autograd.gradcheck(layer, (x,), eps=1e-6, atol=1e-5, rtol=1e-3)


def test_13_train_eval_same():
    """13 train/eval 都使用当前 token 的统计量，参数保持原对象"""
    layer = configured(4)
    params = dict(layer.named_parameters())
    x = H.randn(2, 3, 4, seed=880)
    layer.train()
    a = layer(x)
    layer(H.randn(2, 3, 4, seed=881) * 20)
    layer.eval()
    H.assert_close(layer(x), a, what="train/eval 输出")
    assert all(dict(layer.named_parameters()).get(name) is p for name, p in params.items()), (
        "不要在 forward 里重新创建 Parameter，否则优化器会失去对当前参数的引用"
    )


def test_14_no_shortcuts():
    """14 用基础张量运算实现，没有内置归一化或循环"""
    H.assert_no_builtin(M, ["nn.LayerNorm", ".layer_norm(", ".batch_norm(",
                            ".normalize(", "nn.BatchNorm", "nn.GroupNorm",
                            "nn.InstanceNorm", "nn.RMSNorm", ".rms_norm("])
    H.assert_no_loops(M, ["__init__", "forward"])
