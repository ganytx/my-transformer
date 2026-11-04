"""L06B1 · LayerNorm：每个 token 独立做特征归一化。

讲义：lessons/06b1-layernorm.md
自测：python check.py 6b1
"""

from __future__ import annotations

import torch
import torch.nn as nn


class LayerNorm(nn.Module):
    """对 (B,T,C) 的最后一维归一化，输出仍为 (B,T,C)。

    构造参数：d_model=C 为正整数；eps 为正数，默认 1e-5。
    非正 d_model 或 eps 应抛 ValueError。

    属性：
      eps：保存传入的小常数。
      weight：形状 (C,) 的 nn.Parameter，初始化为全 1，对应 gamma。
      bias：形状 (C,) 的 nn.Parameter，初始化为全 0，对应 beta。

    每个 (b,t) 单独计算：
      mu = sum_c x_c / C
      var = sum_c (x_c - mu)^2 / C
      y_c = weight_c * (x_c - mu) / sqrt(var + eps) + bias_c

    不修改输入，保留 dtype/device 和输入、参数的梯度。
    支持 C=1、常量向量和不连续输入。
    禁止内置 LayerNorm / layer_norm / batch_norm / normalize 及循环；
    使用基础张量运算翻译公式即可。
    """

    def __init__(self, d_model: int, eps: float = 1e-5):
        super().__init__()
        if eps <= 0 or d_model <= 0:raise ValueError("d_model=C 应为正整数 eps 应为正数") 
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(d_model))
        self.bias = nn.Parameter(torch.zeros(d_model))


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mu = x.mean(dim=-1, keepdim = True)
        var = x.var(dim=-1, keepdim = True, correction = 0)
        z = (x - mu) / torch.sqrt(var + self.eps)
        y = self.weight * z + self.bias
        return y
