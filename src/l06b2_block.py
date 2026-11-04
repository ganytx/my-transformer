"""L06B2 · FFN 与 Transformer 块。

先读 lessons/06b2-block.md；自测 python check.py 6b2。
"""

from __future__ import annotations

import torch
import torch.nn as nn

from src.l06a_multihead import MultiHeadAttention
from src.l06b1_layernorm import LayerNorm


class FeedForward(nn.Module):
    """逐 token 的两层 MLP：(B,T,C) -> (B,T,C)。

    注册 fc1=Linear(C,4*C,bias=True)、act=nn.GELU()、
    fc2=Linear(4*C,C,bias=True)，顺序为 fc1 -> act -> fc2。
    GELU 使用默认精确形式 approximate='none'，本题允许直接调用。
    C=d_model 必须为正整数（测试只传整数），否则抛 ValueError。
    所有位置共享同一组参数，参数量为 8*C*C+5*C。
    """

    def __init__(self, d_model: int):
        super().__init__()
        if d_model<=0:raise ValueError("d_model 必须为正整数")
        self.fc1 = nn.Linear(d_model, 4*d_model)
        self.fc2 = nn.Linear(4*d_model, d_model)
        self.act = nn.GELU(approximate='none')

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.act(self.fc1(x)))


class TransformerBlock(nn.Module):
    """使用 Pre-LN 的因果 Transformer 块：(B,T,C) -> (B,T,C)。

    注册四个子模块，属性名必须为：
      ln1：LayerNorm(C, eps)，用于注意力之前
      attn：MultiHeadAttention(C, H)
      ln2：另一份独立的 LayerNorm(C, eps)，用于 FFN 之前
      ffn：FeedForward(C)

    数学定义：u = x + attn(ln1(x))；y = u + ffn(ln2(u))。
    两次残差都保留未归一化的主路径；第二步使用更新后的 u。
    C/H 必须为正，C 能被 H 整除，eps>0，否则抛 ValueError。
    本题不加 dropout、位置编码或最后额外的 LayerNorm。
    必须复用前两关的自定义模块；不调用现成 Transformer 层。
    不修改输入，不 detach 梯度，不使用循环。
    """

    def __init__(self, d_model: int, n_heads: int, eps: float = 1e-5):
        super().__init__()
        if d_model<=0 or n_heads<=0 or d_model%n_heads!=0 or eps<=0: raise ValueError("C/H 必须为正，C 能被 H 整除，eps>0")
        self.ln1 = LayerNorm(d_model, eps)
        self.ln2 = LayerNorm(d_model, eps)   
        self.attn = MultiHeadAttention(d_model, n_heads)
        self.ffn = FeedForward(d_model)     


    def forward(self, x: torch.Tensor) -> torch.Tensor:
        u = x + self.attn(self.ln1(x))
        y = u + self.ffn(self.ln2(u))
        return y
