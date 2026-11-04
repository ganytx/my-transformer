"""L06A · 多头因果注意力。

先读 lessons/06a-multihead.md，再实现本文件的三个接口。
自测：python check.py 6a
"""

from __future__ import annotations

import torch
import torch.nn as nn

from src.l05_attention import scaled_dot_product_attention


def split_heads(x: torch.Tensor, n_heads: int) -> torch.Tensor:
    """(B,T,C) -> (B,H,T,D)，H=n_heads，D=C//H。

    严格满足 out[b,h,t,j] == x[b,t,h*D+j]。
    H 必须为正且整除 C，否则抛 ValueError。
    保留 dtype/device/梯度，不修改输入；不要求输出内存连续。
    """
    B = x.size(0)
    T = x.size(1)
    C = x.size(2)
    if n_heads<=0 or C%n_heads != 0:raise ValueError("H 必须为正且整除 C")
    D = C//n_heads
    return x.reshape(B,T,n_heads,D).transpose(1,2) 


def merge_heads(x: torch.Tensor) -> torch.Tensor:
    """(B,H,T,D) -> (B,T,H*D)，是 split_heads 的逆操作。

    必须支持 transpose 产生的不连续输入。
    保留 dtype/device/梯度，不修改输入。
    """
    B = x.size(0)
    H = x.size(1)
    T = x.size(2)
    D = x.size(3)
    return x.transpose(1,2).reshape(B,T,H*D)


class MultiHeadAttention(nn.Module):
    """多头因果自注意力：(B,T,C) -> (B,T,C)。

    注册 Wq/Wk/Wv/Wo 四个 nn.Linear(C,C,bias=False)。
    保存 n_heads 和 head_dim 两个整数属性。
    C=d_model、H=n_heads 必须为正且 C 能被 H 整除，否则抛 ValueError。
    参数量为 4*C*C。

    前向复用本文件 split_heads/merge_heads 和 L05 注意力函数。
    用 (B*H,T,D) 调用 L05，再恢复各头；禁止循环遍历头。
    本关不加位置编码、残差、LayerNorm 或 dropout。
    """

    def __init__(self, d_model: int, n_heads: int):
        super().__init__()
        if d_model<=0 or n_heads<=0 or d_model%n_heads!=0:raise ValueError("H 必须为正且整除 C")
        self.d_model = d_model
        self.n_heads = n_heads
        self.head_dim = d_model // n_heads
        self.Wq = nn.Linear(d_model, d_model, bias = False)
        self.Wk = nn.Linear(d_model, d_model, bias = False)
        self.Wv = nn.Linear(d_model, d_model, bias = False)
        self.Wo = nn.Linear(d_model, d_model, bias = False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B = x.size(0)
        T = x.size(1)
        D = self.d_model//self.n_heads
        x1 = split_heads(self.Wq(x), self.n_heads).reshape(B*self.n_heads,T,D) #(B*H,T,D)
        x2 = split_heads(self.Wk(x), self.n_heads).reshape(B*self.n_heads,T,D)
        x3 = split_heads(self.Wv(x), self.n_heads).reshape(B*self.n_heads,T,D)
        y = scaled_dot_product_attention(x1, x2, x3).reshape(B,self.n_heads,T,D)        
        return self.Wo(merge_heads(y))
