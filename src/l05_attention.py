"""L05 · 单头因果注意力

第 4 讲的模型只能看固定的 8 个字符，而且窗口一长参数就爆。
这一讲换一种"看前面"的方式：**让每个位置直接跟前面所有位置打分，按分数加权取信息**。

    Attention(Q, K, V) = softmax( Q Kᵀ / √d + mask ) V

三个新东西，逐个说：

## Q、K、V 是什么

同一个输入 x（(B,T,d)）经过三个不同的线性变换，变成三份"角色":

    Q = x @ Wqᵀ    "我在找什么"        (B, T, d)
    K = x @ Wkᵀ    "我能提供什么"      (B, T, d)
    V = x @ Wvᵀ    "如果被选中，我给什么" (B, T, d)

`Q @ Kᵀ` 是 (B, T, T)：第 i 行第 j 列 = 位置 i 的"问题"和位置 j 的"简历"有多匹配。
softmax 把每一行变成概率分布，再用它去加权 V。

## 为什么要除以 √d

如果 q、k 的每个分量独立、方差 1，那么点积 `Σ_d q_i k_i` 的方差是 d。
d=64 时分数标准差是 8，softmax 会被推到"几乎全部质量给最大值"的饱和区，
梯度接近 0，训练不动。除以 √d 把方差拉回 1。

## 因果掩码

语言模型不能看未来。做法是把上三角（j > i 的位置）的分数设成 -inf，
这样 softmax 之后那些位置的权重正好是 0。

**不要用 0 代替 -inf**：0 在 softmax 里是"和满分一样好"的一票，会让模型偷看未来。

## 位置编码

注意力是**置换等变**的：把输入顺序打乱，输出只是跟着重排。
也就是说，它自己根本不知道"谁在前谁在后"——第 4 讲里拼接的顺序提供了位置，这里没有。

所以必须显式加上位置信息：

    PE[pos, 2i]   = sin(pos / 10000^(2i/d))
    PE[pos, 2i+1] = cos(pos / 10000^(2i/d))

## 你要写的

    scaled_dot_product_attention(q, k, v, causal=True) -> (B,T,d)
    sinusoidal_pe(T, d_model) -> (T, d_model)
    class SingleHeadAttention(d_model)   # 四个无 bias 的 Linear，前向用因果注意力

规则：
  * softmax 用你自己第 1 讲的（文件头已 import）；
  * 禁止 F.scaled_dot_product_attention / nn.MultiheadAttention / torch.softmax。

本讲 API：

    math.sqrt(d)                      标量
    q @ k.transpose(-1, -2)           (B,T,d) @ (B,d,T) -> (B,T,T)
    scores.masked_fill(mask, float("-inf"))
    torch.tril(torch.ones(T,T))       下三角矩阵（含对角线）
    torch.arange(T)[:, None]          列向量，用来广播
    pe[:, 0::2] = ...                 隔一个赋一个（步长切片）
    nn.Linear(d, d, bias=False)       无偏置线性层

写完运行：  python check.py 5
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn

from src.l01_tensors import softmax


def scaled_dot_product_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor,
                                 causal: bool = True) -> torch.Tensor:
    """缩放点积注意力。

    参数
    ----
    q, k, v : (B, T, d) 的浮点张量，三者的 B、T 相同
    causal  : True 时每个位置只能看到自己和之前的位置

    返回
    ----
    (B, T, d)

    要求
    ----
    * 打分要除以 sqrt(d)；
    * 因果掩码用 -inf 填，不能填 0；
    * 不许原地修改 q/k/v。
    """
    d = q.size(-1)
    T = q.size(-2)
    scores = q @ k.transpose(-1,-2) / math.sqrt(d)
    if causal:scores = scores.masked_fill(torch.triu(torch.ones(T, T, dtype=torch.bool, device=scores.device),diagonal=1,),float("-inf"))
    return softmax(scores) @ v


def sinusoidal_pe(t: int, d_model: int) -> torch.Tensor:
    """正弦位置编码表，形状 (t, d_model)。

    第 pos 行、第 2i 列 = sin(pos / 10000^(2i/d_model))
    第 pos 行、第 2i+1 列 = cos(pos / 10000^(2i/d_model))

    要求：d_model 为奇数时也要能跑（最后一列的奇数/偶数部分少一个，不算错）。
    """
    """
    sp = torch.zeros(t,d_model)
    for pos in range(t):
        for i in range(d_model):
            if i&1 == 1:sp[pos,i] = math.cos(pos / 10000**((i-1)/d_model))
            else:sp[pos,i] = math.sin(pos / 10000**(i/d_model))
    """
    pe = torch.zeros(t, d_model)
    pos = torch.arange(t).unsqueeze(1)
    i = torch.arange(0, d_model, 2)
    div = torch.exp(-math.log(10000)*(i//2)*2/d_model)
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div[:pe.size(-1)//2])
    return pe



class SingleHeadAttention(nn.Module):
    """单头因果自注意力。

    结构：x -> 四个 d×d 的无 bias 线性层（Wq / Wk / Wv / Wo）-> 因果注意力 -> 输出

    * 参数量必须是 4*d*d；
    * forward(x) 里**不要**加位置编码——位置是调用方的事（检查里有专门一条验证这点）。
    """

    def __init__(self, d_model: int):
        super().__init__()
        self.Wq = nn.Linear(d_model, d_model, bias = False)
        self.Wk = nn.Linear(d_model, d_model, bias = False)
        self.Wv = nn.Linear(d_model, d_model, bias = False)
        self.Wo = nn.Linear(d_model, d_model, bias = False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.Wo(scaled_dot_product_attention(self.Wq(x), self.Wk(x), self.Wv(x)))

