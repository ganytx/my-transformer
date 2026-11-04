"""L07A · Mini-GPT 前向与语言模型损失。

规格：lessons/07a-gpt.md；自测：python check.py 7a。
本关由你决定具体的张量连接代码。
"""
from __future__ import annotations

import torch
import torch.nn as nn

from src.l02_train import cross_entropy
from src.l05_attention import sinusoidal_pe
from src.l06b1_layernorm import LayerNorm
from src.l06b2_block import TransformerBlock


class MiniGPT(nn.Module):
    """输入 idx:(B,T) long，返回原始 logits:(B,T,V)。

    V=vocab_size，S=block_size，C=d_model，H=n_heads，N=n_layers。
    五个配置参数均为正整数，C 能被 H 整除，否则 ValueError。
    forward 要求 1<=T<=S，否则 ValueError；其他输入保证有效。

    按讲义注册：token_emb、pe、blocks、ln_f、lm_head。
    保存 vocab_size 和 block_size 两个整数属性。
    每个 TransformerBlock 独立；不共享输入 embedding 和输出投影权重。
    """

    def __init__(self, vocab_size: int, block_size: int, d_model: int,
                 n_heads: int, n_layers: int):
        super().__init__()
        if vocab_size<=0 or block_size<=0 or d_model<=0 or n_heads<=0 or n_layers<=0 or d_model%n_heads!=0:raise ValueError("正数")
        self.token_emb = nn.Embedding(vocab_size, d_model)
        self.register_buffer("pe", sinusoidal_pe(block_size, d_model))
        self.blocks = nn.ModuleList([TransformerBlock(d_model, n_heads) for _ in range(n_layers)])
        self.ln_f = LayerNorm(d_model)
        self.lm_head = nn.Linear(d_model, vocab_size, bias = False)
        self.vocab_size = vocab_size
        self.block_size = block_size
        self.d_model = d_model        

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        T = idx.size(-1)
        if not (1<=T and T<=self.block_size):raise ValueError("要求 1<=T<=S")
        token = self.token_emb(idx) + self.pe[:T]  #(B,T,C)
        logits = token
        for block in self.blocks:
            logits = block(logits)
        return self.lm_head(self.ln_f(logits))


def language_model_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """(B,T,V) logits 与 (B,T) long targets -> 0 维平均交叉熵。

    targets 已经是对齐的下一个 token 标签，不要在这里再次移位。
    所有 B*T 个位置等权参与；复用 L02 cross_entropy，保留梯度。
    支持非连续输入；不修改 logits/targets；输入形状和标签范围保证有效。
    """
    B = logits.size(0)
    T = logits.size(1)
    V = logits.size(2)
    return cross_entropy(logits.reshape(B*T,V),targets.reshape(B*T))
