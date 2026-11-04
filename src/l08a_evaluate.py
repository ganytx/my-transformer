"""L08A · 可重复的全语料评估。讲义：lessons/08a-evaluate.md。

自测：python check.py 8a
"""
from __future__ import annotations

import math
from collections.abc import Iterable

import torch
import torch.nn as nn

from src.l07a_gpt import language_model_loss


def make_eval_batches(data: torch.Tensor, batch_size: int,
                      block_size: int) -> list[tuple[torch.Tensor, torch.Tensor]]:
    """将一维 long 语料按顺序划分成确定的评估批次。

    L 个 token 形成 L-1 个预测任务：(data[i] 所在前缀 -> data[i+1])。
    从 i=0 开始，按 block_size 分成不重叠的输入窗口。
    完整窗口按顺序每 batch_size 个合成一批，不足一批也保留。
    最后若有不足 block_size 的短窗口，单独作为 B=1 的一批。
    每个目标位置 1..L-1 恰好计算一次；不随机采样、不补 padding。
    返回 x/y 均为 (B,T)，保持 data 的 dtype/device，不修改 data。
    batch_size/block_size 为正整数且 L>=2，否则 ValueError。
    """
    raise NotImplementedError("按固定窗口覆盖全部预测任务，包括最后的短窗口")


def evaluate(model: nn.Module, batches: Iterable[tuple[torch.Tensor, torch.Tensor]]) -> dict:
    """评估有限的非空 batch 序列，返回 loss、perplexity、n_tokens。

    每批 x/y:(B,T)，标签已对齐；不同批次的 B/T 可以不同。
    将 batch 移到模型参数所在 device，复用 language_model_loss。
    loss：按全部目标 token 等权平均，Python float。
    perplexity：exp(loss)，Python float；超出浮点可表示范围时返回正 inf。
    n_tokens：目标 token 总数，Python int。

    期间 eval/no_grad，正常或异常退出都恢复原 model.training。
    不更新参数，不清除/改动已有梯度，不改变输入或全局随机状态。
    batches 为空，或其中某批 B/T 为零时抛 ValueError。
    其他输入保证有效，T 不超过模型上下文上限，模型至少有一个参数。
    """
    raise NotImplementedError("汇总各批损失，按目标数量加权后计算 perplexity")
