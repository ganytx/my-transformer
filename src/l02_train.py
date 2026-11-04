"""L02 · 训练闭环

第 1 讲你写的是"纯函数"：输入张量，输出张量，没有参数，没有学习。

这一讲开始，代码里第一次出现**参数**（parameter）和**学习**：

    前向：logits = model(x)            # 一批输入 -> 每个类别的得分
    损失：loss   = cross_entropy(logits, y)   # 得分离正确答案有多远
    反向：loss.backward()              # 每个参数该往哪边调、调多少
    更新：optimizer.step()             # 真的调一下

你要写四个东西：

    1) class MLP(nn.Module)          —— 用 nn.Linear 搭两层 + ReLU
    2) cross_entropy(logits, targets) —— 平均负对数似然，复用第 1 讲的 log_softmax
    3) train_step(model, x, y, optimizer) —— 一次参数更新，返回这一步的 loss
    4) train_model(model, x, y, steps, lr) —— 循环调用 train_step，返回 loss 曲线

规则（检查会看）：
  * cross_entropy 里禁止 F.cross_entropy / nn.CrossEntropyLoss / F.nll_loss / F.log_softmax；
    请复用你自己在 src/l01_tensors.py 里写的 log_softmax（它已经通过了 23 项检查）。
  * train_step 里必须真的调用 backward()（不是手算梯度）。

写完运行：  python check.py 2
"""

from __future__ import annotations

import torch
import torch.nn as nn

from src.l01_tensors import log_softmax


class MLP(nn.Module):
    """两层全连接网络：Linear(d_in -> d_hidden) -> ReLU -> Linear(d_hidden -> d_out)。

    要求
    ----
    * 用 nn.Linear 建这两层（不要在 __init__ 里手搓 torch.randn 当权重）。
      注意 nn.Linear 里权重的形状是 (out_features, in_features)，不是反过来。
    * forward 里自己写前向，不要只 return self.net(x) 了事——你要能说出每一步。
    """

    def __init__(self, d_in: int, d_hidden: int, d_out: int):
        super().__init__()
        self.fc1 = nn.Linear(d_in, d_hidden)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(d_hidden, d_out)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.relu(self.fc1(x)))
        


def cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """多分类交叉熵（对 batch 取平均）。

    参数
    ----
    logits  : shape (B, C)      模型的原始得分，**不是**概率
    targets : shape (B,)        dtype=torch.long，每个元素是 0..C-1 的类别下标

    返回
    ----
    0 维张量（标量）：-(1/B) * Σ log p(target_i)

    要求
    ----
    * 数值稳定：logits 里有 1e4 这样的值也不能出现 inf/nan。
    * 必须用 log_softmax 算 log p，不能先 softmax 再 log（会下溢）。
    """
    lp = log_softmax(logits)
    n = lp.size(0)
    return -(lp[torch.arange(n, device=logits.device), targets]).mean()


def train_step(model: nn.Module, x: torch.Tensor, y: torch.Tensor,
               optimizer: torch.optim.Optimizer) -> float:
    """做一次完整的参数更新，并返回这一步的 loss（Python float）。

    顺序很重要，想一想每一句在解决什么问题：
        optimizer.zero_grad() -> 前向 -> 算 loss -> loss.backward() -> optimizer.step()

    如果漏掉 zero_grad()，梯度会被累加，训练会变成什么样？
    """
    optimizer.zero_grad()
    res = model(x)
    loss = cross_entropy(res,y)
    loss.backward()
    optimizer.step()
    return loss.item()



def train_model(model: nn.Module, x: torch.Tensor, y: torch.Tensor,
                steps: int = 300, lr: float = 0.1) -> list[float]:
    """在 (x, y) 上训练 steps 步，返回每一步的 loss 列表。

    要求
    ----
    * 在函数内部用 torch.optim.SGD(model.parameters(), lr=lr) 建优化器。
    * 返回的 list 长度必须等于 steps，元素是 Python float。
    """
    optimizer=torch.optim.SGD(model.parameters(), lr=lr)
    losslist = []
    for i in range(steps):
        l = train_step(model, x, y, optimizer)
        losslist.append(l)
    return losslist

