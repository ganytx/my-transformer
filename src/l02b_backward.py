"""2B · 手写反向传播（给 autograd 做一次开盖检查）

2A 里你调用 loss.backward()，梯度就出现了。这一讲把那个黑盒拆开：
**自己推、自己算**，然后和 autograd 的结果逐元素对比。

要算的是这个函数的梯度：

    z1 = x @ W1.T + b1            # (B, H)
    h  = relu(z1)                 # (B, H)
    z2 = h @ W2.T + b2            # (B, C)
    p  = softmax(z2)              # (B, C)
    L  = -mean( log p[正确类别] )  # 标量

返回 ∂L/∂W1, ∂L/∂b1, ∂L/∂W2, ∂L/∂b2（不是"更新后的参数"，是梯度）。

规则（检查会看）：
  * 不许调用反向传播：源码里不能出现 .backward( / torch.autograd / requires_grad；
  * 只许用张量运算，不许循环；
  * 可以用你自己写的 softmax（src/l01_tensors.py），也可以自己再推一遍。

写完运行：  python check.py 2b

提示：每一个梯度的形状，和它对应的参数**一模一样**。
      推导时拿这句话去卡形状，比背公式管用得多。

本讲会用到的东西（都是"看一眼就懂、不看就想不起来"的那类）：

    t.transpose(-1, -2)   交换最后两维，二维等价于 .T
    a @ b                 矩阵乘（也可写 torch.matmul）
    (x > 0)               布尔张量，可以当开关用；和非布尔张量相乘前要 .to(dtype)
    t.sum(dim=0)          沿 batch 那一维求和，结果少一维
    torch.zeros_like(t)   同形状全 0
    torch.arange(B)       0..B-1 的下标张量
    out[i, j] = 1.0       高级索引赋值（one-hot 就这么造）
"""

from __future__ import annotations

import torch

from src.l01_tensors import softmax

def log_softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """沿 dim 做 log_softmax，即 log(softmax(x))，但必须直接算出来。

    为什么需要它：softmax 之后再做 log，小概率会先下溢成 0，log(0) = -inf。
    log_softmax 先减 max 再取 log，把整个计算都留在 log 域里。

    返回同 shape、同 dtype 的张量；沿 dim 做 exp 后每个切片的和为 1。
    """
    res = (x - x.max(dim = dim, keepdim = True).values)
    res = res - ((res.exp()).sum(dim = dim, keepdim = True)).log()
    return res


def manual_grads(x: torch.Tensor, y: torch.Tensor,
                 W1: torch.Tensor, b1: torch.Tensor,
                 W2: torch.Tensor, b2: torch.Tensor) -> dict[str, torch.Tensor]:
    """手算上面那个损失对四个参数的梯度。

    参数
    ----
    x  : (B, D)  输入
    y  : (B,)    long，类别下标
    W1 : (H, D)  第一层权重（注意是 (out, in)）
    b1 : (H,)
    W2 : (C, H)
    b2 : (C,)

    返回
    ----
    {"W1": (H,D), "b1": (H,), "W2": (C,H), "b2": (C,)}，全部是 ∂L/∂参数。

    要求
    ----
    * 结果必须和 autograd 的 .grad 在 1e-9 内一致（float64 输入）。
    * 不能调用 .backward()，也不能开 requires_grad —— 这一讲要你自己走一遍链式法则。
    """
    z1 = x @ W1.transpose(0, 1) + b1 #(B,H)
    h = torch.clamp(z1,min=0)        #(B,H)
    z2 = h @ W2.transpose(0, 1) + b2 #(B,C)

    p = softmax(z2)
    B = x.size(0)
    dLdz2 = p.clone() / B #(B,C)
    dLdz2[torch.arange(B, device=z2.device), y] -= 1.0 / B
    dLdW2 = dLdz2.transpose(0, 1) @ h        #(C,H)
    dLdb2 = dLdz2.sum(dim=0) #(C)
    dLdh = dLdz2 @ W2 #(B,H)
    dLdz1 = dLdh * (z1>0)    #(B,H)
    dLdW1 = dLdz1.transpose(0, 1) @ x        #(H,D)
    dLdb1 = dLdz1.sum(dim=0) #(H)
    return {"W1": dLdW1, "b1": dLdb1, "W2": dLdW2, "b2": dLdb2}

