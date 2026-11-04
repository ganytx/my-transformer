"""L01 · 张量与广播

在这一关里，你要把两个"数学上很简单、写起来很容易错"的东西翻译成张量运算：

    softmax(x)_i = exp(x_i) / sum_j exp(x_j)
    dist(a, b)   = ||a - b||

规则（测试会检查）：
  1. 四个函数内部禁止出现 for / while / 列表推导式；
  2. 禁止调用 torch 的现成实现：softmax / log_softmax / logsumexp / cdist
     （`x.softmax(-1)` 这种写法同样被禁）；
  3. 不许修改输入张量。

写完运行：  python oj.py check L01
"""

from __future__ import annotations

import torch


def softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """沿 dim 做 softmax。

    参数
    ----
    x   : 任意形状的浮点张量
    dim : 归一化的轴（可以是负数，-1 表示最后一维）

    返回
    ----
    与 x 同 shape、同 dtype 的张量；沿 dim 的每一"条"概率和为 1。

    要求
    ----
    * 数值稳定：x 里出现 1e4 甚至 1e9 时也不能产生 inf / nan。
    * 纯函数：不能原地修改 x。
    """
    res = (x - x.max(dim = dim, keepdim = True).values).exp()
    res = res / res.sum(dim = dim, keepdim = True)
    return res 

 


def log_softmax(x: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """沿 dim 做 log_softmax，即 log(softmax(x))，但必须直接算出来。

    为什么需要它：softmax 之后再做 log，小概率会先下溢成 0，log(0) = -inf。
    log_softmax 先减 max 再取 log，把整个计算都留在 log 域里。

    返回同 shape、同 dtype 的张量；沿 dim 做 exp 后每个切片的和为 1。
    """
    res = (x - x.max(dim = dim, keepdim = True).values)
    res = res - ((res.exp()).sum(dim = dim, keepdim = True)).log()
    return res


def pairwise_sqdist(x: torch.Tensor) -> torch.Tensor:
    """批量平方距离矩阵。

    参数
    ----
    x : shape (B, N, D) 的浮点张量，B 个样本、每个样本 N 个 D 维向量。

    返回
    ----
    shape (B, N, N) 的张量，out[b, i, j] = ||x[b, i] - x[b, j]||^2

    要求
    ----
    * 只能用张量运算整体算出来（不能三重循环）。
    * 结果要正好对称，对角线正好是 0（浮点误差请自己处理掉）。
    """
    s1 = ((x*x).sum(-1)).unsqueeze(-1)
    s2 = s1.transpose(-1, -2)
    s3 = x @ x.transpose(-1, -2)
    res = s1 + s2 - 2 * s3
    res = (res  + res.transpose(-1, -2))/2
    res = torch.clamp(res, min=0)
    res.diagonal(dim1=-2, dim2=-1).fill_(0.0) 
    return res

def pairwise_dist(x: torch.Tensor) -> torch.Tensor:
    """批量欧氏距离矩阵：out[b, i, j] = ||x[b, i] - x[b, j]||。

    可以直接复用你自己写的 pairwise_sqdist。
    """
    return torch.sqrt(pairwise_sqdist(x))

