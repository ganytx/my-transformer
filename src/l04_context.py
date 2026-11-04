"""L04 · 上下文窗口：第一个"真的在学习结构"的模型

第 3 讲的 bigram 只记得前**一个**字符。你已经看到它的两个症状：
  * 低温采样变成 "the the the"（因为它只会走最大概率那条路）
  * 常温采样迅速崩坏（因为一步走偏，下一步就没有纠错能力）

这一讲给它一个长度为 k 的窗口：用前 k 个字符预测下一个。

## 和 bigram 的根本区别

bigram 的表是 (V, V)：每种"前一个字符"占一行，互不相干。
这一讲每个字符变成一个 **d 维向量**（d 远小于 V），窗口里的 k 个向量**拼起来**送进 MLP：

    e  = emb(x)              # (B, k, d)   查表 -> 每个字符一个向量
    h  = e.reshape(B, k*d)   # (B, k*d)    拼成一个长向量
    y  = fc2(relu(fc1(h)))   # (B, V)      预测下一个字符

**这一步的意义**：参数不再是"每种上下文一行"，而是"每个字符一个向量 + 一套权重"。
所以模型可以**泛化**到训练时没见过的上下文组合——这正是 embedding 的价值。

（顺带：拼接的**顺序**本身携带了位置信息。记住这一点，第 5 讲做注意力时，
注意力是"置换不变"的，届时就必须显式加位置编码。）

## 你要写的

    get_batch(data, batch_size, block_size, generator=None) -> (x, y)
        x: (B, k) long   窗口
        y: (B,)   long   紧跟在窗口后面的那个字符

    class ContextLM(vocab_size, block_size, n_embd=16, n_hidden=128)
        forward(x: (B,k)) -> logits (B, V)

    train_context_model(model, data, steps=2000, batch_size=32, block_size=8, lr=3e-3)
        -> list[float]

    sample_context(model, idx, n_tokens, block_size, temperature=1.0) -> (B, T+n)
        要求 idx.size(1) >= block_size；每次只看最后 block_size 个字符

规则：
  * 损失用你自己第 2 讲的 cross_entropy；
  * 优化器用 torch.optim.AdamW（这一讲先这样用：它给每个参数单独调步长，
    小模型上比 SGD 省心得多。第 10 讲会拆开讲它到底怎么调的）；
  * 前向里必须真的把 k 个位置的向量拼起来——不能只用最后一个位置（那是 bigram）。

本讲会用到（PyTorch 速查）：

    nn.Embedding(V, d)        查表：输入 (B,k) 的 long -> (B,k,d)
    e.reshape(B, -1)          (B,k,d) -> (B, k*d)，把后两维压平
    torch.randint(0, hi, (n,)) 随机起点
    ix[:, None] + torch.arange(k)   (B,1)+(k,) -> (B,k) 的所有窗口下标
    out[:, -k:]               最后 k 个字符
    torch.optim.AdamW(params, lr=...)  Adam 的变体，带解耦权重衰减

写完运行：  python check.py 4
"""

from __future__ import annotations

import torch
import torch.nn as nn

from src.l02_train import cross_entropy
from src.l01_tensors import softmax

def get_batch(data: torch.Tensor, batch_size: int, block_size: int,
              generator: torch.Generator | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """随机抽 batch_size 个长度为 block_size 的窗口。

    返回
    ----
    x: (batch_size, block_size)  long
    y: (batch_size,)             long，x[i] 窗口之后紧跟的那一个字符

    要求：起点 i 满足 i + block_size + 1 <= len(data)。
    """
    l = data.size(0)
    ridx = torch.randint(0,l-block_size,(batch_size,))
    idx = ridx[:, None] + torch.arange(block_size)          #(B,1) + (block_size,) 
    x = data[idx]
    y = data[ridx+block_size]
    return (x,y)

class ContextLM(nn.Module):
    """窗口语言模型：embedding -> 拼接 -> MLP -> 词表打分。

    __init__ 参数
    -------------
    vocab_size : V
    block_size : k（窗口长度，要和前向输入的最后一维一致）
    n_embd     : d（每个字符的向量维度）
    n_hidden   : MLP 隐层宽度

    参数量应为：V*d + (k*d)*n_hidden + n_hidden + n_hidden*V + V
    """

    def __init__(self, vocab_size: int, block_size: int, n_embd: int = 16, n_hidden: int = 128):
        super().__init__()
        self.embd = nn.Embedding(vocab_size, n_embd)
        self.fc1 = nn.Linear(block_size*n_embd, n_hidden)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(n_hidden, vocab_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e = self.embd(x).flatten(start_dim=1)
        return self.fc2(self.relu(self.fc1(e)))


def train_context_model(model: nn.Module, data: torch.Tensor, steps: int = 2000,
                        batch_size: int = 32, block_size: int = 8,
                        lr: float = 3e-3) -> list[float]:
    """训练 steps 步，返回每一步的 loss（Python float）。内部建 AdamW 优化器。"""
    losslist = []
    optimizer = torch.optim.AdamW(model.parameters(),lr=lr)
    for _ in range(steps):
        x,y = get_batch(data, batch_size, block_size)
        optimizer.zero_grad()
        logits = model(x) #(B,V,)
        loss = cross_entropy(logits, y)
        loss.backward()
        optimizer.step()
        losslist.append(loss.item())
    return losslist



def sample_context(model: nn.Module, idx: torch.Tensor, n_tokens: int,
                   block_size: int, temperature: float = 1.0) -> torch.Tensor:
    """用滑动窗口自回归采样。

    每一步取最后 block_size 个字符作为模型输入；温度作用在 logits 上。
    返回 (B, T + n_tokens)，前缀原样保留。
    """
    B = idx.size(0)
    T = idx.size(1)
    out = torch.zeros(B, T + n_tokens, dtype = idx.dtype, device = idx.device) #(B,T)
    out[:, :T] = idx
    for i in range(n_tokens):
        if i+block_size<=T+n_tokens:
            x = out[:, i:i+block_size]
        else:
            x = out[:, i:T+n_tokens]
            pad = idx[:, :1].expand(B,block_size-(T+n_tokens-i))
            x = torch.cat([pad, x], dim=1)

        next_logits = model(x)       #(B,V)
        if temperature < 1e-8:
            next_tokens = next_logits.argmax(dim=-1, keepdim = True)
            out[:, i+T:i+T+1] = next_tokens
        else:           
            next_tokens = torch.multinomial(softmax(next_logits / temperature, dim=-1), num_samples=1)
            out[:, i+T:i+T+1] = next_tokens
    return out
