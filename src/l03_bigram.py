"""L03 · 字符级 bigram 语言模型

前两讲你训练的是一个"看输入、猜类别"的小分类器。这一讲开始做**语言模型**：
给一段文本，预测下一个字符。

## 先想清楚"预测下一个字符"是什么意思

文本 "hell" 里藏着 3 个训练样本：

    h → e
    e → l
    l → l

所以语言模型不需要人工标注：**把文本切开，前一个字符就是输入，后一个就是答案**。

## bigram = 只看前一个字符

最简的语言模型假设：下一个字符只取决于**当前**这一个字符。于是每个字符只需要一张
"我后面最可能跟什么"的分数表 —— 一个 V×V 的矩阵：

    logits = table[idx]        # idx: (B, T) 的整数张量 -> (B, T, V)

注意这就是第 6 讲"token embedding"的原型：table 的每一行都是一个 token 的向量。
区别只是这里向量长度也等于词表大小。

它一定很差（"q" 后面跟 "u" 都知道，但 "th" 后面跟什么就完全不知道了），
**这正是第 5 讲要上注意力的原因**：注意力让每个位置能看到前面所有字符，而不只是一个。

## 你要写的

    build_vocab(text) -> (stoi, itos)
    encode(text, stoi) -> LongTensor
    decode(ids, itos) -> str
    class BigramLM(nn.Module)          # forward: (B,T) -> (B,T,V)
    train_bigram(model, data, steps, batch_size, block_size, lr) -> list[float]
    sample(model, idx, n_tokens, temperature=1.0) -> LongTensor

数据在 data/alice.txt（《爱丽丝梦游仙境》全文，144602 字符，75 个不同字符）。

规则：
  * 损失要用你第 2 讲写的 cross_entropy（先 reshape 成 (B*T, V) 和 (B*T,)），
    不许调 F.cross_entropy / nn.CrossEntropyLoss；
  * train_bigram 里每步必须重新随机抽取 batch（不能整段一次性算完）。

写完运行：  python check.py 3
"""

from __future__ import annotations

import torch
import torch.nn as nn

from src.l01_tensors import softmax
from src.l02_train import cross_entropy



def build_vocab(text: str) -> tuple[dict[str, int], dict[int, str]]:
    """建立字符表和反查表。

    返回 (stoi, itos)：
      stoi: 字符 -> 整数下标，0..V-1
      itos: 整数下标 -> 字符
    要求：覆盖 text 里出现的每一个字符；两个表互为逆映射。
    """
    t1 = {}
    t2 = {}
    V = 0
    for i in text:
        if i not in t1:
            t1[i] = V
            t2[V] = i
            V = V+1 
    return (t1,t2)

def encode(text: str, stoi: dict[str, int]) -> torch.Tensor:
    """把字符串变成 LongTensor，形状 (len(text),)。"""
    l = len(text)
    lt = torch.zeros(l,dtype=torch.long)
    for i in range(l):
        lt[i]=stoi[text[i]]
    return lt



def decode(ids: torch.Tensor, itos: dict[int, str]) -> str:
    """把整数张量变回字符串。和一维、二维都可以，二维时按行拼接。"""
    return ''.join([itos[i] for i in ids.flatten().tolist()])


class BigramLM(nn.Module):
    """最简语言模型：每个 token 一张"下一个 token 的打分表"。

    forward(idx) -> logits
        idx    : (B, T) 的 long 张量
        logits : (B, T, V)，logits[b, t] 是第 t 个位置之后下一个字符的打分

    参数总量必须是 V*V（不要加 bias、不要加隐藏层）。
    """

    def __init__(self, vocab_size: int):
        super().__init__()
        self.fc=nn.Linear(vocab_size, vocab_size, bias=False)

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        return self.fc.weight[idx]


def train_bigram(model: nn.Module, data: torch.Tensor, steps: int = 2000,
                 batch_size: int = 32, block_size: int = 8,
                 lr: float = 0.5) -> list[float]:
    """在 data 上训练 steps 步，返回每一步的 loss（Python float 列表）。

    每一步：
      1. 随机抽 batch_size 个起点 i；
      2. x = data[i : i+block_size]，y = data[i+1 : i+block_size+1]；
      3. 前向、算损失、反向、更新。

    起点 i 的范围要保证 i+block_size+1 <= len(data)。
    """
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    l = data.size(0)
    losslist = []
    for _ in range(steps):
        optimizer.zero_grad()
        ridx = torch.randint(0,l-block_size,(batch_size,))
        idx = ridx[:, None] + torch.arange(block_size) #(B,1)+(block,)=(B,block)
        x = data[idx] 
        y = data[idx + 1]    #(B,blocksize) 
        logits = model(x)    #(B,blocksize,V) 
        loss = cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))
        loss.backward()
        optimizer.step()
        losslist.append(loss.item())
    return losslist



def sample(model: nn.Module, idx: torch.Tensor, n_tokens: int,
           temperature: float = 1.0) -> torch.Tensor:
    """自回归采样：每次用当前的最后一个字符预测下一个，接上去，再预测。

    参数
    ----
    idx         : (B, T) 的 long 张量，已有的"开头"
    n_tokens    : 要新生成多少个字符
    temperature : 温度。1.0 = 按原分布采样；趋近 0 = 变成贪婪（总选最大）；越大越随机

    返回
    ----
    (B, T+n_tokens) 的 long 张量，前 T 列必须原样保留。

    提示：temperature 作用在 logits 上（logits / temperature），不是作用在概率上。
    """
    B = idx.size(0)
    T = idx.size(1)
    out = torch.zeros(B, T + n_tokens, dtype = idx.dtype, device = idx.device) #(B,T)
    out[:, :T] = idx
    for i in range(n_tokens):
        logits = model(out[:, i:i+T])       #(B,T,V)
        next_logits = logits[:, -1, :]   #(B,V)
        if temperature < 1e-8:
            next_tokens = next_logits.argmax(dim=-1, keepdim = True)
            out[:, i+T:i+T+1] = next_tokens
        else:           
            next_tokens = torch.multinomial(softmax(next_logits / temperature, dim=-1), num_samples=1)
            out[:, i+T:i+T+1] = next_tokens
    return out

