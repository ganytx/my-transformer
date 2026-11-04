"""L07B · 训练与自回归生成。讲义：lessons/07b-train-generate.md。

自测 python check.py 7b；通过后运行 python play7.py。
"""
from __future__ import annotations

import torch
import torch.nn as nn

from src.l01_tensors import softmax
from src.l07a_gpt import language_model_loss


def get_batch(data: torch.Tensor, batch_size: int, block_size: int) -> tuple[torch.Tensor, torch.Tensor]:
    """从一维 long 语料随机取 B 个长度 T 的窗口，返回 x/y:(B,T)。

    每个起点 i 独立均匀抽取（允许重复），0 <= i < len(data)-T。
    x[b,t]=data[i+t]，y[b,t]=data[i+t+1]；保持 data 的 dtype/device。
    B/T 必须为正，len(data)>=T+1，否则 ValueError。不能修改 data。
    要用张量索引，不逐 token 循环。
    """
    l = data.size(0)
    if block_size<=0 or batch_size<=0 or l<=block_size: raise ValueError("B/T 必须为正，len(data)>=T+1")
    ridx = torch.randint(0,l-block_size,(batch_size,))
    idx = ridx[:, None] + torch.arange(block_size)          #(B,1) + (block_size,) 
    x = data[idx]
    y = data[idx+1]
    return (x,y)


def train_language_model(model: nn.Module, data: torch.Tensor, steps: int,
                         batch_size: int, block_size: int, lr: float = 1e-3) -> list[float]:
    """训练 model 并返回每一步更新前的 loss，元素为 Python float。

    一次调用创建一个 torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)。
    model 进入 train 模式；每步重新调用本文件 get_batch，batch 移到模型所在 device。
    复用 language_model_loss，完成清梯度、前向、反向、更新。
    data 可以保留在 CPU；不修改 data。不需要裁剪梯度或调度学习率。
    steps>=0，B>0，1<=T<=model.block_size，lr>0，len(data)>=T+1，
    否则 ValueError。有效配置下 steps=0 返回 []，不改模型参数。
    """
    if not (steps>=0 and batch_size>0 and 1<=block_size and block_size<=model.block_size and lr>0 and len(data)>=block_size+1 ):raise ValueError("steps>=0，B>0，1<=T<=model.block_size，lr>0，len(data)>=T+1") 
    device = next(model.parameters()).device
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    losslist = []
    model.train()
    for _ in range(steps):
        optimizer.zero_grad()
        x,y = get_batch(data, batch_size, block_size)
        x = x.to(device)
        y = y.to(device)
        logits = model(x)
        loss = language_model_loss(logits, y)
        loss.backward()
        optimizer.step()
        losslist.append(loss.item())


        percent = _ / steps
        bar = '█' * int(percent * 20) + '-' * (20 - int(percent * 20))
        print(f'\r[{bar}] {percent*100:5.1f}%', end='', flush=True)
    print(f'\r[{bar}] {100:5.1f}%', end='\n', flush=True)
    return losslist


def generate(model: nn.Module, idx: torch.Tensor, max_new_tokens: int,
             temperature: float = 1.0) -> torch.Tensor:
    """输入前缀 idx:(B,T)，生成并返回 (B,T+max_new_tokens) long。

    保留完整原前缀且不修改 idx。idx 和模型在同一 device，T>=1。
    每步仅把当前序列最后 model.block_size 个 token 送入模型，
    用最后位置的 logits 选出下一个 token，接到完整输出序列末尾。
    temperature=0：argmax 贪婪解码；>0：logits/temperature 后用 L01 softmax，
    再用 torch.multinomial 抽一个 token。禁止把 temperature 除在概率上。
    生成期间 eval 且禁用梯度；结束后恢复调用前 model.training 状态。
    max_new_tokens>=0、temperature>=0，否则 ValueError；空前缀也抛 ValueError。
    输入有效且 max_new_tokens=0 时返回等值前缀，不调用模型。
    """
    if max_new_tokens<0 or temperature<0 or idx.size(-1)==0:raise ValueError("max_new_tokens>=0、temperature>=0")
    if max_new_tokens == 0:return idx
    B = idx.size(0)
    T = idx.size(1)
    out = torch.zeros(B, T + max_new_tokens, dtype = idx.dtype, device = idx.device) #(B,T)
    out[:, :T] = idx
    was_training = model.training
    model.eval()
    try:
        with torch.no_grad():    
            for i in range(max_new_tokens):
                end = i+T
                start = max(0, end - model.block_size)
                logits = model(out[:, start:end])       #(B,T,V)
                next_logits = logits[:, -1, :]   #(B,V)
                if temperature == 0:
                    next_tokens = next_logits.argmax(dim=-1, keepdim = True)
                    out[:, i+T:i+T+1] = next_tokens
                else:           
                    next_tokens = torch.multinomial(softmax(next_logits / temperature, dim=-1), num_samples=1)
                    out[:, i+T:i+T+1] = next_tokens
    finally:
        model.train(was_training)
    return out
