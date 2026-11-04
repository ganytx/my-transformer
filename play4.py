#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""play4.py —— 上下文窗口模型：训练、对比、采样。调参随便玩。

    python play4.py                        默认 k=8, d=16, hidden=128, 训练 3000 步
    python play4.py --k 16 --d 32 --hidden 256 --steps 5000
    python play4.py --sweep                扫 k = 1,2,4,8,16，打一张对比表

第一次引入**验证集**：前 6 万字符训练，接下去 3 万字符只用来评估。
训练 loss 降不代表模型变好，验证 loss 降才是。
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import torch

from src.l02_train import cross_entropy
from src.l04_context import ContextLM, get_batch, sample_context, train_context_model

ROOT = Path(__file__).resolve().parent
TRAIN_CHARS = 60000
VAL_CHARS = 30000


def load(split=1.0):
    text = (ROOT / "data" / "alice.txt").read_text("utf-8")
    text = text[: int(len(text) * split)]
    stoi = {c: i for i, c in enumerate(sorted(set(text)))}
    itos = {i: c for c, i in stoi.items()}
    data = torch.tensor([stoi[c] for c in text], dtype=torch.long)
    return text, stoi, itos, data


@torch.no_grad()
def evaluate(model, data, k, n_batches=40, batch_size=256):
    """在验证集上算 loss 和准确率。"""
    g = torch.Generator().manual_seed(12345)
    total_loss, correct, total = 0.0, 0, 0
    for _ in range(n_batches):
        x, y = get_batch(data, batch_size, k, g)
        logits = model(x)
        total_loss += cross_entropy(logits, y).item()
        correct += (logits.argmax(dim=-1) == y).sum().item()
        total += y.numel()
    return total_loss / n_batches, correct / total


def train_one(k, d, hid, steps, lr, data_tr, data_val, seed=0, quiet=False):
    v = 75
    torch.manual_seed(seed)
    model = ContextLM(v, k, d, hid)
    n_param = sum(p.numel() for p in model.parameters())
    t0 = time.time()
    hist = train_context_model(model, data_tr, steps=steps, batch_size=32, block_size=k, lr=lr)
    val_loss, val_acc = evaluate(model, data_val, k)
    el = time.time() - t0
    if not quiet:
        print(f"  k={k:<3} 参数 {n_param:>7,}  训练 loss {hist[-1]:.4f}  "
              f"验证 loss {val_loss:.4f}  验证准确率 {val_acc:.1%}  用时 {el:.1f}s")
    return model, hist, val_loss, val_acc, n_param


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=8)
    ap.add_argument("--d", type=int, default=16)
    ap.add_argument("--hidden", type=int, default=128)
    ap.add_argument("--steps", type=int, default=3000)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--sweep", action="store_true")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--start", type=str, default="Alice was beginning to")
    args = ap.parse_args()

    text, stoi, itos, data = load()
    data_tr = data[:TRAIN_CHARS]
    data_val = data[TRAIN_CHARS:TRAIN_CHARS + VAL_CHARS]
    v = len(stoi)
    print(f"词表 {v}，训练 {len(data_tr)} 字符，验证 {len(data_val)} 字符")
    print(f"完全瞎猜的 loss = ln({v}) = {math.log(v):.4f}；bigram 大约 2.60；k=8 参考约 1.80\n")

    if args.sweep:
        print("扫窗口长度（其余超参相同）：")
        rows = []
        for k in (1, 2, 4, 8, 16):
            _, hist, vl, va, np_ = train_one(k, args.d, args.hidden, args.steps, args.lr,
                                             data_tr, data_val)
            rows.append((k, np_, vl, va))
        print("\n  k   参数       验证loss   比 k=1 改善")
        base = rows[0][2]
        for k, np_, vl, va in rows:
            print(f"  {k:<3} {np_:>7,}   {vl:.4f}     {(base - vl) / base:>6.1%}")
        print("\n  （k 一直加下去，收益会越来越小——因为窗口是死的，见第 5 讲。）")
        return

    print("训练：")
    model, hist, val_loss, val_acc, n_param = train_one(
        args.k, args.d, args.hidden, args.steps, args.lr, data_tr, data_val)

    print("\nloss 曲线（训练集）：")
    step = max(1, args.steps // 10)
    for i in range(0, args.steps, step):
        bar = "#" * int((math.log(v) - hist[i]) * 14)
        print(f"    {i:>5}  {hist[i]:.4f}  {bar}")
    print(f"    {args.steps - 1:>5}  {hist[-1]:.4f}  ← 训练")
    print(f"           {val_loss:.4f}  ← 验证（模型没见过的文本）")

    print("\n采样（k={}，不同温度）：".format(args.k))
    start = torch.tensor([[stoi[c] for c in args.start]])
    for t in (0.5, 0.8, 1.0):
        torch.manual_seed(1)
        out = sample_context(model, start, args.n, args.k, t)
        s = "".join(itos[int(i)] for i in out[0]).replace("\n", " ⏎ ")
        print(f"\n  ── temperature={t} ──")
        print("    " + s[:300])


if __name__ == "__main__":
    main()
