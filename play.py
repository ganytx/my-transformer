#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""play.py —— 训练 + 观察效果。不是判题，是给你看的。

    python play.py                  默认：训练 3000 步，看几种温度的采样
    python play.py --steps 8000     练久一点
    python play.py --start "Alice"  换个开头
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

import torch

from src.l03_bigram import BigramLM, build_vocab, decode, encode, sample, train_bigram

ROOT = Path(__file__).resolve().parent


def top_next_chars(model, itos, stoi, chars, k=5):
    """看看模型给某些字符排的前 k 个后继。"""
    v = len(stoi)
    idx = torch.tensor([[stoi[c] for c in chars]])
    with torch.no_grad():
        logits = model(idx)[0]                      # (len(chars), V)
        probs = torch.softmax(logits, dim=-1)
    for ch, row in zip(chars, probs):
        top = row.topk(k)
        parts = [f"{itos[int(i)]!r} {p:.0%}" for p, i in zip(top.values, top.indices)]
        print(f"    {ch!r} → " + "   ".join(parts))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--steps", type=int, default=3000)
    p.add_argument("--start", type=str, default="Alice")
    p.add_argument("--n", type=int, default=300)
    args = p.parse_args()

    text = (ROOT / "data" / "alice.txt").read_text("utf-8")
    stoi, itos = build_vocab(text)
    data = encode(text, stoi)
    v = len(stoi)
    print(f"语料 {len(text)} 字符，词表 {v} 个字符 → 完全瞎猜的 loss = ln({v}) = {math.log(v):.4f}")

    torch.manual_seed(0)
    model = BigramLM(v)
    t0 = time.time()
    hist = train_bigram(model, data, steps=args.steps, batch_size=32, block_size=8, lr=0.5)
    print(f"\n训练 {args.steps} 步，用时 {time.time() - t0:.1f}s")

    step = max(1, args.steps // 10)
    print("loss 曲线：")
    for i in range(0, args.steps, step):
        bar = "#" * int((math.log(v) - hist[i]) * 12)
        print(f"    {i:>5}  {hist[i]:.4f}  {bar}")
    print(f"    {args.steps - 1:>5}  {hist[-1]:.4f}  ← 最终")

    print("\n模型学到的「下一个字符」排行：")
    ch = []
    for i in range(26):
        ch.append(chr(ord('A')+i))
    ch.append(' ')
    ch.append('.')
    top_next_chars(model, itos, stoi, ch, k=10)

    print("\n同一个开头，不同温度的采样：")
    start = torch.tensor([[stoi[c] for c in args.start]])
    for t in (0.3, 0.7, 1.0, 1.5):
        torch.manual_seed(1)
        out = sample(model, start, args.n, temperature=t)
        s = decode(out[0], itos).replace("\n", " ⏎ ")
        print(f"\n  ── temperature={t} ──")
        print("    " + s[:300])

    print("\n（这就是「只记得前一个字符」的上限：有单词的样子，没有句子的结构。）")


if __name__ == "__main__":
    main()
