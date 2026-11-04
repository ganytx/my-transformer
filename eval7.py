"""使用学生实现评估已有 Mini-GPT checkpoint：python eval7.py path/to/model.pt。"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import torch

from src.l03_bigram import encode
from src.l07a_gpt import MiniGPT
from src.l08a_evaluate import make_eval_batches, evaluate

ROOT = Path(__file__).resolve().parent


def main(argv=None):
    parser = argparse.ArgumentParser(description="固定非重叠窗口协议下评估 checkpoint；不训练。")
    parser.add_argument("checkpoints", type=Path, nargs="+")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--split", choices=("train", "validation"), default="validation")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--output", type=Path, help="可选 JSON 输出路径；不覆盖已有文件")
    args = parser.parse_args(argv)
    if min(args.batch_size, args.threads) <= 0:
        parser.error("batch-size/threads 必须为正")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("没有可用 CUDA，请选择 cpu")
    if args.output is not None and args.output.exists():
        parser.error("output 已存在，请选择新文件名")
    torch.set_num_threads(args.threads)
    text = (ROOT / "data/alice.txt").read_text("utf-8")
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    results = []
    for path in args.checkpoints:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True)
        model = MiniGPT(**checkpoint["config"])
        model.load_state_dict(checkpoint["model_state"])
        model.to(args.device)
        data = encode(text, checkpoint["stoi"])
        cut = int(len(data) * 0.9)
        subset = data[:cut] if args.split == "train" else data[cut:]
        report = evaluate(model, make_eval_batches(subset, args.batch_size, model.block_size))
        result = dict(checkpoint=str(path.resolve()), split=args.split, corpus_sha256=digest,
                      protocol="nonoverlapping_windows_all_targets",
                      block_size=model.block_size, vocab_size=model.vocab_size,
                      batch_size=args.batch_size, device=args.device, **report)
        results.append(result)
        print(f"{path.parent.name} / {args.split} / S={model.block_size}: "
              f"tokens={report['n_tokens']}, loss={report['loss']:.6f}, "
              f"perplexity={report['perplexity']:.4f}", flush=True)
    print("仅在词表、文本划分和上下文协议相同时直接比较。")
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        # JSON 不支持无穷大；极端情况下保留为明确的字符串。
        serializable = [dict(r, perplexity="Infinity") if r["perplexity"] == float("inf") else r
                        for r in results]
        with args.output.open("x", encoding="utf-8") as handle:
            json.dump(serializable, handle, ensure_ascii=False, indent=2, allow_nan=False)
        print(f"报告已保存：{args.output.resolve()}")
    return results


if __name__ == "__main__":
    main()
