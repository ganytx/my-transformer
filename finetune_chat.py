"""复用学生训练函数，将已有英文模型微调到教学问答格式。"""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import time

import torch

from dialogue_runtime import ROOT, load_model, reply_to
from src.l03_bigram import encode
from src.l07b_train_generate import train_language_model
from src.l09a_dialogue import format_dialogue


def read_pairs(path):
    """读取单行问答 JSONL，在开始装载/训练模型之前报告格式错误。"""
    pairs = []
    for number, line in enumerate(path.read_text(encoding="utf-8-sig").splitlines(), 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
        except json.JSONDecodeError as exc:
            raise ValueError(
                f"{path} 第 {number} 行不是 JSON。此入口需要每行含 user/assistant 的 JSONL；"
                "DailyDialog 请使用 train.jsonl / validation.jsonl，不能传格式化的 .txt。"
            ) from exc
        if not isinstance(record, dict) or not all(isinstance(record.get(k), str) for k in ("user", "assistant")):
            raise ValueError(f"{path} 第 {number} 行必须包含字符串字段 user 和 assistant")
        pair = record["user"].strip(), record["assistant"].strip()
        try:
            format_dialogue([(record["user"], record["assistant"])])
        except ValueError as exc:
            raise ValueError(f"{path} 第 {number} 行：user/assistant 必须非空且不含换行") from exc
        pairs.append(pair)
    if not pairs:
        raise ValueError(f"{path} 没有有效问答")
    return pairs


def probe(model, stoi, pairs):
    results = []
    for question, expected in pairs:
        answer, raw = reply_to(model, stoi, [], question, max_new_tokens=96, temperature=0.)
        results.append(dict(question=question, expected=expected, answer=answer,
                            exact_match=answer.casefold() == expected.casefold(), raw_continuation=raw))
    return results


def main(argv=None):
    parser = argparse.ArgumentParser(description="使用你指定的英文问答数据微调已有模型。")
    parser.add_argument("--base", type=Path, required=True, help="已有模型的 checkpoint")
    parser.add_argument("--data", type=Path, required=True, help="训练 JSONL，每行含 user/assistant")
    parser.add_argument("--eval-data", type=Path, help="可选独立问答评估 JSONL")
    parser.add_argument("--probe-count", type=int, default=16, help="分别检查训练/评估集前多少条回答，默认 16")
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--context", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--output", type=Path, default=ROOT / "runs")
    args = parser.parse_args(argv)
    if args.steps < 0 or min(args.batch_size, args.context, args.probe_count) <= 0 or args.lr <= 0:
        parser.error("steps 非负，batch-size/context/probe-count/lr 为正")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("没有可用 CUDA")
    try:
        pairs = read_pairs(args.data)
        unseen = read_pairs(args.eval_data) if args.eval_data is not None else []
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    model, stoi, config = load_model(args.base, args.device, context=args.context)
    # 使用学生格式化函数，训练和推理遵守同一个协议。
    text = format_dialogue(pairs)
    missing = sorted(set(text) - set(stoi))
    if missing:
        parser.error(f"训练数据有当前词表未收录的字符：{missing!r}")
    longest = max(len(format_dialogue([pair])) for pair in pairs)
    if longest > args.context:
        parser.error(f"最长完整问答有 {longest} 个字符，请将 context 至少设为该长度")
    data = encode(text, stoi)
    if len(data) <= args.context:
        parser.error("语料长度必须大于 context")
    if args.eval_data is not None:
        if {q for q, _ in pairs} & {q for q, _ in unseen}:
            parser.error("评估问题与训练问题重复，请先按对话划分并去重")
        # 先检查评估格式，避免训练结束才因输入错误失败。
        format_dialogue(unseen)
        for question, _ in unseen:
            prompt = format_dialogue([], question)
            if len(prompt) > args.context or set(prompt) - set(stoi):
                parser.error("评估问题的提示超出 context 或包含词表外字符")
    seen_probe = pairs[:args.probe_count]
    unseen_probe = unseen[:args.probe_count]
    print(f"装载 {args.base.parent.name}；保留全部可训练权重；上下文 {config['block_size']} 字符", flush=True)
    print(f"训练问答 {len(pairs)} 条，合计 {len(data)} 字符。先记录微调前回答。", flush=True)
    before = probe(model, stoi, seen_probe)
    print(f"微调前训练题原文匹配：{sum(r['exact_match'] for r in before)}/{len(before)}", flush=True)
    print(f"开始一次连续的 {args.steps} 步 AdamW 微调，lr={args.lr}；不重建跨步优化器。", flush=True)
    started = time.perf_counter()
    history = train_language_model(model, data, args.steps, args.batch_size, args.context, args.lr)
    if args.device == "cuda":
        torch.cuda.synchronize()
    seconds = time.perf_counter() - started
    print(f"训练结束，用时 {seconds:.1f}s。检查训练题与未见问题。", flush=True)
    after = probe(model, stoi, seen_probe)
    heldout = probe(model, stoi, unseen_probe)
    stamp = datetime.now(timezone(timedelta(hours=8))).strftime("%Y%m%d_%H%M%S_%f")
    output = args.output.resolve() / f"dialogue_{stamp}"
    output.mkdir(parents=True, exist_ok=False)
    metadata = dict(kind="toy_dialogue", base_checkpoint=str(args.base.resolve()), config=config,
                    data_path=str(args.data.resolve()),
                    eval_data_path=str(args.eval_data.resolve()) if args.eval_data else None,
                    probe_count=args.probe_count,
                    steps=args.steps, batch_size=args.batch_size, lr=args.lr, seed=args.seed,
                    context=args.context, train_seconds=seconds, train_examples=len(pairs),
                    train_characters=len(data), train_loss=history,
                    before_seen=before, after_seen=after, unseen=heldout,
                    note="完整角色文本的语言模型微调。原文匹配只用于教学诊断，不代表通用问答能力。")
    (output / "metrics.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "training_text.txt").write_text(text, encoding="utf-8")
    torch.save(dict(kind="toy_dialogue", config=config, stoi=stoi,
                    model_state={k: v.detach().cpu() for k, v in model.state_dict().items()}), output / "model.pt")
    print(f"微调后训练题原文匹配：{sum(r['exact_match'] for r in after)}/{len(after)}")
    if heldout:
        print(f"独立评估问题参考答案原文匹配：{sum(r['exact_match'] for r in heldout)}/{len(heldout)}")
    else:
        print("未提供 --eval-data；当前回答检查只覆盖训练问题，不代表泛化表现。")
    for result in after[:5]:
        print(f"User: {result['question']}\nAssistant: {result['answer']}\n")
    print(f"已保存：{output}\n运行 python chat7.py 开始交互。")
    return output


if __name__ == "__main__":
    main()
