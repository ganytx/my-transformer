"""运行学生的 Mini-GPT：默认使用 smol-smoltalk 的 2000 条简短问答。"""
from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
import time

import torch

from src.l03_bigram import build_vocab, encode, decode
from src.l07a_gpt import MiniGPT, language_model_loss
from src.l07b_train_generate import get_batch, train_language_model, generate

ROOT = Path(__file__).resolve().parent
DEFAULT_DATA = ROOT / "data/smol-smoltalk/short_replies_2000/train.txt"
DEFAULT_VALIDATION = ROOT / "data/smol-smoltalk/short_replies_2000/validation.txt"


def main(argv=None):
    parser = argparse.ArgumentParser(description="使用你写的函数训练和生成；保存推理检查点，不支持断点续训。",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA, help="UTF-8 训练文本；默认 smol-smoltalk 2000 条短问答，建议 context512")
    parser.add_argument("--val-data", type=Path, help="默认语料配独立100条验证题；自定义data时省略则按字符位置分前90%%/后10%%")
    parser.add_argument("--steps", type=int, default=4000)
    parser.add_argument("--batch-size", type=int, default=48)
    parser.add_argument("--context", type=int, default=128)
    parser.add_argument("--d-model", type=int, default=128)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--layers", type=int, default=8)
    parser.add_argument("--lr", type=float, default=0.0002)
    parser.add_argument("--new-tokens", type=int, default=160)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--prompt", default="User: ")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--output", type=Path, default=ROOT / "runs")
    args = parser.parse_args(argv)
    if args.val_data is None and args.data.resolve() == DEFAULT_DATA.resolve():
        args.val_data = DEFAULT_VALIDATION
    if args.steps < 0 or args.new_tokens < 0 or args.temperature < 0 or args.lr <= 0:
        parser.error("steps/new-tokens/temperature 必须非负，lr 必须为正")
    if min(args.batch_size, args.context, args.d_model, args.heads, args.layers, args.threads) <= 0:
        parser.error("batch-size/context/d-model/heads/layers/threads 必须为正")
    if args.d_model % args.heads:
        parser.error("d-model 必须能被 heads 整除")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("当前 Python 环境没有可用 CUDA，请用 --device cpu")
    print("[准备] 参数检查通过，开始准备数据。", flush=True)
    torch.set_num_threads(args.threads)
    torch.manual_seed(args.seed)
    preparation_started = time.perf_counter()
    print(f"[读取] 训练文本：{args.data}", flush=True)
    text = args.data.read_text(encoding="utf-8")
    if args.val_data is None:
        split = int(len(text) * 0.9)
        train_text, val_text = text[:split], text[split:]
    else:
        if args.data.resolve() == args.val_data.resolve():
            parser.error("训练与验证文件必须不同，并由你事先按完整文档/对话划分")
        print(f"[读取] 验证文本：{args.val_data}", flush=True)
        train_text, val_text = text, args.val_data.read_text(encoding="utf-8")
    # 词表只由训练集建立，验证集不参与任何参数更新。
    print(f"[词表] 训练 {len(train_text):,} 字符，验证 {len(val_text):,} 字符；建立训练词表……", flush=True)
    stoi, itos = build_vocab(train_text)
    print(f"[词表] 完成，共 {len(stoi)} 个字符。", flush=True)
    missing = sorted(set(val_text) - set(stoi))
    if missing:
        parser.error(f"验证集含训练词表之外的字符：{missing!r}；请统一字符规范或另行设计固定词表")
    if not args.prompt or any(ch not in stoi for ch in args.prompt):
        parser.error("prompt 必须非空，且每个字符都在语料词表中")
    print("[编码] 正在 CPU 上编码训练文本", flush=True)
    encoding_started = time.perf_counter()
    train_data = encode(train_text, stoi)
    print(f"[编码] 训练文本完成，用时 {time.perf_counter() - encoding_started:.1f}s；开始编码验证文本……", flush=True)
    val_data = encode(val_text, stoi)
    print(f"[编码] 完成；数据准备共用时 {time.perf_counter() - preparation_started:.1f}s。", flush=True)
    if min(len(train_data), len(val_data)) <= args.context:
        parser.error("context 太长，训练集和验证集都至少需要 context+1 个字符")
    config = dict(vocab_size=len(stoi), block_size=args.context, d_model=args.d_model,
                  n_heads=args.heads, n_layers=args.layers)
    print(f"[模型] 初始化模型并移到 {args.device}……", flush=True)
    model = MiniGPT(**config).to(args.device)
    vx, vy = get_batch(val_data, args.batch_size, args.context)
    vx, vy = vx.to(args.device), vy.to(args.device)

    def validation():
        original_mode = model.training
        try:
            model.eval()
            with torch.no_grad():
                return language_model_loss(model(vx), vy).item()
        finally:
            model.train(original_mode)

    print("[验证] 计算训练前的验证 loss……", flush=True)
    before = validation()
    print(f"参数 {sum(p.numel() for p in model.parameters()):,}；设备 {args.device}；"
          f"训练/验证字符数 {len(train_data):,}/{len(val_data):,}", flush=True)
    print(f"训练 {args.steps} 步，完成后报告结果；初始验证 loss={before:.4f}", flush=True)
    started = time.perf_counter()
    history = train_language_model(model, train_data, steps=args.steps, batch_size=args.batch_size,
                                   block_size=args.context, lr=args.lr)
    if args.device == "cuda":
        torch.cuda.synchronize()
    seconds = time.perf_counter() - started
    after = validation()
    prefix = encode(args.prompt, stoi)[None].to(args.device)
    generated = generate(model, prefix, args.new_tokens, args.temperature)
    sample = decode(generated[0].cpu(), itos)
    stamp = datetime.now(timezone(timedelta(hours=8))).strftime("%Y%m%d_%H%M%S_%f")
    output = args.output.resolve() / f"mini_gpt_{stamp}"
    output.mkdir(parents=True, exist_ok=False)
    metrics = dict(config=config, steps=args.steps, batch_size=args.batch_size, lr=args.lr,
                   data_path=str(args.data.resolve()),
                   val_data_path=str(args.val_data.resolve()) if args.val_data else None,
                   split_protocol="separate_files" if args.val_data else "sequential_90_10",
                   train_characters=len(train_data), validation_characters=len(val_data),
                   seed=args.seed, device=args.device, temperature=args.temperature,
                   prompt=args.prompt, new_tokens=args.new_tokens, train_seconds=seconds,
                   validation_loss_before=before, validation_loss_after=after, train_loss=history)
    (output / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")
    (output / "sample.txt").write_text(sample, encoding="utf-8")
    torch.save(dict(config=config, stoi=stoi,
                    model_state={k: v.detach().cpu() for k, v in model.state_dict().items()}),
               output / "model.pt")
    print(f"耗时 {seconds:.1f}s；同一验证 batch loss：{before:.4f} → {after:.4f}")
    print("这是小批量诊断，系统评估留到第 8 讲。")
    print("\n生成文本（含前缀）：\n" + sample)
    print(f"\n已保存：{output}")
    return output


if __name__ == "__main__":
    main()
