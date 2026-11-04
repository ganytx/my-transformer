"""加载兼容的 MiniGPT 权重，以对话格式生成；不要求先微调。"""
import argparse
from pathlib import Path

import torch

from dialogue_runtime import latest_checkpoint, load_model, reply_to


def main(argv=None):
    parser = argparse.ArgumentParser(description="/reset 清历史，/quit 退出。")
    parser.add_argument("--checkpoint", type=Path, help="任一兼容的 model.pt；默认取 runs 中修改时间最新的权重")
    parser.add_argument("--prompt", help="单次问题；省略时进入多轮终端")
    parser.add_argument("--tokens", type=int, default=96)
    parser.add_argument("--temperature", type=float, default=0.)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args(argv)
    if args.tokens <= 0 or args.temperature < 0:
        parser.error("tokens 为正，temperature 非负")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("没有可用 CUDA")
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    try:
        checkpoint = args.checkpoint or latest_checkpoint()
        model, stoi, _ = load_model(checkpoint, args.device)
    except (ValueError, FileNotFoundError) as exc:
        parser.error(str(exc))
    model.eval()
    print(f"模型：{checkpoint.parent.name}；上下文上限 {model.block_size} 字符")
    history = []

    def answer(question):
        response, _ = reply_to(model, stoi, history, question, args.tokens, args.temperature)
        if response:
            history.append((question.strip(), response))
            print("Assistant: " + response)
        else:
            print("[模型没有生成非空回复；可尝试 /reset 或更短的问题。]")

    if args.prompt is not None:
        try:
            answer(args.prompt)
        except ValueError as exc:
            parser.error(str(exc))
        return
    print("/reset 清空历史；/quit 退出。")
    while True:
        try:
            question = input("\nUser: ")
        except (EOFError, KeyboardInterrupt):
            print("\n已退出。")
            break
        command = question.strip()
        if command == "/quit":
            break
        if command == "/reset":
            history.clear()
            print("[历史已清空。]")
            continue
        try:
            answer(question)
        except ValueError as exc:
            print(f"[输入问题：{exc}]")


if __name__ == "__main__":
    main()
