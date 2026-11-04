"""现有小说模型的交互续写入口；尚不是问答模型。"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from src.l03_bigram import encode, decode
from src.l07a_gpt import MiniGPT
from src.l07b_train_generate import generate

ROOT = Path(__file__).resolve().parent


def main(argv=None):
    parser = argparse.ArgumentParser(description="与当前字符模型交互续写，每次输入是独立前缀。")
    parser.add_argument("--checkpoint", type=Path, required=True, help="指定你训练的 model.pt")
    parser.add_argument("--prompt", help="单次续写；省略时进入终端交互")
    parser.add_argument("--tokens", type=int, default=160)
    parser.add_argument("--temperature", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    args = parser.parse_args(argv)
    if args.tokens < 0 or args.temperature < 0:
        parser.error("tokens 和 temperature 必须非负")
    if not args.checkpoint.is_file():
        parser.error("找不到 checkpoint，请通过 --checkpoint 指定 model.pt")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("当前没有可用 CUDA")
    torch.set_num_threads(1)
    torch.manual_seed(args.seed)
    saved = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    model = MiniGPT(**saved["config"]).to(args.device)
    model.load_state_dict(saved["model_state"])
    model.eval()
    stoi = saved["stoi"]
    itos = {value: key for key, value in stoi.items()}
    print(f"加载：{args.checkpoint.name}（{args.checkpoint.parent.name}）")
    print("当前模型学习的是英文小说续写，尚未训练问答。每次输入是独立前缀。")

    def complete(prompt):
        if not prompt:
            raise ValueError("前缀不能为空")
        missing = sorted(set(prompt) - set(stoi))
        if missing:
            raise ValueError(f"词表不包含这些字符：{missing!r}；请使用当前英文语料中的字符")
        idx = encode(prompt, stoi)[None].to(args.device)
        out = generate(model, idx, args.tokens, args.temperature)
        return decode(out[0, idx.shape[1]:].cpu(), itos)

    if args.prompt is not None:
        try:
            print("续写：" + complete(args.prompt))
        except ValueError as exc:
            parser.error(str(exc))
        return
    print("输入英文前缀，输入 /quit 退出。")
    while True:
        try:
            prompt = input("\n前缀> ")
        except (EOFError, KeyboardInterrupt):
            print("\n已退出。")
            break
        if prompt.strip() == "/quit":
            break
        try:
            print("续写> " + complete(prompt))
        except ValueError as exc:
            print(str(exc))


if __name__ == "__main__":
    main()
