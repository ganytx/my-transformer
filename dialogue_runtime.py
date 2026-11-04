"""对话脚本的装载和终端适配；模型、格式化和采样均复用学生实现。"""
from pathlib import Path

import torch

from src.l03_bigram import encode, decode
from src.l07a_gpt import MiniGPT
from src.l07b_train_generate import generate
from src.l09a_dialogue import format_dialogue, extract_reply

ROOT = Path(__file__).resolve().parent


def load_model(path, device="cpu", context=None):
    saved = torch.load(path, map_location="cpu", weights_only=True)
    config = dict(saved["config"])
    if context is not None:
        if context <= 0:
            raise ValueError("context 必须为正")
        config["block_size"] = context
    model = MiniGPT(**config)
    state = saved["model_state"]
    if config["block_size"] != saved["config"]["block_size"]:
        # 固定正弦表随窗口重新生成；所有可训练参数仍从 checkpoint 装载。
        result = model.load_state_dict({k: v for k, v in state.items() if k != "pe"}, strict=False)
        if result.missing_keys != ["pe"] or result.unexpected_keys:
            raise ValueError(f"checkpoint 结构不匹配：{result}")
    else:
        model.load_state_dict(state)
    for name, parameter in model.named_parameters():
        if not torch.equal(parameter.detach(), state[name]):
            raise ValueError(f"未正确恢复参数 {name}")
    return model.to(device), saved["stoi"], config


def reply_to(model, stoi, history, question, max_new_tokens=96, temperature=0.):
    """保留可容纳的完整历史轮次，只把新生成部分交给学生的回复截取函数。"""
    kept = list(history)
    prompt = format_dialogue(kept, question)
    while len(prompt) > model.block_size and kept:
        kept.pop(0)
        prompt = format_dialogue(kept, question)
    if len(prompt) > model.block_size:
        raise ValueError(f"本轮提示超过 {model.block_size} 个字符，请缩短问题")
    missing = sorted(set(prompt) - set(stoi))
    if missing:
        raise ValueError(f"当前英文词表不包含：{missing!r}")
    device = next(model.parameters()).device
    idx = encode(prompt, stoi)[None].to(device)
    out = generate(model, idx, max_new_tokens, temperature)
    itos = {value: key for key, value in stoi.items()}
    continuation = decode(out[0, idx.shape[1]:].cpu(), itos)
    answer = extract_reply(continuation)
    # 本次数据协议要求单行问答；显示和下一轮历史使用同一个规范化结果。
    answer = " ".join(answer.split())
    return answer, continuation


def latest_checkpoint():
    paths = list((ROOT / "runs").glob("*/model.pt"))
    if not paths:
        raise ValueError("runs 中没有 model.pt；请先自行训练，或用 --checkpoint 指定已有模型")
    return max(paths, key=lambda path: (path.stat().st_mtime_ns, str(path)))
