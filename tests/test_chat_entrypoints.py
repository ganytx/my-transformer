"""教师运行入口的回归检查：仅装载/数据解析，不训练。"""
import json
from pathlib import Path
from unittest.mock import patch

import pytest
import torch

import chat7
import dialogue_runtime as runtime
import finetune_chat
from src.l07a_gpt import MiniGPT


@pytest.mark.parametrize("kind", [None, "toy_dialogue"])
def test_chat_accepts_both_training_entrypoints(monkeypatch, capsys, kind):
    config = dict(vocab_size=2, block_size=8, d_model=4, n_heads=2, n_layers=1)
    model = MiniGPT(**config)
    checkpoint = dict(config=config, stoi={"a": 0, "b": 1}, model_state=model.state_dict())
    if kind is not None:
        checkpoint["kind"] = kind
    path = Path("in_memory_checkpoint.pt")
    monkeypatch.setattr(torch, "load", lambda *args, **kwargs: checkpoint)
    # /quit exercises loading through the actual CLI without generating or training.
    with patch("builtins.input", return_value="/quit"):
        chat7.main(["--checkpoint", str(path)])
    assert "上下文上限 8" in capsys.readouterr().out
    restored, _, _ = runtime.load_model(path)
    assert all(torch.equal(v, restored.state_dict()[k]) for k, v in model.state_dict().items())


def test_jsonl_accepts_metadata_and_blank_lines(monkeypatch):
    path = Path("pairs.jsonl")
    record = dict(user=" hello ", assistant="hi", source_id="train:0")
    monkeypatch.setattr(Path, "read_text", lambda *args, **kwargs: json.dumps(record) + "\n\n")
    assert finetune_chat.read_pairs(path) == [("hello", "hi")]


@pytest.mark.parametrize("content, message", [
    ("User: hi\nAssistant: hello\n", "JSONL"),
    ('{"user": 7, "assistant": "hi"}', "字符串字段"),
    ('{"user": "hi", "assistant": ""}', "非空"),
    ('{"user": "hi\\nthere", "assistant": "hi"}', "换行"),
    ("\n", "没有有效问答"),
])
def test_bad_data_has_actionable_error(monkeypatch, content, message):
    path = Path("bad.txt")
    monkeypatch.setattr(Path, "read_text", lambda *args, **kwargs: content)
    with pytest.raises(ValueError, match=message):
        finetune_chat.read_pairs(path)
