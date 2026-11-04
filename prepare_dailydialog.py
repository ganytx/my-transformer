"""将已下载的 DailyDialog 原始 ZIP 转为短对话；不创建模型、不启动训练。"""
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
import unicodedata
import zipfile

from src.l09a_dialogue import format_dialogue

ROOT = Path(__file__).resolve().parent
DEST = ROOT / "data/dailydialog"
REVISION = "5214b2a66405abf87fd229e5c1007985501ffe3e"
MAX_CHARACTERS = 256


def normalize(text):
    text = text.translate(str.maketrans({
        "‘": "'", "’": "'", "“": '"', "”": '"',
        "–": "-", "—": "-", "…": "...", "。": ".",
    }))
    text = unicodedata.normalize("NFKC", text)
    text = " ".join(text.split())
    # 原始语料将许多标点分词，用于字符模型时合并常见空格。
    text = re.sub(r"\s+([,.!?;:])", r"\1", text)
    text = re.sub(r"(?<=[A-Za-z])\s*'\s*(?=[A-Za-z])", "'", text)
    return text


def main():
    manifest = dict(
        dataset="DailyDialog", original_card="https://huggingface.co/datasets/li2017dailydialog/daily_dialog",
        mirror="https://huggingface.co/datasets/roskoN/dailydialog",
        revision=REVISION, license="CC-BY-NC-SA-4.0", max_formatted_characters=MAX_CHARACTERS,
        protocol="Original splits; opening two utterances only; normalized ASCII; whole pair length <=256; unique casefolded user across all splits; heldout vocabulary must be covered by train.",
        splits={},
    )
    seen_questions = set()
    train_characters = None
    for split in ("train", "validation", "test"):
        archive_path = DEST / "raw" / f"{split}.zip"
        with zipfile.ZipFile(archive_path) as archive:
            lines = archive.read(f"{split}/dialogues_{split}.txt").decode("utf-8").splitlines()
        pairs = []
        skipped = Counter()
        for index, line in enumerate(lines):
            turns = line.split("__eou__")
            if len(turns) < 2:
                skipped["not_enough_turns"] += 1
                continue
            question, answer = map(normalize, turns[:2])
            if not question or not answer:
                skipped["empty"] += 1
                continue
            if not all(32 <= ord(c) <= 126 for c in question + answer):
                skipped["non_ascii"] += 1
                continue
            formatted = format_dialogue([(question, answer)])
            if len(formatted) > MAX_CHARACTERS:
                skipped["too_long"] += 1
                continue
            key = question.casefold()
            if key in seen_questions:
                skipped["duplicate_user"] += 1
                continue
            if train_characters is not None and set(formatted) - train_characters:
                skipped["unseen_character"] += 1
                continue
            seen_questions.add(key)
            pairs.append(dict(user=question, assistant=answer, source_id=f"{split}:{index}"))
        text = format_dialogue([(r["user"], r["assistant"]) for r in pairs])
        if split == "train":
            train_characters = set(text)
        assert len(text) > MAX_CHARACTERS
        (DEST / f"{split}.txt").write_text(text, encoding="utf-8", newline="\n")
        (DEST / f"{split}.jsonl").write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in pairs),
            encoding="utf-8", newline="\n",
        )
        manifest["splits"][split] = dict(
            source_dialogues=len(lines), prepared_pairs=len(pairs), characters=len(text),
            max_pair_characters=max(len(format_dialogue([(r["user"], r["assistant"])])) for r in pairs),
            skipped=dict(skipped), raw_bytes=archive_path.stat().st_size,
            raw_sha256=hashlib.sha256(archive_path.read_bytes()).hexdigest(),
            prepared_text_sha256=hashlib.sha256(text.encode("utf-8")).hexdigest(),
            download_url=f"https://huggingface.co/datasets/roskoN/dailydialog/resolve/{REVISION}/{split}.zip",
        )
    (DEST / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
