"""从全量原始训练集抽取 20000 段完整对话和 1000 段验证对话；不训练。"""
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import re
import unicodedata

ROOT = Path(__file__).resolve().parent
BASE = ROOT / "data/smol-smoltalk"
DEST = BASE / "sample_20k"
LABELS = {"system": "System", "user": "User", "assistant": "Assistant"}
UNSUPPORTED = re.compile(r"[^\x20-\x7e\n\t]")


def normalize(text):
    text = text.translate(str.maketrans({"‘": "'", "’": "'", "“": '"', "”": '"',
                                       "–": "-", "—": "-", "…": "...", "\u00a0": " "}))
    return unicodedata.normalize("NFKC", text).replace("\r\n", "\n").replace("\r", "\n").strip()


def format_messages(messages):
    return "".join(f"{LABELS[m['role']]}: {m['content']}\n\n" for m in messages)


def main():
    rng = random.Random(42)
    reservoir = []
    seen = set()
    skipped = Counter()
    eligible = 0
    source = BASE / "full/train.jsonl"
    with source.open(encoding="utf-8") as handle:
        for row_id, line in enumerate(handle):
            row = json.loads(line)
            original = row["messages"]
            if (not original or original[-1]["role"] != "assistant"
                    or any(m["role"] not in LABELS for m in original)
                    or not any(m["role"] == "user" for m in original)):
                skipped["roles"] += 1
                continue
            messages = [{"role": m["role"], "content": normalize(m["content"])} for m in original]
            if any(not m["content"] for m in messages):
                skipped["empty_content"] += 1
                continue
            text = format_messages(messages)
            if UNSUPPORTED.search(text):
                skipped["non_ascii_or_control"] += 1
                continue
            opening = next(m["content"] for m in messages if m["role"] == "user")
            key = hashlib.sha256(" ".join(opening.casefold().split()).encode()).hexdigest()
            if key in seen:
                skipped["duplicate_opening"] += 1
                continue
            seen.add(key)
            eligible += 1
            record = dict(messages=messages, source=row["source"], source_id=f"original_train:{row_id}")
            if len(reservoir) < 21000:
                reservoir.append(record)
            else:
                index = rng.randrange(eligible)
                if index < 21000:
                    reservoir[index] = record
    assert len(reservoir) == 21000, eligible
    rng.shuffle(reservoir)
    train, candidates = reservoir[:20000], reservoir[20000:]
    # 仅训练集建词表。未知验证字符不复制进训练集。
    train_vocab = set("".join(format_messages(r["messages"]) for r in train))
    validation = [r for r in candidates if set(format_messages(r["messages"])) <= train_vocab]
    assert len(validation) == 1000, "Validation contains unseen characters; review normalization"
    DEST.mkdir(parents=True, exist_ok=True)
    summary = {}
    for split, records in (("train", train), ("validation", validation)):
        text = "".join(format_messages(r["messages"]) for r in records)
        (DEST / f"{split}.txt").write_text(text, encoding="utf-8", newline="\n")
        (DEST / f"{split}.jsonl").write_text(
            "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records), encoding="utf-8", newline="\n")
        summary[split] = dict(conversations=len(records), messages=sum(len(r["messages"]) for r in records),
                              characters=len(text), max_conversation_characters=max(len(format_messages(r["messages"])) for r in records),
                              sources=dict(Counter(r["source"] for r in records)),
                              text_sha256=hashlib.sha256(text.encode()).hexdigest())
    manifest = dict(seed=42, source="../full/train.jsonl", source_revision="f73fe857d519ff6ac5af2ea67c4d3834da7b8bcc",
                    eligible_unique_conversations=eligible, excluded=dict(skipped), vocab_size=len(train_vocab), splits=summary,
                    protocol="Uniform reservoir sample of 21000 eligible unique openings, shuffled and split 20000/1000. Complete system/user/assistant messages retained; punctuation normalized; printable ASCII plus newline/tab; no length restriction or answer truncation. Official test untouched.")
    (DEST / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(manifest, indent=2), flush=True)


if __name__ == "__main__":
    main()
