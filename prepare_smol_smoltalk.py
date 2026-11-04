"""流式整理官方 smol-smoltalk：完整 messages JSONL + 当前字符模型可用的短对话。"""
from collections import Counter
import hashlib
import json
from pathlib import Path
import sys
import unicodedata

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / ".private/data_deps"))
import pyarrow.parquet as pq

from src.l09a_dialogue import format_dialogue

DEST = ROOT / "data/smol-smoltalk"
MAX_CHARS = 512


def normalize(text):
    # 不压平多行内容，不截断答案；只统一常见排版标点。
    text = text.translate(str.maketrans({
        "‘": "'", "’": "'", "“": '"', "”": '"',
        "–": "-", "—": "-", "…": "...", "\u00a0": " ",
    }))
    return unicodedata.normalize("NFKC", text).strip()


def digest(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def main():
    source_manifest = json.loads((DEST / "download_manifest.json").read_text(encoding="utf-8"))
    (DEST / "full").mkdir(exist_ok=True)
    (DEST / "short_512").mkdir(exist_ok=True)
    selected = {split: [] for split in ("train", "validation", "test")}
    seen = set()
    counts = {}
    full_manifest = {}
    for split in ("train", "test"):
        skipped = Counter()
        sources = Counter()
        rows = 0
        full_hash = hashlib.sha256()
        paths = sorted((DEST / "raw").glob(f"{split}-*.parquet"))
        assert len(paths) == (4 if split == "train" else 1)
        with (DEST / "full" / f"{split}.jsonl").open("wb") as full:
            for path in paths:
                parquet = pq.ParquetFile(path)
                row_number = 0
                for batch in parquet.iter_batches(batch_size=512):
                    for row in batch.to_pylist():
                        encoded = (json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n").encode("utf-8")
                        full.write(encoded)
                        full_hash.update(encoded)
                        row_id = f"{path.name}:{row_number}"
                        row_number += 1
                        rows += 1
                        sources[row["source"]] += 1
                        messages = row["messages"]
                        if len(messages) < 2 or [m["role"] for m in messages[:2]] != ["user", "assistant"]:
                            skipped["initial_roles_or_system_message"] += 1
                            continue
                        q, a = (normalize(m["content"]) for m in messages[:2])
                        if not q or not a:
                            skipped["empty"] += 1
                            continue
                        if "\n" in q + a or "\r" in q + a:
                            skipped["multiline"] += 1
                            continue
                        if not all(32 <= ord(c) <= 126 for c in q + a):
                            skipped["non_ascii_or_control"] += 1
                            continue
                        formatted = format_dialogue([(q, a)])
                        if len(formatted) > MAX_CHARS:
                            skipped["longer_than_512"] += 1
                            continue
                        # 相同规范化开场跨集合只保留一次；不丢弃不同答案中的部分字符。
                        key = digest(" ".join(q.casefold().split()))
                        if key in seen:
                            skipped["duplicate_opening"] += 1
                            continue
                        seen.add(key)
                        target = "test" if split == "test" else ("validation" if int(key[:8], 16) % 20 == 0 else "train")
                        selected[target].append(dict(user=q, assistant=a, source=row["source"], source_id=row_id))
                    if rows and rows % 51200 == 0:
                        print(f"{split}: exported {rows:,} full conversations", flush=True)
                assert row_number == parquet.metadata.num_rows
                print(f"Prepared {path.name}: {row_number:,} conversations", flush=True)
        assert rows == (460341 if split == "train" else 24229)
        counts[split] = dict(original_conversations=rows, excluded_from_short=dict(skipped), sources=dict(sources))
        full_manifest[split] = dict(rows=rows, bytes=(DEST / "full" / f"{split}.jsonl").stat().st_size, sha256=full_hash.hexdigest())

    train_text = format_dialogue([(r["user"], r["assistant"]) for r in selected["train"]])
    vocab = set(train_text)
    short_manifest = {}
    for split, records in selected.items():
        before = len(records)
        if split != "train":
            records = [r for r in records if set(format_dialogue([(r["user"], r["assistant"])])) <= vocab]
        assert records, f"No usable {split} examples"
        text = format_dialogue([(r["user"], r["assistant"]) for r in records])
        jsonl = "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records)
        (DEST / "short_512" / f"{split}.txt").write_text(text, encoding="utf-8", newline="\n")
        (DEST / "short_512" / f"{split}.jsonl").write_text(jsonl, encoding="utf-8", newline="\n")
        short_manifest[split] = dict(
            pairs=len(records), characters=len(text), bytes=len(text.encode("utf-8")),
            max_formatted_characters=max(len(format_dialogue([(r["user"], r["assistant"])])) for r in records),
            unknown_character_exclusions=before-len(records), text_sha256=digest(text),
            sources=dict(Counter(r["source"] for r in records)),
        )
    manifest = dict(
        repo=source_manifest["repo"], revision=source_manifest["revision"], license="Apache-2.0",
        preparation="full is a lossless messages/source export. short_512 keeps only complete opening user/assistant pairs, with no initial system message, no internal newlines, printable ASCII after punctuation normalization, <=512 formatted characters. No answer truncation. Whitespace/case-normalized opening deduplication across splits; 5% hash buckets of original train become validation, original test stays test. Heldout characters must be covered by train.",
        original=counts, full=full_manifest, short_512=short_manifest, short_vocab_size=len(vocab),
        pyarrow_version="26.0.1",
    )
    (DEST / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(short_manifest, ensure_ascii=False, indent=2), flush=True)
    print("Preparation complete. No model loaded; no training invoked.", flush=True)


if __name__ == "__main__":
    main()
