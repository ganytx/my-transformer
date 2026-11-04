"""从已有 smol-smoltalk 中选 2000 条短问答及 100 条独立验证题，不训练。"""
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import re
from prepare_smoltalk_20k import normalize
from src.l09a_dialogue import format_dialogue

ROOT = Path(__file__).resolve().parent
DEST = ROOT / "data/smol-smoltalk/short_replies_2000"
SOURCES = {"everyday-conversations": 0, "smol-magpie-ultra-short": 1, "openhermes-50k": 2, "self-oss-instruct": 3}
GREETINGS = {"hi", "hello", "hey", "hi there", "hello there", "hey there", "good morning", "good afternoon", "good evening"}
META = re.compile(r"\b(rewrite|summarize|summarise|translate|translation|classify|classification|paraphrase|sentence|string|regex|javascript|python|sql|html)\b", re.I)
BAD_CHAR = re.compile(r"[^\x20-\x7e]")


def key(text):
    return " ".join(text.casefold().split())


def main():
    candidates = {}
    exclusions = Counter()
    with (ROOT / "data/smol-smoltalk/full/train.jsonl").open(encoding="utf-8") as handle:
        for row_id, line in enumerate(handle):
            row = json.loads(line)
            if row["source"] not in SOURCES:
                continue
            messages = row["messages"]
            if len(messages) < 2 or [m["role"] for m in messages[:2]] != ["user", "assistant"]:
                continue
            # 只允许首轮，或仅含寒暄的首轮之后的下一条问答，不截取依赖实质历史的后续轮次。
            indices = [0]
            greeting = normalize(messages[0]["content"]).casefold().strip(".!? ")
            greeting_answer = normalize(messages[1]["content"])
            if (greeting in GREETINGS and len(greeting_answer) <= 96
                    and greeting_answer.casefold().startswith(("hello", "hi", "hey", "good"))
                    and len(messages) >= 4 and [m["role"] for m in messages[2:4]] == ["user", "assistant"]):
                indices.append(2)
            for index in indices:
                q, a = (normalize(m["content"]) for m in messages[index:index+2])
                if not q or not a or BAD_CHAR.search(q+a):
                    exclusions["empty_multiline_or_non_ascii"] += 1
                    continue
                if len(a) > 160 or len(q) > 320 or len(format_dialogue([(q, a)])) > 512:
                    exclusions["too_long"] += 1
                    continue
                if META.search(q) or "```" in q+a:
                    exclusions["text_transformation_or_code"] += 1
                    continue
                if index and q.casefold().startswith(("that", "what about", "and ", "yes", "no,", "it ", "they ", "these ", "those ")):
                    exclusions["likely_history_dependent"] += 1
                    continue
                record = dict(user=q, assistant=a, source=row["source"], source_id=f"original_train:{row_id}", message_index=index)
                k = key(q)
                if k not in candidates or SOURCES[row["source"]] < SOURCES[candidates[k]["source"]]:
                    candidates[k] = record
    records = sorted(candidates.values(), key=lambda r: (SOURCES[r["source"]], hashlib.sha256(key(r["user"]).encode()).hexdigest()))
    assert len(records) >= 2100, len(records)
    selected = records[:2100]
    random.Random(42).shuffle(selected)
    train, val = selected[:2000], selected[2000:]
    vocab = set(format_dialogue([(r["user"],r["assistant"]) for r in train]))
    assert set(format_dialogue([(r["user"],r["assistant"]) for r in val])) <= vocab
    DEST.mkdir(parents=True, exist_ok=True)
    stats = {}
    for split, rows in (("train",train),("validation",val)):
        text = format_dialogue([(r["user"],r["assistant"]) for r in rows])
        (DEST/f"{split}.txt").write_text(text,encoding="utf-8",newline="\n")
        (DEST/f"{split}.jsonl").write_text("".join(json.dumps(r,ensure_ascii=False)+"\n" for r in rows),encoding="utf-8",newline="\n")
        stats[split] = dict(pairs=len(rows),characters=len(text),sources=dict(Counter(r["source"] for r in rows)),
                            max_answer_characters=max(len(r["assistant"]) for r in rows),
                            max_formatted_characters=max(len(format_dialogue([(r["user"],r["assistant"])])) for r in rows),
                            sha256=hashlib.sha256(text.encode()).hexdigest())
    manifest = dict(source="HuggingFaceTB/smol-smoltalk",revision="f73fe857d519ff6ac5af2ea67c4d3834da7b8bcc",seed=42,
                    eligible_unique_questions=len(candidates),exclusions=dict(exclusions),splits=stats,
                    protocol="First user/assistant pair or first substantive pair after greeting-only exchange; no substantive history removed. ASCII single-line, answer<=160 characters, question<=320, formatted pair<=512. Exclude common rewriting/summarization/code tasks. Prefer everyday conversations, then short magpie, openhermes and self-oss. Unique normalized questions, disjoint train/validation; reference replies kept without truncation, punctuation normalized. Context dependence filter is heuristic, not a semantic guarantee.")
    (DEST/'manifest.json').write_text(json.dumps(manifest,indent=2)+'\n',encoding="utf-8")
    print(json.dumps(manifest,indent=2),flush=True)
    print('First 12 selected training questions:',flush=True)
    for r in train[:12]:
        print(json.dumps(dict(user=r['user'],assistant=r['assistant']),ensure_ascii=True),flush=True)


if __name__ == '__main__':
    main()
