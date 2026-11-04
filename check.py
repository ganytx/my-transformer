#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""check.py —— 这一讲写得对不对，跑一下就知道。

    python check.py 1       只跑第 1 讲
    python check.py         跑所有已发布讲次
    python check.py 1 -v    失败时打印完整 traceback

它不是"判题平台"，只是一面镜子：把每一讲该满足的条件逐条列出来，
哪一条没过、期望什么、你给了什么、差多少。
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import time
import unicodedata
from pathlib import Path

ROOT = Path(__file__).resolve().parent

# (编号, 标题, 测试文件, 你要写的文件)
STEPS = [
    ("1", "张量与广播", "tests/test_l01_tensors.py", "src/l01_tensors.py"),
    ("2", "训练闭环", "tests/test_l02_train.py", "src/l02_train.py"),
    ("2b", "手写反向传播", "tests/test_l02b_backward.py", "src/l02b_backward.py"),
    ("3", "字符级 bigram", "tests/test_l03_bigram.py", "src/l03_bigram.py"),
    ("4", "上下文窗口", "tests/test_l04_context.py", "src/l04_context.py"),
    ("5", "单头因果注意力", "tests/test_l05_attention.py", "src/l05_attention.py"),
    ("6a", "多头因果注意力", "tests/test_l06a_multihead.py", "src/l06a_multihead.py"),
    ("6b1", "LayerNorm", "tests/test_l06b1_layernorm.py", "src/l06b1_layernorm.py"),
    ("6b2", "FFN 与 Transformer 块", "tests/test_l06b2_block.py", "src/l06b2_block.py"),
    ("7a", "Mini-GPT 前向与损失", "tests/test_l07a_gpt.py", "src/l07a_gpt.py"),
    ("7b", "训练与自回归生成", "tests/test_l07b_train_generate.py", "src/l07b_train_generate.py"),
    ("8a", "全语料评估与困惑度", "tests/test_l08a_evaluate.py", "src/l08a_evaluate.py"),
    ("9a", "英文对话格式与回复边界", "tests/test_l09a_dialogue.py", "src/l09a_dialogue.py"),
]

# 用户优先做英文对话；评估关卡保留，可随时用 python check.py 8a 单独运行。
DEFERRED_STEPS = {"8a"}


def dwidth(s: str) -> int:
    return sum(2 if unicodedata.east_asian_width(ch) in "WF" else 1 for ch in s)


def pad(s: str, width: int) -> str:
    return s + " " * max(0, width - dwidth(s))


def run_pytest(paths, verbose=False):
    fd, tmp = tempfile.mkstemp(suffix=".json", prefix="check_")
    os.close(fd)
    cmd = [sys.executable, "-m", "pytest", *paths, "-q", "--no-header"]
    env = os.environ.copy()
    env["CHECK_REPORT"] = tmp
    env["PYTHONIOENCODING"] = "utf-8"
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    t0 = time.time()
    proc = subprocess.run(cmd, cwd=str(ROOT), env=env, capture_output=True,
                          text=True, encoding="utf-8", errors="replace")
    elapsed = time.time() - t0
    try:
        data = json.loads(Path(tmp).read_text("utf-8"))
    except (OSError, json.JSONDecodeError):
        data = {}
    finally:
        try:
            os.unlink(tmp)
        except OSError:
            pass
    return data.get("records", []), proc, elapsed


def print_records(records, verbose=False, max_detail=3, width=46):
    bad = [r for r in records if r["outcome"] != "passed"]
    for i, r in enumerate(records, 1):
        mark = "✔" if r["outcome"] == "passed" else ("✘" if r["when"] == "call" else "⚠")
        doc = r["doc"] or r["nodeid"].split("::")[-1]
        head, _, tail = doc.partition(" ")
        if tail and head.isdigit():
            doc = tail
        sec = f" {r['duration']:.2f}s" if r["duration"] >= 0.5 else ""
        print(f"  {mark} {pad(f'{i:02d} {doc}', width)}{sec}")
    if bad:
        print()
        for r in bad[:max_detail]:
            kind = "断言失败" if r["when"] == "call" else "导入 / 初始化失败"
            print(f"  ── {kind} · {r['doc'] or r['nodeid']} ──")
            body = r.get("repr") or ""
            if not verbose:
                keep = [ln.strip() for ln in body.splitlines() if ln.strip().startswith("E ")]
                if not keep:
                    keep = [ln for ln in body.splitlines() if ln.strip()][-10:]
                body = "\n".join(keep)
            for ln in body.splitlines()[:26]:
                print("     " + ln)
            print()
        if len(bad) > max_detail:
            print(f"  （另外 {len(bad) - max_detail} 处没通过，先修上面这几个）\n")
    return len(records) - len(bad), len(records)


def main(argv=None) -> int:
    args = [a for a in (argv if argv is not None else sys.argv[1:]) if not a.startswith("-")]
    verbose = "-v" in (argv if argv is not None else sys.argv[1:])

    if not STEPS:
        print("还没有发布任何讲次。")
        return 0

    if args:
        want = args[0]
        picked = [s for s in STEPS if s[0] == want]
        if not picked:
            print(f"没有第 {want} 讲。已发布：{' '.join(s[0] for s in STEPS)}")
            return 2
    else:
        picked = [s for s in STEPS if (ROOT / s[2]).exists() and s[0] not in DEFERRED_STEPS]
        if DEFERRED_STEPS:
            print("暂缓关卡不计入本次检查：" + ", ".join(sorted(DEFERRED_STEPS)) + "（仍可指定编号单独检查）")

    all_ok = True
    for num, title, test_rel, src_rel in picked:
        test = ROOT / test_rel
        if not test.exists():
            print(f"第 {num} 讲还没发布。")
            continue
        print("═" * 62)
        print(f"  第 {num} 讲 · {title}")
        print(f"  你要改的文件：{src_rel}")
        print("═" * 62)
        print()
        records, proc, elapsed = run_pytest([str(test)], verbose=verbose)
        if not records:
            print("  测试在收集阶段就失败了（语法错误，或者 import 不到 src 里的模块）：")
            print()
            text = (proc.stdout or "") + (proc.stderr or "")
            for ln in [ln for ln in text.splitlines() if ln.strip()][-20:]:
                print("     " + ln)
            print()
            all_ok = False
            continue
        passed, total = print_records(records, verbose=verbose)
        print(f"  ─ 通过 {passed}/{total} · 用时 {elapsed:.1f}s ─")
        if passed == total:
            print("  全部通过。把代码发我，我要问三个问题再放你过。")
        else:
            print("  先修上面列出的问题。想看完整 traceback：python check.py %s -v" % num)
        print()
        all_ok &= passed == total

    return 0 if all_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
