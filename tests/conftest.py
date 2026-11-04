"""pytest 全局配置。

- 把仓库根目录和 tests/ 放进 sys.path；
- 固定线程数，避免多线程归约顺序造成的浮点抖动；
- 当环境变量 CHECK_REPORT 存在时，把每个测试的结果写成 JSON 给 check.py 渲染。
"""

from __future__ import annotations

import json
import os
import re
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT), str(ROOT / "tests")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import torch  # noqa: E402

torch.set_num_threads(1)

_REPORT = os.environ.get("CHECK_REPORT")
_DOCS: dict[str, str] = {}
_RECORDS: list[dict] = []
_ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")


def pytest_collection_modifyitems(config, items):
    for item in items:
        func = getattr(item, "function", None)
        doc = (getattr(func, "__doc__", "") or "").strip()
        _DOCS[item.nodeid] = doc.splitlines()[0].strip() if doc else ""


def _short(rep, limit=1500):
    text = _ANSI.sub("", str(rep.longrepr))
    lines = [ln for ln in text.splitlines()]
    while lines and not lines[-1].strip():
        lines.pop()
    text = "\n".join(lines)
    if len(text) > limit:
        text = text[:limit] + "\n  ...（已截断，用 -v 看完整信息）"
    return text


@pytest.hookimpl(wrapper=True)
def pytest_runtest_makereport(item, call):
    rep = yield
    if rep.when == "call" or rep.outcome == "failed":
        _RECORDS.append(
            {
                "nodeid": rep.nodeid,
                "when": rep.when,
                "outcome": rep.outcome,
                "duration": round(getattr(rep, "duration", 0.0), 4),
                "doc": _DOCS.get(rep.nodeid, ""),
                "repr": _short(rep) if rep.outcome != "passed" else "",
            }
        )
    return rep


def pytest_sessionfinish(session, exitstatus):
    if not _REPORT:
        return
    payload = {"exitstatus": int(exitstatus), "records": _RECORDS}
    try:
        Path(_REPORT).write_text(json.dumps(payload, ensure_ascii=False), "utf-8")
    except OSError:
        pass


@pytest.fixture(autouse=True)
def _fresh_rng():
    torch.manual_seed(0)
    yield
