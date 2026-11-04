"""判题公共工具库。

tests/ 下所有测试都从这里取断言函数、固定种子和黄金数据。
学生不需要读这个文件，但可以读——判题标准是公开的。
"""

from __future__ import annotations

import ast
import hashlib
import inspect
import io
import json
import textwrap
import tokenize
from pathlib import Path

import torch

# --------------------------------------------------------------------------- #
# 全局判题参数（改动这里等于改动判题标准，参考解必须同步验证）
# --------------------------------------------------------------------------- #
ROOT = Path(__file__).resolve().parents[1]
GOLDEN_DIR = ROOT / "tests" / "golden"
MANIFEST = GOLDEN_DIR / "manifest.json"

ATOL = 1e-9          # 绝对容差（float64 下的数值比对）
RTOL = 1e-7          # 相对容差
DTYPE = torch.float64


def randn(*shape, seed=0, dtype=DTYPE):
    """固定种子的标准正态张量（与 numpy/pytorch 默认 RNG 解耦）。"""
    g = torch.Generator().manual_seed(seed)
    return torch.randn(*shape, generator=g, dtype=dtype)


def rand(*shape, seed=0, dtype=DTYPE, lo=0.0, hi=1.0):
    g = torch.Generator().manual_seed(seed)
    return lo + (hi - lo) * torch.rand(*shape, generator=g, dtype=dtype)


def arange_t(*shape, seed=0, dtype=DTYPE):
    """整数张量，常用于构造 token id。"""
    g = torch.Generator().manual_seed(seed)
    n = 1
    for s in shape:
        n *= s
    return torch.randint(0, 100, (*shape,), generator=g).to(torch.long)


# --------------------------------------------------------------------------- #
# 调用包装：接口缺失 / 未实现 都变成可读的失败信息（而不是 collection error）
# --------------------------------------------------------------------------- #
def call(module, name, *args, **kwargs):
    if not hasattr(module, name):
        raise AssertionError(
            f"{module.__name__} 里找不到 `{name}`。\n"
            f"  接口名必须和讲义一致，不要改名、不要删函数。"
        )
    fn = getattr(module, name)
    if not callable(fn):
        raise AssertionError(f"`{name}` 存在但不是函数（当前类型 {type(fn).__name__}）。")
    try:
        return fn(*args, **kwargs)
    except NotImplementedError as exc:
        raise AssertionError(
            f"`{name}` 还没有实现（NotImplementedError）：{exc}\n"
            f"  去 {module.__name__.replace('.', '/')}.py 里把它写完。"
        ) from None


def make(module, name, *args, **kwargs):
    """实例化一个类（同样把接口缺失变成可读错误）。"""
    return call(module, name, *args, **kwargs)


# --------------------------------------------------------------------------- #
# 断言
# --------------------------------------------------------------------------- #
def _typename(x):
    return type(x).__name__


def expect_tensor(x, what="返回值"):
    if not isinstance(x, torch.Tensor):
        raise AssertionError(
            f"{what}应该是 torch.Tensor，实际是 {_typename(x)}。\n"
            f"  如果是 Python float / list / numpy 数组，请用 torch.tensor(...) 转换后再返回。"
        )
    return x


def assert_shape(actual, expected, what="输出"):
    expect_tensor(actual, what)
    exp = tuple(expected) if not isinstance(expected, torch.Size) else tuple(expected)
    if tuple(actual.shape) != exp:
        raise AssertionError(
            f"{what}的 shape 不对：期望 {exp}，实际 {tuple(actual.shape)}。\n"
            f"  常见原因：忘了 keepdim、sum 的方向错了、或者少加了一个 batch 维。"
        )
    return actual


def assert_dtype(actual, expected, what="输出"):
    expect_tensor(actual, what)
    if actual.dtype != expected:
        raise AssertionError(
            f"{what}的 dtype 不对：期望 {expected}，实际 {actual.dtype}。\n"
            f"  softmax 这类运算不应改变 dtype；不要用 .float() 或 .double() 强转。"
        )
    return actual


def assert_finite(actual, what="输出"):
    expect_tensor(actual, what)
    if not torch.isfinite(actual).all():
        bad = (~torch.isfinite(actual)).nonzero()
        i = tuple(bad[0].tolist())
        raise AssertionError(
            f"{what}里出现了 inf / nan（共 {bad.shape[0]} 个），第一个位置 {i}，值为 {actual[i]}。\n"
            f"  exp 在大数上会溢出：先用该轴的最大值做平移。"
        )
    return actual


def assert_close(actual, expected, what="输出", atol=ATOL, rtol=RTOL, case=""):
    """逐元素比对，失败时给出最大误差位置与数值（即"最小失败案例"）。"""
    expect_tensor(actual, what)
    expect_tensor(expected, "期望输出")
    if tuple(actual.shape) != tuple(expected.shape):
        raise AssertionError(
            f"{what}的 shape 不对：期望 {tuple(expected.shape)}，实际 {tuple(actual.shape)}。{case}"
        )
    a = actual.detach().to(torch.float64)
    e = expected.detach().to(torch.float64)
    if not torch.isfinite(a).all():
        assert_finite(a, what)
    diff = (a - e).abs()
    tol = atol + rtol * e.abs()
    bad = diff > tol
    if bad.any():
        flat = diff.masked_fill(~bad, -1.0)
        idx = torch.argmax(flat.flatten()).item()
        pos = tuple(torch.unravel_index(torch.tensor(idx), diff.shape))
        n_bad = int(bad.sum())
        raise AssertionError(
            f"{what}数值不对：{n_bad} / {diff.numel()} 个元素超出容差（atol={atol:g}, rtol={rtol:g}）。"
            + (f"\n  场景：{case}" if case else "")
            + f"\n  最大误差 {diff[pos].item():.3e}，出现在索引 {pos}："
            + f"\n    期望 {e[pos].item():.10g}"
            + f"\n    实际 {a[pos].item():.10g}"
        )
    return actual


def assert_equal_tensor(actual, expected, what="输出"):
    expect_tensor(actual, what)
    if tuple(actual.shape) != tuple(expected.shape):
        raise AssertionError(f"{what}的 shape 不对：期望 {tuple(expected.shape)}，实际 {tuple(actual.shape)}。")
    if not torch.equal(actual.to(expected.dtype), expected):
        raise AssertionError(f"{what}与期望不完全相同。")
    return actual


def assert_true(cond, msg):
    if not cond:
        raise AssertionError(msg)


# --------------------------------------------------------------------------- #
# 源码检查（反作弊 / 实现约束）
# --------------------------------------------------------------------------- #
def module_source(module):
    try:
        return inspect.getsource(module)
    except (OSError, TypeError) as exc:  # pragma: no cover
        raise AssertionError(f"读不到 {module.__name__} 的源码：{exc}") from None


def _blank_comments_and_strings(source: str) -> str:
    """把注释和字符串字面量替换成空格（保持字符位置不变），便于做子串检查。"""
    chars = list(source)
    lines = source.splitlines(keepends=True)

    def offset(pos):
        row, col = pos
        return sum(len(l) for l in lines[: row - 1]) + col

    try:
        for tok in tokenize.generate_tokens(io.StringIO(source).readline):
            if tok.type in (tokenize.COMMENT, getattr(tokenize, "FSTRING_MIDDLE", -1)):
                s, e = offset(tok.start), offset(tok.end)
                for i in range(s, e):
                    if chars[i] not in "\r\n":
                        chars[i] = " "
            elif tok.type == tokenize.STRING:
                s, e = offset(tok.start), offset(tok.end)
                for i in range(s, e):
                    if chars[i] not in "\r\n":
                        chars[i] = " "
    except (tokenize.TokenError, IndentationError):
        return source
    return "".join(chars)


def assert_no_builtin(module, banned, why="本题要求自己实现，不能直接调用 PyTorch 的现成实现"):
    """banned 是子串列表，例如 ['torch.softmax', 'F.softmax', '.logsumexp(']。"""
    src = _blank_comments_and_strings(module_source(module))
    hits = [b for b in banned if b in src]
    if hits:
        raise AssertionError(
            f"{module.__name__} 里出现了被禁用的调用：{', '.join(hits)}\n"
            f"  {why}。\n"
            f"  这类题的意义就在于自己把公式翻译成张量运算。"
        )


def assert_no_loops(module, func_names, why="本题要求用张量运算整体完成"):
    """禁止在指定函数里出现 for / while / 推导式。"""
    src = textwrap.dedent(module_source(module))
    tree = ast.parse(src)
    banned_nodes = (ast.For, ast.While, ast.ListComp, ast.SetComp, ast.DictComp, ast.GeneratorExp)
    names = set(func_names) if not isinstance(func_names, str) else {func_names}
    found = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in names:
            for sub in ast.walk(node):
                if isinstance(sub, banned_nodes):
                    found.append((node.name, type(sub).__name__, getattr(sub, "lineno", -1)))
    if found:
        f, kind, line = found[0]
        kind_cn = {
            "For": "for 循环",
            "While": "while 循环",
            "ListComp": "列表推导式",
            "SetComp": "集合推导式",
            "DictComp": "字典推导式",
            "GeneratorExp": "生成器表达式",
        }.get(kind, kind)
        raise AssertionError(
            f"函数 `{f}` 里用了{kind_cn}（源码第 {line} 行左右）。\n"
            f"  {why}：\n"
            f"  想一想：这个循环的每一轮做的事，能不能用一次 reshape / 一次矩阵乘 / 一次自动广播表达？"
        )


# --------------------------------------------------------------------------- #
# 黄金数据：期望输出由独立参考实现预先算好，测试时只做比对
# --------------------------------------------------------------------------- #
_CASES: dict[str, tuple] = {}


def golden_input(level: str, name: str, apply, tensor: torch.Tensor) -> torch.Tensor:
    """登记一个黄金用例：apply(参考实现模块, 输入张量) -> 期望输出。

    测试文件在模块级别调用它获得输入，`.private/gen_golden.py` 也用同一份登记表生成期望值，
    这样输入永远不会两边不一致。
    """
    _CASES[f"{level}/{name}"] = (level, name, apply, tensor)
    return tensor


def all_cases() -> dict:
    return dict(_CASES)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_manifest() -> dict:
    if not MANIFEST.exists():
        return {}
    return json.loads(MANIFEST.read_text("utf-8"))


def golden(level: str, name: str) -> torch.Tensor:
    path = GOLDEN_DIR / level / f"{name}.pt"
    if not path.exists():
        raise AssertionError(
            f"缺少黄金数据 {path.relative_to(ROOT)}。\n"
            f"  这是判题环境损坏，请在仓库根目录运行 `python oj.py doctor` 并告诉我。"
        )
    manifest = load_manifest()
    key = f"{level}/{name}.pt"
    want = manifest.get(key)
    got = _sha256(path)
    if want is None:
        raise AssertionError(f"黄金数据 {key} 没有登记在 manifest.json 里，判题环境损坏了。")
    if want != got:
        raise AssertionError(
            f"黄金数据 {key} 的哈希对不上——文件被改动过。\n"
            f"  改黄金数据是没有用的：删掉 tests/golden 重新生成需要参考实现。\n"
            f"  请把这条错误告诉我。"
        )
    return torch.load(path, map_location="cpu", weights_only=True)
