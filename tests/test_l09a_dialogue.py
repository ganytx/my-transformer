"""9A：训练/生成提示保持一致，回复边界不串角色。"""
import copy

import pytest

import harness as H
from src import l09a_dialogue as M


def format_dialogue(turns, user_message=None):
    return H.call(M, "format_dialogue", turns, user_message)


def reply(text):
    return H.call(M, "extract_reply", text)


def test_01_one_turn():
    """01 完整问答精确使用角色名、空格和两个结束换行"""
    assert format_dialogue([("hi", "hello!")]) == "User: hi\nAssistant: hello!\n\n"


def test_02_history_and_pending():
    """02 完整历史后追加待回答问题，末尾只到 Assistant 的空格"""
    turns = [("hi", "hello!"), ("who are you?", "i am tiny.")]
    before = copy.deepcopy(turns)
    want = ("User: hi\nAssistant: hello!\n\n"
            "User: who are you?\nAssistant: i am tiny.\n\n"
            "User: what can you do?\nAssistant: ")
    assert format_dialogue(turns, "what can you do?") == want
    assert turns == before


def test_03_empty_history():
    """03 空历史可用于首轮提示，纯空历史返回空串"""
    assert format_dialogue([]) == ""
    assert format_dialogue([], "hi") == "User: hi\nAssistant: "


def test_04_trim():
    """04 去掉输入两端空白，保留正文内部空格和大小写"""
    assert format_dialogue([("  Hi there  ", "\tHello there!  ")], " next question ") == (
        "User: Hi there\nAssistant: Hello there!\n\nUser: next question\nAssistant: "
    )


def test_05_invalid_content():
    """05 空内容及原字符串的换行被拒绝，避免混入额外角色行"""
    for bad in ("", "  ", "hi\nthere", "hello\r", "\nhello"):
        with pytest.raises(ValueError):
            format_dialogue([(bad, "ok")])
        with pytest.raises(ValueError):
            format_dialogue([("hi", bad)])
        with pytest.raises(ValueError):
            format_dialogue([], bad)


def test_06_stop_at_earliest_marker():
    """06 三种结束标记均生效，多个标记取最早位置"""
    for marker in ("\n\n", "\nUser:", "\nAssistant:"):
        assert reply(" hello! " + marker + " invented next turn") == "hello!"
    assert reply("first\nUser: next\n\nAssistant: later") == "first"
    assert reply("first\n\nmore\nUser: later") == "first"


def test_07_reply_boundaries():
    """07 无标记或空回复正常处理，不误伤行内词语或不完整标记"""
    assert reply("") == ""
    assert reply("  hi!  ") == "hi!"
    assert reply("\n\nUser: next") == ""
    assert reply("the word User: is here") == "the word User: is here"
    assert reply("hello\nUs") == "hello\nUs"


def test_08_training_and_generation_agree():
    """08 待回答提示接上答案与结束符，应与完整训练轮次完全一致"""
    history = [("first question", "first answer")]
    question, answer = "another question", "another answer"
    prefix = format_dialogue(history, question)
    complete = format_dialogue(history + [(question, answer)])
    assert prefix + answer + "\n\n" == complete
    assert reply(complete[len(prefix):]) == answer
