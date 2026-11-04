"""9A：英文对话格式和回复边界。自测 python check.py 9a。"""
from __future__ import annotations


def format_dialogue(turns: list[tuple[str, str]], user_message: str | None = None) -> str:
    """将完整问答轮次转换成训练文本，或构造等待模型回答的提示。

    每个 (question, answer) 精确输出：
      User: {question}\nAssistant: {answer}\n\n
    若 user_message 不为 None，末尾追加：
      User: {user_message}\nAssistant: 
    注意待回答前缀最后保留一个空格，没有答案或末尾换行。
    每段输入去除两端空白；若去除后为空，或原字符串包含 \n/\r，抛 ValueError。
    turns 为空且 user_message=None 时返回空串。不修改 turns。
    输入其余类型保证有效。
    """
    
    if user_message!=None and ('\n' in user_message or '\r' in user_message or not user_message.strip()):raise ValueError("")
    t = []
    for q,a in turns:
        if '\n' in q or '\r' in q or not q.strip():raise ValueError("")
        if '\n' in a or '\r' in a or not a.strip():raise ValueError("")
        question = q.strip()
        answer = a.strip()
        t.append(f'User: {question}\nAssistant: {answer}\n\n')
    if user_message!=None:t.append(f'User: {user_message.strip()}\nAssistant: ')
    return ''.join(t)

def extract_reply(continuation: str) -> str:
    """从新生成的文本中取出第一条回复。

    continuation 仅含新生成部分，不含 prompt。
    找到最早出现的结束标记：\n\n、\nUser:、\nAssistant:。
    返回其之前的文本并 strip；没有结束标记则对整段 strip。
    空输入返回空串。未完成的标记如 \nUs 不算结束标记。
    """
    if not continuation:return ""
    markers = ("\n\n", "\nUser:", "\nAssistant:")
    end = len(continuation)
    for marker in markers:
        idx = continuation.find(marker)
        if idx != -1 and idx < end:
            end = idx
    return continuation[:end].strip()
