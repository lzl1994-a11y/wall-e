"""Original-text clause boundaries used by task routing and authorization.

These helpers never choose an executable action. They keep model-provided
evidence attached to its original instruction, including adjacent negation.
Offsets always refer to whitespace-compacted text.
"""

from __future__ import annotations

import re
from typing import Any


_NEXT_OPERATION = (
    r"看|观察|识别|辨认|拍|找|搜|低头|抬头|挥手|点头|摇头|转|向|往|"
    r"前进|后退|停止|关闭|播放|放下|抬|举|告诉"
)
_BOUNDARY = re.compile(
    r"[，,。.!！？?；;：:]+|然后(?:再)?|接着|随后|最后|之后|以后|"
    r"并且|但是|不过|但|"
    rf"(?<!不)(?<!别)(?<!要)(?<!用)再(?=(?:{_NEXT_OPERATION}))|"
    # ASR may omit punctuation before a separate prohibition, e.g.
    # '低头看看不要拍照'. Keep the negator with its following operation.
    rf"(?=(?:不要|别|不准|不许|禁止|不用|不必|不需要)(?:再)?(?:{_NEXT_OPERATION}|保存|存下))"
)
_NEGATION = re.compile(
    r"别(?:再)?|不要(?:再)?|不准|不许|禁止|不需要|不用|不必|无需|"
    r"不想|算了|取消|作罢"
)


def compact_text(value: Any) -> str:
    return "".join(str(value or "").split())


def task_clause_spans(value: Any) -> tuple[tuple[int, int], ...]:
    """Split explicit command boundaries; lists and conditional markers stay.

    In particular, the enumeration comma in ``不要低头、挥手`` does not
    release negation, and ``如果``/``就`` are not sequential boundaries.
    """
    text = compact_text(value)
    spans = []
    start = 0
    for boundary in _BOUNDARY.finditer(text):
        if start < boundary.start():
            spans.append((start, boundary.start()))
        start = boundary.end()
    if start < len(text):
        spans.append((start, len(text)))
    return tuple(spans)


def clause_context(value: Any, start: int, end: int) -> str:
    """Expand a literal span to the complete original clause(s) it touches."""
    text = compact_text(value)
    touched = [(left, right) for left, right in task_clause_spans(text)
               if left < end and start < right]
    if not touched:
        return text[start:end]
    return text[touched[0][0]:touched[-1][1]]


def has_negation(value: Any) -> bool:
    """Conservative negative-scope hint, never a positive command matcher."""
    return bool(_NEGATION.search(compact_text(value)))
