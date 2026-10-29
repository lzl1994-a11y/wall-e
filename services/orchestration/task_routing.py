"""Choose one outer planning route without inventing executable actions.

The one-shot conditional planner can represent one observation and one action.
Requests with independent steps or constraints outside that shape must stay in
the complete ordered planner, even if they also contain a conditional phrase.
"""

from __future__ import annotations

import re
from typing import Any, Literal

from services.action.action_intent_guard import is_non_execution_request
from services.orchestration.conditional_task import is_conditional_task_request
from services.orchestration.task_text import compact_text, task_clause_spans


TaskRoute = Literal["conditional_task", "ordered_task"]

_MOTION_CUE = re.compile(
    r"低头|把头低|抬头|转头|扭头|转身|转过身|转过去|转一圈|转个圈|转圈|"
    r"转(?:向|到|一下)|左转|右转|向左|向右|往左|往右|回正|挥手|招手|抬手|举手|"
    r"点头|摇头|放下.*手|前进|后退|往前|往后|向前|向后|走.*步|跳舞|"
    r"(?:抬|举)(?:起|一下)?(?:左|右|双|两只).*(?:手|胳膊)|(?:左|右|双|两只).*手.*(?:抬|举)"
)
_ORDER_CUE = re.compile(r"先|然后|再|接着|之后|以后|随后|并且|同时")
_VISUAL_CUE = re.compile(
    r"看看|看一看|看一下|看下|看一眼|再看(?!着我|住我)|才能看|才看|观察|识别|辨认|拍照|拍.*照片|"
    r"拍.*照|找|寻找|搜索|在哪|有什么|瞅|瞧"
)
_OTHER_ACTION_CUE = re.compile(
    r"(?:播放|停止|停下|关闭|关掉|打开|开启|启动|启用|退出).{0,8}"
    r"(?:音乐|歌曲|视觉|跟踪|跟随|动作)|"
    r"(?:音乐|歌曲|视觉|跟踪|跟随).{0,8}(?:播放|停止|停下|关闭|关掉)|"
    r"跟着我|跟随我|盯着我|看着我|笑一个|开心表情|好奇表情|"
    r"拍照|拍(?:张|一张|个)?照|保存.{0,6}(?:照片|图片|相片)"
)
_OBSERVATION_CUE = re.compile(r"看看|看一看|看一下|看下|看一眼|观察|识别|辨认|找|寻找|搜索|瞅|瞧")
_EXPLICIT_CONDITION = re.compile(r"如果|假如|要是|一旦|只要|当(?=.{1,80}时)")
_SENSORY_CONDITION = re.compile(r"看到|看见|发现|检测到|识别到")
_CONSEQUENT = re.compile(r"就|便|则|然后|请你|请(?!问)|帮我|让你")


def _clauses(text: str) -> list[str]:
    return [text[start:end] for start, end in task_clause_spans(text)]


def _independent_action(clause: str) -> bool:
    """Exclude actions describing the subject of an observation.

    Negated independent actions still count: the prohibition is a constraint
    that the one-action conditional representation cannot silently discard.
    """
    matches = [match for pattern in (_MOTION_CUE, _OTHER_ACTION_CUE)
               if (match := pattern.search(clause)) is not None]
    if not matches:
        return False
    first = min(match.start() for match in matches)
    observed = _OBSERVATION_CUE.search(clause)
    return observed is None or first < observed.start()


def _condition_start(text: str) -> int:
    explicit = _EXPLICIT_CONDITION.search(text)
    if explicit is not None:
        return explicit.start()
    sensory = _SENSORY_CONDITION.search(text)
    if sensory is not None:
        return sensory.start()
    # The legacy matcher also accepts "前面没人的话就举手". Its condition
    # starts after the preceding independent clause, if one was specified.
    suffix = text.find("的话")
    if suffix >= 0:
        spans = task_clause_spans(text[:suffix])
        return spans[-1][0] if spans else 0
    return 0


def _compound_conditional(text: str) -> bool:
    start = _condition_start(text)
    if any(_independent_action(part) for part in _clauses(text[:start])):
        return True
    conditional = text[start:]
    result = _CONSEQUENT.search(conditional)
    if result is None:
        return False
    consequence = conditional[result.end():]
    # A second condition or several requested result steps cannot be encoded
    # by the existing single-action conditional plan. Ordered validation may
    # reject an unsupported combination; a partial conditional plan may not run.
    if _EXPLICIT_CONDITION.search(consequence):
        return True
    parts = _clauses(consequence)
    requested = [part for part in parts
                 if _independent_action(part) or _VISUAL_CUE.search(part)]
    if len(requested) > 1:
        return True
    for part in requested:
        if _independent_action(part) and sum(
            len(list(pattern.finditer(part))) for pattern in (_MOTION_CUE, _OTHER_ACTION_CUE)
        ) > 1:
            return True
        motion, visual = _MOTION_CUE.search(part), _VISUAL_CUE.search(part)
        if motion is not None and visual is not None and motion.start() < visual.start():
            return True
    return False


def select_task_route(text: Any) -> TaskRoute | None:
    """Return the shared outer route; this is not an action authorizer."""
    compact = compact_text(text)
    if not compact or is_non_execution_request(compact):
        return None
    if is_conditional_task_request(compact):
        return "ordered_task" if _compound_conditional(compact) else "conditional_task"
    if (_VISUAL_CUE.search(compact)
            and (_MOTION_CUE.search(compact) or _ORDER_CUE.search(compact))):
        return "ordered_task"
    return None


__all__ = ["TaskRoute", "select_task_route"]
