"""Recognize an explicit sequence followed by a one-shot camera observation."""

from __future__ import annotations

import re
from typing import Any

from services.action.action_intent_guard import (
    resolve_explicit_sequence_action,
    validate_action_call,
)


_ORDER_SEPARATOR = re.compile(r"(?:然后|接着|随后|之后|以后|再)")
_OBSERVATION = re.compile(r"(?:看看|看一看|看一下|看下|看一眼|瞅瞅|瞧瞧|观察|识别|辨认)")
_BARE_LOOK = re.compile(r"^(?:才能|才)?看(?:吧|呀|啊)?[。！？!?]*$")
_NON_CAMERA_LOOK = re.compile(r"(?:看电影|看电视|看视频|看剧|看书|看新闻|看着我|盯着我|注视我|跟着我)")


def action_before_inspection(user_prompt: Any) -> list[dict[str, Any]] | None:
    """Build a grounded pose/sequence step only for an explicit ordered request.

    Unknown or ambiguous wording stays on the existing dialog path. This does
    not invent arbitrary actions or make camera observations conditional.
    """
    text = "".join(str(user_prompt or "").split())
    if not text or _NON_CAMERA_LOOK.search(text):
        return None
    segments = _ORDER_SEPARATOR.split(text)
    if len(segments) > 1:
        action_texts = [part.strip("，,；;。") for part in segments[:-1]]
        observation = segments[-1].lstrip("，,；;。")
    else:
        observation_match = _OBSERVATION.search(text)
        if not observation_match:
            return None
        action_texts = [text[:observation_match.start()].rstrip("，,；;。")]
        observation = text[observation_match.start():]
    if not action_texts or len(action_texts) > 4 or not (_OBSERVATION.search(observation) or _BARE_LOOK.fullmatch(observation)):
        return None
    actions = []
    for action_text in action_texts:
        if not action_text or re.search(r"并且|同时|并|和|又", action_text):
            return None
        action = resolve_explicit_sequence_action(action_text)
        if action is None:
            return None
        name, arguments = action
        if not validate_action_call(text, name, arguments)[0]:
            return None
        actions.append({"name": name, "arguments": arguments, "grounding": action_text})
    return actions
