"""Validated plans and decisions for one-shot conditional robot tasks.

The language model may describe the observation and condition, but it cannot
invent executable code.  A plan contains exactly one camera observation and
at most one pre-registered robot action.  Runtime validation is repeated at
the graph boundary before any ROS command is published.
"""

from __future__ import annotations

import json
import re
from typing import Any, Literal, TypedDict

from services.action_intent_guard import (
    registered_sequence_names,
    validate_action_arguments,
)


CONDITIONAL_TASK_TOOL_NAME = "run_conditional_task"

# A single still image is not a safe basis for autonomous chassis motion.
# Locomotion can be added later behind continuous perception and obstacle
# avoidance.  The remaining tools are bounded or stop/safety operations.
CONDITIONAL_ACTION_TOOLS = frozenset({
    "express_emotion",
    "play_sequence",
    "set_tracking_mode",
    "set_vision_gate",
    "stop_all",
})

_CONDITIONAL_MARKER_RE = re.compile(
    r"(?:如果|假如|要是|一旦|只要|当.+?时|看到|看见|发现|检测到|识别到).{0,80}"
    r"(?:就|便|则|然后|请你|你就|帮我|让你)"
)


class ConditionalTaskPlan(TypedDict):
    observation: str
    condition: str
    action_name: str
    action_arguments: dict[str, Any]


class ConditionalDecision(TypedDict):
    decision: Literal["yes", "no", "uncertain"]
    evidence: str


def is_conditional_task_request(text: Any) -> bool:
    """Return whether text has an explicit observe-condition-action shape.

    This matcher only prevents the existing one-shot camera shortcut from
    swallowing a compound request.  The model still creates the semantic plan
    and the runtime validator remains the authority for side effects.
    """
    compact = "".join(str(text or "").split())
    return bool(compact and _CONDITIONAL_MARKER_RE.search(compact))


def normalize_conditional_task_plan(value: Any) -> ConditionalTaskPlan:
    """Validate and normalize a model-proposed conditional task plan."""
    if not isinstance(value, dict):
        raise ValueError("conditional_plan_not_object")
    allowed_keys = {"observation", "condition", "action_name", "action_arguments"}
    if set(value) != allowed_keys:
        raise ValueError("conditional_plan_fields_invalid")

    observation = value.get("observation")
    condition = value.get("condition")
    action_name = value.get("action_name")
    action_arguments = value.get("action_arguments")
    if not isinstance(observation, str) or not observation.strip():
        raise ValueError("conditional_observation_missing")
    if not isinstance(condition, str) or not condition.strip():
        raise ValueError("conditional_condition_missing")
    if len(observation.strip()) > 500 or len(condition.strip()) > 500:
        raise ValueError("conditional_text_too_long")
    if action_name not in CONDITIONAL_ACTION_TOOLS:
        raise ValueError("conditional_action_not_allowed")
    allowed, reason = validate_action_arguments(action_name, action_arguments)
    if not allowed:
        raise ValueError(f"conditional_action_{reason}")

    return {
        "observation": observation.strip(),
        "condition": condition.strip(),
        "action_name": action_name,
        "action_arguments": dict(action_arguments),
    }


def parse_conditional_decision(value: Any) -> ConditionalDecision:
    """Parse the vision model's closed decision vocabulary, failing closed."""
    if isinstance(value, str):
        text = value.strip()
        if text.startswith("```"):
            text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text, flags=re.IGNORECASE)
        try:
            value = json.loads(text)
        except (TypeError, json.JSONDecodeError):
            return {"decision": "uncertain", "evidence": "invalid_model_output"}
    if not isinstance(value, dict):
        return {"decision": "uncertain", "evidence": "invalid_model_output"}

    decision = value.get("decision")
    evidence = value.get("evidence", "")
    if decision not in {"yes", "no", "uncertain"}:
        return {"decision": "uncertain", "evidence": "invalid_model_decision"}
    if not isinstance(evidence, str):
        evidence = ""
    return {"decision": decision, "evidence": evidence.strip()[:500]}


def conditional_task_tool_schema() -> dict[str, Any]:
    """Return provider-visible constraints for the compound-task tool."""
    return {
        "type": "object",
        "properties": {
            "observation": {
                "type": "string",
                "maxLength": 500,
                "description": "需要摄像头观察的问题，例如前方有什么、用户手里有什么",
            },
            "condition": {
                "type": "string",
                "maxLength": 500,
                "description": "仅依据当前画面判断的条件，保留肯定或否定含义",
            },
            "action_name": {
                "type": "string",
                "enum": sorted(CONDITIONAL_ACTION_TOOLS),
                "description": (
                    "条件明确成立时执行的一个动作工具。举手、点头、挥手、转头等预设"
                    "身体动作必须使用 play_sequence；情绪身体表达使用 express_emotion；"
                    "持续注视或跟随使用 set_tracking_mode；停止全部动作使用 stop_all"
                ),
            },
            "action_arguments": {
                "type": "object",
                "properties": {
                    "sequence_name": {
                        "type": "string",
                        "enum": list(registered_sequence_names()),
                        "description": (
                            "play_sequence 专用：举右手=right_hand_up，举左手=left_hand_up，"
                            "举双手=arms_up，举手示意=raise_hand，点头=basic_nod，"
                            "挥手=wave_hello，放下双手=arms_down，回正=look_center"
                        ),
                    },
                    "emotion": {
                        "type": "string",
                        "enum": [
                            "curious", "happy", "sad", "surprised", "disdain", "angry"
                        ],
                        "description": "express_emotion 专用",
                    },
                    "mode": {
                        "type": "string",
                        "enum": ["follow_me", "look_at_me", "idle"],
                        "description": "set_tracking_mode 专用",
                    },
                    "enabled": {
                        "type": "boolean",
                        "description": "set_vision_gate 专用",
                    },
                },
                "additionalProperties": False,
                "description": (
                    "对应动作的参数对象；只能填写该动作需要的一个参数。stop_all 使用空对象"
                ),
            },
        },
        "required": ["observation", "condition", "action_name", "action_arguments"],
        "additionalProperties": False,
    }


__all__ = [
    "CONDITIONAL_ACTION_TOOLS",
    "CONDITIONAL_TASK_TOOL_NAME",
    "ConditionalDecision",
    "ConditionalTaskPlan",
    "conditional_task_tool_schema",
    "is_conditional_task_request",
    "normalize_conditional_task_plan",
    "parse_conditional_decision",
]
