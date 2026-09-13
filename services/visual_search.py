"""Structured visual-search plans and the ROS leaf-node protocol."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any
from uuid import uuid4

from services.action_plan import ActionPlanStep
from services.action_registry import get_action_skill


VISUAL_SEARCH_TOOL_NAME = "search_environment"
VISUAL_SEARCH_REQUEST_TOPIC = "/visual_search/request"
VISUAL_SEARCH_STATUS_TOPIC = "/visual_search/status"
VISUAL_SEARCH_RESULT_TOOL_NAME = "visual_search_result"

SEARCH_DIRECTIONS = frozenset({"spin", "left", "right"})


def normalize_visual_search_arguments(value: Any) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise ValueError("visual_search_not_object")
    allowed = {"target", "question", "search_direction", "motion_duration", "max_views"}
    if not set(value) <= allowed or "target" not in value:
        raise ValueError("visual_search_fields_invalid")
    target = value.get("target")
    question = value.get("question", "")
    direction = value.get("search_direction", "spin")
    duration = value.get("motion_duration", 1)
    max_views = value.get("max_views", 3)
    if not isinstance(target, str) or not target.strip() or len(target.strip()) > 200:
        raise ValueError("visual_search_target_invalid")
    if not isinstance(question, str) or len(question.strip()) > 500:
        raise ValueError("visual_search_question_invalid")
    if direction not in SEARCH_DIRECTIONS:
        raise ValueError("visual_search_direction_invalid")
    if isinstance(duration, bool) or not isinstance(duration, int) or not 1 <= duration <= 3:
        raise ValueError("visual_search_duration_invalid")
    if isinstance(max_views, bool) or not isinstance(max_views, int) or not 2 <= max_views <= 4:
        raise ValueError("visual_search_views_invalid")
    return {
        "target": target.strip(),
        "question": question.strip() or f"{target.strip()}在哪里",
        "search_direction": direction,
        "motion_duration": duration,
        "max_views": max_views,
    }


@dataclass(frozen=True)
class VisualSearchPlan:
    plan_id: str
    turn_id: str
    target: str
    question: str
    max_views: int
    steps: tuple[ActionPlanStep, ...]
    schema_version: int = 3
    root_type: str = "VisualSearch"

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "plan_id": self.plan_id,
            "turn_id": self.turn_id,
            "root_type": self.root_type,
            "target": self.target,
            "question": self.question,
            "max_views": self.max_views,
            "steps": [step.to_dict() for step in self.steps],
            "on_failure": "report_not_found",
        }


def compile_visual_search_plan(*, turn_id: str, arguments: dict[str, Any]) -> VisualSearchPlan:
    normalized = normalize_visual_search_arguments(arguments)
    skill = get_action_skill("move_chassis")
    if skill is None or not skill.action_bus:
        raise ValueError("visual_search_motion_skill_unavailable")
    max_views = normalized["max_views"]
    step = ActionPlanStep(
        step_id="step-01",
        name="move_chassis",
        arguments={
            "direction": normalized["search_direction"],
            "duration": normalized["motion_duration"],
        },
        grounding="",
        depends_on=(),
        resources=skill.plan_resources,
        timeout_ms=skill.timeout_ms,
        max_attempts=max_views - 1,
    )
    return VisualSearchPlan(
        plan_id=f"search-{uuid4().hex}",
        turn_id=str(turn_id or ""),
        target=normalized["target"],
        question=normalized["question"],
        max_views=max_views,
        steps=(step,),
    )


def visual_search_result_tool() -> dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": VISUAL_SEARCH_RESULT_TOOL_NAME,
            "description": "报告当前图片中是否找到搜索目标及其位置，不执行动作。",
            "parameters": {
                "type": "object",
                "properties": {
                    "status": {
                        "type": "string",
                        "enum": ["found", "not_found", "uncertain"],
                    },
                    "evidence": {"type": "string", "maxLength": 500},
                    "response": {
                        "type": "string",
                        "maxLength": 500,
                        "description": "适合直接向用户播报的简短观察结果",
                    },
                },
                "required": ["status", "evidence", "response"],
                "additionalProperties": False,
            },
        },
    }


def parse_visual_search_result(value: Any) -> dict[str, str] | None:
    if not isinstance(value, dict):
        return None
    status = value.get("status")
    evidence = value.get("evidence")
    response = value.get("response")
    if (
        status not in {"found", "not_found", "uncertain"}
        or not isinstance(evidence, str)
        or not isinstance(response, str)
        or len(evidence) > 500
        or len(response) > 500
    ):
        return None
    return {
        "status": status,
        "evidence": evidence.strip(),
        "response": response.strip(),
    }


def parse_visual_search_request(payload: Any) -> dict[str, Any] | None:
    try:
        data = json.loads(payload) if isinstance(payload, str) else payload
    except (TypeError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict):
        return None
    request_id = data.get("request_id")
    plan_id = data.get("plan_id")
    target = data.get("target")
    question = data.get("question")
    attempt = data.get("attempt")
    max_views = data.get("max_views")
    if (
        not isinstance(request_id, str) or not request_id
        or not isinstance(plan_id, str) or not plan_id
        or not isinstance(target, str) or not target
        or not isinstance(question, str) or not question
        or isinstance(attempt, bool) or not isinstance(attempt, int) or attempt < 1
        or isinstance(max_views, bool) or not isinstance(max_views, int)
        or not 2 <= max_views <= 4 or attempt > max_views
    ):
        return None
    return dict(data)


def encode_visual_search_status(request: dict[str, Any], result: dict[str, str]) -> str:
    parsed = parse_visual_search_result(result)
    if parsed is None:
        raise ValueError("invalid_visual_search_result")
    return json.dumps({
        "request_id": request["request_id"],
        "plan_id": request["plan_id"],
        **parsed,
    }, ensure_ascii=False, separators=(",", ":"))


__all__ = [
    "SEARCH_DIRECTIONS",
    "VISUAL_SEARCH_REQUEST_TOPIC",
    "VISUAL_SEARCH_RESULT_TOOL_NAME",
    "VISUAL_SEARCH_STATUS_TOPIC",
    "VISUAL_SEARCH_TOOL_NAME",
    "VisualSearchPlan",
    "compile_visual_search_plan",
    "encode_visual_search_status",
    "normalize_visual_search_arguments",
    "parse_visual_search_request",
    "parse_visual_search_result",
    "visual_search_result_tool",
]
