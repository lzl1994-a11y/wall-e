"""Structured visual-search plans and the ROS leaf-node protocol."""

from __future__ import annotations

import json
import base64
from dataclasses import dataclass
from collections.abc import Callable
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
    allowed = {
        "target", "question", "search_direction", "motion_duration", "max_views",
        "on_found_actions",
    }
    if not set(value) <= allowed or "target" not in value:
        raise ValueError("visual_search_fields_invalid")
    target = value.get("target")
    question = value.get("question", "")
    direction = value.get("search_direction", "spin")
    duration = value.get("motion_duration", 1)
    max_views = value.get("max_views", 3)
    on_found_actions = value.get("on_found_actions", [])
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
    if not isinstance(on_found_actions, list) or len(on_found_actions) > 7:
        raise ValueError("visual_search_completion_actions_invalid")
    normalized_actions: list[dict[str, Any]] = []
    for index, action in enumerate(on_found_actions, start=1):
        if not isinstance(action, dict) or set(action) != {"name", "arguments"}:
            raise ValueError(f"visual_search_completion_action_{index}_invalid")
        name = action.get("name")
        arguments = action.get("arguments")
        if not isinstance(name, str) or not name.strip() or not isinstance(arguments, dict):
            raise ValueError(f"visual_search_completion_action_{index}_invalid")
        skill = get_action_skill(name.strip())
        if skill is None or not skill.action_bus:
            raise ValueError(f"visual_search_completion_action_{index}_unknown")
        normalized_actions.append({"name": name.strip(), "arguments": dict(arguments)})
    return {
        "target": target.strip(),
        "question": question.strip() or f"{target.strip()}在哪里",
        "search_direction": direction,
        "motion_duration": duration,
        "max_views": max_views,
        "on_found_actions": normalized_actions,
    }


@dataclass(frozen=True)
class VisualSearchPlan:
    plan_id: str
    turn_id: str
    target: str
    question: str
    max_views: int
    steps: tuple[ActionPlanStep, ...]
    on_found_actions: tuple[dict[str, Any], ...] = ()
    schema_version: int = 3
    root_type: str = "VisualSearch"

    def to_dict(self) -> dict[str, Any]:
        payload = {
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
        if self.on_found_actions:
            payload["on_found_actions"] = [dict(item) for item in self.on_found_actions]
        return payload


@dataclass(frozen=True)
class VisualSearchPreparation:
    plan: VisualSearchPlan | None = None
    rejection: dict[str, Any] | None = None


@dataclass(frozen=True)
class VisualSearchViewDecision:
    result: dict[str, str]
    error: str | None = None


class VisualSearchViewWorkflow:
    """Capture and assess one correlated visual-search view without ROS."""

    def __init__(
        self,
        *,
        capture: Callable[[], Any],
        evaluate: Callable[[str, str, str], dict[str, str]],
    ) -> None:
        self._capture = capture
        self._evaluate = evaluate

    def invoke(self, request: dict[str, Any]) -> VisualSearchViewDecision:
        try:
            preview = self._capture()
            frame = getattr(preview, "last_frame", None)
            if getattr(preview, "busy", False) or not frame:
                return VisualSearchViewDecision(result={
                    "status": "uncertain",
                    "evidence": getattr(preview, "error", None)
                    or "camera_frame_unavailable",
                    "response": "这次没有取得可判断的画面。",
                })
            result = self._evaluate(
                request["target"],
                request["question"],
                base64.b64encode(frame).decode("ascii"),
            )
            parsed = parse_visual_search_result(result)
            if parsed is None:
                raise ValueError("visual_search_result_invalid")
            return VisualSearchViewDecision(result=parsed)
        except Exception as exc:
            return VisualSearchViewDecision(
                result={
                    "status": "uncertain",
                    "evidence": "visual_search_view_failed",
                    "response": "这次画面没有分析成功。",
                },
                error=str(exc),
            )


class VisualSearchWorkflow:
    """Prepare and normalize visual-search execution without ROS coupling."""

    def __init__(
        self,
        *,
        authorize: Callable[[str, dict[str, Any]], tuple[bool, str]],
    ) -> None:
        self._authorize = authorize

    def prepare(self, *, turn_id: str, arguments: Any) -> VisualSearchPreparation:
        try:
            plan = compile_visual_search_plan(turn_id=turn_id, arguments=arguments)
        except (TypeError, ValueError) as exc:
            return VisualSearchPreparation(rejection={
                "status": "rejected",
                "action": VISUAL_SEARCH_TOOL_NAME,
                "reason": str(exc),
            })
        for step in plan.steps[1:]:
            allowed, reason = self._authorize(step.name, step.arguments)
            if not allowed:
                return VisualSearchPreparation(rejection={
                    "status": "rejected",
                    "action": VISUAL_SEARCH_TOOL_NAME,
                    "reason": f"completion_action_invalid:{step.step_id}:{reason}",
                })
        return VisualSearchPreparation(plan=plan)

    @staticmethod
    def complete(
        plan: VisualSearchPlan,
        execution: dict[str, Any] | None,
    ) -> dict[str, Any]:
        if execution is None:
            return {
                "status": "failed",
                "action": VISUAL_SEARCH_TOOL_NAME,
                "reason": "native_behavior_tree_unavailable",
            }
        search_results = [
            item for item in execution.get("results", [])
            if item.get("action") == VISUAL_SEARCH_TOOL_NAME
        ]
        if execution.get("status") == "success" and search_results:
            result = dict(search_results[-1])
            result["status"] = "completed"
            return result
        return {
            "status": "failed",
            "action": VISUAL_SEARCH_TOOL_NAME,
            "reason": execution.get("error") or execution.get("status") or "search_failed",
        }


def compile_visual_search_plan(*, turn_id: str, arguments: dict[str, Any]) -> VisualSearchPlan:
    normalized = normalize_visual_search_arguments(arguments)
    skill = get_action_skill("move_chassis")
    if skill is None or not skill.action_bus:
        raise ValueError("visual_search_motion_skill_unavailable")
    max_views = normalized["max_views"]
    steps = [ActionPlanStep(
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
    )]
    previous_step_id = steps[0].step_id
    for index, action in enumerate(normalized["on_found_actions"], start=2):
        skill = get_action_skill(action["name"])
        if skill is None:  # Normalization above guarantees this.
            raise ValueError(f"visual_search_completion_action_{index}_unknown")
        step_id = f"step-{index:02d}"
        steps.append(ActionPlanStep(
            step_id=step_id,
            name=action["name"],
            arguments=dict(action["arguments"]),
            grounding="",
            depends_on=(previous_step_id,),
            resources=skill.plan_resources,
            timeout_ms=skill.timeout_ms,
            max_attempts=skill.max_attempts,
        ))
        previous_step_id = step_id
    completion_actions = tuple(normalized["on_found_actions"])
    return VisualSearchPlan(
        plan_id=f"search-{uuid4().hex}",
        turn_id=str(turn_id or ""),
        target=normalized["target"],
        question=normalized["question"],
        max_views=max_views,
        steps=tuple(steps),
        on_found_actions=completion_actions,
        schema_version=4 if completion_actions else 3,
        root_type="VisualSearchThen" if completion_actions else "VisualSearch",
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
    "VisualSearchPreparation",
    "VisualSearchViewDecision",
    "VisualSearchViewWorkflow",
    "VisualSearchWorkflow",
    "compile_visual_search_plan",
    "encode_visual_search_status",
    "normalize_visual_search_arguments",
    "parse_visual_search_request",
    "parse_visual_search_result",
    "visual_search_result_tool",
]
