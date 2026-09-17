"""Pure service for preparing, parsing, canonicalizing, and validating conditional task plans."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional, Sequence

from services.action_intent_guard import (
    canonicalize_conditional_action,
    validate_action_call,
)
from services.conditional_task import CONDITIONAL_TASK_TOOL_NAME

STATUS_IGNORED = "ignored"
STATUS_MALFORMED = "malformed"
STATUS_REJECTED = "rejected"
STATUS_ACCEPTED = "accepted"

CONDITIONAL_FALLBACK_SYSTEM_PROMPT = (
    "你是机器人条件任务规划器，只做受限 JSON 转换，不观察环境、不执行动作。"
)
CONDITIONAL_FALLBACK_MAX_TOKENS = 384


@dataclass(frozen=True)
class ConditionalPlanDecision:
    status: str
    plan: Optional[Any] = None
    rejection_reason: Optional[str] = None

    @property
    def is_ignored(self) -> bool:
        return self.status == STATUS_IGNORED

    @property
    def is_malformed(self) -> bool:
        return self.status == STATUS_MALFORMED

    @property
    def is_rejected(self) -> bool:
        return self.status == STATUS_REJECTED

    @property
    def is_accepted(self) -> bool:
        return self.status == STATUS_ACCEPTED

    @classmethod
    def ignored(cls) -> ConditionalPlanDecision:
        return cls(status=STATUS_IGNORED)

    @classmethod
    def malformed(cls) -> ConditionalPlanDecision:
        return cls(status=STATUS_MALFORMED)

    @classmethod
    def rejected(
        cls, reason: str, *, plan: Optional[Any] = None
    ) -> ConditionalPlanDecision:
        return cls(status=STATUS_REJECTED, plan=plan, rejection_reason=reason)

    @classmethod
    def accepted(cls, plan: dict[str, Any]) -> ConditionalPlanDecision:
        return cls(status=STATUS_ACCEPTED, plan=plan)


@dataclass(frozen=True)
class ConditionalFallbackRequest:
    prompt: str
    system_prompt: str = CONDITIONAL_FALLBACK_SYSTEM_PROMPT
    history: list[dict[str, Any]] = field(default_factory=list)
    tools_enabled: bool = False
    structured_answer: bool = False
    max_tokens_override: int = CONDITIONAL_FALLBACK_MAX_TOKENS


class LLMConditionalPlanning:
    FALLBACK_SYSTEM_PROMPT = CONDITIONAL_FALLBACK_SYSTEM_PROMPT
    FALLBACK_MAX_TOKENS = CONDITIONAL_FALLBACK_MAX_TOKENS

    @classmethod
    def build_fallback_prompt(cls, user_prompt: str) -> str:
        return (
            "把下面的现实条件任务转换成一个 JSON 对象。只能输出 JSON，不要回答任务，"
            "不要声称已经观察或执行。对象必须且只能包含 observation、condition、"
            "action_name、action_arguments。condition 保留用户完整的肯定或否定条件。"
            "action_name 只能是 play_sequence、express_emotion、set_tracking_mode、"
            "set_vision_gate、stop_all。常用 play_sequence 参数：举手=raise_hand，"
            "点头=basic_nod，挥手=wave_hello，双手放下=arms_down，回正=look_center；"
            "action_arguments 必须是对应动作的参数对象。禁止使用 move_chassis。\n"
            f"用户任务：{user_prompt}"
        )

    @classmethod
    def build_fallback_request(cls, user_prompt: str) -> ConditionalFallbackRequest:
        return ConditionalFallbackRequest(
            prompt=cls.build_fallback_prompt(user_prompt),
            system_prompt=cls.FALLBACK_SYSTEM_PROMPT,
            history=[],
            tools_enabled=False,
            structured_answer=False,
            max_tokens_override=cls.FALLBACK_MAX_TOKENS,
        )

    @classmethod
    def parse_fallback_json(cls, text: Any) -> dict[str, Any]:
        if isinstance(text, str):
            raw = text.strip()
        elif isinstance(text, (list, tuple)):
            raw = "".join(str(chunk) for chunk in text).strip()
        elif text is None:
            raw = ""
        else:
            raw = str(text).strip()

        if raw.startswith("```") and raw.endswith("```"):
            lines = raw.splitlines()
            raw = "\n".join(lines[1:-1]).strip() if len(lines) >= 3 else raw

        try:
            value = json.loads(raw)
        except json.JSONDecodeError:
            start, end = raw.find("{"), raw.rfind("}")
            if start < 0 or end <= start:
                raise ValueError("conditional_json_plan_missing")
            value = json.loads(raw[start:end + 1])

        if not isinstance(value, dict):
            raise ValueError("conditional_json_plan_not_object")

        return value

    @classmethod
    def evaluate_plan(
        cls,
        raw_plan: Any,
        *,
        user_prompt: str,
    ) -> ConditionalPlanDecision:
        plan = canonicalize_conditional_action(user_prompt, raw_plan)
        allowed, reason = validate_action_call(
            user_prompt,
            CONDITIONAL_TASK_TOOL_NAME,
            plan,
        )
        if not allowed:
            return ConditionalPlanDecision.rejected(
                reason=reason or "conditional_plan_rejected",
                plan=plan,
            )
        return ConditionalPlanDecision.accepted(plan=plan)

    @classmethod
    def evaluate_native_event(
        cls,
        data: Any,
        *,
        user_prompt: str,
    ) -> ConditionalPlanDecision:
        if not isinstance(data, Mapping) or data.get("type") != "tool_call":
            return ConditionalPlanDecision.ignored()
        if data.get("name") != CONDITIONAL_TASK_TOOL_NAME:
            return ConditionalPlanDecision.ignored()

        try:
            raw_plan = json.loads(data.get("arguments") or "{}")
        except (TypeError, json.JSONDecodeError):
            return ConditionalPlanDecision.malformed()

        return cls.evaluate_plan(raw_plan, user_prompt=user_prompt)


def build_conditional_fallback_prompt(user_prompt: str) -> str:
    return LLMConditionalPlanning.build_fallback_prompt(user_prompt)


def build_conditional_fallback_request(user_prompt: str) -> ConditionalFallbackRequest:
    return LLMConditionalPlanning.build_fallback_request(user_prompt)


def parse_conditional_fallback_json(text: Any) -> dict[str, Any]:
    return LLMConditionalPlanning.parse_fallback_json(text)


def evaluate_conditional_plan(
    raw_plan: Any,
    *,
    user_prompt: str,
) -> ConditionalPlanDecision:
    return LLMConditionalPlanning.evaluate_plan(raw_plan, user_prompt=user_prompt)


def evaluate_native_conditional_event(
    data: Any,
    *,
    user_prompt: str,
) -> ConditionalPlanDecision:
    return LLMConditionalPlanning.evaluate_native_event(data, user_prompt=user_prompt)
