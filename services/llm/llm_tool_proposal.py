"""Pure service for evaluating LLM-proposed tool calls."""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Mapping, Optional

from services.action.action_intent_guard import validate_action_call
from services.orchestration.conditional_task import CONDITIONAL_TASK_TOOL_NAME

ROUTE_ACTION = "action"
ROUTE_CAMERA_PHOTO = "camera_photo"
ROUTE_CAMERA_INSPECTION = "camera_inspection"
ROUTE_CONDITIONAL_TASK = "conditional_task"

STREAM_CONTROL_BREAK = "break"
STREAM_CONTROL_CONTINUE = "continue"
STREAM_CONTROL_PROCEED = "proceed"

REJECTION_MALFORMED_ARGUMENTS = "malformed_arguments"
REJECTION_POLICY = "policy"


@dataclass(frozen=True)
class ToolProposalDecision:
    is_accepted: bool
    route: Optional[str] = None
    action_name: str = ""
    arguments: Any = field(default_factory=dict)
    plan: Optional[dict[str, Any]] = None
    rejection_reason: Optional[str] = None
    rejection_kind: Optional[str] = None
    stream_control: str = STREAM_CONTROL_PROCEED

    @property
    def is_rejected(self) -> bool:
        return not self.is_accepted

    @property
    def should_break(self) -> bool:
        return self.stream_control == STREAM_CONTROL_BREAK

    @property
    def should_continue(self) -> bool:
        return self.stream_control == STREAM_CONTROL_CONTINUE

    @property
    def is_malformed_arguments(self) -> bool:
        return self.rejection_kind == REJECTION_MALFORMED_ARGUMENTS

    @property
    def is_action(self) -> bool:
        return self.is_accepted and self.route == ROUTE_ACTION

    @property
    def is_camera_photo(self) -> bool:
        return self.is_accepted and self.route == ROUTE_CAMERA_PHOTO

    @property
    def is_camera_inspection(self) -> bool:
        return self.is_accepted and self.route == ROUTE_CAMERA_INSPECTION

    @property
    def is_conditional_task(self) -> bool:
        return self.is_accepted and self.route == ROUTE_CONDITIONAL_TASK

    @classmethod
    def accept(
        cls,
        *,
        route: str,
        action_name: str,
        arguments: dict[str, Any],
        plan: Optional[dict[str, Any]] = None,
    ) -> ToolProposalDecision:
        return cls(
            is_accepted=True,
            route=route,
            action_name=action_name,
            arguments=arguments,
            plan=plan,
            stream_control=STREAM_CONTROL_PROCEED,
        )

    @classmethod
    def reject(
        cls,
        *,
        action_name: str,
        reason: str,
        stream_control: str = STREAM_CONTROL_BREAK,
        arguments: Any = None,
        rejection_kind: str = REJECTION_POLICY,
    ) -> ToolProposalDecision:
        return cls(
            is_accepted=False,
            route=None,
            action_name=action_name,
            arguments={} if arguments is None else arguments,
            plan=None,
            rejection_reason=reason,
            rejection_kind=rejection_kind,
            stream_control=stream_control,
        )


class LLMToolProposalEvaluator:
    @classmethod
    def evaluate(
        cls,
        tool_call: Any,
        *,
        user_prompt: str,
        conditional_request: bool = False,
    ) -> ToolProposalDecision:
        if not isinstance(tool_call, Mapping):
            return ToolProposalDecision.reject(
                action_name="",
                reason="invalid_arguments",
                stream_control=STREAM_CONTROL_BREAK,
                rejection_kind=REJECTION_MALFORMED_ARGUMENTS,
            )

        action_name = str(tool_call.get("name") or "")
        raw_arguments = tool_call.get("arguments")

        # Parse arguments JSON or accept dict
        if isinstance(raw_arguments, Mapping):
            action_arguments = dict(raw_arguments)
        elif isinstance(raw_arguments, str) or raw_arguments is None:
            try:
                parsed = json.loads(raw_arguments or "{}")
            except (TypeError, json.JSONDecodeError):
                return ToolProposalDecision.reject(
                    action_name=action_name,
                    reason="invalid_arguments",
                    stream_control=STREAM_CONTROL_BREAK,
                    rejection_kind=REJECTION_MALFORMED_ARGUMENTS,
                )
            action_arguments = parsed
        else:
            return ToolProposalDecision.reject(
                action_name=action_name,
                reason="invalid_arguments",
                stream_control=STREAM_CONTROL_BREAK,
                rejection_kind=REJECTION_MALFORMED_ARGUMENTS,
            )

        # Atomic compound task protection
        if conditional_request and action_name != CONDITIONAL_TASK_TOOL_NAME:
            return ToolProposalDecision.reject(
                action_name=action_name,
                reason="compound_task_must_stay_atomic",
                stream_control=STREAM_CONTROL_CONTINUE,
                arguments=action_arguments,
            )

        # Semantic intent and argument allowlist validation
        allowed, rejection_reason = validate_action_call(
            user_prompt,
            action_name,
            action_arguments,
        )
        if not allowed:
            return ToolProposalDecision.reject(
                action_name=action_name,
                reason=rejection_reason or "action_disallowed",
                stream_control=STREAM_CONTROL_BREAK,
                arguments=action_arguments,
            )

        # Route determination for valid calls
        if action_name == "inspect_camera":
            if action_arguments.get("save_photo") is True:
                return ToolProposalDecision.accept(
                    route=ROUTE_CAMERA_PHOTO,
                    action_name=action_name,
                    arguments=action_arguments,
                )
            return ToolProposalDecision.accept(
                route=ROUTE_CAMERA_INSPECTION,
                action_name=action_name,
                arguments=action_arguments,
            )

        if action_name == CONDITIONAL_TASK_TOOL_NAME:
            return ToolProposalDecision.accept(
                route=ROUTE_CONDITIONAL_TASK,
                action_name=action_name,
                arguments=action_arguments,
                plan=action_arguments,
            )

        return ToolProposalDecision.accept(
            route=ROUTE_ACTION,
            action_name=action_name,
            arguments=action_arguments,
        )


def evaluate_tool_proposal(
    tool_call: Any,
    *,
    user_prompt: str,
    conditional_request: bool = False,
) -> ToolProposalDecision:
    return LLMToolProposalEvaluator.evaluate(
        tool_call,
        user_prompt=user_prompt,
        conditional_request=conditional_request,
    )
