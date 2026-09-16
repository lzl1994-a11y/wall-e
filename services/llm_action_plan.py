"""Execute LLM-proposed action lists through native or correlated transports."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from services.behavior_tree_workflow import (
    BehaviorTreeActionWorkflow,
    NativeBehaviorTreeWorkflow,
)


class LLMActionPlanWorkflow:
    """Prefer native plans and retain the bounded correlated-action fallback."""

    def __init__(
        self,
        *,
        native_available: Callable[[], bool],
        authorize: Callable[[str, str, dict[str, Any]], tuple[bool, str]],
        execute_plan: Callable[[Any], dict[str, Any] | None],
        execute_action: Callable[[str, dict[str, Any], str], Any],
        cancelled: Callable[[], bool],
    ) -> None:
        self._native_available = native_available
        self._authorize = authorize
        self._execute_plan = execute_plan
        self._execute_action = execute_action
        self._cancelled = cancelled

    def invoke(
        self,
        *,
        actions: list[dict[str, Any]],
        user_prompt: str,
        turn_id: str,
    ) -> dict[str, Any]:
        if self._native_available():
            native_state = NativeBehaviorTreeWorkflow(
                authorize=self._authorize,
                execute_plan=self._execute_plan,
            ).invoke(
                turn_id=turn_id,
                user_prompt=user_prompt,
                actions=actions,
            )
            if native_state is not None:
                return native_state
        return BehaviorTreeActionWorkflow(
            authorize=self._authorize,
            execute=lambda name, arguments: self._execute_action(
                name, arguments, turn_id
            ),
            cancelled=self._cancelled,
        ).invoke(
            turn_id=turn_id,
            user_prompt=user_prompt,
            actions=actions,
        )


__all__ = ["LLMActionPlanWorkflow"]
