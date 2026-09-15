"""Dialogue-specific routing for native behavior-tree action plans."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from services.behavior_tree_workflow import NativeBehaviorTreeWorkflow
from services.conditional_task import CONDITIONAL_TASK_TOOL_NAME
from services.visual_search import VISUAL_SEARCH_TOOL_NAME


SPECIAL_DIALOG_ACTIONS = frozenset({
    "inspect_camera",
    CONDITIONAL_TASK_TOOL_NAME,
    VISUAL_SEARCH_TOOL_NAME,
})


class DialogNativePlanWorkflow:
    """Submit ordinary dialog actions natively and leave special tools routed."""

    def __init__(
        self,
        *,
        authorize: Callable[[str, str, dict[str, Any]], tuple[bool, str]],
        execute_plan: Callable[[Any], dict[str, Any] | None],
    ) -> None:
        self._workflow = NativeBehaviorTreeWorkflow(
            authorize=authorize,
            execute_plan=execute_plan,
        )

    def invoke(
        self,
        *,
        turn_id: str,
        user_prompt: str,
        actions: list[dict[str, Any]],
    ) -> dict[str, Any] | None:
        if any(
            action.get("name") in SPECIAL_DIALOG_ACTIONS
            for action in actions
            if isinstance(action, dict)
        ):
            return None
        return self._workflow.invoke(
            turn_id=turn_id,
            user_prompt=user_prompt,
            actions=actions,
        )


__all__ = ["DialogNativePlanWorkflow", "SPECIAL_DIALOG_ACTIONS"]
