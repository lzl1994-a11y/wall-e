"""ROS-independent routing for voice-dialogue tool calls."""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from services.action.action_intent_guard import validate_action_arguments
from services.orchestration.conditional_task import CONDITIONAL_TASK_TOOL_NAME
from services.vision.visual_search import VISUAL_SEARCH_TOOL_NAME


ToolResult = str | dict[str, Any]


class DialogToolRouter:
    """Routes dialog tools while keeping device effects behind callbacks."""

    def __init__(
        self,
        *,
        inspect_camera: Callable[[Any], ToolResult],
        run_conditional_task: Callable[[Any], ToolResult],
        run_visual_search: Callable[[Any], ToolResult],
        execute_action: Callable[[str, dict[str, Any]], dict[str, Any]],
    ) -> None:
        self._inspect_camera = inspect_camera
        self._run_conditional_task = run_conditional_task
        self._run_visual_search = run_visual_search
        self._execute_action = execute_action

    def dispatch(self, name: str, arguments: Any) -> ToolResult:
        if name == "inspect_camera":
            return self._inspect_camera(arguments)
        if name == CONDITIONAL_TASK_TOOL_NAME:
            return self._run_conditional_task(arguments)
        if name == VISUAL_SEARCH_TOOL_NAME:
            return self._run_visual_search(arguments)

        normalized_arguments = arguments if isinstance(arguments, dict) else None
        allowed, reason = validate_action_arguments(name, normalized_arguments)
        if not allowed:
            return {
                "status": "rejected",
                "action": name,
                "reason": reason,
            }
        return self._execute_action(name, normalized_arguments)
