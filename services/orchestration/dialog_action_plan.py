"""Shared native-first action sequencing for text and audio dialogue.

Each dialogue entry point supplies its native submission callback and its own
authorization policy. Only a pre-submit decline (``None``) permits the
compatibility sequence; a submitted plan's failure must never execute it again.
Speech, model requests, and semantic camera workflows stay with their callers.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any

from services.orchestration.behavior_tree_workflow import BehaviorTreeActionWorkflow


class DialogActionPlanWorkflow:
    """Run a bounded sequence when native submission declines this action list."""

    def __init__(
        self,
        *,
        submit_native: Callable[
            [str, str, list[dict[str, Any]]], dict[str, Any] | None
        ],
        authorize: Callable[[str, str, dict[str, Any]], tuple[bool, str]],
        execute_action: Callable[[str, dict[str, Any], str], Any],
        cancelled: Callable[[], bool],
    ) -> None:
        self._submit_native = submit_native
        self._authorize = authorize
        self._execute_action = execute_action
        self._cancelled = cancelled

    def invoke(
        self,
        *,
        actions: list[dict[str, Any]],
        user_prompt: str,
        turn_id: str,
    ) -> dict[str, Any]:
        native_state = self._submit_native(turn_id, user_prompt, actions)
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


__all__ = ["DialogActionPlanWorkflow"]
