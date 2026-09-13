"""Standard ROS 2 Action client adapter for Wali task plans.

The adapter deliberately has no direct rclpy imports so its lifecycle logic can
be tested without ROS. Nodes provide an ActionClient and the generated goal
class at their ROS boundary.
"""

from __future__ import annotations

import json
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

from services.behavior_tree_execution import CorrelatedPlanExecutor
from services.behavior_tree_protocol import parse_plan_status


WALI_TASK_ACTION_NAME = "wali_task"
WALI_TASK_TREE_NAME = "WaliTask"


def create_wali_task_action_client(node: Any, root: Path) -> tuple[Any, type[Any]]:
    """Create the generated ExecuteTree client from this checkout's overlay."""
    import sys

    for pattern in (
        "install/btcpp_ros2_interfaces/local/lib/python*/site-packages",
        "install/btcpp_ros2_interfaces/local/lib/python*/dist-packages",
        "install/btcpp_ros2_interfaces/lib/python*/site-packages",
        "install/btcpp_ros2_interfaces/lib/python*/dist-packages",
    ):
        for candidate in root.glob(pattern):
            text = str(candidate)
            if text not in sys.path:
                sys.path.insert(0, text)
    from btcpp_ros2_interfaces.action import ExecuteTree
    from rclpy.action import ActionClient

    return ActionClient(node, ExecuteTree, WALI_TASK_ACTION_NAME), ExecuteTree


class Ros2ActionPlanExecutor:
    """Submit one plan through ExecuteTree and await its correlated result."""

    def __init__(self, action_client: Any, goal_type: type[Any]):
        self._client = action_client
        self._goal_type = goal_type

    def try_execute(
        self,
        plan: Any,
        *,
        timeout: float | None = None,
        cancelled: Callable[[], bool] | None = None,
    ) -> dict[str, Any] | None:
        """Return None only if the Action server is unavailable before submit."""
        if not self._client.wait_for_server(timeout_sec=0.75):
            return None

        goal = self._goal_type.Goal()
        goal.target_tree = WALI_TASK_TREE_NAME
        goal.payload = json.dumps(plan.to_dict(), ensure_ascii=False, separators=(",", ":"))
        condition = threading.Condition()
        state: dict[str, Any] = {"goal": None, "result": None, "error": ""}

        def on_result(future: Any) -> None:
            try:
                wrapped = future.result()
                state["result"] = wrapped
            except Exception as exc:  # ROS future boundary
                state["error"] = f"action_result_failed:{exc}"
            with condition:
                condition.notify_all()

        def on_goal(future: Any) -> None:
            try:
                goal_handle = future.result()
                state["goal"] = goal_handle
                if goal_handle is None or not goal_handle.accepted:
                    state["error"] = "action_goal_rejected"
                else:
                    goal_handle.get_result_async().add_done_callback(on_result)
            except Exception as exc:  # ROS future boundary
                state["error"] = f"action_goal_failed:{exc}"
            with condition:
                condition.notify_all()

        try:
            self._client.send_goal_async(goal).add_done_callback(on_goal)
        except Exception as exc:
            return self._failure(plan, f"action_goal_send_failed:{exc}")

        policy_budget = sum(
            step.timeout_ms * step.max_attempts / 1000.0 for step in plan.steps
        )
        deadline = time.monotonic() + (
            float(timeout) if timeout is not None else max(22.0, policy_budget + 2.0)
        )
        cancel_sent = False
        with condition:
            while time.monotonic() < deadline:
                if state["result"] is not None or state["error"]:
                    break
                goal_handle = state["goal"]
                if cancelled and cancelled() and goal_handle is not None and not cancel_sent:
                    cancel_sent = True
                    try:
                        goal_handle.cancel_goal_async()
                    except Exception:
                        pass
                condition.wait(timeout=min(0.1, max(0.01, deadline - time.monotonic())))

        if state["error"]:
            return self._failure(plan, state["error"])
        wrapped = state["result"]
        if wrapped is not None:
            status = parse_plan_status(wrapped.result.return_message)
            if status is not None:
                return {**status, "plan": plan.to_dict(), "stopped": status["status"] != "success"}
            return self._failure(plan, "action_result_invalid_payload")
        if cancel_sent:
            return self._interrupted(plan, "ros2_action_cancel_timeout")
        goal_handle = state["goal"]
        if goal_handle is not None:
            try:
                goal_handle.cancel_goal_async()
            except Exception:
                pass
        return self._failure(plan, "ros2_action_timeout")

    @staticmethod
    def _failure(plan: Any, reason: str) -> dict[str, Any]:
        return {
            "plan_id": plan.plan_id,
            "plan": plan.to_dict(),
            "status": "failure",
            "results": CorrelatedPlanExecutor._synthetic_results(plan, "failed", reason),
            "stopped": True,
            "error": reason,
            "source": "ros2_action_client",
        }

    @staticmethod
    def _interrupted(plan: Any, reason: str) -> dict[str, Any]:
        return {
            "plan_id": plan.plan_id,
            "plan": plan.to_dict(),
            "status": "halted",
            "results": CorrelatedPlanExecutor._synthetic_results(
                plan, "interrupted", reason
            ),
            "stopped": True,
            "error": reason,
            "source": "ros2_action_client",
        }


__all__ = [
    "Ros2ActionPlanExecutor",
    "WALI_TASK_ACTION_NAME",
    "WALI_TASK_TREE_NAME",
    "create_wali_task_action_client",
]
