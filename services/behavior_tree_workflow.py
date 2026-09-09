"""Phase-one compatibility behavior-tree runtime for ordinary actions.

The compatibility classes in this module are deliberately not the native
BehaviorTree.CPP engine.  They retain the same bounded ActionPlan plus
Sequence/ActionLeaf contract as a pre-submit fallback when the native ROS 2
tree owner is unavailable, without rewriting any underlying skill.
"""

from __future__ import annotations

from collections.abc import Callable
from enum import Enum
from typing import Any

from services.action_plan import (
    ActionPlan,
    ActionPlanStep,
    PlanValidationError,
    compile_action_plan,
)


class NodeStatus(str, Enum):
    IDLE = "idle"
    RUNNING = "running"
    SUCCESS = "success"
    FAILURE = "failure"
    HALTED = "halted"


class ActionLeaf:
    """Authorize one plan step and wait for its correlated terminal result."""

    def __init__(
        self,
        step: ActionPlanStep,
        *,
        user_prompt: str,
        authorize: Callable[[str, str, dict[str, Any]], tuple[bool, str]],
        execute: Callable[[str, dict[str, Any]], Any],
        cancelled: Callable[[], bool] | None,
    ):
        self.step = step
        self._user_prompt = user_prompt
        self._authorize = authorize
        self._execute = execute
        self._cancelled = cancelled
        self.status = NodeStatus.IDLE

    def tick(self) -> dict[str, Any]:
        if self._cancelled is not None and self._cancelled():
            self.status = NodeStatus.HALTED
            return self._record("interrupted", reason="turn_cancelled")

        try:
            allowed, reason = self._authorize(
                self._user_prompt,
                self.step.name,
                self.step.arguments,
            )
        except Exception as exc:
            self.status = NodeStatus.FAILURE
            return self._record(
                "rejected",
                reason=f"authorization_failed:{exc}",
            )
        if not allowed:
            self.status = NodeStatus.FAILURE
            return self._record("rejected", reason=reason or "action_not_authorized")

        self.status = NodeStatus.RUNNING
        attempts: list[dict[str, Any]] = []
        for attempt in range(1, self.step.max_attempts + 1):
            if self._cancelled is not None and self._cancelled():
                self.status = NodeStatus.HALTED
                record = self._record("interrupted", reason="turn_cancelled")
                record["attempts"] = attempts
                return record
            try:
                raw_result = self._execute(self.step.name, dict(self.step.arguments))
                if isinstance(raw_result, dict):
                    result = dict(raw_result)
                elif isinstance(raw_result, str) and raw_result.strip():
                    result = {
                        "status": "completed",
                        "action": self.step.name,
                        "response": raw_result.strip(),
                    }
                else:
                    result = {
                        "status": "failed",
                        "action": self.step.name,
                        "reason": "missing_action_result",
                    }
            except Exception as exc:
                result = {
                    "status": "failed",
                    "action": self.step.name,
                    "reason": str(exc),
                }

            result.setdefault("action", self.step.name)
            result.setdefault("status", "failed")
            attempt_record = dict(result)
            attempt_record["attempt"] = attempt
            attempts.append(attempt_record)
            retryable = result["status"] in {"failed", "timeout"}
            if retryable and attempt < self.step.max_attempts:
                continue

            record = self.step.to_dict()
            record.update(result)
            self.status = (
                NodeStatus.SUCCESS
                if result["status"] == "completed"
                else NodeStatus.HALTED
                if result["status"] == "interrupted"
                else NodeStatus.FAILURE
            )
            record["node_status"] = self.status.value
            record["attempts"] = attempts
            return record

        raise RuntimeError("unreachable action retry state")

    def _record(self, status: str, *, reason: str) -> dict[str, Any]:
        return {
            **self.step.to_dict(),
            "status": status,
            "action": self.step.name,
            "reason": reason,
            "node_status": self.status.value,
        }


class SequenceNode:
    """Tick action leaves in order and fail closed on the first non-success."""

    def __init__(self, plan: ActionPlan, leaves: list[ActionLeaf]):
        self.plan = plan
        self.leaves = leaves
        self.status = NodeStatus.IDLE

    def tick(self) -> list[dict[str, Any]]:
        self.status = NodeStatus.RUNNING
        results: list[dict[str, Any]] = []
        for index, leaf in enumerate(self.leaves):
            result = leaf.tick()
            results.append(result)
            if leaf.status != NodeStatus.SUCCESS:
                self.status = (
                    NodeStatus.HALTED
                    if leaf.status == NodeStatus.HALTED
                    else NodeStatus.FAILURE
                )
                results.extend(
                    {
                        **pending.step.to_dict(),
                        "status": "skipped",
                        "action": pending.step.name,
                        "reason": "prior_action_not_completed",
                        "node_status": NodeStatus.IDLE.value,
                    }
                    for pending in self.leaves[index + 1:]
                )
                return results
        self.status = NodeStatus.SUCCESS
        return results


class BehaviorTreeActionWorkflow:
    """Compile and execute an ordinary LLM action list as a linear BT plan."""

    def __init__(
        self,
        *,
        authorize: Callable[[str, str, dict[str, Any]], tuple[bool, str]],
        execute: Callable[[str, dict[str, Any]], Any],
        cancelled: Callable[[], bool] | None = None,
    ):
        self._authorize = authorize
        self._execute = execute
        self._cancelled = cancelled

    def invoke(
        self,
        *,
        turn_id: str,
        user_prompt: str,
        actions: list[dict[str, Any]],
    ) -> dict[str, Any]:
        try:
            plan = compile_action_plan(
                turn_id=turn_id,
                user_prompt=user_prompt,
                actions=actions,
            )
        except PlanValidationError as exc:
            return {
                "plan_id": "",
                "plan": None,
                "results": [{
                    "status": "rejected",
                    "action": "",
                    "reason": str(exc),
                    "node_status": NodeStatus.FAILURE.value,
                }],
                "status": NodeStatus.FAILURE.value,
                "stopped": True,
                "error": str(exc),
            }

        leaves = [
            ActionLeaf(
                step,
                user_prompt=plan.user_prompt,
                authorize=self._authorize,
                execute=self._execute,
                cancelled=self._cancelled,
            )
            for step in plan.steps
        ]
        root = SequenceNode(plan, leaves)
        results = root.tick()
        state: dict[str, Any] = {
            "plan_id": plan.plan_id,
            "plan": plan.to_dict(),
            "results": results,
            "status": root.status.value,
            "stopped": root.status != NodeStatus.SUCCESS,
        }
        if root.status != NodeStatus.SUCCESS:
            first_failure = next(
                (
                    result
                    for result in results
                    if result.get("status") not in ("completed", "skipped")
                ),
                {},
            )
            state["error"] = str(
                first_failure.get("reason") or first_failure.get("status") or "action_failed"
            )
        return state


class NativeBehaviorTreeWorkflow:
    """Authorize a bounded plan, then submit it to the native ROS tree owner."""

    def __init__(
        self,
        *,
        authorize: Callable[[str, str, dict[str, Any]], tuple[bool, str]],
        execute_plan: Callable[[ActionPlan], dict[str, Any] | None],
    ):
        self._authorize = authorize
        self._execute_plan = execute_plan

    def invoke(
        self,
        *,
        turn_id: str,
        user_prompt: str,
        actions: list[dict[str, Any]],
    ) -> dict[str, Any] | None:
        try:
            plan = compile_action_plan(
                turn_id=turn_id,
                user_prompt=user_prompt,
                actions=actions,
            )
        except PlanValidationError as exc:
            return {
                "plan_id": "",
                "plan": None,
                "results": [{
                    "status": "rejected",
                    "action": "",
                    "reason": str(exc),
                    "node_status": NodeStatus.FAILURE.value,
                }],
                "status": NodeStatus.FAILURE.value,
                "stopped": True,
                "error": str(exc),
            }

        for index, step in enumerate(plan.steps):
            try:
                allowed, reason = self._authorize(
                    plan.user_prompt,
                    step.name,
                    step.arguments,
                )
            except Exception as exc:
                allowed, reason = False, f"authorization_failed:{exc}"
            if allowed:
                continue
            results = []
            for current_index, current in enumerate(plan.steps):
                if current_index == index:
                    status = "rejected"
                    item_reason = reason or "action_not_authorized"
                else:
                    status = "skipped"
                    item_reason = "plan_not_submitted"
                results.append({
                    **current.to_dict(),
                    "status": status,
                    "action": current.name,
                    "reason": item_reason,
                    "node_status": (
                        NodeStatus.FAILURE.value
                        if current_index == index
                        else NodeStatus.IDLE.value
                    ),
                })
            return {
                "plan_id": plan.plan_id,
                "plan": plan.to_dict(),
                "results": results,
                "status": NodeStatus.FAILURE.value,
                "stopped": True,
                "error": reason or "action_not_authorized",
            }
        result = self._execute_plan(plan)
        if result is None:
            return None
        results = result.get("results") if isinstance(result, dict) else None
        valid_results = (
            isinstance(results, list)
            and len(results) == len(plan.steps)
            and all(
                isinstance(item, dict)
                and item.get("step_id") == step.step_id
                and (item.get("name") or item.get("action")) == step.name
                for item, step in zip(results, plan.steps)
            )
        )
        if result.get("plan_id") == plan.plan_id and valid_results:
            return result

        status = result.get("status", "failure") if isinstance(result, dict) else "failure"
        reason = (
            str(result.get("error") or "invalid_native_plan_result")
            if isinstance(result, dict)
            else "invalid_native_plan_result"
        )
        if status == "halted":
            first_status, node_status = "interrupted", NodeStatus.HALTED.value
        elif status == "rejected":
            first_status, node_status = "rejected", NodeStatus.FAILURE.value
        else:
            first_status, node_status = "failed", NodeStatus.FAILURE.value
        normalized = []
        for index, step in enumerate(plan.steps):
            normalized.append({
                **step.to_dict(),
                "status": first_status if index == 0 else "skipped",
                "action": step.name,
                "reason": reason if index == 0 else "prior_action_not_completed",
                "node_status": node_status if index == 0 else NodeStatus.IDLE.value,
            })
        return {
            "plan_id": plan.plan_id,
            "plan": plan.to_dict(),
            "results": normalized,
            "status": "halted" if status == "halted" else "failure",
            "stopped": True,
            "error": reason,
            "source": "native_behavior_tree_client",
        }


__all__ = [
    "ActionLeaf",
    "BehaviorTreeActionWorkflow",
    "NativeBehaviorTreeWorkflow",
    "NodeStatus",
    "SequenceNode",
]
