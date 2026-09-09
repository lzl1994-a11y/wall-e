"""Thread-safe client for the native BehaviorTree.CPP ROS node."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from typing import Any

from services.action_plan import ActionPlan
from services.behavior_tree_protocol import (
    TERMINAL_PLAN_STATUSES,
    encode_plan_cancel,
    encode_plan_request,
    parse_plan_status,
)


class CorrelatedPlanExecutor:
    """Submit one plan and wait for its correlated terminal tree status."""

    def __init__(self):
        self._condition = threading.Condition()
        self._statuses: dict[str, dict[str, Any]] = {}

    def accept_status(self, payload: Any) -> bool:
        status = parse_plan_status(payload)
        if status is None:
            return False
        with self._condition:
            self._statuses[status["plan_id"]] = status
            self._condition.notify_all()
        return True

    def try_execute(
        self,
        plan: ActionPlan,
        *,
        publish: Callable[[str], None],
        cancel_publish: Callable[[str], None],
        owner_available: Callable[[], bool],
        timeout: float | None = None,
        cancelled: Callable[[], bool] | None = None,
    ) -> dict[str, Any] | None:
        """Return ``None`` only when no native owner is available before submit."""
        owner_deadline = time.monotonic() + 0.75
        while time.monotonic() < owner_deadline:
            if owner_available():
                break
            if cancelled and cancelled():
                return self._interrupted(plan, "turn_cancelled_before_submit")
            time.sleep(0.05)
        else:
            return None

        deadline = time.monotonic() + (
            float(timeout) if timeout is not None else max(22.0, len(plan.steps) * 22.0)
        )
        with self._condition:
            self._statuses.pop(plan.plan_id, None)
        try:
            publish(encode_plan_request(plan))
        except Exception as exc:
            return self._failure(plan, f"plan_publish_failed:{exc}")

        cancel_sent = False
        latest = None
        with self._condition:
            while time.monotonic() < deadline:
                latest = self._statuses.get(plan.plan_id)
                if latest and latest["status"] in TERMINAL_PLAN_STATUSES:
                    break
                if cancelled and cancelled() and not cancel_sent:
                    cancel_sent = True
                    try:
                        cancel_publish(encode_plan_cancel(plan.plan_id))
                    except Exception:
                        pass
                remaining = max(0.01, deadline - time.monotonic())
                self._condition.wait(timeout=min(0.1, remaining))
            latest = self._statuses.pop(plan.plan_id, latest)

        if latest and latest["status"] in TERMINAL_PLAN_STATUSES:
            return {
                **latest,
                "plan": plan.to_dict(),
                "stopped": latest["status"] != "success",
            }
        if cancel_sent:
            return self._interrupted(plan, "native_cancel_timeout")
        return self._failure(plan, "native_plan_timeout")

    @staticmethod
    def _failure(plan: ActionPlan, reason: str) -> dict[str, Any]:
        return {
            "plan_id": plan.plan_id,
            "plan": plan.to_dict(),
            "status": "failure",
            "results": CorrelatedPlanExecutor._synthetic_results(
                plan, "failed", reason
            ),
            "stopped": True,
            "error": reason,
            "source": "native_behavior_tree_client",
        }

    @staticmethod
    def _interrupted(plan: ActionPlan, reason: str) -> dict[str, Any]:
        return {
            "plan_id": plan.plan_id,
            "plan": plan.to_dict(),
            "status": "halted",
            "results": CorrelatedPlanExecutor._synthetic_results(
                plan, "interrupted", reason
            ),
            "stopped": True,
            "error": reason,
            "source": "native_behavior_tree_client",
        }

    @staticmethod
    def _synthetic_results(
        plan: ActionPlan,
        first_status: str,
        reason: str,
    ) -> list[dict[str, Any]]:
        results = []
        for index, step in enumerate(plan.steps):
            results.append({
                **step.to_dict(),
                "status": first_status if index == 0 else "skipped",
                "action": step.name,
                "reason": reason if index == 0 else "prior_action_not_completed",
                "node_status": (
                    "halted" if index == 0 and first_status == "interrupted"
                    else "failure" if index == 0
                    else "idle"
                ),
            })
        return results


__all__ = ["CorrelatedPlanExecutor"]
