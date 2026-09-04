"""Thread-safe correlated ROS action completion tracking."""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from typing import Any

from services.action_command import build_action_cmd, new_action_request_id
from services.action_status import TERMINAL_ACTION_STATUSES, parse_action_status


class CorrelatedActionExecutor:
    """Publish one action and wait for its owning ROS node's terminal status."""

    def __init__(self):
        self._condition = threading.Condition()
        self._statuses: dict[str, dict[str, str]] = {}

    def accept_status(self, payload: Any) -> bool:
        status = parse_action_status(payload)
        if status is None:
            return False
        with self._condition:
            self._statuses[status["request_id"]] = status
            self._condition.notify_all()
        return True

    def execute(
        self,
        name: str,
        arguments: dict[str, Any],
        *,
        publish: Callable[[str], None],
        owner_available: Callable[[], bool],
        timeout: float = 20.0,
        source: str = "llm_workflow",
    ) -> dict[str, Any]:
        started = time.monotonic()
        deadline = started + max(0.1, float(timeout))
        owner_deadline = min(deadline, started + 2.0)
        while time.monotonic() < owner_deadline:
            if owner_available():
                break
            time.sleep(0.05)
        else:
            return {
                "status": "failed",
                "action": name,
                "reason": "ros_action_owner_unavailable",
            }

        request_id = new_action_request_id()
        payload = build_action_cmd(
            name,
            arguments,
            request_id=request_id,
            source=source,
        )
        with self._condition:
            self._statuses.pop(request_id, None)
        try:
            publish(payload)
        except Exception as exc:
            return {
                "status": "failed",
                "action": name,
                "request_id": request_id,
                "reason": f"publish_failed:{exc}",
            }

        latest = None
        with self._condition:
            while time.monotonic() < deadline:
                latest = self._statuses.get(request_id)
                if (
                    latest
                    and latest["name"] == name
                    and latest["status"] in TERMINAL_ACTION_STATUSES
                ):
                    break
                self._condition.wait(timeout=max(0.01, deadline - time.monotonic()))
            latest = self._statuses.pop(request_id, latest)

        if (
            latest is None
            or latest["name"] != name
            or latest["status"] not in TERMINAL_ACTION_STATUSES
        ):
            return {
                "status": "timeout",
                "action": name,
                "request_id": request_id,
                "reason": "no_terminal_executor_status",
                **({"last_status": latest["status"]} if latest else {}),
            }
        return {
            "status": latest["status"],
            "action": name,
            "request_id": request_id,
            "executor": latest["source"],
            **({"reason": latest["detail"]} if latest["detail"] else {}),
        }


__all__ = ["CorrelatedActionExecutor"]
