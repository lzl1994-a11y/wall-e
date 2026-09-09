"""JSON topic protocol shared by Python plan clients and the native BT node."""

from __future__ import annotations

import json
from typing import Any

from services.action_plan import ActionPlan


BEHAVIOR_TREE_EXECUTE_TOPIC = "/behavior_tree/execute"
BEHAVIOR_TREE_CANCEL_TOPIC = "/behavior_tree/cancel"
BEHAVIOR_TREE_STATUS_TOPIC = "/behavior_tree/status"

PLAN_STATUSES = frozenset({"accepted", "running", "success", "failure", "halted", "rejected"})
TERMINAL_PLAN_STATUSES = frozenset({"success", "failure", "halted", "rejected"})


def encode_plan_request(plan: ActionPlan) -> str:
    return json.dumps(plan.to_dict(), ensure_ascii=False, separators=(",", ":"))


def encode_plan_cancel(plan_id: str) -> str:
    if not isinstance(plan_id, str) or not plan_id.strip():
        raise ValueError("plan_id must be a non-empty string")
    return json.dumps({"plan_id": plan_id.strip()}, separators=(",", ":"))


def parse_plan_status(payload: Any) -> dict[str, Any] | None:
    try:
        data = json.loads(payload) if isinstance(payload, str) else payload
    except (TypeError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict):
        return None
    plan_id = data.get("plan_id")
    status = data.get("status")
    if (
        not isinstance(plan_id, str)
        or not plan_id.strip()
        or status not in PLAN_STATUSES
    ):
        return None
    results = data.get("results", [])
    if not isinstance(results, list) or any(not isinstance(item, dict) for item in results):
        return None
    return {
        "plan_id": plan_id.strip(),
        "status": status,
        "results": [dict(item) for item in results],
        "error": str(data.get("error") or ""),
        "source": str(data.get("source") or "native_behavior_tree"),
    }


__all__ = [
    "BEHAVIOR_TREE_CANCEL_TOPIC",
    "BEHAVIOR_TREE_EXECUTE_TOPIC",
    "BEHAVIOR_TREE_STATUS_TOPIC",
    "PLAN_STATUSES",
    "TERMINAL_PLAN_STATUSES",
    "encode_plan_cancel",
    "encode_plan_request",
    "parse_plan_status",
]
